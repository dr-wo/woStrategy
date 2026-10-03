"""Offline Part 1 reproduction; all outputs go to --output, no network or data writes.

PYTHONPATH must select the three feature worktrees (see README.md).
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import types
import subprocess

import numpy as np
import pandas as pd
from wodata.live.models import LiveTimingMessage
from wodata.live.reducer import _decode_compressed_payload
from wodata.live.recovery import load_recording
from woplanner.live.telemetry import _analysis_store, _snapshot_at_cut, CausalAnalysisCut
from woplanner.analysis.service import (live_position_traffic_shadow, _live_traffic_gaps,
    _snapshot_telemetry_frame, _estimate_live_circuit_length)
from wostrategy.analysis.position_traffic import PositionReference
from wostrategy.analysis import traffic


def stamp(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')) if value else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    a=np.asarray(values,float); a=a[np.isfinite(a)]
    return dict(n=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(max(a)),min=float(min(a))) if len(a) else {'n':0}


def reference(path, window, frame):
    selected=[]
    for r in json.loads(path.read_text())['records']:
        if not window[0] <= pd.Timedelta(r['original_timestamp']).total_seconds() <= window[1]: continue
        for p in r['payload'].get('Position',[]):
            v=p['Entries'].get('1')
            if v and str(v.get('Status')).lower() in ['ontrack','0']:
                selected.append((stamp(p['Timestamp']).timestamp(),v['X']/10,v['Y']/10))
    data=pd.DataFrame(selected,columns=['t','x','y']).sort_values('t').drop_duplicates('t')
    xy=data[['x','y']].to_numpy(); xy=xy[np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>0]]
    end=float(data.t.max())
    return PositionReference(tuple(map(tuple,np.vstack([xy,xy[0]]))),frame,str(path),sha(path),end,end,
        'historical_preload_assumption; archive materialization does not prove live availability')


def replay(path, leader_lap, canonical=False):
    store=_analysis_store(); history=[]; timing=[]; positions_arrival={}; start=time.perf_counter()
    if canonical:
        def iterator():
            for i,line in enumerate(path.open(),1):
                v=json.loads(line); ts=stamp(v.get('original_timestamp'))
                clock=v.get('message_session_time_seconds') if v.get('preserve_message_session_time') else v.get('session_time_seconds')
                yield i, LiveTimingMessage(v['topic'],v['payload'],ts,ts or datetime(2000,1,1,tzinfo=timezone.utc),
                    source='f1_archive_replay',sequence=i,session_time_seconds=clock)
    else:
        def iterator():
            for item in load_recording(path): yield item.source_order,item.message
    for i,m in iterator():
        history.append((i,m)); store.reduce(m)
        if m.topic=='TimingData' and m.source_timestamp_utc:
            timing.append((m.source_timestamp_utc.timestamp(),i,m.payload))
        if m.topic=='Position.z':
            payload=_decode_compressed_payload(m.payload)
            for p in payload.get('Position',[]):
                t=stamp(p['Timestamp']).timestamp()
                positions_arrival.setdefault(t,i)
        # Internal committed-lap state avoids materializing a huge snapshot per message.
        if any(n==leader_lap and v.get('position')==1 for laps in store._laps.values() for n,v in laps.items()): break
    snapshot=store.snapshot()
    leader=next(x for x in snapshot.completed_laps if x.lap_number==leader_lap and x.position==1)
    clock=leader.session_time_seconds if leader.session_time_seconds is not None else leader.source_timestamp_utc.timestamp()
    cut=CausalAnalysisCut(leader_lap,clock,i,0,leader.source_timestamp_utc)
    snapshot=_snapshot_at_cut(snapshot,cut)
    return snapshot,cut,history,timing,positions_arrival,time.perf_counter()-start


def independent(samples, timing, arrivals, codes, *, side="rear"):
    fields={}
    for t,line,payload in timing:
        for number, patch in payload.get('Lines',{}).items():
            for key in ['Position','NumberOfLaps','IntervalToPositionAhead','InPit']:
                if key in patch:
                    value=patch[key];value=value.get('Value') if isinstance(value,dict) else value
                    fields.setdefault((codes.get(number,number),key),[]).append((t,line,value))
    for values in fields.values(): values.sort()
    def get(driver,key,t,line):
        values=fields.get((driver,key),[])
        j=np.searchsorted([v[0] for v in values],t,side='right')-1
        while j>=0 and values[j][1]>line:j-=1
        return (values[j][2],t-values[j][0]) if j>=0 else (None,None)
    rows=[]
    for r in samples.itertuples():
        if r.NeighbourStatus not in ['common_time','partial_field']:continue
        t=r.SessionTime; line=arrivals.get(t,float('inf'))
        other = r.DriverBehind if side == 'rear' else r.DriverAhead
        interval_driver = other if side == 'rear' else r.Driver
        rank,rank_age=get(r.Driver,'Position',t,line); rear_rank,_=get(other,'Position',t,line)
        lap,_=get(r.Driver,'NumberOfLaps',t,line); rear_lap,_=get(other,'NumberOfLaps',t,line)
        gap,age=get(interval_driver,'IntervalToPositionAhead',t,line)
        pit,_=get(r.Driver,'InPit',t,line); rear_pit,_=get(other,'InPit',t,line)
        try: eligible=int(rear_rank)==int(rank)+(1 if side=='rear' else -1) and lap is not None and lap==rear_lap and age<=3 and pit is False and rear_pit is False
        except (TypeError,ValueError):eligible=False
        try:gap=float(gap)
        except (TypeError,ValueError):gap=np.nan
        rows.append(dict(Driver=r.Driver,SessionTime=t,LapNumber=r.LapNumber,Rear=other,Side=side,
            Rank=rank,RearRank=rear_rank,Counter=lap,RearCounter=rear_lap,TimingGap=gap,TimingAge=age,
            Passage=r.RearPassageSeconds if side=='rear' else r.AheadPassageSeconds,Eligible=eligible,InPit=pit,RearInPit=rear_pit))
    frame=pd.DataFrame(rows)
    if frame.empty:return frame,{}
    comparisons=frame.loc[frame.Eligible & frame.Passage.notna() & frame.TimingGap.notna()]
    error=comparisons.Passage-comparisons.TimingGap
    threshold={}
    for value in [1.0,2.5]:
        p=comparisons.Passage.le(value); t=comparisons.TimingGap.le(value)
        threshold[str(value)]=dict(n=len(p),disagree=int((p!=t).sum()),passage_only=int((p & ~t).sum()),timing_only=int((~p & t).sum()))
    return frame,dict(eligible_position_observations=int(frame.Eligible.sum()),difference_s=stats(error),absolute_difference_s=stats(abs(error)),thresholds=threshold)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--skip-legacy',action='store_true');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases=[('r12',12,'recovered_replay.jsonl',18,(1176.146,1249.189),True),
           ('r13',13,'live_timing_raw.jsonl',8,(1101.571,1184.541),False)]
    results={}
    for name,round_number,filename,leader_lap,window,canonical in cases:
        source=args.data_root/f'wodata/live_timing/schema_v1/year=2026/round={round_number}/session=R/{filename}'
        q=args.data_root/f'wodata/live_archive/schema_v1/year=2026/round={round_number}/session=Q/raw/Position.z.json'
        ref=reference(q,window,name)
        snapshot,cut,history,timing,arrivals,elapsed=replay(source,leader_lap,canonical)
        print(name,'cut',cut,'positions',len(snapshot.position_samples),'replay seconds',elapsed,flush=True)
        # Analyse the small focal R12 window; retain preceding position history for passage.
        codes={d.driver_number:d.driver_code for d in snapshot.drivers}
        targets={'PIA','RUS','LEC'} if name=='r12' else {'PIA','NOR','RUS'}
        start=time.perf_counter()
        shadow=live_position_traffic_shadow(snapshot,reference=ref,coordinate_frame=name,cutoff=cut.leader_lap_end_session_time,targets=targets)
        shadow_seconds=time.perf_counter()-start
        samples=shadow.samples
        if name=='r12': samples=samples.loc[samples.Driver.eq('PIA') & samples.LapNumber.between(9,12)].copy()
        samples.to_csv(args.output/f'{name}_samples.csv',index=False)
        comparisons,metrics=independent(samples,timing,arrivals,codes)
        comparisons.to_csv(args.output/f'{name}_timing.csv',index=False)
        ahead_comparisons,ahead_metrics=independent(samples,timing,arrivals,codes,side='ahead')
        ahead_comparisons.to_csv(args.output/f'{name}_ahead_timing.csv',index=False)
        selected=_live_traffic_gaps(snapshot)
        result=dict(source=str(source),source_sha256=sha(source),cut=vars(cut),reference={k:v for k,v in vars(ref).items() if k!='points_m'},
            reference_vertices=len(ref.points_m),reference_length_m=float(np.linalg.norm(np.diff(ref.points_m,axis=0),axis=1).sum()),
            replay_seconds=elapsed,shadow_seconds=shadow_seconds,positions=len(snapshot.position_samples),
            position_span_s=stats([x.session_time_seconds for x in snapshot.position_samples]),
            neighbour_status=samples.NeighbourStatus.value_counts().to_dict(),rear_counts=samples.DriverBehind.value_counts().to_dict(),
            passage_status=samples.RearPassageStatus.value_counts().to_dict(),timing_comparison=metrics,
            ahead_timing_comparison=ahead_metrics,
            production_status=selected.TrafficStatus.value_counts().to_dict(),
            laps=shadow.laps.loc[lambda f:f.Driver.eq('PIA') & f.LapNumber.between(9,12)].to_dict('records') if name=='r12' else shadow.laps.to_dict('records'))
        if name=='r12':
            # Fresh empty stores; only packet arrivals after each join. No warmed laps,
            # timing, telemetry, or positions are imported from the full-prefix store.
            result['cold_starts']=[]
            for join in ['2026-08-23T13:45:00+00:00','2026-08-23T13:46:14+00:00','2026-08-23T13:48:00+00:00']:
                j=stamp(join).timestamp();cold=_analysis_store(); frontier=-np.inf
                for _,m in history:
                    if m.source_timestamp_utc:frontier=max(frontier,m.source_timestamp_utc.timestamp())
                    if frontier>=j and m.topic=='Position.z':cold.reduce(m)
                cold_snapshot=_snapshot_at_cut(cold.snapshot(),cut)
                # Static identity mapping is explicit; it contains no dynamic race state.
                cold_snapshot=replace(cold_snapshot,drivers=tuple(
                    replace(d,position=None,source_timestamp_utc=None,retired=False,retirement_lap=None)
                    for d in snapshot.drivers))
                c=live_position_traffic_shadow(cold_snapshot,reference=ref,coordinate_frame=name,cutoff=cut.leader_lap_end_session_time,targets={'PIA'})
                full=shadow.samples.loc[shadow.samples.Driver.eq('PIA')]
                merged=c.samples.merge(full,on=['Driver','SessionTime'],suffixes=('_cold','_full'))
                both=merged.RearPassageSeconds_cold.notna() & merged.RearPassageSeconds_full.notna()
                result['cold_starts'].append(dict(join=join,common_frames=len(merged),
                    neighbour_mismatch=int((merged.DriverBehind_cold.fillna('?')!=merged.DriverBehind_full.fillna('?')).sum()),
                    missing_passage_vs_full=int((merged.RearPassageSeconds_cold.isna() & merged.RearPassageSeconds_full.notna()).sum()),
                    passage_max_abs_difference=float(abs(merged.loc[both,'RearPassageSeconds_cold']-merged.loc[both,'RearPassageSeconds_full']).max())))
            telemetry=_snapshot_telemetry_frame(snapshot)
            start=time.perf_counter(); repaired=traffic.evaluate_direct_fastf1_style(telemetry,group_columns=('Driver','LapNumber'))
            repaired.samples.loc[repaired.samples.DirectDriverAhead.eq('PIA') & repaired.samples.LapNumber.between(8,13)].to_csv(args.output/'r12_direct_ownership.csv',index=False)
            result['direct_seconds']=time.perf_counter()-start
            result['direct_repaired']=repaired.laps.loc[lambda f:f.Driver.eq('PIA') & f.LapNumber.between(9,12)].to_dict('records')
            if not args.skip_legacy:
                # Frozen pre-task code, executed only by this offline reproduction.
                source_text=subprocess.check_output(['git','show','57495559a860814a96ade41f30dc74df699c9ed6:src/wostrategy/analysis/traffic.py'],text=True,cwd=Path(__file__).resolve().parents[2])
                module=types.ModuleType('part1_legacy_traffic');sys.modules[module.__name__]=module;exec(compile(source_text,'legacy_traffic.py','exec'),module.__dict__)
                legacy=module.evaluate_direct_fastf1_style(telemetry,group_columns=('Driver','LapNumber'))
                physical=module.evaluate_physical_traffic(telemetry,config=module.TrafficEvaluationConfig(_estimate_live_circuit_length(telemetry)),group_columns=('Driver','LapNumber')).laps
                combined=module.combine_lap_traffic_components(legacy.laps,physical,group_columns=('Driver','LapNumber'))
                result['legacy_source_sha256']=hashlib.sha256(source_text.encode()).hexdigest()
                result['legacy_selected']=combined.loc[lambda f:f.Driver.eq('PIA') & f.LapNumber.between(9,12)].to_dict('records')
                result['legacy_contributors']={str(lap):legacy.samples.loc[lambda f:f.DirectDriverAhead.eq('PIA') & f.LapNumber.eq(lap)].dropna(subset=['DirectTimeDeltaToDriverAhead']).Driver.value_counts().to_dict() for lap in range(9,13)}
        results[name]=result
        (args.output/'results.json').write_text(json.dumps(results,indent=2,default=str))
        print(name,'done',metrics,flush=True)
    paths={name:str(sys.modules[name].__file__) for name in ['wodata.live.reducer','woplanner.live.telemetry','woplanner.analysis.service','wostrategy.analysis.traffic','wostrategy.analysis.position_traffic']}
    (args.output/'source_identity.json').write_text(json.dumps({name:dict(path=path,sha256=sha(Path(path))) for name,path in paths.items()},indent=2))

if __name__=='__main__': main()
