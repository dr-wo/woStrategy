"""Fixed descriptive traffic calculation extracted from the completed research.

No runtime research imports. See circuit_traffic_history for public orchestration.
The extraction changes helper references only; eligibility and numerical recipe
are frozen. Timing reversals are not independently verified competitive passes.
"""
from __future__ import annotations
import itertools
import json
import re
import numpy as np
import pandas as pd

CONTRACT=dict(timing_margin_s=.2,max_start_finish_headway_s=10.,
              require_position_corroboration=True,require_next_sector_persistence=True,
              require_single_boundary_order_reversal=True,
              local_window='each car entire co-traversed lap containing timing reversal',
              weather='dry green only; wet/rainfall-unknown not in accepted sample',
              restart='first lap after status 4/5/6/7 excluded',
              compound='exclude pair if either whole-lap window has multiple candidate pairings',
              reference='unchanged existing fresh-set team-pooled ridge and four-boundary timing screen',
              reference_block='exclude every source lap overlapping union of target laps plus two maximum target lap durations on each side',
              age='StintLapNumber-1 for observed interaction lap; total TyreLife-1 retained separately',
              lap_trend='existing own-car LapNumber coordinate, not common-time track-only correction',
              blue='assigned BLUE race-control message for lapped car within union of target laps, using original stream session time',
              causal=False,independently_verified_competitive_pass=False)


OUTPUTS=[('same_lap_passer_cost','same_lap','a_cost_s','Passing car local residual'),
         ('same_lap_passed_cost','same_lap','b_cost_s','Passed car local residual'),
         ('same_lap_advantage','same_lap','advantage_s','Estimated reference advantage'),
         ('lapping_lapper_cost','lapping','a_cost_s','Lapping car local residual'),
         ('lapping_lapped_cost','lapping','b_cost_s','Lapped car local residual')]


def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [safe(x) for x in v]
    if isinstance(v,(np.integer,np.bool_)):return v.item()
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v

def finite(value):
    return value is not None and pd.notna(value) and np.isfinite(float(value))


def eligible_reasons(row):
    reasons = []
    for flag in ['Deleted', 'FastF1Generated', 'Rainfall']:
        if pd.notna(row.get(flag)) and bool(row[flag]): reasons.append(flag.lower())
    if not pd.notna(row.get('IsAccurate')) or not bool(row.IsAccurate): reasons.append('not_accurate')
    if str(row.TrackStatus) != '1': reasons.append('not_green')
    if pd.notna(row.PitInTime) or pd.notna(row.PitOutTime): reasons.append('pit')
    if row.Compound not in ['SOFT', 'MEDIUM', 'HARD']: reasons.append('not_dry_compound')
    if row.LapNumber <= 1: reasons.append('lap_one')
    return reasons

def status_flags(x):
    return eligible_reasons(x)


def crossing_screen(d,rule):
    """Four timing boundaries, nearest OTHER driver at each boundary.

    A clock separation screen is only a conservative proxy, not certified free air.
    It uses no target outcome and no physical reconstruction or vehicle dynamics.
    """
    d=d.copy()
    boundary={'START':'LapStartTime','S1':'Sector1SessionTime','S2':'Sector2SessionTime','END':'Time'}
    for name,col in boundary.items():
        table=d[['Driver',col]].copy()
        table['clock']=d[col].dt.total_seconds()
        table=table.dropna(subset=['clock']).sort_values('clock',kind='stable')
        clocks=table.clock.to_numpy(float); drivers=table.Driver.astype(str).to_numpy()
        ahead=np.full(len(clocks),np.nan); behind=np.full(len(clocks),np.nan)
        for i in range(len(clocks)):
            j=i-1
            while j>=0 and drivers[j]==drivers[i]:j-=1
            k=i+1
            while k<len(clocks) and drivers[k]==drivers[i]:k+=1
            if j>=0:ahead[i]=clocks[i]-clocks[j]
            if k<len(clocks):behind[i]=clocks[k]-clocks[i]
        d.loc[table.index,name+'_ahead_s']=ahead
        d.loc[table.index,name+'_behind_s']=behind
    d['lap_time_s']=d.LapTime.dt.total_seconds()
    sums=sum(d[f'Sector{k}Time'].dt.total_seconds() for k in (1,2,3))
    d['screen_reason']=''
    reasons=[[] for _ in range(len(d))]
    for pos,(_,row) in enumerate(d.iterrows()):
        why=status_flags(row)
        if not np.isfinite(row.lap_time_s) or abs(row.lap_time_s-sums.loc[row.name])>rule['sector_sum_tolerance_s']:why.append('lap_sector_sum')
        if not finite(row.StintLapNumber) or not finite(row.TyreLife) or not finite(row.Stint):why.append('missing_tyre_state')
        for b in boundary:
            if not finite(row[b+'_ahead_s']) or row[b+'_ahead_s']<rule['ahead_s']:why.append(b+'_ahead')
            if not finite(row[b+'_behind_s']) or row[b+'_behind_s']<rule['behind_s']:why.append(b+'_behind')
        reasons[pos]=why
    d['screen_reason']=['|'.join(x) if x else 'screened' for x in reasons]
    d['screened']=d.screen_reason.eq('screened')
    d['stint_age_zero']=pd.to_numeric(d.StintLapNumber,errors='coerce')-1
    d['total_age_zero']=pd.to_numeric(d.TyreLife,errors='coerce')-1
    d['used_offset']=pd.to_numeric(d.TyreLife,errors='coerce')-pd.to_numeric(d.StintLapNumber,errors='coerce')
    return d

REF_RULE=dict(ahead_s=2.5,behind_s=1.0,sector_sum_tolerance_s=.20,
              source='unchanged_previous_timing_screen',exclude_target_lap_radius=2,
              training_used_offset=0,minimum_team_laps=5,minimum_compound_laps=8,
              minimum_team_compound_for_direct_slope=4,minimum_distinct_ages_for_direct_slope=3,
              baseline='team',compound_reference='HARD',age_basis='StintLapNumber_minus_one',
              age_reference=0.0,race_lap_reference='median_screened_training_lap',
              scheduled_distance_input=None,fuel_rate=None,
              ridge_team=2.0,ridge_compound_delta=4.0,ridge_compound_age=16.0,
              ridge_team_compound_age=48.0,ridge_net_time=64.0,
              prior_age_s_per_lap=.10,prior_net_time_s_per_lap=-.05)


def fit_pool(source,masked_lo,masked_hi,rule):
    """Fixed ridge fit of existing team/compound/age structure on screened laps."""
    train=source.loc[source.usable_training&~source.LapNumber.between(masked_lo,masked_hi)].copy()
    if train.empty:return None
    teams=sorted(train.Team.astype(str).unique());compounds=sorted(train.Compound.astype(str).unique())
    pairs=list(zip(train.Team.astype(str),train.Compound.astype(str)))
    counts=train.groupby(['Team','Compound']).size().to_dict()
    distinct=train.groupby(['Team','Compound']).stint_age_zero.nunique().to_dict()
    direct=[p for p in sorted(set(pairs)) if counts[p]>=rule['minimum_team_compound_for_direct_slope'] and distinct[p]>=rule['minimum_distinct_ages_for_direct_slope']]
    ref_comp=rule['compound_reference'] if rule['compound_reference'] in compounds else compounds[0]
    age=train.stint_age_zero.to_numpy(float)
    race_ref=float(train.LapNumber.median())
    time=(train.LapNumber.to_numpy(float)-race_ref)/10.0
    columns=[];penalties=[];priors=[];names=[]
    for team in teams:
        columns.append(train.Team.eq(team).to_numpy(float));penalties.append(rule['ridge_team']);priors.append(0.0);names.append(('team_baseline',team))
    for comp in compounds:
        if comp==ref_comp:continue
        columns.append(train.Compound.eq(comp).to_numpy(float));penalties.append(rule['ridge_compound_delta']);priors.append(0.0);names.append(('compound_delta',comp))
    for comp in compounds:
        columns.append(train.Compound.eq(comp).to_numpy(float)*age/10);penalties.append(rule['ridge_compound_age']);priors.append(rule['prior_age_s_per_lap']*10);names.append(('compound_age',comp))
    for team,comp in direct:
        columns.append(((train.Team.eq(team))&(train.Compound.eq(comp))).to_numpy(float)*age/10)
        penalties.append(rule['ridge_team_compound_age']);priors.append(0.0);names.append(('team_compound_age',team,comp))
    columns.append(time);penalties.append(rule['ridge_net_time']);priors.append(rule['prior_net_time_s_per_lap']*10);names.append(('net_race_time',))
    X=np.column_stack(columns)
    y=train.lap_time_s.to_numpy(float);grand=float(y.mean())
    penalty=np.asarray(penalties,float);prior=np.asarray(priors,float)
    coef=np.linalg.solve(X.T@X+np.diag(penalty),X.T@(y-grand)+penalty*prior)
    out=dict(masked_lo=int(masked_lo),masked_hi=int(masked_hi),race_lap_reference=race_ref,
             age_reference=rule['age_reference'],compound_reference=ref_comp,
             grand_screened_mean_s=grand,training_laps=len(train),training_lap_min=int(train.LapNumber.min()),
             training_lap_max=int(train.LapNumber.max()),feature_names=names,coefficients=coef.tolist(),
             parameter_source='screened_event_local_team_pooled_ridge',
             training_max_time_s=float(train.t.max()),
             team_counts=train.groupby('Team').size().to_dict(),compound_counts=train.groupby('Compound').size().to_dict(),
             team_compound_counts=counts,team_compound_age_min=train.groupby(['Team','Compound']).stint_age_zero.min().to_dict(),
             team_compound_age_max=train.groupby(['Team','Compound']).stint_age_zero.max().to_dict(),
             compound_age_min=train.groupby('Compound').stint_age_zero.min().to_dict(),
             compound_age_max=train.groupby('Compound').stint_age_zero.max().to_dict(),
             own_driver_compound_counts=train.groupby(['Driver','Compound']).size().to_dict(),
             direct_team_compounds=direct)
    return out


def predict(model,team,driver,compound,target_lap,stint_age_zero,total_age_zero,rule):
    row=dict(supported=False,reason=None,team=team,driver=driver,compound=compound,
             target_lap=int(target_lap),stint_age_zero=float(stint_age_zero) if finite(stint_age_zero) else None,
             total_age_zero=float(total_age_zero) if finite(total_age_zero) else None,
             reference_level='unsupported',driver_specific_measurement=False,
             age_basis=rule['age_basis'],age_reference=rule['age_reference'],scheduled_distance_input=None)
    if model is None:row['reason']='no_screened_training_laps';return row
    row.update(masked_lo=model['masked_lo'],masked_hi=model['masked_hi'],training_laps=model['training_laps'],
               race_lap_reference=model['race_lap_reference'],compound_reference=model['compound_reference'],
               model_training_max_time_s=model['training_max_time_s'])
    if not finite(stint_age_zero) or not finite(total_age_zero):row['reason']='missing_tyre_age';return row
    if abs(float(total_age_zero)-float(stint_age_zero))>.01:
        row['reason']='used_tyre_state_unsupported';return row
    team_n=int(model['team_counts'].get(team,0));comp_n=int(model['compound_counts'].get(compound,0))
    tc=(team,compound);pair_n=int(model['team_compound_counts'].get(tc,0))
    own_n=int(model['own_driver_compound_counts'].get((driver,compound),0))
    row.update(team_training_laps=team_n,compound_training_laps=comp_n,
               same_team_compound_laps=pair_n,own_driver_compound_laps=own_n)
    if team_n<rule['minimum_team_laps']:row['reason']='insufficient_team_screened_evidence';return row
    if comp_n<rule['minimum_compound_laps']:row['reason']='insufficient_compound_screened_evidence';return row
    names=model['feature_names'];coef=np.asarray(model['coefficients'],float)
    coeff={tuple(k):v for k,v in zip(names,coef)}
    age=float(stint_age_zero)
    team_base=float(coeff.get(('team_baseline',team),0))
    delta=float(coeff.get(('compound_delta',compound),0))
    age_global=float(coeff[('compound_age',compound)])*age/10
    age_direct=float(coeff.get(('team_compound_age',team,compound),0))*age/10
    net=float(coeff[('net_race_time',)])*(target_lap-model['race_lap_reference'])/10
    direct=tc in model['direct_team_compounds']
    if pair_n==0:level='team_other_compound_plus_field_compound'
    elif direct and own_n>=4:level='team_compound_with_own_driver_contributing'
    elif direct:level='team_compound_borrowed_teammate_evidence'
    else:level='team_baseline_global_compound_age'
    if direct:
        age_min=float(model['team_compound_age_min'][tc]);age_max=float(model['team_compound_age_max'][tc])
    else:
        age_min=float(model['compound_age_min'][compound]);age_max=float(model['compound_age_max'][compound])
    extrap=not(age_min<=age<=age_max)
    row.update(supported=True,reason='supported',reference_level=level,
               uses_team_compound_age_deviation=direct,
               age_support_level='team_compound' if direct else 'global_compound',
               age_support_min=age_min,age_support_max=age_max,age_extrapolated=extrap,
               race_lap_extrapolated=not(model['training_lap_min']<=target_lap<=model['training_lap_max']),
               baseline_team_s=model['grand_screened_mean_s']+team_base,
               compound_delta_s=delta,compound_age_s=age_global,team_compound_age_s=age_direct,
               net_race_time_s=net,
               predicted_pace_s=model['grand_screened_mean_s']+team_base+delta+age_global+age_direct+net)
    return row

TIMING_MARGIN_S = CONTRACT["timing_margin_s"]

def secs(s):
    return pd.to_timedelta(s).dt.total_seconds()


def prepare_laps(df):
    d = pd.DataFrame(df).copy().sort_values(["Driver", "LapNumber"])
    for c in ["Time", "LapStartTime", "PitInTime", "PitOutTime", "Sector1SessionTime", "Sector2SessionTime"]:
        d[c + "S"] = secs(d[c])
    d["accurate"] = d.IsAccurate.eq(True) & ~d.Deleted.fillna(False).astype(bool) & ~d.FastF1Generated.fillna(True).astype(bool)
    d["dry"] = d.Compound.isin(["SOFT", "MEDIUM", "HARD"]) & d.Rainfall.eq(False)
    d["green"] = d.TrackStatus.astype(str).eq("1")
    d["neutral"] = d.TrackStatus.astype(str).str.contains("[4567]", regex=True)
    d["restart"] = d.groupby("Driver").neutral.shift().eq(True) & ~d.neutral
    d["pitlap"] = d.PitInTime.notna() | d.PitOutTime.notna()
    d["eligible"] = d.accurate & d.dry & d.green & ~d.restart & ~d.pitlap & d.LapNumber.gt(1)
    return d.reset_index(drop=True)


def timing_candidates(laps):
    """Independent of physical detector: all adjacent-lap timing reversals."""
    lookup = {(str(r.Driver), int(r.LapNumber)): r for r in laps.itertuples()}
    rows = []
    drivers = sorted(laps.Driver.unique())
    for a, b in itertools.combinations(drivers, 2):
        for n in range(2, int(laps.LapNumber.max())+1):
            entries = [lookup.get((x, k)) for k in [n-1, n] for x in [a, b]]
            if any(x is None for x in entries):
                continue
            ap, bp, ac, bc = entries
            before, after = ap.TimeS-bp.TimeS, ac.TimeS-bc.TimeS
            if not np.isfinite(before+after) or before*after >= 0:
                continue
            f, o = (a,b) if before > 0 else (b,a)
            fp, op, fc, oc = [lookup[(x,k)] for k in [n-1,n] for x in [f,o]]
            signed_before, signed_after = fp.TimeS-op.TimeS, fc.TimeS-oc.TimeS
            vals = [("previous_finish", signed_before, min(fp.TimeS,op.TimeS), max(fp.TimeS,op.TimeS))]
            for label, attr in [("S1", "Sector1SessionTimeS"), ("S2", "Sector2SessionTimeS"), ("finish", "TimeS")]:
                ft, ot = getattr(fc,attr),getattr(oc,attr)
                if np.isfinite(ft+ot): vals.append((label,ft-ot,min(ft,ot),max(ft,ot)))
            fn,on = lookup.get((f,n+1)),lookup.get((o,n+1))
            persistence = False
            next_delta = np.nan
            if fn is not None and on is not None:
                ft,ot=fn.Sector1SessionTimeS,on.Sector1SessionTimeS
                next_delta=ft-ot
                persistence = bool(np.isfinite(next_delta) and next_delta < -TIMING_MARGIN_S and not(fn.pitlap or on.pitlap) and fn.green and on.green)
            positive = [v for v in vals if v[1] > TIMING_MARGIN_S]
            negative = [v for v in vals if v[1] < -TIMING_MARGIN_S]
            sector_swap = any(v[0] in ["S1","S2"] and abs(v[1])>TIMING_MARGIN_S for v in vals)
            first_neg = next((k for k,v in enumerate(vals) if v[1] < -TIMING_MARGIN_S),None)
            lower = vals[max(first_neg-1,0)][2] if first_neg is not None else min(fp.TimeS,op.TimeS)
            upper = vals[first_neg][3] if first_neg is not None else max(fc.TimeS,oc.TimeS)
            positions = bool(fp.Position > op.Position and fc.Position < oc.Position)
            reason=[]
            if any(x.pitlap for x in [fp,op,fc,oc]):reason.append("pit_lap")
            if any(x.neutral or x.restart for x in [fp,op,fc,oc]):reason.append("neutralisation_or_restart")
            if any(not x.green for x in [fp,op,fc,oc]):reason.append("non_green")
            if any(not(x.accurate and x.dry) for x in [fp,op,fc,oc]):reason.append("quality_or_wet")
            if not positions:reason.append("position_not_corroborated")
            if signed_before<=TIMING_MARGIN_S or signed_after>=-TIMING_MARGIN_S:reason.append("small_timing_margin")
            if not sector_swap:reason.append("missing_sector_corroboration")
            if not persistence:reason.append("no_next_boundary_persistence")
            rows.append({"candidate_id":f"{f}-{o}-L{n}","follower":f,"opponent":o,"lap":n,
                "window_start_s":min(fp.TimeS,op.TimeS),"window_end_s":max(fc.TimeS,oc.TimeS),
                "cross_lower_s":lower,"cross_upper_s":upper,
                "before_delta_s":signed_before,"after_delta_s":signed_after,"next_sector_delta_s":next_delta,
                "positions_before":f"{int(fp.Position) if pd.notna(fp.Position) else '?'}>{int(op.Position) if pd.notna(op.Position) else '?'}",
                "positions_after":f"{int(fc.Position) if pd.notna(fc.Position) else '?'}<{int(oc.Position) if pd.notna(oc.Position) else '?'}",
                "timing_corroborated":not reason,"reference_label":"corroborated_on_track_order_gain" if not reason else "not_confirmed",
                "reference_reasons":";".join(reason),"boundary_evidence":json.dumps(vals)})
    return pd.DataFrame(rows)

def number(v):
    if pd.isna(v):return None
    try:return str(int(float(v)))
    except ValueError:return str(v)


def message_cars(row):
    cars=set()
    n=number(row.get('RacingNumber')); cars.update([n] if n else [])
    for match in re.finditer(r'\bCARS?\s+([0-9]+(?:\s*(?:AND|,|&)\s*[0-9]+)*)',str(row.get('Message','')).upper()):
        cars.update(re.findall(r'\d+',match.group(1)))
    return cars


def incident_messages(rc,lo,hi,car_numbers):
    if rc.empty:return []
    messages=[]
    for _,x in rc.loc[rc.session_time_s.between(lo,hi)].iterrows():
        text=str(x.get('Message','')).upper()
        if (re.search(r'COLLISION|CONTACT|SPUN|STOPPED|OFF TRACK|INCIDENT',text)
                and message_cars(x).intersection(car_numbers)):
            messages.append(str(x.get('stream_message_id','')))
    return messages


def boundary_evidence(a,b,ap,bp,an,bn):
    return [('previous_finish',ap.TimeS-bp.TimeS,min(ap.TimeS,bp.TimeS),max(ap.TimeS,bp.TimeS)),
            ('S1',a.Sector1SessionTimeS-b.Sector1SessionTimeS,min(a.Sector1SessionTimeS,b.Sector1SessionTimeS),max(a.Sector1SessionTimeS,b.Sector1SessionTimeS)),
            ('S2',a.Sector2SessionTimeS-b.Sector2SessionTimeS,min(a.Sector2SessionTimeS,b.Sector2SessionTimeS),max(a.Sector2SessionTimeS,b.Sector2SessionTimeS)),
            ('finish',a.TimeS-b.TimeS,min(a.TimeS,b.TimeS),max(a.TimeS,b.TimeS)),
            ('next_S1',an.Sector1SessionTimeS-bn.Sector1SessionTimeS,min(an.Sector1SessionTimeS,bn.Sector1SessionTimeS),max(an.Sector1SessionTimeS,bn.Sector1SessionTimeS))] if an is not None and bn is not None else []


def reasons_for_pair(a,b,ap,bp,an,bn,kind):
    reasons=[]
    if any(x is None for x in [a,b,ap,bp]):return ['missing_pair_laps'],[]
    for label,x in [('a_prev',ap),('b_prev',bp),('a',a),('b',b)]:
        if x.pitlap:reasons.append(label+':pit')
        if not x.green:reasons.append(label+':non_green')
        if x.restart:reasons.append(label+':restart')
        if not x.accurate:reasons.append(label+':inaccurate')
        if not x.dry:reasons.append(label+':wet_or_weather_unknown')
        if x.LapNumber<=1:reasons.append(label+':first_lap')
    if kind=='same_lap':
        if not(ap.Position>bp.Position and a.Position<b.Position):reasons.append('position_not_reversed')
    else:
        if not(ap.Position<bp.Position and a.Position<b.Position):reasons.append('leader_position_not_corroborated')
    margin=CONTRACT['timing_margin_s'];maximum=CONTRACT['max_start_finish_headway_s']
    before=ap.TimeS-bp.TimeS;after=a.TimeS-b.TimeS
    if not(margin<before<=maximum and -maximum<=after<-margin):reasons.append('start_finish_headway_or_margin')
    if an is None or bn is None:reasons.append('missing_next_sector_persistence');return reasons,[]
    if not(an.green and bn.green and not(an.pitlap or bn.pitlap) and an.accurate and bn.accurate):
        reasons.append('next_sector_quality_or_pit')
    vals=boundary_evidence(a,b,ap,bp,an,bn)
    if any(not np.isfinite(v[1]) for v in vals):reasons.append('missing_sector_clock')
    signs=[1 if v[1]>margin else -1 if v[1]<-margin else 0 for v in vals]
    nonzero=[s for s in signs if s]
    if not nonzero or nonzero[0]!=1 or nonzero[-1]!=-1:reasons.append('no_persistent_boundary_reversal')
    if sum(x!=y for x,y in zip(nonzero,nonzero[1:]))!=1:reasons.append('pass_repass_or_boundary_ambiguity')
    if vals[-1][1]>=-margin:reasons.append('no_next_sector_persistence')
    return reasons,vals


def pairings(laps,rc,year,rnd):
    candidates=timing_candidates(laps)
    lookup={(x.Driver,int(x.LapNumber)):x for x in laps.itertuples()}
    seeds=[]
    for x in candidates.itertuples():
        seeds.append(dict(kind='same_lap',a_driver=x.follower,b_driver=x.opponent,
                          a_lap=int(x.lap),b_lap=int(x.lap),
                          legacy_timing_corroborated=bool(x.timing_corroborated),
                          legacy_reason=x.reference_reasons))
    # Co-traversed lapping laps are matched by start-clock proximity, never equal lap number.
    for ad,bd in itertools.permutations(sorted(laps.Driver.unique()),2):
        aa=laps.loc[laps.Driver.eq(ad)].dropna(subset=['LapStartTimeS']).sort_values('LapStartTimeS')
        bb=laps.loc[laps.Driver.eq(bd)].dropna(subset=['LapStartTimeS']).sort_values('LapStartTimeS')
        aligned=pd.merge_asof(aa[['LapStartTimeS','LapNumber']],
            bb[['LapStartTimeS','LapNumber']].rename(columns={'LapStartTimeS':'b_start_s','LapNumber':'b_lap'}),
            left_on='LapStartTimeS',right_on='b_start_s',direction='backward',
            tolerance=CONTRACT['max_start_finish_headway_s'])
        for x in aligned.dropna(subset=['b_lap']).itertuples():
            na,nb=int(x.LapNumber),int(x.b_lap)
            if na<=nb:continue
            a,b=lookup.get((ad,na)),lookup.get((bd,nb))
            if a is None or b is None or not(np.isfinite(a.TimeS+b.TimeS) and a.TimeS<b.TimeS):continue
            seeds.append(dict(kind='lapping',a_driver=ad,b_driver=bd,a_lap=na,b_lap=nb,
                              legacy_timing_corroborated=None,legacy_reason='session_time_co_traversal'))
    rows=[]
    for seed in seeds:
        ad,bd,na,nb=seed['a_driver'],seed['b_driver'],seed['a_lap'],seed['b_lap']
        a,b=lookup.get((ad,na)),lookup.get((bd,nb))
        ap,bp=lookup.get((ad,na-1)),lookup.get((bd,nb-1))
        an,bn=lookup.get((ad,na+1)),lookup.get((bd,nb+1))
        reasons,vals=reasons_for_pair(a,b,ap,bp,an,bn,seed['kind'])
        if seed['kind']=='same_lap' and not seed['legacy_timing_corroborated']:
            reasons.append('legacy_timing_corroboration_failed:'+str(seed['legacy_reason']))
        if a is None or b is None:continue
        lo=min(a.LapStartTimeS,b.LapStartTimeS);hi=max(a.TimeS,b.TimeS)
        incidents=incident_messages(rc,lo,hi,{number(a.DriverNumber),number(b.DriverNumber)})
        if incidents:reasons.append('assigned_incident_message_in_local_window')
        blues=[];unassigned=[]
        if seed['kind']=='lapping' and not rc.empty:
            for _,msg in rc.loc[rc.session_time_s.between(lo,hi)&rc.get('Flag',pd.Series('',index=rc.index)).eq('BLUE')].iterrows():
                if number(msg.get('RacingNumber'))==number(b.DriverNumber):blues.append(str(msg.stream_message_id))
                elif not number(msg.get('RacingNumber')):unassigned.append(str(msg.stream_message_id))
        row=dict(seed,year=year,round=rnd,
                 interaction_id=f'{year}_R{rnd}:{seed["kind"]}:{ad}-{bd}:L{na}-{nb}',
                 identifiable=not reasons,identification_reason='timing_corroborated_on_track_order_reversal' if not reasons else '|'.join(sorted(set(reasons))),
                 independently_verified_competitive_pass=False,
                 lap_offset=na-nb,local_start_s=lo,local_end_s=hi,
                 a_start_s=a.LapStartTimeS,a_end_s=a.TimeS,b_start_s=b.LapStartTimeS,b_end_s=b.TimeS,
                 boundary_evidence=json.dumps(safe(vals)),incident_message_ids='|'.join(incidents),
                 evidenced_blue_flag=bool(blues),blue_message_ids='|'.join(blues),
                 blue_evidence_source='assigned_BLUE_stream_message_in_local_window' if blues else
                     ('inferred_lapping_no_assigned_BLUE_match' if not rc.empty else 'inferred_lapping_RC_unavailable'),
                 unassigned_blue_message_ids='|'.join(unassigned),race_control_available=not rc.empty)
        for p,x in [('a',a),('b',b)]:
            for field in ['Team','Compound','Stint','StintLapNumber','TyreLife','FreshTyre','DriverNumber','Position','TrackStatus','LapTime']:
                v=getattr(x,field);row[p+'_'+field]=v.total_seconds() if field=='LapTime' and pd.notna(v) else v
            row[p+'_rainfall']=bool(x.Rainfall) if pd.notna(x.Rainfall) else None
        rows.append(row)
    return pd.DataFrame(rows),lookup


def prepare_reference(raw,prepared,rc):
    d=crossing_screen(raw,REF_RULE)
    extra={(x.Driver,int(x.LapNumber)):(x.restart,x.dry) for x in prepared.itertuples()}
    d['restart_buffer']=[extra[(x.Driver,int(x.LapNumber))][0] for x in d.itertuples()]
    d['confirmed_dry']=[extra[(x.Driver,int(x.LapNumber))][1] for x in d.itertuples()]
    d['incident_message']=[bool(incident_messages(rc,x.LapStartTime.total_seconds(),x.Time.total_seconds(),
                            {number(x.DriverNumber)})) if pd.notna(x.LapStartTime) and pd.notna(x.Time) else True
                            for x in d.itertuples()]
    d['source_quality']=d.screened & ~d.restart_buffer & d.confirmed_dry & ~d.incident_message
    d['fresh_coordinate_supported']=d.used_offset.abs().lt(.01)&d.FreshTyre.eq(True)
    d['usable_training']=d.source_quality & d.fresh_coordinate_supported
    d['t']=d.Time.dt.total_seconds()
    return d


def reference_for(source,a,b):
    duration=max(a.TimeS-a.LapStartTimeS,b.TimeS-b.LapStartTimeS)
    lo=min(a.LapStartTimeS,b.LapStartTimeS)-2*duration
    hi=max(a.TimeS,b.TimeS)+2*duration
    masked=source.copy()
    overlaps=masked.LapStartTime.dt.total_seconds().le(hi)&masked.Time.dt.total_seconds().ge(lo)
    masked.loc[overlaps,'usable_training']=False
    model=fit_pool(masked,0,0,REF_RULE)
    result=dict(reference_mask_start_s=lo,reference_mask_end_s=hi,
                reference_mask_kind='session_time_union_with_two_lap_buffer',
                excluded_source_laps=int((source.usable_training & overlaps).sum()),
                reference_training_lap_ids='|'.join(f'{x.Driver}:L{int(x.LapNumber)}' for x in
                    source.loc[source.usable_training & ~overlaps].itertuples()),
                reference_coefficients=json.dumps(safe(dict(names=model['feature_names'],values=model['coefficients']))) if model else None)
    for p,x in [('a',a),('b',b)]:
        got=predict(model,x.Team,x.Driver,x.Compound,int(x.LapNumber),
                           x.StintLapNumber-1,x.TyreLife-1,REF_RULE)
        if got['supported']:
            age=got['stint_age_zero'];lower=got['age_support_min'];upper=got['age_support_max']
            got['age_distance_beyond_bounds_laps']=max(lower-age,age-upper,0)
        got['future_reference_evidence']=bool(model and model['training_max_time_s']>max(a.TimeS,b.TimeS))
        result.update({p+'_ref_'+k:v for k,v in got.items()})
    return result

def analyse_session(raw, rc, *, year, round_number):
    """Return all candidate pairs, screened sources and lap coverage for one race."""
    raw = pd.DataFrame(raw).copy().reset_index(drop=True)
    all_pairs=[]; screen_rows=[]; laps_summary=[]
    prepared=prepare_laps(raw)
    pairs,lookup=pairings(prepared,rc,year,round_number)
    source=prepare_reference(raw,prepared,rc)
    screen_rows.append(source[['Driver','LapNumber','Team','Compound','StintLapNumber','TyreLife','used_offset',
                               'screened','screen_reason','restart_buffer','confirmed_dry','incident_message',
                               'source_quality','usable_training']].assign(year=year,round=round_number))
    if not pairs.empty:
        # Include ambiguous/pit reversals in the collision ledger: never repeatedly allocate a lap.
        counts={}
        for x in pairs.itertuples():
            for key in [(x.a_driver,x.a_lap),(x.b_driver,x.b_lap)]:counts[key]=counts.get(key,0)+1
        rows=[]
        for _,x in pairs.iterrows():
            rec=x.to_dict()
            a=lookup[(x.a_driver,int(x.a_lap))];b=lookup[(x.b_driver,int(x.b_lap))]
            rec['a_candidate_pairings_in_lap']=counts[(x.a_driver,x.a_lap)]
            rec['b_candidate_pairings_in_lap']=counts[(x.b_driver,x.b_lap)]
            rec['unresolved_compound_interaction']=bool(max(rec['a_candidate_pairings_in_lap'],rec['b_candidate_pairings_in_lap'])>1)
            if x.identifiable:rec.update(reference_for(source,a,b))
            for p,car in [('a',a),('b',b)]:
                why=[]
                if not x.identifiable:why.append('not_identifiable:'+x.identification_reason)
                if rec['unresolved_compound_interaction']:why.append('unresolved_compound_interaction')
                if x.identifiable and not rec.get(p+'_ref_supported',False):why.append('reference:'+str(rec.get(p+'_ref_reason')))
                observed=car.LapTime.total_seconds() if pd.notna(car.LapTime) else np.nan
                if not np.isfinite(observed) or abs(observed-(car.TimeS-car.LapStartTimeS))>.25:
                    why.append('whole_lap_clock_mismatch')
                rec[p+'_observed_window_s']=observed
                rec[p+'_cost_usable']=not why
                rec[p+'_cost_exclusion']='|'.join(why)
                rec[p+'_cost_s']=observed-rec[p+'_ref_predicted_pace_s'] if not why else None
            why=[]
            if not x.identifiable:why.append('not_identifiable')
            if rec['unresolved_compound_interaction']:why.append('unresolved_compound_interaction')
            if x.identifiable and not(rec.get('a_ref_supported',False) and rec.get('b_ref_supported',False)):
                why.append('one_or_both_references_missing')
            rec['advantage_usable']=not why;rec['advantage_exclusion']='|'.join(why)
            rec['advantage_s']=rec['b_ref_predicted_pace_s']-rec['a_ref_predicted_pace_s'] if not why else None
            rows.append(rec)
        all_pairs.append(pd.DataFrame(rows))
    laps_summary.append(dict(year=year,round=round_number,laps=len(raw),
                             dry_green_nonpit_accurate_laps=int(prepared.eligible.sum()),
                             non_green_laps=int((~prepared.green).sum()),wet_or_unknown_laps=int((~prepared.dry).sum()),
                             pit_laps=int(prepared.pitlap.sum()),restart_buffer_laps=int(prepared.restart.sum()),
                             screened_training_laps=int(source.usable_training.sum()),
                             race_control_messages=len(rc),blue_messages=int(rc.get('Flag',pd.Series(dtype=str)).eq('BLUE').sum())))

    ledger = pd.concat(all_pairs,ignore_index=True) if all_pairs else pd.DataFrame(columns=["kind","year","identifiable","interaction_id","a_cost_s","b_cost_s","advantage_s"])
    return ledger, pd.concat(screen_rows,ignore_index=True), laps_summary[0]
