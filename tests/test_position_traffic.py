from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from wostrategy.analysis.position_traffic import PositionReference, position_traffic_shadow
from wostrategy.analysis.traffic import (
    _attach_direct_rear_ownership, _summarize_direct_samples, DirectTrafficEvaluationConfig,
    traffic_clean_mask,
)

REF = PositionReference(((10, 10), (110, 10), (110, 110), (10, 110), (10, 10)),
                        "test", "previous-session", "abc", -10, -5, "preloaded", 1)

def point(phase):
    phase %= 400
    if phase < 100: return 10 + phase, 10
    if phase < 200: return 110, 10 + phase - 100
    if phase < 300: return 110 - (phase - 200), 110
    return 10, 110 - (phase - 300)

def data():
    rows = []
    for t in np.arange(0, 5.01, .25):
        for driver, offset in [('A', 390), ('B', 380), ('C', 20)]:
            x, y = point(offset + t * 20)
            rows.append(dict(Driver=driver, SessionTime=t, X=x, Y=y, Z=1, OnTrack=True))
    return pd.DataFrame(rows)

def run(frame=None, **kwargs):
    return position_traffic_shadow(data() if frame is None else frame, reference=kwargs.pop('reference', REF),
        coordinate_frame="test", cutoff=kwargs.pop('cutoff', 5), expected_drivers={'A','B','C'}, targets={'A'}, **kwargs)

def test_wrap_lapped_candidates_and_past_passage():
    result = run()
    assert result.samples.DriverBehind.eq('B').all()
    assert result.samples.DriverAhead.eq('C').all()
    assert np.allclose(result.samples.RearPassageSeconds.dropna(), .5)
    assert result.samples.iloc[0].RearPassageStatus == 'insufficient_passage_history'
    assert result.samples.TrafficStatus.eq('unknown').all()

@pytest.mark.parametrize('kind', ['missing','stale','invalid','placeholder','offroute','unknown_status','duplicate'])
def test_bad_field_never_becomes_clear_track(kind):
    frame=data(); mask=frame.Driver.eq('B') & frame.SessionTime.eq(5)
    if kind in ['missing','stale']: frame=frame.loc[~mask]
    elif kind=='invalid': frame.loc[mask,'X']=np.nan
    elif kind=='placeholder': frame.loc[mask,['X','Y','Z']]=0
    elif kind=='offroute': frame.loc[mask,'X']=1000
    elif kind=='unknown_status':
        frame['OnTrack']=frame.OnTrack.astype(object)
        frame.loc[mask,'OnTrack']=None
    elif kind=='duplicate': frame=pd.concat([frame,frame.loc[mask]],ignore_index=True)
    assert not run(frame).samples.iloc[-1].FieldComplete
    assert run(frame).samples.iloc[-1].TrafficStatus == 'unknown'

def test_pit_excluded_but_shadow_does_not_claim_pit_route_calibration():
    frame=data();frame['InPit']=frame.Driver.eq('B')
    r=run(frame).samples.iloc[-1]
    assert r.DriverBehind=='C'
    assert r.PitContext=='explicit'
    assert r.TrafficStatus=='unknown'

def test_no_future_reference_or_frame_mismatch():
    assert run(reference=None).status=='unavailable'
    assert run(reference=replace(REF,available_from=6)).reason=='reference_not_available_at_cut'
    assert run(reference=replace(REF,coordinate_frame='other')).reason=='reference_frame_mismatch'

def test_cold_start_loses_history_then_agrees_without_warmed_state():
    full=run().samples
    cold=run(data().loc[lambda f:f.SessionTime.ge(2)]).samples
    assert np.isnan(cold.iloc[0].RearPassageSeconds)
    assert np.allclose(cold.loc[cold.SessionTime.ge(2.5),'RearPassageSeconds'],
                       full.loc[full.SessionTime.ge(2.5),'RearPassageSeconds'])

def test_batch_order_future_membership_and_target_boundaries():
    windows=pd.DataFrame([dict(Driver='A',LapNumber=9,Start=0,End=2,Basis='publication'),
                          dict(Driver='A',LapNumber=10,Start=2,End=5,Basis='publication')])
    a=run(lap_windows=windows).samples
    b=run(data().iloc[::-1],lap_windows=windows).samples
    pd.testing.assert_frame_equal(a,b)
    assert np.isnan(a.iloc[0].LapNumber)
    assert a.loc[a.SessionTime.eq(2),'LapNumber'].item()==9
    assert a.loc[a.SessionTime.eq(2.25),'LapNumber'].item()==10
    assert run(cutoff=2).samples.SessionTime.max()==2

def test_speed_jump_rejects_passage_without_fabricating_uncertainty():
    frame=data().loc[lambda f:f.SessionTime.isin([0,.25])].copy()
    frame.loc[frame.Driver.eq('A') & frame.SessionTime.eq(.25),['X','Y']]=point(440)
    frame.loc[frame.Driver.eq('B') & frame.SessionTime.eq(.25),['X','Y']]=point(435)
    result=run(frame).samples.iloc[-1]
    assert result.NeighbourStatus=='common_time'
    assert result.RearPassageStatus=='invalid_passage_bracket'

def test_direct_rear_uses_target_label_and_rejects_ambiguous_boundary():
    telemetry=pd.DataFrame(dict(Driver=['A']*4,LapNumber=[9,9,10,10],_SessionSeconds=[0,.5,1,1.5]))
    samples=pd.DataFrame(dict(Driver=['B']*4,LapNumber=[8]*4,SessionTime=[.25,.75,1,2],
        DirectDriverAhead=['A']*4,DirectTimeDeltaToDriverAhead=[.5]*4,DirectDistanceToDriverAhead=[10]*4))
    cfg=DirectTrafficEvaluationConfig()
    r=_attach_direct_rear_ownership(samples,telemetry,config=cfg)
    assert r.DirectRearTargetLapNumber.iloc[0]==9
    assert np.isnan(r.DirectRearTargetLapNumber.iloc[1])
    assert r.DirectRearTargetLapNumber.iloc[2]==10
    assert np.isnan(r.DirectRearTargetLapNumber.iloc[3])
    assert r.DirectRearSampleTime.equals(samples.SessionTime)


def test_rear_summary_counts_observations_on_target_laps_only():
    config=DirectTrafficEvaluationConfig()
    telemetry=pd.DataFrame(dict(Driver=['A']*4,LapNumber=[9,9,10,10],_SessionSeconds=[0,.5,1,1.5]))
    samples=pd.DataFrame([
        dict(Driver='A',LapNumber=lap,SessionTime=t,DirectDriverAhead=None,
             DirectTimeDeltaToDriverAhead=np.nan,DirectDistanceToDriverAhead=np.nan)
        for lap,t in [(9,.25),(10,1.25)]
    ]+[
        dict(Driver='B',LapNumber=8,SessionTime=t,DirectDriverAhead='A',
             DirectTimeDeltaToDriverAhead=gap,DirectDistanceToDriverAhead=10)
        for t,gap in [(.25,.5),(.75,99),(1.25,.8)]
    ])
    rows=_attach_direct_rear_ownership(samples,telemetry,config=config)
    summary=_summarize_direct_samples(rows,groups=('Driver','LapNumber'),config=config).set_index(['Driver','LapNumber'])
    assert summary.loc[('A',9),'MeanTimeDeltaToDriverBehind']==.5
    assert summary.loc[('A',10),'MeanTimeDeltaToDriverBehind']==.8
    assert ('A',8) not in summary.index


def test_reducer_malformed_status_cannot_certify_complete_field():
    from datetime import datetime, timezone
    from wodata.live import LiveSessionStore, LiveTimingMessage
    now=datetime(2026,1,1,tzinfo=timezone.utc)
    message=LiveTimingMessage('Position.z', {'Position':[{'Timestamp':now.isoformat(),
        'Entries':{d:dict(X=x,Y=10,Z=1,Status=status) for d,x,status in
                   [('A',50,'OnTrack'),('B',40,'garbage'),('C',70,'OnTrack')]}}]},now,now,
        session_time_seconds=0)
    snapshot=LiveSessionStore().apply(message)
    frame=pd.DataFrame([dict(Driver=s.driver_number,SessionTime=s.session_time_seconds,
        X=s.x,Y=s.y,Z=s.z,OnTrack=s.on_track) for s in snapshot.position_samples])
    result=run(frame).samples.iloc[0]
    assert not result.FieldComplete
    assert result.UnavailableDrivers=='B'
    assert result.TrafficStatus=='unknown'
