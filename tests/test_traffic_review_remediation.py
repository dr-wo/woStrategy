import numpy as np
import pandas as pd
import pytest
from wostrategy.analysis.traffic import (
    DirectTrafficEvaluationConfig, _attach_direct_rear_ownership,
    evaluate_direct_fastf1_style, summarize_lap_gap_metrics, traffic_clean_mask,
    TRAFFIC_EVALUATOR_VERSION,
)


def legacy_frame():
    return pd.DataFrame([
        dict(Driver=d,DriverNumber=n,LapNumber=lap,SessionTime=t,DriverAhead=a,
             TimeDeltaToDriverAhead=gap,DistanceToDriverAhead=gap*50)
        for t in [0.,.25,.5] for d,n,lap,a,gap in [('A','1',10,'3',5.),('B','2',9,'1',.5)]
    ])


def test_review_legacy_target_lap_10_follower_lap_9():
    rows=summarize_lap_gap_metrics(legacy_frame(),group_columns=('Driver','LapNumber'))
    row=rows[rows.Driver.eq('A') & rows.LapNumber.eq(10)]
    assert row.MeanTimeDeltaToDriverBehind.item()==.5
    assert not traffic_clean_mask(row,minimum_ahead_seconds=2.5,minimum_behind_seconds=1.).item()


@pytest.mark.parametrize('missing', ['SessionTime','DriverAhead','DriverNumber'])
def test_review_legacy_unsupported_ownership_cannot_be_clean(missing):
    rows=summarize_lap_gap_metrics(legacy_frame().drop(columns=missing),group_columns=('Driver','LapNumber'))
    row=rows[rows.Driver.eq('A')]
    assert row.TrafficStatus.item()=='unknown'
    assert not traffic_clean_mask(row,minimum_ahead_seconds=2.5,minimum_behind_seconds=1.).item()


def test_review_conflicting_exact_labels_order_independent():
    telemetry=pd.DataFrame(dict(Driver=['A','A','A','A'],LapNumber=[10,10,11,11],
                                _SessionSeconds=[0.,.25,.25,.5]))
    samples=pd.DataFrame(dict(Driver=['B'],LapNumber=[9],SessionTime=[.25],DirectDriverAhead=['A']))
    for trace in [telemetry,telemetry.iloc[::-1]]:
        row=_attach_direct_rear_ownership(samples,trace,config=DirectTrafficEvaluationConfig()).iloc[0]
        assert pd.isna(row.DirectRearTargetLapNumber)
    same=telemetry.assign(LapNumber=10)
    row=_attach_direct_rear_ownership(samples,same,config=DirectTrafficEvaluationConfig()).iloc[0]
    assert row.DirectRearTargetLapNumber==10


def test_review_extra_year_with_explicit_group_columns():
    frame=pd.DataFrame([dict(Year=2026,Driver=d,LapNumber=lap,SessionTime=start+i*.25,Speed=180.)
        for d,lap,start in [('A',10,0.),('B',9,.5)] for i in range(20)])
    result=evaluate_direct_fastf1_style(frame,group_columns=('Driver','LapNumber'))
    assert not result.laps.empty
    assert 'Year' not in result.laps


def test_review_old_ownership_version_is_not_current():
    assert TRAFFIC_EVALUATOR_VERSION != 'unified-traffic-v3'


def test_legacy_fallback_retains_target_ownership_when_shared_cannot_run():
    frame=legacy_frame().loc[lambda f:f.SessionTime.lt(.5)].assign(Speed=180.)
    rows=summarize_lap_gap_metrics(frame,group_columns=('Driver','LapNumber'))
    target=rows[rows.Driver.eq('A')].iloc[0]
    assert target.SharedDirectState=='UNAVAILABLE'
    assert target.MeanTimeDeltaToDriverBehind==.5


def test_conflicts_survive_progress_preparation_and_input_reversal():
    frame=pd.DataFrame([dict(Driver=d,LapNumber=lap,SessionTime=start+i*.25,Speed=180.)
        for d,lap,start in [('A',10,0.),('B',9,.5)] for i in range(20)])
    conflict=frame[frame.Driver.eq('A') & frame.SessionTime.eq(1.)].assign(LapNumber=11)
    frame=pd.concat([frame,conflict],ignore_index=True)
    results=[]
    for source in [frame,frame.iloc[::-1]]:
        result=evaluate_direct_fastf1_style(source,group_columns=('Driver','LapNumber'))
        row=result.samples.loc[lambda f:f.Driver.eq('B') & f.SessionTime.eq(1.)].iloc[0]
        assert row.DirectRearTargetDriver=='A'
        assert pd.isna(row.DirectRearTargetLapNumber)
        results.append(result.samples.sort_values(['Driver','SessionTime','LapNumber']).reset_index(drop=True))
    pd.testing.assert_frame_equal(*results)


def test_internal_sessions_are_separate_when_output_groups_omit_year():
    frame=pd.DataFrame([dict(Year=year,Driver=d,LapNumber=lap+offset,
                           SessionTime=start+i*.25,Speed=180.)
        for year,offset in [(2025,0),(2026,10)]
        for d,lap,start in [('A',10,0.),('B',9,.5)] for i in range(20)])
    result=evaluate_direct_fastf1_style(frame,group_columns=('Driver','LapNumber'))
    for lap,owned in [(9,10),(19,20)]:
        rows=result.samples.loc[lambda f:f.Driver.eq('B') & f.LapNumber.eq(lap)]
        assert set(rows.DirectRearTargetLapNumber.dropna())=={owned}
