import copy

import pytest
from wostrategy.analysis.circuit_traffic_history import summarize_records, prepare_history
from wodata.circuit_traffic import read_circuit_traffic, write_circuit_traffic


def record(year, value, suffix='', **changes):
    row=dict(year=year,kind='same_lap',interaction_id=f'{year}:{value}:{suffix}',
        identifiable=True,unresolved_compound_interaction=False,advantage_usable=True,
        a_cost_usable=True,b_cost_usable=True,a_cost_s=value,b_cost_s=value+1 if value is not None else None,advantage_s=value,
        a_ref_age_extrapolated=False,b_ref_age_extrapolated=False,evidenced_blue_flag=False)
    return {**row,**changes}


def result(rows=None, target=2026):
    rows=rows if rows is not None else [record(2018,1),record(2022,-.8),record(2022,.2),record(2023,.4)]
    return summarize_records(circuit='Monza',target_season=target,records=rows,
        coverage=[dict(year=y,status='available') for y in sorted({r['year'] for r in rows})])


def output(value, group='all', name='same_lap_advantage'):
    return next(o for g in value['groups'] if g['group']==group for o in g['outputs'] if o['output']==name)


def test_partition_window_missingness_and_pooled_record_statistics():
    value=result()
    a,b,all_group=value['groups']
    assert a['years']==[2022,2023,2024,2025]
    assert b['years']==[2018,2019,2020,2021]
    assert set(a['years']).isdisjoint(b['years'])
    assert sorted(a['years']+b['years'])==all_group['years']==list(range(2018,2026))
    assert value['available_years']==[2018,2022,2023]
    assert value['unavailable_years']==[2019,2020,2021,2024,2025]
    assert value['status']=='partial'
    o=output(value)
    assert o['usable_N']==4
    assert o['mean_s']==pytest.approx(.2)  # not mean of annual means
    assert o['observed_successful_pass_minimum_advantage_s']==-.8
    assert o['observed_successful_pass_median_advantage_s']==pytest.approx(.3)
    assert o['minimum_source_interaction_ids']==['2022:-0.8:']
    assert output(value,'2022_2025')['observed_successful_pass_median_advantage_s']==.2
    for idx in range(5):
        assert all_group['outputs'][idx]['histogram']['counts']==[
            x+y for x,y in zip(a['outputs'][idx]['histogram']['counts'],b['outputs'][idx]['histogram']['counts'])]


@pytest.mark.parametrize('year',[2017,2026,2027])
def test_outside_window_and_target_rejected(year):
    with pytest.raises(ValueError,match='outside'):
        result([record(year,0)])


def test_empty_group_missing_reference_and_compound_are_not_zero():
    value=result([],target=2034)
    assert value['groups'][0]['years']==[]
    assert value['status']=='unavailable'
    o=output(value)
    assert o['usable_N']==0 and o['mean_s'] is None
    assert o['observed_successful_pass_minimum_advantage_s'] is None
    assert o['observed_successful_pass_median_advantage_s'] is None
    rows=[record(2022,None,a_cost_s=None,b_cost_s=None,advantage_usable=False,a_cost_usable=False,b_cost_usable=False),
          record(2022,-100,identifiable=False),record(2022,-200,unresolved_compound_interaction=True),record(2022,-.1)]
    value=result(rows)
    assert output(value)['usable_N']==1
    assert output(value)['observed_successful_pass_minimum_advantage_s']==-.1
    assert len(value['interactions'])==4


def test_tied_minimum_trace_and_exact_even_median():
    value=result([record(2021,-1,'a'),record(2021,-1,'b'),record(2022,2),record(2023,4)])
    assert output(value)['observed_successful_pass_median_advantage_s']==.5
    assert len(output(value)['minimum_source_interaction_ids'])==2


def test_round_trip_reuse_and_stale_recipe(tmp_path):
    value=result()
    path=write_circuit_traffic(value,data_root=tmp_path)
    got,where=read_circuit_traffic(circuit='monza',target_season=2026,data_root=tmp_path,expected_recipe_id=value['recipe_id'])
    assert where==path
    assert got=={**value,'schema_version':'schema_v1'}
    def forbidden(year):raise AssertionError('cached analysis must not load or refit')
    again,_,reused=prepare_history(circuit='Monza',target_season=2026,data_root=tmp_path,source_loader=forbidden)
    assert reused and again==got
    with pytest.raises(ValueError,match='stale'):
        read_circuit_traffic(circuit='Monza',target_season=2026,data_root=tmp_path,expected_recipe_id='wrong')
