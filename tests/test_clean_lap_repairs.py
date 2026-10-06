import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from wostrategy.analysis.clean_lap_repairs import REPAIR_DATA, persist_clean_laps, repair_clean_laps


def fixture_frame():
    repair = json.loads(REPAIR_DATA.read_text())['repairs'][0]
    rows = repair['corrections'] + repair['unresolved']
    return pd.DataFrame([dict(Driver=r['driver'], DriverNumber=int(r['driver_number']),
        LapNumber=r['lap_number'], Stint=r['stint'], Compound=r['compound'], TyreLife=np.nan,
        Year=2026, Round=16, SessionName='R', EventName='Bahrain Grand Prix',
        EventCountry='Bahrain', TyreAgeLaps=99., LapTimeSeconds=90.12345678901234,
        FreshTyre=False) for r in rows])


def test_reviewed_physical_ages_and_unresolved():
    before = fixture_frame()
    after, audit = repair_clean_laps(before, year=2026, race=16, session='R')
    assert before.TyreLife.isna().all()
    assert audit['applied_rows'] == 14
    assert after.loc[after.Driver.eq('ALB'), 'TyreLife'].tolist() == list(range(4, 10))
    assert after.loc[after.Driver.eq('COL'), 'TyreLife'].tolist() == list(range(7, 11))
    assert after.loc[after.Driver.eq('LAW'), 'TyreLife'].tolist() == list(range(2, 6))
    assert after.loc[after.Driver.eq('VER'), 'TyreLife'].isna().all()
    assert_frame_equal(before.drop(columns='TyreLife'), after.drop(columns='TyreLife'), check_exact=True)
    for item in audit['repairs'][0]['rows']:
        if item['status'] == 'applied':
            assert item['tyre_life'] == item['prior_tyre_laps'] + item['lap_number'] - item['first_replacement_lap'] + 1
            assert item['sources'] and item['derivation']
    for _, stint in after.dropna(subset=['TyreLife']).groupby(['Driver', 'Stint']):
        assert (stint.TyreLife.diff().dropna() == stint.LapNumber.diff().dropna()).all()
    again, repeated = repair_clean_laps(after, year=2026, race=16, session='R')
    assert repeated['applied_rows'] == 0
    assert_frame_equal(after, again, check_exact=True)


def test_null_only_and_identity_guards():
    frame = fixture_frame()
    frame.loc[0, 'TyreLife'] = 999.
    frame.loc[1, 'EventName'] = 'Different event'
    frame.loc[2, 'DriverNumber'] = 999
    frame.loc[3, 'Stint'] = 99
    frame.loc[4, 'Compound'] = 'HARD'
    frame.loc[5, 'SessionName'] = 'Qualifying'
    after, _ = repair_clean_laps(frame, year=2026, race=16, session='R')
    assert after.loc[0, 'TyreLife'] == 999.
    assert after.loc[1:5, 'TyreLife'].isna().all()
    unrelated, audit = repair_clean_laps(frame, year=2026, race=15, session='R')
    assert_frame_equal(frame, unrelated, check_exact=True)
    assert audit['repairs'] == []
    frame = pd.concat([frame, frame.iloc[[6]]], ignore_index=True)
    after, audit = repair_clean_laps(frame, year=2026, race=16, session='R')
    assert after.loc[frame.Driver.eq('COL') & frame.LapNumber.eq(38), 'TyreLife'].isna().all()


def test_persisted_provenance_precision_and_cached_idempotence(tmp_path):
    frame = fixture_frame()
    path = tmp_path / 'clean.csv'
    paths = persist_clean_laps(frame, path, year=2026, race=16, session='R')
    saved = pd.read_csv(path, float_precision='round_trip')
    assert_frame_equal(frame.drop(columns='TyreLife'), saved.drop(columns='TyreLife'), check_exact=True)
    audit = json.loads(paths[0].read_text())
    assert audit['applied_rows'] == 14
    assert audit['repair_data_sha256']
    assert audit['repairs'][0]['sources']['pirelli_pitstop']['sha256']
    original = path.read_bytes(), paths[0].read_bytes()
    persist_clean_laps(saved, path, year=2026, race=16, session='R', cached=True)
    assert original == (path.read_bytes(), paths[0].read_bytes())
    assert frame.TyreLife.isna().all()


def test_cached_normal_producer_materializes_only_clean_life(tmp_path):
    from wostrategy.script.race_performance_review import load_cached_monte_carlo_outputs
    prefix = 'race_performance_2026_16_R'
    clean = tmp_path / f'{prefix}_clean_laps.csv'
    fixture_frame().to_csv(clean, index=False)
    summary = tmp_path / f'{prefix}_team_baseline_summary.csv'
    pd.DataFrame([dict(Team='Williams', TeamBaselineMode='average-drivers', Median=90.)]).to_csv(summary,index=False)
    source_summary = summary.read_bytes()
    output = load_cached_monte_carlo_outputs(year=2026,race=16,session='R',output_dir=tmp_path,team_baseline_mode='average-drivers')
    assert output['event_name'] == 'Bahrain Grand Prix'
    assert pd.read_csv(clean).TyreLife.notna().sum() == 14
    assert clean.with_suffix('.repairs.json') in output['paths']
    assert summary.read_bytes() == source_summary
    original = clean.read_bytes()
    output = load_cached_monte_carlo_outputs(year=2026,race=16,session='R',output_dir=tmp_path,team_baseline_mode='average-drivers')
    assert clean.read_bytes() == original
    assert clean.with_suffix('.repairs.json') in output['paths']


def test_cached_missing_overlay_preserves_arbitrary_tokens(tmp_path):
    import csv
    import io
    frame = fixture_frame()
    path = tmp_path / 'clean.csv'
    frame.to_csv(path,index=False)
    text = path.read_text().replace('90.12345678901234', '90.1234567890123456789012345')
    path.write_text(text)
    old = list(csv.reader(io.StringIO(text)))
    cached = pd.read_csv(path)  # Same default parsing as normal cached producer.
    persist_clean_laps(cached,path,year=2026,race=16,session='R',cached=True)
    new = list(csv.reader(io.StringIO(path.read_text())))
    column = old[0].index('TyreLife')
    changes = [(r,c) for r in range(len(old)) for c in range(len(old[r])) if old[r][c] != new[r][c]]
    assert len(changes) == 14 and all(c == column and old[r][c] == '' for r,c in changes)


def test_fresh_normal_producer_hook_does_not_mutate_result(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from wostrategy.script import race_performance_review as producer
    empty = pd.DataFrame({'unchanged_numeric': [0.1234567890123456]})
    frame = fixture_frame()
    attributes = ['wet_lap_summary','sample_parameters','compound_degradation','compound_delta','team_compound_degradation','baseline_pace']
    result = SimpleNamespace(clean_laps=frame,summaries={key:empty for key in ['compound_degradation','compound_delta','team_compound_degradation']},**{name:empty for name in attributes})
    monkeypatch.setattr(producer,'build_tyre_information_summary',lambda **kwargs: empty)
    monkeypatch.setattr(producer,'write_tyre_information',lambda frame,path: frame.to_csv(path,index=False))
    paths = producer.save_race_outputs(result=result,team_baseline_samples=empty,team_baseline_summary=empty,sample_diagnostics=empty,pit_loss_summary=empty,year=2026,race=16,session='R',output_dir=tmp_path)
    clean = tmp_path/'race_performance_2026_16_R_clean_laps.csv'
    assert pd.read_csv(clean).TyreLife.notna().sum() == 14
    assert frame.TyreLife.isna().all()
    assert clean.with_suffix('.repairs.json') in paths
    expected = empty.to_csv(index=False)
    assert (tmp_path/'race_performance_2026_16_R_sample_parameters.csv').read_text() == expected


def test_cached_stale_frame_refuses_nonnull_and_reordering(tmp_path):
    import pytest
    stale = fixture_frame()
    path = tmp_path/'clean.csv'
    current = stale.copy()
    current.loc[0,'TyreLife'] = 25.
    current.to_csv(path,index=False)
    original = path.read_bytes()
    with pytest.raises(ValueError,match='non-empty cached TyreLife'):
        persist_clean_laps(stale,path,year=2026,race=16,session='R',cached=True)
    assert path.read_bytes() == original
    reordered = stale.iloc[::-1]
    reordered.to_csv(path,index=False)
    original = path.read_bytes()
    with pytest.raises(ValueError,match='identity changed'):
        persist_clean_laps(stale,path,year=2026,race=16,session='R',cached=True)
    assert path.read_bytes() == original
