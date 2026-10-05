import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wostrategy.analysis.bsf_junction.constants import TEAMS
from wostrategy.analysis.bsf_junction.workflow import (
    canonical_reference_inputs, get_bsf_event, read_bsf_event, generate_bsf_season,
)
from wostrategy.plots.bsf_junction import season_plot_data, plot_bsf_season


def producer_inputs(root):
    root.mkdir(parents=True, exist_ok=True)
    clean = pd.DataFrame([
        dict(Team=t, Driver=f'D{j}', Stint=1, Compound='MEDIUM', FreshTyre=True,
             LapNumber=k, TyreLife=k, TyreAgeLaps=k-1, LapStartTime=f'0 days 00:0{k}:00',
             LapTimeSeconds=100+j+0.1*k, FuelProxyLapsRemaining=10-k)
        for j,t in enumerate(TEAMS) for k in range(1,5)
    ])
    params = pd.DataFrame([
        dict(SampleId=i, FuelRateSecondsPerLap=0, TrackRateSecondsPerLap=track,
             FuelRef=0, RaceLapRef=0, TyreAgeRef=0, WeightStrategy='gaussian', BaselineGroup='team')
        for i,track in enumerate([0,0.1])
    ])
    delta = pd.DataFrame([dict(SampleId=i, Compound='MEDIUM', CompoundDeltaSeconds=0) for i in range(2)])
    deg = pd.DataFrame([dict(SampleId=i, Team=t, Compound='MEDIUM', TeamCompoundDegSecondsPerLap=0)
                        for i in range(2) for t in TEAMS])
    sources = {k: root / f'{k}.csv' for k in ['clean','parameters','delta','team_deg']}
    for key,table in [('clean',clean),('parameters',params),('delta',delta),('team_deg',deg)]:
        table.to_csv(sources[key], index=False)
    sources['metadata'] = root / 'metadata.json'
    sources['metadata'].write_text(json.dumps({'clean_lap_noise_sigma':0.5}))
    return sources


def test_saved_bank_reference_is_posterior_mean_and_inputs_unchanged(tmp_path):
    from wostrategy.analysis.bsf_junction import calculate_bsf_reference
    from wostrategy.analysis.bsf_junction.input_coverage import complete_clean_laps
    sources = producer_inputs(tmp_path / 'producer')
    before = {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in sources.items()}
    clean = pd.read_csv(sources['clean'])
    normalized, _ = complete_clean_laps(clean)
    pd.testing.assert_frame_equal(normalized, clean)
    prediction, base, counts = canonical_reference_inputs(clean, sources)
    # Candidate 0 residual SSE / lap = .0125; candidate 1 exactly fits.
    weights = np.exp(np.array([-.0125/(2*.5**2),0]))
    weights /= weights.sum()
    expected = np.arange(11)+100+weights[0]*.25
    np.testing.assert_allclose(base, expected, atol=1e-12, rtol=0)
    assert counts.tolist() == [4]*11
    np.testing.assert_allclose(prediction,
        expected.repeat(4)+weights[1]*.1*clean.LapNumber.to_numpy(), atol=1e-12, rtol=0)
    table, meta = get_bsf_event(season=2027,round_number=1,source_paths=sources,
                              artifact_root=tmp_path / 'bsf')
    expected_table, _ = calculate_bsf_reference(clean, prediction, base, counts,
                                               season=2027, round_number=1)
    pd.testing.assert_frame_equal(table, expected_table)
    assert len(table) == 11 and table.fallback_dependent.all()
    assert meta['method'] == 'projected_mean' and meta['source_selection'] == 'explicit_replay'
    assert meta['status'] == 'available' and meta['warnings'] == []
    assert meta['input_coverage']['original_rows'] == 44
    assert meta['input_coverage']['excluded_rows'] == 0
    assert {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in sources.items()} == before
    _, second = get_bsf_event(season=2027,round_number=1,source_paths=sources,
                             artifact_root=tmp_path/'second_cache')
    assert meta['table_sha256'] == second['table_sha256']
    assert meta['score_terms_sha256'] == second['score_terms_sha256']


def test_persistence_cache_history_and_corruption(tmp_path, monkeypatch):
    sources = producer_inputs(tmp_path / 'producer')
    kwargs = dict(season=2027,round_number=1,source_paths=sources,artifact_root=tmp_path/'bsf')
    original, meta = get_bsf_event(**kwargs)
    import wostrategy.analysis.bsf_junction.workflow as workflow
    def should_not_compute(*args,**kw):
        raise AssertionError('unchanged inputs must reuse persisted event')
    monkeypatch.setattr(workflow, 'canonical_reference_inputs', should_not_compute)
    cached,_ = get_bsf_event(**kwargs)
    np.testing.assert_array_equal(cached.score_s_per_lap, original.score_s_per_lap)
    for path in sources.values():
        path.unlink()
    # Historical rendering reads the persisted identity, never a newer bank.
    persisted,_ = read_bsf_event(season=2027,round_number=1,artifact_root=tmp_path/'bsf')
    np.testing.assert_array_equal(cached.score_s_per_lap, persisted.score_s_per_lap)
    Path(meta['score_terms']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        read_bsf_event(season=2027,round_number=1,artifact_root=tmp_path/'bsf')


def test_missing_input_no_substitute_or_canonical_write(tmp_path):
    with pytest.raises(FileNotFoundError,match='producer inputs unavailable'):
        get_bsf_event(season=2027,round_number=1,source_paths={k:tmp_path/'missing'
            for k in ['clean','parameters','delta','team_deg','metadata']},
                      artifact_root=tmp_path/'bsf')
    assert not (tmp_path/'bsf').exists()


@pytest.mark.parametrize('setting',['weight','sigma'])
def test_unsupported_producer_settings_fail_instead_of_changing_method(tmp_path,setting):
    sources = producer_inputs(tmp_path/'producer')
    if setting == 'weight':
        table = pd.read_csv(sources['parameters'])
        table['WeightStrategy'] = 'inverse_rmse'
        table.to_csv(sources['parameters'],index=False)
    else:
        sources['metadata'].write_text(json.dumps({'clean_lap_noise_sigma':1.0}))
    with pytest.raises(ValueError,match='Frozen BSF reference requires'):
        get_bsf_event(season=2027,round_number=1,source_paths=sources,artifact_root=tmp_path/'bsf')
    assert not (tmp_path/'bsf').exists()


def test_plot_contract_excludes_prior_and_fallback_links(tmp_path, monkeypatch):
    fixture = Path(__file__).parent/'fixtures/bsf_junction/BSF_TRACKER_2026_R1_R15.csv'
    history = pd.read_csv(fixture)
    contract = season_plot_data(history.sample(frac=1,random_state=42),'bsf_relative_to_mercedes_s')
    assert list(contract) == TEAMS
    assert sum(len(z['points']) for z in contract.values()) == 148
    assert sum(len(z['prior_rounds']) for z in contract.values()) == 17
    assert sum(len(z['P_segments']) for z in contract.values()) == 32
    from matplotlib.figure import Figure
    captured = []
    def capture(fig,path,**kwargs):
        if str(path).endswith('.png'):
            captured.append(fig)
    monkeypatch.setattr(Figure,'savefig',capture)
    output = plot_bsf_season(history,tmp_path,season=2026)
    assert set(output) == {'tracker','diagnostic','support'}
    for fig in captured[:2]:
        assert sum(len(collection.get_offsets()) for collection in fig.axes[0].collections) == 148
        assert len(fig.axes[1].texts) == 165
    summary = json.loads((tmp_path/'bsf_junction_reference.plot_contract.json').read_text())
    assert summary['tracker']['P_to_P_dashed_segments'] == 32
    assert summary['tracker']['N_only_in_provenance_strip'] == 17
    assert summary['unit'] == 's/lap' and summary['Mercedes_exact_zero']


def test_season_missing_history_is_explicit_and_current_failure_propagates(tmp_path, monkeypatch):
    import wostrategy.analysis.bsf_junction.workflow as workflow
    fixture = Path(__file__).parent/'fixtures/bsf_junction/BSF_TRACKER_2026_R1_R15.csv'
    current = pd.read_csv(fixture).query('round == 3').copy()
    def event(**kwargs):
        if kwargs['round_number'] == 3:
            return current, {'method':'projected_mean'}
        raise FileNotFoundError('historical input unavailable')
    monkeypatch.setattr(workflow,'get_bsf_event',event)
    monkeypatch.setattr('wostrategy.plots.bsf_junction.plot_bsf_season',lambda *a,**k:{})
    output = generate_bsf_season(season=2026,round_number=3,output_dir=tmp_path/'run',
                                figures_dir=tmp_path/'figures',artifact_root=tmp_path/'cache')
    assert output['included_rounds'] == [3]
    assert set(output['unavailable_rounds']) == {'1','2'}
    assert output['status'] == 'partial_history'
    assert len(output['relative_team_performance']) == 11
    with pytest.raises(FileNotFoundError):
        generate_bsf_season(season=2026,round_number=4,output_dir=tmp_path/'failed',
                            figures_dir=tmp_path/'failed_figures',artifact_root=tmp_path/'cache')
    assert not (tmp_path/'failed').exists()


@pytest.mark.parametrize('field,value', [
    ('TyreLife', np.nan), ('TyreLife', np.inf), ('TyreAgeLaps', np.nan),
    ('FuelProxyLapsRemaining', np.nan), ('LapTimeSeconds', np.inf),
    ('LapStartTime', 'invalid clock'), ('Compound', np.nan),
])
def test_incomplete_rows_are_excluded_before_frozen_calculation(tmp_path, field, value):
    from wostrategy.analysis.bsf_junction import calculate_bsf_reference
    sources = producer_inputs(tmp_path/'producer')
    clean = pd.read_csv(sources['clean'])
    if field in ['TyreLife', 'TyreAgeLaps', 'FuelProxyLapsRemaining', 'LapTimeSeconds']:
        clean[field] = clean[field].astype(float)
    clean.loc[0, field] = value
    clean.to_csv(sources['clean'], index=False)
    before = {key: path.read_bytes() for key, path in sources.items()}
    valid = clean.iloc[1:].reset_index(drop=True)
    prediction, base, counts = canonical_reference_inputs(valid, sources)
    expected, _ = calculate_bsf_reference(valid, prediction, base, counts,
                                         season=2027, round_number=1)
    args = dict(season=2027, round_number=1, source_paths=sources, artifact_root=tmp_path/'cache')
    table, meta = get_bsf_event(**args)
    pd.testing.assert_frame_equal(table, expected)
    assert meta['status'] == 'degraded_input'
    coverage = meta['input_coverage']
    assert (coverage['original_rows'], coverage['retained_rows'], coverage['excluded_rows']) == (44, 43, 1)
    assert coverage['affected_drivers'] == ['D0']
    assert coverage['incomplete_covariates'] == {field: 1}
    assert coverage['retained_source_rows'] == list(range(1, 44))
    assert coverage['excluded_clean_laps'] == [dict(source_row=0, driver='D0', team=TEAMS[0],
                                                  lap_number=1, incomplete_covariates=[field])]
    assert meta['graph_support']['provenance'] == table.support_flag.value_counts().reindex(
        ['P', 'F', 'N'], fill_value=0).to_dict()
    assert 'reduced evidence coverage' in meta['warnings'][0]
    assert 'fully supported graph estimate' in meta['warnings'][0]
    assert {key: path.read_bytes() for key, path in sources.items()} == before
    # Both cache paths retain degradation; no regeneration is needed to report it.
    assert get_bsf_event(**args)[1] == meta
    assert read_bsf_event(season=2027, round_number=1, artifact_root=tmp_path/'cache')[1] == meta


def test_multiple_missing_fields_count_rows_once_and_team_without_evidence_is_prior(tmp_path):
    sources = producer_inputs(tmp_path/'producer')
    clean = pd.read_csv(sources['clean'])
    clean.loc[:3, ['TyreLife', 'FuelProxyLapsRemaining']] = np.nan
    clean.to_csv(sources['clean'], index=False)
    table, meta = get_bsf_event(season=2027, round_number=1, source_paths=sources,
                                artifact_root=tmp_path/'cache')
    assert meta['input_coverage']['excluded_rows'] == 4
    assert meta['input_coverage']['incomplete_covariates'] == {'TyreLife': 4, 'FuelProxyLapsRemaining': 4}
    assert table.set_index('team').loc[TEAMS[0], 'support_flag'] == 'N'


@pytest.mark.parametrize('field', ['Stint', 'LapNumber', 'TyreLife', 'TyreAgeLaps',
                                  'LapTimeSeconds', 'FuelProxyLapsRemaining'])
@pytest.mark.parametrize('token', [' ', 'malformed'])
def test_malformed_csv_token_normalizes_retained_numeric_covariates(tmp_path, field, token):
    from wostrategy.analysis.bsf_junction import calculate_bsf_reference
    from wostrategy.analysis.bsf_junction.input_coverage import complete_clean_laps
    sources = producer_inputs(tmp_path/'producer')
    original = pd.read_csv(sources['clean'])
    contaminated = original.copy()
    contaminated[field] = contaminated[field].astype(object)
    contaminated.loc[17, field] = token
    contaminated.to_csv(sources['clean'], index=False)
    before = {key: path.read_bytes() for key, path in sources.items()}
    persisted = pd.read_csv(sources['clean'])
    assert persisted[field].dtype == object
    snapshot = persisted.copy(deep=True)
    retained_positions = [i for i in range(44) if i != 17]
    expected_clean = original.iloc[retained_positions].reset_index(drop=True)
    retained, coverage = complete_clean_laps(persisted)
    assert pd.api.types.is_numeric_dtype(retained[field])
    pd.testing.assert_frame_equal(retained, expected_clean, check_dtype=False)
    pd.testing.assert_frame_equal(persisted, snapshot)
    assert coverage['retained_source_rows'] == retained_positions
    assert coverage['excluded_clean_laps'] == [dict(
        source_row=17, driver='D4', team=TEAMS[4], lap_number=persisted.loc[17, 'LapNumber'],
        incomplete_covariates=[field])]
    assert coverage['incomplete_covariates'] == {field: 1}
    assert 'unparseable' in coverage['reasons'][0]
    prediction, base, counts = canonical_reference_inputs(expected_clean, sources)
    expected, _ = calculate_bsf_reference(expected_clean, prediction, base, counts,
                                         season=2027, round_number=1)
    table, meta = get_bsf_event(season=2027, round_number=1, source_paths=sources,
                               artifact_root=tmp_path/'cache')
    pd.testing.assert_frame_equal(table, expected)
    assert meta['status'] == 'degraded_input'
    assert (coverage['original_rows'], coverage['retained_rows'], coverage['excluded_rows']) == (44, 43, 1)
    assert meta['input_coverage']['retained_source_rows'] == retained_positions
    assert {key: path.read_bytes() for key, path in sources.items()} == before


def test_contaminated_csv_does_not_round_retained_noninteger_tyre_life(tmp_path):
    from wostrategy.analysis.bsf_junction.input_coverage import complete_clean_laps
    sources = producer_inputs(tmp_path/'producer')
    clean = pd.read_csv(sources['clean'])
    clean['TyreLife'] = clean['TyreLife'].astype(object)
    clean.loc[0, 'TyreLife'] = ' '
    clean.loc[1, 'TyreLife'] = '2.5'
    clean.to_csv(sources['clean'], index=False)
    before = {key: path.read_bytes() for key, path in sources.items()}
    retained, coverage = complete_clean_laps(pd.read_csv(sources['clean']))
    assert coverage['excluded_rows'] == 1
    assert coverage['retained_source_rows'] == list(range(1, 44))
    assert retained.loc[0, 'TyreLife'] == 2.5
    with pytest.raises(ValueError, match='Frozen matcher requires integer TyreLife'):
        get_bsf_event(season=2027, round_number=1, source_paths=sources,
                      artifact_root=tmp_path/'cache')
    assert not (tmp_path/'cache').exists()
    assert {key: path.read_bytes() for key, path in sources.items()} == before


@pytest.mark.parametrize('condition', ['empty', 'all_missing_life', 'missing_column', 'noninteger_life'])
def test_non_runnable_inputs_still_fail_without_artifacts(tmp_path, condition):
    sources = producer_inputs(tmp_path/'producer')
    clean = pd.read_csv(sources['clean'])
    if condition == 'empty':
        clean = clean.iloc[:0]
    elif condition == 'all_missing_life':
        clean['TyreLife'] = np.nan
    elif condition == 'missing_column':
        clean = clean.drop(columns='TyreLife')
    else:
        clean['TyreLife'] = clean['TyreLife'].astype(float)
        clean.loc[0, 'TyreLife'] = 1.5
    clean.to_csv(sources['clean'], index=False)
    before = {key: path.read_bytes() for key, path in sources.items()}
    with pytest.raises(ValueError, match='No usable|schema missing|integer TyreLife'):
        get_bsf_event(season=2027, round_number=1, source_paths=sources,
                      artifact_root=tmp_path/'cache')
    assert not (tmp_path/'cache').exists()
    assert {key: path.read_bytes() for key, path in sources.items()} == before


def test_degraded_season_keeps_figures_warning_and_history_status(tmp_path):
    sources = producer_inputs(tmp_path/'producer')
    clean = pd.read_csv(sources['clean'])
    clean.loc[0, 'TyreLife'] = np.nan
    clean.to_csv(sources['clean'], index=False)
    output = generate_bsf_season(season=2027, round_number=1,
        output_dir=tmp_path/'run', figures_dir=tmp_path/'figures', artifact_root=tmp_path/'cache',
        source_paths_by_round={1: sources})
    assert output['status'] == 'degraded_input'
    assert output['history_status'] == 'complete'
    assert output['degraded_rounds'] == [1]
    assert '1/44' in output['warnings'][0]
    assert 'physical `TyreLife`' in output['warnings'][0]
    assert all(Path(path).is_file() for path in output['figures'].values())
    assert 'DEGRADED INPUT' in (tmp_path/'figures/bsf_junction_reference_mercedes.svg').read_text()
    contract = json.loads((tmp_path/'figures/bsf_junction_reference.plot_contract.json').read_text())
    assert contract['degraded_rounds'] == [1]
