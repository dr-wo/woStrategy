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
    sources = producer_inputs(tmp_path / 'producer')
    before = {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in sources.items()}
    clean = pd.read_csv(sources['clean'])
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
    assert len(table) == 11 and table.fallback_dependent.all()
    assert meta['method'] == 'projected_mean' and meta['source_selection'] == 'explicit_replay'
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
