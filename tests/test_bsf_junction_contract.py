"""Closed-source and method-version regressions, including the real R15 bank."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_bsf_junction_workflow import producer_inputs
from wostrategy.analysis.bsf_junction import contract, workflow
from wostrategy.analysis.bsf_junction.constants import TEAMS

ORACLE = Path(__file__).parent / 'fixtures/bsf_junction/BSF_TRACKER_2026_R1_R15.csv'
DATA = Path(__file__).resolve().parents[2] / 'woData'


def assert_r15(table):
    expected = pd.read_csv(ORACLE).query('round == 15').set_index('team').loc[TEAMS]
    actual = table.set_index('team').loc[TEAMS]
    np.testing.assert_allclose(actual.score_s_per_lap, expected.score_s_per_lap,
                               atol=1e-10, rtol=0)
    assert actual.support_flag.tolist() == expected.support_flag.tolist()


def test_live_kernel_change_rejects_generation_and_read_without_recomputation(tmp_path, monkeypatch):
    from wostrategy.analysis.bsf_junction import _support
    sources = producer_inputs(tmp_path / 'producer')
    args = dict(season=2027, round_number=1, source_paths=sources, artifact_root=tmp_path / 'bsf')
    _, meta = workflow.get_bsf_event(**args)
    saved = Path(meta['table']).read_bytes()
    monkeypatch.setattr(workflow, 'canonical_reference_inputs',
                        lambda *a: pytest.fail('incompatible v1 must not recompute'))
    monkeypatch.setattr(_support, 'LADDER', [(0, 0)])
    from wostrategy.analysis.bsf_junction import calculate_bsf_reference
    for call in [lambda: workflow.get_bsf_event(**args),
                 lambda: calculate_bsf_reference(None, None, None, None, season=2027, round_number=1),
                 lambda: workflow.read_bsf_event(season=2027, round_number=1, artifact_root=tmp_path / 'bsf')]:
        with pytest.raises(ValueError, match='explicit version bump'):
            call()
    assert Path(meta['table']).read_bytes() == saved


@pytest.mark.parametrize('change', ['fingerprint', 'version', 'legacy_kernel'])
def test_incompatible_artifact_cannot_silently_recompute(tmp_path, monkeypatch, change):
    sources = producer_inputs(tmp_path / 'producer')
    args = dict(season=2027, round_number=1, source_paths=sources, artifact_root=tmp_path / 'bsf')
    _, meta = workflow.get_bsf_event(**args)
    path = Path(meta['table']).with_suffix('.metadata.json')
    if change == 'fingerprint':
        meta['numerical_kernel_fingerprint'] = 'unapproved'
    elif change == 'version':
        meta['method_version'] = 'unapproved_v2'
    else:
        del meta['numerical_kernel_fingerprint']
        meta['kernel_hashes']['_graph.py'] = 'unapproved'
    path.write_text(json.dumps(meta))
    before = Path(meta['table']).read_bytes()
    monkeypatch.setattr(workflow, 'canonical_reference_inputs',
                        lambda *a: pytest.fail('incompatible artifact must not recompute'))
    for call in [lambda: workflow.get_bsf_event(**args),
                 lambda: workflow.read_bsf_event(season=2027, round_number=1, artifact_root=tmp_path / 'bsf')]:
        with pytest.raises(ValueError, match='(incompatible|identity mismatch)'):
            call()
    assert Path(meta['table']).read_bytes() == before


def test_non_numerical_caption_change_keeps_approved_contract(monkeypatch):
    from wostrategy.analysis.bsf_junction import constants
    before = contract.require_approved_kernel()
    monkeypatch.setattr(constants, 'CAPTION', 'New report wording')
    assert contract.require_approved_kernel() == before
    monkeypatch.setattr(constants, 'VERSION', 'unapproved_v2')
    with pytest.raises(ValueError, match='Unapproved BSF method version'):
        contract.require_approved_kernel()


@pytest.fixture(scope='module')
def reconstructed_r15(tmp_path_factory):
    approved = contract.frozen_round_contract(2026, 15)
    if not all((DATA / z['path']).is_file() for z in approved['sources'].values()):
        pytest.skip('Original published R15 bank is an external integration dependency')
    root = tmp_path_factory.mktemp('original_r15')
    args = dict(season=2026, round_number=15, data_root=DATA, artifact_root=root)
    table, metadata = workflow.get_bsf_event(**args)
    assert metadata['source_selection'] == 'approved_frozen_history'
    assert_r15(table)
    return args, table, metadata


def test_original_r15_cache_hit_matches_oracle(reconstructed_r15, monkeypatch):
    args, original, metadata = reconstructed_r15
    monkeypatch.setattr(workflow, 'canonical_reference_inputs',
                        lambda *a: pytest.fail('approved cache hit must not fit'))
    table, _ = workflow.get_bsf_event(**args)
    assert_r15(table)
    np.testing.assert_array_equal(table.score_s_per_lap, original.score_s_per_lap)


def test_original_r15_cache_loss_rebuild_matches_oracle(reconstructed_r15):
    args, original, metadata = reconstructed_r15
    # Delete only this temporary integration cache, never the accepted data cache.
    for path in (Path(metadata['table']), Path(metadata['score_terms']),
                 Path(metadata['table']).with_suffix('.metadata.json')):
        path.unlink()
    table, second = workflow.get_bsf_event(**args)
    assert_r15(table)
    np.testing.assert_array_equal(table.score_s_per_lap, original.score_s_per_lap)
    assert second['table_sha256'] == metadata['table_sha256']


def test_newer_r15_cannot_overwrite_approved_cache(reconstructed_r15):
    args, _, metadata = reconstructed_r15
    path = Path(metadata['table'])
    before = path.read_bytes()
    newer = workflow.canonical_source_paths(season=2026, round_number=15, data_root=DATA)
    with pytest.raises(ValueError, match='not approved for frozen history'):
        workflow.get_bsf_event(**args, source_paths=newer)
    assert path.read_bytes() == before
    assert_r15(workflow.get_bsf_event(**args)[0])


def test_current_only_r15_has_no_replacement_reconstruction(tmp_path):
    current = workflow.canonical_source_paths(season=2026, round_number=15, data_root=DATA)
    if not all(p.is_file() for p in current.values()):
        pytest.skip('Current per-event bank is an external integration dependency')
    targets = workflow.canonical_source_paths(season=2026, round_number=15, data_root=tmp_path / 'current_only')
    for role, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(current[role])
    cache = tmp_path / 'bsf'
    with pytest.raises(FileNotFoundError, match='original published bank unavailable'):
        workflow.get_bsf_event(season=2026, round_number=15,
                               data_root=tmp_path / 'current_only', artifact_root=cache)
    assert not list(cache.rglob('bsf_junction_reference.csv'))
    # Locating a different file under the original archive name is also rejected.
    pinned = contract.frozen_round_contract(2026, 15)['sources']['delta']
    replacement = tmp_path / 'current_only' / pinned['path']
    replacement.parent.mkdir(parents=True, exist_ok=True)
    replacement.symlink_to(current['delta'])
    with pytest.raises(ValueError, match='Original published bank identity mismatch'):
        workflow.get_bsf_event(season=2026, round_number=15,
                               data_root=tmp_path / 'current_only', artifact_root=cache)
    assert not list(cache.rglob('bsf_junction_reference.csv'))


def test_absent_closed_original_is_explicit_missing_history(tmp_path, monkeypatch):
    current = pd.read_csv(ORACLE).query('round == 15').copy()
    current['round'] = 16
    actual_get = workflow.get_bsf_event
    def event(**kwargs):
        return (current, {}) if kwargs['round_number'] == 16 else actual_get(**kwargs)
    monkeypatch.setattr(workflow, 'get_bsf_event', event)
    monkeypatch.setattr('wostrategy.plots.bsf_junction.plot_bsf_season', lambda *a, **kw: {})
    output = workflow.generate_bsf_season(season=2026, round_number=16,
        data_root=tmp_path / 'absent', artifact_root=tmp_path / 'cache',
        output_dir=tmp_path / 'run', figures_dir=tmp_path / 'figures')
    assert output['status'] == 'partial_history'
    assert output['included_rounds'] == [16]
    assert '15' in output['unavailable_rounds']
    assert len(output['relative_team_performance']) == 11


def test_legacy_closed_cache_requires_exact_source_roles_and_kernel(tmp_path):
    source = DATA / 'wostrategy/bsf_junction_reference/schema_v1/year=2026/round=15/session=R'
    if not source.is_dir():
        pytest.skip('Accepted legacy R15 cache is an external integration dependency')
    import shutil
    destination = tmp_path / 'year=2026/round=15/session=R'
    shutil.copytree(source, destination)
    args = dict(season=2026, round_number=15, artifact_root=tmp_path)
    assert_r15(workflow.read_bsf_event(**args)[0])
    path = destination / 'bsf_junction_reference.metadata.json'
    metadata = json.loads(path.read_text())
    altered = copy.deepcopy(metadata)
    key = next(iter(altered['source_hashes']))
    altered['source_hashes'][key] = 'replacement'
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='approved frozen history'):
        workflow.get_bsf_event(**args)
    altered = copy.deepcopy(metadata)
    altered['kernel_hashes']['_transfer.py'] = 'replacement'
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='legacy BSF numerical kernel'):
        workflow.get_bsf_event(**args)
