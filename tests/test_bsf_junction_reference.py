from pathlib import Path
import gzip, hashlib, json
import numpy as np
import pandas as pd
import pytest

from wostrategy.analysis.bsf_junction import calculate_bsf_reference, mercedes_relative_history

FIXTURES = Path(__file__).parent / 'fixtures/bsf_junction'


@pytest.fixture(scope='module')
def frozen_history():
    provenance = json.loads((FIXTURES / 'provenance.json').read_text())
    oracle = FIXTURES / 'BSF_TRACKER_2026_R1_R15.csv'
    assert hashlib.sha256(oracle.read_bytes()).hexdigest() == provenance['oracle_csv_sha256']
    with gzip.open(FIXTURES / 'frozen_inputs.json.gz', 'rt') as stream:
        inputs = json.load(stream)
    tables = []
    for event in inputs:
        table, _ = calculate_bsf_reference(
            pd.DataFrame(event['clean_laps']), event['row_predictions'],
            event['base_absolute'], event['counts'], season=2026,
            round_number=event['round'],
        )
        tables.append(table)
    return pd.concat(tables, ignore_index=True), pd.read_csv(oracle)


def test_frozen_165_scores_support_and_provenance(frozen_history):
    actual, oracle = frozen_history
    actual = actual.sort_values(['round', 'team']).reset_index(drop=True)
    oracle = oracle.sort_values(['round', 'team']).reset_index(drop=True)
    assert len(actual) == 165
    assert actual.groupby('round').size().eq(11).all()
    np.testing.assert_allclose(actual.score_s_per_lap, oracle.score_s_per_lap,
                               atol=1e-10, rtol=0)
    for field in ['support_flag', 'support_class', 'clean_laps',
                  'direct_local_edge_count', 'projected_support_nodes',
                  'exact_local_edge_count', 'available_transfer_edge_count']:
        assert actual[field].tolist() == oracle[field].tolist(), field
    for field in ['canonical_score', 'junction_transfer_uncentered_s',
                  'fallback_centered_s', 'reference_sensitivity_score_range']:
        np.testing.assert_allclose(actual[field], oracle[field], atol=1e-10, rtol=0)
    assert actual.fallback_dependent.all()
    assert actual.groupby('round').score_s_per_lap.mean().abs().max() < 1e-12


def test_mercedes_contrasts_and_diagnostic_match_frozen_values(frozen_history):
    actual, _ = frozen_history
    relative = mercedes_relative_history(actual.sample(frac=1, random_state=41))
    oracle = pd.DataFrame(json.loads(
        (FIXTURES / 'MERCEDES_RELATIVE_DERIVED_VALUES.json').read_text())['rows'])
    oracle = oracle.sort_values(['round', 'team']).reset_index(drop=True)
    for field in ['bsf_relative_to_mercedes_s', 'canonical_relative_to_mercedes_s',
                  'bsf_minus_canonical_relative_to_mercedes_s']:
        np.testing.assert_allclose(relative[field], oracle[field].astype(float),
                                   atol=1e-10, rtol=0)
        assert relative.loc[relative.team.eq('Mercedes'), field].eq(0).all()
    assert relative[['round', 'team']].values.tolist() == oracle[['round', 'team']].values.tolist()
    assert (~relative.plotted_as_performance).sum() == 17
    assert set(relative.loc[~relative.plotted_as_performance, 'support_flag']) == {'N'}


def test_history_rejects_missing_or_duplicate_team(frozen_history):
    history, _ = frozen_history
    with pytest.raises(ValueError, match='all eleven'):
        mercedes_relative_history(history.iloc[1:])
    with pytest.raises(ValueError, match='Duplicate'):
        mercedes_relative_history(pd.concat([history, history.iloc[:1]]))
