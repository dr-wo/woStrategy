from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import TEAMS, METHOD, VERSION
from ._support import build_edges
from ._transfer import ExactTransfer, COLS
from ._graph import ConditionGraph, projected_scores
from .contract import require_approved_kernel


class _Adapter:
    def __init__(self, d):
        self.d = d
        self.model = ExactTransfer(d)

    def predict(self, a, b):
        result = self.model.predict(
            self.d.iloc[list(a)][COLS + ['row_id']],
            self.d.iloc[list(b)][COLS + ['row_id']], return_donors=True,
        )
        raw = sorted({i for donor in result['donors']
                      for side in ['source_rows', 'target_rows'] for i in donor[side]})
        return dict(estimate=result['prediction'], n_teams=result['n_donor_teams'],
                    raw_rows=raw, **result)


def calculate_bsf_reference(clean_laps, row_predictions, base_absolute, counts,
                            *, season, round_number):
    """Apply the frozen kernel to explicit canonical reference inputs.

    Row IDs follow the persisted clean-lap file order; no filtering, rematching
    of admitted cases, smoothing, fitting of a team solver or retuning occurs.
    Returns the eleven-team table plus auditable exact support/path terms.
    """
    require_approved_kernel()
    d = clean_laps.copy().reset_index(drop=True)
    required = COLS + ['LapTimeSeconds']
    if d[required].isna().any().any():
        raise ValueError('BSF requires complete persisted clean-lap covariates')
    if not set(d.Team).issubset(TEAMS):
        raise ValueError('Roster differs from the frozen eleven-team contract')
    if d.duplicated(['Driver', 'Stint', 'LapNumber']).any():
        raise ValueError('Duplicate named driver/stint/race-lap rows')
    for name in ['LapNumber', 'TyreLife', 'TyreAgeLaps']:
        if not np.equal(d[name], d[name].astype(int)).all():
            raise ValueError(f'Frozen matcher requires integer {name}')
    d['row_id'] = np.arange(len(d))
    # Producer CSVs use numeric stint labels; keep the frozen float identity
    # when an all-integer event is inferred as int64 by the CSV reader.
    d['Stint'] = d.Stint.astype(float)
    d['clock_seconds'] = pd.to_timedelta(d.LapStartTime).dt.total_seconds()
    if not np.isfinite(d.clock_seconds).all():
        raise ValueError('BSF requires finite endpoint clocks')
    if 'EventName' not in d:
        d['EventName'] = f'{season} R{round_number}'
    prediction = np.asarray(row_predictions, float)
    base = np.asarray(base_absolute, float)
    counts = np.asarray(counts, int)
    if prediction.shape != (len(d),) or base.shape != (11,) or counts.shape != (11,):
        raise ValueError('Canonical reference input dimensions do not match')
    if not np.isfinite(prediction).all() or not np.isfinite(base).all():
        raise ValueError('Canonical reference predictions/baselines must be finite')
    actual_counts = d.groupby('Team').size().reindex(TEAMS, fill_value=0).to_numpy()
    if not np.array_equal(counts, actual_counts):
        raise ValueError('Canonical counts differ from the persisted clean laps')
    library = build_edges({round_number: d}, threshold=4, clock_limit=60)
    graph = ConditionGraph(d, library, transfer_model=_Adapter(d))
    scores, metadata, terms = projected_scores(graph, prediction, base, counts, shrink=1.)
    component = graph.components()
    reference = metadata[0]['reference_node']
    reached = next((c for c in component if reference in c), [])
    observed, junction, fallback = [], [], []
    for i, team in enumerate(TEAMS):
        tt = [t for t in terms if t['team'] == team]
        observed.append(float(np.mean([
            sum(graph.edges[e]['delta'] * sign for e, sign in t['path']
                if graph.edges[e]['kind'] == 'observation') for t in tt])) if tt else 0.)
        junction.append(float(np.mean([
            sum(graph.edges[e]['delta'] * sign for e, sign in t['path']
                if graph.edges[e]['kind'] == 'transfer') for t in tt])) if tt else 0.)
        fallback.append(float(np.mean([
            t['canonical_reference_prediction'] + t['projection_delta']
            for t in tt])) if tt else float(base[i]))
    observed, junction, fallback = map(np.asarray, (observed, junction, fallback))
    uncentered = observed + junction + fallback
    raw_increment = uncentered - base
    incidence = {}
    for edge in graph.edges:
        if edge['kind'] == 'observation':
            for node in [edge['a'], edge['b']]:
                incidence[node] = incidence.get(node, 0) + 1
    alternatives = []
    for ref in sorted(incidence, key=lambda k: (-incidence[k], k))[:3]:
        ss, _, _ = projected_scores(graph, prediction, base, counts, reference_override=ref)
        alternatives.append(ss)
    ranges = np.ptp(alternatives, axis=0) if alternatives else np.zeros(11)
    rows = []
    for i, row in enumerate(metadata):
        flag = ('N' if row['clean_laps'] == 0 else
                'P' if row['projected_support_nodes'] else 'F')
        tt = [t for t in terms if t['team'] == row['team']]
        used = [graph.edges[e] for t in tt for e, _ in t['path']
                if graph.edges[e]['kind'] == 'transfer']
        rows.append(dict(
            year=int(season), round=int(round_number), method=METHOD, method_version=VERSION,
            **row, score_s_per_lap=float(scores[i]), support_flag=flag,
            gauge='all_eleven_teams_arithmetic_mean_zero; lower_is_faster',
            no_usable_clean_evidence=flag == 'N',
            reliability_flag=f'{flag};unvalidated;fallback-dependent',
            fallback_status={'P': 'canonical_projection_and_shared_anchor',
                             'F': 'full_canonical_fallback',
                             'N': 'no_clean_field_mean_prior'}[flag],
            observed_path_uncentered_s=float(observed[i]),
            junction_transfer_uncentered_s=float(junction[i]),
            fallback_uncentered_s=float(fallback[i]),
            observed_path_centered_s=float(observed[i] - observed.mean()),
            junction_transfer_centered_s=float(junction[i] - junction.mean()),
            fallback_centered_s=float(fallback[i] - fallback.mean()),
            precenter_absolute_s=float(uncentered[i]),
            event_centering_offset_s=float(uncentered.mean()),
            bsf_minus_canonical_s=float(scores[i] - row['canonical_score']),
            graph_increment_before_gauge_s=float(raw_increment[i]),
            common_gauge_adjustment_vs_canonical_s=float(-raw_increment.mean()),
            graph_components=len(component), reference_component_nodes=len(reached),
            own_nodes_outside_reference_component=sum(
                n['team'] == row['team'] and key not in reached
                for key, n in graph.nodes.items()),
            reference_sensitivity_score_range=float(ranges[i]),
            reference_sensitivity_method='first3_by_observation_degree_then_lexical_id; not CI',
            minimum_donor_teams_on_used_junctions=min(
                (e['donors']['n_teams'] for e in used), default=None),
            transfer_classes=';'.join(sorted({e['donors']['transfer_class'] for e in used})),
            direct_count_note='library_edges; overlapping rows are not independent samples',
            decomposition_note='algebraic attribution only; centered terms include field gauge',
        ))
    for term in terms:
        term['expanded_path'] = [dict(sign=sign, **graph.edges[e],
                                      source_node=graph.nodes[graph.edges[e]['a']],
                                      target_node=graph.nodes[graph.edges[e]['b']])
                                 for e, sign in term['path']]
    result = pd.DataFrame(rows)
    if not np.allclose(result.score_s_per_lap,
                       result.observed_path_centered_s + result.junction_transfer_centered_s
                       + result.fallback_centered_s, atol=1e-10, rtol=0):
        raise ValueError('BSF contribution decomposition failed')
    return result, terms


def mercedes_relative_history(history):
    """Presentation-only same-round contrasts, keeping every P/F/N row."""
    history = history.sort_values(['round', 'team']).copy()
    if history.empty:
        raise ValueError('BSF history has no completed events')
    if not history.support_flag.isin(['P', 'F', 'N']).all():
        raise ValueError('Unknown BSF provenance class')
    if not np.isfinite(history[['score_s_per_lap', 'canonical_score']].to_numpy()).all():
        raise ValueError('Non-finite persisted BSF/canonical scores')
    if history.duplicated(['round', 'team']).any():
        raise ValueError('Duplicate BSF event/team rows')
    for _, group in history.groupby('round'):
        if set(group.team) != set(TEAMS) or len(group) != 11:
            raise ValueError('History requires all eleven teams at each included event')
    reference = history[history.team.eq('Mercedes')].set_index('round')
    history['bsf_relative_to_mercedes_s'] = (
        history.score_s_per_lap - history['round'].map(reference.score_s_per_lap))
    history['canonical_relative_to_mercedes_s'] = (
        history.canonical_score - history['round'].map(reference.canonical_score))
    history['bsf_minus_canonical_relative_to_mercedes_s'] = (
        history.bsf_relative_to_mercedes_s - history.canonical_relative_to_mercedes_s)
    history['mercedes_support'] = history['round'].map(reference.support_flag)
    history['plotted_as_performance'] = history.support_flag.ne('N')
    return history.reset_index(drop=True)
