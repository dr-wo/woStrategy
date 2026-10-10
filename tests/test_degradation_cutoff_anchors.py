import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from wodata.contracts import StrategySearchState
from wostrategy.algorithm.exact_strategy_search import StrategyModel, StrategyRules
from wostrategy.analysis.degradation_cutoff import calculate_degradation_cutoffs
from wostrategy.plots.degradation_cutoff import plot_degradation_cutoff


def _anchors(*values):
    return [dict(session=f'FP{i + 1}', medium_degradation=x, source_family='fp_diagnostic')
            for i, x in enumerate(values)]


def _linear(x, stops, rules):
    return [0, .02, .11, .29, .59][stops] - stops * x


def _run(anchors=(), *, max_stops=4, evaluator=_linear, **kwargs):
    calls = []
    def evaluate(x, stops, rules):
        calls.append((x, stops))
        assert stops <= min(4, max_stops)
        if kwargs.get('scan_max') is not None:
            assert x <= kwargs['scan_max']
        return evaluator(x, stops, rules)
    result = calculate_degradation_cutoffs(
        model=StrategyModel({'MEDIUM': 0}, {'MEDIUM': .039}),
        state=StrategySearchState(0, 56, 'MEDIUM', 0, None, max_stops), pit_loss=23.7,
        coverage_anchors=_anchors(*anchors), evaluator=evaluate, **kwargs,
    )
    return result, calls


@pytest.mark.parametrize('anchors', [(), (None, float('nan'), float('inf'), -1, 'bad'), (.04,)])
def test_no_extension_preserves_initial_calculation(anchors):
    result, calls = _run(anchors)
    original, _ = _run()
    assert result.sweep == original.sweep
    assert result.primary_0_to_1 == original.primary_0_to_1
    assert result.primary_1_to_2 == original.primary_1_to_2
    assert not result.anchor_coverage['upper_extension_triggered']
    assert all(stops < 4 for _, stops in calls)


@pytest.mark.parametrize('anchors,pair,cutoff', [((.15,), '2_to_3', .18),
    ((.15, .22, .16), '3_to_4', .30)])
def test_first_boundary_to_right_only(anchors, pair, cutoff):
    result, calls = _run(anchors)
    meta = result.anchor_coverage
    assert meta['right_transition']['transition'] == pair
    assert meta['right_transition']['cutoff'] == pytest.approx(cutoff, abs=1e-7)
    assert cutoff <= result.effective_scan_max <= cutoff + .005
    assert meta['anchor_max'] == max(anchors)
    assert meta['upper_extension_triggered']
    assert result.primary_1_to_2 == _run()[0].primary_1_to_2
    if pair == '2_to_3':
        assert not any(stops == 4 for _, stops in calls)
    assert result.to_dict()['cutoff_table'][-1]['status'] == 'found_after_upper_evidence_extension'
    assert all(entry['samples'] <= 5 for entry in meta['extension_history'])


@pytest.mark.parametrize('epsilon', [-1e-12, 0, 1e-12])
def test_initial_boundary_equality_uses_original_tolerance(epsilon):
    original, _ = _run()
    result, calls = _run((original.primary_1_to_2 + epsilon,))
    assert result.anchor_coverage['stop_reason'] == 'anchor_at_boundary'
    assert result.sweep == original.sweep
    assert not any(stops == 4 for _, stops in calls)


def test_extended_boundary_equality_does_not_seek_four_stops():
    result, calls = _run((.18 + 1e-12,))
    assert result.anchor_coverage['at_transition']['transition'] == '2_to_3'
    assert not any(stops == 4 for _, stops in calls)


def test_anchor_beyond_supported_limit():
    result, calls = _run((.4,))
    assert result.anchor_coverage['stop_reason'] == 'unsupported_beyond_4_stop'
    assert result.primary_3_to_4 == pytest.approx(.3, abs=1e-7)
    assert max(s for _, s in calls) == 4
    assert result.effective_scan_max < .31


@pytest.mark.parametrize('max_stops', [0, 1, 2, 3])
def test_event_limit(max_stops):
    result, calls = _run((.4,), max_stops=max_stops)
    assert result.anchor_coverage['stop_reason'] == 'event_max_stops_reached'
    assert len(result.to_dict()['cutoff_table']) == max_stops
    assert all(s <= max_stops for _, s in calls)


def test_explicit_max_is_hard_even_for_refinement():
    result, calls = _run((.22,), scan_max=.2)
    assert result.effective_scan_max == .2
    assert max(x for x, _ in calls) <= .2
    assert result.anchor_coverage['stop_reason'] == 'explicit_scan_max_reached'
    assert result.anchor_coverage['explicit_scan_max'] == .2
    assert result.anchor_coverage['left_transition']['transition'] == '2_to_3'
    boundary = result.anchor_coverage['boundary_coverage']['upper_boundary']
    assert boundary['higher_minus_lower_cost'] == pytest.approx(.1)


def test_lazy_four_stop_boundary_inside_explicit_range():
    result, calls = _run((.22,), scan_max=.35)
    assert result.primary_3_to_4 == pytest.approx(.3, abs=1e-7)
    assert result.primary_3_to_4_status == 'found_initial_scan'
    assert not result.anchor_coverage['upper_extension_triggered']
    assert max(x for x, _ in calls) <= .35


def test_high_precision_explicit_max_is_not_rounded_up():
    limit = .1170000000006
    result, calls = _run((.22,), scan_max=limit)
    assert result.effective_scan_max <= limit
    assert max(x for x, _ in calls) <= limit


@pytest.mark.parametrize('value,reason', [(None, 'fixed_stop_class_unavailable'),
    (float('nan'), 'nonfinite_strategy_cost')])
def test_missing_next_class(value, reason):
    result, _ = _run((.15,), evaluator=lambda x,s,r: value if s == 3 else _linear(x,s,r))
    assert result.anchor_coverage['stop_reason'] == reason
    assert not result.anchor_coverage['upper_extension_triggered']


def test_isolated_comparable_samples_are_insufficient():
    def evaluator(x, s, r):
        if s == 3 and x < .117:
            return None
        return _linear(x, s, r)
    result, _ = _run((.15,), evaluator=evaluator)
    assert result.anchor_coverage['stop_reason'] == 'insufficient_comparable_samples'


def test_discontinuous_crossing_is_not_fabricated():
    def evaluator(x, s, r):
        return _linear(x, 2, r) + (1 if x < .18 else -1) if s == 3 else _linear(x,s,r)
    result, _ = _run((.15,), evaluator=evaluator)
    assert result.anchor_coverage['stop_reason'] == 'unstable_crossing'
    assert result.primary_2_to_3 is None


def test_nonfinite_inside_refinement_is_not_fabricated():
    def evaluator(x, s, r):
        if s == 3 and .179 < x < .181:
            return float('nan')
        return _linear(x,s,r)
    result, _ = _run((.15,), evaluator=evaluator)
    assert result.anchor_coverage['stop_reason'] == 'unstable_crossing'
    assert result.primary_2_to_3 is None


def test_no_crossing_stops_at_transparent_computational_budget():
    result, _ = _run((.15,), max_upper_extension_blocks=2,
                     evaluator=lambda x,s,r: _linear(x,2,r)+1 if s == 3 else _linear(x,s,r))
    assert result.anchor_coverage['stop_reason'] == 'computational_guard_reached'
    assert len(result.anchor_coverage['extension_history']) == 2
    assert result.effective_scan_max == pytest.approx(.167)


def test_plot_actual_curves_and_context():
    result, _ = _run((.22,))
    figure, axis = plot_degradation_cutoff(result, fp_diagnostic_markers=_anchors(.22, float('nan')))
    try:
        labels = axis.get_legend_handles_labels()[1]
        assert 'Best 4-stop' in labels and '3→4 cutoff' in labels
        assert 'Best 5-stop' not in labels
        text = ' '.join(t.get_text() for t in figure.texts)
        assert 'scalar reference only' in text
        assert '3→4' in text
        curve = next(line for line in axis.lines if line.get_label() == 'Best 4-stop')
        assert curve.get_xdata()[-1] == result.effective_scan_max
    finally:
        plt.close(figure)


@pytest.mark.parametrize('event', [15, 16])
@pytest.mark.parametrize('minimum_compounds', [0, 2])
def test_saved_event_anchors(event, minimum_compounds):
    path = Path(__file__).parent / 'fixtures' / f'degradation_cutoff_r{event}.json'
    saved = json.loads(path.read_text())
    kwargs = dict(model=StrategyModel(saved['compound_delta'], saved['degradation']),
                  state=StrategySearchState(0,saved['total_laps'],'MEDIUM',0,None,saved['max_stops']),
                  pit_loss=saved['pit_loss'], strategy_rules=StrategyRules(minimum_distinct_dry_compounds=minimum_compounds))
    original = calculate_degradation_cutoffs(**kwargs)
    result = calculate_degradation_cutoffs(**kwargs, coverage_anchors=saved['coverage_anchors'])
    assert result.initial_scan_min == original.initial_scan_min == .00975
    assert result.initial_scan_max == original.initial_scan_max == pytest.approx(.117)
    assert result.primary_0_to_1 == original.primary_0_to_1
    assert result.primary_1_to_2 == original.primary_1_to_2
    if event == 15:
        assert result.sweep == original.sweep
        assert not result.anchor_coverage['upper_extension_triggered']
    else:
        expected = .192744 if minimum_compounds else .182551
        assert result.primary_2_to_3 == pytest.approx(expected, abs=1e-6)
        if minimum_compounds:
            assert result.primary_3_to_4 is None
            assert not any(4 in p.costs for p in result.sweep)
            assert result.anchor_coverage['right_transition']['transition'] == '2_to_3'
        else:
            assert result.primary_3_to_4 > result.anchor_coverage['anchor_max']
            assert result.anchor_coverage['right_transition']['transition'] == '3_to_4'
            assert result.effective_scan_max <= result.primary_3_to_4 + .005
        assert result.anchor_coverage['stop_reason'] == 'right_boundary_found'
