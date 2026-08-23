import pytest

from wodata.contracts import StrategySearchState
from wostrategy.algorithm.exact_strategy_search import (
    StrategyModel, StrategyRules, search_best_fixed_stop_count,
)
from wostrategy.analysis.degradation_cutoff import calculate_degradation_cutoffs
from wostrategy.plots.degradation_cutoff import plot_degradation_cutoff


def _state() -> StrategySearchState:
    return StrategySearchState(
        selected_driver=None, current_completed_lap=0, total_laps=12,
        current_compound="MEDIUM", current_tyre_age=0, max_future_stops=3,
    )


def test_rules_compliant_fixed_stop_search_uses_two_compounds() -> None:
    model = StrategyModel(
        {"SOFT": -0.4, "MEDIUM": 0.0, "HARD": 0.3},
        {"SOFT": 0.2, "MEDIUM": 0.1, "HARD": 0.05},
    )
    result = search_best_fixed_stop_count(
        _state(), model, 2.0, stop_count=1, allow_any_start=True,
        strategy_rules=StrategyRules(minimum_distinct_dry_compounds=2),
    )
    assert result
    assert len(set(result[0].plan.compounds)) >= 2


def test_non_monotonic_cutoff_keeps_primary_references_and_warning() -> None:
    model = StrategyModel(
        {"SOFT": -0.4, "MEDIUM": 0.0, "HARD": 0.3},
        {"SOFT": 0.2, "MEDIUM": 0.1, "HARD": 0.05},
    )
    def evaluator(value, stops, rules):
        if stops == 1:
            return 2.0 + value if value < 0.2 else 1.0 + 8.0 * value
        if stops == 2:
            return 2.1 if value < 0.15 else 1.4 + 2.0 * value
        return 3.0 - value * 8.0
    result = calculate_degradation_cutoffs(
        model=model, state=_state(), pit_loss=2.0, scan_min=0.05, scan_max=0.4,
        coarse_step=0.01, evaluator=evaluator,
    )
    assert result.primary_1_to_2 is not None
    assert result.primary_2_to_3 is not None
    assert result.non_monotonic
    assert result.warnings


def _cutoff_model() -> StrategyModel:
    return StrategyModel(
        {"SOFT": -0.4, "MEDIUM": 0.0, "HARD": 0.3},
        {"SOFT": 0.4, "MEDIUM": 0.2, "HARD": 0.1},
    )


def _crossing_evaluator(crossing: float):
    def evaluator(value, stops, rules):
        if stops == 1:
            return value
        if stops == 2:
            return crossing
        return 1.0
    return evaluator


def test_one_to_two_cutoff_inside_initial_automatic_range() -> None:
    result = calculate_degradation_cutoffs(
        model=_cutoff_model(), state=_state(), pit_loss=2.0,
        scan_max=0.3, coarse_step=0.01,
        evaluator=_crossing_evaluator(0.1),
    )
    assert result.primary_1_to_2 == pytest.approx(0.1, abs=0.001)
    assert result.primary_1_to_2_status == "found_initial_scan"
    assert result.effective_scan_min == result.initial_scan_min


def test_one_to_two_cutoff_below_initial_range_is_found_after_extension() -> None:
    result = calculate_degradation_cutoffs(
        model=_cutoff_model(), state=_state(), pit_loss=2.0,
        scan_max=0.3, coarse_step=0.01,
        evaluator=_crossing_evaluator(0.02),
    )
    assert result.initial_scan_min == pytest.approx(0.05)
    assert result.effective_scan_min < result.initial_scan_min
    assert result.primary_1_to_2 == pytest.approx(0.02, abs=0.001)
    assert result.primary_1_to_2_status == "found_after_lower_extension"


def test_no_positive_one_to_two_crossing_extends_to_zero_with_clear_status() -> None:
    evaluated = []

    def evaluator(value, stops, rules):
        evaluated.append(value)
        return 1.0 if stops == 1 else 0.0 if stops == 2 else 2.0

    result = calculate_degradation_cutoffs(
        model=_cutoff_model(), state=_state(), pit_loss=2.0,
        scan_max=0.3, coarse_step=0.01, evaluator=evaluator,
    )
    assert result.primary_1_to_2 is None
    assert result.primary_1_to_2_status == "no_positive_degradation_crossing"
    assert result.effective_scan_min == 0.0
    assert min(evaluated) == 0.0
    assert all(value >= 0 for value in evaluated)
    assert any("automatic scan to zero" in warning for warning in result.warnings)


def test_explicit_scan_min_is_a_hard_lower_bound() -> None:
    evaluated = []
    base = _crossing_evaluator(0.02)

    def evaluator(value, stops, rules):
        evaluated.append(value)
        return base(value, stops, rules)

    result = calculate_degradation_cutoffs(
        model=_cutoff_model(), state=_state(), pit_loss=2.0,
        scan_min=0.05, scan_max=0.3, coarse_step=0.01, evaluator=evaluator,
    )
    assert result.primary_1_to_2 is None
    assert result.primary_1_to_2_status == "not_found_above_explicit_scan_min"
    assert result.effective_scan_min == 0.05
    assert min(evaluated) >= 0.05


def test_cutoff_plot_labels_prediction_and_both_crossings() -> None:
    def evaluator(value, stops, rules):
        return {1: value, 2: 0.1, 3: 0.4 - value}[stops]

    result = calculate_degradation_cutoffs(
        model=_cutoff_model(), state=_state(), pit_loss=2.0,
        scan_min=0.05, scan_max=0.4, coarse_step=0.01, evaluator=evaluator,
    )
    figure, axis = plot_degradation_cutoff(result)
    try:
        labels = axis.get_legend_handles_labels()[1]
        assert "Predicted MEDIUM degradation" in labels
        assert "1→2 cutoff" in labels
        assert "2→3 cutoff" in labels
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)
