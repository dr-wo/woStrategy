import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from wodata.contracts import StrategySearchState
from wostrategy.algorithm.exact_strategy_search import StrategyModel, StrategyRules
from wostrategy.analysis.degradation_cutoff import calculate_degradation_cutoffs
from wostrategy.plots.degradation_cutoff import plot_degradation_cutoff


def _calculate(evaluator, **kwargs):
    options = dict(scan_min=0.05, scan_max=0.15, coarse_step=0.05)
    options.update(kwargs)
    return calculate_degradation_cutoffs(
        model=StrategyModel({"MEDIUM": 0.0}, {"MEDIUM": 0.2}),
        state=StrategySearchState(0, 12, "MEDIUM", 0, None, 3),
        pit_loss=2.0, evaluator=evaluator, **options,
    )


def test_initial_found_and_rules_unavailable_keep_existing_statuses():
    evaluator = lambda value, stops, rules: {0: 2 * value - 0.075,
                                            1: value, 2: 0.1, 3: 1.0}[stops]
    found = _calculate(evaluator)
    assert found.to_dict()["cutoff_table"][0]["status"] == "found_initial_scan"
    assert found.primary_0_to_1 == pytest.approx(0.075, abs=0.001)
    assert "s/lap (found_initial_scan)" in found.to_dict()["cutoff_table"][0]["display_text"]
    rules = _calculate(evaluator, strategy_rules=StrategyRules(minimum_distinct_dry_compounds=2))
    row = rules.to_dict()["cutoff_table"][0]
    assert row["status"] == row["coverage"]["status"] == "unavailable_rules_compliant"
    assert row["display_text"] == "0→1: unavailable under rules."
    assert row["coverage"]["upper_boundary"]["higher_minus_lower_cost"] is None


@pytest.mark.parametrize("two,three,status", [
    (2.0, 5.0, "both_classes_feasible"),
    (None, 5.0, "lower_class_unavailable"),
    (2.0, None, "higher_class_unavailable"),
    (None, None, "both_classes_unavailable"),
    (float("nan"), 5.0, "insufficient_comparable_samples"),
])
def test_unresolved_coverage_distinguishes_feasibility(two, three, status):
    result = _calculate(lambda value, stops, rules: {0: 0.0, 1: 1.0, 2: two, 3: three}[stops])
    payload = json.loads(json.dumps(result.to_dict()))
    row = payload["cutoff_table"][2]
    assert row["status"] == "not_found_in_scan_range"
    assert row["cutoff_medium_degradation"] is None
    assert payload["initial_scan_max"] == payload["effective_scan_max"] == 0.15
    assert row["coverage"]["status"] == status
    assert row["coverage"]["scan_min"] == 0.05
    assert row["coverage"]["scan_max"] == 0.15
    assert "not found through M degradation = 0.15" in row["display_text"]
    assert "cutoff >" not in row["display_text"]
    boundary = row["coverage"]["upper_boundary"]
    if status == "both_classes_feasible":
        assert boundary["higher_minus_lower_cost"] == 3.0
        assert boundary["lower_cost_stop_count"] == 2
        assert row["coverage"]["comparable_interval_count"] == 2
        assert "2-stop was 3.00 s lower-cost than 3-stop" in row["display_text"]
    else:
        assert boundary["higher_minus_lower_cost"] is None
        assert boundary["lower_cost_stop_count"] is None
        assert "lower-cost" not in row["display_text"]


def test_boundary_unavailable_does_not_erase_earlier_comparable_samples():
    result = _calculate(lambda value, stops, rules: (
        None if stops == 3 and value == 0.15 else float(stops)
    ))
    coverage = result.to_dict()["cutoff_table"][2]["coverage"]
    assert coverage["status"] == "higher_class_unavailable"
    assert coverage["comparable_sample_count"] == 2
    assert coverage["comparable_interval_count"] == 1
    assert coverage["upper_boundary"]["lower_stop_feasible"] is True
    assert coverage["upper_boundary"]["higher_stop_feasible"] is False


def test_isolated_comparable_samples_do_not_establish_crossing_coverage():
    result = _calculate(lambda value, stops, rules: (
        None if stops == 2 and value == 0.1 else float(stops)
    ))
    row = result.to_dict()["cutoff_table"][2]
    assert row["coverage"]["status"] == "insufficient_comparable_samples"
    assert row["coverage"]["comparable_sample_count"] == 2
    assert row["coverage"]["comparable_interval_count"] == 0
    assert row["coverage"]["upper_boundary"]["higher_minus_lower_cost"] == 1.0
    assert "Insufficient adjacent comparable samples" in row["display_text"]


def test_negative_boundary_margin_does_not_claim_the_lower_stop_class_wins():
    result = _calculate(lambda value, stops, rules: {0: 0.0, 1: 1.0, 2: 5.0, 3: 2.0}[stops])
    row = result.to_dict()["cutoff_table"][2]
    assert row["coverage"]["upper_boundary"]["higher_minus_lower_cost"] == -3.0
    assert "3-stop was 3.00 s lower-cost than 2-stop" in row["display_text"]


def test_lower_extension_coverage_is_transition_specific():
    result = _calculate(
        lambda value, stops, rules: {0: 2 * value - 0.02, 1: value,
                                     2: 0.01, 3: 1.0}[stops],
        scan_min=None, coarse_step=0.005,
    )
    payload = result.to_dict()
    assert result.effective_scan_min < result.initial_scan_min
    assert payload["cutoff_table"][1]["status"] == "found_after_lower_extension"
    assert payload["cutoff_table"][1]["coverage"]["scan_min"] == result.effective_scan_min
    assert payload["cutoff_table"][2]["coverage"]["scan_min"] == result.initial_scan_min
    assert payload["initial_scan_max"] == payload["effective_scan_max"] == 0.15
    legacy = replace(result, initial_scan_max=None, effective_scan_max=None)
    assert legacy.to_dict()["effective_scan_max"] == 0.15


def test_plot_unresolved_note_preserves_curves_without_a_fake_cutoff():
    result = _calculate(lambda value, stops, rules: {0: 2 * value - 0.075,
                                                    1: value, 2: 0.1, 3: 1.0}[stops])
    figure, axis = plot_degradation_cutoff(result)
    try:
        labels = axis.get_legend_handles_labels()[1]
        assert "0→1 cutoff" in labels
        assert "2→3 cutoff" not in labels
        curve = next(line for line in axis.lines if line.get_label() == "Best 3-stop")
        assert curve.get_xdata()[-1] == 0.15
        text = " ".join(note.get_text().replace("\n", " ") for note in figure.texts)
        assert "2→3: cutoff not found through M degradation = 0.15" in text
        assert "2-stop was 0.90 s lower-cost than 3-stop" in text
    finally:
        plt.close(figure)


def test_plot_does_not_attribute_other_zero_stop_unavailability_to_rules():
    result = _calculate(lambda value, stops, rules: None if stops == 0 else float(stops))
    figure, axis = plot_degradation_cutoff(result)
    try:
        labels = axis.get_legend_handles_labels()[1]
        assert "0-stop unavailable" in labels
        assert "0-stop unavailable (rules)" not in labels
        assert "0→1 cutoff N/A (rules)" not in labels
    finally:
        plt.close(figure)


@pytest.mark.parametrize("branch,min_compounds", [("unrestricted", 0), ("rules_compliant", 2)])
def test_saved_r16_cutoffs_and_upper_boundary_remain_numerically_unchanged(branch, min_compounds):
    saved = json.loads((Path(__file__).parent / "fixtures/degradation_cutoff_r16.json").read_text())
    result = calculate_degradation_cutoffs(
        model=StrategyModel(saved["compound_delta"], saved["degradation"]),
        state=StrategySearchState(0, saved["total_laps"], "MEDIUM", 0, None, saved["max_stops"]),
        pit_loss=saved["pit_loss"],
        strategy_rules=StrategyRules(minimum_distinct_dry_compounds=min_compounds),
    )
    expected = saved[branch]
    assert [result.primary_0_to_1, result.primary_1_to_2, result.primary_2_to_3] == expected["cutoffs"]
    payload = result.to_dict()
    assert payload["initial_scan_min"] == payload["effective_scan_min"] == saved["initial_scan_min"]
    assert payload["initial_scan_max"] == pytest.approx(saved["scan_max"], abs=1e-12)
    assert payload["effective_scan_max"] == saved["scan_max"]
    assert len(result.sweep) == 23
    coverage = payload["cutoff_table"][2]["coverage"]
    assert coverage["status"] == "both_classes_feasible"
    boundary = coverage["upper_boundary"]
    assert boundary["lower_stop_feasible"] is boundary["higher_stop_feasible"] is True
    assert [boundary["lower_stop_cost"], boundary["higher_stop_cost"]] == pytest.approx(expected["boundary_costs"])
    assert boundary["higher_minus_lower_cost"] == pytest.approx(expected["boundary_margin"])
    assert boundary["lower_cost_stop_count"] == 2
