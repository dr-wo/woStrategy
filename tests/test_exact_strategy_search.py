import itertools

import pytest

from wodata.contracts import StrategySearchState
from wostrategy.algorithm.exact_strategy_search import (
    StrategyModel,
    StrategyPlan,
    format_strategy,
    optimise_same_sequence,
    revalue_fixed_strategy,
    search_best_continuations,
    search_best_compound_sequences,
    stint_cost,
)


def model():
    return StrategyModel(
        {"MEDIUM": -0.25, "HARD": 0.0},
        {"MEDIUM": 0.10, "HARD": 0.04},
        analysis_id="a1",
        version="v1",
    )


def test_closed_form_stint_cost_equals_explicit_sum():
    expected = sum(-0.25 + 0.10 * age for age in range(8, 14))
    assert stint_cost(6, "MEDIUM", 7, model()) == pytest.approx(expected)


def test_fixed_revaluation_preserves_structure_and_age_boundary():
    state = StrategySearchState(10, 20, "MEDIUM", 5)
    plan = StrategyPlan(("MEDIUM", "HARD"), (14,))
    result = revalue_fixed_strategy(plan, state, model(), pit_loss=18.0)
    expected = stint_cost(4, "MEDIUM", 5, model()) + 18 + stint_cost(6, "HARD", 0, model())
    assert result.plan == plan
    assert result.remaining_cost == pytest.approx(expected)
    assert result.formatted_strategy == "M-(14)H"


def test_same_sequence_changes_pit_laps_only_and_returns_top_k():
    state = StrategySearchState(0, 8, "MEDIUM", 0)
    original = StrategyPlan(("MEDIUM", "HARD"), (4,))
    results = optimise_same_sequence(original, state, model(), pit_loss=1.0, k=4)
    assert len(results) == 4
    assert all(item.plan.compounds == original.compounds for item in results)
    assert [item.remaining_cost for item in results] == sorted(
        item.remaining_cost for item in results
    )


def test_dynamic_programming_matches_exhaustive_small_race():
    state = StrategySearchState(0, 5, "MEDIUM", 0, max_future_stops=1)
    results = search_best_continuations(state, model(), pit_loss=0.8, k=10)
    exhaustive = [(stint_cost(5, "MEDIUM", 0, model()), ("MEDIUM",), ())]
    for pit_lap, next_compound in itertools.product(range(1, 5), ("MEDIUM", "HARD")):
        cost = (
            stint_cost(pit_lap, "MEDIUM", 0, model())
            + 0.8
            + stint_cost(5 - pit_lap, next_compound, 0, model())
        )
        exhaustive.append((cost, ("MEDIUM", next_compound), (pit_lap,)))
    expected = min(exhaustive)
    assert results[0].remaining_cost == pytest.approx(expected[0])
    assert results[0].plan.compounds == expected[1]
    assert results[0].plan.pit_laps == expected[2]
    assert len({(r.plan.compounds, r.plan.pit_laps) for r in results}) == len(results)


def test_format_uses_agreed_compact_presentation_without_parsing_it():
    assert format_strategy(
        StrategyPlan(("MEDIUM", "HARD", "HARD"), (20, 45))
    ) == "M-(20)H-45(H)"


def test_compound_sequence_search_keeps_only_best_pit_timing_per_sequence():
    state = StrategySearchState(0, 8, "MEDIUM", 0, max_future_stops=1)
    results = search_best_compound_sequences(state, model(), pit_loss=1.0, k=10)
    assert len(results) == 3
    assert {result.plan.compounds for result in results} == {
        ("MEDIUM",), ("MEDIUM", "MEDIUM"), ("MEDIUM", "HARD")
    }


def test_pre_race_sequence_search_can_choose_starting_compound():
    state = StrategySearchState(0, 8, "MEDIUM", 0, max_future_stops=1)
    results = search_best_compound_sequences(
        state, model(), pit_loss=1.0, k=10, allow_any_start=True
    )
    assert {result.plan.compounds for result in results} == {
        ("MEDIUM",), ("HARD",),
        ("MEDIUM", "MEDIUM"), ("MEDIUM", "HARD"),
        ("HARD", "MEDIUM"), ("HARD", "HARD"),
    }
