from dataclasses import replace
import json

import pytest

from wostrategy.algorithm.exact_strategy_search import StrategyModel, modelled_tyre_pace, stint_cost
from wostrategy.analysis.modelled_undercut import modelled_undercut_curve


def test_curve_uses_strategy_costing_and_age_zero_anchor():
    model = StrategyModel({"SOFT": -.4, "HARD": .3}, {"SOFT": .2, "HARD": .05},
                          fitted_start_age={"SOFT": 3, "HARD": 2})
    result = modelled_undercut_curve(model, 8)
    assert len(result["points"]) == 4 * 9
    for point in result["points"]:
        source, target, age = (point[k] for k in ("source_compound", "replacement_compound", "source_tyre_age"))
        fresh = model.compound_delta[target]
        assert point["fresh_replacement_modelled_tyre_pace_seconds"] == fresh
        # Direct marginal strategy cost, at every positive age.
        expected = (stint_cost(age, source, 0, model) - stint_cost(age - 1, source, 0, model)
                    if age else model.compound_delta[source])
        assert point["source_modelled_tyre_pace_seconds"] == pytest.approx(expected)
        assert point["modelled_undercut_power_seconds"] == pytest.approx(expected - fresh)
    assert modelled_tyre_pace("SOFT", 0, model) == -.4


def test_sign_same_compound_cross_compound_and_degradation():
    model = StrategyModel({"SOFT": -.4, "HARD": .3}, {"SOFT": .2, "HARD": 0})
    def powers(m):
        return {(p["source_compound"], p["replacement_compound"], p["source_tyre_age"]):
                p["modelled_undercut_power_seconds"] for p in modelled_undercut_curve(m, 5)["points"]}
    original = powers(model)
    assert original["SOFT", "HARD", 0] == pytest.approx(-.7)
    assert original["HARD", "SOFT", 0] == pytest.approx(.7)
    assert original["SOFT", "SOFT", 0] == 0
    assert original["SOFT", "SOFT", 5] == pytest.approx(1)
    assert original["HARD", "HARD", 5] == 0
    changed = powers(replace(model, degradation={"SOFT": .3, "HARD": 0}))
    for (source, target, age), power in original.items():
        assert changed[source, target, age] - power == pytest.approx(.1 * age if source == "SOFT" else 0)


def test_deterministic_order_and_serialization():
    a = StrategyModel({"SOFT": -.4, "HARD": .3}, {"SOFT": .2, "HARD": .05})
    b = StrategyModel(dict(reversed(list(a.compound_delta.items()))), dict(reversed(list(a.degradation.items()))))
    assert json.dumps(modelled_undercut_curve(a, 3), sort_keys=True) == json.dumps(modelled_undercut_curve(b, 3), sort_keys=True)


@pytest.mark.parametrize("age", [-1, 1.5, True])
def test_invalid_age_grid(age):
    with pytest.raises(ValueError):
        modelled_undercut_curve(StrategyModel({"HARD": 0}, {"HARD": .1}), age)
