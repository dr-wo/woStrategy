"""Tyre-only undercut curves from the effective strategy model."""
from wostrategy.algorithm.exact_strategy_search import StrategyModel, modelled_tyre_pace


def modelled_undercut_curve(model: StrategyModel, max_age: int) -> dict:
    """All ordered compound pairs, sampled at integer model ages 0..max_age."""
    if isinstance(max_age, bool) or not isinstance(max_age, int) or max_age < 0:
        raise ValueError("max_age must be a non-negative integer")
    compounds = sorted(model.compound_delta)
    points = []
    for source in compounds:
        for replacement in compounds:
            fresh = modelled_tyre_pace(replacement, 0, model)
            for age in range(max_age + 1):
                old = modelled_tyre_pace(source, age, model)
                points.append({
                    "source_compound": source,
                    "replacement_compound": replacement,
                    "source_tyre_age": age,
                    "source_modelled_tyre_pace_seconds": old,
                    "fresh_replacement_modelled_tyre_pace_seconds": fresh,
                    "modelled_undercut_power_seconds": old - fresh,
                })
    return {
        "schema_version": 1,
        "title": "Modelled / predicted undercut power",
        "definition": "source tyre pace at model age a minus replacement tyre pace at model age 0",
        "age_convention": "Model tyre age; fresh replacement is the theoretical age-zero anchor, not an observed out-lap or first clean racing lap",
        "replacement_tyre_age": 0,
        "source_age_grid": {"min": 0, "max": max_age, "step": 1},
        "pace_convention": "Tyre-only relative pace term in seconds; excludes common fuel and track baseline",
        "sign_convention": "Positive: replacement faster; zero: equal; negative: replacement slower",
        "input_source": "effective StrategyModel used by this report run; no Live observations or sensitivity state",
        "model": {
            "analysis_id": model.analysis_id, "version": model.version,
            "compound_delta": dict(model.compound_delta),
            "degradation": dict(model.degradation),
        },
        "points": points,
    }
