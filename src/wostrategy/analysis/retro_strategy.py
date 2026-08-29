"""Post-race strategy adapters using the existing exact optimiser."""

from __future__ import annotations

from typing import Any, Mapping

from wodata.contracts import StrategySearchState
from wostrategy.algorithm.exact_strategy_search import (
    StrategyModel,
    StrategyRules,
    search_best_compound_sequences,
)


def calculate_retro_green_optimum(
    *,
    retro_tyre_estimate: Mapping[str, Any],
    total_laps: int,
    green_pit_loss: float,
    result_count: int = 10,
    max_stops: int = 3,
    strategy_rules: StrategyRules | None = None,
) -> dict[str, Any]:
    """Optimise a hypothetical green race using retrospective tyre estimates."""
    compounds = dict(retro_tyre_estimate.get("compounds", {}))
    model = StrategyModel(
        compound_delta={
            compound: float(values["performance_delta_to_medium"])
            for compound, values in compounds.items()
        },
        degradation={
            compound: float(values["degradation_seconds_per_lap"])
            for compound, values in compounds.items()
        },
        analysis_id=str(
            dict(retro_tyre_estimate.get("provenance", {})).get(
                "artifact_directory", "retro_race_performance_review"
            )
        ),
        version="retro-green-optimum-v1",
    )
    rules = strategy_rules or StrategyRules(minimum_distinct_dry_compounds=2)
    state = StrategySearchState(
        current_completed_lap=0,
        total_laps=int(total_laps),
        current_compound="MEDIUM",
        current_tyre_age=0,
        selected_driver=None,
        max_future_stops=int(max_stops),
        provenance="retro_tyre_estimate",
    )
    results = search_best_compound_sequences(
        state,
        model,
        float(green_pit_loss),
        max_stops=max_stops,
        k=result_count,
        allow_any_start=True,
        strategy_rules=rules,
    )
    return {
        "name": "Retro Green Optimum",
        "internal_id": "retro_green_optimum",
        "uses_actual_event_timeline": False,
        "rules": {"minimum_distinct_dry_compounds": rules.minimum_distinct_dry_compounds},
        "green_pit_loss": float(green_pit_loss),
        "total_laps": int(total_laps),
        "strategies": [
            {
                "rank": item.rank,
                "compounds": list(item.plan.compounds),
                "pit_laps": list(item.plan.pit_laps),
                "remaining_cost": item.remaining_cost,
                "delta_to_best": item.delta_to_best,
                "formatted_strategy": item.formatted_strategy,
            }
            for item in results
        ],
    }


__all__ = ["calculate_retro_green_optimum"]
