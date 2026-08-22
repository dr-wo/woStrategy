from dataclasses import replace

import pandas as pd
import pytest

from wostrategy.algorithm.monte_carlo_race_performance import (
    EVALUATOR_ACCELERATED,
    EVALUATOR_LEGACY,
    WEIGHT_STRATEGY_BEST_RMSE_RELATIVE,
    WEIGHT_STRATEGY_GAUSSIAN,
    MonteCarloRacePerformanceAlgorithm,
    MonteCarloRacePerformanceConfig,
)
from wostrategy.model.live_retro_performance import prepare_live_retro_laps


def _laps():
    rows = []
    for driver, team, offset in (("A", "One", 0.0), ("B", "One", 0.3), ("C", "Two", 0.8)):
        for lap in range(1, 13):
            compound = "SOFT" if lap <= 3 else "MEDIUM" if lap <= 7 else "HARD"
            rows.append({
                "Driver": driver, "Team": team, "LapNumber": lap,
                "LapTimeSeconds": 90 + offset + 0.04 * lap + 0.01 * (lap % 3),
                "Compound": compound, "StintLapNumber": (lap - 1) % 4 + 1,
                "TyreLife": (lap - 1) % 4, "RunId": f"R:{driver}:{compound}",
            })
    return prepare_live_retro_laps(pd.DataFrame(rows), race_lap_count=60)


@pytest.mark.parametrize("baseline_group", ["driver", "team"])
@pytest.mark.parametrize(
    "weight_strategy", [WEIGHT_STRATEGY_GAUSSIAN, WEIGHT_STRATEGY_BEST_RMSE_RELATIVE]
)
@pytest.mark.parametrize("enforce_order", [False, True])
def test_accelerated_backend_is_roundoff_equivalent_to_legacy(
    baseline_group, weight_strategy, enforce_order
):
    config = MonteCarloRacePerformanceConfig(
        sample_count=257, random_seed=2026, sampling_strategy="latin-hypercube",
        baseline_group=baseline_group, weight_strategy=weight_strategy,
        track_temperature_celsius=35 if enforce_order else 15,
        degradation_order_track_temperature_celsius=20,
        compound_degradation_bounds={
            "SOFT": (0.01, 0.20), "MEDIUM": (0.005, 0.16), "HARD": (0.0, 0.12)
        },
        compound_delta_bounds={
            "SOFT": (-1.2, -0.1), "MEDIUM": (-0.8, 0.0), "HARD": (0.0, 0.0)
        },
    )
    legacy = MonteCarloRacePerformanceAlgorithm(
        replace(config, evaluator_backend=EVALUATOR_LEGACY)
    ).run(_laps())
    accelerated = MonteCarloRacePerformanceAlgorithm(
        replace(config, evaluator_backend=EVALUATOR_ACCELERATED, candidate_chunk_size=31)
    ).run(_laps())
    for field in legacy.__dataclass_fields__:
        pd.testing.assert_frame_equal(
            getattr(legacy, field), getattr(accelerated, field),
            check_exact=False, atol=5e-13, rtol=5e-13,
        )


def test_accelerated_backend_is_the_default_and_legacy_remains_selectable():
    assert MonteCarloRacePerformanceConfig().evaluator_backend == EVALUATOR_ACCELERATED
    assert replace(
        MonteCarloRacePerformanceConfig(), evaluator_backend=EVALUATOR_LEGACY
    ).evaluator_backend == EVALUATOR_LEGACY
