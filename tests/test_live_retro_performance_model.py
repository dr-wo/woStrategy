import numpy as np
import pandas as pd

from wostrategy.algorithm.monte_carlo_race_performance import (
    MonteCarloRacePerformanceAlgorithm,
    MonteCarloRacePerformanceConfig,
)
from wostrategy.model.live_retro_performance import (
    prepare_live_retro_laps,
    retro_design_rank,
    run_live_retro_model,
)


def laps():
    rows = []
    for driver, team, offset in (("A", "One", 0.0), ("B", "Two", 1.0)):
        for lap in range(1, 9):
            compound = "MEDIUM" if lap <= 4 else "HARD"
            rows.append({"Driver": driver, "Team": team, "LapNumber": lap,
                         "LapTimeSeconds": 90 + offset + .1 * lap,
                         "Compound": compound, "TyreLife": (lap - 1) % 4,
                         "StintLapNumber": (lap - 1) % 4 + 1,
                         "RunId": f"R:{driver}:{1 if lap <= 4 else 2}"})
    return pd.DataFrame(rows)


def config():
    return MonteCarloRacePerformanceConfig(
        sample_count=128, random_seed=7, sampling_strategy="latin-hypercube",
        compound_delta_reference="HARD", track_temperature_celsius=30,
        degradation_order_track_temperature_celsius=20,
    )


def test_live_retro_wrapper_is_exact_retro_algorithm_on_same_laps():
    prepared = prepare_live_retro_laps(laps(), race_lap_count=8)
    direct = MonteCarloRacePerformanceAlgorithm(config()).run(prepared)
    live = run_live_retro_model(laps(), config(), as_of_leader_lap=8)
    pd.testing.assert_frame_equal(live.retro_result.sample_parameters, direct.sample_parameters)
    pd.testing.assert_frame_equal(live.retro_result.compound_degradation,
                                  direct.compound_degradation)
    pd.testing.assert_frame_equal(live.retro_result.compound_delta, direct.compound_delta)
    pd.testing.assert_frame_equal(live.retro_result.baseline_pace, direct.baseline_pace)


def test_retro_fuel_and_track_are_aliased_after_baseline_profiling():
    prepared = prepare_live_retro_laps(laps(), race_lap_count=8)
    rank = retro_design_rank(prepared, baseline_group="driver")
    assert rank["fuel_track_alias"]
    assert rank["fuel_track_profiled_rank"] == 1
    assert np.isclose(rank["fuel_track_profiled_correlation"], -1.0)
