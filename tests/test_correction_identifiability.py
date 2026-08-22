import numpy as np
import pandas as pd

from wostrategy.model.correction_identifiability import (
    LinearFuelModel,
    LinearTrackEvolutionModel,
    diagnose_correction_identifiability,
)


class QuadraticTrackModel:
    name = "QuadraticTrackEvolutionModel"
    parameter_names = ("track_curve",)

    def evaluate(self, observations, parameters):
        lap = observations["LapNumber"].to_numpy(float)
        return parameters["track_curve"] * (lap - lap.mean()) ** 2


def _observations():
    rows = []
    for driver in ("A", "B"):
        for lap in range(1, 9):
            rows.append({
                "Team": driver, "Driver": driver, "LapNumber": lap,
                "FuelProxyLapsRemaining": 60 - lap,
            })
    return pd.DataFrame(rows)


def test_linear_fuel_track_exact_alias_is_recovered_after_baseline_projection():
    observations = _observations()
    result = diagnose_correction_identifiability(
        observations,
        (LinearFuelModel(), LinearTrackEvolutionModel(observations.LapNumber.mean())),
        {"fuel_rate": 0.05, "track_rate": 0.02},
    )
    assert result.component_names == ("FuelModel", "TrackEvolutionModel")
    assert result.rank == 1
    assert np.isclose(result.sensitivity_correlation[0][1], -1.0)
    assert result.normalized_singular_values[1] < 1e-15
    assert np.isinf(result.condition_number)


def test_nonlinear_component_uses_generic_local_jacobian_and_recovers_rank_two():
    result = diagnose_correction_identifiability(
        _observations(), (LinearFuelModel(), QuadraticTrackModel()),
        {"fuel_rate": 0.05, "track_curve": 0.002},
    )
    assert result.rank == 2
    assert abs(result.sensitivity_correlation[0][1]) < 1e-8
