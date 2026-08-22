from datetime import datetime, timezone

import numpy as np
import pandas as pd

from wostrategy.model.pre_race_performance import WeekendModelConfig
from wostrategy.model.race_specific_performance import (
    BASELINE_PER_DRIVER,
    COMPOUNDS,
    FrozenPreRacePrior,
    MarginalPrior,
    compound_connectivity,
    run_race_specific_model,
)


def prior():
    degradation = {
        "HARD": MarginalPrior(.02, .04, .07, "s/lap"),
        "MEDIUM": MarginalPrior(.05, .08, .12, "s/lap"),
        "SOFT": MarginalPrior(.10, .15, .20, "s/lap"),
    }
    performance = {
        "HARD": MarginalPrior(.2, .4, .6, "s"),
        "MEDIUM": MarginalPrior(-.1, 0, .1, "s"),
        "SOFT": MarginalPrior(-.7, -.5, -.3, "s"),
    }
    return FrozenPreRacePrior(
        1, 2026, 11, "pre", ("FP1", "FP2", "FP3"),
        "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z",
        "marginal_split_normal_approximation_no_joint_covariance",
        MarginalPrior(.01, .03, .05, "s/lap"), degradation, performance,
    )


def race_laps():
    return pd.DataFrame([
        {"SessionName": "R", "RunId": "1", "LapNumber": lap,
         "LapTime": 90 + .1 * lap, "Compound": "MEDIUM", "TyreLife": lap,
         "track_index": lap}
        for lap in range(1, 5)
    ])


def test_race_specific_model_has_six_coordinates_but_four_active_dimensions():
    result = run_race_specific_model(
        race_laps(), prior(), WeekendModelConfig(
            sample_count=1000, random_seed=7, fuel_rate_bounds=(0, .1),
            track_rate_bounds=(-.05, .05), default_degradation_bounds=(0, .3),
            default_compound_delta_bounds=(-1, 1), clean_lap_noise_sigma=.5,
        ), as_of_leader_lap=9, created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert len(result.sample_bank.dimension_names) == 6
    assert result.quality["active_likelihood_dimension_count"] == 4
    assert result.aggregate_snapshot.contributing_sessions == ("R",)
    assert result.quality["compound_evidence"]["HARD"]["status"] == "prior_only"
    assert result.quality["compound_evidence"]["MEDIUM"]["status"] == "weak_race_evidence"
    deltas = [row for row in result.aggregate_snapshot.parameters if row.parameter == "compound_delta"]
    assert {row.support_status for row in deltas} == {"prior_only"}
    assert all(row.p90 > row.p10 for row in deltas)


def test_unseen_compounds_remain_close_to_transfer_prior():
    result = run_race_specific_model(
        race_laps(), prior(), WeekendModelConfig(
            sample_count=10_000, random_seed=9, fuel_rate_bounds=(0, .1),
            track_rate_bounds=(-.05, .05), default_degradation_bounds=(0, .3),
            default_compound_delta_bounds=(-1, 1), clean_lap_noise_sigma=.5,
        ), as_of_leader_lap=9,
    )
    rows = {row.compound: row for row in result.aggregate_snapshot.parameters
            if row.parameter == "degradation"}
    assert abs(rows["HARD"].median - .04) < .02
    assert abs(rows["SOFT"].median - .15) < .02
    assert rows["HARD"].support_status == rows["SOFT"].support_status == "prior_only"
    assert np.all(np.diff(result.sample_bank.candidates[:, 1:4], axis=1) >= 0)


def test_per_driver_baseline_updates_connected_compound_performance():
    laps = pd.DataFrame(
        {
            "Driver": ["A"] * 8 + ["B"] * 8,
            "Stint": [1] * 4 + [2] * 4 + [1] * 4 + [2] * 4,
            "LapNumber": list(range(1, 9)) * 2,
            "LapTimeSeconds": (
                [90.0, 90.1, 90.2, 90.3, 90.7, 90.8, 90.9, 91.0]
                + [91.0, 91.1, 91.2, 91.3, 91.7, 91.8, 91.9, 92.0]
            ),
            "Compound": ["MEDIUM"] * 4 + ["HARD"] * 4 + ["MEDIUM"] * 4 + ["HARD"] * 4,
            "TyreLife": [1, 2, 3, 4] * 4,
            "SessionName": ["R"] * 16,
        }
    )
    result = run_race_specific_model(
        laps,
        prior(),
        WeekendModelConfig(sample_count=2048, random_seed=11),
        as_of_leader_lap=8,
        baseline_architecture=BASELINE_PER_DRIVER,
    )
    parameters = {(row.parameter, row.compound): row for row in result.aggregate_snapshot.parameters}
    assert parameters[("compound_delta", "HARD")].support_status == "race_informed"
    assert parameters[("compound_delta", "MEDIUM")].support_status == "race_informed"
    assert parameters[("compound_delta", "SOFT")].support_status == "prior_only"
    assert result.quality["active_likelihood_dimension_count"] == 6


def test_compound_connectivity_only_needs_a_minority_connector():
    prepared = pd.DataFrame(
        {
            "driver": ["A", "A", "B", "C"],
            "compound": ["MEDIUM", "HARD", "MEDIUM", "HARD"],
        }
    )
    result = compound_connectivity(prepared)
    assert result["fully_connected"]
    assert result["connector_drivers"] == 1
    assert result["drivers_by_compound_count"] == {1: 2, 2: 1, 3: 0}
