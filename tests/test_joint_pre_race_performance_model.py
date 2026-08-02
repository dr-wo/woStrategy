import numpy as np
import pandas as pd

from wostrategy.model.pre_race_performance import (
    WeekendModelConfig,
    derive_race_seed,
    generate_joint_sample_bank,
    run_joint_weekend_model,
)


def session_laps(session, fuel_rate, track_rate, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for run in range(3):
        for run_lap in range(1, 9):
            compound = "MEDIUM" if run_lap <= 4 else "HARD"
            tyre_age = run_lap if run_lap <= 4 else run_lap - 4
            degradation = 0.09 if compound == "MEDIUM" else 0.04
            delta = -0.25 if compound == "MEDIUM" else 0.0
            track_index = run * 12 + run_lap
            lap_time = (
                80 + run + fuel_rate * run_lap + track_rate * track_index
                + degradation * tyre_age + delta + rng.normal(0, 0.003)
            )
            rows.append(
                {
                    "SessionName": session,
                    "RunId": f"{run}",
                    "LapNumber": track_index,
                    "LapTime": lap_time,
                    "Compound": compound,
                    "TyreLife": tyre_age,
                    "track_index": track_index,
                }
            )
    return pd.DataFrame(rows)


def config():
    return WeekendModelConfig(
        sample_count=1200,
        random_seed=41,
        fuel_rate_bounds=(0.0, 0.08),
        track_rate_bounds=(-0.02, 0.02),
        default_degradation_bounds=(0.0, 0.15),
        default_compound_delta_bounds=(-0.5, 0.1),
        clean_lap_noise_sigma=0.02,
    )


def test_joint_bank_has_one_shared_fuel_and_session_specific_dimensions():
    bank = generate_joint_sample_bank(
        ("FP1", "FP2", "FP3"),
        {name: ("SOFT", "MEDIUM", "HARD") for name in ("FP1", "FP2", "FP3")},
        config(),
    )
    assert bank.dimension_names.count("fuel_rate") == 1
    assert "track_rate:FP1" in bank.dimension_names
    assert "track_rate:FP2" in bank.dimension_names
    assert "degradation:FP3:HARD" in bank.dimension_names


def test_joint_bank_can_enforce_dry_compound_pace_and_degradation_order():
    ordered_config = WeekendModelConfig(
        sample_count=200,
        random_seed=41,
        reference_compound="MEDIUM",
        default_degradation_bounds=(0.0, 0.30),
        default_compound_delta_bounds=(-1.0, 1.0),
        enforce_compound_order=True,
    )
    bank = generate_joint_sample_bank(
        ("FP1",),
        {"FP1": ("SOFT", "MEDIUM", "HARD")},
        ordered_config,
    )
    values = {
        name: bank.candidates[:, index]
        for index, name in enumerate(bank.dimension_names)
    }

    assert np.all(
        values["degradation:FP1:SOFT"]
        >= values["degradation:FP1:MEDIUM"]
    )
    assert np.all(
        values["degradation:FP1:MEDIUM"]
        >= values["degradation:FP1:HARD"]
    )
    assert np.all(values["compound_delta:FP1:SOFT"] <= 0.0)
    assert np.all(values["compound_delta:FP1:HARD"] >= 0.0)


def test_joint_cost_is_additive_and_bank_stays_fixed_as_sessions_arrive():
    fp1 = session_laps("FP1", 0.035, -0.006, 1)
    fp2 = session_laps("FP2", 0.035, 0.004, 2)
    first = run_joint_weekend_model({"FP1": fp1}, config())
    second = run_joint_weekend_model({"FP1": fp1, "FP2": fp2}, config())

    assert first.sample_bank.sample_bank_id == second.sample_bank.sample_bank_id
    assert first.contributing_sessions == ("FP1",)
    assert first.excluded_sessions["FP2"] == "unavailable"
    assert second.contributing_sessions == ("FP1", "FP2")
    costs = second.candidate_costs
    np.testing.assert_allclose(costs["TotalCost"], costs["FP1Cost"] + costs["FP2Cost"])
    fuel_rows = [row for row in second.aggregate_snapshot.parameters if row.parameter == "fuel_rate"]
    assert len(fuel_rows) == 1
    assert fuel_rows[0].session == "FP1+FP2"
    assert {row.session for row in second.aggregate_snapshot.parameters if row.parameter == "track_rate"} == {"FP1", "FP2"}


def test_race_seed_is_stable_for_event_and_changes_by_race():
    assert derive_race_seed(2026, 8) == derive_race_seed(2026, 8)
    assert derive_race_seed(2026, 8) != derive_race_seed(2026, 9)
    assert derive_race_seed(2026, 8) != 2026
