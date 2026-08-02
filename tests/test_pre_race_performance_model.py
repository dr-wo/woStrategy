import numpy as np
import pandas as pd
import pytest

from wostrategy.model.pre_race_performance import (
    WeekendModelConfig,
    aggregate_weekend_models,
    generate_fixed_sample_bank,
    run_weekend_model,
)


def synthetic_laps():
    rows = []
    rng = np.random.default_rng(7)
    for run in range(4):
        for lap in range(1, 13):
            compound = "MEDIUM" if lap <= 6 else "HARD"
            age = lap if lap <= 6 else lap - 6
            delta = -0.30 if compound == "MEDIUM" else 0.0
            degradation = 0.10 if compound == "MEDIUM" else 0.05
            rows.append(
                {
                    "SessionName": "FP2",
                    "RunId": f"run-{run}",
                    "LapNumber": lap + run * 12,
                    "LapTime": 80 + run + 0.01 * lap + delta + degradation * age
                    + rng.normal(0, 0.005),
                    "Compound": compound,
                    "TyreLife": age,
                    "track_index": lap,
                }
            )
    return pd.DataFrame(rows)


def config(seed=11):
    return WeekendModelConfig(
        sample_count=2500,
        random_seed=seed,
        fuel_rate_bounds=(0.0, 0.0),
        track_rate_bounds=(0.0, 0.02),
        degradation_bounds={"MEDIUM": (0.05, 0.15), "HARD": (0.0, 0.10)},
        compound_delta_bounds={"MEDIUM": (-0.5, -0.1)},
        clean_lap_noise_sigma=0.02,
    )


def test_fixed_bank_and_snapshot_are_deterministic_for_same_input():
    laps = synthetic_laps()
    bank = generate_fixed_sample_bank(["MEDIUM", "HARD"], config())
    first = run_weekend_model(laps, config(), bank, created_at=pd.Timestamp("2026-07-30", tz="UTC"))
    second = run_weekend_model(laps, config(), bank, created_at=pd.Timestamp("2026-07-30", tz="UTC"))
    assert first.analysis_id == second.analysis_id
    assert first.sample_bank_id == second.sample_bank_id
    assert [row.median for row in first.parameters] == [row.median for row in second.parameters]


def test_weekend_model_recovers_synthetic_parameters_with_uncertainty():
    snapshot = run_weekend_model(synthetic_laps(), config())
    values = {(row.parameter, row.compound): row for row in snapshot.parameters}
    assert values[("degradation", "MEDIUM")].median == pytest.approx(0.10, abs=0.04)
    assert values[("degradation", "HARD")].median == pytest.approx(0.05, abs=0.04)
    assert values[("compound_delta", "MEDIUM")].median == pytest.approx(-0.30, abs=0.12)
    assert all(row.p10 <= row.median <= row.p90 for row in snapshot.parameters)
    assert all(row.ess > 0 for row in snapshot.parameters)


def test_no_result_when_tyre_age_never_varies():
    laps = synthetic_laps()
    laps["TyreLife"] = 1
    with pytest.raises(ValueError, match="Tyre age never varies"):
        run_weekend_model(laps, config())


def test_aggregate_uses_latest_valid_parameter_and_records_provenance():
    first = run_weekend_model(synthetic_laps(), config(seed=1), source_scope="FP1")
    second = run_weekend_model(synthetic_laps(), config(seed=2), source_scope="FP2")
    aggregate = aggregate_weekend_models([first, second])
    assert aggregate.source_scope == "aggregate"
    assert aggregate.source_analysis_ids == (first.analysis_id, second.analysis_id)
    assert {row.session for row in aggregate.parameters} == {"FP2"}
