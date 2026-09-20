from pathlib import Path

import numpy as np
import pandas as pd

from wostrategy.script.race_performance_weight_predict import (
    all_team_plot_rows,
    load_weight_prediction_row,
    total_race_laps_from_clean_laps,
)


def test_total_race_laps_uses_fuel_proxy_when_available():
    clean_laps = pd.DataFrame(
        {
            "LapNumber": [10, 20, 30],
            "FuelProxyLapsRemaining": [40, 30, 20],
        }
    )

    assert total_race_laps_from_clean_laps(clean_laps) == 50.0


def test_load_weight_prediction_row_keeps_heuristic_value_but_labels_equivalent_rate(
    tmp_path,
):
    prefix = tmp_path / "race_performance_2026_5_R"
    pd.DataFrame(
        [
            {
                "Team": "Mercedes",
                "P10": 89.0,
                "Median": 90.0,
                "P90": 91.0,
                "SampleCount": 100,
                "WeightSum": 10.0,
                "MeanRMSESeconds": 0.3,
                "TeamBaselineMode": "average-drivers",
            },
            {
                "Team": "Ferrari",
                "P10": 94.0,
                "Median": 95.0,
                "P90": 96.0,
                "SampleCount": 100,
                "WeightSum": 10.0,
                "MeanRMSESeconds": 0.3,
                "TeamBaselineMode": "average-drivers",
            }
        ]
    ).to_csv(f"{prefix}_team_baseline_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "P10": 0.03,
                "Median": 0.05,
                "P90": 0.07,
                "SampleCount": 100,
                "WeightSum": 10.0,
            }
        ]
    ).to_csv(f"{prefix}_summary_fuel_rate.csv", index=False)
    pd.DataFrame(
        {
            "LapNumber": [1, 50],
            "FuelProxyLapsRemaining": [49, 0],
            "EventName": ["Example GP", "Example GP"],
        }
    ).to_csv(f"{prefix}_clean_laps.csv", index=False)

    row = load_weight_prediction_row(
        input_dir=Path(tmp_path),
        year=2026,
        race=5,
        session="R",
        team="Mercedes",
        reference_team="Ferrari",
        weight_delta_kg=5.0,
        full_fuel_weight_kg=100.0,
    )

    assert row["TotalRaceLaps"] == 50.0
    assert row["FuelKgPerLap"] == 2.0
    assert row["FuelProxyRateSecondsPerRemainingLap"] == 0.05
    assert row["AssumedKgPerFuelProxyLap"] == 2.0
    assert row["EquivalentFuelSecondsPerKg"] == 0.025
    assert row["FuelSecondsPerKg"] == 0.025
    assert row["FuelSecondsPerKgSemanticRole"] == "legacy_alias"
    assert row["FuelWeightProjectionMethod"] == (
        "fuel_proxy_rate / assumed_kg_per_proxy_lap"
    )
    assert row["FuelWeightProjectionCalibrationStatus"] == (
        "heuristic_equivalent_not_physical_calibration"
    )
    assert row["FuelTrackConfoundingStatus"] == "not_resolved_by_conversion"
    assert np.isclose(row["ProjectedBaselineSeconds"], 90.125)
    assert row["ProjectedDeltaSeconds"] == 0.125
    assert row["ReferenceTeam"] == "Ferrari"
    assert row["ReferenceBaselineSeconds"] == 95.0
    assert np.isclose(row["OriginalPercentageToReference"], 90.0 / 95.0 * 100.0)
    assert np.isclose(row["ProjectedPercentageToReference"], 90.125 / 95.0 * 100.0)
    assert row["EventName"] == "Example GP"


def test_weight_prediction_direct_display_calls_equivalent_rate_heuristic(
    tmp_path, capsys,
):
    prefix = tmp_path / "race_performance_2026_5_R"
    pd.DataFrame([
        {"Team": "Mercedes", "P10": 89.0, "Median": 90.0, "P90": 91.0,
         "TeamBaselineMode": "average-drivers"},
    ]).to_csv(f"{prefix}_team_baseline_summary.csv", index=False)
    pd.DataFrame([{"Median": .05}]).to_csv(
        f"{prefix}_summary_fuel_rate.csv", index=False
    )
    pd.DataFrame({
        "LapNumber": [1, 50], "FuelProxyLapsRemaining": [49, 0],
        "EventName": ["Example GP", "Example GP"],
    }).to_csv(f"{prefix}_clean_laps.csv", index=False)

    from wostrategy.script.race_performance_weight_predict import (
        run_race_performance_weight_prediction,
    )

    summary = run_race_performance_weight_prediction(
        year=2026, races=[5], session="R", team="Mercedes",
        reference_team="Mercedes", weight_delta_kg=5.0,
        full_fuel_weight_kg=100.0, input_dir=tmp_path,
        output_path=tmp_path / "weight.png", show=False,
    )

    output = capsys.readouterr().out
    assert "heuristic equivalent" in output
    assert "not a physical fuel-sensitivity calibration" in output
    assert "EquivalentFuelSecondsPerKg" in output
    assert summary["ProjectedDeltaSeconds"].iloc[0] == 0.125


def test_all_team_plot_rows_include_actual_teams_and_projected_team():
    rows = all_team_plot_rows(
        team_summary=pd.DataFrame(
            [
                {"Team": "Mercedes", "Median": 90.0},
                {"Team": "Ferrari", "Median": 95.0},
            ]
        ),
        year=2026,
        race=5,
        session="R",
        event_name="Example GP",
        reference_team="Mercedes",
        reference_baseline=90.0,
        projected_team="Ferrari",
        projected_baseline=94.0,
        weight_delta_kg=-2.0,
    )

    actual = [row for row in rows if row["Series"] == "actual"]
    projected = [row for row in rows if row["Series"] == "projected"]
    assert {row["Team"] for row in actual} == {"Mercedes", "Ferrari"}
    assert len(projected) == 1
    assert projected[0]["Team"] == "Ferrari"
    assert projected[0]["WeightDeltaKg"] == -2.0
    assert np.isclose(projected[0]["PercentageToReference"], 94.0 / 90.0 * 100.0)
