from __future__ import annotations

import numpy as np
import pandas as pd

from wostrategy.algorithm.sampling import LATIN_HYPERCUBE_SAMPLER
from wostrategy.algorithm.monte_carlo_race_performance import RacePerformanceAlgorithmResult
from wostrategy.analysis.race_performance_review import (
    MonteCarloRacePerformanceConfig,
    MonteCarloRacePerformanceResult,
    calculate_monte_carlo_race_performance_review,
    wet_lap_proportion_by_driver,
)
from wostrategy.script.race_performance_review import missing_clean_gap_columns
from wostrategy.script.race_performance_review import parse_race_selector
from wostrategy.script.race_performance_review import relative_team_pace_rows
from wostrategy.script.race_performance_review import is_no_clean_laps_error
from wostrategy.analysis.long_run_performance import select_consecutive_clean_air_runs


def test_monte_carlo_race_performance_review_returns_sample_outputs():
    result = calculate_monte_carlo_race_performance_review(
        _example_laps(),
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(sample_count=12, random_seed=1),
    )

    assert result != "Wet"
    assert len(result.sample_parameters) == 12
    assert set(result.sample_parameters.columns).issuperset(
        {
            "SampleId",
            "FuelRateSecondsPerLap",
            "TrackRateSecondsPerLap",
            "RMSESeconds",
            "Weight",
        }
    )
    assert set(result.compound_degradation["Compound"]) == {"HARD", "MEDIUM"}
    assert set(result.team_compound_degradation.columns).issuperset(
        {"Team", "Compound", "TeamCompoundDegSecondsPerLap", "VariationSecondsPerLap"}
    )
    assert set(result.baseline_pace.columns).issuperset(
        {"Team", "Driver", "CorrectedBaselinePaceSeconds"}
    )
    assert set(result.summaries) == {
        "fuel_rate",
        "track_rate",
        "compound_degradation",
        "team_compound_degradation",
        "baseline_pace",
    }
    assert result.summaries["fuel_rate"]["Median"].notna().all()


def test_monte_carlo_race_performance_rejects_negative_fuel_rate_bounds():
    try:
        calculate_monte_carlo_race_performance_review(
            _example_laps(),
            min_clean_air_laps=2,
            clean_mean_time_delta_seconds=3.0,
            clean_mean_time_delta_behind_seconds=1.0,
            quick_lap_threshold=1.05,
            config=MonteCarloRacePerformanceConfig(
                sample_count=4,
                fuel_rate_bounds=(-0.01, 0.05),
            ),
        )
    except ValueError as exc:
        assert "fuel_rate_bounds must be non-negative" in str(exc)
    else:
        raise AssertionError("negative fuel_rate_bounds should fail")


def test_select_consecutive_clean_air_runs_accepts_exact_minimum_length():
    laps = pd.DataFrame(
        {
            "Year": [2026, 2026, 2026, 2026],
            "Round": [5, 5, 5, 5],
            "SessionName": ["R", "R", "R", "R"],
            "Driver": ["HAD", "HAD", "HAD", "HAD"],
            "Stint": [4, 4, 4, 4],
            "LapNumber": [64, 65, 66, 67],
            "IsCleanAirLongRunLap": [True, True, True, True],
        }
    )

    result = select_consecutive_clean_air_runs(laps, min_clean_air_laps=4)

    assert result["LapNumber"].tolist() == [64, 65, 66, 67]


def test_team_compound_variation_uses_absolute_minimum_for_zero_base_degradation():
    result = calculate_monte_carlo_race_performance_review(
        _example_laps(),
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(
            sample_count=20,
            default_compound_degradation_bounds=(0.0, 0.0),
            team_variation_fraction=0.5,
            team_variation_absolute_min=0.01,
            random_seed=2,
        ),
    )

    variation = result.team_compound_degradation["VariationSecondsPerLap"].abs()
    assert np.isclose(result.compound_degradation["CompoundDegSecondsPerLap"], 0.0).all()
    assert (variation <= 0.010000001).all()
    assert (variation > 0.0).any()


def test_hot_track_temperature_enforces_compound_degradation_order():
    result = calculate_monte_carlo_race_performance_review(
        _three_compound_laps(),
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(
            sample_count=20,
            random_seed=9,
            track_temperature_celsius=35.0,
            degradation_order_track_temperature_celsius=20.0,
        ),
    )

    rates = result.compound_degradation.pivot(
        index="SampleId",
        columns="Compound",
        values="CompoundDegSecondsPerLap",
    )
    assert (rates["SOFT"] >= rates["MEDIUM"]).all()
    assert (rates["MEDIUM"] >= rates["HARD"]).all()
    assert result.sample_parameters["DegradationOrderEnforced"].all()
    assert set(result.sample_parameters["TrackTemperatureCelsius"]) == {35.0}


def test_track_temperature_below_threshold_does_not_enforce_degradation_order():
    result = calculate_monte_carlo_race_performance_review(
        _three_compound_laps(),
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(
            sample_count=8,
            random_seed=10,
            track_temperature_celsius=15.0,
            degradation_order_track_temperature_celsius=20.0,
        ),
    )

    assert not result.sample_parameters["DegradationOrderEnforced"].any()
    assert set(result.sample_parameters["TrackTemperatureCelsius"]) == {15.0}


def test_monte_carlo_race_performance_review_can_fit_team_baseline():
    result = calculate_monte_carlo_race_performance_review(
        _example_laps(),
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(
            sample_count=8,
            baseline_group="team",
            random_seed=3,
        ),
    )

    assert "Driver" not in result.baseline_pace.columns
    assert set(result.baseline_pace["Team"]) == {"Alfa", "Beta"}
    assert set(result.summaries["baseline_pace"]["Team"]) == {"Alfa", "Beta"}


def test_monte_carlo_race_performance_review_records_sampling_strategy():
    result = calculate_monte_carlo_race_performance_review(
        _example_laps(),
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(
            sample_count=8,
            random_seed=3,
            sampling_strategy=LATIN_HYPERCUBE_SAMPLER,
        ),
    )

    assert set(result.sample_parameters["SamplingStrategy"]) == {LATIN_HYPERCUBE_SAMPLER}


def test_monte_carlo_race_performance_review_keeps_mostly_dry_mixed_race():
    laps = _example_laps()
    laps.loc[laps["LapNumber"] == 1, "Compound"] = "INTERMEDIATE"

    result = calculate_monte_carlo_race_performance_review(
        laps,
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(sample_count=8, random_seed=4),
        wet_lap_proportion_skip_threshold=0.5,
    )

    assert result != "Wet"
    assert np.isclose(result.wet_lap_summary["WetLapProportion"], 0.25).all()
    assert set(result.clean_laps["Compound"]) == {"HARD", "MEDIUM"}


def test_monte_carlo_race_performance_review_keeps_clean_lap_with_missing_behind_gap():
    laps = _example_laps()
    mask = (laps["Driver"] == "AAA") & (laps["LapNumber"] == 2)
    laps.loc[mask, "MeanTimeDeltaToDriverAhead"] = 8.0
    laps.loc[mask, "MeanTimeDeltaToDriverBehind"] = pd.NA

    result = calculate_monte_carlo_race_performance_review(
        laps,
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(sample_count=8, random_seed=6),
    )

    aaa_laps = result.clean_laps.loc[result.clean_laps["Driver"] == "AAA", "LapNumber"]
    assert 2 in aaa_laps.tolist()


def test_monte_carlo_race_performance_review_keeps_clean_lap_with_missing_ahead_gap():
    laps = _example_laps()
    mask = (laps["Driver"] == "AAA") & (laps["LapNumber"].isin([2, 3]))
    laps.loc[mask, "MeanTimeDeltaToDriverAhead"] = pd.NA
    laps.loc[mask, "MeanTimeDeltaToDriverBehind"] = 8.0

    result = calculate_monte_carlo_race_performance_review(
        laps,
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(sample_count=8, random_seed=7),
    )

    aaa_laps = result.clean_laps.loc[result.clean_laps["Driver"] == "AAA", "LapNumber"]
    assert {2, 3}.issubset(set(aaa_laps.tolist()))


def test_monte_carlo_race_performance_review_rejects_lap_with_no_gap_evidence():
    laps = _example_laps()
    mask = (laps["Driver"] == "AAA") & (laps["LapNumber"].isin([2, 3]))
    laps.loc[mask, "MeanTimeDeltaToDriverAhead"] = pd.NA
    laps.loc[mask, "MeanTimeDeltaToDriverBehind"] = pd.NA

    result = calculate_monte_carlo_race_performance_review(
        laps,
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(sample_count=8, random_seed=8),
    )

    aaa_laps = result.clean_laps.loc[result.clean_laps["Driver"] == "AAA", "LapNumber"]
    assert 2 not in aaa_laps.tolist()
    assert 3 not in aaa_laps.tolist()


def test_monte_carlo_race_performance_review_skips_mostly_wet_race():
    laps = _example_laps()
    laps.loc[laps["LapNumber"].isin([1, 2, 3]), "Compound"] = "INTERMEDIATE"

    result = calculate_monte_carlo_race_performance_review(
        laps,
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(sample_count=8, random_seed=5),
        wet_lap_proportion_skip_threshold=0.5,
    )

    assert result == "Wet"


def test_wet_lap_proportion_by_driver_counts_driver_laps():
    laps = _example_laps()
    laps.loc[(laps["Driver"] == "AAA") & (laps["LapNumber"].isin([1, 2])), "Compound"] = "WET"

    wet_summary = wet_lap_proportion_by_driver(laps)
    aaa = wet_summary.loc[wet_summary["Driver"] == "AAA"].iloc[0]

    assert aaa["WetLapCount"] == 2
    assert aaa["TotalLapCount"] == 4
    assert aaa["WetLapProportion"] == 0.5


def test_script_gap_column_detection_requires_telemetry_gap_data():
    laps = _example_laps().drop(
        columns=["MeanTimeDeltaToDriverAhead", "MeanTimeDeltaToDriverBehind"]
    )

    missing = missing_clean_gap_columns(
        laps,
        clean_mean_time_delta_behind_seconds=1.0,
    )

    assert missing == ["MeanTimeDeltaToDriverAhead", "MeanTimeDeltaToDriverBehind"]


def test_script_gap_column_detection_rejects_empty_telemetry_gap_data():
    laps = _example_laps()
    laps["MeanTimeDeltaToDriverAhead"] = pd.NA
    laps["MeanTimeDeltaToDriverBehind"] = pd.NA

    missing = missing_clean_gap_columns(
        laps,
        clean_mean_time_delta_behind_seconds=1.0,
    )

    assert missing == [
        "MeanTimeDeltaToDriverAhead (all missing)",
        "MeanTimeDeltaToDriverBehind (all missing)",
    ]


def test_parse_race_selector_accepts_dash_and_bracket_ranges():
    assert parse_race_selector("1-3") == [1, 2, 3]
    assert parse_race_selector("[1, 3]") == [1, 2, 3]
    assert parse_race_selector("7") == [7]


def test_is_no_clean_laps_error_matches_only_expected_filter_failure():
    assert is_no_clean_laps_error(
        ValueError("No consecutive clean-air race laps matched the configured filters.")
    )
    assert not is_no_clean_laps_error(ValueError("No clean telemetry columns."))


def test_relative_team_pace_rows_use_reference_team_median_baseline():
    result = MonteCarloRacePerformanceResult(
        all_laps=pd.DataFrame({"EventName": ["Example GP"]}),
        clean_laps=pd.DataFrame(),
        wet_lap_summary=pd.DataFrame(),
        sample_parameters=pd.DataFrame(),
        compound_degradation=pd.DataFrame(),
        team_compound_degradation=pd.DataFrame(),
        baseline_pace=pd.DataFrame(),
        summaries={},
    )
    team_summary = pd.DataFrame(
        [
            {
                "Team": "Alfa",
                "P10": 89.5,
                "Median": 90.0,
                "P90": 90.5,
                "SampleCount": 10,
                "WeightSum": 4.0,
                "MeanRMSESeconds": 0.2,
                "TeamBaselineMode": "average-drivers",
            },
            {
                "Team": "Beta",
                "P10": 90.5,
                "Median": 91.8,
                "P90": 92.0,
                "SampleCount": 10,
                "WeightSum": 3.0,
                "MeanRMSESeconds": 0.25,
                "TeamBaselineMode": "average-drivers",
            },
        ]
    )

    rows = relative_team_pace_rows(
        result=result,
        team_baseline_summary=team_summary,
        year=2026,
        race=3,
        reference_team="Alfa",
    )

    beta = next(row for row in rows if row["Team"] == "Beta")
    assert beta["EventName"] == "Example GP"
    assert beta["PercentageToReferenceTeam"] == 102.0
    assert beta["CorrectedBaselinePaceSeconds"] == 91.8


def test_monte_carlo_race_performance_review_accepts_custom_algorithm():
    algorithm = FakeRacePerformanceAlgorithm()

    result = calculate_monte_carlo_race_performance_review(
        _example_laps(),
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        algorithm=algorithm,
    )

    assert algorithm.clean_lap_count == len(result.clean_laps)
    assert result.sample_parameters["FuelRateSecondsPerLap"].tolist() == [0.01]
    assert result.summaries["baseline_pace"]["Median"].tolist() == [90.5]


def _example_laps() -> pd.DataFrame:
    rows = []
    driver_specs = [
        ("Alfa", "AAA", "MEDIUM", 90.0),
        ("Alfa", "AAB", "MEDIUM", 90.2),
        ("Beta", "BBB", "HARD", 91.0),
        ("Beta", "BBC", "HARD", 91.1),
    ]
    for team, driver, compound, base_time in driver_specs:
        for lap_number in range(1, 5):
            rows.append(
                {
                    "Year": 2026,
                    "Round": 1,
                    "SessionName": "R",
                    "Driver": driver,
                    "Team": team,
                    "LapNumber": lap_number,
                    "LapTime": pd.to_timedelta(base_time + (0.08 * lap_number), unit="s"),
                    "Compound": compound,
                    "Stint": 1,
                    "StintLapNumber": lap_number,
                    "TyreLife": lap_number,
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "MeanTimeDeltaToDriverAhead": 5.0,
                    "MeanTimeDeltaToDriverBehind": 2.0,
                }
            )
    return pd.DataFrame(rows)


def _three_compound_laps() -> pd.DataFrame:
    rows = []
    driver_specs = [
        ("Alfa", "AAA", "SOFT", 90.0),
        ("Beta", "BBB", "MEDIUM", 91.0),
        ("Gamma", "CCC", "HARD", 92.0),
    ]
    for team, driver, compound, base_time in driver_specs:
        for lap_number in range(1, 4):
            rows.append(
                {
                    "Year": 2026,
                    "Round": 4,
                    "SessionName": "R",
                    "Driver": driver,
                    "Team": team,
                    "LapNumber": lap_number,
                    "LapTime": pd.to_timedelta(base_time + (0.05 * lap_number), unit="s"),
                    "Compound": compound,
                    "Stint": 1,
                    "StintLapNumber": lap_number,
                    "TyreLife": lap_number,
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "MeanTimeDeltaToDriverAhead": 5.0,
                    "MeanTimeDeltaToDriverBehind": 2.0,
                }
            )
    return pd.DataFrame(rows)


class FakeRacePerformanceAlgorithm:
    clean_lap_count: int | None = None

    def run(self, clean_laps: pd.DataFrame) -> RacePerformanceAlgorithmResult:
        self.clean_lap_count = len(clean_laps)
        return RacePerformanceAlgorithmResult(
            sample_parameters=pd.DataFrame(
                [
                    {
                        "SampleId": 0,
                        "FuelRateSecondsPerLap": 0.01,
                        "TrackRateSecondsPerLap": 0.02,
                        "RMSESeconds": 0.1,
                        "Score": 0.1,
                        "Weight": 1.0,
                    }
                ]
            ),
            compound_degradation=pd.DataFrame(
                [
                    {
                        "SampleId": 0,
                        "Compound": "MEDIUM",
                        "CompoundDegSecondsPerLap": 0.03,
                    }
                ]
            ),
            team_compound_degradation=pd.DataFrame(
                [
                    {
                        "SampleId": 0,
                        "Team": "Alfa",
                        "Compound": "MEDIUM",
                        "TeamCompoundDegSecondsPerLap": 0.04,
                    }
                ]
            ),
            baseline_pace=pd.DataFrame(
                [
                    {
                        "SampleId": 0,
                        "BaselineGroup": "team",
                        "BaselineGroupKey": "Alfa",
                        "Team": "Alfa",
                        "CorrectedBaselinePaceSeconds": 90.5,
                    }
                ]
            ),
        )
