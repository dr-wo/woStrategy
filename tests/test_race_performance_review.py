from __future__ import annotations

import numpy as np
import pandas as pd

from wostrategy.algorithm.sampling import LATIN_HYPERCUBE_SAMPLER
from wostrategy.algorithm.monte_carlo_race_performance import RacePerformanceAlgorithmResult
from wostrategy.model.tyre_degragation import TYRE_AGE_LAPS_COLUMN
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
from wostrategy.script.race_performance_review import load_cached_monte_carlo_outputs
from wostrategy.script.race_performance_review import format_effective_sample_size
from wostrategy.script.race_performance_review import sample_diagnostics_summary
from wostrategy.plots.race_performance import plot_relative_team_pace
from wostrategy.analysis.long_run_performance import (
    select_clean_air_stints_as_whole,
    select_consecutive_clean_air_runs,
)


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
    assert set(result.compound_delta["Compound"]) == {"HARD", "MEDIUM"}
    assert result.compound_delta.loc[
        result.compound_delta["Compound"] == "HARD",
        "CompoundDeltaSeconds",
    ].eq(0.0).all()
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
        "compound_delta",
        "team_compound_degradation",
        "baseline_pace",
    }
    assert result.summaries["fuel_rate"]["Median"].notna().all()


def test_best_rmse_relative_weight_strategy_normalizes_weights():
    result = calculate_monte_carlo_race_performance_review(
        _example_laps(),
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(
            sample_count=12,
            random_seed=14,
            clean_lap_noise_sigma=0.5,
            weight_strategy="best-rmse-relative",
            weight_effective_sample_count=6.0,
        ),
    )

    samples = result.sample_parameters
    best_row = samples.loc[samples["RMSESeconds"].idxmin()]
    max_weight_row = samples.loc[samples["Weight"].idxmax()]
    assert np.isclose(samples["Weight"].sum(), 1.0)
    assert max_weight_row["SampleId"] == best_row["SampleId"]
    assert set(samples["WeightStrategy"]) == {"best-rmse-relative"}
    assert set(samples["WeightEffectiveSampleCount"]) == {6.0}


def test_gaussian_weight_strategy_keeps_existing_unnormalized_weights():
    result = calculate_monte_carlo_race_performance_review(
        _example_laps(),
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(
            sample_count=8,
            random_seed=15,
            clean_lap_noise_sigma=0.5,
            weight_strategy="gaussian",
        ),
    )

    samples = result.sample_parameters
    expected = np.exp(-(samples["RMSESeconds"] ** 2) / (2 * 0.5**2))
    assert np.allclose(samples["Weight"], expected)
    assert not np.isclose(samples["Weight"].sum(), 1.0)


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


def test_limit_negative_track_correction_clamps_track_rate_samples_to_non_negative():
    result = calculate_monte_carlo_race_performance_review(
        _example_laps(),
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(
            sample_count=20,
            random_seed=17,
            track_rate_bounds=(-0.05, 0.05),
            limit_negative_track_correction=True,
        ),
    )

    samples = result.sample_parameters
    assert samples["TrackRateSecondsPerLap"].ge(0.0).all()
    assert samples["TrackRateSecondsPerLap"].gt(0.0).any()
    assert samples["LimitNegativeTrackCorrection"].eq(True).all()


def test_limit_negative_track_correction_rejects_fully_negative_track_bounds():
    try:
        calculate_monte_carlo_race_performance_review(
            _example_laps(),
            min_clean_air_laps=2,
            clean_mean_time_delta_seconds=3.0,
            clean_mean_time_delta_behind_seconds=1.0,
            quick_lap_threshold=1.05,
            config=MonteCarloRacePerformanceConfig(
                sample_count=4,
                track_rate_bounds=(-0.05, -0.01),
                limit_negative_track_correction=True,
            ),
        )
    except ValueError as exc:
        assert "track_rate_bounds upper bound must be non-negative" in str(exc)
    else:
        raise AssertionError("fully negative track_rate_bounds should fail")


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


def test_select_clean_air_stints_as_whole_ignores_clean_lap_gaps():
    laps = pd.DataFrame(
        {
            "Year": [2026] * 6,
            "Round": [5] * 6,
            "SessionName": ["R"] * 6,
            "Driver": ["VER"] * 6,
            "Stint": [2] * 6,
            "LapNumber": [34, 35, 36, 37, 38, 39],
            "IsCleanAirLongRunLap": [True, False, True, True, False, True],
        }
    )

    consecutive = select_consecutive_clean_air_runs(laps, min_clean_air_laps=4)
    whole_stint = select_clean_air_stints_as_whole(laps, min_clean_air_laps=4)

    assert consecutive.empty
    assert whole_stint["LapNumber"].tolist() == [34, 36, 37, 39]
    assert whole_stint["LongRunId"].nunique() == 1
    assert whole_stint["LongRunLapNumber"].tolist() == [1, 2, 3, 4]


def test_monte_carlo_race_performance_review_can_treat_stint_as_whole():
    laps = _example_laps()
    laps.loc[laps["LapNumber"] == 2, "MeanTimeDeltaToDriverAhead"] = 1.0

    try:
        calculate_monte_carlo_race_performance_review(
            laps,
            min_clean_air_laps=3,
            clean_mean_time_delta_seconds=3.0,
            clean_mean_time_delta_behind_seconds=1.0,
            quick_lap_threshold=1.05,
            config=MonteCarloRacePerformanceConfig(sample_count=4, random_seed=12),
        )
    except ValueError as exc:
        assert "No consecutive clean-air race laps matched" in str(exc)
    else:
        raise AssertionError("consecutive mode should reject split clean stints")

    result = calculate_monte_carlo_race_performance_review(
        laps,
        min_clean_air_laps=3,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        treat_stint_as_whole=True,
        config=MonteCarloRacePerformanceConfig(sample_count=4, random_seed=12),
    )

    assert result != "Wet"
    assert len(result.clean_laps) == 12
    assert result.clean_laps.groupby(["Driver", "Stint"])["LongRunId"].nunique().eq(1).all()


def test_monte_carlo_race_performance_review_excludes_non_green_track_status():
    laps = _example_laps()
    laps["TrackStatus"] = "1"
    laps.loc[
        (laps["Driver"] == "AAA") & (laps["LapNumber"] == 2),
        "TrackStatus",
    ] = "16"
    laps.loc[
        (laps["Driver"] == "AAB") & (laps["LapNumber"] == 3),
        "TrackStatus",
    ] = "671"

    result = calculate_monte_carlo_race_performance_review(
        laps,
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(sample_count=4, random_seed=18),
    )

    assert result != "Wet"
    assert result.clean_laps["TrackStatus"].eq("1").all()
    excluded = result.clean_laps.loc[
        result.clean_laps["Driver"].isin(["AAA", "AAB"]),
        ["Driver", "LapNumber"],
    ]
    assert ("AAA", 2) not in set(excluded.itertuples(index=False, name=None))
    assert ("AAB", 3) not in set(excluded.itertuples(index=False, name=None))


def test_monte_carlo_race_performance_review_uses_stint_tyre_age_by_default():
    laps = _example_laps()
    laps["TyreLife"] = laps["StintLapNumber"] + 3

    result = calculate_monte_carlo_race_performance_review(
        laps,
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(sample_count=4, random_seed=16),
    )

    assert set(result.clean_laps[TYRE_AGE_LAPS_COLUMN]) == {0, 1, 2, 3}


def test_monte_carlo_race_performance_review_can_use_overall_tyre_age():
    laps = _example_laps()
    laps["TyreLife"] = laps["StintLapNumber"] + 3

    result = calculate_monte_carlo_race_performance_review(
        laps,
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        tyre_age_mode="overall",
        config=MonteCarloRacePerformanceConfig(sample_count=4, random_seed=16),
    )

    assert set(result.clean_laps[TYRE_AGE_LAPS_COLUMN]) == {3, 4, 5, 6}


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


def test_compound_delta_is_sampled_relative_to_reference_compound():
    result = calculate_monte_carlo_race_performance_review(
        _example_laps(),
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(
            sample_count=10,
            default_compound_delta_bounds=(-0.5, 0.5),
            compound_delta_reference="HARD",
            random_seed=11,
        ),
    )

    deltas = result.compound_delta.pivot(
        index="SampleId",
        columns="Compound",
        values="CompoundDeltaSeconds",
    )
    assert deltas["HARD"].eq(0.0).all()
    assert (deltas["MEDIUM"].abs() <= 0.500000001).all()
    assert deltas["MEDIUM"].abs().gt(0.0).any()
    assert set(result.summaries["compound_delta"]["Compound"]) == {"HARD", "MEDIUM"}


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


def test_hot_track_temperature_enforces_compound_delta_order():
    result = calculate_monte_carlo_race_performance_review(
        _three_compound_laps(),
        min_clean_air_laps=2,
        clean_mean_time_delta_seconds=3.0,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.05,
        config=MonteCarloRacePerformanceConfig(
            sample_count=20,
            random_seed=13,
            track_temperature_celsius=35.0,
            degradation_order_track_temperature_celsius=20.0,
            default_compound_delta_bounds=(-1.0, 1.0),
            compound_delta_reference="HARD",
        ),
    )

    deltas = result.compound_delta.pivot(
        index="SampleId",
        columns="Compound",
        values="CompoundDeltaSeconds",
    )
    assert (deltas["SOFT"] <= deltas["MEDIUM"]).all()
    assert (deltas["MEDIUM"] <= deltas["HARD"]).all()
    assert deltas["HARD"].eq(0.0).all()


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
        team_baseline_summary=team_summary,
        year=2026,
        race=3,
        reference_team="Alfa",
        event_name="Example GP",
        sample_diagnostics=pd.DataFrame({"WeightedRMSESeconds": [0.23]}),
    )

    beta = next(row for row in rows if row["Team"] == "Beta")
    assert beta["EventName"] == "Example GP"
    assert beta["PercentageToReferenceTeam"] == 102.0
    assert beta["CorrectedBaselinePaceSeconds"] == 91.8
    assert beta["WeightedRMSESeconds"] == 0.23
    assert beta["RaceTeamCount"] == 2


def test_relative_team_pace_rows_use_sample_paired_relative_baseline_when_available():
    team_summary = pd.DataFrame(
        [
            {
                "Team": "Alfa",
                "P10": 100.0,
                "Median": 200.0,
                "P90": 300.0,
                "SampleCount": 3,
                "WeightSum": 3.0,
                "MeanRMSESeconds": 0.2,
                "TeamBaselineMode": "best-driver",
            },
            {
                "Team": "Beta",
                "P10": 100.0,
                "Median": 200.0,
                "P90": 300.0,
                "SampleCount": 3,
                "WeightSum": 3.0,
                "MeanRMSESeconds": 0.25,
                "TeamBaselineMode": "best-driver",
            },
        ]
    )
    team_samples = pd.DataFrame(
        [
            {
                "SampleId": 1,
                "Team": "Alfa",
                "CorrectedBaselinePaceSeconds": 100.0,
                "Weight": 1.0,
            },
            {
                "SampleId": 2,
                "Team": "Alfa",
                "CorrectedBaselinePaceSeconds": 200.0,
                "Weight": 1.0,
            },
            {
                "SampleId": 3,
                "Team": "Alfa",
                "CorrectedBaselinePaceSeconds": 300.0,
                "Weight": 1.0,
            },
            {
                "SampleId": 1,
                "Team": "Beta",
                "CorrectedBaselinePaceSeconds": 300.0,
                "Weight": 1.0,
            },
            {
                "SampleId": 2,
                "Team": "Beta",
                "CorrectedBaselinePaceSeconds": 100.0,
                "Weight": 1.0,
            },
            {
                "SampleId": 3,
                "Team": "Beta",
                "CorrectedBaselinePaceSeconds": 200.0,
                "Weight": 1.0,
            },
        ]
    )

    rows = relative_team_pace_rows(
        team_baseline_summary=team_summary,
        team_baseline_samples=team_samples,
        year=2026,
        race=3,
        reference_team="Alfa",
        event_name="Example GP",
    )

    beta = next(row for row in rows if row["Team"] == "Beta")
    assert beta["CorrectedBaselinePaceSeconds"] == 200.0
    assert beta["RelativeToReferenceSeconds"] == -100.0
    assert beta["PercentageToReferenceTeam"] == 200.0 / 3.0


def test_plot_relative_team_pace_can_draw_rmse_background_and_low_team_hatch():
    summary = pd.DataFrame(
        [
            {
                "Year": 2026,
                "Race": 1,
                "Team": f"Team {team}",
                "PercentageToReferenceTeam": 100.0 + team,
                "WeightedRMSESeconds": 0.25,
            }
            for team in range(5)
        ]
        + [
            {
                "Year": 2026,
                "Race": 2,
                "Team": "Alfa",
                "PercentageToReferenceTeam": 101.0,
                "WeightedRMSESeconds": 0.8,
            },
            {
                "Year": 2026,
                "Race": 2,
                "Team": "Beta",
                "PercentageToReferenceTeam": 102.0,
                "WeightedRMSESeconds": 0.8,
            },
        ]
    )

    fig, ax = plot_relative_team_pace(
        summary,
        reference_team="Team 0",
        plot_rmse_background=True,
    )

    assert len(fig.axes) == 2
    assert any(patch.get_hatch() == "///" for patch in ax.patches)


def test_sample_diagnostics_summary_reports_weighted_quality_metrics():
    sample_parameters = pd.DataFrame(
        {
            "SampleId": [0, 1, 2],
            "RMSESeconds": [0.1, 0.2, 0.5],
            "Weight": [1.0, 0.5, 0.0],
        }
    )

    diagnostics = sample_diagnostics_summary(sample_parameters)
    row = diagnostics.iloc[0]

    expected_weighted_rmse = np.sqrt(((0.1**2 * 1.0) + (0.2**2 * 0.5)) / 1.5)
    expected_ess = 1.5**2 / (1.0**2 + 0.5**2)
    assert row["SampleCount"] == 3
    assert row["BestRMSESeconds"] == 0.1
    assert np.isclose(row["WeightedRMSESeconds"], expected_weighted_rmse)
    assert np.isclose(row["EffectiveSampleSize"], expected_ess)
    assert np.isclose(row["EffectiveSampleFraction"], expected_ess / 3)


def test_format_effective_sample_size_prints_ess_summary():
    diagnostics = pd.DataFrame(
        [
            {
                "EffectiveSampleSize": 12.345,
                "EffectiveSampleFraction": 0.12345,
            }
        ]
    )

    assert format_effective_sample_size(diagnostics) == (
        "Effective sample size: ESS=12.35, fraction=0.123"
    )


def test_load_cached_monte_carlo_outputs_reads_team_summary_and_diagnostics(tmp_path):
    prefix = tmp_path / "race_performance_2026_3_R"
    pd.DataFrame(
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
            }
        ]
    ).to_csv(f"{prefix}_team_baseline_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "SampleCount": 10,
                "WeightedRMSESeconds": 0.2,
                "EffectiveSampleSize": 5.0,
            }
        ]
    ).to_csv(f"{prefix}_sample_diagnostics.csv", index=False)
    pd.DataFrame({"EventName": ["Example GP"]}).to_csv(
        f"{prefix}_clean_laps.csv",
        index=False,
    )

    cached = load_cached_monte_carlo_outputs(
        year=2026,
        race=3,
        session="R",
        output_dir=tmp_path,
        team_baseline_mode="average-drivers",
    )

    assert cached is not None
    assert cached["team_baseline_summary"]["Team"].tolist() == ["Alfa"]
    assert cached["sample_diagnostics"]["WeightedRMSESeconds"].tolist() == [0.2]
    assert cached["event_name"] == "Example GP"


def test_load_cached_monte_carlo_outputs_reads_degradation_outputs(tmp_path):
    prefix = tmp_path / "race_performance_2026_3_R"
    pd.DataFrame(
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
            }
        ]
    ).to_csv(f"{prefix}_team_baseline_summary.csv", index=False)
    pd.DataFrame(
        [{"SampleId": 0, "Compound": "MEDIUM", "CompoundDegSecondsPerLap": 0.04}]
    ).to_csv(f"{prefix}_compound_degradation.csv", index=False)
    pd.DataFrame(
        [{"SampleId": 0, "Compound": "MEDIUM", "CompoundDeltaSeconds": -0.2}]
    ).to_csv(f"{prefix}_compound_delta.csv", index=False)
    pd.DataFrame(
        [
            {
                "SampleId": 0,
                "Team": "Alfa",
                "Compound": "MEDIUM",
                "TeamCompoundDegSecondsPerLap": 0.05,
            }
        ]
    ).to_csv(f"{prefix}_team_compound_degradation.csv", index=False)
    pd.DataFrame(
        [{"Compound": "MEDIUM", "Median": 0.04, "SampleCount": 1, "WeightSum": 1.0}]
    ).to_csv(f"{prefix}_summary_compound_degradation.csv", index=False)
    pd.DataFrame(
        [{"Compound": "MEDIUM", "Median": -0.2, "SampleCount": 1, "WeightSum": 1.0}]
    ).to_csv(f"{prefix}_summary_compound_delta.csv", index=False)
    pd.DataFrame(
        [
            {
                "Team": "Alfa",
                "Compound": "MEDIUM",
                "Median": 0.05,
                "SampleCount": 1,
                "WeightSum": 1.0,
            }
        ]
    ).to_csv(f"{prefix}_summary_team_compound_degradation.csv", index=False)

    cached = load_cached_monte_carlo_outputs(
        year=2026,
        race=3,
        session="R",
        output_dir=tmp_path,
        team_baseline_mode="average-drivers",
    )

    assert cached is not None
    assert cached["compound_degradation"]["CompoundDegSecondsPerLap"].tolist() == [0.04]
    assert cached["compound_delta"]["CompoundDeltaSeconds"].tolist() == [-0.2]
    assert cached["team_compound_degradation"]["Team"].tolist() == ["Alfa"]
    assert cached["summary_compound_degradation"]["Median"].tolist() == [0.04]
    assert cached["summary_compound_delta"]["Median"].tolist() == [-0.2]
    assert cached["summary_team_compound_degradation"]["Median"].tolist() == [0.05]


def test_load_cached_monte_carlo_outputs_rebuilds_best_driver_from_driver_baselines(tmp_path):
    prefix = tmp_path / "race_performance_2026_3_R"
    pd.DataFrame(
        [
            {
                "Team": "Alfa",
                "P10": 90.5,
                "Median": 90.5,
                "P90": 90.5,
                "SampleCount": 1,
                "WeightSum": 1.0,
                "MeanRMSESeconds": 0.2,
                "TeamBaselineMode": "average-drivers",
            }
        ]
    ).to_csv(f"{prefix}_team_baseline_summary.csv", index=False)
    pd.DataFrame(
        [
            {"SampleId": 0, "Weight": 1.0, "RMSESeconds": 0.2},
            {"SampleId": 1, "Weight": 1.0, "RMSESeconds": 0.3},
        ]
    ).to_csv(f"{prefix}_sample_parameters.csv", index=False)
    pd.DataFrame(
        [
            {
                "SampleId": 0,
                "BaselineGroup": "driver",
                "BaselineGroupKey": "Alfa||AAA",
                "Team": "Alfa",
                "Driver": "AAA",
                "CorrectedBaselinePaceSeconds": 90.0,
            },
            {
                "SampleId": 0,
                "BaselineGroup": "driver",
                "BaselineGroupKey": "Alfa||AAB",
                "Team": "Alfa",
                "Driver": "AAB",
                "CorrectedBaselinePaceSeconds": 91.0,
            },
            {
                "SampleId": 1,
                "BaselineGroup": "driver",
                "BaselineGroupKey": "Alfa||AAA",
                "Team": "Alfa",
                "Driver": "AAA",
                "CorrectedBaselinePaceSeconds": 89.0,
            },
            {
                "SampleId": 1,
                "BaselineGroup": "driver",
                "BaselineGroupKey": "Alfa||AAB",
                "Team": "Alfa",
                "Driver": "AAB",
                "CorrectedBaselinePaceSeconds": 92.0,
            },
        ]
    ).to_csv(f"{prefix}_baseline_pace.csv", index=False)

    cached = load_cached_monte_carlo_outputs(
        year=2026,
        race=3,
        session="R",
        output_dir=tmp_path,
        team_baseline_mode="best-driver",
    )

    assert cached is not None
    samples = cached["team_baseline_samples"].sort_values("SampleId")
    assert samples["CorrectedBaselinePaceSeconds"].tolist() == [90.0, 89.0]
    assert samples["TeamBaselineMode"].unique().tolist() == ["best-driver"]
    assert cached["team_baseline_summary"]["TeamBaselineMode"].unique().tolist() == [
        "best-driver"
    ]


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
            compound_delta=pd.DataFrame(
                [
                    {
                        "SampleId": 0,
                        "Compound": "MEDIUM",
                        "CompoundDeltaSeconds": 0.0,
                        "CompoundDeltaReference": "MEDIUM",
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
