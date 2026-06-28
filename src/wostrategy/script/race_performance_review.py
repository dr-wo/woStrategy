from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from wostrategy.algorithm.monte_carlo_race_performance import (
    MonteCarloRacePerformanceAlgorithm,
    MonteCarloRacePerformanceConfig,
    WEIGHT_STRATEGIES,
    WEIGHT_STRATEGY_GAUSSIAN,
)
from wostrategy.algorithm.sampling import (
    LATIN_HYPERCUBE_SAMPLER,
    SAMPLING_STRATEGIES,
)
from wostrategy.analysis.race_performance_review import (
    MonteCarloRacePerformanceResult,
    calculate_monte_carlo_race_performance_review,
    wet_lap_proportion_by_driver,
)
from wostrategy.analysis.long_run_performance import (
    TYRE_AGE_MODE_STINT,
    TYRE_AGE_MODES,
)
from wostrategy.plots.race_performance import (
    RacePerformancePlotter,
    result_output_path as race_result_output_path,
    save_relative_team_pace_figures,
)
from wostrategy.tools import load_all_session_laps_with_telemetry_gap_summary

TEAM_MODE_BEST_DRIVER = "best-driver"
TEAM_MODE_AVERAGE_DRIVERS = "average-drivers"
TEAM_MODE_DIRECT_TEAM = "direct-team"
TEAM_MODES = (TEAM_MODE_BEST_DRIVER, TEAM_MODE_AVERAGE_DRIVERS, TEAM_MODE_DIRECT_TEAM)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "cache" / "race_performance_review"


SCRIPT_CONFIG = {
    "year": 2026,
    "race": "8",
    "session": "R",
    "sample_count": 80000,
    "sampling_strategy": LATIN_HYPERCUBE_SAMPLER,
    "fuel_rate_bounds": (0.0, 0.1),
    "track_rate_bounds": (-0.05, 0.05),
    "limit_negative_track_correction": True,
    "default_compound_degradation_bounds": (0.0, 0.5),
    "compound_degradation_bounds_json": None,
    "default_compound_delta_bounds": (-1.2, 0.5),
    "compound_delta_bounds_json": None,
    "compound_delta_reference": "HARD",
    "team_variation_fraction": 0.5,
    "team_variation_absolute_min": 0.005,
    "clean_lap_noise_sigma": 0.5,
    "weight_strategy": WEIGHT_STRATEGY_GAUSSIAN,
    "weight_effective_sample_count": 20,
    "team_baseline_mode": TEAM_MODE_AVERAGE_DRIVERS,
    "fuel_ref": 0.0,
    "race_lap_ref": None,
    "tyre_age_ref": 0.0,
    "tyre_age_mode": TYRE_AGE_MODE_STINT,
    "track_temperature": None,
    "degradation_order_track_temperature": 15.0,
    "random_seed": None,
    "progress_interval": 10000,
    "quick_lap_threshold": 1.10,
    "min_clean_air_laps": 3,
    "treat_stint_as_whole": False,
    "clean_mean_time_delta_seconds": 2.5,
    "clean_mean_time_delta_behind_seconds": 1.0,
    "wet_lap_proportion_skip_threshold": 0.5,
    "dry_compounds": ("SOFT", "MEDIUM", "HARD"),
    "output_dir": DEFAULT_OUTPUT_DIR,
    "telemetry_cache_dir": None,
    "force_refresh_telemetry": False,
    "use_cached_monte_carlo": False,
    "test": False,
    "reference_team": "Mercedes",
    "plot": True,
    "plot_uncertainty_band": False,
    "plot_rmse_background": False,
    "plot_output": None,
    "show": False,
}


def run_race_performance_review(
    *,
    year: int,
    races: list[int],
    session: str,
    sample_count: int,
    sampling_strategy: str,
    fuel_rate_bounds: tuple[float, float],
    track_rate_bounds: tuple[float, float],
    limit_negative_track_correction: bool,
    default_compound_degradation_bounds: tuple[float, float],
    compound_degradation_bounds: dict[str, tuple[float, float]],
    default_compound_delta_bounds: tuple[float, float],
    compound_delta_bounds: dict[str, tuple[float, float]],
    compound_delta_reference: str,
    team_variation_fraction: float,
    team_variation_absolute_min: float,
    clean_lap_noise_sigma: float,
    weight_strategy: str,
    weight_effective_sample_count: float | None,
    team_baseline_mode: str,
    fuel_ref: float,
    race_lap_ref: float | None,
    tyre_age_ref: float,
    tyre_age_mode: str,
    track_temperature: float | None,
    degradation_order_track_temperature: float | None,
    random_seed: int | None,
    progress_interval: int | None,
    quick_lap_threshold: float,
    min_clean_air_laps: int,
    treat_stint_as_whole: bool,
    clean_mean_time_delta_seconds: float,
    clean_mean_time_delta_behind_seconds: float | None,
    wet_lap_proportion_skip_threshold: float,
    dry_compounds: tuple[str, ...],
    output_dir: str | Path,
    telemetry_cache_dir: str | Path | None,
    force_refresh_telemetry: bool,
    use_cached_monte_carlo: bool,
    test: bool,
) -> dict[str, object]:
    if team_baseline_mode not in TEAM_MODES:
        raise ValueError(f"Unknown team_baseline_mode {team_baseline_mode!r}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_outputs: list[Path] = []
    race_results: dict[int, MonteCarloRacePerformanceResult] = {}
    race_team_baseline_summaries: dict[int, pd.DataFrame] = {}
    race_team_baseline_samples: dict[int, pd.DataFrame] = {}
    race_event_names: dict[int, str | None] = {}
    race_sample_diagnostics: dict[int, pd.DataFrame] = {}
    range_sample_frames: list[pd.DataFrame] = []
    range_team_baseline_frames: list[pd.DataFrame] = []
    range_sample_diagnostic_frames: list[pd.DataFrame] = []
    range_compound_degradation_frames: list[pd.DataFrame] = []
    range_compound_delta_frames: list[pd.DataFrame] = []
    range_team_compound_degradation_frames: list[pd.DataFrame] = []
    range_summary_compound_degradation_frames: list[pd.DataFrame] = []
    range_summary_compound_delta_frames: list[pd.DataFrame] = []
    range_summary_team_compound_degradation_frames: list[pd.DataFrame] = []

    baseline_group = "team" if team_baseline_mode == TEAM_MODE_DIRECT_TEAM else "driver"
    print("Monte Carlo race performance review")
    print(f"Year: {year}")
    print(f"Races: {races}")
    print(f"Session: {session}")
    print(f"Samples: {sample_count}")
    print(f"Sampling strategy: {sampling_strategy}")
    print(f"Fuel rate bounds: {fuel_rate_bounds} s/lap-proxy")
    print(f"Track rate bounds: {track_rate_bounds} s/race-lap")
    print(f"Limit negative track correction: {limit_negative_track_correction}")
    print(f"Default compound degradation bounds: {default_compound_degradation_bounds} s/lap")
    if compound_degradation_bounds:
        print(f"Compound-specific degradation bounds: {compound_degradation_bounds}")
    print(
        "Default compound delta bounds: "
        f"{default_compound_delta_bounds} s vs {compound_delta_reference.upper()}"
    )
    if compound_delta_bounds:
        print(f"Compound-specific delta bounds: {compound_delta_bounds}")
    print(
        "Team variation: "
        f"fraction={team_variation_fraction}, absolute_min={team_variation_absolute_min} s/lap"
    )
    print(f"Noise sigma: {clean_lap_noise_sigma} s")
    print(
        "Weighting: "
        f"strategy={weight_strategy}, "
        f"N_eff={weight_effective_sample_count or 'clean lap count'}"
    )
    print(f"Team baseline mode: {team_baseline_mode} (algorithm baseline={baseline_group})")
    print(
        f"References: fuel={fuel_ref}, "
        f"race_lap={race_lap_ref or 'mean'}, tyre_age={tyre_age_ref}"
    )
    print(f"Tyre age mode: {tyre_age_mode}")
    print(
        "Degradation order temperature: "
        f"track={track_temperature or 'from data'}, "
        f"threshold={degradation_order_track_temperature}"
    )
    print(
        "Wet-race skip threshold: "
        f"median driver wet proportion > {wet_lap_proportion_skip_threshold}"
    )
    clean_lap_mode = "whole stint" if treat_stint_as_whole else "consecutive chunks"
    print(f"Clean-lap selection mode: {clean_lap_mode}")
    print(f"Use cached Monte Carlo results: {use_cached_monte_carlo}")
    print("Missing telemetry gap columns: skip race and save empty diagnostic result")

    for race_index, race in enumerate(races):
        if use_cached_monte_carlo:
            cached = load_cached_monte_carlo_outputs(
                year=year,
                race=race,
                session=session,
                output_dir=output_dir,
                team_baseline_mode=team_baseline_mode,
            )
            if cached is not None:
                race_team_baseline_summaries[race] = cached["team_baseline_summary"]
                race_event_names[race] = cached["event_name"]
                saved_outputs.extend(cached["paths"])
                if cached["team_baseline_samples"] is not None:
                    team_baseline_samples = cached["team_baseline_samples"].copy()
                    race_team_baseline_samples[race] = team_baseline_samples
                    team_baseline_samples["Round"] = race
                    range_team_baseline_frames.append(team_baseline_samples)
                if cached["sample_diagnostics"] is not None:
                    diagnostics = cached["sample_diagnostics"].copy()
                    race_sample_diagnostics[race] = diagnostics
                    diagnostics["Round"] = race
                    range_sample_diagnostic_frames.append(diagnostics)
                    print("\nCached Monte Carlo sample diagnostics")
                    print(diagnostics.to_string(index=False))
                append_cached_degradation_frames(
                    cached,
                    race=race,
                    range_compound_degradation_frames=range_compound_degradation_frames,
                    range_compound_delta_frames=range_compound_delta_frames,
                    range_team_compound_degradation_frames=(
                        range_team_compound_degradation_frames
                    ),
                    range_summary_compound_degradation_frames=(
                        range_summary_compound_degradation_frames
                    ),
                    range_summary_compound_delta_frames=range_summary_compound_delta_frames,
                    range_summary_team_compound_degradation_frames=(
                        range_summary_team_compound_degradation_frames
                    ),
                )
                print(
                    f"\nUsing cached Monte Carlo result for "
                    f"{year} race={race} session={session}"
                )
                print(cached["team_baseline_summary"].to_string(index=False))
                continue

        print(f"\nLoading {year} race={race} session={session}")
        laps = load_all_session_laps_with_telemetry_gap_summary(
            year=year,
            rounds=[race],
            session_names=[session],
            test=test,
            telemetry_cache_dir=telemetry_cache_dir,
            force_refresh_telemetry=force_refresh_telemetry,
        )
        if laps.empty:
            print(f"{year} race {race} {session}: no laps loaded, skipping.")
            continue
        wet_lap_summary = wet_lap_proportion_by_driver(laps)
        print_wet_lap_summary(wet_lap_summary)
        missing_gap_columns = missing_clean_gap_columns(
            laps,
            clean_mean_time_delta_behind_seconds=clean_mean_time_delta_behind_seconds,
        )
        if missing_gap_columns:
            skipped_paths = save_empty_race_outputs(
                year=year,
                race=race,
                session=session,
                output_dir=output_dir,
                reason="missing telemetry gap columns required for clean-air filtering",
                details=", ".join(missing_gap_columns),
                wet_lap_summary=wet_lap_summary,
            )
            saved_outputs.extend(skipped_paths)
            print(
                f"{year} race {race} {session}: missing telemetry gap columns "
                f"{missing_gap_columns}; skipping Monte Carlo."
            )
            continue

        config = MonteCarloRacePerformanceConfig(
            sample_count=sample_count,
            fuel_rate_bounds=fuel_rate_bounds,
            track_rate_bounds=track_rate_bounds,
            limit_negative_track_correction=limit_negative_track_correction,
            compound_degradation_bounds=compound_degradation_bounds,
            default_compound_degradation_bounds=default_compound_degradation_bounds,
            compound_delta_bounds=compound_delta_bounds,
            default_compound_delta_bounds=default_compound_delta_bounds,
            compound_delta_reference=compound_delta_reference,
            team_variation_fraction=team_variation_fraction,
            team_variation_absolute_min=team_variation_absolute_min,
            clean_lap_noise_sigma=clean_lap_noise_sigma,
            weight_strategy=weight_strategy,
            weight_effective_sample_count=weight_effective_sample_count,
            baseline_group=baseline_group,
            fuel_ref=fuel_ref,
            race_lap_ref=race_lap_ref,
            tyre_age_ref=tyre_age_ref,
            track_temperature_celsius=track_temperature,
            degradation_order_track_temperature_celsius=degradation_order_track_temperature,
            random_seed=None if random_seed is None else random_seed + race_index,
            sampling_strategy=sampling_strategy,
        )
        algorithm = MonteCarloRacePerformanceAlgorithm(
            config,
            progress_callback=_progress_printer(race),
            progress_interval=progress_interval,
        )
        try:
            result = calculate_monte_carlo_race_performance_review(
                laps,
                min_clean_air_laps=min_clean_air_laps,
                clean_mean_time_delta_seconds=clean_mean_time_delta_seconds,
                clean_mean_time_delta_behind_seconds=clean_mean_time_delta_behind_seconds,
                quick_lap_threshold=quick_lap_threshold,
                treat_stint_as_whole=treat_stint_as_whole,
                tyre_age_mode=tyre_age_mode,
                algorithm=algorithm,
                dry_compounds=dry_compounds,
                wet_lap_proportion_skip_threshold=wet_lap_proportion_skip_threshold,
            )
        except ValueError as exc:
            if not is_no_clean_laps_error(exc):
                raise
            skipped_paths = save_empty_race_outputs(
                year=year,
                race=race,
                session=session,
                output_dir=output_dir,
                reason="no consecutive clean-air race laps matched filters",
                details=(
                    f"min_clean_air_laps={min_clean_air_laps}, "
                    f"clean_mean_time_delta_seconds={clean_mean_time_delta_seconds}, "
                    "clean_mean_time_delta_behind_seconds="
                    f"{clean_mean_time_delta_behind_seconds}, "
                    f"quick_lap_threshold={quick_lap_threshold}, "
                    f"treat_stint_as_whole={treat_stint_as_whole}"
                ),
                wet_lap_summary=wet_lap_summary,
            )
            saved_outputs.extend(skipped_paths)
            print(
                f"{year} race {race} {session}: no consecutive clean-air laps "
                "matched the configured filters; skipping Monte Carlo."
            )
            continue
        if result == "Wet":
            skipped_paths = save_empty_race_outputs(
                year=year,
                race=race,
                session=session,
                output_dir=output_dir,
                reason="median driver wet lap proportion exceeded threshold",
                details=f"threshold={wet_lap_proportion_skip_threshold}",
                wet_lap_summary=wet_lap_summary,
            )
            saved_outputs.extend(skipped_paths)
            print(
                f"{year} race {race} {session}: median driver wet proportion "
                f"exceeds {wet_lap_proportion_skip_threshold}, skipping."
            )
            continue

        race_results[race] = result
        race_event_names[race] = event_name_from_laps(result.all_laps)
        print_clean_lap_summary(result.clean_laps)
        sample_diagnostics = sample_diagnostics_summary(result.sample_parameters)
        race_sample_diagnostics[race] = sample_diagnostics
        print_sample_diagnostics(result.sample_parameters, sample_diagnostics)
        print_parameter_summaries(result)

        team_baseline_samples = team_baseline_samples_from_result(
            result,
            team_baseline_mode=team_baseline_mode,
        )
        race_team_baseline_samples[race] = team_baseline_samples
        team_baseline_summary = weighted_team_baseline_summary(team_baseline_samples)
        race_team_baseline_summaries[race] = team_baseline_summary
        print("\nTeam corrected baseline pace summary")
        print(team_baseline_summary.to_string(index=False))

        saved_outputs.extend(
            save_race_outputs(
                result=result,
                team_baseline_samples=team_baseline_samples,
                team_baseline_summary=team_baseline_summary,
                sample_diagnostics=sample_diagnostics,
                year=year,
                race=race,
                session=session,
                output_dir=output_dir,
            )
        )

        sample_parameters = result.sample_parameters.copy()
        sample_parameters["Round"] = race
        range_sample_frames.append(sample_parameters)
        team_baseline_samples = team_baseline_samples.copy()
        team_baseline_samples["Round"] = race
        range_team_baseline_frames.append(team_baseline_samples)
        sample_diagnostics = sample_diagnostics.copy()
        sample_diagnostics["Round"] = race
        range_sample_diagnostic_frames.append(sample_diagnostics)
        append_result_degradation_frames(
            result,
            race=race,
            range_compound_degradation_frames=range_compound_degradation_frames,
            range_compound_delta_frames=range_compound_delta_frames,
            range_team_compound_degradation_frames=range_team_compound_degradation_frames,
            range_summary_compound_degradation_frames=(
                range_summary_compound_degradation_frames
            ),
            range_summary_compound_delta_frames=range_summary_compound_delta_frames,
            range_summary_team_compound_degradation_frames=(
                range_summary_team_compound_degradation_frames
            ),
        )

    if not race_team_baseline_summaries:
        print("\nNo Monte Carlo results were produced for the requested races.")
        print("Saved CSV outputs:")
        for path in saved_outputs:
            print(f"  {path}")
        return {
            "race_results": race_results,
            "team_baseline_summaries": race_team_baseline_summaries,
            "team_baseline_samples": race_team_baseline_samples,
            "race_event_names": race_event_names,
            "race_sample_diagnostics": race_sample_diagnostics,
            "saved_outputs": saved_outputs,
        }

    if len(race_team_baseline_summaries) > 1 and range_team_baseline_frames:
        range_label = race_range_label(races)
        all_team_baselines = pd.concat(range_team_baseline_frames, ignore_index=True)
        all_team_summary = weighted_team_baseline_summary(all_team_baselines)
        team_path = output_dir / (
            f"race_performance_team_baselines_{year}_{range_label}_{session}.csv"
        )
        summary_path = output_dir / (
            f"race_performance_team_summary_{year}_{range_label}_{session}.csv"
        )
        all_team_baselines.to_csv(team_path, index=False)
        all_team_summary.to_csv(summary_path, index=False)
        saved_outputs.extend([team_path, summary_path])
        if range_sample_frames:
            all_samples = pd.concat(range_sample_frames, ignore_index=True)
            sample_path = output_dir / (
                f"race_performance_samples_{year}_{range_label}_{session}.csv"
            )
            all_samples.to_csv(sample_path, index=False)
            saved_outputs.append(sample_path)
        if range_sample_diagnostic_frames:
            all_diagnostics = pd.concat(
                range_sample_diagnostic_frames,
                ignore_index=True,
            )
            diagnostics_path = output_dir / (
                f"race_performance_sample_diagnostics_{year}_{range_label}_{session}.csv"
            )
            all_diagnostics.to_csv(diagnostics_path, index=False)
            saved_outputs.append(diagnostics_path)
        saved_outputs.extend(
            save_range_degradation_outputs(
                year=year,
                races=races,
                session=session,
                output_dir=output_dir,
                compound_degradation_frames=range_compound_degradation_frames,
                compound_delta_frames=range_compound_delta_frames,
                team_compound_degradation_frames=range_team_compound_degradation_frames,
                summary_compound_degradation_frames=(
                    range_summary_compound_degradation_frames
                ),
                summary_compound_delta_frames=range_summary_compound_delta_frames,
                summary_team_compound_degradation_frames=(
                    range_summary_team_compound_degradation_frames
                ),
            )
        )
        print(f"\nAggregate team corrected baseline pace summary ({range_label})")
        print(all_team_summary.to_string(index=False))

    print("\nSaved CSV outputs:")
    for path in saved_outputs:
        print(f"  {path}")

    return {
        "race_results": race_results,
        "team_baseline_summaries": race_team_baseline_summaries,
        "team_baseline_samples": race_team_baseline_samples,
        "race_event_names": race_event_names,
        "race_sample_diagnostics": race_sample_diagnostics,
        "saved_outputs": saved_outputs,
    }


def team_baseline_samples_from_result(
    result: MonteCarloRacePerformanceResult,
    *,
    team_baseline_mode: str,
) -> pd.DataFrame:
    return team_baseline_samples_from_baselines(
        baseline_pace=result.baseline_pace,
        sample_parameters=result.sample_parameters,
        team_baseline_mode=team_baseline_mode,
    )


def team_baseline_samples_from_baselines(
    *,
    baseline_pace: pd.DataFrame,
    sample_parameters: pd.DataFrame,
    team_baseline_mode: str,
) -> pd.DataFrame:
    baselines = baseline_pace.copy()
    weights = sample_parameters[["SampleId", "Weight", "RMSESeconds"]]
    if team_baseline_mode == TEAM_MODE_DIRECT_TEAM:
        team_samples = baselines.loc[
            :, ["SampleId", "Team", "CorrectedBaselinePaceSeconds"]
        ].copy()
    else:
        if "Driver" not in baselines.columns:
            raise ValueError(
                f"{team_baseline_mode} requires driver-level algorithm baselines."
            )
        group = baselines.groupby(["SampleId", "Team"], as_index=False)
        if team_baseline_mode == TEAM_MODE_BEST_DRIVER:
            team_samples = group["CorrectedBaselinePaceSeconds"].min()
        elif team_baseline_mode == TEAM_MODE_AVERAGE_DRIVERS:
            team_samples = group["CorrectedBaselinePaceSeconds"].mean()
        else:
            raise ValueError(f"Unknown team baseline mode {team_baseline_mode!r}.")

    team_samples = team_samples.merge(weights, on="SampleId", how="left")
    team_samples["TeamBaselineMode"] = team_baseline_mode
    return team_samples.sort_values(["Team", "SampleId"]).reset_index(drop=True)


def weighted_team_baseline_summary(team_baseline_samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for team, group in team_baseline_samples.groupby("Team", sort=True):
        clean = group.dropna(subset=["CorrectedBaselinePaceSeconds", "Weight"])
        clean = clean.loc[clean["Weight"] > 0]
        values = clean["CorrectedBaselinePaceSeconds"]
        weights = clean["Weight"]
        rows.append(
            {
                "Team": team,
                "P10": weighted_quantile(values, weights, 0.10),
                "Median": weighted_quantile(values, weights, 0.50),
                "P90": weighted_quantile(values, weights, 0.90),
                "SampleCount": int(len(clean)),
                "WeightSum": float(weights.sum()) if not clean.empty else 0.0,
                "MeanRMSESeconds": float(clean["RMSESeconds"].mean()) if not clean.empty else pd.NA,
                "TeamBaselineMode": (
                    clean["TeamBaselineMode"].iloc[0] if not clean.empty else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["Median", "Team"]).reset_index(drop=True)


def sample_diagnostics_summary(sample_parameters: pd.DataFrame) -> pd.DataFrame:
    clean = sample_parameters.dropna(subset=["RMSESeconds", "Weight"]).copy()
    sample_count = int(len(sample_parameters))
    if clean.empty:
        return pd.DataFrame(
            [
                {
                    "SampleCount": sample_count,
                    "BestRMSESeconds": math.nan,
                    "WeightedRMSESeconds": math.nan,
                    "MedianRMSESeconds": math.nan,
                    "P10RMSESeconds": math.nan,
                    "P90RMSESeconds": math.nan,
                    "WeightSum": 0.0,
                    "EffectiveSampleSize": 0.0,
                    "EffectiveSampleFraction": 0.0,
                    "Top1PctWeightShare": math.nan,
                }
            ]
        )

    rmse = clean["RMSESeconds"].astype(float)
    weights = clean["Weight"].astype(float)
    positive_weights = weights.loc[weights > 0]
    weight_sum = float(positive_weights.sum())
    if weight_sum > 0:
        weighted_rmse = math.sqrt(
            float(((rmse.loc[positive_weights.index] ** 2) * positive_weights).sum())
            / weight_sum
        )
        effective_sample_size = float(weight_sum**2 / (positive_weights**2).sum())
        effective_sample_fraction = effective_sample_size / sample_count
        top_count = max(1, math.ceil(sample_count * 0.01))
        top_weight_share = float(
            positive_weights.sort_values(ascending=False).head(top_count).sum()
            / weight_sum
        )
    else:
        weighted_rmse = math.nan
        effective_sample_size = 0.0
        effective_sample_fraction = 0.0
        top_weight_share = math.nan

    return pd.DataFrame(
        [
            {
                "SampleCount": sample_count,
                "BestRMSESeconds": float(rmse.min()),
                "WeightedRMSESeconds": weighted_rmse,
                "MedianRMSESeconds": float(rmse.median()),
                "P10RMSESeconds": float(rmse.quantile(0.10)),
                "P90RMSESeconds": float(rmse.quantile(0.90)),
                "WeightSum": weight_sum,
                "EffectiveSampleSize": effective_sample_size,
                "EffectiveSampleFraction": effective_sample_fraction,
                "Top1PctWeightShare": top_weight_share,
            }
        ]
    )


def append_result_degradation_frames(
    result: MonteCarloRacePerformanceResult,
    *,
    race: int,
    range_compound_degradation_frames: list[pd.DataFrame],
    range_compound_delta_frames: list[pd.DataFrame],
    range_team_compound_degradation_frames: list[pd.DataFrame],
    range_summary_compound_degradation_frames: list[pd.DataFrame],
    range_summary_compound_delta_frames: list[pd.DataFrame],
    range_summary_team_compound_degradation_frames: list[pd.DataFrame],
) -> None:
    _append_round_frame(
        range_compound_degradation_frames,
        result.compound_degradation,
        race=race,
    )
    _append_round_frame(range_compound_delta_frames, result.compound_delta, race=race)
    _append_round_frame(
        range_team_compound_degradation_frames,
        result.team_compound_degradation,
        race=race,
    )
    _append_round_frame(
        range_summary_compound_degradation_frames,
        result.summaries["compound_degradation"],
        race=race,
    )
    _append_round_frame(
        range_summary_compound_delta_frames,
        result.summaries["compound_delta"],
        race=race,
    )
    _append_round_frame(
        range_summary_team_compound_degradation_frames,
        result.summaries["team_compound_degradation"],
        race=race,
    )


def append_cached_degradation_frames(
    cached: dict[str, object],
    *,
    race: int,
    range_compound_degradation_frames: list[pd.DataFrame],
    range_compound_delta_frames: list[pd.DataFrame],
    range_team_compound_degradation_frames: list[pd.DataFrame],
    range_summary_compound_degradation_frames: list[pd.DataFrame],
    range_summary_compound_delta_frames: list[pd.DataFrame],
    range_summary_team_compound_degradation_frames: list[pd.DataFrame],
) -> None:
    _append_round_frame(
        range_compound_degradation_frames,
        cached.get("compound_degradation"),
        race=race,
    )
    _append_round_frame(
        range_compound_delta_frames,
        cached.get("compound_delta"),
        race=race,
    )
    _append_round_frame(
        range_team_compound_degradation_frames,
        cached.get("team_compound_degradation"),
        race=race,
    )
    _append_round_frame(
        range_summary_compound_degradation_frames,
        cached.get("summary_compound_degradation"),
        race=race,
    )
    _append_round_frame(
        range_summary_compound_delta_frames,
        cached.get("summary_compound_delta"),
        race=race,
    )
    _append_round_frame(
        range_summary_team_compound_degradation_frames,
        cached.get("summary_team_compound_degradation"),
        race=race,
    )


def _append_round_frame(
    frames: list[pd.DataFrame],
    frame: object,
    *,
    race: int,
) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return
    output = frame.copy()
    output["Round"] = race
    frames.append(output)


def save_range_degradation_outputs(
    *,
    year: int,
    races: list[int],
    session: str,
    output_dir: Path,
    compound_degradation_frames: list[pd.DataFrame],
    compound_delta_frames: list[pd.DataFrame],
    team_compound_degradation_frames: list[pd.DataFrame],
    summary_compound_degradation_frames: list[pd.DataFrame],
    summary_compound_delta_frames: list[pd.DataFrame],
    summary_team_compound_degradation_frames: list[pd.DataFrame],
) -> list[Path]:
    range_label = race_range_label(races)
    outputs = {
        f"race_performance_compound_degradation_{year}_{range_label}_{session}.csv": (
            compound_degradation_frames
        ),
        f"race_performance_compound_delta_{year}_{range_label}_{session}.csv": (
            compound_delta_frames
        ),
        f"race_performance_team_compound_degradation_{year}_{range_label}_{session}.csv": (
            team_compound_degradation_frames
        ),
        f"race_performance_summary_compound_degradation_{year}_{range_label}_{session}.csv": (
            summary_compound_degradation_frames
        ),
        f"race_performance_summary_compound_delta_{year}_{range_label}_{session}.csv": (
            summary_compound_delta_frames
        ),
        f"race_performance_summary_team_compound_degradation_{year}_{range_label}_{session}.csv": (
            summary_team_compound_degradation_frames
        ),
    }

    paths: list[Path] = []
    for filename, frames in outputs.items():
        if not frames:
            continue
        path = output_dir / filename
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        paths.append(path)
    return paths


def cached_output_path(
    *,
    output_dir: Path,
    year: int,
    race: int,
    session: str,
    suffix: str,
) -> Path:
    return output_dir / f"race_performance_{year}_{race}_{session}_{suffix}.csv"


def load_cached_monte_carlo_outputs(
    *,
    year: int,
    race: int,
    session: str,
    output_dir: Path,
    team_baseline_mode: str,
) -> dict[str, object] | None:
    team_summary_path = cached_output_path(
        output_dir=output_dir,
        year=year,
        race=race,
        session=session,
        suffix="team_baseline_summary",
    )
    if not team_summary_path.exists():
        return None

    team_baseline_summary = pd.read_csv(team_summary_path)
    if team_baseline_summary.empty:
        return None

    baseline_pace_path = cached_output_path(
        output_dir=output_dir,
        year=year,
        race=race,
        session=session,
        suffix="baseline_pace",
    )
    sample_parameters_path = cached_output_path(
        output_dir=output_dir,
        year=year,
        race=race,
        session=session,
        suffix="sample_parameters",
    )
    team_baseline_samples_path = cached_output_path(
        output_dir=output_dir,
        year=year,
        race=race,
        session=session,
        suffix="team_baseline_samples",
    )
    sample_diagnostics_path = cached_output_path(
        output_dir=output_dir,
        year=year,
        race=race,
        session=session,
        suffix="sample_diagnostics",
    )
    clean_laps_path = cached_output_path(
        output_dir=output_dir,
        year=year,
        race=race,
        session=session,
        suffix="clean_laps",
    )
    degradation_paths = {
        "compound_degradation": cached_output_path(
            output_dir=output_dir,
            year=year,
            race=race,
            session=session,
            suffix="compound_degradation",
        ),
        "compound_delta": cached_output_path(
            output_dir=output_dir,
            year=year,
            race=race,
            session=session,
            suffix="compound_delta",
        ),
        "team_compound_degradation": cached_output_path(
            output_dir=output_dir,
            year=year,
            race=race,
            session=session,
            suffix="team_compound_degradation",
        ),
        "summary_compound_degradation": cached_output_path(
            output_dir=output_dir,
            year=year,
            race=race,
            session=session,
            suffix="summary_compound_degradation",
        ),
        "summary_compound_delta": cached_output_path(
            output_dir=output_dir,
            year=year,
            race=race,
            session=session,
            suffix="summary_compound_delta",
        ),
        "summary_team_compound_degradation": cached_output_path(
            output_dir=output_dir,
            year=year,
            race=race,
            session=session,
            suffix="summary_team_compound_degradation",
        ),
    }

    paths = [team_summary_path]
    team_baseline_samples = None
    if baseline_pace_path.exists() and sample_parameters_path.exists():
        baseline_pace = pd.read_csv(baseline_pace_path)
        sample_parameters = pd.read_csv(sample_parameters_path)
        try:
            team_baseline_samples = team_baseline_samples_from_baselines(
                baseline_pace=baseline_pace,
                sample_parameters=sample_parameters,
                team_baseline_mode=team_baseline_mode,
            )
        except ValueError:
            return None
        team_baseline_summary = weighted_team_baseline_summary(team_baseline_samples)
        paths.extend([baseline_pace_path, sample_parameters_path])
    elif team_baseline_samples_path.exists():
        team_baseline_samples = pd.read_csv(team_baseline_samples_path)
        cached_mode = (
            team_baseline_samples["TeamBaselineMode"].iloc[0]
            if "TeamBaselineMode" in team_baseline_samples.columns
            and not team_baseline_samples.empty
            else None
        )
        if cached_mode != team_baseline_mode:
            return None
        paths.append(team_baseline_samples_path)
    else:
        cached_mode = (
            team_baseline_summary["TeamBaselineMode"].iloc[0]
            if "TeamBaselineMode" in team_baseline_summary.columns
            and not team_baseline_summary.empty
            else None
        )
        if cached_mode != team_baseline_mode:
            return None

    sample_diagnostics = None
    if sample_diagnostics_path.exists():
        sample_diagnostics = pd.read_csv(sample_diagnostics_path)
        paths.append(sample_diagnostics_path)

    event_name = None
    if clean_laps_path.exists():
        clean_laps = pd.read_csv(clean_laps_path)
        event_name = event_name_from_laps(clean_laps)
        paths.append(clean_laps_path)

    degradation_outputs = {}
    for name, path in degradation_paths.items():
        if not path.exists():
            degradation_outputs[name] = None
            continue
        degradation_outputs[name] = pd.read_csv(path)
        paths.append(path)

    return {
        "team_baseline_summary": team_baseline_summary,
        "team_baseline_samples": team_baseline_samples,
        "sample_diagnostics": sample_diagnostics,
        "event_name": event_name,
        "paths": paths,
        **degradation_outputs,
    }


def relative_team_pace_rows(
    *,
    team_baseline_summary: pd.DataFrame,
    team_baseline_samples: pd.DataFrame | None = None,
    year: int,
    race: int,
    reference_team: str,
    event_name: str | None,
    sample_diagnostics: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    if team_baseline_samples is not None and not team_baseline_samples.empty:
        return relative_team_pace_rows_from_samples(
            team_baseline_summary=team_baseline_summary,
            team_baseline_samples=team_baseline_samples,
            year=year,
            race=race,
            reference_team=reference_team,
            event_name=event_name,
            sample_diagnostics=sample_diagnostics,
        )

    reference_row = team_baseline_summary.loc[
        team_baseline_summary["Team"] == reference_team
    ]
    if reference_row.empty:
        available = ", ".join(sorted(team_baseline_summary["Team"].dropna().unique()))
        raise ValueError(
            f"Reference team {reference_team!r} not found in race {race}. "
            f"Available teams: {available}"
        )

    reference_seconds = float(reference_row["Median"].iloc[0])
    weighted_rmse = weighted_rmse_from_diagnostics(sample_diagnostics)
    race_team_count = int(team_baseline_summary["Team"].nunique())
    records: list[dict[str, object]] = []
    for _, row in team_baseline_summary.iterrows():
        median = float(row["Median"])
        p10 = float(row["P10"])
        p90 = float(row["P90"])
        records.append(
            {
                "Year": year,
                "Race": race,
                "EventName": event_name,
                "Team": row["Team"],
                "ReferenceTeam": reference_team,
                "TeamBaselineMode": row["TeamBaselineMode"],
                "CorrectedBaselinePaceSeconds": median,
                "P10": p10,
                "Median": median,
                "P90": p90,
                "PercentageToReferenceTeam": median / reference_seconds * 100.0,
                "P10PercentageToReferenceTeam": p10 / reference_seconds * 100.0,
                "P90PercentageToReferenceTeam": p90 / reference_seconds * 100.0,
                "SampleCount": row["SampleCount"],
                "WeightSum": row["WeightSum"],
                "MeanRMSESeconds": row["MeanRMSESeconds"],
                "WeightedRMSESeconds": weighted_rmse,
                "RaceTeamCount": race_team_count,
            }
        )
    return records


def relative_team_pace_rows_from_samples(
    *,
    team_baseline_summary: pd.DataFrame,
    team_baseline_samples: pd.DataFrame,
    year: int,
    race: int,
    reference_team: str,
    event_name: str | None,
    sample_diagnostics: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    required = {"SampleId", "Team", "CorrectedBaselinePaceSeconds", "Weight"}
    missing = required.difference(team_baseline_samples.columns)
    if missing:
        return relative_team_pace_rows(
            team_baseline_summary=team_baseline_summary,
            team_baseline_samples=None,
            year=year,
            race=race,
            reference_team=reference_team,
            event_name=event_name,
            sample_diagnostics=sample_diagnostics,
        )

    reference_samples = team_baseline_samples.loc[
        team_baseline_samples["Team"] == reference_team,
        ["SampleId", "CorrectedBaselinePaceSeconds"],
    ].rename(columns={"CorrectedBaselinePaceSeconds": "ReferenceBaselinePaceSeconds"})
    if reference_samples.empty:
        available = ", ".join(sorted(team_baseline_summary["Team"].dropna().unique()))
        raise ValueError(
            f"Reference team {reference_team!r} not found in race {race}. "
            f"Available teams: {available}"
        )

    summary_by_team = team_baseline_summary.set_index("Team", drop=False)
    weighted_rmse = weighted_rmse_from_diagnostics(sample_diagnostics)
    race_team_count = int(team_baseline_summary["Team"].nunique())
    records: list[dict[str, object]] = []

    for team, team_samples in team_baseline_samples.groupby("Team", sort=True):
        paired = team_samples.merge(reference_samples, on="SampleId", how="inner")
        paired = paired.dropna(
            subset=[
                "CorrectedBaselinePaceSeconds",
                "ReferenceBaselinePaceSeconds",
                "Weight",
            ]
        )
        paired = paired.loc[paired["Weight"] > 0]
        if paired.empty:
            continue

        relative_seconds = (
            paired["CorrectedBaselinePaceSeconds"]
            - paired["ReferenceBaselinePaceSeconds"]
        )
        percentage = (
            paired["CorrectedBaselinePaceSeconds"]
            / paired["ReferenceBaselinePaceSeconds"]
            * 100.0
        )
        summary_row = summary_by_team.loc[team] if team in summary_by_team.index else None
        median = weighted_quantile(
            paired["CorrectedBaselinePaceSeconds"],
            paired["Weight"],
            0.50,
        )
        p10 = weighted_quantile(
            paired["CorrectedBaselinePaceSeconds"],
            paired["Weight"],
            0.10,
        )
        p90 = weighted_quantile(
            paired["CorrectedBaselinePaceSeconds"],
            paired["Weight"],
            0.90,
        )
        records.append(
            {
                "Year": year,
                "Race": race,
                "EventName": event_name,
                "Team": team,
                "ReferenceTeam": reference_team,
                "TeamBaselineMode": (
                    summary_row["TeamBaselineMode"]
                    if summary_row is not None and "TeamBaselineMode" in summary_row
                    else pd.NA
                ),
                "CorrectedBaselinePaceSeconds": median,
                "P10": p10,
                "Median": median,
                "P90": p90,
                "RelativeToReferenceSeconds": weighted_quantile(
                    relative_seconds,
                    paired["Weight"],
                    0.50,
                ),
                "P10RelativeToReferenceSeconds": weighted_quantile(
                    relative_seconds,
                    paired["Weight"],
                    0.10,
                ),
                "P90RelativeToReferenceSeconds": weighted_quantile(
                    relative_seconds,
                    paired["Weight"],
                    0.90,
                ),
                "PercentageToReferenceTeam": weighted_quantile(
                    percentage,
                    paired["Weight"],
                    0.50,
                ),
                "P10PercentageToReferenceTeam": weighted_quantile(
                    percentage,
                    paired["Weight"],
                    0.10,
                ),
                "P90PercentageToReferenceTeam": weighted_quantile(
                    percentage,
                    paired["Weight"],
                    0.90,
                ),
                "SampleCount": int(len(paired)),
                "WeightSum": float(paired["Weight"].sum()),
                "MeanRMSESeconds": (
                    summary_row["MeanRMSESeconds"]
                    if summary_row is not None and "MeanRMSESeconds" in summary_row
                    else pd.NA
                ),
                "WeightedRMSESeconds": weighted_rmse,
                "RaceTeamCount": race_team_count,
            }
        )
    return records


def weighted_rmse_from_diagnostics(sample_diagnostics: pd.DataFrame | None) -> float:
    if sample_diagnostics is None or sample_diagnostics.empty:
        return math.nan
    if "WeightedRMSESeconds" not in sample_diagnostics.columns:
        return math.nan
    values = pd.to_numeric(
        sample_diagnostics["WeightedRMSESeconds"],
        errors="coerce",
    ).dropna()
    if values.empty:
        return math.nan
    return float(values.iloc[0])


def plot_race_performance_results(
    *,
    team_baseline_summaries: dict[int, pd.DataFrame],
    team_baseline_samples: dict[int, pd.DataFrame] | None = None,
    race_event_names: dict[int, str | None] | None,
    race_sample_diagnostics: dict[int, pd.DataFrame] | None,
    year: int,
    reference_team: str,
    output_path: str | Path | None,
    plot_uncertainty_band: bool = False,
    plot_rmse_background: bool = False,
) -> tuple[pd.DataFrame, dict[str, tuple[plt.Figure, plt.Axes]]]:
    records: list[dict[str, object]] = []
    team_baseline_samples = team_baseline_samples or {}
    race_event_names = race_event_names or {}
    race_sample_diagnostics = race_sample_diagnostics or {}
    for race, team_baseline_summary in sorted(team_baseline_summaries.items()):
        if team_baseline_summary.empty:
            continue
        records.extend(
            relative_team_pace_rows(
                team_baseline_summary=team_baseline_summary,
                team_baseline_samples=team_baseline_samples.get(race),
                year=year,
                race=race,
                reference_team=reference_team,
                event_name=race_event_names.get(race),
                sample_diagnostics=race_sample_diagnostics.get(race),
            )
        )

    if not records:
        raise ValueError("No race performance results available for plotting.")

    summary = pd.DataFrame(records)
    figures = RacePerformancePlotter(
        reference_team=reference_team,
        plot_uncertainty_band=plot_uncertainty_band,
        plot_rmse_background=plot_rmse_background,
    ).plot_relative_team_pace(summary)
    if output_path is not None:
        save_relative_team_pace_figures(figures, output_path)
        save_relative_plot_csv(summary, output_path)

    print_relative_plot_summary(summary)
    return summary, figures


def print_relative_plot_summary(summary: pd.DataFrame) -> None:
    print("\nRace performance plot rows")
    print("=" * 26)
    columns = [
        "Race",
        "Team",
        "CorrectedBaselinePaceSeconds",
        "PercentageToReferenceTeam",
    ]
    print(summary.sort_values(["Race", "PercentageToReferenceTeam"])[columns].to_string(index=False))


def relative_plot_csv_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_relative_pace.csv")


def save_relative_plot_csv(summary: pd.DataFrame, output_path: str | Path) -> Path:
    output_file = relative_plot_csv_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_file, index=False)
    return output_file


def event_name_from_laps(laps: pd.DataFrame) -> str | None:
    for column in ("EventName", "RaceName", "EventLocation", "EventCountry"):
        if column not in laps.columns:
            continue
        values = laps[column].dropna().astype(str)
        if not values.empty:
            return values.iloc[0]
    return None


def weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    if values.empty:
        return float("nan")
    order = values.to_numpy(dtype="float64").argsort()
    sorted_values = values.to_numpy(dtype="float64")[order]
    sorted_weights = weights.to_numpy(dtype="float64")[order]
    total_weight = float(sorted_weights.sum())
    if total_weight <= 0:
        return float("nan")
    cumulative = sorted_weights.cumsum()
    return float(sorted_values[cumulative.searchsorted(quantile * total_weight, side="left")])


def save_race_outputs(
    *,
    result: MonteCarloRacePerformanceResult,
    team_baseline_samples: pd.DataFrame,
    team_baseline_summary: pd.DataFrame,
    sample_diagnostics: pd.DataFrame,
    year: int,
    race: int,
    session: str,
    output_dir: Path,
) -> list[Path]:
    prefix = f"race_performance_{year}_{race}_{session}"
    outputs = {
        f"{prefix}_clean_laps.csv": result.clean_laps,
        f"{prefix}_wet_lap_summary.csv": result.wet_lap_summary,
        f"{prefix}_sample_parameters.csv": result.sample_parameters,
        f"{prefix}_compound_degradation.csv": result.compound_degradation,
        f"{prefix}_compound_delta.csv": result.compound_delta,
        f"{prefix}_team_compound_degradation.csv": result.team_compound_degradation,
        f"{prefix}_baseline_pace.csv": result.baseline_pace,
        f"{prefix}_team_baseline_samples.csv": team_baseline_samples,
        f"{prefix}_team_baseline_summary.csv": team_baseline_summary,
        f"{prefix}_sample_diagnostics.csv": sample_diagnostics,
    }
    outputs.update(
        {
            f"{prefix}_summary_{summary_name}.csv": summary
            for summary_name, summary in result.summaries.items()
        }
    )
    paths: list[Path] = []
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def print_clean_lap_summary(clean_laps: pd.DataFrame) -> None:
    print(f"Clean laps used: {len(clean_laps)}")
    by_team = (
        clean_laps.groupby("Team", as_index=False)
        .agg(CleanLapCount=("LapNumber", "size"), DriverCount=("Driver", "nunique"))
        .sort_values(["CleanLapCount", "Team"], ascending=[False, True])
    )
    print("\nClean lap coverage by team")
    print(by_team.to_string(index=False))


def print_wet_lap_summary(wet_lap_summary: pd.DataFrame) -> None:
    if wet_lap_summary.empty:
        print("Wet lap summary unavailable: no driver compound rows.")
        return
    median_wet = float(wet_lap_summary["WetLapProportion"].median())
    max_wet = float(wet_lap_summary["WetLapProportion"].max())
    print(f"Wet lap proportion: median={median_wet:.3f}, max={max_wet:.3f}")
    print("Wet lap coverage by driver")
    print(
        wet_lap_summary.sort_values(
            ["WetLapProportion", "Driver"],
            ascending=[False, True],
        ).to_string(index=False)
    )


def missing_clean_gap_columns(
    laps: pd.DataFrame,
    *,
    clean_mean_time_delta_behind_seconds: float | None,
) -> list[str]:
    required_columns = ["MeanTimeDeltaToDriverAhead"]
    if clean_mean_time_delta_behind_seconds is not None:
        required_columns.append("MeanTimeDeltaToDriverBehind")
    missing: list[str] = []
    for column in required_columns:
        if column not in laps.columns:
            missing.append(column)
        elif laps[column].notna().sum() == 0:
            missing.append(f"{column} (all missing)")
    return missing


def is_no_clean_laps_error(exc: ValueError) -> bool:
    return "No consecutive clean-air race laps matched" in str(exc)


def save_empty_race_outputs(
    *,
    year: int,
    race: int,
    session: str,
    output_dir: Path,
    reason: str,
    details: str,
    wet_lap_summary: pd.DataFrame,
) -> list[Path]:
    prefix = f"race_performance_{year}_{race}_{session}"
    skipped = pd.DataFrame(
        [
            {
                "Year": year,
                "Round": race,
                "SessionName": session,
                "Skipped": True,
                "SkipReason": reason,
                "SkipDetails": details,
            }
        ]
    )
    empty_team_summary = pd.DataFrame(
        columns=[
            "Team",
            "P10",
            "Median",
            "P90",
            "SampleCount",
            "WeightSum",
            "MeanRMSESeconds",
            "TeamBaselineMode",
        ]
    )
    paths = [
        output_dir / f"{prefix}_skipped.csv",
        output_dir / f"{prefix}_wet_lap_summary.csv",
        output_dir / f"{prefix}_team_baseline_summary.csv",
    ]
    skipped.to_csv(paths[0], index=False)
    wet_lap_summary.to_csv(paths[1], index=False)
    empty_team_summary.to_csv(paths[2], index=False)
    return paths


def print_sample_diagnostics(
    sample_parameters: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    best = sample_parameters.nsmallest(5, "RMSESeconds")
    weight_sum = float(sample_parameters["Weight"].sum())
    print(f"\nWeight sum: {weight_sum:.6f}")
    print(format_effective_sample_size(diagnostics))
    print("\nMonte Carlo sample diagnostics")
    print(diagnostics.to_string(index=False))
    print("Best RMSE samples")
    print(
        best[
            [
                "SampleId",
                "FuelRateSecondsPerLap",
                "TrackRateSecondsPerLap",
                "RMSESeconds",
                "Weight",
            ]
        ].to_string(index=False)
    )


def format_effective_sample_size(diagnostics: pd.DataFrame) -> str:
    if diagnostics.empty:
        return "Effective sample size: unavailable"
    row = diagnostics.iloc[0]
    ess = row.get("EffectiveSampleSize", pd.NA)
    ess_fraction = row.get("EffectiveSampleFraction", pd.NA)
    if pd.isna(ess):
        return "Effective sample size: unavailable"
    if pd.isna(ess_fraction):
        return f"Effective sample size: ESS={float(ess):.2f}"
    return (
        "Effective sample size: "
        f"ESS={float(ess):.2f}, fraction={float(ess_fraction):.3f}"
    )


def print_parameter_summaries(result: MonteCarloRacePerformanceResult) -> None:
    print("\nFuel correction summary")
    print(result.summaries["fuel_rate"].to_string(index=False))
    print("\nTrack correction summary")
    print(result.summaries["track_rate"].to_string(index=False))
    print("\nCompound degradation summary")
    print(result.summaries["compound_degradation"].to_string(index=False))
    print("\nCompound delta summary")
    print(result.summaries["compound_delta"].to_string(index=False))
    print("\nTeam-compound degradation summary")
    print(result.summaries["team_compound_degradation"].to_string(index=False))


def _progress_printer(race: int):
    def _print_progress(progress: dict[str, object]) -> None:
        print(
            f"Race {race}: sample {progress['sample']}/{progress['sample_count']} "
            f"rmse={float(progress['rmse_seconds']):.4f}s "
            f"best={float(progress['best_rmse_seconds']):.4f}s "
            f"weight_sum={float(progress['weight_sum']):.4f}"
        )

    return _print_progress


def main() -> None:
    args = parse_args()
    races = parse_race_selector(args.race)
    run_result = run_race_performance_review(
        year=args.year,
        races=races,
        session=args.session,
        sample_count=args.sample_count,
        sampling_strategy=args.sampling_strategy,
        fuel_rate_bounds=parse_bounds(args.fuel_rate_bounds),
        track_rate_bounds=parse_bounds(args.track_rate_bounds),
        limit_negative_track_correction=args.limit_negative_track_correction,
        default_compound_degradation_bounds=parse_bounds(args.tyre_deg_bounds),
        compound_degradation_bounds=parse_compound_bounds(args.compound_deg_bounds_json),
        default_compound_delta_bounds=parse_bounds(args.tyre_delta_bounds),
        compound_delta_bounds=parse_compound_bounds(args.compound_delta_bounds_json),
        compound_delta_reference=args.compound_delta_reference,
        team_variation_fraction=args.team_variation_fraction,
        team_variation_absolute_min=args.team_variation_absolute_min,
        clean_lap_noise_sigma=args.clean_lap_noise_sigma,
        weight_strategy=args.weight_strategy,
        weight_effective_sample_count=parse_optional_float(
            args.weight_effective_sample_count
        ),
        team_baseline_mode=args.team_baseline_mode,
        fuel_ref=args.fuel_ref,
        race_lap_ref=parse_optional_float(args.race_lap_ref),
        tyre_age_ref=args.tyre_age_ref,
        tyre_age_mode=args.tyre_age_mode,
        track_temperature=parse_optional_float(args.track_temperature),
        degradation_order_track_temperature=parse_optional_float(
            args.degradation_order_track_temperature
        ),
        random_seed=args.random_seed,
        progress_interval=parse_optional_int(args.progress_interval),
        quick_lap_threshold=args.quick_lap_threshold,
        min_clean_air_laps=args.min_clean_air_laps,
        treat_stint_as_whole=args.treat_stint_as_whole,
        clean_mean_time_delta_seconds=args.clean_mean_time_delta_seconds,
        clean_mean_time_delta_behind_seconds=parse_optional_float(
            args.clean_mean_time_delta_behind_seconds
        ),
        wet_lap_proportion_skip_threshold=args.wet_lap_proportion_skip_threshold,
        dry_compounds=tuple(args.dry_compounds),
        output_dir=args.output_dir,
        telemetry_cache_dir=args.telemetry_cache_dir,
        force_refresh_telemetry=args.force_refresh_telemetry,
        use_cached_monte_carlo=args.use_cached_monte_carlo,
        test=args.test,
    )
    if args.plot and run_result["team_baseline_summaries"]:
        output_path = args.plot_output
        if output_path is None:
            output_path = default_plot_output_path(
                year=args.year,
                races=races,
                session=args.session,
                reference_team=args.reference_team,
                team_baseline_mode=args.team_baseline_mode,
            )
        summary, figures = plot_race_performance_results(
            team_baseline_summaries=run_result["team_baseline_summaries"],
            team_baseline_samples=run_result["team_baseline_samples"],
            race_event_names=run_result["race_event_names"],
            race_sample_diagnostics=run_result["race_sample_diagnostics"],
            year=args.year,
            reference_team=args.reference_team,
            output_path=output_path,
            plot_uncertainty_band=args.plot_uncertainty_band,
            plot_rmse_background=args.plot_rmse_background,
        )
        print("\nSaved race performance plot:")
        print(f"  {race_result_output_path(Path(output_path))}")
        print(f"  relative pace csv: {relative_plot_csv_path(output_path)}")
        print(f"Rows plotted: {len(summary)}")
        if args.show:
            plt.show()
        else:
            for fig, _ in figures.values():
                plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo race performance review and save team CSV outputs."
    )
    parser.add_argument("--year", type=int, default=SCRIPT_CONFIG["year"])
    parser.add_argument(
        "--race",
        default=SCRIPT_CONFIG["race"],
        help="Race number or inclusive range, for example '7' or '1-7'.",
    )
    parser.add_argument("--session", default=SCRIPT_CONFIG["session"])
    parser.add_argument("--sample-count", type=int, default=SCRIPT_CONFIG["sample_count"])
    parser.add_argument(
        "--sampling-strategy",
        choices=SAMPLING_STRATEGIES,
        default=SCRIPT_CONFIG["sampling_strategy"],
        help=(
            "random draws independent values; latin-hypercube stratifies each "
            "dimension; halton uses a deterministic low-discrepancy sequence."
        ),
    )
    parser.add_argument(
        "--fuel-rate-bounds",
        nargs=2,
        default=SCRIPT_CONFIG["fuel_rate_bounds"],
        help="Lower and upper fuel correction rate bounds.",
    )
    parser.add_argument(
        "--track-rate-bounds",
        nargs=2,
        default=SCRIPT_CONFIG["track_rate_bounds"],
        help="Lower and upper track evolution correction rate bounds.",
    )
    parser.add_argument(
        "--limit-negative-track-correction",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["limit_negative_track_correction"],
        help=(
            "Clamp sampled track correction rates to be non-negative. This "
            "prevents the correction from making later-race laps longer."
        ),
    )
    parser.add_argument(
        "--tyre-deg-bounds",
        nargs=2,
        default=SCRIPT_CONFIG["default_compound_degradation_bounds"],
        help="Default lower and upper compound degradation rate bounds.",
    )
    parser.add_argument(
        "--compound-deg-bounds-json",
        default=SCRIPT_CONFIG["compound_degradation_bounds_json"],
        help='Optional JSON, e.g. \'{"SOFT":[0.02,0.10],"HARD":[0.0,0.05]}\'',
    )
    parser.add_argument(
        "--tyre-delta-bounds",
        nargs=2,
        default=SCRIPT_CONFIG["default_compound_delta_bounds"],
        help=(
            "Default lower and upper compound lap-time delta bounds in seconds "
            "relative to --compound-delta-reference."
        ),
    )
    parser.add_argument(
        "--compound-delta-bounds-json",
        default=SCRIPT_CONFIG["compound_delta_bounds_json"],
        help='Optional JSON, e.g. \'{"SOFT":[-1.0,-0.1],"MEDIUM":[-0.5,0.2]}\'',
    )
    parser.add_argument(
        "--compound-delta-reference",
        default=SCRIPT_CONFIG["compound_delta_reference"],
        help="Compound whose sampled lap-time delta is fixed at zero.",
    )
    parser.add_argument(
        "--team-variation-fraction",
        type=float,
        default=SCRIPT_CONFIG["team_variation_fraction"],
    )
    parser.add_argument(
        "--team-variation-absolute-min",
        type=float,
        default=SCRIPT_CONFIG["team_variation_absolute_min"],
    )
    parser.add_argument(
        "--clean-lap-noise-sigma",
        type=float,
        default=SCRIPT_CONFIG["clean_lap_noise_sigma"],
    )
    parser.add_argument(
        "--weight-strategy",
        choices=WEIGHT_STRATEGIES,
        default=SCRIPT_CONFIG["weight_strategy"],
        help=(
            "gaussian uses exp(-rmse^2 / (2 sigma^2)); "
            "best-rmse-relative uses normalized exp(-N_eff * "
            "(rmse^2 - best_rmse^2) / (2 sigma^2))."
        ),
    )
    parser.add_argument(
        "--weight-effective-sample-count",
        default=SCRIPT_CONFIG["weight_effective_sample_count"],
        help=(
            "N_eff for best-rmse-relative weighting. Use 'none' to use the "
            "clean lap count."
        ),
    )
    parser.add_argument(
        "--team-baseline-mode",
        choices=TEAM_MODES,
        default=SCRIPT_CONFIG["team_baseline_mode"],
        help=(
            "best-driver and average-drivers fit driver baselines then aggregate "
            "to teams; direct-team fits one baseline per team."
        ),
    )
    parser.add_argument("--fuel-ref", type=float, default=SCRIPT_CONFIG["fuel_ref"])
    parser.add_argument(
        "--race-lap-ref",
        default=SCRIPT_CONFIG["race_lap_ref"],
        help="Race lap reference, or 'none' to use mean clean race lap.",
    )
    parser.add_argument("--tyre-age-ref", type=float, default=SCRIPT_CONFIG["tyre_age_ref"])
    parser.add_argument(
        "--tyre-age-mode",
        choices=TYRE_AGE_MODES,
        default=SCRIPT_CONFIG["tyre_age_mode"],
        help=(
            "stint uses StintLapNumber/counted laps within the stint; overall "
            "uses TyreLife from the session data."
        ),
    )
    parser.add_argument(
        "--track-temperature",
        default=SCRIPT_CONFIG["track_temperature"],
        help=(
            "Actual track temperature in Celsius for degradation-order gating; "
            "use 'none' to infer from lap columns when available."
        ),
    )
    parser.add_argument(
        "--degradation-order-track-temperature",
        default=SCRIPT_CONFIG["degradation_order_track_temperature"],
        help=(
            "If track temperature is above this Celsius threshold, force base "
            "degradation order SOFT >= MEDIUM >= HARD; use 'none' to disable."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=SCRIPT_CONFIG["random_seed"])
    parser.add_argument(
        "--progress-interval",
        default=SCRIPT_CONFIG["progress_interval"],
        help="Print progress every N samples; use 'none' for first/last only.",
    )
    parser.add_argument(
        "--quick-lap-threshold",
        type=float,
        default=SCRIPT_CONFIG["quick_lap_threshold"],
    )
    parser.add_argument(
        "--min-clean-air-laps",
        type=int,
        default=SCRIPT_CONFIG["min_clean_air_laps"],
    )
    parser.add_argument(
        "--treat-stint-as-whole",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["treat_stint_as_whole"],
        help=(
            "Group all clean laps from the same driver/stint into one run. "
            "When disabled, only consecutive clean-air chunks are selected."
        ),
    )
    parser.add_argument(
        "--clean-mean-time-delta-seconds",
        type=float,
        default=SCRIPT_CONFIG["clean_mean_time_delta_seconds"],
    )
    parser.add_argument(
        "--clean-mean-time-delta-behind-seconds",
        default=SCRIPT_CONFIG["clean_mean_time_delta_behind_seconds"],
        help="Use 'none' to disable the behind-car clean-air requirement.",
    )
    parser.add_argument(
        "--wet-lap-proportion-skip-threshold",
        type=float,
        default=SCRIPT_CONFIG["wet_lap_proportion_skip_threshold"],
        help=(
            "Skip the race only when the median driver wet/inter lap proportion "
            "is greater than this threshold."
        ),
    )
    parser.add_argument("--dry-compounds", nargs="+", default=SCRIPT_CONFIG["dry_compounds"])
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_CONFIG["output_dir"])
    parser.add_argument("--reference-team", default=SCRIPT_CONFIG["reference_team"])
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["plot"],
        help="Save a relative team race performance plot.",
    )
    parser.add_argument(
        "--plot-uncertainty-band",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["plot_uncertainty_band"],
        help="Draw the P10/P90 uncertainty band around each team line.",
    )
    parser.add_argument(
        "--plot-rmse-background",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["plot_rmse_background"],
        help=(
            "Shade each race by weighted RMSE. Races with fewer than five teams "
            "are marked with a black striped background."
        ),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=SCRIPT_CONFIG["plot_output"],
        help="Plot output stem/path. Defaults to temp/race_performance_tracker_*.png.",
    )
    parser.add_argument("--show", action="store_true", default=SCRIPT_CONFIG["show"])
    parser.add_argument(
        "--telemetry-cache-dir",
        type=Path,
        default=SCRIPT_CONFIG["telemetry_cache_dir"],
    )
    parser.add_argument(
        "--force-refresh-telemetry",
        action="store_true",
        default=SCRIPT_CONFIG["force_refresh_telemetry"],
    )
    parser.add_argument(
        "--use-cached-monte-carlo",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["use_cached_monte_carlo"],
        help=(
            "Reuse cached per-race Monte Carlo CSVs when available and only "
            "calculate missing races. Disable to rerun all requested races."
        ),
    )
    parser.add_argument("--test", action="store_true", default=SCRIPT_CONFIG["test"])
    return parser.parse_args()


def parse_race_selector(value: str) -> list[int]:
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        parts = [part.strip() for part in text[1:-1].split(",")]
        if len(parts) != 2:
            raise ValueError("--race bracket range must look like '[<start>, <end>]'.")
        start, end = int(parts[0]), int(parts[1])
        if end < start:
            raise ValueError("--race range end must be >= start.")
        return list(range(start, end + 1))
    if "-" in text:
        start_text, end_text = text.split("-", maxsplit=1)
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError("--race range end must be >= start.")
        return list(range(start, end + 1))
    return [int(text)]


def parse_bounds(values: tuple[float, float] | list[str] | tuple[str, str]) -> tuple[float, float]:
    lower, upper = float(values[0]), float(values[1])
    if lower > upper:
        raise ValueError(f"Bounds lower value {lower} must be <= upper value {upper}.")
    return lower, upper


def parse_compound_bounds(value: str | None) -> dict[str, tuple[float, float]]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--compound-deg-bounds-json must decode to a JSON object.")
    output: dict[str, tuple[float, float]] = {}
    for compound, bounds in parsed.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Bounds for compound {compound!r} must be a two-item list.")
        output[str(compound).upper()] = parse_bounds(bounds)
    return output


def parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", ""}:
        return None
    return float(value)


def parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", ""}:
        return None
    return int(value)


def race_range_label(races: list[int]) -> str:
    if len(races) == 1:
        return str(races[0])
    return f"{min(races)}-{max(races)}"


def default_plot_output_path(
    *,
    year: int,
    races: list[int],
    session: str,
    reference_team: str,
    team_baseline_mode: str,
) -> Path:
    return Path("temp") / (
        f"race_performance_tracker_{year}_{race_range_label(races)}_{session}_"
        f"{safe_name(reference_team)}_{team_baseline_mode}.png"
    )


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", ".") else "-" for char in value)


if __name__ == "__main__":
    main()
