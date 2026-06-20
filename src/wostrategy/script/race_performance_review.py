from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from wostrategy.algorithm.monte_carlo_race_performance import (
    MonteCarloRacePerformanceAlgorithm,
    MonteCarloRacePerformanceConfig,
)
from wostrategy.algorithm.sampling import RANDOM_SAMPLER, SAMPLING_STRATEGIES, LATIN_HYPERCUBE_SAMPLER, HALTON_SAMPLER
from wostrategy.analysis.race_performance_review import (
    MonteCarloRacePerformanceResult,
    calculate_monte_carlo_race_performance_review,
    wet_lap_proportion_by_driver,
)
from wostrategy.tools import load_all_session_laps_with_telemetry_gap_summary

TEAM_MODE_BEST_DRIVER = "best-driver"
TEAM_MODE_AVERAGE_DRIVERS = "average-drivers"
TEAM_MODE_DIRECT_TEAM = "direct-team"
TEAM_MODES = (TEAM_MODE_BEST_DRIVER, TEAM_MODE_AVERAGE_DRIVERS, TEAM_MODE_DIRECT_TEAM)


SCRIPT_CONFIG = {
    "year": 2026,
    "race": "7",
    "session": "R",
    "sample_count": 50000,
    "sampling_strategy": LATIN_HYPERCUBE_SAMPLER,
    "fuel_rate_bounds": (0.0, 0.1),
    "track_rate_bounds": (-0.05, 0.05),
    "default_compound_degradation_bounds": (0.0, 0.5),
    "compound_degradation_bounds_json": None,
    "team_variation_fraction": 0.5,
    "team_variation_absolute_min": 0.005,
    "clean_lap_noise_sigma": 0.5,
    "team_baseline_mode": TEAM_MODE_AVERAGE_DRIVERS,
    "fuel_ref": 0.0,
    "race_lap_ref": None,
    "tyre_age_ref": 0.0,
    "random_seed": None,
    "progress_interval": 100,
    "quick_lap_threshold": 1.10,
    "min_clean_air_laps": 4,
    "clean_mean_time_delta_seconds": 2.5,
    "clean_mean_time_delta_behind_seconds": 1.0,
    "wet_lap_proportion_skip_threshold": 0.5,
    "dry_compounds": ("SOFT", "MEDIUM", "HARD"),
    "output_dir": "cache/race_performance_review",
    "telemetry_cache_dir": None,
    "force_refresh_telemetry": False,
    "test": False,
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
    default_compound_degradation_bounds: tuple[float, float],
    compound_degradation_bounds: dict[str, tuple[float, float]],
    team_variation_fraction: float,
    team_variation_absolute_min: float,
    clean_lap_noise_sigma: float,
    team_baseline_mode: str,
    fuel_ref: float,
    race_lap_ref: float | None,
    tyre_age_ref: float,
    random_seed: int | None,
    progress_interval: int | None,
    quick_lap_threshold: float,
    min_clean_air_laps: int,
    clean_mean_time_delta_seconds: float,
    clean_mean_time_delta_behind_seconds: float | None,
    wet_lap_proportion_skip_threshold: float,
    dry_compounds: tuple[str, ...],
    output_dir: str | Path,
    telemetry_cache_dir: str | Path | None,
    force_refresh_telemetry: bool,
    test: bool,
) -> dict[str, object]:
    if team_baseline_mode not in TEAM_MODES:
        raise ValueError(f"Unknown team_baseline_mode {team_baseline_mode!r}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_outputs: list[Path] = []
    race_results: dict[int, MonteCarloRacePerformanceResult] = {}
    range_sample_frames: list[pd.DataFrame] = []
    range_team_baseline_frames: list[pd.DataFrame] = []

    baseline_group = "team" if team_baseline_mode == TEAM_MODE_DIRECT_TEAM else "driver"
    print("Monte Carlo race performance review")
    print(f"Year: {year}")
    print(f"Races: {races}")
    print(f"Session: {session}")
    print(f"Samples: {sample_count}")
    print(f"Sampling strategy: {sampling_strategy}")
    print(f"Fuel rate bounds: {fuel_rate_bounds} s/lap-proxy")
    print(f"Track rate bounds: {track_rate_bounds} s/race-lap")
    print(f"Default compound degradation bounds: {default_compound_degradation_bounds} s/lap")
    if compound_degradation_bounds:
        print(f"Compound-specific degradation bounds: {compound_degradation_bounds}")
    print(
        "Team variation: "
        f"fraction={team_variation_fraction}, absolute_min={team_variation_absolute_min} s/lap"
    )
    print(f"Noise sigma: {clean_lap_noise_sigma} s")
    print(f"Team baseline mode: {team_baseline_mode} (algorithm baseline={baseline_group})")
    print(
        f"References: fuel={fuel_ref}, "
        f"race_lap={race_lap_ref or 'mean'}, tyre_age={tyre_age_ref}"
    )
    print(
        "Wet-race skip threshold: "
        f"median driver wet proportion > {wet_lap_proportion_skip_threshold}"
    )
    print("Missing telemetry gap columns: skip race and save empty diagnostic result")

    for race_index, race in enumerate(races):
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
            compound_degradation_bounds=compound_degradation_bounds,
            default_compound_degradation_bounds=default_compound_degradation_bounds,
            team_variation_fraction=team_variation_fraction,
            team_variation_absolute_min=team_variation_absolute_min,
            clean_lap_noise_sigma=clean_lap_noise_sigma,
            baseline_group=baseline_group,
            fuel_ref=fuel_ref,
            race_lap_ref=race_lap_ref,
            tyre_age_ref=tyre_age_ref,
            random_seed=None if random_seed is None else random_seed + race_index,
            sampling_strategy=sampling_strategy,
        )
        algorithm = MonteCarloRacePerformanceAlgorithm(
            config,
            progress_callback=_progress_printer(race),
            progress_interval=progress_interval,
        )
        result = calculate_monte_carlo_race_performance_review(
            laps,
            min_clean_air_laps=min_clean_air_laps,
            clean_mean_time_delta_seconds=clean_mean_time_delta_seconds,
            clean_mean_time_delta_behind_seconds=clean_mean_time_delta_behind_seconds,
            quick_lap_threshold=quick_lap_threshold,
            algorithm=algorithm,
            dry_compounds=dry_compounds,
            wet_lap_proportion_skip_threshold=wet_lap_proportion_skip_threshold,
        )
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
        print_clean_lap_summary(result.clean_laps)
        print_sample_diagnostics(result.sample_parameters)
        print_parameter_summaries(result)

        team_baseline_samples = team_baseline_samples_from_result(
            result,
            team_baseline_mode=team_baseline_mode,
        )
        team_baseline_summary = weighted_team_baseline_summary(team_baseline_samples)
        print("\nTeam corrected baseline pace summary")
        print(team_baseline_summary.to_string(index=False))

        saved_outputs.extend(
            save_race_outputs(
                result=result,
                team_baseline_samples=team_baseline_samples,
                team_baseline_summary=team_baseline_summary,
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

    if not race_results:
        print("\nNo Monte Carlo results were produced for the requested races.")
        print("Saved CSV outputs:")
        for path in saved_outputs:
            print(f"  {path}")
        return {
            "race_results": race_results,
            "saved_outputs": saved_outputs,
        }

    if len(race_results) > 1:
        range_label = race_range_label(races)
        all_samples = pd.concat(range_sample_frames, ignore_index=True)
        all_team_baselines = pd.concat(range_team_baseline_frames, ignore_index=True)
        all_team_summary = weighted_team_baseline_summary(all_team_baselines)
        sample_path = output_dir / f"race_performance_samples_{year}_{range_label}_{session}.csv"
        team_path = output_dir / (
            f"race_performance_team_baselines_{year}_{range_label}_{session}.csv"
        )
        summary_path = output_dir / (
            f"race_performance_team_summary_{year}_{range_label}_{session}.csv"
        )
        all_samples.to_csv(sample_path, index=False)
        all_team_baselines.to_csv(team_path, index=False)
        all_team_summary.to_csv(summary_path, index=False)
        saved_outputs.extend([sample_path, team_path, summary_path])
        print(f"\nAggregate team corrected baseline pace summary ({range_label})")
        print(all_team_summary.to_string(index=False))

    print("\nSaved CSV outputs:")
    for path in saved_outputs:
        print(f"  {path}")

    return {
        "race_results": race_results,
        "saved_outputs": saved_outputs,
    }


def team_baseline_samples_from_result(
    result: MonteCarloRacePerformanceResult,
    *,
    team_baseline_mode: str,
) -> pd.DataFrame:
    baselines = result.baseline_pace.copy()
    weights = result.sample_parameters[["SampleId", "Weight", "RMSESeconds"]]
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
        f"{prefix}_team_compound_degradation.csv": result.team_compound_degradation,
        f"{prefix}_baseline_pace.csv": result.baseline_pace,
        f"{prefix}_team_baseline_samples.csv": team_baseline_samples,
        f"{prefix}_team_baseline_summary.csv": team_baseline_summary,
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


def print_sample_diagnostics(sample_parameters: pd.DataFrame) -> None:
    best = sample_parameters.nsmallest(5, "RMSESeconds")
    weight_sum = float(sample_parameters["Weight"].sum())
    print(f"\nWeight sum: {weight_sum:.6f}")
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


def print_parameter_summaries(result: MonteCarloRacePerformanceResult) -> None:
    print("\nFuel correction summary")
    print(result.summaries["fuel_rate"].to_string(index=False))
    print("\nTrack correction summary")
    print(result.summaries["track_rate"].to_string(index=False))
    print("\nCompound degradation summary")
    print(result.summaries["compound_degradation"].to_string(index=False))
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
    run_race_performance_review(
        year=args.year,
        races=races,
        session=args.session,
        sample_count=args.sample_count,
        sampling_strategy=args.sampling_strategy,
        fuel_rate_bounds=parse_bounds(args.fuel_rate_bounds),
        track_rate_bounds=parse_bounds(args.track_rate_bounds),
        default_compound_degradation_bounds=parse_bounds(args.tyre_deg_bounds),
        compound_degradation_bounds=parse_compound_bounds(args.compound_deg_bounds_json),
        team_variation_fraction=args.team_variation_fraction,
        team_variation_absolute_min=args.team_variation_absolute_min,
        clean_lap_noise_sigma=args.clean_lap_noise_sigma,
        team_baseline_mode=args.team_baseline_mode,
        fuel_ref=args.fuel_ref,
        race_lap_ref=parse_optional_float(args.race_lap_ref),
        tyre_age_ref=args.tyre_age_ref,
        random_seed=args.random_seed,
        progress_interval=parse_optional_int(args.progress_interval),
        quick_lap_threshold=args.quick_lap_threshold,
        min_clean_air_laps=args.min_clean_air_laps,
        clean_mean_time_delta_seconds=args.clean_mean_time_delta_seconds,
        clean_mean_time_delta_behind_seconds=parse_optional_float(
            args.clean_mean_time_delta_behind_seconds
        ),
        wet_lap_proportion_skip_threshold=args.wet_lap_proportion_skip_threshold,
        dry_compounds=tuple(args.dry_compounds),
        output_dir=args.output_dir,
        telemetry_cache_dir=args.telemetry_cache_dir,
        force_refresh_telemetry=args.force_refresh_telemetry,
        test=args.test,
    )


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
    parser.add_argument("--test", action="store_true", default=SCRIPT_CONFIG["test"])
    return parser.parse_args()


def parse_race_selector(value: str) -> list[int]:
    text = str(value).strip()
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


if __name__ == "__main__":
    main()
