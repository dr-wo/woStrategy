from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from wostrategy.analysis.long_run_performance import (
    LONG_RUN_MODEL_EXPONENTIAL_TRACK,
    LONG_RUN_MODEL_LINEAR_COMPONENTS,
    LongRunPerformanceResult,
    calculate_long_run_performance,
)
from wostrategy.plots.long_run_performance import (
    plot_driver_long_run_fits,
    plot_long_run_performance_trend,
)
from wostrategy.model.track_evolution import LINEAR_TRACK_EVOLUTION_MODEL
from wostrategy.script.quali_performance_tracker import (
    SCRIPT_CONFIG as QUALI_SCRIPT_CONFIG,
    run_quali_performance_tracker,
)
from wostrategy.tools import load_all_session_laps_with_telemetry_gap_summary
from wostrategy.utils import match_team_name, reference_team_or_wcc_leader


SCRIPT_CONFIG = {
    "year": 2026,
    "race_range": [4, 4],
    "section": "R",
    # None means use the leading WCC team after race_range[1].
    "reference_team": None,
    "quick_lap_threshold": 1.10,
    "min_clean_air_laps": 4,
    "clean_mean_time_delta_seconds": 3.0,
    "clean_mean_time_delta_behind_seconds": 1.0,
    # Preset options:
    #   LONG_RUN_MODEL_LINEAR_COMPONENTS
    #   LONG_RUN_MODEL_EXPONENTIAL_TRACK
    # Ignored when model_config or model_config_json is set.
    "model": LONG_RUN_MODEL_LINEAR_COMPONENTS,
    # Optional Python dictionary for custom model combinations. Example:
    # {
    #     "name": "custom_linear_tyre_fuel_exponential_track",
    #     "terms": {
    #         "tyre": {
    #             "model": "linear",
    #             "x_column": "TyreAgeLaps",
    #             "parameter": "tyre_slope_seconds_per_lap",
    #             "label": "tyre_age",
    #             "reference_value": 0.0,
    #         },
    #         "fuel": {
    #             "model": "linear",
    #             "x_column": "FuelLapNumber",
    #             "parameter": "fuel_slope_seconds_per_lap",
    #             "label": "fuel_lap",
    #             "reference_value": 1.0,
    #         },
    #         "track": {
    #             "model": "exponential",
    #             "x_column": "LapNumber",
    #             "amplitude_parameter": "track_amplitude_seconds",
    #             "decay_parameter": "track_decay_rate",
    #             "label": "track_x",
    #         },
    #     },
    #     "intercept": {"parameter": "intercept_seconds"},
    # }
    "model_config": None,
    # Optional CLI-style JSON string for custom model combinations.
    # Takes precedence over model_config when both are set.
    "model_config_json": None,
    "dry_compounds": ("SOFT", "MEDIUM", "HARD"),
    "track_x_column": "LapNumber",
    "min_fit_laps": None,
    "outlier_sigma": 1,
    "min_fit_laps_after_outlier_filter": 4,
    "combined_loss_slope_outlier_sigma": 1.5,
    "combined_loss_slope_outlier_min_fits": 4,
    # "quali" runs/loads a linear qualifying track evolution rate and feeds it
    # into race long-run correction. "race" keeps the previous race-only fit.
    "track_evolution_rate_source": "quali",
    "quali_track_evolution_path": None,
    "quali_track_evolution_fit": LINEAR_TRACK_EVOLUTION_MODEL,
    "output_dir": "temp",
    "telemetry_cache_dir": None,
    "force_refresh_telemetry": False,
    "test": False,
    "show": False,
}


def run_long_run_performance_analysis(
    *,
    year: int,
    races: list[int | str],
    section: int | str,
    min_clean_air_laps: int,
    clean_mean_time_delta_seconds: float,
    clean_mean_time_delta_behind_seconds: float | None,
    quick_lap_threshold: float,
    reference_team: str | None,
    model: str,
    model_config: dict[str, Any] | None,
    dry_compounds: tuple[str, ...],
    track_x_column: str,
    min_fit_laps: int | None,
    outlier_sigma: float | None,
    min_fit_laps_after_outlier_filter: int,
    combined_loss_slope_outlier_sigma: float | None,
    combined_loss_slope_outlier_min_fits: int,
    track_evolution_rate_source: str,
    quali_track_evolution_path: str | Path | None,
    quali_track_evolution_fit: str,
    output_dir: str | Path,
    telemetry_cache_dir: str | Path | None,
    force_refresh_telemetry: bool,
    test: bool,
    show: bool,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_reference_team = _requested_reference_team(
        year=year,
        end_race=races[-1],
        reference_team=reference_team,
    )

    race_results = {}
    team_performance_frames = []
    figures = {}
    saved_outputs: list[tuple[str, Path]] = []
    for race in races:
        track_evolution_rate = _track_evolution_rate_for_race(
            year=year,
            race=race,
            source=track_evolution_rate_source,
            output_dir=output_dir,
            configured_path=quali_track_evolution_path,
            quali_track_evolution_fit=quali_track_evolution_fit,
            dry_compounds=dry_compounds,
            telemetry_cache_dir=telemetry_cache_dir,
            force_refresh_telemetry=force_refresh_telemetry,
            test=test,
        )
        laps = load_all_session_laps_with_telemetry_gap_summary(
            year=year,
            rounds=[race],
            session_names=[section],
            test=test,
            telemetry_cache_dir=telemetry_cache_dir,
            force_refresh_telemetry=force_refresh_telemetry,
        )
        result = calculate_long_run_performance(
            laps,
            min_clean_air_laps=min_clean_air_laps,
            clean_mean_time_delta_seconds=clean_mean_time_delta_seconds,
            clean_mean_time_delta_behind_seconds=clean_mean_time_delta_behind_seconds,
            quick_lap_threshold=quick_lap_threshold,
            model_name=model,
            model_config=model_config,
            dry_compounds=dry_compounds,
            track_x_column=track_x_column,
            min_fit_laps=min_fit_laps,
            outlier_sigma=outlier_sigma,
            min_fit_laps_after_outlier_filter=min_fit_laps_after_outlier_filter,
            combined_loss_slope_outlier_sigma=combined_loss_slope_outlier_sigma,
            combined_loss_slope_outlier_min_fits=(
                combined_loss_slope_outlier_min_fits
            ),
            track_evolution_rate_seconds_per_lap=track_evolution_rate,
        )
        if result == "Wet":
            print(f"{year} race {race} {section}: wet tyre used, skipping long-run analysis.")
            continue
        race_results[race] = result
        try:
            race_reference_team = _resolve_reference_team(
                requested_team=requested_reference_team,
                team_performance=result.team_performance,
            )
        except ValueError:
            _print_reference_resolution_debug(
                result,
                requested_reference_team=requested_reference_team,
                race=race,
            )
            raise
        team_frame = result.team_performance.copy()
        team_frame["Year"] = year
        team_frame["Round"] = race
        team_frame["SessionName"] = section
        event_name = _event_name_from_laps(laps)
        if event_name is not None:
            team_frame["EventName"] = event_name
        team_frame = _add_relative_team_performance(
            team_frame,
            reference_team=race_reference_team,
        )

        filtered_path = output_dir / f"long_run_filtered_{year}_{race}_{section}.csv"
        fits_path = output_dir / f"long_run_fits_{year}_{race}_{section}.csv"
        correction_path = output_dir / (
            f"long_run_driver_estimate_sanity_{year}_{race}_{section}.csv"
        )
        stats_path = output_dir / (
            f"long_run_track_evolution_correction_stats_{year}_{race}_{section}.csv"
        )
        reference_path = output_dir / (
            f"long_run_team_compound_reference_{year}_{race}_{section}.csv"
        )
        team_path = output_dir / f"long_run_team_performance_{year}_{race}_{section}.csv"
        result.filtered_laps.to_csv(filtered_path, index=False)
        result.fit_summary.to_csv(fits_path, index=False)
        result.team_compound_correction_summary.to_csv(correction_path, index=False)
        result.compound_correction_stats.to_csv(stats_path, index=False)
        result.team_compound_summary.to_csv(reference_path, index=False)
        team_frame.to_csv(team_path, index=False)
        saved_outputs.extend(
            [
                ("filtered laps csv", filtered_path),
                ("driver stint tyre-zero csv", fits_path),
                ("driver estimate sanity csv", correction_path),
                ("track-evolution correction stats csv", stats_path),
                ("team compound reference csv", reference_path),
                ("race team performance csv", team_path),
            ]
        )

        team_performance_frames.append(team_frame)

        driver_fig, driver_axes = plot_driver_long_run_fits(
            result.filtered_laps,
            result.fits,
            track_x_column=track_x_column,
            title=f"Long-Run Fits {year} R{race} {section}",
        )
        driver_plot_path = output_dir / f"long_run_driver_fits_{year}_{race}_{section}.png"
        driver_fig.savefig(driver_plot_path, dpi=150, bbox_inches="tight")
        figures[f"driver_fits_{race}"] = (driver_fig, driver_axes)
        saved_outputs.append(("driver fit plot png", driver_plot_path))

        print(f"Race {race}: saved {filtered_path}, {fits_path}, {team_path}")
        _print_driver_tyre_zero_estimates(result)
        _print_team_compound_corrections(result)
        _print_compound_correction_stats(result)
        _print_team_compound_reference_estimates(result)
        print(f"\nTeam performance relative to {race_reference_team}")
        print(team_frame.to_string(index=False))

    if not team_performance_frames:
        raise ValueError("No dry long-run results available for the requested race range.")

    all_team_performance = pd.concat(team_performance_frames, ignore_index=True)
    range_label = _race_range_label(races)
    resolved_reference_team = _resolve_reference_team(
        requested_team=requested_reference_team,
        team_performance=all_team_performance,
    )
    all_team_path = output_dir / (
        f"long_run_team_performance_{year}_{range_label}_{section}.csv"
    )
    all_team_performance.to_csv(all_team_path, index=False)
    saved_outputs.append(("aggregate team performance csv", all_team_path))

    trend_fig, trend_ax = plot_long_run_performance_trend(
        all_team_performance,
        reference_team=resolved_reference_team,
        title=(
            f"Long-Run Performance Trend {year} {range_label} {section} "
            f"Relative to {resolved_reference_team}"
        ),
    )
    trend_path = output_dir / (
        f"long_run_performance_trend_{year}_{range_label}_{section}.png"
    )
    trend_fig.savefig(trend_path, dpi=150, bbox_inches="tight")
    figures["performance_trend"] = (trend_fig, trend_ax)
    saved_outputs.append(("performance trend plot png", trend_path))

    if show:
        plt.show()
    else:
        for fig, _ in figures.values():
            plt.close(fig)

    print(f"Reference team: {resolved_reference_team}")
    _print_all_component_fit_reports(race_results)
    _print_compound_data_summary(race_results)
    _print_saved_outputs(saved_outputs)
    return {
        "race_results": race_results,
        "team_performance": all_team_performance,
        "reference_team": resolved_reference_team,
        "figures": figures,
    }


def main() -> None:
    args = _parse_args()
    race_range = _parse_race_range(args.race_range)
    races = list(range(race_range[0], race_range[1] + 1))
    run_long_run_performance_analysis(
        year=args.year,
        races=races,
        section=args.section,
        min_clean_air_laps=args.min_clean_air_laps,
        clean_mean_time_delta_seconds=args.clean_mean_time_delta_seconds,
        clean_mean_time_delta_behind_seconds=args.clean_mean_time_delta_behind_seconds,
        quick_lap_threshold=args.quick_lap_threshold,
        reference_team=args.reference_team,
        model=args.model,
        model_config=_resolve_model_config(args.model_config, args.model_config_json),
        dry_compounds=tuple(args.dry_compounds),
        track_x_column=args.track_x_column,
        min_fit_laps=args.min_fit_laps,
        outlier_sigma=args.outlier_sigma,
        min_fit_laps_after_outlier_filter=args.min_fit_laps_after_outlier_filter,
        combined_loss_slope_outlier_sigma=args.combined_loss_slope_outlier_sigma,
        combined_loss_slope_outlier_min_fits=(
            args.combined_loss_slope_outlier_min_fits
        ),
        track_evolution_rate_source=args.track_evolution_rate_source,
        quali_track_evolution_path=args.quali_track_evolution_path,
        quali_track_evolution_fit=args.quali_track_evolution_fit,
        output_dir=args.output_dir,
        telemetry_cache_dir=args.telemetry_cache_dir,
        force_refresh_telemetry=args.force_refresh_telemetry,
        test=args.test,
        show=args.show,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze race long-run performance over a configurable race range."
    )
    parser.add_argument("--year", type=int, default=SCRIPT_CONFIG["year"])
    parser.add_argument("--race-range", nargs=2, default=SCRIPT_CONFIG["race_range"])
    parser.add_argument("--section", default=SCRIPT_CONFIG["section"])
    parser.add_argument(
        "--reference-team",
        default=SCRIPT_CONFIG["reference_team"],
        help=(
            "Team to use as 100% reference. Default: WCC leader after the "
            "end race in --race-range."
        ),
    )
    parser.add_argument(
        "--min-clean-air-laps",
        type=int,
        default=SCRIPT_CONFIG["min_clean_air_laps"],
        help="Keep only consecutive clean-air runs longer than this lap count.",
    )
    parser.add_argument(
        "--quick-lap-threshold",
        type=float,
        default=SCRIPT_CONFIG["quick_lap_threshold"],
        help=(
            "Only use race laps within this multiple of each driver's fastest "
            "non-pit lap; 1.10 means 110%."
        ),
    )
    parser.add_argument(
        "--clean-mean-time-delta-seconds",
        type=float,
        default=SCRIPT_CONFIG["clean_mean_time_delta_seconds"],
    )
    parser.add_argument(
        "--clean-mean-time-delta-behind-seconds",
        type=_parse_optional_float,
        default=SCRIPT_CONFIG["clean_mean_time_delta_behind_seconds"],
        help=(
            "Require average time gap to the car behind to be greater than this "
            "value; use 'none' to disable battling-from-behind filtering."
        ),
    )
    parser.add_argument(
        "--model",
        choices=(LONG_RUN_MODEL_LINEAR_COMPONENTS, LONG_RUN_MODEL_EXPONENTIAL_TRACK),
        default=SCRIPT_CONFIG["model"],
        help="Predefined model preset. Ignored when --model-config-json is provided.",
    )
    parser.add_argument(
        "--model-config-json",
        default=SCRIPT_CONFIG["model_config_json"],
        help=(
            "JSON dictionary defining combined model terms. Example: "
            '\'{"name":"custom","terms":{"tyre":{"model":"linear",'
            '"x_column":"TyreAgeLaps","parameter":"tyre_slope",'
            '"reference_value":0},"fuel":{"model":"linear",'
            '"x_column":"FuelLapNumber","parameter":"fuel_slope",'
            '"reference_value":1},"track":{"model":"exponential",'
            '"x_column":"LapNumber"}}}\''
        ),
    )
    parser.add_argument(
        "--dry-compounds",
        nargs="+",
        default=SCRIPT_CONFIG["dry_compounds"],
    )
    parser.add_argument("--track-x-column", default=SCRIPT_CONFIG["track_x_column"])
    parser.add_argument("--min-fit-laps", type=int, default=SCRIPT_CONFIG["min_fit_laps"])
    parser.add_argument(
        "--outlier-sigma",
        type=_parse_optional_float,
        default=SCRIPT_CONFIG["outlier_sigma"],
        help=(
            "MAD-based residual threshold for driver stint tyre-life fits. "
            "Use 'none' to disable outlier filtering."
        ),
    )
    parser.add_argument(
        "--min-fit-laps-after-outlier-filter",
        type=int,
        default=SCRIPT_CONFIG["min_fit_laps_after_outlier_filter"],
    )
    parser.add_argument(
        "--combined-loss-slope-outlier-sigma",
        type=_parse_optional_float,
        default=SCRIPT_CONFIG["combined_loss_slope_outlier_sigma"],
        help=(
            "Robust-sigma threshold for rejecting driver stint tyre slopes before "
            "performance aggregation. Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--combined-loss-slope-outlier-min-fits",
        type=int,
        default=SCRIPT_CONFIG["combined_loss_slope_outlier_min_fits"],
        help="Minimum driver stint estimate count before tyre-slope rejection runs.",
    )
    parser.add_argument(
        "--track-evolution-rate-source",
        choices=("quali", "race"),
        default=SCRIPT_CONFIG["track_evolution_rate_source"],
        help=(
            "Use a linear quali track evolution rate in race correction, or use "
            "race-only correction with no external track rate."
        ),
    )
    parser.add_argument(
        "--quali-track-evolution-path",
        type=Path,
        default=SCRIPT_CONFIG["quali_track_evolution_path"],
        help=(
            "Optional CSV path for saved quali track evolution rates. If the "
            "requested race is missing, the script runs quali analysis and updates it."
        ),
    )
    parser.add_argument(
        "--quali-track-evolution-fit",
        choices=(LINEAR_TRACK_EVOLUTION_MODEL,),
        default=SCRIPT_CONFIG["quali_track_evolution_fit"],
        help="Quali track evolution fit to run for long-run correction.",
    )
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
    parser.add_argument("--show", action="store_true", default=SCRIPT_CONFIG["show"])
    parser.set_defaults(model_config=SCRIPT_CONFIG["model_config"])
    return parser.parse_args()


def _parse_race_range(values: list[int | str]) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError("race_range must contain exactly [start_race, end_race].")
    start_value = _parse_round(str(values[0]))
    end_value = _parse_round(str(values[1]))
    if not isinstance(start_value, int) or not isinstance(end_value, int):
        raise ValueError("long-run race range currently requires numeric race numbers.")
    if end_value < start_value:
        raise ValueError("end race must be greater than or equal to start race.")
    return start_value, end_value


def _parse_round(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def _parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def _parse_model_config(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--model-config-json is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--model-config-json must decode to a JSON object.")
    return parsed


def _track_evolution_rate_for_race(
    *,
    year: int,
    race: int | str,
    source: str,
    output_dir: Path,
    configured_path: str | Path | None,
    quali_track_evolution_fit: str,
    dry_compounds: tuple[str, ...],
    telemetry_cache_dir: str | Path | None,
    force_refresh_telemetry: bool,
    test: bool,
) -> float | None:
    if source == "race":
        return None
    if source != "quali":
        raise ValueError("track_evolution_rate_source must be 'quali' or 'race'.")

    path = (
        Path(configured_path)
        if configured_path is not None
        else output_dir / f"quali_track_evolution_{year}_{race}_Q.csv"
    )
    existing_rate = _load_track_evolution_rate(path, year=year, race=race)
    if existing_rate is not None:
        print(
            f"Race {race}: using saved quali track evolution rate "
            f"{existing_rate:.4f}s/lap from {path}"
        )
        return existing_rate

    result = run_quali_performance_tracker(
        year=year,
        race=race,
        quick_lap_threshold=QUALI_SCRIPT_CONFIG["quick_lap_threshold"],
        clean_min_time_delta_seconds=QUALI_SCRIPT_CONFIG[
            "clean_min_time_delta_seconds"
        ],
        clean_mean_time_delta_seconds=QUALI_SCRIPT_CONFIG[
            "clean_mean_time_delta_seconds"
        ],
        dry_compounds=dry_compounds,
        new_tyre_only=QUALI_SCRIPT_CONFIG["new_tyre_only"],
        last_quali_part_only=QUALI_SCRIPT_CONFIG["last_quali_part_only"],
        top_driver_count=QUALI_SCRIPT_CONFIG["top_driver_count"],
        track_evolution_fit=quali_track_evolution_fit,
        teammate_delta_threshold_percent=QUALI_SCRIPT_CONFIG[
            "teammate_delta_threshold_percent"
        ],
        calculate_best_sectors=False,
        telemetry_cache_dir=telemetry_cache_dir,
        force_refresh_telemetry=force_refresh_telemetry,
        test=test,
    )
    if result == "Wet":
        print(
            f"Race {race}: quali track evolution unavailable because wet tyres "
            "were used; falling back to race-only correction."
        )
        return None

    rate = float(result.evolution_rate_seconds_per_lap)
    if not pd.notna(rate):
        raise ValueError(
            f"Race {race}: quali track evolution fit did not produce a finite "
            "seconds-per-lap rate."
        )
    _upsert_track_evolution_rate(
        path,
        _track_evolution_summary_row(result=result, year=year, race=race),
    )
    print(
        f"Race {race}: saved quali track evolution rate {rate:.4f}s/lap to {path}"
    )
    return rate


def _track_evolution_summary_row(
    *,
    result: object,
    year: int,
    race: int | str,
) -> dict[str, object]:
    return {
        "Year": year,
        "Race": race,
        "SessionName": "Q",
        "DominantCompound": result.dominant_compound,
        "EvolutionFitModel": result.evolution_fit_model,
        "EvolutionRateSecondsPerLap": result.evolution_rate_seconds_per_lap,
        "ReferenceSessionLapOrder": result.reference_session_lap_order,
        "LapTimeOnly": result.lap_time_only,
    }


def _load_track_evolution_rate(
    path: Path,
    *,
    year: int,
    race: int | str,
) -> float | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"Year", "Race", "EvolutionRateSecondsPerLap"}
    if required.difference(frame.columns):
        return None
    matches = frame.loc[
        (frame["Year"].astype(str) == str(year))
        & (frame["Race"].astype(str) == str(race))
    ]
    if matches.empty:
        return None
    rate = pd.to_numeric(matches["EvolutionRateSecondsPerLap"], errors="coerce").iloc[-1]
    if not pd.notna(rate):
        return None
    return float(rate)


def _upsert_track_evolution_rate(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        if {"Year", "Race", "SessionName"}.issubset(existing.columns):
            keep = ~(
                (existing["Year"].astype(str) == str(row["Year"]))
                & (existing["Race"].astype(str) == str(row["Race"]))
                & (existing["SessionName"].astype(str) == str(row["SessionName"]))
            )
            existing = existing.loc[keep]
        output = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        output = new_row
    output.to_csv(path, index=False)


def _resolve_model_config(
    model_config: dict[str, Any] | None,
    model_config_json: str | None,
) -> dict[str, Any] | None:
    if model_config_json is not None:
        return _parse_model_config(model_config_json)
    return model_config


def _requested_reference_team(
    *,
    year: int,
    end_race: int | str,
    reference_team: str | None,
) -> str:
    requested_team = reference_team_or_wcc_leader(
        year=year,
        end_race=end_race,
        reference_team=reference_team,
    )
    if reference_team is None:
        print(
            f"Reference team not configured; using WCC leader after race "
            f"{end_race}: {requested_team}"
        )
    return requested_team


def _resolve_reference_team(
    *,
    requested_team: str,
    team_performance: pd.DataFrame,
) -> str:
    return match_team_name(requested_team, team_performance["Team"].dropna().unique())


def _event_name_from_laps(laps: pd.DataFrame) -> str | None:
    for column in ("EventName", "RaceName", "EventLocation", "EventCountry"):
        if column not in laps.columns:
            continue
        values = laps[column].dropna().astype(str)
        if not values.empty:
            return values.iloc[0]
    return None


def _print_reference_resolution_debug(
    result: LongRunPerformanceResult,
    *,
    requested_reference_team: str,
    race: int | str,
) -> None:
    print("\nReference team resolution failed")
    print("=" * 32)
    print(f"Race: {race}")
    print(f"Requested reference team: {requested_reference_team}")
    print(
        "Teams in loaded race laps: "
        f"{_join_debug_values(result.all_laps['Team'].dropna().unique())}"
    )
    print(
        "Teams after clean quick-lap filtering: "
        f"{_join_debug_values(result.filtered_laps['Team'].dropna().unique())}"
    )
    print(
        "Teams with final performance rows: "
        f"{_join_debug_values(result.team_performance['Team'].dropna().unique())}"
    )

    print("\nFiltered clean quick laps by team/compound")
    print("-" * 42)
    _print_debug_frame(
        _count_debug_frame(
            result.filtered_laps,
            group_columns=["Team", "Compound"],
            value_column="LapTimeSeconds",
            count_column="FilteredLapCount",
        )
    )

    print("\nDriver stint tyre-life-zero estimates by team/compound")
    print("-" * 55)
    _print_debug_frame(_driver_fit_debug_frame(result))

    print("\nTeam/compound estimate sanity")
    print("-" * 29)
    _print_debug_frame(_team_compound_correction_debug_frame(result))

    print("\nTeam/compound reference estimates used for performance")
    print("-" * 55)
    if result.team_compound_summary.empty:
        print("  none")
    else:
        columns = [
            "Team",
            "Compound",
            "ReferenceTyreLifeZeroLapNumber",
            "ReferenceLapSelection",
            "EstimatedReferenceLapSeconds",
            "TrackEvolutionRateSecondsPerLap",
            "AverageTrackEvolutionCorrectionSeconds",
            "DriverStintEstimateCount",
        ]
        _print_debug_frame(result.team_compound_summary.loc[:, columns])

    print("\nFinal team performance input")
    print("-" * 28)
    _print_debug_frame(result.team_performance)


def _driver_fit_debug_frame(result: LongRunPerformanceResult) -> pd.DataFrame:
    if result.fit_summary.empty:
        return pd.DataFrame()
    return (
        result.fit_summary.groupby(["Team", "Compound"], as_index=False)
        .agg(
            DriverStintEstimateCount=("EstimatedTyreLifeZeroSeconds", "size"),
            OutlierLapCount=("OutlierLapCount", "sum"),
            DistinctTyreLifeZeroLapCount=("TyreLifeZeroLapNumber", "nunique"),
            MinTyreLifeZeroLapNumber=("TyreLifeZeroLapNumber", "min"),
            MaxTyreLifeZeroLapNumber=("TyreLifeZeroLapNumber", "max"),
            Drivers=("Driver", _join_debug_values),
        )
        .sort_values(["Team", "Compound"])
        .reset_index(drop=True)
    )


def _team_compound_correction_debug_frame(
    result: LongRunPerformanceResult,
) -> pd.DataFrame:
    if result.fit_summary.empty:
        return pd.DataFrame()
    eligibility = (
        result.fit_summary.groupby(["Team", "Compound"], as_index=False)
        .agg(
            DriverStintEstimateCount=("EstimatedTyreLifeZeroSeconds", "size"),
            DistinctTyreLifeZeroLapCount=("TyreLifeZeroLapNumber", "nunique"),
            MinTyreLifeZeroLapNumber=("TyreLifeZeroLapNumber", "min"),
            MaxTyreLifeZeroLapNumber=("TyreLifeZeroLapNumber", "max"),
        )
        .sort_values(["Team", "Compound"])
        .reset_index(drop=True)
    )
    corrections = result.team_compound_correction_summary.loc[
        :,
        [
            "Team",
            "Compound",
            "EstimateIncludedInPerformance",
            "EstimateOutlierReason",
        ],
    ].copy()
    sanity = (
        corrections.groupby(["Team", "Compound"], as_index=False)
        .agg(
            IncludedEstimateCount=("EstimateIncludedInPerformance", "sum"),
            ExcludedEstimateCount=(
                "EstimateIncludedInPerformance",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            EstimateOutlierReasons=("EstimateOutlierReason", _join_debug_values),
        )
    )
    debug = eligibility.merge(sanity, on=["Team", "Compound"], how="left")
    debug["IncludedEstimateCount"] = debug["IncludedEstimateCount"].fillna(0).astype(int)
    debug["ExcludedEstimateCount"] = debug["ExcludedEstimateCount"].fillna(0).astype(int)
    debug["MissingReason"] = "included"
    debug.loc[debug["IncludedEstimateCount"] == 0, "MissingReason"] = (
        "all estimates excluded"
    )
    return debug


def _count_debug_frame(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_column: str,
    count_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(group_columns, as_index=False)[value_column]
        .size()
        .rename(columns={"size": count_column})
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def _print_debug_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("  none")
        return
    print(frame.to_string(index=False))


def _join_debug_values(values) -> str:
    cleaned = sorted(str(value) for value in pd.Series(values).dropna().unique())
    return ", ".join(cleaned) if cleaned else "none"


def _add_relative_team_performance(
    team_performance: pd.DataFrame,
    *,
    reference_team: str,
) -> pd.DataFrame:
    output = team_performance.copy()
    output["ReferenceTeam"] = reference_team
    output["ReferenceLongRunPerformanceSeconds"] = pd.NA
    output["DeltaToReferenceTeamSeconds"] = pd.NA
    output["PercentageToReferenceTeam"] = pd.NA

    for race, group in output.groupby("Round", sort=False):
        reference_row = group.loc[group["Team"] == reference_team]
        if reference_row.empty:
            raise ValueError(
                f"Reference team {reference_team!r} is not available for race {race}."
            )
        reference_seconds = float(
            reference_row["LongRunPerformanceSeconds"].iloc[0]
        )
        mask = output["Round"] == race
        output.loc[mask, "ReferenceLongRunPerformanceSeconds"] = reference_seconds
        output.loc[mask, "DeltaToReferenceTeamSeconds"] = (
            output.loc[mask, "LongRunPerformanceSeconds"] - reference_seconds
        )
        output.loc[mask, "PercentageToReferenceTeam"] = (
            output.loc[mask, "LongRunPerformanceSeconds"] / reference_seconds * 100.0
        )

    numeric_columns = [
        "ReferenceLongRunPerformanceSeconds",
        "DeltaToReferenceTeamSeconds",
        "PercentageToReferenceTeam",
    ]
    output[numeric_columns] = output[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return output


def _print_all_component_fit_reports(
    race_results: dict[int | str, LongRunPerformanceResult],
) -> None:
    print("\nLong-run component fit reports")
    for race, result in race_results.items():
        _print_component_fit_report(race=race, result=result)


def _print_compound_data_summary(
    race_results: dict[int | str, LongRunPerformanceResult],
) -> None:
    print("\nLong-run compound data summary")
    print("=" * 34)
    for race, result in race_results.items():
        print(f"\nRace {race}")
        teams = sorted(result.all_laps["Team"].dropna().unique())
        for team in teams:
            print(f"  {team}")
            used_compounds = _team_race_compounds(result, team=str(team))
            if not used_compounds:
                print("    no dry race compounds found")
                continue
            for compound in used_compounds:
                driver_bits = _compound_driver_sample_bits(
                    result,
                    team=str(team),
                    compound=compound,
                )
                reference_lap_times = _compound_reference_lap_times(
                    result,
                    team=str(team),
                    compound=compound,
                )
                reference_label = (
                    ", ".join(reference_lap_times)
                    if reference_lap_times
                    else "n/a"
                )
                if driver_bits:
                    print(
                        f"    {compound:<6} reference lap time={reference_label} "
                        f"data gathered: {', '.join(driver_bits)}"
                    )
                else:
                    print(
                        f"    {compound:<6} reference lap time=n/a  "
                        "no fitted clean-air data"
                    )


def _team_race_compounds(
    result: LongRunPerformanceResult,
    *,
    team: str,
) -> list[str]:
    laps = result.all_laps.loc[result.all_laps["Team"] == team].copy()
    if "IsOutLap" in laps.columns:
        laps = laps.loc[~laps["IsOutLap"]]
    if "IsInLap" in laps.columns:
        laps = laps.loc[~laps["IsInLap"]]
    return sorted(laps["Compound"].dropna().astype(str).str.upper().unique())


def _compound_driver_sample_bits(
    result: LongRunPerformanceResult,
    *,
    team: str,
    compound: str,
) -> list[str]:
    driver_samples: dict[str, dict[str, int]] = {}
    for (fit_team, driver, fit_compound, _), fit in sorted(result.fits.items()):
        if fit_team != team or fit_compound != compound:
            continue
        samples = driver_samples.setdefault(driver, {"laps": 0, "fits": 0})
        samples["laps"] += fit.lap_count
        samples["fits"] += 1
    return [
        f"{driver} {samples['laps']} laps/{samples['fits']} fits"
        for driver, samples in sorted(driver_samples.items())
    ]


def _compound_reference_lap_times(
    result: LongRunPerformanceResult,
    *,
    team: str,
    compound: str,
) -> list[str]:
    if result.fit_summary.empty:
        return []
    rows = result.fit_summary.loc[
        (result.fit_summary["Team"] == team)
        & (result.fit_summary["Compound"] == compound)
        & result.fit_summary["EstimatedTyreLifeZeroSeconds"].notna()
    ].sort_values(["Driver", "LongRunId"])
    return [
        (
            f"{row.Driver} run {int(row.LongRunId)} "
            f"L0={row.TyreLifeZeroLapNumber:.1f} "
            f"{format_lap_time(row.EstimatedTyreLifeZeroSeconds)}"
        )
        for row in rows.itertuples(index=False)
    ]


def _print_driver_tyre_zero_estimates(result: LongRunPerformanceResult) -> None:
    print("\nDriver stint tyre-life-zero estimates")
    print("=" * 39)
    if result.fit_summary.empty:
        print("  No driver stint estimates.")
        return
    columns = [
        "Team",
        "Driver",
        "Compound",
        "Stint",
        "LongRunId",
        "OriginalLapCount",
        "LapCount",
        "OutlierLapCount",
        "TyreLifeZeroLapNumber",
        "EstimatedTyreLifeZeroSeconds",
        "TyreSlopeSecondsPerLap",
        "EstimateIncludedInPerformance",
        "EstimateOutlierReason",
        "RMSESeconds",
    ]
    frame = result.fit_summary.loc[:, columns].copy()
    frame["EstimatedTyreLifeZero"] = frame["EstimatedTyreLifeZeroSeconds"].map(
        format_lap_time
    )
    frame = frame.drop(columns=["EstimatedTyreLifeZeroSeconds"])
    print(frame.to_string(index=False))


def _print_team_compound_corrections(result: LongRunPerformanceResult) -> None:
    print("\nDriver stint estimate sanity")
    print("=" * 28)
    if result.team_compound_correction_summary.empty:
        print("  No driver stint estimate diagnostics.")
        return
    columns = [
        "Team",
        "Driver",
        "Compound",
        "Stint",
        "LongRunId",
        "LapCount",
        "TyreLifeZeroLapNumber",
        "EstimatedTyreLifeZeroSeconds",
        "TyreSlopeSecondsPerLap",
        "EstimateIncludedInPerformance",
        "EstimateOutlierReason",
        "RMSESeconds",
    ]
    frame = result.team_compound_correction_summary.loc[:, columns].copy()
    frame["EstimatedTyreLifeZero"] = frame["EstimatedTyreLifeZeroSeconds"].map(
        format_lap_time
    )
    frame = frame.drop(columns=["EstimatedTyreLifeZeroSeconds"])
    print(frame.to_string(index=False))


def _print_compound_correction_stats(result: LongRunPerformanceResult) -> None:
    print("\nTrack-evolution correction stats")
    print("=" * 31)
    if result.compound_correction_stats.empty:
        print("  No track-evolution correction stats.")
        return
    columns = [
        "CorrectionGroup",
        "DriverStintEstimateCount",
        "IncludedDriverStintEstimateCount",
        "ExcludedDriverStintEstimateCount",
        "TrackEvolutionRateSecondsPerLap",
    ]
    print(result.compound_correction_stats.loc[:, columns].to_string(index=False))


def _print_team_compound_reference_estimates(result: LongRunPerformanceResult) -> None:
    print("\nTeam compound estimates at reference tyre-life-zero lap")
    print("=" * 56)
    if result.team_compound_summary.empty:
        print("  No team/compound reference estimates.")
        return
    frame = result.team_compound_summary.copy()
    frame["EstimatedReferenceLapTime"] = frame["EstimatedReferenceLapSeconds"].map(
        format_lap_time
    )
    columns = [
        "Team",
        "Compound",
        "ReferenceTyreLifeZeroLapNumber",
        "ReferenceLapSelection",
        "ReferenceLapTeamCount",
        "ReferenceLapRecordCount",
        "EstimatedReferenceLapTime",
        "TrackEvolutionRateSecondsPerLap",
        "AverageTrackEvolutionCorrectionSeconds",
        "DriverStintEstimateCount",
    ]
    print(frame.loc[:, columns].to_string(index=False))


def format_lap_time(value: float | pd.Timedelta | object) -> str:
    if isinstance(value, pd.Timedelta):
        seconds = value.total_seconds()
    else:
        seconds = float(value)
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes}:{remainder:06.3f}"


def _print_component_fit_report(
    *,
    race: int | str,
    result: LongRunPerformanceResult,
) -> None:
    print(f"\nLong-run component fit report - Race {race}")
    if not result.fits:
        print("  No fitted driver/compound groups.")
        return

    for (team, driver, compound, _), fit in sorted(result.fits.items()):
        print(
            f"\n  {team} | {driver} | {compound} | run {fit.long_run_id} "
            f"| stint {fit.stint} "
            f"| samples={fit.lap_count} | runs={fit.run_count} "
            f"| RMSE={fit.rmse_seconds:.3f}s"
        )
        print(f"    Formula: {fit.formula}")
        fit_laps = _fitted_laps_for_group(result.filtered_laps, team, driver, compound, fit)
        for component_name, component_config in fit.model_fit.config["terms"].items():
            component_label = str(component_config.get("label", component_name))
            model_name = str(component_config["model"])
            x_column = str(component_config["x_column"])
            component_laps = fit_laps.dropna(subset=[x_column, fit.model_fit.y_column])
            if component_laps.empty:
                x_range = "n/a"
                samples = 0
            else:
                samples = len(component_laps)
                x_min = float(component_laps[x_column].min())
                x_max = float(component_laps[x_column].max())
                x_range = f"{x_min:.1f}..{x_max:.1f}"
            parameters = _component_parameter_summary(component_config, fit.parameters)
            print(
                f"    - {component_name:<8} {model_name:<11} "
                f"x={component_label:<10} samples={samples:<3} "
                f"range={x_range:<11} {parameters}"
            )


def _fitted_laps_for_group(
    laps: pd.DataFrame,
    team: str,
    driver: str,
    compound: str,
    fit,
) -> pd.DataFrame:
    group = laps.loc[
        (laps["Team"] == team)
        & (laps["Driver"] == driver)
        & (laps["Compound"] == compound)
    ].copy()
    if "Stint" in group.columns:
        group = group.loc[group["Stint"] == fit.stint]
    elif "LongRunId" in group.columns:
        group = group.loc[group["LongRunId"] == fit.long_run_id]
    return group.dropna(subset=[*fit.model_fit.x_columns, fit.model_fit.y_column])


def _component_parameter_summary(
    component_config: dict[str, Any],
    parameters: dict[str, float],
) -> str:
    model_name = str(component_config["model"])
    if model_name == "linear":
        parameter = str(component_config["parameter"])
        return f"{parameter}={parameters[parameter]:.5f}s/lap"
    if model_name == "exponential":
        amplitude = str(component_config["amplitude_parameter"])
        decay = str(component_config["decay_parameter"])
        return (
            f"{amplitude}={parameters[amplitude]:.5f}s, "
            f"{decay}={parameters[decay]:.6f}"
        )
    return "parameters=n/a"


def _print_saved_outputs(saved_outputs: list[tuple[str, Path]]) -> None:
    if not saved_outputs:
        return
    print("\nSaved outputs")
    print("=" * 13)
    for label, path in saved_outputs:
        print(f"  {label:<32} {path}")


def _race_range_label(races: list[int | str]) -> str:
    if len(races) == 1:
        return str(races[0])
    return f"{races[0]}-{races[-1]}"


if __name__ == "__main__":
    main()
