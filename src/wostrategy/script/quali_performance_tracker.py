from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wostrategy.analysis.quali_performance import (
    AVERAGE_MODE_NOTE,
    CORRECTED_SECTOR_SECONDS,
    DRIVER_COUNT,
    FORMAL_QUALIFYING_SESSION,
    QUALIFYING_PART,
    RESULT_TYPE,
    TEAMMATE_DELTA_PERCENT,
    TEAMMATE_DELTA_SECONDS,
    QualiPerformanceAnalyzer,
    QualiPerformanceResult,
    add_corrected_sector_times as _add_corrected_sector_times,
    add_track_evolution_correction_from_rate as _add_track_evolution_correction,
    calculate_quali_performance as analysis_calculate_quali_performance,
    format_qualifying_part as _format_qualifying_part,
    relative_team_pace_rows,
    team_best_sector_rows as _team_best_sector_rows,
    team_fastest_and_average_rows as _team_fastest_and_average_rows,
)
from wostrategy.analysis import (
    EXPONENTIAL_TRACK_EVOLUTION_MODEL,
    LINEAR_TRACK_EVOLUTION_MODEL,
    TRACK_EVO_CORRECTED_LAP_TIME_SECONDS,
    TRACK_EVO_CORRECTION_SECONDS,
)
from wostrategy.plots.quali_performance import (
    QualiPerformancePlotter,
    result_output_path as _result_output_path,
    save_relative_team_pace_figures,
)
from wostrategy.tools import (
    load_all_session_laps,
    load_all_session_laps_with_telemetry_gap_summary,
)

LAP_TIME_ONLY = "LapTimeOnly"


SCRIPT_CONFIG = {
    "year": 2026,
    "race_range": [1, 7],
    "target_team": "Mercedes",
    "quick_lap_threshold": 1.07,
    "clean_min_time_delta_seconds": None,
    "clean_mean_time_delta_seconds": 3.0,
    "dry_compounds": ("SOFT", "MEDIUM", "HARD"),
    "new_tyre_only": True,
    "last_quali_part_only": True,
    "top_driver_count": 10,
    # "track_evolution_fit": LINEAR_TRACK_EVOLUTION_MODEL,
    "track_evolution_fit": EXPONENTIAL_TRACK_EVOLUTION_MODEL,
    "teammate_delta_threshold_percent": 0.6,
    "calculate_best_sectors": True,
    "allow_lap_time_only": True,
    "telemetry_cache_dir": None,
    "force_refresh_telemetry": False,
    "test": False,
    "output": None,
    "show": False,
}


def calculate_quali_performance(
    laps: pd.DataFrame,
    *,
    quick_lap_threshold: float,
    clean_min_time_delta_seconds: float | None,
    clean_mean_time_delta_seconds: float | None,
    dry_compounds: tuple[str, ...] = SCRIPT_CONFIG["dry_compounds"],
    new_tyre_only: bool = SCRIPT_CONFIG["new_tyre_only"],
    last_quali_part_only: bool = SCRIPT_CONFIG["last_quali_part_only"],
    top_driver_count: int | None = SCRIPT_CONFIG["top_driver_count"],
    track_evolution_fit: str = SCRIPT_CONFIG["track_evolution_fit"],
    lap_time_only: bool = False,
) -> QualiPerformanceResult | str:
    return analysis_calculate_quali_performance(
        laps,
        quick_lap_threshold=quick_lap_threshold,
        clean_min_time_delta_seconds=clean_min_time_delta_seconds,
        clean_mean_time_delta_seconds=clean_mean_time_delta_seconds,
        dry_compounds=dry_compounds,
        new_tyre_only=new_tyre_only,
        last_quali_part_only=last_quali_part_only,
        top_driver_count=top_driver_count,
        track_evolution_fit=track_evolution_fit,
        lap_time_only=lap_time_only,
    )


def run_quali_performance_tracker(
    *,
    year: int,
    race: int | str,
    quick_lap_threshold: float = SCRIPT_CONFIG["quick_lap_threshold"],
    clean_min_time_delta_seconds: float | None = SCRIPT_CONFIG["clean_min_time_delta_seconds"],
    clean_mean_time_delta_seconds: float | None = SCRIPT_CONFIG["clean_mean_time_delta_seconds"],
    dry_compounds: tuple[str, ...] = SCRIPT_CONFIG["dry_compounds"],
    new_tyre_only: bool = SCRIPT_CONFIG["new_tyre_only"],
    last_quali_part_only: bool = SCRIPT_CONFIG["last_quali_part_only"],
    top_driver_count: int | None = SCRIPT_CONFIG["top_driver_count"],
    track_evolution_fit: str = SCRIPT_CONFIG["track_evolution_fit"],
    teammate_delta_threshold_percent: float | None = SCRIPT_CONFIG[
        "teammate_delta_threshold_percent"
    ],
    calculate_best_sectors: bool = SCRIPT_CONFIG["calculate_best_sectors"],
    allow_lap_time_only: bool = SCRIPT_CONFIG["allow_lap_time_only"],
    telemetry_cache_dir: str | Path | None = SCRIPT_CONFIG["telemetry_cache_dir"],
    force_refresh_telemetry: bool = SCRIPT_CONFIG["force_refresh_telemetry"],
    test: bool = SCRIPT_CONFIG["test"],
) -> QualiPerformanceResult | str:
    """Load quali laps/telemetry, calculate track-evolution-corrected team bests."""
    lap_time_only = False
    try:
        laps = load_all_session_laps_with_telemetry_gap_summary(
            year=year,
            rounds=[race],
            session_names=[FORMAL_QUALIFYING_SESSION],
            test=test,
            telemetry_cache_dir=telemetry_cache_dir,
            force_refresh_telemetry=force_refresh_telemetry,
        )
    except Exception as exc:
        if not allow_lap_time_only:
            raise
        print(
            f"{year} race {race} {FORMAL_QUALIFYING_SESSION}: "
            f"telemetry loading failed ({exc}), falling back to lap-time-only mode."
        )
        laps = _load_quali_lap_times(year=year, race=race, test=test)
        lap_time_only = True

    if not lap_time_only and allow_lap_time_only and not _has_clean_gap_columns(
        laps,
        clean_min_time_delta_seconds=clean_min_time_delta_seconds,
        clean_mean_time_delta_seconds=clean_mean_time_delta_seconds,
    ):
        print(
            f"{year} race {race} {FORMAL_QUALIFYING_SESSION}: "
            "telemetry gap columns unavailable, falling back to lap-time-only mode."
        )
        laps = _load_quali_lap_times(year=year, race=race, test=test)
        lap_time_only = True

    if laps.empty:
        raise ValueError(
            f"No laps loaded for year={year}, race={race}, "
            f"section={FORMAL_QUALIFYING_SESSION}"
        )

    analyzer = QualiPerformanceAnalyzer(
        quick_lap_threshold=quick_lap_threshold,
        clean_min_time_delta_seconds=clean_min_time_delta_seconds,
        clean_mean_time_delta_seconds=clean_mean_time_delta_seconds,
        dry_compounds=dry_compounds,
        new_tyre_only=new_tyre_only,
        last_quali_part_only=last_quali_part_only,
        top_driver_count=top_driver_count,
        track_evolution_fit=track_evolution_fit,
        lap_time_only=lap_time_only,
    )
    result = analyzer.calculate(laps)
    if result == "Wet":
        print(
            f"{year} race {race} {FORMAL_QUALIFYING_SESSION}: "
            "Wet tyre used, skipping dry quali tracker."
        )
        return result

    print_evolution_fit_summary(result)
    print_quickest_team_laps(result.quickest_teams)
    return result


def plot_quali_performance_range(
    *,
    year: int,
    race_start: int,
    race_end: int,
    target_team: str,
    quick_lap_threshold: float = SCRIPT_CONFIG["quick_lap_threshold"],
    clean_min_time_delta_seconds: float | None = SCRIPT_CONFIG["clean_min_time_delta_seconds"],
    clean_mean_time_delta_seconds: float | None = SCRIPT_CONFIG["clean_mean_time_delta_seconds"],
    dry_compounds: tuple[str, ...] = SCRIPT_CONFIG["dry_compounds"],
    new_tyre_only: bool = SCRIPT_CONFIG["new_tyre_only"],
    last_quali_part_only: bool = SCRIPT_CONFIG["last_quali_part_only"],
    top_driver_count: int | None = SCRIPT_CONFIG["top_driver_count"],
    track_evolution_fit: str = SCRIPT_CONFIG["track_evolution_fit"],
    teammate_delta_threshold_percent: float | None = SCRIPT_CONFIG[
        "teammate_delta_threshold_percent"
    ],
    calculate_best_sectors: bool = SCRIPT_CONFIG["calculate_best_sectors"],
    allow_lap_time_only: bool = SCRIPT_CONFIG["allow_lap_time_only"],
    telemetry_cache_dir: str | Path | None = SCRIPT_CONFIG["telemetry_cache_dir"],
    force_refresh_telemetry: bool = SCRIPT_CONFIG["force_refresh_telemetry"],
    test: bool = SCRIPT_CONFIG["test"],
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[plt.Figure, plt.Axes]]]:
    """Run multiple quali sessions and plot team pace as target-team percentage."""
    records: list[dict[str, object]] = []
    for race in range(race_start, race_end + 1):
        print(f"\nProcessing {year} race {race} {FORMAL_QUALIFYING_SESSION}")
        result = run_quali_performance_tracker(
            year=year,
            race=race,
            quick_lap_threshold=quick_lap_threshold,
            clean_min_time_delta_seconds=clean_min_time_delta_seconds,
            clean_mean_time_delta_seconds=clean_mean_time_delta_seconds,
            dry_compounds=dry_compounds,
            new_tyre_only=new_tyre_only,
            last_quali_part_only=last_quali_part_only,
            top_driver_count=top_driver_count,
            track_evolution_fit=track_evolution_fit,
            allow_lap_time_only=allow_lap_time_only,
            telemetry_cache_dir=telemetry_cache_dir,
            force_refresh_telemetry=force_refresh_telemetry,
            test=test,
        )
        if result == "Wet":
            continue

        race_records = relative_team_pace_rows(
            result=result,
            year=year,
            race=race,
            target_team=target_team,
            teammate_delta_threshold_percent=teammate_delta_threshold_percent,
            calculate_best_sectors=calculate_best_sectors,
        )
        for record in race_records:
            record[LAP_TIME_ONLY] = result.lap_time_only
        records.extend(race_records)

    if not records:
        raise ValueError("No dry quali results available for the requested race range.")

    summary = pd.DataFrame(records)
    figures = QualiPerformancePlotter(target_team=target_team).plot_relative_team_pace(summary)
    if output_path is not None:
        save_relative_team_pace_figures(figures, output_path)
        save_relative_usage_csv(summary, output_path)

    print_relative_usage_summary(summary)
    return summary, figures


def print_quickest_team_laps(quickest_teams: pd.DataFrame) -> None:
    print("\nQuickest Corrected Quali Laps by Team")
    print("=" * 43)
    for _, row in quickest_teams.iterrows():
        print(f"\n{row['Team']}")
        for slot in (1, 2):
            driver = row.get(f"Driver{slot}")
            if pd.isna(driver):
                continue
            lap_time = row.get(f"Driver{slot}CorrectedLapTimeSeconds")
            lap_number = row.get(f"Driver{slot}LapNumber")
            quali_part = _format_qualifying_part(row.get(f"Driver{slot}QualifyingPart"))
            print(
                f"  {driver:<4}  {quali_part:<2}  lap {int(lap_number):>3}  "
                f"{format_lap_time(lap_time)}"
            )


def print_evolution_fit_summary(result: QualiPerformanceResult) -> None:
    driver_label = (
        "all eligible drivers"
        if result.evolution_drivers is None
        else ", ".join(result.evolution_drivers)
    )
    print("\nTrack Evolution Fit")
    print("=" * 19)
    print(f"Drivers: {driver_label}")
    print(f"Dominant compound: {result.dominant_compound}")
    print(f"Fit model: {result.evolution_fit_model}")
    if result.evolution_fit_model == LINEAR_TRACK_EVOLUTION_MODEL:
        print(f"Evolution rate: {result.evolution_rate_seconds_per_lap:.4f} s/lap")
    else:
        params = result.evolution_fit_parameters
        print(
            "Evolution fit: "
            f"A={params['amplitude_seconds']:.3f}, "
            f"k={params['decay_rate']:.5f}, "
            f"B={params['offset_seconds']:.3f}, "
            f"rmse={params['rmse_seconds']:.3f}s"
        )
    print(f"Reference session lap order: {result.reference_session_lap_order}")


def print_relative_usage_summary(summary: pd.DataFrame) -> None:
    print("\nDrivers/Laps Used in Relative Pace Plot")
    print("=" * 40)
    sort_columns = ["Team", RESULT_TYPE, "Race"]
    for (team, result_type), team_rows in (
        summary.sort_values(sort_columns).groupby(["Team", RESULT_TYPE], sort=True)
    ):
        usage = ", ".join(
            _format_usage_row(row)
            for row in team_rows.itertuples(index=False)
        )
        print(f"{team} ({result_type}): {usage}")


def usage_output_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_usage.csv")


def save_relative_usage_csv(summary: pd.DataFrame, output_path: str | Path) -> Path:
    output_file = usage_output_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    usage = summary.copy()
    usage["SourceLaps"] = usage.apply(_format_usage_row, axis=1)
    usage.to_csv(output_file, index=False)
    return output_file


def format_lap_time(value: float | pd.Timedelta | object) -> str:
    if isinstance(value, pd.Timedelta):
        seconds = value.total_seconds()
    else:
        seconds = float(value)
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes}:{remainder:06.3f}"


def _format_usage_row(row: object) -> str:
    if getattr(row, RESULT_TYPE) == "best_sectors":
        return f"R{int(row.Race)} {getattr(row, AVERAGE_MODE_NOTE)}"

    usage = f"R{int(row.Race)} {row.Driver} {row.QualifyingPart} L{row.LapNumber}"
    note = getattr(row, AVERAGE_MODE_NOTE)
    if isinstance(note, str) and note:
        usage = f"{usage} [{note}]"
    return usage


def _load_quali_lap_times(*, year: int, race: int | str, test: bool) -> pd.DataFrame:
    return load_all_session_laps(
        year=year,
        rounds=[race],
        session_names=[FORMAL_QUALIFYING_SESSION],
        test=test,
    )


def _has_clean_gap_columns(
    laps: pd.DataFrame,
    *,
    clean_min_time_delta_seconds: float | None,
    clean_mean_time_delta_seconds: float | None,
) -> bool:
    min_defined = clean_min_time_delta_seconds is not None
    mean_defined = clean_mean_time_delta_seconds is not None
    if min_defined and mean_defined:
        raise ValueError(
            "Define at most one clean gap filter when allowing lap-time-only fallback."
        )
    if not min_defined and not mean_defined:
        return False

    column = (
        "MinTimeDeltaToDriverAhead"
        if min_defined
        else "MeanTimeDeltaToDriverAhead"
    )
    return column in laps.columns and laps[column].notna().any()


def print_lap_time_only_summary(summaries) -> None:
    lap_only_races: set[int] = set()
    for summary in summaries:
        if LAP_TIME_ONLY not in summary.columns:
            continue
        lap_only_rows = summary.loc[summary[LAP_TIME_ONLY].fillna(False)]
        lap_only_races.update(int(race) for race in lap_only_rows["Race"].dropna().unique())

    race_label = ", ".join(f"R{race}" for race in sorted(lap_only_races)) or "none"
    print(f"Lap-time-only races: {race_label}")


def main() -> None:
    args = _parse_args()
    race_range = _parse_race_range(args.race_range)
    output_path = args.output
    if output_path is None:
        output_path = Path("temp") / (
            f"quali_performance_tracker_{args.year}_"
            f"{race_range[0]}-{race_range[1]}_{_safe_name(args.target_team)}.png"
        )

    summaries = {}
    figure_sets = {}
    for fit_model_name in _selected_fit_names(args.track_evolution_fit):
        model_output_path = _fit_output_path(output_path, fit_model_name, args.track_evolution_fit)
        summary, figures = plot_quali_performance_range(
            year=args.year,
            race_start=race_range[0],
            race_end=race_range[1],
            target_team=args.target_team,
            quick_lap_threshold=args.quick_lap_threshold,
            clean_min_time_delta_seconds=args.clean_min_time_delta_seconds,
            clean_mean_time_delta_seconds=args.clean_mean_time_delta_seconds,
            dry_compounds=tuple(args.dry_compounds),
            new_tyre_only=args.new_tyre_only,
            last_quali_part_only=args.last_quali_part_only,
            top_driver_count=args.top_driver_count,
            track_evolution_fit=fit_model_name,
            teammate_delta_threshold_percent=args.teammate_delta_threshold_percent,
            calculate_best_sectors=args.calculate_best_sectors,
            allow_lap_time_only=args.allow_lap_time_only,
            telemetry_cache_dir=args.telemetry_cache_dir,
            force_refresh_telemetry=args.force_refresh_telemetry,
            test=args.test,
            output_path=model_output_path,
        )
        summaries[fit_model_name] = summary
        figure_sets[fit_model_name] = (model_output_path, figures)

    print("\nSaved relative pace plots:")
    for fit_model_name, (model_output_path, figures) in figure_sets.items():
        for result_type in figures:
            print(
                f"  {fit_model_name} {result_type}: "
                f"{_result_output_path(model_output_path, result_type)}"
            )
        print(f"  {fit_model_name} usage csv: {usage_output_path(model_output_path)}")
    print(f"Rows plotted: {sum(len(summary) for summary in summaries.values())}")
    print_lap_time_only_summary(summaries.values())
    if args.show:
        plt.show()
    else:
        for _, figures in figure_sets.values():
            for fig, _ in figures.values():
                plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track dry quali performance corrected for track evolution."
    )
    parser.add_argument("--year", type=int, default=SCRIPT_CONFIG["year"])
    parser.add_argument(
        "--race-range",
        default=str(SCRIPT_CONFIG["race_range"]),
        help="Inclusive race range, e.g. '[1, 8]'",
    )
    parser.add_argument("--target-team", default=SCRIPT_CONFIG["target_team"])
    parser.add_argument(
        "--quick-lap-threshold",
        type=float,
        default=SCRIPT_CONFIG["quick_lap_threshold"],
    )
    parser.add_argument(
        "--clean-min-time-delta-seconds",
        type=_parse_optional_float,
        default=SCRIPT_CONFIG["clean_min_time_delta_seconds"],
    )
    parser.add_argument(
        "--clean-mean-time-delta-seconds",
        type=_parse_optional_float,
        default=SCRIPT_CONFIG["clean_mean_time_delta_seconds"],
    )
    parser.add_argument(
        "--dry-compounds",
        nargs=3,
        default=SCRIPT_CONFIG["dry_compounds"],
    )
    parser.add_argument(
        "--new-tyre-only",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["new_tyre_only"],
        help="Only use push laps on new tyres for evolution and team bests.",
    )
    parser.add_argument(
        "--last-quali-part-only",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["last_quali_part_only"],
        help=(
            "For each driver, calculate performance only from their last qualifying "
            "part, e.g. use Q2 only for a driver eliminated in Q2."
        ),
    )
    parser.add_argument(
        "--top-driver-count",
        type=_parse_optional_int,
        default=SCRIPT_CONFIG["top_driver_count"],
        help="Use only the top X drivers to fit track evolution; use 'none' to disable.",
    )
    parser.add_argument(
        "--track-evolution-fit",
        choices=(LINEAR_TRACK_EVOLUTION_MODEL, EXPONENTIAL_TRACK_EVOLUTION_MODEL),
        default=SCRIPT_CONFIG["track_evolution_fit"],
        help=(
            "Track evolution fit used for correction. Selecting exponential also writes "
            "linear comparison plots."
        ),
    )
    parser.add_argument(
        "--teammate-delta-threshold-percent",
        type=_parse_optional_float,
        default=SCRIPT_CONFIG["teammate_delta_threshold_percent"],
        help=(
            "In average mode, fall back to the fastest teammate if the two "
            "driver results differ by more than this percentage; use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--calculate-best-sectors",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["calculate_best_sectors"],
        help="Build and plot a team best-sector lap from corrected S1/S2/S3 results.",
    )
    parser.add_argument(
        "--allow-lap-time-only",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["allow_lap_time_only"],
        help=(
            "Try telemetry gap summaries first. If unavailable, select push laps from "
            "lap time only and fit track evolution from those unfiltered lap-time "
            "push laps."
        ),
    )
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
    parser.add_argument("--output", type=Path, default=SCRIPT_CONFIG["output"])
    parser.add_argument("--show", action="store_true", default=SCRIPT_CONFIG["show"])
    return parser.parse_args()


def _parse_race_range(value: str) -> tuple[int, int]:
    cleaned = value.strip().removeprefix("[").removesuffix("]")
    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("race range must look like '[<start>, <end>]'")
    start, end = int(parts[0]), int(parts[1])
    if end < start:
        raise argparse.ArgumentTypeError("race range end must be greater than or equal to start")
    return start, end


def _parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def _parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", ".") else "-" for char in value)


def _selected_fit_names(track_evolution_fit: str) -> tuple[str, ...]:
    if track_evolution_fit == EXPONENTIAL_TRACK_EVOLUTION_MODEL:
        return (LINEAR_TRACK_EVOLUTION_MODEL, EXPONENTIAL_TRACK_EVOLUTION_MODEL)
    return (track_evolution_fit,)


def _fit_output_path(output_path: Path, fit_model_name: str, selected_fit: str) -> Path:
    if selected_fit == EXPONENTIAL_TRACK_EVOLUTION_MODEL:
        return output_path.with_name(f"{output_path.stem}_{fit_model_name}{output_path.suffix}")
    return output_path


if __name__ == "__main__":
    main()
