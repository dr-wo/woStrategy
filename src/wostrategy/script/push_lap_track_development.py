from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wostrategy.analysis.push_laps import (
    PushLapSelector,
    add_push_lap_flags,
    fresh_tyre_mask as _fresh_tyre_mask,
    get_dominant_compound as _get_dominant_compound,
    select_top_drivers as _select_top_drivers,
)
from wostrategy.analysis import (
    EXPONENTIAL_TRACK_EVOLUTION_MODEL,
    LINEAR_TRACK_EVOLUTION_MODEL,
    TrackEvolutionModel,
    get_track_evolution_model,
)
from wostrategy.core import get_session_telemetry_cache_path
from wostrategy.plots.track_development import (
    plot_compound_lap_time_fits,
    plot_top_driver_summary,
)
from wostrategy.tools import (
    load_all_session_laps,
    load_all_session_laps_with_telemetry_gap_summary,
)


SCRIPT_CONFIG = {
    "year": 2026,
    "race": 7,
    "section": "Q",
    "quick_lap_threshold": 1.07,
    "clean_mean_time_delta_seconds": 3,
    "dry_compounds": ("SOFT", "MEDIUM", "HARD"),
    "top_driver_count": 10,
    "new_tyre_only": True,
    # "track_evolution_fit": LINEAR_TRACK_EVOLUTION_MODEL,
    "track_evolution_fit": EXPONENTIAL_TRACK_EVOLUTION_MODEL,
    "output": None,
    "telemetry_cache_dir": None,
    "force_refresh_telemetry": True,
    "allow_lap_time_only": True,
    "test": False,
    "show": False,
}


def plot_push_lap_track_development(
    *,
    year: int,
    race: int | str,
    section: int | str,
    quick_lap_threshold: float,
    clean_min_time_delta_seconds: float | None,
    clean_mean_time_delta_seconds: float | None,
    dry_compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
    top_driver_count: int | None = None,
    new_tyre_only: bool = SCRIPT_CONFIG["new_tyre_only"],
    track_evolution_fit: str = SCRIPT_CONFIG["track_evolution_fit"],
    output_path: str | Path | None = None,
    telemetry_cache_dir: str | Path | None = None,
    force_refresh_telemetry: bool = False,
    allow_lap_time_only: bool = SCRIPT_CONFIG["allow_lap_time_only"],
    test: bool = False,
):
    """Load one session and create push-lap track development plots."""
    cache_path = get_session_telemetry_cache_path(
        year=year,
        round_number=race,
        session_name=section,
        cache_dir=telemetry_cache_dir,
    )
    print(f"Telemetry cache path: {cache_path}")

    lap_time_only = False
    try:
        laps = load_all_session_laps_with_telemetry_gap_summary(
            year=year,
            rounds=[race],
            session_names=[section],
            test=test,
            telemetry_cache_dir=telemetry_cache_dir,
            force_refresh_telemetry=force_refresh_telemetry,
        )
    except Exception as exc:
        if not allow_lap_time_only:
            raise
        print(
            f"{year} round {race} session {section}: telemetry loading failed "
            f"({exc}), falling back to lap-time-only mode."
        )
        laps = load_all_session_laps(
            year=year,
            rounds=[race],
            session_names=[section],
            test=test,
        )
        lap_time_only = True
    print(f"Telemetry cache saved/loaded from: {cache_path}")

    if not lap_time_only and allow_lap_time_only and not _has_clean_gap_columns(
        laps,
        clean_min_time_delta_seconds=clean_min_time_delta_seconds,
        clean_mean_time_delta_seconds=clean_mean_time_delta_seconds,
    ):
        print(
            f"{year} round {race} session {section}: telemetry gap columns unavailable, "
            "falling back to lap-time-only mode."
        )
        lap_time_only = True

    selector = PushLapSelector(
        quick_lap_threshold=quick_lap_threshold,
        clean_min_time_delta_seconds=clean_min_time_delta_seconds,
        clean_mean_time_delta_seconds=clean_mean_time_delta_seconds,
        dry_compounds=dry_compounds,
        new_tyre_only=new_tyre_only,
        lap_time_only=lap_time_only,
    )
    flagged_laps = selector.add_flags(laps)
    push_laps = selector.select_push_laps(flagged_laps)
    top_drivers = _select_top_drivers(flagged_laps, top_driver_count)
    selected_models = _selected_plot_models(track_evolution_fit)

    figures = {}
    dry_compounds = tuple(compound.upper() for compound in dry_compounds)
    for fit_model_name, fit_model in selected_models.items():
        title_suffix = f" ({fit_model.fit_label})"
        time_fig, time_axes, time_fits = plot_compound_lap_time_fits(
            push_laps,
            dry_compounds=dry_compounds,
            fit_model=fit_model,
            x_column="LapStartMinutes",
            x_label="Session elapsed time at lap start (min)",
            slope_unit="s/min",
            title=f"Push Lap Track Development by Time and Compound{title_suffix}",
        )
        lap_order_fig, lap_order_axes, lap_order_fits = plot_compound_lap_time_fits(
            push_laps,
            dry_compounds=dry_compounds,
            fit_model=fit_model,
            x_column="SessionLapOrder",
            x_label="Total session lap number",
            slope_unit="s/lap",
            title=f"Push Lap Track Development by Total Lap and Compound{title_suffix}",
        )
        figures[f"{fit_model_name}_session_time"] = (time_fig, time_axes, time_fits)
        figures[f"{fit_model_name}_total_lap"] = (
            lap_order_fig,
            lap_order_axes,
            lap_order_fits,
        )

        if top_drivers is not None:
            summary_time_fig, summary_time_axes, summary_time_fits = plot_top_driver_summary(
                push_laps,
                top_drivers=top_drivers,
                fit_model=fit_model,
                x_column="LapStartMinutes",
                x_label="Session elapsed time at lap start (min)",
                slope_unit="s/min",
                title=(
                    f"Push Lap Summary by Time, Top {len(top_drivers)} Drivers"
                    f"{title_suffix}"
                ),
            )
            summary_lap_fig, summary_lap_axes, summary_lap_fits = plot_top_driver_summary(
                push_laps,
                top_drivers=top_drivers,
                fit_model=fit_model,
                x_column="SessionLapOrder",
                x_label="Total session lap number",
                slope_unit="s/lap",
                title=(
                    f"Push Lap Summary by Total Lap, Top {len(top_drivers)} Drivers"
                    f"{title_suffix}"
                ),
            )
            figures[f"{fit_model_name}_summary_session_time"] = (
                summary_time_fig,
                summary_time_axes,
                summary_time_fits,
            )
            figures[f"{fit_model_name}_summary_total_lap"] = (
                summary_lap_fig,
                summary_lap_axes,
                summary_lap_fits,
            )

    if output_path:
        output_paths = _output_paths(
            output_path,
            include_summary=top_drivers is not None,
            model_names=tuple(selected_models),
        )
        for figure_key, (fig, _, _) in figures.items():
            fig.savefig(output_paths[figure_key], dpi=150, bbox_inches="tight")

    print(f"Quick laps: {int(flagged_laps['IsQuickLap'].sum())}")
    print(f"Lap-time-only mode: {lap_time_only}")
    print(f"Clean quick laps: {int(flagged_laps['IsCleanLap'].sum())}")
    tyre_label = "new-tyre " if new_tyre_only else ""
    print(f"Dry compound {tyre_label}push laps: {len(push_laps)}")
    if top_drivers is not None:
        print(f"Top {len(top_drivers)} drivers: {', '.join(top_drivers)}")
    _print_fit_summary(figures, selected_models, dry_compounds, top_drivers)
    return figures, push_laps


def _print_fit_summary(
    figures: dict[str, tuple[plt.Figure, object, dict[str, dict[str, float] | None]]],
    selected_models: dict[str, TrackEvolutionModel],
    dry_compounds: tuple[str, ...],
    top_drivers: list[str] | None,
) -> None:
    for fit_model_name in selected_models:
        print(f"\n{fit_model_name.title()} track evolution fits")
        if top_drivers is not None:
            _print_fit_rate(
                "Top drivers",
                "time",
                figures[f"{fit_model_name}_summary_session_time"][2].get("top_drivers"),
            )
            _print_fit_rate(
                "Top drivers dominant compound",
                "time",
                figures[f"{fit_model_name}_summary_session_time"][2].get("top_dominant"),
            )
            _print_fit_rate(
                "Top drivers",
                "total lap",
                figures[f"{fit_model_name}_summary_total_lap"][2].get("top_drivers"),
            )
            _print_fit_rate(
                "Top drivers dominant compound",
                "total lap",
                figures[f"{fit_model_name}_summary_total_lap"][2].get("top_dominant"),
            )
        for compound in dry_compounds:
            _print_fit_rate(
                compound,
                "time",
                figures[f"{fit_model_name}_session_time"][2].get(compound),
            )
            _print_fit_rate(
                compound,
                "total lap",
                figures[f"{fit_model_name}_total_lap"][2].get(compound),
            )


def _output_paths(
    output_path: str | Path,
    *,
    include_summary: bool,
    model_names: tuple[str, ...],
) -> dict[str, Path]:
    output_path = Path(output_path)
    output_paths = {}
    for model_name in model_names:
        model_prefix = f"{model_name}_"
        file_prefix = "" if model_names == (LINEAR_TRACK_EVOLUTION_MODEL,) else f"_{model_name}"
        output_paths[f"{model_prefix}session_time"] = output_path.with_name(
            f"{output_path.stem}{file_prefix}{output_path.suffix}"
        )
        output_paths[f"{model_prefix}total_lap"] = output_path.with_name(
            f"{output_path.stem}{file_prefix}_total_lap{output_path.suffix}"
        )
        if include_summary:
            output_paths[f"{model_prefix}summary_session_time"] = output_path.with_name(
                f"{output_path.stem}{file_prefix}_summary{output_path.suffix}"
            )
            output_paths[f"{model_prefix}summary_total_lap"] = output_path.with_name(
                f"{output_path.stem}{file_prefix}_summary_total_lap{output_path.suffix}"
            )
    return output_paths


def _print_fit_rate(
    compound: str,
    axis_label: str,
    fit: dict[str, float] | None,
) -> None:
    if fit is None:
        print(f"{compound} {axis_label} track development rate: not enough push laps")
        return
    if "slope" in fit:
        print(
            f"{compound} {axis_label} track development rate: "
            f"{-fit['slope']:.4f} {fit['slope_unit']}"
        )
        return
    print(
        f"{compound} {axis_label} track evolution fit: "
        f"A={fit['amplitude_seconds']:.3f}, "
        f"k={fit['decay_rate']:.5f}, "
        f"B={fit['offset_seconds']:.3f}, "
        f"rmse={fit['rmse_seconds']:.3f}s"
    )


def main() -> None:
    args = _parse_args()
    race = _parse_round(str(args.race))

    output_path = args.output
    if output_path is None:
        output_name = f"push_lap_track_development_{args.year}_{race}_{args.section}.png"
        output_path = Path("temp") / output_name
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figures, _ = plot_push_lap_track_development(
        year=args.year,
        race=race,
        section=args.section,
        quick_lap_threshold=args.quick_lap_threshold,
        clean_min_time_delta_seconds=args.clean_min_time_delta_seconds,
        clean_mean_time_delta_seconds=args.clean_mean_time_delta_seconds,
        dry_compounds=tuple(args.dry_compounds),
        top_driver_count=args.top_driver_count,
        new_tyre_only=args.new_tyre_only,
        track_evolution_fit=args.track_evolution_fit,
        output_path=output_path,
        telemetry_cache_dir=args.telemetry_cache_dir,
        force_refresh_telemetry=args.force_refresh_telemetry,
        allow_lap_time_only=args.allow_lap_time_only,
        test=args.test,
    )

    if args.show:
        plt.show()
    else:
        for fig, _, _ in figures.values():
            plt.close(fig)
    output_paths = _output_paths(
        output_path,
        include_summary=args.top_driver_count is not None,
        model_names=tuple(_selected_plot_models(args.track_evolution_fit)),
    )
    print("Saved push lap plots:")
    for figure_key in figures:
        print(f"  {figure_key}: {output_paths[figure_key]}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit push lap times against session elapsed time."
    )
    parser.add_argument("--year", type=int, default=SCRIPT_CONFIG["year"])
    parser.add_argument("--race", default=SCRIPT_CONFIG["race"])
    parser.add_argument("--section", default=SCRIPT_CONFIG["section"])
    parser.add_argument(
        "--quick-lap-threshold",
        type=float,
        default=SCRIPT_CONFIG["quick_lap_threshold"],
    )
    parser.add_argument(
        "--clean-min-time-delta-seconds",
        type=_parse_optional_float,
        default=SCRIPT_CONFIG.get("clean_min_time_delta_seconds"),
    )
    parser.add_argument(
        "--clean-mean-time-delta-seconds",
        type=_parse_optional_float,
        default=SCRIPT_CONFIG.get("clean_mean_time_delta_seconds"),
    )
    parser.add_argument(
        "--dry-compounds",
        nargs=3,
        default=SCRIPT_CONFIG["dry_compounds"],
    )
    parser.add_argument(
        "--top-driver-count",
        type=int,
        default=SCRIPT_CONFIG.get("top_driver_count"),
    )
    parser.add_argument(
        "--new-tyre-only",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["new_tyre_only"],
        help="Only use push laps on new tyres for track evolution fits.",
    )
    parser.add_argument(
        "--track-evolution-fit",
        choices=(LINEAR_TRACK_EVOLUTION_MODEL, EXPONENTIAL_TRACK_EVOLUTION_MODEL),
        default=SCRIPT_CONFIG["track_evolution_fit"],
        help=(
            "Track evolution fit to use. Selecting exponential also writes linear "
            "comparison plots."
        ),
    )
    parser.add_argument("--output", type=Path, default=SCRIPT_CONFIG["output"])
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
        "--allow-lap-time-only",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["allow_lap_time_only"],
        help="Fall back to lap-time-only push-lap selection when telemetry gaps are unavailable.",
    )
    parser.add_argument("--test", action="store_true", default=SCRIPT_CONFIG["test"])
    parser.add_argument("--show", action="store_true", default=SCRIPT_CONFIG["show"])
    return parser.parse_args()


def _parse_round(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def _parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


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


def _selected_plot_models(track_evolution_fit: str) -> dict[str, TrackEvolutionModel]:
    if track_evolution_fit == EXPONENTIAL_TRACK_EVOLUTION_MODEL:
        return {
            LINEAR_TRACK_EVOLUTION_MODEL: get_track_evolution_model(
                LINEAR_TRACK_EVOLUTION_MODEL
            ),
            EXPONENTIAL_TRACK_EVOLUTION_MODEL: get_track_evolution_model(
                EXPONENTIAL_TRACK_EVOLUTION_MODEL
            ),
        }
    return {track_evolution_fit: get_track_evolution_model(track_evolution_fit)}


if __name__ == "__main__":
    main()
