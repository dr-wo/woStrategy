from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from wodata.artifacts import (
    write_joint_candidate_costs,
    write_model_snapshot,
    write_weekend_model_config,
)

from wostrategy.plots.pre_race_performance import plot_pre_race_performance_summary
from wostrategy.core.pre_race_session_data import (
    load_cached_weekend_sessions,
    prepare_weekend_sessions,
)
from wostrategy.model.pre_race_performance import (
    WeekendModelConfig,
    derive_race_seed,
    run_joint_weekend_model,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


# Edit these values, then run this module without command-line options.
# Selected sessions are loaded from woData's cache first. Cache misses are
# downloaded through FastF1 and written back under data_root/fastf1/.
SCRIPT_CONFIG = {
    "year": 2026,
    "round_number": 11,
    "sessions": ("FP1", "FP2", "FP3"),  # Any non-empty FP combination.
    # Optional expert/replay input. Example: {"FP2": Path("temp/fp2.csv")}.
    # Leave as None for normal cache-first FastF1 loading.
    "input_csv_by_session": None,
    "sample_count": 100000,
    # None derives a stable seed from this year+round. Set an int only when
    # intentionally validating with an independent race-specific sample bank.
    "primary_seed": None,
    "fuel_rate_bounds": (0.0, 0.10),
    "track_rate_bounds": (-0.05, 0.05),
    "default_degradation_bounds": (0.0, 0.20),
    "default_compound_delta_bounds": (-1.0, 1.0),
    "reference_compound": "MEDIUM",
    # Match race_performance_review: SOFT quickest/highest degradation,
    # then MEDIUM, then HARD.
    "enforce_compound_order": True,
    "clean_lap_noise_sigma": 0.35,
    "session_noise_sigma": {},
    # Existing race-review filtering settings, applied independently per session.
    "quick_lap_threshold": 1.10,
    "min_clean_air_laps": 4,
    "treat_stint_as_whole": False,
    "clean_mean_time_delta_seconds": 2.5,
    "clean_mean_time_delta_behind_seconds": 1.0,
    "dry_compounds": ("SOFT", "MEDIUM", "HARD"),
    "force_refresh_cache": False,
    "data_root": WORKSPACE_ROOT / "woData",
    "plot_output": WORKSPACE_ROOT / "temp" / "weekend_model_aggregate.png",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a joint FP weekend model using woData cache-first FastF1 loading."
        )
    )
    parser.add_argument("--year", type=int, default=SCRIPT_CONFIG["year"])
    parser.add_argument(
        "--round", dest="round_number", type=int, default=SCRIPT_CONFIG["round_number"]
    )
    parser.add_argument("--sessions", nargs="+", default=SCRIPT_CONFIG["sessions"])
    parser.add_argument("--sample-count", type=int, default=SCRIPT_CONFIG["sample_count"])
    parser.add_argument("--seed", type=int, default=SCRIPT_CONFIG["primary_seed"])
    parser.add_argument(
        "--fuel-rate-bounds", type=float, nargs=2, default=SCRIPT_CONFIG["fuel_rate_bounds"]
    )
    parser.add_argument(
        "--track-rate-bounds", type=float, nargs=2, default=SCRIPT_CONFIG["track_rate_bounds"]
    )
    parser.add_argument(
        "--degradation-bounds", type=float, nargs=2,
        default=SCRIPT_CONFIG["default_degradation_bounds"],
    )
    parser.add_argument(
        "--compound-delta-bounds", type=float, nargs=2,
        default=SCRIPT_CONFIG["default_compound_delta_bounds"],
    )
    parser.add_argument("--reference-compound", default=SCRIPT_CONFIG["reference_compound"])
    parser.add_argument(
        "--enforce-compound-order",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["enforce_compound_order"],
    )
    parser.add_argument(
        "--noise-sigma", type=float, default=SCRIPT_CONFIG["clean_lap_noise_sigma"]
    )
    parser.add_argument("--data-root", type=Path, default=SCRIPT_CONFIG["data_root"])
    parser.add_argument("--plot-output", type=Path, default=SCRIPT_CONFIG["plot_output"])
    parser.add_argument(
        "--force-refresh-cache", action="store_true",
        default=SCRIPT_CONFIG["force_refresh_cache"],
    )
    return parser


def run_from_script_config(config: dict | None = None) -> int:
    """Run the complete weekend workflow directly from an editable dictionary."""
    values = dict(SCRIPT_CONFIG if config is None else config)
    return run_weekend_model_script(
        year=values["year"],
        round_number=values["round_number"],
        sessions=values["sessions"],
        input_csv_by_session=values.get("input_csv_by_session"),
        sample_count=values["sample_count"],
        primary_seed=values.get("primary_seed"),
        fuel_rate_bounds=values["fuel_rate_bounds"],
        track_rate_bounds=values["track_rate_bounds"],
        default_degradation_bounds=values["default_degradation_bounds"],
        default_compound_delta_bounds=values["default_compound_delta_bounds"],
        reference_compound=values["reference_compound"],
        enforce_compound_order=values.get("enforce_compound_order", True),
        clean_lap_noise_sigma=values["clean_lap_noise_sigma"],
        session_noise_sigma=values.get("session_noise_sigma", {}),
        quick_lap_threshold=values["quick_lap_threshold"],
        min_clean_air_laps=values["min_clean_air_laps"],
        treat_stint_as_whole=values["treat_stint_as_whole"],
        clean_mean_time_delta_seconds=values["clean_mean_time_delta_seconds"],
        clean_mean_time_delta_behind_seconds=values[
            "clean_mean_time_delta_behind_seconds"
        ],
        dry_compounds=values["dry_compounds"],
        force_refresh_cache=values["force_refresh_cache"],
        data_root=values["data_root"],
        plot_output=values.get("plot_output"),
    )


def run_weekend_model_script(
    *,
    year: int,
    round_number: int,
    sessions: Sequence[str],
    input_csv_by_session: Mapping[str, str | Path] | None,
    sample_count: int,
    primary_seed: int | None,
    fuel_rate_bounds: tuple[float, float],
    track_rate_bounds: tuple[float, float],
    default_degradation_bounds: tuple[float, float],
    default_compound_delta_bounds: tuple[float, float],
    reference_compound: str,
    enforce_compound_order: bool,
    clean_lap_noise_sigma: float,
    session_noise_sigma: Mapping[str, float],
    quick_lap_threshold: float,
    min_clean_air_laps: int,
    treat_stint_as_whole: bool,
    clean_mean_time_delta_seconds: float,
    clean_mean_time_delta_behind_seconds: float | None,
    dry_compounds: tuple[str, ...],
    force_refresh_cache: bool,
    data_root: str | Path,
    plot_output: str | Path | None,
) -> int:
    selected_sessions = tuple(dict.fromkeys(str(value).upper() for value in sessions))
    invalid = set(selected_sessions).difference({"FP1", "FP2", "FP3"})
    if not selected_sessions or invalid:
        raise ValueError(
            "sessions must be a non-empty combination of FP1, FP2 and FP3; "
            f"invalid={sorted(invalid)}"
        )
    data_root = Path(data_root).expanduser()
    print(
        f"[1/4] Loading {year} round {round_number}: "
        f"{', '.join(selected_sessions)} (woData cache first)"
    )
    source_status: dict[str, str] = {}
    if input_csv_by_session:
        raw_sessions = {}
        normalized_csv = {
            str(name).upper(): Path(path).expanduser()
            for name, path in input_csv_by_session.items()
        }
        for session in selected_sessions:
            path = normalized_csv.get(session)
            if path is None or not path.is_file():
                source_status[session] = "CSV unavailable"
                continue
            raw_sessions[session] = _read_session_csv(path, session)
            source_status[session] = f"CSV {path}"
    else:
        loaded = load_cached_weekend_sessions(
            year=int(year), round_number=int(round_number), sessions=selected_sessions,
            data_root=data_root, force_refresh=bool(force_refresh_cache),
        )
        raw_sessions = dict(loaded.sessions)
        source_status.update(loaded.sources)
    for session in selected_sessions:
        raw_lap_count = len(raw_sessions.get(session, ()))
        print(
            f"      {session}: {raw_lap_count} raw laps; "
            f"{source_status.get(session, 'unavailable')}"
        )
    print("[2/4] Selecting dry, representative clean laps independently by session")
    prepared_sessions, filter_exclusions = prepare_weekend_sessions(
        raw_sessions,
        min_clean_air_laps=int(min_clean_air_laps),
        clean_mean_time_delta_seconds=float(clean_mean_time_delta_seconds),
        clean_mean_time_delta_behind_seconds=clean_mean_time_delta_behind_seconds,
        quick_lap_threshold=float(quick_lap_threshold),
        treat_stint_as_whole=bool(treat_stint_as_whole),
        dry_compounds=tuple(dry_compounds),
    )
    exclusion_reasons = {
        session: filter_exclusions.get(session, source_status.get(session, "unavailable"))
        for session in selected_sessions
        if session not in prepared_sessions
    }
    usable_summary = ", ".join(
        f"{session}={len(laps)}"
        for session, laps in prepared_sessions.items()
    ) or "none"
    print(f"      Usable laps: {usable_summary}")
    seed = (
        int(primary_seed)
        if primary_seed is not None
        else derive_race_seed(int(year), int(round_number))
    )
    config = WeekendModelConfig(
        sample_count=int(sample_count), random_seed=seed,
        fuel_rate_bounds=tuple(fuel_rate_bounds), track_rate_bounds=tuple(track_rate_bounds),
        default_degradation_bounds=tuple(default_degradation_bounds),
        default_compound_delta_bounds=tuple(default_compound_delta_bounds),
        reference_compound=reference_compound,
        enforce_compound_order=bool(enforce_compound_order),
        clean_lap_noise_sigma=float(clean_lap_noise_sigma),
        session_noise_sigma={str(k).upper(): float(v) for k, v in session_noise_sigma.items()},
    )
    print(
        f"[3/4] Running joint Monte Carlo: {config.sample_count:,} samples, "
        f"seed={seed}, reference={config.reference_compound.upper()}, "
        f"compound_order={'on' if config.enforce_compound_order else 'off'}"
    )
    result = run_joint_weekend_model(
        prepared_sessions,
        config,
        expected_sessions=("FP1", "FP2", "FP3", "R"),
        expected_compounds=tuple(dry_compounds),
        excluded_session_reasons=exclusion_reasons,
    )
    for snapshot in result.session_snapshots.values():
        write_model_snapshot(
            snapshot, year=int(year), round_number=int(round_number), data_root=data_root
        )
    destination = write_model_snapshot(
        result.aggregate_snapshot,
        year=int(year), round_number=int(round_number), data_root=data_root,
    )
    write_joint_candidate_costs(
        result.candidate_costs.to_dict("records"),
        year=int(year), round_number=int(round_number), data_root=data_root,
    )
    write_weekend_model_config(
        {
            "artifact_schema_version": 2,
            "season": int(year),
            "round_number": int(round_number),
            "source_sessions": list(selected_sessions),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "producer_api": "wostrategy.script.pre_race_analysis.run_weekend_model_script",
            "sample_count": config.sample_count,
            "random_seed": config.random_seed,
            "fuel_rate_bounds": list(config.fuel_rate_bounds),
            "track_rate_bounds": list(config.track_rate_bounds),
            "default_degradation_bounds": list(config.default_degradation_bounds),
            "default_compound_delta_bounds": list(config.default_compound_delta_bounds),
            "reference_compound": config.reference_compound,
            "enforce_compound_order": config.enforce_compound_order,
            "clean_lap_noise_sigma": config.clean_lap_noise_sigma,
            "session_noise_sigma": dict(config.session_noise_sigma),
            "expected_sessions": ["FP1", "FP2", "FP3", "R"],
            "expected_compounds": list(dry_compounds),
            "quick_lap_threshold": float(quick_lap_threshold),
            "min_clean_air_laps": int(min_clean_air_laps),
            "clean_mean_time_delta_seconds": float(clean_mean_time_delta_seconds),
            "clean_mean_time_delta_behind_seconds": clean_mean_time_delta_behind_seconds,
            "clean_lap_mode": "telemetry-gap-pre-race; timing-proxy-live",
        },
        year=int(year), round_number=int(round_number), data_root=data_root,
    )
    if plot_output:
        plot_path = Path(plot_output).expanduser()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_pre_race_performance_summary(result.aggregate_snapshot, str(plot_path))
    print("[4/4] Joint model complete")
    _print_key_results(result.aggregate_snapshot)
    print(f"      Contributing sessions: {', '.join(result.contributing_sessions)}")
    for session, reason in result.excluded_sessions.items():
        print(f"      Excluded {session}: {reason}")
    print(f"      Saved {result.aggregate_snapshot.analysis_id} to {destination}")
    return 0


def _print_key_results(snapshot) -> None:
    """Print a compact median-only summary without flooding long runs."""
    fuel = next(
        (row for row in snapshot.parameters if row.parameter == "fuel_rate"), None
    )
    if fuel is not None:
        print(f"      Shared fuel effect: {fuel.median:+.4f} {fuel.unit}")
    for session in snapshot.contributing_sessions:
        rows = [row for row in snapshot.parameters if row.session == session]
        track = next((row for row in rows if row.parameter == "track_rate"), None)
        degradation = {
            str(row.compound): row.median
            for row in rows
            if row.parameter == "degradation" and row.compound
        }
        deltas = {
            str(row.compound): row.median
            for row in rows
            if row.parameter == "compound_delta" and row.compound
        }
        parts = []
        if track is not None:
            parts.append(f"track={track.median:+.4f} {track.unit}")
        if degradation:
            parts.append(
                "deg S/M/H="
                + "/".join(
                    f"{degradation.get(compound, float('nan')):.4f}"
                    for compound in ("SOFT", "MEDIUM", "HARD")
                )
                + " s/lap"
            )
        if deltas:
            parts.append(
                "delta S/M/H="
                + "/".join(
                    f"{deltas.get(compound, float('nan')):+.4f}"
                    for compound in ("SOFT", "MEDIUM", "HARD")
                )
                + " s"
            )
        print(f"      {session}: " + "; ".join(parts))


def _read_session_csv(path: Path, session: str) -> pd.DataFrame:
    laps = pd.read_csv(path)
    for column in ("LapTime", "SessionTime", "Time", "PitOutTime", "PitInTime"):
        if column in laps.columns:
            laps[column] = pd.to_timedelta(laps[column], errors="coerce")
    laps["SessionName"] = session
    return laps


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    values = dict(SCRIPT_CONFIG)
    values.update(
        year=args.year, round_number=args.round_number, sessions=args.sessions,
        sample_count=args.sample_count, primary_seed=args.seed,
        fuel_rate_bounds=tuple(args.fuel_rate_bounds),
        track_rate_bounds=tuple(args.track_rate_bounds),
        default_degradation_bounds=tuple(args.degradation_bounds),
        default_compound_delta_bounds=tuple(args.compound_delta_bounds),
        reference_compound=args.reference_compound,
        enforce_compound_order=args.enforce_compound_order,
        clean_lap_noise_sigma=args.noise_sigma, data_root=args.data_root,
        plot_output=args.plot_output, force_refresh_cache=args.force_refresh_cache,
    )
    return run_from_script_config(values)


if __name__ == "__main__":
    raise SystemExit(main())
