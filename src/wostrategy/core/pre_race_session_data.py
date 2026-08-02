from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd

from wodata import (
    get_fastf1_raw_cache_dir,
    get_fastf1_session_laps_path,
    get_fastf1_telemetry_cache_dir,
)

from wostrategy.analysis.long_run_performance import (
    TYRE_AGE_MODE_STINT,
    _prepare_laps,
    select_clean_air_stints_as_whole,
    select_consecutive_clean_air_runs,
)
from wostrategy.tools.load_sessions import load_all_session_laps_with_telemetry_gap_summary


@dataclass(frozen=True)
class WeekendSessionLoadResult:
    sessions: Mapping[str, pd.DataFrame]
    sources: Mapping[str, str]
    cache_paths: Mapping[str, Path]


def load_cached_weekend_sessions(
    *,
    year: int,
    round_number: int,
    sessions: Sequence[str],
    data_root: str | Path,
    force_refresh: bool = False,
    session_loader: Callable[..., pd.DataFrame] | None = None,
) -> WeekendSessionLoadResult:
    """Read enriched FP laps from woData first, downloading only cache misses."""
    normalized_sessions = tuple(dict.fromkeys(str(value).upper() for value in sessions))
    loader = session_loader or _load_fastf1_session
    output: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    paths = {
        session: get_fastf1_session_laps_path(
            year=year,
            round_number=round_number,
            session=session,
            data_root=data_root,
        )
        for session in normalized_sessions
    }
    for session, path in paths.items():
        if path.exists() and not force_refresh:
            output[session] = pd.read_pickle(path)
            sources[session] = "wodata-cache"
            continue
        try:
            laps = loader(
                year=year,
                round_number=round_number,
                session=session,
                data_root=Path(data_root),
                force_refresh=force_refresh,
            )
        except Exception as exc:
            sources[session] = f"FastF1 load failed: {exc}"
            continue
        if laps is None or laps.empty:
            sources[session] = (
                "FastF1 returned no laps; no enriched woData cache exists at "
                f"{path}. Check year/round/session and network access."
            )
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        laps.to_pickle(path)
        output[session] = laps
        sources[session] = "fastf1-download"
    return WeekendSessionLoadResult(output, sources, paths)


def prepare_weekend_sessions(
    sessions: Mapping[str, pd.DataFrame],
    *,
    min_clean_air_laps: int,
    clean_mean_time_delta_seconds: float,
    clean_mean_time_delta_behind_seconds: float | None,
    quick_lap_threshold: float,
    treat_stint_as_whole: bool = False,
    tyre_age_mode: str = TYRE_AGE_MODE_STINT,
    dry_compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Reuse race-review clean-lap/run selection independently per FP session."""
    selected_sessions: dict[str, pd.DataFrame] = {}
    excluded: dict[str, str] = {}
    selector = (
        select_clean_air_stints_as_whole
        if treat_stint_as_whole
        else select_consecutive_clean_air_runs
    )
    for raw_name, laps in sessions.items():
        session = str(raw_name).upper()
        try:
            prepared = _prepare_laps(
                laps,
                clean_mean_time_delta_seconds=clean_mean_time_delta_seconds,
                clean_mean_time_delta_behind_seconds=clean_mean_time_delta_behind_seconds,
                quick_lap_threshold=quick_lap_threshold,
                dry_compounds=dry_compounds,
                tyre_age_mode=tyre_age_mode,
            )
            selected = selector(prepared, min_clean_air_laps=min_clean_air_laps)
            if selected.empty:
                raise ValueError("no clean consecutive runs matched the configured filters")
        except (KeyError, TypeError, ValueError) as exc:
            excluded[session] = str(exc)
            continue
        selected = selected.copy()
        selected["SessionName"] = session
        selected["RunId"] = (
            session
            + ":"
            + selected["Driver"].astype(str)
            + ":"
            + selected["LongRunId"].astype("Int64").astype(str)
        )
        selected_sessions[session] = selected
    return selected_sessions, excluded


def _load_fastf1_session(
    *,
    year: int,
    round_number: int,
    session: str,
    data_root: Path,
    force_refresh: bool,
) -> pd.DataFrame:
    import fastf1

    raw_cache = get_fastf1_raw_cache_dir(data_root)
    telemetry_cache = get_fastf1_telemetry_cache_dir(data_root)
    raw_cache.mkdir(parents=True, exist_ok=True)
    telemetry_cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(raw_cache), force_renew=force_refresh)
    return load_all_session_laps_with_telemetry_gap_summary(
        year,
        rounds=[round_number],
        session_names=[session],
        telemetry_cache_dir=str(telemetry_cache),
        force_refresh_telemetry=force_refresh,
        force_refresh_session_cache=False,
    )
