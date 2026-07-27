from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
from wodata import get_cache_path

from wostrategy.tools.load_sessions import load_all_session_laps

RoundLike = Union[int, str]
SessionNameLike = Union[int, str]
SCHEMA = "schema_v2"
SECTOR_TRACK_STATUS_COLUMNS = {
    "Sector1TrackStatus",
    "Sector2TrackStatus",
    "Sector3TrackStatus",
}
REQUIRED_PLANNER_COLUMNS = {"SessionStartPosition", *SECTOR_TRACK_STATUS_COLUMNS}


def load_race_laps_for_planner(
    year: int,
    round_number: int,
    session: str = "R",
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load race laps for woPlanner, using woData cache paths with old fallback."""
    cache_path = get_planner_laps_cache_path(
        year=year,
        round_number=round_number,
        session=session,
        data_root=data_root,
    )
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if _is_valid_planner_cache(cached):
            return cached

    legacy_path = get_legacy_planner_laps_cache_path(
        year=year,
        round_number=round_number,
        session=session,
    )
    if legacy_path.exists():
        cached = pd.read_pickle(legacy_path)
        if _is_valid_planner_cache(cached):
            return cached

    laps = load_all_session_laps(
        year,
        rounds=[round_number],
        session_names=[session],
        enrich_session=_add_sector_track_statuses,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    laps.to_pickle(cache_path)
    return laps


def _is_valid_planner_cache(laps: pd.DataFrame) -> bool:
    return REQUIRED_PLANNER_COLUMNS.issubset(laps.columns)


def _add_sector_track_statuses(session) -> None:
    laps = session.laps
    status_timeline = getattr(session.data, "track_status", pd.DataFrame())
    required_lap_columns = {"LapStartTime", "Sector1Time", "Sector2Time", "Sector3Time"}
    required_status_columns = {"Time", "Status"}
    if laps.empty or not required_lap_columns.issubset(laps.columns):
        _ensure_empty_sector_track_status_columns(laps)
        return
    if status_timeline.empty or not required_status_columns.issubset(status_timeline.columns):
        _ensure_empty_sector_track_status_columns(laps)
        return

    status_data = status_timeline.loc[:, ["Time", "Status"]].copy()
    status_data["Time"] = pd.to_timedelta(status_data["Time"], errors="coerce")
    status_data["Status"] = status_data["Status"].astype(str)
    status_data = status_data.dropna(subset=["Time"]).sort_values("Time", kind="mergesort")
    if status_data.empty:
        _ensure_empty_sector_track_status_columns(laps)
        return

    for column in SECTOR_TRACK_STATUS_COLUMNS:
        laps[column] = pd.NA

    for index, row in laps.iterrows():
        lap_start = pd.to_timedelta(row.get("LapStartTime"), errors="coerce")
        s1 = pd.to_timedelta(row.get("Sector1Time"), errors="coerce")
        s2 = pd.to_timedelta(row.get("Sector2Time"), errors="coerce")
        s3 = pd.to_timedelta(row.get("Sector3Time"), errors="coerce")
        if pd.isna(lap_start) or pd.isna(s1) or pd.isna(s2) or pd.isna(s3):
            continue
        s1_end = lap_start + s1
        s2_end = s1_end + s2
        s3_end = s2_end + s3
        laps.at[index, "Sector1TrackStatus"] = _track_status_codes_for_interval(
            status_data, lap_start, s1_end
        )
        laps.at[index, "Sector2TrackStatus"] = _track_status_codes_for_interval(
            status_data, s1_end, s2_end
        )
        laps.at[index, "Sector3TrackStatus"] = _track_status_codes_for_interval(
            status_data, s2_end, s3_end
        )


def _ensure_empty_sector_track_status_columns(laps: pd.DataFrame) -> None:
    for column in SECTOR_TRACK_STATUS_COLUMNS:
        if column not in laps.columns:
            laps[column] = pd.NA


def _track_status_codes_for_interval(
    status_data: pd.DataFrame,
    start: pd.Timedelta,
    end: pd.Timedelta,
) -> str:
    if pd.isna(start) or pd.isna(end) or end <= start:
        return ""
    times = status_data["Time"]
    active_before = status_data.loc[times <= start, "Status"]
    if active_before.empty:
        statuses = [str(status_data.iloc[0]["Status"])]
    else:
        statuses = [str(active_before.iloc[-1])]
    interval_changes = status_data.loc[(times > start) & (times < end), "Status"]
    statuses.extend(str(value) for value in interval_changes)

    unique_statuses = []
    for status in statuses:
        if status and status not in unique_statuses:
            unique_statuses.append(status)
    return "".join(unique_statuses)


def get_planner_laps_cache_path(
    *,
    year: int,
    round_number: int,
    session: str,
    data_root: str | Path | None = None,
) -> Path:
    return get_cache_path(
        namespace="wostrategy",
        dataset="planner_race_laps",
        schema=SCHEMA,
        year=year,
        round_number=round_number,
        session=session,
        filename="laps.pkl",
        data_root=Path(data_root) if data_root is not None else None,
    )


def get_legacy_planner_laps_cache_path(
    *,
    year: int,
    round_number: RoundLike,
    session: SessionNameLike,
) -> Path:
    cache_root = Path(__file__).resolve().parents[4] / "cache" / "planner_race_laps"
    return cache_root / f"{year}_{round_number}_{session}.pkl"
