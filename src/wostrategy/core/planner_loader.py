from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
from wodata import get_cache_path

from wostrategy.tools.load_sessions import load_all_session_laps

RoundLike = Union[int, str]
SessionNameLike = Union[int, str]
SCHEMA = "schema_v1"
REQUIRED_PLANNER_COLUMNS = {"SessionStartPosition"}


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
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    laps.to_pickle(cache_path)
    return laps


def _is_valid_planner_cache(laps: pd.DataFrame) -> bool:
    return REQUIRED_PLANNER_COLUMNS.issubset(laps.columns)


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
    cache_root = Path(__file__).resolve().parents[3] / "cache" / "planner_race_laps"
    return cache_root / f"{year}_{round_number}_{session}.pkl"
