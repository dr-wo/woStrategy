from __future__ import annotations

from typing import Union

import pandas as pd

from wostrategy.core.session import Session
from wostrategy.core.session_loader import (
    load_session_laps,
    load_session_laps_with_telemetry_gap_summary,
)


def load_all_session_laps(
    year: int,
    rounds: list[Union[int, str]],
    session_names: list[Union[int, str]],
    test: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Load all laps for the requested sessions without long-stint filtering."""
    return load_session_laps(
        year=year,
        rounds=rounds,
        session_names=session_names,
        session_factory=lambda round_number, session_name: Session(
            year=year,
            round=round_number,
            session_name=session_name,
            test=test,
            **kwargs,
        ),
        log_label="Loading full laps for",
        skip_label="Skipping full laps for",
        empty_columns=["Year", "Round", "SessionName", "Driver", "Team"],
    )


def load_all_session_laps_with_telemetry_gap_summary(
    year: int,
    rounds: list[Union[int, str]],
    session_names: list[Union[int, str]],
    test: bool = False,
    telemetry_cache_dir: str | None = None,
    force_refresh_telemetry: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Load laps and add cached per-lap front-car gap summary metrics."""
    return load_session_laps_with_telemetry_gap_summary(
        year=year,
        rounds=rounds,
        session_names=session_names,
        session_factory=lambda round_number, session_name: Session(
            year=year,
            round=round_number,
            session_name=session_name,
            test=test,
            **kwargs,
        ),
        telemetry_cache_dir=telemetry_cache_dir,
        force_refresh_telemetry=force_refresh_telemetry,
        log_label="Loading full laps with telemetry gaps for",
        skip_label="Skipping full laps with telemetry gaps for",
        empty_columns=["Year", "Round", "SessionName", "Driver", "Team"],
    )
