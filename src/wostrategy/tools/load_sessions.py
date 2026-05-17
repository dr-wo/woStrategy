from __future__ import annotations

from typing import Union

import pandas as pd

from wostrategy.core.session import Session
from wostrategy.core.session_loader import load_session_laps


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
