from __future__ import annotations

from typing import Mapping, Union

import pandas as pd

SessionKey = tuple[Union[int, str], Union[int, str]]


def add_session_value_column(
    laps: pd.DataFrame,
    *,
    values: Mapping[SessionKey, float],
    target_column: str,
    default: float = 0.0,
    round_column: str = "Round",
    session_column: str = "SessionName",
) -> pd.DataFrame:
    """Map per-session scalar values into a dataframe column."""
    laps[target_column] = [
        values.get((row[0], row[1]), default)
        for row in laps[[round_column, session_column]].itertuples(index=False, name=None)
    ]
    return laps
