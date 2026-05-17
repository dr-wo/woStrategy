from __future__ import annotations

from typing import Callable, Union

import pandas as pd

RoundLike = Union[int, str]
SessionNameLike = Union[int, str]


def load_session_laps(
    *,
    year: int,
    rounds: list[RoundLike],
    session_names: list[SessionNameLike],
    session_factory: Callable[[RoundLike, SessionNameLike], object],
    enrich_session: Callable[[object], None] | None = None,
    log_label: str = "Loading",
    skip_label: str = "Skipping",
    empty_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load laps from multiple sessions and append Year/Round/SessionName columns."""
    all_laps: list[pd.DataFrame] = []

    for round_number in rounds:
        for session_name in session_names:
            print(f"{log_label} {year} round={round_number} session={session_name}")
            try:
                session = session_factory(round_number, session_name)
                if enrich_session is not None:
                    enrich_session(session)
            except Exception as exc:
                print(
                    f"{skip_label} year={year}, round={round_number}, "
                    f"session={session_name}: {exc}"
                )
                continue

            if session.laps.empty:
                continue

            session_laps = session.laps.copy()
            session_laps["Year"] = year
            session_laps["Round"] = round_number
            session_laps["SessionName"] = session_name
            all_laps.append(session_laps)

    if all_laps:
        return pd.concat(all_laps, ignore_index=True)

    return pd.DataFrame(columns=empty_columns or ["Year", "Round", "SessionName"])
