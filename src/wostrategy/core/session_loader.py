from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

import pandas as pd

from .telemetry_loader import (
    TelemetryDataLoader,
    load_or_cache_session_telemetry,
    summarize_lap_gap_metrics,
)

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
            session_laps = _add_session_result_rank(session_laps, session)
            all_laps.append(session_laps)

    if all_laps:
        return pd.concat(all_laps, ignore_index=True)

    return pd.DataFrame(columns=empty_columns or ["Year", "Round", "SessionName"])


def load_session_laps_with_telemetry_gap_summary(
    *,
    year: int,
    rounds: list[RoundLike],
    session_names: list[SessionNameLike],
    session_factory: Callable[[RoundLike, SessionNameLike], object],
    enrich_session: Callable[[object], None] | None = None,
    telemetry_loader: TelemetryDataLoader | None = None,
    telemetry_cache_dir: str | Path | None = None,
    force_refresh_telemetry: bool = False,
    log_label: str = "Loading laps and telemetry gaps",
    skip_label: str = "Skipping laps and telemetry gaps",
    empty_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load laps and add per-lap min/mean front-car time delta from telemetry.

    Full per-sample telemetry is cached per session before aggregation. Cache
    files are named ``<year>_<race>_<session>`` under ``cache/telemetry`` by
    default.
    """
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
            session_laps = _add_session_result_rank(session_laps, session)

            try:
                telemetry = load_or_cache_session_telemetry(
                    session,
                    year=year,
                    round_number=round_number,
                    session_name=session_name,
                    telemetry_loader=telemetry_loader,
                    cache_dir=telemetry_cache_dir,
                    force_refresh=force_refresh_telemetry,
                )
                gap_summary = summarize_lap_gap_metrics(telemetry)
            except Exception as exc:
                print(
                    f"{skip_label} telemetry year={year}, round={round_number}, "
                    f"session={session_name}: {exc}"
                )
                gap_summary = pd.DataFrame()

            if not gap_summary.empty:
                session_laps = session_laps.merge(
                    gap_summary,
                    on=["Year", "Round", "SessionName", "Driver", "LapNumber"],
                    how="left",
                )

            all_laps.append(session_laps)

    if all_laps:
        return pd.concat(all_laps, ignore_index=True)

    return pd.DataFrame(columns=empty_columns or ["Year", "Round", "SessionName"])


def _add_session_result_rank(session_laps: pd.DataFrame, session: object) -> pd.DataFrame:
    fastf1_session = getattr(session, "data", session)
    results = getattr(fastf1_session, "results", None)
    if results is None or results.empty:
        return session_laps

    driver_column = _first_existing_column(results, ("Abbreviation", "Driver"))
    rank_column = _first_existing_column(results, ("Position", "ClassifiedPosition"))
    if driver_column is None or rank_column is None:
        return session_laps

    result_rank = results.loc[:, [driver_column, rank_column]].copy()
    result_rank = result_rank.rename(
        columns={
            driver_column: "Driver",
            rank_column: "SessionResultRank",
        }
    )
    result_rank["SessionResultRank"] = pd.to_numeric(
        result_rank["SessionResultRank"],
        errors="coerce",
    )
    result_rank = result_rank.dropna(subset=["Driver", "SessionResultRank"])
    if result_rank.empty:
        return session_laps

    result_rank = result_rank.sort_values("SessionResultRank").drop_duplicates(
        subset=["Driver"],
        keep="first",
    )
    return session_laps.merge(result_rank, on="Driver", how="left")


def _first_existing_column(data: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in data.columns:
            return candidate
    return None
