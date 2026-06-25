from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd
from fastf1 import _api as fastf1_api


SESSION_COLUMNS = {
    "FP1": ("Practice 1",),
    "FP2/SS": ("Practice 2", "Sprint Qualifying", "Sprint Shootout"),
    "FP3/SR": ("Practice 3", "Sprint", "Sprint Race"),
    "Q": ("Qualifying",),
    "R": ("Race",),
}

PROBE_FULL_SESSION = "full-session"
PROBE_API_ENDPOINTS = "api-endpoints"
PROBE_SAMPLE_LAPS = "sample-laps"

SCRIPT_CONFIG = {
    "year": 2026,
    "race_start": 1,
    "race_end": 7,
    "output": None,
    "include_testing": False,
    "force_refresh": True,
    "probe_mode": PROBE_API_ENDPOINTS,
    "sample_laps_per_session": 8,
    "skip_future": True,
    "verbose": True,
}


def telemetry_availability_report(
    *,
    year: int,
    race_start: int = SCRIPT_CONFIG["race_start"],
    race_end: int | None = SCRIPT_CONFIG["race_end"],
    output_path: str | Path | None = SCRIPT_CONFIG["output"],
    include_testing: bool = SCRIPT_CONFIG["include_testing"],
    force_refresh: bool = SCRIPT_CONFIG["force_refresh"],
    probe_mode: str = SCRIPT_CONFIG["probe_mode"],
    sample_laps_per_session: int = SCRIPT_CONFIG["sample_laps_per_session"],
    skip_future: bool = SCRIPT_CONFIG["skip_future"],
    verbose: bool = SCRIPT_CONFIG["verbose"],
) -> pd.DataFrame:
    """Return a race/session table showing whether FastF1 telemetry is available."""
    if not verbose:
        try:
            fastf1.set_log_level("ERROR")
        except AttributeError:
            pass

    schedule = fastf1.get_event_schedule(year, include_testing=include_testing)
    schedule = schedule.loc[schedule["RoundNumber"].notna()].copy()
    schedule["RoundNumber"] = schedule["RoundNumber"].astype(int)
    if race_end is None:
        race_end = int(schedule["RoundNumber"].max())
    if race_end < race_start:
        raise ValueError("race_end must be greater than or equal to race_start.")

    schedule = schedule.loc[
        (schedule["RoundNumber"] >= race_start)
        & (schedule["RoundNumber"] <= race_end)
    ].copy()

    rows = []
    for _, event in schedule.sort_values("RoundNumber").iterrows():
        round_number = int(event["RoundNumber"])
        event_name = str(event["EventName"])
        row: dict[str, object] = {
            "Race": round_number,
            "Grand Prix": event_name,
        }
        for column, session_candidates in SESSION_COLUMNS.items():
            session_name = _session_name_for_column(event, session_candidates)
            if session_name is None:
                row[column] = "n/a"
                continue
            if skip_future and _session_is_future(event, session_name):
                row[column] = "tbd"
                continue

            result = _check_session_telemetry(
                year=year,
                round_number=round_number,
                session_name=session_name,
                force_refresh=force_refresh,
                probe_mode=probe_mode,
                sample_laps_per_session=sample_laps_per_session,
            )
            row[column] = result["Status"]
            if verbose and result.get("Message"):
                print(f"R{round_number} {event_name} {session_name}: {result['Message']}")
        rows.append(row)

    report = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)
        print(f"Wrote telemetry availability CSV to: {output_path}")

    print("\nTelemetry Availability")
    print("=" * 22)
    print(
        "yes = car and position data available, car = car data only, "
        "pos = position data only, partial = incomplete driver coverage, "
        "no = no telemetry loaded, tbd = future session skipped, "
        "n/a = session not scheduled"
    )
    print(report.to_string(index=False))
    return report


def _session_name_for_column(event: pd.Series, candidates: tuple[str, ...]) -> str | None:
    session_columns = [f"Session{index}" for index in range(1, 6)]
    for session_column in session_columns:
        session_name = event.get(session_column)
        if pd.isna(session_name):
            continue
        normalized = _normalize_session_name(session_name)
        for candidate in candidates:
            if normalized == _normalize_session_name(candidate):
                return str(session_name)
    return None


def _session_is_future(event: pd.Series, session_name: str) -> bool:
    session_columns = [f"Session{index}" for index in range(1, 6)]
    for session_column in session_columns:
        if _normalize_session_name(event.get(session_column)) != _normalize_session_name(
            session_name
        ):
            continue
        session_date = pd.to_datetime(
            event.get(f"{session_column}DateUtc"),
            utc=True,
            errors="coerce",
        )
        return pd.notna(session_date) and session_date > pd.Timestamp.now(tz="UTC")
    return False


def _check_session_telemetry(
    *,
    year: int,
    round_number: int,
    session_name: str,
    force_refresh: bool,
    probe_mode: str,
    sample_laps_per_session: int,
) -> dict[str, Any]:
    try:
        cache_context = fastf1.Cache.disabled() if force_refresh else nullcontext()
        with cache_context:
            session = fastf1.get_session(year, round_number, session_name)
            if probe_mode == PROBE_API_ENDPOINTS:
                return _api_endpoint_telemetry_status(session)
            if probe_mode == PROBE_SAMPLE_LAPS:
                session.load(laps=True, telemetry=True, weather=False, messages=False)
            else:
                session.load(laps=False, telemetry=True, weather=False, messages=False)
        if probe_mode == PROBE_SAMPLE_LAPS:
            return _sample_lap_telemetry_status(
                session,
                sample_laps_per_session=sample_laps_per_session,
            )
        car_data = getattr(session, "_car_data", None)
        status = _telemetry_status(car_data, drivers=getattr(session, "drivers", []))
        return status
    except Exception as exc:
        return {"Status": "no", "Message": str(exc)}


def _api_endpoint_telemetry_status(session: object) -> dict[str, Any]:
    car_data = None
    pos_data = None
    car_error = None
    pos_error = None

    try:
        car_data = fastf1_api.car_data(session.api_path)
    except Exception as exc:
        car_error = exc

    try:
        pos_data = fastf1_api.position_data(session.api_path)
    except Exception as exc:
        pos_error = exc

    car_drivers = _nonempty_telemetry_drivers(car_data)
    pos_drivers = _nonempty_telemetry_drivers(pos_data)
    if car_drivers and pos_drivers:
        if car_drivers == pos_drivers:
            return {
                "Status": "yes",
                "Message": f"car+position telemetry for {len(car_drivers)} drivers",
            }
        return {
            "Status": "partial",
            "Message": (
                f"car={len(car_drivers)} drivers, position={len(pos_drivers)} drivers"
            ),
        }
    if car_drivers:
        return {
            "Status": "car",
            "Message": _endpoint_message("car only", car_drivers, pos_error),
        }
    if pos_drivers:
        return {
            "Status": "pos",
            "Message": _endpoint_message("position only", pos_drivers, car_error),
        }
    return {
        "Status": "no",
        "Message": (
            f"car={_error_message(car_error)}, position={_error_message(pos_error)}"
        ),
    }


def _nonempty_telemetry_drivers(data: object) -> set[str]:
    if not isinstance(data, dict):
        return set()
    return {
        str(driver)
        for driver, frame in data.items()
        if isinstance(frame, pd.DataFrame) and not frame.empty
    }


def _endpoint_message(label: str, drivers: set[str], error: Exception | None) -> str:
    message = f"{label} for {len(drivers)} drivers"
    if error is not None:
        message = f"{message}; other endpoint error: {_error_message(error)}"
    return message


def _error_message(error: Exception | None) -> str:
    if error is None:
        return "empty"
    return f"{type(error).__name__}: {error}"


def _sample_lap_telemetry_status(
    session: object,
    *,
    sample_laps_per_session: int,
) -> dict[str, Any]:
    laps = getattr(session, "laps", pd.DataFrame())
    if laps.empty:
        return {"Status": "no", "Message": "no laps loaded"}
    if "LapTime" in laps.columns:
        laps = laps.loc[laps["LapTime"].notna()].copy()
    if laps.empty:
        return {"Status": "no", "Message": "no timed laps loaded"}

    sample_laps = _sample_laps(laps, sample_laps_per_session)
    loaded = 0
    failed = 0
    for _, lap in sample_laps.iterlaps():
        try:
            telemetry = lap.get_telemetry()
        except Exception:
            failed += 1
            continue
        if isinstance(telemetry, pd.DataFrame) and not telemetry.empty:
            loaded += 1
        else:
            failed += 1

    if loaded == len(sample_laps):
        return {"Status": "yes", "Message": f"sample lap telemetry {loaded}/{len(sample_laps)}"}
    if loaded > 0:
        return {
            "Status": "partial",
            "Message": f"sample lap telemetry {loaded}/{len(sample_laps)}",
        }
    return {
        "Status": "no",
        "Message": f"sample lap telemetry 0/{len(sample_laps)}; failures {failed}",
    }


def _sample_laps(laps: pd.DataFrame, sample_laps_per_session: int) -> pd.DataFrame:
    if sample_laps_per_session <= 0 or len(laps) <= sample_laps_per_session:
        return laps
    if {"Driver", "LapTime"}.issubset(laps.columns):
        ranked = laps.sort_values(["Driver", "LapTime"]).groupby("Driver", sort=False).head(1)
        if len(ranked) >= sample_laps_per_session:
            return ranked.head(sample_laps_per_session)
        remaining = laps.loc[~laps.index.isin(ranked.index)].head(
            sample_laps_per_session - len(ranked)
        )
        return pd.concat([ranked, remaining])
    return laps.head(sample_laps_per_session)


def _telemetry_status(car_data: object, *, drivers: list[str]) -> dict[str, Any]:
    if car_data is None:
        return {"Status": "no", "Message": "telemetry missing"}
    if not isinstance(car_data, dict):
        if isinstance(car_data, pd.DataFrame) and not car_data.empty:
            return {"Status": "yes", "Message": f"telemetry rows={len(car_data)}"}
        return {"Status": "no", "Message": "telemetry empty"}

    nonempty_drivers = {
        str(driver)
        for driver, frame in car_data.items()
        if isinstance(frame, pd.DataFrame) and not frame.empty
    }
    listed_drivers = {str(driver) for driver in drivers}
    if not nonempty_drivers:
        return {"Status": "no", "Message": "telemetry missing or empty for all drivers"}
    if listed_drivers and nonempty_drivers >= listed_drivers:
        return {
            "Status": "yes",
            "Message": f"telemetry for {len(nonempty_drivers)}/{len(listed_drivers)} drivers",
        }
    if listed_drivers:
        missing_count = len(listed_drivers - nonempty_drivers)
        return {
            "Status": "partial",
            "Message": (
                f"telemetry for {len(nonempty_drivers)}/{len(listed_drivers)} drivers; "
                f"missing {missing_count}"
            ),
        }
    return {
        "Status": "partial",
        "Message": f"telemetry for {len(nonempty_drivers)} drivers; no driver list loaded",
    }


def _normalize_session_name(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def main() -> None:
    args = _parse_args()
    telemetry_availability_report(
        year=args.year,
        race_start=args.race_start,
        race_end=args.race_end,
        output_path=args.output,
        include_testing=args.include_testing,
        force_refresh=args.force_refresh,
        probe_mode=args.probe_mode,
        sample_laps_per_session=args.sample_laps_per_session,
        skip_future=args.skip_future,
        verbose=args.verbose,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check FastF1 telemetry availability by race and session."
    )
    parser.add_argument("--year", type=int, default=SCRIPT_CONFIG["year"])
    parser.add_argument(
        "--race-start",
        type=int,
        default=SCRIPT_CONFIG["race_start"],
    )
    parser.add_argument(
        "--race-end",
        type=int,
        default=SCRIPT_CONFIG["race_end"],
    )
    parser.add_argument("--output", type=Path, default=SCRIPT_CONFIG["output"])
    parser.add_argument(
        "--include-testing",
        action="store_true",
        default=SCRIPT_CONFIG["include_testing"],
    )
    parser.add_argument(
        "--force-refresh",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["force_refresh"],
        help="Disable FastF1 cache while probing each session telemetry pull.",
    )
    parser.add_argument(
        "--probe-mode",
        choices=(PROBE_API_ENDPOINTS, PROBE_FULL_SESSION, PROBE_SAMPLE_LAPS),
        default=SCRIPT_CONFIG["probe_mode"],
        help=(
            "'api-endpoints' checks raw FastF1 car and position endpoints; "
            "'full-session' checks FastF1 car_data for every listed driver; "
            "'sample-laps' checks lap.get_telemetry() for a small timed-lap sample."
        ),
    )
    parser.add_argument(
        "--sample-laps-per-session",
        type=int,
        default=SCRIPT_CONFIG["sample_laps_per_session"],
    )
    parser.add_argument(
        "--skip-future",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["skip_future"],
        help="Skip future sessions and mark them as tbd.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=SCRIPT_CONFIG["verbose"],
        help="Print per-session failure details while scanning.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
