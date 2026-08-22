from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from wostrategy.model.track_evolution import (
    LINEAR_TRACK_EVOLUTION_MODEL,
    fit_compound_track_evolution,
    get_track_evolution_model,
)


LAP_ROLES = ("OUT", "PREP", "PUSH", "COOL", "IN", "OTHER")
WET_COMPOUNDS = frozenset({"WET", "INTERMEDIATE", "INTER", "INTERS"})
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


@dataclass(frozen=True)
class PreQualiAnalysisResult:
    laps: pd.DataFrame
    durations: pd.DataFrame
    track_progression: pd.DataFrame
    push_threshold_fraction: float

    @property
    def planning_defaults(self) -> dict[str, float | None]:
        return {
            str(row["category"]): (
                None if pd.isna(row["selected_default_s"]) else float(row["selected_default_s"])
            )
            for _, row in self.durations.iterrows()
        }


def analyze_pre_quali(
    laps: pd.DataFrame,
    *,
    year: int,
    round_number: int,
    push_threshold_fraction: float = 0.02,
    sessions: Iterable[str] = ("FP1", "FP2", "FP3"),
    dry_compounds: Iterable[str] = DRY_COMPOUNDS,
) -> PreQualiAnalysisResult:
    """Classify qualifying-style FP runs and derive weekend planning defaults.

    Push eligibility is deliberately session-relative (not driver-relative). The
    run parser accepts only ``OUT [PREP] PUSH (COOL PUSH)* IN``; ambiguous rows
    remain present but are excluded from representative values.
    """
    if push_threshold_fraction < 0:
        raise ValueError("push_threshold_fraction must be non-negative")
    required = {
        "Driver", "LapNumber", "LapTime", "LapStartTime", "PitOutTime", "PitInTime"
    }
    missing = required.difference(laps.columns)
    if missing:
        raise ValueError(f"Laps are missing required columns: {', '.join(sorted(missing))}")

    prepared = laps.copy()
    session_column = _first_column(prepared, ("SessionName", "Session"))
    if session_column is None:
        raise ValueError("Laps are missing required session column: SessionName")
    prepared["_session"] = prepared[session_column].astype(str).str.upper()
    wanted = {str(value).upper() for value in sessions}
    prepared = prepared.loc[prepared["_session"].isin(wanted)].copy()
    prepared["_original_order"] = np.arange(len(prepared))
    prepared["_lap_time_s"] = _seconds(prepared["LapTime"])
    prepared["_compound"] = prepared.get("Compound", pd.Series(index=prepared.index, dtype=object)).astype("string").str.upper()
    prepared["_base_valid"], prepared["_base_reason"] = _base_validity(prepared)
    fastest_candidates = prepared["_lap_time_s"].where(
        prepared["_base_valid"]
        & prepared["PitOutTime"].isna()
        & prepared["PitInTime"].isna()
    )
    fastest_by_session = fastest_candidates.groupby(prepared["_session"]).min()
    fastest = prepared["_session"].map(fastest_by_session)
    prepared["_session_fastest_s"] = fastest
    prepared["_delta_fraction"] = prepared["_lap_time_s"] / fastest - 1.0
    prepared["_push_candidate"] = (
        prepared["_base_valid"]
        & prepared["_lap_time_s"].notna()
        & (prepared["_lap_time_s"] <= fastest * (1.0 + push_threshold_fraction) + 1e-9)
        & prepared["PitOutTime"].isna()
        & prepared["PitInTime"].isna()
    )

    classified = []
    for (session, driver), group in prepared.groupby(["_session", "Driver"], sort=False):
        classified.append(_classify_driver(group, str(session), str(driver)))
    if classified:
        output = pd.concat(classified).sort_values("_original_order")
    else:
        output = prepared.assign(
            lap_role="OTHER", run_id=pd.NA, is_valid_run=False,
            is_valid_for_average=False, invalid_reason="NO_QUALIFYING_RUN",
            planning_duration_s=np.nan,
        )

    result_laps = pd.DataFrame(
        {
            "year": int(year),
            "round": int(round_number),
            "session": output["_session"],
            "driver": output["Driver"].astype(str),
            "team": output.get("Team", pd.Series("", index=output.index)),
            "lap_number": output["LapNumber"],
            "run_id": output["run_id"],
            "lap_role": output["lap_role"],
            "planning_duration_s": output["planning_duration_s"],
            "raw_lap_time_s": output["_lap_time_s"],
            "session_fastest_s": output["_session_fastest_s"],
            "delta_to_session_fastest_fraction": output["_delta_fraction"],
            "is_valid_run": output["is_valid_run"].astype(bool),
            "is_valid_for_average": output["is_valid_for_average"].astype(bool),
            "invalid_reason": output["invalid_reason"],
            "pit_out_time": output["PitOutTime"],
            "pit_in_time": output["PitInTime"],
            "lap_start_time": output["LapStartTime"],
            "lap_end_time": _lap_end(output),
            "compound": output["_compound"],
        }
    ).reset_index(drop=True)
    durations = summarize_planning_durations(result_laps)
    progression = estimate_track_progression(result_laps)
    return PreQualiAnalysisResult(result_laps, durations, progression, push_threshold_fraction)


def _classify_driver(group: pd.DataFrame, session: str, driver: str) -> pd.DataFrame:
    group = group.sort_values(["LapStartTime", "LapNumber", "_original_order"]).copy()
    group["lap_role"] = "OTHER"
    group["run_id"] = pd.NA
    group["is_valid_run"] = False
    group["is_valid_for_average"] = False
    group["invalid_reason"] = group["_base_reason"]
    group["planning_duration_s"] = np.nan
    run_indexes: list[object] = []
    run_number = 0

    def finish(valid_end: bool) -> None:
        nonlocal run_indexes
        if not run_indexes:
            return
        roles = group.loc[run_indexes, "lap_role"].tolist()
        valid_structure = _valid_role_pattern(roles) and valid_end
        base_ok = bool(group.loc[run_indexes, "_base_valid"].all())
        valid = valid_structure and base_ok
        group.loc[run_indexes, "is_valid_run"] = valid
        if valid:
            group.loc[run_indexes, "invalid_reason"] = ""
            group.loc[run_indexes, "is_valid_for_average"] = True
        else:
            reason = "INVALID_RUN_STRUCTURE" if not valid_structure else "UNUSABLE_LAP_IN_RUN"
            empty_reason = group.loc[run_indexes, "invalid_reason"].fillna("").eq("")
            group.loc[group.loc[run_indexes].index[empty_reason], "invalid_reason"] = reason
        run_indexes = []

    seen_push = False
    slow_since_push = 0
    for index, row in group.iterrows():
        out_lap = pd.notna(row["PitOutTime"])
        in_lap = pd.notna(row["PitInTime"])
        if out_lap:
            finish(False)
            run_number += 1
            run_indexes = [index]
            seen_push = False
            slow_since_push = 0
            group.loc[index, ["lap_role", "run_id"]] = ["OUT", f"{session}-{driver}-{run_number}"]
            group.loc[index, "planning_duration_s"] = _duration(row.get("Time"), row["PitOutTime"])
            if in_lap:
                group.loc[index, "invalid_reason"] = "OUT_AND_IN_SAME_LAP"
                finish(False)
            continue
        if not run_indexes:
            group.loc[index, "invalid_reason"] = group.loc[index, "invalid_reason"] or "MISSING_OUT"
            continue
        group.loc[index, "run_id"] = f"{session}-{driver}-{run_number}"
        run_indexes.append(index)
        if in_lap:
            group.loc[index, "lap_role"] = "IN"
            group.loc[index, "planning_duration_s"] = _duration(row["PitInTime"], row["LapStartTime"])
            finish(True)
            continue
        if bool(row["_push_candidate"]):
            group.loc[index, "lap_role"] = "PUSH"
            group.loc[index, "planning_duration_s"] = row["_lap_time_s"]
            if seen_push and slow_since_push != 1:
                group.loc[index, "invalid_reason"] = "MISSING_OR_MULTIPLE_COOL_LAPS"
            seen_push = True
            slow_since_push = 0
        else:
            slow_since_push += 1
            role = "COOL" if seen_push else "PREP"
            group.loc[index, "lap_role"] = role if slow_since_push == 1 else "OTHER"
            group.loc[index, "planning_duration_s"] = row["_lap_time_s"]
            if slow_since_push > 1:
                group.loc[index, "invalid_reason"] = "MULTIPLE_PREP_OR_COOL_LAPS"
    finish(False)
    group.loc[group["lap_role"].eq("OTHER"), "is_valid_for_average"] = False
    return group


def _valid_role_pattern(roles: list[str]) -> bool:
    if len(roles) < 3 or roles[0] != "OUT" or roles[-1] != "IN":
        return False
    middle = roles[1:-1]
    cursor = 1 if middle and middle[0] == "PREP" else 0
    if cursor >= len(middle) or middle[cursor] != "PUSH":
        return False
    cursor += 1
    while cursor < len(middle):
        if middle[cursor:cursor + 2] != ["COOL", "PUSH"]:
            return False
        cursor += 2
    return True


def summarize_planning_durations(laps: pd.DataFrame) -> pd.DataFrame:
    role_map = {"out_lap_s": ("OUT",), "in_lap_s": ("IN",),
                "cool_lap_s": ("PREP", "COOL"), "push_lap_s": ("PUSH",)}
    rows = []
    valid = laps.loc[laps["is_valid_for_average"] & laps["planning_duration_s"].notna()]
    for category, roles in role_map.items():
        raw = valid.loc[valid["lap_role"].isin(roles), "planning_duration_s"].astype(float)
        selected = _iqr_filter(raw)
        rows.append({
            "category": category, "count": int(len(raw)),
            "selected_count": int(len(selected)),
            "mean_s": raw.mean(), "median_s": raw.median(), "std_s": raw.std(ddof=1),
            "min_s": raw.min(), "max_s": raw.max(),
            "selected_default_s": selected.median(), "selection_method": "IQR-filtered median",
        })
    return pd.DataFrame(rows)


def estimate_track_progression(laps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session, group in laps.loc[laps["lap_role"].eq("PUSH") & laps["is_valid_run"]].groupby("session"):
        fit_laps = group.copy().sort_values(["lap_start_time", "driver"])
        fit_laps["SessionLapOrder"] = np.arange(1, len(fit_laps) + 1)
        fit_laps["LapTimeSeconds"] = fit_laps["raw_lap_time_s"]
        fit_laps["Compound"] = fit_laps["compound"]
        compound = _dominant(fit_laps["Compound"])
        try:
            fit = fit_compound_track_evolution(
                fit_laps, compound=compound,
                model=get_track_evolution_model(LINEAR_TRACK_EVOLUTION_MODEL),
                x_column="SessionLapOrder", y_column="LapTimeSeconds", slope_unit="s/field push",
            )
            rate = fit.evolution_rate_seconds_per_lap
            quality = "estimated"
        except (ValueError, np.linalg.LinAlgError):
            rate, quality = np.nan, "insufficient_support"
        rows.append({"session": session, "rate_s_per_field_push": rate,
                     "sample_count": len(fit_laps), "compound": compound, "quality": quality})
    return pd.DataFrame(rows, columns=["session", "rate_s_per_field_push", "sample_count", "compound", "quality"])


def estimate_push_sequence_track_progression(lap_times_s: Iterable[float]) -> float:
    """Fit the existing linear track-evolution model to ordered field pushes."""
    values = [float(value) for value in lap_times_s if pd.notna(value)]
    if len(values) < 2:
        raise ValueError("At least two qualifying push laps are required")
    frame = pd.DataFrame({
        "SessionLapOrder": np.arange(1, len(values) + 1),
        "LapTimeSeconds": values,
        "Compound": "DRY",
    })
    fit = fit_compound_track_evolution(
        frame, compound="DRY",
        model=get_track_evolution_model(LINEAR_TRACK_EVOLUTION_MODEL),
        x_column="SessionLapOrder", y_column="LapTimeSeconds", slope_unit="s/field push",
    )
    return float(fit.evolution_rate_seconds_per_lap)


def _base_validity(laps: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    partial_pit_lap = laps["PitOutTime"].notna() | laps["PitInTime"].notna()
    valid = laps["_lap_time_s"].notna() | partial_pit_lap
    reason = pd.Series("", index=laps.index, dtype=object)
    for column in ("IsAccurate", "IsValid"):
        if column in laps:
            bad = laps[column].notna() & ~laps[column].astype(bool)
            valid &= ~bad
            reason.loc[bad] = "INACCURATE_OR_INVALID"
    if "Deleted" in laps:
        bad = laps["Deleted"].fillna(False).astype(bool)
        valid &= ~bad
        reason.loc[bad] = "DELETED"
    wet = laps["_compound"].isin(WET_COMPOUNDS)
    valid &= ~wet
    reason.loc[wet] = "WET_COMPOUND"
    missing = laps["_lap_time_s"].isna() & laps["PitOutTime"].isna() & laps["PitInTime"].isna()
    reason.loc[missing] = "MISSING_LAP_TIME"
    return valid, reason


def _seconds(values: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(values):
        return values.dt.total_seconds()
    return pd.to_numeric(values, errors="coerce")


def _duration(end: object, start: object) -> float:
    if pd.isna(end) or pd.isna(start):
        return np.nan
    difference = end - start
    return float(difference.total_seconds() if hasattr(difference, "total_seconds") else difference)


def _lap_end(laps: pd.DataFrame) -> pd.Series:
    if "Time" in laps:
        return laps["Time"]
    return laps["LapStartTime"] + laps["LapTime"]


def _iqr_filter(values: pd.Series) -> pd.Series:
    if len(values) < 4:
        return values
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return values.loc[values.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]


def _dominant(values: pd.Series) -> str:
    usable = values.dropna().astype(str)
    return str(usable.mode().iloc[0]) if not usable.empty else "SOFT"


def _first_column(data: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in data), None)
