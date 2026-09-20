"""Build the fixed 2026 R14 FP2 push-cool offline evidence set.

This is an isolated R&D generator. It reads two existing woData cache files,
checks five predeclared pit-bounded runs and writes small derived CSV tables. It
never refreshes caches, searches for replacement samples or estimates physical
fuel/tyre quantities.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from wodata import (
    get_data_root,
    get_fastf1_session_laps_path,
    get_fastf1_telemetry_cache_dir,
)
from wostrategy.analysis.push_laps import add_push_lap_flags


YEAR = 2026
ROUND = 14
SESSION = "FP2"
QUICK_LAP_THRESHOLD = 1.10
CLEAN_AHEAD_SECONDS = 2.5
CLEAN_BEHIND_SECONDS = 1.0
FULL_THROTTLE_THRESHOLD = 98.0


@dataclass(frozen=True)
class SequenceSpec:
    driver: str
    first_lap: int
    last_lap: int
    push_laps: tuple[int, ...]
    cool_like_laps: tuple[int, ...]

    @property
    def run_id(self) -> str:
        return (
            f"{YEAR}-R{ROUND:02d}-{SESSION}-{self.driver}-"
            f"L{self.first_lap}-L{self.last_lap}"
        )


SEQUENCES = (
    SequenceSpec("HAM", 1, 5, (2, 4), (3,)),
    SequenceSpec("RUS", 4, 10, (5, 7, 9), (6, 8)),
    SequenceSpec("GAS", 1, 7, (2, 4, 6), (3, 5)),
    SequenceSpec("TSU", 1, 8, (2, 4, 7), (3, 5, 6)),
    SequenceSpec("SAI", 4, 10, (5, 8), (6, 7, 9)),
)

EXCLUDED_CANDIDATES = (
    SequenceSpec("PER", 1, 8, (2, 4, 7), (3, 5, 6)),
    SequenceSpec("VER", 1, 8, (2, 4, 7), (3, 5, 6)),
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timedelta_text(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def _timedelta_seconds(value: object) -> float:
    return float("nan") if pd.isna(value) else float(pd.Timedelta(value).total_seconds())


def _text_or_empty(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def _has_text(value: object) -> bool:
    return not pd.isna(value) and bool(str(value))


def _csv_list(values: tuple[int, ...] | list[int]) -> str:
    return ";".join(str(int(value)) for value in values)


def _load_sources(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    laps_path = get_fastf1_session_laps_path(
        year=YEAR,
        round_number=ROUND,
        session=SESSION,
        data_root=data_root,
    )
    telemetry_path = get_fastf1_telemetry_cache_dir(data_root) / f"{YEAR}_{ROUND}_{SESSION}"
    missing = [path for path in (laps_path, telemetry_path) if not path.is_file()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Required local push-cool source cache is missing; no download or "
            f"sample substitution is allowed: {formatted}"
        )
    return pd.read_pickle(laps_path), pd.read_pickle(telemetry_path), laps_path, telemetry_path


def _flag_laps(laps: pd.DataFrame) -> pd.DataFrame:
    return add_push_lap_flags(
        laps,
        quick_lap_threshold=QUICK_LAP_THRESHOLD,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=CLEAN_AHEAD_SECONDS,
        clean_min_time_delta_behind_seconds=None,
        clean_mean_time_delta_behind_seconds=CLEAN_BEHIND_SECONDS,
    )


def _telemetry_summary(telemetry: pd.DataFrame) -> dict[str, object]:
    if telemetry.empty:
        raise ValueError("Selected lap has no telemetry rows")
    missing = {"Throttle", "Speed"}.difference(telemetry.columns)
    if missing:
        raise ValueError(f"Selected lap telemetry is missing columns: {sorted(missing)}")
    throttle = pd.to_numeric(telemetry["Throttle"], errors="coerce")
    speed = pd.to_numeric(telemetry["Speed"], errors="coerce")
    if throttle.notna().sum() == 0 or speed.notna().sum() == 0:
        raise ValueError("Selected lap has no usable throttle/speed observations")
    return {
        "TelemetryRows": int(len(telemetry)),
        "TelemetryMeanThrottlePct": float(throttle.mean()),
        "TelemetryFullThrottleFraction": float(throttle.ge(FULL_THROTTLE_THRESHOLD).mean()),
        "TelemetryMeanSpeedKph": float(speed.mean()),
        "TelemetryMaxSpeedKph": float(speed.max()),
    }


def _manual_role(spec: SequenceSpec, lap_number: int) -> tuple[str, str]:
    if lap_number == spec.first_lap:
        return "out_lap", "PitOutTime present on declared run boundary"
    if lap_number == spec.last_lap:
        return "in_lap", "PitInTime present on declared run boundary"
    if lap_number in spec.push_laps:
        return (
            "push_lap",
            "existing quick+clean push flag; green; accurate; not deleted; traffic v2 ok",
        )
    if lap_number in spec.cool_like_laps:
        return (
            "cool_like",
            "existing slow_lap between/after pushes; green; traffic v2 ok; "
            "descriptive throttle/speed lower than adjacent push running",
        )
    raise ValueError(f"No declared role for {spec.run_id} lap {lap_number}")


def build_lap_table(
    flagged: pd.DataFrame,
    telemetry: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    traffic_version = str(flagged.attrs.get("traffic_evaluator_version", ""))
    traffic_mode = str(flagged.attrs.get("traffic_combination_mode", ""))
    for spec in SEQUENCES:
        run = flagged.loc[
            (flagged["Driver"] == spec.driver)
            & flagged["LapNumber"].between(spec.first_lap, spec.last_lap)
        ].sort_values("LapNumber").copy()
        run["ElapsedToNextLapStartSeconds"] = (
            run["LapStartTime"].shift(-1) - run["LapStartTime"]
        ).dt.total_seconds()
        for run_order, (_, lap) in enumerate(run.iterrows(), start=1):
            lap_number = int(lap["LapNumber"])
            role, basis = _manual_role(spec, lap_number)
            lap_telemetry = telemetry.loc[
                (telemetry["Driver"] == spec.driver)
                & (telemetry["LapNumber"] == lap_number)
            ]
            summary = _telemetry_summary(lap_telemetry)
            rows.append(
                {
                    "Year": int(lap["Year"]),
                    "Round": int(lap["Round"]),
                    "SessionName": str(lap["SessionName"]),
                    "EventName": str(lap["EventName"]),
                    "EventCountry": str(lap["EventCountry"]),
                    "EventLocation": str(lap["EventLocation"]),
                    "Driver": spec.driver,
                    "Team": str(lap["Team"]),
                    "RunId": spec.run_id,
                    "RunOrder": run_order,
                    "RunIntervalCount": len(spec.push_laps) - 1,
                    "LapNumber": lap_number,
                    "Stint": int(lap["Stint"]),
                    "LapStartTime": _timedelta_text(lap["LapStartTime"]),
                    "LapStartTimeSeconds": _timedelta_seconds(lap["LapStartTime"]),
                    "PitOutTime": _timedelta_text(lap["PitOutTime"]),
                    "PitInTime": _timedelta_text(lap["PitInTime"]),
                    "RawLapTime": _timedelta_text(lap["LapTime"]),
                    "RawLapTimeSeconds": _timedelta_seconds(lap["LapTime"]),
                    "RawLapTimeMissing": bool(pd.isna(lap["LapTime"])),
                    "ElapsedToNextLapStartSeconds": float(
                        lap["ElapsedToNextLapStartSeconds"]
                    ),
                    "ManualRole": role,
                    "ManualRoleBasis": basis,
                    "ExistingLapPatternRole": str(lap["LapPatternRole"]),
                    "IsQuickLap": bool(lap["IsQuickLap"]),
                    "IsCleanLap": bool(lap["IsCleanLap"]),
                    "IsPushLap": bool(lap["IsPushLap"]),
                    "Compound": str(lap["Compound"]),
                    "TyreLife": float(lap["TyreLife"]),
                    "FreshTyre": bool(lap["FreshTyre"]),
                    "TrackStatus": str(lap["TrackStatus"]),
                    "IsAccurate": bool(lap["IsAccurate"]),
                    "Deleted": bool(lap["Deleted"]),
                    "DeletedReason": _text_or_empty(lap["DeletedReason"]),
                    "TrafficStatus": str(lap["TrafficStatus"]),
                    "MeanTimeDeltaToDriverAhead": float(
                        lap["MeanTimeDeltaToDriverAhead"]
                    ),
                    "MeanTimeDeltaToDriverBehind": float(
                        lap["MeanTimeDeltaToDriverBehind"]
                    ),
                    "TrafficEvaluatorVersion": traffic_version,
                    "TrafficCombinationMode": traffic_mode,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def build_run_table(lap_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in SEQUENCES:
        run = lap_table.loc[lap_table["RunId"] == spec.run_id].sort_values("LapNumber")
        first = run.iloc[0]
        last = run.iloc[-1]
        rows.append(
            {
                "Year": YEAR,
                "Round": ROUND,
                "SessionName": SESSION,
                "EventName": first["EventName"],
                "Driver": spec.driver,
                "Team": first["Team"],
                "RunId": spec.run_id,
                "Stint": int(first["Stint"]),
                "FirstLap": spec.first_lap,
                "LastLap": spec.last_lap,
                "LapCount": len(run),
                "PushLapNumbers": _csv_list(spec.push_laps),
                "CoolLikeLapNumbers": _csv_list(spec.cool_like_laps),
                "PushToPushIntervalCount": len(spec.push_laps) - 1,
                "PitOutTime": first["PitOutTime"],
                "PitInTime": last["PitInTime"],
                "Compound": first["Compound"],
                "TrafficEvaluatorVersion": first["TrafficEvaluatorVersion"],
                "DependenceNote": (
                    "intervals from this run share car/run state and are not independent repeats"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_interval_table(lap_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in SEQUENCES:
        run = lap_table.loc[lap_table["RunId"] == spec.run_id].sort_values("LapNumber")
        for interval_index, (left_lap, right_lap) in enumerate(
            zip(spec.push_laps, spec.push_laps[1:]), start=1
        ):
            left = run.loc[run["LapNumber"] == left_lap].iloc[0]
            right = run.loc[run["LapNumber"] == right_lap].iloc[0]
            cool = run.loc[
                run["LapNumber"].between(left_lap + 1, right_lap - 1)
                & run["ManualRole"].eq("cool_like")
            ]
            cool_rows = int(cool["TelemetryRows"].sum())
            weighted_throttle = float(
                np.average(cool["TelemetryMeanThrottlePct"], weights=cool["TelemetryRows"])
            )
            weighted_full = float(
                np.average(cool["TelemetryFullThrottleFraction"], weights=cool["TelemetryRows"])
            )
            weighted_speed = float(
                np.average(cool["TelemetryMeanSpeedKph"], weights=cool["TelemetryRows"])
            )
            rows.append(
                {
                    "Year": YEAR,
                    "Round": ROUND,
                    "SessionName": SESSION,
                    "EventName": left["EventName"],
                    "Driver": spec.driver,
                    "RunId": spec.run_id,
                    "RunIntervalCount": len(spec.push_laps) - 1,
                    "IntervalIndexWithinRun": interval_index,
                    "IntervalId": f"{spec.run_id}-P{left_lap}-P{right_lap}",
                    "FromPushLap": left_lap,
                    "ToPushLap": right_lap,
                    "FromRawLapTimeSeconds": float(left["RawLapTimeSeconds"]),
                    "ToRawLapTimeSeconds": float(right["RawLapTimeSeconds"]),
                    "ObservedPaceDeltaSeconds": float(
                        right["RawLapTimeSeconds"] - left["RawLapTimeSeconds"]
                    ),
                    "CoolLapCount": len(cool),
                    "CoolLapNumbers": _csv_list(cool["LapNumber"].astype(int).tolist()),
                    "PhysicalTyreAgeDelta": float(right["TyreLife"] - left["TyreLife"]),
                    "PushSpacingLaps": right_lap - left_lap,
                    "PushStartElapsedSeconds": float(
                        right["LapStartTimeSeconds"] - left["LapStartTimeSeconds"]
                    ),
                    "CoolElapsedSeconds": float(
                        cool["ElapsedToNextLapStartSeconds"].sum()
                    ),
                    "CoolTelemetryRows": cool_rows,
                    "CoolTelemetryMeanThrottlePct": weighted_throttle,
                    "CoolTelemetryFullThrottleFraction": weighted_full,
                    "CoolTelemetryMeanSpeedKph": weighted_speed,
                    "DependenceNote": (
                        "shares run-level state with other intervals carrying the same RunId"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_exclusion_table(flagged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in EXCLUDED_CANDIDATES:
        run = flagged.loc[
            (flagged["Driver"] == spec.driver)
            & flagged["LapNumber"].between(spec.first_lap, spec.last_lap)
        ].sort_values("LapNumber")
        deleted = run.loc[run["Deleted"].fillna(False).astype(bool)]
        deleted_push = deleted.loc[deleted["IsPushLap"]]
        if deleted_push.empty:
            raise ValueError(f"Expected deleted push evidence is absent for {spec.run_id}")
        rows.append(
            {
                "Year": YEAR,
                "Round": ROUND,
                "SessionName": SESSION,
                "Driver": spec.driver,
                "CandidateRun": spec.run_id,
                "CandidatePushLaps": _csv_list(spec.push_laps),
                "DeletedLapNumbers": _csv_list(deleted["LapNumber"].astype(int).tolist()),
                "DeletedPushLapNumbers": _csv_list(
                    deleted_push["LapNumber"].astype(int).tolist()
                ),
                "ExclusionReason": (
                    "candidate contains a deleted push; production push flagger does not "
                    "apply Deleted, so exclusion is retained here without changing it"
                ),
            }
        )
    return pd.DataFrame(rows)


def _scaled_condition(matrix: np.ndarray) -> float:
    norms = np.linalg.norm(matrix, axis=0)
    scaled = matrix / np.where(norms > 0, norms, 1.0)
    singular = np.linalg.svd(scaled, compute_uv=False)
    if singular[-1] < 1e-12:
        return float("inf")
    return float(singular[0] / singular[-1])


def _fit_combination(
    label: str,
    matrix: np.ndarray,
    response: np.ndarray,
    coefficient_names: tuple[str, ...],
) -> dict[str, object]:
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, response, rcond=None)
    residual = response - matrix @ coefficients
    row: dict[str, object] = {
        "AnalysisType": "combination_fit",
        "Assumption": label,
        "Rank": int(rank),
        "ColumnCount": int(matrix.shape[1]),
        "ScaledConditionNumber": _scaled_condition(matrix),
        "RMSESeconds": float(np.sqrt(np.mean(residual**2))),
    }
    row.update(dict(zip(coefficient_names, coefficients)))
    return row


def build_identifiability_table(intervals: pd.DataFrame) -> pd.DataFrame:
    response = intervals["ObservedPaceDeltaSeconds"].to_numpy(float)
    cool_count = intervals["CoolLapCount"].to_numpy(float)
    tyre_age = intervals["PhysicalTyreAgeDelta"].to_numpy(float)
    session_min = intervals["PushStartElapsedSeconds"].to_numpy(float) / 60.0
    cool_min = intervals["CoolElapsedSeconds"].to_numpy(float) / 60.0
    low_intensity_min = cool_min * (
        1.0 - intervals["CoolTelemetryMeanThrottlePct"].to_numpy(float) / 100.0
    )
    rows: list[dict[str, object]] = []
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        fuel_exposure = 1.0 + rho * cool_count
        matrix = np.column_stack((-fuel_exposure, tyre_age, session_min))
        rows.append(
            {
                "AnalysisType": "fixed_rho_design",
                "Assumption": "no thermal term; rho fixed",
                "Rho": rho,
                "Rank": int(np.linalg.matrix_rank(matrix)),
                "ColumnCount": int(matrix.shape[1]),
                "ScaledConditionNumber": _scaled_condition(matrix),
            }
        )

    base = _fit_combination(
        "unknown rho; no thermal term",
        np.column_stack((np.ones(len(intervals)), cool_count, session_min)),
        response,
        ("CombinationA_TyreMinusFuel", "CombinationB_TyreMinusRhoFuel", "SessionSPerMin"),
    )
    rows.append(base)
    rows.append(
        _fit_combination(
            "unknown rho; thermal linear in cool duration",
            np.column_stack((np.ones(len(intervals)), cool_count, session_min, cool_min)),
            response,
            (
                "CombinationA_TyreMinusFuel",
                "CombinationB_TyreMinusRhoFuel",
                "SessionSPerMin",
                "ThermalCoefficient",
            ),
        )
    )
    rows.append(
        _fit_combination(
            "unknown rho; thermal linear in low-throttle elapsed",
            np.column_stack(
                (np.ones(len(intervals)), cool_count, session_min, low_intensity_min)
            ),
            response,
            (
                "CombinationA_TyreMinusFuel",
                "CombinationB_TyreMinusRhoFuel",
                "SessionSPerMin",
                "ThermalCoefficient",
            ),
        )
    )
    combination_a = float(base["CombinationA_TyreMinusFuel"])
    combination_b = float(base["CombinationB_TyreMinusRhoFuel"])
    for rho in (0.0, 0.25, 0.5, 0.75, 0.9):
        fuel = (combination_b - combination_a) / (1.0 - rho)
        rows.append(
            {
                "AnalysisType": "assumed_rho_mapping",
                "Assumption": "mapping from the same no-thermal combination fit",
                "Rho": rho,
                "NormalizedFuelCoefficient": fuel,
                "LinearTyreCoefficient": combination_a + fuel,
            }
        )
    output = pd.DataFrame(rows)
    output["ObservationCount"] = len(intervals)
    output["CoolCountTyreAgeCorrelation"] = float(
        intervals["CoolLapCount"].corr(intervals["PhysicalTyreAgeDelta"])
    )
    output["IntervalCoolElapsedCorrelation"] = float(
        intervals["PushStartElapsedSeconds"].corr(intervals["CoolElapsedSeconds"])
    )
    return output


def validate_evidence_tables(
    runs: pd.DataFrame,
    laps: pd.DataFrame,
    intervals: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> None:
    expected_run_ids = {spec.run_id for spec in SEQUENCES}
    if set(runs["RunId"]) != expected_run_ids or set(laps["RunId"]) != expected_run_ids:
        raise ValueError("Persisted run identities differ from the five declared sequences")
    if len(runs) != 5 or len(laps) != 34 or len(intervals) != 8:
        raise ValueError(
            f"Expected 5 runs, 34 physical laps and 8 intervals; got "
            f"{len(runs)}, {len(laps)}, {len(intervals)}"
        )
    for spec in SEQUENCES:
        run = laps.loc[laps["RunId"] == spec.run_id].sort_values("LapNumber")
        expected_laps = list(range(spec.first_lap, spec.last_lap + 1))
        if run["LapNumber"].astype(int).tolist() != expected_laps:
            raise ValueError(f"{spec.run_id} lost or gained a physical lap")
        if not _has_text(run.iloc[0]["PitOutTime"]) or not _has_text(
            run.iloc[-1]["PitInTime"]
        ):
            raise ValueError(f"{spec.run_id} lacks its declared pit boundary")
        if run.iloc[1:-1][["PitOutTime", "PitInTime"]].fillna("").ne("").any().any():
            raise ValueError(f"{spec.run_id} crosses an intermediate pit visit")
        selected_pushes = run.loc[run["ManualRole"].eq("push_lap")]
        if selected_pushes["Deleted"].astype(bool).any():
            raise ValueError(f"{spec.run_id} contains a deleted selected push")
        if not selected_pushes["IsPushLap"].astype(bool).all():
            raise ValueError(f"{spec.run_id} selected push no longer matches existing flagger")
        missing_raw = run["RawLapTimeMissing"].astype(bool)
        if run.loc[missing_raw, "RawLapTimeSeconds"].notna().any():
            raise ValueError(f"{spec.run_id} overwrote a missing raw LapTime")
        if run.loc[run["ManualRole"].eq("cool_like")].empty:
            raise ValueError(f"{spec.run_id} lost its cool-like physical laps")
        expected_pattern_roles = {
            "out_lap": "out_lap",
            "push_lap": "push_lap",
            "cool_like": "slow_lap",
            "in_lap": "in_lap",
        }
        actual_pattern_roles = run["ManualRole"].map(expected_pattern_roles)
        if not run["ExistingLapPatternRole"].eq(actual_pattern_roles).all():
            raise ValueError(f"{spec.run_id} manual roles diverge from the existing lap pattern")
        for cool_lap in run.loc[run["ManualRole"].eq("cool_like")].itertuples():
            previous_push = run.loc[
                run["ManualRole"].eq("push_lap")
                & run["LapNumber"].lt(cool_lap.LapNumber)
            ].tail(1)
            next_push = run.loc[
                run["ManualRole"].eq("push_lap")
                & run["LapNumber"].gt(cool_lap.LapNumber)
            ].head(1)
            adjacent_pushes = pd.concat((previous_push, next_push))
            if adjacent_pushes.empty:
                raise ValueError(f"{spec.run_id} cool-like lap lacks an adjacent push")
            if not (
                cool_lap.TelemetryMeanThrottlePct
                < adjacent_pushes["TelemetryMeanThrottlePct"].min()
                and cool_lap.TelemetryMeanSpeedKph
                < adjacent_pushes["TelemetryMeanSpeedKph"].min()
            ):
                raise ValueError(
                    f"{spec.run_id} lap {cool_lap.LapNumber} lacks its descriptive "
                    "cool-like telemetry contrast"
                )

    for interval in intervals.itertuples(index=False):
        run = laps.loc[laps["RunId"] == interval.RunId]
        left = run.loc[run["LapNumber"] == interval.FromPushLap].iloc[0]
        right = run.loc[run["LapNumber"] == interval.ToPushLap].iloc[0]
        recalculated = right["RawLapTimeSeconds"] - left["RawLapTimeSeconds"]
        if not np.isclose(recalculated, interval.ObservedPaceDeltaSeconds, atol=1e-12):
            raise ValueError(f"{interval.IntervalId} pace delta is not reproducible")
        between = run.loc[
            run["LapNumber"].between(interval.FromPushLap + 1, interval.ToPushLap - 1)
        ]
        if len(between) != interval.CoolLapCount:
            raise ValueError(f"{interval.IntervalId} omitted an intermediate physical lap")
        if not between["ManualRole"].eq("cool_like").all():
            raise ValueError(f"{interval.IntervalId} contains an undeclared intermediate role")

    if set(exclusions["Driver"]) != {"PER", "VER"}:
        raise ValueError("Deleted-push exclusions for PER and VER were not retained")
    if exclusions["DeletedPushLapNumbers"].astype(str).str.len().eq(0).any():
        raise ValueError("Deleted-push exclusion lacks the source lap number")


def _repository_head(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_source_manifest(
    laps: pd.DataFrame,
    telemetry: pd.DataFrame,
    laps_path: Path,
    telemetry_path: Path,
    data_root: Path,
    repository_root: Path,
) -> pd.DataFrame:
    generator_path = Path(__file__).resolve()
    code_paths = (
        generator_path,
        repository_root / "src/wostrategy/analysis/push_laps.py",
        repository_root / "src/wostrategy/model/pace_contract.py",
    )
    rows = [
        {
            "SourceKind": "enriched_session_laps",
            "PathBase": "data_root",
            "RelativePath": str(laps_path.relative_to(data_root)),
            "SizeBytes": laps_path.stat().st_size,
            "SHA256": file_sha256(laps_path),
            "Rows": len(laps),
            "Columns": len(laps.columns),
            "TrafficEvaluatorVersion": laps.attrs.get("traffic_evaluator_version", ""),
            "TrafficCombinationMode": laps.attrs.get("traffic_combination_mode", ""),
            "RepositoryHead": _repository_head(repository_root),
        },
        {
            "SourceKind": "derived_per_sample_telemetry",
            "PathBase": "data_root",
            "RelativePath": str(telemetry_path.relative_to(data_root)),
            "SizeBytes": telemetry_path.stat().st_size,
            "SHA256": file_sha256(telemetry_path),
            "Rows": len(telemetry),
            "Columns": len(telemetry.columns),
            "TrafficEvaluatorVersion": telemetry.attrs.get("traffic_evaluator_version", ""),
            "TrafficCombinationMode": telemetry.attrs.get("traffic_combination_mode", ""),
            "RepositoryHead": _repository_head(repository_root),
        },
    ]
    for path in code_paths:
        rows.append(
            {
                "SourceKind": "analysis_code",
                "PathBase": "repository_root",
                "RelativePath": str(path.relative_to(repository_root)),
                "SizeBytes": path.stat().st_size,
                "SHA256": file_sha256(path),
                "Rows": pd.NA,
                "Columns": pd.NA,
                "TrafficEvaluatorVersion": pd.NA,
                "TrafficCombinationMode": pd.NA,
                "RepositoryHead": _repository_head(repository_root),
            }
        )
    return pd.DataFrame(rows)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.12g")


def generate_evidence(data_root: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    laps, telemetry, laps_path, telemetry_path = _load_sources(data_root)
    flagged = _flag_laps(laps)
    # pandas does not preserve attrs through every transformation; keep the
    # historical cache semantics explicit on the flagged frame.
    flagged.attrs.update(laps.attrs)
    lap_table = build_lap_table(flagged, telemetry)
    run_table = build_run_table(lap_table)
    interval_table = build_interval_table(lap_table)
    exclusion_table = build_exclusion_table(flagged)
    identifiability_table = build_identifiability_table(interval_table)
    validate_evidence_tables(run_table, lap_table, interval_table, exclusion_table)

    repository_root = Path(__file__).resolve().parents[3]
    manifest = build_source_manifest(
        laps,
        telemetry,
        laps_path,
        telemetry_path,
        data_root,
        repository_root,
    )
    outputs = {
        "sequence_runs.csv": run_table,
        "sequence_laps.csv": lap_table,
        "push_to_push_intervals.csv": interval_table,
        "excluded_candidates.csv": exclusion_table,
        "identifiability_sensitivity.csv": identifiability_table,
        "source_manifest.csv": manifest,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        _write_csv(frame, output_dir / filename)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="woData root containing the existing FastF1 caches (default: wodata resolution)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("doc/push_cool_sequence_evidence"),
        help="directory for the small derived CSV evidence set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = get_data_root(args.data_root).resolve()
    outputs = generate_evidence(data_root, args.output_dir.resolve())
    print(
        "Saved push-cool evidence: "
        f"{len(outputs['sequence_runs.csv'])} runs, "
        f"{len(outputs['sequence_laps.csv'])} physical laps, "
        f"{len(outputs['push_to_push_intervals.csv'])} intervals"
    )


if __name__ == "__main__":
    main()
