from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from wodata import get_data_root, get_fastf1_session_laps_path
from wodata.artifacts import weekend_model_root
from wostrategy.analysis.traffic import (
    COMBINATION_MODE_COMPATIBILITY_MIN,
    TRAFFIC_EVALUATOR_VERSION,
)


SESSIONS = ("FP1", "FP2", "FP3")
SUMMARY_FILENAME = "historical_fp_backfill_summary.csv"
ESTIMATOR_VERSION = "joint-weekend-model"


def _analysis_code_hash() -> str:
    source_root = Path(__file__).resolve().parents[1]
    paths = (
        source_root / "script/pre_race_analysis.py",
        source_root / "core/pre_race_session_data.py",
        source_root / "model/pre_race_performance.py",
    )
    digest = sha256()
    for path in paths:
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def inspect_fp_session_artifact(
    *,
    season: int,
    round_number: int,
    session: str,
    data_root: str | Path | None = None,
) -> dict[str, object]:
    """Inspect canonical raw and model artifacts without downloading anything."""
    root = get_data_root(data_root)
    session = str(session).upper()
    raw_path = get_fastf1_session_laps_path(
        year=season,
        round_number=round_number,
        session=session,
        data_root=root,
    )
    artifact_dir = weekend_model_root(season, round_number, root) / "sessions" / session
    parameters_path = artifact_dir / "latest_parameters.csv"
    support_path = artifact_dir / "latest_support.csv"
    manifest_path = artifact_dir / "manifest.json"

    event = ""
    event_format = ""
    raw_status = "missing"
    raw_reason = ""
    raw_hash = ""
    if raw_path.is_file():
        try:
            laps = pd.read_pickle(raw_path)
            event_values = laps.get("EventName", pd.Series(dtype=object)).dropna()
            event = str(event_values.iloc[0]) if not event_values.empty else ""
            format_values = laps.get("EventFormat", pd.Series(dtype=object)).dropna()
            event_format = str(format_values.iloc[0]) if not format_values.empty else ""
            if laps.empty:
                raw_status = "invalid/stale"
                raw_reason = "canonical enriched lap cache is empty"
            elif (
                laps.attrs.get("traffic_evaluator_version")
                != TRAFFIC_EVALUATOR_VERSION
                or laps.attrs.get("traffic_combination_mode")
                != COMBINATION_MODE_COMPATIBILITY_MIN
            ):
                raw_status = "invalid/stale"
                raw_reason = "traffic cache metadata does not match current evaluator"
            else:
                raw_status = "cached"
                raw_hash = sha256(raw_path.read_bytes()).hexdigest()
        except (OSError, ValueError, TypeError) as exc:
            raw_status = "invalid/stale"
            raw_reason = f"cache read failed: {exc}"

    analysis_status = "missing"
    analysis_reason = ""
    manifest: dict[str, object] = {}
    parameters = pd.DataFrame()
    if any(path.exists() for path in (parameters_path, support_path, manifest_path)):
        if not all(path.is_file() for path in (parameters_path, support_path, manifest_path)):
            analysis_status = "partial"
            analysis_reason = "parameters, support, and manifest are not all present"
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                parameters = pd.read_csv(parameters_path)
                support = pd.read_csv(support_path)
                required_manifest = {
                    "schema_version", "analysis_id", "source_scope",
                    "input_fingerprint", "config_hash", "created_at", "status",
                }
                missing_manifest = required_manifest.difference(manifest)
                if missing_manifest:
                    raise ValueError(
                        f"manifest missing fields {sorted(missing_manifest)}"
                    )
                if str(manifest["source_scope"]).upper() != session:
                    raise ValueError("manifest source scope does not match session")
                contributors = {
                    str(value).upper()
                    for value in manifest.get("contributing_sessions", ())
                }
                if contributors != {session}:
                    raise ValueError(
                        "session snapshot is not leakage-safe: contributing sessions "
                        f"are {sorted(contributors)}"
                    )
                if parameters.empty or support.empty:
                    raise ValueError("parameters or support table is empty")
                for column in ("analysis_id", "config_hash"):
                    expected = str(manifest[column])
                    actual = set(parameters[column].dropna().astype(str))
                    if actual != {expected}:
                        raise ValueError(f"{column} does not match manifest")
                expected_fingerprint = str(manifest["input_fingerprint"])
                local_rows = parameters.loc[
                    ~parameters["parameter"].eq("fuel_rate"), "input_fingerprint"
                ].dropna().astype(str)
                if set(local_rows) != {expected_fingerprint}:
                    raise ValueError("input_fingerprint does not match manifest")
                if str(manifest["status"]).upper() not in {"UPDATED", "FROZEN"}:
                    raise ValueError(f"manifest status is {manifest['status']!r}")
                analysis_status = "complete"
            except (OSError, json.JSONDecodeError, ValueError, KeyError, pd.errors.ParserError) as exc:
                analysis_status = "invalid/stale"
                analysis_reason = f"artifact validation failed: {exc}"

    degradation = (
        parameters.loc[parameters["parameter"].eq("degradation")]
        if analysis_status == "complete"
        else pd.DataFrame()
    )
    supported_compounds = sorted(
        degradation.get("compound", pd.Series(dtype=object)).dropna().astype(str).unique()
    )
    supported_teams = sorted(
        parameters.get("team", pd.Series(dtype=object)).dropna().astype(str).unique()
    ) if analysis_status == "complete" else []
    return {
        "season": int(season),
        "round": int(round_number),
        "event": event,
        "event_format": event_format,
        "session": session,
        "raw_data_status": raw_status,
        "raw_data_cache_path": str(raw_path),
        "analysis_status": analysis_status,
        "analysis_artifact_path": str(artifact_dir),
        "long_run_supported": bool(analysis_status == "complete" and not degradation.empty),
        "supported_compounds": ";".join(supported_compounds),
        "supported_teams": ";".join(supported_teams),
        "source_manifest_hash": (
            sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else ""
        ),
        "source_data_hash": raw_hash,
        "analysis_id": str(manifest.get("analysis_id", "")),
        "analysis_config_hash": str(manifest.get("config_hash", "")),
        "estimator_version": ESTIMATOR_VERSION,
        "analysis_code_hash": _analysis_code_hash(),
        "replay_created_at": str(manifest.get("created_at", "")),
        "failure_reason": analysis_reason or raw_reason,
    }


def discover_historical_fp_artifacts(
    *,
    season: int,
    rounds: Iterable[int],
    sessions: Sequence[str] = SESSIONS,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    rows = [
        inspect_fp_session_artifact(
            season=season,
            round_number=round_number,
            session=session,
            data_root=data_root,
        )
        for round_number in sorted({int(value) for value in rounds})
        for session in sessions
    ]
    summary = pd.DataFrame(rows)
    if not summary.empty:
        for column in ("event", "event_format"):
            summary[column] = summary.groupby("round")[column].transform(
                lambda values: values.replace("", pd.NA).ffill().bfill().fillna("")
            )
        absent = (
            summary["event_format"].str.lower().str.contains("sprint")
            & summary["session"].isin(("FP2", "FP3"))
            & summary["raw_data_status"].eq("missing")
        )
        summary.loc[absent, "raw_data_status"] = "session_absent"
        summary.loc[absent, "failure_reason"] = (
            "source data unavailable: session absent for sprint event"
        )
    return summary


def classify_rounds(summary: pd.DataFrame) -> dict[int, str]:
    result: dict[int, str] = {}
    for round_number, group in summary.groupby("round", sort=True):
        statuses = set(group["analysis_status"].astype(str))
        if statuses == {"complete"}:
            status = "complete"
        elif "invalid/stale" in statuses:
            status = "invalid/stale"
        elif "complete" in statuses or "partial" in statuses:
            status = "partial"
        else:
            status = "missing"
        result[int(round_number)] = status
    return result


def write_backfill_summary(
    summary: pd.DataFrame,
    *,
    season: int,
    data_root: str | Path | None = None,
    failures: Mapping[tuple[int, str], str] | None = None,
) -> Path:
    output = summary.copy()
    if failures:
        for (round_number, session), reason in failures.items():
            mask = output["round"].eq(int(round_number)) & output["session"].eq(str(session).upper())
            output.loc[mask, "failure_reason"] = str(reason)
    path = (
        get_data_root(data_root)
        / "wostrategy"
        / "weekend_model"
        / "schema_v1"
        / f"year={int(season)}"
        / SUMMARY_FILENAME
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.sort_values(["round", "session"]).to_csv(path, index=False)
    return path
