"""Stable access to the current replaceable pre-race tyre provider."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

from wodata import get_data_root

from wostrategy.model.tyre_prediction import PreRaceTyrePrediction, TyreCompoundPrediction


class MissingPreRaceTyrePrediction(FileNotFoundError):
    pass


class StalePreRaceTyrePrediction(RuntimeError):
    pass


def get_pre_race_tyre_prediction(
    *,
    season: int,
    round_number: int,
    data_root: str | Path | None = None,
    fresh_after: str | datetime | None = None,
    refresh: Callable[[], object] | None = None,
) -> PreRaceTyrePrediction:
    """Load and normalize the current cache without exposing provider details.

    If `fresh_after` proves the cache stale, the owning producer callback is used
    when supplied. No orchestration-layer timestamp interpretation is required.
    """
    path = _prediction_path(season, round_number, data_root)
    payload = _read(path) if path.exists() else None
    if payload is None or not _is_fresh(payload, fresh_after):
        if refresh is not None:
            refresh()
            payload = _read(path) if path.exists() else None
        if payload is None:
            raise MissingPreRaceTyrePrediction(
                f"No pre-race tyre prediction for {season} round {round_number}; "
                "run the owning woStrategy tyre prediction producer first."
            )
        if not _is_fresh(payload, fresh_after):
            raise StalePreRaceTyrePrediction(
                f"Pre-race tyre prediction {path} predates required source evidence."
            )
    return _normalize(payload, path)


def _prediction_path(season: int, round_number: int, data_root) -> Path:
    return (
        get_data_root(data_root) / "wostrategy" / "tyre_prediction" / "schema_v1"
        / f"year={int(season)}" / "predictions" / f"round={int(round_number)}"
        / "prediction.json"
    )


def _read(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _as_time(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


def _is_fresh(payload: dict[str, object], fresh_after: str | datetime | None) -> bool:
    if fresh_after is None:
        return True
    timestamp = payload.get("prediction_timestamp")
    return timestamp is not None and _as_time(str(timestamp)) >= _as_time(fresh_after)


def _normalize(payload: dict[str, object], path: Path) -> PreRaceTyrePrediction:
    allocation = {str(k).upper(): str(v) for k, v in dict(payload["compound_allocation"]).items()}
    raw = dict(payload["compounds"])
    by_role = {str(row["compound_role"]).upper(): dict(row) for row in raw.values()}
    if "MEDIUM" not in by_role:
        raise ValueError(f"{path} has no MEDIUM role")
    medium_performance = by_role["MEDIUM"].get("performance_mean")
    if medium_performance is None:
        raise ValueError(f"{path} has no MEDIUM performance estimate")
    compounds = {}
    for role in ("SOFT", "MEDIUM", "HARD"):
        row = by_role.get(role)
        if row is None or row.get("degradation_mean") is None or row.get("performance_mean") is None:
            continue
        compounds[role] = TyreCompoundPrediction(
            performance_delta_to_medium=float(row["performance_mean"]) - float(medium_performance),
            degradation_seconds_per_lap=float(row["degradation_mean"]),
            performance_uncertainty=_optional_float(row.get("performance_std")),
            degradation_uncertainty=_optional_float(row.get("degradation_std")),
            # Legacy field retained for compatibility.  It describes descriptor
            # domain coverage, not statistical identifiability from FP.
            identifiable=not bool(payload.get("descriptor_boundary_warning", False)),
            diagnostics={"absolute_compound": allocation.get(role), "selected_default": row.get("selected_default")},
        )
    artifact_id = str(payload.get("prediction_fingerprint") or path.resolve())
    return PreRaceTyrePrediction(
        event=str(payload.get("event") or f"{payload['season']}-{payload['round']}"),
        season=int(payload["season"]), round_number=int(payload["round"]),
        reference_compound="MEDIUM", compounds=compounds,
        generated_at=str(payload["prediction_timestamp"]), provider="current_pre_race_cache",
        artifact_id=artifact_id, model_version=str(payload.get("model") or "") or None,
        provenance={"path": str(path), "training_data_hash": payload.get("training_data_hash"),
                    "target_preview_hash": payload.get("target_preview_hash"),
                    "trained_through_round": payload.get("trained_through_round"),
                    "training_rounds": payload.get("training_rounds"),
                    "default_prediction_family": payload.get("default_prediction_family"),
                    "production_policy": payload.get("production_policy"),
                    "pirelli_preview_effect": (
                        "compound_allocation_only_for_selected_historical_baseline"
                        if payload.get("default_prediction_family") == "historical_baseline"
                        else "event_descriptors_and_compound_allocation"
                    )},
        freshness={"producer_metadata_available": True},
    )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


__all__ = [
    "MissingPreRaceTyrePrediction", "StalePreRaceTyrePrediction",
    "get_pre_race_tyre_prediction",
]
