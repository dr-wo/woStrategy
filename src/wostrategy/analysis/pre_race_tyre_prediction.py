"""Stable access to the current replaceable pre-race tyre provider."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Callable

from wodata import get_data_root

from wostrategy.model.tyre_prediction import (
    LEGACY_PERFORMANCE_FRAME,
    RELATIVE_PERFORMANCE_FRAME,
    PreRaceTyrePrediction,
    TyreCompoundPrediction,
    canonicalize_operational_tyre_prediction,
    select_source_performance_reference,
)


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
    """Load the current cache as a validated canonical operational prediction.

    If `fresh_after` proves the cache stale, the owning producer callback is used
    when supplied. No orchestration-layer timestamp interpretation is required.
    """
    source = load_pre_race_tyre_prediction_source(
        season=season,
        round_number=round_number,
        data_root=data_root,
        fresh_after=fresh_after,
        refresh=refresh,
    )
    return canonicalize_operational_tyre_prediction(
        source,
        expected_season=season,
        expected_round_number=round_number,
        expected_event_id=f"{int(season):04d}-{int(round_number):02d}",
    )


def load_pre_race_tyre_prediction_source(
    *,
    season: int,
    round_number: int,
    data_root: str | Path | None = None,
    fresh_after: str | datetime | None = None,
    refresh: Callable[[], object] | None = None,
) -> PreRaceTyrePrediction:
    """Load source values and provenance without changing their additive gauge."""
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
    return _source_prediction(payload, path)


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


def _source_prediction(payload: dict[str, object], path: Path) -> PreRaceTyrePrediction:
    allocation = {str(k).upper(): str(v) for k, v in dict(payload["compound_allocation"]).items()}
    raw = dict(payload["compounds"])
    by_role = {str(row["compound_role"]).upper(): dict(row) for row in raw.values()}
    compounds = {}
    for role in ("SOFT", "MEDIUM", "HARD"):
        row = by_role.get(role)
        if row is None:
            continue
        compounds[role] = TyreCompoundPrediction(
            performance_delta_to_medium=row.get("performance_mean"),
            degradation_seconds_per_lap=row.get("degradation_mean"),
            performance_uncertainty=_optional_float(row.get("performance_std")),
            degradation_uncertainty=_optional_float(row.get("degradation_std")),
            # Legacy field retained for compatibility.  It describes descriptor
            # domain coverage, not statistical identifiability from FP.
            identifiable=not bool(payload.get("descriptor_boundary_warning", False)),
            diagnostics={"absolute_compound": allocation.get(role), "selected_default": row.get("selected_default")},
        )
    artifact_id = str(payload.get("prediction_fingerprint") or path.resolve())
    declared_reference = _declared_performance_reference(payload)
    selected_reference, selection_rule, zero_candidates = (
        select_source_performance_reference(
            compounds, declared_reference=declared_reference
        )
    )
    prediction_provenance = dict(payload.get("provenance") or {})
    return PreRaceTyrePrediction(
        event=str(payload.get("event") or f"{payload['season']}-{payload['round']}"),
        season=int(payload["season"]), round_number=int(payload["round"]),
        reference_compound=selected_reference, compounds=compounds,
        generated_at=str(payload["prediction_timestamp"]), provider="current_pre_race_cache",
        artifact_id=artifact_id, model_version=str(payload.get("model") or "") or None,
        provenance={"path": str(path), "training_data_hash": payload.get("training_data_hash"),
                    "target_preview_hash": payload.get("target_preview_hash"),
                    "trained_through_round": payload.get("trained_through_round"),
                    "training_rounds": payload.get("training_rounds"),
                    "default_prediction_family": payload.get("default_prediction_family"),
                    "prediction_status": payload.get("prediction_status"),
                    "classification": prediction_provenance.get("classification"),
                    "independent_of_current_fp": prediction_provenance.get(
                        "independent_of_current_fp"
                    ),
                    "fp_evidence_automatic_update": prediction_provenance.get(
                        "fp_evidence_automatic_update"
                    ),
                    "source_round": prediction_provenance.get("source_round"),
                    "allocation_source_url": prediction_provenance.get(
                        "allocation_source_url"
                    ),
                    "prediction_provenance": {
                        key: prediction_provenance.get(key)
                        for key in (
                            "classification", "independent_of_current_fp",
                            "fp_evidence_automatic_update", "source_round",
                            "source_training_cutoff_round", "source_training_rounds",
                            "prediction_model_sha256", "allocation_source_url",
                            "descriptor_models_available", "descriptor_values_used",
                            "target_preview_available", "reason",
                        )
                    },
                    "production_policy": payload.get("production_policy"),
                    "pirelli_preview_effect": (
                        "compound_allocation_only_for_selected_historical_baseline"
                        if payload.get("default_prediction_family") == "historical_baseline"
                        else "event_descriptors_and_compound_allocation"
                    ),
                    "source_reference_selection": {
                        "rule": selection_rule,
                        "zero_candidates": list(zero_candidates),
                    }},
        freshness={"producer_metadata_available": True},
        performance_frame=(
            RELATIVE_PERFORMANCE_FRAME
            if declared_reference is not None
            else LEGACY_PERFORMANCE_FRAME
        ),
        reference_declared=declared_reference is not None,
        formal_relative_output=declared_reference is not None,
    )


def _declared_performance_reference(payload: dict[str, object]) -> str | None:
    semantics = payload.get("performance_semantics")
    if isinstance(semantics, dict) and semantics.get("reference_compound"):
        return str(semantics["reference_compound"]).upper()
    coordinate = str(payload.get("performance_coordinate") or "").upper()
    match = re.search(r"RELATIVE TO (?:EVENT )?(SOFT|MEDIUM|HARD) COMPOUND", coordinate)
    return match.group(1) if match else None


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


__all__ = [
    "MissingPreRaceTyrePrediction", "StalePreRaceTyrePrediction",
    "get_pre_race_tyre_prediction", "load_pre_race_tyre_prediction_source",
]
