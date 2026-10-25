"""Write a transparent allocation-only fallback from an existing P0/D0 fit."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping

from wodata import get_data_root
from wodata.contracts import canonical_hash

from wostrategy.analysis.retro_tyre_results import tyre_prediction_year_root
from wostrategy.model.cross_event_tyre_prediction import (
    AdjacentGapPerformanceModel,
    LinearDegradationModel,
    MODEL_VERSION,
)


_ROLES = ("HARD", "MEDIUM", "SOFT")
PRIOR_PRODUCER_VERSION = "allocation-only-historical-p0d0-v1"


def write_historical_baseline_prior(
    *,
    season: int,
    round_number: int,
    event: str,
    circuit: str,
    compound_allocation: Mapping[str, str],
    allocation_source_url: str,
    data_root: str | Path | None = None,
    source_round_number: int | None = None,
) -> Path:
    """Create a prior-only operational prediction from a frozen historical P0/D0.

    This path deliberately uses the saved descriptor-free P0/D0 model objects
    without fitting or updating them. It requires an externally verified dry
    compound allocation and leaves descriptor-informed families unavailable.
    Existing event artifacts are immutable: an identical fingerprint is reused;
    a different artifact at the canonical path is never overwritten.
    """
    season = int(season)
    round_number = int(round_number)
    event = str(event).strip()
    circuit = str(circuit).strip()
    if season < 1900 or round_number < 1 or not event or not circuit:
        raise ValueError("A valid season, round, event and circuit are required.")
    if not allocation_source_url.startswith("https://"):
        raise ValueError("allocation_source_url must be an HTTPS source URL.")
    allocation = {
        str(key).upper(): str(value).upper()
        for key, value in compound_allocation.items()
    }
    if set(allocation) != set(_ROLES):
        raise ValueError("compound_allocation must contain HARD, MEDIUM and SOFT roles.")
    if any(not re.fullmatch(r"C[1-5]", value) for value in allocation.values()):
        raise ValueError("Dry compound allocation values must be Pirelli compounds C1–C5.")
    if len(set(allocation.values())) != 3:
        raise ValueError("Each dry compound role must have a distinct allocation.")
    compound_ranks = {role: int(value[1:]) for role, value in allocation.items()}
    if not compound_ranks["HARD"] < compound_ranks["MEDIUM"] < compound_ranks["SOFT"]:
        raise ValueError("HARD, MEDIUM and SOFT allocations must follow the ordered Pirelli range.")

    root = get_data_root(data_root)
    source_round = int(source_round_number) if source_round_number is not None else round_number - 1
    if source_round >= round_number or source_round < 1:
        raise ValueError("The source model round must precede the target event.")
    source_dir = tyre_prediction_year_root(season, root) / "predictions" / f"round={source_round}"
    source_model_path = source_dir / "model.json"
    source_prediction_path = source_dir / "prediction.json"
    if not source_model_path.is_file() or not source_prediction_path.is_file():
        raise FileNotFoundError(f"Missing frozen source prediction/model for round {source_round}.")
    model_bytes = source_model_path.read_bytes()
    source_model = json.loads(model_bytes)
    source_prediction = json.loads(source_prediction_path.read_text(encoding="utf-8"))

    if (
        int(source_model.get("season") or -1) != season
        or int(source_model.get("target_round") or -1) != source_round
        or int(source_prediction.get("season") or -1) != season
        or int(source_prediction.get("round") or -1) != source_round
    ):
        raise ValueError(
            "Source model/prediction event identity does not match the declared source round."
        )
    if (
        source_model.get("model_version") != MODEL_VERSION
        or source_prediction.get("model") != MODEL_VERSION
        or not source_prediction.get("prediction_fingerprint")
    ):
        raise ValueError("Source model/prediction fingerprint metadata is incompatible.")

    training_cutoff = int(source_model.get("trained_through_round") or -1)
    if training_cutoff < 1 or training_cutoff >= round_number:
        raise ValueError("The source model training cutoff must precede the target event.")
    if int(source_prediction.get("trained_through_round") or -1) != training_cutoff:
        raise ValueError("Source prediction and frozen model training cutoffs disagree.")
    training_rounds = source_prediction.get("training_rounds")
    if not isinstance(training_rounds, list) or not training_rounds:
        raise ValueError("The frozen source prediction has no exact training-round provenance.")
    rounds = [int(value) for value in training_rounds]
    if (
        rounds != sorted(set(rounds))
        or min(rounds) < 1
        or max(rounds) != training_cutoff
        or any(value >= round_number for value in rounds)
    ):
        raise ValueError("Source training rounds are inconsistent with the target cutoff.")
    training_data_hash = str(source_model.get("training_data_hash") or "")
    if not training_data_hash:
        raise ValueError("Source model has no training-data hash.")
    if training_data_hash != str(source_prediction.get("training_data_hash") or ""):
        raise ValueError("Source model and prediction training-data hashes disagree.")

    regressions = dict(source_model.get("regressions") or {})
    try:
        performance_model = AdjacentGapPerformanceModel.from_dict(regressions["P0"])
        degradation_model = LinearDegradationModel.from_dict(regressions["D0"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The frozen source model lacks supported P0/D0 regressions.") from exc
    if performance_model.include_environment or degradation_model.include_environment:
        raise ValueError("Allocation-only priors require descriptor-free P0/D0 model objects.")
    if performance_model.code != "P0" or degradation_model.code != "D0":
        raise ValueError("The source regressions are not the validated P0/D0 baseline.")

    values: dict[str, dict[str, object]] = {}
    hard = allocation["HARD"]
    for role in _ROLES:
        absolute = allocation[role]
        performance = performance_model.predict(
            absolute_compound=absolute,
            hard_compound=hard,
            tyre_stress=0.0,
            asphalt_grip=0.0,
        ).mean
        degradation = degradation_model.predict(
            absolute_compound=absolute,
            tyre_stress=0.0,
            asphalt_abrasion=0.0,
            asphalt_grip=0.0,
        ).mean
        if not math.isfinite(performance) or not math.isfinite(degradation) or degradation < 0:
            raise ValueError(f"Source model produced invalid values for {role}.")
        values[absolute] = {
            "absolute_compound": absolute,
            "compound_role": role,
            "performance_mean": float(performance),
            "performance_std": None,
            "degradation_mean": float(degradation),
            "degradation_std": None,
            "selected_default": "historical_baseline",
            "historical_baseline": {
                "performance": float(performance),
                "degradation": float(degradation),
            },
        }

    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    fingerprint = canonical_hash({
        "prior_producer_version": PRIOR_PRODUCER_VERSION,
        "season": season,
        "round": round_number,
        "event": event,
        "circuit": circuit,
        "allocation": allocation,
        "allocation_source_url": allocation_source_url,
        "source_round": source_round,
        "training_cutoff": training_cutoff,
        "training_rounds": rounds,
        "training_data_hash": training_data_hash,
        "source_model_sha256": model_sha256,
        "model": {
            "performance": performance_model.to_dict(),
            "degradation": degradation_model.to_dict(),
        },
    })
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": "schema_v1",
        "season": season,
        "round": round_number,
        "event": f"{season}-{round_number:02d}-{event}",
        "circuit": circuit,
        "prediction_timestamp": timestamp,
        "trained_through_round": training_cutoff,
        "training_rounds": rounds,
        "training_data_hash": training_data_hash,
        "target_preview_hash": None,
        "model": str(source_model.get("model_version") or "linear_v1.1"),
        "prior_producer_version": PRIOR_PRODUCER_VERSION,
        "default_prediction_family": "historical_baseline",
        "prediction_status": "prior_only",
        "compound_allocation": allocation,
        "compounds": values,
        "prediction_fingerprint": fingerprint,
        "performance_coordinate": f"seconds relative to event HARD compound ({hard})",
        "performance_semantics": {
            "contract": "canonical_tyre_source_semantics_v1",
            "quantity": "compound_performance_delta",
            "unit": "s",
            "independent_variable": "dry_compound_choice",
            "positive_sign": "compound is slower than HARD",
            "frame": "relative_to_compound",
            "reference_compound": "HARD",
            "reference_value": 0.0,
            "uncertainty": "unavailable",
        },
        "degradation_semantics": {
            "contract": "canonical_tyre_source_semantics_v1",
            "quantity": "degradation_rate",
            "unit": "s/tyre-age-lap",
            "independent_variable": "tyre_age_laps",
            "positive_sign": "greater tyre age increases modelled lap time",
            "frame": "absolute_rate",
            "reference_compound": None,
            "uncertainty": "unavailable",
        },
        "provenance": {
            "classification": "prior_only",
            "independent_of_current_fp": True,
            "fp_evidence_automatic_update": False,
            "source_round": source_round,
            "source_training_cutoff_round": training_cutoff,
            "source_training_rounds": rounds,
            "training_data_hash": training_data_hash,
            "prediction_model_path": str(source_model_path),
            "prediction_model_sha256": model_sha256,
            "source_prediction_fingerprint": source_prediction.get("prediction_fingerprint"),
            "source_prediction_timestamp": source_prediction.get("prediction_timestamp"),
            "production_policy": source_model.get("production_policy"),
            "model_formulations": {"performance": "P0", "degradation": "D0"},
            "descriptor_models_available": False,
            "descriptor_values_used": None,
            "target_preview_available": False,
            "target_preview_hash": None,
            "allocation_source_url": allocation_source_url,
            "reason": (
                "No accepted target preview was supplied; this uses only the frozen "
                "validated descriptor-free historical baseline."
            ),
        },
    }
    event_root = tyre_prediction_year_root(season, root) / "predictions" / f"round={round_number}"
    versions = event_root / "versions"
    canonical_path = event_root / "prediction.json"
    version_path = versions / f"{fingerprint}.json"
    if canonical_path.is_file():
        existing = json.loads(canonical_path.read_text(encoding="utf-8"))
        if existing.get("prediction_fingerprint") != fingerprint:
            raise FileExistsError(
                "Refusing to overwrite a different tyre prediction artifact: "
                f"{canonical_path}"
            )
    _write_once(version_path, payload, fingerprint=fingerprint)
    _write_once(canonical_path, payload, fingerprint=fingerprint)
    return canonical_path


def _write_once(path: Path, payload: Mapping[str, object], *, fingerprint: str) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("prediction_fingerprint") != fingerprint:
            raise FileExistsError(
                f"Refusing to overwrite a different tyre prediction artifact: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{fingerprint[:12]}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("prediction_fingerprint") != fingerprint:
                raise FileExistsError(
                    f"Refusing to overwrite a different tyre prediction artifact: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["write_historical_baseline_prior"]
