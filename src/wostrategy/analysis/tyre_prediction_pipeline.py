from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

import pandas as pd
from wodata.contracts import canonical_hash
from wodata.pirelli_preview import (
    PreviewIngestionReport,
    PreviewVisionExtractor,
    ingest_previews,
    preview_csv_path,
)

from wostrategy.analysis.retro_tyre_results import (
    RetroExtractionReport,
    RetroTyreResultExtractor,
    observations_csv_path,
    tyre_prediction_year_root,
)
from wostrategy.analysis.tyre_descriptor_domain import descriptor_domain_diagnostics
from wostrategy.analysis.tyre_prediction_validation import (
    PRODUCTION_POLICY,
    PIRELLI_INFORMED_POLICY,
    ValidationConfig,
    ValidationResult,
    build_diagnostic_summary,
    readiness_at_cutoff,
    rolling_validate,
    write_validation_artifacts,
)
from wostrategy.analysis.tyre_prediction_prospective import (
    score_available_prospective_predictions,
)
from wostrategy.model.cross_event_tyre_prediction import (
    FEATURES,
    MODEL_VERSION,
    LinearDegradationModel,
    ReadinessConfig,
    RelativeLinearPerformanceModel,
)
from wostrategy.model.observation_weighting import WEIGHT_POLICIES, UniformWeightPolicy


PREDICTION_HISTORY_COLUMNS = (
    "season",
    "round",
    "event",
    "absolute_compound",
    "compound_role",
    "prediction_timestamp",
    "trained_through_round",
    "training_data_hash",
    "target_preview_hash",
    "model_version",
    "performance_model_name",
    "performance_weighting_policy",
    "degradation_model_name",
    "degradation_weighting_policy",
    "performance_prediction",
    "degradation_prediction",
    "p0_performance_prediction",
    "p1_performance_prediction",
    "d0_degradation_prediction",
    "d1_degradation_prediction",
    "prediction_fingerprint",
)


@dataclass(frozen=True)
class PipelineConfig:
    performance_ridge_alpha: float = 0.0
    degradation_ridge_alpha: float = 0.0
    readiness: ReadinessConfig = ReadinessConfig()
    validation_weight_policies: tuple[str, ...] = WEIGHT_POLICIES
    evidence_field: str = "support_clean_lap_count"
    evidence_transformation: str = "sqrt"


@dataclass(frozen=True)
class PredictionArtifact:
    round_number: int
    path: Path
    history_rows_added: int


@dataclass(frozen=True)
class TyrePredictionPipelineReport:
    preview: PreviewIngestionReport
    retro: RetroExtractionReport
    prediction_candidates: tuple[int, ...]
    predictions: tuple[PredictionArtifact, ...]
    readiness_paths: tuple[Path, ...]
    training_data_path: Path
    validation_path: Path
    validation_predictions_path: Path
    coefficient_history_path: Path
    diagnostic_summary_path: Path
    prospective_validation_path: Path


def run_tyre_prediction_pipeline(
    *,
    season: int,
    vision_extractor: PreviewVisionExtractor | None = None,
    data_root: str | Path | None = None,
    legacy_retro_root: str | Path | None = None,
    config: PipelineConfig | None = None,
) -> TyrePredictionPipelineReport:
    config = config or PipelineConfig()
    preview_report = ingest_previews(
        season=season, extractor=vision_extractor, data_root=data_root
    )
    previews = pd.read_csv(preview_csv_path(season, data_root))
    accepted = previews.loc[previews["review_status"].eq("accepted")].copy()
    retro_extractor = RetroTyreResultExtractor(
        data_root=data_root, legacy_root=legacy_retro_root
    )
    retro_report = retro_extractor.update(
        season=season, previews=accepted.to_dict("records")
    )
    observations = pd.read_csv(observations_csv_path(season, data_root))
    training = build_training_data(previews=accepted, observations=observations)
    training_path = tyre_prediction_year_root(season, data_root) / "training_data.csv"
    _write_frame(training, training_path)
    validation = rolling_validate(
        training,
        ValidationConfig(
            readiness=config.readiness,
            weight_policies=config.validation_weight_policies,
            evidence_field=config.evidence_field,
            evidence_transformation=config.evidence_transformation,
            performance_ridge_alpha=config.performance_ridge_alpha,
            degradation_ridge_alpha=config.degradation_ridge_alpha,
        ),
    )
    year_root = tyre_prediction_year_root(season, data_root)
    validation_path = year_root / "validation.json"
    validation_predictions_path = year_root / "validation_predictions.csv"
    coefficient_history_path = year_root / "coefficient_history.csv"
    diagnostic_summary_path = year_root / "diagnostic_summary.json"
    prospective_validation_path = year_root / "prospective_validation.csv"
    prospective = score_available_prospective_predictions(
        year_root=year_root, observations=training, output_path=prospective_validation_path
    )
    validation = ValidationResult(
        payload=validation.payload,
        predictions=validation.predictions,
        coefficient_history=validation.coefficient_history,
        diagnostic_summary=build_diagnostic_summary(validation, prospective),
    )
    write_validation_artifacts(
        validation, json_path=validation_path, csv_path=validation_predictions_path,
        coefficient_history_path=coefficient_history_path,
        diagnostic_summary_path=diagnostic_summary_path,
    )

    observed_rounds = set(int(value) for value in observations["round"].unique())
    candidates = tuple(
        int(value) for value in sorted(accepted["round"].unique())
        if int(value) not in observed_rounds
    )
    predictions: list[PredictionArtifact] = []
    readiness_paths: list[Path] = []
    validation_context = _prediction_validation_context(validation.payload)
    for target_round in candidates:
        target = accepted.loc[accepted["round"].eq(target_round)].iloc[0]
        historical = training.loc[training["round"] < target_round].copy()
        readiness_by_model = readiness_at_cutoff(historical, config.readiness)
        descriptor_domains = {
            target_name: descriptor_domain_diagnostics(
                target=target_name, historical=historical, target_values=target.to_dict()
            )
            for target_name in ("performance", "degradation")
        }
        target_root = (
            tyre_prediction_year_root(season, data_root)
            / "predictions"
            / f"round={target_round}"
        )
        readiness_path = target_root / "readiness.json"
        readiness = {
            "schema_version": "schema_v1",
            "season": int(season),
            "target_round": target_round,
            "training_rounds": sorted(int(value) for value in historical["round"].unique()),
            "models": {
                name: diagnostics.to_dict()
                for name, diagnostics in readiness_by_model.items()
            },
            "selection": {
                "performance": {**PRODUCTION_POLICY["performance"], "basis": "fixed production policy"},
                "degradation": {**PRODUCTION_POLICY["degradation"], "basis": "fixed production policy"},
            },
            "pirelli_informed": PIRELLI_INFORMED_POLICY,
            "descriptor_domain": descriptor_domains,
        }
        _write_json(readiness_path, readiness)
        readiness_paths.append(readiness_path)
        if not any(diagnostics.ready for diagnostics in readiness_by_model.values()):
            continue
        predictions.append(_fit_and_predict(
            season=season,
            target=target,
            historical=historical,
            readiness_by_model=readiness_by_model,
            validation_context=validation_context,
            config=config,
            target_root=target_root,
            year_root=year_root,
        ))
    return TyrePredictionPipelineReport(
        preview=preview_report,
        retro=retro_report,
        prediction_candidates=candidates,
        predictions=tuple(predictions),
        readiness_paths=tuple(readiness_paths),
        training_data_path=training_path,
        validation_path=validation_path,
        validation_predictions_path=validation_predictions_path,
        coefficient_history_path=coefficient_history_path,
        diagnostic_summary_path=diagnostic_summary_path,
        prospective_validation_path=prospective_validation_path,
    )


def build_training_data(*, previews: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "season", "round", *FEATURES,
        "hard_compound", "medium_compound", "soft_compound", "source_hash",
    ]
    joined = observations.merge(
        previews[feature_columns].rename(columns={"source_hash": "preview_source_hash"}),
        on=["season", "round"], how="inner", validate="many_to_one",
    )
    return joined.sort_values(["season", "round", "compound_role"]).reset_index(drop=True)


def _prediction_validation_context(payload: Mapping[str, object]) -> dict[str, object]:
    by_code = {
        str(row["model_code"]): row
        for row in payload.get("metrics", [])
        if str(row["weighting_policy"]) == "uniform"
    }
    def family(performance_code: str, degradation_code: str) -> dict[str, object]:
        return {
            "performance": {
                key: by_code.get(performance_code, {}).get(key)
                for key in ("prediction_count", "MAE", "RMSE", "bias")
            },
            "degradation": {
                key: by_code.get(degradation_code, {}).get(key)
                for key in ("prediction_count", "MAE", "RMSE", "bias")
            },
        }
    return {
        "validation_type": "historical_rolling",
        "weighting_policy": "uniform",
        "historical_baseline": family("P0", "D0"),
        "pirelli_informed": family("P1", "D1"),
        "current_observation": (
            "historical_baseline currently has lower rolling validation error for "
            "both performance and degradation; both estimates remain exposed"
        ),
    }


def _fit_and_predict(
    *,
    season: int,
    target: pd.Series,
    historical: pd.DataFrame,
    readiness_by_model: Mapping[str, object],
    validation_context: Mapping[str, object],
    config: PipelineConfig,
    target_root: Path,
    year_root: Path,
) -> PredictionArtifact:
    uniform = UniformWeightPolicy()
    candidates = {
        "P0": ("performance_baseline", RelativeLinearPerformanceModel(
            include_environment=False, ridge_alpha=config.performance_ridge_alpha,
            weight_policy=uniform)),
        "P1": ("performance_pirelli_v1", RelativeLinearPerformanceModel(
            include_environment=True, ridge_alpha=config.performance_ridge_alpha,
            weight_policy=uniform)),
        "D0": ("degradation_baseline", LinearDegradationModel(
            include_environment=False, ridge_alpha=config.degradation_ridge_alpha,
            weight_policy=uniform)),
        "D1": ("degradation_pirelli_v1", LinearDegradationModel(
            include_environment=True, ridge_alpha=config.degradation_ridge_alpha,
            weight_policy=uniform)),
    }
    models: dict[str, object] = {}
    for code, (name, model) in candidates.items():
        if readiness_by_model[name].ready:
            model.fit(historical)
            models[code] = model

    training_hash = _training_hash(historical)
    trained_through = int(historical["round"].max()) if not historical.empty else None
    model_payload = {
        "schema_version": "schema_v1",
        "model_version": MODEL_VERSION,
        "season": int(season),
        "target_round": int(target["round"]),
        "trained_through_round": trained_through,
        "training_data_hash": training_hash,
        "regressions": {code: model.to_dict() for code, model in models.items()},
        "production_policy": PRODUCTION_POLICY,
        "pirelli_informed_policy": PIRELLI_INFORMED_POLICY,
    }
    _write_json(target_root / "model.json", model_payload)

    target_features = {feature: float(target[feature]) for feature in FEATURES}
    descriptor_domains = {
        target_name: descriptor_domain_diagnostics(
            target=target_name, historical=historical, target_values=target.to_dict()
        )
        for target_name in ("performance", "degradation")
    }
    hard_compound = str(target["hard_compound"])
    roles = {
        "HARD": hard_compound,
        "MEDIUM": str(target["medium_compound"]),
        "SOFT": str(target["soft_compound"]),
    }
    regression_predictions: dict[str, dict[str, object]] = {}
    for code, model in models.items():
        target_name = "performance" if code.startswith("P") else "degradation"
        per_role: dict[str, dict[str, object]] = {}
        for role, compound in roles.items():
            if target_name == "performance":
                value = model.predict(
                    absolute_compound=compound, hard_compound=hard_compound,
                    **target_features,
                ).mean
            else:
                value = model.predict(
                    absolute_compound=compound, **target_features
                ).mean
            per_role[role] = {
                "absolute_compound": compound,
                "compound_role": role,
                "predicted_value": float(value),
            }
        regression_predictions[code] = {
            "target": target_name,
            "model_name": model.name,
            "weighting_policy": model.weight_policy.name,
            "is_production": code in {"P0", "D0"},
            "prediction_family": (
                "historical_baseline" if code in {"P0", "D0"} else "pirelli_informed"
            ),
            "descriptor_domain": descriptor_domains[target_name],
            "compounds": per_role,
        }

    compounds: dict[str, dict[str, object]] = {}
    for role, compound in roles.items():
        performance_value = _regression_value(regression_predictions, "P0", role)
        degradation_value = _regression_value(regression_predictions, "D0", role)
        compounds[compound] = {
            "absolute_compound": compound,
            "compound_role": role,
            "performance_mean": performance_value,
            "performance_std": None,
            "degradation_mean": degradation_value,
            "degradation_std": None,
            "selected_default": "historical_baseline",
            "historical_baseline": {
                "performance": _regression_value(regression_predictions, "P0", role),
                "degradation": _regression_value(regression_predictions, "D0", role),
            },
            "pirelli_informed": {
                "performance": _regression_value(regression_predictions, "P1", role),
                "degradation": _regression_value(regression_predictions, "D1", role),
            },
        }
    fingerprint_values = {
        "season": int(season),
        "round": int(target["round"]),
        "training_data_hash": training_hash,
        "target_preview_hash": str(target["source_hash"]),
        "model_version": MODEL_VERSION,
        "performance_ridge_alpha": config.performance_ridge_alpha,
        "degradation_ridge_alpha": config.degradation_ridge_alpha,
        "production_policy": PRODUCTION_POLICY,
        "pirelli_informed_policy": PIRELLI_INFORMED_POLICY,
        "validation_context": dict(validation_context),
        "descriptor_domain": descriptor_domains,
        "regression_predictions": regression_predictions,
        "compounds": compounds,
    }
    fingerprint = canonical_hash(fingerprint_values)
    prediction_path = target_root / "prediction.json"
    existing = _read_json(prediction_path) if prediction_path.exists() else None
    prediction_timestamp = (
        str(existing["prediction_timestamp"])
        if existing and existing.get("prediction_fingerprint") == fingerprint
        else _utc_now()
    )
    prediction_payload = {
        "schema_version": "schema_v1",
        "season": int(season),
        "round": int(target["round"]),
        "event": str(target["event"]),
        "circuit": str(target["circuit"]),
        "prediction_timestamp": prediction_timestamp,
        "trained_through_round": trained_through,
        "training_rounds": sorted(int(value) for value in historical["round"].unique()),
        "training_data_hash": training_hash,
        "target_preview_hash": str(target["source_hash"]),
        "model": MODEL_VERSION,
        "performance_model": {**PRODUCTION_POLICY["performance"], "basis": "fixed production policy"},
        "degradation_model": {**PRODUCTION_POLICY["degradation"], "basis": "fixed production policy"},
        "production_policy": PRODUCTION_POLICY,
        "pirelli_informed_policy": PIRELLI_INFORMED_POLICY,
        "default_prediction_family": "historical_baseline",
        "prediction_families": {
            "historical_baseline": {
                "performance_model": "P0",
                "degradation_model": "D0",
                "description": (
                    "Uses transferable historical compound behaviour only. Currently "
                    "better validated on the limited 2026 rolling sample."
                ),
            },
            "pirelli_informed": {
                "performance_model": "P1",
                "degradation_model": "D1",
                "description": (
                    "Adds simplified tyre-stress and surface dependence. It represents "
                    "the intended physical dependence more explicitly but has not yet "
                    "demonstrated superior rolling predictive accuracy."
                ),
            },
        },
        "validation_context": dict(validation_context),
        "features": target_features,
        "compound_allocation": roles,
        "descriptor_boundary_warning": bool(
            descriptor_domains["degradation"]["any_boundary_descriptor"]
        ),
        "boundary_descriptors": descriptor_domains["degradation"]["boundary_descriptors"],
        "descriptor_boundary_note": (
            "Endpoint ratings 1 and 5 are saturated ordinal categories; underlying "
            "physical severity may extend beyond the nominal coding. Predictions are "
            "not modified by this diagnostic warning."
        ),
        "descriptor_domain": descriptor_domains,
        "performance_coordinate": "seconds relative to event HARD compound",
        "compounds": compounds,
        "regression_predictions": regression_predictions,
        "prediction_fingerprint": fingerprint,
    }
    version_path = target_root / "versions" / f"{fingerprint}.json"
    if not version_path.exists():
        _write_json(version_path, prediction_payload)
    _write_json(prediction_path, prediction_payload)
    history_path = year_root / "prediction_history.csv"
    added = _append_prediction_history(history_path, prediction_payload)
    return PredictionArtifact(int(target["round"]), prediction_path, added)


def _regression_value(
    outputs: Mapping[str, object], model_code: str, role: str
) -> float | None:
    if model_code not in outputs:
        return None
    return float(outputs[model_code]["compounds"][role]["predicted_value"])


def _training_hash(frame: pd.DataFrame) -> str:
    columns = [
        "season", "round", "compound_role", "absolute_compound", "event_hard_compound",
        *FEATURES, "performance", "degradation", "source_result_hash", "preview_source_hash",
    ]
    records = frame[columns].sort_values(
        ["season", "round", "compound_role"]
    ).to_dict("records")
    return canonical_hash(records)


def _append_prediction_history(path: Path, prediction: Mapping[str, object]) -> int:
    existing: list[dict[str, str]] = []
    existing_columns: list[str] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing = list(reader)
            existing_columns = list(reader.fieldnames or [])
    if existing_columns and tuple(existing_columns) != PREDICTION_HISTORY_COLUMNS:
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=PREDICTION_HISTORY_COLUMNS, lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(existing)
        temporary.replace(path)
    fingerprints = {
        (row["prediction_fingerprint"], row["absolute_compound"]) for row in existing
    }
    additions: list[dict[str, object]] = []
    for compound, values in dict(prediction["compounds"]).items():
        key = (str(prediction["prediction_fingerprint"]), compound)
        if key in fingerprints:
            continue
        additions.append({
            "season": prediction["season"],
            "round": prediction["round"],
            "event": prediction["event"],
            "absolute_compound": compound,
            "compound_role": values["compound_role"],
            "prediction_timestamp": prediction["prediction_timestamp"],
            "trained_through_round": prediction["trained_through_round"],
            "training_data_hash": prediction["training_data_hash"],
            "target_preview_hash": prediction["target_preview_hash"],
            "model_version": prediction["model"],
            "performance_model_name": (
                prediction["performance_model"]["model_name"]
                if prediction.get("performance_model") else None
            ),
            "performance_weighting_policy": (
                prediction["performance_model"]["weighting_policy"]
                if prediction.get("performance_model") else None
            ),
            "degradation_model_name": (
                prediction["degradation_model"]["model_name"]
                if prediction.get("degradation_model") else None
            ),
            "degradation_weighting_policy": (
                prediction["degradation_model"]["weighting_policy"]
                if prediction.get("degradation_model") else None
            ),
            "performance_prediction": values["performance_mean"],
            "degradation_prediction": values["degradation_mean"],
            "p0_performance_prediction": _history_regression_value(prediction, "P0", values["compound_role"]),
            "p1_performance_prediction": _history_regression_value(prediction, "P1", values["compound_role"]),
            "d0_degradation_prediction": _history_regression_value(prediction, "D0", values["compound_role"]),
            "d1_degradation_prediction": _history_regression_value(prediction, "D1", values["compound_role"]),
            "prediction_fingerprint": prediction["prediction_fingerprint"],
        })
    if not additions:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_HISTORY_COLUMNS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerows(additions)
    return len(additions)


def _history_regression_value(
    prediction: Mapping[str, object], model_code: str, role: object
) -> float | None:
    outputs = dict(prediction.get("regression_predictions", {}))
    if model_code not in outputs:
        return None
    return float(dict(outputs[model_code]["compounds"])[str(role)]["predicted_value"])


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _write_json(path: Path, values: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
