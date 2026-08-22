from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from wodata.contracts import canonical_hash

from wostrategy.analysis.tyre_descriptor_domain import descriptor_domain_diagnostics

from wostrategy.model.cross_event_tyre_prediction import (
    COMPOUNDS,
    DEGRADATION_FEATURES,
    MODEL_VERSION,
    PERFORMANCE_FEATURES,
    AdjacentGapPerformanceModel,
    LinearDegradationModel,
    ReadinessConfig,
    diagnose_degradation_readiness,
    diagnose_performance_readiness,
)
from wostrategy.model.observation_weighting import WEIGHT_POLICIES, get_weight_policy


VALIDATION_SCHEMA_VERSION = "schema_v1"
PRODUCTION_POLICY = {
    "performance": {"formulation_code": "P0", "model_name": "performance_baseline", "weighting_policy": "uniform"},
    "degradation": {"formulation_code": "D0", "model_name": "degradation_baseline", "weighting_policy": "uniform"},
}
PIRELLI_INFORMED_POLICY = {
    "performance": {"formulation_code": "P1", "model_name": "performance_pirelli_v1", "weighting_policy": "uniform"},
    "degradation": {"formulation_code": "D1", "model_name": "degradation_pirelli_v1", "weighting_policy": "uniform"},
}
# Backwards-compatible internal name; user-facing artifacts use Pirelli-informed.
SHADOW_POLICY = PIRELLI_INFORMED_POLICY
MODEL_CODES = {
    "performance_baseline": "P0",
    "performance_pirelli_v1": "P1",
    "degradation_baseline": "D0",
    "degradation_pirelli_v1": "D1",
}
COEFFICIENT_COLUMNS = (
    "alpha_C1_C2", "alpha_C2_C3", "alpha_C3_C4", "alpha_C4_C5",
    "alpha_C1", "alpha_C2", "alpha_C3", "alpha_C4", "alpha_C5",
    "beta_stress", "beta_abrasion", "beta_grip",
)
VALIDATION_COLUMNS = (
    "validation_type", "season", "round", "predicted_round", "event",
    "compound_role", "absolute_compound", "model_code", "model_name",
    "model_version", "weighting_policy", "predicted_value", "observed_value",
    "residual", "absolute_residual", "tyre_stress", "asphalt_abrasion",
    "asphalt_grip", "hard_compound", "medium_compound", "soft_compound",
    "support_clean_lap_count", "support_run_count", "support_stint_count",
    "observed_p10", "observed_median", "observed_p90", "observed_w80",
    "absolute_residual_over_w80", "predicted_gap_contributions",
    "observed_increment_from_medium", "descriptor_tuple", "boundary_descriptors",
    "any_boundary_descriptor", "boundary_descriptor_count", "stress_is_boundary",
    "abrasion_is_boundary", "grip_is_boundary", "stress_level_seen_in_training",
    "abrasion_level_seen_in_training", "grip_level_seen_in_training",
    "exact_descriptor_tuple_seen", "exact_descriptor_tuple_prior_count",
    "training_round_start", "training_round_end",
    "training_data_hash",
)
COEFFICIENT_HISTORY_COLUMNS = (
    "validation_type", "fit_context", "season", "predicted_round",
    "training_round_start", "training_round_end", "model_code", "model_name",
    "model_version", "weighting_policy", "observation_count",
    "informative_observation_count", "design_rank", "parameter_count",
    "condition_number", "ridge_alpha", "training_data_hash",
    "coefficient_names", "coefficients_json", *COEFFICIENT_COLUMNS,
)


@dataclass(frozen=True)
class ValidationConfig:
    readiness: ReadinessConfig = ReadinessConfig()
    weight_policies: tuple[str, ...] = WEIGHT_POLICIES
    evidence_field: str = "support_clean_lap_count"
    evidence_transformation: str = "sqrt"
    performance_ridge_alpha: float = 0.0
    degradation_ridge_alpha: float = 0.0


@dataclass(frozen=True)
class ValidationResult:
    payload: dict[str, object]
    predictions: pd.DataFrame
    coefficient_history: pd.DataFrame
    diagnostic_summary: dict[str, object]


def rolling_validate(
    observations: pd.DataFrame, config: ValidationConfig | None = None
) -> ValidationResult:
    config = config or ValidationConfig()
    ordered = observations.sort_values(["season", "round", "compound_role"]).reset_index(drop=True)
    prediction_records: list[dict[str, object]] = []
    coefficient_records: list[dict[str, object]] = []
    cutoff_readiness: list[dict[str, object]] = []
    for season in sorted(int(value) for value in ordered["season"].unique()):
        season_rows = ordered.loc[ordered["season"].eq(season)]
        rounds = sorted(int(value) for value in season_rows["round"].unique())
        for predicted_round in rounds[1:]:
            historical = season_rows.loc[season_rows["round"] < predicted_round].copy()
            target = season_rows.loc[season_rows["round"].eq(predicted_round)].copy()
            training_hash = validation_training_hash(historical)
            readiness = readiness_at_cutoff(historical, config.readiness)
            cutoff_readiness.append(
                {
                    "season": season,
                    "predicted_round": predicted_round,
                    "training_round_start": _optional_round(historical, "min"),
                    "training_round_end": _optional_round(historical, "max"),
                    "training_data_hash": training_hash,
                    **{name: value.to_dict() for name, value in readiness.items()},
                }
            )
            fitted = _fit_eligible_models(historical, readiness, config)
            for model in fitted:
                coefficient_records.append(
                    _coefficient_record(
                        model, historical, training_hash,
                        fit_context="historical_rolling", predicted_round=predicted_round,
                    )
                )
                if model.code.startswith("P"):
                    prediction_records.extend(
                        _performance_records(model, target, historical, training_hash)
                    )
                else:
                    prediction_records.extend(
                        _degradation_records(model, target, historical, training_hash)
                    )

        current_hash = validation_training_hash(season_rows)
        current_readiness = readiness_at_cutoff(season_rows, config.readiness)
        for model in _fit_eligible_models(season_rows, current_readiness, config):
            coefficient_records.append(
                _coefficient_record(
                    model, season_rows, current_hash,
                    fit_context="current", predicted_round=None,
                )
            )

    predictions = pd.DataFrame(prediction_records, columns=VALIDATION_COLUMNS)
    if not predictions.empty:
        predictions = predictions.sort_values(
            ["season", "round", "model_code", "weighting_policy", "compound_role"]
        ).reset_index(drop=True)
    coefficients = pd.DataFrame(coefficient_records, columns=COEFFICIENT_HISTORY_COLUMNS)
    if not coefficients.empty:
        coefficients = coefficients.sort_values(
            ["season", "training_round_end", "fit_context", "model_code", "weighting_policy"]
        ).reset_index(drop=True)
    metrics = validation_metrics(predictions)
    rolling_comparison = {
        target: numerical_comparison(predictions, target)
        for target in ("performance", "degradation")
    }
    payload: dict[str, object] = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "validation_type": "historical_rolling",
        "model_version": MODEL_VERSION,
        "production_policy": PRODUCTION_POLICY,
        "pirelli_informed_policy": PIRELLI_INFORMED_POLICY,
        "promotion_policy": "explicit development decision only; no automatic metric-based promotion",
        "model_definitions": _model_definitions(),
        "weighting": _weighting_metadata(config),
        "evidence_support": _evidence_metadata(config),
        "historical_cutoffs": cutoff_readiness,
        "predictions": _records_for_json(predictions),
        "metrics": metrics,
        "rolling_comparison_diagnostic_only": rolling_comparison,
        "selection": {
            "performance": {**PRODUCTION_POLICY["performance"], "basis": "fixed production policy"},
            "degradation": {**PRODUCTION_POLICY["degradation"], "basis": "fixed production policy"},
        },
    }
    interim = ValidationResult(payload, predictions, coefficients, {})
    summary = build_diagnostic_summary(interim, prospective=pd.DataFrame())
    return ValidationResult(payload, predictions, coefficients, summary)


def write_validation_artifacts(
    result: ValidationResult,
    *,
    json_path: Path,
    csv_path: Path,
    coefficient_history_path: Path | None = None,
    diagnostic_summary_path: Path | None = None,
) -> None:
    _write_json(json_path, result.payload)
    _write_frame(csv_path, result.predictions)
    if coefficient_history_path is not None:
        _write_frame(coefficient_history_path, result.coefficient_history)
    if diagnostic_summary_path is not None:
        _write_json(diagnostic_summary_path, result.diagnostic_summary)


def build_diagnostic_summary(
    result: ValidationResult, prospective: pd.DataFrame
) -> dict[str, object]:
    predictions = result.predictions
    uniform = predictions.loc[predictions.get("weighting_policy", pd.Series(dtype=str)).eq("uniform")]
    metrics = [row for row in validation_metrics(uniform)]
    largest: dict[str, object] = {}
    descriptor_diagnostics: dict[str, object] = {}
    for model_name, group in uniform.groupby("model_name", sort=True):
        largest[str(model_name)] = {
            "largest_absolute_residual_observations": _records_for_json(
                group.sort_values(
                    ["absolute_residual", "season", "round", "absolute_compound"],
                    ascending=[False, True, True, True],
                ).head(5)
            ),
            "largest_mean_residual_rounds": _rank_rounds(group, "absolute_bias"),
            "largest_RMSE_rounds": _rank_rounds(group, "RMSE"),
        }
        features = PERFORMANCE_FEATURES if str(model_name).startswith("performance") else DEGRADATION_FEATURES
        descriptor_diagnostics[str(model_name)] = {
            feature: _grouped_metrics(group, feature) for feature in features
        }
    coefficient_stability: dict[str, object] = {}
    uniform_coefficients = result.coefficient_history.loc[
        result.coefficient_history.get("weighting_policy", pd.Series(dtype=str)).eq("uniform")
    ]
    requested = {
        "P1": ("beta_stress", "beta_grip"),
        "D1": ("beta_stress", "beta_abrasion", "beta_grip"),
    }
    for model_code, names in requested.items():
        model_rows = uniform_coefficients.loc[uniform_coefficients["model_code"].eq(model_code)]
        coefficient_stability[model_code] = {
            name: [
                {
                    "training_round_end": int(row.training_round_end),
                    "fit_context": str(row.fit_context),
                    "value": float(getattr(row, name)),
                }
                for row in model_rows.itertuples(index=False)
                if pd.notna(getattr(row, name))
            ]
            for name in names
        }
    prospective_metrics = validation_metrics(prospective) if not prospective.empty else []
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "production_regressions": PRODUCTION_POLICY,
        "pirelli_informed_regressions": PIRELLI_INFORMED_POLICY,
        "historical_rolling": {
            "metrics": metrics,
            "coefficient_stability": coefficient_stability,
            "largest_residuals": largest,
            "descriptor_diagnostics": descriptor_diagnostics,
            "absolute_residual_vs_w80": _posterior_width_diagnostics(uniform),
            "descriptor_domain_analysis": _descriptor_domain_analysis(uniform),
        },
        "prospective": {
            "prediction_count": int(len(prospective)),
            "metrics": prospective_metrics,
        },
    }


def validation_metrics(predictions: pd.DataFrame) -> list[dict[str, object]]:
    if predictions.empty:
        return []
    output: list[dict[str, object]] = []
    for (model_code, model_name, policy), group in predictions.groupby(
        ["model_code", "model_name", "weighting_policy"], sort=True
    ):
        output.append(
            {
                "target": _target_for_model(str(model_name)),
                "model_code": str(model_code),
                "model_name": str(model_name),
                "weighting_policy": str(policy),
                **_metric_values(group),
                "per_round": _grouped_metrics(group, "round"),
                "per_compound": _grouped_metrics(group, "absolute_compound"),
            }
        )
    return output


def numerical_comparison(predictions: pd.DataFrame, target: str) -> dict[str, object] | None:
    if predictions.empty:
        return None
    relevant = predictions.loc[predictions["model_name"].astype(str).str.startswith(target)].copy()
    if relevant.empty:
        return None
    candidates = sorted(
        (str(model), str(policy))
        for model, policy in relevant[["model_name", "weighting_policy"]]
        .drop_duplicates().itertuples(index=False, name=None)
    )
    key_sets = [
        set(zip(group["season"], group["round"], group["absolute_compound"]))
        for _, group in relevant.groupby(["model_name", "weighting_policy"], sort=True)
    ]
    common_keys = set.intersection(*key_sets) if key_sets else set()
    comparison = []
    for model, policy in candidates:
        group = relevant.loc[
            relevant["model_name"].eq(model) & relevant["weighting_policy"].eq(policy)
        ].copy()
        if common_keys:
            group = group.loc[
                [
                    (row.season, row.round, row.absolute_compound) in common_keys
                    for row in group.itertuples(index=False)
                ]
            ]
        comparison.append(
            {"model_name": model, "model_code": MODEL_CODES[model],
             "weighting_policy": policy, **_metric_values(group)}
        )
    numerical_best = min(comparison, key=lambda item: float(item["RMSE"]))
    return {
        "purpose": "diagnostic only; does not change the fixed production policy",
        "common_prediction_count": len(common_keys),
        "candidates": comparison,
        "numerical_best": numerical_best,
    }


def readiness_at_cutoff(
    historical: pd.DataFrame, config: ReadinessConfig
) -> dict[str, object]:
    return {
        "performance_baseline": diagnose_performance_readiness(
            historical, config, include_environment=False
        ),
        "performance_pirelli_v1": diagnose_performance_readiness(
            historical, config, include_environment=True
        ),
        "degradation_baseline": diagnose_degradation_readiness(
            historical, config, include_environment=False
        ),
        "degradation_pirelli_v1": diagnose_degradation_readiness(
            historical, config, include_environment=True
        ),
    }


def validation_training_hash(frame: pd.DataFrame) -> str:
    columns = [
        "season", "round", "compound_role", "absolute_compound", "event_hard_compound",
        *DEGRADATION_FEATURES, "performance", "degradation", "support_clean_lap_count",
        "source_result_hash", "preview_source_hash",
    ]
    present = [column for column in columns if column in frame.columns]
    records = frame[present].sort_values(
        ["season", "round", "compound_role"]
    ).replace({np.nan: None}).to_dict("records")
    return canonical_hash(records)


def _fit_eligible_models(
    historical: pd.DataFrame,
    readiness: Mapping[str, object],
    config: ValidationConfig,
) -> list[AdjacentGapPerformanceModel | LinearDegradationModel]:
    models: list[AdjacentGapPerformanceModel | LinearDegradationModel] = []
    for policy_name in config.weight_policies:
        policy = get_weight_policy(
            policy_name,
            evidence_field=config.evidence_field,
            evidence_transformation=config.evidence_transformation,
        )
        try:
            policy.weights(historical)
        except ValueError:
            continue
        candidates = (
            ("performance_baseline", AdjacentGapPerformanceModel(
                include_environment=False, ridge_alpha=config.performance_ridge_alpha,
                weight_policy=policy)),
            ("performance_pirelli_v1", AdjacentGapPerformanceModel(
                include_environment=True, ridge_alpha=config.performance_ridge_alpha,
                weight_policy=policy)),
            ("degradation_baseline", LinearDegradationModel(
                include_environment=False, ridge_alpha=config.degradation_ridge_alpha,
                weight_policy=policy)),
            ("degradation_pirelli_v1", LinearDegradationModel(
                include_environment=True, ridge_alpha=config.degradation_ridge_alpha,
                weight_policy=policy)),
        )
        for name, model in candidates:
            if readiness[name].ready:
                model.fit(historical)
                models.append(model)
    return models


def _coefficient_record(
    model: AdjacentGapPerformanceModel | LinearDegradationModel,
    historical: pd.DataFrame,
    training_hash: str,
    *,
    fit_context: str,
    predicted_round: int | None,
) -> dict[str, object]:
    payload = model.to_dict()
    labels = list(payload["coefficient_labels"])
    values = [float(value) for value in payload["coefficients"]]
    coefficients = dict(zip(labels, values))
    metadata = model.training_metadata
    record: dict[str, object] = {
        "validation_type": "historical_rolling",
        "fit_context": fit_context,
        "season": int(historical["season"].iloc[0]),
        "predicted_round": predicted_round,
        "training_round_start": _optional_round(historical, "min"),
        "training_round_end": _optional_round(historical, "max"),
        "model_code": model.code,
        "model_name": model.name,
        "model_version": MODEL_VERSION,
        "weighting_policy": model.weight_policy.name,
        "observation_count": int(len(historical)),
        "informative_observation_count": int(metadata["observation_count"]),
        "design_rank": int(metadata["design_rank"]),
        "parameter_count": int(metadata["parameter_count"]),
        "condition_number": float(metadata["condition_number"]),
        "ridge_alpha": float(model.ridge_alpha),
        "training_data_hash": training_hash,
        "coefficient_names": ",".join(labels),
        "coefficients_json": json.dumps(coefficients, sort_keys=True, separators=(",", ":")),
        **dict.fromkeys(COEFFICIENT_COLUMNS, None),
        **coefficients,
    }
    return record


def _performance_records(
    model: AdjacentGapPerformanceModel,
    target: pd.DataFrame,
    historical: pd.DataFrame,
    training_hash: str,
) -> list[dict[str, object]]:
    informative = target.loc[
        target["absolute_compound"].astype(str) != target["event_hard_compound"].astype(str)
    ]
    records = []
    for row in informative.itertuples(index=False):
        predicted = model.predict(
            absolute_compound=str(row.absolute_compound),
            hard_compound=str(row.event_hard_compound),
            tyre_stress=float(row.tyre_stress),
            asphalt_grip=float(row.asphalt_grip),
        ).mean
        records.append(
            _prediction_record(
                row, target, historical, model.code, model.name, model.weight_policy.name,
                predicted, float(row.performance), getattr(row, "performance_p10", None),
                getattr(row, "performance_p90", None), training_hash,
                gap_contributions=_gap_contributions(model, row),
                observed_increment=_observed_increment_from_medium(row, target),
            )
        )
    return records


def _degradation_records(
    model: LinearDegradationModel,
    target: pd.DataFrame,
    historical: pd.DataFrame,
    training_hash: str,
) -> list[dict[str, object]]:
    records = []
    for row in target.itertuples(index=False):
        predicted = model.predict(
            absolute_compound=str(row.absolute_compound), tyre_stress=float(row.tyre_stress),
            asphalt_abrasion=float(row.asphalt_abrasion), asphalt_grip=float(row.asphalt_grip),
        ).mean
        records.append(
            _prediction_record(
                row, target, historical, model.code, model.name, model.weight_policy.name,
                predicted, float(row.degradation), getattr(row, "degradation_p10", None),
                getattr(row, "degradation_p90", None), training_hash,
            )
        )
    return records


def _prediction_record(
    row: object,
    target: pd.DataFrame,
    historical: pd.DataFrame,
    model_code: str,
    model_name: str,
    weighting_policy: str,
    predicted: float,
    observed: float,
    observed_p10: object,
    observed_p90: object,
    training_hash: str,
    *,
    gap_contributions: Mapping[str, float] | None = None,
    observed_increment: float | None = None,
) -> dict[str, object]:
    p10 = _optional_float(observed_p10)
    p90 = _optional_float(observed_p90)
    allocations = _allocation(target, row)
    domain_target = "performance" if model_code.startswith("P") else "degradation"
    domain = descriptor_domain_diagnostics(
        target=domain_target, historical=historical, target_values=row._asdict()
    )
    w80 = p90 - p10 if p10 is not None and p90 is not None else None
    residual = float(predicted - observed)
    return {
        "validation_type": "historical_rolling", "season": int(row.season),
        "round": int(row.round), "predicted_round": int(row.round),
        "event": str(getattr(row, "event", "")), "compound_role": str(row.compound_role),
        "absolute_compound": str(row.absolute_compound), "model_code": model_code,
        "model_name": model_name, "model_version": MODEL_VERSION,
        "weighting_policy": weighting_policy, "predicted_value": float(predicted),
        "observed_value": float(observed), "residual": residual,
        "absolute_residual": abs(residual), "tyre_stress": float(row.tyre_stress),
        "asphalt_abrasion": float(row.asphalt_abrasion), "asphalt_grip": float(row.asphalt_grip),
        **allocations,
        "support_clean_lap_count": _optional_int(getattr(row, "support_clean_lap_count", None)),
        "support_run_count": _optional_int(getattr(row, "support_run_count", None)),
        "support_stint_count": _optional_int(getattr(row, "support_stint_count", None)),
        "observed_p10": p10, "observed_median": float(observed), "observed_p90": p90,
        "observed_w80": w80,
        "absolute_residual_over_w80": abs(residual) / w80 if w80 and w80 > 0 else None,
        "predicted_gap_contributions": (
            json.dumps(gap_contributions, sort_keys=True, separators=(",", ":"))
            if gap_contributions else None
        ),
        "observed_increment_from_medium": observed_increment,
        "descriptor_tuple": json.dumps(domain["descriptor_tuple"], separators=(",", ":")),
        "boundary_descriptors": json.dumps(domain["boundary_descriptors"], separators=(",", ":")),
        "any_boundary_descriptor": domain["any_boundary_descriptor"],
        "boundary_descriptor_count": domain["boundary_descriptor_count"],
        "stress_is_boundary": domain["stress_is_boundary"],
        "abrasion_is_boundary": domain["abrasion_is_boundary"],
        "grip_is_boundary": domain["grip_is_boundary"],
        "stress_level_seen_in_training": domain["stress_level_seen_in_training"],
        "abrasion_level_seen_in_training": domain["abrasion_level_seen_in_training"],
        "grip_level_seen_in_training": domain["grip_level_seen_in_training"],
        "exact_descriptor_tuple_seen": domain["exact_descriptor_tuple_seen"],
        "exact_descriptor_tuple_prior_count": domain["exact_descriptor_tuple_prior_count"],
        "training_round_start": _optional_round(historical, "min"),
        "training_round_end": _optional_round(historical, "max"),
        "training_data_hash": training_hash,
    }


def _gap_contributions(model: AdjacentGapPerformanceModel, row: object) -> dict[str, float]:
    hard_index = COMPOUNDS.index(str(row.event_hard_compound))
    compound_index = COMPOUNDS.index(str(row.absolute_compound))
    output: dict[str, float] = {}
    for index in range(min(hard_index, compound_index), max(hard_index, compound_index)):
        left, right = COMPOUNDS[index], COMPOUNDS[index + 1]
        value = model.predict(
            absolute_compound=right, hard_compound=left,
            tyre_stress=float(row.tyre_stress), asphalt_grip=float(row.asphalt_grip),
        ).mean
        output[f"{left}->{right}"] = value if compound_index >= hard_index else -value
    return output


def _observed_increment_from_medium(row: object, target: pd.DataFrame) -> float | None:
    if str(row.compound_role).upper() != "SOFT":
        return None
    medium = target.loc[target["compound_role"].astype(str).str.upper().eq("MEDIUM")]
    if medium.empty:
        return None
    return float(row.performance) - float(medium.iloc[0]["performance"])


def _allocation(target: pd.DataFrame, row: object) -> dict[str, str | None]:
    by_role = {
        str(role).upper(): str(compound)
        for role, compound in target[["compound_role", "absolute_compound"]]
        .drop_duplicates().itertuples(index=False, name=None)
    }
    return {
        "hard_compound": str(getattr(row, "hard_compound", by_role.get("HARD") or row.event_hard_compound)),
        "medium_compound": str(getattr(row, "medium_compound", by_role.get("MEDIUM"))) if getattr(row, "medium_compound", by_role.get("MEDIUM")) is not None else None,
        "soft_compound": str(getattr(row, "soft_compound", by_role.get("SOFT"))) if getattr(row, "soft_compound", by_role.get("SOFT")) is not None else None,
    }


def _rank_rounds(group: pd.DataFrame, ranking: str) -> list[dict[str, object]]:
    rows = []
    for round_number, values in group.groupby("round", sort=True):
        metrics = _metric_values(values)
        rows.append({"round": int(round_number), "absolute_bias": abs(float(metrics["bias"])), **metrics})
    return sorted(rows, key=lambda row: (-float(row[ranking]), int(row["round"])))[:5]


def _posterior_width_diagnostics(frame: pd.DataFrame) -> list[dict[str, object]]:
    output = []
    for model_name, group in frame.groupby("model_name", sort=True):
        usable = group.dropna(subset=["absolute_residual", "observed_w80"])
        correlation = None
        if len(usable) >= 2 and usable["observed_w80"].nunique() > 1:
            correlation = float(np.corrcoef(usable["absolute_residual"], usable["observed_w80"])[0, 1])
        output.append({
            "model_name": str(model_name), "observation_count": int(len(usable)),
            "pearson_correlation": correlation,
            "interpretation": "descriptive only; posterior width is not used as regression truth confidence",
        })
    return output


def _descriptor_domain_analysis(frame: pd.DataFrame) -> dict[str, object]:
    output: dict[str, object] = {}
    for target, model_codes in (("performance", ("P0", "P1")),
                                ("degradation", ("D0", "D1"))):
        target_rows = frame.loc[frame["model_code"].isin(model_codes)]
        splits = {
            "interior": ~target_rows["any_boundary_descriptor"].astype(bool),
            "boundary_exposed": target_rows["any_boundary_descriptor"].astype(bool),
            "tuple_seen": target_rows["exact_descriptor_tuple_seen"].astype(bool),
            "tuple_unseen": ~target_rows["exact_descriptor_tuple_seen"].astype(bool),
        }
        output[target] = {}
        for split_name, mask in splits.items():
            subset = target_rows.loc[mask]
            output[target][split_name] = {
                code: _metric_values(subset.loc[subset["model_code"].eq(code)])
                for code in model_codes
            }
        output[target]["interpretation"] = (
            "diagnostic only; descriptor endpoints are saturated ordinal categories "
            "and subset sizes are too small for automatic model changes"
        )
    return output


def _metric_values(group: pd.DataFrame) -> dict[str, object]:
    residual = group["residual"].to_numpy(dtype=float)
    return {
        "prediction_count": int(len(residual)),
        "MAE": float(np.mean(np.abs(residual))) if len(residual) else math.nan,
        "RMSE": float(np.sqrt(np.mean(residual**2))) if len(residual) else math.nan,
        "mean_error": float(np.mean(residual)) if len(residual) else math.nan,
        "bias": float(np.mean(residual)) if len(residual) else math.nan,
    }


def _grouped_metrics(group: pd.DataFrame, column: str) -> list[dict[str, object]]:
    return [
        {column: _json_scalar(key), **_metric_values(values)}
        for key, values in group.groupby(column, sort=True)
    ]


def _model_definitions() -> dict[str, object]:
    return {
        "performance_baseline": {"formulation_code": "P0", "parameters": 4, "features": [],
            "definition": "four transferable adjacent compound gap intercepts"},
        "performance_pirelli_v1": {"formulation_code": "P1", "parameters": 6,
            "features": list(PERFORMANCE_FEATURES),
            "definition": "P0 plus shared stress and grip sensitivities"},
        "degradation_baseline": {"formulation_code": "D0", "parameters": 5, "features": [],
            "definition": "five absolute compound intercepts"},
        "degradation_pirelli_v1": {"formulation_code": "D1", "parameters": 8,
            "features": list(DEGRADATION_FEATURES),
            "definition": "D0 plus shared stress, abrasion, and grip sensitivities"},
    }


def _weighting_metadata(config: ValidationConfig) -> dict[str, object]:
    return {
        "concept": "weighting policy is separate from regression formulation",
        "policies_compared": list(config.weight_policies),
        "production_policy": "uniform", "alternative_policies_are_shadow_diagnostics": True,
        "evidence_field": config.evidence_field,
        "evidence_transformation": config.evidence_transformation,
        "event_total_weight": 1.0, "posterior_interval_weighting": False,
    }


def _evidence_metadata(config: ValidationConfig) -> dict[str, object]:
    return {
        "selected_for_shadow_weighting": {
            "field": config.evidence_field,
            "source": "persisted clean_laps.csv grouped by compound role",
            "meaning": "selected clean-air long-run laps directly supporting the compound",
            "transformation": config.evidence_transformation,
        },
        "available_not_selected": [
            {"field": "support_run_count", "meaning": "unique Driver/LongRunId groups"},
            {"field": "support_stint_count", "meaning": "unique Driver/Stint groups"},
            {"field": "retro_effective_sample_size", "meaning": "event-level posterior diagnostic"},
            {"field": "retro_weighted_rmse", "meaning": "event-level Retro fit diagnostic"},
        ],
        "posterior_intervals": {"fields": ["P10", "Median", "P90", "W80=P90-P10"],
            "use": "diagnostic only", "used_for_weighting": False},
    }


def _records_for_json(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {key: _json_scalar(value) for key, value in record.items()}
        for record in frame.replace({np.nan: None}).to_dict("records")
    ]


def _json_scalar(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _optional_round(frame: pd.DataFrame, operation: str) -> int | None:
    if frame.empty:
        return None
    value = frame["round"].min() if operation == "min" else frame["round"].max()
    return int(value)


def _target_for_model(model_name: str) -> str:
    return "performance" if model_name.startswith("performance") else "degradation"


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _write_json(path: Path, values: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
