from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROSPECTIVE_COLUMNS = (
    "validation_type", "season", "round", "event", "compound_role",
    "absolute_compound", "target", "model_code", "model_name", "model_version",
    "weighting_policy", "prediction_fingerprint", "prediction_timestamp",
    "trained_through_round", "training_data_hash", "predicted_value", "observed_value",
    "residual", "absolute_residual", "tyre_stress", "asphalt_abrasion", "asphalt_grip",
    "hard_compound", "medium_compound", "soft_compound", "support_clean_lap_count",
    "support_run_count", "support_stint_count", "observed_p10", "observed_median",
    "observed_p90", "observed_w80", "source_result_hash", "result_available_timestamp",
    "descriptor_tuple", "boundary_descriptors", "any_boundary_descriptor",
    "boundary_descriptor_count", "stress_is_boundary", "abrasion_is_boundary",
    "grip_is_boundary", "stress_level_seen_in_training",
    "abrasion_level_seen_in_training", "grip_level_seen_in_training",
    "exact_descriptor_tuple_seen", "exact_descriptor_tuple_prior_count",
)


def score_available_prospective_predictions(
    *, year_root: Path, observations: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    existing = _read_existing(output_path)
    existing_keys = {
        (str(row.prediction_fingerprint), str(row.model_code), str(row.target),
         str(row.compound_role), str(row.absolute_compound))
        for row in existing.itertuples(index=False)
    } if not existing.empty else set()
    additions: list[dict[str, object]] = []
    observed_rounds = set(int(value) for value in observations["round"].unique())
    for prediction in _latest_prediction_versions(year_root):
        round_number = int(prediction["round"])
        if round_number not in observed_rounds:
            continue
        event_rows = observations.loc[observations["round"].eq(round_number)]
        for model_code, output in dict(prediction.get("regression_predictions", {})).items():
            target = str(output["target"])
            for role, predicted_row in dict(output["compounds"]).items():
                if target == "performance" and str(role).upper() == "HARD":
                    continue
                compound = str(predicted_row["absolute_compound"])
                key = (str(prediction["prediction_fingerprint"]), str(model_code), target,
                       str(role), compound)
                if key in existing_keys:
                    continue
                matched = event_rows.loc[
                    event_rows["compound_role"].astype(str).str.upper().eq(str(role).upper())
                    & event_rows["absolute_compound"].astype(str).eq(compound)
                ]
                if matched.empty:
                    continue
                observed = matched.iloc[0]
                additions.append(
                    _score_row(prediction, output, model_code, role, predicted_row, observed)
                )
                existing_keys.add(key)
    if additions:
        additions_frame = pd.DataFrame(additions, columns=PROSPECTIVE_COLUMNS)
        combined = (
            additions_frame
            if existing.empty
            else pd.concat([existing, additions_frame], ignore_index=True)
        )
    else:
        combined = existing
    if combined.empty:
        combined = pd.DataFrame(columns=PROSPECTIVE_COLUMNS)
    combined = combined.reindex(columns=PROSPECTIVE_COLUMNS).sort_values(
        ["season", "round", "prediction_timestamp", "model_code", "compound_role"]
    ).reset_index(drop=True)
    _write_frame(output_path, combined)
    return _read_existing(output_path)


def _latest_prediction_versions(year_root: Path) -> list[dict[str, object]]:
    latest: dict[tuple[int, int], dict[str, object]] = {}
    for path in sorted((year_root / "predictions").glob("round=*/versions/*.json")):
        prediction = _read_json(path)
        if not prediction.get("regression_predictions"):
            continue
        key = (int(prediction["season"]), int(prediction["round"]))
        candidate_order = (
            int(prediction.get("trained_through_round") or -1),
            str(prediction.get("prediction_timestamp") or ""),
            str(prediction.get("prediction_fingerprint") or ""),
        )
        current = latest.get(key)
        current_order = (
            int(current.get("trained_through_round") or -1),
            str(current.get("prediction_timestamp") or ""),
            str(current.get("prediction_fingerprint") or ""),
        ) if current is not None else None
        if current_order is None or candidate_order > current_order:
            latest[key] = prediction
    return [latest[key] for key in sorted(latest)]


def _score_row(
    prediction: dict[str, object],
    output: dict[str, object],
    model_code: str,
    role: str,
    predicted_row: dict[str, object],
    observed: pd.Series,
) -> dict[str, object]:
    target = str(output["target"])
    prefix = "performance" if target == "performance" else "degradation"
    predicted_value = float(predicted_row["predicted_value"])
    observed_value = float(observed[prefix])
    p10 = _optional_float(observed.get(f"{prefix}_p10"))
    p90 = _optional_float(observed.get(f"{prefix}_p90"))
    residual = predicted_value - observed_value
    allocation = dict(prediction["compound_allocation"])
    features = dict(prediction["features"])
    domain = dict(output.get("descriptor_domain", {}))
    return {
        "validation_type": "prospective",
        "season": int(prediction["season"]), "round": int(prediction["round"]),
        "event": str(prediction["event"]), "compound_role": str(role),
        "absolute_compound": str(predicted_row["absolute_compound"]), "target": target,
        "model_code": str(model_code), "model_name": str(output["model_name"]),
        "model_version": str(prediction["model"]),
        "weighting_policy": str(output["weighting_policy"]),
        "prediction_fingerprint": str(prediction["prediction_fingerprint"]),
        "prediction_timestamp": str(prediction["prediction_timestamp"]),
        "trained_through_round": prediction["trained_through_round"],
        "training_data_hash": str(prediction["training_data_hash"]),
        "predicted_value": predicted_value, "observed_value": observed_value,
        "residual": residual, "absolute_residual": abs(residual),
        "tyre_stress": float(features["tyre_stress"]),
        "asphalt_abrasion": float(features["asphalt_abrasion"]),
        "asphalt_grip": float(features["asphalt_grip"]),
        "hard_compound": str(allocation["HARD"]),
        "medium_compound": str(allocation["MEDIUM"]),
        "soft_compound": str(allocation["SOFT"]),
        "support_clean_lap_count": _optional_int(observed.get("support_clean_lap_count")),
        "support_run_count": _optional_int(observed.get("support_run_count")),
        "support_stint_count": _optional_int(observed.get("support_stint_count")),
        "observed_p10": p10, "observed_median": observed_value, "observed_p90": p90,
        "observed_w80": p90 - p10 if p10 is not None and p90 is not None else None,
        "source_result_hash": str(observed["source_result_hash"]),
        "result_available_timestamp": str(observed.get("source_result_timestamp", "")),
        "descriptor_tuple": json.dumps(domain.get("descriptor_tuple"), separators=(",", ":")),
        "boundary_descriptors": json.dumps(domain.get("boundary_descriptors"), separators=(",", ":")),
        "any_boundary_descriptor": domain.get("any_boundary_descriptor"),
        "boundary_descriptor_count": domain.get("boundary_descriptor_count"),
        "stress_is_boundary": domain.get("stress_is_boundary"),
        "abrasion_is_boundary": domain.get("abrasion_is_boundary"),
        "grip_is_boundary": domain.get("grip_is_boundary"),
        "stress_level_seen_in_training": domain.get("stress_level_seen_in_training"),
        "abrasion_level_seen_in_training": domain.get("abrasion_level_seen_in_training"),
        "grip_level_seen_in_training": domain.get("grip_level_seen_in_training"),
        "exact_descriptor_tuple_seen": domain.get("exact_descriptor_tuple_seen"),
        "exact_descriptor_tuple_prior_count": domain.get("exact_descriptor_tuple_prior_count"),
    }


def _read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=PROSPECTIVE_COLUMNS)
    frame = pd.read_csv(path)
    return frame.reindex(columns=PROSPECTIVE_COLUMNS)


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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
    frame.replace({np.nan: None}).to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)
