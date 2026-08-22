from __future__ import annotations

import json

import pandas as pd
import pytest

from wostrategy.analysis.tyre_prediction_prospective import (
    score_available_prospective_predictions,
)


def test_scores_original_version_once_without_rewriting_prediction(tmp_path):
    year_root = tmp_path / "year=2026"
    version = year_root / "predictions/round=2/versions/original.json"
    version.parent.mkdir(parents=True)
    prediction = {
        "season": 2026, "round": 2, "event": "Event 2", "model": "linear_v1.1",
        "prediction_timestamp": "2026-01-01T00:00:00+00:00",
        "prediction_fingerprint": "original", "trained_through_round": 1,
        "training_data_hash": "training-1",
        "features": {"tyre_stress": 3, "asphalt_abrasion": 2, "asphalt_grip": 4},
        "compound_allocation": {"HARD": "C2", "MEDIUM": "C3", "SOFT": "C4"},
        "regression_predictions": {
            "P0": _output("performance", "performance_baseline", [0, -0.2, -0.5]),
            "P1": _output("performance", "performance_pirelli_v1", [0, -0.25, -0.55]),
            "D0": _output("degradation", "degradation_baseline", [0.1, 0.2, 0.3]),
            "D1": _output("degradation", "degradation_pirelli_v1", [0.11, 0.21, 0.31]),
        },
    }
    older = json.loads(json.dumps(prediction))
    older["prediction_timestamp"] = "2025-12-31T00:00:00+00:00"
    older["prediction_fingerprint"] = "older"
    older["regression_predictions"]["P0"]["compounds"]["MEDIUM"]["predicted_value"] = -99
    (version.parent / "older.json").write_text(json.dumps(older), encoding="utf-8")
    version.write_text(json.dumps(prediction), encoding="utf-8")
    original_bytes = version.read_bytes()
    observations = _observations()
    output = year_root / "prospective_validation.csv"

    first = score_available_prospective_predictions(
        year_root=year_root, observations=observations, output_path=output
    )
    second = score_available_prospective_predictions(
        year_root=year_root, observations=observations, output_path=output
    )

    assert len(first) == 10  # P0/P1 exclude HARD; D0/D1 score all three roles.
    pd.testing.assert_frame_equal(first, second)
    assert version.read_bytes() == original_bytes
    assert set(first["validation_type"]) == {"prospective"}
    assert set(first["model_code"]) == {"P0", "P1", "D0", "D1"}
    assert first["exact_descriptor_tuple_seen"].all()
    p0_medium = first.loc[
        first["model_code"].eq("P0") & first["compound_role"].eq("MEDIUM")
    ].iloc[0]
    assert p0_medium.residual == pytest.approx(-0.2 - (-0.3))


def _output(target, model_name, values):
    descriptor_tuple = [3, 4] if target == "performance" else [3, 2, 4]
    return {
        "target": target, "model_name": model_name, "weighting_policy": "uniform",
        "descriptor_domain": {
            "descriptor_tuple": descriptor_tuple, "boundary_descriptors": [],
            "any_boundary_descriptor": False, "boundary_descriptor_count": 0,
            "stress_is_boundary": False,
            "abrasion_is_boundary": None if target == "performance" else False,
            "grip_is_boundary": False, "stress_level_seen_in_training": True,
            "abrasion_level_seen_in_training": None if target == "performance" else True,
            "grip_level_seen_in_training": True, "exact_descriptor_tuple_seen": True,
            "exact_descriptor_tuple_prior_count": 1,
        },
        "compounds": {
            role: {"absolute_compound": compound, "compound_role": role,
                   "predicted_value": value}
            for role, compound, value in zip(
                ("HARD", "MEDIUM", "SOFT"), ("C2", "C3", "C4"), values
            )
        },
    }


def _observations():
    rows = []
    for role, compound, performance, degradation in (
        ("HARD", "C2", 0.0, 0.12),
        ("MEDIUM", "C3", -0.3, 0.22),
        ("SOFT", "C4", -0.6, 0.32),
    ):
        rows.append({
            "season": 2026, "round": 2, "compound_role": role,
            "absolute_compound": compound, "performance": performance,
            "degradation": degradation, "performance_p10": performance - 0.1,
            "performance_p90": performance + 0.1,
            "degradation_p10": degradation - 0.02, "degradation_p90": degradation + 0.02,
            "support_clean_lap_count": 10, "support_run_count": 2,
            "support_stint_count": 2, "source_result_hash": "retro-2",
            "source_result_timestamp": "2026-01-02T00:00:00+00:00",
        })
    return pd.DataFrame(rows)
