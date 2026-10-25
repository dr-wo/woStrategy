import json
from pathlib import Path

import numpy as np
import pytest

from wostrategy.analysis.pre_race_tyre_prediction import get_pre_race_tyre_prediction
from wostrategy.analysis.tyre_prediction_prior import write_historical_baseline_prior
from wostrategy.model.cross_event_tyre_prediction import (
    AdjacentGapPerformanceModel,
    LinearDegradationModel,
)


ALLOCATION = {"HARD": "C2", "MEDIUM": "C3", "SOFT": "C4"}
ALLOCATION_URL = "https://example.test/official-allocation"


def _source_artifacts(root: Path, *, cutoff=16, include_environment=False) -> None:
    directory = (
        root / "wostrategy/tyre_prediction/schema_v1/year=2026/predictions/round=17"
    )
    directory.mkdir(parents=True, exist_ok=True)
    performance = AdjacentGapPerformanceModel(include_environment=include_environment)
    performance.coefficients = np.array(
        [-0.40, -0.27, -0.32, -0.33]
        if not include_environment
        else [-0.40, -0.27, -0.32, -0.33, 0.01, 0.02]
    )
    degradation = LinearDegradationModel(include_environment=include_environment)
    degradation.coefficients = np.array(
        [0.06, 0.09, 0.145, 0.209, 0.222]
        if not include_environment else [0.06, 0.09, 0.145, 0.209, 0.222, 0.01, 0.02, 0.03]
    )
    model = {
        "schema_version": "schema_v1", "model_version": "linear_v1.1",
        "season": 2026, "target_round": 17,
        "trained_through_round": cutoff,
        "training_data_hash": "training-hash-17",
        "regressions": {"P0": performance.to_dict(), "D0": degradation.to_dict()},
        "production_policy": {
            "performance": {"formulation_code": "P0"},
            "degradation": {"formulation_code": "D0"},
        },
    }
    prediction = {
        "season": 2026, "round": 17, "trained_through_round": cutoff,
        "training_rounds": list(range(1, cutoff + 1)),
        "training_data_hash": "training-hash-17",
        "prediction_fingerprint": "source-fingerprint",
        "prediction_timestamp": "2026-10-01T00:00:00+00:00",
        "model": "linear_v1.1",
    }
    (directory / "model.json").write_text(json.dumps(model), encoding="utf-8")
    (directory / "prediction.json").write_text(json.dumps(prediction), encoding="utf-8")


def _write(root: Path, **kwargs) -> Path:
    return write_historical_baseline_prior(
        season=2026, round_number=18, event="United States Grand Prix",
        circuit="Circuit of The Americas", compound_allocation=ALLOCATION,
        allocation_source_url=ALLOCATION_URL, data_root=root, **kwargs,
    )


def test_writes_prior_only_from_frozen_descriptor_free_models_and_reloads(tmp_path: Path) -> None:
    _source_artifacts(tmp_path)
    path = _write(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["prediction_status"] == "prior_only"
    assert payload["provenance"]["classification"] == "prior_only"
    assert payload["provenance"]["independent_of_current_fp"] is True
    assert payload["provenance"]["fp_evidence_automatic_update"] is False
    assert payload["provenance"]["descriptor_values_used"] is None
    assert payload["provenance"]["source_training_rounds"] == list(range(1, 17))
    assert payload["training_data_hash"] == "training-hash-17"
    assert all(item["performance_std"] is None for item in payload["compounds"].values())
    assert all(item["degradation_std"] is None for item in payload["compounds"].values())
    assert payload["compounds"]["C2"]["performance_mean"] == 0.0

    prediction = get_pre_race_tyre_prediction(
        season=2026, round_number=18, data_root=tmp_path,
    )
    assert prediction.reference_compound == "MEDIUM"
    assert prediction.compounds["MEDIUM"].performance_delta_to_medium == 0.0
    assert prediction.compounds["HARD"].performance_uncertainty is None
    assert prediction.provenance["prediction_status"] == "prior_only"
    assert prediction.provenance["prediction_provenance"]["independent_of_current_fp"] is True


@pytest.mark.parametrize(
    "allocation",
    [
        {"HARD": "C2", "MEDIUM": "C3"},
        {"HARD": "C2", "MEDIUM": "C2", "SOFT": "C4"},
        {"HARD": "C0", "MEDIUM": "C3", "SOFT": "C4"},
    ],
)
def test_rejects_invalid_allocation(tmp_path: Path, allocation) -> None:
    _source_artifacts(tmp_path)
    with pytest.raises(ValueError):
        write_historical_baseline_prior(
            season=2026, round_number=18, event="United States Grand Prix",
            circuit="Circuit of The Americas", compound_allocation=allocation,
            allocation_source_url=ALLOCATION_URL, data_root=tmp_path,
        )


def test_rejects_source_cutoff_at_or_after_target(tmp_path: Path) -> None:
    _source_artifacts(tmp_path, cutoff=18)
    with pytest.raises(ValueError, match="training cutoff"):
        _write(tmp_path)


def test_rejects_source_event_identity_mismatch(tmp_path: Path) -> None:
    _source_artifacts(tmp_path)
    path = (
        tmp_path / "wostrategy/tyre_prediction/schema_v1/year=2026/"
        "predictions/round=17/model.json"
    )
    payload = json.loads(path.read_text())
    payload["target_round"] = 16
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="event identity"):
        _write(tmp_path)


def test_rejects_unordered_dry_compound_roles(tmp_path: Path) -> None:
    _source_artifacts(tmp_path)
    with pytest.raises(ValueError, match="ordered Pirelli range"):
        write_historical_baseline_prior(
            season=2026, round_number=18, event="United States Grand Prix",
            circuit="Circuit of The Americas",
            compound_allocation={"HARD": "C3", "MEDIUM": "C2", "SOFT": "C4"},
            allocation_source_url=ALLOCATION_URL, data_root=tmp_path,
        )


def test_rejects_descriptor_informed_source_models(tmp_path: Path) -> None:
    _source_artifacts(tmp_path, include_environment=True)
    with pytest.raises(ValueError, match="descriptor-free"):
        _write(tmp_path)


def test_reuses_same_fingerprint_and_refuses_different_overwrite(tmp_path: Path) -> None:
    _source_artifacts(tmp_path)
    path = _write(tmp_path)
    before = path.read_bytes()
    assert _write(tmp_path) == path
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_historical_baseline_prior(
            season=2026, round_number=18, event="Different Event",
            circuit="Circuit of The Americas", compound_allocation=ALLOCATION,
            allocation_source_url=ALLOCATION_URL, data_root=tmp_path,
        )
