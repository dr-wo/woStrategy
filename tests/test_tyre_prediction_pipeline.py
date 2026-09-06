from __future__ import annotations

import json
import pandas as pd

from wostrategy.analysis.tyre_prediction_pipeline import (
    PipelineConfig,
    run_tyre_prediction_pipeline,
)
from wostrategy.model.cross_event_tyre_prediction import ReadinessConfig


ALLOCATIONS = (
    ("C1", "C2", "C3"), ("C2", "C3", "C4"), ("C3", "C4", "C5"),
    ("C1", "C3", "C5"), ("C2", "C4", "C5"), ("C1", "C4", "C5"),
    ("C2", "C3", "C5"), ("C3", "C4", "C5"), ("C2", "C3", "C4"),
)


class RoundExtractor:
    model_name = "fake-vision"

    def __init__(self):
        self.calls = 0

    def extract(self, image_path, *, season, round_number):
        self.calls += 1
        hard, medium, soft = ALLOCATIONS[round_number - 1]
        return {
            "event": f"Event {round_number}", "circuit": f"Circuit {round_number}",
            "tyre_stress": 1 + round_number % 5,
            "asphalt_abrasion": 1 + (round_number * 2) % 5,
            "asphalt_grip": 1 + (round_number * 3) % 5,
            "traction": None, "lateral": None,
            "hard_compound": hard, "medium_compound": medium, "soft_compound": soft,
        }


def test_pipeline_second_run_is_idempotent(tmp_path):
    preview_dir = tmp_path / "pirelli_preview/year=2026"
    preview_dir.mkdir(parents=True)
    for round_number in range(1, 10):
        (preview_dir / f"1920_{round_number:02d}-xx26-preview-en.jpg").write_bytes(
            f"image-{round_number}".encode()
        )
    for round_number in range(1, 9):
        _write_retro(tmp_path, round_number)
    extractor = RoundExtractor()
    config = PipelineConfig(
        readiness=ReadinessConfig(
            minimum_performance_observations=1,
            minimum_degradation_observations=20,
        )
    )

    first = run_tyre_prediction_pipeline(
        season=2026, vision_extractor=extractor, data_root=tmp_path,
        legacy_retro_root=tmp_path / "legacy", config=config,
    )
    assert extractor.calls == 9
    assert first.prediction_candidates == (9,)
    assert len(first.predictions) == 1
    assert first.predictions[0].history_rows_added == 3
    year_root = tmp_path / "wostrategy/tyre_prediction/schema_v1/year=2026"
    tracked = [
        preview_dir / "pirelli_preview_2026.csv",
        year_root / "retro_tyre_observations_2026.csv",
        year_root / "training_data.csv",
        year_root / "prediction_history.csv",
        year_root / "validation.json",
        year_root / "validation_predictions.csv",
        year_root / "coefficient_history.csv",
        year_root / "diagnostic_summary.json",
        year_root / "prospective_validation.csv",
        first.predictions[0].path,
        *sorted((first.predictions[0].path.parent / "versions").glob("*.json")),
        first.fp_calibration_path,
        year_root / "fp_degradation_calibration/versions/through_round=8.json",
    ]
    before = {path: path.read_bytes() for path in tracked}

    second = run_tyre_prediction_pipeline(
        season=2026, vision_extractor=extractor, data_root=tmp_path,
        legacy_retro_root=tmp_path / "legacy", config=config,
    )
    assert extractor.calls == 9
    assert second.retro.unchanged_events == 8
    assert second.predictions[0].history_rows_added == 0
    assert {path: path.read_bytes() for path in tracked} == before
    history = pd.read_csv(year_root / "prediction_history.csv")
    assert len(history) == 3
    prediction = json.loads(first.predictions[0].path.read_text())
    calibration = json.loads(first.fp_calibration_path.read_text())
    assert calibration["completed_retro_through_round"] == 8
    assert calibration["calibration_status"] == "diagnostic_only"
    assert calibration["production_weight_selected"] is False
    assert set(prediction["regression_predictions"]) == {"P0", "P1", "D0", "D1"}
    assert prediction["production_policy"]["performance"]["weighting_policy"] == "uniform"
    assert prediction["production_policy"]["degradation"]["weighting_policy"] == "uniform"
    assert prediction["default_prediction_family"] == "historical_baseline"
    assert prediction["descriptor_boundary_warning"] is True
    assert "tyre_stress" in prediction["boundary_descriptors"]
    for values in prediction["compounds"].values():
        assert set(values["historical_baseline"]) == {"performance", "degradation"}
        assert set(values["pirelli_informed"]) == {"performance", "degradation"}
        assert values["performance_mean"] == values["historical_baseline"]["performance"]
        assert values["degradation_mean"] == values["historical_baseline"]["degradation"]
    assert prediction["validation_context"]["historical_baseline"]["performance"]["RMSE"] is not None
    assert prediction["descriptor_domain"]["performance"]["exact_descriptor_tuple_prior_count"] >= 0


def _write_retro(root, round_number):
    directory = (
        root / "wostrategy/race_performance_review/schema_v1"
        / f"year=2026/round={round_number}/session=R"
    )
    directory.mkdir(parents=True)
    rows = []
    for index, role in enumerate(("HARD", "MEDIUM", "SOFT")):
        performance = -0.3 * index
        degradation = 0.04 * round_number + 0.05 * index
        rows.append({
            "Scope": "global", "Team": None, "Compound": role,
            "ReferenceCompound": "HARD",
            "CompoundDeltaP10Seconds": performance - 0.1,
            "CompoundDeltaMedianSeconds": performance,
            "CompoundDeltaP90Seconds": performance + 0.1,
            "DegradationP10SecondsPerLap": degradation - 0.01,
            "DegradationMedianSecondsPerLap": degradation,
            "DegradationP90SecondsPerLap": degradation + 0.01,
        })
    pd.DataFrame(rows).to_csv(directory / "tyre_information.csv", index=False)
