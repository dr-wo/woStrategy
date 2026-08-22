from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wodata.pirelli_preview import OpenAIPreviewVisionExtractor

from wostrategy.analysis.tyre_prediction_pipeline import (
    PipelineConfig,
    run_tyre_prediction_pipeline,
)
from wostrategy.model.cross_event_tyre_prediction import ReadinessConfig


def main() -> None:
    args = parse_args()
    extractor = None
    if args.vision_model:
        extractor = OpenAIPreviewVisionExtractor(model=args.vision_model)
    report = run_tyre_prediction_pipeline(
        season=args.year,
        vision_extractor=extractor,
        data_root=args.data_root,
        legacy_retro_root=args.legacy_retro_root,
        config=PipelineConfig(
            performance_ridge_alpha=args.performance_ridge_alpha,
            degradation_ridge_alpha=args.degradation_ridge_alpha,
            readiness=ReadinessConfig(
                minimum_performance_observations=args.minimum_performance_observations,
                minimum_degradation_observations=args.minimum_degradation_observations,
            ),
        ),
    )
    print(json.dumps(_report_dict(report), indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update Pirelli previews, extract Retro tyre history, and predict tyres."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--legacy-retro-root", type=Path)
    parser.add_argument(
        "--vision-model", default=os.environ.get("WODATA_PIRELLI_VISION_MODEL")
    )
    parser.add_argument("--performance-ridge-alpha", type=float, default=0.0)
    parser.add_argument("--degradation-ridge-alpha", type=float, default=0.0)
    parser.add_argument("--minimum-performance-observations", type=int, default=8)
    parser.add_argument("--minimum-degradation-observations", type=int, default=12)
    return parser.parse_args()


def _report_dict(report) -> dict[str, object]:
    return {
        "preview": {
            **report.preview.__dict__,
            "csv_path": str(report.preview.csv_path),
        },
        "retro": {
            **report.retro.__dict__,
            "csv_path": str(report.retro.csv_path),
        },
        "prediction_candidates": list(report.prediction_candidates),
        "predictions": [
            {
                "round": item.round_number,
                "path": str(item.path),
                "history_rows_added": item.history_rows_added,
            }
            for item in report.predictions
        ],
        "readiness_paths": [str(path) for path in report.readiness_paths],
        "training_data_path": str(report.training_data_path),
        "validation_path": str(report.validation_path),
        "validation_predictions_path": str(report.validation_predictions_path),
        "coefficient_history_path": str(report.coefficient_history_path),
        "diagnostic_summary_path": str(report.diagnostic_summary_path),
        "prospective_validation_path": str(report.prospective_validation_path),
    }


if __name__ == "__main__":
    main()
