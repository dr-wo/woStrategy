from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from wostrategy.analysis.fp_pre_race_report import (
    build_fp_tyre_evidence_report,
    load_causal_calibration_snapshot,
    render_fp_tyre_evidence_report,
    write_fp_tyre_evidence_report,
)
from wostrategy.analysis.fp_race_diagnostic import (
    build_latest_calibration_snapshot,
    refresh_fp_degradation_calibration,
)


def test_report_handles_missing_sessions_and_renders_diagnostic_limitations(tmp_path):
    _write_prediction(tmp_path, target_round=9)
    prediction_path = (
        tmp_path / "wostrategy/tyre_prediction/schema_v1/year=2026"
        / "predictions/round=9/prediction.json"
    )
    frozen_prediction = prediction_path.read_bytes()
    _write_calibration(tmp_path, through_round=8)
    _write_fp2_snapshot(tmp_path, target_round=9)
    report = build_fp_tyre_evidence_report(
        season=2026,
        round_number=9,
        data_root=tmp_path,
        supported_compounds_by_session={"FP2": ("MEDIUM",)},
    )
    by_session = {row["session"]: row for row in report["sessions"]}
    assert by_session["FP1"]["status"] == "unavailable"
    assert "sprint-format" in by_session["FP1"]["reason"]
    assert by_session["FP2"]["status"] == "available"
    assert by_session["FP3"]["status"] == "unavailable"
    comparisons = by_session["FP2"]["compounds"][0]["prior_comparisons"]
    assert {row["prior_family"] for row in comparisons} == {
        "historical_baseline", "pirelli_informed"
    }
    assert all(row["calibration_status"] == "diagnostic_only" for row in comparisons)
    assert report["production_strategy_inputs_modified"] is False
    rendered = render_fp_tyre_evidence_report(report)
    assert "## FP Tyre Evidence" in rendered
    assert "diagnostic only" in rendered
    assert "leave-one-event-out" in rendered
    assert "do not modify Strategy Prediction" in rendered
    assert "Historical degradation" in rendered
    assert "Pirelli-informed degradation" in rendered
    assert "| historical_baseline |" not in rendered
    assert "| pirelli_informed |" not in rendered
    json_path, markdown_path = write_fp_tyre_evidence_report(
        season=2026, round_number=9, data_root=tmp_path,
        supported_compounds_by_session={"FP2": ("MEDIUM",)},
    )
    assert json_path.is_file() and markdown_path.is_file()
    assert prediction_path.read_bytes() == frozen_prediction


def test_report_without_calibration_does_not_fabricate_update(tmp_path):
    _write_prediction(tmp_path, target_round=9)
    _write_fp2_snapshot(tmp_path, target_round=9)
    report = build_fp_tyre_evidence_report(
        season=2026, round_number=9, data_root=tmp_path
    )
    comparison = report["sessions"][1]["compounds"][0]["prior_comparisons"][0]
    assert comparison["diagnostic_candidate_weight"] is None
    assert comparison["diagnostic_partial_update"] is None
    assert comparison["calibration_status"] == "unavailable"
    assert "No causal calibration history" in render_fp_tyre_evidence_report(report)


def test_race_n_refresh_does_not_rewrite_n_and_n_plus_one_reads_n(tmp_path):
    _write_prediction(tmp_path, target_round=8)
    frozen = (
        tmp_path / "wostrategy/tyre_prediction/schema_v1/year=2026"
        / "predictions/round=8/prediction.json"
    )
    before = frozen.read_bytes()
    latest = refresh_fp_degradation_calibration(
        season=2026,
        completed_retro_through_round=8,
        data_root=tmp_path,
    )
    assert frozen.read_bytes() == before
    snapshot = json.loads(latest.read_text())
    assert snapshot["completed_retro_through_round"] == 8
    consumed, path = load_causal_calibration_snapshot(
        season=2026, target_round=9, data_root=tmp_path
    )
    assert consumed["completed_retro_through_round"] == 8
    assert path.name == "through_round=8.json"
    unavailable, _ = load_causal_calibration_snapshot(
        season=2026, target_round=8, data_root=tmp_path
    )
    assert unavailable is None


def test_latest_snapshot_keeps_fp_sessions_as_independent_tracks():
    rows = []
    for session, preferred in (("FP1", 0.0), ("FP2", 0.5), ("FP3", 0.25)):
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            rows.append({
                "session": session,
                "prior_family": "historical_baseline",
                "update_type": "direct",
                "trained_through_round": 8,
                "training_rounds": "[7, 8]",
                "event_count": 2,
                "prediction_count": 4,
                "candidate_weights": "[0.0, 0.25, 0.5, 0.75, 1.0]",
                "candidate_weight": weight,
                "mae": 0.1, "rmse": 0.2, "bias": -0.1,
                "mean_event_mae": 0.1,
                "loeo_selection_count": 1 if weight == preferred else 0,
                "diagnostic_grid_preferred_weight": preferred,
                "is_diagnostic_grid_preferred": weight == preferred,
                "loeo_modal_weight": preferred,
                "loeo_modal_share": 0.5,
                "loeo_selected_weights": json.dumps([preferred]),
                "loeo_stability": "unstable",
                "calibration_status": "diagnostic_only",
                "calibration_version": "calibration_v1",
                "calibration_method": "fixed_grid_event_mean_mae_with_loeo_v1",
            })
    snapshot = build_latest_calibration_snapshot(
        pd.DataFrame(rows), season=2026, completed_retro_through_round=8
    )
    assert {
        row["session"]: row["diagnostic_grid_preferred_weight"]
        for row in snapshot["tracks"]
    } == {"FP1": 0.0, "FP2": 0.5, "FP3": 0.25}


def test_status_document_references_repository_paths():
    repository = Path(__file__).resolve().parents[1]
    document = repository / "doc/FP_TYRE_ESTIMATION_STATUS_AND_ROADMAP.md"
    text = document.read_text(encoding="utf-8")
    for relative in (
        "src/wostrategy/analysis/fp_race_diagnostic.py",
        "src/wostrategy/analysis/fp_pre_race_report.py",
    ):
        assert relative in text
        assert (repository / relative).is_file()


def _write_prediction(root, *, target_round):
    path = (
        root / "wostrategy/tyre_prediction/schema_v1/year=2026"
        / f"predictions/round={target_round}/prediction.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": "schema_v1", "season": 2026, "round": target_round,
        "event": f"Event {target_round}",
        "prediction_timestamp": "2026-01-01T00:00:00+00:00",
        "prediction_fingerprint": "frozen", "model": "test",
        "compound_allocation": {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"},
        "compounds": {
            "C3": {"compound_role": "HARD", "historical_baseline": {"degradation": 0.05}, "pirelli_informed": {"degradation": 0.06}},
            "C4": {"compound_role": "MEDIUM", "historical_baseline": {"degradation": 0.08}, "pirelli_informed": {"degradation": 0.09}},
            "C5": {"compound_role": "SOFT", "historical_baseline": {"degradation": 0.12}, "pirelli_informed": {"degradation": 0.13}},
        },
    }), encoding="utf-8")


def _write_calibration(root, *, through_round):
    destination = (
        root / "wostrategy/tyre_prediction/schema_v1/year=2026"
        / "fp_degradation_calibration/versions"
        / f"through_round={through_round}.json"
    )
    destination.parent.mkdir(parents=True)
    tracks = []
    for family in ("historical_baseline", "pirelli_informed"):
        tracks.append({
            "session": "FP2", "prior_family": family, "update_type": "direct",
            "calibration_status": "diagnostic_only", "trained_through_round": through_round,
            "event_count": 3, "diagnostic_grid_preferred_weight": 0.5,
            "loeo_stability": "unstable",
        })
    destination.write_text(json.dumps({
        "season": 2026, "completed_retro_through_round": through_round,
        "calibration_version": "calibration_v1", "tracks": tracks,
    }), encoding="utf-8")


def _write_fp2_snapshot(root, *, target_round):
    directory = (
        root / "wostrategy/weekend_model/schema_v1/year=2026"
        / f"round={target_round}/sessions/FP2"
    )
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({
        "analysis_id": "fp2", "included_sessions": ["FP2"],
        "information_cutoff": "FP2",
    }), encoding="utf-8")
    pd.DataFrame([{
        "parameter": "degradation", "compound": "MEDIUM", "median": 0.10,
        "support_status": "measured", "usable_lap_count": 12,
        "usable_run_count": 2,
    }]).to_csv(directory / "latest_parameters.csv", index=False)
