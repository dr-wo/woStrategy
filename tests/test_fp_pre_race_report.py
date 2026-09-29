from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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


def test_temperature_context_uses_only_persisted_supporting_laps(tmp_path):
    _write_prediction(tmp_path, target_round=9)
    _write_fp2_snapshot(tmp_path, target_round=9)
    fp2_evidence = pd.DataFrame([
        {
            "run_id": "FP2:driver-a:1", "lap_number": 10,
            "compound": "MEDIUM", "air_temp_c": 25.2, "track_temp_c": 0.0,
        },
        {
            "run_id": "FP2:driver-a:1", "lap_number": 11,
            "compound": "MEDIUM", "air_temp_c": None, "track_temp_c": 40.4,
        },
        {
            "run_id": "FP2:driver-b:2", "lap_number": 20,
            "compound": "SOFT", "air_temp_c": 26.0, "track_temp_c": None,
        },
        {
            "run_id": "FP2:driver-b:2", "lap_number": 21,
            "compound": "SOFT", "air_temp_c": 26.4, "track_temp_c": None,
        },
    ])
    excluded_fp3 = pd.DataFrame([{
        "run_id": "FP3:excluded:1", "lap_number": 99,
        "compound": "MEDIUM", "air_temp_c": 99.0, "track_temp_c": 99.0,
    }])

    report = build_fp_tyre_evidence_report(
        season=2026,
        round_number=9,
        data_root=tmp_path,
        temperature_evidence_by_session={"FP2": fp2_evidence, "FP3": excluded_fp3},
        temperature_analysis_id_by_session={"FP2": "fp2", "FP3": "excluded"},
        temperature_model_created_at_by_session={"FP2": "2026-09-26T12:00:00+00:00"},
        temperature_source_by_session={
            "FP2": {"cache_origin": "wodata-cache", "cache_path": "fp2/laps.pkl"}
        },
    )

    by_session = {row["session"]: row for row in report["sessions"]}
    context = by_session["FP2"]["temperature_context"]
    assert context["analysis_id"] == "fp2"
    assert context["selection_stage"] == "post_normalisation_final_model_evidence"
    assert context["coverage"] == {
        "evidence_laps": 4, "air_temp_laps": 3, "track_temp_laps": 2,
    }
    medium = context["compound_contexts"][0]
    assert [row["lap_number"] for row in medium["supporting_laps"]] == [10, 11]
    assert medium["air_temp_c"]["status"] == "partial"
    assert medium["track_temp_c"]["status"] == "available"
    assert medium["track_temp_c"]["minimum"] == 0.0
    assert medium["track_temp_c"]["recorded_zero_observed"] is True
    soft = context["compound_contexts"][1]
    assert soft["air_temp_c"]["status"] == "available"
    assert soft["track_temp_c"]["status"] == "unavailable"
    assert by_session["FP3"]["temperature_context"]["status"] == (
        "unavailable_no_tyre_evidence"
    )
    assert "99.0" not in json.dumps(report)
    rendered = render_fp_tyre_evidence_report(report)
    assert "TrackTemp 0.0–40.4°C (2/2 laps, recorded 0.0°C retained as unusual)" in rendered
    assert "AirTemp 25.2°C (1/2 laps)" in rendered
    json_path, markdown_path = write_fp_tyre_evidence_report(
        season=2026,
        round_number=9,
        data_root=tmp_path,
        temperature_evidence_by_session={"FP2": fp2_evidence},
        temperature_analysis_id_by_session={"FP2": "fp2"},
        temperature_model_created_at_by_session={"FP2": "2026-09-26T12:00:00+00:00"},
        temperature_source_by_session={
            "FP2": {"cache_origin": "wodata-cache", "cache_path": "fp2/laps.pkl"}
        },
    )
    persisted = json.loads(json_path.read_text())
    assert persisted["sessions"][1]["temperature_context"]["coverage"] == (
        context["coverage"]
    )
    assert "recorded 0.0°C retained as unusual" in markdown_path.read_text()


def test_legacy_model_artifact_does_not_reconstruct_temperature(tmp_path):
    _write_prediction(tmp_path, target_round=9)
    _write_fp2_snapshot(tmp_path, target_round=9)
    report = build_fp_tyre_evidence_report(
        season=2026, round_number=9, data_root=tmp_path
    )
    context = report["sessions"][1]["temperature_context"]
    assert context["status"] == "legacy_provenance_unavailable"
    assert "not reconstructed" in context["reason"]


def test_temperature_provenance_from_another_analysis_is_rejected(tmp_path):
    _write_prediction(tmp_path, target_round=9)
    _write_fp2_snapshot(tmp_path, target_round=9)
    report = build_fp_tyre_evidence_report(
        season=2026,
        round_number=9,
        data_root=tmp_path,
        temperature_evidence_by_session={
            "FP2": pd.DataFrame([{
                "run_id": "old:1", "lap_number": 1, "compound": "MEDIUM",
                "air_temp_c": 25.0, "track_temp_c": 40.0,
            }])
        },
        temperature_analysis_id_by_session={"FP2": "older-analysis"},
        temperature_model_created_at_by_session={"FP2": "2026-09-26T12:00:00+00:00"},
    )
    context = report["sessions"][1]["temperature_context"]
    assert context["status"] == "stale_provenance_unavailable"
    assert "compound_contexts" not in context


@pytest.mark.parametrize("missing", ["manifest", "analysis_id", "created_at"])
def test_missing_current_artifact_identity_leaves_temperature_unbound(tmp_path, missing):
    _write_fp2_snapshot(tmp_path, target_round=9)
    path = (tmp_path / "wostrategy/weekend_model/schema_v1/year=2026"
            / "round=9/sessions/FP2/manifest.json")
    if missing == "manifest":
        path.unlink()
    else:
        manifest = json.loads(path.read_text())
        del manifest[missing]
        path.write_text(json.dumps(manifest))
    report = build_fp_tyre_evidence_report(
        season=2026, round_number=9, data_root=tmp_path,
        temperature_evidence_by_session={"FP2": pd.DataFrame([{
            "run_id": "FP2:1", "lap_number": 1, "compound": "MEDIUM",
            "air_temp_c": 25.0, "track_temp_c": 40.0,
        }])},
        temperature_analysis_id_by_session={"FP2": "fp2"},
        temperature_model_created_at_by_session={"FP2": "2026-09-26T12:00:00+00:00"},
    )
    context = report["sessions"][1]["temperature_context"]
    assert context["status"] == "unavailable_unbound_provenance"
    assert "compound_contexts" not in context


def test_same_analysis_from_an_older_artifact_instance_is_rejected(tmp_path):
    _write_fp2_snapshot(tmp_path, target_round=9)
    report = build_fp_tyre_evidence_report(
        season=2026, round_number=9, data_root=tmp_path,
        temperature_evidence_by_session={"FP2": pd.DataFrame([{
            "run_id": "FP2:1", "lap_number": 1, "compound": "MEDIUM",
            "air_temp_c": 25.0, "track_temp_c": 40.0,
        }])},
        temperature_analysis_id_by_session={"FP2": "fp2"},
        temperature_model_created_at_by_session={"FP2": "2026-09-25T12:00:00+00:00"},
    )
    context = report["sessions"][1]["temperature_context"]
    assert context["status"] == "stale_provenance_unavailable"
    assert context["analysis_id"] == context["evidence_analysis_id"]
    assert "compound_contexts" not in context


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
        "information_cutoff": "FP2", "input_fingerprint": "fp2-input",
        "created_at": "2026-09-26T12:00:00+00:00",
    }), encoding="utf-8")
    pd.DataFrame([{
        "parameter": "degradation", "compound": "MEDIUM", "median": 0.10,
        "support_status": "measured", "usable_lap_count": 12,
        "usable_run_count": 2,
    }]).to_csv(directory / "latest_parameters.csv", index=False)
