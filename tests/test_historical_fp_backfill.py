from __future__ import annotations

import json

import pandas as pd

from wodata import get_fastf1_session_laps_path
from wodata.artifacts import weekend_model_root
from wostrategy.analysis.historical_fp_backfill import (
    classify_rounds,
    discover_historical_fp_artifacts,
    inspect_fp_session_artifact,
)
from wostrategy.analysis.traffic import (
    COMBINATION_MODE_COMPATIBILITY_MIN,
    TRAFFIC_EVALUATOR_VERSION,
)


def _write_valid_session(tmp_path, session="FP1", event_format="conventional"):
    raw = get_fastf1_session_laps_path(
        year=2026, round_number=1, session=session, data_root=tmp_path
    )
    raw.parent.mkdir(parents=True)
    laps = pd.DataFrame({
        "EventName": ["Example Grand Prix"],
        "SessionName": [session],
        "EventFormat": [event_format],
    })
    laps.attrs["traffic_evaluator_version"] = TRAFFIC_EVALUATOR_VERSION
    laps.attrs["traffic_combination_mode"] = COMBINATION_MODE_COMPATIBILITY_MIN
    laps.to_pickle(raw)

    artifact = weekend_model_root(2026, 1, tmp_path) / "sessions" / session
    artifact.mkdir(parents=True)
    common = {
        "analysis_id": "analysis-1",
        "input_fingerprint": "input-1",
        "config_hash": "config-1",
    }
    pd.DataFrame([{
        **common, "parameter": "degradation", "compound": "HARD",
        "team": "", "support_status": "measured",
    }]).to_csv(artifact / "latest_parameters.csv", index=False)
    pd.DataFrame([{
        "analysis_id": "analysis-1", "parameter": "degradation",
        "compound": "HARD", "team": "", "support_status": "measured",
    }]).to_csv(artifact / "latest_support.csv", index=False)
    (artifact / "manifest.json").write_text(json.dumps({
        "schema_version": "schema_v1", "analysis_id": "analysis-1",
        "source_scope": session, "input_fingerprint": "input-1",
        "config_hash": "config-1", "created_at": "2026-01-01T00:00:00+00:00",
        "status": "UPDATED", "contributing_sessions": [session],
    }), encoding="utf-8")


def test_discovery_validates_normal_cache_and_artifact_metadata(tmp_path):
    _write_valid_session(tmp_path)
    row = inspect_fp_session_artifact(
        season=2026, round_number=1, session="FP1", data_root=tmp_path
    )
    assert row["raw_data_status"] == "cached"
    assert row["analysis_status"] == "complete"
    assert row["event"] == "Example Grand Prix"
    assert row["supported_compounds"] == "HARD"
    assert row["analysis_id"] == "analysis-1"


def test_discovery_classifies_complete_partial_missing_and_stale(tmp_path):
    for session in ("FP1", "FP2", "FP3"):
        _write_valid_session(tmp_path, session)
    summary = discover_historical_fp_artifacts(
        season=2026, rounds=[1, 2], data_root=tmp_path
    )
    assert classify_rounds(summary) == {1: "complete", 2: "missing"}

    manifest = weekend_model_root(2026, 1, tmp_path) / "sessions/FP2/manifest.json"
    payload = json.loads(manifest.read_text())
    payload["config_hash"] = "different"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    summary = discover_historical_fp_artifacts(
        season=2026, rounds=[1], data_root=tmp_path
    )
    assert classify_rounds(summary) == {1: "invalid/stale"}


def test_discovery_is_read_only_and_does_not_download(tmp_path):
    summary = discover_historical_fp_artifacts(
        season=2026, rounds=[1], data_root=tmp_path
    )
    assert set(summary["raw_data_status"]) == {"missing"}
    assert not (tmp_path / "fastf1").exists()


def test_discovery_records_sprint_only_session_absence(tmp_path):
    _write_valid_session(tmp_path, event_format="sprint_shootout")
    summary = discover_historical_fp_artifacts(
        season=2026, rounds=[1], data_root=tmp_path
    ).set_index("session")
    assert summary.loc["FP2", "raw_data_status"] == "session_absent"
    assert "sprint event" in summary.loc["FP3", "failure_reason"]


def test_discovery_rejects_later_session_contributors(tmp_path):
    _write_valid_session(tmp_path)
    manifest = weekend_model_root(2026, 1, tmp_path) / "sessions/FP1/manifest.json"
    payload = json.loads(manifest.read_text())
    payload["contributing_sessions"] = ["FP1", "FP2"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    row = inspect_fp_session_artifact(
        season=2026, round_number=1, session="FP1", data_root=tmp_path
    )
    assert row["analysis_status"] == "invalid/stale"
    assert "leakage-safe" in row["failure_reason"]
