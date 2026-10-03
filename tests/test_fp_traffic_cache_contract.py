"""Exercise the historical entry points, with source acquisition isolated.

The fixture supplies selected runs rather than re-testing programme selection.
Diagnostic fitting, persisted-model reads, output and calibration reuse are real.
"""
import json

import pandas as pd
import pytest

from wodata import get_fastf1_session_laps_path
from wostrategy.analysis import fp_race_diagnostic as diagnostic
from wostrategy.analysis.traffic import (
    COMBINATION_MODE_COMPATIBILITY_MIN,
    TRAFFIC_EVALUATOR_VERSION,
)
from wostrategy.core.pre_race_session_data import load_cached_weekend_sessions
from wostrategy.analysis.traffic_contract import current_traffic_contract


def _current_derived_fixture(**kwargs):
    frame = pd.DataFrame({"LapNumber": [1]})
    frame.attrs.update(current_traffic_contract())
    return frame


def _files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def event(tmp_path, monkeypatch):
    model_root = tmp_path / "wostrategy/weekend_model/schema_v1/year=2026/round=11"
    saved = model_root / "sessions/FP1"
    saved.mkdir(parents=True)
    pd.DataFrame([
        {"parameter": "degradation", "compound": compound, "median": value}
        for compound, value in [("HARD", .1), ("MEDIUM", .2)]
    ]).to_csv(saved / "latest_parameters.csv", index=False)
    (saved / "manifest.json").write_text(json.dumps({"contributing_sessions": ["FP1"]}))
    year_root = diagnostic.tyre_prediction_year_root(2026, tmp_path)
    year_root.mkdir(parents=True)
    (year_root / "validation_predictions.csv").write_text("fixture acquisition marker\n")
    prior = {
        family: {"degradation": {"HARD": .1, "MEDIUM": .2}, "performance": {"MEDIUM": -.3}}
        for family in ("historical_baseline", "pirelli_informed")
    }
    monkeypatch.setattr(diagnostic, "_read_priors_and_targets", lambda *args: (
        prior, {"HARD": .12, "MEDIUM": .22}, {"MEDIUM": -.3}, "Fixture GP",
        year_root / "validation_predictions.csv",
    ))
    calls = []

    def evidence(frames, **kwargs):
        calls.append(frames["FP1"].copy())
        return {"longrun_laps": [
            {"session": "FP1", "team": "Team", "driver": "AAA", "run_id": stint,
             "compound": compound, "stint": stint, "lap_number": stint * 10 + age,
             "tyre_age": age, "lap_time_s": 80.0 + slope * age,
             "used_for_quantitative_inference": True}
            for stint, compound, slope in [(1, "HARD", .1), (2, "MEDIUM", .2)]
            for age in range(1, 6)
        ]}

    monkeypatch.setattr(diagnostic, "analyse_fp_tyre_evidence_sessions", evidence)
    monkeypatch.setattr(diagnostic, "normal_model_coordinates", lambda *args: pd.DataFrame())
    path = get_fastf1_session_laps_path(
        year=2026, round_number=11, session="FP1", data_root=tmp_path,
    )
    path.parent.mkdir(parents=True)
    return tmp_path, model_root, path, calls


def _run(entry, root):
    if entry == "diagnostic":
        return diagnostic.run_historical_fp_race_diagnostic(season=2026, data_root=root)
    return diagnostic.refresh_fp_degradation_calibration(
        season=2026, completed_retro_through_round=11, data_root=root,
    )


@pytest.mark.parametrize("entry", ["diagnostic", "calibration"])
@pytest.mark.parametrize("version,mode", [
    ("unified-traffic-v2", COMBINATION_MODE_COMPATIBILITY_MIN),
    ("unified-traffic-v3", COMBINATION_MODE_COMPATIBILITY_MIN),
    (TRAFFIC_EVALUATOR_VERSION, "incompatible_mode"),
    (None, COMBINATION_MODE_COMPATIBILITY_MIN),
    (TRAFFIC_EVALUATOR_VERSION, None),
])
def test_direct_fp_entry_rejects_incompatible_cache_before_analysis(event, entry, version, mode):
    root, _, path, calls = event
    frame = pd.DataFrame({"LapNumber": [1]})
    frame.attrs.update(traffic_evaluator_version=version, traffic_combination_mode=mode)
    frame.to_pickle(path)
    before = _files(root)
    with pytest.raises(ValueError, match="Incompatible FP traffic cache"):
        _run(entry, root)
    assert calls == []
    assert _files(root) == before  # Cache, saved models and existing outputs untouched.


@pytest.mark.parametrize("entry", ["diagnostic", "calibration"])
def test_current_fp_cache_fresh_and_repeated_analysis_match_without_model_writes(event, entry):
    root, model_root, path, calls = event
    saved_models = _files(model_root)
    frozen = diagnostic.calibration_snapshot_root(2026, root) / "versions/through_round=11.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text('{"historical_calibration": "do not rewrite"}')
    frozen_bytes = frozen.read_bytes()
    loaded = load_cached_weekend_sessions(
        year=2026, round_number=11, sessions=("FP1",), data_root=root,
        session_loader=_current_derived_fixture,
    )
    assert loaded.sources["FP1"] == "fastf1-download"
    cache_bytes = path.read_bytes()
    _run(entry, root)
    first = _files(diagnostic.diagnostic_root(2026, root))
    summary = json.loads(first["fp_race_diagnostic_summary.json"])
    assert summary["enriched_lap_traffic_contract"]["traffic_evaluator_version"] == TRAFFIC_EVALUATOR_VERSION
    assert "not refitted" in summary["saved_fp_parameter_semantics"]
    assert str(path) in summary["provenance_sha256"]
    history = pd.read_csv(diagnostic.diagnostic_root(2026, root) / "fp_degradation_calibration_history.csv")
    assert not history.empty
    assert history.traffic_evaluator_version.eq(TRAFFIC_EVALUATOR_VERSION).all()

    def unexpected_refresh(**kwargs):
        pytest.fail("Compatible cache must not refresh")

    cached = load_cached_weekend_sessions(
        year=2026, round_number=11, sessions=("FP1",), data_root=root,
        session_loader=unexpected_refresh,
    )
    assert cached.sources["FP1"] == "wodata-cache"
    _run(entry, root)
    assert _files(diagnostic.diagnostic_root(2026, root)) == first
    pd.testing.assert_frame_equal(calls[0], calls[1])
    pd.testing.assert_frame_equal(calls[0], cached.sessions["FP1"])

    # Without the acquisition prerequisite, refresh reuses the derived history.
    validation = diagnostic.tyre_prediction_year_root(2026, root) / "validation_predictions.csv"
    validation.unlink()
    latest = diagnostic.refresh_fp_degradation_calibration(
        season=2026, completed_retro_through_round=11, data_root=root,
    )
    first_snapshot = latest.read_bytes()
    diagnostic.refresh_fp_degradation_calibration(
        season=2026, completed_retro_through_round=11, data_root=root,
    )
    assert latest.read_bytes() == first_snapshot
    assert len(calls) == 2
    assert _files(model_root) == saved_models
    assert path.read_bytes() == cache_bytes
    assert frozen.read_bytes() == frozen_bytes


@pytest.mark.parametrize("version", [None, "unified-traffic-v2", "unified-traffic-v3"])
def test_old_calibration_history_cannot_bypass_fp_input_contract(event, version):
    root, _, _, _ = event
    load_cached_weekend_sessions(
        year=2026, round_number=11, sessions=("FP1",), data_root=root,
        session_loader=_current_derived_fixture,
    )
    _run("calibration", root)
    (diagnostic.tyre_prediction_year_root(2026, root) / "validation_predictions.csv").unlink()
    output = diagnostic.diagnostic_root(2026, root)
    history = pd.read_csv(output / "fp_degradation_calibration_history.csv").drop(
        columns=["traffic_evaluator_version", "traffic_combination_mode"], errors="ignore",
    )
    if version is not None:
        history["traffic_evaluator_version"] = version
        history["traffic_combination_mode"] = COMBINATION_MODE_COMPATIBILITY_MIN
    history.to_csv(output / "fp_degradation_calibration_history.csv", index=False)
    before = _files(root)
    with pytest.raises(ValueError, match="Incompatible FP traffic calibration history"):
        _run("calibration", root)
    assert _files(root) == before
