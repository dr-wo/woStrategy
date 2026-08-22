from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wostrategy.analysis.fp_tyre_evidence import (
    FPTyreEvidenceCalculationConfig,
    analyse_fp_tyre_evidence_sessions,
    calculate_fp_tyre_evidence,
    load_fp_tyre_evidence,
)


R11_CACHE = (
    Path(__file__).resolve().parents[2]
    / "woData/fastf1/session_laps/schema_v1/year=2026/round=11"
)


def r11_sessions():
    if not R11_CACHE.is_dir():
        pytest.skip("Local 2026 R11 FP cache is unavailable")
    return {
        session: pd.read_pickle(R11_CACHE / f"session={session}/laps.pkl")
        for session in ("FP1", "FP2", "FP3")
    }


def test_r11_strict_longrun_contrast_regression_and_programme_exclusion():
    evidence = analyse_fp_tyre_evidence_sessions(
        r11_sessions(), year=2026, round_number=11
    )

    assert evidence["longrun_support"]["strict_longrun_runs"] == 8
    assert evidence["longrun_support"]["strict_clean_laps"] == 58
    assert evidence["identifiability"]["rank"] == 3
    assert evidence["identifiability"]["nullity"] == 1
    contrasts = {
        row["parameter"]: row for row in evidence["degradation_contrasts"]
    }
    assert contrasts["Delta_HM"]["strict_central"] == pytest.approx(-0.056086, abs=5e-4)
    assert contrasts["Delta_SM"]["strict_central"] == pytest.approx(-0.189669, abs=5e-4)
    assert {row["support_status"] for row in evidence["performance_support"]} == {
        "unsupported"
    }
    used = [row for row in evidence["longrun_laps"] if row["used_for_quantitative_inference"]]
    assert used
    assert {row["programme"] for row in used} == {"LongRun"}
    assert not {"Setting", "QualiSim", "Uncertain"}.intersection(
        row["programme"] for row in used
    )


def test_partial_weekend_records_only_sessions_used():
    sessions = r11_sessions()
    evidence = analyse_fp_tyre_evidence_sessions(
        {"FP1": sessions["FP1"], "FP2": sessions["FP2"]},
        year=2026,
        round_number=11,
    )

    assert evidence["sessions_used"] == ["FP1", "FP2"]
    assert "FP3" not in evidence["sessions_used"]


def test_calculate_is_explicit_and_cache_load_is_read_only(tmp_path):
    sessions = r11_sessions()
    calls = []

    def loader(*, session, **_kwargs):
        calls.append(session)
        return sessions[session]

    assert load_fp_tyre_evidence(2026, 11, tmp_path) is None
    assert calls == []

    calculated = calculate_fp_tyre_evidence(
        year=2026,
        round_number=11,
        data_root=tmp_path,
        config=FPTyreEvidenceCalculationConfig(sessions=("FP1", "FP2")),
        session_loader=loader,
    )

    assert calls == ["FP1", "FP2"]
    assert calculated["sessions_used"] == ["FP1", "FP2"]
    cached = load_fp_tyre_evidence(2026, 11, tmp_path)
    assert cached["analysis_id"] == calculated["analysis_id"]
    assert calls == ["FP1", "FP2"]

