from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wostrategy.script.push_cool_sequence_evidence import (
    _load_sources,
    validate_evidence_tables,
)


EVIDENCE_DIR = (
    Path(__file__).resolve().parents[1] / "doc" / "push_cool_sequence_evidence"
)


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(EVIDENCE_DIR / name)


def test_persisted_sequence_evidence_has_fixed_scope_and_recomputable_deltas() -> None:
    runs = _read("sequence_runs.csv")
    laps = _read("sequence_laps.csv")
    intervals = _read("push_to_push_intervals.csv")
    exclusions = _read("excluded_candidates.csv")

    validate_evidence_tables(runs, laps, intervals, exclusions)

    assert (len(runs), len(laps), len(intervals)) == (5, 34, 8)
    assert set(runs["Driver"]) == {"HAM", "RUS", "GAS", "TSU", "SAI"}
    assert laps.loc[laps["ManualRole"].eq("push_lap"), "Deleted"].eq(False).all()
    assert set(laps["TrafficEvaluatorVersion"]) == {"unified-traffic-v2"}

    expected_deltas = np.array(
        [-0.767, -0.114, -0.493, -1.660, -0.014, -1.105, -0.481, -1.310]
    )
    assert np.allclose(intervals["ObservedPaceDeltaSeconds"], expected_deltas)


def test_missing_lap_time_remains_distinct_from_elapsed_time() -> None:
    laps = _read("sequence_laps.csv")
    missing = laps["RawLapTimeMissing"].astype(bool)

    assert missing.sum() == 11
    assert laps.loc[missing, "RawLapTime"].isna().all()
    assert laps.loc[missing, "RawLapTimeSeconds"].isna().all()
    assert laps.loc[missing, "ElapsedToNextLapStartSeconds"].notna().sum() == 9


def test_identifiability_snapshot_retains_rank_loss_and_dependence() -> None:
    intervals = _read("push_to_push_intervals.csv")
    sensitivity = _read("identifiability_sensitivity.csv")

    repeated_runs = intervals.groupby("RunId").size()
    assert repeated_runs.loc[repeated_runs.gt(1)].to_dict() == {
        "2026-R14-FP2-GAS-L1-L7": 2,
        "2026-R14-FP2-RUS-L4-L10": 2,
        "2026-R14-FP2-TSU-L1-L8": 2,
    }
    fixed_rho = sensitivity.loc[sensitivity["AnalysisType"].eq("fixed_rho_design")]
    assert fixed_rho.loc[fixed_rho["Rho"].lt(1), "Rank"].eq(3).all()
    assert fixed_rho.loc[fixed_rho["Rho"].eq(1), "Rank"].eq(2).all()
    assert sensitivity["CoolCountTyreAgeCorrelation"].eq(1.0).all()


def test_missing_sources_fail_without_substitution(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no download or sample substitution"):
        _load_sources(tmp_path)
