from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from wostrategy.analysis import fp_tyre_evidence as evidence
from wostrategy.analysis.traffic_contract import current_traffic_contract
from wostrategy.core import session_loader
from wostrategy.core.pre_race_session_data import load_cached_weekend_sessions


def current_laps():
    """Representative three-compound fixture with explicitly current derived gaps."""
    rows = []
    for stint, compound, slope in [(1, "HARD", .1), (2, "MEDIUM", .2), (3, "SOFT", .3)]:
        for age in range(1, 11):
            lap = (stint - 1) * 10 + age
            rows.append(dict(
                Driver="AAA", Team="Team", LapNumber=lap, Stint=stint,
                Compound=compound, TyreLife=age, TrackStatus="1",
                LapTime=pd.Timedelta(seconds=80 + slope * age),
                Time=pd.Timedelta(seconds=lap * 90), PitOutTime=pd.NaT, PitInTime=pd.NaT,
                MeanTimeDeltaToDriverAhead=5., MeanTimeDeltaToDriverBehind=5.,
                TrafficStatus="ok",
            ))
    frame = pd.DataFrame(rows)
    frame = pd.concat([frame, frame.assign(Driver="BBB", Team="Other")], ignore_index=True)
    frame.attrs.update(current_traffic_contract())
    return frame


def incompatible(kind):
    frame = current_laps()
    if kind in ("v2", "v3"):
        frame.attrs["traffic_evaluator_version"] = "unified-traffic-" + kind
    elif kind == "missing":
        frame.attrs.clear()
    elif kind == "mode":
        frame.attrs["traffic_combination_mode"] = "wrong-mode"
    else:
        frame["TrafficEvaluatorVersion"] = "unified-traffic-v3"
    return frame


@pytest.mark.parametrize("kind", ["v2", "v3", "missing", "mode", "mixed_rows"])
def test_materialized_contract_rejects_before_selection(kind, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Incompatible materialized input reached quantitative preparation")
    monkeypatch.setattr(evidence, "_prepare_runs", forbidden)
    with pytest.raises(ValueError, match="Incompatible FP traffic session FP2"):
        evidence.analyse_fp_tyre_evidence_sessions(
            {"FP2": incompatible(kind)}, year=2026, round_number=11,
        )


def test_mixed_sessions_rejected_before_any_selection(monkeypatch):
    monkeypatch.setattr(evidence, "_prepare_runs", lambda *_: pytest.fail("selection ran"))
    with pytest.raises(ValueError, match="session FP3"):
        evidence.analyse_fp_tyre_evidence_sessions(
            {"FP2": current_laps(), "FP3": incompatible("v3")}, year=2026, round_number=11,
        )


def test_current_materialized_calculation_and_empty_behavior():
    kwargs = dict(year=2026, round_number=11, calculated_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    frame = current_laps()
    before = frame.copy(deep=True)
    first = evidence.analyse_fp_tyre_evidence_sessions({"FP2": frame}, **kwargs)
    second = evidence.analyse_fp_tyre_evidence_sessions(
        {"FP2": frame, "FP3": pd.DataFrame(), "FP1": None}, **kwargs,
    )
    assert first == second
    assert first["longrun_support"]["strict_longrun_runs"] == 6
    assert first["longrun_support"]["strict_clean_laps"] == 60
    assert len(first["degradation_contrasts"]) == 3
    assert first["session_traffic_contracts"]["FP2"] == current_traffic_contract()
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(ValueError, match="At least one non-empty"):
        evidence.analyse_fp_tyre_evidence_sessions({"FP1": pd.DataFrame()}, **kwargs)
    with pytest.raises(ValueError, match="No strict LongRun"):
        evidence.analyse_fp_tyre_evidence_sessions(
            {"FP2": frame.assign(MeanTimeDeltaToDriverAhead=0.0)}, **kwargs,
        )


def test_single_compound_support_returns_unidentified_contrasts_without_crashing():
    frame = current_laps().loc[lambda rows: rows["Compound"].eq("MEDIUM")].copy()
    result = evidence.analyse_fp_tyre_evidence_sessions(
        {"FP2": frame}, year=2026, round_number=11,
    )

    assert result["longrun_support"]["strict_longrun_runs"] > 0
    assert len(result["degradation_contrasts"]) == 3
    assert {row["status"] for row in result["degradation_contrasts"]} == {"unsupported"}
    assert all(row["strict_central"] is None for row in result["degradation_contrasts"])
    assert result["identifiability"]["rank"] < 3
    assert result["identifiability"]["nullity"] > 1


@pytest.mark.parametrize("kind", ["v2", "v3", "missing", "mode", "mixed_rows"])
def test_injected_loader_cannot_relabel_or_persist_incompatible_laps(tmp_path, kind):
    frame = incompatible(kind)
    before = frame.copy(deep=True)
    # An existing evidence artefact must survive the failed explicit calculation.
    saved = evidence.fp_tyre_evidence_path(2026, 11, tmp_path)
    saved.parent.mkdir(parents=True)
    saved.write_text('{"historical": true}')
    with pytest.raises(ValueError, match="Incompatible FP traffic loader output for FP2"):
        evidence.calculate_fp_tyre_evidence(
            year=2026, round_number=11, data_root=tmp_path,
            config=evidence.FPTyreEvidenceCalculationConfig(sessions=("FP2",)),
            session_loader=lambda **_: frame,
        )
    assert not list(tmp_path.rglob("*.pkl"))
    assert saved.read_text() == '{"historical": true}'
    pd.testing.assert_frame_equal(frame, before)
    assert frame.attrs == before.attrs


def test_current_injected_loader_preserves_cache_contents_and_repeats(tmp_path):
    frame = current_laps()
    config = evidence.FPTyreEvidenceCalculationConfig(sessions=("FP2",))
    first = evidence.calculate_fp_tyre_evidence(
        year=2026, round_number=11, data_root=tmp_path, config=config,
        session_loader=lambda **_: frame,
    )
    path, = tmp_path.rglob("laps.pkl")
    cached = pd.read_pickle(path)
    pd.testing.assert_frame_equal(cached, frame)
    assert cached.attrs == frame.attrs
    def unexpected(**_):
        pytest.fail("Compatible cached laps unexpectedly refreshed")
    second = evidence.calculate_fp_tyre_evidence(
        year=2026, round_number=11, data_root=tmp_path, config=config, session_loader=unexpected,
    )
    assert first["longrun_laps"] == second["longrun_laps"]
    assert first["degradation_contrasts"] == second["degradation_contrasts"]
    assert first["input_fingerprint"] == second["input_fingerprint"]
    assert second["session_traffic_contracts"]["FP2"] == frame.attrs
    assert evidence.load_fp_tyre_evidence(2026, 11, tmp_path) == second


@pytest.mark.parametrize("fail", [False, True])
def test_derivation_certifies_only_after_recomputing_gaps(tmp_path, monkeypatch, fail):
    # Stale source-lap metrics cannot survive a successful genuine aggregation.
    raw = pd.DataFrame({"Driver": ["A", "B"], "LapNumber": [10, 9],
                        "MeanTimeDeltaToDriverBehind": [999., 999.]})
    raw.attrs.update(traffic_evaluator_version="unified-traffic-v2",
                     traffic_combination_mode="compatibility_min")
    telemetry = pd.DataFrame([
        dict(Year=2026, Round=11, SessionName="FP2", Driver=d, DriverNumber=n,
             LapNumber=lap, SessionTime=t, DriverAhead=a,
             TimeDeltaToDriverAhead=gap, DistanceToDriverAhead=gap * 50)
        for t in [0., .25, .5]
        for d, n, lap, a, gap in [("A", "1", 10, "3", 5.), ("B", "2", 9, "1", .5)]
    ])
    def raw_telemetry(*args, **kwargs):
        if fail:
            raise ValueError("telemetry unavailable")
        return telemetry
    monkeypatch.setattr(session_loader, "load_or_cache_session_telemetry", raw_telemetry)
    # The actual summarize_lap_gap_metrics function runs; it is not stubbed.
    def derive(**_):
        return session_loader.load_session_laps_with_telemetry_gap_summary(
            year=2026, rounds=[11], session_names=["FP2"],
            session_factory=lambda *_: SimpleNamespace(laps=raw),
        )
    if fail:
        with pytest.raises(ValueError, match="Incompatible FP traffic loader output"):
            load_cached_weekend_sessions(
                year=2026, round_number=11, sessions=("FP2",), data_root=tmp_path,
                session_loader=derive,
            )
        assert not list(tmp_path.rglob("laps.pkl"))
    else:
        loaded = load_cached_weekend_sessions(
            year=2026, round_number=11, sessions=("FP2",), data_root=tmp_path,
            session_loader=derive,
        )
        cached = pd.read_pickle(loaded.cache_paths["FP2"])
        assert cached.attrs == current_traffic_contract()
        assert cached.loc[cached.Driver.eq("A"), "MeanTimeDeltaToDriverBehind"].item() == .5
        assert not cached.MeanTimeDeltaToDriverBehind.eq(999.).any()
    assert raw.attrs["traffic_evaluator_version"] == "unified-traffic-v2"
    assert raw.MeanTimeDeltaToDriverBehind.tolist() == [999., 999.]
