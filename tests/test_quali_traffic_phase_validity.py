"""Anchors, not distance fractions, establish qualifying circular phase."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wostrategy.analysis.traffic import (
    TrafficCoveragePolicy, TrafficEvaluationConfig, _prepare_progress,
    evaluate_physical_traffic, validated_complete_lap_span,
)


CONFIG = TrafficEvaluationConfig(
    1000., require_phase_validity=True,
    coverage_policy=TrafficCoveragePolicy(require_behind=False),
)


def traversal(driver, start, lap=1):
    times = np.arange(start, start + 10.01, .5)
    frame = pd.DataFrame(dict(Driver=driver, SessionTime=times, LapNumber=lap,
                              Speed=360., CandidateEligible=True))
    frame["LapStartTime"] = np.nan
    frame["LapEndTime"] = np.nan
    frame.loc[frame.index[0], "LapStartTime"] = start
    frame.loc[frame.index[-1], "LapEndTime"] = start + 10
    return frame


def evaluate(b=None, a=None, config=CONFIG):
    frame = pd.concat([traversal("A", 0) if a is None else a,
                       traversal("B", -1) if b is None else b,
                       traversal("C", -2)], ignore_index=True)
    return evaluate_physical_traffic(frame, config=config, target_lap=("A", 1))


def sample(result, time=5):
    return result.samples.loc[result.samples.SessionTime.eq(time)].iloc[0]


def assert_unavailable(result):
    lap = result.laps.iloc[0]
    assert lap.TrafficStatus == "unknown"
    assert lap.TrafficStatusReason == "physical phase unavailable"
    assert pd.isna(lap.PhysicalMeanTimeDeltaToDriverAhead)


def test_complete_anchored_geometry_is_numerically_unchanged():
    fixed = evaluate()
    legacy = evaluate(config=replace(CONFIG, require_phase_validity=False))
    columns = [c for c in legacy.samples if c.startswith("Physical")]
    pd.testing.assert_frame_equal(fixed.samples[columns], legacy.samples[columns])
    pd.testing.assert_frame_equal(fixed.laps, legacy.laps)
    assert sample(fixed).PhysicalDriverAhead == "B"
    assert sample(fixed).PhysicalDistanceToDriverAhead == 100.


@pytest.mark.parametrize("case", ["tail", "prefix", "one_tail_row", "one_prefix_row",
                                  "missing_anchors", "distance_is_not_an_anchor",
                                  "gap", "missing_speed", "pit", "outlap"])
def test_invalid_competitor_cannot_disappear_or_manufacture_clean_air(case):
    b = traversal("B", -1)
    if case == "tail":
        b = b[b.SessionTime <= 6]
    elif case == "prefix":
        b = b[b.SessionTime >= 3]
    elif case == "one_tail_row":
        b = b.iloc[:-1]
    elif case == "one_prefix_row":
        b = b.iloc[1:]
    elif case in {"missing_anchors", "distance_is_not_an_anchor"}:
        b = b.drop(columns=["LapStartTime", "LapEndTime"])
        if case == "distance_is_not_an_anchor":
            b["Distance"] = (b.SessionTime + 1) * 100
    elif case == "gap":
        b = b[~b.SessionTime.between(1, 2)]
    elif case == "missing_speed":
        b.loc[b.SessionTime.eq(2), "Speed"] = np.nan
    elif case == "pit":
        b.loc[b.SessionTime.eq(2), "CandidateEligible"] = False
    elif case == "outlap":
        b["PhaseRouteValid"] = False
    result = evaluate(b)
    assert_unavailable(result)
    assert pd.isna(sample(result).PhysicalDriverAhead)
    assert pd.isna(sample(result).PhysicalDistanceToDriverAhead)


def test_invalid_target_and_anchored_partial_without_offset_are_unavailable():
    a = traversal("A", 0)
    assert_unavailable(evaluate(a=a.iloc[1:]))
    # Start anchor alone cannot justify scaling even at 95% of a circuit.
    a.loc[a.index[-1], "LapEndTime"] = np.nan
    assert_unavailable(evaluate(a=a))


def test_explicitly_off_track_invalid_candidate_is_irrelevant():
    b = traversal("B", -1).iloc[:-1].copy()
    b["CandidateEligible"] = False
    result = evaluate(b)
    assert result.laps.iloc[0].TrafficStatus == "ok"
    assert sample(result).PhysicalDriverAhead == "C"
    assert sample(result).PhysicalDistanceToDriverAhead == 200.


def test_single_uncertain_on_track_sample_vetoes_covered_lap_mean():
    b = traversal("B", -1).iloc[:-1].copy()
    b["CandidateEligible"] = False
    b.loc[b.SessionTime.eq(5), "CandidateEligible"] = True
    result = evaluate(b)
    assert result.laps.iloc[0].AheadCoverageFraction > .5
    assert_unavailable(result)


def test_boundary_interpolation_requires_both_spans_valid():
    # B crosses the timing line between A's timestamps, at 4.25 seconds.
    b = pd.concat([traversal("B", -5.75), traversal("B", 4.25, lap=2)], ignore_index=True)
    def boundary_result():
        return evaluate_physical_traffic(pd.concat([traversal("A", 0), b], ignore_index=True),
                                         config=CONFIG, target_lap=("A", 1))
    result = boundary_result()
    assert sample(result, 4).PhysicalDistanceToDriverAhead == pytest.approx(575)
    assert sample(result, 4.5).PhysicalDistanceToDriverAhead == pytest.approx(575)
    b.loc[b.LapNumber.eq(2), "LapEndTime"] = np.nan
    result = boundary_result()
    assert pd.isna(sample(result, 4.5).PhysicalDriverAhead)
    assert_unavailable(result)


def test_unsupported_interval_between_anchored_laps_cannot_drop_competitor():
    b = pd.concat([traversal("B", -6), traversal("B", 6, lap=2)], ignore_index=True)
    result = evaluate(b)
    # Both B traversals have anchors, but its phase at t=5 is unsupported.
    assert pd.isna(sample(result).PhysicalDriverAhead)
    assert_unavailable(result)


def returning_field(*, fresh_progress=False, returned=True):
    b = traversal("B", 94.)  # Last valid historical phase is at 104 s.
    if fresh_progress:
        b = pd.concat([b, traversal("B", 105., lap=3)], ignore_index=True)
    frame = pd.concat([traversal("A", 100.), b, traversal("C", 98.)], ignore_index=True)
    states = [(94., True), (104.25, False)]
    if returned:
        states.append((105., True))
    states = pd.DataFrame(states, columns=["SessionTime", "CandidateEligible"]).assign(Driver="B")
    return frame, states


def test_return_requires_current_phase_and_preserves_pit_interval():
    frame, states = returning_field()
    before = evaluate_physical_traffic(frame, config=CONFIG, target_lap=("A", 1),
                                      candidate_states=states.iloc[:-1])
    after = evaluate_physical_traffic(frame, config=CONFIG, target_lap=("A", 1),
                                     candidate_states=states)
    pd.testing.assert_frame_equal(before.samples[before.samples.SessionTime < 105],
                                  after.samples[after.samples.SessionTime < 105])
    assert sample(after, 104.5).PhysicalDriverAhead == "C"
    assert sample(after, 104.5).PhysicalDistanceToDriverAhead == pytest.approx(200.)
    assert sample(after, 104.5).PhysicalTimeDeltaToDriverAhead == pytest.approx(2.)
    # Historical valid endpoints cannot establish phase at the return time.
    assert sample(after, 105.).TrafficPhaseUnavailable
    assert pd.isna(sample(after, 105.).PhysicalDriverAhead)
    assert pd.isna(sample(after, 105.).PhysicalTimeDeltaToDriverAhead)
    assert_unavailable(after)


def test_return_with_supported_current_progress_resumes_numeric_participation():
    frame, states = returning_field(fresh_progress=True)
    result = evaluate_physical_traffic(frame, config=CONFIG, target_lap=("A", 1),
                                      candidate_states=states)
    assert sample(result, 104.5).PhysicalDriverAhead == "C"
    # At 109 s C no longer has a current interpolation; B's new traversal does.
    assert not sample(result, 109.).TrafficPhaseUnavailable
    assert sample(result, 109.).PhysicalDriverAhead == "B"
    assert sample(result, 109.).PhysicalDistanceToDriverAhead == pytest.approx(500.)


@pytest.mark.parametrize("driver", ["OCO", "BEA"])
def test_saved_r16_invalid_spans_do_not_emit_frozen_false_identities(driver):
    root = Path(__file__).parent / "fixtures/quali_traffic_phase"
    meta = json.loads((root / "provenance.json").read_text())[driver]
    source = root / f"{driver}.csv.gz"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == meta["sha256"]
    frame = pd.read_csv(source)
    config = replace(CONFIG, circuit_length_m=meta["circuit_length_m"])
    legacy = evaluate_physical_traffic(frame, config=replace(config, require_phase_validity=False),
                                       target_lap=(driver, 6))
    fixed = evaluate_physical_traffic(frame, config=config, target_lap=(driver, 6))
    prepared = _prepare_progress(frame, config=config)
    assert_unavailable(fixed)
    assert legacy.laps.iloc[0].PhysicalMeanTimeDeltaToDriverAhead == pytest.approx(meta["previous_mean_s"])
    for case in meta["cases"]:
        assert case["right_unfinished"] and not case["left_evicted"]
        before = sample(legacy, case["time"])
        after = sample(fixed, case["time"])
        assert before.PhysicalDriverAhead == case["ahead"]
        assert before.PhysicalDistanceToDriverAhead == pytest.approx(case["distance_m"])
        assert before.PhysicalTimeDeltaToDriverAhead == pytest.approx(case["gap_s"])
        assert pd.isna(after.PhysicalDriverAhead)
        assert pd.isna(after.PhysicalTimeDeltaToDriverAhead)
        candidate = prepared[prepared.Driver.eq(case["ahead"]) & prepared.LapNumber.eq(4)]
        assert not candidate._PhaseValid.any()
        assert validated_complete_lap_span(candidate, config=config) is None
