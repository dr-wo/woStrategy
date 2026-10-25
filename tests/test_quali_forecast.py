"""Field scenarios use causal pace, chronological priors and coherent vectors."""
from dataclasses import replace
import json

import pytest

from wostrategy.analysis.quali_forecast import (
    DriverPace, ForecastPrior, forecast_final_cutoff, forecast_partial_field_bounds,
)


def record(event, *, part="Q1", lead=300., improvement=(1., 1., 1., 1.), previous=False,
           current=True):
    return {"event_id": f"2026-R{event}", "part": part, "remaining_s": lead,
            "split": "train", "drivers": [
                {"driver": f"D{i}", "current_best_s": 100. + i if current else None,
                 "previous_part_best_s": 100. + i if previous else None,
                 "final_best_s": 100. + i - delta,
                 "future_credible_attempt": delta > 0}
                for i, delta in enumerate(improvement)]}


def prior(*records):
    return ForecastPrior.from_records(records, revision="test-train-only")


def field():
    return tuple(DriverPace(f"D{i}", 100. + i) for i in range(4))


def predict(calibration, drivers=None, **kwargs):
    inputs = dict(event_id="2026-R18", part="Q1", remaining_s=300., target_rank=3)
    inputs.update(kwargs)
    return forecast_final_cutoff(field() if drivers is None else drivers, prior=calibration, **inputs)


def test_no_prior_never_relabels_observed_cut_as_forecast():
    result = predict(None)
    assert result.status == "UNAVAILABLE"
    assert result.estimate_s is result.lower_s is result.upper_s is None
    assert "not substituted" in result.limitations[0]


@pytest.mark.parametrize("event_id", [None, "R18", "2026-R1", "2026-R2", "2025-R24"])
def test_missing_same_or_future_event_calibration_is_rejected(event_id):
    calibration = prior(record(1), record(2))
    result = predict(calibration, event_id=event_id)
    assert result.status == "UNAVAILABLE"
    assert result.estimate_s is None


def test_forecast_uses_entire_field_and_non_boundary_driver_can_change_target():
    calibration = prior(record(1, improvement=(0, 0, 0, 3)),
                        record(2, improvement=(0, 0, 0, 4)))
    base = predict(calibration)
    # D3 is currently P4, outside target P3, but its calibrated future pace
    # moves it ahead of D2 and changes the field's eventual P3.
    blocked = predict(calibration, tuple(replace(d, future_attempt_possible=False)
                                       if d.driver == "D3" else d for d in field()))
    assert base.status == "EXPERIMENTAL"
    assert base.estimate_s < 102.
    assert blocked.estimate_s == 102.


def test_joint_scenarios_keep_shared_improvements_correlated():
    a = record(1, improvement=tuple((100. + i) * .01 for i in range(4)))
    b = record(2, improvement=tuple((100. + i) * .02 for i in range(4)))
    result = predict(prior(a, b))
    assert result.scenario_count == 2
    assert result.training_events == ("2026-R1", "2026-R2")
    for scenario, ratio in zip(result.scenarios, (.99, .98)):
        finals = dict(scenario.driver_final_bests_s)
        assert [finals[f"D{i}"] / (100. + i) for i in range(4)] == pytest.approx([ratio] * 4)
    assert result.lower_s < result.estimate_s < result.upper_s
    assert "prospective coverage is unvalidated" in result.limitations[0]


def test_early_q2_without_observed_cutoff_uses_only_causal_q1_anchors():
    calibration = prior(record(1, part="Q2", lead=900., current=False, previous=True),
                        record(2, part="Q2", lead=900., current=False, previous=True,
                               improvement=(2, 2, 2, 2)))
    drivers = tuple(DriverPace(f"D{i}", previous_part_best_s=100. + i) for i in range(4))
    result = predict(calibration, drivers, part="Q2", remaining_s=900.)
    assert result.status == "EXPERIMENTAL"
    assert result.current_pace_count == 0
    assert result.previous_part_pace_count == 4
    assert result.estimate_s == pytest.approx(100.5)
    assert result.source == "HISTORICAL_JOINT_FIELD_PRIOR"


def test_missing_unknown_driver_pace_cannot_be_filled_from_final_training_rank():
    calibration = prior(record(1), record(2))
    drivers = field()[:3] + (DriverPace("D3"),)
    result = predict(calibration, drivers)
    assert result.status == "UNAVAILABLE"
    assert "D3" in result.limitations[0]
    # A caller-certified no-future/no-time driver does not require invented pace.
    result = predict(calibration, field()[:3] + (DriverPace("D3", future_attempt_possible=False),))
    assert result.status == "EXPERIMENTAL"


def test_prior_serialization_preserves_detached_provenance_and_validates_manifest():
    calibration = prior(record(1), record(2))
    serialized = calibration.to_dict()
    restored = ForecastPrior.from_dict(json.loads(json.dumps(serialized)))
    assert restored == calibration
    assert predict(restored) == predict(calibration)
    serialized["metadata"]["method"] = "mutated"
    assert calibration.metadata["method"] != "mutated"
    with pytest.raises(ValueError, match="method"):
        ForecastPrior.from_dict(serialized)
    serialized = calibration.to_dict()
    serialized["training_events"].append("2026-R18")
    with pytest.raises(ValueError, match="manifest"):
        ForecastPrior.from_dict(serialized)


def test_training_does_not_accept_holdout_rows_or_backfill_missing_anchors():
    with pytest.raises(ValueError, match="training"):
        prior(dict(record(1), split="holdout"))
    missing = record(1, current=False)
    calibration = prior(missing, record(2))
    assert calibration.training_events == ("2026-R2",)
    assert calibration.metadata["skipped_records"][0]["reason"] == "INCOMPLETE_PACE_OR_FINAL_TARGET_COVERAGE"
    assert predict(calibration).status == "UNAVAILABLE"


def test_driver_order_is_irrelevant_and_lead_interpolation_is_bounded():
    calibration = prior(record(1, lead=0., improvement=(0, 0, 0, 0)),
                        record(1, lead=300.), record(2, lead=0., improvement=(0, 0, 0, 0)),
                        record(2, lead=300., improvement=(2, 2, 2, 2)))
    result = predict(calibration, remaining_s=150.)
    assert predict(calibration, tuple(reversed(field())), remaining_s=150.) == result
    assert result.estimate_s == pytest.approx(101.25)
    assert predict(calibration, remaining_s=301.).status == "UNAVAILABLE"


def test_zero_clock_does_not_remove_inflight_laps_but_finalised_part_converges():
    calibration = prior(record(1, lead=0.), record(2, lead=0., improvement=(2, 2, 2, 2)))
    at_flag = predict(calibration, remaining_s=0.)
    assert at_flag.estimate_s < 102.
    final = predict(None, remaining_s=0., part_finalised=True)
    assert final.status == "OBSERVED_FINAL"
    assert final.estimate_s == final.lower_s == final.upper_s == 102.


def test_current_best_bounds_prior_transfer_when_q2_has_started():
    calibration = prior(record(1, part="Q2", lead=900., current=False, previous=True,
                               improvement=(-2, -2, -2, -2)),
                        record(2, part="Q2", lead=900., current=False, previous=True,
                               improvement=(-1, -1, -1, -1)))
    drivers = tuple(DriverPace(f"D{i}", 99. + i, 100. + i) for i in range(4))
    result = predict(calibration, drivers, part="Q2", remaining_s=900.)
    assert result.status == "EXPERIMENTAL"
    assert result.estimate_s == 101.
    assert all(value <= dict((d.driver, d.current_best_s) for d in drivers)[code]
               for scenario in result.scenarios for code, value in scenario.driver_final_bests_s)


def test_fixed_rank_and_invalid_inputs_do_not_fall_back_to_dynamic_current_cut():
    calibration = prior(record(1), record(2))
    assert predict(calibration, target_rank=16).status == "UNAVAILABLE"
    assert predict(calibration, part="Q3").status == "UNAVAILABLE"
    assert predict(calibration, remaining_s=float("nan")).status == "UNAVAILABLE"
    assert predict(calibration, field() + (field()[0],)).status == "UNAVAILABLE"


def test_partial_envelope_assigns_no_unknown_pace_point_or_probability():
    calibration = prior(record(1), record(2, improvement=(2, 2, 2, 2)))
    drivers = field() + (DriverPace("UNKNOWN"),)
    result = forecast_partial_field_bounds(drivers, event_id="2026-R18", part="Q1",
        remaining_s=300., prior=calibration, target_rank=3)
    assert result.status == "SCENARIO_RANGE"
    assert result.source == "PARTIAL_FIELD_SCENARIO_ENVELOPE"
    assert result.estimate_s is result.interval_coverage is None
    assert result.lower_s < result.upper_s
    assert result.scenario_count == 2
    assert result.training_events == ("2026-R1", "2026-R2")
    assert result.scenarios == ()  # No invented final-best trajectory for UNKNOWN.
    assert "no probability coverage" in " ".join(result.limitations)


def test_partial_envelope_contains_complete_scenarios_for_all_unknown_rank_placements():
    calibration = prior(record(1, improvement=(0, 4, 0, 3)),
                        record(2, improvement=(2, 0, 4, 0)))
    drivers = field() + (DriverPace("UNKNOWN"), DriverPace("UNKNOWN2"))
    bounds = forecast_partial_field_bounds(drivers, event_id="2026-R18", part="Q1",
        remaining_s=300., prior=calibration, target_rank=3)
    for pace in (50., 99.5, 100.25, 101.5, 102.5, 103.5, 200.):
        for pace2 in (50., 100.5, 102.25, 200.):
            complete = predict(calibration, field() + (DriverPace("UNKNOWN", pace),
                                DriverPace("UNKNOWN2", pace2)))
            for scenario in complete.scenarios:
                assert bounds.lower_s <= scenario.cutoff_s <= bounds.upper_s


def test_partial_envelope_retains_chronological_and_finite_support_guards():
    calibration = prior(record(1), record(2))
    drivers = field()[:2] + (DriverPace("UNKNOWN"), DriverPace("UNKNOWN2"))
    result = forecast_partial_field_bounds(drivers, event_id="2026-R18", part="Q1",
        remaining_s=300., prior=calibration, target_rank=3)
    assert result.status == "UNAVAILABLE"
    assert "unbounded" in result.limitations[0]
    result = forecast_partial_field_bounds(field() + (DriverPace("UNKNOWN"),),
        event_id="2026-R1", part="Q1", remaining_s=300., prior=calibration, target_rank=3)
    assert result.status == "UNAVAILABLE"
    assert "strictly precede" in result.limitations[0]
    assert forecast_partial_field_bounds(field(), event_id="2026-R18", part="Q1",
        remaining_s=300., prior=calibration, target_rank=3) == predict(calibration)
