import json
from pathlib import Path

import pytest

from wostrategy.analysis.pre_race_tyre_prediction import (
    get_pre_race_tyre_prediction,
    load_pre_race_tyre_prediction_source,
)
from wostrategy.model.tyre_prediction import (
    COMMON_GAUGE_FRAME,
    RELATIVE_PERFORMANCE_FRAME,
    PreRaceTyrePrediction,
    TyreCompoundPrediction,
    canonicalize_operational_tyre_prediction,
    select_source_performance_reference,
)


def _write_prediction(root: Path, timestamp: str) -> None:
    path = root / "wostrategy/tyre_prediction/schema_v1/year=2026/predictions/round=1/prediction.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "season": 2026, "round": 1, "event": "Test", "prediction_timestamp": timestamp,
        "model": "test", "prediction_fingerprint": timestamp,
        "compound_allocation": {"SOFT": "C5", "MEDIUM": "C4", "HARD": "C3"},
        "descriptor_domain": {"any_boundary_descriptor": False},
        "compounds": {
            "C5": {"compound_role": "SOFT", "performance_mean": -0.3, "performance_std": 0.1, "degradation_mean": 0.2, "degradation_std": 0.03},
            "C4": {"compound_role": "MEDIUM", "performance_mean": 0.0, "performance_std": 0.1, "degradation_mean": 0.1, "degradation_std": 0.02},
            "C3": {"compound_role": "HARD", "performance_mean": 0.2, "performance_std": 0.1, "degradation_mean": 0.05, "degradation_std": 0.01},
        },
    }), encoding="utf-8")


def test_stale_cache_calls_owner_refresh_and_normalizes_medium_reference(tmp_path: Path) -> None:
    _write_prediction(tmp_path, "2026-01-01T00:00:00+00:00")
    refreshed = []
    def refresh():
        refreshed.append(True)
        _write_prediction(tmp_path, "2026-01-02T00:00:00+00:00")
    result = get_pre_race_tyre_prediction(
        season=2026, round_number=1, data_root=tmp_path,
        fresh_after="2026-01-01T12:00:00+00:00", refresh=refresh,
    )
    assert refreshed == [True]
    assert result.compounds["MEDIUM"].performance_delta_to_medium == 0.0
    assert result.compounds["SOFT"].degradation_uncertainty == 0.03


def _prediction(compounds, *, season=2026, round_number=1, event="2026-01"):
    return PreRaceTyrePrediction(
        event=event,
        season=season,
        round_number=round_number,
        reference_compound="MEDIUM",
        compounds=compounds,
        generated_at="2026-01-01T00:00:00+00:00",
        provider="test",
        artifact_id="test",
    )


def _compound(performance, degradation):
    return TyreCompoundPrediction(performance, degradation)


def test_operational_contract_accepts_arbitrary_baseline_once() -> None:
    source = _prediction({
        "SOFT": _compound(-0.20, 0.20),
        "MEDIUM": _compound(0.30, 0.10),
        "HARD": _compound(0.80, 0.05),
    })
    before = {
        key: value.performance_delta_to_medium
        for key, value in source.compounds.items()
    }

    result = canonicalize_operational_tyre_prediction(
        source,
        expected_season=2026,
        expected_round_number=1,
        expected_event_id="2026-01",
    )
    after = {
        key: value.performance_delta_to_medium
        for key, value in result.compounds.items()
    }

    assert after == pytest.approx({"SOFT": -0.50, "MEDIUM": 0.0, "HARD": 0.50})
    assert after["SOFT"] - after["MEDIUM"] == pytest.approx(
        before["SOFT"] - before["MEDIUM"]
    )
    assert after["HARD"] - after["MEDIUM"] == pytest.approx(
        before["HARD"] - before["MEDIUM"]
    )
    assert after["HARD"] - after["SOFT"] == pytest.approx(
        before["HARD"] - before["SOFT"]
    )
    assert canonicalize_operational_tyre_prediction(
        result,
        expected_season=2026,
        expected_round_number=1,
        expected_event_id="2026-01",
    ).compounds == result.compounds


@pytest.mark.parametrize(
    ("compounds", "message"),
    [
        ({"MEDIUM": _compound(0.0, 0.1)}, "SOFT, HARD"),
        ({"MEDIUM": _compound(0.0, 0.1), "HARD": _compound(0.3, 0.05)}, "SOFT"),
        ({"SOFT": _compound(-0.3, 0.2), "MEDIUM": _compound(0.0, 0.1)}, "HARD"),
        ({"SOFT": _compound(None, 0.2), "MEDIUM": _compound(0.0, 0.1), "HARD": _compound(0.3, 0.05)}, "is required"),
        ({"SOFT": _compound(-0.3, 0.2), "MEDIUM": _compound(0.0, None), "HARD": _compound(0.3, 0.05)}, "is required"),
        ({"SOFT": _compound(float("nan"), 0.2), "MEDIUM": _compound(0.0, 0.1), "HARD": _compound(0.3, 0.05)}, "must be finite"),
        ({"SOFT": _compound(-0.3, 0.2), "MEDIUM": _compound(float("inf"), 0.1), "HARD": _compound(0.3, 0.05)}, "must be finite"),
        ({"SOFT": _compound(-0.3, -0.01), "MEDIUM": _compound(0.0, 0.1), "HARD": _compound(0.3, 0.05)}, "cannot be negative"),
    ],
)
def test_operational_contract_rejects_unusable_values(compounds, message) -> None:
    with pytest.raises(ValueError, match=message):
        canonicalize_operational_tyre_prediction(
            _prediction(compounds),
            expected_season=2026,
            expected_round_number=1,
            expected_event_id="2026-01",
        )


@pytest.mark.parametrize(
    ("season", "round_number", "event", "message"),
    [
        (2025, 1, "2025-01", "season 2025"),
        (2026, 2, "2026-02", "round 2"),
        (2026, 1, "2025-01", "event 2025-01"),
    ],
)
def test_operational_contract_rejects_wrong_event_identity(
    season, round_number, event, message,
) -> None:
    compounds = {
        "SOFT": _compound(-0.3, 0.2),
        "MEDIUM": _compound(0.0, 0.1),
        "HARD": _compound(0.3, 0.05),
    }
    with pytest.raises(ValueError, match=message):
        canonicalize_operational_tyre_prediction(
            _prediction(
                compounds, season=season, round_number=round_number, event=event,
            ),
            expected_season=2026,
            expected_round_number=1,
            expected_event_id="2026-01",
        )


def test_source_loader_preserves_raw_baseline_while_public_loader_canonicalizes(
    tmp_path: Path,
) -> None:
    _write_prediction(tmp_path, "2026-01-01T00:00:00+00:00")
    path = (
        tmp_path / "wostrategy/tyre_prediction/schema_v1/year=2026/"
        "predictions/round=1/prediction.json"
    )
    payload = json.loads(path.read_text())
    payload["compounds"]["C5"]["performance_mean"] = -0.2
    payload["compounds"]["C4"]["performance_mean"] = 0.3
    payload["compounds"]["C3"]["performance_mean"] = 0.8
    path.write_text(json.dumps(payload))

    source = load_pre_race_tyre_prediction_source(
        season=2026, round_number=1, data_root=tmp_path,
    )
    operational = get_pre_race_tyre_prediction(
        season=2026, round_number=1, data_root=tmp_path,
    )

    assert source.compounds["MEDIUM"].performance_delta_to_medium == 0.3
    assert operational.compounds["SOFT"].performance_delta_to_medium == pytest.approx(-0.5)
    assert operational.compounds["MEDIUM"].performance_delta_to_medium == 0.0
    assert operational.compounds["HARD"].performance_delta_to_medium == pytest.approx(0.5)


def test_source_loader_honours_declared_hard_reference(tmp_path: Path) -> None:
    _write_prediction(tmp_path, "2026-01-01T00:00:00+00:00")
    path = (
        tmp_path / "wostrategy/tyre_prediction/schema_v1/year=2026/"
        "predictions/round=1/prediction.json"
    )
    payload = json.loads(path.read_text())
    payload["performance_coordinate"] = "seconds relative to event HARD compound"
    payload["compounds"]["C5"]["performance_mean"] = -0.5
    payload["compounds"]["C4"]["performance_mean"] = -0.2
    payload["compounds"]["C3"]["performance_mean"] = 0.0
    path.write_text(json.dumps(payload))

    source = load_pre_race_tyre_prediction_source(
        season=2026, round_number=1, data_root=tmp_path,
    )
    operational = get_pre_race_tyre_prediction(
        season=2026, round_number=1, data_root=tmp_path,
    )

    assert source.reference_compound == "HARD"
    assert source.reference_declared is True
    assert source.compounds["HARD"].performance_delta_to_medium == 0.0
    assert operational.reference_compound == "MEDIUM"
    assert operational.compounds["SOFT"].performance_delta_to_medium == pytest.approx(-.3)
    assert operational.compounds["HARD"].performance_delta_to_medium == pytest.approx(.2)


@pytest.mark.parametrize(
    ("values", "expected", "rule"),
    [
        ({"SOFT": 0.0, "MEDIUM": .4, "HARD": .8}, "SOFT", "unique_zero"),
        ({"SOFT": -.4, "MEDIUM": 0.0, "HARD": .4}, "MEDIUM", "unique_zero"),
        ({"SOFT": -.8, "MEDIUM": -.4, "HARD": 0.0}, "HARD", "unique_zero"),
        ({"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}, "MEDIUM", "multiple_zeros_prefer_medium_then_hard_then_soft"),
        ({"SOFT": 0.0, "MEDIUM": .4, "HARD": 0.0}, "HARD", "multiple_zeros_prefer_medium_then_hard_then_soft"),
        ({"SOFT": -.3, "MEDIUM": .2, "HARD": .7}, "MEDIUM", "no_zero_default_medium"),
    ],
)
def test_source_reference_selection_rules(values, expected, rule) -> None:
    reference, selected_by, _ = select_source_performance_reference(values)
    assert reference == expected
    assert selected_by == rule


def test_declared_reference_takes_priority_without_normalizing_values() -> None:
    values = {"SOFT": -.3, "MEDIUM": .2, "HARD": .7}
    reference, rule, _ = select_source_performance_reference(
        values, declared_reference="HARD"
    )
    assert reference == "HARD"
    assert rule == "declared"
    assert values == {"SOFT": -.3, "MEDIUM": .2, "HARD": .7}


def test_formal_relative_output_requires_declared_reference_zero() -> None:
    with pytest.raises(ValueError, match="reference HARD must be zero"):
        PreRaceTyrePrediction(
            event="2026-01", season=2026, round_number=1,
            reference_compound="HARD",
            compounds={
                "SOFT": _compound(-.5, .2),
                "MEDIUM": _compound(0.0, .1),
                "HARD": _compound(.5, .05),
            },
            generated_at="now", provider="test", artifact_id="test",
            performance_frame=RELATIVE_PERFORMANCE_FRAME,
            reference_declared=True,
            formal_relative_output=True,
        )


def test_formal_relative_reference_validator_uses_documented_tolerance() -> None:
    compounds = {
        "SOFT": _compound(-.5, .2),
        "MEDIUM": _compound(-.2, .1),
        "HARD": _compound(1e-9, .05),
    }
    accepted = PreRaceTyrePrediction(
        event="2026-01", season=2026, round_number=1,
        reference_compound="HARD", compounds=compounds,
        generated_at="now", provider="test", artifact_id="within-tolerance",
        performance_frame=RELATIVE_PERFORMANCE_FRAME,
        reference_declared=True, formal_relative_output=True,
    )
    assert accepted.compounds["HARD"].performance_delta_to_medium == 1e-9

    with pytest.raises(ValueError, match="must be zero within"):
        PreRaceTyrePrediction(
            event="2026-01", season=2026, round_number=1,
            reference_compound="HARD",
            compounds={**compounds, "HARD": _compound(1.01e-9, .05)},
            generated_at="now", provider="test", artifact_id="outside-tolerance",
            performance_frame=RELATIVE_PERFORMANCE_FRAME,
            reference_declared=True, formal_relative_output=True,
        )


def test_common_gauge_and_legacy_inputs_are_not_rejected_as_relative_outputs() -> None:
    compounds = {
        "SOFT": _compound(-.2, .2),
        "MEDIUM": _compound(.3, .1),
        "HARD": _compound(.8, .05),
    }
    common = PreRaceTyrePrediction(
        event="2026-01", season=2026, round_number=1,
        reference_compound="HARD", compounds=compounds,
        generated_at="now", provider="test", artifact_id="common",
        performance_frame=COMMON_GAUGE_FRAME, reference_declared=True,
    )
    legacy = _prediction(compounds)
    assert common.compounds["HARD"].performance_delta_to_medium == .8
    assert legacy.compounds["MEDIUM"].performance_delta_to_medium == .3


def test_operational_output_carries_semantics_and_one_normalization_record() -> None:
    result = canonicalize_operational_tyre_prediction(
        _prediction({
            "SOFT": _compound(-.2, .2),
            "MEDIUM": _compound(.3, .1),
            "HARD": _compound(.8, .05),
        }),
        expected_season=2026,
        expected_round_number=1,
        expected_event_id="2026-01",
    )
    assert result.performance_frame == RELATIVE_PERFORMANCE_FRAME
    assert result.provenance["performance_semantics"]["reference_compound"] == "MEDIUM"
    assert result.provenance["operational_normalization"]["normalization_count"] == 1
    assert result.compounds["MEDIUM"].performance_delta_to_medium == 0.0
