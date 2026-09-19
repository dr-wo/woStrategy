import json
from pathlib import Path

import pytest

from wostrategy.analysis.pre_race_tyre_prediction import (
    get_pre_race_tyre_prediction,
    load_pre_race_tyre_prediction_source,
)
from wostrategy.model.tyre_prediction import (
    PreRaceTyrePrediction,
    TyreCompoundPrediction,
    canonicalize_operational_tyre_prediction,
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
