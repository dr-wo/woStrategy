import json
from pathlib import Path

from wostrategy.analysis.pre_race_tyre_prediction import get_pre_race_tyre_prediction


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
