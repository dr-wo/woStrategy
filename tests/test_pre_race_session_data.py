import pandas as pd

from wostrategy.analysis.traffic import TRAFFIC_EVALUATOR_VERSION
from wostrategy.core.pre_race_session_data import load_cached_weekend_sessions


def test_loader_uses_wodata_cache_before_fastf1(tmp_path):
    calls = []

    def loader(**kwargs):
        calls.append(kwargs["session"])
        return pd.DataFrame({"SessionName": [kwargs["session"]], "LapNumber": [1]})

    first = load_cached_weekend_sessions(
        year=2026, round_number=8, sessions=("FP1", "FP2"), data_root=tmp_path,
        session_loader=loader,
    )
    second = load_cached_weekend_sessions(
        year=2026, round_number=8, sessions=("FP1", "FP2"), data_root=tmp_path,
        session_loader=lambda **kwargs: (_ for _ in ()).throw(AssertionError("downloaded")),
    )

    assert calls == ["FP1", "FP2"]
    assert first.sources == {"FP1": "fastf1-download", "FP2": "fastf1-download"}
    assert second.sources == {"FP1": "wodata-cache", "FP2": "wodata-cache"}
    assert all(str(path).startswith(str(tmp_path)) for path in second.cache_paths.values())


def test_loader_refreshes_v2_enriched_traffic_cache(tmp_path):
    calls = []

    def loader(**kwargs):
        calls.append(kwargs["session"])
        return pd.DataFrame(
            {"SessionName": [kwargs["session"]], "LapNumber": [len(calls)]}
        )

    first = load_cached_weekend_sessions(
        year=2026,
        round_number=8,
        sessions=("FP1",),
        data_root=tmp_path,
        session_loader=loader,
    )
    cache_path = first.cache_paths["FP1"]
    stale = pd.read_pickle(cache_path)
    stale.attrs["traffic_evaluator_version"] = "unified-traffic-v2"
    stale.to_pickle(cache_path)

    refreshed = load_cached_weekend_sessions(
        year=2026,
        round_number=8,
        sessions=("FP1",),
        data_root=tmp_path,
        session_loader=loader,
    )

    assert calls == ["FP1", "FP1"]
    assert refreshed.sources == {"FP1": "fastf1-download"}
    assert refreshed.sessions["FP1"]["LapNumber"].tolist() == [2]
    assert (
        refreshed.sessions["FP1"].attrs["traffic_evaluator_version"]
        == TRAFFIC_EVALUATOR_VERSION
    )


def test_loader_preserves_fastf1_failure_reason(tmp_path):
    result = load_cached_weekend_sessions(
        year=2026,
        round_number=11,
        sessions=("FP1",),
        data_root=tmp_path,
        session_loader=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("timing endpoint unavailable")
        ),
    )

    assert result.sessions == {}
    assert result.sources["FP1"] == "FastF1 load failed: timing endpoint unavailable"


def test_loader_explains_empty_fastf1_result(tmp_path):
    result = load_cached_weekend_sessions(
        year=2026,
        round_number=11,
        sessions=("FP1",),
        data_root=tmp_path,
        session_loader=lambda **kwargs: pd.DataFrame(),
    )

    reason = result.sources["FP1"]
    assert "FastF1 returned no laps" in reason
    assert "year=2026/round=11/session=FP1/laps.pkl" in reason
