from __future__ import annotations

from pathlib import Path

import pandas as pd

from wostrategy.core.planner_loader import (
    get_legacy_planner_laps_cache_path,
    get_planner_laps_cache_path,
    load_race_laps_for_planner,
)


def test_legacy_planner_cache_path_uses_repository_cache():
    repository_root = Path(__file__).resolve().parents[2]

    path = get_legacy_planner_laps_cache_path(
        year=2026,
        round_number=12,
        session="R",
    )

    assert path == repository_root / "cache" / "planner_race_laps" / "2026_12_R.pkl"


def test_get_planner_laps_cache_path_uses_wodata_layout(tmp_path):
    path = get_planner_laps_cache_path(
        year=2026,
        round_number=12,
        session="R",
        data_root=tmp_path,
    )

    assert path == (
        tmp_path
        / "wostrategy"
        / "planner_race_laps"
        / "schema_v2"
        / "year=2026"
        / "round=12"
        / "session=R"
        / "laps.pkl"
    )


def test_load_race_laps_for_planner_reads_wodata_cache(tmp_path):
    cache_path = get_planner_laps_cache_path(
        year=2026,
        round_number=12,
        session="R",
        data_root=tmp_path,
    )
    cache_path.parent.mkdir(parents=True)
    expected = pd.DataFrame(
        {
            "Driver": ["AAA"],
            "LapNumber": [1],
            "SessionStartPosition": [1],
            "Sector1TrackStatus": [pd.NA],
            "Sector2TrackStatus": [pd.NA],
            "Sector3TrackStatus": [pd.NA],
        }
    )
    expected.to_pickle(cache_path)

    result = load_race_laps_for_planner(
        year=2026,
        round_number=12,
        session="R",
        data_root=tmp_path,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_load_race_laps_for_planner_refreshes_cache_without_start_position(
    monkeypatch,
    tmp_path,
):
    cache_path = get_planner_laps_cache_path(
        year=2026,
        round_number=12,
        session="R",
        data_root=tmp_path,
    )
    cache_path.parent.mkdir(parents=True)
    pd.DataFrame({"Driver": ["AAA"], "LapNumber": [1]}).to_pickle(cache_path)
    fresh = pd.DataFrame(
        {
            "Driver": ["BBB"],
            "LapNumber": [1],
            "SessionStartPosition": [1],
        }
    )

    monkeypatch.setattr(
        "wostrategy.core.planner_loader.load_all_session_laps",
        lambda *args, **kwargs: fresh,
    )

    result = load_race_laps_for_planner(
        year=2026,
        round_number=12,
        session="R",
        data_root=tmp_path,
    )

    pd.testing.assert_frame_equal(result, fresh)
