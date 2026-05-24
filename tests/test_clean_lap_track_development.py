from __future__ import annotations

import pandas as pd
import pytest

from wostrategy.script.push_lap_track_development import (
    _get_dominant_compound,
    _select_top_drivers,
    add_push_lap_flags,
)


def test_add_push_lap_flags_marks_single_push_between_out_and_in_laps():
    laps = _laps(
        lap_numbers=[1, 2, 3],
        lap_times=[None, 80.0, None],
        pit_out=[True, False, False],
        pit_in=[False, False, True],
        min_gaps=[None, 6.0, None],
    )

    result = add_push_lap_flags(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=5.0,
        clean_mean_time_delta_seconds=None,
    )

    assert result["IsPushLap"].tolist() == [False, True, False]
    assert result["LapPatternRole"].tolist() == ["out_lap", "push_lap", "in_lap"]


def test_add_push_lap_flags_marks_repeated_push_laps_after_slow_laps():
    laps = _laps(
        lap_numbers=[1, 2, 3, 4, 5, 6, 7],
        lap_times=[None, 90.0, 80.0, 88.0, 89.0, 81.0, None],
        pit_out=[True, False, False, False, False, False, False],
        pit_in=[False, False, False, False, False, False, True],
        min_gaps=[None, 6.0, 6.0, 6.0, 6.0, 7.0, None],
    )

    result = add_push_lap_flags(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=5.0,
        clean_mean_time_delta_seconds=None,
    )

    assert result["IsQuickLap"].tolist() == [False, False, True, False, False, True, False]
    assert result["IsPushLap"].tolist() == [False, False, True, False, False, True, False]
    assert result["LapPatternRole"].tolist() == [
        "out_lap",
        "slow_lap",
        "push_lap",
        "slow_lap",
        "slow_lap",
        "push_lap",
        "in_lap",
    ]


def test_add_push_lap_flags_allows_consecutive_push_laps():
    laps = _laps(
        lap_numbers=[1, 2, 3, 4],
        lap_times=[None, 80.0, 81.0, None],
        pit_out=[True, False, False, False],
        pit_in=[False, False, False, True],
        min_gaps=[None, 6.0, 6.0, None],
    )

    result = add_push_lap_flags(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=5.0,
        clean_mean_time_delta_seconds=None,
    )

    assert result["IsPushLap"].tolist() == [False, True, True, False]
    assert result["LapPatternRole"].tolist() == [
        "out_lap",
        "push_lap",
        "push_lap",
        "in_lap",
    ]


def test_add_push_lap_flags_uses_driver_fastest_lap_from_whole_session():
    laps = pd.concat(
        [
            _laps(
                driver="VER",
                lap_numbers=[1, 2, 3],
                lap_times=[None, 80.0, None],
                pit_out=[True, False, False],
                pit_in=[False, False, True],
                min_gaps=[None, 6.0, None],
            ),
            _laps(
                driver="VER",
                stint=2,
                lap_numbers=[4, 5, 6],
                lap_times=[None, 84.0, None],
                pit_out=[True, False, False],
                pit_in=[False, False, True],
                min_gaps=[None, 6.0, None],
            ),
        ],
        ignore_index=True,
    )

    result = add_push_lap_flags(
        laps,
        quick_lap_threshold=1.03,
        clean_min_time_delta_seconds=5.0,
        clean_mean_time_delta_seconds=None,
    )

    assert result.loc[result["LapNumber"] == 2, "IsPushLap"].iloc[0]
    assert not result.loc[result["LapNumber"] == 5, "IsPushLap"].iloc[0]


def test_add_push_lap_flags_rejects_dirty_quick_laps():
    laps = _laps(
        lap_numbers=[1, 2, 3],
        lap_times=[None, 80.0, None],
        pit_out=[True, False, False],
        pit_in=[False, False, True],
        min_gaps=[None, 4.9, None],
    )

    result = add_push_lap_flags(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=5.0,
        clean_mean_time_delta_seconds=None,
    )

    assert not result["IsPushLap"].any()
    assert result["LapPatternRole"].tolist() == ["out_lap", "slow_lap", "in_lap"]


def test_add_push_lap_flags_accepts_zero_clean_delta_as_no_gap_filter():
    laps = _laps(
        lap_numbers=[1, 2, 3],
        lap_times=[None, 80.0, None],
        pit_out=[True, False, False],
        pit_in=[False, False, True],
        min_gaps=[None, 0.0, None],
    )

    result = add_push_lap_flags(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=0,
        clean_mean_time_delta_seconds=None,
    )

    assert result["IsPushLap"].tolist() == [False, True, False]


def test_add_push_lap_flags_can_use_mean_delta_clean_filter():
    laps = _laps(
        lap_numbers=[1, 2, 3],
        lap_times=[None, 80.0, None],
        pit_out=[True, False, False],
        pit_in=[False, False, True],
        min_gaps=[None, 4.0, None],
        mean_gaps=[None, 8.0, None],
    )

    result = add_push_lap_flags(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=7.0,
    )

    assert result["IsPushLap"].tolist() == [False, True, False]


def test_add_push_lap_flags_requires_exactly_one_clean_delta_filter():
    laps = _laps(
        lap_numbers=[1, 2, 3],
        lap_times=[None, 80.0, None],
        pit_out=[True, False, False],
        pit_in=[False, False, True],
        min_gaps=[None, 6.0, None],
    )

    with pytest.raises(ValueError, match="Define exactly one clean gap filter"):
        add_push_lap_flags(
            laps,
            quick_lap_threshold=1.07,
            clean_min_time_delta_seconds=None,
            clean_mean_time_delta_seconds=None,
        )

    with pytest.raises(ValueError, match="Define exactly one clean gap filter"):
        add_push_lap_flags(
            laps,
            quick_lap_threshold=1.07,
            clean_min_time_delta_seconds=5.0,
            clean_mean_time_delta_seconds=7.0,
        )


def test_add_push_lap_flags_adds_session_lap_order_by_lap_start_time():
    laps = pd.concat(
        [
            _laps(
                driver="VER",
                lap_numbers=[1, 2],
                lap_times=[80.0, 81.0],
                pit_out=[False, False],
                pit_in=[False, False],
                min_gaps=[6.0, 6.0],
            ),
            _laps(
                driver="LEC",
                lap_numbers=[1, 2],
                lap_times=[82.0, 83.0],
                pit_out=[False, False],
                pit_in=[False, False],
                min_gaps=[6.0, 6.0],
                start_minutes=[1.5, 3.5],
            ),
        ],
        ignore_index=True,
    )

    result = add_push_lap_flags(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=5.0,
        clean_mean_time_delta_seconds=None,
    )

    ordered = result.sort_values("SessionLapOrder")
    assert ordered["Driver"].tolist() == ["VER", "LEC", "VER", "LEC"]
    assert ordered["LapNumber"].tolist() == [1, 1, 2, 2]


def test_select_top_drivers_uses_fastest_session_lap():
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "VER", "LEC", "LEC", "HAM"],
            "LapTimeSeconds": [81.0, 80.0, 79.5, 82.0, 83.0],
        }
    )

    assert _select_top_drivers(laps, 2) == ["LEC", "VER"]


def test_select_top_drivers_prefers_session_result_rank():
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "VER", "LEC", "LEC", "HAM"],
            "LapTimeSeconds": [79.0, 78.0, 77.5, 82.0, 83.0],
            "SessionResultRank": [2, 2, 3, 3, 1],
        }
    )

    assert _select_top_drivers(laps, 2) == ["HAM", "VER"]


def test_select_top_drivers_allows_count_larger_than_available_drivers():
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "LEC", "HAM"],
            "LapTimeSeconds": [79.0, 78.0, 80.0],
            "SessionResultRank": [2, 1, 3],
        }
    )

    assert _select_top_drivers(laps, 15) == ["LEC", "VER", "HAM"]


def test_get_dominant_compound_requires_more_than_half():
    dominant_laps = pd.DataFrame({"Compound": ["SOFT", "SOFT", "MEDIUM"]})
    no_dominant_laps = pd.DataFrame({"Compound": ["SOFT", "MEDIUM"]})

    assert _get_dominant_compound(dominant_laps) == "SOFT"
    assert _get_dominant_compound(no_dominant_laps) is None


def _laps(
    *,
    lap_numbers,
    lap_times,
    pit_out,
    pit_in,
    min_gaps,
    mean_gaps=None,
    driver="VER",
    stint=1,
    start_minutes=None,
) -> pd.DataFrame:
    if start_minutes is None:
        start_minutes = lap_numbers
    if mean_gaps is None:
        mean_gaps = min_gaps

    return pd.DataFrame(
        {
            "Driver": driver,
            "Stint": stint,
            "LapNumber": lap_numbers,
            "LapTime": pd.to_timedelta(lap_times, unit="s"),
            "LapStartTime": pd.to_timedelta(start_minutes, unit="m"),
            "PitOutTime": _pit_times(pit_out, lap_numbers),
            "PitInTime": _pit_times(pit_in, lap_numbers),
            "MinTimeDeltaToDriverAhead": min_gaps,
            "MeanTimeDeltaToDriverAhead": mean_gaps,
        }
    )


def _pit_times(mask, lap_numbers) -> pd.Series:
    values = [
        pd.Timedelta(minutes=lap_number) if is_pit_lap else pd.NaT
        for is_pit_lap, lap_number in zip(mask, lap_numbers)
    ]
    return pd.Series(values)
