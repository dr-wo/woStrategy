from __future__ import annotations

import pandas as pd
import pytest

from wostrategy.model.track_evolution import TRACK_EVOLUTION_FIT_MODEL
from wostrategy.analysis.quali_performance import QUICK_LAP_NUMBER
from wostrategy.script.quali_performance_tracker import (
    CORRECTED_SECTOR_SECONDS,
    EXPONENTIAL_TRACK_EVOLUTION_MODEL,
    LINEAR_TRACK_EVOLUTION_MODEL,
    QUALIFYING_PART,
    TRACK_EVO_CORRECTION_SECONDS,
    TRACK_EVO_CORRECTED_LAP_TIME_SECONDS,
    _add_corrected_sector_times,
    _add_track_evolution_correction,
    _has_clean_gap_columns,
    _parse_race_range,
    _team_best_sector_rows,
    _team_fastest_and_average_rows,
    calculate_quali_performance,
)


def test_calculate_quali_performance_returns_wet_when_wet_tyres_used():
    laps = _laps([("Ferrari", "LEC", "INTERMEDIATE", 80.0, 2)])

    result = calculate_quali_performance(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=3.0,
        last_quali_part_only=False,
        track_evolution_fit=LINEAR_TRACK_EVOLUTION_MODEL,
    )

    assert result == "Wet"


def test_track_evolution_correction_uses_last_push_lap_reference():
    laps = pd.DataFrame(
        {
            "LapTimeSeconds": [80.0, 81.0],
            "SessionLapOrder": [100, 300],
        }
    )

    result = _add_track_evolution_correction(
        laps,
        evolution_rate_seconds_per_lap=0.01,
        reference_session_lap_order=300,
    )

    assert result[TRACK_EVO_CORRECTION_SECONDS].tolist() == [2.0, 0.0]
    assert result[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS].tolist() == [78.0, 81.0]


def test_corrected_sector_times_keep_original_sector_ratio():
    laps = pd.DataFrame(
        {
            "LapTimeSeconds": [80.0],
            TRACK_EVO_CORRECTED_LAP_TIME_SECONDS: [76.0],
            "Sector1Time": pd.to_timedelta([20.0], unit="s"),
            "Sector2Time": pd.to_timedelta([40.0], unit="s"),
            "Sector3Time": pd.to_timedelta([20.0], unit="s"),
        }
    )

    result = _add_corrected_sector_times(laps)

    assert result[CORRECTED_SECTOR_SECONDS["S1"]].iloc[0] == pytest.approx(19.0)
    assert result[CORRECTED_SECTOR_SECONDS["S2"]].iloc[0] == pytest.approx(38.0)
    assert result[CORRECTED_SECTOR_SECONDS["S3"]].iloc[0] == pytest.approx(19.0)


def test_calculate_quali_performance_reports_two_driver_team_bests():
    laps = _laps(
        [
            ("Ferrari", "LEC", "SOFT", 80.0, 2),
            ("Ferrari", "HAM", "SOFT", 81.0, 5),
            ("McLaren", "NOR", "SOFT", 82.0, 8),
            ("McLaren", "PIA", "SOFT", 83.0, 11),
        ]
    )

    result = calculate_quali_performance(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=3.0,
    )

    assert result != "Wet"
    assert result.dominant_compound == "SOFT"
    assert result.reference_session_lap_order == 11
    assert result.quickest_teams["Team"].tolist() == ["Ferrari", "McLaren"]

    ferrari = result.quickest_teams.loc[result.quickest_teams["Team"] == "Ferrari"].iloc[0]
    assert {ferrari["Driver1"], ferrari["Driver2"]} == {"LEC", "HAM"}
    assert {int(ferrari["Driver1LapNumber"]), int(ferrari["Driver2LapNumber"])} == {2, 5}
    assert pd.notna(ferrari["Driver1QualifyingPart"])


def test_calculate_quali_performance_requires_dominant_compound():
    laps = _laps(
        [
            ("Ferrari", "LEC", "SOFT", 80.0, 2),
            ("Ferrari", "HAM", "MEDIUM", 81.0, 5),
            ("McLaren", "NOR", "SOFT", 82.0, 8),
            ("McLaren", "PIA", "MEDIUM", 83.0, 11),
        ]
    )

    with pytest.raises(ValueError, match="more than 50%"):
        calculate_quali_performance(
            laps,
            quick_lap_threshold=1.07,
            clean_min_time_delta_seconds=None,
            clean_mean_time_delta_seconds=3.0,
        )


def test_calculate_quali_performance_filters_used_tyres_when_enabled():
    laps = _laps(
        [
            ("Ferrari", "LEC", "SOFT", 80.0, 2, True),
            ("Ferrari", "HAM", "SOFT", 79.0, 5, False),
            ("McLaren", "NOR", "SOFT", 82.0, 8, True),
        ]
    )

    result = calculate_quali_performance(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=3.0,
        new_tyre_only=True,
        track_evolution_fit=LINEAR_TRACK_EVOLUTION_MODEL,
    )

    assert result != "Wet"
    assert result.quickest_drivers["Driver"].tolist() == ["LEC", "NOR"]


def test_calculate_quali_performance_can_use_lap_time_only_without_telemetry_gaps():
    laps = _laps(
        [
            ("Ferrari", "LEC", "SOFT", 80.0, 2, True, 1),
            ("Ferrari", "HAM", "SOFT", 81.0, 5, True, 2),
            ("McLaren", "NOR", "SOFT", 82.0, 8, True, 3),
        ]
    ).drop(columns=["MinTimeDeltaToDriverAhead", "MeanTimeDeltaToDriverAhead"])

    result = calculate_quali_performance(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=None,
        lap_time_only=True,
        top_driver_count=None,
        track_evolution_fit=LINEAR_TRACK_EVOLUTION_MODEL,
    )

    assert result != "Wet"
    assert result.dominant_compound == "SOFT"
    assert result.laps["IsCleanLap"].equals(result.laps["IsQuickLap"])
    assert set(result.quickest_drivers["Driver"]) == {"LEC", "HAM", "NOR"}


def test_has_clean_gap_columns_requires_requested_telemetry_column_with_data():
    laps = pd.DataFrame({"MeanTimeDeltaToDriverAhead": [pd.NA, 4.5]})

    assert _has_clean_gap_columns(
        laps,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=3.0,
    )
    assert not _has_clean_gap_columns(
        laps,
        clean_min_time_delta_seconds=3.0,
        clean_mean_time_delta_seconds=None,
    )
    assert not _has_clean_gap_columns(
        pd.DataFrame({"MeanTimeDeltaToDriverAhead": [pd.NA]}),
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=3.0,
    )


def test_calculate_quali_performance_uses_top_drivers_for_evolution_fit():
    laps = _laps(
        [
            ("Ferrari", "LEC", "SOFT", 80.0, 2, True, 1),
            ("Red Bull Racing", "VER", "SOFT", 70.0, 5, True, 3),
            ("McLaren", "NOR", "SOFT", 79.4, 8, True, 2),
        ]
    )

    result = calculate_quali_performance(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=3.0,
        top_driver_count=2,
        track_evolution_fit=LINEAR_TRACK_EVOLUTION_MODEL,
    )

    assert result != "Wet"
    assert result.evolution_drivers == ["LEC", "NOR"]
    assert result.evolution_rate_seconds_per_lap == pytest.approx(0.1)
    ver = result.quickest_drivers.loc[result.quickest_drivers["Driver"] == "VER"].iloc[0]
    assert ver[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS] == pytest.approx(69.7)


def test_calculate_quali_performance_can_use_exponential_evolution_fit():
    laps = _laps(
        [
            ("Ferrari", "LEC", "SOFT", 82.2, 2, True, 1),
            ("McLaren", "NOR", "SOFT", 80.8, 5, True, 2),
            ("Mercedes", "RUS", "SOFT", 80.2, 8, True, 3),
        ]
    )

    result = calculate_quali_performance(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=3.0,
        track_evolution_fit=EXPONENTIAL_TRACK_EVOLUTION_MODEL,
    )

    assert result != "Wet"
    assert result.evolution_fit_model == EXPONENTIAL_TRACK_EVOLUTION_MODEL
    assert result.evolution_fit_parameters["decay_rate"] > 0
    assert set(result.laps[TRACK_EVOLUTION_FIT_MODEL]) == {EXPONENTIAL_TRACK_EVOLUTION_MODEL}
    assert result.laps[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS].notna().all()


def test_calculate_quali_performance_can_fit_evolution_by_quick_lap_number():
    laps = pd.concat(
        [
            _driver_stint("Ferrari", "LEC", "SOFT", 80.0, 2, True, 1, "Q1"),
            _slow_laps("Ferrari", "LEC", [4, 5, 6]),
            _driver_stint("McLaren", "NOR", "SOFT", 79.0, 8, True, 2, "Q1"),
            _slow_laps("McLaren", "NOR", [10, 11, 12]),
            _driver_stint("Mercedes", "RUS", "SOFT", 78.0, 14, True, 3, "Q1"),
        ],
        ignore_index=True,
    )

    result = calculate_quali_performance(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=3.0,
        top_driver_count=None,
        track_evolution_fit=LINEAR_TRACK_EVOLUTION_MODEL,
        track_evolution_quick_lap_number=True,
    )

    assert result != "Wet"
    assert result.track_evolution_x_column == QUICK_LAP_NUMBER
    assert result.reference_session_lap_order == 3
    assert result.evolution_rate_seconds_per_lap == pytest.approx(1.0)
    corrected_times = result.quickest_drivers[
        TRACK_EVO_CORRECTED_LAP_TIME_SECONDS
    ].tolist()
    assert corrected_times == pytest.approx([78.0, 78.0, 78.0])


def test_calculate_quali_performance_keeps_existing_qualifying_part():
    laps = _laps(
        [
            ("Ferrari", "LEC", "SOFT", 80.0, 2, True, None, "Q1"),
            ("McLaren", "NOR", "SOFT", 79.7, 5, True, None, "Q2"),
        ]
    )

    result = calculate_quali_performance(
        laps,
        quick_lap_threshold=1.07,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=3.0,
        track_evolution_fit=LINEAR_TRACK_EVOLUTION_MODEL,
    )

    assert result != "Wet"
    assert set(result.quickest_drivers[QUALIFYING_PART]) == {"Q1", "Q2"}


def test_calculate_quali_performance_can_use_only_driver_last_quali_part():
    laps = pd.concat(
        [
            _driver_stint("Ferrari", "LEC", "SOFT", 80.0, 2, True, None, "Q1"),
            _driver_stint("Ferrari", "LEC", "SOFT", 85.0, 5, True, None, "Q2"),
            _driver_stint("McLaren", "NOR", "SOFT", 82.0, 8, True, None, "Q2"),
        ],
        ignore_index=True,
    )

    all_parts_result = calculate_quali_performance(
        laps,
        quick_lap_threshold=1.20,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=3.0,
        last_quali_part_only=False,
        track_evolution_fit=LINEAR_TRACK_EVOLUTION_MODEL,
    )
    last_part_result = calculate_quali_performance(
        laps,
        quick_lap_threshold=1.20,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=3.0,
        last_quali_part_only=True,
        track_evolution_fit=LINEAR_TRACK_EVOLUTION_MODEL,
    )

    assert all_parts_result != "Wet"
    assert last_part_result != "Wet"

    all_parts_lec = all_parts_result.quickest_drivers.loc[
        all_parts_result.quickest_drivers["Driver"] == "LEC"
    ].iloc[0]
    last_part_lec = last_part_result.quickest_drivers.loc[
        last_part_result.quickest_drivers["Driver"] == "LEC"
    ].iloc[0]
    assert all_parts_lec[QUALIFYING_PART] == "Q1"
    assert int(all_parts_lec["LapNumber"]) == 2
    assert last_part_lec[QUALIFYING_PART] == "Q2"
    assert int(last_part_lec["LapNumber"]) == 5
    assert set(
        last_part_result.eligible_laps.loc[
            last_part_result.eligible_laps["Driver"] == "LEC", QUALIFYING_PART
        ]
    ) == {"Q2"}


def test_last_quali_part_uses_latest_entered_part_even_without_push_lap():
    laps = pd.concat(
        [
            _driver_stint("Ferrari", "LEC", "SOFT", 80.0, 2, True, None, "Q2"),
            _driver_stint("Ferrari", "LEC", "SOFT", 100.0, 5, True, None, "Q3"),
            _driver_stint("Ferrari", "HAM", "SOFT", 81.0, 8, True, None, "Q3"),
            _driver_stint("McLaren", "NOR", "SOFT", 82.0, 11, True, None, "Q3"),
        ],
        ignore_index=True,
    )

    result = calculate_quali_performance(
        laps,
        quick_lap_threshold=1.20,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=3.0,
        last_quali_part_only=True,
        track_evolution_fit=LINEAR_TRACK_EVOLUTION_MODEL,
    )

    assert result != "Wet"
    assert "LEC" not in set(result.quickest_drivers["Driver"])
    assert not (
        (result.eligible_laps["Driver"] == "LEC")
        & (result.eligible_laps[QUALIFYING_PART] == "Q2")
    ).any()
    assert set(
        result.eligible_laps.loc[result.eligible_laps["Driver"] == "HAM", QUALIFYING_PART]
    ) == {"Q3"}


def test_last_quali_part_caps_split_labels_by_session_result_rank():
    laps = pd.concat(
        [
            _driver_stint("Cadillac", "PER", "SOFT", 80.0, 2, True, 19, "Q1"),
            _driver_stint("Cadillac", "PER", "SOFT", 79.0, 5, True, 19, "Q2"),
            _driver_stint("Ferrari", "LEC", "SOFT", 78.0, 8, True, 1, "Q3"),
        ],
        ignore_index=True,
    )
    laps.loc[
        (laps["Driver"] == "PER")
        & (laps[QUALIFYING_PART] == "Q2")
        & (laps["LapTime"] == pd.Timedelta(seconds=79.0)),
        "MeanTimeDeltaToDriverAhead",
    ] = 2.0

    result = calculate_quali_performance(
        laps,
        quick_lap_threshold=1.20,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=3.0,
        last_quali_part_only=True,
        track_evolution_fit=LINEAR_TRACK_EVOLUTION_MODEL,
    )

    assert result != "Wet"
    per = result.quickest_drivers.loc[result.quickest_drivers["Driver"] == "PER"].iloc[0]
    assert per[QUALIFYING_PART] == "Q1"
    assert int(per["LapNumber"]) == 2


def test_last_quali_part_clamps_over_ranked_laps_instead_of_dropping_them():
    laps = pd.concat(
        [
            _driver_stint("Haas F1 Team", "BEA", "SOFT", 80.0, 2, True, 13, "Q2"),
            _driver_stint("Haas F1 Team", "BEA", "SOFT", 79.0, 5, True, 13, "Q3"),
            _driver_stint("Ferrari", "LEC", "SOFT", 78.0, 8, True, 1, "Q3"),
        ],
        ignore_index=True,
    )

    result = calculate_quali_performance(
        laps,
        quick_lap_threshold=1.20,
        clean_min_time_delta_seconds=None,
        clean_mean_time_delta_seconds=3.0,
        last_quali_part_only=True,
        track_evolution_fit=LINEAR_TRACK_EVOLUTION_MODEL,
    )

    assert result != "Wet"
    bea = result.quickest_drivers.loc[result.quickest_drivers["Driver"] == "BEA"].iloc[0]
    assert bea[QUALIFYING_PART] == "Q2"
    assert int(bea["LapNumber"]) == 5


def test_team_fastest_and_average_rows_uses_two_driver_mean_or_single_result():
    quickest_drivers = pd.DataFrame(
        {
            "Team": ["Ferrari", "Ferrari", "McLaren"],
            "Driver": ["LEC", "HAM", "NOR"],
            "LapNumber": [2, 5, 8],
            QUALIFYING_PART: ["Q3", "Q2", "Q3"],
            TRACK_EVO_CORRECTED_LAP_TIME_SECONDS: [80.0, 82.0, 79.0],
        }
    )

    result = _team_fastest_and_average_rows(
        quickest_drivers,
        teammate_delta_threshold_percent=3.0,
    )

    ferrari_average = result.loc[
        (result["Team"] == "Ferrari") & (result["ResultType"] == "average")
    ].iloc[0]
    assert ferrari_average[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS] == pytest.approx(81.0)
    assert ferrari_average["Driver"] == "LEC/HAM"
    assert ferrari_average["DriverCount"] == 2

    mclaren_average = result.loc[
        (result["Team"] == "McLaren") & (result["ResultType"] == "average")
    ].iloc[0]
    assert mclaren_average[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS] == pytest.approx(79.0)
    assert mclaren_average["Driver"] == "NOR"
    assert mclaren_average["DriverCount"] == 1


def test_team_average_falls_back_to_fastest_when_teammate_delta_is_too_large():
    quickest_drivers = pd.DataFrame(
        {
            "Team": ["Ferrari", "Ferrari"],
            "Driver": ["LEC", "HAM"],
            "LapNumber": [2, 5],
            QUALIFYING_PART: ["Q3", "Q2"],
            TRACK_EVO_CORRECTED_LAP_TIME_SECONDS: [80.0, 82.0],
        }
    )

    result = _team_fastest_and_average_rows(
        quickest_drivers,
        teammate_delta_threshold_percent=1.0,
    )

    ferrari_average = result.loc[result["ResultType"] == "average"].iloc[0]
    assert ferrari_average[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS] == pytest.approx(80.0)
    assert ferrari_average["Driver"] == "LEC"
    assert ferrari_average["DriverCount"] == 1
    assert "used fastest" in ferrari_average["AverageModeNote"]


def test_team_best_sector_rows_uses_best_corrected_sector_across_team():
    eligible_laps = pd.DataFrame(
        {
            "Team": ["Ferrari", "Ferrari"],
            "Driver": ["LEC", "HAM"],
            "LapNumber": [2, 5],
            QUALIFYING_PART: ["Q3", "Q3"],
            CORRECTED_SECTOR_SECONDS["S1"]: [20.0, 21.0],
            CORRECTED_SECTOR_SECONDS["S2"]: [41.0, 39.0],
            CORRECTED_SECTOR_SECONDS["S3"]: [20.5, 20.0],
        }
    )

    result = _team_best_sector_rows(eligible_laps)

    ferrari = result.iloc[0]
    assert ferrari["ResultType"] == "best_sectors"
    assert ferrari[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS] == pytest.approx(79.0)
    assert "S1 LEC" in ferrari["AverageModeNote"]
    assert "S2 HAM" in ferrari["AverageModeNote"]
    assert "S3 HAM" in ferrari["AverageModeNote"]


def test_parse_race_range():
    assert _parse_race_range("[2, 6]") == (2, 6)
    assert _parse_race_range("2,6") == (2, 6)


def _laps(push_lap_specs) -> pd.DataFrame:
    frames = []
    for spec in push_lap_specs:
        team, driver, compound, lap_time, push_order, *rest = spec
        fresh_tyre = rest[0] if rest else True
        session_result_rank = rest[1] if len(rest) > 1 else None
        qualifying_part = rest[2] if len(rest) > 2 else None
        frames.append(
            _driver_stint(
                team,
                driver,
                compound,
                lap_time,
                push_order,
                fresh_tyre,
                session_result_rank,
                qualifying_part,
            )
        )
    return pd.concat(frames, ignore_index=True)


def _driver_stint(
    team: str,
    driver: str,
    compound: str,
    push_lap_time: float,
    push_order: int,
    fresh_tyre: bool,
    session_result_rank,
    qualifying_part,
) -> pd.DataFrame:
    lap_numbers = [push_order - 1, push_order, push_order + 1]
    lap_times = [100.0, push_lap_time, 120.0]
    return pd.DataFrame(
        {
            "Team": team,
            "Driver": driver,
            "Stint": 1,
            "LapNumber": lap_numbers,
            "LapTime": pd.to_timedelta(lap_times, unit="s"),
            "LapStartTime": pd.to_timedelta(lap_numbers, unit="m"),
            "PitOutTime": [pd.Timedelta(minutes=lap_numbers[0]), pd.NaT, pd.NaT],
            "PitInTime": [pd.NaT, pd.NaT, pd.Timedelta(minutes=lap_numbers[2])],
            "Compound": compound,
            "FreshTyre": [fresh_tyre, fresh_tyre, fresh_tyre],
            "SessionResultRank": [session_result_rank] * 3,
            QUALIFYING_PART: [qualifying_part] * 3,
            "MinTimeDeltaToDriverAhead": [None, 6.0, None],
            "MeanTimeDeltaToDriverAhead": [None, 6.0, None],
        }
    )


def _slow_laps(team: str, driver: str, lap_numbers: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Team": team,
            "Driver": driver,
            "Stint": 1,
            "LapNumber": lap_numbers,
            "LapTime": pd.to_timedelta([120.0] * len(lap_numbers), unit="s"),
            "LapStartTime": pd.to_timedelta(lap_numbers, unit="m"),
            "PitOutTime": [pd.NaT] * len(lap_numbers),
            "PitInTime": [pd.NaT] * len(lap_numbers),
            "Compound": ["SOFT"] * len(lap_numbers),
            "FreshTyre": [True] * len(lap_numbers),
            "SessionResultRank": [pd.NA] * len(lap_numbers),
            QUALIFYING_PART: ["Q1"] * len(lap_numbers),
            "MinTimeDeltaToDriverAhead": [None] * len(lap_numbers),
            "MeanTimeDeltaToDriverAhead": [6.0] * len(lap_numbers),
        }
    )
