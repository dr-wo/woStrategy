from __future__ import annotations

import pandas as pd
import pytest

from wostrategy.model.track_evolution import TRACK_EVOLUTION_FIT_MODEL
from wostrategy.script.quali_performance_tracker import (
    CORRECTED_SECTOR_SECONDS,
    EXPONENTIAL_TRACK_EVOLUTION_MODEL,
    LINEAR_TRACK_EVOLUTION_MODEL,
    QUALIFYING_PART,
    TRACK_EVO_CORRECTION_SECONDS,
    TRACK_EVO_CORRECTED_LAP_TIME_SECONDS,
    _add_corrected_sector_times,
    _add_track_evolution_correction,
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
