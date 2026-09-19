from __future__ import annotations

import numpy as np
import pandas as pd

from wostrategy.analysis.long_run_performance import prepare_and_select_clean_air_laps
from wostrategy.analysis.traffic import (
    PHYSICAL_CIRCULAR,
    TrafficEvaluationConfig,
    combine_lap_traffic_components,
    evaluate_direct_fastf1_style,
    evaluate_physical_traffic,
)


def _nor_traffic_components() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for driver in ("NOR", "COINCIDENT"):
        for lap_number in range(7, 11):
            lap_start = (lap_number - 7) * 100.0
            for offset in np.arange(0.0, 100.0, 1.0):
                target_distance = offset * 50.0
                rows.append(
                    {
                        "Driver": driver,
                        "LapNumber": lap_number,
                        "SessionTime": lap_start + offset,
                        "Speed": 180.0,
                        "Distance": (
                            target_distance
                            if driver == "NOR"
                            else (target_distance + 150.0) % 5000.0
                        ),
                    }
                )
    telemetry = pd.DataFrame(rows)
    direct = evaluate_direct_fastf1_style(telemetry).laps
    physical = evaluate_physical_traffic(
        telemetry,
        config=TrafficEvaluationConfig(circuit_length_m=5000.0),
    ).laps
    return direct, physical


def _physical_laps() -> pd.DataFrame:
    rows = []
    for lap_number in range(7, 11):
        rows.append(
            {
                "Driver": "ANT",
                "LapNumber": lap_number,
                "PhysicalMeanTimeDeltaToDriverAhead": 4.0,
                "PhysicalMinTimeDeltaToDriverAhead": 4.0,
                "PhysicalMeanDistanceToDriverAhead": 200.0,
                "PhysicalMinDistanceToDriverAhead": 200.0,
                "PhysicalMeanTimeDeltaToDriverBehind": 3.0,
                "PhysicalMinTimeDeltaToDriverBehind": 3.0,
                "PhysicalMeanDistanceToDriverBehind": 150.0,
                "PhysicalMinDistanceToDriverBehind": 150.0,
                "TrafficStatus": "ok",
                "CoverageFraction": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _completed_laps(combined: pd.DataFrame, *, leader_lap: int) -> pd.DataFrame:
    completed = combined.loc[
        combined["Driver"].eq("NOR")
        | (combined["Driver"].eq("ANT") & combined["LapNumber"].le(leader_lap - 1))
    ].copy()
    completed["Year"] = 2026
    completed["Round"] = 14
    completed["SessionName"] = "R"
    completed["Team"] = completed["Driver"].map({"NOR": "McLaren", "ANT": "Mercedes"})
    completed["Stint"] = 1
    completed["LapTime"] = pd.to_timedelta(90.0, unit="s")
    completed["Compound"] = "MEDIUM"
    completed["PitOutTime"] = pd.NaT
    completed["PitInTime"] = pd.NaT
    return completed


def _select(combined: pd.DataFrame, *, leader_lap: int) -> pd.DataFrame:
    _, selected = prepare_and_select_clean_air_laps(
        _completed_laps(combined, leader_lap=leader_lap),
        min_clean_air_laps=4,
        clean_mean_time_delta_seconds=2.5,
        clean_mean_time_delta_behind_seconds=1.0,
        quick_lap_threshold=1.1,
    )
    return selected


def test_r14_zero_direct_candidate_falls_back_to_physical_and_preserves_causal_cut():
    direct, nor_physical = _nor_traffic_components()
    physical = pd.concat([nor_physical, _physical_laps()], ignore_index=True)
    combined = combine_lap_traffic_components(
        direct,
        physical,
        group_columns=("Driver", "LapNumber"),
    )

    nor = combined.loc[combined["Driver"].eq("NOR")].sort_values("LapNumber")
    assert nor["DirectMeanTimeDeltaToDriverAhead"].isna().all()
    assert nor["PhysicalMeanTimeDeltaToDriverAhead"].gt(2.5).all()
    assert nor["MeanTimeDeltaToDriverAhead"].equals(
        nor["PhysicalMeanTimeDeltaToDriverAhead"]
    )
    assert nor["MeanTimeDeltaToDriverAheadMethod"].eq(PHYSICAL_CIRCULAR).all()

    leader_l10 = _select(combined, leader_lap=10)
    assert leader_l10.loc[leader_l10["Driver"].eq("NOR"), "LapNumber"].tolist() == [
        7,
        8,
        9,
        10,
    ]
    assert leader_l10.loc[leader_l10["Driver"].eq("ANT")].empty
    assert _completed_laps(combined, leader_lap=10).loc[
        lambda laps: laps["Driver"].eq("ANT"), "LapNumber"
    ].tolist() == [7, 8, 9]

    leader_l11 = _select(combined, leader_lap=11)
    assert leader_l11.loc[leader_l11["Driver"].eq("ANT"), "LapNumber"].tolist() == [
        7,
        8,
        9,
        10,
    ]
