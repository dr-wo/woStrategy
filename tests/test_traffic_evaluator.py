from __future__ import annotations

import numpy as np
import pandas as pd

from wostrategy.analysis.traffic import (
    TrafficEvaluationConfig,
    combine_lap_traffic_components,
    evaluate_direct_fastf1_style,
    evaluate_physical_traffic,
)


LENGTH = 1000.0
SPEED_KPH = 180.0


def _phase_frame(specs: list[tuple[str, int, float]]) -> pd.DataFrame:
    # Directly create timestamped constant-speed laps with starts shifted in
    # session time; this gives each driver the requested phase at t=0.
    rows = []
    for driver, official_lap, phase in specs:
        lap_start = -phase / (SPEED_KPH / 3.6)
        for time in np.arange(lap_start, lap_start + 20.01, 0.5):
            rows.append(
                {
                    "Driver": driver,
                    "SessionTime": time,
                    "LapNumber": official_lap,
                    "Speed": SPEED_KPH,
                }
            )
    return pd.DataFrame(rows)


def _sample(result, driver: str, time: float = 0.0):
    rows = result.samples.loc[
        result.samples["Driver"].eq(driver)
        & np.isclose(pd.to_numeric(result.samples["SessionTime"]), time)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def test_car_two_official_laps_behind_can_be_physically_1_5_seconds_behind():
    telemetry = _phase_frame([("TARGET", 10, 500.0), ("LAPPED", 8, 425.0)])
    result = evaluate_physical_traffic(telemetry, config=TrafficEvaluationConfig(LENGTH))
    row = _sample(result, "TARGET")
    assert row["PhysicalDriverBehind"] == "LAPPED"
    assert np.isclose(row["PhysicalTimeDeltaToDriverBehind"], 1.5)


def test_lapped_car_can_be_physically_ahead():
    telemetry = _phase_frame([("TARGET", 10, 500.0), ("LAPPED", 9, 600.0)])
    result = evaluate_physical_traffic(telemetry, config=TrafficEvaluationConfig(LENGTH))
    row = _sample(result, "TARGET")
    assert row["PhysicalDriverAhead"] == "LAPPED"
    assert np.isclose(row["PhysicalDistanceToDriverAhead"], 100.0)


def test_opposite_sides_of_finish_line_use_short_circular_gap():
    telemetry = _phase_frame([("TARGET", 4, 975.0), ("OTHER", 4, 25.0)])
    result = evaluate_physical_traffic(telemetry, config=TrafficEvaluationConfig(LENGTH))
    row = _sample(result, "TARGET")
    assert row["PhysicalDriverAhead"] == "OTHER"
    assert np.isclose(row["PhysicalDistanceToDriverAhead"], 50.0)


def test_same_other_car_has_correct_ahead_and_behind_direction():
    telemetry = _phase_frame(
        [("TARGET", 5, 500.0), ("AHEAD", 3, 575.0), ("BEHIND", 7, 450.0)]
    )
    result = evaluate_physical_traffic(telemetry, config=TrafficEvaluationConfig(LENGTH))
    row = _sample(result, "TARGET")
    assert row["PhysicalDriverAhead"] == "AHEAD"
    assert row["PhysicalDriverBehind"] == "BEHIND"


def test_shared_direct_estimator_matches_finish_line_anchored_constant_speed_case():
    telemetry = _phase_frame(
        [("TARGET", 5, 500.0), ("AHEAD", 5, 600.0), ("BEHIND", 5, 425.0)]
    )

    result = evaluate_direct_fastf1_style(telemetry)
    row = result.samples.loc[
        result.samples["Driver"].eq("TARGET")
        & np.isclose(pd.to_numeric(result.samples["SessionTime"]), 0.0)
    ].iloc[0]

    assert row["DirectDriverAhead"] == "AHEAD"
    assert np.isclose(row["DirectDistanceToDriverAhead"], 100.0)
    assert np.isclose(row["DirectTimeDeltaToDriverAhead"], 2.0)


def test_shared_direct_estimator_is_independent_of_batch_delivery_order():
    telemetry = _phase_frame([("TARGET", 5, 500.0), ("AHEAD", 5, 600.0)])

    batch = evaluate_direct_fastf1_style(telemetry).laps.sort_values(
        ["Driver", "LapNumber"]
    ).reset_index(drop=True)
    replay = evaluate_direct_fastf1_style(telemetry.iloc[::-1]).laps.sort_values(
        ["Driver", "LapNumber"]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(batch, replay)


def test_shared_direct_integration_does_not_drop_finish_line_sample_interval():
    rows = []
    for driver, first_start in (("TARGET", 0.0), ("AHEAD", -2.0)):
        for lap_number in (1, 2, 3):
            lap_start = first_start + (lap_number - 1) * 20.0
            for time in np.arange(lap_start, lap_start + 20.0, 0.5):
                rows.append(
                    {
                        "Driver": driver,
                        "SessionTime": time,
                        "LapNumber": lap_number,
                        "Speed": SPEED_KPH,
                    }
                )

    result = evaluate_direct_fastf1_style(pd.DataFrame(rows))
    target_lap = result.samples.loc[
        result.samples["Driver"].eq("TARGET")
        & result.samples["LapNumber"].eq(2)
    ]

    assert np.allclose(target_lap["DirectDistanceToDriverAhead"], 100.0)
    assert np.allclose(target_lap["DirectTimeDeltaToDriverAhead"], 2.0)


def test_combination_preserves_partial_direct_coverage_status():
    telemetry = pd.DataFrame(
        [
            {"Driver": "TARGET", "SessionTime": time, "LapNumber": 1, "Speed": 180.0}
            for time in np.arange(0.0, 5.0, 0.5)
        ]
        + [
            {"Driver": "OTHER", "SessionTime": time, "LapNumber": 1, "Speed": 180.0}
            for time in np.arange(0.0, 1.5, 0.5)
        ]
    )
    direct = evaluate_direct_fastf1_style(telemetry).laps

    selected = combine_lap_traffic_components(
        direct,
        pd.DataFrame(columns=["Driver", "LapNumber"]),
        group_columns=("Driver", "LapNumber"),
    )
    target = selected.loc[selected["Driver"].eq("TARGET")].iloc[0]

    assert target["DirectTrafficStatus"] == "partial_coverage"
    assert target["TrafficStatus"] == "partial_coverage"
    assert np.isclose(target["CoverageFraction"], 0.3)
