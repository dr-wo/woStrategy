from __future__ import annotations

import numpy as np
import pandas as pd

from wostrategy.script.telemetry_intensity_validation import (
    PUSH_THRESHOLD,
    SLOW_ANCHOR_THRESHOLD,
    build_distance_profiles,
    summarize_lap_telemetry,
    timing_band,
)


def _telemetry_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Distance": [0.0, 10.0, 20.0, 30.0],
            "RelativeDistance": [0.0, 1 / 3, 2 / 3, 1.0],
            "SessionTime": pd.to_timedelta([0.00, 0.24, 0.48, 0.72], unit="s"),
            "Source": ["car", "car", "car", "car"],
            "Throttle": [0.0, 100.0, 50.0, 0.0],
            "Speed": [100.0, 200.0, 160.0, 200.0],
            "RPM": [9000.0, 11000.0, 10000.0, 9000.0],
            "nGear": [4, 7, 6, 7],
            "Brake": [False, False, False, False],
            "DRS": [0, 0, 0, 0],
        }
    )


def test_spatial_metrics_use_distance_and_report_native_resolution():
    result = summarize_lap_telemetry(_telemetry_frame())

    assert result["TelemetryUsable"]
    assert np.isclose(result["DistanceWeightedThrottlePct"], 50.0)
    assert np.isclose(result["FullThrottleDistancePct"], 100 / 3)
    assert np.isclose(result["HighSpeedOffThrottleNoBrakeDistancePct"], 100 / 3)
    assert np.isclose(result["DistanceCoverageFraction"], 1.0)
    assert np.isclose(result["NativeCarHz"], 1 / 0.24)


def test_timing_bands_keep_existing_push_boundary_and_fixed_slow_anchor():
    assert timing_band(PUSH_THRESHOLD) == "push_anchor"
    assert timing_band(PUSH_THRESHOLD + 0.001) == "ambiguous_non_push"
    assert timing_band(SLOW_ANCHOR_THRESHOLD) == "slow_non_push_anchor"


def test_distance_profiles_use_native_car_rows_only():
    telemetry = _telemetry_frame()
    inserted = telemetry.iloc[[1]].copy()
    inserted["Source"] = "pos"
    inserted["RelativeDistance"] = 0.5
    inserted["Throttle"] = 999.0
    cache = type("Cache", (), {"telemetry": pd.concat([telemetry, inserted])})()
    results = pd.DataFrame(
        [
            {
                "RunId": "run",
                "Round": 1,
                "EventName": "Event",
                "Driver": "DRV",
                "Stint": 1,
                "LapNumber": 1,
                "RelativePace": 1.0,
                "ExistingTimingLabel": "push_like",
                "DescriptiveTimingBand": "push_anchor",
                "TelemetryUsable": True,
            }
        ]
    )
    cache.telemetry["Driver"] = "DRV"
    cache.telemetry["LapNumber"] = 1.0

    profile = build_distance_profiles({1: cache}, results)

    midpoint = profile.loc[np.isclose(profile["RelativeDistance"], 0.5), "Throttle"]
    assert np.isclose(midpoint.iloc[0], 75.0)
    assert profile["Throttle"].max() <= 100.0
