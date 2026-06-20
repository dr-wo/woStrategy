from __future__ import annotations

import pandas as pd

from wostrategy.core.telemetry_loader import (
    DistanceInterpolationTimeDeltaEstimator,
    TelemetryDataLoader,
    get_session_telemetry_cache_path,
    load_or_cache_session_telemetry,
    load_session_telemetry,
    summarize_lap_gap_metrics,
)
from wostrategy.core.session_loader import load_session_laps_with_telemetry_gap_summary


def test_distance_interpolation_time_delta_estimator_adds_seconds_column():
    telemetry = pd.DataFrame(
        {
            "Distance": [0.0, 50.0, 100.0, 150.0, 200.0],
            "DistanceToDriverAhead": [100.0, 75.0, 50.0, 25.0, 10.0],
            "Time": pd.to_timedelta([0.0, 5.0, 10.0, 15.0, 20.0], unit="s"),
        }
    )

    result = DistanceInterpolationTimeDeltaEstimator().add_time_delta(telemetry)

    assert result["TimeDeltaToDriverAhead"].tolist() == [10.0, 7.5, 5.0, 2.5, 1.0]


def test_telemetry_loader_wraps_lap_telemetry_and_adds_lap_metadata():
    session = FakeSession(
        [
            FakeLap(
                {
                    "Driver": "VER",
                    "Team": "Red Bull Racing",
                    "LapNumber": 1,
                    "LapTime": pd.Timedelta(seconds=80),
                },
                pd.DataFrame(
                    {
                        "Distance": [0.0, 100.0, 200.0],
                        "DistanceToDriverAhead": [50.0, 50.0, 50.0],
                        "Time": pd.to_timedelta([0.0, 10.0, 20.0], unit="s"),
                    }
                ),
            )
        ]
    )

    result = TelemetryDataLoader().load_session(
        session,
        year=2026,
        round_number=1,
        session_name="R",
    )

    assert len(result) == 3
    assert result["Year"].tolist() == [2026, 2026, 2026]
    assert result["Round"].tolist() == [1, 1, 1]
    assert result["SessionName"].tolist() == ["R", "R", "R"]
    assert result["Driver"].tolist() == ["VER", "VER", "VER"]
    assert result["TimeDeltaToDriverAhead"].tolist() == [5.0, 5.0, 5.0]


def test_load_session_telemetry_batches_sessions():
    def session_factory(round_number, session_name):
        assert round_number == 1
        assert session_name == "R"
        return FakeSession(
            [
                FakeLap(
                    {"Driver": "LEC", "LapNumber": 2},
                    pd.DataFrame(
                        {
                            "Distance": [0.0, 100.0],
                            "DistanceToDriverAhead": [100.0, 50.0],
                            "Time": pd.to_timedelta([0.0, 10.0], unit="s"),
                        }
                    ),
                )
            ]
        )

    result = load_session_telemetry(
        year=2026,
        rounds=[1],
        session_names=["R"],
        session_factory=session_factory,
    )

    assert result[["Year", "Round", "SessionName", "Driver"]].iloc[0].to_dict() == {
        "Year": 2026,
        "Round": 1,
        "SessionName": "R",
        "Driver": "LEC",
    }
    assert result["TimeDeltaToDriverAhead"].iloc[0] == 10.0


def test_load_or_cache_session_telemetry_writes_and_reuses_cache(tmp_path):
    first_session = FakeSession(
        [
            FakeLap(
                {"Driver": "LEC", "LapNumber": 1},
                pd.DataFrame(
                    {
                        "Distance": [0.0, 100.0],
                        "DistanceToDriverAhead": [50.0, 50.0],
                        "Time": pd.to_timedelta([0.0, 10.0], unit="s"),
                    }
                ),
            )
        ]
    )

    first = load_or_cache_session_telemetry(
        first_session,
        year=2026,
        round_number=1,
        session_name="R",
        cache_dir=tmp_path,
    )
    cache_path = get_session_telemetry_cache_path(
        year=2026,
        round_number=1,
        session_name="R",
        cache_dir=tmp_path,
    )

    assert cache_path.name == "2026_1_R"
    assert cache_path.exists()

    second_session = FakeSession(
        [
            FakeLap(
                {"Driver": "LEC", "LapNumber": 1},
                pd.DataFrame(
                    {
                        "Distance": [0.0, 100.0],
                        "DistanceToDriverAhead": [999.0, 999.0],
                        "Time": pd.to_timedelta([0.0, 10.0], unit="s"),
                    }
                ),
            )
        ]
    )
    second = load_or_cache_session_telemetry(
        second_session,
        year=2026,
        round_number=1,
        session_name="R",
        cache_dir=tmp_path,
    )

    assert second["DistanceToDriverAhead"].tolist() == first["DistanceToDriverAhead"].tolist()


def test_load_or_cache_session_telemetry_refreshes_stale_cache_without_time_delta(tmp_path):
    cache_path = get_session_telemetry_cache_path(
        year=2026,
        round_number=1,
        session_name="R",
        cache_dir=tmp_path,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Distance": [0.0, 100.0],
            "DistanceToDriverAhead": [999.0, 999.0],
            "Time": pd.to_timedelta([0.0, 10.0], unit="s"),
        }
    ).to_pickle(cache_path)
    session = FakeSession(
        [
            FakeLap(
                {"Driver": "LEC", "LapNumber": 1},
                pd.DataFrame(
                    {
                        "Distance": [0.0, 100.0],
                        "DistanceToDriverAhead": [50.0, 50.0],
                        "Time": pd.to_timedelta([0.0, 10.0], unit="s"),
                    }
                ),
            )
        ]
    )

    result = load_or_cache_session_telemetry(
        session,
        year=2026,
        round_number=1,
        session_name="R",
        cache_dir=tmp_path,
    )
    refreshed = pd.read_pickle(cache_path)

    assert result["DistanceToDriverAhead"].tolist() == [50.0, 50.0]
    assert result["TimeDeltaToDriverAhead"].tolist() == [5.0, 5.0]
    assert "TimeDeltaToDriverAhead" in refreshed.columns


def test_summarize_lap_gap_metrics_aggregates_by_lap():
    telemetry = pd.DataFrame(
        {
            "Year": [2026, 2026, 2026],
            "Round": [1, 1, 1],
            "SessionName": ["R", "R", "R"],
            "Driver": ["VER", "VER", "VER"],
            "LapNumber": [1, 1, 1],
            "TimeDeltaToDriverAhead": [2.0, 4.0, 6.0],
            "DistanceToDriverAhead": [20.0, 10.0, 30.0],
        }
    )

    result = summarize_lap_gap_metrics(telemetry)

    assert result["MinTimeDeltaToDriverAhead"].iloc[0] == 2.0
    assert result["MeanTimeDeltaToDriverAhead"].iloc[0] == 4.0
    assert result["MinDistanceToDriverAhead"].iloc[0] == 10.0
    assert result["MeanDistanceToDriverAhead"].iloc[0] == 20.0


def test_summarize_lap_gap_metrics_derives_driver_behind_gap():
    telemetry = pd.DataFrame(
        {
            "Year": [2026, 2026, 2026, 2026],
            "Round": [1, 1, 1, 1],
            "SessionName": ["R", "R", "R", "R"],
            "Driver": ["VER", "VER", "LEC", "LEC"],
            "DriverNumber": ["1", "1", "16", "16"],
            "DriverAhead": ["", "", "1", "1"],
            "LapNumber": [1, 1, 1, 1],
            "TimeDeltaToDriverAhead": [8.0, 9.0, 2.0, 4.0],
            "DistanceToDriverAhead": [80.0, 90.0, 20.0, 40.0],
        }
    )

    result = summarize_lap_gap_metrics(telemetry)
    ver = result.loc[result["Driver"] == "VER"].iloc[0]

    assert ver["MinTimeDeltaToDriverBehind"] == 2.0
    assert ver["MeanTimeDeltaToDriverBehind"] == 3.0
    assert ver["MinDistanceToDriverBehind"] == 20.0
    assert ver["MeanDistanceToDriverBehind"] == 30.0


def test_summarize_lap_gap_metrics_derives_lapped_car_behind_gap_from_track_position():
    telemetry = pd.DataFrame(
        {
            "Year": [2026, 2026, 2026, 2026],
            "Round": [7, 7, 7, 7],
            "SessionName": ["R", "R", "R", "R"],
            "Driver": ["GAS", "GAS", "VER", "VER"],
            "DriverNumber": ["10", "10", "1", "1"],
            "DriverAhead": ["1", "1", "1", "1"],
            "LapNumber": [50, 50, 51, 51],
            "SessionTime": pd.to_timedelta([100.0, 101.0, 100.0, 101.0], unit="s"),
            "Distance": [1000.0, 1100.0, 900.0, 980.0],
            "Speed": [180.0, 180.0, 180.0, 180.0],
            "TimeDeltaToDriverAhead": [15.0, 15.0, 19.0, 19.0],
            "DistanceToDriverAhead": [800.0, 800.0, 1000.0, 1000.0],
        }
    )

    result = summarize_lap_gap_metrics(telemetry)
    gas = result.loc[result["Driver"] == "GAS"].iloc[0]

    assert gas["MinDistanceToDriverBehind"] == 100.0
    assert gas["MeanDistanceToDriverBehind"] == 110.0
    assert gas["MinTimeDeltaToDriverBehind"] == 2.0
    assert gas["MeanTimeDeltaToDriverBehind"] == 2.2


def test_load_session_laps_with_telemetry_gap_summary_merges_metrics_and_caches(tmp_path):
    def session_factory(round_number, session_name):
        assert round_number == 1
        assert session_name == "R"
        return FakeLapSummarySession(
            pd.DataFrame(
                {
                    "Driver": ["VER"],
                    "LapNumber": [1],
                    "LapTime": [pd.Timedelta(seconds=80)],
                }
            ),
            {
                ("VER", 1): pd.DataFrame(
                    {
                        "Distance": [0.0, 100.0, 200.0],
                        "DistanceToDriverAhead": [100.0, 50.0, 25.0],
                        "Time": pd.to_timedelta([0.0, 10.0, 20.0], unit="s"),
                    }
                )
            },
        )

    result = load_session_laps_with_telemetry_gap_summary(
        year=2026,
        rounds=[1],
        session_names=["R"],
        session_factory=session_factory,
        telemetry_cache_dir=tmp_path,
    )

    assert result["MinTimeDeltaToDriverAhead"].iloc[0] == 2.5
    assert round(result["MeanTimeDeltaToDriverAhead"].iloc[0], 6) == 5.833333
    assert result["MinDistanceToDriverAhead"].iloc[0] == 25.0
    assert round(result["MeanDistanceToDriverAhead"].iloc[0], 6) == 58.333333
    assert (tmp_path / "2026_1_R").exists()


def test_load_session_laps_with_telemetry_gap_summary_adds_session_result_rank(tmp_path):
    def session_factory(round_number, session_name):
        assert round_number == 1
        assert session_name == "Q"
        return FakeLapSummarySession(
            pd.DataFrame(
                {
                    "Driver": ["VER", "LEC"],
                    "LapNumber": [1, 1],
                    "LapTime": [pd.Timedelta(seconds=80), pd.Timedelta(seconds=79)],
                }
            ),
            {
                ("VER", 1): _telemetry(),
                ("LEC", 1): _telemetry(),
            },
            results=pd.DataFrame(
                {
                    "Abbreviation": ["VER", "LEC"],
                    "Position": [1, 2],
                }
            ),
        )

    result = load_session_laps_with_telemetry_gap_summary(
        year=2026,
        rounds=[1],
        session_names=["Q"],
        session_factory=session_factory,
        telemetry_cache_dir=tmp_path,
    )

    ranks = result.set_index("Driver")["SessionResultRank"].to_dict()
    assert ranks == {"VER": 1, "LEC": 2}


class FakeLap(pd.Series):
    _metadata = ["_telemetry"]

    @property
    def _constructor(self):
        return FakeLap

    def __init__(self, data, telemetry):
        super().__init__(data)
        self._telemetry = telemetry

    def get_telemetry(self, **kwargs):
        assert kwargs == {}
        return self._telemetry


class FakeLaps:
    def __init__(self, laps):
        self._laps = laps

    def iterlaps(self):
        for index, lap in enumerate(self._laps):
            yield index, lap


class FakeSession:
    def __init__(self, laps):
        self.laps = FakeLaps(laps)


class FakeLapSummaryLaps(pd.DataFrame):
    _metadata = ["_telemetry_by_lap"]

    @property
    def _constructor(self):
        return FakeLapSummaryLaps

    def __init__(self, *args, telemetry_by_lap=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._telemetry_by_lap = telemetry_by_lap or {}

    def iterlaps(self):
        for index, row in self.iterrows():
            key = (row["Driver"], row["LapNumber"])
            yield index, FakeLap(row.to_dict(), self._telemetry_by_lap[key])


class FakeLapSummarySession:
    def __init__(self, laps, telemetry_by_lap, results=None):
        self.laps = FakeLapSummaryLaps(laps, telemetry_by_lap=telemetry_by_lap)
        self.results = results if results is not None else pd.DataFrame()


def _telemetry():
    return pd.DataFrame(
        {
            "Distance": [0.0, 100.0, 200.0],
            "DistanceToDriverAhead": [100.0, 50.0, 25.0],
            "Time": pd.to_timedelta([0.0, 10.0, 20.0], unit="s"),
        }
    )
