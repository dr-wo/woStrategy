from __future__ import annotations

import pandas as pd

from wostrategy.core.session import Session


def test_session_race_lap_number_falls_back_when_circuit_distance_is_missing():
    session = Session.__new__(Session)
    session._session = FakeFastF1Session(pd.NA)

    assert session._lap_distance() is None

    session.lap_distance = None
    assert session._race_lap_number() == 78


def test_session_race_lap_number_uses_valid_circuit_distance():
    session = Session.__new__(Session)
    session._session = FakeFastF1Session(5000.0)

    assert session._lap_distance() == 5000.0

    session.lap_distance = 5000.0
    assert session._race_lap_number() == 78


def test_load_session_disables_fastf1_cache_when_force_refresh_requested(monkeypatch):
    session = Session.__new__(Session)
    session.test = False
    session.year = 2026
    session.round = 10
    session.session_name = "R"
    session.force_refresh_session_cache = True
    fake_session = FakeLoadSession()
    cache_state = {"entered": False, "load_inside_disabled_cache": False}

    def fake_get_session(*args, **kwargs):
        assert args == (2026, 10, "R")
        assert kwargs == {}
        return fake_session

    class FakeDisabledCache:
        def __enter__(self):
            cache_state["entered"] = True
            fake_session.cache_disabled = True

        def __exit__(self, exc_type, exc, traceback):
            fake_session.cache_disabled = False

    monkeypatch.setattr("wostrategy.core.session.fastf1.get_session", fake_get_session)
    monkeypatch.setattr(
        "wostrategy.core.session.fastf1.Cache.disabled",
        lambda: FakeDisabledCache(),
    )
    monkeypatch.setattr(
        session,
        "_fill_missing_outlap_laptimes",
        lambda loaded_session: None,
    )

    loaded = session._load_session()

    assert loaded is fake_session
    assert cache_state["entered"]
    assert fake_session.loaded
    assert fake_session.load_inside_disabled_cache


class FakeFastF1Session:
    def __init__(self, lap_distance) -> None:
        self._lap_distance = lap_distance

    def get_circuit_info(self):
        return FakeCircuitInfo(self._lap_distance)


class FakeCircuitInfo:
    def __init__(self, lap_distance) -> None:
        self.marshal_sectors = pd.DataFrame({"Distance": [lap_distance]})


class FakeLoadSession:
    def __init__(self) -> None:
        self.loaded = False
        self.cache_disabled = False
        self.load_inside_disabled_cache = False

    def load(self) -> None:
        self.loaded = True
        self.load_inside_disabled_cache = self.cache_disabled
