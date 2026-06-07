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


class FakeFastF1Session:
    def __init__(self, lap_distance) -> None:
        self._lap_distance = lap_distance

    def get_circuit_info(self):
        return FakeCircuitInfo(self._lap_distance)


class FakeCircuitInfo:
    def __init__(self, lap_distance) -> None:
        self.marshal_sectors = pd.DataFrame({"Distance": [lap_distance]})
