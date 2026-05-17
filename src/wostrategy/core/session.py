from __future__ import annotations

from math import ceil
from typing import Union

import fastf1
import pandas as pd
from fastf1.core import Session as FastF1Session


class Session:
    """Wrapper for FastF1 Session with chainable filter methods."""

    def __init__(
        self,
        year: int,
        round: Union[int, str],
        session_name: str,
        test: bool = False,
        **kwargs,
    ):
        self.year = year
        self.round = round
        self.session_name = session_name
        self.test = test
        self._session = self._load_session(**kwargs)
        self._original_laps = self._session.laps
        self.laps = self._session.laps
        self.lap_distance = self._session.get_circuit_info().marshal_sectors["Distance"].max()
        self.race_lap_number = max(ceil(3e5 / self.lap_distance), 78)
        self._add_stint_lap_number()

    def _load_session(self, **kwargs) -> FastF1Session:
        if not self.test:
            data = fastf1.get_session(self.year, self.round, self.session_name, **kwargs)
        else:
            data = fastf1.get_testing_event(self.year, self.round).get_session(
                self.session_name, **kwargs
            )
        data.load()
        self._fill_missing_outlap_laptimes(data)
        return data

    def _fill_missing_outlap_laptimes(self, session: FastF1Session) -> None:
        laps = session.laps
        required_columns = {"Driver", "LapTime", "PitOutTime", "LapStartTime"}
        if not required_columns.issubset(laps.columns):
            return

        for driver in laps["Driver"].dropna().unique():
            driver_laps = laps.loc[laps["Driver"] == driver].sort_values("LapStartTime")
            next_lap_start = driver_laps["LapStartTime"].shift(-1)
            fill_mask = (
                driver_laps["LapTime"].isna()
                & driver_laps["PitOutTime"].notna()
                & next_lap_start.notna()
            )
            if not fill_mask.any():
                continue

            filled_laptimes = next_lap_start[fill_mask] - driver_laps.loc[fill_mask, "PitOutTime"]
            filled_laptimes = filled_laptimes[filled_laptimes > pd.Timedelta(0)]
            if filled_laptimes.empty:
                continue

            session.laps.loc[filled_laptimes.index, "LapTime"] = filled_laptimes

    def _add_stint_lap_number(self) -> None:
        self.laps["StintLapNumber"] = self.laps.groupby(["Driver", "Stint"], sort=False).cumcount() + 1

    @property
    def data(self) -> FastF1Session:
        return self._session

    def __call__(self):
        return self._session

    def reset(self) -> "Session":
        self.laps = self._original_laps
        return self

    def drivers(self, drivers: list[str]) -> "Session":
        self.laps = self.laps.pick_drivers(drivers)
        return self

    def quicklaps(self, threshold: float = 1.07) -> "Session":
        self.laps = self.laps.pick_quicklaps(threshold)
        return self

    def compounds(self, compounds: list[str]) -> "Session":
        self.laps = self.laps.pick_compounds(compounds)
        return self

    def track_status(self, status: str) -> "Session":
        self.laps = self.laps.pick_track_status(status)
        return self

    def stint(self, stint: int) -> "Session":
        self.laps = self.laps[self.laps["Stint"] == stint]
        return self

    def clean_laps(self, car_ahead: float) -> "Session":
        keep_indices = []
        for lap_index, lap in self.laps.iterlaps():
            lap_tel = lap.get_car_data()
            if lap_tel.empty:
                continue
            if "DistanceToDriverAhead" not in lap_tel or "Speed" not in lap_tel:
                continue

            dist = lap_tel["DistanceToDriverAhead"]
            spd = lap_tel["Speed"]
            valid = (spd > 1) & dist.notna() & (dist > 0)
            timegap = pd.Series(float("inf"), index=lap_tel.index)
            timegap.loc[valid] = dist[valid] / (spd[valid] / 3.6)
            if (timegap > car_ahead).all():
                keep_indices.append(lap_index)
        self.laps = self.laps.loc[keep_indices]
        return self

    def effective_stint(self) -> "Session":
        self.laps["EffectiveStint"] = pd.NA
        self.laps["EffectiveStintLapNumber"] = pd.NA
        gap_threshold = pd.Timedelta(seconds=50)

        for driver in self.laps["Driver"].unique():
            print(f"Processing driver {driver}")
            driver_mask = self.laps["Driver"] == driver
            driver_laps = self.laps.loc[driver_mask]
            if driver_laps.empty:
                continue

            stint_order = (
                driver_laps.groupby("Stint", sort=False)["LapStartTime"]
                .min()
                .sort_values()
                .index
                .tolist()
            )
            print(f"  Stints found: {stint_order}")

            print("  Get rid of quali-sim stints based on lap info")
            quali_sim_stints = set()
            for stint in stint_order:
                print(
                    f"    Analyzing stint {stint} to check if it is quali lap, "
                    f"total lap count: {len(driver_laps.loc[driver_laps['Stint'] == stint])}"
                )
                stint_laps = driver_laps.loc[driver_laps["Stint"] == stint]
                lap_count = len(stint_laps)

                if lap_count <= 3:
                    quali_sim_stints.add(stint)
                    print(f"      Marking stint {stint} as quali-sim (only {lap_count} laps)")
                    continue

                valid_lap_times = stint_laps["LapTime"].dropna()
                if valid_lap_times.empty:
                    print(f"      No valid lap times for stint {stint}, marking as quali-sim")
                    continue

                best_lap_time = valid_lap_times.min()
                slow_laps = valid_lap_times[valid_lap_times > 1.07 * best_lap_time]
                if len(slow_laps) / len(valid_lap_times) > 0.33:
                    quali_sim_stints.add(stint)
                    print(
                        f"      Marking stint {stint} as quali-sim "
                        f"({len(slow_laps)}/{len(valid_lap_times)} slow laps)"
                    )

            effective_stint_number = 1
            prev_stint_end_time = None
            prev_stint_quali_sim = False
            prev_effective_stint = None
            effective_lap_offset: dict[int, int] = {}

            for stint in stint_order:
                stint_mask = driver_mask & (self.laps["Stint"] == stint)
                stint_laps = self.laps.loc[stint_mask]
                if stint_laps.empty:
                    continue

                is_quali_sim = stint in quali_sim_stints
                stint_start = stint_laps["PitOutTime"].min()
                stint_end = stint_laps["PitInTime"].max()
                tyre = (
                    stint_laps["Compound"].mode().iloc[0]
                    if not stint_laps["Compound"].mode().empty
                    else "Unknown"
                )
                print(
                    f"      Stint {stint} start: {stint_start}, end: {stint_end}, "
                    f"duration: {stint_end - stint_start}, previous_stint_end: {prev_stint_end_time}, "
                    f"is_quali_sim: {is_quali_sim}, tyre: {tyre}"
                )

                start_tyre_life = stint_laps["TyreLife"].min()
                combine_with_previous = (
                    prev_stint_end_time is not None
                    and not prev_stint_quali_sim
                    and not is_quali_sim
                    and (stint_start - prev_stint_end_time) < gap_threshold
                    and start_tyre_life == 1
                )

                if prev_stint_end_time and ((stint_start - prev_stint_end_time) < gap_threshold):
                    print(
                        f"        Stint {stint} has very small gap to previous stint "
                        f"({stint_start - prev_stint_end_time}, pre_end: {prev_stint_end_time}, "
                        f"start: {stint_start}), combining regardless of quali-sim status"
                    )

                only_gap_threshold = (
                    prev_stint_end_time is not None
                    and not prev_stint_quali_sim
                    and not is_quali_sim
                    and (stint_start - prev_stint_end_time) > gap_threshold
                )
                if prev_stint_end_time and only_gap_threshold:
                    print(
                        f"        Stint {stint} is NOT quali-sim but gap to previous stint "
                        f"is too large ({stint_start - prev_stint_end_time}), not combining"
                    )

                if combine_with_previous:
                    effective_stint = prev_effective_stint
                    lap_offset = effective_lap_offset[effective_stint]
                else:
                    effective_stint = effective_stint_number
                    effective_stint_number += 1
                    lap_offset = 0

                self.laps.loc[stint_mask, "EffectiveStint"] = effective_stint
                self.laps.loc[stint_mask, "EffectiveStintLapNumber"] = (
                    stint_laps["StintLapNumber"] + lap_offset
                )

                effective_lap_offset[effective_stint] = lap_offset + len(stint_laps)
                prev_stint_end_time = stint_laps["PitInTime"].max()
                prev_stint_quali_sim = is_quali_sim
                prev_effective_stint = effective_stint

        self.laps["EffectiveStint"] = self.laps["EffectiveStint"].astype("Int64")
        self.laps["EffectiveStintLapNumber"] = self.laps["EffectiveStintLapNumber"].astype("Int64")
        return self
