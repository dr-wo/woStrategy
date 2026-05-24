from typing import Optional
import pandas as pd
import numpy as np

from wostrategy.core.session import Session


class FuelCorrection:
    pass

    def add_fuel_correction(self) -> None:
        """Add fuel correction column to laps data."""
        pass


class FixedRateFuelCorrection:
    def __init__(self, session: Session, init_fuel: int = 50, race_fuel: float = 105.0) -> None:

        self.session = session
        self.ref_init_fuel = init_fuel  # reference intial fuel level (default practice fuel)
        self.race_fuel = race_fuel  # in kg
        self.rate = 0.02            # seconds per lap per kg
        self.per_lap_rate = self._per_lap_rate()    # seconds per lap
        self.fuel_per_lap = self.race_fuel / self.session.race_lap_number  # in kg, how much fuel used per lap
        self._fuel_correct_reference = 2 * self.fuel_per_lap + 1  # fuel correction to what fuel level (in kg)

        self.init_fuel = self._get_init_fuel()

    def _per_lap_rate(self) -> float:
        """Calculate per-lap fuel consumption rate."""
        fuel_per_lap = self.race_fuel / self.session.race_lap_number  # in kg
        return fuel_per_lap * self.rate                   # in seconds per lap

    def _get_init_fuel(self) -> float:
        """Estimate initial fuel level based on fastest lap time."""
        if self.session.session_name in ['R']:
            return self.race_fuel
        elif self.session.session_name in ['SS']:
            # In sprint, init fuel is race / 305 * 100 km + 1 kg buffer
            return self.race_fuel / 305 * 100 + 1
        elif self.session.session_name in ['SQ', 'Q']:
            # In quali, init fuel is lap of the stint * fuel per lap + 1 kg buffer
            stint_length = self.session.laps['StintLapNumber'].max()
            return stint_length * self.fuel_per_lap + 1.
        elif self.session.session_name in ['FP1', 'FP2', 'FP3']:
            stint_length = self.session.laps['StintLapNumber'].max()
            return stint_length * self.fuel_per_lap + 1.
        return self.ref_init_fuel


    def update_rate(self, rate: float) -> None:
        """Update the fuel correction rate (seconds per lap)."""
        self.rate = rate

    def add_remaining_fuel_column(self) -> None:
        """Add remaining fuel column to laps data."""
        self.session.laps['RemainingFuel'] = self.init_fuel - (self.session.laps['StintLapNumber'] * (self.race_fuel / self.session.race_lap_number))

    def add_fuel_correction(self) -> None:
        """Add fuel correction column to laps data. Fuel correction is rate * stint lap number. """
        # First check if it is a low fuel run

        unit_time_correction = pd.to_timedelta(self.rate, unit='s')  # Convert seconds to nanoseconds
        print('unit time correction:', unit_time_correction)
        self.session.laps['FuelCorrection'] = unit_time_correction * self.session.laps['StintLapNumber']
