from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wostrategy.core.session import Session

LINEAR_FUEL_CONSUMPTION_MODEL = "linear"
FUEL_LAP_NUMBER_COLUMN = "FuelLapNumber"
FUEL_SLOPE_PARAMETER = "fuel_slope_seconds_per_lap"


@dataclass(frozen=True)
class FuelConsumptionFit:
    model_name: str
    parameters: dict[str, float]
    x_column: str
    y_column: str

    def to_summary(self) -> dict[str, float | str]:
        return {"model": self.model_name, **self.parameters}


class FuelConsumptionModel(ABC):
    name: str
    fit_label: str

    def fit(
        self,
        laps: pd.DataFrame,
        *,
        x_column: str = FUEL_LAP_NUMBER_COLUMN,
        y_column: str = "LapTimeSeconds",
    ) -> FuelConsumptionFit:
        fit_laps = laps.dropna(subset=[x_column, y_column])
        if len(fit_laps) < self.minimum_points:
            raise ValueError(
                f"Not enough laps to fit {self.name} fuel model "
                f"({len(fit_laps)} available, {self.minimum_points} required)."
            )

        x = fit_laps[x_column].to_numpy(dtype="float64")
        y = fit_laps[y_column].to_numpy(dtype="float64")
        return FuelConsumptionFit(
            model_name=self.name,
            parameters=self._fit_parameters(x, y),
            x_column=x_column,
            y_column=y_column,
        )

    @property
    def minimum_points(self) -> int:
        return 2

    @abstractmethod
    def _fit_parameters(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def predict_values(self, x: np.ndarray, fit: FuelConsumptionFit) -> np.ndarray:
        raise NotImplementedError

    def predict(self, x: float | pd.Series | np.ndarray, fit: FuelConsumptionFit) -> np.ndarray:
        return self.predict_values(np.asarray(x, dtype="float64"), fit)

    def equation_label(self, fit: FuelConsumptionFit) -> str:
        return f"{self.fit_label}: {fit.parameters}"


class LinearFuelConsumptionModel(FuelConsumptionModel):
    name = LINEAR_FUEL_CONSUMPTION_MODEL
    fit_label = "Linear fuel fit"

    def _fit_parameters(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        slope, intercept = np.polyfit(x, y, 1)
        return {
            FUEL_SLOPE_PARAMETER: float(slope),
            "intercept_seconds": float(intercept),
        }

    def predict_values(self, x: np.ndarray, fit: FuelConsumptionFit) -> np.ndarray:
        slope = fit.parameters[FUEL_SLOPE_PARAMETER]
        intercept = fit.parameters["intercept_seconds"]
        return (slope * x) + intercept

    def equation_label(self, fit: FuelConsumptionFit) -> str:
        slope = fit.parameters[FUEL_SLOPE_PARAMETER]
        intercept = fit.parameters["intercept_seconds"]
        return f"lap_time = {slope:.4f} * fuel_lap + {intercept:.3f}"


def get_fuel_consumption_model(name: str) -> FuelConsumptionModel:
    normalized = name.strip().lower()
    models: dict[str, FuelConsumptionModel] = {
        LINEAR_FUEL_CONSUMPTION_MODEL: LinearFuelConsumptionModel(),
    }
    try:
        return models[normalized]
    except KeyError as exc:
        options = ", ".join(sorted(models))
        raise ValueError(f"Unknown fuel consumption fit {name!r}. Options: {options}") from exc


def get_fuel_consumption_term_config(
    name: str = LINEAR_FUEL_CONSUMPTION_MODEL,
) -> dict[str, object]:
    normalized = name.strip().lower()
    if normalized == LINEAR_FUEL_CONSUMPTION_MODEL:
        return {
            "model": LINEAR_FUEL_CONSUMPTION_MODEL,
            "x_column": FUEL_LAP_NUMBER_COLUMN,
            "parameter": FUEL_SLOPE_PARAMETER,
            "label": "fuel_lap",
            "reference_value": 1.0,
        }
    options = LINEAR_FUEL_CONSUMPTION_MODEL
    raise ValueError(f"Unknown fuel consumption term {name!r}. Options: {options}")


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
        self.fuel_per_lap = self.race_fuel / self.session.race_lap_number
        self._fuel_correct_reference = 2 * self.fuel_per_lap + 1

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
        fuel_used = (
            self.session.laps['StintLapNumber']
            * (self.race_fuel / self.session.race_lap_number)
        )
        self.session.laps['RemainingFuel'] = self.init_fuel - fuel_used

    def add_fuel_correction(self) -> None:
        """Add fuel correction column to laps data. Fuel correction is rate * stint lap number. """
        # First check if it is a low fuel run

        unit_time_correction = pd.to_timedelta(self.rate, unit='s')
        print('unit time correction:', unit_time_correction)
        self.session.laps['FuelCorrection'] = (
            unit_time_correction * self.session.laps['StintLapNumber']
        )
