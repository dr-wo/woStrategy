from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wostrategy.core.session import Session

LINEAR_TYRE_DEGRADATION_MODEL = "linear"
TYRE_AGE_LAPS_COLUMN = "TyreAgeLaps"
TYRE_SLOPE_PARAMETER = "tyre_slope_seconds_per_lap"


@dataclass(frozen=True)
class TyreDegradationFit:
    model_name: str
    parameters: dict[str, float]
    x_column: str
    y_column: str

    def to_summary(self) -> dict[str, float | str]:
        return {"model": self.model_name, **self.parameters}


class TyreDegradationModel(ABC):
    name: str
    fit_label: str

    def fit(
        self,
        laps: pd.DataFrame,
        *,
        x_column: str = TYRE_AGE_LAPS_COLUMN,
        y_column: str = "LapTimeSeconds",
    ) -> TyreDegradationFit:
        fit_laps = laps.dropna(subset=[x_column, y_column])
        if len(fit_laps) < self.minimum_points:
            raise ValueError(
                f"Not enough laps to fit {self.name} tyre model "
                f"({len(fit_laps)} available, {self.minimum_points} required)."
            )

        x = fit_laps[x_column].to_numpy(dtype="float64")
        y = fit_laps[y_column].to_numpy(dtype="float64")
        return TyreDegradationFit(
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
    def predict_values(self, x: np.ndarray, fit: TyreDegradationFit) -> np.ndarray:
        raise NotImplementedError

    def predict(self, x: float | pd.Series | np.ndarray, fit: TyreDegradationFit) -> np.ndarray:
        return self.predict_values(np.asarray(x, dtype="float64"), fit)

    def equation_label(self, fit: TyreDegradationFit) -> str:
        return f"{self.fit_label}: {fit.parameters}"


class LinearTyreDegradationModel(TyreDegradationModel):
    name = LINEAR_TYRE_DEGRADATION_MODEL
    fit_label = "Linear tyre fit"

    def _fit_parameters(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        slope, intercept = np.polyfit(x, y, 1)
        return {
            TYRE_SLOPE_PARAMETER: float(slope),
            "intercept_seconds": float(intercept),
        }

    def predict_values(self, x: np.ndarray, fit: TyreDegradationFit) -> np.ndarray:
        slope = fit.parameters[TYRE_SLOPE_PARAMETER]
        intercept = fit.parameters["intercept_seconds"]
        return (slope * x) + intercept

    def equation_label(self, fit: TyreDegradationFit) -> str:
        slope = fit.parameters[TYRE_SLOPE_PARAMETER]
        intercept = fit.parameters["intercept_seconds"]
        return f"lap_time = {slope:.4f} * tyre_age + {intercept:.3f}"


def get_tyre_degradation_model(name: str) -> TyreDegradationModel:
    normalized = name.strip().lower()
    models: dict[str, TyreDegradationModel] = {
        LINEAR_TYRE_DEGRADATION_MODEL: LinearTyreDegradationModel(),
    }
    try:
        return models[normalized]
    except KeyError as exc:
        options = ", ".join(sorted(models))
        raise ValueError(f"Unknown tyre degradation fit {name!r}. Options: {options}") from exc


def get_tyre_degradation_term_config(
    name: str = LINEAR_TYRE_DEGRADATION_MODEL,
) -> dict[str, object]:
    normalized = name.strip().lower()
    if normalized == LINEAR_TYRE_DEGRADATION_MODEL:
        return {
            "model": LINEAR_TYRE_DEGRADATION_MODEL,
            "x_column": TYRE_AGE_LAPS_COLUMN,
            "parameter": TYRE_SLOPE_PARAMETER,
            "label": "tyre_age",
            "reference_value": 0.0,
        }
    options = LINEAR_TYRE_DEGRADATION_MODEL
    raise ValueError(f"Unknown tyre degradation term {name!r}. Options: {options}")


def main() -> None:
    year = 2021
    round_number = 21

    for session_name in ["FP1", "FP2", "FP3"]:
        session = Session(year, round_number, session_name)
        session.quicklaps()


if __name__ == "__main__":
    main()
