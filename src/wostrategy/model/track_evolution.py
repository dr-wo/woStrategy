from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TRACK_EVO_CORRECTION_SECONDS = "track_evo_correction_seconds"
TRACK_EVOLUTION_FIT_MODEL = "track_evolution_fit_model"
TRACK_EVOLUTION_SECONDS_PER_LAP = "track_evolution_seconds_per_lap"
TRACK_EVO_CORRECTED_LAP_TIME = "track_evo_corrected_lap_time"
TRACK_EVO_CORRECTED_LAP_TIME_SECONDS = "track_evo_corrected_lap_time_seconds"

LINEAR_TRACK_EVOLUTION_MODEL = "linear"
EXPONENTIAL_TRACK_EVOLUTION_MODEL = "exponential"
TRACK_X_COLUMN = "LapNumber"
TRACK_SLOPE_PARAMETER = "track_slope_seconds_per_lap"
TRACK_AMPLITUDE_PARAMETER = "track_amplitude_seconds"
TRACK_DECAY_PARAMETER = "track_decay_rate"


@dataclass(frozen=True)
class TrackEvolutionFit:
    model_name: str
    parameters: dict[str, float]
    x_column: str
    y_column: str
    slope_unit: str

    @property
    def evolution_rate_seconds_per_lap(self) -> float:
        if "slope" not in self.parameters:
            return float("nan")
        return float(-self.parameters["slope"])

    def to_summary(self) -> dict[str, float | str]:
        return {
            "model": self.model_name,
            "slope_unit": self.slope_unit,
            **self.parameters,
        }


class TrackEvolutionModel(ABC):
    name: str
    fit_label: str

    def fit(
        self,
        laps: pd.DataFrame,
        *,
        x_column: str = "SessionLapOrder",
        y_column: str = "LapTimeSeconds",
        slope_unit: str = "s/lap",
    ) -> TrackEvolutionFit:
        fit_laps = laps.dropna(subset=[x_column, y_column])
        if len(fit_laps) < self.minimum_points:
            raise ValueError(
                f"Not enough laps to fit {self.name} track evolution "
                f"({len(fit_laps)} available, {self.minimum_points} required)."
            )

        x = fit_laps[x_column].to_numpy(dtype="float64")
        y = fit_laps[y_column].to_numpy(dtype="float64")
        parameters = self._fit_parameters(x, y)
        return TrackEvolutionFit(
            model_name=self.name,
            parameters=parameters,
            x_column=x_column,
            y_column=y_column,
            slope_unit=slope_unit,
        )

    @property
    def minimum_points(self) -> int:
        return 2

    @abstractmethod
    def _fit_parameters(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def predict_values(self, x: np.ndarray, fit: TrackEvolutionFit) -> np.ndarray:
        raise NotImplementedError

    def predict(self, x: float | pd.Series | np.ndarray, fit: TrackEvolutionFit) -> np.ndarray:
        return self.predict_values(np.asarray(x, dtype="float64"), fit)

    def correction_seconds(
        self,
        x: pd.Series | np.ndarray,
        *,
        fit: TrackEvolutionFit,
        reference_x: float,
    ) -> np.ndarray:
        predicted = self.predict(x, fit)
        reference_prediction = float(self.predict(reference_x, fit))
        return predicted - reference_prediction

    def equation_label(self, fit: TrackEvolutionFit) -> str:
        return f"{self.fit_label}: {fit.parameters}"


class LinearTrackEvolutionModel(TrackEvolutionModel):
    name = LINEAR_TRACK_EVOLUTION_MODEL
    fit_label = "Linear fit"

    def _fit_parameters(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        slope, intercept = np.polyfit(x, y, 1)
        return {
            "slope": float(slope),
            "intercept_seconds": float(intercept),
        }

    def predict_values(self, x: np.ndarray, fit: TrackEvolutionFit) -> np.ndarray:
        slope = fit.parameters["slope"]
        intercept = fit.parameters["intercept_seconds"]
        return (slope * x) + intercept

    def equation_label(self, fit: TrackEvolutionFit) -> str:
        slope = fit.parameters["slope"]
        intercept = fit.parameters["intercept_seconds"]
        return (
            f"lap_time = {slope:.4f} * x + {intercept:.3f}\n"
            f"track development = {-slope:.4f} {fit.slope_unit}"
        )


class ExponentialTrackEvolutionModel(TrackEvolutionModel):
    name = EXPONENTIAL_TRACK_EVOLUTION_MODEL
    fit_label = "Exponential fit"

    @property
    def minimum_points(self) -> int:
        return 3

    def _fit_parameters(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        x_span = float(np.nanmax(x) - np.nanmin(x))
        if x_span <= 0:
            raise ValueError("Cannot fit exponential track evolution with zero x span.")

        # For each k, y = A * exp(-k*x) + B is linear in A and B.
        lower_k = 1e-6 / x_span
        upper_k = 10.0 / x_span
        k_grid = np.geomspace(lower_k, upper_k, num=400)
        best_error = float("inf")
        best_parameters: dict[str, float] | None = None

        for k in k_grid:
            basis = np.exp(-k * x)
            design = np.column_stack([basis, np.ones_like(x)])
            try:
                a, b = np.linalg.lstsq(design, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

            residuals = y - ((a * basis) + b)
            error = float(np.mean(residuals**2))
            if error < best_error:
                best_error = error
                best_parameters = {
                    "amplitude_seconds": float(a),
                    "decay_rate": float(k),
                    "offset_seconds": float(b),
                    "rmse_seconds": float(np.sqrt(error)),
                }

        if best_parameters is None:
            raise ValueError("Could not fit exponential track evolution.")
        return best_parameters

    def predict_values(self, x: np.ndarray, fit: TrackEvolutionFit) -> np.ndarray:
        amplitude = fit.parameters["amplitude_seconds"]
        decay_rate = fit.parameters["decay_rate"]
        offset = fit.parameters["offset_seconds"]
        return (amplitude * np.exp(-decay_rate * x)) + offset

    def equation_label(self, fit: TrackEvolutionFit) -> str:
        amplitude = fit.parameters["amplitude_seconds"]
        decay_rate = fit.parameters["decay_rate"]
        offset = fit.parameters["offset_seconds"]
        rmse = fit.parameters["rmse_seconds"]
        return (
            f"lap_time = {amplitude:.3f} * e^(-{decay_rate:.5f}x) + {offset:.3f}\n"
            f"fit rmse = {rmse:.3f}s"
        )


def get_track_evolution_model(name: str) -> TrackEvolutionModel:
    normalized = name.strip().lower()
    models: dict[str, TrackEvolutionModel] = {
        LINEAR_TRACK_EVOLUTION_MODEL: LinearTrackEvolutionModel(),
        EXPONENTIAL_TRACK_EVOLUTION_MODEL: ExponentialTrackEvolutionModel(),
    }
    try:
        return models[normalized]
    except KeyError as exc:
        options = ", ".join(sorted(models))
        raise ValueError(f"Unknown track evolution fit {name!r}. Options: {options}") from exc


def get_track_evolution_term_config(
    name: str = LINEAR_TRACK_EVOLUTION_MODEL,
    *,
    x_column: str = TRACK_X_COLUMN,
) -> dict[str, object]:
    normalized = name.strip().lower()
    if normalized == LINEAR_TRACK_EVOLUTION_MODEL:
        return {
            "model": LINEAR_TRACK_EVOLUTION_MODEL,
            "x_column": x_column,
            "parameter": TRACK_SLOPE_PARAMETER,
            "label": "track_x",
        }
    if normalized == EXPONENTIAL_TRACK_EVOLUTION_MODEL:
        return {
            "model": EXPONENTIAL_TRACK_EVOLUTION_MODEL,
            "x_column": x_column,
            "amplitude_parameter": TRACK_AMPLITUDE_PARAMETER,
            "decay_parameter": TRACK_DECAY_PARAMETER,
            "label": "track_x",
        }
    options = ", ".join([LINEAR_TRACK_EVOLUTION_MODEL, EXPONENTIAL_TRACK_EVOLUTION_MODEL])
    raise ValueError(f"Unknown track evolution term {name!r}. Options: {options}")


def dominant_compound(compounds: pd.Series, *, require_majority: bool) -> str | None:
    shares = compounds.value_counts(normalize=True, dropna=True)
    if shares.empty:
        if require_majority:
            raise ValueError("No compounds available to determine dominant compound.")
        return None

    if shares.iloc[0] > 0.5:
        return str(shares.index[0])

    if require_majority:
        counts = compounds.value_counts(dropna=True).to_dict()
        raise ValueError(f"No compound was used by more than 50% of dry push laps: {counts}")
    return None


def fit_compound_track_evolution(
    push_laps: pd.DataFrame,
    *,
    compound: str,
    model: TrackEvolutionModel,
    x_column: str = "SessionLapOrder",
    y_column: str = "LapTimeSeconds",
    slope_unit: str = "s/lap",
) -> TrackEvolutionFit:
    compound_laps = push_laps.loc[push_laps["Compound"] == compound]
    try:
        return model.fit(
            compound_laps,
            x_column=x_column,
            y_column=y_column,
            slope_unit=slope_unit,
        )
    except ValueError as exc:
        raise ValueError(
            f"Not enough {compound} push laps to calculate {model.name} track evolution."
        ) from exc


def add_track_evolution_correction(
    laps: pd.DataFrame,
    *,
    model: TrackEvolutionModel,
    fit: TrackEvolutionFit,
    reference_session_lap_order: float,
) -> pd.DataFrame:
    required_columns = {fit.x_column, fit.y_column}
    _require_columns(laps, required_columns)

    corrected = laps.copy()
    corrected[TRACK_EVOLUTION_FIT_MODEL] = model.name
    corrected[TRACK_EVOLUTION_SECONDS_PER_LAP] = fit.evolution_rate_seconds_per_lap
    corrected[TRACK_EVO_CORRECTION_SECONDS] = model.correction_seconds(
        corrected[fit.x_column],
        fit=fit,
        reference_x=reference_session_lap_order,
    )
    corrected[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS] = (
        corrected[fit.y_column] - corrected[TRACK_EVO_CORRECTION_SECONDS]
    )
    corrected[TRACK_EVO_CORRECTED_LAP_TIME] = pd.to_timedelta(
        corrected[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS], unit="s"
    )
    return corrected


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Laps are missing required columns: {', '.join(sorted(missing))}")
