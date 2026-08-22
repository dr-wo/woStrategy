from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from wostrategy.algorithm.monte_carlo_race_performance import (
    BASELINE_DRIVER,
    FUEL_PROXY_LAPS_REMAINING,
)


class CorrectionComponent(Protocol):
    """A physical correction component that exposes local parameter sensitivity."""

    name: str
    parameter_names: tuple[str, ...]

    def evaluate(self, observations: pd.DataFrame, parameters: Mapping[str, float]) -> np.ndarray:
        """Return this component's correction contribution for every observation."""

    def sensitivity(
        self, observations: pd.DataFrame, parameters: Mapping[str, float]
    ) -> np.ndarray:
        """Optionally return exact local Jacobian columns."""


@dataclass(frozen=True)
class LinearFuelModel:
    name: str = "FuelModel"
    parameter_names: tuple[str, ...] = ("fuel_rate",)

    def evaluate(self, observations, parameters):
        return float(parameters["fuel_rate"]) * observations[FUEL_PROXY_LAPS_REMAINING].to_numpy(float)

    def sensitivity(self, observations, parameters):
        return observations[[FUEL_PROXY_LAPS_REMAINING]].to_numpy(float)


@dataclass(frozen=True)
class LinearTrackEvolutionModel:
    race_lap_ref: float
    name: str = "TrackEvolutionModel"
    parameter_names: tuple[str, ...] = ("track_rate",)

    def evaluate(self, observations, parameters):
        return float(parameters["track_rate"]) * (
            observations["LapNumber"].to_numpy(float) - self.race_lap_ref
        )

    def sensitivity(self, observations, parameters):
        return (
            observations[["LapNumber"]].to_numpy(float) - self.race_lap_ref
        )


@dataclass(frozen=True)
class IdentifiabilityDiagnostics:
    parameter_names: tuple[str, ...]
    component_names: tuple[str, ...]
    observation_count: int
    baseline_group_count: int
    rank: int
    singular_values: tuple[float, ...]
    normalized_singular_values: tuple[float, ...]
    condition_number: float
    sensitivity_correlation: tuple[tuple[float, ...], ...]
    projection: str = "candidate_profiled_baseline_group_mean"
    scope: str = "local_jacobian"


def numerical_jacobian(
    component: CorrectionComponent,
    observations: pd.DataFrame,
    parameters: Mapping[str, float],
    *,
    relative_step: float = 1e-6,
) -> np.ndarray:
    """Central-difference sensitivity; analytical components use the same interface."""
    columns = []
    for name in component.parameter_names:
        value = float(parameters[name])
        step = relative_step * max(abs(value), 1.0)
        high = dict(parameters); high[name] = value + step
        low = dict(parameters); low[name] = value - step
        columns.append((component.evaluate(observations, high)
                        - component.evaluate(observations, low)) / (2 * step))
    return np.column_stack(columns)


def component_jacobian(
    component: CorrectionComponent,
    observations: pd.DataFrame,
    parameters: Mapping[str, float],
    *,
    relative_step: float = 1e-6,
) -> np.ndarray:
    """Use an analytical sensitivity when supplied, otherwise finite differences."""
    analytical = getattr(component, "sensitivity", None)
    if analytical is not None:
        result = np.asarray(analytical(observations, parameters), dtype=float)
        if result.shape != (len(observations), len(component.parameter_names)):
            raise ValueError(
                f"{component.name} sensitivity shape {result.shape} does not match "
                f"({len(observations)}, {len(component.parameter_names)})."
            )
        return result
    return numerical_jacobian(
        component, observations, parameters, relative_step=relative_step
    )


def baseline_project(matrix: np.ndarray, group_codes: np.ndarray) -> np.ndarray:
    projected = np.asarray(matrix, dtype=float).copy()
    for code in range(int(group_codes.max()) + 1):
        mask = group_codes == code
        projected[mask] -= projected[mask].mean(axis=0, keepdims=True)
    return projected


def diagnose_correction_identifiability(
    observations: pd.DataFrame,
    components: Sequence[CorrectionComponent],
    parameters: Mapping[str, float],
    *,
    baseline_group: str = BASELINE_DRIVER,
    relative_step: float = 1e-6,
) -> IdentifiabilityDiagnostics:
    group_columns = ("Team", "Driver") if baseline_group == BASELINE_DRIVER else ("Team",)
    labels = observations[list(group_columns)].astype(str).agg("||".join, axis=1)
    group_codes, groups = pd.factorize(labels, sort=True)
    matrices = [component_jacobian(component, observations, parameters,
                                   relative_step=relative_step)
                for component in components]
    names = tuple(name for component in components for name in component.parameter_names)
    component_names = tuple(component.name for component in components
                            for _ in component.parameter_names)
    projected = baseline_project(np.column_stack(matrices), group_codes)
    singular = np.linalg.svd(projected, compute_uv=False)
    largest = float(singular[0]) if len(singular) else 0.0
    tolerance = max(projected.shape, default=0) * np.finfo(float).eps * largest
    rank = int(np.sum(singular > tolerance))
    normalized = singular / largest if largest > 0 else np.zeros_like(singular)
    condition = (float(singular[0] / singular[-1])
                 if len(singular) and singular[-1] > tolerance else float("inf"))
    norms = np.linalg.norm(projected, axis=0)
    normalized_columns = np.divide(projected, norms, out=np.zeros_like(projected), where=norms > 0)
    correlation = normalized_columns.T @ normalized_columns
    return IdentifiabilityDiagnostics(
        names, component_names, len(observations), len(groups), rank,
        tuple(map(float, singular)), tuple(map(float, normalized)), condition,
        tuple(tuple(map(float, row)) for row in correlation),
    )


def diagnose_retro_fuel_track(observations: pd.DataFrame, *, baseline_group: str):
    reference = float(observations["LapNumber"].mean())
    return diagnose_correction_identifiability(
        observations,
        (LinearFuelModel(), LinearTrackEvolutionModel(reference)),
        {"fuel_rate": 0.05, "track_rate": 0.02},
        baseline_group=baseline_group,
    )
