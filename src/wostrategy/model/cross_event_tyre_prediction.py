from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from wostrategy.model.observation_weighting import (
    ObservationWeightPolicy,
    UniformWeightPolicy,
    get_weight_policy,
)


MODEL_VERSION = "linear_v1.1"
COMPOUNDS = ("C1", "C2", "C3", "C4", "C5")
ADJACENT_GAPS = ("C1-C2", "C2-C3", "C3-C4", "C4-C5")
PERFORMANCE_FEATURES = ("tyre_stress", "asphalt_grip")
DEGRADATION_FEATURES = ("tyre_stress", "asphalt_abrasion", "asphalt_grip")
# Kept as the union used by the training-data join and prediction payload.
FEATURES = DEGRADATION_FEATURES


@dataclass(frozen=True)
class ReadinessDiagnostics:
    ready: bool
    reasons: tuple[str, ...]
    observed_compounds: tuple[str, ...]
    feature_values: dict[str, tuple[float, ...]]
    observation_count: int
    informative_observation_count: int
    design_rank: int
    parameter_count: int
    comparison_edges: tuple[tuple[str, str], ...] = ()
    comparison_components: tuple[tuple[str, ...], ...] = ()
    adjacent_gap_counts: dict[str, int] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Prediction:
    mean: float
    std: float | None = None


class TyrePredictionModel(Protocol):
    name: str

    def fit(self, observations: pd.DataFrame) -> None: ...

    def predict(self, **kwargs: object) -> Prediction: ...

    def save(self, path: str | Path) -> None: ...


@dataclass(frozen=True)
class ReadinessConfig:
    minimum_performance_observations: int = 8
    minimum_degradation_observations: int = 12
    minimum_unique_feature_values: int = 2
    require_all_compounds: bool = True
    require_each_adjacent_gap: bool = True


def informative_performance_observations(observations: pd.DataFrame) -> pd.DataFrame:
    return observations.loc[
        observations["absolute_compound"].astype(str)
        != observations["event_hard_compound"].astype(str)
    ].copy()


def performance_design_matrix(
    observations: pd.DataFrame, *, include_environment: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    informative = informative_performance_observations(observations)
    parameter_count = 6 if include_environment else 4
    if informative.empty:
        return np.empty((0, parameter_count), dtype=float), np.empty(0, dtype=float)
    matrix = np.vstack(
        [
            _performance_basis(
                str(row.absolute_compound),
                str(row.event_hard_compound),
                float(row.tyre_stress),
                float(row.asphalt_grip),
                include_environment=include_environment,
            )
            for row in informative.itertuples(index=False)
        ]
    )
    return matrix, informative["performance"].to_numpy(dtype=float)


def degradation_design_matrix(
    observations: pd.DataFrame, *, include_environment: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    parameter_count = 8 if include_environment else 5
    if observations.empty:
        return np.empty((0, parameter_count), dtype=float), np.empty(0, dtype=float)
    matrix = np.vstack(
        [
            _degradation_basis(
                str(row.absolute_compound),
                [float(getattr(row, feature)) for feature in DEGRADATION_FEATURES],
                include_environment=include_environment,
            )
            for row in observations.itertuples(index=False)
        ]
    )
    return matrix, observations["degradation"].to_numpy(dtype=float)


def diagnose_performance_readiness(
    observations: pd.DataFrame,
    config: ReadinessConfig | None = None,
    *,
    include_environment: bool = True,
) -> ReadinessDiagnostics:
    config = config or ReadinessConfig()
    matrix, _ = performance_design_matrix(
        observations, include_environment=include_environment
    )
    compounds = _observed_compounds(observations)
    feature_names = PERFORMANCE_FEATURES if include_environment else ()
    feature_values = _feature_values(observations, feature_names)
    edges = _comparison_edges(observations)
    components = _graph_components(compounds, edges)
    gap_counts = _adjacent_gap_counts(observations)
    rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
    parameter_count = matrix.shape[1]
    reasons: list[str] = []
    if config.require_all_compounds and compounds != COMPOUNDS:
        reasons.append(f"compound coverage is {list(compounds)}, expected {list(COMPOUNDS)}")
    for feature, values in feature_values.items():
        if len(values) < config.minimum_unique_feature_values:
            reasons.append(f"{feature} has insufficient variation: {list(values)}")
    if compounds and len(components) != 1:
        reasons.append(f"compound comparison graph is disconnected: {list(components)}")
    if config.require_each_adjacent_gap:
        missing = [gap for gap, count in gap_counts.items() if count == 0]
        if missing:
            reasons.append(f"adjacent gaps lack representation: {missing}")
    if len(matrix) < config.minimum_performance_observations:
        reasons.append(
            f"only {len(matrix)} informative observations; "
            f"minimum is {config.minimum_performance_observations}"
        )
    if rank < parameter_count:
        reasons.append(f"relative design rank is {rank}/{parameter_count}")
    return ReadinessDiagnostics(
        ready=not reasons,
        reasons=tuple(reasons),
        observed_compounds=compounds,
        feature_values=feature_values,
        observation_count=len(observations),
        informative_observation_count=len(matrix),
        design_rank=rank,
        parameter_count=parameter_count,
        comparison_edges=edges,
        comparison_components=components,
        adjacent_gap_counts=gap_counts,
    )


def diagnose_degradation_readiness(
    observations: pd.DataFrame,
    config: ReadinessConfig | None = None,
    *,
    include_environment: bool = True,
) -> ReadinessDiagnostics:
    config = config or ReadinessConfig()
    matrix, _ = degradation_design_matrix(
        observations, include_environment=include_environment
    )
    compounds = _observed_compounds(observations)
    feature_names = DEGRADATION_FEATURES if include_environment else ()
    feature_values = _feature_values(observations, feature_names)
    rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
    parameter_count = matrix.shape[1]
    reasons: list[str] = []
    if config.require_all_compounds and compounds != COMPOUNDS:
        reasons.append(f"compound coverage is {list(compounds)}, expected {list(COMPOUNDS)}")
    for feature, values in feature_values.items():
        if len(values) < config.minimum_unique_feature_values:
            reasons.append(f"{feature} has insufficient variation: {list(values)}")
    if len(matrix) < config.minimum_degradation_observations:
        reasons.append(
            f"only {len(matrix)} observations; minimum is {config.minimum_degradation_observations}"
        )
    if rank < parameter_count:
        reasons.append(f"degradation design rank is {rank}/{parameter_count}")
    return ReadinessDiagnostics(
        ready=not reasons,
        reasons=tuple(reasons),
        observed_compounds=compounds,
        feature_values=feature_values,
        observation_count=len(observations),
        informative_observation_count=len(observations),
        design_rank=rank,
        parameter_count=parameter_count,
    )


class AdjacentGapPerformanceModel:
    def __init__(
        self,
        *,
        include_environment: bool = True,
        ridge_alpha: float = 0.0,
        weight_policy: ObservationWeightPolicy | None = None,
    ) -> None:
        if ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")
        self.include_environment = bool(include_environment)
        self.ridge_alpha = float(ridge_alpha)
        self.weight_policy = weight_policy or UniformWeightPolicy()
        self.coefficients: np.ndarray | None = None
        self.training_metadata: dict[str, object] = {}

    @property
    def name(self) -> str:
        return "performance_pirelli_v1" if self.include_environment else "performance_baseline"

    @property
    def code(self) -> str:
        return "P1" if self.include_environment else "P0"

    def fit(self, observations: pd.DataFrame) -> None:
        informative = informative_performance_observations(observations)
        matrix, target = performance_design_matrix(
            informative, include_environment=self.include_environment
        )
        if matrix.shape[0] == 0:
            raise ValueError("No informative non-HARD performance observations.")
        weights = self.weight_policy.weights(informative)
        self.coefficients = _weighted_linear_fit(
            matrix, target, weights, self.ridge_alpha
        )
        self.training_metadata = _fit_metadata(
            matrix, target, self.coefficients, weights, self.weight_policy.metadata()
        )

    def predict(
        self,
        *,
        absolute_compound: str,
        hard_compound: str,
        tyre_stress: float,
        asphalt_grip: float,
        **_: object,
    ) -> Prediction:
        if str(absolute_compound).upper() == str(hard_compound).upper():
            return Prediction(0.0, None)
        design = _performance_basis(
            absolute_compound,
            hard_compound,
            tyre_stress,
            asphalt_grip,
            include_environment=self.include_environment,
        )
        return Prediction(float(design @ self._required_coefficients()), None)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "formulation_code": self.code,
            "model_version": MODEL_VERSION,
            "definition": "adjacent compound gaps with shared environmental sensitivities",
            "performance_coordinate": "seconds relative to event HARD compound",
            "include_environment": self.include_environment,
            "features": list(PERFORMANCE_FEATURES if self.include_environment else ()),
            "ridge_alpha": self.ridge_alpha,
            "weighting_policy": self.weight_policy.metadata(),
            "coefficients": self._required_coefficients().tolist(),
            "coefficient_labels": _performance_coefficient_labels(self.include_environment),
            "training_metadata": self.training_metadata,
        }

    def save(self, path: str | Path) -> None:
        _write_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "AdjacentGapPerformanceModel":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "AdjacentGapPerformanceModel":
        policy_values = dict(values.get("weighting_policy", {"name": "uniform"}))
        model = cls(
            include_environment=bool(values.get("include_environment", True)),
            ridge_alpha=float(values["ridge_alpha"]),
            weight_policy=get_weight_policy(
                str(policy_values.get("name", "uniform")),
                evidence_field=str(policy_values.get("evidence_field", "support_clean_lap_count")),
                evidence_transformation=str(policy_values.get("transformation", "sqrt")),
            ),
        )
        model.coefficients = np.asarray(values["coefficients"], dtype=float)
        model.training_metadata = dict(values.get("training_metadata", {}))
        return model

    def _required_coefficients(self) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Performance model has not been fitted.")
        return self.coefficients


# Backwards-compatible import name; its implementation is now the adjacent-gap model.
RelativeLinearPerformanceModel = AdjacentGapPerformanceModel


class LinearDegradationModel:
    def __init__(
        self,
        *,
        include_environment: bool = True,
        ridge_alpha: float = 0.0,
        weight_policy: ObservationWeightPolicy | None = None,
    ) -> None:
        if ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")
        self.include_environment = bool(include_environment)
        self.ridge_alpha = float(ridge_alpha)
        self.weight_policy = weight_policy or UniformWeightPolicy()
        self.coefficients: np.ndarray | None = None
        self.training_metadata: dict[str, object] = {}

    @property
    def name(self) -> str:
        return "degradation_pirelli_v1" if self.include_environment else "degradation_baseline"

    @property
    def code(self) -> str:
        return "D1" if self.include_environment else "D0"

    def fit(self, observations: pd.DataFrame) -> None:
        matrix, target = degradation_design_matrix(
            observations, include_environment=self.include_environment
        )
        if matrix.shape[0] == 0:
            raise ValueError("No degradation observations.")
        weights = self.weight_policy.weights(observations)
        self.coefficients = _weighted_linear_fit(
            matrix, target, weights, self.ridge_alpha
        )
        self.training_metadata = _fit_metadata(
            matrix, target, self.coefficients, weights, self.weight_policy.metadata()
        )

    def predict(
        self,
        *,
        absolute_compound: str,
        tyre_stress: float,
        asphalt_abrasion: float,
        asphalt_grip: float,
        **_: object,
    ) -> Prediction:
        basis = _degradation_basis(
            absolute_compound,
            [tyre_stress, asphalt_abrasion, asphalt_grip],
            include_environment=self.include_environment,
        )
        return Prediction(float(basis @ self._required_coefficients()), None)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "formulation_code": self.code,
            "model_version": MODEL_VERSION,
            "definition": "absolute compound baselines plus shared linear track descriptors",
            "include_environment": self.include_environment,
            "features": list(DEGRADATION_FEATURES if self.include_environment else ()),
            "ridge_alpha": self.ridge_alpha,
            "weighting_policy": self.weight_policy.metadata(),
            "coefficients": self._required_coefficients().tolist(),
            "coefficient_labels": [
                *(f"alpha_{compound}" for compound in COMPOUNDS),
                *(
                    ("beta_stress", "beta_abrasion", "beta_grip")
                    if self.include_environment else ()
                ),
            ],
            "training_metadata": self.training_metadata,
        }

    def save(self, path: str | Path) -> None:
        _write_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "LinearDegradationModel":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "LinearDegradationModel":
        policy_values = dict(values.get("weighting_policy", {"name": "uniform"}))
        model = cls(
            include_environment=bool(values.get("include_environment", True)),
            ridge_alpha=float(values["ridge_alpha"]),
            weight_policy=get_weight_policy(
                str(policy_values.get("name", "uniform")),
                evidence_field=str(policy_values.get("evidence_field", "support_clean_lap_count")),
                evidence_transformation=str(policy_values.get("transformation", "sqrt")),
            ),
        )
        model.coefficients = np.asarray(values["coefficients"], dtype=float)
        model.training_metadata = dict(values.get("training_metadata", {}))
        return model

    def _required_coefficients(self) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Degradation model has not been fitted.")
        return self.coefficients


def _performance_basis(
    compound: str,
    hard_compound: str,
    tyre_stress: float,
    asphalt_grip: float,
    *,
    include_environment: bool,
) -> np.ndarray:
    path = _adjacent_path(compound, hard_compound)
    if not include_environment:
        return path
    signed_gap_count = float(path.sum())
    return np.concatenate(
        [path, [signed_gap_count * float(tyre_stress), signed_gap_count * float(asphalt_grip)]]
    )


def _adjacent_path(compound: str, reference_compound: str) -> np.ndarray:
    compound = _normalise_compound(compound)
    reference_compound = _normalise_compound(reference_compound)
    compound_index = COMPOUNDS.index(compound)
    reference_index = COMPOUNDS.index(reference_compound)
    path = np.zeros(4, dtype=float)
    if compound_index > reference_index:
        path[reference_index:compound_index] = 1.0
    elif compound_index < reference_index:
        path[compound_index:reference_index] = -1.0
    return path


def _degradation_basis(
    compound: str, features: Sequence[float], *, include_environment: bool
) -> np.ndarray:
    compound = _normalise_compound(compound)
    parameter_count = 8 if include_environment else 5
    basis = np.zeros(parameter_count, dtype=float)
    basis[COMPOUNDS.index(compound)] = 1.0
    if include_environment:
        basis[5:] = np.asarray(features, dtype=float)
    return basis


def _normalise_compound(compound: str) -> str:
    normalised = str(compound).upper()
    if normalised not in COMPOUNDS:
        raise ValueError(f"Unknown absolute compound {normalised!r}.")
    return normalised


def _weighted_linear_fit(
    matrix: np.ndarray, target: np.ndarray, weights: np.ndarray, alpha: float
) -> np.ndarray:
    if len(weights) != len(target) or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Regression weights must be finite, positive, and match observations.")
    root_weight = np.sqrt(weights)
    weighted_matrix = matrix * root_weight[:, None]
    weighted_target = target * root_weight
    if alpha == 0:
        return np.linalg.lstsq(weighted_matrix, weighted_target, rcond=None)[0]
    penalty = np.eye(matrix.shape[1], dtype=float) * float(alpha)
    return np.linalg.solve(
        weighted_matrix.T @ weighted_matrix + penalty,
        weighted_matrix.T @ weighted_target,
    )


def _fit_metadata(
    matrix: np.ndarray,
    target: np.ndarray,
    coefficients: np.ndarray,
    weights: np.ndarray,
    weighting_policy: Mapping[str, object],
) -> dict[str, object]:
    residual = target - matrix @ coefficients
    weighted_matrix = matrix * np.sqrt(weights)[:, None]
    return {
        "observation_count": int(len(target)),
        "design_rank": int(np.linalg.matrix_rank(matrix)),
        "parameter_count": int(matrix.shape[1]),
        "condition_number": float(np.linalg.cond(weighted_matrix)),
        "residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "weighted_residual_rmse": float(
            np.sqrt(np.average(residual**2, weights=weights))
        ),
        "weight_sum": float(weights.sum()),
        "weighting_policy": dict(weighting_policy),
    }


def _observed_compounds(observations: pd.DataFrame) -> tuple[str, ...]:
    values = set(observations.get("absolute_compound", pd.Series(dtype=str)).astype(str))
    return tuple(compound for compound in COMPOUNDS if compound in values)


def _feature_values(
    observations: pd.DataFrame, features: Sequence[str]
) -> dict[str, tuple[float, ...]]:
    return {
        feature: tuple(sorted(float(value) for value in observations[feature].dropna().unique()))
        for feature in features
    }


def _comparison_edges(observations: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    edges = {
        tuple(sorted((str(row.absolute_compound), str(row.event_hard_compound))))
        for row in observations.itertuples(index=False)
        if str(row.absolute_compound) != str(row.event_hard_compound)
    }
    return tuple(sorted(edges))


def _adjacent_gap_counts(observations: pd.DataFrame) -> dict[str, int]:
    counts = dict.fromkeys(ADJACENT_GAPS, 0)
    for row in informative_performance_observations(observations).itertuples(index=False):
        path = np.abs(_adjacent_path(str(row.absolute_compound), str(row.event_hard_compound)))
        for index, included in enumerate(path):
            if included:
                counts[ADJACENT_GAPS[index]] += 1
    return counts


def _graph_components(
    nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, ...], ...]:
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    components: list[tuple[str, ...]] = []
    unseen = set(adjacency)
    while unseen:
        stack = [min(unseen)]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node].difference(component))
        unseen.difference_update(component)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def _performance_coefficient_labels(include_environment: bool) -> list[str]:
    labels = [f"alpha_{gap.replace('-', '_')}" for gap in ADJACENT_GAPS]
    if include_environment:
        labels.extend(("beta_stress", "beta_grip"))
    return labels


def _write_json(path: Path, values: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
