from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


UNIFORM = "uniform"
EVENT_NORMALISED = "event_normalised"
EVIDENCE_EVENT_NORMALISED = "evidence_event_normalised"
WEIGHT_POLICIES = (UNIFORM, EVENT_NORMALISED, EVIDENCE_EVENT_NORMALISED)


class ObservationWeightPolicy(Protocol):
    name: str

    def weights(self, observations: pd.DataFrame) -> np.ndarray:
        ...

    def metadata(self) -> dict[str, object]:
        ...


@dataclass(frozen=True)
class UniformWeightPolicy:
    name: str = UNIFORM

    def weights(self, observations: pd.DataFrame) -> np.ndarray:
        return np.ones(len(observations), dtype=float)

    def metadata(self) -> dict[str, object]:
        return {"name": self.name}


@dataclass(frozen=True)
class EventNormalisedWeightPolicy:
    name: str = EVENT_NORMALISED

    def weights(self, observations: pd.DataFrame) -> np.ndarray:
        return _normalise_by_event(observations, np.ones(len(observations), dtype=float))

    def metadata(self) -> dict[str, object]:
        return {"name": self.name, "event_total_weight": 1.0}


@dataclass(frozen=True)
class EvidenceEventNormalisedWeightPolicy:
    evidence_field: str = "support_clean_lap_count"
    transformation: str = "sqrt"
    name: str = EVIDENCE_EVENT_NORMALISED

    def weights(self, observations: pd.DataFrame) -> np.ndarray:
        if self.evidence_field not in observations.columns:
            raise ValueError(f"Missing evidence field {self.evidence_field!r}.")
        evidence = pd.to_numeric(
            observations[self.evidence_field], errors="coerce"
        ).to_numpy(dtype=float)
        if np.any(~np.isfinite(evidence)) or np.any(evidence <= 0):
            raise ValueError(
                f"{self.evidence_field} must be finite and positive for evidence weighting."
            )
        if self.transformation == "sqrt":
            quality = np.sqrt(evidence)
        elif self.transformation == "log1p":
            quality = np.log1p(evidence)
        else:
            raise ValueError("Evidence transformation must be 'sqrt' or 'log1p'.")
        return _normalise_by_event(observations, quality)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "evidence_field": self.evidence_field,
            "transformation": self.transformation,
            "event_total_weight": 1.0,
            "posterior_interval_weighting": False,
        }


def get_weight_policy(
    name: str,
    *,
    evidence_field: str = "support_clean_lap_count",
    evidence_transformation: str = "sqrt",
) -> ObservationWeightPolicy:
    if name == UNIFORM:
        return UniformWeightPolicy()
    if name == EVENT_NORMALISED:
        return EventNormalisedWeightPolicy()
    if name == EVIDENCE_EVENT_NORMALISED:
        return EvidenceEventNormalisedWeightPolicy(
            evidence_field=evidence_field,
            transformation=evidence_transformation,
        )
    raise ValueError(f"Unknown weight policy {name!r}; expected one of {WEIGHT_POLICIES}.")


def _normalise_by_event(observations: pd.DataFrame, quality: np.ndarray) -> np.ndarray:
    if len(observations) != len(quality):
        raise ValueError("Quality vector length does not match observations.")
    if "season" in observations.columns:
        keys = list(zip(observations["season"], observations["round"]))
    else:
        keys = observations["round"].tolist()
    totals: dict[object, float] = {}
    for key, value in zip(keys, quality):
        totals[key] = totals.get(key, 0.0) + float(value)
    return np.asarray(
        [float(value) / totals[key] for key, value in zip(keys, quality)], dtype=float
    )
