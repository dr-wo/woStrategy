"""Provider-neutral pre-race tyre prediction contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TyreCompoundPrediction:
    performance_delta_to_medium: float
    degradation_seconds_per_lap: float
    performance_uncertainty: float | None = None
    degradation_uncertainty: float | None = None
    available: bool = True
    identifiable: bool | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreRaceTyrePrediction:
    event: str
    season: int
    round_number: int
    reference_compound: str
    compounds: Mapping[str, TyreCompoundPrediction]
    generated_at: str
    provider: str
    artifact_id: str
    model_version: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    freshness: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reference_compound.upper() != "MEDIUM":
            raise ValueError("Pre-race tyre prediction reference compound must be MEDIUM")
        normalized = {str(key).upper(): value for key, value in self.compounds.items()}
        if "MEDIUM" not in normalized:
            raise ValueError("A MEDIUM prediction is required")
        if abs(normalized["MEDIUM"].performance_delta_to_medium) > 1e-12:
            raise ValueError("MEDIUM performance delta must be zero")
        object.__setattr__(self, "compounds", normalized)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["PreRaceTyrePrediction", "TyreCompoundPrediction"]
