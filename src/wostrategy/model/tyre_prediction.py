"""Provider-neutral pre-race tyre prediction contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import re
from typing import Any, Mapping


REQUIRED_DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
_CANONICAL_EVENT_ID = re.compile(r"^(?P<season>\d{4})-(?P<round>\d{1,2})(?:-.+)?$")


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
        object.__setattr__(self, "compounds", normalized)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonicalize_operational_tyre_prediction(
    prediction: PreRaceTyrePrediction,
    *,
    expected_season: int,
    expected_round_number: int,
    expected_event_id: str | None = None,
) -> PreRaceTyrePrediction:
    """Validate and normalize one resolved tyre prediction for operational use.

    Source predictions may use any common additive performance gauge. The
    returned contract always uses MEDIUM=0 and is safe for Strategy, Planner,
    report, and plot consumers.
    """
    expected_season = int(expected_season)
    expected_round_number = int(expected_round_number)
    if int(prediction.season) != expected_season:
        raise ValueError(
            f"tyre prediction season {prediction.season} does not match requested "
            f"season {expected_season}"
        )
    if int(prediction.round_number) != expected_round_number:
        raise ValueError(
            f"tyre prediction round {prediction.round_number} does not match requested "
            f"round {expected_round_number}"
        )
    event_match = _CANONICAL_EVENT_ID.fullmatch(str(prediction.event).strip())
    if event_match is not None and expected_event_id is not None:
        source_event_id = (
            f"{int(event_match.group('season')):04d}-"
            f"{int(event_match.group('round')):02d}"
        )
        if source_event_id != expected_event_id:
            raise ValueError(
                f"tyre prediction event {source_event_id} does not match requested "
                f"event {expected_event_id}"
            )

    missing = [
        compound for compound in REQUIRED_DRY_COMPOUNDS
        if compound not in prediction.compounds
    ]
    if missing:
        raise ValueError(
            "tyre prediction is missing required compound(s): " + ", ".join(missing)
        )

    validated: dict[str, TyreCompoundPrediction] = {}
    for compound in REQUIRED_DRY_COMPOUNDS:
        item = prediction.compounds[compound]
        if not item.available:
            raise ValueError(f"tyre prediction {compound} is marked unavailable")
        performance = _required_finite(
            item.performance_delta_to_medium,
            f"{compound} performance_delta_to_medium",
        )
        degradation = _required_finite(
            item.degradation_seconds_per_lap,
            f"{compound} degradation_seconds_per_lap",
        )
        if degradation < 0:
            raise ValueError(
                f"{compound} degradation_seconds_per_lap cannot be negative"
            )
        validated[compound] = replace(
            item,
            performance_delta_to_medium=performance,
            degradation_seconds_per_lap=degradation,
        )

    medium_baseline = validated["MEDIUM"].performance_delta_to_medium
    normalized = {
        compound: replace(
            item,
            performance_delta_to_medium=(
                item.performance_delta_to_medium - medium_baseline
            ),
        )
        for compound, item in validated.items()
    }
    provenance = dict(prediction.provenance)
    provenance["operational_normalization"] = {
        "contract": "required_smh_finite_nonnegative_degradation_medium_baseline_v1",
        "input_medium_performance_baseline": medium_baseline,
    }
    return replace(
        prediction,
        season=expected_season,
        round_number=expected_round_number,
        reference_compound="MEDIUM",
        compounds=normalized,
        provenance=provenance,
    )


def _required_finite(value: object, name: str) -> float:
    if value is None:
        raise ValueError(f"tyre prediction {name} is required")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tyre prediction {name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"tyre prediction {name} must be finite")
    return number


__all__ = [
    "PreRaceTyrePrediction",
    "REQUIRED_DRY_COMPOUNDS",
    "TyreCompoundPrediction",
    "canonicalize_operational_tyre_prediction",
]
