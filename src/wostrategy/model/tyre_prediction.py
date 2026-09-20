"""Provider-neutral pre-race tyre prediction contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import re
from typing import Any, Mapping


REQUIRED_DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
REFERENCE_SELECTION_ORDER = ("MEDIUM", "HARD", "SOFT")
REFERENCE_ZERO_TOLERANCE = 1e-9
RELATIVE_PERFORMANCE_FRAME = "relative_to_compound"
COMMON_GAUGE_FRAME = "common_additive_gauge"
LEGACY_PERFORMANCE_FRAME = "legacy_unspecified"
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
    performance_frame: str = LEGACY_PERFORMANCE_FRAME
    reference_declared: bool | None = None
    formal_relative_output: bool = False

    def __post_init__(self) -> None:
        reference = str(self.reference_compound).upper()
        if reference not in REQUIRED_DRY_COMPOUNDS:
            raise ValueError(
                "Pre-race tyre prediction reference compound must be SOFT, MEDIUM or HARD"
            )
        object.__setattr__(self, "reference_compound", reference)
        normalized = {str(key).upper(): value for key, value in self.compounds.items()}
        object.__setattr__(self, "compounds", normalized)
        if self.performance_frame == RELATIVE_PERFORMANCE_FRAME and self.formal_relative_output:
            validate_relative_performance_reference(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreRaceTyrePrediction":
        """Restore the contract without changing its frame or formal status."""
        values = dict(payload)
        compounds = values.get("compounds", {})
        values["compounds"] = {
            str(name): (
                item
                if isinstance(item, TyreCompoundPrediction)
                else TyreCompoundPrediction(**dict(item))
            )
            for name, item in dict(compounds).items()
        }
        return cls(**values)


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

    prior_selection = prediction.provenance.get("source_reference_selection")
    has_prior_selection = isinstance(prior_selection, Mapping)
    selected_reference, reference_rule, zero_candidates = (
        select_source_performance_reference(
            validated,
            declared_reference=(
                prediction.reference_compound
                if prediction.reference_declared is True or has_prior_selection
                else None
            ),
        )
    )
    if has_prior_selection and prediction.reference_declared is not True:
        reference_rule = str(prior_selection.get("rule") or reference_rule)
        zero_candidates = tuple(
            str(value).upper()
            for value in prior_selection.get("zero_candidates", zero_candidates)
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
    uncertainty_kind = (
        "marginal_per_compound"
        if any(
            item.performance_uncertainty is not None
            or item.degradation_uncertainty is not None
            for item in validated.values()
        )
        else "unavailable"
    )
    provenance["source_performance_semantics"] = {
        "contract": "canonical_tyre_source_semantics_v1",
        "quantity": "compound_performance_delta",
        "unit": "s",
        "independent_variable": "dry_compound_choice",
        "positive_sign": "compound is slower than the comparison compound",
        "frame": prediction.performance_frame,
        "reference_compound": selected_reference,
        "reference_declared": prediction.reference_declared is True,
        "reference_selection_rule": reference_rule,
        "zero_candidates": list(zero_candidates),
        "numerical_transformation": "none",
    }
    provenance["performance_semantics"] = {
        "contract": "canonical_tyre_operational_semantics_v1",
        "quantity": "compound_performance_delta",
        "unit": "s",
        "independent_variable": "dry_compound_choice",
        "positive_sign": "compound is slower than MEDIUM",
        "frame": RELATIVE_PERFORMANCE_FRAME,
        "reference_compound": "MEDIUM",
        "reference_value": 0.0,
        "uncertainty": uncertainty_kind,
        "numerical_transformation": "subtract_source_medium_value_once",
    }
    provenance["degradation_semantics"] = {
        "contract": "canonical_tyre_operational_semantics_v1",
        "quantity": "degradation_rate",
        "unit": "s/tyre-age-lap",
        "independent_variable": "tyre_age_laps",
        "positive_sign": "greater tyre age increases modelled lap time",
        "frame": "absolute_rate",
        "reference_compound": None,
        "uncertainty": uncertainty_kind,
        "numerical_transformation": "none",
    }
    provenance["operational_normalization"] = {
        "contract": "required_smh_finite_nonnegative_degradation_medium_baseline_v2",
        "input_medium_performance_baseline": medium_baseline,
        "source_reference_compound": selected_reference,
        "source_reference_selection_rule": reference_rule,
        "normalization_count": 1,
    }
    return replace(
        prediction,
        season=expected_season,
        round_number=expected_round_number,
        reference_compound="MEDIUM",
        compounds=normalized,
        provenance=provenance,
        performance_frame=RELATIVE_PERFORMANCE_FRAME,
        reference_declared=True,
        formal_relative_output=True,
    )


def select_source_performance_reference(
    compounds: Mapping[str, TyreCompoundPrediction | float],
    *,
    declared_reference: str | None = None,
    tolerance: float = REFERENCE_ZERO_TOLERANCE,
) -> tuple[str, str, tuple[str, ...]]:
    """Select source gauge metadata without changing any numeric values."""
    normalized = {str(key).upper(): value for key, value in compounds.items()}
    zeros = tuple(
        compound
        for compound in REQUIRED_DRY_COMPOUNDS
        if compound in normalized
        and (value := _finite_performance_value(normalized[compound])) is not None
        and abs(value) <= tolerance
    )
    if declared_reference is not None:
        reference = str(declared_reference).upper()
        if reference not in REQUIRED_DRY_COMPOUNDS:
            raise ValueError("Declared tyre reference must be SOFT, MEDIUM or HARD")
        return reference, "declared", zeros
    if len(zeros) == 1:
        return zeros[0], "unique_zero", zeros
    if len(zeros) > 1:
        reference = next(item for item in REFERENCE_SELECTION_ORDER if item in zeros)
        return reference, "multiple_zeros_prefer_medium_then_hard_then_soft", zeros
    return "MEDIUM", "no_zero_default_medium", zeros


def validate_relative_performance_reference(
    prediction: PreRaceTyrePrediction,
    *,
    tolerance: float = REFERENCE_ZERO_TOLERANCE,
) -> None:
    """Validate only formal relative outputs, not common-gauge or legacy inputs."""
    reference = str(prediction.reference_compound).upper()
    if reference not in prediction.compounds:
        raise ValueError(
            f"Relative tyre prediction is missing its {reference} reference compound"
        )
    value = _required_finite(
        prediction.compounds[reference].performance_delta_to_medium,
        f"{reference} relative performance reference",
    )
    if abs(value) > tolerance:
        raise ValueError(
            f"Relative tyre prediction reference {reference} must be zero within "
            f"{tolerance:g}; got {value}"
        )


def _finite_performance_value(
    value: TyreCompoundPrediction | float,
) -> float | None:
    raw = value.performance_delta_to_medium if isinstance(value, TyreCompoundPrediction) else value
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
    "COMMON_GAUGE_FRAME",
    "LEGACY_PERFORMANCE_FRAME",
    "REFERENCE_ZERO_TOLERANCE",
    "RELATIVE_PERFORMANCE_FRAME",
    "REQUIRED_DRY_COMPOUNDS",
    "TyreCompoundPrediction",
    "canonicalize_operational_tyre_prediction",
    "select_source_performance_reference",
    "validate_relative_performance_reference",
]
