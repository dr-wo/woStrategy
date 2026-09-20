"""Semantic metadata for the existing pace models.

These builders describe the coordinates already used by each model. They are
not an equation framework and do not alter filtering, sampling or likelihoods.
"""

from __future__ import annotations

from typing import Mapping


def _uncertainty(sample_bank_id: str) -> dict[str, object]:
    return {
        "published_summary": "marginal_quantiles",
        "joint_samples_available": True,
        "joint_sample_bank_id": sample_bank_id,
        "sample_ids_are_local_to_sample_bank": True,
    }


def _identifiability(status: str) -> dict[str, object]:
    return {
        "status": status,
        "components_remain_numerically_available": True,
        "components_are_not_independent_measurements": True,
    }


def race_pace_semantics(
    *,
    reference_compound: str,
    sample_bank_id: str,
    architecture: str,
    fuel_reference: float,
    race_lap_reference: float,
    fuel_track_identifiability: str,
) -> tuple[tuple[str, object], ...]:
    """Describe the established Race correction and published baseline."""
    quantities = {
        "fuel_rate": {
            "unit": "s/fuel-proxy-remaining-lap",
            "stored_unit": "s/lap",
            "independent_variable": "FuelProxyLapsRemaining",
            "formula_term": "fuel_rate * (FuelProxyLapsRemaining - FuelRef)",
            "reference_value": float(fuel_reference),
            "positive_sign": "more fuel-proxy remaining increases modelled lap time",
            "not_a_mass_sensitivity": True,
            "estimation_role": "joint_model_coordinate",
        },
        "track_rate": {
            "unit": "s/race-lap-index",
            "stored_unit": "s/lap",
            "independent_variable": "LapNumber",
            "formula_term": "track_rate * (LapNumber - RaceLapRef)",
            "reference_value": float(race_lap_reference),
            "positive_sign": "later race lap increases modelled lap time",
            "negative_sign": "later race lap decreases modelled lap time",
            "estimation_role": "joint_model_coordinate",
        },
        "degradation": {
            "unit": "s/tyre-age-lap",
            "stored_unit": "s/lap",
            "independent_variable": "TyreAgeLaps",
            "positive_sign": "greater tyre age increases modelled lap time",
            "estimation_role": "joint_model_coordinate",
        },
        "compound_delta": {
            "unit": "s",
            "frame": "relative_to_compound",
            "reference_compound": str(reference_compound).upper(),
            "positive_sign": "compound is slower than the reference compound",
            "estimation_role": "joint_model_coordinate",
        },
        "baseline_pace": {
            "unit": "s",
            "frame": "corrected_race_baseline_at_declared_references",
            "published": True,
            "scope": "configured driver/team baseline group",
            "estimation_role": "profiled_baseline",
        },
    }
    return (
        ("contract", "canonical_pace_semantics_v1"),
        ("architecture", architecture),
        ("observation", "raw_lap_time_seconds"),
        ("filtering", {"role": "evidence_selection", "is_numeric_pace_transformation": False}),
        ("numeric_transformation", {
            "role": "remove sampled Race effects before profiling the published baseline",
            "joint_estimation_does_not_prohibit_future_effect_calculation": True,
        }),
        ("quantities", quantities),
        ("published_quantities", ("fuel/track/tyre marginals", "corrected baseline pace")),
        ("uncertainty", _uncertainty(sample_bank_id)),
        ("fuel_track_identifiability", _identifiability(fuel_track_identifiability)),
    )


def fp_pace_semantics(
    *,
    reference_compound: str,
    sample_bank_id: str,
    architecture: str,
    track_index_construction: Mapping[str, str],
    fuel_track_identifiability: str,
) -> tuple[tuple[str, object], ...]:
    """Describe the FP model with its candidate-local profiled run intercepts."""
    published_quantities = (
        (
            "fuel_rate marginal",
            "track_rate/degradation/compound_delta marginals",
        )
        if architecture == "single-session-fp-v1"
        else (
            "shared fuel_rate marginal",
            "session-local track_rate/degradation/compound_delta marginals",
        )
    )
    quantities = {
        "fuel_rate": {
            "unit": "s/run-lap-index",
            "stored_unit": "s/lap",
            "independent_variable": "run_lap_index (one-based within each run)",
            "formula_term": "fuel_rate * run_lap_index",
            "positive_sign": "later lap within a run increases modelled lap time",
            "not_a_mass_sensitivity": True,
            "estimation_role": "joint_model_coordinate",
        },
        "track_rate": {
            "unit": "s/track-index",
            "stored_unit": "s/lap",
            "independent_variable": "track_index",
            "formula_term": "track_rate[session] * track_index",
            "positive_sign": "higher track index increases modelled lap time",
            "negative_sign": "higher track index decreases modelled lap time",
            "estimation_role": "session_local_joint_model_coordinate",
        },
        "degradation": {
            "unit": "s/tyre-age-lap",
            "stored_unit": "s/lap",
            "independent_variable": "tyre_age",
            "formula_term": "degradation[session, compound] * tyre_age",
            "positive_sign": "greater tyre age increases modelled lap time",
            "estimation_role": "session_local_joint_model_coordinate",
        },
        "compound_delta": {
            "unit": "s",
            "frame": "relative_to_compound",
            "reference_compound": str(reference_compound).upper(),
            "positive_sign": "compound is slower than the reference compound",
            "estimation_role": "session_local_joint_model_coordinate",
        },
        "run_intercept": {
            "unit": "s",
            "fit_scope": "one intercept per run",
            "fit_timing": "re-fitted for every candidate",
            "published": False,
            "role": "profiled nuisance coordinate; not a common absolute baseline",
        },
    }
    return (
        ("contract", "canonical_pace_semantics_v1"),
        ("architecture", architecture),
        ("observation", "raw_lap_time_seconds"),
        ("filtering", {"role": "evidence_selection", "is_numeric_pace_transformation": False}),
        ("numeric_transformation", {
            "role": "candidate effect construction followed by per-run intercept profiling",
            "joint_estimation_does_not_prohibit_future_effect_calculation": True,
        }),
        ("quantities", quantities),
        ("track_index_construction", dict(track_index_construction)),
        ("published_quantities", published_quantities),
        ("uncertainty", _uncertainty(sample_bank_id)),
        ("fuel_track_identifiability", _identifiability(fuel_track_identifiability)),
    )


__all__ = ["fp_pace_semantics", "race_pace_semantics"]
