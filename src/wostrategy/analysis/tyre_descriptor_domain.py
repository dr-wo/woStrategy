from __future__ import annotations

from typing import Mapping

import pandas as pd


DESCRIPTOR_FEATURES = {
    "performance": ("tyre_stress", "asphalt_grip"),
    "degradation": ("tyre_stress", "asphalt_abrasion", "asphalt_grip"),
}
SHORT_NAMES = {
    "tyre_stress": "stress",
    "asphalt_abrasion": "abrasion",
    "asphalt_grip": "grip",
}
BOUNDARY_LEVELS = {1.0, 5.0}


def descriptor_domain_diagnostics(
    *, target: str, historical: pd.DataFrame, target_values: Mapping[str, object]
) -> dict[str, object]:
    if target not in DESCRIPTOR_FEATURES:
        raise ValueError(f"Unknown descriptor-domain target {target!r}.")
    features = DESCRIPTOR_FEATURES[target]
    values = tuple(float(target_values[feature]) for feature in features)
    event_features = historical[["season", "round", *features]].drop_duplicates(
        ["season", "round"]
    ) if not historical.empty else pd.DataFrame(columns=["season", "round", *features])
    result: dict[str, object] = {
        "descriptor_features": list(features),
        "descriptor_tuple": list(values),
        "descriptor_scale_interpretation": "ordinal 1-5 categories; endpoints 1 and 5 are saturated bins",
    }
    boundary_descriptors = []
    for feature in ("tyre_stress", "asphalt_abrasion", "asphalt_grip"):
        short = SHORT_NAMES[feature]
        if feature not in features:
            result[f"{short}_is_boundary"] = None
            result[f"{short}_level_seen_in_training"] = None
            continue
        value = float(target_values[feature])
        is_boundary = value in BOUNDARY_LEVELS
        result[f"{short}_is_boundary"] = is_boundary
        result[f"{short}_level_seen_in_training"] = bool(
            not event_features.empty and event_features[feature].eq(value).any()
        )
        if is_boundary:
            boundary_descriptors.append(feature)
    tuple_matches = pd.Series(False, index=event_features.index)
    if not event_features.empty:
        tuple_matches = pd.Series(True, index=event_features.index)
        for feature, value in zip(features, values):
            tuple_matches &= event_features[feature].eq(value)
    prior_count = int(tuple_matches.sum())
    result.update(
        {
            "any_boundary_descriptor": bool(boundary_descriptors),
            "boundary_descriptor_count": len(boundary_descriptors),
            "boundary_descriptors": boundary_descriptors,
            "exact_descriptor_tuple_seen": prior_count > 0,
            "exact_descriptor_tuple_prior_count": prior_count,
        }
    )
    return result
