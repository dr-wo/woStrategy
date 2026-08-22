from __future__ import annotations

import pandas as pd

from wostrategy.analysis.tyre_descriptor_domain import descriptor_domain_diagnostics


def historical() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2026, "round": 1, "tyre_stress": 2,
         "asphalt_abrasion": 3, "asphalt_grip": 4},
        {"season": 2026, "round": 2, "tyre_stress": 5,
         "asphalt_abrasion": 2, "asphalt_grip": 2},
        {"season": 2026, "round": 2, "tyre_stress": 5,
         "asphalt_abrasion": 2, "asphalt_grip": 2},
    ])


def test_boundary_categories_and_interior_categories():
    for value in (1, 5):
        result = descriptor_domain_diagnostics(
            target="performance", historical=historical(),
            target_values={"tyre_stress": value, "asphalt_grip": 3},
        )
        assert result["stress_is_boundary"] is True
        assert result["any_boundary_descriptor"] is True
    for value in (2, 3, 4):
        result = descriptor_domain_diagnostics(
            target="performance", historical=historical(),
            target_values={"tyre_stress": value, "asphalt_grip": 3},
        )
        assert result["stress_is_boundary"] is False


def test_level_and_tuple_coverage_use_only_passed_prior_events():
    prior = historical()
    result = descriptor_domain_diagnostics(
        target="degradation", historical=prior,
        target_values={"tyre_stress": 5, "asphalt_abrasion": 2, "asphalt_grip": 2},
    )
    assert result["stress_level_seen_in_training"] is True
    assert result["abrasion_level_seen_in_training"] is True
    assert result["grip_level_seen_in_training"] is True
    assert result["exact_descriptor_tuple_seen"] is True
    assert result["exact_descriptor_tuple_prior_count"] == 1  # unique event, not rows

    unseen = descriptor_domain_diagnostics(
        target="degradation", historical=prior,
        target_values={"tyre_stress": 5, "asphalt_abrasion": 3, "asphalt_grip": 2},
    )
    assert unseen["stress_level_seen_in_training"] is True
    assert unseen["abrasion_level_seen_in_training"] is True
    assert unseen["grip_level_seen_in_training"] is True
    assert unseen["exact_descriptor_tuple_seen"] is False
    assert unseen["exact_descriptor_tuple_prior_count"] == 0
