from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wostrategy.model.cross_event_tyre_prediction import (
    COMPOUNDS,
    LinearDegradationModel,
    ReadinessConfig,
    RelativeLinearPerformanceModel,
    diagnose_degradation_readiness,
    diagnose_performance_readiness,
)


def synthetic_observations():
    adjacent_baselines = np.array([-0.22, -0.27, -0.31, -0.36])
    beta_stress = -0.012
    beta_grip = 0.007
    rows = []
    allocations = [
        ("C1", "C2", "C3"), ("C2", "C3", "C4"), ("C3", "C4", "C5"),
        ("C1", "C3", "C5"), ("C2", "C4", "C5"), ("C1", "C4", "C5"),
    ]
    round_number = 0
    for repeat in range(4):
        for hard, medium, soft in allocations:
            round_number += 1
            features = np.array([
                1 + (round_number % 5),
                1 + ((round_number * 2) % 5),
                1 + ((round_number * 3) % 5),
            ], dtype=float)
            for role, compound in (("HARD", hard), ("MEDIUM", medium), ("SOFT", soft)):
                hard_index = COMPOUNDS.index(hard)
                compound_index = COMPOUNDS.index(compound)
                if compound_index >= hard_index:
                    path = np.zeros(4)
                    path[hard_index:compound_index] = 1
                else:
                    path = np.zeros(4)
                    path[compound_index:hard_index] = -1
                rows.append({
                    "season": 2026,
                    "round": round_number,
                    "compound_role": role,
                    "absolute_compound": compound,
                    "event_hard_compound": hard,
                    "tyre_stress": features[0],
                    "asphalt_abrasion": features[1],
                    "asphalt_grip": features[2],
                    "performance": float(
                        path @ adjacent_baselines
                        + path.sum() * (beta_stress * features[0] + beta_grip * features[2])
                    ),
                    "degradation": (
                        0.05 * (COMPOUNDS.index(compound) + 1)
                        + 0.01 * features[0] + 0.02 * features[1] - 0.005 * features[2]
                    ),
                    "support_clean_lap_count": 4 + (round_number + compound_index) % 12,
                    "source_result_hash": f"retro-{round_number}",
                    "preview_source_hash": f"preview-{round_number}",
                })
    return pd.DataFrame(rows)


def test_readiness_rejects_missing_coverage_feature_variation_and_rank():
    observations = synthetic_observations()
    missing = observations.loc[~observations["absolute_compound"].eq("C5")]
    assert not diagnose_performance_readiness(missing).ready
    constant = observations.copy()
    constant[["tyre_stress", "asphalt_abrasion", "asphalt_grip"]] = 3
    assert not diagnose_degradation_readiness(constant).ready
    assert not diagnose_performance_readiness(constant).ready


def test_performance_readiness_rejects_disconnected_comparison_graph():
    observations = synthetic_observations()
    first = observations.loc[
        observations["absolute_compound"].isin(["C1", "C2"])
        & observations["event_hard_compound"].eq("C1")
    ]
    second = observations.loc[
        observations["absolute_compound"].isin(["C3", "C4", "C5"])
        & observations["event_hard_compound"].eq("C3")
    ]
    disconnected = pd.concat([first, second], ignore_index=True)
    diagnostics = diagnose_performance_readiness(
        disconnected,
        ReadinessConfig(minimum_performance_observations=1),
    )
    assert not diagnostics.ready
    assert len(diagnostics.comparison_components) == 2
    assert any("disconnected" in reason for reason in diagnostics.reasons)


def test_suitable_coverage_is_ready_and_models_recover_known_structure():
    observations = synthetic_observations()
    config = ReadinessConfig(minimum_performance_observations=16,
                             minimum_degradation_observations=20)
    performance_readiness = diagnose_performance_readiness(observations, config)
    degradation_readiness = diagnose_degradation_readiness(observations, config)
    assert performance_readiness.ready, performance_readiness.reasons
    assert degradation_readiness.ready, degradation_readiness.reasons

    performance = RelativeLinearPerformanceModel(ridge_alpha=0.0)
    degradation = LinearDegradationModel(ridge_alpha=0.0)
    performance.fit(observations)
    degradation.fit(observations)
    kwargs = dict(tyre_stress=4, asphalt_abrasion=2, asphalt_grip=5)
    assert performance.predict(
        absolute_compound="C3", hard_compound="C2", **kwargs
    ).mean == pytest.approx(-0.283, abs=1e-8)
    assert performance.predict(
        absolute_compound="C2", hard_compound="C2", **kwargs
    ).mean == 0.0
    assert degradation.predict(absolute_compound="C3", **kwargs).mean == pytest.approx(0.205)
    assert performance.predict(
        absolute_compound="C4", hard_compound="C1", **kwargs
    ).mean == pytest.approx(-0.839, abs=1e-8)
    assert performance.coefficients.tolist() == pytest.approx(
        [-0.22, -0.27, -0.31, -0.36, -0.012, 0.007], abs=1e-8
    )
    assert performance.coefficients.shape == (6,)
    assert degradation.coefficients.shape == (8,)


def test_models_round_trip_through_replaceable_interface(tmp_path):
    observations = synthetic_observations()
    performance = RelativeLinearPerformanceModel(ridge_alpha=0.1)
    degradation = LinearDegradationModel(ridge_alpha=0.2)
    performance.fit(observations)
    degradation.fit(observations)
    performance_path = tmp_path / "performance.json"
    degradation_path = tmp_path / "degradation.json"
    performance.save(performance_path)
    degradation.save(degradation_path)
    kwargs = dict(tyre_stress=4, asphalt_abrasion=2, asphalt_grip=5)
    assert RelativeLinearPerformanceModel.load(performance_path).predict(
        absolute_compound="C3", hard_compound="C2", **kwargs
    ) == performance.predict(absolute_compound="C3", hard_compound="C2", **kwargs)
    assert LinearDegradationModel.load(degradation_path).predict(
        absolute_compound="C3", **kwargs
    ) == degradation.predict(absolute_compound="C3", **kwargs)


@pytest.mark.parametrize(
    ("hard", "medium", "soft"),
    [("C1", "C2", "C3"), ("C2", "C3", "C4"), ("C3", "C4", "C5")],
)
def test_relative_coordinate_sums_adjacent_gaps_and_keeps_hard_zero(hard, medium, soft):
    model = RelativeLinearPerformanceModel(include_environment=False)
    model.coefficients = np.array([-0.2, -0.3, -0.4, -0.5])
    kwargs = dict(tyre_stress=3, asphalt_grip=4)
    hard_index = COMPOUNDS.index(hard)
    assert model.predict(absolute_compound=hard, hard_compound=hard, **kwargs).mean == 0.0
    assert model.predict(
        absolute_compound=medium, hard_compound=hard, **kwargs
    ).mean == pytest.approx(model.coefficients[hard_index])
    assert model.predict(
        absolute_compound=soft, hard_compound=hard, **kwargs
    ).mean == pytest.approx(model.coefficients[hard_index:hard_index + 2].sum())
