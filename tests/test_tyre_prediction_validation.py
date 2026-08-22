from __future__ import annotations

import pandas as pd
import pytest

from wostrategy.analysis.tyre_prediction_validation import (
    ValidationConfig,
    rolling_validate,
)
from wostrategy.model.cross_event_tyre_prediction import COMPOUNDS, ReadinessConfig


def history() -> pd.DataFrame:
    allocations = [
        ("C1", "C2", "C3"),
        ("C2", "C3", "C4"),
        ("C3", "C4", "C5"),
        ("C1", "C3", "C5"),
        ("C2", "C4", "C5"),
        ("C1", "C4", "C5"),
        ("C2", "C3", "C5"),
        ("C3", "C4", "C5"),
    ]
    rows = []
    gaps = [-0.2, -0.25, -0.3, -0.35]
    for round_number, (hard, medium, soft) in enumerate(allocations, start=1):
        stress = 1 + round_number % 5
        abrasion = 1 + round_number * 2 % 5
        grip = 1 + round_number * 3 % 5
        for role, compound in (("HARD", hard), ("MEDIUM", medium), ("SOFT", soft)):
            left, right = COMPOUNDS.index(hard), COMPOUNDS.index(compound)
            direction = 1 if right >= left else -1
            performance = direction * sum(gaps[min(left, right):max(left, right)])
            performance += (right - left) * (-0.01 * stress + 0.005 * grip)
            rows.append(
                {
                    "season": 2026,
                    "round": round_number,
                    "event": f"Event {round_number}",
                    "compound_role": role,
                    "absolute_compound": compound,
                    "event_hard_compound": hard,
                    "tyre_stress": stress,
                    "asphalt_abrasion": abrasion,
                    "asphalt_grip": grip,
                    "performance": performance,
                    "degradation": 0.04 * (right + 1) + 0.01 * stress
                    + 0.002 * abrasion - 0.003 * grip,
                    "support_clean_lap_count": 5 + round_number + right,
                    "support_run_count": 2 + right,
                    "support_stint_count": 1 + right,
                    "performance_p10": performance - 0.1,
                    "performance_p90": performance + 0.1,
                    "degradation_p10": 0.04 * (right + 1) + 0.01 * stress
                    + 0.002 * abrasion - 0.003 * grip - 0.02,
                    "degradation_p90": 0.04 * (right + 1) + 0.01 * stress
                    + 0.002 * abrasion - 0.003 * grip + 0.02,
                    "source_result_hash": f"retro-{round_number}",
                    "preview_source_hash": f"preview-{round_number}",
                }
            )
    return pd.DataFrame(rows)


def config() -> ValidationConfig:
    return ValidationConfig(
        readiness=ReadinessConfig(
            minimum_performance_observations=1,
            minimum_degradation_observations=1,
        )
    )


def test_rolling_validation_has_no_future_leakage_and_excludes_hard_performance():
    result = rolling_validate(history(), config())
    predictions = result.predictions
    assert not predictions.empty
    assert (predictions["training_round_end"] < predictions["predicted_round"]).all()
    performance = predictions.loc[predictions["model_name"].str.startswith("performance")]
    assert not performance["compound_role"].eq("HARD").any()
    generated_rounds = set(predictions["predicted_round"])
    readiness_rounds = {
        row["predicted_round"]
        for row in result.payload["historical_cutoffs"]
        if any(
            row[name]["ready"]
            for name in (
                "performance_baseline", "performance_pirelli_v1",
                "degradation_baseline", "degradation_pirelli_v1",
            )
        )
    }
    assert generated_rounds.issubset(readiness_rounds)


def test_changing_last_round_does_not_change_earlier_predictions():
    original = rolling_validate(history(), config()).predictions
    changed_history = history()
    changed_history.loc[changed_history["round"].eq(8), ["performance", "degradation"]] += 99
    changed = rolling_validate(changed_history, config()).predictions
    keys = [
        "season", "predicted_round", "compound_role", "absolute_compound",
        "model_name", "weighting_policy",
    ]
    before = original.loc[original["predicted_round"] < 8, keys + ["predicted_value"]]
    after = changed.loc[changed["predicted_round"] < 8, keys + ["predicted_value"]]
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))


def test_coefficient_history_has_one_deterministic_record_per_eligible_fit():
    first = rolling_validate(history(), config())
    second = rolling_validate(history(), config())
    pd.testing.assert_frame_equal(first.coefficient_history, second.coefficient_history)
    rolling = first.coefficient_history.loc[
        first.coefficient_history["fit_context"].eq("historical_rolling")
        & first.coefficient_history["weighting_policy"].eq("uniform")
    ]
    assert not rolling.duplicated(["predicted_round", "model_code"]).any()
    assert (rolling["training_round_end"] < rolling["predicted_round"]).all()
    expected = {
        "P0": {"alpha_C1_C2", "alpha_C2_C3", "alpha_C3_C4", "alpha_C4_C5"},
        "P1": {"alpha_C1_C2", "alpha_C2_C3", "alpha_C3_C4", "alpha_C4_C5",
               "beta_stress", "beta_grip"},
        "D0": {"alpha_C1", "alpha_C2", "alpha_C3", "alpha_C4", "alpha_C5"},
        "D1": {"alpha_C1", "alpha_C2", "alpha_C3", "alpha_C4", "alpha_C5",
               "beta_stress", "beta_abrasion", "beta_grip"},
    }
    for row in rolling.itertuples(index=False):
        assert set(row.coefficient_names.split(",")) == expected[row.model_code]


def test_residual_diagnostics_are_enriched_ranked_and_keep_hard_unscored():
    result = rolling_validate(history(), config())
    diagnostics = result.predictions
    performance = diagnostics.loc[diagnostics["model_code"].isin(["P0", "P1"])]
    assert not performance["compound_role"].eq("HARD").any()
    row = diagnostics.iloc[0]
    assert row.event.startswith("Event ")
    assert row.absolute_residual == abs(row.predicted_value - row.observed_value)
    assert row.hard_compound in COMPOUNDS
    assert row.support_clean_lap_count > 0
    ranked = result.diagnostic_summary["historical_rolling"]["largest_residuals"]
    for values in ranked.values():
        residuals = [item["absolute_residual"]
                     for item in values["largest_absolute_residual_observations"]]
        assert residuals == sorted(residuals, reverse=True)


def test_production_policy_does_not_promote_numerically_better_challenger():
    result = rolling_validate(history(), config())
    assert result.payload["selection"]["performance"]["formulation_code"] == "P0"
    assert result.payload["selection"]["degradation"]["formulation_code"] == "D0"
    assert result.payload["selection"]["performance"]["weighting_policy"] == "uniform"
    assert result.payload["selection"]["degradation"]["weighting_policy"] == "uniform"


def test_descriptor_domain_split_metrics_match_enriched_records():
    result = rolling_validate(history(), config())
    rows = result.predictions.loc[result.predictions["weighting_policy"].eq("uniform")]
    analysis = result.diagnostic_summary["historical_rolling"]["descriptor_domain_analysis"]
    splits = {
        "interior": ~rows["any_boundary_descriptor"],
        "boundary_exposed": rows["any_boundary_descriptor"],
        "tuple_seen": rows["exact_descriptor_tuple_seen"],
        "tuple_unseen": ~rows["exact_descriptor_tuple_seen"],
    }
    for target, codes in (("performance", ("P0", "P1")),
                          ("degradation", ("D0", "D1"))):
        for split_name, mask in splits.items():
            for code in codes:
                subset = rows.loc[mask & rows["model_code"].eq(code)]
                metrics = analysis[target][split_name][code]
                assert metrics["prediction_count"] == len(subset)
                if not subset.empty:
                    assert metrics["MAE"] == pytest.approx(subset["residual"].abs().mean())
    assert rows["stress_level_seen_in_training"].notna().all()
    assert rows.loc[rows["model_code"].isin(["P0", "P1"]),
                    "abrasion_level_seen_in_training"].isna().all()
