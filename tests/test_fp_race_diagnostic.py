from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from wostrategy.analysis.fp_race_diagnostic import (
    assert_chronology,
    assert_pre_event_training_cutoff,
    baseline_corrected_delta,
    calibration_history,
    centre_programme_offset_observations,
    classify_topology,
    corrected_run_summaries,
    fit_run_summaries,
    joint_partial_fit,
    leave_one_event_out_weight_selection,
    leave_one_compound_out_k,
    partial_degradation_update,
    partial_k_update,
    programme_offset_stability,
    programme_offset_variance,
    rolling_programme_offset_validation,
    same_team_corrected_comparisons,
)


@pytest.mark.parametrize(
    ("pairs", "category", "bridge"),
    [
        ([('A', 'HARD')], "single_compound_single_team", False),
        ([('A', 'HARD'), ('B', 'MEDIUM')], "multiple_compounds_different_teams_only", False),
        ([('A', 'HARD'), ('A', 'MEDIUM')], "two_compounds_same_team_bridge", True),
        ([('A', 'HARD'), ('B', 'HARD'), ('B', 'MEDIUM')], "multiple_teams_with_overlap", True),
        ([('A', 'HARD'), ('A', 'MEDIUM'), ('B', 'MEDIUM'), ('B', 'SOFT')], "three_compounds_connected", True),
    ],
)
def test_topology_graph_cases(pairs, category, bridge):
    result = classify_topology(pd.DataFrame(pairs, columns=["team", "compound"]))
    assert result["topology_category"] == category
    assert result["same_team_multi_compound_bridge"] is bridge


def test_topology_does_not_invent_bridge_from_missing_team_names():
    result = classify_topology(pd.DataFrame([
        ("", "HARD"), ("", "MEDIUM"), ("Mercedes", "SOFT"),
    ], columns=["team", "compound"]))
    assert result["topology_category"] == "single_compound_single_team"
    assert result["same_team_multi_compound_bridge"] is False


def test_known_k_transfer_holds_out_compound():
    prior = {"HARD": 0.10, "MEDIUM": 0.20, "SOFT": 0.30}
    fp = {compound: 1.5 * value for compound, value in prior.items()}
    result = leave_one_compound_out_k(priors=prior, fp=fp, race_targets=fp)
    assert len(result) == 3
    assert result["k_event"].tolist() == pytest.approx([1.5, 1.5, 1.5])
    assert result["residual"].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_team_baseline_correction_sign():
    result = baseline_corrected_delta(
        left_pace=80.0, right_pace=80.6, left_baseline=79.8, right_baseline=80.1
    )
    assert result["raw_pace_delta"] == pytest.approx(0.6)
    assert result["team_baseline_delta"] == pytest.approx(0.3)
    assert result["compound_delta_estimate"] == pytest.approx(0.3)


def test_hard_relative_joint_fit_recovers_identifiable_synthetic_case():
    rows = []
    baseline = {"A": 80.0, "B": 80.4}
    performance = {"HARD": 0.0, "MEDIUM": -0.3, "SOFT": -0.6}
    degradation = {"HARD": 0.1, "MEDIUM": 0.2, "SOFT": 0.3}
    for team in baseline:
        for compound in performance:
            for age in range(5):
                rows.append({"team": team, "compound": compound, "tyre_age": age,
                             "lap_time_s": baseline[team] + performance[compound] + degradation[compound] * age})
    result = joint_partial_fit(
        pd.DataFrame(rows), team_baselines=baseline,
        performance_priors=performance, degradation_priors=degradation,
        team_strength=1e-9, performance_strength=1e-9, degradation_strength=1e-9,
    )
    assert result.unregularised_identifiable
    assert result.estimates["P:HARD"] == 0.0
    assert result.estimates["P:MEDIUM"] == pytest.approx(-0.3, abs=1e-7)
    assert result.estimates["D:SOFT"] == pytest.approx(0.3, abs=1e-7)


def test_joint_fit_reports_sparse_case_as_underdetermined():
    rows = pd.DataFrame([
        {"team": "A", "compound": "HARD", "tyre_age": age, "lap_time_s": 80 + 0.1 * age}
        for age in range(4)
    ] + [
        {"team": "B", "compound": "MEDIUM", "tyre_age": age, "lap_time_s": 80.2 + 0.2 * age}
        for age in range(4)
    ])
    result = joint_partial_fit(
        rows, team_baselines={"A": 80, "B": 80.5},
        performance_priors={"HARD": 0, "MEDIUM": -0.3},
        degradation_priors={"HARD": 0.1, "MEDIUM": 0.2},
    )
    assert not result.unregularised_identifiable
    assert result.design_rank < result.parameter_count


def test_chronology_rejects_later_session_and_never_mentions_race():
    assert_chronology(["FP1", "FP2"], "FP2")
    with pytest.raises(ValueError, match="Later-session leakage"):
        assert_chronology(["FP1", "FP3"], "FP2")


def test_training_cutoff_rejects_target_race_leakage():
    assert_pre_event_training_cutoff([1, 2, 3], 4)
    with pytest.raises(ValueError, match="Target/future Race leakage"):
        assert_pre_event_training_cutoff([1, 4], 4)


def test_deterministic_k_records_are_byte_identical():
    args = dict(
        priors={"HARD": .1, "MEDIUM": .2},
        fp={"HARD": .12, "MEDIUM": .24},
        race_targets={"HARD": .11, "MEDIUM": .22},
    )
    first = leave_one_compound_out_k(**args).to_csv(index=False).encode()
    second = leave_one_compound_out_k(**args).to_csv(index=False).encode()
    assert hashlib.sha256(first).digest() == hashlib.sha256(second).digest()


def test_empty_session_evidence_retains_topology_columns():
    runs = fit_run_summaries(pd.DataFrame({
        "used_for_quantitative_inference": pd.Series(dtype=bool),
    }))
    result = classify_topology(runs[["team", "compound"]])
    assert result["supported_team_count"] == 0
    assert result["supported_compound_count"] == 0


def test_partial_degradation_update_is_expected_convex_combination():
    assert partial_degradation_update(prior=0.10, fp=0.30, weight=0.25) == pytest.approx(0.15)
    assert partial_degradation_update(prior=0.10, fp=0.30, weight=0.0) == pytest.approx(0.10)
    assert partial_degradation_update(prior=0.10, fp=0.30, weight=1.0) == pytest.approx(0.30)


def test_partial_k_uses_shrunk_effective_k():
    result = partial_k_update(prior=0.20, raw_k=1.5, weight=0.25)
    assert result["effective_k"] == pytest.approx(1.125)
    assert result["prediction"] == pytest.approx(0.225)


def test_leave_one_event_out_selection_never_uses_held_out_errors():
    rows = []
    for round_number in (1, 2):
        rows.extend([
            {"round": round_number, "event": f"E{round_number}", "session": "FP2",
             "prior_family": "historical_baseline", "update_weight": 0.0,
             "prediction_count": 1, "mae": 0.01, "rmse": 0.01, "bias": 0.01},
            {"round": round_number, "event": f"E{round_number}", "session": "FP2",
             "prior_family": "historical_baseline", "update_weight": 1.0,
             "prediction_count": 1, "mae": 1.0, "rmse": 1.0, "bias": 1.0},
        ])
    rows.extend([
        {"round": 3, "event": "held", "session": "FP2",
         "prior_family": "historical_baseline", "update_weight": 0.0,
         "prediction_count": 1, "mae": 10.0, "rmse": 10.0, "bias": 10.0},
        {"round": 3, "event": "held", "session": "FP2",
         "prior_family": "historical_baseline", "update_weight": 1.0,
         "prediction_count": 1, "mae": 0.0, "rmse": 0.0, "bias": 0.0},
    ])
    result = leave_one_event_out_weight_selection(pd.DataFrame(rows))
    held = result.loc[result["held_out_round"].eq(3)].iloc[0]
    assert held.selected_weight == 0.0
    assert held.mae == 10.0


def test_k_is_calculated_separately_for_each_prior_family():
    fp = {"HARD": 0.20, "MEDIUM": 0.30}
    historical = leave_one_compound_out_k(
        priors={"HARD": 0.10, "MEDIUM": 0.15}, fp=fp, race_targets=fp
    )
    pirelli = leave_one_compound_out_k(
        priors={"HARD": 0.20, "MEDIUM": 0.20}, fp=fp, race_targets=fp
    )
    assert historical["k_event"].tolist() == pytest.approx([2.0, 2.0])
    assert pirelli["k_event"].tolist() == pytest.approx([1.5, 1.0])


def test_corrected_intercept_recovers_same_team_compound_delta_without_baseline():
    fuel_rate = 0.05
    track_rate = -0.02
    evidence_rows = []
    coordinate_rows = []
    for compound, delta, track_offset in (
        ("HARD", 0.0, 0), ("MEDIUM", -0.30, 10)
    ):
        run_id = f"FP2:A:{compound}"
        for tyre_age in range(1, 7):
            run_lap_index = tyre_age
            track_index = track_offset + tyre_age - 1
            lap_time = (
                80.0 + delta + 0.10 * tyre_age
                + fuel_rate * run_lap_index + track_rate * track_index
            )
            evidence_rows.append({
                "session": "FP2", "team": "A", "driver": "DRV",
                "run_id": run_id, "stint": 1, "compound": compound,
                "lap_number": track_index + 1, "lap_time_s": lap_time,
                "tyre_age": tyre_age, "used_for_quantitative_inference": True,
            })
            coordinate_rows.append({
                "session": "FP2", "driver": "DRV", "compound": compound,
                "lap_number": track_index + 1, "run_lap_index": run_lap_index,
                "track_index": track_index,
            })
    runs = corrected_run_summaries(
        pd.DataFrame(evidence_rows), pd.DataFrame(coordinate_rows),
        fuel_rate=fuel_rate, track_rate=track_rate,
    )
    comparison = same_team_corrected_comparisons(
        runs, {"MEDIUM": -0.30}
    ).iloc[0]
    assert comparison.corrected_prediction == pytest.approx(-0.30)
    assert comparison.corrected_residual == pytest.approx(0.0)
    assert comparison.uncorrected_prediction != pytest.approx(-0.30)


def test_calibration_history_is_versioned_through_each_available_round():
    rows = []
    for round_number, target in ((1, 0.1), (3, 0.3)):
        for weight, prediction in ((0.0, 0.1), (1.0, 0.2)):
            rows.append({
                "season": 2026, "round": round_number, "event": f"E{round_number}",
                "session": "FP2", "prior_family": "historical_baseline",
                "update_weight": weight, "prediction": prediction,
                "race_target": target,
            })
    history = calibration_history(pd.DataFrame(rows), update_type="direct")
    first = history.loc[history["trained_through_round"].eq(1)]
    latest = history.loc[history["trained_through_round"].eq(3)]
    assert first["event_count"].unique().tolist() == [1]
    assert first["training_rounds"].unique().tolist() == ["[1]"]
    assert latest["event_count"].unique().tolist() == [2]
    assert latest["calibration_status"].unique().tolist() == ["diagnostic_only"]
    assert not latest["is_diagnostic_grid_preferred"].isna().any()


def test_programme_offset_decomposition_centres_each_event_session():
    observations = centre_programme_offset_observations(pd.DataFrame([
        {"season": 2026, "round": 1, "session": "FP2", "team": "A",
         "run_slot": 1, "raw_programme_residual": 1.0},
        {"season": 2026, "round": 1, "session": "FP2", "team": "B",
         "run_slot": 1, "raw_programme_residual": 3.0},
        {"season": 2026, "round": 2, "session": "FP2", "team": "A",
         "run_slot": 1, "raw_programme_residual": 2.0},
        {"season": 2026, "round": 2, "session": "FP2", "team": "B",
         "run_slot": 1, "raw_programme_residual": 4.0},
    ]))
    assert observations.groupby("round")["relative_programme_offset"].mean().tolist() == pytest.approx([0.0, 0.0])
    stability = programme_offset_stability(observations)
    team_a = stability.loc[
        stability["grouping_level"].eq("team_session")
        & stability["team"].eq("A")
    ].iloc[0]
    assert bool(team_a.sufficient_event_support)
    assert team_a.median_relative_offset == pytest.approx(-1.0)
    variance = programme_offset_variance(observations).iloc[0]
    assert variance.event_session_common_variance == pytest.approx(0.25)
    assert variance.team_specific_residual_variance == pytest.approx(1.0)


def test_rolling_programme_offset_uses_only_earlier_events():
    observations = pd.DataFrame([
        {"round": round_number, "session": "FP2", "team": "A",
         "run_slot": slot, "relative_programme_offset": offset}
        for round_number in (1, 2)
        for slot, offset in ((1, 0.2), (2, 0.5))
    ] + [
        {"round": 3, "session": "FP2", "team": "A", "run_slot": 1,
         "relative_programme_offset": 100.0},
        {"round": 3, "session": "FP2", "team": "A", "run_slot": 2,
         "relative_programme_offset": -100.0},
    ])
    runs = pd.DataFrame([
        {"round": 3, "session": "FP2", "team": "A", "compound": "HARD",
         "run_id": "hard", "run_slot": 1, "corrected_intercept": 80.0},
        {"round": 3, "session": "FP2", "team": "A", "compound": "MEDIUM",
         "run_id": "medium", "run_slot": 2, "corrected_intercept": 80.4},
    ])
    bridges = pd.DataFrame([{
        "season": 2026, "round": 3, "event": "held", "session": "FP2",
        "team": "A", "compound": "MEDIUM", "uncorrected_prediction": 0.4,
        "corrected_prediction": 0.4, "race_target": 0.1,
        "uncorrected_residual": 0.3, "corrected_residual": 0.3,
        "historical_pre_race_prediction": 0.1,
        "historical_pre_race_residual": 0.0,
        "pirelli_pre_race_prediction": 0.2,
        "pirelli_pre_race_residual": 0.1,
    }])
    result = rolling_programme_offset_validation(runs, bridges, observations)
    held = result.iloc[0]
    assert bool(held.eligible)
    assert held.trained_through_round == 2
    assert held.programme_adjusted_prediction == pytest.approx(0.1)
    assert held.programme_adjusted_residual == pytest.approx(0.0)
