from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from wodata import get_data_root
from wodata.contracts import canonical_hash

from wostrategy.analysis.fp_tyre_evidence import analyse_fp_tyre_evidence_sessions
from wostrategy.analysis.long_run_performance import fit_long_run_components
from wostrategy.core.pre_race_session_data import prepare_weekend_sessions
from wostrategy.model.pre_race_performance import _prepare_analysis_laps
from wostrategy.analysis.retro_tyre_results import (
    observations_csv_path,
    tyre_prediction_year_root,
)
from wostrategy.model.tyre_degragation import TYRE_AGE_LAPS_COLUMN
from wostrategy.analysis.tyre_prediction_validation import (
    PIRELLI_INFORMED_POLICY,
    PRODUCTION_POLICY,
)


SCHEMA_VERSION = "schema_v1"
ALGORITHM_VERSION = "historical-fp-race-diagnostic-v4"
SESSION_ORDER = {"FP1": 1, "FP2": 2, "FP3": 3}
COMPOUNDS = ("HARD", "MEDIUM", "SOFT")
UPDATE_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)
CALIBRATION_METHOD = "fixed_grid_event_mean_mae_with_loeo_v1"
CALIBRATION_VERSION = "calibration_v1"
CALIBRATION_STATUS = "diagnostic_only"
PROGRAMME_OFFSET_MIN_EVENTS = 2


@dataclass(frozen=True)
class JointFitResult:
    estimates: dict[str, float]
    design_rank: int
    parameter_count: int
    data_row_count: int
    unregularised_identifiable: bool
    regularisation: dict[str, float]


def diagnostic_root(season: int, data_root: str | Path | None = None) -> Path:
    return (
        get_data_root(data_root)
        / "wostrategy"
        / "tyre_prediction"
        / SCHEMA_VERSION
        / f"year={int(season)}"
        / "fp_race_diagnostic"
    )


def calibration_snapshot_root(
    season: int, data_root: str | Path | None = None
) -> Path:
    return (
        tyre_prediction_year_root(season, data_root)
        / "fp_degradation_calibration"
    )


def error_metrics(
    records: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    target_column: str = "race_target",
) -> dict[str, float | int | None]:
    if records.empty:
        return {"prediction_count": 0, "mae": None, "rmse": None, "bias": None}
    valid = records[[prediction_column, target_column]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if valid.empty:
        return {"prediction_count": 0, "mae": None, "rmse": None, "bias": None}
    residual = valid[prediction_column] - valid[target_column]
    return {
        "prediction_count": int(len(residual)),
        "mae": float(residual.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "bias": float(residual.mean()),
    }


def classify_topology(edges: pd.DataFrame) -> dict[str, object]:
    required = {"team", "compound"}
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"Topology edges are missing columns: {sorted(missing)}")
    unique = (
        edges.dropna(subset=["team", "compound"])[["team", "compound"]]
        .astype(str)
    )
    unique["team"] = unique["team"].str.strip()
    unique["compound"] = unique["compound"].str.strip().str.upper()
    unique = (
        unique.loc[unique["team"].ne("") & unique["compound"].ne("")]
        .drop_duplicates()
        .sort_values(["team", "compound"])
    )
    teams = sorted(unique["team"].unique())
    compounds = sorted(unique["compound"].str.upper().unique())
    adjacency: dict[str, set[str]] = {}
    for row in unique.itertuples(index=False):
        team_node = f"team:{row.team}"
        compound_node = f"compound:{str(row.compound).upper()}"
        adjacency.setdefault(team_node, set()).add(compound_node)
        adjacency.setdefault(compound_node, set()).add(team_node)
    components: list[list[str]] = []
    unseen = set(adjacency)
    while unseen:
        stack = [min(unseen)]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency.get(node, set()).difference(component))
        unseen.difference_update(component)
        components.append(sorted(component))
    team_degrees = unique.groupby("team")["compound"].nunique() if not unique.empty else pd.Series(dtype=int)
    compound_degrees = unique.groupby("compound")["team"].nunique() if not unique.empty else pd.Series(dtype=int)
    same_team_bridge = bool((team_degrees >= 2).any())
    if len(compounds) <= 1:
        category = "single_compound_multiple_teams" if len(teams) > 1 else "single_compound_single_team"
    elif same_team_bridge and len(compounds) == 3 and len(components) == 1:
        category = "three_compounds_connected"
    elif same_team_bridge and bool((compound_degrees >= 2).any()):
        category = "multiple_teams_with_overlap"
    elif same_team_bridge:
        category = "two_compounds_same_team_bridge"
    else:
        category = "multiple_compounds_different_teams_only"
    return {
        "teams": teams,
        "compounds": compounds,
        "team_compound_edges": [f"{r.team}->{str(r.compound).upper()}" for r in unique.itertuples(index=False)],
        "supported_team_count": len(teams),
        "supported_compound_count": len(compounds),
        "same_team_multi_compound_bridge": same_team_bridge,
        "cross_team_only_comparison": len(compounds) >= 2 and not same_team_bridge,
        "component_count": len(components),
        "components": components,
        "compound_graph_connected": len(compounds) > 0 and len(components) == 1,
        "topology_category": category,
    }


def event_k(priors: Mapping[str, float], fp: Mapping[str, float]) -> float | None:
    ratios = [
        float(fp[c]) / float(priors[c])
        for c in sorted(set(priors).intersection(fp))
        if np.isfinite(float(fp[c])) and np.isfinite(float(priors[c])) and float(priors[c]) > 0
    ]
    return float(np.median(ratios)) if ratios else None


def partial_degradation_update(*, prior: float, fp: float, weight: float) -> float:
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("update weight must be between zero and one")
    return (1.0 - weight) * float(prior) + weight * float(fp)


def partial_k_update(*, prior: float, raw_k: float, weight: float) -> dict[str, float]:
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("update weight must be between zero and one")
    effective_k = 1.0 + weight * (float(raw_k) - 1.0)
    return {
        "effective_k": effective_k,
        "prediction": effective_k * float(prior),
    }


def event_level_metrics(records: pd.DataFrame) -> pd.DataFrame:
    columns = ["season", "round", "event", "session", "prior_family", "update_weight"]
    rows = []
    if records.empty:
        return pd.DataFrame()
    for key, group in records.groupby(columns, dropna=False, sort=True):
        rows.append({
            **dict(zip(columns, key)),
            **error_metrics(group),
        })
    return pd.DataFrame(rows)


def leave_one_event_out_weight_selection(event_metrics: pd.DataFrame) -> pd.DataFrame:
    """Select a fixed-grid weight using training events only, then score held out."""
    rows: list[dict[str, object]] = []
    if event_metrics.empty:
        return pd.DataFrame()
    for (session, family), group in event_metrics.groupby(
        ["session", "prior_family"], sort=True
    ):
        event_keys = group[["round", "event"]].drop_duplicates().sort_values("round")
        for held in event_keys.itertuples(index=False):
            training = group.loc[group["round"].ne(held.round)]
            if training.empty:
                continue
            training_scores = (
                training.groupby("update_weight", as_index=False)
                .agg(
                    training_mean_event_mae=("mae", "mean"),
                    training_event_count=("round", "nunique"),
                )
                .sort_values(
                    ["training_mean_event_mae", "update_weight"],
                    kind="mergesort",
                )
            )
            selected = training_scores.iloc[0]
            held_metric = group.loc[
                group["round"].eq(held.round)
                & group["update_weight"].eq(selected.update_weight)
            ]
            if held_metric.empty:
                continue
            metric = held_metric.iloc[0]
            rows.append({
                "session": session,
                "prior_family": family,
                "held_out_round": int(held.round),
                "held_out_event": held.event,
                "selected_weight": float(selected.update_weight),
                "training_event_count": int(selected.training_event_count),
                "training_mean_event_mae": float(selected.training_mean_event_mae),
                "prediction_count": int(metric.prediction_count),
                "mae": float(metric.mae),
                "rmse": float(metric.rmse),
                "bias": float(metric.bias),
            })
    return pd.DataFrame(rows)


def calibration_history(
    records: pd.DataFrame,
    *,
    update_type: str,
) -> pd.DataFrame:
    """Recompute diagnostic fixed-grid calibration through each available race."""
    rows: list[dict[str, object]] = []
    if records.empty:
        return pd.DataFrame()
    candidate_weights = json.dumps(list(UPDATE_WEIGHTS))
    for (session, family), group in records.groupby(
        ["session", "prior_family"], sort=True
    ):
        for through_round in sorted(group["round"].astype(int).unique()):
            prefix = group.loc[group["round"].astype(int).le(through_round)].copy()
            event_metrics = event_level_metrics(prefix)
            loeo = leave_one_event_out_weight_selection(event_metrics)
            selections = (
                loeo["selected_weight"].value_counts().sort_index()
                if not loeo.empty else pd.Series(dtype="int64")
            )
            unique_selections = [float(value) for value in selections.index]
            modal_weight = None
            modal_share = None
            if not selections.empty:
                largest = int(selections.max())
                modal_weight = float(min(
                    weight for weight, count in selections.items()
                    if int(count) == largest
                ))
                modal_share = float(largest / int(selections.sum()))
            candidate_metrics = []
            for weight, candidate in prefix.groupby("update_weight", sort=True):
                metrics = error_metrics(candidate)
                event_rows = event_metrics.loc[
                    event_metrics["update_weight"].eq(weight)
                ]
                candidate_metrics.append({
                    "weight": float(weight),
                    **metrics,
                    "mean_event_mae": float(event_rows["mae"].mean()),
                })
            preferred = min(
                candidate_metrics,
                key=lambda value: (value["mean_event_mae"], value["weight"]),
            )["weight"]
            if len(unique_selections) <= 1 and len(loeo) >= 2:
                stability = "stable"
            elif loeo.empty:
                stability = "insufficient_events"
            else:
                stability = "unstable"
            for metric in candidate_metrics:
                weight = float(metric.pop("weight"))
                rows.append({
                    "calibration_method": CALIBRATION_METHOD,
                    "calibration_version": CALIBRATION_VERSION,
                    "calibration_status": CALIBRATION_STATUS,
                    "trained_through_round": int(through_round),
                    "event_count": int(prefix["round"].nunique()),
                    "training_rounds": json.dumps(
                        sorted(prefix["round"].astype(int).unique().tolist())
                    ),
                    "candidate_weights": candidate_weights,
                    "prior_family": family,
                    "session": session,
                    "update_type": update_type,
                    "candidate_weight": weight,
                    **metric,
                    "diagnostic_grid_preferred_weight": float(preferred),
                    "is_diagnostic_grid_preferred": weight == float(preferred),
                    "loeo_fold_count": int(len(loeo)),
                    "loeo_selection_count": int(selections.get(weight, 0)),
                    "loeo_selected_weights": json.dumps(unique_selections),
                    "loeo_modal_weight": modal_weight,
                    "loeo_modal_share": modal_share,
                    "loeo_stability": stability,
                })
    return pd.DataFrame(rows)


def leave_one_compound_out_k(
    *,
    priors: Mapping[str, float],
    fp: Mapping[str, float],
    race_targets: Mapping[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    observed = sorted(set(fp).intersection(priors).intersection(race_targets))
    for held_out in observed:
        remaining = {compound: fp[compound] for compound in observed if compound != held_out}
        k = event_k(priors, remaining)
        if k is None:
            continue
        target = float(race_targets[held_out])
        prior = float(priors[held_out])
        prediction = k * prior
        rows.append({
            "held_out_compound": held_out,
            "supporting_compounds": ";".join(sorted(remaining)),
            "k_event": k,
            "prior_prediction": prior,
            "prediction": prediction,
            "race_target": target,
            "prior_residual": prior - target,
            "residual": prediction - target,
        })
    return pd.DataFrame(rows)


def baseline_corrected_delta(
    *,
    left_pace: float,
    right_pace: float,
    left_baseline: float,
    right_baseline: float,
) -> dict[str, float]:
    raw = float(right_pace) - float(left_pace)
    baseline = float(right_baseline) - float(left_baseline)
    return {
        "raw_pace_delta": raw,
        "team_baseline_delta": baseline,
        "compound_delta_estimate": raw - baseline,
    }


def fit_run_summaries(longrun_laps: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "session", "team", "driver", "run_id", "compound",
        "clean_lap_count", "tyre_age_span", "slope_seconds_per_lap",
        "tyre_age_zero_pace_seconds", "rmse_seconds",
    ]
    rows: list[dict[str, object]] = []
    used = longrun_laps.loc[longrun_laps["used_for_quantitative_inference"]].copy()
    if used.empty:
        return pd.DataFrame(columns=columns)
    for key, group in used.groupby(["session", "team", "driver", "run_id", "compound"], sort=True):
        clean = group.dropna(subset=["tyre_age", "lap_time_s"])
        if len(clean) < 2 or clean["tyre_age"].nunique() < 2:
            continue
        fit_input = pd.DataFrame({
            "Driver": clean["driver"], "Team": clean["team"],
            "Compound": clean["compound"], "Stint": clean["stint"],
            "LongRunId": 1, "LapNumber": clean["lap_number"],
            TYRE_AGE_LAPS_COLUMN: clean["tyre_age"],
            "LapTimeSeconds": clean["lap_time_s"],
        })
        fit = fit_long_run_components(fit_input)
        rows.append({
            "session": key[0], "team": key[1], "driver": key[2], "run_id": key[3],
            "compound": str(key[4]).upper(), "clean_lap_count": int(len(clean)),
            "tyre_age_span": float(clean["tyre_age"].max() - clean["tyre_age"].min()),
            "slope_seconds_per_lap": fit.parameters["tyre_slope_seconds_per_lap"],
            "tyre_age_zero_pace_seconds": fit.estimated_first_lap_seconds,
            "rmse_seconds": fit.rmse_seconds,
        })
    return pd.DataFrame(rows, columns=columns)


def corrected_run_summaries(
    longrun_laps: pd.DataFrame,
    model_coordinates: pd.DataFrame,
    *,
    fuel_rate: float,
    track_rate: float,
) -> pd.DataFrame:
    """Correct raw tyre-age-zero intercepts in the model's exact coordinates."""
    columns = [
        "session", "team", "driver", "run_id", "stint", "compound",
        "run_start_lap", "run_slot", "run_slot_source",
        "clean_lap_count", "coordinate_match_count", "raw_intercept",
        "fuel_correction", "track_correction", "corrected_intercept",
        "fuel_rate", "track_rate", "coordinate_source",
    ]
    used = longrun_laps.loc[
        longrun_laps["used_for_quantitative_inference"]
    ].copy()
    if used.empty or model_coordinates.empty:
        return pd.DataFrame(columns=columns)
    coordinates = model_coordinates[
        ["session", "driver", "lap_number", "compound", "run_lap_index", "track_index"]
    ].copy()
    for name in ("session", "driver", "compound"):
        coordinates[name] = coordinates[name].astype(str)
        used[name] = used[name].astype(str)
    matched = used.merge(
        coordinates,
        on=["session", "driver", "lap_number", "compound"],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    keys = ["session", "team", "driver", "run_id", "stint", "compound"]
    for key, group in matched.groupby(keys, sort=True, dropna=False):
        clean = group.dropna(
            subset=["tyre_age", "lap_time_s", "run_lap_index", "track_index"]
        ).copy()
        if len(clean) < 2 or clean["tyre_age"].nunique() < 2:
            continue
        fit_input = pd.DataFrame({
            "Driver": clean["driver"],
            "Team": clean["team"],
            "Compound": clean["compound"],
            "Stint": clean["stint"],
            "LongRunId": 1,
            "LapNumber": clean["lap_number"],
            TYRE_AGE_LAPS_COLUMN: clean["tyre_age"],
            "LapTimeSeconds": clean["lap_time_s"],
        })
        raw_fit = fit_long_run_components(fit_input)
        selected = clean.loc[list(raw_fit.fit_lap_indices)]
        tyre_age = selected["tyre_age"].to_numpy(dtype="float64")
        fuel_effect = float(fuel_rate) * selected["run_lap_index"].to_numpy(
            dtype="float64"
        )
        track_effect = float(track_rate) * selected["track_index"].to_numpy(
            dtype="float64"
        )
        fuel_intercept = float(np.polyfit(tyre_age, fuel_effect, 1)[1])
        track_intercept = float(np.polyfit(tyre_age, track_effect, 1)[1])
        raw_intercept = float(raw_fit.estimated_first_lap_seconds)
        rows.append({
            "session": key[0],
            "team": key[1],
            "driver": key[2],
            "run_id": key[3],
            "stint": key[4],
            "compound": str(key[5]).upper(),
            "run_start_lap": float(clean["lap_number"].min()),
            "run_slot": None,
            "run_slot_source": "chronological_supported_long_run_within_driver",
            "clean_lap_count": int(len(selected)),
            "coordinate_match_count": int(len(clean)),
            "raw_intercept": raw_intercept,
            "fuel_correction": fuel_intercept,
            "track_correction": track_intercept,
            "corrected_intercept": (
                raw_intercept - fuel_intercept - track_intercept
            ),
            "fuel_rate": float(fuel_rate),
            "track_rate": float(track_rate),
            "coordinate_source": "normal_pre_race_prepared_laps",
        })
    output = pd.DataFrame(rows, columns=columns)
    if output.empty:
        return output
    output = output.sort_values(
        ["session", "team", "driver", "run_start_lap", "run_id"],
        kind="mergesort",
    )
    output["run_slot"] = (
        output.groupby(["session", "team", "driver"], sort=False)
        .cumcount()
        .add(1)
        .astype(int)
    )
    return output.reset_index(drop=True)


def same_team_corrected_comparisons(
    runs: pd.DataFrame,
    race_targets: Mapping[str, float],
) -> pd.DataFrame:
    columns = [
        "team", "compound", "hard_raw_intercept", "comparison_raw_intercept",
        "hard_corrected_intercept", "comparison_corrected_intercept",
        "hard_fuel_correction", "comparison_fuel_correction",
        "hard_track_correction", "comparison_track_correction",
        "uncorrected_prediction", "corrected_prediction", "race_target",
        "uncorrected_residual", "corrected_residual", "same_team_pair",
    ]
    if runs.empty:
        return pd.DataFrame(columns=columns)
    valid = runs.loc[runs["team"].astype(str).str.strip().ne("")].copy()
    pace = valid.groupby(["team", "compound"], as_index=False).agg(
        raw_intercept=("raw_intercept", "mean"),
        corrected_intercept=("corrected_intercept", "mean"),
        fuel_correction=("fuel_correction", "mean"),
        track_correction=("track_correction", "mean"),
    )
    rows = []
    for team, group in pace.groupby("team", sort=True):
        hard = group.loc[group["compound"].eq("HARD")]
        if hard.empty:
            continue
        hard_row = hard.iloc[0]
        for compound in ("MEDIUM", "SOFT"):
            target = race_targets.get(compound)
            comparison = group.loc[group["compound"].eq(compound)]
            if target is None or comparison.empty:
                continue
            comparison_row = comparison.iloc[0]
            raw_prediction = float(
                comparison_row.raw_intercept - hard_row.raw_intercept
            )
            corrected_prediction = float(
                comparison_row.corrected_intercept - hard_row.corrected_intercept
            )
            rows.append({
                "team": team,
                "compound": compound,
                "hard_raw_intercept": float(hard_row.raw_intercept),
                "comparison_raw_intercept": float(comparison_row.raw_intercept),
                "hard_corrected_intercept": float(hard_row.corrected_intercept),
                "comparison_corrected_intercept": float(
                    comparison_row.corrected_intercept
                ),
                "hard_fuel_correction": float(hard_row.fuel_correction),
                "comparison_fuel_correction": float(
                    comparison_row.fuel_correction
                ),
                "hard_track_correction": float(hard_row.track_correction),
                "comparison_track_correction": float(
                    comparison_row.track_correction
                ),
                "uncorrected_prediction": raw_prediction,
                "corrected_prediction": corrected_prediction,
                "race_target": float(target),
                "uncorrected_residual": raw_prediction - float(target),
                "corrected_residual": corrected_prediction - float(target),
                "same_team_pair": True,
            })
    return pd.DataFrame(rows, columns=columns)


def centre_programme_offset_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Remove event/session pace level; the remainder is not literal fuel mass."""
    if observations.empty:
        return observations.copy()
    output = observations.copy()
    output["event_session_common_offset"] = output.groupby(
        ["season", "round", "session"], sort=True
    )["raw_programme_residual"].transform("mean")
    output["relative_programme_offset"] = (
        output["raw_programme_residual"]
        - output["event_session_common_offset"]
    )
    return output


def programme_offset_stability(
    observations: pd.DataFrame,
    *,
    minimum_events: int = PROGRAMME_OFFSET_MIN_EVENTS,
) -> pd.DataFrame:
    """Describe transfer stability without interpreting offsets as fuel kg."""
    columns = [
        "grouping_level", "team", "session", "run_slot", "observation_count",
        "event_count", "mean_relative_offset", "median_relative_offset",
        "standard_deviation", "median_absolute_deviation", "minimum_offset",
        "maximum_offset", "event_to_event_range", "sufficient_event_support",
    ]
    rows: list[dict[str, object]] = []
    if observations.empty:
        return pd.DataFrame(columns=columns)
    grouping_options = [("team_session", ["team", "session"])]
    if observations["run_slot"].nunique() > 1:
        grouping_options.append(
            ("team_session_run_slot", ["team", "session", "run_slot"])
        )
    for level, group_columns in grouping_options:
        for key, group in observations.groupby(group_columns, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            identity = dict(zip(group_columns, values))
            event_values = group.groupby("round", sort=True)[
                "relative_programme_offset"
            ].median()
            median = float(event_values.median())
            mad = float(np.median(np.abs(event_values.to_numpy() - median)))
            rows.append({
                "grouping_level": level,
                "team": identity["team"],
                "session": identity["session"],
                "run_slot": identity.get("run_slot"),
                "observation_count": int(len(group)),
                "event_count": int(len(event_values)),
                "mean_relative_offset": float(event_values.mean()),
                "median_relative_offset": median,
                "standard_deviation": float(event_values.std(ddof=0)),
                "median_absolute_deviation": mad,
                "minimum_offset": float(event_values.min()),
                "maximum_offset": float(event_values.max()),
                "event_to_event_range": float(event_values.max() - event_values.min()),
                "sufficient_event_support": len(event_values) >= minimum_events,
            })
    return pd.DataFrame(rows, columns=columns)


def programme_offset_variance(observations: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "session", "observation_count", "event_count", "total_variance",
        "event_session_common_variance", "team_specific_residual_variance",
        "event_session_common_fraction", "team_specific_residual_fraction",
    ]
    rows: list[dict[str, object]] = []
    if observations.empty:
        return pd.DataFrame(columns=columns)
    for session, group in observations.groupby("session", sort=True):
        total = float(group["raw_programme_residual"].var(ddof=0))
        common = float(group["event_session_common_offset"].var(ddof=0))
        residual = float(group["relative_programme_offset"].var(ddof=0))
        denominator = common + residual
        rows.append({
            "session": session,
            "observation_count": int(len(group)),
            "event_count": int(group["round"].nunique()),
            "total_variance": total,
            "event_session_common_variance": common,
            "team_specific_residual_variance": residual,
            "event_session_common_fraction": (
                common / denominator if denominator > 0 else None
            ),
            "team_specific_residual_fraction": (
                residual / denominator if denominator > 0 else None
            ),
        })
    return pd.DataFrame(rows, columns=columns)


def rolling_programme_offset_validation(
    corrected_runs: pd.DataFrame,
    bridges: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    minimum_training_events: int = PROGRAMME_OFFSET_MIN_EVENTS,
) -> pd.DataFrame:
    """Apply only earlier-event effective programme offsets to held-out bridges."""
    rows: list[dict[str, object]] = []
    if bridges.empty:
        return pd.DataFrame()
    for bridge in bridges.itertuples(index=False):
        held_runs = corrected_runs.loc[
            corrected_runs["round"].eq(bridge.round)
            & corrected_runs["session"].eq(bridge.session)
            & corrected_runs["team"].eq(bridge.team)
            & corrected_runs["compound"].isin(("HARD", bridge.compound))
        ].copy()
        training = observations.loc[
            observations["round"].lt(bridge.round)
            & observations["session"].eq(bridge.session)
            & observations["team"].eq(bridge.team)
        ].copy()
        assert_pre_event_training_cutoff(training["round"], int(bridge.round))
        corrected_values: list[dict[str, object]] = []
        unsupported: list[str] = []
        for run in held_runs.itertuples(index=False):
            slot_training = training.loc[training["run_slot"].eq(run.run_slot)]
            if (
                training["run_slot"].nunique() > 1
                and slot_training["round"].nunique() >= minimum_training_events
            ):
                selected = slot_training
                level = "team_session_run_slot"
            elif training["round"].nunique() >= minimum_training_events:
                selected = training
                level = "team_session"
            else:
                unsupported.append(str(run.run_id))
                continue
            effective_offset = float(selected["relative_programme_offset"].median())
            corrected_values.append({
                "compound": run.compound,
                "programme_adjusted_intercept": (
                    float(run.corrected_intercept) - effective_offset
                ),
                "effective_offset": effective_offset,
                "grouping_level": level,
                "training_event_count": int(selected["round"].nunique()),
                "trained_through_round": int(selected["round"].max()),
            })
        eligible = not unsupported and set(held_runs["compound"]) == {
            "HARD", bridge.compound
        }
        adjusted_prediction = None
        selected_levels = None
        training_event_count = int(training["round"].nunique())
        trained_through_round = (
            int(training["round"].max()) if not training.empty else None
        )
        if eligible:
            adjusted = pd.DataFrame(corrected_values)
            adjusted_prediction = float(
                adjusted.loc[
                    adjusted["compound"].eq(bridge.compound),
                    "programme_adjusted_intercept",
                ].mean()
                - adjusted.loc[
                    adjusted["compound"].eq("HARD"),
                    "programme_adjusted_intercept",
                ].mean()
            )
            selected_levels = json.dumps(sorted(set(adjusted["grouping_level"])))
            training_event_count = int(adjusted["training_event_count"].min())
            trained_through_round = int(adjusted["trained_through_round"].max())
        rows.append({
            "season": int(bridge.season),
            "round": int(bridge.round),
            "event": bridge.event,
            "session": bridge.session,
            "team": bridge.team,
            "compound": bridge.compound,
            "eligible": bool(eligible),
            "eligibility_reason": (
                "supported" if eligible
                else "fewer_than_two_prior_events_for_one_or_more_runs"
            ),
            "unsupported_run_ids": json.dumps(sorted(unsupported)),
            "offset_grouping_levels": selected_levels,
            "training_event_count": training_event_count,
            "trained_through_round": trained_through_round,
            "uncorrected_prediction": float(bridge.uncorrected_prediction),
            "corrected_prediction": float(bridge.corrected_prediction),
            "programme_adjusted_prediction": adjusted_prediction,
            "race_target": float(bridge.race_target),
            "uncorrected_residual": float(bridge.uncorrected_residual),
            "corrected_residual": float(bridge.corrected_residual),
            "programme_adjusted_residual": (
                adjusted_prediction - float(bridge.race_target)
                if adjusted_prediction is not None else None
            ),
            "historical_pre_race_prediction": float(
                bridge.historical_pre_race_prediction
            ),
            "historical_pre_race_residual": float(
                bridge.historical_pre_race_residual
            ),
            "pirelli_pre_race_prediction": float(
                bridge.pirelli_pre_race_prediction
            ),
            "pirelli_pre_race_residual": float(bridge.pirelli_pre_race_residual),
            "physical_interpretation": (
                "effective pace offset: starting load + engine mode + programme conventions; not fuel kg"
            ),
        })
    return pd.DataFrame(rows)


def normal_model_coordinates(
    session_frames: Mapping[str, pd.DataFrame],
    model_config: Mapping[str, object],
) -> pd.DataFrame:
    prepared, _ = prepare_weekend_sessions(
        session_frames,
        min_clean_air_laps=int(model_config.get("min_clean_air_laps", 4)),
        clean_mean_time_delta_seconds=float(
            model_config.get("clean_mean_time_delta_seconds", 2.5)
        ),
        clean_mean_time_delta_behind_seconds=model_config.get(
            "clean_mean_time_delta_behind_seconds", 1.0
        ),
        quick_lap_threshold=float(model_config.get("quick_lap_threshold", 1.10)),
        treat_stint_as_whole=False,
        dry_compounds=tuple(model_config.get(
            "expected_compounds", ("SOFT", "MEDIUM", "HARD")
        )),
    )
    frames = []
    for session, laps in prepared.items():
        coordinates = _prepare_analysis_laps(laps)
        coordinates["session"] = session
        frames.append(coordinates)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def performance_comparisons(
    runs: pd.DataFrame,
    team_baselines: Mapping[str, float],
    race_targets: Mapping[str, float],
) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    runs = runs.loc[runs["team"].astype(str).str.strip().ne("")].copy()
    if runs.empty:
        return pd.DataFrame()
    pace = runs.groupby(["team", "compound"], as_index=False).agg(
        pace_seconds=("tyre_age_zero_pace_seconds", "mean"),
        run_count=("run_id", "nunique"), clean_lap_count=("clean_lap_count", "sum"),
    )
    hard = pace.loc[pace["compound"].eq("HARD")]
    rows: list[dict[str, object]] = []
    for compound in ("MEDIUM", "SOFT"):
        target = race_targets.get(compound)
        if target is None or hard.empty:
            continue
        candidate = pace.loc[pace["compound"].eq(compound)]
        for h in hard.itertuples(index=False):
            for c in candidate.itertuples(index=False):
                same_team = h.team == c.team
                if not same_team and (h.team not in team_baselines or c.team not in team_baselines):
                    continue
                correction = baseline_corrected_delta(
                    left_pace=h.pace_seconds, right_pace=c.pace_seconds,
                    left_baseline=team_baselines.get(h.team, 0.0),
                    right_baseline=team_baselines.get(c.team, 0.0),
                )
                rows.append({
                    "strategy": "Perf-1" if same_team else "Perf-2",
                    "hard_team": h.team, "comparison_team": c.team, "compound": compound,
                    "hard_pace_seconds": h.pace_seconds, "comparison_pace_seconds": c.pace_seconds,
                    "team_baseline_source": "preceding_completed_race_retro_team_baseline",
                    **correction, "prediction": correction["compound_delta_estimate"],
                    "race_target": float(target),
                    "residual": correction["compound_delta_estimate"] - float(target),
                })
    return pd.DataFrame(rows)


def joint_partial_fit(
    observations: pd.DataFrame,
    *,
    team_baselines: Mapping[str, float],
    performance_priors: Mapping[str, float],
    degradation_priors: Mapping[str, float],
    team_strength: float = 4.0,
    performance_strength: float = 2.0,
    degradation_strength: float = 2.0,
) -> JointFitResult:
    required = {"team", "compound", "tyre_age", "lap_time_s"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Joint observations are missing columns: {sorted(missing)}")
    frame = observations.dropna(subset=list(required)).copy()
    frame = frame.loc[
        frame["team"].astype(str).str.strip().ne("")
        & frame["compound"].astype(str).str.strip().ne("")
    ].copy()
    teams = sorted(frame["team"].astype(str).unique())
    compounds = [c for c in COMPOUNDS if c in set(frame["compound"].astype(str).str.upper())]
    names = [f"B:{team}" for team in teams]
    names += [f"P:{c}" for c in compounds if c != "HARD"]
    names += [f"D:{c}" for c in compounds]
    index = {name: position for position, name in enumerate(names)}
    x = np.zeros((len(frame), len(names)))
    for row_index, row in enumerate(frame.itertuples(index=False)):
        team = str(row.team); compound = str(row.compound).upper()
        x[row_index, index[f"B:{team}"]] = 1.0
        if compound != "HARD" and f"P:{compound}" in index:
            x[row_index, index[f"P:{compound}"]] = 1.0
        x[row_index, index[f"D:{compound}"]] = float(row.tyre_age)
    y = frame["lap_time_s"].astype(float).to_numpy()
    rank = int(np.linalg.matrix_rank(x))
    prior_rows: list[np.ndarray] = []
    prior_values: list[float] = []
    centred_baselines = {
        team: float(team_baselines.get(team, np.mean(y)))
        for team in teams
    }
    if centred_baselines:
        offset = np.mean(list(centred_baselines.values())) - np.mean(y)
        centred_baselines = {key: value - offset for key, value in centred_baselines.items()}
    for name in names:
        vector = np.zeros(len(names)); vector[index[name]] = 1.0
        prefix, value = name.split(":", 1)
        if prefix == "B":
            strength, prior = team_strength, centred_baselines[value]
        elif prefix == "P":
            strength, prior = performance_strength, float(performance_priors[value])
        else:
            strength, prior = degradation_strength, float(degradation_priors[value])
        prior_rows.append(np.sqrt(strength) * vector)
        prior_values.append(np.sqrt(strength) * prior)
    augmented_x = np.vstack([x, *prior_rows])
    augmented_y = np.concatenate([y, prior_values])
    # Small transparent active-set solve: degradation coefficients alone are
    # constrained non-negative; team and performance coordinates stay free.
    free = list(range(len(names)))
    fixed: dict[int, float] = {}
    while True:
        adjusted_y = augmented_y.copy()
        for position, value in fixed.items():
            adjusted_y -= augmented_x[:, position] * value
        candidate = np.zeros(len(names))
        if free:
            candidate[free] = np.linalg.lstsq(augmented_x[:, free], adjusted_y, rcond=None)[0]
        for position, value in fixed.items():
            candidate[position] = value
        negative = [
            position for name, position in index.items()
            if name.startswith("D:") and position in free and candidate[position] < 0
        ]
        if not negative:
            beta = candidate
            break
        worst = min(negative, key=lambda position: candidate[position])
        free.remove(worst)
        fixed[worst] = 0.0
    estimates = {name: float(beta[position]) for name, position in index.items()}
    estimates["P:HARD"] = 0.0
    return JointFitResult(
        estimates=estimates, design_rank=rank, parameter_count=len(names),
        data_row_count=len(frame), unregularised_identifiable=rank == len(names),
        regularisation={"team": team_strength, "performance": performance_strength, "degradation": degradation_strength},
    )


def assert_chronology(sessions_used: Iterable[str], information_state: str) -> None:
    cutoff = SESSION_ORDER[str(information_state).upper()]
    later = [session for session in sessions_used if SESSION_ORDER[str(session).upper()] > cutoff]
    if later:
        raise ValueError(f"Later-session leakage for {information_state}: {sorted(later)}")


def assert_pre_event_training_cutoff(training_rounds: Iterable[int], target_round: int) -> None:
    leaked = sorted({int(value) for value in training_rounds if int(value) >= int(target_round)})
    if leaked:
        raise ValueError(
            f"Target/future Race leakage for round {target_round}: training rounds {leaked}"
        )


def _read_priors_and_targets(season: int, round_number: int, data_root: Path):
    path = tyre_prediction_year_root(season, data_root) / "validation_predictions.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame["round"].eq(round_number) & frame["weighting_policy"].eq("uniform")
    ].copy()
    assert_pre_event_training_cutoff(frame["training_round_end"].dropna(), round_number)
    families = {
        "historical_baseline": (PRODUCTION_POLICY["performance"]["formulation_code"], PRODUCTION_POLICY["degradation"]["formulation_code"]),
        "pirelli_informed": (PIRELLI_INFORMED_POLICY["performance"]["formulation_code"], PIRELLI_INFORMED_POLICY["degradation"]["formulation_code"]),
    }
    output = {}
    for family, (p_code, d_code) in families.items():
        p = frame.loc[frame["model_code"].eq(p_code)].set_index("compound_role")["predicted_value"].to_dict()
        p["HARD"] = 0.0
        d = frame.loc[frame["model_code"].eq(d_code)].set_index("compound_role")["predicted_value"].to_dict()
        output[family] = {"performance": p, "degradation": d}
    targets = frame.drop_duplicates("compound_role").set_index("compound_role")["observed_value"].to_dict()
    retro = pd.read_csv(tyre_prediction_year_root(season, data_root) / f"retro_tyre_observations_{season}.csv")
    event = retro.loc[retro["round"].eq(round_number)].iloc[0]["event"]
    performance_targets = retro.loc[retro["round"].eq(round_number)].set_index("compound_role")["performance"].to_dict()
    performance_targets["HARD"] = 0.0
    return output, targets, performance_targets, str(event), path


def _fp_parameters(data_root: Path, season: int, round_number: int, session: str) -> pd.DataFrame:
    directory = data_root / "wostrategy/weekend_model" / SCHEMA_VERSION / f"year={season}" / f"round={round_number}" / "sessions" / session
    path = directory / "latest_parameters.csv"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    contributors = {
        str(value).upper() for value in manifest.get("contributing_sessions", ())
    }
    if contributors != {str(session).upper()}:
        raise ValueError(
            f"Persisted {session} snapshot is not leakage-safe; "
            f"contributors={sorted(contributors)}"
        )
    return pd.read_csv(path)


def _previous_team_baselines(data_root: Path, season: int, round_number: int) -> tuple[dict[str, float], Path]:
    previous = round_number - 1
    root = data_root / "wostrategy/race_performance_review" / SCHEMA_VERSION / f"year={season}" / f"round={previous}" / "session=R"
    path = root / f"race_performance_{season}_{previous}_R_team_baseline_summary.csv"
    if not path.is_file():
        return {}, path
    frame = pd.read_csv(path)
    return frame.set_index("Team")["Median"].astype(float).to_dict(), path


def _race_team_baselines(
    data_root: Path, season: int, round_number: int
) -> tuple[dict[str, float], Path]:
    root = (
        data_root / "wostrategy/race_performance_review" / SCHEMA_VERSION
        / f"year={season}" / f"round={round_number}" / "session=R"
    )
    path = root / (
        f"race_performance_{season}_{round_number}_R_team_baseline_summary.csv"
    )
    if not path.is_file():
        return {}, path
    frame = pd.read_csv(path)
    return frame.set_index("Team")["Median"].astype(float).to_dict(), path


def run_historical_fp_race_diagnostic(
    *,
    season: int = 2026,
    data_root: str | Path | None = None,
    trained_through_round: int | None = None,
) -> dict[str, Path]:
    root = get_data_root(data_root)
    event_roots = sorted((root / "wostrategy/weekend_model" / SCHEMA_VERSION / f"year={season}").glob("round=*"))
    candidates: list[int] = []
    for event_root in event_roots:
        available = {p.parent.name for p in event_root.glob("sessions/FP*/latest_parameters.csv")}
        if available:
            candidates.append(int(event_root.name.split("=")[1]))
    if trained_through_round is not None:
        candidates = [
            value for value in candidates
            if value <= int(trained_through_round)
        ]
    candidates.sort()
    degradation_rows: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []
    performance_rows: list[dict[str, object]] = []
    corrected_run_rows: list[dict[str, object]] = []
    corrected_performance_rows: list[dict[str, object]] = []
    programme_observation_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    provenance: list[str] = []
    skipped: list[str] = []
    for round_number in candidates:
        try:
            priors, deg_targets, perf_targets, event, prior_path = _read_priors_and_targets(season, round_number, root)
            baselines, baseline_path = _previous_team_baselines(root, season, round_number)
            race_baselines, race_baseline_path = _race_team_baselines(
                root, season, round_number
            )
            current_event_root = (
                root / "wostrategy/weekend_model" / SCHEMA_VERSION
                / f"year={season}" / f"round={round_number}"
            )
            lap_root = root / "fastf1/session_laps" / SCHEMA_VERSION / f"year={season}" / f"round={round_number}"
            session_frames = {
                session: pd.read_pickle(lap_root / f"session={session}/laps.pkl")
                for session in SESSION_ORDER
                if (lap_root / f"session={session}/laps.pkl").is_file()
                and (current_event_root / "sessions" / session / "latest_parameters.csv").is_file()
            }
        except (FileNotFoundError, IndexError, pd.errors.EmptyDataError) as exc:
            skipped.append(f"round {round_number}: {exc}")
            continue
        provenance.append(str(prior_path))
        if baseline_path.is_file():
            provenance.append(str(baseline_path))
        if race_baseline_path.is_file():
            provenance.append(str(race_baseline_path))
        # Reuse the existing strict programme/clean-lap calculation once for the
        # weekend.  Some individual sessions have no estimable degradation
        # contrast, while their selected strict laps remain valid diagnostics.
        try:
            weekend_evidence = analyse_fp_tyre_evidence_sessions(
                session_frames, year=season, round_number=round_number
            )
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append(f"round {round_number}: FP evidence unsupported: {exc}")
            continue
        weekend_laps = pd.DataFrame(weekend_evidence["longrun_laps"])
        model_config_path = current_event_root / "model_config.json"
        model_config = (
            json.loads(model_config_path.read_text(encoding="utf-8"))
            if model_config_path.is_file()
            else {}
        )
        model_coordinates = normal_model_coordinates(session_frames, model_config)
        for session in session_frames:
            assert_chronology([session], session)
            session_laps = weekend_laps.loc[weekend_laps["session"].eq(session)].copy()
            runs = fit_run_summaries(session_laps)
            topology = classify_topology(runs[["team", "compound"]])
            topology_rows.append({
                "season": season, "round": round_number, "event": event, "session": session,
                **topology, "clean_lap_count": int(runs["clean_lap_count"].sum()),
                "run_count": int(runs["run_id"].nunique()), "stint_count": int(runs["run_id"].nunique()),
            })
            try:
                fp_parameters = _fp_parameters(root, season, round_number, session)
            except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
                skipped.append(f"round {round_number} {session}: {exc}")
                continue
            provenance.append(str(root / "wostrategy/weekend_model" / SCHEMA_VERSION / f"year={season}" / f"round={round_number}" / "sessions" / session / "latest_parameters.csv"))
            supported_compounds = set(topology["compounds"])
            fp_deg = (
                fp_parameters.loc[
                    fp_parameters["parameter"].eq("degradation")
                    & fp_parameters["compound"].isin(supported_compounds)
                ]
                .set_index("compound")["median"].astype(float).to_dict()
            )
            fuel_rows = fp_parameters.loc[
                fp_parameters["parameter"].eq("fuel_rate"), "median"
            ]
            track_rows = fp_parameters.loc[
                fp_parameters["parameter"].eq("track_rate"), "median"
            ]
            if not fuel_rows.empty and not track_rows.empty:
                session_coordinates = (
                    model_coordinates.loc[model_coordinates["session"].eq(session)]
                    if "session" in model_coordinates
                    else pd.DataFrame()
                )
                corrected_runs = corrected_run_summaries(
                    session_laps,
                    session_coordinates,
                    fuel_rate=float(fuel_rows.iloc[0]),
                    track_rate=float(track_rows.iloc[0]),
                )
                for row in corrected_runs.to_dict("records"):
                    corrected_run_rows.append({
                        "season": season,
                        "round": round_number,
                        "event": event,
                        **row,
                    })
                    team = str(row["team"])
                    compound = str(row["compound"])
                    if team in race_baselines and compound in perf_targets:
                        race_baseline = float(race_baselines[team])
                        retro_performance = float(perf_targets[compound])
                        programme_observation_rows.append({
                            "season": season,
                            "round": round_number,
                            "event": event,
                            **row,
                            "race_team_baseline": race_baseline,
                            "retro_compound_performance": retro_performance,
                            "raw_programme_residual": (
                                float(row["corrected_intercept"])
                                - race_baseline
                                - retro_performance
                            ),
                            "coordinate_compatibility": (
                                "tyre_age_zero_seconds; Race team baseline + HARD-relative Retro performance"
                            ),
                            "physical_interpretation": (
                                "effective pace offset: starting load + engine mode + programme conventions; not fuel kg"
                            ),
                        })
                corrected_comparisons = same_team_corrected_comparisons(
                    corrected_runs, perf_targets
                )
                for row in corrected_comparisons.to_dict("records"):
                    compound = str(row["compound"])
                    historical = priors["historical_baseline"]["performance"].get(
                        compound
                    )
                    pirelli = priors["pirelli_informed"]["performance"].get(
                        compound
                    )
                    corrected_performance_rows.append({
                        "season": season,
                        "round": round_number,
                        "event": event,
                        "session": session,
                        **row,
                        "historical_pre_race_prediction": historical,
                        "historical_pre_race_residual": (
                            float(historical) - float(row["race_target"])
                            if historical is not None else None
                        ),
                        "pirelli_pre_race_prediction": pirelli,
                        "pirelli_pre_race_residual": (
                            float(pirelli) - float(row["race_target"])
                            if pirelli is not None else None
                        ),
                    })
            support_fields = {
                "clean_lap_count": int(runs["clean_lap_count"].sum()),
                "run_count": int(runs["run_id"].nunique()),
                "stint_count": int(runs["run_id"].nunique()),
                "team_count": int(topology["supported_team_count"]),
                "compound_count": int(topology["supported_compound_count"]),
                "topology_category": topology["topology_category"],
            }
            for family, values in priors.items():
                for compound in sorted(set(fp_deg).intersection(deg_targets).intersection(values["degradation"])):
                    prior_value = float(values["degradation"][compound])
                    fp_value = float(fp_deg[compound])
                    common = {"season": season, "round": round_number, "event": event, "session": session,
                              "prior_family": family, "compound": compound,
                              **support_fields,
                              "pre_race_value": prior_value,
                              "fp_value": fp_value,
                              "race_target": float(deg_targets[compound])}
                    for weight in UPDATE_WEIGHTS:
                        prediction = partial_degradation_update(
                            prior=prior_value, fp=fp_value, weight=weight
                        )
                        strategy = (
                            "Deg-0" if weight == 0.0
                            else "Deg-1" if weight == 1.0
                            else "Deg-partial"
                        )
                        degradation_rows.append({
                            **common,
                            "strategy": strategy,
                            "update_weight": weight,
                            "partially_updated_value": prediction,
                            "prediction": prediction,
                            "residual": prediction - float(deg_targets[compound]),
                        })
                loo = leave_one_compound_out_k(priors=values["degradation"], fp=fp_deg, race_targets=deg_targets)
                for row in loo.to_dict("records"):
                    for weight in UPDATE_WEIGHTS:
                        partial = partial_k_update(
                            prior=float(row["prior_prediction"]),
                            raw_k=float(row["k_event"]),
                            weight=weight,
                        )
                        transfer_rows.append({
                            "season": season,
                            "round": round_number,
                            "event": event,
                            "session": session,
                            "prior_family": family,
                            **support_fields,
                            **row,
                            "raw_k": float(row["k_event"]),
                            "update_weight": weight,
                            "effective_k": partial["effective_k"],
                            "prediction": partial["prediction"],
                            "residual": partial["prediction"] - float(row["race_target"]),
                        })
                comparisons = performance_comparisons(runs, baselines, perf_targets)
                for row in comparisons.to_dict("records"):
                    performance_rows.append({"season": season, "round": round_number, "event": event,
                                             "session": session, "prior_family": family,
                                             "topology_category": topology["topology_category"], **row})
                for compound in ("MEDIUM", "SOFT"):
                    if compound in perf_targets and compound in values["performance"]:
                        performance_rows.append({"season": season, "round": round_number, "event": event,
                                                 "session": session, "prior_family": family, "strategy": "Perf-0",
                                                 "topology_category": topology["topology_category"],
                                                 "compound": compound, "prediction": values["performance"][compound],
                                                 "race_target": perf_targets[compound],
                                                 "residual": values["performance"][compound] - perf_targets[compound]})
                used = session_laps.copy()
                used = used.loc[used["used_for_quantitative_inference"]]
                used = used.loc[
                    used["team"].astype(str).str.strip().ne("")
                    & used["compound"].astype(str).str.strip().ne("")
                ]
                if used.empty or not baselines:
                    continue
                observed_compounds = set(used["compound"].astype(str).str.upper())
                if not observed_compounds.issubset(values["degradation"]):
                    continue
                if not observed_compounds.difference({"HARD"}).issubset(
                    values["performance"]
                ):
                    continue
                joint = joint_partial_fit(
                    used,
                    team_baselines=baselines,
                    performance_priors=values["performance"],
                    degradation_priors=values["degradation"],
                )
                for compound in COMPOUNDS:
                    if f"D:{compound}" in joint.estimates and compound in deg_targets:
                        joint_rows.append({"season": season, "round": round_number, "event": event,
                                           "session": session, "prior_family": family, "target_type": "degradation",
                                           "compound": compound, "prediction": joint.estimates[f"D:{compound}"],
                                           "race_target": deg_targets[compound], "design_rank": joint.design_rank,
                                           "parameter_count": joint.parameter_count, "data_row_count": joint.data_row_count,
                                           "unregularised_identifiable": joint.unregularised_identifiable,
                                           "regularisation": json.dumps(joint.regularisation, sort_keys=True)})
                    if compound != "HARD" and f"P:{compound}" in joint.estimates and compound in perf_targets:
                        joint_rows.append({"season": season, "round": round_number, "event": event,
                                           "session": session, "prior_family": family, "target_type": "performance",
                                           "compound": compound, "prediction": joint.estimates[f"P:{compound}"],
                                           "race_target": perf_targets[compound], "design_rank": joint.design_rank,
                                           "parameter_count": joint.parameter_count, "data_row_count": joint.data_row_count,
                                           "unregularised_identifiable": joint.unregularised_identifiable,
                                           "regularisation": json.dumps(joint.regularisation, sort_keys=True)})
    degradation_frame = pd.DataFrame(degradation_rows, columns=(
        None if degradation_rows else [
            "season", "round", "event", "session", "prior_family",
            "compound", "strategy", "update_weight", "prediction",
            "race_target", "residual", "topology_category",
        ]
    ))
    transfer_frame = pd.DataFrame(transfer_rows, columns=(
        None if transfer_rows else [
            "season", "round", "event", "session", "prior_family",
            "held_out_compound", "update_weight", "prediction",
            "race_target", "residual", "topology_category",
        ]
    ))
    degradation_event_frame = event_level_metrics(degradation_frame)
    transfer_event_frame = event_level_metrics(transfer_frame)
    calibration_frame = pd.concat(
        [
            calibration_history(degradation_frame, update_type="direct"),
            calibration_history(transfer_frame, update_type="partial_k"),
        ],
        ignore_index=True,
    )
    corrected_runs_frame = pd.DataFrame(corrected_run_rows)
    corrected_performance_frame = pd.DataFrame(corrected_performance_rows)
    programme_observations = centre_programme_offset_observations(
        pd.DataFrame(programme_observation_rows)
    )
    programme_stability = programme_offset_stability(programme_observations)
    programme_variance = programme_offset_variance(programme_observations)
    rolling_programme = rolling_programme_offset_validation(
        corrected_runs_frame,
        corrected_performance_frame,
        programme_observations,
    )
    frames = {
        "fp_race_degradation_validation.csv": degradation_frame,
        "fp_degradation_partial_update_event_metrics.csv": degradation_event_frame,
        "fp_degradation_partial_update_loeo.csv": (
            leave_one_event_out_weight_selection(degradation_event_frame)
        ),
        "fp_degradation_transfer_validation.csv": transfer_frame,
        "fp_degradation_transfer_partial_update_event_metrics.csv": transfer_event_frame,
        "fp_degradation_transfer_partial_update_loeo.csv": (
            leave_one_event_out_weight_selection(transfer_event_frame)
        ),
        "fp_degradation_calibration_history.csv": calibration_frame,
        "fp_performance_validation.csv": pd.DataFrame(performance_rows),
        "fp_corrected_run_intercepts.csv": corrected_runs_frame,
        "fp_corrected_same_team_performance_validation.csv": corrected_performance_frame,
        "fp_programme_offset_observations.csv": programme_observations,
        "fp_programme_offset_stability.csv": programme_stability,
        "fp_programme_offset_variance.csv": programme_variance,
        "fp_programme_offset_rolling_validation.csv": rolling_programme,
        "fp_information_topology.csv": pd.DataFrame(topology_rows),
        "fp_joint_fit_validation.csv": pd.DataFrame(joint_rows),
    }
    output_root = diagnostic_root(season, root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, frame in frames.items():
        path = output_root / filename
        frame.to_csv(path, index=False)
        paths[filename] = path
    summary = _build_summary(
        frames, season=season, candidates=candidates, skipped=skipped,
        provenance=provenance,
    )
    summary_path = output_root / "fp_race_diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[summary_path.name] = summary_path
    return paths


def build_latest_calibration_snapshot(
    history: pd.DataFrame,
    *,
    season: int,
    completed_retro_through_round: int,
) -> dict[str, object]:
    """Freeze the latest diagnostic state for use by later events only."""
    tracks: list[dict[str, object]] = []
    if not history.empty:
        required = {"session", "prior_family", "update_type", "trained_through_round"}
        missing = required.difference(history.columns)
        if missing:
            raise ValueError(f"Calibration history is missing columns: {sorted(missing)}")
        causal = history.loc[
            pd.to_numeric(history["trained_through_round"], errors="coerce")
            .le(int(completed_retro_through_round))
        ].copy()
        for key, group in causal.groupby(
            ["session", "prior_family", "update_type"], sort=True
        ):
            latest_round = int(group["trained_through_round"].max())
            latest = group.loc[group["trained_through_round"].eq(latest_round)].copy()
            preferred = latest.loc[latest["is_diagnostic_grid_preferred"].astype(bool)]
            state = preferred.iloc[0] if not preferred.empty else latest.iloc[0]
            candidates = []
            for row in latest.sort_values("candidate_weight").to_dict("records"):
                candidates.append({
                    "weight": float(row["candidate_weight"]),
                    "prediction_count": int(row["prediction_count"]),
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "bias": float(row["bias"]),
                    "mean_event_mae": float(row["mean_event_mae"]),
                    "loeo_selection_count": int(row["loeo_selection_count"]),
                })
            tracks.append({
                "session": key[0],
                "prior_family": key[1],
                "update_type": key[2],
                "calibration_version": str(state["calibration_version"]),
                "calibration_method": str(state["calibration_method"]),
                "trained_through_round": latest_round,
                "training_rounds": json.loads(str(state["training_rounds"])),
                "event_count": int(state["event_count"]),
                "prediction_count": int(state["prediction_count"]),
                "candidate_weights": json.loads(str(state["candidate_weights"])),
                "candidate_metrics": candidates,
                "diagnostic_grid_preferred_weight": float(
                    state["diagnostic_grid_preferred_weight"]
                ),
                "loeo_modal_weight": (
                    None if pd.isna(state["loeo_modal_weight"])
                    else float(state["loeo_modal_weight"])
                ),
                "loeo_modal_share": (
                    None if pd.isna(state["loeo_modal_share"])
                    else float(state["loeo_modal_share"])
                ),
                "loeo_selected_weights": json.loads(
                    str(state["loeo_selected_weights"])
                ),
                "loeo_stability": str(state["loeo_stability"]),
                "calibration_status": CALIBRATION_STATUS,
                "production_weight_selected": False,
            })
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "season": int(season),
        "completed_retro_through_round": int(completed_retro_through_round),
        "calibration_version": CALIBRATION_VERSION,
        "calibration_method": CALIBRATION_METHOD,
        "calibration_status": CALIBRATION_STATUS,
        "candidate_weights": list(UPDATE_WEIGHTS),
        "production_weight_selected": False,
        "tracks": tracks,
        "future_calibration_note": (
            "V1 keeps a fixed diagnostic grid. A continuous event-level calibration "
            "may replace it only after substantially more independent events."
        ),
    }
    payload["calibration_fingerprint"] = canonical_hash(payload)
    return payload


def refresh_fp_degradation_calibration(
    *,
    season: int,
    completed_retro_through_round: int,
    data_root: str | Path | None = None,
) -> Path:
    """Refresh future-event calibration after Retro persistence; never edits predictions."""
    root = get_data_root(data_root)
    history_path = diagnostic_root(season, root) / "fp_degradation_calibration_history.csv"
    event_root = (
        root / "wostrategy/weekend_model" / SCHEMA_VERSION / f"year={season}"
    )
    has_fp_history = any(event_root.glob("round=*/sessions/FP*/latest_parameters.csv"))
    validation_path = tyre_prediction_year_root(season, root) / "validation_predictions.csv"
    if has_fp_history and validation_path.is_file():
        run_historical_fp_race_diagnostic(
            season=season,
            data_root=root,
            trained_through_round=completed_retro_through_round,
        )
    history = pd.DataFrame()
    if history_path.is_file() and history_path.stat().st_size > 0:
        try:
            history = pd.read_csv(history_path)
        except pd.errors.EmptyDataError:
            history = pd.DataFrame()
    destination = calibration_snapshot_root(season, root)
    observations_path = observations_csv_path(season, root)
    completed_rounds = [int(completed_retro_through_round)]
    if observations_path.is_file():
        retro = pd.read_csv(observations_path)
        completed_rounds = sorted({
            int(value) for value in retro["round"].dropna()
            if int(value) <= int(completed_retro_through_round)
        }) or completed_rounds
    latest_snapshot = None
    for cutoff in completed_rounds:
        snapshot = build_latest_calibration_snapshot(
            history,
            season=season,
            completed_retro_through_round=cutoff,
        )
        version_path = (
            destination / "versions" / f"through_round={cutoff}.json"
        )
        # A cutoff version is a frozen historical information state. Initial
        # integration backfills missing versions; later refreshes append only.
        if not version_path.exists():
            version_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = version_path.with_name(f".{version_path.name}.tmp")
            temporary.write_text(
                json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(version_path)
        if cutoff == int(completed_retro_through_round):
            latest_snapshot = snapshot
    if latest_snapshot is None:
        latest_snapshot = build_latest_calibration_snapshot(
            history,
            season=season,
            completed_retro_through_round=completed_retro_through_round,
        )
    latest_path = destination / "latest_calibration.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = latest_path.with_name(f".{latest_path.name}.tmp")
    temporary.write_text(
        json.dumps(latest_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(latest_path)
    return latest_path


def _build_summary(frames, *, season, candidates, skipped, provenance):
    degradation = frames["fp_race_degradation_validation.csv"]
    transfer = frames["fp_degradation_transfer_validation.csv"]
    degradation_event = frames["fp_degradation_partial_update_event_metrics.csv"]
    degradation_loeo = frames["fp_degradation_partial_update_loeo.csv"]
    transfer_event = frames["fp_degradation_transfer_partial_update_event_metrics.csv"]
    transfer_loeo = frames["fp_degradation_transfer_partial_update_loeo.csv"]
    calibration = frames["fp_degradation_calibration_history.csv"]
    performance = frames["fp_performance_validation.csv"]
    corrected_runs = frames["fp_corrected_run_intercepts.csv"]
    corrected_performance = frames[
        "fp_corrected_same_team_performance_validation.csv"
    ]
    programme_observations = frames["fp_programme_offset_observations.csv"]
    programme_stability = frames["fp_programme_offset_stability.csv"]
    programme_variance = frames["fp_programme_offset_variance.csv"]
    programme_rolling = frames["fp_programme_offset_rolling_validation.csv"]
    topology = frames["fp_information_topology.csv"]
    joint = frames["fp_joint_fit_validation.csv"]
    def grouped(frame, columns):
        rows = []
        if frame.empty:
            return rows
        for key, group in frame.groupby(columns, dropna=False, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            rows.append({**dict(zip(columns, values)), **error_metrics(group)})
        return rows
    transfer_prior = transfer.loc[transfer["update_weight"].eq(0.0)].copy()
    transfer_full = transfer.loc[transfer["update_weight"].eq(1.0)].copy()
    degradation_endpoints = degradation.loc[
        degradation["update_weight"].isin((0.0, 1.0))
    ]
    latest_calibration = pd.DataFrame()
    if not calibration.empty:
        latest_round = calibration.groupby(
            ["session", "prior_family", "update_type"], sort=True
        )["trained_through_round"].transform("max")
        latest_calibration = calibration.loc[
            calibration["trained_through_round"].eq(latest_round)
        ]
    provenance_hashes = {}
    for raw in sorted(set(provenance)):
        path = Path(raw)
        if path.is_file():
            provenance_hashes[raw] = sha256(path.read_bytes()).hexdigest()
    paired_performance = []
    if not performance.empty:
        baseline = performance.loc[performance["strategy"].eq("Perf-0"), [
            "season", "round", "session", "prior_family", "compound",
            "prediction", "race_target",
        ]].rename(columns={"prediction": "baseline_prediction"})
        for fp_strategy in ("Perf-1", "Perf-2"):
            fp = performance.loc[performance["strategy"].eq(fp_strategy)].copy()
            paired = fp.merge(
                baseline, on=["season", "round", "session", "prior_family", "compound"],
                how="inner", suffixes=("", "_baseline"),
            )
            for family, group in paired.groupby("prior_family", sort=True):
                paired_performance.append({
                    "fp_strategy": fp_strategy, "prior_family": family,
                    "fp": error_metrics(group),
                    "pre_race": error_metrics(
                        group, prediction_column="baseline_prediction",
                        target_column="race_target",
                    ),
                })
    def family_gap(frame, group_columns):
        rows = []
        if frame.empty:
            return rows
        key_columns = [column for column in group_columns if column in frame]
        for key, group in frame.groupby(key_columns, dropna=False, sort=True):
            pivot = group.pivot_table(
                index=[
                    column for column in (
                        "season", "round", "session", "compound",
                        "held_out_compound", "hard_team", "comparison_team",
                        "target_type",
                    )
                    if column in group and group[column].notna().any()
                ],
                columns="prior_family", values="prediction", aggfunc="first",
            )
            if {"historical_baseline", "pirelli_informed"}.issubset(pivot.columns):
                gap = (pivot["historical_baseline"] - pivot["pirelli_informed"]).abs()
                values = key if isinstance(key, tuple) else (key,)
                rows.append({**dict(zip(key_columns, values)), "comparison_count": int(len(gap)),
                             "mean_absolute_family_gap": float(gap.mean())})
        return rows
    payload = {
        "schema_version": SCHEMA_VERSION, "algorithm_version": ALGORITHM_VERSION,
        "season": int(season), "candidate_rounds": candidates,
        "skipped": skipped, "degradation_metrics": grouped(degradation_endpoints, ["session", "prior_family", "strategy"]),
        "degradation_partial_update_metrics": grouped(
            degradation, ["session", "prior_family", "update_weight"]
        ),
        "degradation_partial_update_event_metrics": degradation_event.to_dict("records"),
        "degradation_partial_update_loeo": degradation_loeo.to_dict("records"),
        "k_transfer_metrics": grouped(transfer_full, ["session", "prior_family"]),
        "k_transfer_prior_metrics": grouped(transfer_prior, ["session", "prior_family"]),
        "k_transfer_partial_update_metrics": grouped(
            transfer, ["session", "prior_family", "update_weight"]
        ),
        "k_transfer_partial_update_event_metrics": transfer_event.to_dict("records"),
        "k_transfer_partial_update_loeo": transfer_loeo.to_dict("records"),
        "degradation_calibration": {
            "calibration_method": CALIBRATION_METHOD,
            "calibration_version": CALIBRATION_VERSION,
            "calibration_status": CALIBRATION_STATUS,
            "candidate_weights": list(UPDATE_WEIGHTS),
            "latest_trained_through_round": (
                int(calibration["trained_through_round"].max())
                if not calibration.empty else None
            ),
            "history_record_count": int(len(calibration)),
            "latest_history": latest_calibration.to_dict("records"),
            "production_weight_selected": False,
        },
        "performance_metrics": grouped(performance, ["session", "prior_family", "strategy"]),
        "performance_support_counts": {
            strategy: int(performance["strategy"].eq(strategy).sum()) if not performance.empty else 0
            for strategy in ("Perf-0", "Perf-1", "Perf-2")
        },
        "performance_paired_common_case_metrics": paired_performance,
        "corrected_intercept_audit": {
            "corrected_run_count": int(len(corrected_runs)),
            "same_team_bridge_count": int(len(corrected_performance)),
            "uncorrected_same_team": error_metrics(
                corrected_performance,
                prediction_column="uncorrected_prediction",
            ),
            "corrected_same_team": error_metrics(
                corrected_performance,
                prediction_column="corrected_prediction",
            ),
            "historical_pre_race": error_metrics(
                corrected_performance,
                prediction_column="historical_pre_race_prediction",
            ),
            "pirelli_pre_race": error_metrics(
                corrected_performance,
                prediction_column="pirelli_pre_race_prediction",
            ),
            "cross_team_evaluated": False,
        },
        "programme_offset_audit": {
            "physical_interpretation": (
                "The residual is an effective pace offset containing unknown starting load, engine mode, and programme conventions; it is not fuel kg."
            ),
            "observation_count": int(len(programme_observations)),
            "supported_stability_groups": int(
                programme_stability["sufficient_event_support"].astype(bool).sum()
            ) if not programme_stability.empty else 0,
            "stability": programme_stability.loc[
                programme_stability["sufficient_event_support"].astype(bool)
            ].to_dict("records") if not programme_stability.empty else [],
            "variance_decomposition": programme_variance.to_dict("records"),
            "rolling_bridge_count": int(len(programme_rolling)),
            "rolling_eligible_bridge_count": int(
                programme_rolling["eligible"].astype(bool).sum()
            ) if not programme_rolling.empty else 0,
            "rolling_programme_adjusted": error_metrics(
                programme_rolling.loc[
                    programme_rolling["eligible"].astype(bool)
                ] if not programme_rolling.empty else programme_rolling,
                prediction_column="programme_adjusted_prediction",
            ),
            "performance_gate_passed": False,
            "cross_team_evaluated": False,
        },
        "topology_frequency": (
            topology["topology_category"].value_counts().sort_index().astype(int).to_dict()
            if not topology.empty else {}
        ),
        "same_team_bridge_state_count": (
            int(topology["same_team_multi_compound_bridge"].astype(bool).sum())
            if not topology.empty else 0
        ),
        "degradation_metrics_by_topology": grouped(
            degradation_endpoints, ["session", "prior_family", "strategy", "topology_category"]
        ),
        "k_transfer_metrics_by_topology": grouped(
            transfer_full, ["session", "prior_family", "topology_category"]
        ),
        "joint_metrics": grouped(joint, ["session", "prior_family", "target_type"]),
        "joint_identifiable_states": int(joint.drop_duplicates(["round", "session", "prior_family"])["unregularised_identifiable"].sum()) if not joint.empty else 0,
        "joint_state_count": int(len(joint.drop_duplicates(["round", "session", "prior_family"]))) if not joint.empty else 0,
        "prior_family_gap": {
            "degradation": family_gap(
                degradation, ["session", "strategy", "update_weight"]
            ),
            "k_transfer": family_gap(transfer, ["session", "update_weight"]),
            "performance": family_gap(performance, ["session", "strategy"]),
            "joint": family_gap(joint, ["session", "target_type"]),
        },
        "provenance_sha256": provenance_hashes,
        "limitations": [
            "Each event/session requires its own persisted FP snapshot and canonical cached FP laps; partial weekends are retained.",
            "Cross-team correction uses the preceding completed race team baseline; the target race baseline is excluded as leakage.",
            "Corrected run intercepts use only strict laps exactly matched to the normal pre-race model coordinates; unmatched runs are unsupported.",
            "Fuel burn slope correction does not identify starting fuel load, engine mode, or long-run programme convention; programme residuals are effective pace offsets, not fuel kg.",
            "Programme-offset rolling validation uses only earlier rounds and requires at least two distinct training events; unsupported held-out bridges are retained explicitly.",
            "No leakage-safe persisted cumulative FP1+FP2 snapshot exists; cumulative scoring is therefore not invented.",
            "Joint-fit estimates are regularised diagnostics; design rank is reported separately and regularisation does not establish identifiability.",
            "No production tyre inputs or upstream estimators are modified.",
        ],
        "interpretation": {
            "fp2_bias": (
                "FP2 currently shows repeatable useful degradation information and a negative-bias tendency across multiple historical events, but the event sample is still too small to claim a universal FP2 bias."
            ),
            "calibration": (
                "Update weight is a versioned diagnostic calibration parameter, not a selected production constant."
            ),
        },
    }
    payload["input_fingerprint"] = canonical_hash({"algorithm": ALGORITHM_VERSION, "provenance": provenance_hashes})
    return payload
