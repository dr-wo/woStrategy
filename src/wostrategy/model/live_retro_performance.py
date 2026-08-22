from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Mapping

import numpy as np
import pandas as pd

from wodata.contracts import ModelSnapshot, ParameterEstimate, canonical_hash
from wostrategy.algorithm.monte_carlo_race_performance import (
    BASELINE_DRIVER,
    FUEL_PROXY_LAPS_REMAINING,
    WEIGHT_STRATEGY_BEST_RMSE_RELATIVE,
    MonteCarloRacePerformanceAlgorithm,
    MonteCarloRacePerformanceConfig,
)
from wostrategy.model.tyre_degragation import TYRE_AGE_LAPS_COLUMN
from wostrategy.model.correction_identifiability import diagnose_retro_fuel_track

from .pre_race_performance import FixedSampleBank
from .race_specific_performance import numerical_quality_status


@dataclass(frozen=True)
class LiveRetroModelResult:
    aggregate_snapshot: ModelSnapshot
    session_snapshots: Mapping[str, ModelSnapshot]
    sample_bank: FixedSampleBank
    candidate_costs: pd.DataFrame
    quality: Mapping[str, object]
    timings: Mapping[str, float]
    retro_result: object


def prepare_live_retro_laps(race_laps: pd.DataFrame, *, race_lap_count: int) -> pd.DataFrame:
    """Add only the columns normally supplied by the retro review preparation."""
    prepared = race_laps.copy()
    prepared["Compound"] = prepared["Compound"].astype(str).str.upper()
    prepared[FUEL_PROXY_LAPS_REMAINING] = (
        float(race_lap_count) - pd.to_numeric(prepared["LapNumber"], errors="coerce")
    )
    if "StintLapNumber" in prepared:
        prepared[TYRE_AGE_LAPS_COLUMN] = (
            pd.to_numeric(prepared["StintLapNumber"], errors="coerce") - 1.0
        )
    else:
        prepared[TYRE_AGE_LAPS_COLUMN] = pd.to_numeric(
            prepared["TyreLife"], errors="coerce"
        )
    return prepared


def retro_design_rank(prepared: pd.DataFrame, *, baseline_group: str) -> dict[str, object]:
    """Backward-compatible summary from the generic correction-model diagnostic."""
    fuel = prepared[FUEL_PROXY_LAPS_REMAINING].to_numpy(float)
    track = prepared["LapNumber"].to_numpy(float)
    diagnostic = diagnose_retro_fuel_track(prepared, baseline_group=baseline_group)
    correlation = diagnostic.sensitivity_correlation[0][1]
    return {
        "fuel_track_raw_rank": int(np.linalg.matrix_rank(np.column_stack((fuel, track)))),
        "fuel_track_profiled_rank": diagnostic.rank,
        "fuel_track_profiled_correlation": correlation,
        "fuel_track_alias": diagnostic.rank < 2,
        "correction_identifiability_parameter_names": diagnostic.parameter_names,
        "correction_identifiability_component_names": diagnostic.component_names,
        "correction_identifiability_singular_values": diagnostic.singular_values,
        "correction_identifiability_normalized_singular_values": (
            diagnostic.normalized_singular_values
        ),
        "correction_identifiability_condition_number": diagnostic.condition_number,
        "correction_identifiability_scope": diagnostic.scope,
        "fuel_track_identifiable_combination": "track_rate - fuel_rate",
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    return float(values[order[min(np.searchsorted(cumulative, probability), len(order) - 1)]])


def run_live_retro_model(
    race_laps: pd.DataFrame,
    config: MonteCarloRacePerformanceConfig,
    *,
    as_of_leader_lap: int,
    race_lap_count: int | None = None,
    created_at: datetime | None = None,
) -> LiveRetroModelResult:
    """Run the established retro algorithm unchanged on one live Race snapshot."""
    import time

    started = time.perf_counter()
    prepared = prepare_live_retro_laps(
        race_laps,
        race_lap_count=as_of_leader_lap if race_lap_count is None else race_lap_count,
    )
    prepared_at = time.perf_counter()
    algorithm = MonteCarloRacePerformanceAlgorithm(config)
    retro = algorithm.run(prepared)
    evaluated_at = time.perf_counter()

    raw_weights = retro.sample_parameters["Weight"].to_numpy(float)
    weight_sum = float(raw_weights.sum())
    weights = raw_weights / weight_sum if weight_sum > 0 else np.full(len(raw_weights), 1 / len(raw_weights))
    ess = float(1.0 / np.sum(weights**2))
    max_weight = float(weights.max())
    rmse = retro.sample_parameters["RMSESeconds"].to_numpy(float)
    weighted_rmse = float(np.sum(weights * rmse))
    compounds = tuple(sorted(prepared["Compound"].unique()))
    team_compounds = tuple(sorted(map(tuple, prepared[["Team", "Compound"]].drop_duplicates().to_numpy())))
    delta_compounds = tuple(c for c in compounds if c != config.compound_delta_reference.upper())
    dimension_names = (
        "fuel_rate", "track_rate",
        *(f"degradation:{compound}" for compound in compounds),
        *(f"compound_delta:{compound}" for compound in delta_compounds),
        *(f"team_degradation:{team}:{compound}" for team, compound in team_compounds),
    )
    degradation_wide = retro.compound_degradation.pivot(
        index="SampleId", columns="Compound", values="CompoundDegSecondsPerLap"
    )
    delta_wide = retro.compound_delta.loc[
        ~retro.compound_delta["Compound"].eq(config.compound_delta_reference.upper())
    ].pivot(index="SampleId", columns="Compound", values="CompoundDeltaSeconds")
    team_wide = retro.team_compound_degradation.pivot(
        index="SampleId", columns=["Team", "Compound"], values="VariationSecondsPerLap"
    )
    candidate_columns = [
        retro.sample_parameters["FuelRateSecondsPerLap"].to_numpy(float),
        retro.sample_parameters["TrackRateSecondsPerLap"].to_numpy(float),
        *(degradation_wide[compound].to_numpy(float) for compound in compounds),
        *(delta_wide[compound].to_numpy(float) for compound in delta_compounds),
        *(team_wide[(team, compound)].to_numpy(float) for team, compound in team_compounds),
    ]
    candidate_matrix = np.column_stack(candidate_columns)
    bounds = (
        config.fuel_rate_bounds,
        config.track_rate_bounds,
        *(config.compound_degradation_bounds.get(
            compound, config.default_compound_degradation_bounds
        ) for compound in compounds),
        *(config.compound_delta_bounds.get(
            compound, config.default_compound_delta_bounds
        ) for compound in delta_compounds),
        *((-1.0, 1.0) for _ in team_compounds),
    )
    bank_definition = {"architecture": "live-retro-v1", "config": asdict(config),
                       "dimensions": dimension_names, "leader_lap": as_of_leader_lap}
    bank_id = "live-retro-bank-" + canonical_hash(bank_definition)[:20]
    bank = FixedSampleBank(bank_id, tuple(dimension_names), tuple(bounds),
                           candidate_matrix, config.random_seed)
    fingerprint = canonical_hash(prepared[["Team", "Driver", "LapNumber", "LapTimeSeconds",
                                            "Compound", TYRE_AGE_LAPS_COLUMN,
                                            FUEL_PROXY_LAPS_REMAINING]].to_dict("records"))
    config_hash = canonical_hash(bank_definition)
    analysis_id = "live-retro-analysis-" + canonical_hash(
        {"input": fingerprint, "config": config_hash, "bank": bank_id}
    )[:20]
    created_at = created_at or datetime.now(timezone.utc)
    rows: list[ParameterEstimate] = []

    def add(parameter: str, values: np.ndarray, *, unit: str, compound=None, team=None,
            reference=None, lap_count=None, run_count=None):
        rows.append(ParameterEstimate(
            analysis_id, bank_id, "aggregate", as_of_leader_lap, parameter, "R",
            team, compound, reference,
            _weighted_quantile(values, weights, .1),
            _weighted_quantile(values, weights, .5),
            _weighted_quantile(values, weights, .9), unit, "measured",
            int(len(prepared) if lap_count is None else lap_count),
            int(prepared["RunId"].nunique() if run_count is None else run_count),
            ess, weighted_rmse, 0.0, 0.0, fingerprint, config_hash, created_at,
        ))

    add("fuel_rate", retro.sample_parameters["FuelRateSecondsPerLap"].to_numpy(float), unit="s/lap")
    add("track_rate", retro.sample_parameters["TrackRateSecondsPerLap"].to_numpy(float), unit="s/lap")
    for compound, frame in retro.compound_degradation.groupby("Compound", sort=True):
        support = prepared.loc[prepared["Compound"].eq(compound)]
        add("degradation", frame["CompoundDegSecondsPerLap"].to_numpy(float), unit="s/lap",
            compound=str(compound), lap_count=len(support), run_count=support["RunId"].nunique())
    for compound, frame in retro.compound_delta.groupby("Compound", sort=True):
        support = prepared.loc[prepared["Compound"].eq(compound)]
        add("compound_delta", frame["CompoundDeltaSeconds"].to_numpy(float), unit="s",
            compound=str(compound), reference=config.compound_delta_reference.upper(),
            lap_count=len(support), run_count=support["RunId"].nunique())
    for (team, compound), frame in retro.team_compound_degradation.groupby(
        ["Team", "Compound"], sort=True
    ):
        support = prepared.loc[prepared["Team"].eq(team) & prepared["Compound"].eq(compound)]
        add("team_degradation", frame["TeamCompoundDegSecondsPerLap"].to_numpy(float),
            unit="s/lap", team=str(team), compound=str(compound),
            lap_count=len(support), run_count=support["RunId"].nunique())
    baseline_group_columns = ["Team", "Driver"] if config.baseline_group == BASELINE_DRIVER else ["Team"]
    for key, frame in retro.baseline_pace.groupby(baseline_group_columns, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        team = str(key_values[0])
        # ParameterEstimate has no driver field; preserve a driver-mode key in
        # team rather than silently aggregating it.
        if len(key_values) > 1:
            team = f"{team}||{key_values[1]}"
        add("baseline_pace", frame["CorrectedBaselinePaceSeconds"].to_numpy(float),
            unit="s", team=team)
    aggregated_at = time.perf_counter()

    rank = retro_design_rank(prepared, baseline_group=config.baseline_group)
    quality = {
        "architecture": "live-retro-v1",
        "baseline_group": config.baseline_group,
        "independent_coordinate_count": len(dimension_names),
        "sample_count": config.sample_count,
        "ess": ess, "ess_fraction": ess / config.sample_count,
        "max_weight": max_weight, "top5_weight": float(np.sort(weights)[-5:].sum()),
        "clean_race_laps": int(len(prepared)),
        "clean_race_runs": int(prepared["RunId"].nunique()),
        "observed_compounds": compounds,
        "team_compound_coordinate_count": len(team_compounds),
        "weight_strategy": config.weight_strategy,
        "evaluator_backend": config.evaluator_backend,
        "candidate_chunk_size": config.candidate_chunk_size,
        **rank,
    }
    quality["numerical_quality_status"] = numerical_quality_status(
        quality["ess_fraction"], quality["max_weight"]
    )
    snapshot = ModelSnapshot(
        analysis_id, bank_id, "aggregate", as_of_leader_lap, tuple(rows), fingerprint,
        config_hash, created_at, contributing_sessions=("R",),
        joint_weighted_cost=float(np.sum(weights * rmse**2 / config.clean_lap_noise_sigma**2)),
        session_weighted_costs=(("R", float(np.sum(weights * rmse**2 /
                                                    config.clean_lap_noise_sigma**2))),),
        sampler_quality=tuple(quality.items()),
    )
    costs = pd.DataFrame({"SampleId": retro.sample_parameters["SampleId"],
                          "Weight": weights,
                          "TotalCost": rmse**2 / config.clean_lap_noise_sigma**2,
                          "RCost": rmse**2 / config.clean_lap_noise_sigma**2})
    timings = {"live_retro_preparation_seconds": prepared_at - started,
               "likelihood_evaluation_seconds": evaluated_at - prepared_at,
               "posterior_aggregation_seconds": aggregated_at - evaluated_at,
               **algorithm.last_timings}
    return LiveRetroModelResult(snapshot, {"R": snapshot}, bank, costs, quality, timings, retro)
