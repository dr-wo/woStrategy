from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

from wodata.contracts import ModelSnapshot, ParameterEstimate, canonical_hash

from .pre_race_performance import FixedSampleBank, WeekendModelConfig


COMPOUNDS = ("HARD", "MEDIUM", "SOFT")
BASELINE_PER_RUN = "per_run"
BASELINE_PER_DRIVER = "per_driver"
BASELINE_GLOBAL_DRIVER = "global_driver"
BASELINE_ARCHITECTURES = (
    BASELINE_PER_RUN,
    BASELINE_PER_DRIVER,
    BASELINE_GLOBAL_DRIVER,
)
NUMERICAL_QUALITY_THRESHOLDS = {
    "warning_ess_fraction": 0.01,
    "degenerate_ess_fraction": 0.001,
    "warning_max_weight": 0.05,
    "degenerate_max_weight": 0.20,
}


def numerical_quality_status(ess_fraction: float, max_weight: float) -> str:
    """Sampler-only warning state; deliberately unrelated to inference confidence."""
    if (ess_fraction < NUMERICAL_QUALITY_THRESHOLDS["degenerate_ess_fraction"]
            or max_weight > NUMERICAL_QUALITY_THRESHOLDS["degenerate_max_weight"]):
        return "degenerate"
    if (ess_fraction < NUMERICAL_QUALITY_THRESHOLDS["warning_ess_fraction"]
            or max_weight > NUMERICAL_QUALITY_THRESHOLDS["warning_max_weight"]):
        return "warning"
    return "adequate"


@dataclass(frozen=True)
class MarginalPrior:
    p10: float
    median: float
    p90: float
    unit: str
    approximation: str = "split_normal_from_marginal_p10_median_p90"


@dataclass(frozen=True)
class FrozenPreRacePrior:
    schema_version: int
    year: int
    round_number: int
    source_analysis_id: str
    source_sessions: tuple[str, ...]
    source_created_at: str
    frozen_at: str
    construction: str
    fuel: MarginalPrior
    degradation: Mapping[str, MarginalPrior]
    performance: Mapping[str, MarginalPrior]


@dataclass(frozen=True)
class RaceSpecificModelResult:
    aggregate_snapshot: ModelSnapshot
    session_snapshots: Mapping[str, ModelSnapshot]
    sample_bank: FixedSampleBank
    candidate_costs: pd.DataFrame
    quality: Mapping[str, object]
    timings: Mapping[str, float]


def _split_normal(unit: np.ndarray, prior: MarginalPrior) -> np.ndarray:
    z = norm.ppf(np.clip(unit, 1e-12, 1 - 1e-12))
    scale_low = max((prior.median - prior.p10) / abs(norm.ppf(.1)), 1e-9)
    scale_high = max((prior.p90 - prior.median) / norm.ppf(.9), 1e-9)
    return prior.median + np.where(z < 0, scale_low, scale_high) * z


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, probability, side="left")]])


def generate_race_specific_bank(
    prior: FrozenPreRacePrior,
    config: WeekendModelConfig,
) -> tuple[FixedSampleBank, dict[str, np.ndarray]]:
    """Generate six Race-state coordinates plus an auxiliary frozen-prior fuel draw."""
    dimensions = (
        "net_lap_slope",
        "degradation:R:HARD",
        "degradation:R:MEDIUM",
        "degradation:R:SOFT",
        "performance_contrast:HARD_MINUS_MEDIUM",
        "performance_contrast:SOFT_MINUS_MEDIUM",
    )
    # These bounds are support metadata. Transfer-prior coordinates are sampled
    # from the frozen marginal approximation and clipped to physical support.
    bounds = (
        (config.fuel_rate_bounds[0] + config.track_rate_bounds[0],
         config.fuel_rate_bounds[1] + config.track_rate_bounds[1]),
        config.degradation_bounds.get("HARD", config.default_degradation_bounds),
        config.degradation_bounds.get("MEDIUM", config.default_degradation_bounds),
        config.degradation_bounds.get("SOFT", config.default_degradation_bounds),
        config.default_compound_delta_bounds,
        config.default_compound_delta_bounds,
    )
    unit = qmc.LatinHypercube(d=8, seed=config.random_seed).random(config.sample_count)
    fuel = np.clip(
        _split_normal(unit[:, 0], prior.fuel),
        config.fuel_rate_bounds[0], config.fuel_rate_bounds[1],
    )
    track = config.track_rate_bounds[0] + unit[:, 1] * (
        config.track_rate_bounds[1] - config.track_rate_bounds[0]
    )
    candidates = np.empty((config.sample_count, len(dimensions)), dtype=float)
    candidates[:, 0] = fuel + track
    for offset, compound in enumerate(COMPOUNDS, start=1):
        low, high = bounds[offset]
        candidates[:, offset] = np.clip(
            _split_normal(unit[:, offset + 1], prior.degradation[compound]), low, high
        )
    # Preserve the physical degradation order. This is the only legitimate way
    # an unseen compound can move from its marginal transfer prior.
    candidates[:, 1:4] = np.sort(candidates[:, 1:4], axis=1)
    hard = _split_normal(unit[:, 5], prior.performance["HARD"])
    medium = _split_normal(unit[:, 6], prior.performance["MEDIUM"])
    soft = _split_normal(unit[:, 7], prior.performance["SOFT"])
    candidates[:, 4] = hard - medium
    candidates[:, 5] = soft - medium
    definition = {
        "architecture": "race-specific-v1",
        "dimensions": dimensions,
        "bounds": bounds,
        "sample_count": config.sample_count,
        "seed": config.random_seed,
        "prior_analysis_id": prior.source_analysis_id,
    }
    bank = FixedSampleBank(
        "race-bank-" + canonical_hash(definition)[:20], dimensions,
        tuple((float(a), float(b)) for a, b in bounds), candidates, config.random_seed,
    )
    # Sum-to-zero physical effects make the output reference invariant.
    contrasts = np.column_stack((candidates[:, 4], np.zeros(config.sample_count), candidates[:, 5]))
    effects = contrasts - contrasts.mean(axis=1, keepdims=True)
    auxiliary = {
        "fuel_rate": fuel,
        "track_rate:R": candidates[:, 0] - fuel,
        **{f"compound_delta:R:{compound}": effects[:, index]
           for index, compound in enumerate(COMPOUNDS)},
    }
    return bank, auxiliary


def compound_evidence(prepared: pd.DataFrame, compound: str) -> dict[str, object]:
    rows = prepared.loc[prepared["compound"].astype(str).str.upper() == compound]
    lap_count = int(len(rows))
    run_count = int(rows["run_id"].nunique()) if lap_count else 0
    age_span = float(rows["tyre_age"].max() - rows["tyre_age"].min()) if lap_count else 0.0
    # Diagnostic/configurable initial rule: direct slope evidence needs variation;
    # two runs, six laps and three tyre-age units are considered Race-informed.
    if lap_count == 0 or age_span <= 0:
        status = "prior_only"
    elif lap_count >= 6 and run_count >= 2 and age_span >= 3:
        status = "race_informed"
    else:
        status = "weak_race_evidence"
    return {"status": status, "clean_laps": lap_count, "clean_runs": run_count,
            "tyre_age_span": age_span}


def compound_connectivity(prepared: pd.DataFrame) -> dict[str, object]:
    """Describe the driver-compound observation graph used by performance contrasts."""
    usage = (
        prepared[["driver", "compound"]]
        .drop_duplicates()
        .assign(compound=lambda frame: frame["compound"].astype(str).str.upper())
    )
    observed = tuple(sorted(set(usage["compound"]) & set(COMPOUNDS)))
    adjacency = {compound: set() for compound in COMPOUNDS}
    driver_compound_counts: dict[str, int] = {}
    for driver, rows in usage.groupby("driver", sort=True):
        used = tuple(sorted(set(rows["compound"]) & set(COMPOUNDS)))
        driver_compound_counts[str(driver)] = len(used)
        for left in used:
            adjacency[left].update(compound for compound in used if compound != left)
    components: list[tuple[str, ...]] = []
    unseen = set(observed)
    while unseen:
        stack = [unseen.pop()]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            component.add(node)
            neighbours = adjacency[node] & unseen
            unseen -= neighbours
            stack.extend(neighbours)
        components.append(tuple(sorted(component)))
    components.sort()
    count_distribution = {
        count: sum(value == count for value in driver_compound_counts.values())
        for count in (1, 2, 3)
    }
    return {
        "observed_compounds": observed,
        "components": tuple(components),
        "fully_connected": len(observed) >= 2 and len(components) == 1,
        "drivers": int(len(driver_compound_counts)),
        "drivers_by_compound_count": count_distribution,
        "connector_drivers": int(sum(value >= 2 for value in driver_compound_counts.values())),
    }


def _candidate_sse(
    prepared: pd.DataFrame,
    bank: FixedSampleBank,
    auxiliary: Mapping[str, np.ndarray],
    baseline_architecture: str,
) -> np.ndarray:
    """Profile nuisance baselines and return candidate SSEs.

    ``global_driver`` and ``per_driver`` span the same fitted subspace: one
    global intercept plus mean-zero driver deviations has exactly as many
    degrees of freedom as one intercept per driver.  Keeping both switch names
    makes that equivalence explicit in architecture diagnostics.
    """
    y = prepared["lap_time_s"].to_numpy(float)
    tyre_age = prepared["tyre_age"].to_numpy(float)
    compounds = prepared["compound"].astype(str).str.upper().to_numpy()
    if baseline_architecture == BASELINE_PER_RUN:
        trend_index = prepared["run_lap_index"].to_numpy(float)
        group = prepared["run_id"]
    elif baseline_architecture in (BASELINE_PER_DRIVER, BASELINE_GLOBAL_DRIVER):
        trend_index = prepared["track_index"].to_numpy(float)
        group = prepared["driver"]
    else:
        options = ", ".join(BASELINE_ARCHITECTURES)
        raise ValueError(f"Unknown Race baseline architecture {baseline_architecture!r}; use {options}.")

    effects = bank.candidates[:, 0, None] * trend_index[None, :]
    for index, compound in enumerate(COMPOUNDS, start=1):
        mask = compounds == compound
        effects[:, mask] += bank.candidates[:, index, None] * tyre_age[mask][None, :]
        if baseline_architecture != BASELINE_PER_RUN:
            effects[:, mask] += auxiliary[f"compound_delta:R:{compound}"][:, None]
    codes, _ = pd.factorize(group, sort=True)
    residual = y[None, :] - effects
    for code in range(int(codes.max()) + 1):
        mask = codes == code
        residual[:, mask] -= residual[:, mask].mean(axis=1, keepdims=True)
    return np.sum(residual**2, axis=1)


def run_race_specific_model(
    race_laps: pd.DataFrame,
    prior: FrozenPreRacePrior,
    config: WeekendModelConfig,
    *,
    as_of_leader_lap: int,
    created_at: datetime | None = None,
    baseline_architecture: Literal["per_run", "per_driver", "global_driver"] = BASELINE_PER_RUN,
) -> RaceSpecificModelResult:
    import time
    from .pre_race_performance import _prepare_analysis_laps

    started = time.perf_counter()
    bank, auxiliary = generate_race_specific_bank(prior, config)
    generated = time.perf_counter()
    prepared = _prepare_analysis_laps(race_laps)
    sse = _candidate_sse(prepared, bank, auxiliary, baseline_architecture)
    cost = sse / config.clean_lap_noise_sigma**2
    log_weights = -.5 * (cost - cost.min())
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    evaluated = time.perf_counter()
    ess = float(1 / np.sum(weights**2))
    max_weight = float(weights.max())
    evidence = {compound: compound_evidence(prepared, compound) for compound in COMPOUNDS}
    connectivity = compound_connectivity(prepared)
    fingerprint = canonical_hash(prepared[["run_id", "lap_number", "lap_time_s", "compound",
                                            "tyre_age", "track_index"]].to_dict("records"))
    config_hash = canonical_hash({"architecture": "race-specific-v2",
                                  "baseline_architecture": baseline_architecture,
                                  "config": asdict(config), "prior": prior.source_analysis_id})
    analysis_id = "race-analysis-" + canonical_hash(
        {"input": fingerprint, "config": config_hash, "bank": bank.sample_bank_id}
    )[:20]
    created_at = created_at or datetime.now(timezone.utc)
    rows: list[ParameterEstimate] = []

    def add(parameter, values, unit, compound=None, status="measured", support=None):
        support = support or {"clean_laps": len(prepared), "clean_runs": prepared.run_id.nunique()}
        rows.append(ParameterEstimate(
            analysis_id, bank.sample_bank_id, "aggregate", as_of_leader_lap,
            parameter, "R", None, compound, "SUM_ZERO",
            _weighted_quantile(values, weights, .1), _weighted_quantile(values, weights, .5),
            _weighted_quantile(values, weights, .9), unit, status,
            int(support["clean_laps"]), int(support["clean_runs"]), ess,
            float(np.sqrt(np.sum(weights * sse) / max(len(prepared), 1))), 0.0, 0.0,
            fingerprint, config_hash, created_at,
        ))

    add("net_lap_slope", bank.candidates[:, 0], "s/lap")
    add("fuel_rate", auxiliary["fuel_rate"], "s/lap", status="prior_only")
    add("track_rate", auxiliary["track_rate:R"], "s/lap")
    for index, compound in enumerate(COMPOUNDS, start=1):
        item = evidence[compound]
        add("degradation", bank.candidates[:, index], "s/lap", compound,
            str(item["status"]), item)
    for compound in COMPOUNDS:
        observed = compound in connectivity["observed_compounds"]
        connected = any(
            compound in component and len(component) >= 2
            for component in connectivity["components"]
        )
        performance_status = (
            "prior_only" if not observed else
            "race_informed" if connectivity["fully_connected"] else
            "weak_race_evidence" if connected else "prior_only"
        )
        if baseline_architecture == BASELINE_PER_RUN:
            performance_status = "prior_only"
        add("compound_delta", auxiliary[f"compound_delta:R:{compound}"], "s", compound,
            performance_status, evidence[compound])
    aggregated = time.perf_counter()
    quality = {
        "architecture": "race-specific-v2", "baseline_architecture": baseline_architecture,
        "physical_quantity_count": 8, "independent_coordinate_count": 6,
        "active_likelihood_dimension_count": 4 if baseline_architecture == BASELINE_PER_RUN else 6,
        "sample_count": config.sample_count, "ess": ess,
        "ess_fraction": ess / config.sample_count, "max_weight": max_weight,
        "top5_weight": float(np.sort(weights)[-5:].sum()),
        "clean_race_laps": int(len(prepared)), "clean_race_runs": int(prepared.run_id.nunique()),
        "compound_evidence": evidence, "compound_connectivity": connectivity,
        "profiled_nuisance_baseline_count": int(
            prepared.run_id.nunique() if baseline_architecture == BASELINE_PER_RUN
            else prepared.driver.nunique()
        ),
        "prior_analysis_id": prior.source_analysis_id,
    }
    quality["numerical_quality_status"] = numerical_quality_status(
        quality["ess_fraction"], quality["max_weight"]
    )
    quality["numerical_quality_thresholds"] = NUMERICAL_QUALITY_THRESHOLDS
    snapshot = ModelSnapshot(
        analysis_id, bank.sample_bank_id, "aggregate", as_of_leader_lap, tuple(rows),
        fingerprint, config_hash, created_at, contributing_sessions=("R",),
        joint_weighted_cost=float(np.sum(weights * cost)),
        session_weighted_costs=(("R", float(np.sum(weights * cost))),),
        sampler_quality=tuple(quality.items()),
    )
    timings = {"candidate_generation_seconds": generated - started,
               "likelihood_evaluation_seconds": evaluated - generated,
               "posterior_aggregation_seconds": aggregated - evaluated}
    costs = pd.DataFrame({"SampleId": np.arange(config.sample_count), "Weight": weights,
                          "TotalCost": cost, "RCost": cost})
    return RaceSpecificModelResult(snapshot, {"R": snapshot}, bank, costs, quality, timings)
