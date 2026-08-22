from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from wodata.contracts import ModelSnapshot, ParameterEstimate, canonical_hash

from ..algorithm.sampling import LatinHypercubeSampler, scale_unit_sample


@dataclass(frozen=True)
class WeekendModelConfig:
    sample_count: int = 5000
    random_seed: int = 2026
    fuel_rate_bounds: tuple[float, float] = (0.0, 0.0)
    track_rate_bounds: tuple[float, float] = (-0.05, 0.05)
    degradation_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    default_degradation_bounds: tuple[float, float] = (0.0, 0.20)
    compound_delta_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    default_compound_delta_bounds: tuple[float, float] = (-1.0, 1.0)
    reference_compound: str = "HARD"
    clean_lap_noise_sigma: float = 0.35
    session_noise_sigma: Mapping[str, float] = field(default_factory=dict)
    boundary_fraction: float = 0.05
    enforce_compound_order: bool = False


@dataclass(frozen=True)
class FixedSampleBank:
    sample_bank_id: str
    dimension_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    candidates: np.ndarray
    seed: int


@dataclass(frozen=True)
class JointWeekendModelResult:
    """Joint posterior plus session projections and candidate diagnostics."""

    aggregate_snapshot: ModelSnapshot
    session_snapshots: Mapping[str, ModelSnapshot]
    sample_bank: FixedSampleBank
    candidate_costs: pd.DataFrame
    contributing_sessions: tuple[str, ...]
    excluded_sessions: Mapping[str, str]


def derive_race_seed(year: int, round_number: int) -> int:
    """Derive a stable event-specific seed instead of using a season number.

    Year and round are both included because a round number alone identifies a
    different event/data set each season. The value remains stable for every
    update during one race weekend.
    """
    digest = canonical_hash({"year": int(year), "round": int(round_number)})
    return int(digest[:8], 16)


def generate_joint_sample_bank(
    expected_sessions: Sequence[str],
    compounds_by_session: Mapping[str, Sequence[str]],
    config: WeekendModelConfig,
) -> FixedSampleBank:
    """Create the full-weekend bank before every expected session is available."""
    sessions = tuple(dict.fromkeys(str(value).upper() for value in expected_sessions))
    if not sessions:
        raise ValueError("At least one expected FP session is required.")
    reference = config.reference_compound.upper()
    dimensions = ["fuel_rate"]
    bounds = [config.fuel_rate_bounds]
    for session in sessions:
        compounds = tuple(
            sorted({str(value).upper() for value in compounds_by_session.get(session, ())})
        )
        if not compounds:
            raise ValueError(f"No expected compounds configured for {session}.")
        dimensions.append(f"track_rate:{session}")
        bounds.append(config.track_rate_bounds)
        for compound in compounds:
            dimensions.append(f"degradation:{session}:{compound}")
            bounds.append(
                config.degradation_bounds.get(compound, config.default_degradation_bounds)
            )
        for compound in compounds:
            if compound == reference:
                continue
            dimensions.append(f"compound_delta:{session}:{compound}")
            bounds.append(
                config.compound_delta_bounds.get(
                    compound, config.default_compound_delta_bounds
                )
            )
    unit = LatinHypercubeSampler().sample(
        sample_count=config.sample_count,
        dimension_count=len(dimensions),
        seed=config.random_seed,
    )
    candidates = np.empty_like(unit)
    for index, parameter_bounds in enumerate(bounds):
        candidates[:, index] = [
            scale_unit_sample(value, parameter_bounds) for value in unit[:, index]
        ]
    if config.enforce_compound_order:
        _apply_ordered_joint_samples(
            unit, candidates, dimensions, bounds, sessions, reference
        )
    definition = {
        "dimensions": dimensions,
        "bounds": bounds,
        "sample_count": config.sample_count,
        "seed": config.random_seed,
        "enforce_compound_order": config.enforce_compound_order,
    }
    return FixedSampleBank(
        "bank-" + canonical_hash(definition)[:20],
        tuple(dimensions),
        tuple((float(low), float(high)) for low, high in bounds),
        candidates,
        config.random_seed,
    )


def _apply_ordered_joint_samples(
    unit: np.ndarray,
    candidates: np.ndarray,
    dimensions: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    sessions: Sequence[str],
    reference_compound: str,
) -> None:
    """Apply the race-review dry-compound ordering to a joint sample bank."""
    index_by_name = {name: index for index, name in enumerate(dimensions)}
    ordered = ("SOFT", "MEDIUM", "HARD")
    if reference_compound not in ordered:
        raise ValueError("Compound ordering requires a SOFT, MEDIUM or HARD reference.")
    reference_index = ordered.index(reference_compound)
    for session in sessions:
        previous = None
        for compound in reversed(ordered):
            index = index_by_name.get(f"degradation:{session}:{compound}")
            if index is None:
                continue
            low, high = bounds[index]
            higher_degradation_compounds = ordered[: ordered.index(compound)]
            high = min(
                [float(high)]
                + [
                    float(bounds[other_index][1])
                    for higher_compound in higher_degradation_compounds
                    if (
                        other_index := index_by_name.get(
                            f"degradation:{session}:{higher_compound}"
                        )
                    )
                    is not None
                ]
            )
            effective_low = np.full(len(unit), float(low))
            if previous is not None:
                effective_low = np.maximum(effective_low, previous)
            if np.any(effective_low > high):
                raise ValueError(
                    "Degradation bounds cannot satisfy SOFT >= MEDIUM >= HARD."
                )
            candidates[:, index] = effective_low + unit[:, index] * (
                high - effective_low
            )
            previous = candidates[:, index]

        previous = None
        for compound_index, compound in enumerate(ordered):
            if compound == reference_compound:
                previous = np.zeros(len(unit), dtype="float64")
                continue
            index = index_by_name.get(f"compound_delta:{session}:{compound}")
            if index is None:
                continue
            low, high = bounds[index]
            effective_low = np.full(len(unit), float(low))
            if previous is not None:
                effective_low = np.maximum(effective_low, previous)
            effective_high = float(high)
            if compound_index < reference_index:
                effective_high = min(effective_high, 0.0)
            elif compound_index > reference_index:
                effective_low = np.maximum(effective_low, 0.0)
            if np.any(effective_low > effective_high):
                raise ValueError(
                    "Compound delta bounds cannot satisfy SOFT <= MEDIUM <= HARD."
                )
            candidates[:, index] = effective_low + unit[:, index] * (
                effective_high - effective_low
            )
            previous = candidates[:, index]


def run_joint_weekend_model(
    session_laps: Mapping[str, pd.DataFrame],
    config: WeekendModelConfig | None = None,
    sample_bank: FixedSampleBank | None = None,
    *,
    expected_sessions: Sequence[str] = ("FP1", "FP2", "FP3"),
    expected_compounds: Sequence[str] = ("SOFT", "MEDIUM", "HARD"),
    excluded_session_reasons: Mapping[str, str] | None = None,
    as_of_leader_lap: int = 0,
    created_at: datetime | None = None,
) -> JointWeekendModelResult:
    """Evaluate one joint posterior with shared fuel and session-local effects."""
    config = config or WeekendModelConfig()
    sessions = tuple(dict.fromkeys(str(value).upper() for value in expected_sessions))
    supplied = {str(name).upper(): laps for name, laps in session_laps.items()}
    expected_compounds_by_session = {
        session: tuple(str(value).upper() for value in expected_compounds)
        for session in sessions
    }
    expected_bank = generate_joint_sample_bank(
        sessions, expected_compounds_by_session, config
    )
    bank = sample_bank or expected_bank
    if bank.dimension_names != expected_bank.dimension_names or bank.bounds != expected_bank.bounds:
        raise ValueError("The joint bank dimensions/bounds do not match this weekend model.")

    prepared_by_session: dict[str, pd.DataFrame] = {}
    excluded: dict[str, str] = {
        str(name).upper(): str(reason)
        for name, reason in (excluded_session_reasons or {}).items()
    }
    reference = config.reference_compound.upper()
    for session in sessions:
        laps = supplied.get(session)
        if laps is None or laps.empty:
            excluded.setdefault(session, "unavailable")
            continue
        try:
            prepared = _prepare_analysis_laps(laps)
            if prepared.empty:
                raise ValueError("no eligible runs remain")
            if prepared["tyre_age"].nunique() < 2:
                raise ValueError("tyre age never varies")
        except (TypeError, ValueError) as exc:
            excluded[session] = str(exc)
            continue
        prepared["session"] = session
        prepared["run_id"] = session + ":" + prepared["run_id"].astype(str)
        prepared_by_session[session] = prepared
    if not prepared_by_session:
        reasons = "; ".join(f"{name}: {reason}" for name, reason in excluded.items())
        raise ValueError(f"No valid FP sessions can contribute to the joint model ({reasons}).")

    candidate_values = [dict(zip(bank.dimension_names, row)) for row in bank.candidates]
    session_costs = {
        session: np.empty(len(bank.candidates), dtype="float64")
        for session in prepared_by_session
    }
    session_sse = {
        session: np.empty(len(bank.candidates), dtype="float64")
        for session in prepared_by_session
    }
    for sample_index, values in enumerate(candidate_values):
        for session, prepared in prepared_by_session.items():
            sse = _joint_session_sse(prepared, session, values, reference)
            sigma = float(
                config.session_noise_sigma.get(session, config.clean_lap_noise_sigma)
            )
            if sigma <= 0:
                raise ValueError(f"Noise sigma for {session} must be positive.")
            session_sse[session][sample_index] = sse
            session_costs[session][sample_index] = sse / (sigma**2)
    total_cost = np.sum(np.vstack(tuple(session_costs.values())), axis=0)
    log_weights = -0.5 * (total_cost - float(total_cost.min()))
    weights = np.exp(log_weights - float(log_weights.max()))
    weight_sum = float(weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Joint candidate weights are numerically invalid.")
    weights /= weight_sum
    ess = float(1.0 / np.sum(weights**2))
    contributing = tuple(prepared_by_session)
    input_fingerprints = {
        session: canonical_hash(
            prepared[
                ["run_id", "lap_number", "lap_time_s", "compound", "tyre_age", "track_index"]
            ].to_dict("records")
        )
        for session, prepared in prepared_by_session.items()
    }
    input_fingerprint = canonical_hash(input_fingerprints)
    config_hash = canonical_hash(
        {"config": asdict(config), "expected_sessions": sessions,
         "expected_compounds": tuple(expected_compounds)}
    )
    analysis_id = "analysis-" + canonical_hash(
        {"input": input_fingerprint, "config": config_hash, "bank": bank.sample_bank_id}
    )[:20]
    created_at = created_at or datetime.now(timezone.utc)
    weighted_total_cost = float(np.sum(weights * total_cost))
    weighted_session_costs = tuple(
        (session, float(np.sum(weights * costs)))
        for session, costs in session_costs.items()
    )

    estimates: list[ParameterEstimate] = []
    fuel_index = bank.dimension_names.index("fuel_rate")
    combined_prepared = pd.concat(tuple(prepared_by_session.values()), ignore_index=True)
    estimates.append(
        _estimate(
            bank.candidates[:, fuel_index], weights, bank.bounds[fuel_index],
            parameter="fuel_rate", compound=None, analysis_id=analysis_id, bank=bank,
            scope="aggregate", reference=reference, prepared=combined_prepared, ess=ess,
            weighted_rmse=float(np.sqrt(sum(np.sum(weights * values) for values in session_sse.values())
                                        / len(combined_prepared))),
            input_fingerprint=input_fingerprint, config_hash=config_hash,
            created_at=created_at, as_of_leader_lap=as_of_leader_lap,
            boundary_fraction=config.boundary_fraction,
            session_label="+".join(contributing),
        )
    )
    for session, prepared in prepared_by_session.items():
        session_rmse = float(
            np.sqrt(np.sum(weights * session_sse[session]) / max(len(prepared), 1))
        )
        for dimension_index, (name, bounds) in enumerate(
            zip(bank.dimension_names, bank.bounds)
        ):
            parts = name.split(":")
            if len(parts) < 2 or parts[1] != session:
                continue
            parameter = parts[0]
            compound = parts[2] if len(parts) == 3 else None
            estimates.append(
                _estimate(
                    bank.candidates[:, dimension_index], weights, bounds,
                    parameter=parameter, compound=compound, analysis_id=analysis_id,
                    bank=bank, scope="aggregate", reference=reference, prepared=prepared,
                    ess=ess, weighted_rmse=session_rmse,
                    input_fingerprint=input_fingerprints[session], config_hash=config_hash,
                    created_at=created_at, as_of_leader_lap=as_of_leader_lap,
                    boundary_fraction=config.boundary_fraction, session_label=session,
                )
            )
        estimates.append(
            _estimate(
                np.zeros(len(weights)), weights, (0.0, 0.0),
                parameter="compound_delta", compound=reference,
                analysis_id=analysis_id, bank=bank, scope="aggregate",
                reference=reference, prepared=prepared, ess=ess,
                weighted_rmse=session_rmse,
                input_fingerprint=input_fingerprints[session], config_hash=config_hash,
                created_at=created_at, as_of_leader_lap=as_of_leader_lap,
                boundary_fraction=config.boundary_fraction, session_label=session,
            )
        )
    aggregate = ModelSnapshot(
        analysis_id, bank.sample_bank_id, "aggregate", as_of_leader_lap,
        tuple(estimates), input_fingerprint, config_hash, created_at,
        contributing_sessions=contributing,
        excluded_sessions=tuple(excluded.items()),
        joint_weighted_cost=weighted_total_cost,
        session_weighted_costs=weighted_session_costs,
    )
    session_snapshots = {}
    shared = estimates[0]
    for session in contributing:
        rows = [shared] + [row for row in estimates[1:] if row.session == session]
        session_snapshots[session] = ModelSnapshot(
            analysis_id, bank.sample_bank_id, session, as_of_leader_lap,
            tuple(replace(row, source_scope=session) for row in rows),
            input_fingerprints[session], config_hash, created_at,
            contributing_sessions=contributing,
            excluded_sessions=tuple(excluded.items()),
            joint_weighted_cost=weighted_total_cost,
            session_weighted_costs=weighted_session_costs,
        )
    candidate_cost_rows = {"SampleId": np.arange(len(weights)), "Weight": weights,
                           "TotalCost": total_cost}
    for session, costs in session_costs.items():
        candidate_cost_rows[f"{session}Cost"] = costs
    candidate_costs = pd.DataFrame(candidate_cost_rows)
    return JointWeekendModelResult(
        aggregate, session_snapshots, bank, candidate_costs, contributing, excluded
    )


def generate_fixed_sample_bank(
    compounds: Sequence[str], config: WeekendModelConfig
) -> FixedSampleBank:
    compounds = tuple(sorted({str(value).upper() for value in compounds}))
    if not compounds:
        raise ValueError("Cannot generate a sample bank without compounds.")
    reference = config.reference_compound.upper()
    dimensions = ["fuel_rate", "track_rate"]
    bounds = [config.fuel_rate_bounds, config.track_rate_bounds]
    for compound in compounds:
        dimensions.append(f"degradation:{compound}")
        bounds.append(config.degradation_bounds.get(compound, config.default_degradation_bounds))
    for compound in compounds:
        if compound == reference:
            continue
        dimensions.append(f"compound_delta:{compound}")
        bounds.append(
            config.compound_delta_bounds.get(compound, config.default_compound_delta_bounds)
        )
    unit = LatinHypercubeSampler().sample(
        sample_count=config.sample_count,
        dimension_count=len(dimensions),
        seed=config.random_seed,
    )
    candidates = np.empty_like(unit)
    for index, parameter_bounds in enumerate(bounds):
        candidates[:, index] = [
            scale_unit_sample(value, parameter_bounds) for value in unit[:, index]
        ]
    definition = {
        "dimensions": dimensions,
        "bounds": bounds,
        "sample_count": config.sample_count,
        "seed": config.random_seed,
    }
    return FixedSampleBank(
        "bank-" + canonical_hash(definition)[:20],
        tuple(dimensions),
        tuple((float(low), float(high)) for low, high in bounds),
        candidates,
        config.random_seed,
    )


def run_weekend_model(
    session_laps: pd.DataFrame,
    config: WeekendModelConfig | None = None,
    sample_bank: FixedSampleBank | None = None,
    *,
    source_scope: str | None = None,
    as_of_leader_lap: int = 0,
    created_at: datetime | None = None,
) -> ModelSnapshot:
    """Evaluate a stable candidate bank against analysis-ready FP/Q laps.

    A nuisance intercept is re-fitted for every run and every candidate. Fuel is
    fixed by default because it is not independently identifiable in short FP runs.
    """
    config = config or WeekendModelConfig()
    prepared = _prepare_analysis_laps(session_laps)
    if prepared.empty:
        raise ValueError("No eligible runs remain for weekend modelling.")
    if prepared["tyre_age"].nunique() < 2:
        raise ValueError("Tyre age never varies in the eligible runs.")
    compounds = tuple(sorted(prepared["compound"].unique()))
    reference = config.reference_compound.upper()
    if reference not in compounds:
        raise ValueError(f"Reference compound {reference} is absent from eligible laps.")
    bank = sample_bank or generate_fixed_sample_bank(compounds, config)
    expected_bank = generate_fixed_sample_bank(compounds, config)
    if bank.dimension_names != expected_bank.dimension_names or bank.bounds != expected_bank.bounds:
        raise ValueError("The fixed sample bank dimensions/bounds do not match this model.")

    y = prepared["lap_time_s"].to_numpy(dtype="float64")
    run_index = prepared["run_lap_index"].to_numpy(dtype="float64")
    track_index = prepared["track_index"].to_numpy(dtype="float64")
    tyre_age = prepared["tyre_age"].to_numpy(dtype="float64")
    compound_values = prepared["compound"].to_numpy(dtype=str)
    run_codes, _ = pd.factorize(prepared["run_id"], sort=True)
    run_count = int(run_codes.max()) + 1
    rmse = np.empty(len(bank.candidates), dtype="float64")
    for sample_index, candidate in enumerate(bank.candidates):
        values = dict(zip(bank.dimension_names, candidate))
        effects = values["fuel_rate"] * run_index + values["track_rate"] * track_index
        effects = effects.copy()
        for compound in compounds:
            mask = compound_values == compound
            effects[mask] += values[f"degradation:{compound}"] * tyre_age[mask]
            if compound != reference:
                effects[mask] += values[f"compound_delta:{compound}"]
        baseline_residual = y - effects
        sums = np.bincount(run_codes, weights=baseline_residual, minlength=run_count)
        counts = np.bincount(run_codes, minlength=run_count)
        intercepts = sums / np.maximum(counts, 1)
        residual = baseline_residual - intercepts[run_codes]
        rmse[sample_index] = float(np.sqrt(np.mean(residual**2)))

    sigma = float(config.clean_lap_noise_sigma)
    if sigma <= 0:
        raise ValueError("clean_lap_noise_sigma must be positive.")
    best_rmse = float(rmse.min())
    log_weights = -0.5 * ((rmse - best_rmse) / sigma) ** 2
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    ess = float(1.0 / np.sum(weights**2))
    weighted_rmse = float(np.sum(weights * rmse))
    input_rows = prepared[
        ["run_id", "lap_number", "lap_time_s", "compound", "tyre_age", "track_index"]
    ].to_dict("records")
    input_fingerprint = canonical_hash(input_rows)
    config_hash = canonical_hash(asdict(config))
    analysis_id = "analysis-" + canonical_hash(
        {"input": input_fingerprint, "config": config_hash, "bank": bank.sample_bank_id}
    )[:20]
    created_at = created_at or datetime.now(timezone.utc)
    scope = source_scope or _source_scope(prepared)
    estimates = []
    for dimension_index, (name, bounds) in enumerate(zip(bank.dimension_names, bank.bounds)):
        parameter, _, compound = name.partition(":")
        estimates.append(
            _estimate(
                bank.candidates[:, dimension_index],
                weights,
                bounds,
                parameter=parameter,
                compound=compound or None,
                analysis_id=analysis_id,
                bank=bank,
                scope=scope,
                reference=reference,
                prepared=prepared,
                ess=ess,
                weighted_rmse=weighted_rmse,
                input_fingerprint=input_fingerprint,
                config_hash=config_hash,
                created_at=created_at,
                as_of_leader_lap=as_of_leader_lap,
                boundary_fraction=config.boundary_fraction,
            )
        )
    if reference not in [item.compound for item in estimates if item.parameter == "compound_delta"]:
        estimates.append(
            _estimate(
                np.zeros(len(weights)), weights, (0.0, 0.0), parameter="compound_delta",
                compound=reference, analysis_id=analysis_id, bank=bank, scope=scope,
                reference=reference, prepared=prepared, ess=ess, weighted_rmse=weighted_rmse,
                input_fingerprint=input_fingerprint, config_hash=config_hash,
                created_at=created_at, as_of_leader_lap=as_of_leader_lap,
                boundary_fraction=config.boundary_fraction,
            )
        )
    return ModelSnapshot(
        analysis_id,
        bank.sample_bank_id,
        scope,
        as_of_leader_lap,
        tuple(estimates),
        input_fingerprint,
        config_hash,
        created_at,
    )


def run_weekend_models(
    laps: pd.DataFrame,
    config: WeekendModelConfig | None = None,
    *,
    session_column: str = "SessionName",
) -> dict[str, ModelSnapshot]:
    if session_column not in laps.columns:
        raise ValueError(f"Missing session column: {session_column}")
    return {
        str(session): run_weekend_model(group, config, source_scope=str(session))
        for session, group in laps.groupby(session_column, sort=False)
    }


def aggregate_weekend_models(
    models: Iterable[ModelSnapshot], policy: str = "latest-valid"
) -> ModelSnapshot:
    models = tuple(models)
    if not models:
        raise ValueError("At least one session model is required.")
    if policy != "latest-valid":
        raise ValueError("The MVP supports only the latest-valid aggregation policy.")
    selected: dict[tuple[str, str | None, str | None], ParameterEstimate] = {}
    for model in models:
        for parameter in model.parameters:
            selected[(parameter.parameter, parameter.compound, parameter.team)] = parameter
    latest = models[-1]
    source_ids = tuple(model.analysis_id for model in models)
    config_hash = canonical_hash({"policy": policy, "sources": source_ids})
    analysis_id = "analysis-" + canonical_hash(config_hash)[:20]
    created_at = max(model.created_at for model in models)
    parameters = tuple(
        ParameterEstimate(
            analysis_id=analysis_id,
            sample_bank_id=row.sample_bank_id,
            source_scope="aggregate",
            as_of_leader_lap=max(model.as_of_leader_lap for model in models),
            parameter=row.parameter,
            session=row.session,
            team=row.team,
            compound=row.compound,
            reference_compound=row.reference_compound,
            p10=row.p10,
            median=row.median,
            p90=row.p90,
            unit=row.unit,
            support_status=row.support_status,
            usable_lap_count=row.usable_lap_count,
            usable_run_count=row.usable_run_count,
            ess=row.ess,
            weighted_rmse=row.weighted_rmse,
            bound_weight_low=row.bound_weight_low,
            bound_weight_high=row.bound_weight_high,
            input_fingerprint=row.input_fingerprint,
            config_hash=config_hash,
            created_at=created_at,
        )
        for row in selected.values()
    )
    return ModelSnapshot(
        analysis_id,
        latest.sample_bank_id,
        "aggregate",
        max(model.as_of_leader_lap for model in models),
        parameters,
        canonical_hash([model.input_fingerprint for model in models]),
        config_hash,
        created_at,
        source_analysis_ids=source_ids,
    )


def _prepare_analysis_laps(laps: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "lap_time_s": ("lap_time_s", "LapTimeSeconds", "LapTime"),
        "compound": ("compound", "Compound"),
        "tyre_age": ("tyre_age", "TyreLife"),
        "lap_number": ("lap_number", "LapNumber"),
    }
    output = pd.DataFrame(index=laps.index)
    for target, candidates in aliases.items():
        source = next((name for name in candidates if name in laps.columns), None)
        if source is None:
            raise ValueError(f"Laps are missing a {target} column.")
        output[target] = laps[source]
    if pd.api.types.is_timedelta64_dtype(output["lap_time_s"]):
        output["lap_time_s"] = output["lap_time_s"].dt.total_seconds()
    else:
        output["lap_time_s"] = pd.to_numeric(output["lap_time_s"], errors="coerce")
    output["tyre_age"] = pd.to_numeric(output["tyre_age"], errors="coerce")
    output["lap_number"] = pd.to_numeric(output["lap_number"], errors="coerce")
    output["compound"] = output["compound"].astype("string").str.upper()
    run_source = next((name for name in ("run_id", "RunId") if name in laps.columns), None)
    if run_source:
        output["run_id"] = laps[run_source].astype(str)
    else:
        parts = []
        for name in ("SessionName", "Driver", "Stint"):
            if name in laps.columns:
                parts.append(laps[name].astype(str))
        if not parts:
            output["run_id"] = "run-0"
        else:
            run_id = parts[0]
            for part in parts[1:]:
                run_id = run_id + ":" + part
            output["run_id"] = run_id
    output = output.dropna(subset=["lap_time_s", "compound", "tyre_age", "lap_number"])
    output = output.loc[output["compound"].isin(("SOFT", "MEDIUM", "HARD"))].copy()
    output = output.sort_values(["run_id", "lap_number"], kind="mergesort")
    output["run_lap_index"] = output.groupby("run_id").cumcount() + 1
    track_source = next(
        (name for name in ("track_index", "SessionTime", "Time") if name in laps.columns), None
    )
    if track_source == "track_index":
        output["track_index"] = pd.to_numeric(laps.loc[output.index, track_source], errors="coerce")
    elif track_source:
        values = pd.to_timedelta(laps.loc[output.index, track_source], errors="coerce").dt.total_seconds()
        output["track_index"] = values.rank(method="dense") - 1
    else:
        output["track_index"] = output["lap_number"] - output["lap_number"].min()
    output["track_index"] = output["track_index"].fillna(0.0)
    if "SessionName" in laps.columns:
        output["session"] = laps.loc[output.index, "SessionName"].astype(str)
    else:
        output["session"] = "unknown"
    # Race-specific likelihoods need the persistent driver identity in addition
    # to the run/stint identity.  Keeping it here is backwards-compatible with
    # the pre-race run-profile likelihood and avoids attempting to recover a
    # driver from an opaque RunId downstream.
    if "Driver" in laps.columns:
        output["driver"] = laps.loc[output.index, "Driver"].astype(str)
    else:
        output["driver"] = output["run_id"]
    if "Team" in laps.columns:
        output["team"] = laps.loc[output.index, "Team"].astype(str)
    return output.reset_index(drop=True)


def _estimate(
    values, weights, bounds, *, parameter, compound, analysis_id, bank, scope, reference,
    prepared, ess, weighted_rmse, input_fingerprint, config_hash, created_at,
    as_of_leader_lap, boundary_fraction, session_label=None,
) -> ParameterEstimate:
    low, high = bounds
    span = high - low
    if span == 0:
        near_low = near_high = 1.0
    else:
        near_low = float(weights[values <= low + span * boundary_fraction].sum())
        near_high = float(weights[values >= high - span * boundary_fraction].sum())
    return ParameterEstimate(
        analysis_id, bank.sample_bank_id, scope, as_of_leader_lap, parameter,
        str(session_label or prepared["session"].iloc[-1]), None, compound, reference,
        _weighted_quantile(values, weights, 0.10),
        _weighted_quantile(values, weights, 0.50),
        _weighted_quantile(values, weights, 0.90),
        "s/lap" if parameter in {"fuel_rate", "track_rate", "degradation"} else "s",
        "measured", int(len(prepared)), int(prepared["run_id"].nunique()), ess,
        weighted_rmse, near_low, near_high, input_fingerprint, config_hash, created_at,
    )


def _joint_session_sse(
    prepared: pd.DataFrame,
    session: str,
    values: Mapping[str, float],
    reference_compound: str,
) -> float:
    """Refit each run intercept and return this session's residual SSE."""
    y = prepared["lap_time_s"].to_numpy(dtype="float64")
    run_index = prepared["run_lap_index"].to_numpy(dtype="float64")
    track_index = prepared["track_index"].to_numpy(dtype="float64")
    tyre_age = prepared["tyre_age"].to_numpy(dtype="float64")
    compounds = prepared["compound"].to_numpy(dtype=str)
    effects = (
        float(values["fuel_rate"]) * run_index
        + float(values[f"track_rate:{session}"]) * track_index
    )
    effects = effects.copy()
    for compound in np.unique(compounds):
        mask = compounds == compound
        effects[mask] += float(values[f"degradation:{session}:{compound}"]) * tyre_age[mask]
        if compound != reference_compound:
            effects[mask] += float(values[f"compound_delta:{session}:{compound}"])
    run_codes, _ = pd.factorize(prepared["run_id"], sort=True)
    run_count = int(run_codes.max()) + 1
    baseline_residual = y - effects
    sums = np.bincount(run_codes, weights=baseline_residual, minlength=run_count)
    counts = np.bincount(run_codes, minlength=run_count)
    intercepts = sums / np.maximum(counts, 1)
    residual = baseline_residual - intercepts[run_codes]
    return float(np.sum(residual**2))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, quantile, side="left")), len(order) - 1)
    return float(values[order[index]])


def _source_scope(prepared: pd.DataFrame) -> str:
    sessions = prepared["session"].dropna().unique()
    return str(sessions[0]) if len(sessions) == 1 else "session"
