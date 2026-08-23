from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

from wodata.contracts import (
    ModelSnapshot,
    StrategyRecord,
    StrategySearchState,
    canonical_hash,
    utc_now,
)


@dataclass(frozen=True)
class StrategyModel:
    compound_delta: Mapping[str, float]
    degradation: Mapping[str, float]
    analysis_id: str = "manual"
    version: str = "manual"
    fitted_start_age: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        compounds = set(self.compound_delta) | set(self.degradation)
        if not compounds:
            raise ValueError("At least one compound is required.")
        missing_delta = compounds.difference(self.compound_delta)
        missing_deg = compounds.difference(self.degradation)
        if missing_delta or missing_deg:
            raise ValueError("Every compound needs both a delta and degradation value.")

    @classmethod
    def from_snapshot(cls, snapshot: ModelSnapshot) -> "StrategyModel":
        deltas: dict[str, float] = {}
        degradation: dict[str, float] = {}
        for row in snapshot.parameters:
            if not row.compound:
                continue
            compound = row.compound.upper()
            if row.parameter == "compound_delta":
                deltas[compound] = float(row.median)
            elif row.parameter == "degradation":
                degradation[compound] = float(row.median)
        for compound in degradation:
            deltas.setdefault(compound, 0.0)
        return cls(deltas, degradation, snapshot.analysis_id, snapshot.config_hash)


@dataclass(frozen=True)
class StrategyPlan:
    compounds: tuple[str, ...]
    pit_laps: tuple[int, ...]

    def __post_init__(self) -> None:
        normalized = tuple(str(value).upper() for value in self.compounds)
        object.__setattr__(self, "compounds", normalized)
        if not normalized:
            raise ValueError("A strategy requires at least one compound.")
        if len(self.pit_laps) != len(normalized) - 1:
            raise ValueError("pit_laps must contain one entry per compound change/set fit.")
        if tuple(sorted(self.pit_laps)) != self.pit_laps or len(set(self.pit_laps)) != len(
            self.pit_laps
        ):
            raise ValueError("pit_laps must be unique and strictly increasing.")


@dataclass(frozen=True)
class StrategyResult:
    rank: int
    plan: StrategyPlan
    remaining_cost: float
    delta_to_best: float
    search_type: str
    formatted_strategy: str


@dataclass(frozen=True)
class StrategyRules:
    """General sporting/availability constraints for deterministic search."""

    minimum_distinct_dry_compounds: int = 0
    available_compounds: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.minimum_distinct_dry_compounds < 0:
            raise ValueError("minimum_distinct_dry_compounds cannot be negative")
        if self.available_compounds is not None:
            object.__setattr__(
                self, "available_compounds",
                tuple(dict.fromkeys(str(value).upper() for value in self.available_compounds)),
            )

    def accepts(self, plan: StrategyPlan) -> bool:
        if self.available_compounds is not None and not set(plan.compounds).issubset(self.available_compounds):
            return False
        dry = {value for value in plan.compounds if value in {"SOFT", "MEDIUM", "HARD"}}
        return len(dry) >= self.minimum_distinct_dry_compounds


def stint_cost(n: int, compound: str, current_tyre_age: int, model: StrategyModel) -> float:
    """Cost n future laps when current age is after the last completed lap.

    The first modelled lap is therefore costed at ``current_tyre_age + 1``.
    """
    if n < 0 or current_tyre_age < 0:
        raise ValueError("Lap count and tyre age cannot be negative.")
    compound = compound.upper()
    delta = float(model.compound_delta[compound])
    degradation = float(model.degradation[compound])
    return n * delta + degradation * (n * current_tyre_age + n * (n + 1) / 2.0)


def format_strategy(plan: StrategyPlan) -> str:
    initials = [_compound_initial(value) for value in plan.compounds]
    text = initials[0]
    for index, (pit_lap, compound) in enumerate(zip(plan.pit_laps, initials[1:])):
        if index == 0:
            text += f"-({pit_lap}){compound}"
        else:
            text += f"-{pit_lap}({compound})"
    return text


def revalue_fixed_strategy(
    strategy: StrategyPlan,
    state: StrategySearchState,
    model: StrategyModel,
    pit_loss: float,
) -> StrategyResult:
    _validate_state(state, model)
    _validate_plan_horizon(strategy, state)
    cost = 0.0
    stint_start_lap = state.current_completed_lap + 1
    age = state.current_tyre_age
    for index, compound in enumerate(strategy.compounds):
        stint_end = strategy.pit_laps[index] if index < len(strategy.pit_laps) else state.total_laps
        lap_count = stint_end - stint_start_lap + 1
        cost += stint_cost(lap_count, compound, age, model)
        if index < len(strategy.pit_laps):
            cost += float(pit_loss)
            stint_start_lap = stint_end + 1
            age = _start_age(model, strategy.compounds[index + 1])
    return _result(strategy, cost, "pre_race_fixed")


def optimise_same_sequence(
    strategy: StrategyPlan,
    state: StrategySearchState,
    model: StrategyModel,
    pit_loss: float,
    k: int = 10,
) -> list[StrategyResult]:
    _validate_state(state, model)
    compounds = strategy.compounds
    if compounds[0] != state.current_compound.upper():
        raise ValueError("The sequence must start on the state's current compound.")
    if state.laps_remaining < len(compounds):
        return []

    @lru_cache(maxsize=None)
    def solve(index: int, next_lap: int, age: int) -> tuple[tuple[float, tuple[int, ...]], ...]:
        remaining = state.total_laps - next_lap + 1
        remaining_stints = len(compounds) - index
        compound = compounds[index]
        if remaining_stints == 1:
            return ((stint_cost(remaining, compound, age, model), ()),)
        candidates: list[tuple[float, tuple[int, ...]]] = []
        max_length = remaining - (remaining_stints - 1)
        for length in range(1, max_length + 1):
            pit_lap = next_lap + length - 1
            next_compound = compounds[index + 1]
            for future_cost, future_pits in solve(
                index + 1, pit_lap + 1, _start_age(model, next_compound)
            ):
                candidates.append(
                    (
                        stint_cost(length, compound, age, model)
                        + float(pit_loss)
                        + future_cost,
                        (pit_lap,) + future_pits,
                    )
                )
        candidates.sort(key=lambda item: (item[0], item[1]))
        return tuple(candidates[:k])

    raw = solve(0, state.current_completed_lap + 1, state.current_tyre_age)
    return _rank(
        [
            (StrategyPlan(compounds, pit_laps), cost)
            for cost, pit_laps in raw
        ],
        "same_sequence",
        k,
    )


def search_best_continuations(
    state: StrategySearchState,
    model: StrategyModel,
    pit_loss: float,
    max_stops: int | None = None,
    k: int = 10,
    permitted_compounds: Sequence[str] | None = None,
) -> list[StrategyResult]:
    _validate_state(state, model)
    if state.laps_remaining == 0:
        return []
    max_stops = state.max_future_stops if max_stops is None else int(max_stops)
    if max_stops < 0 or k <= 0:
        raise ValueError("max_stops cannot be negative and k must be positive.")
    compounds = tuple(
        dict.fromkeys(
            value.upper() for value in (permitted_compounds or tuple(model.degradation))
        )
    )
    unknown = set(compounds).difference(model.degradation)
    if unknown:
        raise ValueError(f"Unknown permitted compounds: {sorted(unknown)}")

    @lru_cache(maxsize=None)
    def solve(
        next_lap: int, compound: str, age: int, stops_remaining: int
    ) -> tuple[tuple[float, tuple[str, ...], tuple[int, ...]], ...]:
        remaining = state.total_laps - next_lap + 1
        candidates = [(stint_cost(remaining, compound, age, model), (compound,), ())]
        if stops_remaining > 0:
            for length in range(1, remaining):
                pit_lap = next_lap + length - 1
                current_cost = stint_cost(length, compound, age, model) + float(pit_loss)
                for next_compound in compounds:
                    next_age = _start_age(model, next_compound)
                    for future_cost, future_compounds, future_pits in solve(
                        pit_lap + 1, next_compound, next_age, stops_remaining - 1
                    ):
                        candidates.append(
                            (
                                current_cost + future_cost,
                                (compound,) + future_compounds,
                                (pit_lap,) + future_pits,
                            )
                        )
        unique = {}
        for candidate in candidates:
            signature = (candidate[1], candidate[2])
            if signature not in unique or candidate[0] < unique[signature][0]:
                unique[signature] = candidate
        ordered = sorted(unique.values(), key=lambda item: (item[0], item[1], item[2]))
        return tuple(ordered[:k])

    raw = solve(
        state.current_completed_lap + 1,
        state.current_compound.upper(),
        state.current_tyre_age,
        max_stops,
    )
    return _rank(
        [(StrategyPlan(compound_sequence, pit_laps), cost) for cost, compound_sequence, pit_laps in raw],
        "continuation",
        k,
    )


def search_best_compound_sequences(
    state: StrategySearchState,
    model: StrategyModel,
    pit_loss: float,
    max_stops: int | None = None,
    k: int = 10,
    permitted_compounds: Sequence[str] | None = None,
    allow_any_start: bool = False,
    strategy_rules: StrategyRules | None = None,
) -> list[StrategyResult]:
    """Return the single best pit schedule for every compound sequence."""
    _validate_state(state, model)
    max_stops = state.max_future_stops if max_stops is None else int(max_stops)
    if max_stops < 0 or k <= 0:
        raise ValueError("max_stops cannot be negative and k must be positive.")
    compounds = tuple(
        dict.fromkeys(
            value.upper() for value in (permitted_compounds or tuple(model.degradation))
        )
    )
    unknown = set(compounds).difference(model.degradation)
    if unknown:
        raise ValueError(f"Unknown permitted compounds: {sorted(unknown)}")
    candidates: list[tuple[StrategyPlan, float]] = []
    starts = compounds if allow_any_start and state.current_completed_lap == 0 else (
        state.current_compound.upper(),
    )
    for current in starts:
        sequence_state = replace(
            state, current_compound=current, current_tyre_age=_start_age(model, current)
        )
        for stop_count in range(max_stops + 1):
            for tail in product(compounds, repeat=stop_count):
                sequence = (current,) + tuple(tail)
                if sequence_state.laps_remaining < len(sequence):
                    continue
                placeholder_pits = tuple(
                    sequence_state.current_completed_lap + index + 1
                    for index in range(stop_count)
                )
                best = optimise_same_sequence(
                    StrategyPlan(sequence, placeholder_pits),
                    sequence_state,
                    model,
                    pit_loss,
                    k=1,
                )
                if best:
                    if strategy_rules is None or strategy_rules.accepts(best[0].plan):
                        candidates.append((best[0].plan, best[0].remaining_cost))
    return _rank(candidates, "compound_sequence", k)


def search_best_fixed_stop_count(
    state: StrategySearchState,
    model: StrategyModel,
    pit_loss: float,
    *,
    stop_count: int,
    k: int = 1,
    permitted_compounds: Sequence[str] | None = None,
    allow_any_start: bool = False,
    strategy_rules: StrategyRules | None = None,
) -> list[StrategyResult]:
    """Find the best schedule/sequence with exactly `stop_count` stops."""
    _validate_state(state, model)
    if stop_count < 0 or k <= 0:
        raise ValueError("stop_count cannot be negative and k must be positive")
    compounds = tuple(dict.fromkeys(
        value.upper() for value in (permitted_compounds or tuple(model.degradation))
    ))
    unknown = set(compounds).difference(model.degradation)
    if unknown:
        raise ValueError(f"Unknown permitted compounds: {sorted(unknown)}")
    starts = compounds if allow_any_start and state.current_completed_lap == 0 else (
        state.current_compound.upper(),
    )
    candidates: list[tuple[StrategyPlan, float]] = []
    for current in starts:
        sequence_state = replace(
            state, current_compound=current, current_tyre_age=_start_age(model, current)
        )
        for tail in product(compounds, repeat=stop_count):
            sequence = (current,) + tuple(tail)
            if sequence_state.laps_remaining < len(sequence):
                continue
            if strategy_rules is not None and not strategy_rules.accepts(
                StrategyPlan(sequence, tuple(
                    sequence_state.current_completed_lap + index + 1
                    for index in range(stop_count)
                ))
            ):
                continue
            placeholder = tuple(
                sequence_state.current_completed_lap + index + 1 for index in range(stop_count)
            )
            best = optimise_same_sequence(
                StrategyPlan(sequence, placeholder), sequence_state, model, pit_loss, k=1
            )
            if best:
                candidates.append((best[0].plan, best[0].remaining_cost))
    return _rank(candidates, "fixed_stop_count", k)


def full_reoptimisation(
    state: StrategySearchState,
    model: StrategyModel,
    pit_loss: float,
    max_stops: int | None = None,
    k: int = 10,
    permitted_compounds: Sequence[str] | None = None,
) -> list[StrategyResult]:
    return [
        replace(result, search_type="full")
        for result in search_best_continuations(
            state, model, pit_loss, max_stops, k, permitted_compounds
        )
    ]


def results_to_records(
    results: Iterable[StrategyResult],
    state: StrategySearchState,
    model: StrategyModel,
    *,
    search_id: str | None = None,
    status: str = "current",
) -> list[StrategyRecord]:
    search_id = search_id or f"search-{uuid4().hex}"
    created_at = utc_now()
    input_hash = canonical_hash(state)
    return [
        StrategyRecord(
            search_id=search_id,
            strategy_id=f"{search_id}-{result.rank}",
            rank=result.rank,
            search_type=result.search_type,
            selected_driver=state.selected_driver,
            as_of_completed_lap=state.current_completed_lap,
            laps_remaining=state.laps_remaining,
            current_compound=(
                result.plan.compounds[0]
                if state.current_completed_lap == 0 and state.selected_driver is None
                else state.current_compound.upper()
            ),
            current_tyre_age=state.current_tyre_age,
            future_stop_count=len(result.plan.pit_laps),
            compound_sequence=result.plan.compounds,
            pit_laps=result.plan.pit_laps,
            formatted_strategy=result.formatted_strategy,
            remaining_cost=result.remaining_cost,
            delta_to_best=result.delta_to_best,
            model_analysis_id=model.analysis_id,
            model_version=model.version,
            input_state_hash=input_hash,
            status=status,
            created_at=created_at,
        )
        for result in results
    ]


def _rank(
    candidates: Iterable[tuple[StrategyPlan, float]], search_type: str, k: int
) -> list[StrategyResult]:
    unique = {}
    for plan, cost in candidates:
        signature = (plan.compounds, plan.pit_laps)
        if signature not in unique or cost < unique[signature][1]:
            unique[signature] = (plan, float(cost))
    ordered = sorted(unique.values(), key=lambda item: (item[1], item[0].compounds, item[0].pit_laps))[:k]
    if not ordered:
        return []
    best = ordered[0][1]
    return [
        StrategyResult(index, plan, cost, cost - best, search_type, format_strategy(plan))
        for index, (plan, cost) in enumerate(ordered, start=1)
    ]


def _result(plan: StrategyPlan, cost: float, search_type: str) -> StrategyResult:
    return StrategyResult(1, plan, float(cost), 0.0, search_type, format_strategy(plan))


def _start_age(model: StrategyModel, compound: str) -> int:
    return int((model.fitted_start_age or {}).get(compound.upper(), 0))


def _validate_state(state: StrategySearchState, model: StrategyModel) -> None:
    if state.current_completed_lap < 0 or state.total_laps < state.current_completed_lap:
        raise ValueError("Invalid current/total lap state.")
    if state.current_tyre_age < 0:
        raise ValueError("current_tyre_age cannot be negative.")
    if state.current_compound.upper() not in model.degradation:
        raise ValueError(f"Current compound {state.current_compound!r} is not in the model.")


def _validate_plan_horizon(plan: StrategyPlan, state: StrategySearchState) -> None:
    if plan.compounds[0] != state.current_compound.upper():
        raise ValueError("The plan must start on the current compound.")
    lower = state.current_completed_lap
    if any(lap <= lower or lap >= state.total_laps for lap in plan.pit_laps):
        raise ValueError("Every future pit lap must be after the current lap and before the finish.")


def _compound_initial(compound: str) -> str:
    value = compound.strip().upper()
    if not value:
        return "?"
    return {"INTERMEDIATE": "I", "WET": "W"}.get(value, value[0])
