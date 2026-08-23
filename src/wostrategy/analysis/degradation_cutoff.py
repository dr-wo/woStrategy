"""MEDIUM-degradation strategy-envelope cutoff analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil
from typing import Callable, Mapping

from wodata.contracts import StrategySearchState

from wostrategy.algorithm.exact_strategy_search import (
    StrategyModel,
    StrategyResult,
    StrategyRules,
    search_best_fixed_stop_count,
)


@dataclass(frozen=True)
class CutoffPoint:
    medium_degradation: float
    costs: Mapping[int, float | None]
    preferred_stop_count: int | None


@dataclass(frozen=True)
class DegradationCutoffResult:
    predicted_medium_degradation: float
    initial_scan_min: float
    effective_scan_min: float
    primary_1_to_2: float | None
    primary_1_to_2_status: str
    primary_2_to_3: float | None
    non_monotonic: bool
    reversal_regions: tuple[tuple[float, float], ...]
    sweep: tuple[CutoffPoint, ...]
    warnings: tuple[str, ...]
    rules: StrategyRules

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calculate_degradation_cutoffs(
    *,
    model: StrategyModel,
    state: StrategySearchState,
    pit_loss: float,
    strategy_rules: StrategyRules | None = None,
    scan_min: float | None = None,
    scan_max: float | None = None,
    coarse_step: float | None = None,
    refine_tolerance: float | None = None,
    evaluator: Callable[[float, int, StrategyRules], float | None] | None = None,
) -> DegradationCutoffResult:
    rules = strategy_rules or StrategyRules()
    predicted = float(model.degradation.get("MEDIUM", 0.0))
    if abs(predicted) < 1e-9:
        raise ValueError(
            "MEDIUM degradation is too close to zero to preserve S/M/H degradation ratios"
        )
    minimum = max(0.0, predicted * 0.25) if scan_min is None else float(scan_min)
    maximum = max(predicted * 3.0, minimum + 0.05) if scan_max is None else float(scan_max)
    step = max((maximum - minimum) / 30.0, 0.005) if coarse_step is None else float(coarse_step)
    tolerance = min(step / 8.0, 0.001) if refine_tolerance is None else float(refine_tolerance)
    if minimum < 0:
        raise ValueError("degradation cutoff scan_min cannot be negative")
    if maximum <= minimum or step <= 0 or tolerance <= 0:
        raise ValueError("degradation cutoff scan bounds/steps must be positive and ordered")

    def cost(value: float, stops: int) -> float | None:
        if value < 0:
            raise ValueError("MEDIUM degradation scan cannot evaluate negative values")
        if evaluator is not None:
            return evaluator(value, stops, rules)
        ratio = value / predicted
        scaled = replace(model, degradation={key: float(item) * ratio for key, item in model.degradation.items()})
        results = search_best_fixed_stop_count(
            state, scaled, pit_loss, stop_count=stops, k=1,
            allow_any_start=state.current_completed_lap == 0 and state.selected_driver is None,
            strategy_rules=rules,
        )
        return results[0].remaining_cost if results else None

    initial_minimum = minimum
    values = _scan_values(minimum, maximum, step)
    points = [_point(value, {stops: cost(value, stops) for stops in (1, 2, 3)}) for value in values]
    initial_crossings = _crossing_brackets(points)
    brackets_12 = _positive_crossing_brackets(
        initial_crossings[(1, 2)], (1, 2), cost,
    )
    brackets_23 = initial_crossings[(2, 3)]
    primary_12_status = "found_initial_scan" if brackets_12 else "not_found"

    # An automatic prediction-centred range is only an initial search window.
    # Progressively extend it toward zero for the 1→2 crossing. An explicit
    # scan_min remains a hard engineering constraint and is never crossed.
    if not brackets_12 and scan_min is None and minimum > 0:
        while not brackets_12 and minimum > 0:
            next_minimum = 0.0 if minimum <= step else minimum / 2.0
            extension_values = _scan_values(next_minimum, minimum, step)
            extension_points = [
                _point(value, {stops: cost(value, stops) for stops in (1, 2, 3)})
                for value in extension_values[:-1]
            ]
            points = extension_points + points
            minimum = next_minimum
            brackets_12 = _positive_crossing_brackets(
                _crossing_brackets(points)[(1, 2)], (1, 2), cost,
            )
        primary_12_status = (
            "found_after_lower_extension"
            if brackets_12 else "no_positive_degradation_crossing"
        )
    elif not brackets_12 and scan_min is not None:
        primary_12_status = "not_found_above_explicit_scan_min"

    refined_12 = [
        _refine_crossing(a, b, (1, 2), cost, tolerance)
        for a, b in brackets_12
    ]
    # Preserve the existing 2→3 logic: only brackets from the original scan
    # participate in primary 2→3 selection.
    refined_23 = [
        _refine_crossing(a, b, (2, 3), cost, tolerance)
        for a, b in brackets_23
    ]
    primary_12 = refined_12[0] if refined_12 else None
    primary_23 = refined_23[0] if refined_23 else None
    preferred = [point.preferred_stop_count for point in points if point.preferred_stop_count is not None]
    reversals = []
    for left, right in zip(points, points[1:]):
        if (left.preferred_stop_count is not None and right.preferred_stop_count is not None
                and right.preferred_stop_count < left.preferred_stop_count):
            reversals.append((left.medium_degradation, right.medium_degradation))
    non_monotonic = bool(reversals)
    warnings = []
    if non_monotonic:
        warnings.append("Strategy envelope is non-monotonic; primary cutoffs are simplified references.")
    if primary_12 is None:
        if primary_12_status == "no_positive_degradation_crossing":
            warnings.append(
                "No positive-degradation 1→2 stop crossing was found after "
                "extending the automatic scan to zero."
            )
        else:
            warnings.append(
                "No 1→2 stop crossing was found at or above the explicit "
                "scan_min hard lower bound."
            )
    if primary_23 is None:
        warnings.append("No 2→3 stop crossing was found inside the scan range.")
    return DegradationCutoffResult(
        predicted, initial_minimum, minimum, primary_12, primary_12_status,
        primary_23, non_monotonic, tuple(reversals), tuple(points),
        tuple(warnings), rules,
    )


def _scan_values(minimum: float, maximum: float, step: float) -> list[float]:
    """Return inclusive non-negative scan points with gaps no larger than step."""
    minimum = max(0.0, float(minimum))
    maximum = max(minimum, float(maximum))
    span = maximum - minimum
    if span == 0:
        return [round(minimum, 12)]
    interval_count = max(1, int(ceil(span / step)))
    interval = span / interval_count
    return [
        round(minimum + interval * index, 12)
        for index in range(interval_count + 1)
    ]


def _positive_crossing_brackets(brackets, pair, cost):
    """Exclude equality at exactly zero, which is not a positive-degradation cutoff."""
    output = []
    for lower, upper in brackets:
        if lower == 0:
            first, second = cost(0.0, pair[0]), cost(0.0, pair[1])
            if first is not None and second is not None and first == second:
                continue
        output.append((lower, upper))
    return output


def _point(value: float, costs: Mapping[int, float | None]) -> CutoffPoint:
    available = {key: item for key, item in costs.items() if item is not None}
    preferred = min(available, key=lambda key: (available[key], key)) if available else None
    return CutoffPoint(value, dict(costs), preferred)


def _crossing_brackets(points: list[CutoffPoint]) -> dict[tuple[int, int], list[tuple[float, float]]]:
    output = {(1, 2): [], (2, 3): []}
    for left, right in zip(points, points[1:]):
        for pair in output:
            a0, b0 = left.costs[pair[0]], left.costs[pair[1]]
            a1, b1 = right.costs[pair[0]], right.costs[pair[1]]
            if None not in (a0, b0, a1, b1) and (a0 - b0) * (a1 - b1) <= 0:
                output[pair].append((left.medium_degradation, right.medium_degradation))
    return output


def _refine_crossing(a: float, b: float, pair, cost, tolerance: float) -> float:
    def difference(value):
        first, second = cost(value, pair[0]), cost(value, pair[1])
        if first is None or second is None:
            raise ValueError("Strategy became unavailable while refining cutoff")
        return first - second
    fa = difference(a)
    while b - a > tolerance:
        middle = (a + b) / 2.0
        fm = difference(middle)
        if fa * fm <= 0:
            b = middle
        else:
            a, fa = middle, fm
    return (a + b) / 2.0


__all__ = ["CutoffPoint", "DegradationCutoffResult", "calculate_degradation_cutoffs"]
