"""MEDIUM-degradation strategy-envelope cutoff analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil, isfinite
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
    primary_0_to_1: float | None
    primary_0_to_1_status: str
    primary_1_to_2: float | None
    primary_1_to_2_status: str
    primary_2_to_3: float | None
    non_monotonic: bool
    reversal_regions: tuple[tuple[float, float], ...]
    sweep: tuple[CutoffPoint, ...]
    warnings: tuple[str, ...]
    rules: StrategyRules
    initial_scan_max: float | None = None
    effective_scan_max: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["cutoff_table"] = [
            {
                "transition": "0_to_1",
                "display_label": "0→1",
                "cutoff_medium_degradation": self.primary_0_to_1,
                "status": self.primary_0_to_1_status,
            },
            {
                "transition": "1_to_2",
                "display_label": "1→2",
                "cutoff_medium_degradation": self.primary_1_to_2,
                "status": self.primary_1_to_2_status,
            },
            {
                "transition": "2_to_3",
                "display_label": "2→3",
                "cutoff_medium_degradation": self.primary_2_to_3,
                "status": (
                    "found_initial_scan"
                    if self.primary_2_to_3 is not None
                    else "not_found_in_scan_range"
                ),
            },
        ]
        # Older in-memory results can infer the upper bound from their sweep.
        upper = self.sweep[-1].medium_degradation if self.sweep else None
        payload["initial_scan_max"] = (
            self.initial_scan_max if self.initial_scan_max is not None else upper
        )
        payload["effective_scan_max"] = (
            self.effective_scan_max if self.effective_scan_max is not None else upper
        )
        for row, pair in zip(payload["cutoff_table"], ((0, 1), (1, 2), (2, 3))):
            # 2→3 selection intentionally excludes the lower-extension samples.
            points = tuple(
                point for point in self.sweep
                if pair != (2, 3)
                or point.medium_degradation >= self.initial_scan_min - 1e-12
            )
            row["coverage"] = _transition_coverage(
                points, pair, rules_unavailable=row["status"] == "unavailable_rules_compliant",
            )
            row["display_text"] = _cutoff_display_text(row)
        return payload


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
        if stops == 0 and rules.minimum_distinct_dry_compounds > 1:
            return None
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
    stop_counts = (0, 1, 2, 3)
    zero_stop_rules_available = rules.minimum_distinct_dry_compounds <= 1
    points = [
        _point(value, {stops: cost(value, stops) for stops in stop_counts})
        for value in values
    ]
    initial_crossings = _crossing_brackets(points)
    brackets_01 = (
        _positive_crossing_brackets(initial_crossings[(0, 1)], (0, 1), cost)
        if zero_stop_rules_available else []
    )
    brackets_12 = _positive_crossing_brackets(
        initial_crossings[(1, 2)], (1, 2), cost,
    )
    brackets_23 = initial_crossings[(2, 3)]
    primary_01_status = (
        "found_initial_scan" if brackets_01
        else "unavailable_rules_compliant" if not zero_stop_rules_available
        else "not_found"
    )
    primary_12_status = "found_initial_scan" if brackets_12 else "not_found"

    # An automatic prediction-centred range is only an initial search window.
    # Progressively extend it toward zero for missing 0→1 and 1→2 crossings.
    # An explicit scan_min remains a hard engineering constraint and is never crossed.
    needs_lower_extension = (
        not brackets_12 or (zero_stop_rules_available and not brackets_01)
    )
    if needs_lower_extension and scan_min is None and minimum > 0:
        while (
            not brackets_12 or (zero_stop_rules_available and not brackets_01)
        ) and minimum > 0:
            next_minimum = 0.0 if minimum <= step else minimum / 2.0
            extension_values = _scan_values(next_minimum, minimum, step)
            extension_points = [
                _point(value, {stops: cost(value, stops) for stops in stop_counts})
                for value in extension_values[:-1]
            ]
            points = extension_points + points
            minimum = next_minimum
            extended_crossings = _crossing_brackets(points)
            if not brackets_01 and zero_stop_rules_available:
                brackets_01 = _positive_crossing_brackets(
                    extended_crossings[(0, 1)], (0, 1), cost,
                )
            if not brackets_12:
                brackets_12 = _positive_crossing_brackets(
                    extended_crossings[(1, 2)], (1, 2), cost,
                )
        if primary_01_status == "not_found":
            primary_01_status = (
                "found_after_lower_extension"
                if brackets_01 else "no_positive_degradation_crossing"
            )
        if primary_12_status == "not_found":
            primary_12_status = (
                "found_after_lower_extension"
                if brackets_12 else "no_positive_degradation_crossing"
            )
    if (not brackets_01 and zero_stop_rules_available and scan_min is not None
            and primary_01_status == "not_found"):
        primary_01_status = "not_found_above_explicit_scan_min"
    if (not brackets_12 and scan_min is not None
            and primary_12_status == "not_found"):
        primary_12_status = "not_found_above_explicit_scan_min"

    refined_01 = [
        _refine_crossing(a, b, (0, 1), cost, tolerance)
        for a, b in brackets_01
    ]
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
    primary_01 = refined_01[0] if refined_01 else None
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
    if primary_01 is None and zero_stop_rules_available:
        if primary_01_status == "no_positive_degradation_crossing":
            warnings.append(
                "No positive-degradation 0→1 stop crossing was found after "
                "extending the automatic scan to zero."
            )
        else:
            warnings.append(
                "No 0→1 stop crossing was found at or above the explicit "
                "scan_min hard lower bound."
            )
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
        predicted_medium_degradation=predicted,
        initial_scan_min=initial_minimum,
        effective_scan_min=minimum,
        primary_0_to_1=primary_01,
        primary_0_to_1_status=primary_01_status,
        primary_1_to_2=primary_12,
        primary_1_to_2_status=primary_12_status,
        primary_2_to_3=primary_23,
        non_monotonic=non_monotonic,
        reversal_regions=tuple(reversals),
        sweep=tuple(points),
        warnings=tuple(warnings),
        rules=rules,
        initial_scan_max=maximum,
        effective_scan_max=points[-1].medium_degradation,
    )


def _transition_coverage(points, pair, *, rules_unavailable=False):
    """Describe sampled evidence, never an untested mathematical bound."""
    def finite_cost(point, stops):
        value = point.costs.get(stops)
        return float(value) if value is not None and isfinite(value) else None

    comparable = [all(finite_cost(point, stops) is not None for stops in pair)
                  for point in points]
    intervals = sum(left and right for left, right in zip(comparable, comparable[1:]))
    boundary = points[-1] if points else None
    costs = [finite_cost(boundary, stops) if boundary else None for stops in pair]
    feasible = [
        (None if boundary is None else
         False if boundary.costs.get(stops) is None else
         True if costs[index] is not None else None)
        for index, stops in enumerate(pair)
    ]
    margin = costs[1] - costs[0] if all(value is not None for value in costs) else None
    if rules_unavailable:
        status = "unavailable_rules_compliant"
    elif feasible == [False, False]:
        status = "both_classes_unavailable"
    elif feasible[0] is False:
        status = "lower_class_unavailable"
    elif feasible[1] is False:
        status = "higher_class_unavailable"
    elif feasible != [True, True] or not intervals:
        status = "insufficient_comparable_samples"
    else:
        status = "both_classes_feasible"
    return {
        "scan_min": points[0].medium_degradation if points else None,
        "scan_max": boundary.medium_degradation if boundary else None,
        "comparable_sample_count": sum(comparable),
        "comparable_interval_count": intervals,
        "status": status,
        "upper_boundary": {
            "medium_degradation": boundary.medium_degradation if boundary else None,
            "lower_stop_count": pair[0],
            "higher_stop_count": pair[1],
            "lower_stop_feasible": feasible[0],
            "higher_stop_feasible": feasible[1],
            "lower_stop_cost": costs[0],
            "higher_stop_cost": costs[1],
            "higher_minus_lower_cost": margin,
            "lower_cost_stop_count": (
                pair[0] if margin is not None and margin > 0 else
                pair[1] if margin is not None and margin < 0 else None
            ),
        },
    }


def _cutoff_display_text(row):
    label = row["display_label"]
    value = row["cutoff_medium_degradation"]
    if value is not None:
        return f"{label}: {value:.6g} s/lap ({row['status']})."
    if row["status"] == "unavailable_rules_compliant":
        return f"{label}: unavailable under rules."
    coverage = row["coverage"]
    upper, lower = coverage["scan_max"], coverage["scan_min"]
    if upper is None:
        return f"{label}: cutoff unavailable; scan coverage unavailable."
    text = (f"{label}: cutoff not found through M degradation = {upper:.6g} s/lap "
            f"(tested from {lower:.6g}).")
    boundary = coverage["upper_boundary"]
    status = coverage["status"]
    if status == "both_classes_feasible":
        margin = boundary["higher_minus_lower_cost"]
        winner = boundary["lower_cost_stop_count"]
        if margin == 0:
            return text + " Both fixed-stop classes had equal cost at the scan limit."
        loser = (boundary["higher_stop_count"] if winner == boundary["lower_stop_count"]
                 else boundary["lower_stop_count"])
        return text + (f" At the scan limit, {winner}-stop was {abs(margin):.2f} s "
                       f"lower-cost than {loser}-stop.")
    unavailable = {
        "lower_class_unavailable": f"{boundary['lower_stop_count']}-stop was unavailable",
        "higher_class_unavailable": f"{boundary['higher_stop_count']}-stop was unavailable",
        "both_classes_unavailable": "Both fixed-stop classes were unavailable",
    }
    if status in unavailable:
        return text + f" {unavailable[status]} at the scan limit; no boundary comparison."
    return text + " Insufficient adjacent comparable samples to test a crossing."


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
    output = {(0, 1): [], (1, 2): [], (2, 3): []}
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
