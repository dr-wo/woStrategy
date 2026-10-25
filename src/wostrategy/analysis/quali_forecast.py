"""Experimental final-cutoff forecasts from chronological joint field scenarios.

The prior contains empirical future-best residual vectors from *other, earlier*
events. Runtime inputs contain only current Q bests and, in Q2, previously
observed Q1 bests. No FP reference, fitted track slope, final result, IO or clock
extrapolation is required here. Whole historical field vectors retain shared
conditions; independently drawing an improvement for each driver would not.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
import json
import math
import re
from typing import Iterable, Mapping


SCHEMA_VERSION = 1
METHOD = "empirical-joint-field-rank-v1"
_EVENT = re.compile(r"^(\d{4})-R([1-9]\d*)$")


def _event_order(event_id: str) -> tuple[int, int]:
    match = _EVENT.fullmatch(event_id) if isinstance(event_id, str) else None
    if not match:
        raise ValueError("Event identity must use YYYY-Rn")
    return tuple(map(int, match.groups()))


def _positive(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _finite(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


@dataclass(frozen=True)
class DriverPace:
    driver: str
    current_best_s: float | None = None
    previous_part_best_s: float | None = None
    # False is a caller-certified no-further-attempt constraint, not an
    # inference from last-lap time, operational UNKNOWN, or missing FP inputs.
    future_attempt_possible: bool | None = None


@dataclass(frozen=True)
class PriorDriver:
    rank_fraction: float
    current_delta_fraction: float | None
    previous_delta_fraction: float | None
    future_attempt_probability: float | None


@dataclass(frozen=True)
class FieldTemplate:
    event_id: str
    part: str
    remaining_s: float
    drivers: tuple[PriorDriver, ...]


@dataclass(frozen=True)
class ForecastPrior:
    revision: str
    templates: tuple[FieldTemplate, ...]
    interval_coverage: float = .8
    metadata_json: str = "{}"

    @property
    def training_events(self) -> tuple[str, ...]:
        return tuple(sorted({t.event_id for t in self.templates}, key=_event_order))

    @property
    def metadata(self) -> dict:
        """Return detached provenance, keeping the frozen prior immutable."""
        return json.loads(self.metadata_json)

    @classmethod
    def from_records(cls, records: Iterable[Mapping], *, revision: str,
                     interval_coverage: float = .8, metadata: Mapping | None = None):
        """Calibrate only supplied training records; reject accidental holdouts.

        A record names an event, part, remaining seconds and the complete known
        participant roster. Each driver has checkpoint current/previous bests,
        eventual final best and an optional future-credible-attempt flag. Final
        targets are converted to residuals here and are absent from runtime API.
        Snapshots without complete pace anchors are explicitly omitted rather
        than filling unknown Q1 driver pace from future results.
        """
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("A prior revision is required")
        if not _finite(interval_coverage) or not 0 < interval_coverage < 1:
            raise ValueError("Interval coverage must be between zero and one")
        templates, skipped, identities = [], [], set()
        for record in records:
            if record.get("split", "train") != "train":
                raise ValueError("Only training records may calibrate a forecast prior")
            event_id, part = record["event_id"], record["part"]
            _event_order(event_id)
            remaining = record["remaining_s"]
            if part not in {"Q1", "Q2"} or not _finite(remaining) or remaining < 0:
                raise ValueError("Training records require Q1/Q2 and nonnegative remaining time")
            identity = (event_id, part, float(remaining))
            if identity in identities:
                raise ValueError("Duplicate event/part/checkpoint calibration record")
            identities.add(identity)
            rows = tuple(record["drivers"])
            names = [r["driver"] for r in rows]
            if not rows or len(set(names)) != len(names):
                raise ValueError("A calibration roster must contain unique drivers")
            anchors = []
            for row in rows:
                current, previous = row.get("current_best_s"), row.get("previous_part_best_s")
                anchor = current if _positive(current) else previous if part == "Q2" and _positive(previous) else None
                if anchor is None or not _positive(row.get("final_best_s")):
                    break
                anchors.append((float(anchor), row["driver"], row))
            if len(anchors) != len(rows):
                skipped.append({"event_id": event_id, "part": part, "remaining_s": remaining,
                                "reason": "INCOMPLETE_PACE_OR_FINAL_TARGET_COVERAGE"})
                continue
            anchors.sort(key=lambda value: (value[0], value[1]))
            values = []
            for index, (_, _, row) in enumerate(anchors):
                current, previous, final = row.get("current_best_s"), row.get("previous_part_best_s"), float(row["final_best_s"])
                future = row.get("future_credible_attempt")
                if future is not None and not isinstance(future, bool):
                    raise ValueError("Future-attempt calibration evidence must be boolean or null")
                values.append(PriorDriver(
                    (index + .5) / len(anchors),
                    min(0., final / current - 1.) if _positive(current) else None,
                    final / previous - 1. if part == "Q2" and _positive(previous) else None,
                    float(future) if future is not None else None,
                ))
            templates.append(FieldTemplate(event_id, part, float(remaining), tuple(values)))
        provenance = dict(metadata or {})
        provenance.update({"method": METHOD, "skipped_records": skipped,
                           "calibration_target": "eventual final best per driver",
                           "opportunity_model": "empirical remaining-time marginal; zero future attempts retained"})
        prior = cls(revision, tuple(sorted(templates, key=lambda t: (_event_order(t.event_id), t.part, t.remaining_s))),
                    float(interval_coverage), json.dumps(provenance, sort_keys=True, allow_nan=False))
        prior._validate()
        return prior

    def _validate(self):
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("A prior revision is required")
        if not _finite(self.interval_coverage) or not 0 < self.interval_coverage < 1:
            raise ValueError("Invalid interval coverage")
        if not isinstance(self.metadata, dict) or self.metadata.get("method") != METHOD:
            raise ValueError("Missing or unsupported calibration method metadata")
        seen = set()
        for template in self.templates:
            _event_order(template.event_id)
            identity = (template.event_id, template.part, template.remaining_s)
            if identity in seen or template.part not in {"Q1", "Q2"}:
                raise ValueError("Invalid or duplicate calibration template")
            seen.add(identity)
            if not _finite(template.remaining_s) or template.remaining_s < 0 or not template.drivers:
                raise ValueError("Invalid calibration lead or roster")
            previous_rank = -1.
            for row in template.drivers:
                if not _finite(row.rank_fraction) or not 0 < row.rank_fraction < 1 or row.rank_fraction <= previous_rank:
                    raise ValueError("Calibration driver ranks must be ordered inside (0, 1)")
                previous_rank = row.rank_fraction
                for value in (row.current_delta_fraction, row.previous_delta_fraction):
                    if value is not None and (not _finite(value) or value <= -1):
                        raise ValueError("Invalid normalized final-best residual")
                if row.current_delta_fraction is not None and row.current_delta_fraction > 0:
                    raise ValueError("Future valid best cannot worsen an observed current best")
                if (row.future_attempt_probability is not None
                        and (not _finite(row.future_attempt_probability)
                             or not 0 <= row.future_attempt_probability <= 1)):
                    raise ValueError("Invalid empirical future-attempt support")

    def to_dict(self) -> dict:
        self._validate()
        return {"schema_version": SCHEMA_VERSION, "revision": self.revision,
                "interval_coverage": self.interval_coverage,
                "training_events": list(self.training_events), "metadata": self.metadata,
                "templates": [{"event_id": t.event_id, "part": t.part, "remaining_s": t.remaining_s,
                    "drivers": [{"rank_fraction": d.rank_fraction,
                        "current_delta_fraction": d.current_delta_fraction,
                        "previous_delta_fraction": d.previous_delta_fraction,
                        "future_attempt_probability": d.future_attempt_probability} for d in t.drivers]}
                    for t in self.templates]}

    @classmethod
    def from_dict(cls, value: Mapping):
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported forecast prior schema")
        prior = cls(value["revision"], tuple(FieldTemplate(
            t["event_id"], t["part"], t["remaining_s"], tuple(PriorDriver(**d) for d in t["drivers"]))
            for t in value["templates"]), value["interval_coverage"],
            json.dumps(value["metadata"], sort_keys=True, allow_nan=False))
        prior._validate()
        if tuple(value.get("training_events", ())) != prior.training_events:
            raise ValueError("Training event manifest does not match prior templates")
        return prior


@dataclass(frozen=True)
class FieldScenario:
    training_event: str
    driver_final_bests_s: tuple[tuple[str, float | None], ...]
    future_attempt_support: tuple[tuple[str, float | None], ...]
    cutoff_s: float


@dataclass(frozen=True)
class FinalCutoffForecast:
    status: str
    source: str
    event_id: str | None
    part: str
    target_rank: int
    remaining_s: float | None
    estimate_s: float | None = None
    lower_s: float | None = None
    upper_s: float | None = None
    interval_coverage: float | None = None
    prior_revision: str | None = None
    current_pace_count: int = 0
    previous_part_pace_count: int = 0
    driver_count: int = 0
    scenarios: tuple[FieldScenario, ...] = ()
    limitations: tuple[str, ...] = ()
    source_events: tuple[str, ...] = ()

    @property
    def scenario_count(self) -> int:
        return len(self.source_events) if self.source_events else len(self.scenarios)

    @property
    def training_events(self) -> tuple[str, ...]:
        return self.source_events or tuple(s.training_event for s in self.scenarios)


def _interpolate(points, position):
    if not points:
        return None
    if position <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if position <= x1:
            return y0 + (y1 - y0) * (position - x0) / (x1 - x0)
    return points[-1][1]


def _template_value(template, rank, name):
    return _interpolate([(d.rank_fraction, getattr(d, name)) for d in template.drivers
                         if getattr(d, name) is not None], rank)


def _quantile(values, fraction):
    ordered = sorted(values)
    coordinate = fraction * (len(ordered) - 1)
    lo, hi = math.floor(coordinate), math.ceil(coordinate)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (coordinate - lo)


def forecast_final_cutoff(drivers: Iterable[DriverPace], *, event_id: str | None,
                          part: str, remaining_s: float | None,
                          prior: ForecastPrior | None, target_rank: int | None = None,
                          part_finalised: bool = False) -> FinalCutoffForecast:
    """Forecast fixed final P16/P10 using complete causal driver pace anchors.

    The interval is a central empirical field-scenario range, not a confidence
    interval or a claim of calibrated prospective coverage. Current-event and
    future-event training data are rejected at this boundary. Clock zero alone
    does not mean finalised: still-running laps can complete after the flag.
    """
    values = tuple(drivers)
    target_rank = target_rank if target_rank is not None else {"Q1": 16, "Q2": 10}.get(part, 0)
    base = dict(event_id=event_id, part=part, target_rank=target_rank,
                remaining_s=float(remaining_s) if _finite(remaining_s) else None,
                driver_count=len(values), current_pace_count=sum(_positive(d.current_best_s) for d in values),
                previous_part_pace_count=sum(not _positive(d.current_best_s) and part == "Q2"
                                            and _positive(d.previous_part_best_s) for d in values))

    def unavailable(reason):
        return FinalCutoffForecast("UNAVAILABLE", "INSUFFICIENT_EVIDENCE", **base,
                                   limitations=(reason,))

    if part not in {"Q1", "Q2"} or not isinstance(target_rank, int) or isinstance(target_rank, bool) or target_rank < 1:
        return unavailable("Only a fixed positive Q1/Q2 final-cutoff rank is supported.")
    if len(values) < target_rank or len({d.driver for d in values}) != len(values):
        return unavailable("A unique qualifying participant roster covering the target rank is required.")
    if any(not isinstance(d.driver, str) or not d.driver.strip()
           or (d.future_attempt_possible is not None and not isinstance(d.future_attempt_possible, bool)) for d in values):
        return unavailable("Invalid driver identity or opportunity constraint.")
    try:
        order = _event_order(event_id)
    except ValueError:
        return unavailable("A valid current event identity is required to enforce chronological calibration.")
    if part_finalised:
        observed = sorted(d.current_best_s for d in values if _positive(d.current_best_s))
        if len(observed) < target_rank:
            return unavailable("Finalised part does not contain enough observed valid times for the target rank.")
        cutoff = observed[target_rank - 1]
        return FinalCutoffForecast("OBSERVED_FINAL", "OBSERVED_FINAL_CLASSIFICATION", **base,
                                   estimate_s=cutoff, lower_s=cutoff, upper_s=cutoff,
                                   limitations=("Finalised classification; no future attempts assumed.",))
    if not _finite(remaining_s) or remaining_s < 0:
        return unavailable("A nonnegative, observed remaining session time is required.")
    if prior is None:
        return unavailable("No calibrated historical field prior is available; current cutoff is not substituted as a forecast.")
    try:
        prior._validate()
    except (ValueError, TypeError, KeyError) as exc:
        return unavailable(f"Invalid calibration prior: {exc}")
    if not prior.training_events or max(map(_event_order, prior.training_events)) >= order:
        return unavailable("All calibration training events must strictly precede the forecast event.")
    anchored = []
    for driver in values:
        anchor = (driver.current_best_s if _positive(driver.current_best_s) else
                  driver.previous_part_best_s if part == "Q2" and _positive(driver.previous_part_best_s) else None)
        if anchor is None and driver.future_attempt_possible is not False:
            return unavailable(f"No causal current-part pace or calibrated previous-Q1 anchor is available for {driver.driver}.")
        anchored.append((anchor if anchor is not None else math.inf, driver.driver, driver))
    anchored.sort(key=lambda item: (item[0], item[1]))
    grouped = {}
    for template in prior.templates:
        if template.part == part:
            grouped.setdefault(template.event_id, []).append(template)
    scenarios = []
    for training_event in sorted(grouped, key=_event_order):
        templates = sorted(grouped[training_event], key=lambda t: t.remaining_s)
        leads = [t.remaining_s for t in templates]
        if remaining_s < leads[0] or remaining_s > leads[-1]:
            continue
        index = bisect_left(leads, remaining_s)
        lower = upper = templates[index]
        if leads[index] != remaining_s:
            lower = templates[index - 1]
        weight = ((remaining_s - lower.remaining_s) / (upper.remaining_s - lower.remaining_s)
                  if upper.remaining_s != lower.remaining_s else 0.)
        finals, opportunities = [], []
        for rank, (_, code, driver) in enumerate(anchored):
            position = (rank + .5) / len(anchored)
            if driver.future_attempt_possible is False:
                final = float(driver.current_best_s) if _positive(driver.current_best_s) else None
                opportunity = 0.
            else:
                name, anchor = "current_delta_fraction", driver.current_best_s
                low = _template_value(lower, position, name) if _positive(anchor) else None
                high = _template_value(upper, position, name) if _positive(anchor) else None
                if low is None or high is None:
                    name, anchor = "previous_delta_fraction", driver.previous_part_best_s
                    low = _template_value(lower, position, name) if part == "Q2" and _positive(anchor) else None
                    high = _template_value(upper, position, name) if part == "Q2" and _positive(anchor) else None
                if low is None or high is None:
                    break
                final = float(anchor) * (1. + low + weight * (high - low))
                if _positive(driver.current_best_s):
                    final = min(final, driver.current_best_s)
                lo_opp = _template_value(lower, position, "future_attempt_probability")
                hi_opp = _template_value(upper, position, "future_attempt_probability")
                opportunity = lo_opp + weight * (hi_opp - lo_opp) if lo_opp is not None and hi_opp is not None else None
            finals.append((code, final))
            opportunities.append((code, opportunity))
        if len(finals) != len(values):
            continue
        timed = sorted(value for _, value in finals if _positive(value))
        if len(timed) >= target_rank:
            scenarios.append(FieldScenario(training_event, tuple(finals), tuple(opportunities), timed[target_rank - 1]))
    if len(scenarios) < 2:
        return unavailable("Fewer than two earlier-event field scenarios support this part, lead time and pace basis.")
    cuts = [s.cutoff_s for s in scenarios]
    tail = (1 - prior.interval_coverage) / 2
    return FinalCutoffForecast(
        "EXPERIMENTAL", "HISTORICAL_JOINT_FIELD_PRIOR", **base,
        estimate_s=_quantile(cuts, .5), lower_s=_quantile(cuts, tail),
        upper_s=_quantile(cuts, 1 - tail), interval_coverage=prior.interval_coverage,
        prior_revision=prior.revision, scenarios=tuple(scenarios),
        limitations=("Experimental empirical scenario range; prospective coverage is unvalidated.",
                     "Remaining-time opportunity uncertainty is historical, not a per-driver pit/cool feasibility measurement.",
                     "FP and live Track Evolution are not added to this calibrated residual model.",
                     "Currently valid bests bound scenarios; future deletions or retractions are not modelled.",
                     "Fractional improvement transfers across circuits; tyre, weather and run position are not explicitly conditioned."),
    )


def forecast_partial_field_bounds(drivers: Iterable[DriverPace], *, event_id: str | None,
                                  part: str, remaining_s: float | None,
                                  prior: ForecastPrior | None,
                                  target_rank: int | None = None) -> FinalCutoffForecast:
    """Enclose configured scenarios without assigning unknown driver pace.

    With m unknown cars, target rank r lies between known ranks r-m and r.
    Known cars' prior ranks can also shift by0..m; extrema include every knot
    of the piecewise-linear rank residuals. Per-car extrema overenclose joint
    constraints, so this is conservative *conditional on the empirical model*.
    It is neither a point forecast nor a probabilistic prediction interval.
    """
    values = tuple(drivers)
    point = forecast_final_cutoff(values, event_id=event_id, part=part,
        remaining_s=remaining_s, prior=prior, target_rank=target_rank)
    # Reuse all identity, chronological, prior, roster and clock checks. A
    # complete field keeps precisely the existing point-model result.
    if not any(message.startswith("No causal current-part pace") for message in point.limitations):
        return point
    known, unknown = [], []
    for driver in values:
        anchor = (driver.current_best_s if _positive(driver.current_best_s) else
                  driver.previous_part_best_s if part == "Q2" and _positive(driver.previous_part_best_s) else None)
        if anchor is not None:
            known.append((anchor, driver.driver, driver))
        elif driver.future_attempt_possible is not False:
            unknown.append(driver.driver)
    known.sort(key=lambda item: (item[0], item[1]))
    m, rank, size = len(unknown), point.target_rank, len(values)
    if rank <= m or len(known) < rank:
        return replace(point, limitations=("Unknown pace leaves at least one cutoff-bound endpoint unbounded; no finite scenario envelope is published.",))
    grouped = {}
    for template in prior.templates:
        if template.part == part:
            grouped.setdefault(template.event_id, []).append(template)
    bounds, events = [], []
    for training_event in sorted(grouped, key=_event_order):
        templates = sorted(grouped[training_event], key=lambda t: t.remaining_s)
        leads = [t.remaining_s for t in templates]
        if remaining_s < leads[0] or remaining_s > leads[-1]:
            continue
        index = bisect_left(leads, remaining_s)
        lower = upper = templates[index]
        if leads[index] != remaining_s:
            lower = templates[index - 1]
        weight = ((remaining_s - lower.remaining_s) / (upper.remaining_s - lower.remaining_s)
                  if upper.remaining_s != lower.remaining_s else 0.)
        points = {name: tuple([(d.rank_fraction, getattr(d, name)) for d in t.drivers
                              if getattr(d, name) is not None] for t in (lower, upper))
                  for name in ("current_delta_fraction", "previous_delta_fraction")}
        low_bests, high_bests = [], []
        for position, (_, _, driver) in enumerate(known):
            if driver.future_attempt_possible is False:
                lo = hi = driver.current_best_s
            else:
                name, anchor = "current_delta_fraction", driver.current_best_s
                if not _positive(anchor) or not all(points[name]):
                    name, anchor = "previous_delta_fraction", driver.previous_part_best_s
                if not _positive(anchor) or not all(points[name]):
                    break
                left, right = (position + .5) / size, (position + m + .5) / size
                knots = {left, right} | {x for group in points[name] for x, _ in group if left < x < right}
                possible = []
                for x in knots:
                    a, b = (_interpolate(group, x) for group in points[name])
                    final = anchor * (1. + a + weight * (b - a))
                    possible.append(min(final, driver.current_best_s) if _positive(driver.current_best_s) else final)
                lo, hi = min(possible), max(possible)
            if not _positive(lo) or not _positive(hi):
                break
            low_bests.append(lo)
            high_bests.append(hi)
        if len(low_bests) == len(known):
            bounds.append((sorted(low_bests)[rank - m - 1], sorted(high_bests)[rank - 1]))
            events.append(training_event)
    if len(bounds) < 2:
        return replace(point, limitations=("Fewer than two earlier-event scenarios support a finite partial-field envelope at this lead.",))
    return replace(point, status="SCENARIO_RANGE", source="PARTIAL_FIELD_SCENARIO_ENVELOPE",
        lower_s=min(a for a, _ in bounds), upper_s=max(b for _, b in bounds),
        interval_coverage=None, prior_revision=prior.revision, source_events=tuple(events),
        limitations=(f"Unknown driver pace: {', '.join(sorted(unknown))}; no point estimate or unknown-driver probabilities are assigned.",
            "Conditional envelope over all unknown-car rank placements and every supported historical field scenario; no probability coverage is asserted.",
            "Per-driver extrema conservatively overenclose joint placement constraints; future conditions may lie outside the configured empirical scenarios.",
            "Currently valid bests persist; future deletions, weather and tyre changes are not explicitly modelled."))


__all__ = ["DriverPace", "ForecastPrior", "FieldScenario", "FinalCutoffForecast", "forecast_final_cutoff",
           "forecast_partial_field_bounds"]
