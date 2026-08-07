from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

TRAFFIC_EVALUATOR_VERSION = "unified-traffic-v2"
COMBINATION_MODE_COMPATIBILITY_MIN = "compatibility_min"

DIRECT_FASTF1_STYLE = "DIRECT_FASTF1_STYLE"
PHYSICAL_CIRCULAR = "PHYSICAL_CIRCULAR"
SECTOR_BOUNDARY_FALLBACK = "SECTOR_BOUNDARY_FALLBACK"

TRAFFIC_STATUS_OK = "ok"
TRAFFIC_STATUS_PARTIAL = "partial_coverage"
TRAFFIC_STATUS_UNKNOWN = "unknown"

NORMALIZED_REQUIRED_COLUMNS = (
    "Driver",
    "SessionTime",
    "LapNumber",
    "Speed",
)


@dataclass(frozen=True)
class TrafficEvaluationConfig:
    circuit_length_m: float
    interpolation_tolerance_seconds: float = 0.5
    maximum_bracketing_interval_seconds: float = 1.0
    minimum_gap_m: float = 1.0
    minimum_coverage_fraction: float = 0.5
    minimum_lap_samples: int = 2
    combination_mode: str = COMBINATION_MODE_COMPATIBILITY_MIN
    evaluator_version: str = TRAFFIC_EVALUATOR_VERSION


@dataclass(frozen=True)
class TrafficEvaluationResult:
    samples: pd.DataFrame
    laps: pd.DataFrame


@dataclass(frozen=True)
class DirectTrafficEvaluationConfig:
    """Quality controls for the normalized FastF1-style direct component."""

    interpolation_tolerance_seconds: float = 0.5
    maximum_bracketing_interval_seconds: float = 1.0
    minimum_coverage_fraction: float = 0.5
    minimum_lap_samples: int = 3
    evaluator_version: str = TRAFFIC_EVALUATOR_VERSION


class DirectTrafficEstimator(Protocol):
    """Pluggable direct-driver estimator over normalized all-driver telemetry."""

    def evaluate(
        self,
        telemetry: pd.DataFrame,
        *,
        group_columns: tuple[str, ...] | None = None,
    ) -> TrafficEvaluationResult: ...


@dataclass(frozen=True)
class FastF1StyleDirectTrafficEstimator:
    """FastF1-compatible finish-line-anchored direct traffic estimator.

    The numerical approach is adapted from FastF1 ``Telemetry.calculate_driver_ahead``
    (MIT licensed). Unlike that method, this implementation accepts normalized
    timestamped buffers and integrates each driver/lap once, so it can be shared by
    retrospective, archive-replay and live adapters without constructing a Session.
    """

    config: DirectTrafficEvaluationConfig = DirectTrafficEvaluationConfig()

    def evaluate(
        self,
        telemetry: pd.DataFrame,
        *,
        group_columns: tuple[str, ...] | None = None,
    ) -> TrafficEvaluationResult:
        return evaluate_direct_fastf1_style(
            telemetry,
            config=self.config,
            group_columns=group_columns,
        )


class TimeDeltaEstimator(Protocol):
    """Strategy object for adding a front-car time delta to telemetry."""

    output_column: str

    def add_time_delta(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        """Return telemetry with the estimator's output column added."""


@dataclass
class DistanceInterpolationTimeDeltaEstimator:
    """Estimate front-car time delta from distance gap and lap distance trace.

    For each telemetry row, this uses ``Distance + DistanceToDriverAhead`` as
    the target distance and interpolates over the current lap's ``Distance`` to
    ``Time`` trace to estimate when the current car reaches that target. Target
    distances beyond the end of the lap are wrapped and offset by the lap time.
    """

    distance_column: str = "Distance"
    distance_delta_column: str = "DistanceToDriverAhead"
    time_column: str = "Time"
    output_column: str = "TimeDeltaToDriverAhead"

    def add_time_delta(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        telemetry = telemetry.copy()
        telemetry[self.output_column] = self.estimate_seconds(telemetry)
        return telemetry

    def estimate_seconds(self, telemetry: pd.DataFrame) -> pd.Series:
        required_columns = {
            self.distance_column,
            self.distance_delta_column,
            self.time_column,
        }
        missing_columns = required_columns.difference(telemetry.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Telemetry is missing required columns: {missing}")

        result = pd.Series(np.nan, index=telemetry.index, dtype="float64")
        if telemetry.empty:
            return result

        distance = pd.to_numeric(telemetry[self.distance_column], errors="coerce")
        distance_delta = pd.to_numeric(telemetry[self.distance_delta_column], errors="coerce")
        time_seconds = self._time_as_seconds(telemetry[self.time_column])

        axis = (
            pd.DataFrame({"Distance": distance, "TimeSeconds": time_seconds})
            .dropna()
            .sort_values("Distance")
            .drop_duplicates(subset="Distance", keep="last")
        )
        if len(axis) < 2:
            return result

        distance_axis = axis["Distance"].to_numpy(dtype="float64")
        time_axis = axis["TimeSeconds"].to_numpy(dtype="float64")
        min_distance = distance_axis[0]
        max_distance = distance_axis[-1]
        lap_distance = max_distance - min_distance
        lap_time = time_axis[-1] - time_axis[0]
        if lap_distance <= 0 or lap_time <= 0:
            return result

        target_distance = distance + distance_delta

        valid = (
            distance.notna()
            & distance_delta.notna()
            & (distance_delta >= 0)
            & time_seconds.notna()
            & (target_distance >= min_distance)
        )
        if not valid.any():
            return result

        valid_target_distance = target_distance.loc[valid].to_numpy(dtype="float64")
        lap_offsets = np.floor((valid_target_distance - min_distance) / lap_distance)
        wrapped_target_distance = (
            (valid_target_distance - min_distance) % lap_distance
        ) + min_distance
        target_time = np.interp(
            wrapped_target_distance,
            distance_axis,
            time_axis,
        ) + (lap_offsets * lap_time)
        delta = target_time - time_seconds.loc[valid].to_numpy(dtype="float64")
        delta = np.where(delta >= 0, delta, np.nan)
        result.loc[valid] = delta
        return result

    def _time_as_seconds(self, values: pd.Series) -> pd.Series:
        if pd.api.types.is_timedelta64_dtype(values):
            return values.dt.total_seconds()

        if pd.api.types.is_numeric_dtype(values):
            return pd.to_numeric(values, errors="coerce")

        timedeltas = pd.to_timedelta(values, errors="coerce")
        return timedeltas.dt.total_seconds()


_GROUP_CANDIDATES = ("Year", "Round", "SessionName", "Driver", "LapNumber")
_METRICS = (
    "MinTimeDeltaToDriverAhead",
    "MeanTimeDeltaToDriverAhead",
    "MinDistanceToDriverAhead",
    "MeanDistanceToDriverAhead",
    "MinTimeDeltaToDriverBehind",
    "MeanTimeDeltaToDriverBehind",
    "MinDistanceToDriverBehind",
    "MeanDistanceToDriverBehind",
)


def evaluate_physical_traffic(
    telemetry: pd.DataFrame,
    *,
    config: TrafficEvaluationConfig,
    group_columns: tuple[str, ...] | None = None,
) -> TrafficEvaluationResult:
    """Evaluate nearest physical cars from original timestamped speed samples.

    Official lap numbers constrain per-lap integration and anchoring. Proximity
    is calculated from circular circuit phase, so classification lap differences
    never add whole circuit lengths to the physical gap.
    """
    _validate_config(config)
    missing = set(NORMALIZED_REQUIRED_COLUMNS).difference(telemetry.columns)
    if missing:
        raise ValueError(f"Telemetry is missing required columns: {sorted(missing)}")
    groups = group_columns or tuple(c for c in _GROUP_CANDIDATES if c in telemetry.columns)
    if "Driver" not in groups or "LapNumber" not in groups:
        raise ValueError("group_columns must include Driver and LapNumber")
    prepared = _prepare_progress(telemetry, config=config)
    if prepared.empty:
        return TrafficEvaluationResult(_empty_samples(groups), _empty_laps(groups))

    sample_frames: list[pd.DataFrame] = []
    session_columns = [c for c in ("Year", "Round", "SessionName") if c in prepared]
    sessions = (
        prepared.groupby(session_columns, dropna=False, sort=False)
        if session_columns
        else [((), prepared)]
    )
    for _, session in sessions:
        sample_frames.extend(_evaluate_session(session, config=config))
    if not sample_frames:
        return TrafficEvaluationResult(
            _empty_samples(groups), _unknown_laps(prepared, groups, config)
        )

    samples = pd.concat(sample_frames, ignore_index=True)
    laps = _summarize_physical_samples(samples, groups=groups, config=config)
    return TrafficEvaluationResult(samples=samples, laps=laps)


def evaluate_direct_fastf1_style(
    telemetry: pd.DataFrame,
    *,
    config: DirectTrafficEvaluationConfig | None = None,
    group_columns: tuple[str, ...] | None = None,
) -> TrafficEvaluationResult:
    """Locate the directly-ahead driver from normalized speed/lap buffers.

    Speed is integrated once per driver/lap. For each target lap, other-driver
    progress is finish-line anchored using FastF1's relevant-lap selection rule,
    interpolated once onto the target timestamps, and compared in one NumPy matrix.
    The direct component intentionally retains classification-lap anchoring; circular
    lapped-car proximity remains the responsibility of ``evaluate_physical_traffic``.
    """
    config = config or DirectTrafficEvaluationConfig()
    _validate_direct_config(config)
    missing = set(NORMALIZED_REQUIRED_COLUMNS).difference(telemetry.columns)
    if missing:
        raise ValueError(f"Telemetry is missing required columns: {sorted(missing)}")
    groups = group_columns or tuple(c for c in _GROUP_CANDIDATES if c in telemetry.columns)
    if "Driver" not in groups or "LapNumber" not in groups:
        raise ValueError("group_columns must include Driver and LapNumber")

    prepared = _prepare_direct_progress(telemetry)
    if prepared.empty:
        return TrafficEvaluationResult(_empty_direct_samples(groups), _empty_laps(groups))

    session_columns = [c for c in ("Year", "Round", "SessionName") if c in prepared]
    sessions = (
        prepared.groupby(session_columns, dropna=False, sort=False)
        if session_columns
        else [((), prepared)]
    )
    sample_frames: list[pd.DataFrame] = []
    for _, session in sessions:
        sample_frames.extend(_evaluate_direct_session(session, config=config, groups=groups))
    if not sample_frames:
        return TrafficEvaluationResult(
            _empty_direct_samples(groups), _unknown_direct_laps(prepared, groups, config)
        )
    samples = pd.concat(sample_frames, ignore_index=True)
    laps = _summarize_direct_samples(samples, groups=groups, config=config)
    return TrafficEvaluationResult(samples=samples, laps=laps)


def _prepare_direct_progress(telemetry: pd.DataFrame) -> pd.DataFrame:
    output = telemetry.copy().reset_index(drop=True)
    output["_SessionSeconds"] = _seconds(output["SessionTime"])
    output["_SpeedKph"] = pd.to_numeric(output["Speed"], errors="coerce")
    output["LapNumber"] = pd.to_numeric(output["LapNumber"], errors="coerce")
    output = output.dropna(subset=["Driver", "LapNumber", "_SessionSeconds", "_SpeedKph"])
    output = output.loc[output["_SpeedKph"] >= 0].copy()
    if output.empty:
        return output
    session_columns = [c for c in ("Year", "Round", "SessionName") if c in output]
    lap_groups = [*session_columns, "Driver", "LapNumber"]
    driver_groups = [*session_columns, "Driver"]
    output = output.sort_values([*driver_groups, "_SessionSeconds"], kind="stable")
    output = output.drop_duplicates([*driver_groups, "_SessionSeconds"], keep="last")
    driver_delta = (
        output.groupby(driver_groups, dropna=False)["_SessionSeconds"].diff().fillna(0.0)
    )
    output["_DirectDriverDistance"] = (
        output["_SpeedKph"].div(3.6).mul(driver_delta.clip(lower=0.0))
    ).groupby([output[c] for c in driver_groups], dropna=False).cumsum()
    output = output.sort_values([*lap_groups, "_SessionSeconds"], kind="stable")
    output = output.drop_duplicates([*lap_groups, "_SessionSeconds"], keep="last")
    lap_anchor = output.groupby(lap_groups, dropna=False)[
        "_DirectDriverDistance"
    ].transform("first")
    output["_DirectLapDistance"] = output["_DirectDriverDistance"] - lap_anchor
    return output


def _evaluate_direct_session(
    session: pd.DataFrame,
    *,
    config: DirectTrafficEvaluationConfig,
    groups: tuple[str, ...],
) -> list[pd.DataFrame]:
    driver_laps: dict[str, dict[float, pd.DataFrame]] = {}
    for (driver, lap_number), lap in session.groupby(
        ["Driver", "LapNumber"], dropna=False, sort=False
    ):
        lap = lap.sort_values("_SessionSeconds", kind="stable").copy()
        if len(lap) < config.minimum_lap_samples:
            continue
        driver_laps.setdefault(str(driver), {})[float(lap_number)] = lap
    driver_metadata = {
        driver: _direct_lap_metadata(laps)
        for driver, laps in driver_laps.items()
    }
    # A trace starting at a given driver/lap is independent of the target car.
    # Cache it so a race evaluates each suffix once instead of rebuilding the
    # same pandas objects for every target driver and target lap.
    trace_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray] | None] = {}
    frames: list[pd.DataFrame] = []
    for target_driver, laps in driver_laps.items():
        for target_lap_number, target in laps.items():
            times = target["_SessionSeconds"].to_numpy(dtype="float64")
            own_distance = target["_DirectLapDistance"].to_numpy(dtype="float64")
            if len(times) < config.minimum_lap_samples:
                continue
            other_drivers: list[str] = []
            candidates: list[np.ndarray] = []
            for other_driver, other_laps in driver_laps.items():
                if other_driver == target_driver:
                    continue
                first_lap_number = _direct_trace_start_lap(
                    driver_metadata[other_driver],
                    target_lap_number=target_lap_number,
                    target_start=float(times[0]),
                )
                if first_lap_number is None:
                    continue
                cache_key = (other_driver, first_lap_number)
                if cache_key not in trace_cache:
                    trace_cache[cache_key] = _build_direct_trace(
                        other_laps,
                        first_lap_number=first_lap_number,
                    )
                trace = trace_cache[cache_key]
                if trace is None:
                    continue
                other_time, other_distance = trace
                interpolated = _interpolate_direct_trace(
                    other_time,
                    other_distance,
                    times,
                    config=config,
                )
                candidates.append(interpolated - own_distance)
                other_drivers.append(other_driver)
            frame = target.loc[:, [c for c in groups if c in target]].copy()
            frame["SessionTime"] = target["SessionTime"].to_numpy()
            frame["DirectDriverAhead"] = pd.NA
            frame["DirectDistanceToDriverAhead"] = np.nan
            if candidates:
                delta = np.column_stack(candidates)
                delta[~np.isfinite(delta) | (delta < 0.0)] = np.inf
                nearest = np.argmin(delta, axis=1)
                nearest_distance = delta[np.arange(len(delta)), nearest]
                valid = np.isfinite(nearest_distance)
                driver_values = np.full(len(delta), None, dtype=object)
                driver_values[valid] = np.asarray(other_drivers, dtype=object)[nearest[valid]]
                frame["DirectDriverAhead"] = driver_values
                frame["DirectDistanceToDriverAhead"] = np.where(
                    valid, nearest_distance, np.nan
                )
            estimator_input = pd.DataFrame(
                {
                    "Distance": own_distance,
                    "DistanceToDriverAhead": frame["DirectDistanceToDriverAhead"],
                    "Time": times - times[0],
                },
                index=frame.index,
            )
            frame["DirectTimeDeltaToDriverAhead"] = (
                DistanceInterpolationTimeDeltaEstimator().estimate_seconds(estimator_input)
            )
            frames.append(frame.reset_index(drop=True))
    return frames


def _direct_lap_metadata(
    laps: dict[float, pd.DataFrame],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(sorted(
        (
            lap_number,
            float(lap["_SessionSeconds"].iloc[0]),
            float(lap["_SessionSeconds"].iloc[-1]),
        )
        for lap_number, lap in laps.items()
    ))


def _direct_trace_start_lap(
    metadata: tuple[tuple[float, float, float], ...],
    *,
    target_lap_number: float,
    target_start: float,
) -> float | None:
    if not metadata:
        return None
    before = [item for item in metadata if item[1] <= target_start]
    selected_number = before[-1][0] if before else metadata[0][0]
    if selected_number < target_lap_number:
        selected_number += 1.0
    selected_numbers = [
        lap_number
        for lap_number, _, end in metadata
        if lap_number >= selected_number and end >= target_start
    ]
    return selected_numbers[0] if selected_numbers else None


def _build_direct_trace(
    laps: dict[float, pd.DataFrame],
    *,
    first_lap_number: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    selected_numbers = sorted(
        lap_number for lap_number in laps if lap_number >= first_lap_number
    )
    time_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    for lap_number in selected_numbers:
        lap = laps[lap_number]
        lap_times = lap["_SessionSeconds"].to_numpy(dtype="float64")
        lap_distance = lap["_DirectDriverDistance"].to_numpy(dtype="float64")
        time_parts.append(lap_times)
        distance_parts.append(lap_distance)
    if not time_parts:
        return None
    times = np.concatenate(time_parts)
    distances = np.concatenate(distance_parts)
    order = np.argsort(times, kind="stable")
    times = times[order]
    distances = distances[order]
    # Match pandas drop_duplicates(..., keep="last") without constructing a
    # temporary DataFrame for every driver/lap suffix.
    keep = np.r_[times[1:] != times[:-1], True]
    times = times[keep]
    distances = distances[keep]
    if len(times) < 2:
        return None
    return times, distances - distances[0]


def _interpolate_direct_trace(
    source_time: np.ndarray,
    source_distance: np.ndarray,
    target_time: np.ndarray,
    *,
    config: DirectTrafficEvaluationConfig,
) -> np.ndarray:
    right = np.searchsorted(source_time, target_time, side="left")
    exact = (right < len(source_time)) & np.isclose(
        source_time[np.clip(right, 0, len(source_time) - 1)], target_time
    )
    left = right - 1
    valid = exact.copy()
    bracketed = (~exact) & (left >= 0) & (right < len(source_time))
    if bracketed.any():
        span = source_time[right[bracketed]] - source_time[left[bracketed]]
        nearest_age = np.minimum(
            target_time[bracketed] - source_time[left[bracketed]],
            source_time[right[bracketed]] - target_time[bracketed],
        )
        valid[bracketed] = (
            (span <= config.maximum_bracketing_interval_seconds)
            & (nearest_age <= config.interpolation_tolerance_seconds)
        )
    result = np.full(len(target_time), np.nan, dtype="float64")
    if exact.any():
        result[exact] = source_distance[right[exact]]
    interpolate = valid & ~exact
    if interpolate.any():
        li = left[interpolate]
        ri = right[interpolate]
        weight = (target_time[interpolate] - source_time[li]) / (
            source_time[ri] - source_time[li]
        )
        result[interpolate] = source_distance[li] + weight * (
            source_distance[ri] - source_distance[li]
        )
    return result


def _summarize_direct_samples(
    samples: pd.DataFrame,
    *,
    groups: tuple[str, ...],
    config: DirectTrafficEvaluationConfig,
) -> pd.DataFrame:
    summary = samples.groupby(list(groups), dropna=False, as_index=False).agg(
        MinTimeDeltaToDriverAhead=("DirectTimeDeltaToDriverAhead", "min"),
        MeanTimeDeltaToDriverAhead=("DirectTimeDeltaToDriverAhead", "mean"),
        MinDistanceToDriverAhead=("DirectDistanceToDriverAhead", "min"),
        MeanDistanceToDriverAhead=("DirectDistanceToDriverAhead", "mean"),
        DirectSampleCount=("SessionTime", "size"),
        DirectMatchedSampleCount=("DirectDistanceToDriverAhead", "count"),
    )
    behind = samples.dropna(
        subset=["DirectDriverAhead", "DirectTimeDeltaToDriverAhead"]
    ).copy()
    if not behind.empty:
        behind["Driver"] = behind["DirectDriverAhead"].astype(str)
        behind = behind.groupby(list(groups), dropna=False, as_index=False).agg(
            MinTimeDeltaToDriverBehind=("DirectTimeDeltaToDriverAhead", "min"),
            MeanTimeDeltaToDriverBehind=("DirectTimeDeltaToDriverAhead", "mean"),
            MinDistanceToDriverBehind=("DirectDistanceToDriverAhead", "min"),
            MeanDistanceToDriverBehind=("DirectDistanceToDriverAhead", "mean"),
        )
        summary = summary.merge(behind, on=list(groups), how="left")
    else:
        for metric in _METRICS[4:]:
            summary[metric] = np.nan
    summary["DirectCoverageFraction"] = np.divide(
        summary["DirectMatchedSampleCount"],
        summary["DirectSampleCount"],
        out=np.zeros(len(summary), dtype="float64"),
        where=summary["DirectSampleCount"].to_numpy() > 0,
    )
    summary["DirectTrafficStatus"] = np.where(
        summary["DirectCoverageFraction"] >= config.minimum_coverage_fraction,
        TRAFFIC_STATUS_OK,
        TRAFFIC_STATUS_PARTIAL,
    )
    summary["DirectTrafficStatusReason"] = np.where(
        summary["DirectTrafficStatus"].eq(TRAFFIC_STATUS_OK),
        "direct synchronized coverage passed",
        "direct synchronized coverage below configured minimum",
    )
    summary["TrafficEvaluatorVersion"] = config.evaluator_version
    return summary


def _unknown_direct_laps(
    prepared: pd.DataFrame,
    groups: tuple[str, ...],
    config: DirectTrafficEvaluationConfig,
) -> pd.DataFrame:
    output = prepared.loc[:, list(groups)].drop_duplicates().copy()
    for metric in _METRICS:
        output[metric] = np.nan
    output["DirectCoverageFraction"] = 0.0
    output["DirectTrafficStatus"] = TRAFFIC_STATUS_UNKNOWN
    output["DirectTrafficStatusReason"] = "no synchronized direct telemetry"
    output["TrafficEvaluatorVersion"] = config.evaluator_version
    return output


def _empty_direct_samples(groups: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *groups,
            "SessionTime",
            "DirectDriverAhead",
            "DirectDistanceToDriverAhead",
            "DirectTimeDeltaToDriverAhead",
        ]
    )


def _validate_direct_config(config: DirectTrafficEvaluationConfig) -> None:
    if config.interpolation_tolerance_seconds < 0:
        raise ValueError("interpolation_tolerance_seconds cannot be negative")
    if config.maximum_bracketing_interval_seconds <= 0:
        raise ValueError("maximum_bracketing_interval_seconds must be positive")
    if not 0 <= config.minimum_coverage_fraction <= 1:
        raise ValueError("minimum_coverage_fraction must be between zero and one")
    if config.minimum_lap_samples < 2:
        raise ValueError("minimum_lap_samples must be at least two")


def _prepare_progress(
    telemetry: pd.DataFrame,
    *,
    config: TrafficEvaluationConfig,
) -> pd.DataFrame:
    output = telemetry.copy().reset_index(drop=True)
    output["_SessionSeconds"] = _seconds(output["SessionTime"])
    output["_SpeedKph"] = pd.to_numeric(output["Speed"], errors="coerce")
    output["LapNumber"] = pd.to_numeric(output["LapNumber"], errors="coerce")
    output = output.dropna(subset=["Driver", "LapNumber", "_SessionSeconds", "_SpeedKph"])
    output = output.loc[output["_SpeedKph"] >= 0].copy()
    if output.empty:
        return output

    session_columns = [c for c in ("Year", "Round", "SessionName") if c in output]
    lap_groups = [*session_columns, "Driver", "LapNumber"]
    output = output.sort_values([*lap_groups, "_SessionSeconds"], kind="stable")
    output["_DeltaSeconds"] = (
        output.groupby(lap_groups, dropna=False)["_SessionSeconds"].diff().fillna(0.0)
    )
    output["_IntegratedLapDistance"] = (
        (output["_SpeedKph"] / 3.6) * output["_DeltaSeconds"].clip(lower=0.0)
    ).groupby([output[c] for c in lap_groups], dropna=False).cumsum()

    lap_span = output.groupby(lap_groups, dropna=False)["_IntegratedLapDistance"].transform("max")
    lap_count = output.groupby(lap_groups, dropna=False)["_IntegratedLapDistance"].transform("size")
    complete = (
        lap_count.ge(config.minimum_lap_samples)
        & lap_span.gt(config.circuit_length_m * 0.5)
        & lap_span.lt(config.circuit_length_m * 1.5)
    )
    scale = pd.Series(1.0, index=output.index, dtype="float64")
    scale.loc[complete] = config.circuit_length_m / lap_span.loc[complete]
    if "Distance" in output.columns:
        supplied_distance = pd.to_numeric(output["Distance"], errors="coerce")
        first_supplied = supplied_distance.groupby(
            [output[c] for c in lap_groups], dropna=False
        ).transform("first")
        phase_anchor = first_supplied.fillna(0.0)
    else:
        phase_anchor = pd.Series(0.0, index=output.index, dtype="float64")
    output["_LapProgress"] = phase_anchor + output["_IntegratedLapDistance"] * scale
    output["_CircuitPhase"] = output["_LapProgress"] % config.circuit_length_m
    # Unwrapped progress is used only to interpolate continuously across a lap
    # boundary. It is reduced back to circuit phase before physical comparison.
    output["_UnwrappedProgress"] = (
        output["LapNumber"].astype("float64") * config.circuit_length_m
        + output["_LapProgress"]
    )
    output["_ProgressQuality"] = np.where(complete, "normalized_complete_lap", "partial_lap")
    return output


def _evaluate_session(
    session: pd.DataFrame,
    *,
    config: TrafficEvaluationConfig,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    by_driver = {
        str(driver): data.sort_values("_SessionSeconds", kind="stable")
        for driver, data in session.groupby("Driver", dropna=False, sort=False)
    }
    for target_driver, target in by_driver.items():
        target = target.copy()
        times = target["_SessionSeconds"].to_numpy(dtype="float64")
        target_phase = target["_CircuitPhase"].to_numpy(dtype="float64")
        target_speed = target["_SpeedKph"].to_numpy(dtype="float64")
        candidates: list[pd.DataFrame] = []
        for other_driver, other in by_driver.items():
            if other_driver == target_driver:
                continue
            interpolation = _interpolate_driver(other, times, config=config)
            valid = interpolation["valid"]
            if not valid.any():
                continue
            other_phase = interpolation["progress"] % config.circuit_length_m
            ahead_distance = (other_phase - target_phase) % config.circuit_length_m
            behind_distance = (target_phase - other_phase) % config.circuit_length_m
            candidate = pd.DataFrame(
                {
                    "_TargetRowId": target.index.to_numpy(),
                    "_OtherDriver": other_driver,
                    "_Valid": valid,
                    "_InterpolationAge": interpolation["age"],
                    "_OtherSpeedKph": interpolation["speed"],
                    "_AheadDistance": ahead_distance,
                    "_BehindDistance": behind_distance,
                }
            )
            candidate.loc[candidate["_AheadDistance"] <= config.minimum_gap_m, "_Valid"] = False
            candidate.loc[candidate["_BehindDistance"] <= config.minimum_gap_m, "_Valid"] = False
            candidates.append(candidate)
        frames.append(_select_nearest(target, candidates, target_speed=target_speed, config=config))
    return frames


def _interpolate_driver(
    other: pd.DataFrame,
    target_times: np.ndarray,
    *,
    config: TrafficEvaluationConfig,
) -> dict[str, np.ndarray]:
    source = (
        other.loc[:, ["_SessionSeconds", "_UnwrappedProgress", "_SpeedKph"]]
        .drop_duplicates("_SessionSeconds", keep="last")
        .sort_values("_SessionSeconds", kind="stable")
    )
    source_times = source["_SessionSeconds"].to_numpy(dtype="float64")
    empty = np.full(len(target_times), np.nan, dtype="float64")
    if len(source_times) == 1:
        age = np.abs(target_times - source_times[0])
        valid = age <= config.interpolation_tolerance_seconds
        progress = np.full(len(target_times), source["_UnwrappedProgress"].iloc[0])
        speed = np.full(len(target_times), source["_SpeedKph"].iloc[0])
        progress[~valid] = np.nan
        speed[~valid] = np.nan
        age[~valid] = np.nan
        return {"progress": progress, "speed": speed, "age": age, "valid": valid}
    if len(source_times) < 2:
        return {
            "progress": empty,
            "speed": empty.copy(),
            "age": empty.copy(),
            "valid": np.zeros(len(target_times), dtype=bool),
        }
    right = np.searchsorted(source_times, target_times, side="left")
    right_clipped = np.clip(right, 1, len(source_times) - 1)
    left = right_clipped - 1
    left_time = source_times[left]
    right_time = source_times[right_clipped]
    span = right_time - left_time
    nearest_age = np.minimum(np.abs(target_times - left_time), np.abs(right_time - target_times))
    valid = (
        (target_times >= source_times[0])
        & (target_times <= source_times[-1])
        & (span > 0)
        & (span <= config.maximum_bracketing_interval_seconds)
        & (nearest_age <= config.interpolation_tolerance_seconds)
    )
    weight = np.divide(
        target_times - left_time,
        span,
        out=np.zeros_like(target_times, dtype="float64"),
        where=span > 0,
    )
    progress_values = source["_UnwrappedProgress"].to_numpy(dtype="float64")
    speed_values = source["_SpeedKph"].to_numpy(dtype="float64")
    progress = progress_values[left] + weight * (
        progress_values[right_clipped] - progress_values[left]
    )
    speed = speed_values[left] + weight * (speed_values[right_clipped] - speed_values[left])
    progress[~valid] = np.nan
    speed[~valid] = np.nan
    nearest_age[~valid] = np.nan
    return {"progress": progress, "speed": speed, "age": nearest_age, "valid": valid}


def _select_nearest(
    target: pd.DataFrame,
    candidates: list[pd.DataFrame],
    *,
    target_speed: np.ndarray,
    config: TrafficEvaluationConfig,
) -> pd.DataFrame:
    result_columns = [c for c in target.columns if not c.startswith("_")]
    result = target.loc[:, result_columns].copy()
    result["PhysicalDriverAhead"] = pd.NA
    result["PhysicalDistanceToDriverAhead"] = np.nan
    result["PhysicalTimeDeltaToDriverAhead"] = np.nan
    result["PhysicalDriverBehind"] = pd.NA
    result["PhysicalDistanceToDriverBehind"] = np.nan
    result["PhysicalTimeDeltaToDriverBehind"] = np.nan
    result["InterpolationAgeSeconds"] = np.nan
    if not candidates:
        return result
    all_candidates = pd.concat(candidates, ignore_index=True)
    all_candidates = all_candidates.loc[all_candidates["_Valid"]].copy()
    if all_candidates.empty:
        return result
    target_speed_by_id = pd.Series(target_speed, index=target.index)
    ahead = all_candidates.sort_values(
        ["_TargetRowId", "_AheadDistance"], kind="stable"
    ).drop_duplicates("_TargetRowId").set_index("_TargetRowId")
    behind = all_candidates.sort_values(
        ["_TargetRowId", "_BehindDistance"], kind="stable"
    ).drop_duplicates("_TargetRowId").set_index("_TargetRowId")
    ahead = ahead.reindex(result.index)
    behind = behind.reindex(result.index)

    ahead_distance = ahead["_AheadDistance"].to_numpy(dtype="float64")
    ahead_speed = target_speed_by_id.reindex(result.index).to_numpy(dtype="float64")
    result["PhysicalDriverAhead"] = ahead["_OtherDriver"].to_numpy()
    result["PhysicalDistanceToDriverAhead"] = ahead_distance
    result["PhysicalTimeDeltaToDriverAhead"] = np.divide(
        ahead_distance,
        ahead_speed / 3.6,
        out=np.full(len(result), np.nan, dtype="float64"),
        where=ahead_speed > 1.0,
    )

    behind_distance = behind["_BehindDistance"].to_numpy(dtype="float64")
    behind_speed = behind["_OtherSpeedKph"].to_numpy(dtype="float64")
    result["PhysicalDriverBehind"] = behind["_OtherDriver"].to_numpy()
    result["PhysicalDistanceToDriverBehind"] = behind_distance
    result["PhysicalTimeDeltaToDriverBehind"] = np.divide(
        behind_distance,
        behind_speed / 3.6,
        out=np.full(len(result), np.nan, dtype="float64"),
        where=behind_speed > 1.0,
    )
    result["InterpolationAgeSeconds"] = np.fmax(
        ahead["_InterpolationAge"].to_numpy(dtype="float64"),
        behind["_InterpolationAge"].to_numpy(dtype="float64"),
    )
    return result


def _summarize_physical_samples(
    samples: pd.DataFrame,
    *,
    groups: tuple[str, ...],
    config: TrafficEvaluationConfig,
) -> pd.DataFrame:
    grouped = samples.groupby(list(groups), dropna=False, as_index=False)
    summary = grouped.agg(
        PhysicalMinTimeDeltaToDriverAhead=("PhysicalTimeDeltaToDriverAhead", "min"),
        PhysicalMeanTimeDeltaToDriverAhead=("PhysicalTimeDeltaToDriverAhead", "mean"),
        PhysicalMinDistanceToDriverAhead=("PhysicalDistanceToDriverAhead", "min"),
        PhysicalMeanDistanceToDriverAhead=("PhysicalDistanceToDriverAhead", "mean"),
        PhysicalMinTimeDeltaToDriverBehind=("PhysicalTimeDeltaToDriverBehind", "min"),
        PhysicalMeanTimeDeltaToDriverBehind=("PhysicalTimeDeltaToDriverBehind", "mean"),
        PhysicalMinDistanceToDriverBehind=("PhysicalDistanceToDriverBehind", "min"),
        PhysicalMeanDistanceToDriverBehind=("PhysicalDistanceToDriverBehind", "mean"),
        TargetSampleCount=("SessionTime", "size"),
        AheadMatchedSampleCount=("PhysicalTimeDeltaToDriverAhead", "count"),
        BehindMatchedSampleCount=("PhysicalTimeDeltaToDriverBehind", "count"),
        MaximumInterpolationAgeSeconds=("InterpolationAgeSeconds", "max"),
    )
    summary["AheadCoverageFraction"] = (
        summary["AheadMatchedSampleCount"] / summary["TargetSampleCount"]
    )
    summary["BehindCoverageFraction"] = (
        summary["BehindMatchedSampleCount"] / summary["TargetSampleCount"]
    )
    summary["CoverageFraction"] = summary[
        ["AheadCoverageFraction", "BehindCoverageFraction"]
    ].min(axis=1)
    summary["TrafficMethod"] = PHYSICAL_CIRCULAR
    summary["TrafficStatus"] = np.select(
        [
            summary["CoverageFraction"] >= config.minimum_coverage_fraction,
            summary["CoverageFraction"] > 0,
        ],
        [TRAFFIC_STATUS_OK, TRAFFIC_STATUS_PARTIAL],
        default=TRAFFIC_STATUS_UNKNOWN,
    )
    summary["TrafficStatusReason"] = np.select(
        [
            summary["TrafficStatus"].eq(TRAFFIC_STATUS_OK),
            summary["TrafficStatus"].eq(TRAFFIC_STATUS_PARTIAL),
        ],
        ["telemetry coverage passed", "telemetry coverage below configured minimum"],
        default="no synchronized other-driver telemetry",
    )
    summary["TrafficEvaluatorVersion"] = config.evaluator_version
    return summary


def combine_lap_traffic_components(
    direct: pd.DataFrame,
    physical: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    combination_mode: str = COMBINATION_MODE_COMPATIBILITY_MIN,
) -> pd.DataFrame:
    if combination_mode != COMBINATION_MODE_COMPATIBILITY_MIN:
        raise ValueError(f"Unsupported traffic combination mode: {combination_mode}")
    direct = direct.copy()
    physical = physical.copy()
    direct_rename = {metric: f"Direct{metric}" for metric in _METRICS if metric in direct}
    direct = direct.rename(columns=direct_rename)
    physical = physical.rename(
        columns={
            "CoverageFraction": "PhysicalCoverageFraction",
            "TrafficMethod": "PhysicalTrafficMethod",
            "TrafficStatus": "PhysicalTrafficStatus",
            "TrafficStatusReason": "PhysicalTrafficStatusReason",
        }
    )
    merged = direct.merge(physical, on=list(group_columns), how="outer")
    for metric in _METRICS:
        direct_column = f"Direct{metric}"
        physical_column = f"Physical{metric}"
        if direct_column not in merged:
            merged[direct_column] = np.nan
        if physical_column not in merged:
            merged[physical_column] = np.nan
        merged[direct_column] = pd.to_numeric(merged[direct_column], errors="coerce")
        merged[physical_column] = pd.to_numeric(merged[physical_column], errors="coerce")
        values = merged[[direct_column, physical_column]]
        merged[metric] = values.min(axis=1, skipna=True)
        merged[f"{metric}Method"] = np.select(
            [
                values[direct_column].notna()
                & values[physical_column].notna()
                & values[direct_column].le(values[physical_column]),
                values[physical_column].notna(),
                values[direct_column].notna(),
            ],
            [DIRECT_FASTF1_STYLE, PHYSICAL_CIRCULAR, DIRECT_FASTF1_STYLE],
            default=pd.NA,
        )
    merged["TrafficCombinationMode"] = combination_mode
    has_physical = merged[[f"Physical{metric}" for metric in _METRICS]].notna().any(
        axis=1
    )
    has_direct = merged[[f"Direct{metric}" for metric in _METRICS]].notna().any(axis=1)
    if "DirectTrafficStatus" not in merged:
        merged["DirectTrafficStatus"] = np.where(
            has_direct, TRAFFIC_STATUS_OK, TRAFFIC_STATUS_UNKNOWN
        )
    else:
        merged["DirectTrafficStatus"] = merged["DirectTrafficStatus"].fillna(
            TRAFFIC_STATUS_UNKNOWN
        )
    if "DirectCoverageFraction" not in merged:
        merged["DirectCoverageFraction"] = has_direct.astype("float64")
    merged["DirectCoverageFraction"] = pd.to_numeric(
        merged["DirectCoverageFraction"], errors="coerce"
    ).fillna(0.0)
    merged["TrafficMethod"] = np.select(
        [has_physical & has_direct, has_physical, has_direct],
        [combination_mode, PHYSICAL_CIRCULAR, DIRECT_FASTF1_STYLE],
        default=pd.NA,
    )
    has_selected = merged[list(_METRICS)].notna().any(axis=1)
    if "PhysicalTrafficStatus" not in merged:
        merged["PhysicalTrafficStatus"] = TRAFFIC_STATUS_UNKNOWN
    if "PhysicalTrafficStatusReason" not in merged:
        merged["PhysicalTrafficStatusReason"] = "no physical traffic telemetry available"
    if "PhysicalCoverageFraction" not in merged:
        merged["PhysicalCoverageFraction"] = 0.0
    merged["PhysicalCoverageFraction"] = pd.to_numeric(
        merged["PhysicalCoverageFraction"], errors="coerce"
    ).fillna(0.0)
    direct_ok = merged["DirectTrafficStatus"].eq(TRAFFIC_STATUS_OK)
    physical_ok = merged["PhysicalTrafficStatus"].eq(TRAFFIC_STATUS_OK)
    merged["CoverageFraction"] = merged[
        ["DirectCoverageFraction", "PhysicalCoverageFraction"]
    ].max(axis=1)
    merged["TrafficStatus"] = np.select(
        [direct_ok | physical_ok, has_selected],
        [TRAFFIC_STATUS_OK, TRAFFIC_STATUS_PARTIAL],
        default=TRAFFIC_STATUS_UNKNOWN,
    )
    merged["TrafficStatusReason"] = np.select(
        [
            direct_ok & physical_ok,
            direct_ok,
            physical_ok,
            has_selected,
        ],
        [
            "direct and physical telemetry coverage passed",
            "direct FastF1-style telemetry coverage passed",
            "physical circular telemetry coverage passed",
            "traffic telemetry coverage below configured minimum",
        ],
        default="no synchronized other-driver telemetry",
    )
    if "TrafficEvaluatorVersion" not in merged.columns:
        merged["TrafficEvaluatorVersion"] = TRAFFIC_EVALUATOR_VERSION
    else:
        merged["TrafficEvaluatorVersion"] = merged[
            "TrafficEvaluatorVersion"
        ].fillna(TRAFFIC_EVALUATOR_VERSION)
    return merged


def _unknown_laps(
    prepared: pd.DataFrame,
    groups: tuple[str, ...],
    config: TrafficEvaluationConfig,
) -> pd.DataFrame:
    laps = prepared.loc[:, list(groups)].drop_duplicates().copy()
    laps["CoverageFraction"] = 0.0
    laps["TrafficMethod"] = PHYSICAL_CIRCULAR
    laps["TrafficStatus"] = TRAFFIC_STATUS_UNKNOWN
    laps["TrafficStatusReason"] = "no synchronized other-driver telemetry"
    laps["TrafficEvaluatorVersion"] = config.evaluator_version
    return laps


def _empty_samples(groups: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[*groups, "SessionTime", "PhysicalDriverAhead", "PhysicalDriverBehind"]
    )


def _empty_laps(groups: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *groups,
            "CoverageFraction",
            "TrafficMethod",
            "TrafficStatus",
            "TrafficStatusReason",
            "TrafficEvaluatorVersion",
        ]
    )


def _seconds(values: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(values):
        return values.dt.total_seconds()
    return pd.to_numeric(values, errors="coerce")


def _validate_config(config: TrafficEvaluationConfig) -> None:
    if not np.isfinite(config.circuit_length_m) or config.circuit_length_m <= 0:
        raise ValueError("circuit_length_m must be positive")
    if config.interpolation_tolerance_seconds < 0:
        raise ValueError("interpolation_tolerance_seconds must be non-negative")
    if config.maximum_bracketing_interval_seconds <= 0:
        raise ValueError("maximum_bracketing_interval_seconds must be positive")
    if not 0 <= config.minimum_coverage_fraction <= 1:
        raise ValueError("minimum_coverage_fraction must be between zero and one")


def traffic_clean_mask(
    laps: pd.DataFrame,
    *,
    minimum_ahead_seconds: float,
    minimum_behind_seconds: float | None,
    ahead_column: str = "MeanTimeDeltaToDriverAhead",
    behind_column: str = "MeanTimeDeltaToDriverBehind",
    allow_missing_side_with_other_evidence: bool = True,
) -> pd.Series:
    """Return the shared lap-level traffic decision without run grouping."""
    required = {ahead_column}
    if minimum_behind_seconds is not None:
        required.add(behind_column)
    missing = required.difference(laps.columns)
    if missing:
        raise ValueError(f"Laps are missing required traffic columns: {sorted(missing)}")
    has_evidence = _has_traffic_evidence(laps)
    ahead = pd.to_numeric(laps[ahead_column], errors="coerce")
    if allow_missing_side_with_other_evidence:
        ahead = ahead.where(ahead.notna(), np.where(has_evidence, np.inf, np.nan))
    result = ahead > float(minimum_ahead_seconds)
    if minimum_behind_seconds is not None:
        behind = pd.to_numeric(laps[behind_column], errors="coerce")
        if allow_missing_side_with_other_evidence:
            behind = behind.where(
                behind.notna(), np.where(has_evidence, np.inf, np.nan)
            )
        result = result & (behind > float(minimum_behind_seconds))
    if "TrafficStatus" in laps.columns:
        result = result & laps["TrafficStatus"].isin(["ok", "sector_fallback"])
    return result.fillna(False).astype(bool)


def _has_traffic_evidence(laps: pd.DataFrame) -> pd.Series:
    columns = [
        column
        for column in laps.columns
        if column.startswith(
            (
                "MinTimeDeltaToDriver",
                "MeanTimeDeltaToDriver",
                "MinDistanceToDriver",
                "MeanDistanceToDriver",
            )
        )
    ]
    if not columns:
        return pd.Series(False, index=laps.index)
    return laps.loc[:, columns].notna().any(axis=1)


def summarize_lap_gap_metrics(
    telemetry: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("Year", "Round", "SessionName", "Driver", "LapNumber"),
    time_delta_column: str = "TimeDeltaToDriverAhead",
    distance_delta_column: str = "DistanceToDriverAhead",
) -> pd.DataFrame:
    """Aggregate full telemetry into per-lap front-car gap metrics."""
    if telemetry.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "MinTimeDeltaToDriverAhead",
                "MeanTimeDeltaToDriverAhead",
                "MinDistanceToDriverAhead",
                "MeanDistanceToDriverAhead",
                "MinTimeDeltaToDriverBehind",
                "MeanTimeDeltaToDriverBehind",
                "MinDistanceToDriverBehind",
                "MeanDistanceToDriverBehind",
            ]
        )

    legacy_direct_summary = pd.DataFrame()
    direct_required = {*group_columns, time_delta_column, distance_delta_column}
    if not direct_required.difference(telemetry.columns):
        legacy_direct_summary = _summarize_lap_gaps_from_driver_ahead(
            telemetry,
            group_columns=group_columns,
            time_delta_column=time_delta_column,
            distance_delta_column=distance_delta_column,
        )

    direct_summary = pd.DataFrame()
    normalized_direct_required = {*group_columns, "SessionTime", "Speed"}
    if not normalized_direct_required.difference(telemetry.columns):
        shared_direct = evaluate_direct_fastf1_style(
            telemetry,
            group_columns=group_columns,
        ).laps
        if (
            not shared_direct.empty
            and shared_direct[[metric for metric in _METRICS if metric in shared_direct]]
            .notna()
            .any(axis=None)
        ):
            direct_summary = shared_direct
    if direct_summary.empty:
        direct_summary = legacy_direct_summary

    physical_required = {*group_columns, "SessionTime", "Distance", "Speed"}
    physical_summary = pd.DataFrame()
    if not physical_required.difference(telemetry.columns):
        physical_summary = _summarize_lap_gaps_from_track_position(
            telemetry,
            group_columns=group_columns,
        )
    if not direct_summary.empty:
        combined = combine_lap_traffic_components(
            direct_summary,
            physical_summary if not physical_summary.empty else pd.DataFrame(columns=group_columns),
            group_columns=group_columns,
            combination_mode=COMBINATION_MODE_COMPATIBILITY_MIN,
        )
        return _attach_legacy_direct_comparison(
            combined,
            legacy_direct_summary,
            group_columns=group_columns,
        )
    if not physical_summary.empty:
        return combine_lap_traffic_components(
            pd.DataFrame(columns=group_columns),
            physical_summary,
            group_columns=group_columns,
            combination_mode=COMBINATION_MODE_COMPATIBILITY_MIN,
        )

    required_columns = {*group_columns, time_delta_column, distance_delta_column}
    missing_columns = required_columns.difference(telemetry.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Telemetry is missing required columns: {missing}")

    return _summarize_lap_gaps_from_driver_ahead(
        telemetry,
        group_columns=group_columns,
        time_delta_column=time_delta_column,
        distance_delta_column=distance_delta_column,
    )


def _attach_legacy_direct_comparison(
    selected: pd.DataFrame,
    legacy: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    output = selected.copy()
    if legacy.empty:
        for metric in _METRICS:
            output[f"LegacyFastF1{metric}"] = np.nan
        output["LegacyFastF1TrafficStatus"] = TRAFFIC_STATUS_UNKNOWN
        return output
    columns = [*group_columns, *[metric for metric in _METRICS if metric in legacy]]
    comparison = legacy.loc[:, columns].rename(
        columns={metric: f"LegacyFastF1{metric}" for metric in _METRICS}
    )
    comparison["LegacyFastF1TrafficStatus"] = TRAFFIC_STATUS_OK
    return output.merge(comparison, on=list(group_columns), how="left")


def _summarize_lap_gaps_from_driver_ahead(
    telemetry: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    time_delta_column: str,
    distance_delta_column: str,
) -> pd.DataFrame:
    summary_input = telemetry.loc[:, [*group_columns, time_delta_column, distance_delta_column]]
    ahead_summary = (
        summary_input.groupby(list(group_columns), dropna=False, as_index=False)
        .agg(
            MinTimeDeltaToDriverAhead=(time_delta_column, "min"),
            MeanTimeDeltaToDriverAhead=(time_delta_column, "mean"),
            MinDistanceToDriverAhead=(distance_delta_column, "min"),
            MeanDistanceToDriverAhead=(distance_delta_column, "mean"),
        )
    )
    behind_summary = _summarize_lap_gap_behind(
        telemetry,
        group_columns=group_columns,
        time_delta_column=time_delta_column,
        distance_delta_column=distance_delta_column,
    )
    if behind_summary.empty:
        ahead_summary["MinTimeDeltaToDriverBehind"] = pd.NA
        ahead_summary["MeanTimeDeltaToDriverBehind"] = pd.NA
        ahead_summary["MinDistanceToDriverBehind"] = pd.NA
        ahead_summary["MeanDistanceToDriverBehind"] = pd.NA
        return ahead_summary

    return ahead_summary.merge(
        behind_summary,
        on=list(group_columns),
        how="left",
    )


def _combine_gap_summaries(
    *,
    preferred: pd.DataFrame,
    fallback: pd.DataFrame,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    columns = [*group_columns, *_gap_summary_columns(group_columns)]
    preferred = preferred.copy()
    fallback = fallback.copy()
    for column in columns:
        if column not in preferred.columns:
            preferred[column] = pd.NA
        if column not in fallback.columns:
            fallback[column] = pd.NA
    preferred = preferred.loc[:, columns].set_index(list(group_columns))
    fallback = fallback.loc[:, columns].set_index(list(group_columns))
    combined = preferred.combine_first(fallback)
    for column in _gap_summary_columns(group_columns):
        combined[column] = pd.concat(
            [preferred[column], fallback[column]],
            axis=1,
        ).min(axis=1, skipna=True)
    combined = combined.reset_index()
    return combined.loc[:, columns]


def _summarize_lap_gap_behind(
    telemetry: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    time_delta_column: str,
    distance_delta_column: str,
) -> pd.DataFrame:
    common_required = {*group_columns, "DriverNumber"}
    if common_required.difference(telemetry.columns):
        return _empty_behind_gap_summary(group_columns)

    driver_ahead_required = {"DriverAhead", time_delta_column, distance_delta_column}
    if driver_ahead_required.difference(telemetry.columns):
        return _empty_behind_gap_summary(group_columns)
    return _summarize_lap_gap_behind_from_driver_ahead(
        telemetry,
        group_columns=group_columns,
        time_delta_column=time_delta_column,
        distance_delta_column=distance_delta_column,
    )


def _summarize_lap_gap_behind_from_driver_ahead(
    telemetry: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    time_delta_column: str,
    distance_delta_column: str,
) -> pd.DataFrame:
    driver_lookup_columns = [
        column
        for column in ("Year", "Round", "SessionName", "LapNumber", "Driver", "DriverNumber")
        if column in telemetry.columns
    ]
    driver_lookup = (
        telemetry.loc[:, driver_lookup_columns]
        .dropna(subset=["Driver", "DriverNumber"])
        .drop_duplicates()
        .copy()
    )
    driver_lookup["DriverNumber"] = driver_lookup["DriverNumber"].astype("string")

    behind_samples = telemetry.dropna(subset=["DriverAhead", time_delta_column]).copy()
    behind_samples["DriverAhead"] = behind_samples["DriverAhead"].astype("string")
    merge_columns = [
        column
        for column in ("Year", "Round", "SessionName", "LapNumber")
        if column in group_columns and column in telemetry.columns
    ]
    target_lookup = driver_lookup.rename(
        columns={
            "Driver": "TargetDriver",
            "DriverNumber": "DriverAhead",
        }
    )
    behind_samples = behind_samples.merge(
        target_lookup[[*merge_columns, "TargetDriver", "DriverAhead"]],
        on=[*merge_columns, "DriverAhead"],
        how="inner",
    )
    if behind_samples.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "MinTimeDeltaToDriverBehind",
                "MeanTimeDeltaToDriverBehind",
                "MinDistanceToDriverBehind",
                "MeanDistanceToDriverBehind",
            ]
        )

    behind_samples["Driver"] = behind_samples["TargetDriver"]
    return (
        behind_samples.groupby(list(group_columns), dropna=False, as_index=False)
        .agg(
            MinTimeDeltaToDriverBehind=(time_delta_column, "min"),
            MeanTimeDeltaToDriverBehind=(time_delta_column, "mean"),
            MinDistanceToDriverBehind=(distance_delta_column, "min"),
            MeanDistanceToDriverBehind=(distance_delta_column, "mean"),
        )
    )


def _summarize_lap_gaps_from_track_position(
    telemetry: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Adapt retrospective FastF1 telemetry to the shared physical evaluator."""
    lap_distance = _estimate_lap_distance(telemetry)
    if not np.isfinite(lap_distance) or lap_distance <= 0:
        return _empty_gap_summary(group_columns)
    result = evaluate_physical_traffic(
        telemetry,
        config=TrafficEvaluationConfig(circuit_length_m=lap_distance),
        group_columns=group_columns,
    )
    return result.laps


def _legacy_summarize_lap_gaps_from_track_position(
    telemetry: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    session_columns = [
        column
        for column in ("Year", "Round", "SessionName")
        if column in group_columns and column in telemetry.columns
    ]
    target_columns = list(
        dict.fromkeys([*group_columns, "SessionTime", "Distance", "Speed"])
    )
    target_samples = (
        telemetry.loc[:, target_columns]
        .dropna(subset=["Driver", "SessionTime", "Distance", "Speed"])
        .copy()
    )
    if target_samples.empty:
        return _empty_gap_summary(group_columns)

    target_samples["_TargetRowId"] = np.arange(len(target_samples))
    target_samples["_TargetDistance"] = pd.to_numeric(
        target_samples["Distance"],
        errors="coerce",
    )
    target_samples["_TargetSpeedKph"] = pd.to_numeric(
        target_samples["Speed"],
        errors="coerce",
    )
    target_samples = target_samples.dropna(subset=["_TargetDistance", "_TargetSpeedKph"])
    target_samples = target_samples.loc[target_samples["_TargetSpeedKph"] > 1.0]
    if target_samples.empty:
        return _empty_gap_summary(group_columns)

    nearest_ahead_samples: list[pd.DataFrame] = []
    nearest_behind_samples: list[pd.DataFrame] = []
    session_groups = (
        target_samples.groupby(session_columns, dropna=False, sort=False)
        if session_columns
        else [((), target_samples)]
    )
    for session_key, session_targets in session_groups:
        session_telemetry = _session_slice(
            telemetry,
            session_columns=session_columns,
            session_key=session_key,
        )
        if session_telemetry.empty:
            continue

        lap_distance = _estimate_lap_distance(session_telemetry)
        if not np.isfinite(lap_distance) or lap_distance <= 0:
            continue

        session_targets = session_targets.sort_values("SessionTime")
        for other_driver, other_samples in session_telemetry.groupby("Driver", dropna=False):
            other_samples = other_samples.loc[
                :,
                ["SessionTime", "Driver", "Distance", "Speed"],
            ].copy()
            other_samples = other_samples.dropna(
                subset=["SessionTime", "Driver", "Distance", "Speed"]
            )
            if other_samples.empty:
                continue
            other_samples["Distance"] = pd.to_numeric(
                other_samples["Distance"],
                errors="coerce",
            )
            other_samples["Speed"] = pd.to_numeric(other_samples["Speed"], errors="coerce")
            other_samples = other_samples.dropna(subset=["Distance", "Speed"])
            other_samples = other_samples.loc[other_samples["Speed"] > 1.0]
            if other_samples.empty:
                continue

            other_samples = other_samples.sort_values("SessionTime")
            matched = pd.merge_asof(
                session_targets,
                other_samples.rename(
                    columns={
                        "Driver": "_OtherDriver",
                        "Distance": "_OtherDistance",
                        "Speed": "_OtherSpeedKph",
                    }
                ),
                on="SessionTime",
                direction="nearest",
                tolerance=pd.Timedelta(milliseconds=500),
            )
            matched = matched.dropna(subset=["_OtherDriver", "_OtherDistance"])
            if matched.empty:
                continue
            matched = matched.loc[matched["Driver"] != matched["_OtherDriver"]].copy()
            if matched.empty:
                continue

            ahead_distance = (
                matched["_OtherDistance"].astype("float64")
                - matched["_TargetDistance"].astype("float64")
            ) % lap_distance
            matched["_AheadDistance"] = ahead_distance
            ahead = matched.loc[matched["_AheadDistance"] > 1.0].copy()
            if not ahead.empty:
                ahead["_AheadTimeDelta"] = ahead["_AheadDistance"] / (
                    ahead["_TargetSpeedKph"].astype("float64") / 3.6
                )
                nearest_ahead_samples.append(
                    ahead.loc[
                        :,
                        [*group_columns, "_TargetRowId", "_AheadDistance", "_AheadTimeDelta"],
                    ]
                )

            behind_distance = (
                matched["_TargetDistance"].astype("float64")
                - matched["_OtherDistance"].astype("float64")
            ) % lap_distance
            matched["_BehindDistance"] = behind_distance
            behind = matched.loc[matched["_BehindDistance"] > 1.0].copy()
            if not behind.empty:
                behind["_BehindTimeDelta"] = behind["_BehindDistance"] / (
                    behind["_OtherSpeedKph"].astype("float64") / 3.6
                )
                nearest_behind_samples.append(
                    behind.loc[
                        :,
                        [
                            *group_columns,
                            "_TargetRowId",
                            "_BehindDistance",
                            "_BehindTimeDelta",
                        ],
                    ]
                )

    if not nearest_ahead_samples and not nearest_behind_samples:
        return _empty_gap_summary(group_columns)

    summaries: list[pd.DataFrame] = []
    if nearest_ahead_samples:
        summaries.append(
            _nearest_gap_summary(
                pd.concat(nearest_ahead_samples, ignore_index=True),
                group_columns=group_columns,
                distance_column="_AheadDistance",
                time_column="_AheadTimeDelta",
                output_prefix="Ahead",
            )
        )
    if nearest_behind_samples:
        summaries.append(
            _nearest_gap_summary(
                pd.concat(nearest_behind_samples, ignore_index=True),
                group_columns=group_columns,
                distance_column="_BehindDistance",
                time_column="_BehindTimeDelta",
                output_prefix="Behind",
            )
        )
    summary = summaries[0]
    for other in summaries[1:]:
        summary = summary.merge(other, on=list(group_columns), how="outer")
    for column in _gap_summary_columns(group_columns):
        if column not in summary.columns:
            summary[column] = pd.NA
    return summary.loc[:, [*group_columns, *_gap_summary_columns(group_columns)]]


def _nearest_gap_summary(
    candidate_samples: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    distance_column: str,
    time_column: str,
    output_prefix: str,
) -> pd.DataFrame:
    nearest_by_target_sample = (
        candidate_samples.sort_values(
            ["_TargetRowId", distance_column],
            kind="stable",
        )
        .drop_duplicates(subset=["_TargetRowId"], keep="first")
    )
    return (
        nearest_by_target_sample.groupby(list(group_columns), dropna=False, as_index=False)
        .agg(
            **{
                f"MinTimeDeltaToDriver{output_prefix}": (time_column, "min"),
                f"MeanTimeDeltaToDriver{output_prefix}": (time_column, "mean"),
                f"MinDistanceToDriver{output_prefix}": (distance_column, "min"),
                f"MeanDistanceToDriver{output_prefix}": (distance_column, "mean"),
            }
        )
    )


def _session_slice(
    telemetry: pd.DataFrame,
    *,
    session_columns: list[str],
    session_key: object,
) -> pd.DataFrame:
    if not session_columns:
        return telemetry
    key_values = session_key if isinstance(session_key, tuple) else (session_key,)
    mask = pd.Series(True, index=telemetry.index)
    for column, value in zip(session_columns, key_values):
        mask = mask & (telemetry[column] == value)
    return telemetry.loc[mask]


def _estimate_lap_distance(telemetry: pd.DataFrame) -> float:
    distance = pd.to_numeric(telemetry["Distance"], errors="coerce")
    global_span = float(distance.max() - distance.min())
    if {"Driver", "LapNumber"}.difference(telemetry.columns):
        return global_span

    lap_spans = (
        telemetry.assign(_Distance=distance)
        .dropna(subset=["Driver", "LapNumber", "_Distance"])
        .groupby(["Driver", "LapNumber"], dropna=False)["_Distance"]
        .agg(lambda values: float(values.max() - values.min()))
    )
    lap_spans = lap_spans.loc[np.isfinite(lap_spans) & (lap_spans > 1000.0)]
    if lap_spans.empty:
        return global_span

    median_span = float(lap_spans.median())
    plausible_spans = lap_spans.loc[
        (lap_spans > median_span * 0.75)
        & (lap_spans < median_span * 1.25)
    ]
    if plausible_spans.empty:
        return median_span
    return float(plausible_spans.median())


def _empty_behind_gap_summary(group_columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *group_columns,
            "MinTimeDeltaToDriverBehind",
            "MeanTimeDeltaToDriverBehind",
            "MinDistanceToDriverBehind",
            "MeanDistanceToDriverBehind",
        ]
    )


def _empty_gap_summary(group_columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=[*group_columns, *_gap_summary_columns(group_columns)])


def _gap_summary_columns(group_columns: tuple[str, ...]) -> list[str]:
    return [
        "MinTimeDeltaToDriverAhead",
        "MeanTimeDeltaToDriverAhead",
        "MinDistanceToDriverAhead",
        "MeanDistanceToDriverAhead",
        "MinTimeDeltaToDriverBehind",
        "MeanTimeDeltaToDriverBehind",
        "MinDistanceToDriverBehind",
        "MeanDistanceToDriverBehind",
    ]
