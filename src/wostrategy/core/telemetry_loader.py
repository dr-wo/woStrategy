from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Union

import numpy as np
import pandas as pd

RoundLike = Union[int, str]
SessionNameLike = Union[int, str]
DEFAULT_TELEMETRY_CACHE_DIR = Path(__file__).resolve().parents[3] / "cache" / "telemetry"


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


@dataclass
class TelemetryDataLoader:
    """Load per-lap FastF1 telemetry and append strategy-oriented columns."""

    time_delta_estimator: TimeDeltaEstimator = field(
        default_factory=DistanceInterpolationTimeDeltaEstimator
    )
    telemetry_method: str = "get_telemetry"
    telemetry_kwargs: dict | None = None
    lap_metadata_columns: tuple[str, ...] = (
        "Driver",
        "DriverNumber",
        "Team",
        "LapNumber",
        "Stint",
        "Compound",
        "TyreLife",
        "LapTime",
        "LapStartTime",
        "PitOutTime",
        "PitInTime",
    )
    skip_lap_errors: bool = True

    def load_session(
        self,
        session: object,
        *,
        year: int | None = None,
        round_number: RoundLike | None = None,
        session_name: SessionNameLike | None = None,
    ) -> pd.DataFrame:
        """Load enriched telemetry for all laps currently exposed by a session."""
        telemetry_frames: list[pd.DataFrame] = []
        laps = getattr(session, "laps")

        for _, lap in laps.iterlaps():
            try:
                telemetry = self._load_lap_telemetry(lap)
                if telemetry.empty:
                    continue

                telemetry = self.time_delta_estimator.add_time_delta(telemetry)
                telemetry = self._add_lap_metadata(
                    telemetry,
                    lap,
                    year=year,
                    round_number=round_number,
                    session_name=session_name,
                )
            except Exception:
                if self.skip_lap_errors:
                    continue
                raise

            telemetry_frames.append(telemetry)

        if telemetry_frames:
            return pd.concat(telemetry_frames, ignore_index=True)

        return pd.DataFrame(columns=self.empty_columns)

    @property
    def empty_columns(self) -> list[str]:
        return [
            "Year",
            "Round",
            "SessionName",
            *self.lap_metadata_columns,
            self.time_delta_estimator.output_column,
        ]

    def _load_lap_telemetry(self, lap: pd.Series) -> pd.DataFrame:
        telemetry_kwargs = self.telemetry_kwargs or {}
        telemetry_getter = getattr(lap, self.telemetry_method)
        telemetry = telemetry_getter(**telemetry_kwargs)
        return pd.DataFrame(telemetry).copy()

    def _add_lap_metadata(
        self,
        telemetry: pd.DataFrame,
        lap: pd.Series,
        *,
        year: int | None,
        round_number: RoundLike | None,
        session_name: SessionNameLike | None,
    ) -> pd.DataFrame:
        telemetry = telemetry.copy()
        if year is not None:
            telemetry["Year"] = year
        if round_number is not None:
            telemetry["Round"] = round_number
        if session_name is not None:
            telemetry["SessionName"] = session_name

        for column in self.lap_metadata_columns:
            if column in lap.index:
                telemetry[column] = lap[column]

        return telemetry


def load_session_telemetry(
    *,
    year: int,
    rounds: list[RoundLike],
    session_names: list[SessionNameLike],
    session_factory: Callable[[RoundLike, SessionNameLike], object],
    enrich_session: Callable[[object], None] | None = None,
    telemetry_loader: TelemetryDataLoader | None = None,
    log_label: str = "Loading telemetry",
    skip_label: str = "Skipping telemetry",
) -> pd.DataFrame:
    """Load telemetry from multiple sessions and append session metadata."""
    loader = telemetry_loader or TelemetryDataLoader()
    all_telemetry: list[pd.DataFrame] = []

    for round_number in rounds:
        for session_name in session_names:
            print(f"{log_label} {year} round={round_number} session={session_name}")
            try:
                session = session_factory(round_number, session_name)
                if enrich_session is not None:
                    enrich_session(session)
                telemetry = loader.load_session(
                    session,
                    year=year,
                    round_number=round_number,
                    session_name=session_name,
                )
            except Exception as exc:
                print(
                    f"{skip_label} year={year}, round={round_number}, "
                    f"session={session_name}: {exc}"
                )
                continue

            if not telemetry.empty:
                all_telemetry.append(telemetry)

    if all_telemetry:
        return pd.concat(all_telemetry, ignore_index=True)

    return pd.DataFrame(columns=loader.empty_columns)


def load_or_cache_session_telemetry(
    session: object,
    *,
    year: int,
    round_number: RoundLike,
    session_name: SessionNameLike,
    telemetry_loader: TelemetryDataLoader | None = None,
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load full session telemetry from cache, or query FastF1 and cache it."""
    cache_path = get_session_telemetry_cache_path(
        year=year,
        round_number=round_number,
        session_name=session_name,
        cache_dir=cache_dir,
    )
    loader = telemetry_loader or TelemetryDataLoader()
    if cache_path.exists() and not force_refresh:
        telemetry = pd.read_pickle(cache_path)
        if loader.time_delta_estimator.output_column in telemetry.columns:
            return telemetry

    telemetry = loader.load_session(
        session,
        year=year,
        round_number=round_number,
        session_name=session_name,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry.to_pickle(cache_path)
    return telemetry


def get_session_telemetry_cache_path(
    *,
    year: int,
    round_number: RoundLike,
    session_name: SessionNameLike,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return cache path named as ``<year>_<race>_<session>``."""
    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_TELEMETRY_CACHE_DIR
    file_name = "_".join(
        [
            str(year),
            _safe_cache_part(round_number),
            _safe_cache_part(session_name),
        ]
    )
    return cache_root / file_name


def summarize_lap_gap_metrics(
    telemetry: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("Year", "Round", "SessionName", "Driver", "LapNumber"),
    time_delta_column: str = "TimeDeltaToDriverAhead",
    distance_delta_column: str = "DistanceToDriverAhead",
) -> pd.DataFrame:
    """Aggregate full telemetry into per-lap front-car gap metrics."""
    required_columns = {*group_columns, time_delta_column, distance_delta_column}
    missing_columns = required_columns.difference(telemetry.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Telemetry is missing required columns: {missing}")

    if telemetry.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "MinTimeDeltaToDriverAhead",
                "MeanTimeDeltaToDriverAhead",
                "MinDistanceToDriverAhead",
                "MeanDistanceToDriverAhead",
            ]
        )

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


def _summarize_lap_gap_behind(
    telemetry: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    time_delta_column: str,
    distance_delta_column: str,
) -> pd.DataFrame:
    common_required = {*group_columns, "DriverNumber"}
    if common_required.difference(telemetry.columns):
        return pd.DataFrame(
            columns=[
                *group_columns,
                "MinTimeDeltaToDriverBehind",
                "MeanTimeDeltaToDriverBehind",
                "MinDistanceToDriverBehind",
                "MeanDistanceToDriverBehind",
            ]
        )

    physical_required = {"SessionTime", "Distance", "Speed"}
    if not physical_required.difference(telemetry.columns):
        physical_summary = _summarize_lap_gap_behind_from_track_position(
            telemetry,
            group_columns=group_columns,
        )
        if not physical_summary.empty:
            return physical_summary

    driver_ahead_required = {"DriverAhead", time_delta_column, distance_delta_column}
    if driver_ahead_required.difference(telemetry.columns):
        return pd.DataFrame(
            columns=[
                *group_columns,
                "MinTimeDeltaToDriverBehind",
                "MeanTimeDeltaToDriverBehind",
                "MinDistanceToDriverBehind",
                "MeanDistanceToDriverBehind",
            ]
        )
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


def _summarize_lap_gap_behind_from_track_position(
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
        dict.fromkeys([*group_columns, "DriverNumber", "SessionTime", "Distance"])
    )
    target_samples = (
        telemetry.loc[:, target_columns]
        .dropna(subset=["Driver", "DriverNumber", "SessionTime", "Distance"])
        .copy()
    )
    if target_samples.empty:
        return _empty_behind_gap_summary(group_columns)

    target_samples["_TargetRowId"] = np.arange(len(target_samples))
    target_samples["DriverNumber"] = target_samples["DriverNumber"].astype("string")
    target_samples["_TargetDistance"] = pd.to_numeric(
        target_samples["Distance"],
        errors="coerce",
    )
    target_samples = target_samples.dropna(subset=["_TargetDistance"])
    if target_samples.empty:
        return _empty_behind_gap_summary(group_columns)

    nearest_samples: list[pd.DataFrame] = []
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

        distance = pd.to_numeric(session_telemetry["Distance"], errors="coerce")
        lap_distance = float(distance.max() - distance.min())
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

            gap_distance = (
                matched["_TargetDistance"].astype("float64")
                - matched["_OtherDistance"].astype("float64")
            ) % lap_distance
            matched["_BehindDistance"] = gap_distance
            matched = matched.loc[matched["_BehindDistance"] > 1.0].copy()
            if matched.empty:
                continue
            matched["_BehindTimeDelta"] = matched["_BehindDistance"] / (
                matched["_OtherSpeedKph"].astype("float64") / 3.6
            )
            nearest_samples.append(
                matched.loc[
                    :,
                    [*group_columns, "_TargetRowId", "_BehindDistance", "_BehindTimeDelta"],
                ]
            )

    if not nearest_samples:
        return _empty_behind_gap_summary(group_columns)

    candidate_samples = pd.concat(nearest_samples, ignore_index=True)
    candidate_samples = candidate_samples.sort_values(
        ["_TargetRowId", "_BehindDistance"],
        kind="stable",
    )
    nearest_by_target_sample = candidate_samples.drop_duplicates(
        subset=["_TargetRowId"],
        keep="first",
    )
    return (
        nearest_by_target_sample.groupby(list(group_columns), dropna=False, as_index=False)
        .agg(
            MinTimeDeltaToDriverBehind=("_BehindTimeDelta", "min"),
            MeanTimeDeltaToDriverBehind=("_BehindTimeDelta", "mean"),
            MinDistanceToDriverBehind=("_BehindDistance", "min"),
            MeanDistanceToDriverBehind=("_BehindDistance", "mean"),
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


def _safe_cache_part(value: object) -> str:
    return "".join(char if char.isalnum() or char in ("-", ".") else "-" for char in str(value))
