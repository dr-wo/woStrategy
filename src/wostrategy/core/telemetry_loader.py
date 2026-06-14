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
    required_columns = {
        *group_columns,
        "DriverNumber",
        "DriverAhead",
        time_delta_column,
        distance_delta_column,
    }
    if required_columns.difference(telemetry.columns):
        return pd.DataFrame(
            columns=[
                *group_columns,
                "MinTimeDeltaToDriverBehind",
                "MeanTimeDeltaToDriverBehind",
                "MinDistanceToDriverBehind",
                "MeanDistanceToDriverBehind",
            ]
        )

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


def _safe_cache_part(value: object) -> str:
    return "".join(char if char.isalnum() or char in ("-", ".") else "-" for char in str(value))
