from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

import pandas as pd
from wodata import get_fastf1_telemetry_cache_dir
from wostrategy.analysis.traffic import (
    DistanceInterpolationTimeDeltaEstimator,
    TimeDeltaEstimator,
)

RoundLike = Union[int, str]
SessionNameLike = Union[int, str]
DEFAULT_TELEMETRY_CACHE_DIR = get_fastf1_telemetry_cache_dir()


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
    from wostrategy.analysis.traffic import (
        COMBINATION_MODE_COMPATIBILITY_MIN,
        TRAFFIC_EVALUATOR_VERSION,
    )

    cache_path = get_session_telemetry_cache_path(
        year=year,
        round_number=round_number,
        session_name=session_name,
        cache_dir=cache_dir,
    )
    loader = telemetry_loader or TelemetryDataLoader()
    if cache_path.exists() and not force_refresh:
        telemetry = pd.read_pickle(cache_path)
        if _is_valid_cached_telemetry(
            telemetry,
            required_time_delta_column=loader.time_delta_estimator.output_column,
        ):
            return telemetry

    telemetry = loader.load_session(
        session,
        year=year,
        round_number=round_number,
        session_name=session_name,
    )
    telemetry.attrs["traffic_evaluator_version"] = TRAFFIC_EVALUATOR_VERSION
    telemetry.attrs["traffic_combination_mode"] = COMBINATION_MODE_COMPATIBILITY_MIN
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry.to_pickle(cache_path)
    return telemetry


def _is_valid_cached_telemetry(
    telemetry: pd.DataFrame,
    *,
    required_time_delta_column: str,
) -> bool:
    from wostrategy.analysis.traffic import (
        COMBINATION_MODE_COMPATIBILITY_MIN,
        TRAFFIC_EVALUATOR_VERSION,
    )

    return (
        not telemetry.empty
        and required_time_delta_column in telemetry.columns
        and telemetry.attrs.get("traffic_evaluator_version") == TRAFFIC_EVALUATOR_VERSION
        and telemetry.attrs.get("traffic_combination_mode")
        == COMBINATION_MODE_COMPATIBILITY_MIN
    )


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


def _safe_cache_part(value: object) -> str:
    return "".join(char if char.isalnum() or char in ("-", ".") else "-" for char in str(value))
