from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol

import numpy as np
import pandas as pd

from wostrategy.algorithm.sampling import (
    RANDOM_SAMPLER,
    get_unit_cube_sampler,
    scale_unit_sample,
)
from wostrategy.model.tyre_degragation import TYRE_AGE_LAPS_COLUMN

BASELINE_DRIVER = "driver"
BASELINE_TEAM = "team"
BASELINE_GROUPS = (BASELINE_DRIVER, BASELINE_TEAM)

FUEL_PROXY_LAPS_REMAINING = "FuelProxyLapsRemaining"
CORRECTED_LAP_TIME_SECONDS = "CorrectedLapTimeSeconds"
DEFAULT_DEGRADATION_ORDER_TRACK_TEMPERATURE_CELSIUS = 20.0
ORDERED_DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
TRACK_TEMPERATURE_COLUMNS = (
    "TrackTemp",
    "TrackTemperature",
    "TrackTempCelsius",
    "TrackTemperatureCelsius",
)


@dataclass(frozen=True)
class MonteCarloRacePerformanceConfig:
    sample_count: int = 1000
    fuel_rate_bounds: tuple[float, float] = (0.0, 0.08)
    track_rate_bounds: tuple[float, float] = (-0.05, 0.05)
    compound_degradation_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    default_compound_degradation_bounds: tuple[float, float] = (0.0, 0.08)
    team_variation_fraction: float = 0.5
    team_variation_absolute_min: float = 0.005
    clean_lap_noise_sigma: float = 0.15
    baseline_group: Literal["driver", "team"] = BASELINE_DRIVER
    fuel_ref: float = 0.0
    race_lap_ref: float | None = None
    tyre_age_ref: float = 0.0
    random_seed: int | None = None
    sampling_strategy: str = RANDOM_SAMPLER
    track_temperature_celsius: float | None = None
    degradation_order_track_temperature_celsius: float | None = (
        DEFAULT_DEGRADATION_ORDER_TRACK_TEMPERATURE_CELSIUS
    )


@dataclass(frozen=True)
class RacePerformanceAlgorithmResult:
    sample_parameters: pd.DataFrame
    compound_degradation: pd.DataFrame
    team_compound_degradation: pd.DataFrame
    baseline_pace: pd.DataFrame


class RacePerformanceReviewAlgorithm(Protocol):
    def run(self, clean_laps: pd.DataFrame) -> RacePerformanceAlgorithmResult:
        """Return sampled parameters and fitted baselines for prepared clean laps."""


class MonteCarloRacePerformanceAlgorithm:
    """Simple weighted Monte Carlo race correction algorithm.

    The input frame must already be prepared and clean-lap filtered. This class
    intentionally knows nothing about FastF1 loading, clean-air selection, or
    report persistence so the modelling algorithm can evolve independently.
    """

    def __init__(
        self,
        config: MonteCarloRacePerformanceConfig | None = None,
        *,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        progress_interval: int | None = None,
    ) -> None:
        self.config = config or MonteCarloRacePerformanceConfig()
        _validate_config(self.config)
        if progress_interval is not None and progress_interval <= 0:
            raise ValueError("progress_interval must be positive or None.")
        self.progress_callback = progress_callback
        self.progress_interval = progress_interval

    def run(self, clean_laps: pd.DataFrame) -> RacePerformanceAlgorithmResult:
        fit_laps = clean_laps.dropna(
            subset=[
                "Driver",
                "Team",
                "Compound",
                "LapNumber",
                "LapTimeSeconds",
                FUEL_PROXY_LAPS_REMAINING,
                TYRE_AGE_LAPS_COLUMN,
            ]
        ).copy()
        if fit_laps.empty:
            raise ValueError("No clean laps had the columns required for Monte Carlo correction.")

        config = self.config
        compounds = tuple(sorted(fit_laps["Compound"].astype(str).unique()))
        team_compounds = tuple(
            sorted(
                (str(row.Team), str(row.Compound))
                for row in fit_laps[["Team", "Compound"]]
                .drop_duplicates()
                .itertuples(index=False)
            )
        )
        baseline_columns = _baseline_columns(config.baseline_group)
        race_lap_ref = (
            float(fit_laps["LapNumber"].mean())
            if config.race_lap_ref is None
            else float(config.race_lap_ref)
        )
        track_temperature = _track_temperature_celsius(fit_laps, config)
        enforce_degradation_order = _should_enforce_degradation_order(
            track_temperature,
            config,
        )

        sample_rows: list[dict[str, object]] = []
        compound_rows: list[dict[str, object]] = []
        team_compound_rows: list[dict[str, object]] = []
        baseline_rows: list[dict[str, object]] = []
        dimension_count = 2 + len(compounds) + len(team_compounds)
        unit_samples = get_unit_cube_sampler(config.sampling_strategy).sample(
            sample_count=config.sample_count,
            dimension_count=dimension_count,
            seed=config.random_seed,
        )

        lap_time = fit_laps["LapTimeSeconds"].to_numpy(dtype="float64")
        fuel_proxy_delta = (
            fit_laps[FUEL_PROXY_LAPS_REMAINING].to_numpy(dtype="float64")
            - config.fuel_ref
        )
        track_delta = fit_laps["LapNumber"].to_numpy(dtype="float64") - race_lap_ref
        tyre_age_delta = (
            fit_laps[TYRE_AGE_LAPS_COLUMN].to_numpy(dtype="float64")
            - config.tyre_age_ref
        )
        group_labels = _baseline_labels(fit_laps, baseline_columns)
        team_compound_index = pd.MultiIndex.from_frame(fit_laps[["Team", "Compound"]])
        best_rmse = float("inf")
        weight_sum = 0.0

        for sample_id in range(config.sample_count):
            sample = unit_samples[sample_id]
            dimension_index = 0
            fuel_rate = scale_unit_sample(
                sample[dimension_index],
                config.fuel_rate_bounds,
            )
            dimension_index += 1
            track_rate = scale_unit_sample(
                sample[dimension_index],
                config.track_rate_bounds,
            )
            dimension_index += 1
            compound_unit_samples: dict[str, float] = {}
            for compound in compounds:
                compound_unit_samples[compound] = float(sample[dimension_index])
                dimension_index += 1
            compound_rates = _sample_compound_rates(
                compound_unit_samples,
                config=config,
                enforce_degradation_order=enforce_degradation_order,
            )
            team_compound_rates: dict[tuple[str, str], float] = {}
            for team, compound in team_compounds:
                compound_rate = compound_rates[compound]
                variation_limit = max(
                    config.team_variation_fraction * abs(compound_rate),
                    config.team_variation_absolute_min,
                )
                variation = scale_unit_sample(
                    sample[dimension_index],
                    (-variation_limit, variation_limit),
                )
                dimension_index += 1
                team_compound_rates[(team, compound)] = compound_rate + variation

            tyre_rate = np.fromiter(
                (
                    team_compound_rates[(str(team), str(compound))]
                    for team, compound in team_compound_index
                ),
                dtype="float64",
                count=len(fit_laps),
            )
            corrected = (
                lap_time
                - (fuel_rate * fuel_proxy_delta)
                - (track_rate * track_delta)
                - (tyre_rate * tyre_age_delta)
            )
            baselines = (
                pd.DataFrame(
                    {
                        "BaselineGroupKey": group_labels,
                        CORRECTED_LAP_TIME_SECONDS: corrected,
                    }
                )
                .groupby("BaselineGroupKey", sort=True)[CORRECTED_LAP_TIME_SECONDS]
                .mean()
            )
            fitted = np.array([baselines[label] for label in group_labels], dtype="float64")
            rmse = float(np.sqrt(np.mean((corrected - fitted) ** 2)))
            weight = float(np.exp(-((rmse**2) / (2 * config.clean_lap_noise_sigma**2))))
            best_rmse = min(best_rmse, rmse)
            weight_sum += weight

            sample_rows.append(
                {
                    "SampleId": sample_id,
                    "FuelRateSecondsPerLap": fuel_rate,
                    "TrackRateSecondsPerLap": track_rate,
                    "RMSESeconds": rmse,
                    "Score": rmse,
                    "Weight": weight,
                    "CleanLapCount": int(len(fit_laps)),
                    "BaselineGroup": config.baseline_group,
                    "FuelRef": float(config.fuel_ref),
                    "RaceLapRef": race_lap_ref,
                    "TyreAgeRef": float(config.tyre_age_ref),
                    "SamplingStrategy": config.sampling_strategy,
                    "TrackTemperatureCelsius": track_temperature,
                    "DegradationOrderTrackTemperatureCelsius": (
                        config.degradation_order_track_temperature_celsius
                    ),
                    "DegradationOrderEnforced": enforce_degradation_order,
                }
            )
            for compound, rate in compound_rates.items():
                compound_rows.append(
                    {
                        "SampleId": sample_id,
                        "Compound": compound,
                        "CompoundDegSecondsPerLap": rate,
                    }
                )
            for (team, compound), rate in team_compound_rates.items():
                compound_rate = compound_rates[compound]
                team_compound_rows.append(
                    {
                        "SampleId": sample_id,
                        "Team": team,
                        "Compound": compound,
                        "CompoundDegSecondsPerLap": compound_rate,
                        "TeamCompoundDegSecondsPerLap": rate,
                        "VariationSecondsPerLap": rate - compound_rate,
                    }
                )
            for key, pace in baselines.items():
                row = {
                    "SampleId": sample_id,
                    "BaselineGroup": config.baseline_group,
                    "BaselineGroupKey": key,
                    "CorrectedBaselinePaceSeconds": float(pace),
                }
                row.update(_split_baseline_key(key, baseline_columns))
                baseline_rows.append(row)
            self._report_progress(
                sample_id=sample_id,
                rmse=rmse,
                best_rmse=best_rmse,
                weight_sum=weight_sum,
            )

        return RacePerformanceAlgorithmResult(
            sample_parameters=pd.DataFrame(sample_rows),
            compound_degradation=pd.DataFrame(compound_rows),
            team_compound_degradation=pd.DataFrame(team_compound_rows),
            baseline_pace=pd.DataFrame(baseline_rows),
        )

    def _report_progress(
        self,
        *,
        sample_id: int,
        rmse: float,
        best_rmse: float,
        weight_sum: float,
    ) -> None:
        if self.progress_callback is None:
            return
        sample_number = sample_id + 1
        is_last = sample_number == self.config.sample_count
        should_report = sample_number == 1 or is_last
        if self.progress_interval is not None:
            should_report = should_report or sample_number % self.progress_interval == 0
        if not should_report:
            return
        self.progress_callback(
            {
                "sample": sample_number,
                "sample_count": self.config.sample_count,
                "rmse_seconds": rmse,
                "best_rmse_seconds": best_rmse,
                "weight_sum": weight_sum,
            }
        )


def _baseline_columns(baseline_group: str) -> tuple[str, ...]:
    if baseline_group == BASELINE_DRIVER:
        return ("Team", "Driver")
    if baseline_group == BASELINE_TEAM:
        return ("Team",)
    options = ", ".join(BASELINE_GROUPS)
    raise ValueError(f"Unknown baseline_group {baseline_group!r}. Options: {options}")


def _baseline_labels(laps: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    return laps.loc[:, list(columns)].astype(str).agg("||".join, axis=1)


def _split_baseline_key(key: str, columns: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(columns, key.split("||")))


def _compound_bounds(
    config: MonteCarloRacePerformanceConfig,
    compound: str,
) -> tuple[float, float]:
    return config.compound_degradation_bounds.get(
        compound,
        config.default_compound_degradation_bounds,
    )


def _sample_compound_rates(
    unit_samples: dict[str, float],
    *,
    config: MonteCarloRacePerformanceConfig,
    enforce_degradation_order: bool,
) -> dict[str, float]:
    if not enforce_degradation_order:
        return {
            compound: scale_unit_sample(unit_sample, _compound_bounds(config, compound))
            for compound, unit_sample in unit_samples.items()
        }

    rates: dict[str, float] = {}
    previous_rate: float | None = None
    for compound in reversed(ORDERED_DRY_COMPOUNDS):
        if compound not in unit_samples:
            continue
        lower, upper = _compound_bounds(config, compound)
        higher_compounds = ORDERED_DRY_COMPOUNDS[
            : ORDERED_DRY_COMPOUNDS.index(compound)
        ]
        upper = min(
            [upper]
            + [
                _compound_bounds(config, higher_compound)[1]
                for higher_compound in higher_compounds
                if higher_compound in unit_samples
            ]
        )
        if previous_rate is not None:
            lower = max(lower, previous_rate)
        rates[compound] = scale_unit_sample(unit_samples[compound], (lower, upper))
        previous_rate = rates[compound]

    for compound, unit_sample in unit_samples.items():
        if compound in rates:
            continue
        rates[compound] = scale_unit_sample(unit_sample, _compound_bounds(config, compound))
    return rates


def _track_temperature_celsius(
    laps: pd.DataFrame,
    config: MonteCarloRacePerformanceConfig,
) -> float | None:
    if config.track_temperature_celsius is not None:
        return float(config.track_temperature_celsius)
    for column in TRACK_TEMPERATURE_COLUMNS:
        if column not in laps.columns:
            continue
        values = pd.to_numeric(laps[column], errors="coerce").dropna()
        if not values.empty:
            return float(values.mean())
    return None


def _should_enforce_degradation_order(
    track_temperature_celsius: float | None,
    config: MonteCarloRacePerformanceConfig,
) -> bool:
    threshold = config.degradation_order_track_temperature_celsius
    if threshold is None or track_temperature_celsius is None:
        return False
    return track_temperature_celsius > threshold


def _validate_config(config: MonteCarloRacePerformanceConfig) -> None:
    if config.sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if config.clean_lap_noise_sigma <= 0:
        raise ValueError("clean_lap_noise_sigma must be positive.")
    if config.team_variation_fraction < 0:
        raise ValueError("team_variation_fraction must be non-negative.")
    if config.team_variation_absolute_min < 0:
        raise ValueError("team_variation_absolute_min must be non-negative.")
    if config.fuel_rate_bounds[0] < 0 or config.fuel_rate_bounds[1] < 0:
        raise ValueError("fuel_rate_bounds must be non-negative.")
    if (
        config.degradation_order_track_temperature_celsius is not None
        and not np.isfinite(config.degradation_order_track_temperature_celsius)
    ):
        raise ValueError(
            "degradation_order_track_temperature_celsius must be finite or None."
        )
    if (
        config.track_temperature_celsius is not None
        and not np.isfinite(config.track_temperature_celsius)
    ):
        raise ValueError("track_temperature_celsius must be finite or None.")
    _validate_bounds("fuel_rate_bounds", config.fuel_rate_bounds)
    _validate_bounds("track_rate_bounds", config.track_rate_bounds)
    _validate_bounds(
        "default_compound_degradation_bounds",
        config.default_compound_degradation_bounds,
    )
    for compound, bounds in config.compound_degradation_bounds.items():
        _validate_bounds(f"compound_degradation_bounds[{compound!r}]", bounds)
    _validate_degradation_order_bounds(config)
    get_unit_cube_sampler(config.sampling_strategy)
    _baseline_columns(config.baseline_group)


def _validate_bounds(name: str, bounds: tuple[float, float]) -> None:
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain lower and upper bounds.")
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower > upper:
        raise ValueError(f"{name} lower bound must be <= upper bound.")


def _validate_degradation_order_bounds(config: MonteCarloRacePerformanceConfig) -> None:
    if config.degradation_order_track_temperature_celsius is None:
        return
    hard_upper = _compound_bounds(config, "HARD")[1]
    medium_upper = _compound_bounds(config, "MEDIUM")[1]
    soft_upper = _compound_bounds(config, "SOFT")[1]
    hard_lower = _compound_bounds(config, "HARD")[0]
    medium_lower = _compound_bounds(config, "MEDIUM")[0]
    if hard_lower > medium_upper:
        raise ValueError(
            "Compound degradation bounds cannot satisfy MEDIUM >= HARD "
            "for hot-track ordering."
        )
    if max(medium_lower, hard_lower) > soft_upper:
        raise ValueError(
            "Compound degradation bounds cannot satisfy SOFT >= MEDIUM "
            "for hot-track ordering."
        )
