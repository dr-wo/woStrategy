from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wostrategy.algorithm.monte_carlo_race_performance import (
    FUEL_PROXY_LAPS_REMAINING,
    MonteCarloRacePerformanceAlgorithm,
    MonteCarloRacePerformanceConfig,
    RacePerformanceReviewAlgorithm,
)
from wostrategy.analysis.long_run_performance import (
    TYRE_AGE_MODE_STINT,
    WET_COMPOUNDS,
    _prepare_laps,
    select_clean_air_stints_as_whole,
    select_consecutive_clean_air_runs,
)
from wostrategy.model.fuel_consumption import FUEL_LAP_NUMBER_COLUMN


@dataclass(frozen=True)
class MonteCarloRacePerformanceResult:
    all_laps: pd.DataFrame
    clean_laps: pd.DataFrame
    wet_lap_summary: pd.DataFrame
    sample_parameters: pd.DataFrame
    compound_degradation: pd.DataFrame
    compound_delta: pd.DataFrame
    team_compound_degradation: pd.DataFrame
    baseline_pace: pd.DataFrame
    summaries: dict[str, pd.DataFrame]


def calculate_monte_carlo_race_performance_review(
    laps: pd.DataFrame,
    *,
    min_clean_air_laps: int,
    clean_mean_time_delta_seconds: float,
    clean_mean_time_delta_behind_seconds: float | None,
    quick_lap_threshold: float,
    treat_stint_as_whole: bool = False,
    tyre_age_mode: str = TYRE_AGE_MODE_STINT,
    config: MonteCarloRacePerformanceConfig | None = None,
    algorithm: RacePerformanceReviewAlgorithm | None = None,
    dry_compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
    wet_lap_proportion_skip_threshold: float = 0.5,
) -> MonteCarloRacePerformanceResult | str:
    """Estimate race pace corrections with simple weighted Monte Carlo sampling.

    This intentionally reuses the existing long-run lap preparation and clean-air
    run selection. It is a pragmatic sampler, not a full Bayesian MCMC workflow.
    """
    if config is not None and algorithm is not None:
        raise ValueError("Pass either config or algorithm, not both.")
    if algorithm is None:
        algorithm = MonteCarloRacePerformanceAlgorithm(config)
    _validate_wet_lap_threshold(wet_lap_proportion_skip_threshold)

    wet_lap_summary = wet_lap_proportion_by_driver(laps)
    if is_wet_race(
        wet_lap_summary,
        wet_lap_proportion_skip_threshold=wet_lap_proportion_skip_threshold,
    ):
        return "Wet"

    prepared = _prepare_laps(
        laps,
        clean_mean_time_delta_seconds=clean_mean_time_delta_seconds,
        clean_mean_time_delta_behind_seconds=clean_mean_time_delta_behind_seconds,
        quick_lap_threshold=quick_lap_threshold,
        dry_compounds=dry_compounds,
        tyre_age_mode=tyre_age_mode,
    )
    prepared = _add_race_fuel_proxy(prepared)
    selector = (
        select_clean_air_stints_as_whole
        if treat_stint_as_whole
        else select_consecutive_clean_air_runs
    )
    clean_laps = selector(
        prepared,
        min_clean_air_laps=min_clean_air_laps,
    )
    if clean_laps.empty:
        raise ValueError("No consecutive clean-air race laps matched the configured filters.")

    algorithm_result = algorithm.run(clean_laps)
    sample_parameters = algorithm_result.sample_parameters
    compound_degradation = algorithm_result.compound_degradation
    compound_delta = algorithm_result.compound_delta
    team_compound_degradation = algorithm_result.team_compound_degradation
    baseline_pace = algorithm_result.baseline_pace
    summaries = summarize_monte_carlo_race_performance(
        sample_parameters=sample_parameters,
        compound_degradation=compound_degradation,
        compound_delta=compound_delta,
        team_compound_degradation=team_compound_degradation,
        baseline_pace=baseline_pace,
    )

    return MonteCarloRacePerformanceResult(
        all_laps=prepared,
        clean_laps=clean_laps,
        wet_lap_summary=wet_lap_summary,
        sample_parameters=sample_parameters,
        compound_degradation=compound_degradation,
        compound_delta=compound_delta,
        team_compound_degradation=team_compound_degradation,
        baseline_pace=baseline_pace,
        summaries=summaries,
    )


def wet_lap_proportion_by_driver(laps: pd.DataFrame) -> pd.DataFrame:
    required = {"Driver", "Compound"}
    missing = required.difference(laps.columns)
    if missing:
        raise ValueError(f"Laps are missing required columns: {sorted(missing)}")

    wet_laps = laps.dropna(subset=["Driver", "Compound"]).copy()
    if wet_laps.empty:
        return pd.DataFrame(
            columns=[
                "Driver",
                "Team",
                "TotalLapCount",
                "WetLapCount",
                "WetLapProportion",
            ]
        )

    wet_laps["Compound"] = wet_laps["Compound"].astype("string").str.upper()
    wet_laps["IsWetLap"] = wet_laps["Compound"].isin(WET_COMPOUNDS)
    group_columns = ["Driver"]
    if "Team" in wet_laps.columns:
        group_columns.append("Team")
    summary = (
        wet_laps.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            TotalLapCount=("Compound", "size"),
            WetLapCount=("IsWetLap", "sum"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    summary["WetLapCount"] = summary["WetLapCount"].astype(int)
    summary["WetLapProportion"] = summary["WetLapCount"] / summary["TotalLapCount"]
    return summary


def is_wet_race(
    wet_lap_summary: pd.DataFrame,
    *,
    wet_lap_proportion_skip_threshold: float,
) -> bool:
    _validate_wet_lap_threshold(wet_lap_proportion_skip_threshold)
    if wet_lap_summary.empty:
        return False
    median_wet_proportion = float(wet_lap_summary["WetLapProportion"].median())
    return median_wet_proportion > wet_lap_proportion_skip_threshold


def summarize_monte_carlo_race_performance(
    *,
    sample_parameters: pd.DataFrame,
    compound_degradation: pd.DataFrame,
    compound_delta: pd.DataFrame,
    team_compound_degradation: pd.DataFrame,
    baseline_pace: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    weights = sample_parameters[["SampleId", "Weight"]]
    return {
        "fuel_rate": _weighted_summary(
            sample_parameters,
            value_column="FuelRateSecondsPerLap",
            weight_column="Weight",
        ),
        "track_rate": _weighted_summary(
            sample_parameters,
            value_column="TrackRateSecondsPerLap",
            weight_column="Weight",
        ),
        "compound_degradation": _weighted_summary(
            compound_degradation.merge(weights, on="SampleId", how="left"),
            value_column="CompoundDegSecondsPerLap",
            weight_column="Weight",
            group_columns=("Compound",),
        ),
        "compound_delta": _weighted_summary(
            compound_delta.merge(weights, on="SampleId", how="left"),
            value_column="CompoundDeltaSeconds",
            weight_column="Weight",
            group_columns=("Compound", "CompoundDeltaReference"),
        ),
        "team_compound_degradation": _weighted_summary(
            team_compound_degradation.merge(weights, on="SampleId", how="left"),
            value_column="TeamCompoundDegSecondsPerLap",
            weight_column="Weight",
            group_columns=("Team", "Compound"),
        ),
        "baseline_pace": _weighted_summary(
            baseline_pace.merge(weights, on="SampleId", how="left"),
            value_column="CorrectedBaselinePaceSeconds",
            weight_column="Weight",
            group_columns=_baseline_summary_columns(baseline_pace),
        ),
    }


def _weighted_summary(
    values: pd.DataFrame,
    *,
    value_column: str,
    weight_column: str,
    group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if group_columns:
        grouped = values.groupby(list(group_columns), dropna=False, sort=True)
    else:
        grouped = [((), values)]
    for key, group in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(group_columns, key_values)}
        clean = group.dropna(subset=[value_column, weight_column])
        clean = clean.loc[clean[weight_column] > 0]
        row.update(
            {
                "P10": _weighted_quantile(clean[value_column], clean[weight_column], 0.10),
                "Median": _weighted_quantile(clean[value_column], clean[weight_column], 0.50),
                "P90": _weighted_quantile(clean[value_column], clean[weight_column], 0.90),
                "SampleCount": int(len(clean)),
                "WeightSum": float(clean[weight_column].sum()) if not clean.empty else 0.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    if values.empty:
        return float("nan")
    order = np.argsort(values.to_numpy(dtype="float64"))
    sorted_values = values.to_numpy(dtype="float64")[order]
    sorted_weights = weights.to_numpy(dtype="float64")[order]
    total_weight = float(sorted_weights.sum())
    if total_weight <= 0:
        return float("nan")
    cumulative = np.cumsum(sorted_weights)
    return float(sorted_values[np.searchsorted(cumulative, quantile * total_weight, side="left")])


def _add_race_fuel_proxy(laps: pd.DataFrame) -> pd.DataFrame:
    output = laps.copy()
    group_columns = [
        column for column in ("Year", "Round", "SessionName") if column in output.columns
    ]
    if group_columns:
        race_lap_count = output.groupby(group_columns)["LapNumber"].transform("max")
    else:
        race_lap_count = output["LapNumber"].max()
    output[FUEL_PROXY_LAPS_REMAINING] = (
        pd.to_numeric(race_lap_count, errors="coerce")
        - pd.to_numeric(output[FUEL_LAP_NUMBER_COLUMN], errors="coerce")
    )
    return output


def _baseline_summary_columns(baseline_pace: pd.DataFrame) -> tuple[str, ...]:
    if "Driver" in baseline_pace.columns:
        return ("Team", "Driver")
    return ("Team",)


def _validate_wet_lap_threshold(threshold: float) -> None:
    if threshold < 0 or threshold > 1:
        raise ValueError("wet_lap_proportion_skip_threshold must be between 0 and 1.")
