from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from wostrategy.model.fuel_consumption import FUEL_LAP_NUMBER_COLUMN
from wostrategy.model.long_run_performance import (
    DEFAULT_LONG_RUN_MODEL_CONFIG,
    EXPONENTIAL_TRACK_LONG_RUN_MODEL_CONFIG,
)
from wostrategy.model.tyre_degragation import TYRE_AGE_LAPS_COLUMN

LONG_RUN_MODEL_LINEAR_COMPONENTS = str(DEFAULT_LONG_RUN_MODEL_CONFIG["name"])
LONG_RUN_MODEL_EXPONENTIAL_TRACK = str(EXPONENTIAL_TRACK_LONG_RUN_MODEL_CONFIG["name"])
WET_COMPOUNDS = {"WET", "INTERMEDIATE", "INTER", "INTERS"}
TYRE_AGE_MODE_STINT = "stint"
TYRE_AGE_MODE_OVERALL = "overall"
TYRE_AGE_MODES = (TYRE_AGE_MODE_STINT, TYRE_AGE_MODE_OVERALL)


@dataclass(frozen=True)
class LinearLapTimeFit:
    model_name: str
    parameters: dict[str, float]
    x_columns: tuple[str, ...]
    y_column: str
    rmse_seconds: float
    formula: str
    config: dict[str, Any]

    def predict_frame(self, laps: pd.DataFrame) -> np.ndarray:
        x_column = self.x_columns[0]
        slope = self.parameters["tyre_slope_seconds_per_lap"]
        intercept = self.parameters["estimated_tyre_life_zero_seconds"]
        return intercept + (slope * laps[x_column].to_numpy(dtype="float64"))

    def predict_values(self, values: dict[str, float]) -> float:
        x_column = self.x_columns[0]
        return float(
            self.parameters["estimated_tyre_life_zero_seconds"]
            + (self.parameters["tyre_slope_seconds_per_lap"] * values[x_column])
        )


@dataclass(frozen=True)
class LongRunFit:
    driver: str
    team: str
    compound: str
    long_run_id: int
    stint: object
    model_name: str
    parameters: dict[str, float]
    model_fit: LinearLapTimeFit
    lap_count: int
    run_count: int
    reference_track_x: float
    estimated_first_lap_seconds: float
    rmse_seconds: float
    formula: str
    tyre_life_zero_lap_number: float
    original_lap_count: int
    outlier_lap_count: int
    fit_lap_indices: tuple[object, ...]

    def predict(self, values: dict[str, float]) -> float:
        return self.model_fit.predict_values(values)

    def predict_frame(self, laps: pd.DataFrame) -> pd.Series:
        return pd.Series(
            self.model_fit.predict_frame(laps),
            index=laps.index,
            dtype="float64",
        )


@dataclass(frozen=True)
class LongRunPerformanceResult:
    all_laps: pd.DataFrame
    filtered_laps: pd.DataFrame
    fit_summary: pd.DataFrame
    team_compound_summary: pd.DataFrame
    team_performance: pd.DataFrame
    fits: dict[tuple[str, str, str, int], LongRunFit]
    team_compound_correction_summary: pd.DataFrame
    compound_correction_stats: pd.DataFrame
    compound_reference_laps: pd.DataFrame


def calculate_long_run_performance(
    laps: pd.DataFrame,
    *,
    min_clean_air_laps: int,
    clean_mean_time_delta_seconds: float,
    clean_mean_time_delta_behind_seconds: float | None,
    quick_lap_threshold: float,
    model_name: str = LONG_RUN_MODEL_LINEAR_COMPONENTS,
    model_config: dict[str, Any] | None = None,
    dry_compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
    track_x_column: str = "LapNumber",
    reference_track_x: float | None = None,
    min_fit_laps: int | None = None,
    outlier_sigma: float | None = 2.5,
    min_fit_laps_after_outlier_filter: int = 6,
    combined_loss_slope_outlier_sigma: float | None = 1.5,
    combined_loss_slope_outlier_min_fits: int = 4,
    track_evolution_rate_seconds_per_lap: float | None = None,
) -> LongRunPerformanceResult | str:
    """Calculate long-run performance from stint estimates and quali track evolution.

    The flow is:
    1. Keep the existing quick-lap and clean-air filtering.
    2. Fit lap time linearly against tyre age for each driver stint.
    3. Estimate each stint at tyre life zero and infer the race lap where that
       tyre life zero occurred.
    4. Remove stint estimates whose fitted tyre slope is far from the grid.
    5. Choose a compound target lap closest to all remaining stint estimates.
    6. Correct each driver's own tyre-life-zero estimate to that target lap using
       the external quali track evolution rate.
    7. Aggregate to team performance by compound coverage and usage.
    """
    del model_name, model_config, track_x_column, reference_track_x

    if laps["Compound"].astype("string").str.upper().isin(WET_COMPOUNDS).any():
        return "Wet"

    prepared = _prepare_laps(
        laps,
        clean_mean_time_delta_seconds=clean_mean_time_delta_seconds,
        clean_mean_time_delta_behind_seconds=clean_mean_time_delta_behind_seconds,
        quick_lap_threshold=quick_lap_threshold,
        dry_compounds=dry_compounds,
    )
    filtered = select_consecutive_clean_air_runs(
        prepared,
        min_clean_air_laps=min_clean_air_laps,
    )
    if filtered.empty:
        raise ValueError("No consecutive clean-air long runs matched the configured filters.")
    prepared = _add_long_run_ids_to_all_laps(prepared, filtered)

    if min_fit_laps is None:
        min_fit_laps = max(2, min_clean_air_laps)

    fits: dict[tuple[str, str, str, int], LongRunFit] = {}
    fit_rows: list[dict[str, object]] = []
    for (team, driver, compound, stint), group in filtered.groupby(
        ["Team", "Driver", "Compound", "Stint"], sort=True
    ):
        if len(group) < min_fit_laps:
            continue
        if group[TYRE_AGE_LAPS_COLUMN].nunique(dropna=True) < 2:
            continue
        fit = fit_long_run_components(
            group,
            outlier_sigma=outlier_sigma,
            min_fit_laps_after_outlier_filter=min_fit_laps_after_outlier_filter,
        )
        fits[(team, driver, compound, fit.long_run_id)] = fit
        row: dict[str, object] = {
            "Team": team,
            "Driver": driver,
            "Compound": compound,
            "LongRunId": fit.long_run_id,
            "Stint": stint,
            "OriginalLapCount": fit.original_lap_count,
            "LapCount": fit.lap_count,
            "OutlierLapCount": fit.outlier_lap_count,
            "TyreLifeZeroLapNumber": fit.tyre_life_zero_lap_number,
            "EstimatedTyreLifeZeroSeconds": fit.estimated_first_lap_seconds,
            "TyreSlopeSecondsPerLap": fit.parameters["tyre_slope_seconds_per_lap"],
            "RMSESeconds": fit.rmse_seconds,
            "Formula": fit.formula,
        }
        fit_rows.append(row)

    if not fit_rows:
        raise ValueError("No driver stints had enough clean quick laps to fit.")

    fit_summary = pd.DataFrame(fit_rows).sort_values(
        ["Compound", "Team", "Driver", "Stint", "LongRunId"]
    )
    fit_summary = _mark_driver_estimate_outliers(
        fit_summary,
        slope_outlier_sigma=combined_loss_slope_outlier_sigma,
        slope_outlier_min_fits=combined_loss_slope_outlier_min_fits,
    )
    included_fit_summary = fit_summary.loc[
        fit_summary["EstimateIncludedInPerformance"]
    ].copy()
    if included_fit_summary.empty:
        raise ValueError("No driver stint estimates remained after sanity filtering.")

    team_compound_correction_summary = _driver_estimate_diagnostic_summary(fit_summary)
    compound_correction_stats = _track_evolution_correction_stats(
        fit_summary,
        track_evolution_rate_seconds_per_lap=track_evolution_rate_seconds_per_lap,
    )
    compound_reference_laps = _compound_reference_laps(included_fit_summary)
    team_compound_summary = _estimate_team_compound_at_reference_lap(
        included_fit_summary,
        track_evolution_rate_seconds_per_lap,
        compound_reference_laps,
    )
    compound_weights = _compound_lap_weights(prepared)
    team_performance = _team_performance_from_compound_availability(
        team_compound_summary,
        compound_weights,
    )

    return LongRunPerformanceResult(
        all_laps=prepared,
        filtered_laps=filtered,
        fit_summary=fit_summary,
        team_compound_summary=team_compound_summary,
        team_performance=team_performance,
        fits=fits,
        team_compound_correction_summary=team_compound_correction_summary,
        compound_correction_stats=compound_correction_stats,
        compound_reference_laps=compound_reference_laps,
    )


def select_consecutive_clean_air_runs(
    laps: pd.DataFrame,
    *,
    min_clean_air_laps: int,
) -> pd.DataFrame:
    if min_clean_air_laps <= 0:
        raise ValueError("min_clean_air_laps must be positive.")

    required = {"Driver", "Stint", "LapNumber", "IsCleanAirLongRunLap"}
    missing = required.difference(laps.columns)
    if missing:
        raise ValueError(f"Laps are missing required columns: {sorted(missing)}")

    selected = laps.copy()
    selected["LongRunId"] = pd.NA
    selected["LongRunLapNumber"] = pd.NA
    group_columns = ["Year", "Round", "SessionName", "Driver", "Stint"]
    group_columns = [column for column in group_columns if column in selected.columns]
    run_number = 0
    selected_indices: list[int] = []

    for _, group in selected.sort_values(group_columns + ["LapNumber"]).groupby(
        group_columns, dropna=False, sort=False
    ):
        clean = group["IsCleanAirLongRunLap"].fillna(False).astype(bool)
        block_id = clean.ne(clean.shift(fill_value=False)).cumsum()
        for _, block in group.loc[clean].groupby(block_id.loc[clean], sort=False):
            if len(block) < min_clean_air_laps:
                continue
            run_number += 1
            selected.loc[block.index, "LongRunId"] = run_number
            selected.loc[block.index, "LongRunLapNumber"] = range(1, len(block) + 1)
            selected_indices.extend(block.index.tolist())

    output = selected.loc[selected_indices].copy()
    output["LongRunId"] = pd.to_numeric(output["LongRunId"], errors="coerce")
    output["LongRunLapNumber"] = pd.to_numeric(output["LongRunLapNumber"], errors="coerce")
    return output


def select_clean_air_stints_as_whole(
    laps: pd.DataFrame,
    *,
    min_clean_air_laps: int,
) -> pd.DataFrame:
    if min_clean_air_laps <= 0:
        raise ValueError("min_clean_air_laps must be positive.")

    required = {"Driver", "Stint", "LapNumber", "IsCleanAirLongRunLap"}
    missing = required.difference(laps.columns)
    if missing:
        raise ValueError(f"Laps are missing required columns: {sorted(missing)}")

    selected = laps.copy()
    selected["LongRunId"] = pd.NA
    selected["LongRunLapNumber"] = pd.NA
    group_columns = ["Year", "Round", "SessionName", "Driver", "Stint"]
    group_columns = [column for column in group_columns if column in selected.columns]
    run_number = 0
    selected_indices: list[int] = []

    for _, group in selected.sort_values(group_columns + ["LapNumber"]).groupby(
        group_columns, dropna=False, sort=False
    ):
        clean_stint_laps = group.loc[
            group["IsCleanAirLongRunLap"].fillna(False).astype(bool)
        ]
        if len(clean_stint_laps) < min_clean_air_laps:
            continue
        run_number += 1
        selected.loc[clean_stint_laps.index, "LongRunId"] = run_number
        selected.loc[clean_stint_laps.index, "LongRunLapNumber"] = range(
            1,
            len(clean_stint_laps) + 1,
        )
        selected_indices.extend(clean_stint_laps.index.tolist())

    output = selected.loc[selected_indices].copy()
    output["LongRunId"] = pd.to_numeric(output["LongRunId"], errors="coerce")
    output["LongRunLapNumber"] = pd.to_numeric(output["LongRunLapNumber"], errors="coerce")
    return output


def fit_long_run_components(
    laps: pd.DataFrame,
    *,
    outlier_sigma: float | None = 2.5,
    min_fit_laps_after_outlier_filter: int = 6,
) -> LongRunFit:
    fit_laps = laps.dropna(subset=[TYRE_AGE_LAPS_COLUMN, "LapNumber", "LapTimeSeconds"])
    if len(fit_laps) < 2:
        raise ValueError("At least two laps are required to fit a driver stint.")

    original_lap_count = len(fit_laps)
    if outlier_sigma is not None:
        filtered_fit_laps = _filter_linear_fit_outliers(
            fit_laps,
            x_column=TYRE_AGE_LAPS_COLUMN,
            y_column="LapTimeSeconds",
            outlier_sigma=outlier_sigma,
            min_fit_laps_after_outlier_filter=min_fit_laps_after_outlier_filter,
        )
    else:
        filtered_fit_laps = fit_laps
    fit_laps = filtered_fit_laps

    x = fit_laps[TYRE_AGE_LAPS_COLUMN].to_numpy(dtype="float64")
    y = fit_laps["LapTimeSeconds"].to_numpy(dtype="float64")
    slope, intercept, rmse = _fit_linear_with_rmse(x, y)
    outlier_lap_count = original_lap_count - len(fit_laps)

    tyre_life_zero_lap_number = float(
        fit_laps["LapNumber"].to_numpy(dtype="float64")[0] - x[0]
    )
    driver = str(fit_laps["Driver"].iloc[0])
    team = str(fit_laps["Team"].iloc[0])
    compound = str(fit_laps["Compound"].iloc[0])
    long_run_id = int(fit_laps["LongRunId"].iloc[0]) if "LongRunId" in fit_laps else 0
    stint = fit_laps["Stint"].iloc[0] if "Stint" in fit_laps else pd.NA
    formula = (
        "lap_time = "
        f"{slope:.5f} * tyre_life + {intercept:.3f}"
    )
    parameters = {
        "tyre_slope_seconds_per_lap": float(slope),
        "estimated_tyre_life_zero_seconds": float(intercept),
    }
    model_fit = LinearLapTimeFit(
        model_name="linear_tyre_life_stint",
        parameters=parameters,
        x_columns=(TYRE_AGE_LAPS_COLUMN,),
        y_column="LapTimeSeconds",
        rmse_seconds=rmse,
        formula=formula,
        config={
            "terms": {
                "tyre": {
                    "model": "linear",
                    "x_column": TYRE_AGE_LAPS_COLUMN,
                    "label": "tyre_life",
                    "parameter": "tyre_slope_seconds_per_lap",
                }
            }
        },
    )
    return LongRunFit(
        driver=driver,
        team=team,
        compound=compound,
        long_run_id=long_run_id,
        stint=stint,
        model_name=model_fit.model_name,
        parameters=parameters,
        model_fit=model_fit,
        lap_count=len(fit_laps),
        run_count=(
            int(fit_laps["LongRunId"].nunique()) if "LongRunId" in fit_laps else 1
        ),
        reference_track_x=tyre_life_zero_lap_number,
        estimated_first_lap_seconds=float(intercept),
        rmse_seconds=rmse,
        formula=formula,
        tyre_life_zero_lap_number=tyre_life_zero_lap_number,
        original_lap_count=original_lap_count,
        outlier_lap_count=outlier_lap_count,
        fit_lap_indices=tuple(fit_laps.index.tolist()),
    )


def add_fitted_lap_times(
    laps: pd.DataFrame,
    fits: dict[tuple[str, str, str, int], LongRunFit],
    *,
    track_x_column: str = "LapNumber",
) -> pd.DataFrame:
    del track_x_column
    output = laps.copy()
    output["FittedLapTimeSeconds"] = np.nan
    for key, fit in fits.items():
        team, driver, compound, long_run_id = key
        mask = (
            (output["Team"] == team)
            & (output["Driver"] == driver)
            & (output["Compound"] == compound)
        )
        if "Stint" in output.columns:
            mask = mask & (output["Stint"] == fit.stint)
        elif "LongRunId" in output.columns:
            mask = mask & (output["LongRunId"] == long_run_id)
        if fit.fit_lap_indices:
            mask = mask & output.index.isin(fit.fit_lap_indices)
        if not mask.any():
            continue
        output.loc[mask, "FittedLapTimeSeconds"] = fit.predict_frame(output.loc[mask])
    return output


def _filter_linear_fit_outliers(
    fit_laps: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    outlier_sigma: float,
    min_fit_laps_after_outlier_filter: int,
) -> pd.DataFrame:
    if outlier_sigma <= 0:
        raise ValueError("outlier_sigma must be positive or None.")
    if min_fit_laps_after_outlier_filter < 2:
        raise ValueError("min_fit_laps_after_outlier_filter must be at least 2.")
    if len(fit_laps) <= min_fit_laps_after_outlier_filter:
        return fit_laps

    x = fit_laps[x_column].to_numpy(dtype="float64")
    y = fit_laps[y_column].to_numpy(dtype="float64")
    slope, intercept, _ = _fit_linear_with_rmse(x, y)
    residuals = y - ((slope * x) + intercept)
    residual_center = float(np.median(residuals))
    absolute_deviation = np.abs(residuals - residual_center)
    mad = float(np.median(absolute_deviation))
    if mad > 0:
        robust_sigma = 1.4826 * mad
    else:
        robust_sigma = float(np.std(residuals))
    if robust_sigma <= 1e-6:
        return fit_laps

    keep = absolute_deviation <= (outlier_sigma * robust_sigma)
    if int(keep.sum()) < min_fit_laps_after_outlier_filter:
        return fit_laps
    return fit_laps.loc[keep].copy()


def _fit_linear_with_rmse(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = (slope * x) + intercept
    rmse = float(np.sqrt(np.mean((y - fitted) ** 2)))
    return float(slope), float(intercept), rmse


def _prepare_laps(
    laps: pd.DataFrame,
    *,
    clean_mean_time_delta_seconds: float,
    clean_mean_time_delta_behind_seconds: float | None,
    quick_lap_threshold: float,
    dry_compounds: tuple[str, ...],
    tyre_age_mode: str = TYRE_AGE_MODE_STINT,
) -> pd.DataFrame:
    if quick_lap_threshold <= 0:
        raise ValueError("quick_lap_threshold must be positive.")
    if tyre_age_mode not in TYRE_AGE_MODES:
        options = ", ".join(TYRE_AGE_MODES)
        raise ValueError(f"Unknown tyre_age_mode {tyre_age_mode!r}. Options: {options}.")

    required = {
        "Driver",
        "Team",
        "LapNumber",
        "LapTime",
        "Compound",
        "Stint",
        "PitOutTime",
        "PitInTime",
        "MeanTimeDeltaToDriverAhead",
    }
    if clean_mean_time_delta_behind_seconds is not None:
        required.add("MeanTimeDeltaToDriverBehind")
    missing = required.difference(laps.columns)
    if missing:
        raise ValueError(f"Laps are missing required columns: {sorted(missing)}")

    prepared = laps.copy()
    prepared["Compound"] = prepared["Compound"].astype("string").str.upper()
    dry_compound_names = tuple(c.upper() for c in dry_compounds)
    prepared = prepared.loc[prepared["Compound"].isin(dry_compound_names)]
    prepared = prepared.dropna(subset=["LapTime", "Driver", "Team", "Compound"]).copy()
    if "TrackStatus" in prepared.columns:
        track_status = prepared["TrackStatus"].astype("string").str.strip()
        prepared["IsGreenTrackStatus"] = track_status.eq("1")
    else:
        prepared["IsGreenTrackStatus"] = True
    prepared["LapTimeSeconds"] = prepared["LapTime"].dt.total_seconds()
    prepared["IsOutLap"] = prepared["PitOutTime"].notna()
    prepared["IsInLap"] = prepared["PitInTime"].notna()
    valid_fast_lap_candidates = ~prepared["IsOutLap"] & ~prepared["IsInLap"]
    driver_fastest = (
        prepared.loc[valid_fast_lap_candidates]
        .groupby("Driver")["LapTimeSeconds"]
        .transform("min")
    )
    prepared["DriverFastestRaceLapSeconds"] = pd.NA
    prepared.loc[valid_fast_lap_candidates, "DriverFastestRaceLapSeconds"] = driver_fastest
    prepared["DriverFastestRaceLapSeconds"] = pd.to_numeric(
        prepared["DriverFastestRaceLapSeconds"],
        errors="coerce",
    )
    prepared["IsQuickLap"] = (
        prepared["LapTimeSeconds"]
        <= quick_lap_threshold * prepared["DriverFastestRaceLapSeconds"]
    )
    if tyre_age_mode == TYRE_AGE_MODE_OVERALL:
        if "TyreLife" not in prepared.columns:
            raise ValueError("tyre_age_mode='overall' requires a TyreLife column.")
        prepared[TYRE_AGE_LAPS_COLUMN] = (
            pd.to_numeric(prepared["TyreLife"], errors="coerce") - 1
        )
    elif "StintLapNumber" in prepared.columns:
        prepared[TYRE_AGE_LAPS_COLUMN] = (
            pd.to_numeric(prepared["StintLapNumber"], errors="coerce") - 1
        )
    else:
        prepared[TYRE_AGE_LAPS_COLUMN] = (
            prepared.sort_values(["Driver", "Stint", "LapNumber"])
            .groupby(["Driver", "Stint"], dropna=False)
            .cumcount()
        )
    prepared[FUEL_LAP_NUMBER_COLUMN] = pd.to_numeric(
        prepared["LapNumber"],
        errors="coerce",
    )
    has_gap_summary = _has_any_gap_summary(prepared)
    ahead_gap = prepared["MeanTimeDeltaToDriverAhead"].where(
        prepared["MeanTimeDeltaToDriverAhead"].notna(),
        np.where(has_gap_summary, np.inf, np.nan),
    )
    clean_air_mask = (
        (ahead_gap > clean_mean_time_delta_seconds)
        & prepared["IsQuickLap"]
        & prepared["IsGreenTrackStatus"]
        & ~prepared["IsOutLap"]
        & ~prepared["IsInLap"]
    )
    if clean_mean_time_delta_behind_seconds is not None:
        # Missing per-lap gap values can mean there was no relevant car ahead or
        # behind, which is clean air. Only allow that interpretation when some
        # telemetry gap summary exists for the lap, so unmerged/no-telemetry laps
        # do not pass the clean-air filter.
        behind_gap = prepared["MeanTimeDeltaToDriverBehind"].where(
            prepared["MeanTimeDeltaToDriverBehind"].notna(),
            np.where(has_gap_summary, np.inf, np.nan),
        )
        clean_air_mask = clean_air_mask & (
            behind_gap > clean_mean_time_delta_behind_seconds
        )
    prepared["IsCleanAirLongRunLap"] = clean_air_mask
    return prepared


def _has_any_gap_summary(laps: pd.DataFrame) -> pd.Series:
    gap_columns = [
        column
        for column in (
            "MinTimeDeltaToDriverAhead",
            "MeanTimeDeltaToDriverAhead",
            "MinDistanceToDriverAhead",
            "MeanDistanceToDriverAhead",
            "MinTimeDeltaToDriverBehind",
            "MeanTimeDeltaToDriverBehind",
            "MinDistanceToDriverBehind",
            "MeanDistanceToDriverBehind",
        )
        if column in laps.columns
    ]
    if not gap_columns:
        return pd.Series(False, index=laps.index)
    return laps.loc[:, gap_columns].notna().any(axis=1)


def _add_long_run_ids_to_all_laps(
    laps: pd.DataFrame,
    filtered_laps: pd.DataFrame,
) -> pd.DataFrame:
    output = laps.copy()
    output["LongRunId"] = pd.NA
    output["LongRunLapNumber"] = pd.NA
    output.loc[filtered_laps.index, "LongRunId"] = filtered_laps["LongRunId"]
    output.loc[filtered_laps.index, "LongRunLapNumber"] = filtered_laps["LongRunLapNumber"]
    output["LongRunId"] = pd.to_numeric(output["LongRunId"], errors="coerce")
    output["LongRunLapNumber"] = pd.to_numeric(output["LongRunLapNumber"], errors="coerce")
    return output


def _mark_driver_estimate_outliers(
    fit_summary: pd.DataFrame,
    *,
    slope_outlier_sigma: float | None,
    slope_outlier_min_fits: int,
) -> pd.DataFrame:
    output = fit_summary.copy()
    output["EstimateIncludedInPerformance"] = True
    output["EstimateOutlierReason"] = ""

    required_values = (
        output["TyreLifeZeroLapNumber"].notna()
        & output["EstimatedTyreLifeZeroSeconds"].notna()
        & output["TyreSlopeSecondsPerLap"].notna()
    )
    output.loc[~required_values, "EstimateIncludedInPerformance"] = False
    output.loc[~required_values, "EstimateOutlierReason"] = "missing fit value"

    if slope_outlier_sigma is None:
        return output
    if slope_outlier_sigma <= 0:
        raise ValueError("combined_loss_slope_outlier_sigma must be positive or None.")
    if slope_outlier_min_fits < 3:
        raise ValueError("combined_loss_slope_outlier_min_fits must be at least 3.")

    for compound, group in output.loc[required_values].groupby("Compound", sort=True):
        if len(group) < slope_outlier_min_fits:
            continue
        slopes = group["TyreSlopeSecondsPerLap"].to_numpy(dtype="float64")
        finite = np.isfinite(slopes)
        if int(finite.sum()) < slope_outlier_min_fits:
            continue

        finite_slopes = slopes[finite]
        center = float(np.median(finite_slopes))
        absolute_deviation = np.abs(finite_slopes - center)
        mad = float(np.median(absolute_deviation))
        if mad > 1e-9:
            robust_sigma = 1.4826 * mad
        else:
            robust_sigma = float(np.std(finite_slopes))
        if robust_sigma <= 1e-9:
            continue

        finite_outliers = absolute_deviation > (slope_outlier_sigma * robust_sigma)
        finite_index = group.iloc[np.where(finite)[0]].index
        reason = (
            f"{compound} tyre slope outside {slope_outlier_sigma:g} robust sigma "
            f"from field median {center:.5f}s/lap"
        )
        for row_index, is_outlier in zip(finite_index, finite_outliers):
            if not is_outlier:
                continue
            output.at[row_index, "EstimateIncludedInPerformance"] = False
            output.at[row_index, "EstimateOutlierReason"] = reason
    return output


def _driver_estimate_diagnostic_summary(fit_summary: pd.DataFrame) -> pd.DataFrame:
    return fit_summary.sort_values(
        ["Compound", "Team", "Driver", "Stint", "LongRunId"]
    ).reset_index(drop=True)


def _track_evolution_correction_stats(
    fit_summary: pd.DataFrame,
    *,
    track_evolution_rate_seconds_per_lap: float | None,
) -> pd.DataFrame:
    included = fit_summary["EstimateIncludedInPerformance"].fillna(False).astype(bool)
    return pd.DataFrame(
        [
            {
                "CorrectionGroup": "QUALI_TRACK_EVOLUTION",
                "DriverStintEstimateCount": int(len(fit_summary)),
                "IncludedDriverStintEstimateCount": int(included.sum()),
                "ExcludedDriverStintEstimateCount": int((~included).sum()),
                "TrackEvolutionRateSecondsPerLap": (
                    np.nan
                    if track_evolution_rate_seconds_per_lap is None
                    else float(track_evolution_rate_seconds_per_lap)
                ),
            }
        ]
    )


def _compound_reference_laps(fit_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for compound, group in fit_summary.dropna(
        subset=["TyreLifeZeroLapNumber"]
    ).groupby("Compound", sort=True):
        values = group["TyreLifeZeroLapNumber"].to_numpy(dtype="float64")
        median_value = float(np.median(values))
        value_summary = (
            group.groupby("TyreLifeZeroLapNumber", as_index=False)
            .agg(
                RecordCount=("TyreLifeZeroLapNumber", "size"),
                TeamCount=("Team", "nunique"),
            )
        )
        candidates = value_summary.copy()
        candidates["DistanceToMedian"] = (
            candidates["TyreLifeZeroLapNumber"].astype("float64") - median_value
        ).abs()
        chosen = candidates.sort_values(
            [
                "DistanceToMedian",
                "TeamCount",
                "RecordCount",
                "TyreLifeZeroLapNumber",
            ],
            ascending=[True, False, False, True],
            kind="mergesort",
        ).iloc[0]
        selection = (
            "closest_shared_to_median"
            if int(chosen["TeamCount"]) > 1
            else "closest_to_median"
        )
        rows.append(
            {
                "Compound": compound,
                "ReferenceTyreLifeZeroLapNumber": float(
                    chosen["TyreLifeZeroLapNumber"]
                ),
                "ReferenceLapSelection": selection,
                "ReferenceLapTeamCount": int(chosen["TeamCount"]),
                "ReferenceLapRecordCount": int(chosen["RecordCount"]),
                "RecordCount": len(group),
                "TeamCount": int(group["Team"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("Compound").reset_index(drop=True)


def _estimate_team_compound_at_reference_lap(
    fit_summary: pd.DataFrame,
    track_evolution_rate_seconds_per_lap: float | None,
    compound_reference_laps: pd.DataFrame,
) -> pd.DataFrame:
    track_rate = (
        0.0
        if track_evolution_rate_seconds_per_lap is None
        else float(track_evolution_rate_seconds_per_lap)
    )
    if not np.isfinite(track_rate):
        raise ValueError("track_evolution_rate_seconds_per_lap must be finite or None.")

    records = fit_summary.dropna(
        subset=["Compound", "TyreLifeZeroLapNumber", "EstimatedTyreLifeZeroSeconds"]
    ).copy()
    records = records.merge(
        compound_reference_laps[
            [
                "Compound",
                "ReferenceTyreLifeZeroLapNumber",
                "ReferenceLapSelection",
                "ReferenceLapTeamCount",
                "ReferenceLapRecordCount",
            ]
        ],
        on="Compound",
        how="left",
    )
    if records.empty:
        return pd.DataFrame(
            columns=[
                "Team",
                "Compound",
                "ReferenceTyreLifeZeroLapNumber",
                "ReferenceLapSelection",
                "ReferenceLapTeamCount",
                "ReferenceLapRecordCount",
                "EstimatedReferenceLapSeconds",
                "TrackEvolutionRateSecondsPerLap",
                "AverageTrackEvolutionCorrectionSeconds",
                "DriverStintEstimateCount",
            ]
        )

    records["TrackEvolutionCorrectionSeconds"] = track_rate * (
        records["ReferenceTyreLifeZeroLapNumber"] - records["TyreLifeZeroLapNumber"]
    )
    records["CorrectedReferenceLapSeconds"] = (
        records["EstimatedTyreLifeZeroSeconds"]
        - records["TrackEvolutionCorrectionSeconds"]
    )
    records["TrackEvolutionRateSecondsPerLap"] = track_rate
    estimated = (
        records.groupby(["Team", "Compound"], as_index=False)
        .agg(
            ReferenceTyreLifeZeroLapNumber=(
                "ReferenceTyreLifeZeroLapNumber",
                "first",
            ),
            ReferenceLapSelection=("ReferenceLapSelection", "first"),
            ReferenceLapTeamCount=("ReferenceLapTeamCount", "first"),
            ReferenceLapRecordCount=("ReferenceLapRecordCount", "first"),
            EstimatedReferenceLapSeconds=("CorrectedReferenceLapSeconds", "mean"),
            TrackEvolutionRateSecondsPerLap=(
                "TrackEvolutionRateSecondsPerLap",
                "first",
            ),
            AverageTrackEvolutionCorrectionSeconds=(
                "TrackEvolutionCorrectionSeconds",
                "mean",
            ),
            DriverStintEstimateCount=("CorrectedReferenceLapSeconds", "size"),
        )
    )
    return estimated.sort_values(
        ["Compound", "EstimatedReferenceLapSeconds", "Team"]
    ).reset_index(drop=True)


def _compound_lap_weights(laps: pd.DataFrame) -> pd.DataFrame:
    race_laps = laps.loc[~laps["IsOutLap"] & ~laps["IsInLap"]].copy()
    return (
        race_laps.groupby(["Team", "Compound"], as_index=False)
        .size()
        .rename(columns={"size": "CompoundLapCount"})
    )


def _team_performance_from_compound_availability(
    team_compound_summary: pd.DataFrame,
    compound_weights: pd.DataFrame,
) -> pd.DataFrame:
    if team_compound_summary.empty:
        return pd.DataFrame(
            columns=[
                "Team",
                "LongRunPerformanceSeconds",
                "WeightedLapCount",
                "CompoundsIncluded",
                "PerformanceBasis",
            ]
        )

    team_count = team_compound_summary["Team"].nunique()
    compound_team_counts = team_compound_summary.groupby("Compound")["Team"].nunique()
    complete_compounds = compound_team_counts.index.tolist()
    team_compound_counts = team_compound_summary.groupby("Team")["Compound"].nunique()
    all_teams_have_all_compounds = (
        len(complete_compounds) > 1
        and (compound_team_counts == team_count).all()
        and (team_compound_counts == len(complete_compounds)).all()
    )

    if all_teams_have_all_compounds:
        selected = team_compound_summary.copy()
        basis = "weighted_all_compounds"
    else:
        selected_compound = compound_team_counts.sort_values(
            ascending=False,
            kind="mergesort",
        ).index[0]
        selected = team_compound_summary.loc[
            team_compound_summary["Compound"] == selected_compound
        ].copy()
        basis = f"single_compound_{selected_compound}"

    merged = selected.merge(compound_weights, on=["Team", "Compound"], how="left")
    merged["CompoundLapCount"] = merged["CompoundLapCount"].fillna(1)
    merged.loc[merged["CompoundLapCount"] <= 0, "CompoundLapCount"] = 1
    merged["WeightedSeconds"] = (
        merged["EstimatedReferenceLapSeconds"] * merged["CompoundLapCount"]
    )
    team_performance = (
        merged.groupby("Team", as_index=False)
        .agg(
            LongRunPerformanceSeconds=("WeightedSeconds", "sum"),
            WeightedLapCount=("CompoundLapCount", "sum"),
            CompoundsIncluded=("Compound", "nunique"),
            Compounds=("Compound", _join_strings),
        )
    )
    team_performance["LongRunPerformanceSeconds"] = (
        team_performance["LongRunPerformanceSeconds"]
        / team_performance["WeightedLapCount"].replace(0, np.nan)
    )
    team_performance["PerformanceBasis"] = basis
    return team_performance.sort_values("LongRunPerformanceSeconds").reset_index(drop=True)


def _join_strings(values: pd.Series) -> str:
    return ",".join(sorted(values.dropna().astype(str).unique()))


__all__ = [
    "LONG_RUN_MODEL_EXPONENTIAL_TRACK",
    "LONG_RUN_MODEL_LINEAR_COMPONENTS",
    "TYRE_AGE_MODE_OVERALL",
    "TYRE_AGE_MODE_STINT",
    "TYRE_AGE_MODES",
    "WET_COMPOUNDS",
    "LinearLapTimeFit",
    "LongRunFit",
    "LongRunPerformanceResult",
    "add_fitted_lap_times",
    "calculate_long_run_performance",
    "fit_long_run_components",
    "select_clean_air_stints_as_whole",
    "select_consecutive_clean_air_runs",
]
