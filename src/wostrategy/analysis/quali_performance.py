from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .push_laps import add_push_lap_flags, fresh_tyre_mask, select_top_drivers
from wostrategy.model.track_evolution import (
    LINEAR_TRACK_EVOLUTION_MODEL,
    TRACK_EVO_CORRECTED_LAP_TIME,
    TRACK_EVO_CORRECTED_LAP_TIME_SECONDS,
    TRACK_EVO_CORRECTION_SECONDS,
    TrackEvolutionFit,
    add_track_evolution_correction,
    dominant_compound,
    fit_compound_track_evolution,
    get_track_evolution_model,
)


FORMAL_QUALIFYING_SESSION = "Q"
WET_COMPOUNDS = {"WET", "INTERMEDIATE", "INTER", "INTERS"}
QUALIFYING_PART = "QualifyingPart"
RESULT_TYPE = "ResultType"
DRIVER_COUNT = "DriverCount"
TEAMMATE_DELTA_SECONDS = "TeammateDeltaSeconds"
TEAMMATE_DELTA_PERCENT = "TeammateDeltaPercent"
AVERAGE_MODE_NOTE = "AverageModeNote"
QUICK_LAP_NUMBER = "QuickLapNumber"
SECTOR_TIME_COLUMNS = {
    "S1": "Sector1Time",
    "S2": "Sector2Time",
    "S3": "Sector3Time",
}
CORRECTED_SECTOR_SECONDS = {
    sector: f"track_evo_corrected_{sector.lower()}_seconds"
    for sector in SECTOR_TIME_COLUMNS
}
CORRECTED_SECTOR_TIME = {
    sector: f"track_evo_corrected_{sector.lower()}"
    for sector in SECTOR_TIME_COLUMNS
}


@dataclass(frozen=True)
class QualiPerformanceResult:
    laps: pd.DataFrame
    eligible_laps: pd.DataFrame
    quickest_drivers: pd.DataFrame
    quickest_teams: pd.DataFrame
    dominant_compound: str
    evolution_fit_model: str
    evolution_fit_parameters: dict[str, float]
    evolution_rate_seconds_per_lap: float
    reference_session_lap_order: int
    evolution_drivers: list[str] | None
    track_evolution_x_column: str = "SessionLapOrder"
    track_evolution_slope_unit: str = "s/lap"
    lap_time_only: bool = False


class QualiPerformanceAnalyzer:
    def __init__(
        self,
        *,
        quick_lap_threshold: float,
        clean_min_time_delta_seconds: float | None,
        clean_mean_time_delta_seconds: float | None,
        dry_compounds: tuple[str, ...],
        new_tyre_only: bool,
        top_driver_count: int | None,
        track_evolution_fit: str,
        last_quali_part_only: bool = False,
        lap_time_only: bool = False,
        track_evolution_quick_lap_number: bool = False,
    ) -> None:
        self.quick_lap_threshold = quick_lap_threshold
        self.clean_min_time_delta_seconds = clean_min_time_delta_seconds
        self.clean_mean_time_delta_seconds = clean_mean_time_delta_seconds
        self.dry_compounds = dry_compounds
        self.new_tyre_only = new_tyre_only
        self.last_quali_part_only = last_quali_part_only
        self.top_driver_count = top_driver_count
        self.track_evolution_fit = track_evolution_fit
        self.lap_time_only = lap_time_only
        self.track_evolution_quick_lap_number = track_evolution_quick_lap_number

    def calculate(self, laps: pd.DataFrame) -> QualiPerformanceResult | str:
        return calculate_quali_performance(
            laps,
            quick_lap_threshold=self.quick_lap_threshold,
            clean_min_time_delta_seconds=self.clean_min_time_delta_seconds,
            clean_mean_time_delta_seconds=self.clean_mean_time_delta_seconds,
            dry_compounds=self.dry_compounds,
            new_tyre_only=self.new_tyre_only,
            top_driver_count=self.top_driver_count,
            track_evolution_fit=self.track_evolution_fit,
            last_quali_part_only=self.last_quali_part_only,
            lap_time_only=self.lap_time_only,
            track_evolution_quick_lap_number=self.track_evolution_quick_lap_number,
        )


def calculate_quali_performance(
    laps: pd.DataFrame,
    *,
    quick_lap_threshold: float,
    clean_min_time_delta_seconds: float | None,
    clean_mean_time_delta_seconds: float | None,
    dry_compounds: tuple[str, ...],
    new_tyre_only: bool,
    top_driver_count: int | None,
    track_evolution_fit: str,
    last_quali_part_only: bool = False,
    lap_time_only: bool = False,
    track_evolution_quick_lap_number: bool = False,
) -> QualiPerformanceResult | str:
    """Return corrected quickest quali laps, or ``Wet`` if wet/inter tyres are present."""
    _require_columns(laps, {"Team", "Driver", "LapNumber", "LapTime", "Compound"})

    prepared = laps.copy()
    prepared = add_qualifying_part(laps, prepared)
    prepared["Compound"] = prepared["Compound"].astype("string").str.upper()
    if prepared["Compound"].isin(WET_COMPOUNDS).any():
        return "Wet"

    prepared = add_push_lap_flags(
        prepared,
        quick_lap_threshold=quick_lap_threshold,
        clean_min_time_delta_seconds=clean_min_time_delta_seconds,
        clean_mean_time_delta_seconds=clean_mean_time_delta_seconds,
        lap_time_only=lap_time_only,
    )
    prepared = add_quick_lap_numbers(prepared)
    dry_compounds = tuple(compound.upper() for compound in dry_compounds)
    push_laps = prepared.loc[
        prepared["IsPushLap"] & prepared["Compound"].isin(dry_compounds)
    ].copy()
    if new_tyre_only:
        _require_columns(prepared, {"FreshTyre"})
        push_laps = push_laps.loc[fresh_tyre_mask(push_laps["FreshTyre"])].copy()
    if push_laps.empty:
        tyre_filter = " on new tyres" if new_tyre_only else ""
        raise ValueError(f"No dry push laps{tyre_filter} available after filtering.")

    evolution_laps, evolution_drivers = filter_evolution_laps_by_top_drivers(
        push_laps,
        prepared,
        top_driver_count,
    )
    dominant = dominant_compound(evolution_laps["Compound"], require_majority=True)
    if dominant is None:
        raise ValueError("No dominant compound found.")
    evolution_model = get_track_evolution_model(track_evolution_fit)
    track_evolution_x_column = (
        QUICK_LAP_NUMBER if track_evolution_quick_lap_number else "SessionLapOrder"
    )
    track_evolution_slope_unit = (
        "s/quick lap" if track_evolution_quick_lap_number else "s/lap"
    )
    evolution_fit = fit_compound_track_evolution(
        evolution_laps,
        compound=dominant,
        model=evolution_model,
        x_column=track_evolution_x_column,
        y_column="LapTimeSeconds",
        slope_unit=track_evolution_slope_unit,
    )
    reference_lap_order = int(push_laps[track_evolution_x_column].max())

    prepared = add_track_evolution_correction(
        prepared,
        model=evolution_model,
        fit=evolution_fit,
        reference_session_lap_order=reference_lap_order,
    )
    prepared = add_corrected_sector_times(prepared)
    corrected_push_laps = prepared.loc[
        prepared["IsPushLap"] & prepared["Compound"].isin(dry_compounds)
    ].copy()
    if new_tyre_only:
        corrected_push_laps = corrected_push_laps.loc[
            fresh_tyre_mask(corrected_push_laps["FreshTyre"])
        ].copy()
    performance_laps = (
        latest_qualifying_part_laps(corrected_push_laps, reference_laps=prepared)
        if last_quali_part_only
        else corrected_push_laps
    )
    quickest_drivers = quickest_driver_laps(performance_laps)
    quickest_teams = quickest_team_laps(quickest_drivers)

    return QualiPerformanceResult(
        laps=prepared,
        eligible_laps=performance_laps,
        quickest_drivers=quickest_drivers,
        quickest_teams=quickest_teams,
        dominant_compound=dominant,
        evolution_fit_model=evolution_model.name,
        evolution_fit_parameters=evolution_fit.parameters,
        evolution_rate_seconds_per_lap=evolution_fit.evolution_rate_seconds_per_lap,
        reference_session_lap_order=reference_lap_order,
        evolution_drivers=evolution_drivers,
        track_evolution_x_column=track_evolution_x_column,
        track_evolution_slope_unit=track_evolution_slope_unit,
        lap_time_only=lap_time_only,
    )


def relative_team_pace_rows(
    *,
    result: QualiPerformanceResult,
    year: int,
    race: int,
    target_team: str,
    teammate_delta_threshold_percent: float | None,
    calculate_best_sectors: bool,
) -> list[dict[str, object]]:
    event_name = _event_name_from_laps(result.laps)
    team_pace = team_fastest_and_average_rows(
        result.quickest_drivers,
        teammate_delta_threshold_percent=teammate_delta_threshold_percent,
    )
    if calculate_best_sectors:
        team_pace = pd.concat(
            [team_pace, team_best_sector_rows(result.eligible_laps)],
            ignore_index=True,
        )

    records: list[dict[str, object]] = []
    for result_type, result_type_rows in team_pace.groupby(RESULT_TYPE, sort=False):
        target_row = result_type_rows.loc[result_type_rows["Team"] == target_team]
        if target_row.empty:
            available = ", ".join(sorted(result_type_rows["Team"].dropna().unique()))
            raise ValueError(
                f"Target team {target_team!r} not found in race {race} "
                f"for {result_type}. Available teams: {available}"
            )
        reference_seconds = float(target_row[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS].iloc[0])

        for _, row in result_type_rows.iterrows():
            records.append(
                {
                    "Year": year,
                    "Race": race,
                    "EventName": event_name,
                    RESULT_TYPE: row[RESULT_TYPE],
                    "Team": row["Team"],
                    "Driver": row["Driver"],
                    DRIVER_COUNT: row[DRIVER_COUNT],
                    TEAMMATE_DELTA_SECONDS: row[TEAMMATE_DELTA_SECONDS],
                    TEAMMATE_DELTA_PERCENT: row[TEAMMATE_DELTA_PERCENT],
                    AVERAGE_MODE_NOTE: row[AVERAGE_MODE_NOTE],
                    "LapNumber": row["LapNumber"],
                    QUALIFYING_PART: row[QUALIFYING_PART],
                    TRACK_EVO_CORRECTED_LAP_TIME_SECONDS: row[
                        TRACK_EVO_CORRECTED_LAP_TIME_SECONDS
                    ],
                    "PercentageToTargetTeam": (
                        float(row[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS])
                        / reference_seconds
                        * 100.0
                    ),
                }
            )
    return records


def _event_name_from_laps(laps: pd.DataFrame) -> str | None:
    for column in ("EventName", "RaceName", "EventLocation", "EventCountry"):
        if column not in laps.columns:
            continue
        values = laps[column].dropna().astype(str)
        if not values.empty:
            return values.iloc[0]
    return None


def compound_evolution_rate(push_laps: pd.DataFrame, compound: str) -> float:
    fit = fit_compound_track_evolution(
        push_laps,
        compound=compound,
        model=get_track_evolution_model(LINEAR_TRACK_EVOLUTION_MODEL),
        x_column="SessionLapOrder",
        y_column="LapTimeSeconds",
        slope_unit="s/lap",
    )
    return fit.evolution_rate_seconds_per_lap


def add_quick_lap_numbers(laps: pd.DataFrame) -> pd.DataFrame:
    _require_columns(laps, {"IsQuickLap", "SessionLapOrder"})
    numbered = laps.copy()
    numbered[QUICK_LAP_NUMBER] = pd.NA
    quick_laps = numbered.loc[numbered["IsQuickLap"]].sort_values(
        ["SessionLapOrder", "Driver"]
    )
    numbered.loc[quick_laps.index, QUICK_LAP_NUMBER] = range(1, len(quick_laps) + 1)
    numbered[QUICK_LAP_NUMBER] = pd.to_numeric(
        numbered[QUICK_LAP_NUMBER],
        errors="coerce",
    )
    return numbered


def filter_evolution_laps_by_top_drivers(
    push_laps: pd.DataFrame,
    ranking_laps: pd.DataFrame,
    top_driver_count: int | None,
) -> tuple[pd.DataFrame, list[str] | None]:
    top_drivers = select_top_drivers(ranking_laps, top_driver_count)
    if top_drivers is None:
        return push_laps, None

    filtered = push_laps.loc[push_laps["Driver"].isin(top_drivers)].copy()
    if filtered.empty:
        raise ValueError("No push laps remain after applying the top-driver filter.")
    return filtered, top_drivers


def add_qualifying_part(source_laps: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    prepared = prepared.copy()
    if QUALIFYING_PART in prepared.columns and prepared[QUALIFYING_PART].notna().any():
        return prepared

    existing_column = _first_existing_column(
        prepared,
        ("QualifyingSession", "SessionPart", "QSession"),
    )
    if existing_column is not None:
        prepared[QUALIFYING_PART] = prepared[existing_column].astype("string")
        return prepared

    split_getter = getattr(source_laps, "split_qualifying_sessions", None)
    if callable(split_getter):
        try:
            split_sessions = split_getter()
        except Exception:
            split_sessions = None
        if split_sessions is not None:
            prepared[QUALIFYING_PART] = pd.NA
            for label, part_laps in zip(("Q1", "Q2", "Q3"), split_sessions):
                if part_laps is None or part_laps.empty:
                    continue
                prepared.loc[prepared.index.intersection(part_laps.index), QUALIFYING_PART] = label
            if prepared[QUALIFYING_PART].notna().any():
                return prepared

    prepared[QUALIFYING_PART] = fallback_qualifying_part(prepared)
    return prepared


def fallback_qualifying_part(laps: pd.DataFrame) -> pd.Series:
    if "LapStartTime" not in laps.columns:
        return pd.Series(pd.NA, index=laps.index, dtype="string")

    ordered_index = laps.dropna(subset=["LapStartTime"]).sort_values("LapStartTime").index
    labels = pd.Series(pd.NA, index=laps.index, dtype="string")
    if ordered_index.empty:
        return labels

    for label, part_index in zip(("Q1", "Q2", "Q3"), np.array_split(ordered_index, 3)):
        labels.loc[part_index] = label
    return labels


def add_track_evolution_correction_from_rate(
    laps: pd.DataFrame,
    *,
    evolution_rate_seconds_per_lap: float,
    reference_session_lap_order: int,
) -> pd.DataFrame:
    fit = TrackEvolutionFit(
        model_name=LINEAR_TRACK_EVOLUTION_MODEL,
        parameters={
            "slope": -float(evolution_rate_seconds_per_lap),
            "intercept_seconds": 0.0,
        },
        x_column="SessionLapOrder",
        y_column="LapTimeSeconds",
        slope_unit="s/lap",
    )
    return add_track_evolution_correction(
        laps,
        model=get_track_evolution_model(LINEAR_TRACK_EVOLUTION_MODEL),
        fit=fit,
        reference_session_lap_order=reference_session_lap_order,
    )


def add_corrected_sector_times(laps: pd.DataFrame) -> pd.DataFrame:
    missing_sector_columns = set(SECTOR_TIME_COLUMNS.values()).difference(laps.columns)
    if missing_sector_columns:
        return laps

    corrected = laps.copy()
    lap_time_seconds = pd.to_numeric(corrected["LapTimeSeconds"], errors="coerce")
    corrected_lap_seconds = pd.to_numeric(
        corrected[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS],
        errors="coerce",
    )
    correction_ratio = corrected_lap_seconds / lap_time_seconds
    correction_ratio = correction_ratio.where(lap_time_seconds > 0)

    for sector, source_column in SECTOR_TIME_COLUMNS.items():
        sector_seconds = timedelta_seconds(corrected[source_column])
        corrected_sector_seconds = sector_seconds * correction_ratio
        corrected[CORRECTED_SECTOR_SECONDS[sector]] = corrected_sector_seconds
        corrected[CORRECTED_SECTOR_TIME[sector]] = pd.to_timedelta(
            corrected_sector_seconds,
            unit="s",
        )

    return corrected


def timedelta_seconds(values: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(values):
        return values.dt.total_seconds()
    return pd.to_timedelta(values, errors="coerce").dt.total_seconds()


def quickest_driver_laps(push_laps: pd.DataFrame) -> pd.DataFrame:
    idx = push_laps.groupby(["Team", "Driver"])[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS].idxmin()
    columns = [
        "Team",
        "Driver",
        "LapNumber",
        QUALIFYING_PART,
        "SessionLapOrder",
        "Compound",
        TRACK_EVO_CORRECTED_LAP_TIME,
        TRACK_EVO_CORRECTED_LAP_TIME_SECONDS,
    ]
    return (
        push_laps.loc[idx, columns]
        .sort_values(["Team", TRACK_EVO_CORRECTED_LAP_TIME_SECONDS, "Driver"])
        .reset_index(drop=True)
    )


def latest_qualifying_part_laps(
    push_laps: pd.DataFrame,
    *,
    reference_laps: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if QUALIFYING_PART not in push_laps.columns:
        return push_laps.copy()

    latest_parts = (
        latest_qualifying_parts_by_driver(reference_laps)
        if reference_laps is not None
        else None
    )
    part_rank = capped_qualifying_part_ranks(push_laps, reference_laps=reference_laps)
    filtered_indexes = []
    for _, driver_laps in push_laps.assign(_QualifyingPartRank=part_rank).groupby(
        ["Team", "Driver"], sort=False
    ):
        team = driver_laps["Team"].iloc[0]
        driver = driver_laps["Driver"].iloc[0]
        latest_rank = (
            latest_parts.get((team, driver))
            if latest_parts is not None
            else None
        )
        if latest_rank is not None:
            filtered_indexes.extend(
                driver_laps.loc[
                    driver_laps["_QualifyingPartRank"] == latest_rank
                ].index.tolist()
            )
            continue

        ranked_laps = driver_laps.dropna(subset=["_QualifyingPartRank"])
        if ranked_laps.empty:
            filtered_indexes.extend(driver_laps.index.tolist())
            continue
        latest_rank = ranked_laps["_QualifyingPartRank"].max()
        filtered_indexes.extend(
            ranked_laps.loc[ranked_laps["_QualifyingPartRank"] == latest_rank].index.tolist()
        )

    filtered = push_laps.loc[filtered_indexes].copy()
    filtered_part_rank = capped_qualifying_part_ranks(
        filtered,
        reference_laps=reference_laps,
    )
    filtered[QUALIFYING_PART] = filtered_part_rank.map(qualifying_part_label)
    return filtered


def latest_qualifying_parts_by_driver(laps: pd.DataFrame) -> dict[tuple[object, object], float]:
    if QUALIFYING_PART not in laps.columns:
        return {}

    required_columns = {"Team", "Driver", QUALIFYING_PART}
    if required_columns.difference(laps.columns):
        return {}

    ranked = laps.loc[:, ["Team", "Driver", QUALIFYING_PART]].copy()
    ranked["_QualifyingPartRank"] = ranked[QUALIFYING_PART].map(qualifying_part_rank)
    ranked = ranked.dropna(subset=["_QualifyingPartRank"])
    if ranked.empty:
        return {}

    latest_parts = (
        ranked.groupby(["Team", "Driver"], dropna=False)["_QualifyingPartRank"]
        .max()
        .to_dict()
    )
    result_rank_limits = latest_qualifying_part_limits_from_result_rank(laps)
    for driver_key, rank_limit in result_rank_limits.items():
        if driver_key not in latest_parts:
            continue
        latest_parts[driver_key] = min(latest_parts[driver_key], rank_limit)
    return latest_parts


def capped_qualifying_part_ranks(
    laps: pd.DataFrame,
    *,
    reference_laps: pd.DataFrame | None = None,
) -> pd.Series:
    part_rank = laps[QUALIFYING_PART].map(qualifying_part_rank)
    rank_limits = latest_qualifying_part_limits_from_result_rank(
        reference_laps if reference_laps is not None else laps
    )
    if not rank_limits:
        return part_rank

    capped = part_rank.copy()
    for driver_key, rank_limit in rank_limits.items():
        team, driver = driver_key
        mask = laps["Team"].eq(team) & laps["Driver"].eq(driver) & capped.notna()
        capped.loc[mask] = capped.loc[mask].clip(upper=rank_limit)
    return capped


def latest_qualifying_part_limits_from_result_rank(
    laps: pd.DataFrame,
) -> dict[tuple[object, object], float]:
    required_columns = {"Team", "Driver", "SessionResultRank"}
    if required_columns.difference(laps.columns):
        return {}

    ranked = laps.loc[:, ["Team", "Driver", "SessionResultRank"]].copy()
    ranked["SessionResultRank"] = pd.to_numeric(
        ranked["SessionResultRank"],
        errors="coerce",
    )
    ranked = ranked.dropna(subset=["Driver", "SessionResultRank"])
    if ranked.empty:
        return {}

    ranked = (
        ranked.sort_values("SessionResultRank")
        .drop_duplicates(subset=["Team", "Driver"], keep="first")
        .copy()
    )
    ranked["_QualifyingPartRankLimit"] = ranked["SessionResultRank"].map(
        qualifying_part_rank_limit_from_result_rank
    )
    ranked = ranked.dropna(subset=["_QualifyingPartRankLimit"])
    return (
        ranked.set_index(["Team", "Driver"])["_QualifyingPartRankLimit"]
        .astype(float)
        .to_dict()
    )


def qualifying_part_rank_limit_from_result_rank(rank: object) -> float:
    if pd.isna(rank):
        return float("nan")
    numeric_rank = float(rank)
    if numeric_rank <= 10:
        return 3.0
    if numeric_rank <= 15:
        return 2.0
    return 1.0


def qualifying_part_rank(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    normalized = str(value).strip().upper()
    if normalized in {"Q1", "1"}:
        return 1.0
    if normalized in {"Q2", "2"}:
        return 2.0
    if normalized in {"Q3", "3"}:
        return 3.0
    return float("nan")


def qualifying_part_label(rank: object) -> str | pd.NA:
    if pd.isna(rank):
        return pd.NA
    numeric_rank = int(float(rank))
    if numeric_rank == 1:
        return "Q1"
    if numeric_rank == 2:
        return "Q2"
    if numeric_rank == 3:
        return "Q3"
    return pd.NA


def quickest_team_laps(quickest_drivers: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for team, team_laps in quickest_drivers.groupby("Team", sort=True):
        row: dict[str, object] = {"Team": team}
        team_laps = team_laps.sort_values(TRACK_EVO_CORRECTED_LAP_TIME_SECONDS)
        for slot, (_, lap) in enumerate(team_laps.head(2).iterrows(), start=1):
            row[f"Driver{slot}"] = lap["Driver"]
            row[f"Driver{slot}LapNumber"] = lap["LapNumber"]
            row[f"Driver{slot}QualifyingPart"] = lap[QUALIFYING_PART]
            row[f"Driver{slot}CorrectedLapTime"] = lap[TRACK_EVO_CORRECTED_LAP_TIME]
            row[f"Driver{slot}CorrectedLapTimeSeconds"] = lap[
                TRACK_EVO_CORRECTED_LAP_TIME_SECONDS
            ]
        rows.append(row)
    return pd.DataFrame(rows)


def team_fastest_and_average_rows(
    quickest_drivers: pd.DataFrame,
    *,
    teammate_delta_threshold_percent: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for team, team_laps in quickest_drivers.groupby("Team", sort=True):
        team_laps = team_laps.sort_values(TRACK_EVO_CORRECTED_LAP_TIME_SECONDS)
        fastest = team_laps.iloc[0]
        rows.append(
            {
                RESULT_TYPE: "fastest",
                "Team": team,
                "Driver": fastest["Driver"],
                DRIVER_COUNT: 1,
                TEAMMATE_DELTA_SECONDS: np.nan,
                TEAMMATE_DELTA_PERCENT: np.nan,
                AVERAGE_MODE_NOTE: "",
                "LapNumber": int(fastest["LapNumber"]),
                QUALIFYING_PART: fastest[QUALIFYING_PART],
                TRACK_EVO_CORRECTED_LAP_TIME_SECONDS: float(
                    fastest[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS]
                ),
            }
        )

        average_laps = team_laps.head(2)
        fastest_seconds = float(fastest[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS])
        teammate_delta = teammate_delta_seconds(average_laps)
        teammate_delta_percent = teammate_delta_percent_value(average_laps)
        use_fastest_for_average = (
            teammate_delta_threshold_percent is not None
            and pd.notna(teammate_delta_percent)
            and teammate_delta_percent > teammate_delta_threshold_percent
        )
        if use_fastest_for_average:
            performance_laps = average_laps.head(1)
            average_seconds = fastest_seconds
            note = (
                f"used fastest; teammate delta {teammate_delta_percent:.3f}% "
                f"> {teammate_delta_threshold_percent:.3f}%"
            )
        else:
            performance_laps = average_laps
            average_seconds = float(
                performance_laps[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS].mean()
            )
            note = (
                "single driver result"
                if len(performance_laps) == 1
                else f"averaged; teammate delta {teammate_delta_percent:.3f}%"
            )
        if average_seconds < fastest_seconds:
            raise ValueError(
                f"Average team result is faster than fastest result for {team}: "
                f"{average_seconds:.3f}s < {fastest_seconds:.3f}s"
            )
        rows.append(
            {
                RESULT_TYPE: "average",
                "Team": team,
                "Driver": "/".join(performance_laps["Driver"].astype(str).tolist()),
                DRIVER_COUNT: len(performance_laps),
                TEAMMATE_DELTA_SECONDS: teammate_delta,
                TEAMMATE_DELTA_PERCENT: teammate_delta_percent,
                AVERAGE_MODE_NOTE: note,
                "LapNumber": "/".join(
                    str(int(lap_number)) for lap_number in performance_laps["LapNumber"]
                ),
                QUALIFYING_PART: "/".join(
                    performance_laps[QUALIFYING_PART].astype(str).tolist()
                ),
                TRACK_EVO_CORRECTED_LAP_TIME_SECONDS: average_seconds,
            }
        )

    return pd.DataFrame(rows)


def team_best_sector_rows(eligible_laps: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        eligible_laps,
        {
            "Team",
            "Driver",
            "LapNumber",
            QUALIFYING_PART,
            *CORRECTED_SECTOR_SECONDS.values(),
        },
    )

    rows: list[dict[str, object]] = []
    for team, team_laps in eligible_laps.groupby("Team", sort=True):
        sector_rows = []
        for sector, corrected_column in CORRECTED_SECTOR_SECONDS.items():
            sector_laps = team_laps.dropna(subset=[corrected_column])
            if sector_laps.empty:
                sector_rows = []
                break
            sector_rows.append((sector, sector_laps.loc[sector_laps[corrected_column].idxmin()]))

        if len(sector_rows) != 3:
            continue

        best_sector_seconds = sum(
            float(row[CORRECTED_SECTOR_SECONDS[sector]])
            for sector, row in sector_rows
        )
        sector_summary = ", ".join(
            (
                f"{sector} {row['Driver']} {format_qualifying_part(row[QUALIFYING_PART])} "
                f"L{int(row['LapNumber'])} "
                f"{float(row[CORRECTED_SECTOR_SECONDS[sector]]):.3f}s"
            )
            for sector, row in sector_rows
        )
        rows.append(
            {
                RESULT_TYPE: "best_sectors",
                "Team": team,
                "Driver": "/".join(str(row["Driver"]) for _, row in sector_rows),
                DRIVER_COUNT: len({str(row["Driver"]) for _, row in sector_rows}),
                TEAMMATE_DELTA_SECONDS: np.nan,
                TEAMMATE_DELTA_PERCENT: np.nan,
                AVERAGE_MODE_NOTE: sector_summary,
                "LapNumber": "/".join(
                    f"{sector}:{int(row['LapNumber'])}" for sector, row in sector_rows
                ),
                QUALIFYING_PART: "/".join(
                    f"{sector}:{format_qualifying_part(row[QUALIFYING_PART])}"
                    for sector, row in sector_rows
                ),
                TRACK_EVO_CORRECTED_LAP_TIME_SECONDS: best_sector_seconds,
            }
        )

    if not rows:
        raise ValueError("No complete corrected sector results available for best-sector mode.")
    return pd.DataFrame(rows)


def teammate_delta_seconds(team_laps: pd.DataFrame) -> float:
    if len(team_laps) < 2:
        return float("nan")
    times = team_laps[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS].astype(float)
    return float(times.max() - times.min())


def teammate_delta_percent_value(team_laps: pd.DataFrame) -> float:
    if len(team_laps) < 2:
        return float("nan")
    times = team_laps[TRACK_EVO_CORRECTED_LAP_TIME_SECONDS].astype(float)
    fastest_seconds = float(times.min())
    if fastest_seconds <= 0:
        return float("nan")
    return float((times.max() - fastest_seconds) / fastest_seconds * 100.0)


def format_qualifying_part(value: object) -> str:
    if pd.isna(value):
        return "Q?"
    return str(value)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Laps are missing required columns: {', '.join(sorted(missing))}")


def _first_existing_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None
