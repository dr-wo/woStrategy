from __future__ import annotations

import pandas as pd

from wostrategy.model.track_evolution import dominant_compound


class PushLapSelector:
    def __init__(
        self,
        *,
        quick_lap_threshold: float,
        clean_min_time_delta_seconds: float | None,
        clean_mean_time_delta_seconds: float | None,
        dry_compounds: tuple[str, ...],
        new_tyre_only: bool,
    ) -> None:
        self.quick_lap_threshold = quick_lap_threshold
        self.clean_min_time_delta_seconds = clean_min_time_delta_seconds
        self.clean_mean_time_delta_seconds = clean_mean_time_delta_seconds
        self.dry_compounds = tuple(compound.upper() for compound in dry_compounds)
        self.new_tyre_only = new_tyre_only

    def add_flags(self, laps: pd.DataFrame) -> pd.DataFrame:
        return add_push_lap_flags(
            laps,
            quick_lap_threshold=self.quick_lap_threshold,
            clean_min_time_delta_seconds=self.clean_min_time_delta_seconds,
            clean_mean_time_delta_seconds=self.clean_mean_time_delta_seconds,
        )

    def select_push_laps(self, laps: pd.DataFrame) -> pd.DataFrame:
        flagged = self.add_flags(laps)
        return select_dry_push_laps(
            flagged,
            dry_compounds=self.dry_compounds,
            new_tyre_only=self.new_tyre_only,
        )


def add_push_lap_flags(
    laps: pd.DataFrame,
    *,
    quick_lap_threshold: float,
    clean_min_time_delta_seconds: float | None,
    clean_mean_time_delta_seconds: float | None,
) -> pd.DataFrame:
    clean_delta_column, clean_delta_threshold = _resolve_clean_delta_filter(
        clean_min_time_delta_seconds=clean_min_time_delta_seconds,
        clean_mean_time_delta_seconds=clean_mean_time_delta_seconds,
    )
    required_columns = {
        "Driver",
        "LapTime",
        "LapStartTime",
        clean_delta_column,
        "PitOutTime",
        "PitInTime",
    }
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Laps are missing required columns: {missing}")

    flagged = laps.copy()
    flagged["IsQuickLap"] = False
    flagged["IsCleanLap"] = False
    flagged["IsPushLap"] = False
    flagged["LapPatternRole"] = pd.NA

    valid_lap_time = flagged["LapTime"].notna()
    valid_lap_start = flagged["LapStartTime"].notna()
    flagged.loc[valid_lap_time, "LapTimeSeconds"] = (
        flagged.loc[valid_lap_time, "LapTime"].dt.total_seconds()
    )
    flagged.loc[valid_lap_start, "LapStartSeconds"] = (
        flagged.loc[valid_lap_start, "LapStartTime"].dt.total_seconds()
    )
    flagged["LapStartMinutes"] = flagged["LapStartSeconds"] / 60.0
    flagged["SessionLapOrder"] = pd.NA
    ordered_laps = flagged.loc[valid_lap_start].sort_values(["LapStartTime", "Driver"])
    flagged.loc[ordered_laps.index, "SessionLapOrder"] = range(1, len(ordered_laps) + 1)
    flagged["SessionLapOrder"] = pd.to_numeric(flagged["SessionLapOrder"], errors="coerce")

    driver_fastest = flagged.groupby("Driver")["LapTimeSeconds"].transform("min")
    flagged["IsQuickLap"] = flagged["LapTimeSeconds"] <= quick_lap_threshold * driver_fastest
    if clean_delta_threshold == 0:
        flagged["IsCleanLap"] = flagged["IsQuickLap"]
    else:
        flagged["IsCleanLap"] = (
            flagged["IsQuickLap"]
            & (flagged[clean_delta_column] > clean_delta_threshold)
        )
    candidate_push = flagged["IsQuickLap"] & flagged["IsCleanLap"]
    flagged["IsOutLap"] = flagged["PitOutTime"].notna()
    flagged["IsInLap"] = flagged["PitInTime"].notna()
    candidate_push = candidate_push & ~flagged["IsOutLap"] & ~flagged["IsInLap"]

    group_columns = ["Driver"]
    if "Stint" in flagged.columns:
        group_columns.append("Stint")

    sort_columns = [column for column in ("LapStartTime", "LapNumber") if column in flagged]
    for _, group in flagged.groupby(group_columns, dropna=False, sort=False):
        if sort_columns:
            group = group.sort_values(sort_columns)

        pattern_active = False
        for index, lap in group.iterrows():
            if lap["IsOutLap"]:
                pattern_active = True
                flagged.loc[index, "LapPatternRole"] = "out_lap"
                continue

            if not pattern_active:
                continue

            if lap["IsInLap"]:
                pattern_active = False
                flagged.loc[index, "LapPatternRole"] = "in_lap"
                continue

            if candidate_push.loc[index]:
                flagged.loc[index, "IsPushLap"] = True
                flagged.loc[index, "LapPatternRole"] = "push_lap"
            else:
                flagged.loc[index, "LapPatternRole"] = "slow_lap"

    return flagged


def select_dry_push_laps(
    laps: pd.DataFrame,
    *,
    dry_compounds: tuple[str, ...],
    new_tyre_only: bool,
) -> pd.DataFrame:
    push_laps = laps.loc[laps["IsPushLap"]].copy()
    if push_laps.empty:
        raise ValueError("No push laps available after filtering.")

    dry_compounds = tuple(compound.upper() for compound in dry_compounds)
    if "Compound" not in push_laps.columns:
        raise ValueError("Push laps are missing required column: Compound")
    push_laps["Compound"] = push_laps["Compound"].astype("string").str.upper()
    push_laps = push_laps.loc[push_laps["Compound"].isin(dry_compounds)].copy()
    if new_tyre_only:
        if "FreshTyre" not in push_laps.columns:
            raise ValueError("Push laps are missing required column: FreshTyre")
        push_laps = push_laps.loc[fresh_tyre_mask(push_laps["FreshTyre"])].copy()
    if push_laps.empty:
        tyre_filter = " on new tyres" if new_tyre_only else ""
        raise ValueError(
            f"No push laps{tyre_filter} available on configured dry compounds."
        )
    return push_laps


def select_top_drivers(laps: pd.DataFrame, top_driver_count: int | None) -> list[str] | None:
    if top_driver_count is None:
        return None
    if top_driver_count <= 0:
        raise ValueError("top_driver_count must be positive when provided.")
    if "Driver" not in laps.columns:
        raise ValueError("Laps need Driver column to rank drivers.")

    if "SessionResultRank" in laps.columns and laps["SessionResultRank"].notna().any():
        driver_rank = (
            laps.dropna(subset=["Driver", "SessionResultRank"])
            .groupby("Driver", as_index=False)["SessionResultRank"]
            .min()
            .sort_values(["SessionResultRank", "Driver"])
        )
        return driver_rank["Driver"].head(top_driver_count).tolist()

    if "LapTimeSeconds" not in laps.columns:
        raise ValueError("Laps need LapTimeSeconds column when result rank is unavailable.")

    driver_fastest = (
        laps.dropna(subset=["Driver", "LapTimeSeconds"])
        .groupby("Driver", as_index=False)["LapTimeSeconds"]
        .min()
        .sort_values(["LapTimeSeconds", "Driver"])
    )
    return driver_fastest["Driver"].head(top_driver_count).tolist()


def get_dominant_compound(laps: pd.DataFrame) -> str | None:
    return dominant_compound(laps["Compound"], require_majority=False)


def fresh_tyre_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)

    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "new"})


def _resolve_clean_delta_filter(
    *,
    clean_min_time_delta_seconds: float | None,
    clean_mean_time_delta_seconds: float | None,
) -> tuple[str, float]:
    min_defined = clean_min_time_delta_seconds is not None
    mean_defined = clean_mean_time_delta_seconds is not None
    if min_defined == mean_defined:
        raise ValueError(
            "Define exactly one clean gap filter: "
            "clean_min_time_delta_seconds or clean_mean_time_delta_seconds."
        )

    if min_defined:
        return "MinTimeDeltaToDriverAhead", float(clean_min_time_delta_seconds)
    return "MeanTimeDeltaToDriverAhead", float(clean_mean_time_delta_seconds)
