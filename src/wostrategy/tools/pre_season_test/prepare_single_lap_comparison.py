from __future__ import annotations

from typing import Union

import pandas as pd

from wostrategy.core.session import Session
from wostrategy.core.session_loader import load_session_laps
from wostrategy.plots.style_maps import build_team_style_maps
from wostrategy.tools.forenoon_afternoon import add_half_day_label
from wostrategy.tools.session_values import add_session_value_column


def load_single_lap_comparison_laps(
    year: int,
    rounds: list[Union[int, str]],
    session_names: list[Union[int, str]],
    test: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Load laps for single-lap comparison workflows."""
    laps = load_session_laps(
        year=year,
        rounds=rounds,
        session_names=session_names,
        session_factory=lambda round_number, session_name: Session(
            year=year,
            round=round_number,
            session_name=session_name,
            test=test,
            **kwargs,
        ),
        log_label="Loading single-lap data for",
        skip_label="Skipping single-lap data for",
    )
    if laps.empty:
        raise ValueError("No laps were loaded for single-lap comparison")
    return laps


def prepare_single_lap_comparison_data(
    laps: pd.DataFrame,
    *,
    correction_map: dict[tuple[Union[int, str], Union[int, str]], float] | None = None,
) -> dict[str, object]:
    """Prepare single-lap comparison inputs without rendering."""
    correction_map = correction_map or {}
    laps = laps.dropna(subset=["LapTime", "Driver", "Team", "LapStartTime"]).copy()
    if laps.empty:
        raise ValueError("No valid laps available for single-lap comparison")

    laps = laps[laps["PitOutTime"].isna() & laps["PitInTime"].isna()].copy()
    if laps.empty:
        raise ValueError("No representative laps remain after removing in-laps/out-laps")

    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    add_half_day_label(laps)
    add_session_value_column(laps, values=correction_map, target_column="CorrectionSeconds")
    laps["AdjustedLapTimeSeconds"] = laps["LapTimeSeconds"]
    afternoon_mask = laps["HalfDay"] == "afternoon"
    laps.loc[afternoon_mask, "AdjustedLapTimeSeconds"] = (
        laps.loc[afternoon_mask, "AdjustedLapTimeSeconds"] - laps.loc[afternoon_mask, "CorrectionSeconds"]
    )

    best_laps = (
        laps.sort_values("AdjustedLapTimeSeconds")
        .groupby("Driver", as_index=False)
        .first()
        .sort_values("AdjustedLapTimeSeconds")
        .reset_index(drop=True)
    )
    if best_laps.empty:
        raise ValueError("No best laps available for single-lap comparison")

    quickest_lap = float(best_laps["AdjustedLapTimeSeconds"].iloc[0])
    best_laps["DeltaToQuickestSeconds"] = best_laps["AdjustedLapTimeSeconds"] - quickest_lap

    team_color_map, _, team_driver_hatch_map = build_team_style_maps(best_laps[["Team", "Driver"]].copy())
    bar_colors = [team_color_map.get(team) for team in best_laps["Team"]]
    bar_hatches = [
        team_driver_hatch_map.get((team, driver))
        for team, driver in best_laps[["Team", "Driver"]].itertuples(index=False)
    ]
    return {
        "best_laps": best_laps,
        "bar_colors": bar_colors,
        "bar_hatches": bar_hatches,
    }
