from __future__ import annotations

import pandas as pd

from wostrategy.plots.style_maps import build_team_style_maps


def prepare_cumulative_laps_by_day_data(laps: pd.DataFrame) -> dict[str, object]:
    """Prepare cumulative-lap plot inputs without rendering."""
    required_columns = {"Round", "SessionName", "Driver", "Team"}
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns for cumulative laps plot: {sorted(missing_columns)}")

    plot_laps = laps.dropna(subset=["Round", "SessionName", "Driver", "Team"]).copy()
    if plot_laps.empty:
        raise ValueError("No valid laps available for cumulative laps plot")

    day_counts = (
        plot_laps.groupby(["Round", "SessionName", "Driver", "Team"])
        .size()
        .reset_index(name="LapCount")
        .sort_values(["Round", "SessionName", "Driver"])
    )
    day_order = (
        day_counts[["Round", "SessionName"]]
        .drop_duplicates()
        .sort_values(["Round", "SessionName"])
        .reset_index(drop=True)
    )
    day_order["DayIndex"] = range(1, len(day_order) + 1)
    day_order["DayLabel"] = [f"R{row.Round} S{row.SessionName}" for row in day_order.itertuples(index=False)]

    cumulative_counts = day_counts.merge(day_order, on=["Round", "SessionName"], how="left")
    driver_cumulative_counts = cumulative_counts.copy()
    driver_cumulative_counts["CumulativeLapCount"] = (
        driver_cumulative_counts.sort_values(["Driver", "DayIndex"]).groupby("Driver")["LapCount"].cumsum()
    )
    team_day_counts = (
        plot_laps.groupby(["Round", "SessionName", "Team"])
        .size()
        .reset_index(name="LapCount")
        .sort_values(["Round", "SessionName", "Team"])
    )
    team_cumulative_counts = team_day_counts.merge(day_order, on=["Round", "SessionName"], how="left")
    team_cumulative_counts["CumulativeLapCount"] = (
        team_cumulative_counts.sort_values(["Team", "DayIndex"]).groupby("Team")["LapCount"].cumsum()
    )

    team_color_map, team_driver_linestyle_map, _ = build_team_style_maps(plot_laps)
    return {
        "day_order": day_order,
        "driver_cumulative_counts": driver_cumulative_counts,
        "team_cumulative_counts": team_cumulative_counts,
        "team_color_map": team_color_map,
        "team_driver_linestyle_map": team_driver_linestyle_map,
    }
