from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from wostrategy.analysis.long_run_performance import (
    LongRunFit,
    add_fitted_lap_times,
)
from wostrategy.plots.race_labels import race_tick_labels
from wostrategy.plots.style_maps import F1_TEAM_COLORS, build_team_style_maps


def plot_driver_long_run_fits(
    laps: pd.DataFrame,
    fits: dict[tuple[str, str, str], LongRunFit],
    *,
    track_x_column: str = "LapNumber",
    title: str | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    plot_laps = add_fitted_lap_times(laps, fits, track_x_column=track_x_column)
    team_color_map, team_driver_linestyle_map, _ = build_team_style_maps(plot_laps)

    driver_rank = (
        plot_laps.dropna(subset=["LapTimeSeconds"])
        .groupby(["Driver", "Team"], as_index=False)["LapTimeSeconds"]
        .mean()
        .sort_values(["LapTimeSeconds", "Driver"])
        .reset_index(drop=True)
    )
    driver_rank["PlotIndex"] = driver_rank.index % 4
    driver_to_axis = dict(zip(driver_rank["Driver"], driver_rank["PlotIndex"]))

    fig, axes_array = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes = list(axes_array.flatten())
    for ax in axes:
        ax.grid(True, which="major", alpha=0.3)
        ax.set_xlabel("Race lap")
        ax.set_ylabel("Lap time (s)")

    for (team, driver, compound), group in plot_laps.groupby(
        ["Team", "Driver", "Compound"], sort=False
    ):
        axis_index = driver_to_axis.get(driver)
        if axis_index is None:
            continue
        ax = axes[int(axis_index)]
        group = group.sort_values(track_x_column)
        color = team_color_map.get(team)
        linestyle = team_driver_linestyle_map.get((team, driver), "-")
        label = f"{driver} {compound}"
        ax.plot(
            group[track_x_column],
            group["LapTimeSeconds"],
            color=color,
            linestyle="",
            marker="o",
            markersize=3,
            alpha=0.6,
            label=label,
        )
        fitted = group.dropna(subset=["FittedLapTimeSeconds"])
        if not fitted.empty:
            ax.plot(
                fitted[track_x_column],
                fitted["FittedLapTimeSeconds"],
                color=color,
                linestyle=linestyle,
                linewidth=2,
                alpha=0.95,
            )

    y_values = plot_laps[["LapTimeSeconds", "FittedLapTimeSeconds"]].stack().dropna()
    if not y_values.empty:
        margin = max(0.5, (float(y_values.max()) - float(y_values.min())) * 0.08)
        for ax in axes:
            ax.set_ylim(float(y_values.min()) - margin, float(y_values.max()) + margin)

    for idx, ax in enumerate(axes):
        drivers = driver_rank.loc[driver_rank["PlotIndex"] == idx, "Driver"].tolist()
        ax.set_title(", ".join(drivers) if drivers else "No drivers")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(title or "Long-Run Race Pace With Fitted Curves")
    fig.tight_layout()
    return fig, axes


def plot_long_run_performance_trend(
    team_performance_by_race: pd.DataFrame,
    *,
    reference_team: str | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    required = {"Round", "Team", "LongRunPerformanceSeconds"}
    missing = required.difference(team_performance_by_race.columns)
    if missing:
        raise ValueError(f"Performance trend is missing columns: {sorted(missing)}")

    team_color_map = _build_team_color_map(team_performance_by_race)
    y_column = (
        "PercentageToReferenceTeam"
        if "PercentageToReferenceTeam" in team_performance_by_race.columns
        else "LongRunPerformanceSeconds"
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    for team, group in team_performance_by_race.groupby("Team", sort=True):
        group = group.sort_values("Round")
        ax.plot(
            group["Round"],
            group[y_column],
            marker="o",
            linewidth=2,
            color=team_color_map.get(team),
            label=team,
        )

    tick_frame = race_tick_labels(team_performance_by_race, round_column="Round")
    ax.set_xlabel("Round")
    if not tick_frame.empty:
        ax.set_xticks(tick_frame["Round"])
        ax.set_xticklabels(tick_frame["Label"], rotation=45, ha="right")
    if y_column == "PercentageToReferenceTeam":
        ax.axhline(100.0, color="black", linewidth=1, alpha=0.5)
        ax.set_ylabel(f"Long-run pace (% of {reference_team})")
    else:
        ax.set_ylabel("Weighted long-run first-lap pace (s)")
    ax.set_title(title or "Long-Run Performance Trend")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig, ax


def _build_team_color_map(frame: pd.DataFrame) -> dict[str, object]:
    team_names = sorted(frame["Team"].dropna().unique())
    fallback_cmap = plt.get_cmap("tab20")
    return {
        team: F1_TEAM_COLORS.get(team, fallback_cmap(idx % fallback_cmap.N))
        for idx, team in enumerate(team_names)
    }


__all__ = [
    "plot_driver_long_run_fits",
    "plot_long_run_performance_trend",
]
