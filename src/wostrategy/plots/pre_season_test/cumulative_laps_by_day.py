from __future__ import annotations

import matplotlib.pyplot as plt

def plot_cumulative_laps_by_day(
    prepared_data: dict[str, object],
    title: str = None,
    output_path: str = None,
):
    day_order = prepared_data["day_order"]
    driver_cumulative_counts = prepared_data["driver_cumulative_counts"]
    team_cumulative_counts = prepared_data["team_cumulative_counts"]
    team_color_map = prepared_data["team_color_map"]
    team_driver_linestyle_map = prepared_data["team_driver_linestyle_map"]

    driver_fig, driver_ax = plt.subplots(figsize=(12, 7))
    for (driver, team), driver_counts in driver_cumulative_counts.groupby(["Driver", "Team"], sort=False):
        driver_counts = driver_counts.sort_values("DayIndex")
        driver_ax.plot(
            driver_counts["DayIndex"],
            driver_counts["CumulativeLapCount"],
            label=f"{driver} ({team})",
            color=team_color_map.get(team),
            linestyle=team_driver_linestyle_map.get((team, driver), "-"),
            linewidth=2,
            marker="o",
        )

    driver_ax.set_xlabel("Test Day")
    driver_ax.set_ylabel("Cumulative Lap Count")
    driver_ax.set_title(title or "Cumulative Driver Laps by Day")
    driver_ax.set_xticks(day_order["DayIndex"])
    driver_ax.set_xticklabels(day_order["DayLabel"], rotation=45, ha="right")
    driver_ax.grid(True, alpha=0.3)
    driver_ax.legend(loc="best", fontsize=9)
    driver_fig.tight_layout()

    team_fig, team_ax = plt.subplots(figsize=(12, 7))
    for team, team_counts in team_cumulative_counts.groupby("Team", sort=False):
        team_counts = team_counts.sort_values("DayIndex")
        team_ax.plot(
            team_counts["DayIndex"],
            team_counts["CumulativeLapCount"],
            label=team,
            color=team_color_map.get(team),
            linewidth=2.5,
            marker="o",
        )

    team_ax.set_xlabel("Test Day")
    team_ax.set_ylabel("Cumulative Lap Count")
    team_ax.set_title("Cumulative Team Laps by Day")
    team_ax.set_xticks(day_order["DayIndex"])
    team_ax.set_xticklabels(day_order["DayLabel"], rotation=45, ha="right")
    team_ax.grid(True, alpha=0.3)
    team_ax.legend(loc="best", fontsize=9)
    team_fig.tight_layout()

    if output_path:
        if "." in output_path:
            base, ext = output_path.rsplit(".", 1)
            driver_fig.savefig(f"{base}_drivers.{ext}", dpi=150, bbox_inches="tight")
            team_fig.savefig(f"{base}_teams.{ext}", dpi=150, bbox_inches="tight")
        else:
            driver_fig.savefig(f"{output_path}_drivers", dpi=150, bbox_inches="tight")
            team_fig.savefig(f"{output_path}_teams", dpi=150, bbox_inches="tight")

    return {
        "drivers": (driver_fig, driver_ax, driver_cumulative_counts),
        "teams": (team_fig, team_ax, team_cumulative_counts),
        "day_order": day_order,
    }
