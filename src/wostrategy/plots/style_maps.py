from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

F1_TEAM_COLORS = {
    "Ferrari": "#dc0000",
    "McLaren": "#ff8700",
    "Red Bull Racing": "#3029ed",
    "Red Bull": "#3029ed",
    "Mercedes": "#00fbe2",
    "Aston Martin": "#006f62",
    "Alpine": "#ff87bc",
    "Williams": "#1f6af4ff",
    "Haas F1 Team": "#6e6e6e",
    "Haas": "#6e6e6e",
    "Cadillac": "#82d8f4",
    "Kick Sauber": "#00e701",
    "Sauber": "#00e701",
    "Audi": "#000000",
    "Racing Bulls": "#5e59ee",
    "RB": "#5e59ee",
}

DRIVER_LINESTYLES_BY_RANK = {
    0: "-",
    1: "--",
    2: ":",
}

DRIVER_BAR_HATCHES_BY_LINESTYLE = {
    "-": None,
    "--": "xx",
    ":": "//",
}


def build_team_style_maps(
    laps: pd.DataFrame,
) -> tuple[dict[str, object], dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    team_names = sorted(laps["Team"].dropna().unique())
    fallback_cmap = plt.get_cmap("tab20")
    fallback_team_colors = {
        team: fallback_cmap(idx % fallback_cmap.N)
        for idx, team in enumerate(team_names)
    }
    team_color_map = {
        team: F1_TEAM_COLORS.get(team, fallback_team_colors[team])
        for team in team_names
    }

    driver_team_summary = (
        laps.groupby(["Team", "Driver"])
        .size()
        .reset_index(name="total_laps")
        .sort_values(["Team", "total_laps", "Driver"], ascending=[True, False, True])
    )
    team_driver_linestyle_map = {}
    for team, team_group in driver_team_summary.groupby("Team", sort=False):
        drivers = team_group["Driver"].tolist()
        for idx, driver in enumerate(drivers):
            team_driver_linestyle_map[(team, driver)] = DRIVER_LINESTYLES_BY_RANK.get(idx, ":")

    team_driver_hatch_map = {
        key: DRIVER_BAR_HATCHES_BY_LINESTYLE.get(linestyle)
        for key, linestyle in team_driver_linestyle_map.items()
    }

    return team_color_map, team_driver_linestyle_map, team_driver_hatch_map


__all__ = [
    "F1_TEAM_COLORS",
    "DRIVER_LINESTYLES_BY_RANK",
    "DRIVER_BAR_HATCHES_BY_LINESTYLE",
    "build_team_style_maps",
]
