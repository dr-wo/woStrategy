from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wostrategy.plots.race_labels import race_tick_labels
from wostrategy.plots.style_maps import F1_TEAM_COLORS

RACE_BASELINE_RESULT_TYPE = "team_baseline"


class RacePerformancePlotter:
    def __init__(self, *, reference_team: str) -> None:
        self.reference_team = reference_team

    def plot_relative_team_pace(
        self,
        summary: pd.DataFrame,
    ) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        return {
            RACE_BASELINE_RESULT_TYPE: plot_relative_team_pace(
                summary,
                reference_team=self.reference_team,
            )
        }


def plot_relative_team_pace(
    summary: pd.DataFrame,
    *,
    reference_team: str,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(13, 7))
    sorted_summary = summary.sort_values("Race")
    for team, team_rows in sorted_summary.groupby("Team", sort=True):
        team_rows = team_rows.sort_values("Race")
        color = F1_TEAM_COLORS.get(team)
        ax.plot(
            team_rows["Race"],
            team_rows["PercentageToReferenceTeam"],
            marker="o",
            linewidth=2,
            label=team,
            color=color,
        )
        if {
            "P10PercentageToReferenceTeam",
            "P90PercentageToReferenceTeam",
        }.issubset(team_rows.columns):
            ax.fill_between(
                team_rows["Race"].to_numpy(dtype="float64"),
                team_rows["P10PercentageToReferenceTeam"].to_numpy(dtype="float64"),
                team_rows["P90PercentageToReferenceTeam"].to_numpy(dtype="float64"),
                color=color,
                alpha=0.12,
                linewidth=0,
            )

    ax.axhline(100.0, color="black", linewidth=1, alpha=0.5)
    tick_frame = race_tick_labels(summary, round_column="Race")
    ax.set_xlabel("Round")
    if not tick_frame.empty:
        ax.set_xticks(tick_frame["Race"])
        ax.set_xticklabels(tick_frame["Label"], rotation=45, ha="right")
    ax.set_ylabel(f"Corrected race baseline pace (% of {reference_team})")
    ax.set_title(
        f"{summary['Year'].iloc[0]} Race Corrected Baseline Pace "
        f"Relative to {reference_team}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="Team", ncols=2)
    fig.tight_layout()
    return fig, ax


def result_output_path(output_path: Path) -> Path:
    return output_path.with_name(
        f"{output_path.stem}_{RACE_BASELINE_RESULT_TYPE}{output_path.suffix}"
    )


def save_relative_team_pace_figures(
    figures: dict[str, tuple[plt.Figure, plt.Axes]],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for _, (fig, _) in figures.items():
        fig.savefig(result_output_path(output_path), dpi=150, bbox_inches="tight")
