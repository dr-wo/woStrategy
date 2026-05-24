from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from wostrategy.analysis.quali_performance import RESULT_TYPE
from wostrategy.plots.style_maps import F1_TEAM_COLORS


class QualiPerformancePlotter:
    def __init__(self, *, target_team: str) -> None:
        self.target_team = target_team

    def plot_relative_team_pace(
        self,
        summary: pd.DataFrame,
    ) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        figures = {
            result_type: plot_relative_team_pace(
                summary.loc[summary[RESULT_TYPE] == result_type].copy(),
                target_team=self.target_team,
                result_type=result_type,
            )
            for result_type in ("fastest", "average", "best_sectors")
            if (summary[RESULT_TYPE] == result_type).any()
        }
        sync_y_limits(figures)
        return figures


def plot_relative_team_pace(
    summary: pd.DataFrame,
    *,
    target_team: str,
    result_type: str,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(13, 7))
    marker = "o" if result_type == "fastest" else "s"
    for team, team_rows in summary.sort_values("Race").groupby("Team", sort=True):
        ax.plot(
            team_rows["Race"],
            team_rows["PercentageToTargetTeam"],
            marker=marker,
            linewidth=2,
            label=team,
            color=F1_TEAM_COLORS.get(team),
        )

    ax.axhline(100.0, color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel("Race number")
    result_label = result_type.replace("_", " ").title()
    ax.set_ylabel(f"{result_label} corrected lap time (% of {target_team})")
    ax.set_title(
        f"{summary['Year'].iloc[0]} Quali {result_label} Pace "
        f"Relative to {target_team}"
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Team", ncols=2)
    fig.tight_layout()
    return fig, ax


def result_output_path(output_path: Path, result_type: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{result_type}{output_path.suffix}")


def save_relative_team_pace_figures(
    figures: dict[str, tuple[plt.Figure, plt.Axes]],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for result_type, (fig, _) in figures.items():
        fig.savefig(result_output_path(output_path, result_type), dpi=150, bbox_inches="tight")


def sync_y_limits(figures: dict[str, tuple[plt.Figure, plt.Axes]]) -> None:
    if len(figures) < 2:
        return

    limits = [ax.get_ylim() for _, ax in figures.values()]
    y_min = min(low for low, _ in limits)
    y_max = max(high for _, high in limits)
    for _, ax in figures.values():
        ax.set_ylim(y_min, y_max)
