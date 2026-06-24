from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pandas as pd

from wostrategy.plots.race_labels import race_tick_labels
from wostrategy.plots.style_maps import F1_TEAM_COLORS

RACE_BASELINE_RESULT_TYPE = "team_baseline"
RMSE_BACKGROUND_CMAP = LinearSegmentedColormap.from_list(
    "race_performance_rmse",
    [(0.0, "green"), (0.5, "green"), (0.75, "orange"), (1.0, "red")],
)
RMSE_BACKGROUND_NORM = Normalize(vmin=0.0, vmax=1.0, clip=True)


class RacePerformancePlotter:
    def __init__(
        self,
        *,
        reference_team: str,
        plot_uncertainty_band: bool = False,
        plot_rmse_background: bool = False,
    ) -> None:
        self.reference_team = reference_team
        self.plot_uncertainty_band = plot_uncertainty_band
        self.plot_rmse_background = plot_rmse_background

    def plot_relative_team_pace(
        self,
        summary: pd.DataFrame,
    ) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        return {
            RACE_BASELINE_RESULT_TYPE: plot_relative_team_pace(
                summary,
                reference_team=self.reference_team,
                plot_uncertainty_band=self.plot_uncertainty_band,
                plot_rmse_background=self.plot_rmse_background,
            )
        }


def plot_relative_team_pace(
    summary: pd.DataFrame,
    *,
    reference_team: str,
    plot_uncertainty_band: bool = False,
    plot_rmse_background: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(13, 7))
    sorted_summary = summary.sort_values("Race")
    if plot_rmse_background:
        _plot_rmse_background(ax, sorted_summary)

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
        if plot_uncertainty_band and {
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
    if plot_rmse_background:
        scalar = ScalarMappable(norm=RMSE_BACKGROUND_NORM, cmap=RMSE_BACKGROUND_CMAP)
        scalar.set_array([])
        colorbar = fig.colorbar(scalar, ax=ax, pad=0.015)
        colorbar.set_label("Weighted RMSE (s)")
    fig.tight_layout()
    return fig, ax


def _plot_rmse_background(ax: plt.Axes, summary: pd.DataFrame) -> None:
    if "WeightedRMSESeconds" not in summary.columns:
        return
    for race, race_rows in summary.groupby("Race", sort=True):
        team_count = int(race_rows["Team"].nunique())
        race_number = float(race)
        if team_count < 5:
            ax.axvspan(
                race_number - 0.45,
                race_number + 0.45,
                facecolor="black",
                alpha=0.08,
                hatch="///",
                edgecolor="black",
                linewidth=0.0,
                zorder=0,
            )
            continue

        rmse_values = pd.to_numeric(
            race_rows["WeightedRMSESeconds"],
            errors="coerce",
        ).dropna()
        if rmse_values.empty:
            continue
        color = RMSE_BACKGROUND_CMAP(RMSE_BACKGROUND_NORM(float(rmse_values.iloc[0])))
        ax.axvspan(
            race_number - 0.45,
            race_number + 0.45,
            facecolor=color,
            alpha=0.18,
            linewidth=0.0,
            zorder=0,
        )


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
