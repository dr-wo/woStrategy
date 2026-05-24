from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wostrategy.analysis import TrackEvolutionModel
from wostrategy.analysis.push_laps import get_dominant_compound
from wostrategy.plots.style_maps import F1_TEAM_COLORS


class TrackDevelopmentPlotter:
    def __init__(self, fit_model: TrackEvolutionModel) -> None:
        self.fit_model = fit_model

    def plot_compound_lap_time_fits(
        self,
        push_laps: pd.DataFrame,
        *,
        dry_compounds: tuple[str, ...],
        x_column: str,
        x_label: str,
        slope_unit: str,
        title: str,
    ):
        return plot_compound_lap_time_fits(
            push_laps,
            dry_compounds=dry_compounds,
            fit_model=self.fit_model,
            x_column=x_column,
            x_label=x_label,
            slope_unit=slope_unit,
            title=title,
        )

    def plot_top_driver_summary(
        self,
        push_laps: pd.DataFrame,
        *,
        top_drivers: list[str],
        x_column: str,
        x_label: str,
        slope_unit: str,
        title: str,
    ):
        return plot_top_driver_summary(
            push_laps,
            top_drivers=top_drivers,
            fit_model=self.fit_model,
            x_column=x_column,
            x_label=x_label,
            slope_unit=slope_unit,
            title=title,
        )


def plot_compound_lap_time_fits(
    push_laps: pd.DataFrame,
    *,
    dry_compounds: tuple[str, ...],
    fit_model: TrackEvolutionModel,
    x_column: str,
    x_label: str,
    slope_unit: str,
    title: str,
):
    if "Compound" not in push_laps.columns:
        raise ValueError("Push laps are missing required column: Compound")
    if "Team" not in push_laps.columns:
        raise ValueError("Push laps are missing required column: Team")
    if x_column not in push_laps.columns:
        raise ValueError(f"Push laps are missing required column: {x_column}")

    push_laps = push_laps.copy()
    push_laps["Compound"] = push_laps["Compound"].astype("string").str.upper()
    driver_marker_styles = _build_driver_marker_styles(push_laps)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    fits: dict[str, dict[str, float] | None] = {}

    _plot_lap_time_points(
        axes_flat[0],
        push_laps,
        title="All dry compounds",
        fit=False,
        fit_model=fit_model,
        x_column=x_column,
        slope_unit=slope_unit,
        driver_marker_styles=driver_marker_styles,
    )

    for ax, compound in zip(axes_flat[1:], dry_compounds):
        compound_laps = push_laps.loc[push_laps["Compound"] == compound].copy()
        fits[compound] = _plot_lap_time_points(
            ax,
            compound_laps,
            title=compound,
            fit=True,
            fit_model=fit_model,
            x_column=x_column,
            slope_unit=slope_unit,
            driver_marker_styles=driver_marker_styles,
        )

    for ax in axes_flat:
        ax.set_xlabel(x_label)
        ax.set_ylabel("Lap time (s)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Driver", loc="upper center", ncols=8)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, axes, fits


def plot_top_driver_summary(
    push_laps: pd.DataFrame,
    *,
    top_drivers: list[str],
    fit_model: TrackEvolutionModel,
    x_column: str,
    x_label: str,
    slope_unit: str,
    title: str,
):
    dominant_compound = get_dominant_compound(push_laps)
    top_driver_laps = push_laps.loc[push_laps["Driver"].isin(top_drivers)].copy()
    if dominant_compound is None:
        dominant_laps = push_laps.iloc[0:0].copy()
        top_dominant_laps = top_driver_laps.iloc[0:0].copy()
        dominant_title = "Dominant compound >50%"
    else:
        dominant_laps = push_laps.loc[push_laps["Compound"] == dominant_compound].copy()
        top_dominant_laps = top_driver_laps.loc[
            top_driver_laps["Compound"] == dominant_compound
        ].copy()
        dominant_title = f"Dominant compound: {dominant_compound}"

    driver_marker_styles = _build_driver_marker_styles(push_laps)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    fits: dict[str, dict[str, float] | None] = {}
    panel_data = (
        ("all", push_laps, "All push laps", False),
        ("dominant", dominant_laps, dominant_title, False),
        ("top_drivers", top_driver_laps, f"Top {len(top_drivers)} drivers", True),
        (
            "top_dominant",
            top_dominant_laps,
            f"Top {len(top_drivers)} drivers, {dominant_title}",
            True,
        ),
    )
    for ax, (fit_key, panel_laps, panel_title, should_fit) in zip(axes_flat, panel_data):
        fits[fit_key] = _plot_lap_time_points(
            ax,
            panel_laps,
            title=panel_title,
            fit=should_fit,
            fit_model=fit_model,
            x_column=x_column,
            slope_unit=slope_unit,
            driver_marker_styles=driver_marker_styles,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Lap time (s)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Driver", loc="upper center", ncols=8)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, axes, fits


def _plot_lap_time_points(
    ax,
    laps: pd.DataFrame,
    *,
    title: str,
    fit: bool,
    fit_model: TrackEvolutionModel,
    x_column: str,
    slope_unit: str,
    driver_marker_styles: dict[str, dict[str, object]],
) -> dict[str, float] | None:
    ax.set_title(f"{title} (n={len(laps)})")
    if laps.empty:
        ax.text(0.5, 0.5, "No push laps", transform=ax.transAxes, ha="center")
        return None

    for driver, driver_laps in laps.groupby("Driver", sort=True):
        style = driver_marker_styles.get(
            driver,
            {"color": "#1f77b4", "filled": True},
        )
        color = style["color"]
        filled = bool(style["filled"])
        ax.scatter(
            driver_laps[x_column],
            driver_laps["LapTimeSeconds"],
            label=driver,
            alpha=0.75,
            s=34,
            marker="o",
            facecolors=color if filled else "none",
            edgecolors=color,
            linewidths=1.2,
        )

    if not fit:
        return None

    fit_laps = laps.dropna(subset=[x_column, "LapTimeSeconds"])
    if len(fit_laps) < 2:
        _add_not_enough_laps_text(ax)
        return None

    x = fit_laps[x_column].to_numpy(dtype="float64")
    try:
        evolution_fit = fit_model.fit(
            fit_laps,
            x_column=x_column,
            y_column="LapTimeSeconds",
            slope_unit=slope_unit,
        )
    except ValueError:
        _add_not_enough_laps_text(ax)
        return None

    sort_order = np.argsort(x)
    fit_y = fit_model.predict(x, evolution_fit)
    ax.plot(
        x[sort_order],
        fit_y[sort_order],
        color="black",
        linewidth=2,
        label=fit_model.fit_label,
    )

    ax.text(
        0.02,
        0.98,
        fit_model.equation_label(evolution_fit),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    return evolution_fit.to_summary()


def _add_not_enough_laps_text(ax) -> None:
    ax.text(
        0.02,
        0.98,
        "Not enough laps for fit",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )


def _build_driver_marker_styles(laps: pd.DataFrame) -> dict[str, dict[str, object]]:
    team_names = sorted(laps["Team"].dropna().unique())
    fallback_cmap = plt.get_cmap("tab20")
    team_colors = {
        team: F1_TEAM_COLORS.get(team, fallback_cmap(index % fallback_cmap.N))
        for index, team in enumerate(team_names)
    }

    driver_team_summary = (
        laps.groupby(["Team", "Driver"])
        .size()
        .reset_index(name="sample_count")
        .sort_values(["Team", "sample_count", "Driver"], ascending=[True, False, True])
    )

    marker_styles: dict[str, dict[str, object]] = {}
    for team, team_group in driver_team_summary.groupby("Team", sort=False):
        drivers = team_group["Driver"].tolist()
        for index, driver in enumerate(drivers):
            marker_styles[driver] = {
                "team": team,
                "color": team_colors.get(team, "#1f77b4"),
                "filled": index % 2 == 0,
            }

    return marker_styles
