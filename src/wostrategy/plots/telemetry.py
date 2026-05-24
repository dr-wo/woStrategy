from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def plot_front_car_delta_circuit_map(
    telemetry: pd.DataFrame,
    *,
    time_delta_column: str = "TimeDeltaToDriverAhead",
    distance_delta_column: str = "DistanceToDriverAhead",
    x_column: str = "X",
    y_column: str = "Y",
    output_path: str | Path | None = None,
):
    """Plot one lap circuit maps colored by time and distance delta ahead."""
    required_columns = {
        x_column,
        y_column,
        time_delta_column,
        distance_delta_column,
    }
    missing_columns = required_columns.difference(telemetry.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Telemetry is missing required columns: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    plot_specs = (
        (axes[0], time_delta_column, "Time delta to car ahead (s)", "viridis"),
        (axes[1], distance_delta_column, "Distance to car ahead (m)", "plasma"),
    )

    for ax, color_column, title, cmap in plot_specs:
        collection = _build_colored_track_collection(
            telemetry=telemetry,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            cmap=cmap,
        )
        ax.add_collection(collection)
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(title)
        fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, axes


def _build_colored_track_collection(
    *,
    telemetry: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str,
    cmap: str,
) -> LineCollection:
    plot_data = telemetry[[x_column, y_column, color_column]].copy()
    plot_data[x_column] = pd.to_numeric(plot_data[x_column], errors="coerce")
    plot_data[y_column] = pd.to_numeric(plot_data[y_column], errors="coerce")
    plot_data[color_column] = pd.to_numeric(plot_data[color_column], errors="coerce")
    plot_data = plot_data.dropna()

    if len(plot_data) < 2:
        raise ValueError(f"Not enough valid telemetry points to plot {color_column}")

    points = plot_data[[x_column, y_column]].to_numpy(dtype="float64")
    segments = list(zip(points[:-1], points[1:]))
    values = plot_data[color_column].iloc[:-1].to_numpy(dtype="float64")
    norm = Normalize(vmin=values.min(), vmax=values.max())

    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=4,
    )
    collection.set_array(values)
    return collection
