from __future__ import annotations

import matplotlib.pyplot as plt


def plot_single_lap_comparison(
    prepared_data: dict[str, object],
    output_path: str = None,
):
    best_laps = prepared_data["best_laps"]
    bar_colors = prepared_data["bar_colors"]
    bar_hatches = prepared_data["bar_hatches"]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(
        best_laps["Driver"],
        best_laps["DeltaToQuickestSeconds"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, hatch in zip(bars, bar_hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Driver")
    ax.set_ylabel("Delta to Quickest Lap (s)")
    ax.set_title("Single-Lap Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax, best_laps
