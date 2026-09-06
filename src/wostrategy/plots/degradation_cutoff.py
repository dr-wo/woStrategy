"""Deterministic engineering plot for degradation strategy envelopes."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt

from wostrategy.analysis.degradation_cutoff import DegradationCutoffResult


def plot_degradation_cutoff(
    result: DegradationCutoffResult,
    *,
    fp_diagnostic_markers: Sequence[Mapping[str, object]] = (),
):
    figure, axis = plt.subplots(figsize=(9, 5.5))
    x = [point.medium_degradation for point in result.sweep]
    colours = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    for stops in (1, 2, 3):
        y = [point.costs.get(stops) for point in result.sweep]
        axis.plot(x, y, label=f"Best {stops}-stop", color=colours[stops], linewidth=2)
    axis.axvline(result.predicted_medium_degradation, color="black", linestyle="--", label="Predicted MEDIUM degradation")
    for value, label, colour in (
        (result.primary_1_to_2, "1→2 cutoff", "#9467bd"),
        (result.primary_2_to_3, "2→3 cutoff", "#8c564b"),
    ):
        if value is not None:
            axis.axvline(value, color=colour, linestyle=":", label=label)
    fp_styles = {
        "FP1": ("#555555", "-.", "o"),
        "FP2": ("#777777", (0, (5, 2, 1, 2)), "s"),
        "FP3": ("#999999", (0, (2, 2, 6, 2)), "^"),
    }
    for marker in fp_diagnostic_markers:
        session = str(marker["session"]).upper()
        value = float(marker["medium_degradation"])
        colour, linestyle, symbol = fp_styles.get(session, ("#777777", "-.", "D"))
        label = f"{session} diagnostic"
        axis.axvline(
            value, color=colour, linestyle=linestyle, linewidth=1.6,
            alpha=0.9, label=label,
        )
        axis.plot(
            [value], [0.02], marker=symbol, color=colour, markerfacecolor="white",
            markersize=7, transform=axis.get_xaxis_transform(), clip_on=False,
        )
    for index, (start, end) in enumerate(result.reversal_regions):
        axis.axvspan(start, end, color="#d62728", alpha=0.15, label="Reversal region" if index == 0 else None)
    axis.set_xlabel("MEDIUM degradation (s/lap)")
    axis.set_ylabel("Best strategy cost (s relative)")
    axis.set_title("Strategy envelope versus MEDIUM degradation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def save_degradation_cutoff_plot(
    result: DegradationCutoffResult,
    path: str | Path,
    *,
    fp_diagnostic_markers: Sequence[Mapping[str, object]] = (),
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_degradation_cutoff(
        result, fp_diagnostic_markers=fp_diagnostic_markers,
    )
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


__all__ = ["plot_degradation_cutoff", "save_degradation_cutoff_plot"]
