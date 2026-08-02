from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from wodata.contracts import ModelSnapshot


def plot_pre_race_performance_summary(snapshot: ModelSnapshot, output: str) -> None:
    """Render compact median and P10-P90 diagnostics for a weekend snapshot."""
    rows = [
        row
        for row in snapshot.parameters
        if row.parameter in {"degradation", "compound_delta"}
    ]
    labels = [
        f"{row.parameter.replace('_', ' ')}\n{row.session} {row.compound or ''}"
        for row in rows
    ]
    medians = np.array([row.median for row in rows], dtype=float)
    lower = medians - np.array([row.p10 for row in rows], dtype=float)
    upper = np.array([row.p90 for row in rows], dtype=float) - medians
    figure, axis = plt.subplots(figsize=(max(7.0, len(rows) * 1.2), 4.8))
    positions = np.arange(len(rows))
    axis.errorbar(positions, medians, yerr=np.vstack([lower, upper]), fmt="o", capsize=5)
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Seconds")
    axis.set_title(f"Weekend model — {snapshot.source_scope}")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
