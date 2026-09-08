"""Pure live-qualifying calculations shared by GUI and orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from wostrategy.analysis.pre_quali import estimate_push_sequence_track_progression


@dataclass(frozen=True)
class LiveQualifyingTrackEstimate:
    rate_s_per_field_push: float
    evidence_count: int
    source_scope: str = "QUALIFYING_ONLY"


def calculate_live_qualifying_track_evolution(
    qualifying_push_lap_times: Sequence[float],
) -> LiveQualifyingTrackEstimate:
    """Apply the established live planner estimator to qualifying evidence only.

    The API deliberately has no FP argument, preventing an FP/Q combined live fit.
    Operational out/cool/in durations are not accepted as evidence.
    """
    values = tuple(float(value) for value in qualifying_push_lap_times)
    rate = estimate_push_sequence_track_progression(values)
    return LiveQualifyingTrackEstimate(rate, len(values))


__all__ = ["LiveQualifyingTrackEstimate", "calculate_live_qualifying_track_evolution"]
