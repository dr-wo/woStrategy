"""Public empirical pit-loss analysis API.

The implementation remains the established race-performance-review calculation;
this module keeps application code from depending on a CLI/script module path.
"""

from __future__ import annotations

import pandas as pd


def pit_loss_split_summary(laps: pd.DataFrame) -> pd.DataFrame:
    from wostrategy.script.race_performance_review import pit_loss_split_summary as calculate

    return calculate(laps)


def median_pit_loss_by_state(laps: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    summary = pit_loss_split_summary(laps)
    output: dict[str, dict[str, float | int]] = {}
    for row in summary.to_dict("records"):
        status = str(row["TrackStatusType"])
        if status not in {"normal", "sc_vsc"}:
            continue
        output[status] = {
            "sample_count": int(row["SampleCount"]),
            "pit_in_s3": float(row["PitInS3LossMedianSeconds"]),
            "pit_out_s1": float(row["PitOutS1LossMedianSeconds"]),
            "total": float(row["PitTotalLossMedianSeconds"]),
        }
    return output


__all__ = ["median_pit_loss_by_state", "pit_loss_split_summary"]
