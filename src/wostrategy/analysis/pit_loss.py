"""Public empirical pit-loss analysis API.

The implementation remains the established race-performance-review calculation;
this module keeps application code from depending on a CLI/script module path.
"""

from __future__ import annotations

import pandas as pd


def pit_loss_split_summary(laps: pd.DataFrame) -> pd.DataFrame:
    from wostrategy.script.race_performance_review import pit_loss_split_summary as calculate

    return calculate(laps)


def median_pit_loss_by_state(laps: pd.DataFrame) -> dict[str, dict[str, object]]:
    summary = pit_loss_split_summary(laps)
    output: dict[str, dict[str, object]] = {
        state: {"status": "unavailable", "sample_count": 0, "samples": []}
        for state in ("normal", "sc", "vsc")
    }
    for row in summary.to_dict("records"):
        status = str(row["TrackStatusType"])
        if status not in output:
            continue
        output[status] = {
            "status": "available",
            "sample_count": int(row["SampleCount"]),
            "pit_in_s3": float(row["PitInS3LossMedianSeconds"]),
            "pit_out_s1": float(row["PitOutS1LossMedianSeconds"]),
            "total": float(row["PitTotalLossMedianSeconds"]),
            "samples": _sample_details(row.get("SampleDetailsJson")),
        }
    return output


def _sample_details(value: object) -> list[dict[str, object]]:
    import json

    if value is None or pd.isna(value):
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return []
    return [dict(item) for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


__all__ = ["median_pit_loss_by_state", "pit_loss_split_summary"]
