from __future__ import annotations

from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from wostrategy.core.session import Session
from wostrategy.core.session_loader import load_session_laps


def add_half_day_label(
    laps: pd.DataFrame,
    *,
    lap_start_column: str = "LapStartTime",
    lap_start_seconds_column: str = "LapStartSeconds",
    half_day_column: str = "HalfDay",
    cutoff_seconds: float = 4.5 * 3600,
) -> pd.DataFrame:
    """Add half-day labels based on lap start time."""
    laps[lap_start_seconds_column] = laps[lap_start_column].dt.total_seconds()
    laps[half_day_column] = laps[lap_start_seconds_column].map(
        lambda seconds: "forenoon"
        if pd.notna(seconds) and seconds < cutoff_seconds
        else "afternoon"
    )
    return laps


def forenoon_afternoon_delta(
    year: int,
    rounds: list[Union[int, str]],
    session_names: list[Union[int, str]],
    output_csv: str,
    min_laps: int = 40,
    test: bool = False,
    **kwargs,
) -> dict[tuple[Union[int, str], Union[int, str]], float]:
    """
    Calculate the average lap-time delta between forenoon and afternoon.
    """
    laps = load_session_laps(
        year=year,
        rounds=rounds,
        session_names=session_names,
        session_factory=lambda round_number, session_name: Session(
            year=year,
            round=round_number,
            session_name=session_name,
            test=test,
            **kwargs,
        ),
        enrich_session=lambda session: session.effective_stint(),
        log_label="Loading",
        skip_label="Skipping",
    )
    if laps.empty:
        raise ValueError("No laps were loaded for forenoon/afternoon comparison")
    laps = laps.dropna(subset=["LapTime", "LapNumber", "Driver", "LapStartDate"]).copy()
    if laps.empty:
        raise ValueError("No valid laps available after dropping missing values")

    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps["DayKey"] = list(zip(laps["Round"], laps["SessionName"]))

    day_fastest_lap = laps.groupby("DayKey")["LapTimeSeconds"].transform("min")
    laps["Slow107"] = laps["LapTimeSeconds"] > (1.07 * day_fastest_lap)
    laps["Slow115"] = laps["LapTimeSeconds"] > (1.15 * day_fastest_lap)

    laps = laps.sort_values(["Round", "SessionName", "Driver", "LapNumber"]).copy()
    laps["PrevSlow107"] = laps.groupby(["Round", "SessionName", "Driver"])["Slow107"].shift(1)
    laps["NextSlow107"] = laps.groupby(["Round", "SessionName", "Driver"])["Slow107"].shift(-1)

    isolated_quali_sim_mask = (
        ~laps["Slow107"] & laps["PrevSlow107"].fillna(False) & laps["NextSlow107"].fillna(False)
    )
    laps = laps.loc[~isolated_quali_sim_mask].copy()
    laps = laps.loc[~laps["Slow115"]].copy()
    if laps.empty:
        raise ValueError("No laps remain after applying the lap-time filters")

    add_half_day_label(laps)
    print("Lap counts by period after filtering:")
    print(laps["HalfDay"].value_counts())

    paired_summary = (
        laps.groupby(["Round", "SessionName", "HalfDay"])
        .agg(avg_lap_time_seconds=("LapTimeSeconds", "mean"), sample_count=("LapTimeSeconds", "size"))
        .reset_index()
        .pivot_table(
            index=["Round", "SessionName"],
            columns="HalfDay",
            values=["avg_lap_time_seconds", "sample_count"],
        )
        .reset_index()
    )
    paired_summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in paired_summary.columns.to_flat_index()
    ]
    for period in ("forenoon", "afternoon"):
        avg_col = f"avg_lap_time_seconds_{period}"
        count_col = f"sample_count_{period}"
        if avg_col not in paired_summary.columns:
            paired_summary[avg_col] = pd.NA
        if count_col not in paired_summary.columns:
            paired_summary[count_col] = 0

    paired_summary = paired_summary.dropna(
        subset=["avg_lap_time_seconds_forenoon", "avg_lap_time_seconds_afternoon"]
    ).copy()

    if paired_summary.empty:
        raise ValueError("No sessions have both forenoon and afternoon data")

    paired_summary["DeltaSeconds"] = (
        paired_summary["avg_lap_time_seconds_afternoon"] - paired_summary["avg_lap_time_seconds_forenoon"]
    )
    paired_summary.to_csv(output_csv, index=False)
    print(f"Saved paired forenoon/afternoon summary to {output_csv}")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        paired_summary["avg_lap_time_seconds_forenoon"],
        paired_summary["avg_lap_time_seconds_afternoon"],
        color="#1f77b4",
        s=70,
        alpha=0.8,
    )

    min_time = min(
        paired_summary["avg_lap_time_seconds_forenoon"].min(),
        paired_summary["avg_lap_time_seconds_afternoon"].min(),
    )
    max_time = max(
        paired_summary["avg_lap_time_seconds_forenoon"].max(),
        paired_summary["avg_lap_time_seconds_afternoon"].max(),
    )
    ax.plot([min_time, max_time], [min_time, max_time], linestyle="--", color="black")
    ax.set_xlabel("Forenoon Average Lap Time (s)")
    ax.set_ylabel("Afternoon Average Lap Time (s)")
    ax.set_title("Forenoon vs Afternoon Pace")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_plot = output_csv.rsplit(".", 1)[0] + ".png"
    fig.savefig(output_plot, dpi=150, bbox_inches="tight")
    print(f"Saved forenoon/afternoon scatter plot to {output_plot}")

    correction_map = {
        (row.Round, row.SessionName): row.DeltaSeconds
        for row in paired_summary[["Round", "SessionName", "DeltaSeconds"]].itertuples(index=False)
    }
    print(f"Per-session afternoon - forenoon delta map: {correction_map}")
    return correction_map
