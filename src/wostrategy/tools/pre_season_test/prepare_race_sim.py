from __future__ import annotations

from typing import Union

import pandas as pd

from wostrategy.plots.style_maps import build_team_style_maps
from wostrategy.tools.forenoon_afternoon import add_half_day_label
from wostrategy.tools.session_values import add_session_value_column


def prepare_race_sim_data(
    laps: pd.DataFrame,
    *,
    min_laps: int = 57,
    reference_laps: int = 57,
    correction_map: dict[tuple[Union[int, str], Union[int, str]], float] | None = None,
    session_offset_map: dict[tuple[Union[int, str], Union[int, str]], float] | None = None,
    benchmark_session_key: tuple[Union[int, str], Union[int, str]] | None = None,
) -> dict[str, object]:
    """Prepare all race-sim plot inputs without rendering."""
    required_columns = {
        "Driver",
        "Team",
        "LapTime",
        "EffectiveStint",
        "EffectiveStintLapNumber",
        "Year",
        "Round",
        "SessionName",
        "LapStartTime",
    }
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns for race sim plot: {sorted(missing_columns)}")

    plot_laps = laps.dropna(
        subset=["LapTime", "EffectiveStint", "EffectiveStintLapNumber", "Driver"]
    ).copy()
    if plot_laps.empty:
        raise ValueError("No valid laps available for plotting")

    plot_laps["LapTimeSeconds"] = plot_laps["LapTime"].dt.total_seconds()

    stint_summary = (
        plot_laps.groupby(["Year", "Round", "SessionName", "Driver", "Team", "EffectiveStint"])
        .agg(lap_count=("EffectiveStintLapNumber", "max"))
        .reset_index()
    )
    long_stints = stint_summary[stint_summary["lap_count"] > min_laps].copy()
    if long_stints.empty:
        raise ValueError(f"No effective stints longer than {min_laps} laps found")

    plot_laps = plot_laps.merge(
        long_stints[
            ["Year", "Round", "SessionName", "Driver", "Team", "EffectiveStint", "lap_count"]
        ],
        on=["Year", "Round", "SessionName", "Driver", "Team", "EffectiveStint"],
        how="inner",
    )
    if reference_laps < 1:
        raise ValueError("reference_laps must be at least 1")

    reference_lap_count = reference_laps - 1
    correction_map = correction_map or {}
    session_offset_map = session_offset_map or {}
    add_half_day_label(plot_laps)
    add_session_value_column(plot_laps, values=correction_map, target_column="DayCorrectionSeconds")
    plot_laps["CorrectedLapTimeSeconds"] = plot_laps["LapTimeSeconds"]
    corrected_afternoon_mask = plot_laps["HalfDay"] == "afternoon"
    plot_laps.loc[corrected_afternoon_mask, "CorrectedLapTimeSeconds"] = (
        plot_laps.loc[corrected_afternoon_mask, "CorrectedLapTimeSeconds"]
        - plot_laps.loc[corrected_afternoon_mask, "DayCorrectionSeconds"]
    )
    add_session_value_column(plot_laps, values=session_offset_map, target_column="SessionOffsetSeconds")
    plot_laps["AlignedCorrectedLapTimeSeconds"] = (
        plot_laps["CorrectedLapTimeSeconds"] - plot_laps["SessionOffsetSeconds"]
    )
    representative_laps = plot_laps[
        (plot_laps["EffectiveStintLapNumber"] >= 2) & (plot_laps["EffectiveStintLapNumber"] <= reference_laps)
    ].copy()
    if representative_laps.empty:
        raise ValueError("No representative laps available after excluding lap 1")

    team_color_map, team_driver_linestyle_map, _ = build_team_style_maps(representative_laps)

    def _build_reference_context(apply_correction: bool) -> dict[str, object]:
        lap_time_column = "AlignedCorrectedLapTimeSeconds" if apply_correction else "LapTimeSeconds"
        reference_candidates = representative_laps.copy()
        if benchmark_session_key is not None:
            reference_candidates = reference_candidates[
                (reference_candidates["Round"] == benchmark_session_key[0])
                & (reference_candidates["SessionName"] == benchmark_session_key[1])
            ].copy()
        reference_summary = (
            reference_candidates.groupby(
                ["Year", "Round", "SessionName", "Driver", "Team", "EffectiveStint", "lap_count"]
            )
            .agg(
                reference_total_time_seconds=(lap_time_column, "sum"),
                reference_completed_laps=("EffectiveStintLapNumber", "nunique"),
                reference_first_lap=("EffectiveStintLapNumber", "min"),
                reference_last_lap=("EffectiveStintLapNumber", "max"),
            )
            .reset_index()
        )
        reference_summary = reference_summary[
            (reference_summary["reference_completed_laps"] == reference_lap_count)
            & (reference_summary["reference_first_lap"] == 2)
            & (reference_summary["reference_last_lap"] == reference_laps)
        ].copy()
        if reference_summary.empty:
            raise ValueError(
                f"No effective stint has a complete representative window from lap 2 to lap {reference_laps}"
            )

        reference_stint = reference_summary.loc[reference_summary["reference_total_time_seconds"].idxmin()]
        reference_avg_lap_time = reference_stint["reference_total_time_seconds"] / reference_lap_count
        reference_label = (
            f"Reference: {reference_stint['Driver']} ({reference_stint['Team']}), "
            f"stint {reference_stint['EffectiveStint']}, "
            f"laps 2-{reference_laps} in {reference_stint['reference_total_time_seconds']:.3f}s, "
            f"avg {reference_avg_lap_time:.3f}s"
        )
        return {
            "lap_time_column": lap_time_column,
            "reference_avg_lap_time": reference_avg_lap_time,
            "reference_label": reference_label,
        }

    return {
        "representative_laps": representative_laps,
        "team_color_map": team_color_map,
        "team_driver_linestyle_map": team_driver_linestyle_map,
        "reference_lap_count": reference_lap_count,
        "correction_map": correction_map,
        "session_offset_map": session_offset_map,
        "reference_context": {
            "uncorrected": _build_reference_context(apply_correction=False),
            "corrected": _build_reference_context(apply_correction=True),
        },
    }
