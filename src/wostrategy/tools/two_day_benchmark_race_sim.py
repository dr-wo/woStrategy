from __future__ import annotations

from typing import Union

from .forenoon_afternoon import add_half_day_label, forenoon_afternoon_delta
from .long_effective_stints import export_long_effective_stints
from .pre_season_test.prepare_race_sim import prepare_race_sim_data
from .session_values import add_session_value_column
from wostrategy.plots.pre_season_test.race_sim import plot_race_sim


def run_two_day_benchmark_race_sim(
    year: int,
    round_number: Union[int, str],
    benchmark_session: Union[int, str],
    comparison_session: Union[int, str],
    output_prefix: str,
    min_laps: int = 57,
    reference_laps: int = 57,
    test: bool = False,
    **kwargs,
):
    """
    Run a dedicated two-day race-sim analysis.
    """
    session_names = [benchmark_session, comparison_session]
    correction_map = forenoon_afternoon_delta(
        year=year,
        rounds=[round_number],
        session_names=session_names,
        output_csv=f"{output_prefix}_ampm_delta.csv",
        min_laps=min_laps,
        test=test,
        **kwargs,
    )

    long_laps = export_long_effective_stints(
        year=year,
        rounds=[round_number],
        session_names=session_names,
        output_csv=f"{output_prefix}_long_stints.csv",
        min_laps=min_laps,
        test=test,
        **kwargs,
    )
    if long_laps.empty:
        raise ValueError("No long-stint laps available for the two-day benchmark race-sim workflow")

    aligned_laps = long_laps.dropna(
        subset=["LapTime", "LapStartTime", "Round", "SessionName", "EffectiveStintLapNumber"]
    ).copy()
    aligned_laps["LapTimeSeconds"] = aligned_laps["LapTime"].dt.total_seconds()
    add_half_day_label(aligned_laps)
    add_session_value_column(aligned_laps, values=correction_map, target_column="CorrectionSeconds")
    aligned_laps["SessionCorrectedLapTimeSeconds"] = aligned_laps["LapTimeSeconds"]
    afternoon_mask = aligned_laps["HalfDay"] == "afternoon"
    aligned_laps.loc[afternoon_mask, "SessionCorrectedLapTimeSeconds"] = (
        aligned_laps.loc[afternoon_mask, "SessionCorrectedLapTimeSeconds"]
        - aligned_laps.loc[afternoon_mask, "CorrectionSeconds"]
    )

    morning_laps = aligned_laps[
        (aligned_laps["HalfDay"] == "forenoon")
        & (aligned_laps["EffectiveStintLapNumber"] >= 2)
        & (aligned_laps["EffectiveStintLapNumber"] <= reference_laps)
    ].copy()
    morning_session_summary = (
        morning_laps.groupby(["Round", "SessionName"])
        .agg(avg_lap_time_seconds=("SessionCorrectedLapTimeSeconds", "mean"))
        .reset_index()
    )
    benchmark_key = (round_number, benchmark_session)
    comparison_key = (round_number, comparison_session)
    benchmark_am = morning_session_summary[
        (morning_session_summary["Round"] == benchmark_key[0])
        & (morning_session_summary["SessionName"] == benchmark_key[1])
    ]
    comparison_am = morning_session_summary[
        (morning_session_summary["Round"] == comparison_key[0])
        & (morning_session_summary["SessionName"] == comparison_key[1])
    ]
    if benchmark_am.empty or comparison_am.empty:
        raise ValueError("Unable to compute AM-to-AM session delta for the selected two-day workflow")

    session_offset = float(comparison_am["avg_lap_time_seconds"].iloc[0] - benchmark_am["avg_lap_time_seconds"].iloc[0])
    session_offset_map = {
        benchmark_key: 0.0,
        comparison_key: session_offset,
    }
    print(f"Two-day session offset map relative to benchmark {benchmark_key}: {session_offset_map}")

    race_sim_prepared = prepare_race_sim_data(
        long_laps,
        min_laps=min_laps,
        reference_laps=reference_laps,
        correction_map=correction_map,
        session_offset_map=session_offset_map,
        benchmark_session_key=benchmark_key,
    )
    plots = plot_race_sim(
        race_sim_prepared,
        title=f"Two-Day Race Sim Benchmark R{round_number} S{benchmark_session} vs S{comparison_session}",
        output_path=f"{output_prefix}_race_sim.png",
    )

    return {
        "correction_map": correction_map,
        "session_offset_map": session_offset_map,
        "long_laps": long_laps,
        "plots": plots,
        "morning_session_summary": morning_session_summary,
    }
