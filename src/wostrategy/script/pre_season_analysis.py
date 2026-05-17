"""Pre-season analysis script wiring tools and plots together."""

from pathlib import Path

from wostrategy.core.session import Session
from wostrategy.tools import (
    export_long_effective_stints,
    forenoon_afternoon_delta,
    load_all_session_laps,
    run_two_day_benchmark_race_sim,
)
from wostrategy.tools.pre_season_test import (
    load_single_lap_comparison_laps,
    prepare_cumulative_laps_by_day_data,
    prepare_single_lap_comparison_data,
)
from wostrategy.plots import plot_cumulative_laps_by_day, plot_race_sim, plot_single_lap_comparison

__all__ = [
    "Session",
    "export_long_effective_stints",
    "load_all_session_laps",
    "plot_race_sim",
    "plot_cumulative_laps_by_day",
    "plot_single_lap_comparison",
    "run_two_day_benchmark_race_sim",
    "forenoon_afternoon_delta",
]


if __name__ == "__main__":
    output_dir = Path("temp")
    output_dir.mkdir(exist_ok=True)

    full_laps = load_all_session_laps(
        year=2026,
        rounds=[1, 2],
        session_names=[1, 2, 3],
        test=True,
    )

    cumulative_laps_plots = plot_cumulative_laps_by_day(
        prepare_cumulative_laps_by_day_data(full_laps),
        output_path=output_dir / "cumulative_laps_by_day.png",
    )

    two_day_race_sim = run_two_day_benchmark_race_sim(
        year=2026,
        round_number=2,
        benchmark_session=2,
        comparison_session=3,
        output_prefix=str(output_dir / "r2s2_r2s3"),
        min_laps=30,
        reference_laps=57,
        test=True,
    )

    single_lap_laps = load_single_lap_comparison_laps(
        year=2026,
        rounds=[2],
        session_names=[2, 3],
        test=True
    )
    single_lap_plot = plot_single_lap_comparison(
        prepare_single_lap_comparison_data(
            single_lap_laps,
            correction_map=two_day_race_sim["correction_map"],
        ),
        output_path=output_dir / "single_lap_comparison.png",
    )

    two_day_race_sim["plots"]["uncorrected"][0].show()
    two_day_race_sim["plots"]["corrected"][0].show()
    cumulative_laps_plots["drivers"][0].show()
    cumulative_laps_plots["teams"][0].show()
    single_lap_plot[0].show()
