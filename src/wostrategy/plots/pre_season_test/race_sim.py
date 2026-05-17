from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

def plot_race_sim(
    prepared_data: dict[str, object],
    title: str = None,
    output_path: str = None,
):
    representative_laps: pd.DataFrame = prepared_data["representative_laps"]  # type: ignore[assignment]
    team_color_map: dict[str, object] = prepared_data["team_color_map"]  # type: ignore[assignment]
    team_driver_linestyle_map: dict[tuple[str, str], str] = prepared_data["team_driver_linestyle_map"]  # type: ignore[assignment]
    correction_map: dict = prepared_data["correction_map"]  # type: ignore[assignment]
    session_offset_map: dict = prepared_data["session_offset_map"]  # type: ignore[assignment]
    reference_context: dict[str, dict[str, object]] = prepared_data["reference_context"]  # type: ignore[assignment]

    def _make_plot(apply_correction: bool):
        ref_key = "corrected" if apply_correction else "uncorrected"
        lap_time_column = reference_context[ref_key]["lap_time_column"]  # type: ignore[index]
        reference_avg_lap_time = float(reference_context[ref_key]["reference_avg_lap_time"])  # type: ignore[index]
        reference_label = str(reference_context[ref_key]["reference_label"])  # type: ignore[index]

        fig, ax = plt.subplots(figsize=(12, 7))

        for (year, round_number, session_name, driver, team, effective_stint), stint_laps in representative_laps.groupby(
            ["Year", "Round", "SessionName", "Driver", "Team", "EffectiveStint"], sort=False
        ):
            color = team_color_map.get(team)
            linestyle = team_driver_linestyle_map.get((team, driver), "-")

            stint_laps = stint_laps.sort_values("EffectiveStintLapNumber").copy()
            has_forenoon = (stint_laps["HalfDay"] == "forenoon").any()
            has_afternoon = (stint_laps["HalfDay"] == "afternoon").any()
            period_label = "AM+PM" if has_forenoon and has_afternoon else "AM only" if has_forenoon else "PM only"
            label = (
                f"R{round_number} S{session_name} {driver} ({team}, stint {effective_stint}, "
                f"{int(stint_laps['lap_count'].iloc[0])} laps)"
            )
            if apply_correction:
                session_correction = float(stint_laps["DayCorrectionSeconds"].iloc[0])
                session_offset = float(stint_laps["SessionOffsetSeconds"].iloc[0])
                cumulative_time = stint_laps["AlignedCorrectedLapTimeSeconds"].cumsum()
                label = f"{label} | {period_label} | corr {session_correction:.3f}s on PM | sess {session_offset:.3f}s"
            else:
                cumulative_time = stint_laps["LapTimeSeconds"].cumsum()

            stint_laps["ReferenceElapsedSeconds"] = (stint_laps["EffectiveStintLapNumber"] - 1) * reference_avg_lap_time
            stint_laps["CumulativeDeltaSeconds"] = stint_laps["ReferenceElapsedSeconds"] - cumulative_time
            x_values = pd.concat(
                [pd.Series([1], dtype="int64"), stint_laps["EffectiveStintLapNumber"].reset_index(drop=True)],
                ignore_index=True,
            )
            y_values = pd.concat(
                [pd.Series([0.0], dtype="float64"), stint_laps["CumulativeDeltaSeconds"].reset_index(drop=True)],
                ignore_index=True,
            )
            ax.plot(x_values, y_values, label=label, color=color, linewidth=2, linestyle=linestyle)

        correction_label = (
            f"Corrected by per-session afternoon delta after 4.5h: {correction_map}; "
            f"session offsets: {session_offset_map}"
            if apply_correction
            else "Uncorrected"
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Effective Stint Lap Number")
        ax.set_ylabel("Delta to Reference Pace (s)")
        ax.set_title(title or f"Race Sim Delta Plot\n{reference_label}\n{correction_label}")
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        return fig, ax

    raw_fig, raw_ax = _make_plot(apply_correction=False)
    corrected_fig, corrected_ax = _make_plot(apply_correction=True)

    if output_path:
        if "." in output_path:
            base, ext = output_path.rsplit(".", 1)
            raw_fig.savefig(f"{base}_uncorrected.{ext}", dpi=150, bbox_inches="tight")
            corrected_fig.savefig(f"{base}_corrected.{ext}", dpi=150, bbox_inches="tight")
        else:
            raw_fig.savefig(f"{output_path}_uncorrected", dpi=150, bbox_inches="tight")
            corrected_fig.savefig(f"{output_path}_corrected", dpi=150, bbox_inches="tight")

    return {
        "uncorrected": (raw_fig, raw_ax),
        "corrected": (corrected_fig, corrected_ax),
        "correction_map": correction_map,
        "session_offset_map": session_offset_map,
    }
