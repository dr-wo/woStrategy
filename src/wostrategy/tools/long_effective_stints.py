from __future__ import annotations

from typing import Union

import pandas as pd

from wostrategy.core.session import Session
from wostrategy.core.session_loader import load_session_laps


def export_long_effective_stints(
    year: int,
    rounds: list[Union[int, str]],
    session_names: list[Union[int, str]],
    output_csv: str,
    min_laps: int = 40,
    test: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Export all laps from effective stints longer than `min_laps` into one CSV."""
    all_laps = load_session_laps(
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
    if all_laps.empty:
        result = pd.DataFrame(
            columns=[
                "Year",
                "Round",
                "SessionName",
                "Driver",
                "EffectiveStint",
                "EffectiveStintLapNumber",
            ]
        )
    else:
        max_laps_per_effective_stint = (
            all_laps.groupby(["Year", "Round", "SessionName", "Driver", "EffectiveStint"])[
                "EffectiveStintLapNumber"
            ]
            .max()
            .reset_index(name="MaxEffectiveStintLapNumber")
        )
        valid_effective_stints = max_laps_per_effective_stint[
            max_laps_per_effective_stint["MaxEffectiveStintLapNumber"] > min_laps
        ][["Year", "Round", "SessionName", "Driver", "EffectiveStint"]]

        result = all_laps.merge(
            valid_effective_stints,
            on=["Year", "Round", "SessionName", "Driver", "EffectiveStint"],
            how="inner",
        ).copy()

    if not result.empty:
        summary = (
            result.groupby(["Driver", "Team"])["EffectiveStintLapNumber"]
            .max()
            .reset_index(name="MaxEffectiveStintLapNumber")
            .sort_values(by=["MaxEffectiveStintLapNumber", "Driver"], ascending=[False, True])
        )
        print("Summary of long effective stints:")
        print(summary.to_string(index=False))

    result.to_csv(output_csv, index=False)
    print(f"Saved {len(result)} laps to {output_csv}")
    return result
