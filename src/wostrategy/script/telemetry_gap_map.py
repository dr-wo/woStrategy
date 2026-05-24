from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from wostrategy.core import Session, TelemetryDataLoader
from wostrategy.plots import plot_front_car_delta_circuit_map


SCRIPT_CONFIG = {
    "year": 2026,
    "race": 4,
    "section": "R",
    "driver": "ANT",
    "lap_number": 12,
    "output": None,
    "test": False,
    "show": False,
}


def plot_lap_front_car_delta_map(
    *,
    year: int,
    race: int | str,
    section: int | str,
    driver: str,
    lap_number: int,
    output_path: str | Path | None = None,
    test: bool = False,
):
    """Load one FastF1 lap and plot front-car time/distance delta maps."""
    session = Session(year=year, round=race, session_name=section, test=test)
    session.drivers([driver.upper()])
    session.laps = session.laps.loc[session.laps["LapNumber"] == lap_number]

    if session.laps.empty:
        raise ValueError(
            f"No lap found for year={year}, race={race}, section={section}, "
            f"driver={driver}, lap_number={lap_number}"
        )

    telemetry = TelemetryDataLoader(skip_lap_errors=False).load_session(
        session,
        year=year,
        round_number=race,
        session_name=section,
    )
    if telemetry.empty:
        raise ValueError(
            f"No telemetry loaded for year={year}, race={race}, section={section}, "
            f"driver={driver}, lap_number={lap_number}"
        )

    return plot_front_car_delta_circuit_map(telemetry, output_path=output_path)


def main() -> None:
    args = _parse_args()
    year = args.year
    race = _parse_round(str(args.race))
    section = args.section
    driver = args.driver
    lap_number = args.lap_number

    output_path = args.output
    if output_path is None:
        output_name = (
            f"telemetry_gap_map_{year}_{race}_{section}_"
            f"{driver}_lap{lap_number}.png"
        )
        output_path = Path("temp") / output_name
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_lap_front_car_delta_map(
        year=year,
        race=race,
        section=section,
        driver=driver,
        lap_number=lap_number,
        output_path=output_path,
        test=args.test,
    )

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved telemetry gap map to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one lap circuit maps colored by front-car time and distance delta."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=SCRIPT_CONFIG["year"],
        help="FastF1 season year.",
    )
    parser.add_argument(
        "--race",
        default=SCRIPT_CONFIG["race"],
        help="FastF1 round number or event name.",
    )
    parser.add_argument(
        "--section",
        default=SCRIPT_CONFIG["section"],
        help="FastF1 session name, for example FP1, FP2, FP3, Q, or R.",
    )
    parser.add_argument(
        "--driver",
        default=SCRIPT_CONFIG["driver"],
        help="Driver abbreviation, for example VER.",
    )
    parser.add_argument(
        "--lap-number",
        type=int,
        default=SCRIPT_CONFIG["lap_number"],
    )
    parser.add_argument("--output", type=Path, default=SCRIPT_CONFIG["output"])
    parser.add_argument(
        "--test",
        action="store_true",
        default=SCRIPT_CONFIG["test"],
        help="Load a FastF1 testing event/session.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=SCRIPT_CONFIG["show"],
        help="Show the plot window after saving.",
    )
    return parser.parse_args()


def _parse_round(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


if __name__ == "__main__":
    main()
