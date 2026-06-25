from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wostrategy.plots.race_labels import race_tick_labels
from wostrategy.plots.style_maps import F1_TEAM_COLORS
from wostrategy.script.race_performance_review import (
    parse_race_selector,
    race_range_label,
    safe_name,
)


SCRIPT_CONFIG = {
    "year": 2026,
    "race": "1-7",
    "session": "R",
    "team": "Red Bull Racing",
    "reference_team": "Mercedes",
    "weight_delta_kg": -12,
    "full_fuel_weight_kg": 95.0,
    "input_dir": "cache/race_performance_review",
    "output": None,
    "show": False,
}


def run_race_performance_weight_prediction(
    *,
    year: int,
    races: list[int],
    session: str,
    team: str,
    reference_team: str | None,
    weight_delta_kg: float,
    full_fuel_weight_kg: float,
    input_dir: str | Path,
    output_path: str | Path | None,
    show: bool = False,
) -> pd.DataFrame:
    if full_fuel_weight_kg <= 0:
        raise ValueError("full_fuel_weight_kg must be positive.")

    input_dir = Path(input_dir)
    reference_team = reference_team or team
    rows: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []
    print("Race performance weight prediction")
    print(f"Year: {year}")
    print(f"Races: {races}")
    print(f"Session: {session}")
    print(f"Team: {team}")
    print(f"Reference team: {reference_team}")
    print(f"Weight delta: {weight_delta_kg:+.3f} kg")
    print(f"Full fuel weight: {full_fuel_weight_kg:.3f} kg")
    print(f"Input directory: {input_dir}")

    for race in races:
        try:
            prediction_row, race_plot_rows = load_weight_prediction_race(
                    input_dir=input_dir,
                    year=year,
                    race=race,
                    session=session,
                    team=team,
                    reference_team=reference_team,
                    weight_delta_kg=weight_delta_kg,
                    full_fuel_weight_kg=full_fuel_weight_kg,
                )
            rows.append(prediction_row)
            plot_rows.extend(race_plot_rows)
        except FileNotFoundError as exc:
            print(f"{year} race {race} {session}: missing cache file; skipping {exc}")
        except ValueError as exc:
            print(f"{year} race {race} {session}: {exc}; skipping.")

    if not rows:
        raise ValueError("No race performance review cache rows were available.")

    summary = pd.DataFrame(rows).sort_values("Race").reset_index(drop=True)
    plot_summary = pd.DataFrame(plot_rows).sort_values(["Race", "Team", "Series"])
    print("\nWeight-adjusted race performance")
    print(
        summary[
            [
                "Race",
                "Team",
                "ReferenceTeam",
                "OriginalBaselineSeconds",
                "ReferenceBaselineSeconds",
                "OriginalPercentageToReference",
                "FuelRateSecondsPerLap",
                "TotalRaceLaps",
                "FuelSecondsPerKg",
                "WeightDeltaKg",
                "ProjectedBaselineSeconds",
                "ProjectedDeltaSeconds",
                "ProjectedPercentageToReference",
            ]
        ].to_string(index=False)
    )

    output_path = Path(output_path) if output_path is not None else default_output_path(
        year=year,
        races=races,
        session=session,
        team=team,
        reference_team=reference_team,
        weight_delta_kg=weight_delta_kg,
    )
    save_weight_prediction_outputs(
        summary,
        plot_summary=plot_summary,
        output_path=output_path,
        show=show,
    )
    return summary


def load_weight_prediction_race(
    *,
    input_dir: Path,
    year: int,
    race: int,
    session: str,
    team: str,
    reference_team: str,
    weight_delta_kg: float,
    full_fuel_weight_kg: float,
) -> dict[str, object]:
    team_summary = pd.read_csv(
        cached_output_path(input_dir, year=year, race=race, session=session, suffix="team_baseline_summary")
    )
    fuel_summary = pd.read_csv(
        cached_output_path(input_dir, year=year, race=race, session=session, suffix="summary_fuel_rate")
    )
    clean_laps = pd.read_csv(
        cached_output_path(input_dir, year=year, race=race, session=session, suffix="clean_laps")
    )

    if team_summary.empty:
        raise ValueError("team baseline summary is empty")
    team_row = team_summary.loc[team_summary["Team"] == team]
    if team_row.empty:
        available = ", ".join(sorted(team_summary["Team"].dropna().astype(str).unique()))
        raise ValueError(f"team {team!r} not found. Available teams: {available}")
    reference_row = team_summary.loc[team_summary["Team"] == reference_team]
    if reference_row.empty:
        available = ", ".join(sorted(team_summary["Team"].dropna().astype(str).unique()))
        raise ValueError(
            f"reference team {reference_team!r} not found. Available teams: {available}"
        )
    if fuel_summary.empty or "Median" not in fuel_summary.columns:
        raise ValueError("fuel-rate summary is empty or missing Median")

    total_laps = total_race_laps_from_clean_laps(clean_laps)
    kg_per_lap = full_fuel_weight_kg / total_laps
    fuel_rate = float(fuel_summary["Median"].iloc[0])
    fuel_seconds_per_kg = fuel_rate / kg_per_lap
    baseline = float(team_row["Median"].iloc[0])
    reference_baseline = float(reference_row["Median"].iloc[0])
    projected_delta = fuel_seconds_per_kg * weight_delta_kg
    projected_baseline = baseline + projected_delta

    prediction_row = {
        "Year": year,
        "Race": race,
        "Session": session,
        "EventName": event_name_from_laps(clean_laps),
        "Team": team,
        "ReferenceTeam": reference_team,
        "TeamBaselineMode": team_row["TeamBaselineMode"].iloc[0]
        if "TeamBaselineMode" in team_row
        else pd.NA,
        "OriginalBaselineSeconds": baseline,
        "OriginalP10Seconds": float(team_row["P10"].iloc[0]),
        "OriginalP90Seconds": float(team_row["P90"].iloc[0]),
        "ReferenceBaselineSeconds": reference_baseline,
        "OriginalPercentageToReference": baseline / reference_baseline * 100.0,
        "FuelRateSecondsPerLap": fuel_rate,
        "TotalRaceLaps": total_laps,
        "FullFuelWeightKg": full_fuel_weight_kg,
        "FuelKgPerLap": kg_per_lap,
        "FuelSecondsPerKg": fuel_seconds_per_kg,
        "WeightDeltaKg": weight_delta_kg,
        "ProjectedDeltaSeconds": projected_delta,
        "ProjectedBaselineSeconds": projected_baseline,
        "ProjectedPercentageToReference": projected_baseline
        / reference_baseline
        * 100.0,
    }
    plot_rows = all_team_plot_rows(
        team_summary=team_summary,
        year=year,
        race=race,
        session=session,
        event_name=event_name_from_laps(clean_laps),
        reference_team=reference_team,
        reference_baseline=reference_baseline,
        projected_team=team,
        projected_baseline=projected_baseline,
        weight_delta_kg=weight_delta_kg,
    )
    return prediction_row, plot_rows


def load_weight_prediction_row(
    *,
    input_dir: Path,
    year: int,
    race: int,
    session: str,
    team: str,
    reference_team: str,
    weight_delta_kg: float,
    full_fuel_weight_kg: float,
) -> dict[str, object]:
    prediction_row, _ = load_weight_prediction_race(
        input_dir=input_dir,
        year=year,
        race=race,
        session=session,
        team=team,
        reference_team=reference_team,
        weight_delta_kg=weight_delta_kg,
        full_fuel_weight_kg=full_fuel_weight_kg,
    )
    return prediction_row


def all_team_plot_rows(
    *,
    team_summary: pd.DataFrame,
    year: int,
    race: int,
    session: str,
    event_name: str | None,
    reference_team: str,
    reference_baseline: float,
    projected_team: str,
    projected_baseline: float,
    weight_delta_kg: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in team_summary.iterrows():
        baseline = float(row["Median"])
        team = str(row["Team"])
        rows.append(
            {
                "Year": year,
                "Race": race,
                "Session": session,
                "EventName": event_name,
                "Team": team,
                "ReferenceTeam": reference_team,
                "Series": "actual",
                "BaselineSeconds": baseline,
                "PercentageToReference": baseline / reference_baseline * 100.0,
                "WeightDeltaKg": 0.0,
            }
        )
    rows.append(
        {
            "Year": year,
            "Race": race,
            "Session": session,
            "EventName": event_name,
            "Team": projected_team,
            "ReferenceTeam": reference_team,
            "Series": "projected",
            "BaselineSeconds": projected_baseline,
            "PercentageToReference": projected_baseline
            / reference_baseline
            * 100.0,
            "WeightDeltaKg": weight_delta_kg,
        }
    )
    return rows


def cached_output_path(
    input_dir: Path,
    *,
    year: int,
    race: int,
    session: str,
    suffix: str,
) -> Path:
    path = input_dir / f"race_performance_{year}_{race}_{session}_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def total_race_laps_from_clean_laps(clean_laps: pd.DataFrame) -> float:
    if clean_laps.empty or "LapNumber" not in clean_laps.columns:
        raise ValueError("clean laps are empty or missing LapNumber")
    lap_number = pd.to_numeric(clean_laps["LapNumber"], errors="coerce")
    if "FuelProxyLapsRemaining" in clean_laps.columns:
        fuel_remaining = pd.to_numeric(
            clean_laps["FuelProxyLapsRemaining"],
            errors="coerce",
        )
        total_laps = (lap_number + fuel_remaining).dropna()
        if not total_laps.empty:
            return float(total_laps.max())
    lap_number = lap_number.dropna()
    if lap_number.empty:
        raise ValueError("clean laps have no numeric LapNumber")
    return float(lap_number.max())


def event_name_from_laps(clean_laps: pd.DataFrame) -> str | None:
    for column in ("EventName", "RaceName", "EventLocation", "EventCountry"):
        if column not in clean_laps.columns:
            continue
        values = clean_laps[column].dropna().astype(str)
        if not values.empty:
            return values.iloc[0]
    return None


def save_weight_prediction_outputs(
    summary: pd.DataFrame,
    *,
    plot_summary: pd.DataFrame,
    output_path: str | Path,
    show: bool = False,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_name(f"{output_path.stem}.csv")
    plot_csv_path = output_path.with_name(f"{output_path.stem}_plot_data.csv")
    summary.to_csv(csv_path, index=False)
    plot_summary.to_csv(plot_csv_path, index=False)

    fig, ax = plot_weight_prediction(plot_summary)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print("\nSaved race performance weight prediction:")
    print(f"  plot: {output_path}")
    print(f"  csv: {csv_path}")
    print(f"  plot data csv: {plot_csv_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_weight_prediction(summary: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    if summary.empty:
        raise ValueError("summary is empty")
    fig, ax = plt.subplots(figsize=(13, 7))
    actual = summary.loc[summary["Series"] == "actual"].copy()
    projected = summary.loc[summary["Series"] == "projected"].copy()
    for team, team_rows in actual.sort_values("Race").groupby("Team", sort=True):
        ax.plot(
            team_rows["Race"],
            team_rows["PercentageToReference"],
            marker="o",
            linewidth=1.8,
            linestyle="-",
            color=F1_TEAM_COLORS.get(team),
            label=team,
        )

    if not projected.empty:
        projected_team = str(projected["Team"].iloc[0])
        weight_delta = float(projected["WeightDeltaKg"].iloc[0])
        ax.plot(
            projected.sort_values("Race")["Race"],
            projected.sort_values("Race")["PercentageToReference"],
            marker="o",
            linewidth=3,
            linestyle="--",
            color=F1_TEAM_COLORS.get(projected_team),
            label=f"{projected_team} {weight_delta:+g} kg",
        )
    ax.axhline(100.0, color="black", linewidth=1, alpha=0.5)
    tick_frame = race_tick_labels(summary, round_column="Race")
    ax.set_xlabel("Round")
    if not tick_frame.empty:
        ax.set_xticks(tick_frame["Race"])
        ax.set_xticklabels(tick_frame["Label"], rotation=45, ha="right")
    reference_team = str(summary["ReferenceTeam"].iloc[0])
    ax.set_ylabel(f"Corrected race baseline pace (% of {reference_team})")
    ax.set_title(
        f"{int(summary['Year'].iloc[0])} Race Performance Weight Projection "
        f"Relative to {reference_team}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="Team", ncols=2)
    fig.tight_layout()
    return fig, ax


def default_output_path(
    *,
    year: int,
    races: list[int],
    session: str,
    team: str,
    reference_team: str,
    weight_delta_kg: float,
) -> Path:
    delta_label = f"{weight_delta_kg:+g}".replace("+", "plus").replace("-", "minus")
    return Path("temp") / (
        f"race_performance_weight_predict_{year}_{race_range_label(races)}_{session}_"
        f"{safe_name(team)}_vs_{safe_name(reference_team)}_{delta_label}kg.png"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project race performance review baselines for a selected team "
            "under a constant fuel-weight delta."
        )
    )
    parser.add_argument("--year", type=int, default=SCRIPT_CONFIG["year"])
    parser.add_argument(
        "--race",
        default=SCRIPT_CONFIG["race"],
        help="Race number or inclusive range, for example '7' or '1-7'.",
    )
    parser.add_argument("--session", default=SCRIPT_CONFIG["session"])
    parser.add_argument("--team", default=SCRIPT_CONFIG["team"])
    parser.add_argument(
        "--reference-team",
        default=SCRIPT_CONFIG["reference_team"],
        help=(
            "Reference team for relative plotting. Defaults to --team, so the "
            "solid line is 100%% and the dashed line shows only the weight effect."
        ),
    )
    parser.add_argument(
        "--weight-delta-kg",
        type=float,
        default=SCRIPT_CONFIG["weight_delta_kg"],
        help="Positive means more weight, negative means less weight.",
    )
    parser.add_argument(
        "--full-fuel-weight-kg",
        type=float,
        default=SCRIPT_CONFIG["full_fuel_weight_kg"],
        help="Full race fuel weight assumption used to convert lap fuel proxy to kg.",
    )
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_CONFIG["input_dir"])
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_CONFIG["output"],
        help="Output PNG path. A CSV with the same stem is saved alongside it.",
    )
    parser.add_argument("--show", action="store_true", default=SCRIPT_CONFIG["show"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    races = parse_race_selector(args.race)
    run_race_performance_weight_prediction(
        year=args.year,
        races=races,
        session=args.session,
        team=args.team,
        reference_team=args.reference_team,
        weight_delta_kg=args.weight_delta_kg,
        full_fuel_weight_kg=args.full_fuel_weight_kg,
        input_dir=args.input_dir,
        output_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
