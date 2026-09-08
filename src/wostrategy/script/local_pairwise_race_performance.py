from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wostrategy.analysis.local_pairwise_race_performance import (
    build_local_pairwise_model,
    load_production_event_inputs,
    production_relative_medians,
)


KNOWN_CASES = {
    (13, "Ferrari"),
    (5, "Red Bull Racing"),
    (11, "Audi"),
    (12, "Audi"),
    (11, "Alpine"),
    (8, "McLaren"),
    (7, "Haas F1 Team"),
}


def run_offline_diagnostic(
    *,
    input_root: str | Path,
    output_dir: str | Path,
    year: int = 2026,
    races: range = range(1, 14),
    window_sizes: tuple[int, ...] = (3, 5, 10),
) -> dict[str, Path]:
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_lists: dict[str, list[pd.DataFrame]] = {
        name: [] for name in ("observations", "edges", "teams", "components", "cycles")
    }
    production_frames: list[pd.DataFrame] = []
    for race in races:
        event_dir = input_root / f"year={year}" / f"round={race}" / "session=R"
        inputs = load_production_event_inputs(event_dir, year=year, race=race)
        production = production_relative_medians(inputs["baseline_samples"])
        production.insert(0, "Round", race)
        production_frames.append(production)
        for window_size in window_sizes:
            result = build_local_pairwise_model(
                inputs["clean_laps"],
                compound_offsets=inputs["compound_offsets"],
                team_degradation=inputs["team_degradation"],
                window_size=window_size,
            )
            for name in frame_lists:
                frame = getattr(result, name).copy()
                frame.insert(0, "WindowSize", window_size)
                frame.insert(0, "Round", race)
                frame.insert(0, "Year", year)
                frame_lists[name].append(frame)

    frames = {
        name: pd.concat(items, ignore_index=True) if items else pd.DataFrame()
        for name, items in frame_lists.items()
    }
    production = pd.concat(production_frames, ignore_index=True)
    window_summary = _window_summary(frames["teams"], frames["components"], frames["cycles"])
    variation = _team_variation_summary(frames["teams"])
    global_summary = _global_summary(
        frames["observations"],
        frames["teams"],
        frames["components"],
        frames["cycles"],
        window_summary,
        variation,
    )
    known_cases = _known_case_summary(
        production,
        frames["observations"],
        frames["edges"],
        frames["teams"],
        frames["components"],
    )

    outputs = {
        "observations": output_dir / "local_pairwise_observations.csv",
        "edges": output_dir / "local_pairwise_edges.csv",
        "teams": output_dir / "local_pairwise_team_windows.csv",
        "components": output_dir / "local_pairwise_components.csv",
        "cycles": output_dir / "local_pairwise_cycles.csv",
        "window_summary": output_dir / "local_pairwise_window_summary.csv",
        "team_variation": output_dir / "local_pairwise_team_variation.csv",
        "global_summary": output_dir / "local_pairwise_global_summary.csv",
        "known_cases": output_dir / "local_pairwise_known_cases.csv",
        "production": output_dir / "production_relative_medians.csv",
    }
    output_frames = {
        **frames,
        "window_summary": window_summary,
        "team_variation": variation,
        "global_summary": global_summary,
        "known_cases": known_cases,
        "production": production,
    }
    for name, path in outputs.items():
        output_frames[name].to_csv(path, index=False)
    return outputs


def _window_summary(
    teams: pd.DataFrame, components: pd.DataFrame, cycles: pd.DataFrame
) -> pd.DataFrame:
    keys = ["Year", "Round", "WindowSize", "WindowStart", "WindowEnd"]
    team_summary = teams.groupby(keys, as_index=False).agg(
        TeamCount=("Team", "nunique"),
        MercedesConnectedTeamCount=("ConnectedToMercedes", "sum"),
        MedianNodeDegree=("NodeDegree", "median"),
        IndirectOnlyMercedesTeamCount=("IndirectOnlyMercedesPath", "sum"),
    )
    component_summary = components.groupby(keys, as_index=False).agg(
        ComponentCount=("ComponentId", "nunique"),
        EdgeCount=("EdgeCount", "sum"),
        MedianComponentResidualRMSESeconds=("ResidualRMSESeconds", "median"),
        MaximumEdgeResidualSeconds=("LargestAbsoluteEdgeResidualSeconds", "max"),
        MedianRawComponentResidualRMSESeconds=("RawResidualRMSESeconds", "median"),
        MaximumRawEdgeResidualSeconds=("RawLargestAbsoluteEdgeResidualSeconds", "max"),
    )
    output = team_summary.merge(component_summary, on=keys, how="outer")
    if cycles.empty:
        output["TriangleCycleCount"] = 0
        output["MaximumLoopClosureSeconds"] = np.nan
    else:
        cycle_summary = cycles.groupby(keys, as_index=False).agg(
            TriangleCycleCount=("AbsoluteLoopClosureSeconds", "size"),
            MaximumLoopClosureSeconds=("AbsoluteLoopClosureSeconds", "max"),
        )
        output = output.merge(cycle_summary, on=keys, how="left")
        output["TriangleCycleCount"] = output["TriangleCycleCount"].fillna(0).astype(int)
    output["HasMercedesConnectedComponent"] = output["MercedesConnectedTeamCount"].gt(0)
    output["HasMultipleComponents"] = output["ComponentCount"].gt(1)
    return output.sort_values(keys).reset_index(drop=True)


def _team_variation_summary(teams: pd.DataFrame) -> pd.DataFrame:
    connected = teams.loc[
        teams["ConnectedToMercedes"] & teams["Team"].ne("Mercedes")
    ].sort_values(["Year", "Round", "WindowSize", "Team", "WindowStart"])
    rows: list[dict[str, object]] = []
    keys = ["Year", "Round", "WindowSize", "Team"]
    for key, group in connected.groupby(keys, sort=True):
        values = group["RelativeToMercedesSeconds"].astype(float)
        raw_values = group["RawRelativeToMercedesSeconds"].astype(float)
        adjacent = values.diff().abs().dropna()
        raw_adjacent = raw_values.diff().abs().dropna()
        rows.append(
            {
                **dict(zip(keys, key)),
                "IdentifiableWindowCount": int(len(group)),
                "LocalMedianSeconds": float(values.median()),
                "LocalMinSeconds": float(values.min()),
                "LocalMaxSeconds": float(values.max()),
                "LocalRangeSeconds": float(values.max() - values.min()),
                "LocalStdSeconds": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "MaximumAdjacentMovementSeconds": (
                    float(adjacent.max()) if len(adjacent) else np.nan
                ),
                "RawLocalMedianSeconds": float(raw_values.median()),
                "RawLocalMinSeconds": float(raw_values.min()),
                "RawLocalMaxSeconds": float(raw_values.max()),
                "RawLocalRangeSeconds": float(raw_values.max() - raw_values.min()),
                "RawMaximumAdjacentMovementSeconds": (
                    float(raw_adjacent.max()) if len(raw_adjacent) else np.nan
                ),
                "CorrectionChangeInLocalRangeSeconds": float(
                    (values.max() - values.min())
                    - (raw_values.max() - raw_values.min())
                ),
            }
        )
    return pd.DataFrame(rows)


def _global_summary(
    observations: pd.DataFrame,
    teams: pd.DataFrame,
    components: pd.DataFrame,
    cycles: pd.DataFrame,
    windows: pd.DataFrame,
    variation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window_size in sorted(teams["WindowSize"].unique()):
        team_group = teams.loc[teams["WindowSize"].eq(window_size)]
        non_reference = team_group.loc[team_group["Team"].ne("Mercedes")]
        component_group = components.loc[
            components["WindowSize"].eq(window_size) & components["EdgeCount"].gt(0)
        ]
        observation_group = observations.loc[observations["WindowSize"].eq(window_size)]
        window_group = windows.loc[windows["WindowSize"].eq(window_size)]
        cycle_group = (
            cycles.loc[cycles["WindowSize"].eq(window_size)]
            if not cycles.empty
            else cycles
        )
        variation_group = variation.loc[
            variation["WindowSize"].eq(window_size)
            & variation["IdentifiableWindowCount"].ge(2)
        ]
        connected_non_reference = non_reference.loc[non_reference["ConnectedToMercedes"]]
        rows.append(
            {
                "WindowSize": int(window_size),
                "NonEmptyWindowCount": int(len(window_group)),
                "MercedesConnectedWindowFraction": float(
                    window_group["HasMercedesConnectedComponent"].mean()
                ),
                "MercedesConnectedNonReferenceTeamWindowFraction": float(
                    non_reference["ConnectedToMercedes"].mean()
                ),
                "IndirectOnlyTeamWindowCount": int(
                    connected_non_reference["IndirectOnlyMercedesPath"].sum()
                ),
                "IndirectOnlyFractionOfMercedesConnectedTeamWindows": float(
                    connected_non_reference["IndirectOnlyMercedesPath"].mean()
                ) if len(connected_non_reference) else np.nan,
                "MedianNodeDegree": float(team_group["NodeDegree"].median()),
                "MedianEdgesPerNonEmptyWindow": float(window_group["EdgeCount"].median()),
                "MultipleComponentWindowFraction": float(
                    window_group["HasMultipleComponents"].mean()
                ),
                "NonMercedesConnectedTeamWindowFraction": float(
                    (~non_reference["ConnectedToMercedes"]).mean()
                ),
                "MedianComponentResidualRMSESeconds": float(
                    component_group["ResidualRMSESeconds"].median()
                ),
                "P90ComponentResidualRMSESeconds": float(
                    component_group["ResidualRMSESeconds"].quantile(0.9)
                ),
                "MedianAbsoluteLoopClosureSeconds": float(
                    cycle_group["AbsoluteLoopClosureSeconds"].median()
                ) if len(cycle_group) else np.nan,
                "P90AbsoluteLoopClosureSeconds": float(
                    cycle_group["AbsoluteLoopClosureSeconds"].quantile(0.9)
                ) if len(cycle_group) else np.nan,
                "LoopClosureOver0.5SecondsFraction": float(
                    cycle_group["AbsoluteLoopClosureSeconds"].gt(0.5).mean()
                ) if len(cycle_group) else np.nan,
                "LoopClosureOver1.0SecondsFraction": float(
                    cycle_group["AbsoluteLoopClosureSeconds"].gt(1.0).mean()
                ) if len(cycle_group) else np.nan,
                "MedianAbsoluteDifferentialTyreCorrectionSeconds": float(
                    observation_group["TyreCorrectionToRawSeconds"].abs().median()
                ),
                "P90AbsoluteDifferentialTyreCorrectionSeconds": float(
                    observation_group["TyreCorrectionToRawSeconds"].abs().quantile(0.9)
                ),
                "TeamRoundsWithTwoWindows": int(len(variation_group)),
                "TeamRoundLocalRangeOver0.5SecondsFraction": float(
                    variation_group["LocalRangeSeconds"].gt(0.5).mean()
                ) if len(variation_group) else np.nan,
                "TeamRoundLocalRangeOver1.0SecondsFraction": float(
                    variation_group["LocalRangeSeconds"].gt(1.0).mean()
                ) if len(variation_group) else np.nan,
                "TeamRoundAdjacentMovementOver0.5SecondsFraction": float(
                    variation_group["MaximumAdjacentMovementSeconds"].gt(0.5).mean()
                ) if len(variation_group) else np.nan,
                "RawTeamRoundLocalRangeOver0.5SecondsFraction": float(
                    variation_group["RawLocalRangeSeconds"].gt(0.5).mean()
                ) if len(variation_group) else np.nan,
                "RawTeamRoundLocalRangeOver1.0SecondsFraction": float(
                    variation_group["RawLocalRangeSeconds"].gt(1.0).mean()
                ) if len(variation_group) else np.nan,
                "CorrectionAmplifiedLocalRangeFraction": float(
                    variation_group["CorrectionChangeInLocalRangeSeconds"].gt(0).mean()
                ) if len(variation_group) else np.nan,
                "MedianCorrectionChangeInLocalRangeSeconds": float(
                    variation_group["CorrectionChangeInLocalRangeSeconds"].median()
                ) if len(variation_group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _known_case_summary(
    production: pd.DataFrame,
    observations: pd.DataFrame,
    edges: pd.DataFrame,
    teams: pd.DataFrame,
    components: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for race, team in sorted(KNOWN_CASES):
        production_row = production.loc[
            production["Round"].eq(race) & production["Team"].eq(team)
        ]
        production_value = (
            float(production_row["ProductionRelativeSeconds"].iloc[0])
            if len(production_row)
            else np.nan
        )
        for window_size in sorted(teams["WindowSize"].unique()):
            team_windows = teams.loc[
                teams["Round"].eq(race)
                & teams["Team"].eq(team)
                & teams["WindowSize"].eq(window_size)
                & teams["ConnectedToMercedes"]
            ].copy()
            values = team_windows["RelativeToMercedesSeconds"].astype(float)
            raw_values = team_windows["RawRelativeToMercedesSeconds"].astype(float)
            direct_edges = edges.loc[
                edges["Round"].eq(race)
                & edges["WindowSize"].eq(window_size)
                & (edges["TeamA"].eq(team) | edges["TeamB"].eq(team))
            ]
            case_observations = observations.loc[
                observations["Round"].eq(race)
                & observations["WindowSize"].eq(window_size)
                & (
                    observations["TeamA"].eq(team)
                    | observations["TeamB"].eq(team)
                )
            ]
            component_values = team_windows.merge(
                components,
                on=["Year", "Round", "WindowSize", "WindowStart", "WindowEnd", "ComponentId"],
                suffixes=("", "_component"),
            )
            rows.append(
                {
                    "Round": race,
                    "Team": team,
                    "WindowSize": int(window_size),
                    "ProductionRelativeSeconds": production_value,
                    "MercedesConnectedWindowCount": int(len(team_windows)),
                    "LocalWindowRanges": ";".join(
                        f"{int(row.WindowStart)}-{int(row.WindowEnd)}:"
                        f"{row.RelativeToMercedesSeconds:+.3f}"
                        for row in team_windows.sort_values("WindowStart").itertuples(index=False)
                    ),
                    "LocalMedianSeconds": float(values.median()) if len(values) else np.nan,
                    "LocalMinSeconds": float(values.min()) if len(values) else np.nan,
                    "LocalMaxSeconds": float(values.max()) if len(values) else np.nan,
                    "LocalRangeSeconds": (
                        float(values.max() - values.min()) if len(values) else np.nan
                    ),
                    "RawLocalWindowRanges": ";".join(
                        f"{int(row.WindowStart)}-{int(row.WindowEnd)}:"
                        f"{row.RawRelativeToMercedesSeconds:+.3f}"
                        for row in team_windows.sort_values("WindowStart").itertuples(index=False)
                    ),
                    "RawLocalMedianSeconds": (
                        float(raw_values.median()) if len(raw_values) else np.nan
                    ),
                    "RawLocalMinSeconds": float(raw_values.min()) if len(raw_values) else np.nan,
                    "RawLocalMaxSeconds": float(raw_values.max()) if len(raw_values) else np.nan,
                    "MedianModelMovementFromRawSeconds": float(
                        (values - raw_values).median()
                    ) if len(values) else np.nan,
                    "DirectEdgeCount": int(len(direct_edges)),
                    "DirectObservationCount": int(len(case_observations)),
                    "MedianAbsoluteDifferentialTyreCorrectionSeconds": float(
                        case_observations["TyreCorrectionToRawSeconds"].abs().median()
                    ) if len(case_observations) else np.nan,
                    "P90AbsoluteDifferentialTyreCorrectionSeconds": float(
                        case_observations["TyreCorrectionToRawSeconds"].abs().quantile(0.9)
                    ) if len(case_observations) else np.nan,
                    "MedianGraphResidualRMSESeconds": float(
                        component_values["ResidualRMSESeconds"].median()
                    ) if len(component_values) else np.nan,
                    "MaximumGraphEdgeResidualSeconds": float(
                        component_values["LargestAbsoluteEdgeResidualSeconds"].max()
                    ) if len(component_values) else np.nan,
                    "MedianGraphStandardErrorSeconds": float(
                        team_windows["GraphStandardErrorSeconds"].median()
                    ) if len(team_windows) else np.nan,
                    "MaximumGraphStandardErrorSeconds": float(
                        team_windows["GraphStandardErrorSeconds"].max()
                    ) if len(team_windows) else np.nan,
                    "IndirectOnlyWindowCount": int(
                        team_windows["IndirectOnlyMercedesPath"].sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline diagnostic for local contemporaneous pairwise Race performance."
    )
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--first-round", type=int, default=1)
    parser.add_argument("--last-round", type=int, default=13)
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument(
        "--input-root",
        default="../woData/wostrategy/race_performance_review/schema_v1",
    )
    parser.add_argument("--output-dir", default="../temp/local_pairwise_race_performance")
    args = parser.parse_args()
    paths = run_offline_diagnostic(
        input_root=args.input_root,
        output_dir=args.output_dir,
        year=args.year,
        races=range(args.first_round, args.last_round + 1),
        window_sizes=tuple(args.window_sizes),
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
