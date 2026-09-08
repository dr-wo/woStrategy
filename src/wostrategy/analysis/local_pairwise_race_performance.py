"""Offline diagnostic for contemporaneous pairwise Race performance.

The prototype deliberately consumes already-selected production clean laps and
posterior-median tyre parameters.  It does not refit or alter the production
Race estimator, infer fuel, extrapolate laps to tyre age zero, or link windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LocalPairwiseResult:
    observations: pd.DataFrame
    edges: pd.DataFrame
    teams: pd.DataFrame
    components: pd.DataFrame
    cycles: pd.DataFrame


def build_local_pairwise_model(
    clean_laps: pd.DataFrame,
    *,
    compound_offsets: pd.DataFrame,
    team_degradation: pd.DataFrame,
    window_size: int = 5,
    reference_team: str = "Mercedes",
    clean_lap_noise_sigma: float = 0.5,
) -> LocalPairwiseResult:
    """Construct exact-lap pairwise edges and independently solve each window."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    required = {"Team", "Driver", "LapNumber", "LapTimeSeconds", "Compound", "TyreAgeLaps"}
    missing = required.difference(clean_laps.columns)
    if missing:
        raise ValueError(f"Clean laps are missing required columns: {sorted(missing)}")

    laps = clean_laps.loc[:, sorted(required)].copy()
    for column in ("LapNumber", "LapTimeSeconds", "TyreAgeLaps"):
        laps[column] = pd.to_numeric(laps[column], errors="coerce")
    laps["Compound"] = laps["Compound"].astype("string").str.upper()
    laps = laps.dropna(subset=list(required)).copy()
    laps["WindowStart"] = (((laps["LapNumber"].astype(int) - 1) // window_size) * window_size) + 1
    laps["WindowEnd"] = laps["WindowStart"] + window_size - 1

    offsets = _parameter_map(
        compound_offsets,
        key_columns=("Compound",),
        value_column="Median",
    )
    degradation = _parameter_map(
        team_degradation,
        key_columns=("Team", "Compound"),
        value_column="Median",
    )
    missing_parameters = sorted(
        {
            (str(row.Team), str(row.Compound))
            for row in laps.itertuples(index=False)
            if (str(row.Compound),) not in offsets
            or (str(row.Team), str(row.Compound)) not in degradation
        }
    )
    if missing_parameters:
        raise ValueError(f"Missing posterior-median tyre parameters: {missing_parameters}")

    observation_rows: list[dict[str, object]] = []
    for (window_start, window_end, lap_number), lap_group in laps.groupby(
        ["WindowStart", "WindowEnd", "LapNumber"], sort=True
    ):
        teams = sorted(lap_group["Team"].astype(str).unique())
        for team_a, team_b in combinations(teams, 2):
            rows_a = lap_group.loc[lap_group["Team"].astype(str).eq(team_a)]
            rows_b = lap_group.loc[lap_group["Team"].astype(str).eq(team_b)]
            for row_a in rows_a.itertuples(index=False):
                for row_b in rows_b.itertuples(index=False):
                    offset_a = offsets[(str(row_a.Compound),)]
                    offset_b = offsets[(str(row_b.Compound),)]
                    degradation_a = degradation[(team_a, str(row_a.Compound))]
                    degradation_b = degradation[(team_b, str(row_b.Compound))]
                    effect_a = offset_a + degradation_a * float(row_a.TyreAgeLaps)
                    effect_b = offset_b + degradation_b * float(row_b.TyreAgeLaps)
                    raw_delta = float(row_a.LapTimeSeconds) - float(row_b.LapTimeSeconds)
                    tyre_effect_delta = effect_a - effect_b
                    observation_rows.append(
                        {
                            "WindowStart": int(window_start),
                            "WindowEnd": int(window_end),
                            "RaceLap": int(lap_number),
                            "TeamA": team_a,
                            "TeamB": team_b,
                            "DriverA": str(row_a.Driver),
                            "DriverB": str(row_b.Driver),
                            "CompoundA": str(row_a.Compound),
                            "CompoundB": str(row_b.Compound),
                            "TyreAgeA": float(row_a.TyreAgeLaps),
                            "TyreAgeB": float(row_b.TyreAgeLaps),
                            "TyreAgeDifference": float(row_a.TyreAgeLaps - row_b.TyreAgeLaps),
                            "RawDeltaSeconds": raw_delta,
                            "TyreEffectDifferenceSeconds": tyre_effect_delta,
                            "TyreCorrectionToRawSeconds": -tyre_effect_delta,
                            "CorrectedDeltaSeconds": raw_delta - tyre_effect_delta,
                        }
                    )
    observations = pd.DataFrame(observation_rows)
    edges = _aggregate_edges(
        observations,
        clean_lap_noise_sigma=clean_lap_noise_sigma,
    )
    teams, components, fitted_edges = _solve_windows(
        laps,
        edges,
        reference_team=reference_team,
    )
    cycles = _cycle_diagnostics(fitted_edges)
    return LocalPairwiseResult(
        observations=observations,
        edges=fitted_edges,
        teams=teams,
        components=components,
        cycles=cycles,
    )


def load_production_event_inputs(
    event_dir: str | Path, *, year: int, race: int
) -> dict[str, pd.DataFrame]:
    """Load only persisted production artifacts; no FastF1/session access occurs."""
    event_dir = Path(event_dir)
    prefix = f"race_performance_{year}_{race}_R"
    return {
        "clean_laps": pd.read_csv(event_dir / f"{prefix}_clean_laps.csv", low_memory=False),
        "compound_offsets": pd.read_csv(event_dir / f"{prefix}_summary_compound_delta.csv"),
        "team_degradation": pd.read_csv(
            event_dir / f"{prefix}_summary_team_compound_degradation.csv"
        ),
        "baseline_samples": pd.read_csv(
            event_dir / f"{prefix}_team_baseline_samples.csv",
            usecols=["SampleId", "Team", "CorrectedBaselinePaceSeconds", "Weight"],
        ),
    }


def production_relative_medians(
    baseline_samples: pd.DataFrame,
    *,
    reference_team: str = "Mercedes",
) -> pd.DataFrame:
    """Return paired posterior medians of team minus reference-team baseline."""
    reference = baseline_samples.loc[
        baseline_samples["Team"].eq(reference_team),
        ["SampleId", "CorrectedBaselinePaceSeconds"],
    ].rename(columns={"CorrectedBaselinePaceSeconds": "ReferenceBaselineSeconds"})
    paired = baseline_samples.merge(reference, on="SampleId", how="inner")
    paired["RelativeToReferenceSeconds"] = (
        paired["CorrectedBaselinePaceSeconds"] - paired["ReferenceBaselineSeconds"]
    )
    rows = []
    for team, group in paired.groupby("Team", sort=True):
        rows.append(
            {
                "Team": team,
                "ProductionRelativeSeconds": _weighted_quantile(
                    group["RelativeToReferenceSeconds"], group["Weight"], 0.5
                ),
            }
        )
    return pd.DataFrame(rows)


def _parameter_map(
    frame: pd.DataFrame,
    *,
    key_columns: tuple[str, ...],
    value_column: str,
) -> dict[tuple[str, ...], float]:
    required = {*key_columns, value_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Parameter frame is missing columns: {sorted(missing)}")
    output: dict[tuple[str, ...], float] = {}
    for row in frame.loc[:, [*key_columns, value_column]].itertuples(index=False, name=None):
        key = tuple(
            str(value).upper() if column == "Compound" else str(value)
            for column, value in zip(key_columns, row[:-1])
        )
        output[key] = float(row[-1])
    return output


def _aggregate_edges(observations: pd.DataFrame, *, clean_lap_noise_sigma: float) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["WindowStart", "WindowEnd", "TeamA", "TeamB"]
    for key, group in observations.groupby(keys, sort=True):
        lap_means = group.groupby("RaceLap", as_index=False).agg(
            RawDeltaSeconds=("RawDeltaSeconds", "mean"),
            TyreCorrectionToRawSeconds=("TyreCorrectionToRawSeconds", "mean"),
            CorrectedDeltaSeconds=("CorrectedDeltaSeconds", "mean"),
        )
        corrected = lap_means["CorrectedDeltaSeconds"].astype(float)
        observation_count = int(len(lap_means))
        dispersion = (
            float(corrected.std(ddof=1)) if observation_count > 1 else float("nan")
        )
        empirical_variance = 0.0 if np.isnan(dispersion) else dispersion**2
        uncertainty = float(
            np.sqrt(
                (empirical_variance + 2.0 * clean_lap_noise_sigma**2)
                / observation_count
            )
        )
        rows.append(
            {
                **dict(zip(keys, key)),
                "OverlapRaceLaps": _integer_ranges(group["RaceLap"]),
                "ContributingDriverPairs": ";".join(
                    sorted(set(group["DriverA"] + "-" + group["DriverB"]))
                ),
                "CompoundPairs": ";".join(
                    sorted(set(group["CompoundA"] + "-" + group["CompoundB"]))
                ),
                "TyreAgeDifferenceMin": float(group["TyreAgeDifference"].min()),
                "TyreAgeDifferenceMedian": float(group["TyreAgeDifference"].median()),
                "TyreAgeDifferenceMax": float(group["TyreAgeDifference"].max()),
                "RawRelativePaceSeconds": float(lap_means["RawDeltaSeconds"].mean()),
                "TyreCorrectionSeconds": float(
                    lap_means["TyreCorrectionToRawSeconds"].mean()
                ),
                "CorrectedRelativePaceSeconds": float(corrected.mean()),
                "ObservationCount": observation_count,
                "DriverPairComparisonCount": int(len(group)),
                "CorrectedDispersionSeconds": dispersion,
                "EdgeUncertaintySeconds": uncertainty,
                "Weight": 1.0 / uncertainty**2,
            }
        )
    return pd.DataFrame(rows)


def _solve_windows(
    laps: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    reference_team: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    team_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    fitted_edge_frames: list[pd.DataFrame] = []
    for (window_start, window_end), window_laps in laps.groupby(
        ["WindowStart", "WindowEnd"], sort=True
    ):
        nodes = sorted(window_laps["Team"].astype(str).unique())
        if edges.empty:
            window_edges = pd.DataFrame()
        else:
            window_edges = edges.loc[
                edges["WindowStart"].eq(window_start) & edges["WindowEnd"].eq(window_end)
            ].copy()
        adjacency = {node: set() for node in nodes}
        for edge in window_edges.itertuples(index=False):
            adjacency[edge.TeamA].add(edge.TeamB)
            adjacency[edge.TeamB].add(edge.TeamA)
        for component_id, component_nodes in enumerate(_connected_components(adjacency), start=1):
            component_set = set(component_nodes)
            if window_edges.empty:
                component_edges = window_edges.copy()
            else:
                component_edges = window_edges.loc[
                    window_edges["TeamA"].isin(component_set)
                    & window_edges["TeamB"].isin(component_set)
                ].copy()
            component_reference = (
                reference_team if reference_team in component_set else component_nodes[0]
            )
            estimates, residuals, standard_errors = _weighted_graph_fit(
                component_nodes,
                component_edges,
                reference=component_reference,
                target_column="CorrectedRelativePaceSeconds",
            )
            raw_estimates, raw_residuals, _ = _weighted_graph_fit(
                component_nodes,
                component_edges,
                reference=component_reference,
                target_column="RawRelativePaceSeconds",
            )
            connected_to_reference = reference_team in component_set
            edge_count = int(len(component_edges))
            if edge_count:
                component_edges["ComponentId"] = component_id
                component_edges["ReferenceTeam"] = component_reference
                component_edges["FittedDeltaSeconds"] = [
                    estimates[a] - estimates[b]
                    for a, b in zip(component_edges["TeamA"], component_edges["TeamB"])
                ]
                component_edges["ResidualSeconds"] = residuals
                component_edges["RawFittedDeltaSeconds"] = [
                    raw_estimates[a] - raw_estimates[b]
                    for a, b in zip(component_edges["TeamA"], component_edges["TeamB"])
                ]
                component_edges["RawResidualSeconds"] = raw_residuals
                fitted_edge_frames.append(component_edges)
                weighted_rmse = float(
                    np.sqrt(
                        np.average(
                            np.square(residuals),
                            weights=component_edges["Weight"].to_numpy(float),
                        )
                    )
                )
                residual_rmse = float(np.sqrt(np.mean(np.square(residuals))))
                maximum_residual = float(np.max(np.abs(residuals)))
                raw_residual_rmse = float(np.sqrt(np.mean(np.square(raw_residuals))))
                raw_maximum_residual = float(np.max(np.abs(raw_residuals)))
            else:
                weighted_rmse = residual_rmse = maximum_residual = 0.0
                raw_residual_rmse = raw_maximum_residual = 0.0
            component_rows.append(
                {
                    "WindowStart": int(window_start),
                    "WindowEnd": int(window_end),
                    "ComponentId": component_id,
                    "ReferenceTeam": component_reference,
                    "ConnectedToMercedes": connected_to_reference,
                    "TeamCount": len(component_nodes),
                    "Teams": ";".join(component_nodes),
                    "EdgeCount": edge_count,
                    "ResidualRMSESeconds": residual_rmse,
                    "WeightedResidualRMSESeconds": weighted_rmse,
                    "LargestAbsoluteEdgeResidualSeconds": maximum_residual,
                    "RawResidualRMSESeconds": raw_residual_rmse,
                    "RawLargestAbsoluteEdgeResidualSeconds": raw_maximum_residual,
                }
            )
            for team in component_nodes:
                team_laps = window_laps.loc[window_laps["Team"].astype(str).eq(team)]
                degree = len(adjacency[team])
                has_direct_reference_edge = reference_team in adjacency[team]
                team_rows.append(
                    {
                        "WindowStart": int(window_start),
                        "WindowEnd": int(window_end),
                        "ComponentId": component_id,
                        "Team": team,
                        "Drivers": ";".join(sorted(team_laps["Driver"].astype(str).unique())),
                        "AcceptedLapCount": int(len(team_laps)),
                        "AcceptedRaceLaps": _integer_ranges(team_laps["LapNumber"]),
                        "NodeDegree": degree,
                        "ReferenceTeam": component_reference,
                        "ConnectedToMercedes": connected_to_reference,
                        "DirectMercedesEdge": team == reference_team or has_direct_reference_edge,
                        "IndirectOnlyMercedesPath": bool(
                            connected_to_reference
                            and team != reference_team
                            and not has_direct_reference_edge
                        ),
                        "ComponentRelativeSeconds": estimates[team],
                        "GraphStandardErrorSeconds": standard_errors[team],
                        "RelativeToMercedesSeconds": (
                            estimates[team] if connected_to_reference else float("nan")
                        ),
                        "RawComponentRelativeSeconds": raw_estimates[team],
                        "RawRelativeToMercedesSeconds": (
                            raw_estimates[team] if connected_to_reference else float("nan")
                        ),
                    }
                )
    fitted_edges = (
        pd.concat(fitted_edge_frames, ignore_index=True)
        if fitted_edge_frames
        else edges.copy()
    )
    return pd.DataFrame(team_rows), pd.DataFrame(component_rows), fitted_edges


def _weighted_graph_fit(
    nodes: list[str],
    edges: pd.DataFrame,
    *,
    reference: str,
    target_column: str,
) -> tuple[dict[str, float], np.ndarray, dict[str, float]]:
    if len(nodes) == 1:
        return {nodes[0]: 0.0}, np.array([], dtype=float), {nodes[0]: 0.0}
    free_nodes = [node for node in nodes if node != reference]
    index = {node: position for position, node in enumerate(free_nodes)}
    design = np.zeros((len(edges), len(free_nodes)), dtype=float)
    target = edges[target_column].to_numpy(float)
    for row_index, edge in enumerate(edges.itertuples(index=False)):
        if edge.TeamA != reference:
            design[row_index, index[edge.TeamA]] = 1.0
        if edge.TeamB != reference:
            design[row_index, index[edge.TeamB]] = -1.0
    sqrt_weight = np.sqrt(edges["Weight"].to_numpy(float))
    solution, _, _, _ = np.linalg.lstsq(
        design * sqrt_weight[:, None], target * sqrt_weight, rcond=None
    )
    estimates = {reference: 0.0, **dict(zip(free_nodes, solution))}
    residuals = design @ solution - target
    information = (design * sqrt_weight[:, None]).T @ (design * sqrt_weight[:, None])
    covariance = np.linalg.pinv(information)
    standard_errors = {
        reference: 0.0,
        **{
            node: float(np.sqrt(max(covariance[position, position], 0.0)))
            for node, position in index.items()
        },
    }
    return estimates, residuals, standard_errors


def _connected_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    components: list[list[str]] = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node].difference(component))
        unseen.difference_update(component)
        components.append(sorted(component))
    return components


def _cycle_diagnostics(edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if edges.empty:
        return pd.DataFrame()
    for (window_start, window_end, component_id), group in edges.groupby(
        ["WindowStart", "WindowEnd", "ComponentId"], sort=True
    ):
        values = {
            (row.TeamA, row.TeamB): float(row.CorrectedRelativePaceSeconds)
            for row in group.itertuples(index=False)
        }
        nodes = sorted(set(group["TeamA"]).union(group["TeamB"]))
        for team_a, team_b, team_c in combinations(nodes, 3):
            if not all(
                pair in values
                for pair in ((team_a, team_b), (team_a, team_c), (team_b, team_c))
            ):
                continue
            closure = values[(team_a, team_b)] + values[(team_b, team_c)] - values[(team_a, team_c)]
            rows.append(
                {
                    "WindowStart": int(window_start),
                    "WindowEnd": int(window_end),
                    "ComponentId": int(component_id),
                    "TeamA": team_a,
                    "TeamB": team_b,
                    "TeamC": team_c,
                    "LoopClosureSeconds": closure,
                    "AbsoluteLoopClosureSeconds": abs(closure),
                }
            )
    return pd.DataFrame(rows)


def _integer_ranges(values: pd.Series) -> str:
    numbers = sorted(set(pd.to_numeric(values, errors="coerce").dropna().astype(int)))
    if not numbers:
        return ""
    ranges: list[str] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = number
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ";".join(ranges)


def _weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    clean = pd.DataFrame({"value": values, "weight": weights}).dropna()
    clean = clean.loc[clean["weight"] > 0].sort_values("value")
    if clean.empty:
        return float("nan")
    cumulative = clean["weight"].cumsum()
    target = quantile * float(clean["weight"].sum())
    return float(clean.loc[cumulative.ge(target), "value"].iloc[0])


__all__ = [
    "LocalPairwiseResult",
    "build_local_pairwise_model",
    "load_production_event_inputs",
    "production_relative_medians",
]
