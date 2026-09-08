from __future__ import annotations

import pandas as pd
import pytest

from wostrategy.analysis.local_pairwise_race_performance import build_local_pairwise_model


def _parameters() -> tuple[pd.DataFrame, pd.DataFrame]:
    offsets = pd.DataFrame(
        {"Compound": ["HARD", "MEDIUM"], "Median": [0.0, -0.2]}
    )
    degradation = pd.DataFrame(
        {
            "Team": ["Mercedes", "Alpha", "Beta"],
            "Compound": ["HARD", "MEDIUM", "MEDIUM"],
            "Median": [0.05, 0.10, 0.10],
        }
    )
    return offsets, degradation


def test_pairwise_correction_uses_only_tyre_effect_difference() -> None:
    offsets, degradation = _parameters()
    laps = pd.DataFrame(
        {
            "Team": ["Mercedes", "Alpha"],
            "Driver": ["MER", "ALP"],
            "LapNumber": [5, 5],
            "LapTimeSeconds": [80.0, 81.0],
            "Compound": ["HARD", "MEDIUM"],
            "TyreAgeLaps": [10, 20],
        }
    )

    result = build_local_pairwise_model(
        laps,
        compound_offsets=offsets,
        team_degradation=degradation,
        window_size=5,
    )

    observation = result.observations.iloc[0]
    assert observation["RawDeltaSeconds"] == pytest.approx(1.0)
    assert observation["TyreEffectDifferenceSeconds"] == pytest.approx(1.3)
    assert observation["CorrectedDeltaSeconds"] == pytest.approx(-0.3)
    alpha = result.teams.loc[result.teams["Team"].eq("Alpha")].iloc[0]
    assert alpha["RelativeToMercedesSeconds"] == pytest.approx(-0.3)
    assert alpha["RawRelativeToMercedesSeconds"] == pytest.approx(1.0)


def test_same_window_indirect_path_connects_team_to_mercedes() -> None:
    offsets, degradation = _parameters()
    laps = pd.DataFrame(
        {
            "Team": ["Mercedes", "Alpha", "Alpha", "Beta"],
            "Driver": ["MER", "ALP", "ALP", "BET"],
            "LapNumber": [1, 1, 2, 2],
            "LapTimeSeconds": [80.0, 80.5, 80.4, 81.0],
            "Compound": ["HARD", "MEDIUM", "MEDIUM", "MEDIUM"],
            "TyreAgeLaps": [1, 1, 2, 2],
        }
    )

    result = build_local_pairwise_model(
        laps,
        compound_offsets=offsets,
        team_degradation=degradation,
        window_size=5,
    )

    beta = result.teams.loc[result.teams["Team"].eq("Beta")].iloc[0]
    assert bool(beta["ConnectedToMercedes"])
    assert bool(beta["IndirectOnlyMercedesPath"])
    assert beta["NodeDegree"] == 1


def test_separate_components_remain_relative_only() -> None:
    offsets, degradation = _parameters()
    laps = pd.DataFrame(
        {
            "Team": ["Mercedes", "Alpha", "Beta"],
            "Driver": ["MER", "ALP", "BET"],
            "LapNumber": [1, 2, 2],
            "LapTimeSeconds": [80.0, 80.4, 81.0],
            "Compound": ["HARD", "MEDIUM", "MEDIUM"],
            "TyreAgeLaps": [1, 2, 2],
        }
    )

    result = build_local_pairwise_model(
        laps,
        compound_offsets=offsets,
        team_degradation=degradation,
        window_size=1,
    )

    beta = result.teams.loc[result.teams["Team"].eq("Beta")].iloc[0]
    assert not bool(beta["ConnectedToMercedes"])
    assert pd.isna(beta["RelativeToMercedesSeconds"])
    assert beta["ComponentRelativeSeconds"] == pytest.approx(0.6)


def test_edge_count_uses_independent_race_laps_not_driver_pair_cartesian_count() -> None:
    offsets, degradation = _parameters()
    laps = pd.DataFrame(
        {
            "Team": ["Mercedes", "Mercedes", "Alpha", "Alpha"],
            "Driver": ["MER1", "MER2", "ALP1", "ALP2"],
            "LapNumber": [5, 5, 5, 5],
            "LapTimeSeconds": [80.0, 80.2, 80.5, 80.7],
            "Compound": ["HARD", "HARD", "MEDIUM", "MEDIUM"],
            "TyreAgeLaps": [5, 5, 5, 5],
        }
    )

    result = build_local_pairwise_model(
        laps,
        compound_offsets=offsets,
        team_degradation=degradation,
        window_size=5,
    )

    edge = result.edges.iloc[0]
    assert edge["ObservationCount"] == 1
    assert edge["DriverPairComparisonCount"] == 4
