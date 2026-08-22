from __future__ import annotations

import pandas as pd
import pytest

from wostrategy.analysis.retro_tyre_results import RetroTyreResultExtractor


def preview():
    return {
        "season": 2026, "round": 1, "event": "Example Grand Prix",
        "hard_compound": "C2", "medium_compound": "C3", "soft_compound": "C4",
    }


def write_tyre_information(root, soft_degradation=0.3):
    directory = root / "wostrategy/race_performance_review/schema_v1/year=2026/round=1/session=R"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        _row("global", None, "HARD", 0.0, 0.1),
        _row("global", None, "MEDIUM", -0.3, 0.2),
        _row("global", None, "SOFT", -0.7, soft_degradation),
        _row("team", "Example Team", "SOFT", -0.7, 0.9),
    ]).to_csv(directory / "tyre_information.csv", index=False)
    return directory


def _row(scope, team, compound, performance, degradation):
    return {
        "Scope": scope, "Team": team, "Compound": compound,
        "ReferenceCompound": "HARD",
        "CompoundDeltaP10Seconds": performance - 0.1,
        "CompoundDeltaMedianSeconds": performance,
        "CompoundDeltaP90Seconds": performance + 0.1,
        "DegradationP10SecondsPerLap": degradation - 0.01,
        "DegradationMedianSecondsPerLap": degradation,
        "DegradationP90SecondsPerLap": degradation + 0.01,
    }


def test_extractor_reads_global_saved_semantics_without_team_contamination(tmp_path):
    directory = write_tyre_information(tmp_path)
    pd.DataFrame([
        {"Compound": "SOFT", "Driver": "AAA", "LongRunId": 1, "Stint": 2},
        {"Compound": "SOFT", "Driver": "AAA", "LongRunId": 1, "Stint": 2},
        {"Compound": "SOFT", "Driver": "BBB", "LongRunId": 2, "Stint": 3},
        {"Compound": "HARD", "Driver": "AAA", "LongRunId": 3, "Stint": 1},
    ]).to_csv(directory / "race_performance_2026_1_R_clean_laps.csv", index=False)
    pd.DataFrame([{
        "EffectiveSampleSize": 1234.5, "WeightedRMSESeconds": 0.42,
    }]).to_csv(directory / "race_performance_2026_1_R_sample_diagnostics.csv", index=False)
    extractor = RetroTyreResultExtractor(data_root=tmp_path, legacy_root=tmp_path / "legacy")
    rows = extractor.extract_event(
        season=2026, round_number=1, source_dir=directory, preview=preview()
    )
    by_role = {row.compound_role: row for row in rows}
    assert by_role["HARD"].performance == 0.0
    assert by_role["SOFT"].performance == -0.7
    assert by_role["SOFT"].degradation == 0.3
    assert by_role["SOFT"].absolute_compound == "C4"
    assert by_role["SOFT"].performance_p10 == pytest.approx(-0.8)
    assert by_role["SOFT"].support_clean_lap_count == 3
    assert by_role["SOFT"].support_run_count == 2
    assert by_role["SOFT"].support_stint_count == 2
    assert by_role["SOFT"].retro_effective_sample_size == 1234.5
    assert by_role["SOFT"].retro_weighted_rmse == 0.42
    assert all(row.degradation != 0.9 for row in rows)


def test_unchanged_artifact_is_not_duplicated_and_changed_artifact_replaces(tmp_path):
    write_tyre_information(tmp_path)
    extractor = RetroTyreResultExtractor(data_root=tmp_path, legacy_root=tmp_path / "legacy")
    first = extractor.update(season=2026, previews=[preview()])
    initial = pd.read_csv(first.csv_path)
    second = extractor.update(season=2026, previews=[preview()])
    repeated = pd.read_csv(second.csv_path)
    assert second.unchanged_events == 1
    assert len(repeated) == 3
    pd.testing.assert_frame_equal(initial, repeated)

    write_tyre_information(tmp_path, soft_degradation=0.4)
    changed = extractor.update(season=2026, previews=[preview()])
    rows = pd.read_csv(changed.csv_path)
    assert changed.replaced_events == 1
    assert rows.loc[rows["compound_role"].eq("SOFT"), "degradation"].item() == 0.4


def test_extractor_falls_back_to_authoritative_weighted_summaries(tmp_path):
    directory = (
        tmp_path / "wostrategy/race_performance_review/schema_v1"
        / "year=2026/round=1/session=R"
    )
    directory.mkdir(parents=True)
    prefix = directory / "race_performance_2026_1_R"
    pd.DataFrame([
        {"Compound": "HARD", "P10": 0.09, "Median": 0.1, "P90": 0.11},
        {"Compound": "MEDIUM", "P10": 0.19, "Median": 0.2, "P90": 0.21},
    ]).to_csv(f"{prefix}_summary_compound_degradation.csv", index=False)
    pd.DataFrame([
        {"Compound": "HARD", "CompoundDeltaReference": "HARD",
         "P10": 0.0, "Median": 0.0, "P90": 0.0},
        {"Compound": "MEDIUM", "CompoundDeltaReference": "HARD",
         "P10": -0.4, "Median": -0.3, "P90": -0.2},
    ]).to_csv(f"{prefix}_summary_compound_delta.csv", index=False)
    rows = RetroTyreResultExtractor(
        data_root=tmp_path, legacy_root=tmp_path / "legacy"
    ).extract_event(
        season=2026, round_number=1, source_dir=directory, preview=preview()
    )
    medium = next(row for row in rows if row.compound_role == "MEDIUM")
    assert medium.performance == -0.3
    assert medium.degradation == 0.2
    assert medium.performance_p10 == -0.4
