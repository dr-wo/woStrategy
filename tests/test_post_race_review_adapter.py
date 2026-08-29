from pathlib import Path
import json

import pandas as pd
import pytest

import wostrategy.analysis.post_race_review as adapter
from wostrategy.analysis.post_race_review import (
    MissingRacePerformanceReview,
    get_post_race_review,
)


def test_cached_review_normalizes_medium_reference_and_median_pit_loss(tmp_path: Path) -> None:
    root = tmp_path / "wostrategy/race_performance_review/schema_v1/year=2026/round=7/session=R"
    root.mkdir(parents=True)
    prefix = "race_performance_2026_7_R"
    pd.DataFrame([
        {"Scope": "global", "Compound": compound, "CompoundDeltaMedianSeconds": delta,
         "CompoundDeltaP10Seconds": delta - 0.1, "CompoundDeltaP90Seconds": delta + 0.1,
         "DegradationP10SecondsPerLap": deg - 0.01, "DegradationMedianSecondsPerLap": deg,
         "DegradationP90SecondsPerLap": deg + 0.01, "DeltaSampleCount": 10,
         "DegradationSampleCount": 10}
        for compound, delta, deg in (("SOFT", -0.5, 0.2), ("MEDIUM", -0.2, 0.1), ("HARD", 0.0, 0.05))
    ]).to_csv(root / "tyre_information.csv", index=False)
    pd.DataFrame([{"Team": "A", "Median": 80.0}]).to_csv(root / f"{prefix}_team_baseline_summary.csv", index=False)
    pd.DataFrame([{"WeightedRMSESeconds": 0.5}]).to_csv(root / f"{prefix}_sample_diagnostics.csv", index=False)
    pd.DataFrame([{"TrackStatusType": "normal", "SampleCount": 3,
                   "PitInS3LossMedianSeconds": 10.0, "PitOutS1LossMedianSeconds": 10.5,
                   "PitTotalLossMedianSeconds": 20.5,
                   "PitTotalLossMeanSeconds": 99.0}]).to_csv(root / f"{prefix}_pit_loss_summary.csv", index=False)
    (root / f"{prefix}_metadata.json").write_text(json.dumps({"sample_count": 10}))

    result = get_post_race_review(season=2026, round_number=7, data_root=tmp_path)

    assert result["retro_tyre_estimate"]["compounds"]["MEDIUM"]["performance_delta_to_medium"] == 0.0
    assert result["empirical_pit_loss"]["normal"]["total"] == 20.5
    assert result["empirical_pit_loss"]["normal"]["statistic"] == "median"


def test_force_refresh_requires_and_records_fresh_monte_carlo(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "wostrategy/race_performance_review/schema_v1/year=2026/round=7/session=R"
    root.mkdir(parents=True)
    prefix = "race_performance_2026_7_R"
    pd.DataFrame([{
        "Scope": "global", "Compound": "MEDIUM",
        "CompoundDeltaMedianSeconds": 0.0, "CompoundDeltaP10Seconds": -0.1,
        "CompoundDeltaP90Seconds": 0.1, "DegradationP10SecondsPerLap": 0.05,
        "DegradationMedianSecondsPerLap": 0.1, "DegradationP90SecondsPerLap": 0.15,
        "DeltaSampleCount": 10, "DegradationSampleCount": 10,
    }]).to_csv(root / "tyre_information.csv", index=False)
    pd.DataFrame([{"Team": "A", "Median": 80.0}]).to_csv(
        root / f"{prefix}_team_baseline_summary.csv", index=False
    )
    pd.DataFrame([{"WeightedRMSESeconds": 0.5}]).to_csv(
        root / f"{prefix}_sample_diagnostics.csv", index=False
    )
    pd.DataFrame(columns=[
        "TrackStatusType", "SampleCount", "PitInS3LossMedianSeconds",
        "PitOutS1LossMedianSeconds", "PitTotalLossMedianSeconds",
    ]).to_csv(root / f"{prefix}_pit_loss_summary.csv", index=False)
    (root / f"{prefix}_metadata.json").write_text("{}")
    calls = []

    def producer(**kwargs):
        calls.append(kwargs)
        return {"race_results": {7: object()}}

    monkeypatch.setattr(adapter, "run_race_performance_review_adapter", producer)
    result = get_post_race_review(
        season=2026, round_number=7, data_root=tmp_path,
        overrides={"sample_count": 12}, force_refresh=True,
    )
    execution = result["retro_tyre_estimate"]["provenance"]["execution"]
    assert calls[0]["use_cached_monte_carlo"] is False
    assert calls[0]["overrides"] == {"sample_count": 12}
    assert execution["monte_carlo_executed"] is True
    assert execution["executed_rounds"] == [7]

    monkeypatch.setattr(
        adapter, "run_race_performance_review_adapter",
        lambda **kwargs: {"race_results": {}},
    )
    with pytest.raises(MissingRacePerformanceReview, match="did not produce a fresh result"):
        get_post_race_review(
            season=2026, round_number=7, data_root=tmp_path, force_refresh=True
        )
