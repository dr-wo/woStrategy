"""Stable production adapter over the established race performance/Retro MC outputs."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from wodata import get_race_performance_review_event_dir, get_race_performance_review_root


class MissingRacePerformanceReview(FileNotFoundError):
    pass


def get_post_race_review(
    *,
    season: int,
    round_number: int,
    data_root: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Return normalized race-performance, Retro tyre, quality and pit-loss data.

    Existing cached producer artifacts are reused. If absent or explicitly
    refreshed, the established `run_race_performance_review` producer is called.
    """
    root = get_race_performance_review_event_dir(
        year=season, round_number=round_number, session="R", data_root=data_root
    )
    required = _paths(root, season, round_number)
    execution: dict[str, Any] | None = None
    if force_refresh or any(not path.exists() for path in required.values()):
        producer_result = run_race_performance_review_adapter(
            season=season,
            races=[round_number],
            data_root=data_root,
            overrides=overrides or {},
            use_cached_monte_carlo=not force_refresh,
        )
        executed_rounds = sorted(int(value) for value in producer_result["race_results"])
        execution = {
            "producer_called": True,
            "monte_carlo_executed": round_number in executed_rounds,
            "executed_rounds": executed_rounds,
            "use_cached_monte_carlo": not force_refresh,
        }
        if force_refresh and round_number not in executed_rounds:
            raise MissingRacePerformanceReview(
                "Canonical Retro MC did not produce a fresh result for "
                f"{season} round {round_number}."
            )
    missing = [path for path in required.values() if not path.exists()]
    if missing:
        raise MissingRacePerformanceReview(
            "Race performance review did not produce required artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    return _normalize(required, season, round_number, execution=execution)


def get_race_performance_review(**kwargs) -> dict[str, Any]:
    review = get_post_race_review(**kwargs)
    return review["race_performance"]


def get_retro_tyre_estimate(**kwargs) -> dict[str, Any]:
    review = get_post_race_review(**kwargs)
    return review["retro_tyre_estimate"]


def get_empirical_median_pit_loss(**kwargs) -> dict[str, Any]:
    review = get_post_race_review(**kwargs)
    return review["empirical_pit_loss"]


def _paths(root: Path, season: int, round_number: int) -> dict[str, Path]:
    prefix = f"race_performance_{season}_{round_number}_R"
    return {
        "tyre": root / "tyre_information.csv",
        "teams": root / f"{prefix}_team_baseline_summary.csv",
        "quality": root / f"{prefix}_sample_diagnostics.csv",
        "pit": root / f"{prefix}_pit_loss_summary.csv",
        "metadata": root / f"{prefix}_metadata.json",
    }


def run_race_performance_review_adapter(
    *,
    season: int,
    races: list[int],
    data_root: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    use_cached_monte_carlo: bool = True,
) -> dict[str, object]:
    """Call the canonical race-review producer with stable defaults and overrides."""
    from wostrategy.script.race_performance_review import (
        SCRIPT_CONFIG,
        run_race_performance_review,
    )

    signature = inspect.signature(run_race_performance_review)
    aliases = {
        "compound_degradation_bounds_json": "compound_degradation_bounds",
        "compound_delta_bounds_json": "compound_delta_bounds",
        "lap_compound_overrides_json": "lap_compound_overrides",
    }
    defaults = dict(SCRIPT_CONFIG)
    defaults.update({
        "year": season,
        "races": list(races),
        "session": "R",
        "output_dir": get_race_performance_review_root(data_root),
        "compound_degradation_bounds": {},
        "compound_delta_bounds": {},
        "lap_compound_overrides": [],
        "use_cached_monte_carlo": use_cached_monte_carlo,
    })
    overrides = dict(overrides or {})
    for plot_only_key in (
        "reference_team",
        "plot",
        "plot_uncertainty_band",
        "plot_rmse_background",
        "plot_output",
        "show",
    ):
        overrides.pop(plot_only_key, None)
    unknown = set(overrides).difference(signature.parameters)
    if unknown:
        raise ValueError(
            "Unsupported race performance override(s): " + ", ".join(sorted(unknown))
        )
    defaults.update(overrides)
    # Workflow identity and execution freshness are orchestration guarantees,
    # not user-tunable analytical settings.
    defaults.update({
        "year": season,
        "races": list(races),
        "session": "R",
        "output_dir": get_race_performance_review_root(data_root),
        "use_cached_monte_carlo": use_cached_monte_carlo,
    })
    kwargs = {}
    for name, parameter in signature.parameters.items():
        if name in defaults:
            kwargs[name] = defaults[name]
        elif name in aliases and aliases[name] in defaults:
            kwargs[name] = defaults[aliases[name]]
        elif parameter.default is inspect.Parameter.empty:
            raise ValueError(f"No default is available for race review parameter {name!r}")
    return run_race_performance_review(**kwargs)


def _normalize(
    paths: Mapping[str, Path],
    season: int,
    round_number: int,
    *,
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tyres = pd.read_csv(paths["tyre"])
    global_tyres = tyres.loc[tyres["Scope"].astype(str).str.lower().eq("global")].copy()
    by_compound = {str(row["Compound"]).upper(): row for row in global_tyres.to_dict("records")}
    if "MEDIUM" not in by_compound:
        raise ValueError("Retro tyre output has no global MEDIUM row")
    medium_delta = float(by_compound["MEDIUM"]["CompoundDeltaMedianSeconds"])
    compounds = {}
    for compound in ("SOFT", "MEDIUM", "HARD"):
        row = by_compound.get(compound)
        if row is None:
            continue
        delta_median = float(row["CompoundDeltaMedianSeconds"]) - medium_delta
        delta_p10 = float(row["CompoundDeltaP10Seconds"]) - medium_delta
        delta_p90 = float(row["CompoundDeltaP90Seconds"]) - medium_delta
        deg_p10 = float(row["DegradationP10SecondsPerLap"])
        deg_p90 = float(row["DegradationP90SecondsPerLap"])
        compounds[compound] = {
            "performance_delta_to_medium": delta_median,
            "degradation_seconds_per_lap": float(row["DegradationMedianSecondsPerLap"]),
            "performance_uncertainty": (delta_p90 - delta_p10) / 2.0,
            "degradation_uncertainty": (deg_p90 - deg_p10) / 2.0,
            "identifiable": int(row["DeltaSampleCount"]) > 0 and int(row["DegradationSampleCount"]) > 0,
            "diagnostics": {
                "performance_p10": delta_p10,
                "performance_p90": delta_p90,
                "degradation_p10": deg_p10,
                "degradation_p90": deg_p90,
                "delta_sample_count": int(row["DeltaSampleCount"]),
                "degradation_sample_count": int(row["DegradationSampleCount"]),
            },
        }
    quality_rows = pd.read_csv(paths["quality"]).to_dict("records")
    teams = pd.read_csv(paths["teams"]).to_dict("records")
    pit_rows = pd.read_csv(paths["pit"]).to_dict("records")
    empirical = {}
    for row in pit_rows:
        status = str(row["TrackStatusType"])
        if status not in {"normal", "sc_vsc"}:
            continue
        empirical[status] = {
            "sample_count": int(row["SampleCount"]),
            "pit_in_s3": float(row["PitInS3LossMedianSeconds"]),
            "pit_out_s1": float(row["PitOutS1LossMedianSeconds"]),
            "total": float(row["PitTotalLossMedianSeconds"]),
            "statistic": "median",
        }
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    provenance = {
        "owner": "woStrategy",
        "producer": "run_race_performance_review",
        "artifact_directory": str(paths["tyre"].parent),
        "metadata": metadata,
        "source_files": {name: str(path) for name, path in paths.items()},
        "execution": dict(execution or {
            "producer_called": False,
            "monte_carlo_executed": False,
            "executed_rounds": [],
            "use_cached_monte_carlo": True,
        }),
    }
    return {
        "race_performance": {
            "team_performance": teams,
            "quality": quality_rows,
            "provenance": provenance,
        },
        "retro_tyre_estimate": {
            "event": f"{season}-{round_number:02d}",
            "season": season,
            "round_number": round_number,
            "reference_compound": "MEDIUM",
            "compounds": compounds,
            "quality": quality_rows,
            "retro_is_estimate_not_ground_truth": True,
            "provenance": provenance,
        },
        "empirical_pit_loss": empirical,
    }


__all__ = [
    "MissingRacePerformanceReview",
    "get_empirical_median_pit_loss",
    "get_post_race_review",
    "get_race_performance_review",
    "get_retro_tyre_estimate",
    "run_race_performance_review_adapter",
]
