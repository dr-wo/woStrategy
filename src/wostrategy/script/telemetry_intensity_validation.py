"""Offline cross-event validation of public telemetry running-state evidence.

This is an isolated R&D report generator.  It reads existing woData caches only;
it does not refresh caches, alter production classifiers, or calculate a fuel or
normalized intensity quantity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wodata import (
    get_data_root,
    get_fastf1_session_laps_path,
    get_fastf1_telemetry_cache_dir,
)
from wostrategy.analysis.fp_tyre_evidence import (
    FPProgrammeClassifierConfig,
    _normalise_sessions,
)


YEAR = 2026
SESSION = "FP1"
ROUNDS = (1, 3, 11)
MANDATORY_RUNS = {1: (("SAI", 2), ("ANT", 2))}
ADDITIONAL_RUNS_PER_EVENT = 3

# Existing FP timing classifier threshold. This is not fitted here.
PUSH_THRESHOLD = FPProgrammeClassifierConfig().push_lap_threshold

# Fixed descriptive anchor declared before inspecting additional telemetry.
# It separates very slow non-push examples for comparison; it is not a new
# production label or a claim that every such lap is physically a cool lap.
SLOW_ANCHOR_THRESHOLD = 1.20

# Telemetry definitions recovered from the preceding diagnosis.
FULL_THROTTLE_PCT = 95.0
OFF_THROTTLE_PCT = 5.0
HIGH_SPEED_KPH = 150.0
MAX_DISTANCE_STEP_M = 50.0
PROFILE_POINTS = 101
DRY_COMPOUNDS = ("HARD", "MEDIUM", "SOFT")
CORE_TELEMETRY_CHANNELS = (
    "Distance",
    "RelativeDistance",
    "SessionTime",
    "Source",
    "Throttle",
    "Speed",
    "RPM",
    "nGear",
    "Brake",
    "DRS",
)


@dataclass(frozen=True)
class SessionCache:
    round_number: int
    event_name: str
    laps_path: Path
    telemetry_path: Path
    laps: pd.DataFrame
    telemetry: pd.DataFrame


def timing_band(relative_pace: float) -> str:
    """Return a descriptive band while preserving the binary production label."""
    if relative_pace <= PUSH_THRESHOLD:
        return "push_anchor"
    if relative_pace < SLOW_ANCHOR_THRESHOLD:
        return "ambiguous_non_push"
    return "slow_non_push_anchor"


def prepare_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Apply the existing FP representative-lap and relative-pace semantics."""
    prepared = _normalise_sessions({SESSION: laps})[SESSION].copy()
    prepared["ExistingTimingLabel"] = np.where(
        prepared["Representative"]
        & prepared["RelativePace"].le(PUSH_THRESHOLD),
        "push_like",
        "non_push",
    )
    prepared["DescriptiveTimingBand"] = pd.NA
    representative = prepared["Representative"] & prepared["RelativePace"].notna()
    prepared.loc[representative, "DescriptiveTimingBand"] = prepared.loc[
        representative, "RelativePace"
    ].map(timing_band)
    return prepared


def _transition_count(labels: Iterable[str]) -> int:
    values = list(labels)
    return sum(left != right for left, right in zip(values, values[1:]))


def candidate_runs(prepared: pd.DataFrame, round_number: int) -> pd.DataFrame:
    """Describe timing-diverse runs without reading any telemetry values."""
    rows: list[dict[str, object]] = []
    representative = prepared.loc[prepared["Representative"]].copy()
    for (driver, stint), group in representative.groupby(
        ["Driver", "Stint"], dropna=False, sort=True
    ):
        group = group.sort_values("LapNumber")
        bands = group["DescriptiveTimingBand"].astype(str)
        row = {
            "Year": YEAR,
            "Round": round_number,
            "SessionName": SESSION,
            "EventName": str(group["EventName"].iloc[0]),
            "Driver": str(driver),
            "Stint": int(stint),
            "RepresentativeLaps": len(group),
            "PushAnchorLaps": int(bands.eq("push_anchor").sum()),
            "AmbiguousNonPushLaps": int(bands.eq("ambiguous_non_push").sum()),
            "SlowNonPushAnchorLaps": int(bands.eq("slow_non_push_anchor").sum()),
            "TimingBandCount": int(bands.nunique()),
            "TimingBandTransitions": _transition_count(bands),
            "FirstRepresentativeLap": int(group["LapNumber"].min()),
            "LastRepresentativeLap": int(group["LapNumber"].max()),
        }
        rows.append(row)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    candidates["EligibleTimingDiverseRun"] = (
        candidates["RepresentativeLaps"].ge(4)
        & candidates["PushAnchorLaps"].ge(1)
        & candidates["SlowNonPushAnchorLaps"].ge(1)
    )
    return candidates


def select_runs(prepared_by_round: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Apply the predeclared timing-only sample-selection rule."""
    selected: list[pd.DataFrame] = []
    for round_number in ROUNDS:
        candidates = candidate_runs(prepared_by_round[round_number], round_number)
        if round_number in MANDATORY_RUNS:
            keys = set(MANDATORY_RUNS[round_number])
            chosen = candidates.loc[
                [
                    (str(row.Driver), int(row.Stint)) in keys
                    for row in candidates.itertuples()
                ]
            ].copy()
            missing = keys.difference(
                (str(row.Driver), int(row.Stint)) for row in chosen.itertuples()
            )
            if missing:
                raise ValueError(f"Mandatory reproduction runs are unavailable: {sorted(missing)}")
            chosen["SelectionSource"] = "mandatory_reproduction"
            chosen["SelectionRank"] = pd.NA
        else:
            eligible = candidates.loc[candidates["EligibleTimingDiverseRun"]].copy()
            eligible = eligible.sort_values(
                [
                    "TimingBandCount",
                    "TimingBandTransitions",
                    "RepresentativeLaps",
                    "Driver",
                    "Stint",
                ],
                ascending=[False, False, False, True, True],
            )
            chosen = eligible.head(ADDITIONAL_RUNS_PER_EVENT).copy()
            if len(chosen) < ADDITIONAL_RUNS_PER_EVENT:
                raise ValueError(
                    f"Round {round_number} has only {len(chosen)} timing-diverse runs"
                )
            chosen["SelectionSource"] = "timing_only_rank"
            chosen["SelectionRank"] = range(1, len(chosen) + 1)
        selected.append(chosen)
    output = pd.concat(selected, ignore_index=True)
    output["RunId"] = [
        f"{YEAR}-R{int(row.Round):02d}-{SESSION}-{row.Driver}-S{int(row.Stint)}"
        for row in output.itertuples()
    ]
    return output.sort_values(["Round", "Driver", "Stint"]).reset_index(drop=True)


def _exclusion_reason(lap: pd.Series) -> str:
    reasons: list[str] = []
    if pd.isna(lap.get("LapTimeSeconds")):
        reasons.append("missing_lap_time")
    if str(lap.get("Compound", "")).upper() not in DRY_COMPOUNDS:
        reasons.append("non_dry_compound")
    if pd.notna(lap.get("PitOutTime")):
        reasons.append("pit_out_lap")
    if pd.notna(lap.get("PitInTime")):
        reasons.append("pit_in_lap")
    if str(lap.get("TrackStatus", "1")) != "1":
        reasons.append(f"track_status_{lap.get('TrackStatus')}")
    return ";".join(reasons) if reasons else "included"


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def summarize_lap_telemetry(telemetry: pd.DataFrame) -> dict[str, object]:
    """Calculate duration-safe spatial summaries for one lap.

    Distance increments are assigned to the current merged telemetry row, matching
    the completed diagnosis. Negative, missing, or implausibly large increments
    are excluded. Frame rows are not treated as independent observations.
    """
    if telemetry.empty:
        return {"TelemetryUsable": False, "TelemetryIssue": "no_telemetry_rows"}
    missing = set(CORE_TELEMETRY_CHANNELS).difference(telemetry.columns)
    if missing:
        return {
            "TelemetryUsable": False,
            "TelemetryIssue": "missing_channels:" + ",".join(sorted(missing)),
        }

    frame = telemetry.sort_values("SessionTime").copy()
    distance = _numeric(frame["Distance"])
    distance_step = distance.diff()
    valid_step = distance_step.ge(0) & distance_step.le(MAX_DISTANCE_STEP_M)
    weights = distance_step.where(valid_step, 0.0).fillna(0.0)
    valid_distance = float(weights.sum())
    distance_span = float(distance.max() - distance.min())
    if not np.isfinite(valid_distance) or valid_distance <= 0:
        return {"TelemetryUsable": False, "TelemetryIssue": "no_valid_distance"}

    throttle = _numeric(frame["Throttle"])
    speed = _numeric(frame["Speed"])
    rpm = _numeric(frame["RPM"])
    gear = _numeric(frame["nGear"])
    brake = frame["Brake"].fillna(False).astype(bool)

    def weighted_mean(values: pd.Series) -> float:
        valid = values.notna() & weights.gt(0)
        if not valid.any():
            return float("nan")
        return float(np.average(values.loc[valid], weights=weights.loc[valid]))

    def distance_pct(mask: pd.Series) -> float:
        valid = mask.fillna(False) & weights.gt(0)
        return float(100.0 * weights.loc[valid].sum() / valid_distance)

    merged_dt = frame["SessionTime"].diff().dt.total_seconds()
    car = frame.loc[frame["Source"].eq("car")].copy()
    car_dt = car["SessionTime"].diff().dt.total_seconds()
    merged_median_dt = float(merged_dt[merged_dt.gt(0)].median())
    native_median_dt = float(car_dt[car_dt.gt(0)].median())
    channel_coverage = min(
        float(frame[column].notna().mean()) for column in CORE_TELEMETRY_CHANNELS
    )
    drs_values = sorted(_numeric(frame["DRS"]).dropna().astype(int).unique().tolist())

    return {
        "TelemetryUsable": True,
        "TelemetryIssue": "",
        "TelemetryRows": len(frame),
        "NativeCarRows": len(car),
        "MergedFrameMedianDtSeconds": merged_median_dt,
        "NativeCarMedianDtSeconds": native_median_dt,
        "MergedFrameHz": 1.0 / merged_median_dt,
        "NativeCarHz": 1.0 / native_median_dt,
        "MergedChannelRowCoverageFraction": channel_coverage,
        "ValidDistanceM": valid_distance,
        "DistanceSpanM": distance_span,
        "DistanceCoverageFraction": (
            valid_distance / distance_span if distance_span > 0 else float("nan")
        ),
        "DistanceWeightedThrottlePct": weighted_mean(throttle),
        "FullThrottleDistancePct": distance_pct(throttle.ge(FULL_THROTTLE_PCT)),
        "HighSpeedOffThrottleNoBrakeDistancePct": distance_pct(
            throttle.le(OFF_THROTTLE_PCT) & ~brake & speed.ge(HIGH_SPEED_KPH)
        ),
        "BrakeDistancePct": distance_pct(brake),
        "DistanceWeightedSpeedKph": weighted_mean(speed),
        "DistanceWeightedRPM": weighted_mean(rpm),
        "Gear7PlusDistancePct": distance_pct(gear.ge(7)),
        "DRSValues": "|".join(str(value) for value in drs_values),
    }


def _lap_metadata(lap: pd.Series, run_id: str) -> dict[str, object]:
    output = {
        "RunId": run_id,
        "Year": int(lap["Year"]),
        "Round": int(lap["Round"]),
        "EventName": str(lap["EventName"]),
        "SessionName": str(lap["SessionName"]),
        "Driver": str(lap["Driver"]),
        "Team": str(lap["Team"]),
        "Stint": int(lap["Stint"]),
        "LapNumber": int(lap["LapNumber"]),
        "Compound": str(lap["Compound"]),
        "TyreLife": lap.get("TyreLife"),
        "LapTimeSeconds": lap.get("LapTimeSeconds"),
        "DriverSessionFastestSeconds": lap.get("DriverSessionFastest"),
        "RelativePace": lap.get("RelativePace"),
        "ExistingTimingLabel": lap.get("ExistingTimingLabel"),
        "DescriptiveTimingBand": lap.get("DescriptiveTimingBand"),
        "TrackStatus": lap.get("TrackStatus"),
        "MeanTimeDeltaToDriverAhead": lap.get("MeanTimeDeltaToDriverAhead"),
        "MeanTimeDeltaToDriverBehind": lap.get("MeanTimeDeltaToDriverBehind"),
        "MinTimeDeltaToDriverAhead": lap.get("MinTimeDeltaToDriverAhead"),
        "MinTimeDeltaToDriverBehind": lap.get("MinTimeDeltaToDriverBehind"),
        "TrafficCoverageFraction": lap.get("CoverageFraction"),
        "TrafficStatus": lap.get("TrafficStatus"),
    }
    for sector in ("Sector1Time", "Sector2Time", "Sector3Time"):
        value = lap.get(sector)
        output[f"{sector}Seconds"] = (
            value.total_seconds() if pd.notna(value) and hasattr(value, "total_seconds") else np.nan
        )
    return output


def analyze_selected_runs(
    caches: dict[int, SessionCache],
    prepared_by_round: dict[int, pd.DataFrame],
    selected_runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for selected in selected_runs.itertuples():
        laps = prepared_by_round[int(selected.Round)]
        run_laps = laps.loc[
            laps["Driver"].eq(selected.Driver) & laps["Stint"].eq(selected.Stint)
        ].sort_values("LapNumber")
        session_telemetry = caches[int(selected.Round)].telemetry
        for _, lap in run_laps.iterrows():
            reason = _exclusion_reason(lap)
            coverage_row = _lap_metadata(lap, selected.RunId)
            coverage_row["IncludedByTimingRule"] = reason == "included"
            coverage_row["ExclusionReason"] = reason
            if reason != "included":
                coverage.append(coverage_row)
                continue
            telemetry = session_telemetry.loc[
                session_telemetry["Driver"].eq(selected.Driver)
                & session_telemetry["LapNumber"].eq(float(lap["LapNumber"]))
            ]
            metrics = summarize_lap_telemetry(telemetry)
            coverage_row.update(metrics)
            if not metrics["TelemetryUsable"]:
                coverage_row["ExclusionReason"] = metrics["TelemetryIssue"]
            coverage.append(coverage_row)
            result = _lap_metadata(lap, selected.RunId)
            result.update(metrics)
            results.append(result)
    return pd.DataFrame(results), pd.DataFrame(coverage)


def add_run_references(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    usable = output.loc[output["TelemetryUsable"]].copy()
    references = (
        usable.sort_values(["RunId", "RelativePace", "LapNumber"])
        .groupby("RunId", as_index=False)
        .first()[["RunId", "LapNumber", "RelativePace"]]
        .rename(
            columns={
                "LapNumber": "RunReferenceLapNumber",
                "RelativePace": "RunReferenceRelativePace",
            }
        )
    )
    return output.merge(references, on="RunId", how="left")


def build_run_band_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "DistanceWeightedThrottlePct",
        "FullThrottleDistancePct",
        "HighSpeedOffThrottleNoBrakeDistancePct",
        "DistanceWeightedSpeedKph",
        "DistanceWeightedRPM",
        "Gear7PlusDistancePct",
    ]
    usable = results.loc[results["TelemetryUsable"]].copy()
    summary = (
        usable.groupby(
            [
                "RunId",
                "Year",
                "Round",
                "EventName",
                "SessionName",
                "Driver",
                "Team",
                "Stint",
                "DescriptiveTimingBand",
            ],
            dropna=False,
        )[metrics]
        .agg(["count", "median", "min", "max"])
    )
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    return summary.reset_index()


def build_run_contrasts(run_band_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id, group in run_band_summary.groupby("RunId"):
        indexed = group.set_index("DescriptiveTimingBand")
        if {"push_anchor", "slow_non_push_anchor"}.difference(indexed.index):
            continue
        push = indexed.loc["push_anchor"]
        slow = indexed.loc["slow_non_push_anchor"]
        rows.append(
            {
                "RunId": run_id,
                "Round": int(push["Round"]),
                "EventName": push["EventName"],
                "Driver": push["Driver"],
                "Stint": int(push["Stint"]),
                "PushMinusSlowDistanceWeightedThrottlePct": (
                    push["DistanceWeightedThrottlePct_median"]
                    - slow["DistanceWeightedThrottlePct_median"]
                ),
                "PushMinusSlowFullThrottleDistancePct": (
                    push["FullThrottleDistancePct_median"]
                    - slow["FullThrottleDistancePct_median"]
                ),
                "SlowMinusPushHighSpeedOffThrottleDistancePct": (
                    slow["HighSpeedOffThrottleNoBrakeDistancePct_median"]
                    - push["HighSpeedOffThrottleNoBrakeDistancePct_median"]
                ),
                "PushMinusSlowDistanceWeightedSpeedKph": (
                    push["DistanceWeightedSpeedKph_median"]
                    - slow["DistanceWeightedSpeedKph_median"]
                ),
                "PushMinusSlowDistanceWeightedRPM": (
                    push["DistanceWeightedRPM_median"]
                    - slow["DistanceWeightedRPM_median"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["Round", "Driver", "Stint"])


def build_ambiguous_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    """Place each ambiguous lap separately between its run's anchor medians.

    The output retains one comparison per channel and deliberately does not
    combine them into a score or force a state label.
    """
    metrics = (
        "DistanceWeightedThrottlePct",
        "FullThrottleDistancePct",
        "HighSpeedOffThrottleNoBrakeDistancePct",
        "DistanceWeightedSpeedKph",
        "DistanceWeightedRPM",
    )
    rows: list[dict[str, object]] = []
    usable = results.loc[results["TelemetryUsable"]]
    for run_id, run in usable.groupby("RunId"):
        push = run.loc[run["DescriptiveTimingBand"].eq("push_anchor")]
        slow = run.loc[run["DescriptiveTimingBand"].eq("slow_non_push_anchor")]
        ambiguous = run.loc[run["DescriptiveTimingBand"].eq("ambiguous_non_push")]
        if push.empty or slow.empty or ambiguous.empty:
            continue
        push_median = push[list(metrics)].median()
        slow_median = slow[list(metrics)].median()
        for lap in ambiguous.itertuples():
            row = {
                "RunId": run_id,
                "Round": int(lap.Round),
                "EventName": lap.EventName,
                "Driver": lap.Driver,
                "Stint": int(lap.Stint),
                "LapNumber": int(lap.LapNumber),
                "RelativePace": lap.RelativePace,
                "MeanTimeDeltaToDriverAhead": lap.MeanTimeDeltaToDriverAhead,
                "MeanTimeDeltaToDriverBehind": lap.MeanTimeDeltaToDriverBehind,
            }
            for metric in metrics:
                value = float(getattr(lap, metric))
                lower = float(min(push_median[metric], slow_median[metric]))
                upper = float(max(push_median[metric], slow_median[metric]))
                if value > upper:
                    position = "above_both_anchor_medians"
                elif value < lower:
                    position = "below_both_anchor_medians"
                else:
                    position = "between_anchor_medians"
                row[f"{metric}Position"] = position
                row[f"{metric}MinusPushMedian"] = value - float(push_median[metric])
            primary_positions = [
                row[f"{metric}Position"]
                for metric in (
                    "DistanceWeightedThrottlePct",
                    "FullThrottleDistancePct",
                    "DistanceWeightedSpeedKph",
                )
            ]
            row["AllPrimaryMetricsBetweenAnchorMedians"] = all(
                position == "between_anchor_medians" for position in primary_positions
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["Round", "Driver", "Stint", "LapNumber"])


def _profile_laps_for_run(run_results: pd.DataFrame) -> list[int]:
    driver = str(run_results["Driver"].iloc[0])
    round_number = int(run_results["Round"].iloc[0])
    stint = int(run_results["Stint"].iloc[0])
    reproduction = {
        (1, "SAI", 2): [11, 12, 13],
        (1, "ANT", 2): [3, 4, 5],
    }
    if (round_number, driver, stint) in reproduction:
        return reproduction[(round_number, driver, stint)]

    usable = run_results.loc[run_results["TelemetryUsable"]].copy()
    push = usable.loc[usable["DescriptiveTimingBand"].eq("push_anchor")]
    slow = usable.loc[usable["DescriptiveTimingBand"].eq("slow_non_push_anchor")]
    ambiguous = usable.loc[usable["DescriptiveTimingBand"].eq("ambiguous_non_push")]
    fastest = push if not push.empty else usable
    chosen = [int(fastest.sort_values("RelativePace").iloc[0]["LapNumber"])]
    if not ambiguous.empty:
        median = ambiguous["RelativePace"].median()
        middle = ambiguous.assign(
            _DistanceFromMedian=(ambiguous["RelativePace"] - median).abs()
        ).sort_values(["_DistanceFromMedian", "LapNumber"])
        chosen.append(int(middle.iloc[0]["LapNumber"]))
    slowest = slow if not slow.empty else usable
    chosen.append(
        int(slowest.sort_values("RelativePace", ascending=False).iloc[0]["LapNumber"])
    )
    return list(dict.fromkeys(chosen))


def build_distance_profiles(
    caches: dict[int, SessionCache], results: pd.DataFrame
) -> pd.DataFrame:
    grid = np.linspace(0.0, 1.0, PROFILE_POINTS)
    rows: list[pd.DataFrame] = []
    for run_id, run_results in results.groupby("RunId"):
        profile_laps = _profile_laps_for_run(run_results)
        for lap_number in profile_laps:
            result = run_results.loc[run_results["LapNumber"].eq(lap_number)].iloc[0]
            telemetry = caches[int(result["Round"])].telemetry
            lap = telemetry.loc[
                telemetry["Driver"].eq(result["Driver"])
                & telemetry["LapNumber"].eq(float(lap_number))
                & telemetry["Source"].eq("car")
            ].copy()
            lap["RelativeDistance"] = _numeric(lap["RelativeDistance"])
            lap = lap.dropna(subset=["RelativeDistance"]).sort_values("RelativeDistance")
            lap = lap.drop_duplicates("RelativeDistance", keep="last")
            if len(lap) < 2:
                continue
            x = lap["RelativeDistance"].to_numpy(dtype=float)
            profile = pd.DataFrame({"RelativeDistance": grid})
            for column in ("Throttle", "Speed", "RPM", "nGear"):
                values = _numeric(lap[column]).to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(values)
                interpolated = np.full_like(grid, np.nan, dtype=float)
                if valid.sum() >= 2:
                    inside = (grid >= x[valid].min()) & (grid <= x[valid].max())
                    interpolated[inside] = np.interp(grid[inside], x[valid], values[valid])
                profile[column] = interpolated
            profile["RunId"] = run_id
            profile["Round"] = int(result["Round"])
            profile["EventName"] = result["EventName"]
            profile["Driver"] = result["Driver"]
            profile["Stint"] = int(result["Stint"])
            profile["LapNumber"] = lap_number
            profile["RelativePace"] = result["RelativePace"]
            profile["ExistingTimingLabel"] = result["ExistingTimingLabel"]
            profile["DescriptiveTimingBand"] = result["DescriptiveTimingBand"]
            rows.append(profile)
    return pd.concat(rows, ignore_index=True)


def plot_metric_recurrence(results: pd.DataFrame, output_path: Path) -> None:
    metrics = (
        ("DistanceWeightedThrottlePct", "Distance-weighted throttle (%)"),
        ("FullThrottleDistancePct", "Full-throttle distance (%)"),
        (
            "HighSpeedOffThrottleNoBrakeDistancePct",
            "High-speed off-throttle/no-brake distance (%)",
        ),
    )
    rounds = sorted(results["Round"].unique())
    fig, axes = plt.subplots(len(rounds), len(metrics), figsize=(15, 11), squeeze=False)
    marker = {
        "push_anchor": "o",
        "ambiguous_non_push": "^",
        "slow_non_push_anchor": "X",
    }
    for row_index, round_number in enumerate(rounds):
        event = results.loc[results["Round"].eq(round_number), "EventName"].iloc[0]
        event_data = results.loc[
            results["Round"].eq(round_number) & results["TelemetryUsable"]
        ]
        run_ids = sorted(event_data["RunId"].unique())
        colors = plt.cm.tab10(np.linspace(0, 0.7, len(run_ids)))
        for column_index, (metric, label) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for color, run_id in zip(colors, run_ids):
                run = event_data.loc[event_data["RunId"].eq(run_id)].sort_values(
                    "RelativePace"
                )
                short = f"{run['Driver'].iloc[0]} S{int(run['Stint'].iloc[0])}"
                axis.plot(
                    run["RelativePace"],
                    run[metric],
                    color=color,
                    alpha=0.45,
                    linewidth=1.0,
                    label=short,
                )
                for band, points in run.groupby("DescriptiveTimingBand"):
                    axis.scatter(
                        points["RelativePace"],
                        points[metric],
                        color=color,
                        marker=marker[str(band)],
                        s=30,
                        edgecolor="black",
                        linewidth=0.35,
                        zorder=3,
                    )
            axis.axvline(PUSH_THRESHOLD, color="grey", linestyle="--", linewidth=0.8)
            axis.axvline(SLOW_ANCHOR_THRESHOLD, color="grey", linestyle=":", linewidth=0.8)
            axis.set_xlabel("Lap time / driver-session fastest representative lap")
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            if column_index == 0:
                axis.set_title(f"R{round_number} {event}\n{label}")
                axis.legend(fontsize=8, loc="best")
            else:
                axis.set_title(label)
    fig.suptitle(
        "Local timing–telemetry relationships (lines connect laps only within one run)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_event_profiles(profiles: pd.DataFrame, round_number: int, output_path: Path) -> None:
    event_profiles = profiles.loc[profiles["Round"].eq(round_number)]
    run_ids = sorted(event_profiles["RunId"].unique())
    fig, axes = plt.subplots(len(run_ids), 2, figsize=(14, 3.4 * len(run_ids)), squeeze=False)
    palette = {
        "push_anchor": "#228833",
        "ambiguous_non_push": "#CCBB44",
        "slow_non_push_anchor": "#4477AA",
    }
    for row_index, run_id in enumerate(run_ids):
        run = event_profiles.loc[event_profiles["RunId"].eq(run_id)]
        for lap_number, lap in run.groupby("LapNumber"):
            band = str(lap["DescriptiveTimingBand"].iloc[0])
            ratio = float(lap["RelativePace"].iloc[0])
            label = f"L{int(lap_number)} {band} ratio={ratio:.3f}"
            color = palette[band]
            axes[row_index, 0].plot(
                lap["RelativeDistance"], lap["Throttle"], color=color, label=label
            )
            axes[row_index, 1].plot(
                lap["RelativeDistance"], lap["Speed"], color=color, label=label
            )
        driver = run["Driver"].iloc[0]
        stint = int(run["Stint"].iloc[0])
        axes[row_index, 0].set_title(f"{driver} stint {stint}: throttle")
        axes[row_index, 1].set_title(f"{driver} stint {stint}: speed")
        axes[row_index, 0].set_ylabel("Throttle (%)")
        axes[row_index, 1].set_ylabel("Speed (km/h)")
        for axis in axes[row_index]:
            axis.set_xlabel("Relative lap distance")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7, loc="best")
    event_name = event_profiles["EventName"].iloc[0]
    fig.suptitle(
        f"2026 R{round_number} {event_name} FP1 — native-car-sample distance profiles",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _load_caches(data_root: Path) -> tuple[dict[int, SessionCache], dict[int, pd.DataFrame]]:
    caches: dict[int, SessionCache] = {}
    prepared: dict[int, pd.DataFrame] = {}
    telemetry_root = get_fastf1_telemetry_cache_dir(data_root)
    for round_number in ROUNDS:
        laps_path = get_fastf1_session_laps_path(
            year=YEAR,
            round_number=round_number,
            session=SESSION,
            data_root=data_root,
        )
        telemetry_path = telemetry_root / f"{YEAR}_{round_number}_{SESSION}"
        if not laps_path.is_file() or not telemetry_path.is_file():
            raise FileNotFoundError(
                f"Offline validation requires existing caches: {laps_path}, {telemetry_path}"
            )
        laps = pd.read_pickle(laps_path)
        telemetry = pd.read_pickle(telemetry_path)
        event_name = str(laps["EventName"].dropna().iloc[0])
        caches[round_number] = SessionCache(
            round_number,
            event_name,
            laps_path,
            telemetry_path,
            laps,
            telemetry,
        )
        prepared[round_number] = prepare_laps(laps)
    return caches, prepared


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_source_manifest(caches: dict[int, SessionCache]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for round_number, cache in sorted(caches.items()):
        for kind, path, frame in (
            ("enriched_session_laps", cache.laps_path, cache.laps),
            ("derived_per_sample_telemetry", cache.telemetry_path, cache.telemetry),
        ):
            rows.append(
                {
                    "Year": YEAR,
                    "Round": round_number,
                    "EventName": cache.event_name,
                    "SessionName": SESSION,
                    "SourceKind": kind,
                    "Path": str(path.resolve()),
                    "SizeBytes": path.stat().st_size,
                    "SHA256": _sha256_file(path),
                    "Rows": len(frame),
                    "Columns": len(frame.columns),
                    "TrafficEvaluatorVersion": frame.attrs.get(
                        "traffic_evaluator_version"
                    ),
                    "TrafficCombinationMode": frame.attrs.get(
                        "traffic_combination_mode"
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_validation(*, data_root: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    caches, prepared = _load_caches(data_root)
    selected = select_runs(prepared)
    results, coverage = analyze_selected_runs(caches, prepared, selected)
    results = add_run_references(results)
    band_summary = build_run_band_summary(results)
    contrasts = build_run_contrasts(band_summary)
    ambiguous_comparisons = build_ambiguous_comparisons(results)
    profiles = build_distance_profiles(caches, results)
    source_manifest = build_source_manifest(caches)

    outputs = {
        "selected_runs": output_dir / "selected_runs.csv",
        "per_lap_results": output_dir / "per_lap_results.csv",
        "coverage": output_dir / "lap_coverage_and_exclusions.csv",
        "run_band_summary": output_dir / "run_band_summary.csv",
        "run_contrasts": output_dir / "run_contrasts.csv",
        "ambiguous_comparisons": output_dir / "ambiguous_lap_comparisons.csv",
        "profile_data": output_dir / "distance_aligned_profile_data.csv",
        "source_manifest": output_dir / "source_manifest.csv",
        "metric_plot": output_dir / "metric_recurrence.png",
    }
    selected.to_csv(outputs["selected_runs"], index=False)
    results.to_csv(outputs["per_lap_results"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    band_summary.to_csv(outputs["run_band_summary"], index=False)
    contrasts.to_csv(outputs["run_contrasts"], index=False)
    ambiguous_comparisons.to_csv(outputs["ambiguous_comparisons"], index=False)
    profiles.to_csv(outputs["profile_data"], index=False)
    source_manifest.to_csv(outputs["source_manifest"], index=False)
    plot_metric_recurrence(results, outputs["metric_plot"])
    for round_number in ROUNDS:
        path = output_dir / f"distance_profiles_2026_r{round_number:02d}_fp1.png"
        plot_event_profiles(profiles, round_number, path)
        outputs[f"profiles_r{round_number}"] = path

    print("Selected runs")
    print(
        selected[
            [
                "Round",
                "EventName",
                "Driver",
                "Stint",
                "RepresentativeLaps",
                "SelectionSource",
                "SelectionRank",
            ]
        ].to_string(index=False)
    )
    print(f"\nUsable telemetry laps: {int(results['TelemetryUsable'].sum())}/{len(results)}")
    print("\nRun-level push minus slow-anchor contrasts")
    print(contrasts.round(3).to_string(index=False))
    print(f"\nWrote validation artifacts to {output_dir}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("doc/telemetry_intensity_validation"),
    )
    args = parser.parse_args()
    run_validation(
        data_root=get_data_root(args.data_root),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
