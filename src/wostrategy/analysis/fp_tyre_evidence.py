from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Sequence
import json

import numpy as np
import pandas as pd

from wodata.artifacts import weekend_model_root
from wodata.contracts import canonical_hash
from wodata import get_data_root

from wostrategy.analysis.long_run_performance import (
    _prepare_laps,
)
from wostrategy.core.pre_race_session_data import load_cached_weekend_sessions


FP_TYRE_EVIDENCE_SCHEMA_VERSION = 1
FP_TYRE_EVIDENCE_ALGORITHM_VERSION = "strict-longrun-contrast-v1"
FP_TYRE_EVIDENCE_FILENAME = "fp_tyre_evidence_v1.json"
DRY_COMPOUNDS = ("HARD", "MEDIUM", "SOFT")
DEFAULT_SESSIONS = ("FP1", "FP2", "FP3")
NUMERICAL_SEEDS = (11, 2026, 424242, 8675309)


@dataclass(frozen=True)
class FPProgrammeClassifierConfig:
    push_lap_threshold: float = 1.03
    setting_substantial_min_timed_laps: int = 4
    setting_max_session_progress: float = 0.35
    setting_min_push_laps: int = 2
    setting_min_fast_slow_fast_cycles: int = 1
    quali_sim_min_push_laps: int = 1
    long_run_min_consecutive_representative_laps: int = 6
    long_run_relative_pace_limit: float = 1.15
    long_run_variability_limit: float = 0.025
    long_run_min_representative_fraction: float = 0.60
    long_run_max_internal_gap_laps: int = 1


@dataclass(frozen=True)
class FPTyreEvidenceCalculationConfig:
    sessions: tuple[str, ...] = DEFAULT_SESSIONS
    min_clean_air_laps: int = 4
    clean_mean_time_delta_seconds: float = 3.0
    clean_mean_time_delta_behind_seconds: float = 1.0
    quick_lap_threshold: float = 1.10
    classifier: FPProgrammeClassifierConfig = FPProgrammeClassifierConfig()


def fp_tyre_evidence_path(
    year: int, round_number: int, data_root: str | Path | None = None
) -> Path:
    return weekend_model_root(year, round_number, data_root) / FP_TYRE_EVIDENCE_FILENAME


def load_fp_tyre_evidence(
    year: int, round_number: int, data_root: str | Path | None = None
) -> dict[str, object] | None:
    """Read a cached production evidence artifact; never calculate or download."""
    path = fp_tyre_evidence_path(year, round_number, data_root)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema_version", 0)) != FP_TYRE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported FP tyre evidence schema in {path}.")
    return payload


def write_fp_tyre_evidence(
    evidence: Mapping[str, object],
    *,
    year: int,
    round_number: int,
    data_root: str | Path | None = None,
) -> Path:
    """Atomically replace the cache only after a complete successful analysis."""
    path = fp_tyre_evidence_path(year, round_number, data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(dict(evidence)), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
    return path


def calculate_fp_tyre_evidence(
    *,
    year: int,
    round_number: int,
    data_root: str | Path | None = None,
    force_refresh: bool = False,
    config: FPTyreEvidenceCalculationConfig | None = None,
    session_loader: Callable[..., pd.DataFrame] | None = None,
) -> dict[str, object]:
    """Explicitly load available FP data, analyse it, and atomically cache evidence.

    This function is intentionally the only public path that may ask the existing
    session loader to retrieve missing FP data.  Cache reads use
    :func:`load_fp_tyre_evidence` and never call this function implicitly.
    """
    resolved = config or FPTyreEvidenceCalculationConfig()
    loaded = load_cached_weekend_sessions(
        year=year,
        round_number=round_number,
        sessions=resolved.sessions,
        data_root=get_data_root(data_root),
        force_refresh=force_refresh,
        session_loader=session_loader,
    )
    sessions = {
        name: frame for name, frame in loaded.sessions.items()
        if frame is not None and not frame.empty
    }
    if not sessions:
        detail = "; ".join(f"{name}: {reason}" for name, reason in loaded.sources.items())
        raise ValueError(f"No FP session data are available. {detail}".strip())
    fingerprints = {
        session: _path_fingerprint(loaded.cache_paths[session])
        for session in sessions
        if loaded.cache_paths[session].is_file()
    }
    evidence = analyse_fp_tyre_evidence_sessions(
        sessions,
        year=year,
        round_number=round_number,
        config=resolved,
        source_fingerprints=fingerprints,
        session_sources=loaded.sources,
    )
    write_fp_tyre_evidence(
        evidence,
        year=year,
        round_number=round_number,
        data_root=data_root,
    )
    return evidence


def analyse_fp_tyre_evidence_sessions(
    sessions: Mapping[str, pd.DataFrame],
    *,
    year: int,
    round_number: int,
    config: FPTyreEvidenceCalculationConfig | None = None,
    source_fingerprints: Mapping[str, Mapping[str, object]] | None = None,
    session_sources: Mapping[str, str] | None = None,
    calculated_at: datetime | None = None,
) -> dict[str, object]:
    """Pure in-memory FP evidence calculation for applications and tests."""
    resolved = config or FPTyreEvidenceCalculationConfig()
    normalized = _normalise_sessions(sessions)
    if not normalized:
        raise ValueError("At least one non-empty FP session is required.")
    laps = pd.concat(normalized.values(), ignore_index=True)
    meta, groups = _prepare_runs(laps)
    ledger = _classify_all(meta, groups, resolved.classifier)
    clean = _clean_longrun_laps(normalized, resolved)
    evidence = _evidence_laps(ledger, groups, clean)
    strict = evidence.loc[evidence["used_for_quantitative_inference"]].copy()
    if strict.empty:
        raise ValueError("No strict LongRun clean laps are available for quantitative evidence.")

    variants = _variant_results(evidence)
    strict_fit = _transformed_fit(_variant_frame(evidence, "strict"))
    identifiability = _identifiability(strict_fit)
    convergence = _convergence(strict_fit)
    contrasts = _contrast_records(variants, convergence)
    performance = _direct_performance(strict)
    support = {
        "strict_longrun_runs": int(strict.run_id.nunique()),
        "strict_clean_laps": int(len(strict)),
        "strict_teams": int(strict.team.nunique()),
        "strict_drivers": int(strict.driver.nunique()),
    }
    sessions_used = list(normalized)
    event_name = next(
        (
            str(frame["EventName"].dropna().iloc[0])
            for frame in normalized.values()
            if "EventName" in frame and not frame["EventName"].dropna().empty
        ),
        f"{year} R{round_number}",
    )
    inventory = ledger[
        [
            "run_id", "session", "team", "driver", "stint", "compound",
            "revised_label", "classification_reason", "classification_evidence_strength",
            "uncertain_breakdown", "total_physical_laps", "total_timed_laps",
            "representative_laps", "representative_fraction",
            "longest_consecutive_representative", "representative_pace_cv",
        ]
    ].to_dict("records")
    details = evidence.to_dict("records")
    fingerprints = dict(source_fingerprints or {})
    input_fingerprint = canonical_hash(
        {
            "sessions": sessions_used,
            "sources": fingerprints,
            "config": asdict(resolved),
        }
    )
    now = calculated_at or datetime.now(timezone.utc)
    payload = {
        "schema_version": FP_TYRE_EVIDENCE_SCHEMA_VERSION,
        "algorithm_version": FP_TYRE_EVIDENCE_ALGORITHM_VERSION,
        "analysis_id": f"fp-tyre-{input_fingerprint[:20]}",
        "year": int(year),
        "round_number": int(round_number),
        "event": event_name,
        "calculated_at": now.astimezone(timezone.utc).isoformat(),
        "sessions_requested": list(resolved.sessions),
        "sessions_used": sessions_used,
        "session_sources": dict(session_sources or {}),
        "source_fingerprints": fingerprints,
        "input_fingerprint": input_fingerprint,
        "calculation_config": asdict(resolved),
        "programme_inventory": inventory,
        "longrun_support": support,
        "identifiability": identifiability,
        "degradation_contrasts": contrasts,
        "performance_support": performance,
        "longrun_laps": details,
        "limitations": [
            "Only strict LongRun clean laps contribute quantitatively.",
            "Setting, QualiSim and Uncertain are inventory-only.",
            "FP track evolution is excluded.",
            "Absolute Fuel and the common degradation level are gauge-unidentified.",
            "Degradation contrasts are advisory and never absolute Strategy inputs.",
            "Direct performance is absent without a defensible strict LongRun connector.",
        ],
    }
    return _jsonable(payload)


def local_fp_tyre_evidence_status(
    evidence: Mapping[str, object],
    *,
    year: int,
    round_number: int,
    data_root: str | Path | None = None,
    sessions: Sequence[str] = DEFAULT_SESSIONS,
) -> dict[str, object]:
    """Compare cached fingerprints with local files only; never contact a network."""
    from wodata import get_fastf1_session_laps_path

    stored = dict(evidence.get("source_fingerprints") or {})
    changed = []
    newly_available = []
    current = {}
    used = {str(value).upper() for value in evidence.get("sessions_used") or ()}
    for raw_session in sessions:
        session = str(raw_session).upper()
        path = get_fastf1_session_laps_path(
            year=year, round_number=round_number, session=session, data_root=data_root
        )
        if not path.is_file():
            continue
        fingerprint = _path_fingerprint(path)
        current[session] = fingerprint
        if session not in used:
            newly_available.append(session)
        elif session in stored and stored[session].get("sha256") != fingerprint["sha256"]:
            changed.append(session)
    stale = bool(changed or newly_available)
    reasons = []
    if changed:
        reasons.append("local source changed: " + ", ".join(changed))
    if newly_available:
        reasons.append("new local session available: " + ", ".join(newly_available))
    return {
        "cached": True,
        "stale": stale,
        "reason": "; ".join(reasons),
        "changed_sessions": changed,
        "newly_available_sessions": newly_available,
        "local_fingerprints": current,
    }


def _normalise_sessions(sessions: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    output = {}
    for raw_name, raw_frame in sessions.items():
        if raw_frame is None or raw_frame.empty:
            continue
        session = str(raw_name).upper()
        frame = raw_frame.copy()
        frame["Session"] = session
        if "LapTimeSeconds" not in frame:
            if "LapTime" not in frame:
                raise ValueError(f"{session} laps have no LapTime column.")
            if pd.api.types.is_timedelta64_dtype(frame["LapTime"]):
                frame["LapTimeSeconds"] = frame["LapTime"].dt.total_seconds()
            else:
                frame["LapTimeSeconds"] = pd.to_numeric(frame["LapTime"], errors="coerce")
        if "SessionSeconds" not in frame:
            source = "Time" if "Time" in frame else "SessionTime"
            frame["SessionSeconds"] = pd.to_timedelta(
                frame[source], errors="coerce"
            ).dt.total_seconds()
        frame["Compound"] = frame["Compound"].astype("string").str.upper()
        frame["DryTimed"] = frame["Compound"].isin(DRY_COMPOUNDS) & frame["LapTimeSeconds"].notna()
        pit_out = frame["PitOutTime"].isna() if "PitOutTime" in frame else True
        pit_in = frame["PitInTime"].isna() if "PitInTime" in frame else True
        track_green = frame["TrackStatus"].astype(str).eq("1") if "TrackStatus" in frame else True
        frame["Representative"] = frame["DryTimed"] & pit_out & pit_in & track_green
        fast = (
            frame.loc[frame["Representative"]]
            .groupby(["Session", "Driver"])["LapTimeSeconds"]
            .min()
        )
        frame["DriverSessionFastest"] = [
            fast.get((session_name, driver), np.nan)
            for session_name, driver in zip(frame["Session"], frame["Driver"])
        ]
        frame["RelativePace"] = frame["LapTimeSeconds"] / frame["DriverSessionFastest"]
        output[session] = frame
    return output


def _prepare_runs(laps: pd.DataFrame):
    groups = {}
    rows = []
    keys = ["Session", "Team", "Driver", "Stint"]
    for key, group in laps.groupby(keys, dropna=False, sort=True):
        group = group.sort_values(["SessionSeconds", "LapNumber"]).copy()
        run_id = ":".join(map(str, key))
        groups[run_id] = group
        dry = group.loc[group["DryTimed"]]
        rows.append(
            {
                "run_id": run_id, "session": key[0], "team": key[1],
                "driver": key[2], "stint": key[3],
                "start_s": group["SessionSeconds"].min(),
                "end_s": group["SessionSeconds"].max(),
                "total_physical_laps": len(group), "total_timed_laps": len(dry),
            }
        )
    meta = pd.DataFrame(rows).sort_values(["session", "driver", "start_s", "stint"])
    meta["run_order"] = meta.groupby(["session", "driver"]).cumcount() + 1
    meta["prior_max_timed_laps"] = (
        meta.groupby(["session", "driver"])["total_timed_laps"]
        .transform(lambda values: values.shift().cummax()).fillna(0)
    )
    meta["session_progress_start"] = (
        meta["start_s"] / meta.groupby("session")["end_s"].transform("max")
    )
    return meta, groups


def _fast_slow_fast_cycles(push: list[bool]) -> int:
    compressed = []
    for value in push:
        symbol = "F" if value else "S"
        if not compressed or compressed[-1] != symbol:
            compressed.append(symbol)
    return sum(
        compressed[index:index + 3] == ["F", "S", "F"]
        for index in range(len(compressed) - 2)
    )


def _max_consecutive(numbers: list[int], max_gap: int) -> int:
    if not numbers:
        return 0
    best = current = 1
    for left, right in zip(numbers, numbers[1:]):
        current = current + 1 if right - left <= max_gap else 1
        best = max(best, current)
    return best


def _classify(group: pd.DataFrame, row, config: FPProgrammeClassifierConfig):
    ordered = group.sort_values("LapNumber").copy()
    timed = ordered.loc[ordered["DryTimed"]]
    ordered["push_cfg"] = ordered["Representative"] & ordered["RelativePace"].le(
        config.push_lap_threshold
    )
    representative = ordered.loc[ordered["Representative"]]
    pushes = int(ordered["push_cfg"].sum())
    cycles = _fast_slow_fast_cycles(
        ordered.loc[ordered["Representative"], "push_cfg"].astype(bool).tolist()
    )
    same_compound = timed["Compound"].nunique() == 1
    fraction = len(representative) / max(len(timed), 1)
    pace_cv = (
        representative["LapTimeSeconds"].std(ddof=1)
        / representative["LapTimeSeconds"].mean()
        if len(representative) >= 2 else np.nan
    )
    maximum_relative = representative["RelativePace"].max() if len(representative) else np.nan
    consecutive = _max_consecutive(
        representative["LapNumber"].astype(int).tolist(),
        config.long_run_max_internal_gap_laps,
    )
    setting_context = (
        row.session == "FP1"
        and row.total_timed_laps >= config.setting_substantial_min_timed_laps
        and row.prior_max_timed_laps < config.setting_substantial_min_timed_laps
        and row.session_progress_start <= config.setting_max_session_progress
    )
    if (
        setting_context
        and pushes >= config.setting_min_push_laps
        and cycles >= config.setting_min_fast_slow_fast_cycles
    ):
        label, strength = "Setting", "strong"
        reason = f"early FP1 substantial stint with {pushes} pushes and {cycles} fast-slow-fast cycles"
    elif (
        same_compound
        and consecutive >= config.long_run_min_consecutive_representative_laps
        and maximum_relative <= config.long_run_relative_pace_limit
        and (np.isnan(pace_cv) or pace_cv <= config.long_run_variability_limit)
        and fraction >= config.long_run_min_representative_fraction
        and cycles == 0
    ):
        label, strength = "LongRun", "strong"
        reason = f"one compound, {consecutive} sustained representative laps, pace CV={pace_cv:.3f}"
    elif pushes >= config.quali_sim_min_push_laps:
        label = "QualiSim"
        strength = "strong" if pushes >= 2 else "moderate"
        reason = f"{pushes} push-like laps without sustained LongRun evidence"
    else:
        label, strength = "Uncertain", "insufficient"
        reason = "no qualifying push and insufficient sustained race-like evidence"
    if label != "Uncertain":
        uncertain = "NotUncertain"
    elif len(ordered) <= 2 or len(representative) <= 1:
        uncertain = "TooShortOrInsufficient"
    elif len(representative) and representative["RelativePace"].median() > config.long_run_relative_pace_limit:
        uncertain = "SlowOrSpecialTest"
    elif same_compound and len(representative) >= 4:
        uncertain = "NearLongRunFailedContinuityOrVariability"
    else:
        uncertain = "GenuinelyAmbiguous"
    compound = timed["Compound"].mode().iloc[0] if len(timed) else "UNKNOWN"
    return {
        "revised_label": label,
        "classification_reason": reason,
        "classification_evidence_strength": strength,
        "uncertain_breakdown": uncertain,
        "push_laps": pushes,
        "fast_slow_fast_cycles": cycles,
        "same_compound": bool(same_compound),
        "representative_laps": len(representative),
        "representative_fraction": fraction,
        "longest_consecutive_representative": consecutive,
        "representative_pace_cv": pace_cv,
        "maximum_representative_relative_pace": maximum_relative,
        "compound": compound,
    }


def _classify_all(meta, groups, config):
    return pd.DataFrame(
        [
            {**row._asdict(), **_classify(groups[row.run_id], row, config)}
            for row in meta.itertuples()
        ]
    )


def _clean_longrun_laps(sessions, config):
    selected = []
    for session, raw in sessions.items():
        prepared = _prepare_laps(
            raw,
            clean_mean_time_delta_seconds=config.clean_mean_time_delta_seconds,
            clean_mean_time_delta_behind_seconds=config.clean_mean_time_delta_behind_seconds,
            quick_lap_threshold=config.quick_lap_threshold,
            dry_compounds=DRY_COMPOUNDS,
        )
        # The completed LongRun audit selected individually clean laps inside a
        # physical stint.  Requiring each clean fragment to form a new block
        # would incorrectly discard otherwise valid laps after an isolated
        # disturbance (and changes the R11 reference from 8/58 to 6/49).
        clean = prepared.loc[prepared["IsCleanAirLongRunLap"].fillna(False)].copy()
        clean["Session"] = session
        selected.append(clean)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def _longest_one_gap(numbers):
    values = sorted(int(value) for value in numbers)
    best = 0
    for start in range(len(values)):
        used = False
        count = 1
        for left, right in zip(values[start:], values[start + 1:]):
            gap = right - left
            if gap <= 1:
                count += 1
            elif gap == 2 and not used:
                used = True
                count += 1
            else:
                break
        best = max(best, count)
    return best


def _robust_cv(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.median(values) == 0:
        return np.nan
    mad = np.median(np.abs(values - np.median(values)))
    return float(1.4826 * mad / np.median(values))


def _evidence_laps(ledger, groups, clean):
    selected_keys = set(
        zip(clean["Session"], clean["Driver"], clean["Stint"], clean["LapNumber"])
    ) if not clean.empty else set()
    rows = []
    for run in ledger.itertuples():
        group = groups[run.run_id].sort_values(["SessionSeconds", "LapNumber"])
        representative = group.loc[group["Representative"] & group["LapTimeSeconds"].notna()]
        one_gap = _longest_one_gap(representative["LapNumber"])
        robust_cv = _robust_cv(representative["LapTimeSeconds"])
        common = (
            bool(run.same_compound)
            and len(representative) >= 6
            and float(run.maximum_representative_relative_pace) <= 1.15
            and float(run.representative_fraction) >= 0.60
            and int(run.fast_slow_fast_cycles) == 0
        )
        one_gap_variant = bool(
            common and one_gap >= 6
            and (pd.isna(run.representative_pace_cv) or run.representative_pace_cv <= 0.025)
        )
        robust_variant = bool(common and one_gap >= 6 and pd.notna(robust_cv) and robust_cv <= 0.025)
        if run.revised_label != "LongRun" and not one_gap_variant and not robust_variant:
            continue
        for index, lap in enumerate(group.itertuples(), 1):
            key = (run.session, run.driver, run.stint, lap.LapNumber)
            strict_clean = run.revised_label == "LongRun" and key in selected_keys
            representative_candidate = bool(lap.Representative and pd.notna(lap.LapTimeSeconds))
            selected = strict_clean if run.revised_label == "LongRun" else representative_candidate
            if strict_clean:
                reason = ""
            elif run.revised_label != "LongRun":
                reason = f"not_strict_longrun:{run.revised_label}"
            elif pd.notna(getattr(lap, "PitOutTime", None)):
                reason = "pit_out"
            elif pd.notna(getattr(lap, "PitInTime", None)):
                reason = "pit_in"
            elif str(getattr(lap, "TrackStatus", "1")) != "1":
                reason = "not_green"
            elif not representative_candidate:
                reason = "not_representative"
            else:
                reason = "audited_clean_or_consecutive_policy_exclusion"
            rows.append(
                {
                    "session": run.session, "team": run.team, "driver": run.driver,
                    "run_id": run.run_id, "stint": run.stint, "compound": run.compound,
                    "lap_number": lap.LapNumber, "session_time_s": lap.SessionSeconds,
                    "lap_time_s": lap.LapTimeSeconds, "tyre_age": lap.TyreLife,
                    "physical_run_index": index,
                    "relative_pace": lap.RelativePace,
                    "pit_out": pd.notna(getattr(lap, "PitOutTime", None)),
                    "pit_in": pd.notna(getattr(lap, "PitInTime", None)),
                    "strict_classifier": run.revised_label == "LongRun",
                    "programme": run.revised_label,
                    "representative_candidate": representative_candidate,
                    "strict_clean_lap": strict_clean,
                    "used_for_quantitative_inference": strict_clean,
                    "one_gap_variant": one_gap_variant,
                    "robust_variability_variant": robust_variant,
                    "selected_for_variant_fit": selected,
                    "exclusion_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def _variant_frame(evidence, variant):
    selected = evidence.loc[
        evidence["selected_for_variant_fit"] & evidence["lap_time_s"].notna()
    ].copy()
    if variant == "strict":
        mask = selected["strict_classifier"] & selected["strict_clean_lap"]
    elif variant == "one_gap_tolerant":
        mask = selected["strict_classifier"] | selected["one_gap_variant"]
    elif variant == "robust_variability":
        mask = selected["strict_classifier"] | selected["robust_variability_variant"]
    else:
        raise ValueError(variant)
    return selected.loc[mask].sort_values(["run_id", "physical_run_index"]).reset_index(drop=True)


def _demean_runs(frame, matrix, values=None):
    result = np.asarray(matrix, float).copy()
    y = None if values is None else np.asarray(values, float).copy()
    for _, indices in frame.groupby("run_id").groups.items():
        index = np.asarray(list(indices), int)
        result[index] -= result[index].mean(axis=0)
        if y is not None:
            y[index] -= y[index].mean()
    return (result, y) if y is not None else result


def _transformed_fit(frame):
    frame = frame.reset_index(drop=True)
    compound = frame["compound"].astype(str)
    age = frame["tyre_age"].to_numpy(float)
    x = np.column_stack(
        (age, compound.eq("HARD") * age, compound.eq("SOFT") * age)
    )
    xp, yp = _demean_runs(frame, x, frame["lap_time_s"].to_numpy(float))
    rank = int(np.linalg.matrix_rank(xp))
    beta = np.linalg.pinv(xp) @ yp
    residual = yp - xp @ beta
    sigma = float(np.sqrt(np.sum(residual**2) / max(len(yp) - rank, 1)))
    covariance = np.linalg.pinv(xp.T @ xp) * sigma**2
    projector = np.linalg.pinv(xp) @ xp
    return {
        "frame": frame, "xp": xp, "yp": yp, "rank": rank, "beta": beta,
        "residual": residual, "sigma": sigma, "covariance": covariance,
        "parameter_projector": projector,
    }


def _contrast_vector(name):
    return {
        "Delta_HM": np.array([0.0, 1.0, 0.0]),
        "Delta_SM": np.array([0.0, 0.0, 1.0]),
        "Delta_SH": np.array([0.0, -1.0, 1.0]),
    }[name]


def _estimable(fit, vector):
    residual = vector - vector @ fit["parameter_projector"]
    return bool(np.linalg.norm(residual) < 1e-8)


def _summary(fit, name):
    vector = _contrast_vector(name)
    if not _estimable(fit, vector):
        return None
    central = float(vector @ fit["beta"])
    standard_error = float(np.sqrt(max(vector @ fit["covariance"] @ vector, 0.0)))
    width = 1.2815515655 * standard_error
    return central, standard_error, central - width, central + width


def _variant_results(evidence):
    results = []
    sensitivities = []
    profiles = []
    fits = {}
    for variant in ("strict", "one_gap_tolerant", "robust_variability"):
        frame = _variant_frame(evidence, variant)
        fit = _transformed_fit(frame)
        fits[variant] = fit
        for name in ("Delta_HM", "Delta_SM", "Delta_SH"):
            summary = _summary(fit, name)
            if summary is None:
                continue
            central, standard_error, p10, p90 = summary
            involved = {
                "Delta_HM": ("HARD", "MEDIUM"),
                "Delta_SM": ("SOFT", "MEDIUM"),
                "Delta_SH": ("SOFT", "HARD"),
            }[name]
            support = frame.loc[frame["compound"].isin(involved)]
            results.append(
                {
                    "variant": variant, "contrast": name, "central": central,
                    "standard_error": standard_error, "p10": p10, "p50": central,
                    "p90": p90, "supporting_runs": int(support.run_id.nunique()),
                    "supporting_teams": int(support.team.nunique()),
                    "supporting_clean_laps": int(len(support)),
                    "tyre_age_span": float(support.tyre_age.max() - support.tyre_age.min()),
                    "fit_rank": fit["rank"], "residual_sigma_s": fit["sigma"],
                }
            )
            if variant == "strict":
                for dimension in ("run_id", "team"):
                    for omitted in support[dimension].dropna().unique():
                        subset = frame.loc[frame[dimension].ne(omitted)]
                        candidate = _transformed_fit(subset)
                        value = _summary(candidate, name)
                        if value is not None:
                            sensitivities.append(
                                {
                                    "contrast": name, "dimension": dimension,
                                    "omitted": str(omitted), "central": value[0],
                                }
                            )
        for target, index in (("Delta_HM", 1), ("Delta_SM", 2)):
            if _summary(fit, target) is None:
                continue
            centre = fit["beta"][index]
            for fixed in np.linspace(centre - 0.35, centre + 0.35, 141):
                free = [position for position in range(3) if position != index]
                target_y = fit["yp"] - fit["xp"][:, index] * fixed
                other = np.linalg.pinv(fit["xp"][:, free]) @ target_y
                beta = np.zeros(3)
                beta[index] = fixed
                beta[free] = other
                sse = float(np.sum((fit["yp"] - fit["xp"] @ beta) ** 2))
                profiles.append(
                    {"variant": variant, "contrast": target, "value": fixed, "sse": sse}
                )
    return {
        "results": pd.DataFrame(results),
        "sensitivities": pd.DataFrame(sensitivities),
        "profiles": pd.DataFrame(profiles),
        "fits": fits,
    }


def _convergence(fit):
    covariance = fit["covariance"]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    active = eigenvalues > max(eigenvalues.max(), 1.0) * 1e-14
    scales = np.sqrt(eigenvalues[active])
    basis = eigenvectors[:, active]
    rows = []
    for count in (20_000, 50_000):
        for seed in NUMERICAL_SEEDS:
            rng = np.random.default_rng(seed)
            z = rng.uniform(-4.0, 4.0, size=(count, len(scales)))
            candidates = fit["beta"] + (z * scales) @ basis.T
            residual = fit["yp"][None, :] - candidates @ fit["xp"].T
            log_weight = -0.5 * np.sum(residual**2, axis=1) / max(fit["sigma"]**2, 1e-12)
            log_weight -= log_weight.max()
            weights = np.exp(log_weight)
            weights /= weights.sum()
            rows.append(
                {
                    "sample_count": count, "seed": seed,
                    "ess": float(1.0 / np.sum(weights**2)),
                    "ess_fraction": float(1.0 / np.sum(weights**2) / count),
                    "max_weight": float(weights.max()),
                    "top5_weight": float(np.sort(weights)[-5:].sum()),
                }
            )
    return rows


def _contrast_records(variants, convergence):
    results = variants["results"]
    sensitivity = variants["sensitivities"]
    profiles = variants["profiles"]
    records = []
    for name in ("Delta_HM", "Delta_SM", "Delta_SH"):
        rows = results.loc[results["contrast"].eq(name)].set_index("variant")
        if "strict" not in rows.index:
            continue
        strict = rows.loc["strict"]
        leave_run = sensitivity.loc[
            sensitivity["contrast"].eq(name) & sensitivity["dimension"].eq("run_id")
        ]
        leave_team = sensitivity.loc[
            sensitivity["contrast"].eq(name) & sensitivity["dimension"].eq("team")
        ]
        profile = profiles.loc[
            profiles["contrast"].eq(name) & profiles["variant"].eq("strict")
        ].copy()
        profile_information = None
        if not profile.empty:
            profile["delta_objective"] = (
                profile["sse"] - profile["sse"].min()
            ) / max(float(strict.residual_sigma_s) ** 2, 1e-12)
            optimum = profile.loc[profile["sse"].idxmin()]
            profile_information = {
                "point_count": len(profile),
                "optimum": optimum["value"],
                "grid_low": profile["value"].min(),
                "grid_high": profile["value"].max(),
                "edge_delta_objective_min": min(
                    profile.iloc[0]["delta_objective"],
                    profile.iloc[-1]["delta_objective"],
                ),
            }
        records.append(
            {
                "parameter": name,
                "unit": "s/lap",
                "provenance_type": "fp_longrun_contrast",
                "automatic_strategy_update": False,
                "strict_central": strict.central,
                "strict_p10": strict.p10,
                "strict_p50": strict.p50,
                "strict_p90": strict.p90,
                "one_gap_central": rows.loc["one_gap_tolerant"].central if "one_gap_tolerant" in rows.index else None,
                "robust_variability_central": rows.loc["robust_variability"].central if "robust_variability" in rows.index else None,
                "supporting_runs": strict.supporting_runs,
                "supporting_teams": strict.supporting_teams,
                "supporting_clean_laps": strict.supporting_clean_laps,
                "tyre_age_span": strict.tyre_age_span,
                "leave_one_run_min": leave_run.central.min() if len(leave_run) else None,
                "leave_one_run_max": leave_run.central.max() if len(leave_run) else None,
                "leave_one_team_min": leave_team.central.min() if len(leave_team) else None,
                "leave_one_team_max": leave_team.central.max() if len(leave_team) else None,
                "profile": profile_information,
                "rank": strict.fit_rank,
                "residual_sigma_s": strict.residual_sigma_s,
                "numerical_convergence": convergence,
            }
        )
    return records


def _identifiability(fit):
    frame = fit["frame"]
    compound = frame["compound"].astype(str)
    age = frame["tyre_age"].to_numpy(float)
    progress = frame["physical_run_index"].to_numpy(float)
    original = np.column_stack(
        (-progress, compound.eq("HARD") * age, compound.eq("MEDIUM") * age, compound.eq("SOFT") * age)
    )
    projected = _demean_runs(frame, original)
    _, singular, vt = np.linalg.svd(projected, full_matrices=False)
    rank = int(np.linalg.matrix_rank(projected))
    null = vt[rank] if rank < len(vt) else np.full(4, np.nan)
    before = projected @ np.array([0.03, 0.06, 0.10, 0.15])
    after = projected @ np.array([0.103, 0.133, 0.173, 0.223])
    return {
        "parameter_order": ["Fuel", "Deg_HARD", "Deg_MEDIUM", "Deg_SOFT"],
        "rank": rank,
        "nullity": 4 - rank,
        "singular_spectrum": singular.tolist(),
        "null_vector": null.tolist(),
        "gauge": "Fuel->Fuel+t and every Deg_c->Deg_c+t",
        "gauge_invariance_max_abs_prediction_change": float(np.max(np.abs(before - after))),
        "track_evolution_active": False,
    }


def _direct_performance(strict):
    records = []
    for name, left in (("Perf_HARD_vs_MEDIUM", "HARD"), ("Perf_SOFT_vs_MEDIUM", "SOFT")):
        pair = {left, "MEDIUM"}
        drivers = []
        teams = []
        for (session, driver), group in strict.groupby(["session", "driver"]):
            if pair.issubset(set(group["compound"].astype(str))):
                drivers.append(f"{session}:{driver}")
        for (session, team), group in strict.groupby(["session", "team"]):
            if pair.issubset(set(group["compound"].astype(str))):
                teams.append(f"{session}:{team}")
        supported = bool(drivers)
        value = p10 = p90 = None
        if supported:
            subset = strict.loc[
                strict.apply(
                    lambda row: f"{row.session}:{row.driver}" in drivers,
                    axis=1,
                )
            ].copy().reset_index(drop=True)
            groups = pd.get_dummies(
                subset["session"].astype(str) + ":" + subset["driver"].astype(str),
                drop_first=False,
                dtype=float,
            ).to_numpy()
            age = subset["tyre_age"].to_numpy(float)
            compound = subset["compound"].astype(str)
            x = np.column_stack(
                [groups, -subset["physical_run_index"].to_numpy(float)]
                + [(compound.eq(item) * age).to_numpy(float) for item in DRY_COMPOUNDS]
                + [(compound.eq("HARD")).to_numpy(float), (compound.eq("SOFT")).to_numpy(float)]
            )
            y = subset["lap_time_s"].to_numpy(float)
            beta = np.linalg.pinv(x) @ y
            residual = y - x @ beta
            rank = np.linalg.matrix_rank(x)
            covariance = np.linalg.pinv(x.T @ x) * np.sum(residual**2) / max(len(y) - rank, 1)
            index = x.shape[1] - (2 if left == "HARD" else 1)
            selector = np.zeros(x.shape[1])
            selector[index] = 1.0
            estimable = np.linalg.norm(selector - selector @ (np.linalg.pinv(x) @ x)) < 1e-8
            if estimable:
                value = float(beta[index])
                standard_error = float(np.sqrt(max(covariance[index, index], 0.0)))
                p10 = value - 1.2815515655 * standard_error
                p90 = value + 1.2815515655 * standard_error
            else:
                supported = False
        records.append(
            {
                "parameter": name,
                "unit": "s",
                "convention": f"{left} pace minus MEDIUM pace; positive means {left} is slower",
                "support_status": "fp_longrun_direct" if supported else "unsupported",
                "fp_direct_value": value if supported else None,
                "p10": p10 if supported else None,
                "p90": p90 if supported else None,
                "same_driver_connectors": drivers,
                "same_team_connectors": teams,
                "limitation": (
                    "Same-driver, same-session strict LongRuns provide a direct comparison."
                    if supported
                    else "No defensible same-driver strict LongRun cross-compound connector."
                ),
            }
        )
    return records


def _path_fingerprint(path: Path) -> dict[str, object]:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path), "sha256": digest.hexdigest(),
        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
