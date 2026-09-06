"""Report-only FP degradation evidence; never mutates Strategy Prediction inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from wodata import get_data_root
from wodata.artifacts import weekend_model_root
from wodata.contracts import canonical_hash

from wostrategy.analysis.fp_race_diagnostic import calibration_snapshot_root
from wostrategy.analysis.retro_tyre_results import tyre_prediction_year_root


REPORT_VERSION = "fp-pre-race-report-v1"
REPORT_JSON = "fp_tyre_evidence_report.json"
REPORT_MARKDOWN = "fp_tyre_evidence_report.md"
EXPECTED_FP_SESSIONS = ("FP1", "FP2", "FP3")
DEGRADATION_FAMILY_LABELS = {
    "historical_baseline": "Historical degradation",
    "pirelli_informed": "Pirelli-informed degradation",
}
LIMITATIONS = (
    "FP degradation is inferred from limited clean long-run evidence.",
    "FP conditions and programmes may differ materially from Race conditions.",
    "Available history shows a tendency for FP degradation estimates to be lower than final Race Retro.",
    "Calibration currently contains only a small number of independent events.",
    "The apparent best update weight is not stable under leave-one-event-out testing.",
    "FP1, FP2 and FP3 have different historical behaviour and remain independently recalibrated.",
    "Displayed candidate updates are diagnostic and never replace Strategy Prediction inputs automatically.",
)


def load_causal_calibration_snapshot(
    *,
    season: int,
    target_round: int,
    data_root: str | Path | None = None,
) -> tuple[dict[str, object] | None, Path | None]:
    """Load the newest frozen calibration strictly earlier than the target race."""
    versions = calibration_snapshot_root(season, data_root) / "versions"
    eligible: list[tuple[int, Path]] = []
    for path in versions.glob("through_round=*.json"):
        try:
            cutoff = int(path.stem.split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        if cutoff < int(target_round):
            eligible.append((cutoff, path))
    if not eligible:
        return None, None
    _, path = max(eligible, key=lambda item: item[0])
    payload = json.loads(path.read_text(encoding="utf-8"))
    cutoff = int(payload["completed_retro_through_round"])
    if cutoff >= int(target_round):
        raise ValueError(
            f"Calibration through round {cutoff} is not causal for round {target_round}"
        )
    return payload, path


def build_fp_tyre_evidence_report(
    *,
    season: int,
    round_number: int,
    data_root: str | Path | None = None,
    supported_compounds_by_session: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    root = get_data_root(data_root)
    prediction_path = (
        tyre_prediction_year_root(season, root) / "predictions"
        / f"round={int(round_number)}" / "prediction.json"
    )
    prediction = (
        json.loads(prediction_path.read_text(encoding="utf-8"))
        if prediction_path.is_file() else {}
    )
    priors = _priors_by_role(prediction)
    calibration, calibration_path = load_causal_calibration_snapshot(
        season=season,
        target_round=round_number,
        data_root=root,
    )
    tracks = {
        (str(row["session"]), str(row["prior_family"]), str(row["update_type"])): row
        for row in (calibration or {}).get("tracks", ())
    }
    normalized_support = {
        str(session).upper(): {str(compound).upper() for compound in compounds}
        for session, compounds in (supported_compounds_by_session or {}).items()
    }
    session_reports = []
    weekend_root = weekend_model_root(season, round_number, root)
    for session in EXPECTED_FP_SESSIONS:
        directory = weekend_root / "sessions" / session
        parameters_path = directory / "latest_parameters.csv"
        manifest_path = directory / "manifest.json"
        if not parameters_path.is_file() or not manifest_path.is_file():
            session_reports.append({
                "session": session,
                "status": "unavailable",
                "reason": (
                    "No usable persisted session evidence; the session may be absent "
                    "on a sprint-format weekend. The pre-race prior remains unchanged."
                ),
                "compounds": [],
            })
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        included = [str(value).upper() for value in manifest.get("included_sessions", ())]
        if any(value != session for value in included):
            raise ValueError(f"Non-causal {session} report input: {included}")
        parameters = pd.read_csv(parameters_path)
        degradation = parameters.loc[parameters["parameter"].eq("degradation")].copy()
        if session in normalized_support:
            degradation = degradation.loc[
                degradation["compound"].astype(str).str.upper().isin(
                    normalized_support[session]
                )
            ]
        compounds = []
        for row in degradation.sort_values("compound").itertuples(index=False):
            role = str(row.compound).upper()
            fp_value = float(row.median)
            families = []
            for family in ("historical_baseline", "pirelli_informed"):
                prior = priors.get(family, {}).get(role)
                track = tracks.get((session, family, "direct"))
                weight = (
                    float(track["diagnostic_grid_preferred_weight"])
                    if track is not None else None
                )
                candidate = (
                    (1.0 - weight) * float(prior) + weight * fp_value
                    if weight is not None and prior is not None else None
                )
                families.append({
                    "prior_family": family,
                    "pre_race_degradation_prior": prior,
                    "fp_derived_degradation": fp_value,
                    "fp_minus_prior": (
                        fp_value - float(prior) if prior is not None else None
                    ),
                    "diagnostic_candidate_weight": weight,
                    "diagnostic_partial_update": candidate,
                    "calibration_status": (
                        str(track["calibration_status"])
                        if track is not None else "unavailable"
                    ),
                    "trained_through_round": (
                        int(track["trained_through_round"])
                        if track is not None else None
                    ),
                    "historical_event_count": (
                        int(track["event_count"]) if track is not None else 0
                    ),
                    "loeo_stability": (
                        str(track["loeo_stability"])
                        if track is not None else "unavailable"
                    ),
                })
            compounds.append({
                "compound_role": role,
                "absolute_compound": dict(prediction.get("compound_allocation", {})).get(role),
                "support_status": str(row.support_status),
                "usable_lap_count": int(row.usable_lap_count),
                "usable_run_count": int(row.usable_run_count),
                "prior_comparisons": families,
            })
        session_reports.append({
            "session": session,
            "status": "available" if compounds else "no_supported_compounds",
            "reason": (
                None if compounds else
                "No supported compound degradation evidence; the pre-race prior remains unchanged."
            ),
            "analysis_id": manifest.get("analysis_id"),
            "information_cutoff": manifest.get("information_cutoff"),
            "compounds": compounds,
        })
    payload: dict[str, object] = {
        "schema_version": "schema_v1",
        "report_version": REPORT_VERSION,
        "season": int(season),
        "round": int(round_number),
        "event": prediction.get("event"),
        "report_status": "diagnostic_only",
        "production_strategy_inputs_modified": False,
        "prediction_artifact": str(prediction_path) if prediction_path.is_file() else None,
        "calibration_artifact": str(calibration_path) if calibration_path else None,
        "calibration_version": (calibration or {}).get("calibration_version"),
        "calibration_trained_through_round": (
            (calibration or {}).get("completed_retro_through_round")
        ),
        "sessions": session_reports,
        "partial_k_calibration": [
            row for row in (calibration or {}).get("tracks", ())
            if row.get("update_type") == "partial_k"
        ],
        "session_interpretation": (
            "Historical calibration currently finds the strongest degradation-update "
            "evidence in FP2. FP1 and FP3 currently have insufficient or inconsistent "
            "evidence for substantial movement. All three sessions remain independent "
            "calibration tracks and are revisited after each completed Race Retro."
        ),
        "performance_note": (
            "FP long-run pace does not update compound performance because unknown "
            "starting fuel, engine mode and programme offsets cannot yet be robustly "
            "removed from public data. Existing P0/P1 pre-race performance is retained."
        ),
        "limitations": list(LIMITATIONS),
    }
    payload["report_fingerprint"] = canonical_hash(payload)
    return payload


def render_fp_tyre_evidence_report(report: Mapping[str, object]) -> str:
    lines = [
        "# Pre-Race Tyre Report",
        "",
        "## FP Tyre Evidence",
        "",
        "Status: **diagnostic only**. Displayed candidates do not modify Strategy Prediction.",
        "",
    ]
    version = report.get("calibration_version")
    cutoff = report.get("calibration_trained_through_round")
    if version is None:
        lines.extend([
            "No causal calibration history is available for this event; no FP adjustment is shown.",
            "",
        ])
    else:
        lines.extend([
            f"Calibration provenance: `{version}`, trained through round {cutoff}.",
            "",
        ])
    lines.extend([str(report["session_interpretation"]), ""])
    for session in report.get("sessions", ()):
        lines.extend([f"### {session['session']}", ""])
        if session["status"] != "available":
            lines.extend([str(session["reason"]), ""])
            continue
        lines.extend([
            "| Compound | Prior family | Prior | FP estimate | FP − prior | Diagnostic w | Experimental update | Events | LOEO |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        for compound in session["compounds"]:
            label = str(compound["compound_role"])
            absolute = compound.get("absolute_compound")
            if absolute:
                label += f" ({absolute})"
            for family in compound["prior_comparisons"]:
                lines.append(
                    "| " + " | ".join([
                        label,
                        _degradation_family_label(family["prior_family"]),
                        _number(family["pre_race_degradation_prior"]),
                        _number(family["fp_derived_degradation"]),
                        _number(family["fp_minus_prior"]),
                        _number(family["diagnostic_candidate_weight"]),
                        _number(family["diagnostic_partial_update"]),
                        str(family["historical_event_count"]),
                        str(family["loeo_stability"]),
                    ]) + " |"
                )
        lines.append("")
    lines.extend([
        "### Experimental partial-K calibration",
        "",
        "Transferred event-level degradation severity remains an experimental diagnostic, not a measured physical constant or production input.",
        "",
        "### Performance",
        "",
        str(report["performance_note"]),
        "",
        "### Limitations",
        "",
    ])
    lines.extend(f"- {value}" for value in report.get("limitations", ()))
    return "\n".join(lines) + "\n"


def write_fp_tyre_evidence_report(
    *,
    season: int,
    round_number: int,
    data_root: str | Path | None = None,
    supported_compounds_by_session: Mapping[str, Sequence[str]] | None = None,
) -> tuple[Path, Path]:
    report = build_fp_tyre_evidence_report(
        season=season,
        round_number=round_number,
        data_root=data_root,
        supported_compounds_by_session=supported_compounds_by_session,
    )
    root = weekend_model_root(season, round_number, data_root)
    json_path = root / REPORT_JSON
    markdown_path = root / REPORT_MARKDOWN
    _atomic_text(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_text(markdown_path, render_fp_tyre_evidence_report(report))
    return json_path, markdown_path


def _priors_by_role(prediction: Mapping[str, object]) -> dict[str, dict[str, float]]:
    output = {"historical_baseline": {}, "pirelli_informed": {}}
    for values in dict(prediction.get("compounds", {})).values():
        role = str(values.get("compound_role", "")).upper()
        if not role:
            continue
        for family in output:
            degradation = dict(values.get(family, {})).get("degradation")
            if degradation is not None:
                output[family][role] = float(degradation)
    return output


def _number(value: object) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _degradation_family_label(value: object) -> str:
    internal_id = str(value)
    return DEGRADATION_FAMILY_LABELS.get(internal_id, internal_id)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "build_fp_tyre_evidence_report",
    "load_causal_calibration_snapshot",
    "render_fp_tyre_evidence_report",
    "write_fp_tyre_evidence_report",
]
