from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
from wodata import get_data_root, get_race_performance_review_root


SCHEMA_VERSION = "schema_v1"
EXTRACTOR_VERSION = "retro-tyre-v1"
OBSERVATION_COLUMNS = (
    "schema_version",
    "season",
    "round",
    "event",
    "compound_role",
    "absolute_compound",
    "event_hard_compound",
    "performance",
    "degradation",
    "performance_p10",
    "performance_p90",
    "degradation_p10",
    "degradation_p90",
    "support_clean_lap_count",
    "support_run_count",
    "support_stint_count",
    "retro_effective_sample_size",
    "retro_weighted_rmse",
    "performance_reference_role",
    "source_result_path",
    "source_result_hash",
    "source_result_timestamp",
    "extractor_version",
    "extracted_at",
)


class RetroTyreExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class RetroTyreObservation:
    schema_version: str
    season: int
    round: int
    event: str
    compound_role: str
    absolute_compound: str
    event_hard_compound: str
    performance: float
    degradation: float
    performance_p10: float | None
    performance_p90: float | None
    degradation_p10: float | None
    degradation_p90: float | None
    support_clean_lap_count: int | None
    support_run_count: int | None
    support_stint_count: int | None
    retro_effective_sample_size: float | None
    retro_weighted_rmse: float | None
    performance_reference_role: str
    source_result_path: str
    source_result_hash: str
    source_result_timestamp: str
    extractor_version: str
    extracted_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RetroExtractionReport:
    events_discovered: int
    rows_extracted: int
    unchanged_events: int
    replaced_events: int
    skipped_events: tuple[str, ...]
    csv_path: Path


def tyre_prediction_year_root(
    season: int, data_root: str | Path | None = None
) -> Path:
    return (
        get_data_root(data_root)
        / "wostrategy"
        / "tyre_prediction"
        / SCHEMA_VERSION
        / f"year={int(season)}"
    )


def observations_csv_path(
    season: int, data_root: str | Path | None = None
) -> Path:
    return tyre_prediction_year_root(season, data_root) / f"retro_tyre_observations_{season}.csv"


class RetroTyreResultExtractor:
    """Read existing Retro result artifacts without importing or invoking Retro MC."""

    def __init__(
        self,
        *,
        data_root: str | Path | None = None,
        legacy_root: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root) if data_root is not None else None
        self.canonical_root = get_race_performance_review_root(data_root)
        self.legacy_root = (
            Path(legacy_root)
            if legacy_root is not None
            else Path(__file__).resolve().parents[4] / "cache" / "race_performance_review"
        )

    def discover(self, season: int) -> dict[int, Path]:
        discovered: dict[int, Path] = {}
        canonical_year = self.canonical_root / f"year={int(season)}"
        for event_dir in sorted(canonical_year.glob("round=*/session=R")):
            try:
                round_number = int(event_dir.parent.name.split("=", 1)[1])
            except (IndexError, ValueError):
                continue
            if _source_files(event_dir, season, round_number):
                discovered[round_number] = event_dir
        for delta_path in sorted(
            self.legacy_root.glob(f"race_performance_{int(season)}_*_R_summary_compound_delta.csv")
        ):
            parts = delta_path.stem.split("_")
            try:
                round_number = int(parts[3])
            except (IndexError, ValueError):
                continue
            if round_number in discovered:
                continue
            if _source_files(self.legacy_root, season, round_number):
                discovered[round_number] = self.legacy_root
        return dict(sorted(discovered.items()))

    def update(
        self,
        *,
        season: int,
        previews: Iterable[Mapping[str, object]],
    ) -> RetroExtractionReport:
        preview_by_round = {int(row["round"]): row for row in previews}
        existing = _read_existing_observations(observations_csv_path(season, self.data_root))
        old_hashes = {
            round_number: str(rows[0]["source_result_hash"])
            for round_number, rows in existing.items() if rows
        }
        discovered = self.discover(season)
        observations: list[RetroTyreObservation] = []
        skipped: list[str] = []
        unchanged = 0
        replaced = 0
        for round_number, source_dir in discovered.items():
            preview = preview_by_round.get(round_number)
            if preview is None:
                skipped.append(f"round {round_number}: no Pirelli preview allocation")
                continue
            try:
                rows = self.extract_event(
                    season=season,
                    round_number=round_number,
                    source_dir=source_dir,
                    preview=preview,
                )
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                skipped.append(f"round {round_number}: {exc}")
                continue
            source_hash = rows[0].source_result_hash
            old_rows = existing.get(round_number, [])
            old_allocation = {
                (row["compound_role"], row["absolute_compound"]) for row in old_rows
            }
            new_allocation = {
                (row.compound_role, row.absolute_compound) for row in rows
            }
            if old_hashes.get(round_number) == source_hash and old_allocation == new_allocation:
                extracted_at_by_role = {
                    row["compound_role"]: row["extracted_at"] for row in old_rows
                }
                rows = [
                    replace(row, extracted_at=extracted_at_by_role[row.compound_role])
                    for row in rows
                ]
                unchanged += 1
            elif round_number in old_hashes:
                replaced += 1
            observations.extend(rows)
        _write_observations(observations, observations_csv_path(season, self.data_root))
        return RetroExtractionReport(
            events_discovered=len(discovered),
            rows_extracted=len(observations),
            unchanged_events=unchanged,
            replaced_events=replaced,
            skipped_events=tuple(skipped),
            csv_path=observations_csv_path(season, self.data_root),
        )

    def extract_event(
        self,
        *,
        season: int,
        round_number: int,
        source_dir: Path,
        preview: Mapping[str, object],
    ) -> list[RetroTyreObservation]:
        files = _source_files(source_dir, season, round_number)
        if not files:
            raise RetroTyreExtractionError("completed Retro tyre summaries are unavailable")
        if files[0].name == "tyre_information.csv":
            values = _read_tyre_information(files[0])
        else:
            values = _read_weighted_summaries(files[0], files[1])
        support_path = _event_artifact_path(
            source_dir, season, round_number, "clean_laps"
        )
        diagnostics_path = _event_artifact_path(
            source_dir, season, round_number, "sample_diagnostics"
        )
        support = _read_compound_support(support_path) if support_path.exists() else {}
        diagnostics = (
            _read_sampler_diagnostics(diagnostics_path)
            if diagnostics_path.exists()
            else {}
        )
        reference_roles = {str(row["reference_role"]).upper() for row in values}
        if reference_roles != {"HARD"}:
            raise RetroTyreExtractionError(
                f"Expected HARD performance reference, found {sorted(reference_roles)}."
            )
        role_map = {
            "HARD": _required_compound(preview, "hard_compound"),
            "MEDIUM": _required_compound(preview, "medium_compound"),
            "SOFT": _required_compound(preview, "soft_compound"),
        }
        event = str(preview.get("event") or _event_name(source_dir, season, round_number))
        if not event or event == "None":
            raise RetroTyreExtractionError("event name is unavailable")
        consumed_files = files + tuple(
            path for path in (support_path, diagnostics_path) if path.exists()
        )
        source_hash = _combined_file_hash(consumed_files)
        timestamp = datetime.fromtimestamp(
            max(path.stat().st_mtime for path in consumed_files), timezone.utc
        ).isoformat()
        extracted_at = _utc_now()
        source_path = ";".join(str(path.resolve()) for path in consumed_files)
        rows: list[RetroTyreObservation] = []
        for row in values:
            role = str(row["role"]).upper()
            if role not in role_map:
                continue
            role_support = support.get(role, {})
            rows.append(RetroTyreObservation(
                schema_version=SCHEMA_VERSION,
                season=int(season),
                round=int(round_number),
                event=event,
                compound_role=role,
                absolute_compound=role_map[role],
                event_hard_compound=role_map["HARD"],
                performance=float(row["performance"]),
                degradation=float(row["degradation"]),
                performance_p10=_optional_float(row.get("performance_p10")),
                performance_p90=_optional_float(row.get("performance_p90")),
                degradation_p10=_optional_float(row.get("degradation_p10")),
                degradation_p90=_optional_float(row.get("degradation_p90")),
                support_clean_lap_count=_optional_int(role_support.get("clean_lap_count")),
                support_run_count=_optional_int(role_support.get("run_count")),
                support_stint_count=_optional_int(role_support.get("stint_count")),
                retro_effective_sample_size=_optional_float(
                    diagnostics.get("EffectiveSampleSize")
                ),
                retro_weighted_rmse=_optional_float(
                    diagnostics.get("WeightedRMSESeconds")
                ),
                performance_reference_role="HARD",
                source_result_path=source_path,
                source_result_hash=source_hash,
                source_result_timestamp=timestamp,
                extractor_version=EXTRACTOR_VERSION,
                extracted_at=extracted_at,
            ))
        if not rows:
            raise RetroTyreExtractionError("no global compound rows were found")
        return rows


def _source_files(directory: Path, season: int, round_number: int) -> tuple[Path, ...]:
    tyre_information = directory / "tyre_information.csv"
    if tyre_information.exists():
        return (tyre_information,)
    prefix = f"race_performance_{int(season)}_{int(round_number)}_R"
    degradation = directory / f"{prefix}_summary_compound_degradation.csv"
    delta = directory / f"{prefix}_summary_compound_delta.csv"
    return (degradation, delta) if degradation.exists() and delta.exists() else ()


def _read_tyre_information(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    required = {
        "Scope", "Compound", "ReferenceCompound", "CompoundDeltaMedianSeconds",
        "DegradationMedianSecondsPerLap",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RetroTyreExtractionError(f"{path.name} is missing columns {sorted(missing)}")
    frame = frame.loc[frame["Scope"].astype(str).str.lower().eq("global")]
    rows: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        if pd.isna(row["CompoundDeltaMedianSeconds"]) or pd.isna(
            row["DegradationMedianSecondsPerLap"]
        ):
            continue
        rows.append({
            "role": row["Compound"],
            "reference_role": row["ReferenceCompound"],
            "performance": row["CompoundDeltaMedianSeconds"],
            "degradation": row["DegradationMedianSecondsPerLap"],
            "performance_p10": row.get("CompoundDeltaP10Seconds"),
            "performance_p90": row.get("CompoundDeltaP90Seconds"),
            "degradation_p10": row.get("DegradationP10SecondsPerLap"),
            "degradation_p90": row.get("DegradationP90SecondsPerLap"),
        })
    return rows


def _read_weighted_summaries(degradation_path: Path, delta_path: Path) -> list[dict]:
    degradation = pd.read_csv(degradation_path).rename(columns={
        "P10": "degradation_p10", "Median": "degradation", "P90": "degradation_p90"
    })
    delta = pd.read_csv(delta_path).rename(columns={
        "P10": "performance_p10", "Median": "performance", "P90": "performance_p90",
        "CompoundDeltaReference": "reference_role",
    })
    merged = delta.merge(
        degradation[["Compound", "degradation", "degradation_p10", "degradation_p90"]],
        on="Compound", how="inner", validate="one_to_one",
    )
    return [
        {
            "role": row["Compound"],
            "reference_role": row["reference_role"],
            "performance": row["performance"],
            "degradation": row["degradation"],
            "performance_p10": row.get("performance_p10"),
            "performance_p90": row.get("performance_p90"),
            "degradation_p10": row.get("degradation_p10"),
            "degradation_p90": row.get("degradation_p90"),
        }
        for row in merged.to_dict("records")
    ]


def _event_name(directory: Path, season: int, round_number: int) -> str | None:
    prefix = f"race_performance_{season}_{round_number}_R"
    path = directory / f"{prefix}_clean_laps.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, nrows=1)
    for column in ("EventName", "RaceName", "EventLocation", "EventCountry"):
        if column in frame and not frame[column].dropna().empty:
            return str(frame[column].dropna().iloc[0])
    return None


def _event_artifact_path(
    directory: Path, season: int, round_number: int, suffix: str
) -> Path:
    return directory / f"race_performance_{season}_{round_number}_R_{suffix}.csv"


def _read_compound_support(path: Path) -> dict[str, dict[str, int]]:
    frame = pd.read_csv(path)
    if "Compound" not in frame.columns:
        return {}
    frame = frame.dropna(subset=["Compound"]).copy()
    frame["Compound"] = frame["Compound"].astype(str).str.upper()
    output: dict[str, dict[str, int]] = {}
    for compound, group in frame.groupby("Compound", sort=True):
        run_count = _unique_group_count(group, ("Driver", "LongRunId"))
        stint_count = _unique_group_count(group, ("Driver", "Stint"))
        output[str(compound)] = {
            "clean_lap_count": int(len(group)),
            "run_count": run_count,
            "stint_count": stint_count,
        }
    return output


def _unique_group_count(frame: pd.DataFrame, columns: tuple[str, ...]) -> int:
    available = [column for column in columns if column in frame.columns]
    if len(available) != len(columns):
        return 0
    return int(frame.dropna(subset=available)[available].drop_duplicates().shape[0])


def _read_sampler_diagnostics(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return {
        column: float(frame.iloc[0][column])
        for column in ("EffectiveSampleSize", "WeightedRMSESeconds")
        if column in frame.columns and pd.notna(frame.iloc[0][column])
    }


def _required_compound(preview: Mapping[str, object], field: str) -> str:
    value = str(preview.get(field) or "").upper()
    if value not in {"C1", "C2", "C3", "C4", "C5"}:
        raise RetroTyreExtractionError(f"Preview has no valid {field} allocation.")
    return value


def _combined_file_hash(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _read_existing_observations(path: Path) -> dict[int, list[dict[str, str]]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["round"]), []).append(row)
    return grouped


def _write_observations(rows: list[RetroTyreObservation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda value: (value.season, value.round, value.compound_role)):
            writer.writerow(row.to_dict())
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
