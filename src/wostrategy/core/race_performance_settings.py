from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from wodata import get_race_performance_review_event_dir, get_race_performance_review_root


RESOLVED_SETTINGS_SCHEMA_VERSION = 1
LATEST_RESOLVED_SETTINGS_FILENAME = "latest_resolved_settings.json"


@dataclass(frozen=True)
class ResolvedRacePerformanceSettings:
    settings: dict[str, object]
    source: str
    path: Path


def write_resolved_race_performance_settings(
    settings: Mapping[str, object],
    *,
    output_root: str | Path,
    year: int,
    races: list[int],
    session: str,
) -> tuple[Path, ...]:
    """Persist the effective review settings before data loading or fitting."""
    root = Path(output_root)
    payload = {
        "schema_version": RESOLVED_SETTINGS_SCHEMA_VERSION,
        "resolved_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "year": int(year),
        "races": [int(race) for race in races],
        "session": str(session).upper(),
        "settings": _json_value(dict(settings)),
    }
    destinations = [root / LATEST_RESOLVED_SETTINGS_FILENAME]
    destinations.extend(
        root
        / f"year={int(year)}"
        / f"round={int(race)}"
        / f"session={str(session).upper()}"
        / LATEST_RESOLVED_SETTINGS_FILENAME
        for race in races
    )
    for destination in destinations:
        _atomic_json(destination, payload)
    return tuple(destinations)


def resolve_race_performance_settings(
    *,
    year: int,
    round_number: int,
    session: str = "R",
    data_root: str | Path | None = None,
) -> ResolvedRacePerformanceSettings | None:
    """Resolve successful event metadata, then event/global invocation settings."""
    event_dir = get_race_performance_review_event_dir(
        year=year,
        round_number=round_number,
        session=session,
        data_root=data_root,
    )
    metadata_path = event_dir / (
        f"race_performance_{int(year)}_{int(round_number)}_{str(session).upper()}_metadata.json"
    )
    candidates = (
        (metadata_path, "event_cached_result"),
        (event_dir / LATEST_RESOLVED_SETTINGS_FILENAME, "event_latest_invocation"),
        (
            get_race_performance_review_root(data_root) / LATEST_RESOLVED_SETTINGS_FILENAME,
            "global_latest_invocation",
        ),
    )
    for path, source in candidates:
        value = _read_json(path)
        if value is None:
            continue
        settings = value.get("settings", value)
        if isinstance(settings, dict):
            return ResolvedRacePerformanceSettings(dict(settings), source, path)
    return None


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True, ensure_ascii=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
