"""Production prerequisite management for Planner's live tyre model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

from wodata import get_data_root, get_fastf1_session_laps_path
from wodata.artifacts import (
    read_weekend_model_config,
    weekend_model_root,
)


REQUIRED_MODEL_CONFIG_FIELDS = (
    "sample_count",
    "random_seed",
    "fuel_rate_bounds",
    "track_rate_bounds",
    "default_degradation_bounds",
    "default_compound_delta_bounds",
    "reference_compound",
    "clean_lap_noise_sigma",
)
PRODUCER_API = "wostrategy.script.pre_race_analysis.run_from_script_config"


@dataclass(frozen=True)
class PreRaceModelConfigResult:
    status: str
    path: Path
    season: int
    round_number: int
    sessions: tuple[str, ...]
    reason: str
    producer_api: str = PRODUCER_API

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["path"] = str(self.path)
        return value


def ensure_pre_race_model_config(
    *,
    season: int,
    round_number: int,
    data_root: str | Path | None = None,
    session_names: Sequence[str] | None = None,
    runner: Callable[[dict], int] | None = None,
) -> PreRaceModelConfigResult:
    """Reuse a fresh live-MC config or run the canonical pre-race producer.

    Numerical work stays in ``pre_race_analysis``.  This function owns only
    identity validation, cache freshness, and production orchestration.
    """
    year = int(season)
    race = int(round_number)
    root = get_data_root(data_root)
    path = weekend_model_root(year, race, root) / "model_config.json"
    sessions = resolve_practice_sessions(session_names)
    existing = read_weekend_model_config(
        year=year, round_number=race, data_root=root
    )
    valid, reason = validate_pre_race_model_config(
        existing,
        season=year,
        round_number=race,
        sessions=sessions,
        path=path,
        data_root=root,
    )
    if valid:
        return PreRaceModelConfigResult(
            "reused", path, year, race, sessions, reason
        )

    if runner is None:
        from wostrategy.script.pre_race_analysis import (
            SCRIPT_CONFIG,
            run_from_script_config,
        )

        runner = run_from_script_config
        values = dict(SCRIPT_CONFIG)
    else:
        # Tests and embedding callers receive the same complete production
        # settings contract without importing/copying model calculations.
        from wostrategy.script.pre_race_analysis import SCRIPT_CONFIG

        values = dict(SCRIPT_CONFIG)
    values.update(
        year=year,
        round_number=race,
        sessions=sessions,
        data_root=root,
        plot_output=None,
        force_refresh_cache=False,
    )
    return_code = runner(values)
    if return_code not in (None, 0):
        raise RuntimeError(
            f"Canonical pre-race model-config producer returned {return_code}"
        )
    generated = read_weekend_model_config(
        year=year, round_number=race, data_root=root
    )
    generated_valid, generated_reason = validate_pre_race_model_config(
        generated,
        season=year,
        round_number=race,
        sessions=sessions,
        path=path,
        data_root=root,
    )
    if not generated_valid:
        raise RuntimeError(
            "Canonical pre-race model-config producer did not create a valid "
            f"artifact for {year} round {race}: {generated_reason}"
        )
    return PreRaceModelConfigResult(
        "generated", path, year, race, sessions, reason
    )


def validate_pre_race_model_config(
    config: Mapping[str, object] | None,
    *,
    season: int,
    round_number: int,
    sessions: Sequence[str],
    path: Path,
    data_root: str | Path | None,
) -> tuple[bool, str]:
    if not config:
        return False, "missing"
    missing = [field for field in REQUIRED_MODEL_CONFIG_FIELDS if field not in config]
    if missing:
        return False, "missing required fields: " + ", ".join(missing)
    try:
        if int(config["sample_count"]) <= 0:
            raise ValueError("sample_count must be positive")
        int(config["random_seed"])
        for name in (
            "fuel_rate_bounds",
            "track_rate_bounds",
            "default_degradation_bounds",
            "default_compound_delta_bounds",
        ):
            bounds = tuple(config[name])
            if len(bounds) != 2 or not all(math.isfinite(float(value)) for value in bounds):
                raise ValueError(f"{name} must contain two finite values")
            if float(bounds[0]) > float(bounds[1]):
                raise ValueError(f"{name} lower bound exceeds upper bound")
        if not str(config["reference_compound"]).strip():
            raise ValueError("reference_compound is empty")
        if not math.isfinite(float(config["clean_lap_noise_sigma"])):
            raise ValueError("clean_lap_noise_sigma is not finite")
    except (TypeError, ValueError) as exc:
        return False, f"invalid: {exc}"
    configured_year = config.get("season", config.get("year"))
    configured_round = config.get("round_number", config.get("round"))
    try:
        if configured_year is not None and int(configured_year) != int(season):
            return False, f"belongs to season {configured_year}, not {season}"
        if configured_round is not None and int(configured_round) != int(round_number):
            return False, f"belongs to round {configured_round}, not {round_number}"
    except (TypeError, ValueError) as exc:
        return False, f"invalid event identity: {exc}"
    configured_sessions = config.get("source_sessions")
    if configured_sessions is not None:
        try:
            normalized = resolve_practice_sessions(tuple(configured_sessions))
        except (TypeError, ValueError) as exc:
            return False, f"invalid source_sessions: {exc}"
        if normalized != tuple(sessions):
            return False, "practice-session set changed"
    if bool(config.get("stale", False)):
        return False, "explicitly marked stale"
    if path.is_file():
        config_mtime = path.stat().st_mtime
        for session in sessions:
            source = get_fastf1_session_laps_path(
                year=season,
                round_number=round_number,
                session=session,
                data_root=data_root,
            )
            if source.is_file() and source.stat().st_mtime > config_mtime:
                return False, f"{session} lap cache is newer than model config"
    return True, "valid and fresh"


def resolve_practice_sessions(session_names: Sequence[str] | None) -> tuple[str, ...]:
    if not session_names:
        return ("FP1", "FP2", "FP3")
    aliases = {
        "FP1": "FP1",
        "PRACTICE 1": "FP1",
        "FP2": "FP2",
        "PRACTICE 2": "FP2",
        "FP3": "FP3",
        "PRACTICE 3": "FP3",
    }
    sessions = tuple(
        dict.fromkeys(
            aliases[value]
            for item in session_names
            if (value := str(item).strip().upper()) in aliases
        )
    )
    if not sessions:
        raise ValueError("Event metadata contains no free-practice session")
    return sessions
