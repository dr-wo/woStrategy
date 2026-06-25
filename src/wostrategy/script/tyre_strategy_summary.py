from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wostrategy.tools import load_all_session_laps


SCRIPT_CONFIG = {
    "year": 2026,
    "race": 7,
    "session": "R",
    "test": False,
    "output": None,
}

CHINESE_COMPOUND_LABELS = {
    "SOFT": "软",
    "MEDIUM": "中",
    "HARD": "硬",
    "INTERMEDIATE": "半",
    "INTER": "半",
    "INTERS": "半",
    "WET": "全",
}
ENGLISH_COMPOUND_LABELS = {
    "SOFT": "S",
    "MEDIUM": "M",
    "HARD": "H",
    "INTERMEDIATE": "I",
    "INTER": "I",
    "INTERS": "I",
    "WET": "W",
}


def tyre_strategy_summary(
    *,
    year: int,
    race: int | str,
    session: int | str = SCRIPT_CONFIG["session"],
    test: bool = SCRIPT_CONFIG["test"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    laps = load_all_session_laps(
        year=year,
        rounds=[race],
        session_names=[session],
        test=test,
    )
    if laps.empty:
        raise ValueError(f"No laps loaded for year={year}, race={race}, session={session}")
    return build_strategy_tables(laps)


def build_strategy_tables(laps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {"Driver", "LapNumber", "Compound"}
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Laps are missing required columns: {missing}")

    driver_column = _driver_name_column(laps)
    records = []
    for driver, driver_laps in laps.dropna(subset=["Driver"]).groupby("Driver", sort=True):
        driver_name = _driver_display_name(driver_laps, driver_column, fallback=str(driver))
        strategy = extract_tyre_strategy(driver_laps)
        if not strategy:
            continue
        records.append(
            {
                "_SortRank": _driver_result_rank(driver_laps),
                "Driver": driver_name,
                "Tyre Strategy": format_strategy(strategy, ENGLISH_COMPOUND_LABELS),
                "车手": driver_name,
                "轮胎策略": format_strategy(strategy, CHINESE_COMPOUND_LABELS),
            }
        )

    if not records:
        raise ValueError("No tyre strategy rows available.")

    rows = pd.DataFrame(records).sort_values(["_SortRank", "Driver"]).reset_index(drop=True)
    english = rows[["Driver", "Tyre Strategy"]].copy()
    chinese = rows[["车手", "轮胎策略"]].copy()
    return chinese, english


def extract_tyre_strategy(driver_laps: pd.DataFrame) -> list[tuple[str, int | None]]:
    laps = driver_laps.dropna(subset=["LapNumber", "Compound"]).copy()
    if laps.empty:
        return []

    laps["LapNumber"] = pd.to_numeric(laps["LapNumber"], errors="coerce")
    sort_columns = ["LapNumber"]
    if "LapStartTime" in laps.columns:
        sort_columns.append("LapStartTime")
    laps = laps.dropna(subset=["LapNumber"]).sort_values(sort_columns)
    if laps.empty:
        return []

    strategy: list[tuple[str, int | None]] = []
    previous_lap: pd.Series | None = None
    previous_compound: str | None = None
    for _, lap in laps.iterrows():
        compound = normalize_compound(lap["Compound"])
        if compound is None:
            continue

        if previous_lap is None:
            strategy.append((compound, None))
        elif _is_new_tyre_segment(
            previous_lap=previous_lap,
            current_lap=lap,
            previous_compound=previous_compound,
            current_compound=compound,
        ):
            strategy.append((compound, _pit_marker_lap(previous_lap, lap)))

        previous_lap = lap
        previous_compound = compound

    return _collapse_duplicate_markers(strategy)


def format_strategy(
    strategy: list[tuple[str, int | None]],
    compound_labels: dict[str, str],
) -> str:
    parts: list[str] = []
    for compound, marker_lap in strategy:
        label = compound_labels.get(compound, compound.title())
        if marker_lap is not None:
            parts.append(f"-({marker_lap})")
        parts.append(label)
    return "".join(parts)


def normalize_compound(value: object) -> str | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip().upper()
    if not normalized:
        return None
    return normalized


def _is_new_tyre_segment(
    *,
    previous_lap: pd.Series,
    current_lap: pd.Series,
    previous_compound: str | None,
    current_compound: str,
) -> bool:
    if previous_compound is not None and current_compound != previous_compound:
        return True

    previous_age = _tyre_life(previous_lap)
    current_age = _tyre_life(current_lap)
    if previous_age is None or current_age is None:
        return False
    return current_age < previous_age


def _pit_marker_lap(previous_lap: pd.Series, current_lap: pd.Series) -> int:
    previous_lap_number = int(previous_lap["LapNumber"])
    current_lap_number = int(current_lap["LapNumber"])
    if "PitInTime" in previous_lap.index and pd.notna(previous_lap["PitInTime"]):
        return previous_lap_number
    return current_lap_number


def _tyre_life(lap: pd.Series) -> float | None:
    if "TyreLife" not in lap.index:
        return None
    value = pd.to_numeric(pd.Series([lap["TyreLife"]]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _collapse_duplicate_markers(strategy: list[tuple[str, int | None]]) -> list[tuple[str, int | None]]:
    collapsed: list[tuple[str, int | None]] = []
    for compound, marker_lap in strategy:
        if collapsed and collapsed[-1] == (compound, marker_lap):
            continue
        collapsed.append((compound, marker_lap))
    return collapsed


def _driver_name_column(laps: pd.DataFrame) -> str:
    for column in ("FullName", "BroadcastName", "Driver"):
        if column in laps.columns:
            return column
    return "Driver"


def _driver_display_name(laps: pd.DataFrame, column: str, *, fallback: str) -> str:
    if column not in laps.columns:
        return fallback
    names = laps[column].dropna().astype(str)
    if names.empty:
        return fallback
    return names.iloc[0]


def _driver_result_rank(laps: pd.DataFrame) -> float:
    if "SessionResultRank" not in laps.columns:
        return float("inf")

    ranks = pd.to_numeric(laps["SessionResultRank"], errors="coerce").dropna()
    if ranks.empty:
        return float("inf")
    return float(ranks.min())


def main() -> None:
    args = _parse_args()
    race = _parse_round(str(args.race))
    session = _parse_round(str(args.session))
    chinese, english = tyre_strategy_summary(
        year=args.year,
        race=race,
        session=session,
        test=args.test,
    )

    print("\n中文")
    print(chinese.to_string(index=False))
    print("\nEnglish")
    print(english.to_string(index=False))

    output_prefix = args.output
    if output_prefix is None:
        output_prefix = _default_output_prefix(args.year, race, session)
    chinese_path, english_path = save_strategy_tables(
        chinese,
        english,
        output_prefix=output_prefix,
    )
    print(f"\nSaved Chinese tyre strategy table to {chinese_path}")
    print(f"Saved English tyre strategy table to {english_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print Chinese and English tyre strategy tables for one feature race."
    )
    parser.add_argument("--year", type=int, default=SCRIPT_CONFIG["year"])
    parser.add_argument(
        "--race",
        default=SCRIPT_CONFIG["race"],
        help="FastF1 round number or event name.",
    )
    parser.add_argument(
        "--session",
        default=SCRIPT_CONFIG["session"],
        help="Feature race session name. Defaults to FastF1 race session 'R'.",
    )
    parser.add_argument("--test", action="store_true", default=SCRIPT_CONFIG["test"])
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_CONFIG["output"],
        help=(
            "Output prefix or CSV path. Defaults to temp/"
            "tyre_strategy_summary_<year>_<race>_<session>."
        ),
    )
    return parser.parse_args()


def _parse_round(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def _default_output_prefix(year: int, race: int | str, session: int | str) -> Path:
    return Path("temp") / f"tyre_strategy_summary_{year}_{race}_{session}"


def save_strategy_tables(
    chinese: pd.DataFrame,
    english: pd.DataFrame,
    *,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    if prefix.suffix.lower() == ".csv":
        base = prefix.with_suffix("")
    else:
        base = prefix

    chinese_path = base.with_name(f"{base.name}_chinese.csv")
    english_path = base.with_name(f"{base.name}_english.csv")
    chinese_path.parent.mkdir(parents=True, exist_ok=True)
    chinese.to_csv(chinese_path, index=False)
    english.to_csv(english_path, index=False)
    return chinese_path, english_path


if __name__ == "__main__":
    main()
