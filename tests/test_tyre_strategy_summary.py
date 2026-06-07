from __future__ import annotations

import pandas as pd

from wostrategy.script.tyre_strategy_summary import (
    build_strategy_tables,
    extract_tyre_strategy,
    format_strategy,
    save_strategy_tables,
    CHINESE_COMPOUND_LABELS,
    ENGLISH_COMPOUND_LABELS,
)


def test_extract_tyre_strategy_ignores_drive_through_when_tyre_life_continues():
    laps = pd.DataFrame(
        {
            "Driver": ["VER"] * 6,
            "LapNumber": [1, 2, 3, 4, 5, 6],
            "Compound": ["SOFT", "SOFT", "SOFT", "SOFT", "MEDIUM", "MEDIUM"],
            "TyreLife": [1, 2, 3, 4, 1, 2],
            "PitInTime": [pd.NaT, pd.Timedelta(minutes=2), pd.NaT, pd.Timedelta(minutes=4), pd.NaT, pd.NaT],
            "PitOutTime": [pd.NaT, pd.NaT, pd.Timedelta(minutes=3), pd.NaT, pd.Timedelta(minutes=5), pd.NaT],
        }
    )

    strategy = extract_tyre_strategy(laps)

    assert strategy == [("SOFT", None), ("MEDIUM", 4)]


def test_extract_tyre_strategy_counts_same_compound_when_tyre_life_resets():
    laps = pd.DataFrame(
        {
            "Driver": ["VER"] * 4,
            "LapNumber": [1, 2, 3, 4],
            "Compound": ["MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM"],
            "TyreLife": [4, 5, 1, 2],
            "PitInTime": [pd.NaT, pd.Timedelta(minutes=2), pd.NaT, pd.NaT],
        }
    )

    strategy = extract_tyre_strategy(laps)

    assert strategy == [("MEDIUM", None), ("MEDIUM", 2)]


def test_format_strategy_outputs_chinese_and_english_compounds():
    strategy = [
        ("SOFT", None),
        ("MEDIUM", 12),
        ("HARD", 33),
        ("INTERMEDIATE", 41),
        ("WET", 47),
    ]

    assert format_strategy(strategy, CHINESE_COMPOUND_LABELS) == "软-(12)中-(33)硬-(41)半-(47)全"
    assert format_strategy(strategy, ENGLISH_COMPOUND_LABELS) == "S-(12)M-(33)H-(41)I-(47)W"


def test_build_strategy_tables_returns_chinese_and_english_tables():
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "VER", "LEC", "LEC"],
            "LapNumber": [1, 2, 1, 2],
            "Compound": ["SOFT", "MEDIUM", "HARD", "HARD"],
            "TyreLife": [1, 1, 1, 2],
            "PitInTime": [pd.Timedelta(minutes=1), pd.NaT, pd.NaT, pd.NaT],
            "SessionResultRank": [2, 2, 1, 1],
        }
    )

    chinese, english = build_strategy_tables(laps)

    assert chinese.columns.tolist() == ["车手", "轮胎策略"]
    assert english.columns.tolist() == ["Driver", "Tyre Strategy"]
    assert english["Driver"].tolist() == ["LEC", "VER"]
    assert chinese["车手"].tolist() == ["LEC", "VER"]
    assert english.set_index("Driver").loc["VER", "Tyre Strategy"] == "S-(1)M"
    assert chinese.set_index("车手").loc["LEC", "轮胎策略"] == "硬"


def test_save_strategy_tables_writes_two_column_csvs(tmp_path):
    chinese = pd.DataFrame({"车手": ["LEC"], "轮胎策略": ["硬"]})
    english = pd.DataFrame({"Driver": ["LEC"], "Tyre Strategy": ["H"]})

    chinese_path, english_path = save_strategy_tables(
        chinese,
        english,
        output_prefix=tmp_path / "strategy.csv",
    )

    assert chinese_path.name == "strategy_chinese.csv"
    assert english_path.name == "strategy_english.csv"
    assert pd.read_csv(chinese_path).shape == (1, 2)
    assert pd.read_csv(english_path).shape == (1, 2)
