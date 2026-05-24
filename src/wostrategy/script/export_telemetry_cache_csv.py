from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wostrategy.core import get_session_telemetry_cache_path


SCRIPT_CONFIG = {
    "year": 2026,
    "race": 4,
    "section": "FP1",
    "output": None,
    "telemetry_cache_dir": None,
}


def export_telemetry_cache_to_csv(
    *,
    year: int,
    race: int | str,
    section: int | str,
    output_path: str | Path | None = None,
    telemetry_cache_dir: str | Path | None = None,
) -> Path:
    cache_path = get_session_telemetry_cache_path(
        year=year,
        round_number=race,
        session_name=section,
        cache_dir=telemetry_cache_dir,
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Telemetry cache does not exist: {cache_path}. "
            "Run a telemetry-gap loader first."
        )

    if output_path is None:
        output_path = Path("temp") / f"{cache_path.name}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    telemetry = pd.read_pickle(cache_path)
    telemetry.to_csv(output_path, index=False)
    print(f"Read telemetry cache from: {cache_path}")
    print(f"Wrote CSV to: {output_path}")
    print(f"Rows: {len(telemetry)}, columns: {len(telemetry.columns)}")
    return output_path


def main() -> None:
    args = _parse_args()
    export_telemetry_cache_to_csv(
        year=args.year,
        race=_parse_round(str(args.race)),
        section=args.section,
        output_path=args.output,
        telemetry_cache_dir=args.telemetry_cache_dir,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a cached session telemetry pickle to CSV."
    )
    parser.add_argument("--year", type=int, default=SCRIPT_CONFIG["year"])
    parser.add_argument("--race", default=SCRIPT_CONFIG["race"])
    parser.add_argument("--section", default=SCRIPT_CONFIG["section"])
    parser.add_argument("--output", type=Path, default=SCRIPT_CONFIG["output"])
    parser.add_argument(
        "--telemetry-cache-dir",
        type=Path,
        default=SCRIPT_CONFIG["telemetry_cache_dir"],
    )
    return parser.parse_args()


def _parse_round(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


if __name__ == "__main__":
    main()
