from __future__ import annotations

import argparse

from wostrategy.analysis.fp_race_diagnostic import run_historical_fp_race_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate persisted FP tyre evidence against later Race Retro targets.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--data-root")
    args = parser.parse_args()
    paths = run_historical_fp_race_diagnostic(season=args.season, data_root=args.data_root)
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
