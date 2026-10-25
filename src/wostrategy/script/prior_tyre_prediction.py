"""Create a prior-only tyre artifact from an earlier frozen P0/D0 prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wostrategy.analysis.tyre_prediction_prior import write_historical_baseline_prior


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--round", required=True, type=int, dest="round_number")
    parser.add_argument("--event", required=True)
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--hard", required=True, help="Verified Pirelli dry HARD compound, e.g. C2")
    parser.add_argument("--medium", required=True, help="Verified Pirelli dry MEDIUM compound")
    parser.add_argument("--soft", required=True, help="Verified Pirelli dry SOFT compound")
    parser.add_argument("--allocation-source-url", required=True)
    parser.add_argument("--source-round", type=int)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    path = write_historical_baseline_prior(
        season=args.year,
        round_number=args.round_number,
        event=args.event,
        circuit=args.circuit,
        compound_allocation={"HARD": args.hard, "MEDIUM": args.medium, "SOFT": args.soft},
        allocation_source_url=args.allocation_source_url,
        data_root=args.data_root,
        source_round_number=args.source_round,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({
        "status": payload["prediction_status"],
        "path": str(path),
        "prediction_fingerprint": payload["prediction_fingerprint"],
        "trained_through_round": payload["trained_through_round"],
        "training_rounds": payload["training_rounds"],
        "training_data_hash": payload["training_data_hash"],
        "compound_allocation": payload["compound_allocation"],
        "uncertainty": "unavailable",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
