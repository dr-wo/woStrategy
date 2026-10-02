"""Prepare persisted SC/Red Flag history before report assembly or Planner use."""
import argparse
from wodata.events import parse_event_id
from wostrategy.analysis.interruption_history import prepare_history, history_markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, help="YYYY-RR")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    _, year, number = parse_event_id(args.event)
    print(history_markdown(prepare_history(year, number, data_root=args.data_root)))


if __name__ == "__main__":
    main()
