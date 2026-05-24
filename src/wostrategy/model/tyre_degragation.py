from wostrategy.core.session import Session


def main() -> None:
    year = 2021
    round_number = 21

    for session_name in ["FP1", "FP2", "FP3"]:
        session = Session(year, round_number, session_name)
        session.quicklaps()


if __name__ == "__main__":
    main()
