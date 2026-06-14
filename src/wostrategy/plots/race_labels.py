from __future__ import annotations

import pandas as pd


def race_tick_labels(
    frame: pd.DataFrame,
    *,
    round_column: str,
    name_columns: tuple[str, ...] = (
        "EventName",
        "RaceName",
        "EventLocation",
        "EventCountry",
    ),
) -> pd.DataFrame:
    label_column = _first_existing_column(frame, name_columns)
    columns = [round_column]
    if label_column is not None:
        columns.append(label_column)

    labels = frame.loc[:, columns].drop_duplicates().sort_values(round_column).copy()
    if labels.empty:
        return pd.DataFrame(columns=[round_column, "Label"])

    if label_column is None:
        labels["Label"] = labels[round_column].map(round_label)
    else:
        labels["Label"] = [
            f"{round_label(getattr(row, round_column))}\n"
            f"{short_event_name(getattr(row, label_column))}"
            for row in labels.itertuples(index=False)
        ]
    return labels.loc[:, [round_column, "Label"]]


def round_label(round_number: object) -> str:
    try:
        return f"R{int(round_number)}"
    except (TypeError, ValueError):
        return f"R{round_number}"


def short_event_name(event_name: object) -> str:
    short_name = str(event_name).replace("Grand Prix", "").strip()
    return short_name or str(event_name)


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns and frame[candidate].notna().any():
            return candidate
    return None


__all__ = ["race_tick_labels", "round_label", "short_event_name"]
