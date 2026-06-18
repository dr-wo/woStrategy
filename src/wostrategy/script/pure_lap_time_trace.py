from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from wostrategy.plots.style_maps import build_team_style_maps
from wostrategy.tools import load_all_session_laps


SCRIPT_CONFIG = {
    "year": 2026,
    "race": 7,
    "session": "R",
    "test": False,
    # Use driver abbreviations as keys for one trace per driver. To plot the
    # same driver more than once, use a unique key and set "driver".
    "traces": {
        "RUS": {"lap": ["37-61"], "off-set": 0.12},
        "ANT": {"lap": ["38-61"], "off-set": 0.09},
        "HAM": {"lap": ["41-61"], "off-set": 0},

        # "VER hard": {"driver": "VER", "lap": ["28-35"], "linestyle": "--"},
        # "LEC": {"lap": ["3-8"], "color": "#dc0000", "label": "LEC adjusted"},
    },
    # Optional secondary-axis cumulative deltas from already-collected trace
    # data. Positive means trace_a lost time to trace_b over the selected
    # collected lap numbers.
    "delta_traces": {
        "RUS vs HAM": {
            "trace_a": "RUS",
            "trace_b": "HAM",
            "lap": ["7-21"],
        },
    },
    # Set to [min_seconds, max_seconds] to override matplotlib autoscaling.
    # Example: [79.5, 84.0]
    "y_range": [80, 83],
    "delta_y_range": None,
    "output": None,
    "show": False,
}


@dataclass(frozen=True)
class LapTimeTrace:
    key: str
    driver: str
    label: str
    laps: tuple[int, ...]
    offset_seconds: float
    color: object | None = None
    linestyle: str | None = None
    marker: str | None = None


@dataclass(frozen=True)
class AccumulatedDeltaTrace:
    key: str
    trace_a: str
    trace_b: str
    label: str
    collected_laps: tuple[int, ...]
    color: object | None = None
    linestyle: str | None = None
    marker: str | None = None


def run_pure_lap_time_trace(
    *,
    year: int,
    race: int | str,
    session: int | str,
    traces: dict[str, Any] | list[dict[str, Any]],
    delta_traces: dict[str, Any] | list[dict[str, Any]] | None = None,
    y_range: tuple[float | None, float | None] | list[float | None] | None = None,
    delta_y_range: tuple[float | None, float | None] | list[float | None] | None = None,
    output: str | Path | None = None,
    test: bool = False,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame, Path | None]:
    laps = load_all_session_laps(
        year=year,
        rounds=[race],
        session_names=[session],
        test=test,
    )
    if laps.empty:
        raise ValueError(f"No laps loaded for year={year}, race={race}, session={session}")

    trace_configs = normalize_trace_configs(traces)
    delta_trace_configs = normalize_delta_trace_configs(delta_traces)
    plot_data = build_pure_lap_time_trace_data(laps, trace_configs)
    delta_plot_data = build_accumulated_delta_trace_data(plot_data, delta_trace_configs)
    if plot_data.empty:
        raise ValueError("No requested lap times were found.")

    fig, ax = plot_pure_lap_time_traces(
        plot_data,
        laps=laps,
        trace_configs=trace_configs,
        delta_plot_data=delta_plot_data,
        delta_trace_configs=delta_trace_configs,
        title=f"Pure Lap Time Trace {year} R{race} {session}",
        y_range=normalize_y_range(y_range),
        delta_y_range=normalize_y_range(delta_y_range),
    )

    output_path = Path(output) if output is not None else _default_output_path(year, race, session)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax, plot_data, output_path


def normalize_trace_configs(traces: dict[str, Any] | list[dict[str, Any]]) -> list[LapTimeTrace]:
    if isinstance(traces, dict):
        items = traces.items()
    elif isinstance(traces, list):
        items = []
        for idx, config in enumerate(traces, start=1):
            if not isinstance(config, dict):
                raise TypeError("Each trace list entry must be a dictionary.")
            key = str(config.get("key") or config.get("label") or f"trace_{idx}")
            items.append((key, config))
    else:
        raise TypeError("traces must be a dictionary or a list of dictionaries.")

    normalized = []
    for key, raw_config in items:
        if not isinstance(raw_config, dict):
            raise TypeError(f"Trace {key!r} must be configured with a dictionary.")

        driver = str(raw_config.get("driver") or raw_config.get("Driver") or key).strip()
        if not driver:
            raise ValueError(f"Trace {key!r} has an empty driver.")

        lap_selector = raw_config.get("lap", raw_config.get("laps"))
        if lap_selector is None:
            raise ValueError(f"Trace {key!r} must define 'lap' or 'laps'.")

        offset = raw_config.get("off-set", raw_config.get("offset", raw_config.get("offset_seconds", 0.0)))
        marker = raw_config.get("marker", "o")
        normalized.append(
            LapTimeTrace(
                key=str(key),
                driver=driver,
                label=str(raw_config.get("label") or key),
                laps=tuple(expand_lap_selector(lap_selector)),
                offset_seconds=float(offset),
                color=raw_config.get("color"),
                linestyle=raw_config.get("linestyle", raw_config.get("line-style")),
                marker=None if marker in (None, "") else str(marker),
            )
        )
    return normalized


def normalize_delta_trace_configs(
    traces: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[AccumulatedDeltaTrace]:
    if traces is None:
        return []
    if isinstance(traces, dict):
        items = traces.items()
    elif isinstance(traces, list):
        items = []
        for idx, config in enumerate(traces, start=1):
            if not isinstance(config, dict):
                raise TypeError("Each delta trace list entry must be a dictionary.")
            key = str(config.get("key") or config.get("label") or f"delta_{idx}")
            items.append((key, config))
    else:
        raise TypeError("delta_traces must be a dictionary, a list of dictionaries, or None.")

    normalized = []
    for key, raw_config in items:
        if not isinstance(raw_config, dict):
            raise TypeError(f"Delta trace {key!r} must be configured with a dictionary.")

        trace_a = str(
            raw_config.get("trace_a")
            or raw_config.get("traceA")
            or raw_config.get("driver_a")
            or raw_config.get("driverA")
            or ""
        ).strip()
        trace_b = str(
            raw_config.get("trace_b")
            or raw_config.get("traceB")
            or raw_config.get("driver_b")
            or raw_config.get("driverB")
            or ""
        ).strip()
        if not trace_a or not trace_b:
            raise ValueError(f"Delta trace {key!r} must define trace_a and trace_b.")

        lap_selector = raw_config.get("lap", raw_config.get("laps"))
        if lap_selector is None:
            raise ValueError(f"Delta trace {key!r} must define collected 'lap' or 'laps'.")

        marker = raw_config.get("marker", None)
        normalized.append(
            AccumulatedDeltaTrace(
                key=str(key),
                trace_a=trace_a,
                trace_b=trace_b,
                label=str(raw_config.get("label") or key),
                collected_laps=tuple(expand_lap_selector(lap_selector)),
                color=raw_config.get("color"),
                linestyle=raw_config.get("linestyle", raw_config.get("line-style")),
                marker=None if marker in (None, "") else str(marker),
            )
        )
    return normalized


def expand_lap_selector(selector: object) -> list[int]:
    if isinstance(selector, (str, int)):
        selector_values: Iterable[object] = [selector]
    else:
        selector_values = selector  # type: ignore[assignment]

    laps: list[int] = []
    for value in selector_values:
        if isinstance(value, int):
            laps.append(value)
            continue

        text = str(value).strip()
        if not text:
            continue
        if "-" in text:
            start_text, end_text = text.split("-", maxsplit=1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if end >= start else -1
            laps.extend(range(start, end + step, step))
        else:
            laps.append(int(text))

    seen = set()
    unique_laps = []
    for lap in laps:
        if lap in seen:
            continue
        seen.add(lap)
        unique_laps.append(lap)
    return unique_laps


def build_pure_lap_time_trace_data(
    laps: pd.DataFrame,
    trace_configs: list[LapTimeTrace],
) -> pd.DataFrame:
    required_columns = {"Driver", "LapNumber", "LapTime"}
    missing_columns = required_columns.difference(laps.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Laps are missing required columns: {missing}")

    frames = []
    for trace in trace_configs:
        driver_laps = laps.loc[
            (laps["Driver"] == trace.driver) & (laps["LapNumber"].isin(trace.laps))
        ].copy()
        if driver_laps.empty:
            print(f"Skipping {trace.label}: no laps found for {trace.driver} laps {list(trace.laps)}")
            continue

        driver_laps["LapNumber"] = pd.to_numeric(driver_laps["LapNumber"], errors="coerce")
        sort_columns = ["LapNumber"]
        if "LapStartTime" in driver_laps.columns:
            sort_columns.append("LapStartTime")
        driver_laps = driver_laps.dropna(subset=["LapNumber", "LapTime"]).sort_values(sort_columns)
        if driver_laps.empty:
            print(f"Skipping {trace.label}: requested laps have no lap times")
            continue

        driver_laps["CollectedLapNumber"] = range(1, len(driver_laps) + 1)
        driver_laps["LapTimeSeconds"] = driver_laps["LapTime"].dt.total_seconds()
        driver_laps["PureLapTimeSeconds"] = driver_laps["LapTimeSeconds"] - trace.offset_seconds
        driver_laps["TraceKey"] = trace.key
        driver_laps["TraceLabel"] = trace.label
        driver_laps["OffsetSeconds"] = trace.offset_seconds
        frames.append(driver_laps)

    if not frames:
        return pd.DataFrame(
            columns=[
                "TraceKey",
                "TraceLabel",
                "Driver",
                "LapNumber",
                "CollectedLapNumber",
                "LapTimeSeconds",
                "PureLapTimeSeconds",
                "OffsetSeconds",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def build_accumulated_delta_trace_data(
    plot_data: pd.DataFrame,
    trace_configs: list[AccumulatedDeltaTrace],
) -> pd.DataFrame:
    if not trace_configs:
        return _empty_accumulated_delta_trace_data()

    required_columns = {
        "TraceKey",
        "TraceLabel",
        "Driver",
        "LapNumber",
        "CollectedLapNumber",
        "PureLapTimeSeconds",
    }
    missing_columns = required_columns.difference(plot_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Plot data is missing required columns: {missing}")

    frames = []
    for trace in trace_configs:
        trace_a_laps = _collected_trace_laps(
            plot_data,
            trace_key=trace.trace_a,
            collected_laps=trace.collected_laps,
            prefix="TraceA",
        )
        trace_b_laps = _collected_trace_laps(
            plot_data,
            trace_key=trace.trace_b,
            collected_laps=trace.collected_laps,
            prefix="TraceB",
        )

        matched_laps = trace_a_laps.merge(trace_b_laps, on="CollectedLapNumber", how="inner")
        matched_laps = matched_laps.sort_values("CollectedLapNumber").reset_index(drop=True)
        if matched_laps.empty:
            print(
                f"Skipping {trace.label}: no matching collected laps found for "
                f"{trace.trace_a} and {trace.trace_b} laps {list(trace.collected_laps)}"
            )
            continue

        matched_laps["LapDeltaSeconds"] = (
            matched_laps["TraceAPureLapTimeSeconds"]
            - matched_laps["TraceBPureLapTimeSeconds"]
        )
        matched_laps["AccumulatedDeltaSeconds"] = matched_laps["LapDeltaSeconds"].cumsum()
        matched_laps["TraceKey"] = trace.key
        matched_laps["TraceLabel"] = trace.label
        matched_laps["TraceA"] = trace.trace_a
        matched_laps["TraceB"] = trace.trace_b
        frames.append(matched_laps)

    if not frames:
        return _empty_accumulated_delta_trace_data()
    return pd.concat(frames, ignore_index=True)


def plot_pure_lap_time_traces(
    plot_data: pd.DataFrame,
    *,
    laps: pd.DataFrame,
    trace_configs: list[LapTimeTrace],
    title: str,
    delta_plot_data: pd.DataFrame | None = None,
    delta_trace_configs: list[AccumulatedDeltaTrace] | None = None,
    y_range: tuple[float | None, float | None] | None = None,
    delta_y_range: tuple[float | None, float | None] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = _resolve_trace_styles(laps, trace_configs)

    for trace in trace_configs:
        trace_data = plot_data.loc[plot_data["TraceKey"] == trace.key]
        if trace_data.empty:
            continue
        style = styles[trace.key]
        ax.plot(
            trace_data["CollectedLapNumber"],
            trace_data["PureLapTimeSeconds"],
            label=trace.label,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2,
        )

    ax.set_title(title)
    ax.set_xlabel("Collected lap number")
    ax.set_ylabel("Lap time minus offset (s)")
    if y_range is not None:
        ax.set_ylim(*y_range)
    ax.grid(True, linestyle=":", alpha=0.5)

    delta_ax = None
    if delta_plot_data is not None and not delta_plot_data.empty and delta_trace_configs:
        delta_ax = ax.twinx()
        delta_styles = _resolve_delta_trace_styles(delta_trace_configs)
        for trace in delta_trace_configs:
            trace_data = delta_plot_data.loc[delta_plot_data["TraceKey"] == trace.key]
            if trace_data.empty:
                continue
            style = delta_styles[trace.key]
            delta_ax.plot(
                trace_data["CollectedLapNumber"],
                trace_data["AccumulatedDeltaSeconds"],
                label=trace.label,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2,
            )
        delta_ax.set_ylabel("Accumulated delta (s)")
        if delta_y_range is not None:
            delta_ax.set_ylim(*delta_y_range)

    handles, labels = ax.get_legend_handles_labels()
    if delta_ax is not None:
        delta_handles, delta_labels = delta_ax.get_legend_handles_labels()
        handles.extend(delta_handles)
        labels.extend(delta_labels)
    ax.legend(handles, labels)
    fig.tight_layout()
    return fig, ax


def _empty_accumulated_delta_trace_data() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "TraceKey",
            "TraceLabel",
            "TraceA",
            "TraceB",
            "TraceADriver",
            "TraceBDriver",
            "TraceALapNumber",
            "TraceBLapNumber",
            "CollectedLapNumber",
            "TraceAPureLapTimeSeconds",
            "TraceBPureLapTimeSeconds",
            "LapDeltaSeconds",
            "AccumulatedDeltaSeconds",
        ]
    )


def _collected_trace_laps(
    plot_data: pd.DataFrame,
    *,
    trace_key: str,
    collected_laps: tuple[int, ...],
    prefix: str,
) -> pd.DataFrame:
    trace_laps = plot_data.loc[
        (plot_data["TraceKey"] == trace_key)
        & (plot_data["CollectedLapNumber"].isin(collected_laps)),
        ["CollectedLapNumber", "Driver", "LapNumber", "PureLapTimeSeconds"],
    ].copy()
    trace_laps = trace_laps.sort_values("CollectedLapNumber").drop_duplicates(
        subset=["CollectedLapNumber"],
        keep="first",
    )
    return trace_laps.rename(
        columns={
            "Driver": f"{prefix}Driver",
            "LapNumber": f"{prefix}LapNumber",
            "PureLapTimeSeconds": f"{prefix}PureLapTimeSeconds",
        }
    )


def normalize_y_range(
    y_range: tuple[float | None, float | None] | list[float | None] | None,
) -> tuple[float | None, float | None] | None:
    if y_range is None:
        return None
    if len(y_range) != 2:
        raise ValueError("y_range must contain exactly two values: [min_seconds, max_seconds].")

    y_min, y_max = y_range
    normalized = (
        None if y_min is None else float(y_min),
        None if y_max is None else float(y_max),
    )
    if normalized[0] is not None and normalized[1] is not None and normalized[0] >= normalized[1]:
        raise ValueError("y_range minimum must be smaller than y_range maximum.")
    return normalized


def _resolve_trace_styles(
    laps: pd.DataFrame,
    trace_configs: list[LapTimeTrace],
) -> dict[str, dict[str, object]]:
    team_color_map, team_driver_linestyle_map, _ = build_team_style_maps(laps)
    fallback_cmap = plt.get_cmap("tab10")
    repeated_driver_counts: dict[str, int] = {}
    styles = {}

    for idx, trace in enumerate(trace_configs):
        driver_laps = laps.loc[laps["Driver"] == trace.driver]
        team = _driver_team(driver_laps)
        repeat_idx = repeated_driver_counts.get(trace.driver, 0)
        repeated_driver_counts[trace.driver] = repeat_idx + 1

        default_color = (
            team_color_map.get(team)
            if team is not None
            else fallback_cmap(idx % fallback_cmap.N)
        )
        default_linestyle = (
            team_driver_linestyle_map.get((team, trace.driver))
            if team is not None
            else None
        )
        if repeat_idx > 0:
            default_linestyle = ["-", "--", ":", "-."][repeat_idx % 4]

        styles[trace.key] = {
            "color": trace.color if trace.color is not None else default_color,
            "linestyle": trace.linestyle or default_linestyle or "-",
            "marker": trace.marker,
        }
    return styles


def _resolve_delta_trace_styles(
    trace_configs: list[AccumulatedDeltaTrace],
) -> dict[str, dict[str, object]]:
    styles = {}
    for trace in trace_configs:
        styles[trace.key] = {
            "color": trace.color if trace.color is not None else "black",
            "linestyle": trace.linestyle or "-.",
            "marker": trace.marker,
        }
    return styles


def _driver_team(driver_laps: pd.DataFrame) -> str | None:
    if driver_laps.empty or "Team" not in driver_laps.columns:
        return None
    teams = driver_laps["Team"].dropna().astype(str)
    if teams.empty:
        return None
    return teams.mode().iloc[0]


def main() -> None:
    args = _parse_args()
    race = _parse_round(str(args.race))
    session = _parse_round(str(args.session))
    traces = _load_traces(args)
    delta_traces = _load_delta_traces(args)
    _, _, plot_data, output_path = run_pure_lap_time_trace(
        year=args.year,
        race=race,
        session=session,
        traces=traces,
        delta_traces=delta_traces,
        y_range=args.y_range,
        delta_y_range=args.delta_y_range,
        output=args.output,
        test=args.test,
        show=args.show,
    )
    print(plot_data[["TraceLabel", "Driver", "LapNumber", "PureLapTimeSeconds"]].to_string(index=False))
    print(f"\nSaved pure lap time trace plot to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot selected driver lap times against collected lap number, with optional "
            "per-trace lap-time offsets."
        )
    )
    parser.add_argument("--year", type=int, default=SCRIPT_CONFIG["year"])
    parser.add_argument("--race", default=SCRIPT_CONFIG["race"], help="FastF1 round number or event name.")
    parser.add_argument("--session", default=SCRIPT_CONFIG["session"], help="FastF1 session name.")
    parser.add_argument(
        "--traces-json",
        help=(
            "JSON dictionary/list defining traces. Example: "
            '\'{"VER": {"lap": [2, "4-6"], "off-set": 0.2}}\''
        ),
    )
    parser.add_argument(
        "--traces-file",
        type=Path,
        help="Path to a JSON file containing the trace dictionary/list.",
    )
    parser.add_argument(
        "--delta-traces-json",
        help=(
            "JSON dictionary/list defining accumulated delta traces. Example: "
            '\'{"RUS vs HAM": {"trace_a": "RUS", "trace_b": "HAM", "lap": ["7-21"]}}\''
        ),
    )
    parser.add_argument(
        "--delta-traces-file",
        type=Path,
        help="Path to a JSON file containing accumulated delta trace definitions.",
    )
    parser.add_argument("--test", action="store_true", default=SCRIPT_CONFIG["test"])
    parser.add_argument("--show", action="store_true", default=SCRIPT_CONFIG["show"])
    parser.add_argument(
        "--y-range",
        nargs=2,
        type=float,
        default=SCRIPT_CONFIG["y_range"],
        metavar=("MIN_SECONDS", "MAX_SECONDS"),
        help="Override y-axis range in seconds, for example: --y-range 79.5 84.0.",
    )
    parser.add_argument(
        "--delta-y-range",
        nargs=2,
        type=float,
        default=SCRIPT_CONFIG["delta_y_range"],
        metavar=("MIN_SECONDS", "MAX_SECONDS"),
        help="Override right-hand accumulated-delta y-axis range in seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_CONFIG["output"],
        help="Output PNG path. Defaults to temp/pure_lap_time_trace_<year>_<race>_<session>.png.",
    )
    return parser.parse_args()


def _load_traces(args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    if args.traces_json and args.traces_file:
        raise ValueError("Use only one of --traces-json or --traces-file.")
    if args.traces_json:
        return json.loads(args.traces_json)
    if args.traces_file:
        return json.loads(args.traces_file.read_text())
    return SCRIPT_CONFIG["traces"]


def _load_delta_traces(args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]] | None:
    if args.delta_traces_json and args.delta_traces_file:
        raise ValueError("Use only one of --delta-traces-json or --delta-traces-file.")
    if args.delta_traces_json:
        return json.loads(args.delta_traces_json)
    if args.delta_traces_file:
        return json.loads(args.delta_traces_file.read_text())
    return SCRIPT_CONFIG["delta_traces"]


def _parse_round(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def _default_output_path(year: int, race: int | str, session: int | str) -> Path:
    return Path("temp") / f"pure_lap_time_trace_{year}_{race}_{session}.png"


if __name__ == "__main__":
    main()
