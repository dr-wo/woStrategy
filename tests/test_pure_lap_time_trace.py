from __future__ import annotations

import pandas as pd

from wostrategy.script.pure_lap_time_trace import (
    build_accumulated_delta_trace_data,
    build_pure_lap_time_trace_data,
    expand_lap_selector,
    normalize_delta_trace_configs,
    normalize_trace_configs,
    normalize_y_range,
    plot_pure_lap_time_traces,
)


def test_expand_lap_selector_keeps_collected_order_and_expands_ranges():
    assert expand_lap_selector([2, "4-6", "9-11", 4]) == [2, 4, 5, 6, 9, 10, 11]


def test_normalize_trace_configs_supports_repeated_driver_and_overrides():
    traces = normalize_trace_configs(
        {
            "VER first": {
                "driver": "VER",
                "lap": [2, "4-5"],
                "off-set": 0.2,
                "color": "#123456",
                "line-style": "--",
            },
            "VER second": {"driver": "VER", "laps": ["9-10"], "offset": 0.1},
        }
    )

    assert [trace.driver for trace in traces] == ["VER", "VER"]
    assert traces[0].laps == (2, 4, 5)
    assert traces[0].offset_seconds == 0.2
    assert traces[0].color == "#123456"
    assert traces[0].linestyle == "--"
    assert traces[1].laps == (9, 10)
    assert traces[1].offset_seconds == 0.1


def test_build_pure_lap_time_trace_data_uses_collected_laps_and_offsets():
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "VER", "VER", "LEC"],
            "Team": ["Red Bull Racing", "Red Bull Racing", "Red Bull Racing", "Ferrari"],
            "LapNumber": [2, 4, 6, 2],
            "LapTime": pd.to_timedelta([81.2, 80.8, 80.5, 82.0], unit="s"),
        }
    )
    traces = normalize_trace_configs({"VER": {"lap": [2, "4-6"], "off-set": 0.2}})

    plot_data = build_pure_lap_time_trace_data(laps, traces)

    assert plot_data["LapNumber"].tolist() == [2, 4, 6]
    assert plot_data["CollectedLapNumber"].tolist() == [1, 2, 3]
    assert plot_data["PureLapTimeSeconds"].round(3).tolist() == [81.0, 80.6, 80.3]


def test_build_accumulated_delta_trace_data_uses_matching_collected_laps():
    laps = pd.DataFrame(
        {
            "Driver": ["RUS", "RUS", "RUS", "ANT", "ANT", "ANT"],
            "Team": ["Mercedes"] * 6,
            "LapNumber": [37, 38, 39, 38, 39, 40],
            "LapTime": pd.to_timedelta([81.0, 82.0, 80.5, 80.5, 82.5, 80.0], unit="s"),
        }
    )
    traces = normalize_trace_configs(
        {
            "RUS": {"lap": ["37-39"]},
            "ANT": {"lap": ["38-40"]},
        }
    )
    delta_traces = normalize_delta_trace_configs(
        {"RUS vs ANT": {"trace_a": "RUS", "trace_b": "ANT", "lap": ["1-3"]}}
    )

    plot_data = build_pure_lap_time_trace_data(laps, traces)
    delta_data = build_accumulated_delta_trace_data(plot_data, delta_traces)

    assert delta_data["TraceALapNumber"].tolist() == [37, 38, 39]
    assert delta_data["TraceBLapNumber"].tolist() == [38, 39, 40]
    assert delta_data["CollectedLapNumber"].tolist() == [1, 2, 3]
    assert delta_data["LapDeltaSeconds"].tolist() == [0.5, -0.5, 0.5]
    assert delta_data["AccumulatedDeltaSeconds"].tolist() == [0.5, 0.0, 0.5]


def test_plot_pure_lap_time_traces_applies_style_overrides():
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "VER", "LEC", "LEC"],
            "Team": ["Red Bull Racing", "Red Bull Racing", "Ferrari", "Ferrari"],
            "LapNumber": [2, 3, 2, 3],
            "LapTime": pd.to_timedelta([81.2, 80.8, 82.0, 81.5], unit="s"),
        }
    )
    traces = normalize_trace_configs(
        {
            "VER": {"lap": [2, 3], "color": "#123456", "linestyle": "-."},
            "LEC": {"lap": [2, 3]},
        }
    )
    plot_data = build_pure_lap_time_trace_data(laps, traces)

    fig, ax = plot_pure_lap_time_traces(
        plot_data,
        laps=laps,
        trace_configs=traces,
        title="test",
    )

    lines = ax.get_lines()
    assert lines[0].get_color() == "#123456"
    assert lines[0].get_linestyle() == "-."
    assert lines[0].get_xdata().tolist() == [1, 2]
    fig.clear()


def test_plot_pure_lap_time_traces_applies_y_range():
    laps = pd.DataFrame(
        {
            "Driver": ["VER", "VER"],
            "Team": ["Red Bull Racing", "Red Bull Racing"],
            "LapNumber": [2, 3],
            "LapTime": pd.to_timedelta([81.2, 80.8], unit="s"),
        }
    )
    traces = normalize_trace_configs({"VER": {"lap": [2, 3]}})
    plot_data = build_pure_lap_time_trace_data(laps, traces)

    fig, ax = plot_pure_lap_time_traces(
        plot_data,
        laps=laps,
        trace_configs=traces,
        title="test",
        y_range=normalize_y_range([79.5, 84.0]),
    )

    assert ax.get_ylim() == (79.5, 84.0)
    fig.clear()


def test_plot_pure_lap_time_traces_adds_delta_axis():
    laps = pd.DataFrame(
        {
            "Driver": ["RUS", "RUS", "ANT", "ANT"],
            "Team": ["Mercedes", "Mercedes", "Mercedes", "Mercedes"],
            "LapNumber": [38, 39, 38, 39],
            "LapTime": pd.to_timedelta([81.0, 82.0, 80.5, 82.5], unit="s"),
        }
    )
    traces = normalize_trace_configs({"RUS": {"lap": ["38-39"]}})
    delta_traces = normalize_delta_trace_configs(
        {"RUS vs ANT": {"trace_a": "RUS", "trace_b": "ANT", "lap": ["1-2"]}}
    )
    traces = normalize_trace_configs({"RUS": {"lap": ["38-39"]}, "ANT": {"lap": ["38-39"]}})
    plot_data = build_pure_lap_time_trace_data(laps, traces)
    delta_data = build_accumulated_delta_trace_data(plot_data, delta_traces)

    fig, _ = plot_pure_lap_time_traces(
        plot_data,
        laps=laps,
        trace_configs=traces,
        delta_plot_data=delta_data,
        delta_trace_configs=delta_traces,
        title="test",
        delta_y_range=normalize_y_range([-1.0, 1.0]),
    )

    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "Accumulated delta (s)"
    assert fig.axes[1].get_ylim() == (-1.0, 1.0)
    assert fig.axes[1].get_lines()[0].get_color() == "black"
    fig.clear()
