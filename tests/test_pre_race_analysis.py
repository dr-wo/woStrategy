from pathlib import Path

import pytest

from wostrategy.script.pre_race_analysis import (
    SCRIPT_CONFIG,
    build_parser,
    run_from_script_config,
)


def test_parser_defaults_to_editable_script_config():
    args = build_parser().parse_args([])
    assert args.year == SCRIPT_CONFIG["year"]
    assert args.round_number == SCRIPT_CONFIG["round_number"]
    assert tuple(args.sessions) == SCRIPT_CONFIG["sessions"]
    assert args.seed == SCRIPT_CONFIG["primary_seed"]
    assert args.data_root == SCRIPT_CONFIG["data_root"]


def test_direct_config_run_reports_when_no_selected_csv_is_available(tmp_path):
    config = dict(SCRIPT_CONFIG)
    config["sessions"] = ("FP2",)
    config["input_csv_by_session"] = {"FP2": tmp_path / "missing.csv"}
    config["plot_output"] = None
    with pytest.raises(ValueError, match="No valid FP sessions"):
        run_from_script_config(config)
