from wostrategy.script.race_performance_review import _combine_pit_status


def test_cross_state_pit_stop_is_mixed() -> None:
    assert _combine_pit_status("normal", "vsc") == "mixed"
    assert _combine_pit_status("sc", "normal") == "mixed"
    assert _combine_pit_status("normal", "normal") == "normal"
    assert _combine_pit_status("sc", "sc") == "sc"
    assert _combine_pit_status("vsc", "vsc") == "vsc"
