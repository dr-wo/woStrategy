from wostrategy.script.race_performance_review import _combine_pit_status


def test_cross_state_pit_stop_is_mixed() -> None:
    assert _combine_pit_status("normal", "sc_vsc") == "mixed"
    assert _combine_pit_status("sc_vsc", "normal") == "mixed"
    assert _combine_pit_status("normal", "normal") == "normal"
    assert _combine_pit_status("sc_vsc", "sc_vsc") == "sc_vsc"
