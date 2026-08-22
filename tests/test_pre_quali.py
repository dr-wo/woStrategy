import pandas as pd

from wostrategy.analysis.pre_quali import analyze_pre_quali


def _laps(pattern, *, driver="AAA", session="FP1", start=0, invalid_push=False, wet=False):
    rows = []
    cursor = float(start)
    for number, (role, duration) in enumerate(pattern, start=1):
        lap_start = pd.Timedelta(seconds=cursor)
        pit_out = lap_start + pd.Timedelta(seconds=10) if role == "OUT" else pd.NaT
        lap_end = lap_start + pd.Timedelta(seconds=duration)
        pit_in = lap_start + pd.Timedelta(seconds=duration - 8) if role == "IN" else pd.NaT
        rows.append({
            "SessionName": session, "Driver": driver, "Team": "Team", "LapNumber": number,
            "LapTime": pd.Timedelta(seconds=duration), "LapStartTime": lap_start,
            "Time": lap_end, "PitOutTime": pit_out, "PitInTime": pit_in,
            "Compound": "INTERMEDIATE" if wet and role == "PUSH" else "SOFT",
            "IsAccurate": not (invalid_push and role == "PUSH"), "Deleted": False,
        })
        cursor += duration
    return rows


def test_exact_two_percent_and_supported_patterns_and_partial_pit_timing():
    rows = _laps([("OUT", 100), ("PUSH", 90), ("IN", 100)], driver="AAA")
    rows += _laps(
        [("OUT", 110), ("PREP", 110), ("PUSH", 91.8), ("COOL", 111),
         ("PUSH", 90.5), ("COOL", 109), ("PUSH", 90.2), ("IN", 105)],
        driver="BBB", start=500,
    )
    result = analyze_pre_quali(pd.DataFrame(rows), year=2026, round_number=1)
    aaa = result.laps.loc[result.laps.driver.eq("AAA")]
    bbb = result.laps.loc[result.laps.driver.eq("BBB")]
    assert aaa.lap_role.tolist() == ["OUT", "PUSH", "IN"]
    assert bbb.lap_role.tolist() == ["OUT", "PREP", "PUSH", "COOL", "PUSH", "COOL", "PUSH", "IN"]
    assert result.laps.is_valid_run.all()
    # Pit exit to lap finish, and lap start to pit entry (not full partial-lap times).
    assert aaa.planning_duration_s.tolist() == [90.0, 90.0, 92.0]


def test_multiple_runs_missing_boundaries_and_unexpected_slow_laps_are_visible():
    rows = _laps([("OUT", 100), ("PUSH", 90), ("IN", 100)], driver="AAA")
    second = _laps([("OUT", 100), ("PREP", 110), ("PREP", 112), ("PUSH", 90), ("IN", 100)], driver="AAA", start=400)
    for offset, row in enumerate(second, start=4): row["LapNumber"] = offset
    missing = _laps([("PUSH", 90), ("IN", 100)], driver="BBB", start=1000)
    result = analyze_pre_quali(pd.DataFrame(rows + second + missing), year=2026, round_number=1)
    aaa = result.laps.loc[result.laps.driver.eq("AAA")]
    assert aaa.run_id.dropna().nunique() == 2
    assert aaa.loc[aaa.run_id.eq("FP1-AAA-1"), "is_valid_run"].all()
    assert not aaa.loc[aaa.run_id.eq("FP1-AAA-2"), "is_valid_run"].any()
    assert "OTHER" in aaa.lap_role.tolist()
    assert "MISSING_OUT" in result.laps.loc[result.laps.driver.eq("BBB"), "invalid_reason"].tolist()


def test_invalid_deleted_and_wet_push_laps_are_excluded_from_averages():
    invalid = _laps([("OUT", 100), ("PUSH", 90), ("IN", 100)], invalid_push=True)
    wet = _laps([("OUT", 100), ("PUSH", 89), ("IN", 100)], driver="BBB", wet=True, start=400)
    deleted = _laps([("OUT", 100), ("PUSH", 88), ("IN", 100)], driver="CCC", start=800)
    deleted[1]["Deleted"] = True
    result = analyze_pre_quali(pd.DataFrame(invalid + wet + deleted), year=2026, round_number=1)
    assert not result.laps.is_valid_for_average.any()
    assert set(result.laps.invalid_reason) >= {"INACCURATE_OR_INVALID", "WET_COMPOUND", "DELETED"}


def test_representative_summary_combines_prep_and_cool_and_uses_overall_weekend():
    rows = _laps([("OUT", 100), ("PREP", 110), ("PUSH", 90), ("COOL", 108), ("PUSH", 89), ("IN", 100)])
    rows += _laps([("OUT", 102), ("PUSH", 88), ("IN", 104)], driver="BBB", session="FP2", start=500)
    result = analyze_pre_quali(pd.DataFrame(rows), year=2026, round_number=1)
    summary = result.durations.set_index("category")
    assert summary.loc["cool_lap_s", "count"] == 2
    assert summary.loc["out_lap_s", "count"] == 2
    assert summary.loc["push_lap_s", "count"] == 3

