
import pandas as pd
import pytest

from wodata.interruption_history import load_history, write_history
from wostrategy.analysis.interruption_history import (
    extract_race, history_rows, prepare_history, select_prior_races,
)


def record(year=2024, number=1, name='Different GP'):
    return {"season": year, "round_number": number, "event_id": f"{year}-{number:02d}",
            "event_identity": {"country": "UK", "location": "Silverstone", "event_name": name},
            "race_start": f"{year}-01-{number:02d}T12:00:00Z"}


def set_leader_laps(data, frame):
    """Declare independent reported leader advances for these controlled fixtures."""
    frame = frame.copy()
    frame['DriverNumber'] = 'leader'
    data['laps'] = frame
    updates = [(0, 'leader', 1, 'LAP 1')]
    previous_time = 0
    for number, group in frame.groupby('LapNumber', sort=True):
        finite = pd.to_numeric(group['Time'], errors='coerce').dropna()
        # Keep the independently reported counter progression chronological,
        # including fixtures with deliberately inconsistent raw completion Time.
        timestamp = max(previous_time + 1, finite.max() if len(finite) else previous_time + 1)
        updates.append((timestamp, 'leader', 1, f'LAP {int(number) + 1}'))
        previous_time = timestamp
    data['timing_stream'] = pd.DataFrame(updates, columns=['Time', 'Driver', 'Position', 'GapToLeader'])


def source(updates=None, laps=True, **extra):
    updates = updates if updates is not None else [(0, '1'), (20, '4'), (21, '4'), (22, '6'),
        (23, '7'), (24, '1'), (25, '4'), (26, '5'), (27, '5'), (28, '1'), (29, '5'), (30, '1')]
    data = {"track_status": pd.DataFrame(updates, columns=['Time', 'Status']),
            "session_status": pd.DataFrame([(10, 'Started'), (100, 'Finished')], columns=['Time', 'Status']),
            "official_start": 10,
            "track_status_integrity": {"status": "verified_complete", "evidence": "synthetic fixture with known complete transition inventory"}}
    frame = pd.DataFrame({'LapNumber': [1, 2, 3], 'Time': [15, 70, 100]}) if laps else pd.DataFrame({'LapNumber': [1], 'Time': [15]})
    set_leader_laps(data, frame)
    data.update(extra)
    return data


def test_original_status_order_multiplicity_and_vsc_exclusion():
    race = extract_race(record(), source())
    assert race['coverage']['status'] == 'complete'
    assert race['safety_car']['count'] == 2
    assert race['red_flag']['count'] == 2
    assert [e['start_lap']['lap'] for e in race['safety_car']['occurrences']] == [2, 2]
    assert [e['start_seconds'] for e in race['red_flag']['occurrences']] == [26, 29]
    assert [e['sequence'] for e in race['red_flag']['occurrences']] == [3, 4]
    assert race['safety_car']['occurrences'][0]['end_seconds'] == 22


def test_identical_timestamps_keep_source_order():
    race = extract_race(record(), source([(0, '1'), (20, '4'), (20, '1'), (20, '4'), (20, '5')]))
    assert race['safety_car']['count'] == 2
    assert race['red_flag']['count'] == 1


@pytest.mark.parametrize('updates', [[(0, '1')], [(0, '1'), (20, '6'), (30, '7'), (40, '1')]])
def test_confirmed_zero_excludes_pure_vsc(updates):
    race = extract_race(record(), source(updates))
    assert race['safety_car']['count'] == 0
    assert race['red_flag']['occurred'] is False
    assert history_rows({'races': [race]})[0]['safety_car'] == 'No SC'


@pytest.mark.parametrize('data', [None, source([], coverage_complete=False), source([(20, '1')])])
def test_missing_or_partial_is_never_zero(data):
    race = extract_race(record(), data)
    assert race['safety_car']['count'] is None
    assert race['safety_car']['occurred'] is None
    assert race['red_flag']['count'] is None


def test_event_with_unavailable_lap_and_partial_total():
    race = extract_race(record(), source(laps=False, coverage_complete=False))
    assert race['safety_car']['occurred'] is True
    assert race['safety_car']['count'] is None
    assert race['safety_car']['observed_count'] == 2
    assert race['safety_car']['occurrences'][0]['start_lap']['status'] == 'unavailable'
    assert 'at least 2' in history_rows({'races': [race]})[0]['safety_car']


def test_missing_end_and_start_are_retained():
    race = extract_race(record(), source([(20, '4')]))
    event = race['safety_car']['occurrences'][0]
    assert event['start_status'] == 'start_transition_unavailable'
    assert event['start_seconds'] is None
    assert event['first_observed_seconds'] == 20
    assert event['start_lap']['lap'] is None
    assert event['end_status'] == 'unavailable'
    assert race['safety_car']['count'] is None


def test_prestart_and_postfinish_events_excluded():
    race = extract_race(record(), source([(0, '4'), (2, '5'), (3, '1'), (101, '4')]))
    assert race['safety_car']['count'] == 0
    assert race['red_flag']['count'] == 0


def test_prestart_interruption_carry_is_excluded_and_degrades_explicitly():
    race = extract_race(record(), source([(0, '4'), (20, '1')]))
    assert race['safety_car']['count'] is None
    assert race['safety_car']['occurrences'] == []
    assert 'Pre-start interruption' in ';'.join(race['coverage']['reasons'])


def test_official_start_missing_does_not_admit_formation_events():
    race = extract_race(record(), source(official_start=None))
    assert race['safety_car']['occurred'] is None


def test_aborted_start_uses_later_official_start():
    data = source([(0, '1'), (20, '5'), (29, '1'), (50, '4'), (55, '1')])
    data['session_status'] = pd.DataFrame([(10, 'Started'), (20, 'Aborted'), (40, 'Started'), (100, 'Finished')], columns=['Time', 'Status'])
    data['race_control_messages'] = pd.DataFrame({'Message': ['START ABORTED'], 'SessionTime': [20]})
    set_leader_laps(data, pd.DataFrame({'LapNumber': [1, 2, 3], 'Time': [60, 80, 100]}))
    race = extract_race(record(), data)
    assert race['coverage']['official_start_seconds'] == 40
    assert race['red_flag']['count'] == 0
    assert race['safety_car']['count'] == 1


def test_raw_lap_boundaries_keep_suspension_time_and_unknown_gaps():
    data = source([(0, '1'), (69, '5'), (200, '1'), (210, '4'), (211, '1')])
    set_leader_laps(data, pd.DataFrame({'LapNumber': [1, 2, 3], 'Time': [40, 70, 240]}))
    data['session_status'] = pd.DataFrame([(10, 'Started'), (69, 'Aborted'), (200, 'Started'), (240, 'Finished')], columns=['Time', 'Status'])
    race = extract_race(record(), data)
    assert race['red_flag']['occurrences'][0]['start_lap']['lap'] == 2
    assert race['safety_car']['occurrences'][0]['start_lap']['lap'] == 3
    data['laps'] = data['laps'].iloc[[0, 2]]
    assert extract_race(record(), data)['safety_car']['occurrences'][0]['start_lap']['lap'] is None


def test_latest_five_prior_races_skips_years_and_includes_same_season():
    target = record(2026, 9)
    records = [record(2026, 9), record(2026, 10), record(2026, 2), record(2025, 3),
               record(2023, 2), record(2020, 4), record(2020, 3), record(2019, 2)]
    elsewhere = record(2024, 3); elsewhere['event_identity']['location'] = 'Brands Hatch'
    chosen = select_prior_races(target, records + [elsewhere, record(2025, 3)])
    assert [(r['season'], r['round_number']) for r in chosen] == [(2026, 2), (2025, 3), (2023, 2), (2020, 4), (2020, 3)]


def test_prepare_persist_and_load_production_path(tmp_path):
    calls = []
    def schedule(year, **kwargs):
        rows = [(2, 'Silverstone'), (9, 'Silverstone')] if year == 2026 else [(3, 'Silverstone')]
        return pd.DataFrame([{'RoundNumber': n, 'Location': place, 'Country': 'UK', 'EventName': f'GP {year} {n}', 'Session5DateUtc': f'{year}-01-{n:02d}T12:00:00Z'} for n, place in rows])
    def loader(year, number):
        calls.append((year, number))
        if year == 2024:
            raise OSError('not cached')
        return source()
    payload = prepare_history(2026, 9, data_root=tmp_path, schedule_loader=schedule, source_loader=loader)
    assert len(payload['races']) == 5
    assert (2026, 9) not in calls
    assert load_history(2026, 9, tmp_path) == payload
    missing = next(r for r in payload['races'] if r['season'] == 2024)
    assert missing['safety_car']['count'] is None


def test_invalid_source_degrades_and_cannot_create_zero():
    data = source([(0, '1'), (20, 'unknown'), (30, '4'), (40, '1')])
    assert extract_race(record(), data)['safety_car']['count'] is None


def test_future_scheduled_race_does_not_leak_even_before_target_round():
    target = record(2028, 10)
    future = record(2028, 2); future['race_start'] = '2028-03-01T12:00:00Z'
    prior = record(2025, 2); prior['race_start'] = '2025-03-01T12:00:00Z'
    assert select_prior_races(target, [future, prior], as_of='2026-09-30T12:00:00Z') == [prior]


def test_aborted_start_message_without_mapping_degrades_explicitly():
    data = source()
    data['race_control_messages'] = pd.DataFrame({'Message': ['ABORTED START']})
    race = extract_race(record(), data)
    assert race['safety_car']['count'] is None
    assert race['safety_car']['occurred'] is None


def test_aborted_start_utc_mapping_uses_raw_session_clock():
    data = source([(0, '1'), (20, '5'), (29, '1'), (50, '4'), (55, '1')])
    data['session_status'] = pd.DataFrame([(10, 'Started'), (40, 'Started'), (100, 'Finished')], columns=['Time', 'Status'])
    set_leader_laps(data, pd.DataFrame({'LapNumber': [1, 2, 3], 'Time': [60, 80, 100]}))
    data['t0_date'] = pd.Timestamp('2024-01-01T12:00:00')
    data['race_control_messages'] = pd.DataFrame({'Message': ['START ABORTED'], 'Time': [pd.Timestamp('2024-01-01T12:00:20')]})
    race = extract_race(record(), data)
    assert race['coverage']['official_start_seconds'] == 40
    assert race['red_flag']['count'] == 0


def test_missing_finished_status_retains_events_without_exact_totals():
    data = source()
    data['session_status'] = data['session_status'].iloc[:1]
    race = extract_race(record(), data)
    assert race['red_flag']['occurred'] is True
    assert race['red_flag']['count'] is None


def test_missing_venue_identity_degrades_in_preparation(tmp_path):
    def schedule(year, **kwargs):
        return pd.DataFrame([{'RoundNumber': 1, 'EventName': 'Unknown Venue'}])
    payload = prepare_history(2026, 1, data_root=tmp_path, schedule_loader=schedule)
    assert payload['status'] == 'unavailable'
    assert 'venue identity unavailable' in payload['reason']


def test_target_timestamp_and_missing_future_season_dates_cannot_leak():
    target = record(2026, 9); target['race_start'] = '2026-05-01T12:00:00Z'
    later = record(2026, 2); later['race_start'] = '2026-06-01T12:00:00Z'
    assert select_prior_races(target, [later], as_of='2026-09-30T12:00:00Z') == []
    assert select_prior_races(record(2028, 9), [record(2027, 2)], as_of='2026-09-30T12:00:00Z') == []


def test_unknown_generated_lap_flag_does_not_erase_known_events():
    data = source()
    set_leader_laps(data, pd.DataFrame({'LapNumber': [1, 2, 3, 4], 'Time': [15, 40, 70, 100], 'FastF1Generated': [False, pd.NA, pd.NA, pd.NA]}))
    race = extract_race(record(), data)
    assert race['safety_car']['count'] == 2
    assert race['safety_car']['occurrences'][0]['start_lap']['lap'] is None


@pytest.mark.parametrize('updates', [[(0, '1')], [(0, '1'), (20, '6'), (30, '7'), (40, '1')]])
def test_lifecycle_and_baseline_do_not_prove_track_status_integrity(updates):
    data = source(updates, track_status_integrity={'status': 'unknown', 'evidence': None})
    race = extract_race(record(), data)
    assert race['coverage']['status'] == 'incomplete'
    for family in ['safety_car', 'red_flag']:
        assert race[family]['count'] is None
        assert race[family]['observed_count'] == 0
        assert race[family]['occurred'] is None
    assert 'No event observed; coverage unproven' in history_rows({'races': [race]})[0]['safety_car']


def test_legacy_completeness_boolean_without_affirmative_evidence_cannot_prove_zero():
    data = source([(0, '1')], coverage_complete=True)
    del data['track_status_integrity']
    assert extract_race(record(), data)['safety_car']['count'] is None


@pytest.mark.parametrize('code,name', [('4', 'safety_car'), ('5', 'red_flag')])
def test_unknown_gap_creates_fragments_not_a_second_minimum_deployment(code, name):
    race = extract_race(record(), source([(0, '1'), (20, code), (30, '?'), (50, code), (60, '1')]))
    family = race[name]
    assert family['count'] is None
    assert family['observed_count'] == 1
    assert len(family['occurrences']) == 1
    occurrence = family['occurrences'][0]
    assert occurrence['boundary_ambiguous']
    assert len(occurrence['fragments']) == 2
    assert occurrence['fragments'][0]['end_status'] == 'ambiguous_transition'
    assert occurrence['fragments'][1]['start_status'] == 'start_transition_unavailable'
    text = history_rows({'races': [race]})[0][name]
    assert 'at least 1' in text and 'may be separate deployments' in text


@pytest.mark.parametrize('code,name', [('4', 'safety_car'), ('5', 'red_flag')])
def test_trusted_intervening_state_after_unknown_gap_supports_a_new_minimum(code, name):
    race = extract_race(record(), source([(0, '1'), (20, code), (30, '?'), (40, '1'), (50, code), (60, '1')]))
    assert race[name]['observed_count'] == 2
    assert race[name]['occurrences'][0]['exit_proof']['track_status'] == '1'


def test_established_race_is_not_replaced_by_aborted_restart():
    data = source([(0, '1'), (20, '4'), (25, '5'), (90, '1')])
    data['session_status'] = pd.DataFrame([(10, 'Started'), (25, 'Aborted'), (40, 'Started'),
        (50, 'Aborted'), (80, 'Started'), (100, 'Finished')], columns=['Time', 'Status'])
    data['race_control_messages'] = pd.DataFrame({'Message': ['START ABORTED'], 'SessionTime': [50]})
    race = extract_race(record(), data)
    assert race['coverage']['official_start_seconds'] == 10
    assert race['safety_car']['observed_count'] == 1
    assert race['red_flag']['occurrences'][0]['start_seconds'] == 25


def test_missing_original_started_evidence_is_not_replaced_by_a_restart():
    data = source()
    data['session_status'] = pd.DataFrame([(40, 'Started'), (100, 'Finished')], columns=['Time', 'Status'])
    race = extract_race(record(), data)
    assert race['coverage']['official_start_seconds'] is None
    assert race['coverage']['start_admission']['status'] == 'ambiguous'
    assert race['safety_car']['occurred'] is None


@pytest.mark.parametrize('session_rows', [
    [(10, 'Started'), (20, 'Aborted'), (40, 'Started'), (100, 'Finished')],
    [(10, 'Started'), (40, 'Started'), (100, 'Finished')],
])
def test_early_suspension_without_initial_abort_evidence_keeps_admission_ambiguous(session_rows):
    data = source([(0, '1'), (20, '5'), (30, '1'), (50, '4'), (60, '1')])
    set_leader_laps(data, pd.DataFrame({'LapNumber': [1, 2, 3], 'Time': [65, 80, 100]}))
    data['session_status'] = pd.DataFrame(session_rows, columns=['Time', 'Status'])
    race = extract_race(record(), data)
    assert race['coverage']['official_start_seconds'] is None
    assert race['red_flag']['occurred'] is None


@pytest.mark.parametrize('numbers,times', [([1, 2, 3], [15, 70, 60]), ([1, 2, 2, 3], [15, 12, 70, 100])])
def test_non_monotonic_chain_does_not_repair_with_a_backmarker(numbers, times):
    data = source()
    set_leader_laps(data, pd.DataFrame({'LapNumber': numbers, 'Time': times}))
    race = extract_race(record(), data)
    assert race['safety_car']['occurred'] is True
    assert race['safety_car']['occurrences'][0]['start_lap']['lap'] is None


def test_needed_missing_lap_boundary_preserves_event_without_mapping():
    data = source([(0, '1'), (75, '4'), (90, '1')])
    set_leader_laps(data, pd.DataFrame({'LapNumber': [1, 3], 'Time': [40, 100]}))
    race = extract_race(record(), data)
    assert race['safety_car']['observed_count'] == 1
    assert race['safety_car']['occurrences'][0]['start_lap']['lap'] is None


def test_nat_race_session_date_falls_back_and_excludes_future_same_season():
    from wodata.interruption_history import schedule_races
    schedule = pd.DataFrame([{'RoundNumber': 1, 'Country': 'UK', 'Location': 'Silverstone',
        'EventName': 'Future edition', 'Session5DateUtc': pd.NaT, 'EventDate': pd.Timestamp('2026-12-01')}])
    records = schedule_races(schedule, 2026)
    assert records[0]['race_start'].startswith('2026-12-01')
    assert records[0]['race_start_precision'] == 'date'
    target = record(2026, 9); target['race_start'] = '2026-12-31T12:00:00Z'
    assert select_prior_races(target, records, as_of='2026-10-01T12:00:00Z') == []
    records[0]['race_start'] = None
    assert select_prior_races(target, records, as_of='2026-10-01T12:00:00Z') == []


def test_date_only_same_day_does_not_prove_race_preceded_cutoff():
    candidate = record(2026, 1)
    candidate.update(race_start='2026-10-01T00:00:00Z', race_start_precision='date')
    target = record(2026, 9); target['race_start'] = '2026-12-31T12:00:00Z'
    assert select_prior_races(target, [candidate], as_of='2026-10-01T12:00:00Z') == []


def test_sc_ending_exactly_on_completion_does_not_include_next_lap():
    data = source([(0, '1'), (20, '4'), (40, '1'), (41, '5'), (50, '1')])
    set_leader_laps(data, pd.DataFrame({'LapNumber': [1, 2, 3], 'Time': [40, 70, 100]}))
    race = extract_race(record(), data)
    assert race['safety_car']['occurrences'][0]['end_lap']['lap'] == 1
    row = history_rows({'races': [race]})[0]
    assert row['safety_car'] == '1: ~L1'
    assert row['red_flag'] == '1: ~L2'


def test_sc_active_during_parts_of_three_laps_and_direct_red_flag_termination():
    data = source([(0, '1'), (20, '4'), (80, '5'), (90, '1')])
    set_leader_laps(data, pd.DataFrame({'LapNumber': [1, 2, 3], 'Time': [40, 70, 100]}))
    race = extract_race(record(), data)
    assert history_rows({'races': [race]})[0]['safety_car'] == '1: ~L1–3 (to Red Flag)'
    assert race['safety_car']['occurrences'][0]['end_track_status'] == '5'


def test_zero_duration_sc_at_boundary_keeps_event_but_not_an_inverted_range():
    data = source([(0, '1'), (40, '4'), (40, '1')])
    set_leader_laps(data, pd.DataFrame({'LapNumber': [1, 2, 3], 'Time': [40, 70, 100]}))
    race = extract_race(record(), data)
    occurrence = race['safety_car']['occurrences'][0]
    assert race['safety_car']['count'] == 1
    assert occurrence['start_lap']['lap'] is None
    assert occurrence['end_lap']['lap'] is None
    assert 'L2–1' not in history_rows({'races': [race]})[0]['safety_car']


@pytest.mark.parametrize('updates', [
    [(0, '1')],
    [(0, '1'), (20, '4'), (30, '?'), (50, '4'), (60, '1')],
    [(0, '1'), (20, '5'), (30, '?'), (50, '5'), (60, '1')],
    [(0, '1'), (20, '4'), (30, '?'), (40, '5'), (50, '4'), (60, '1')],
    [(0, '1'), (20, '4'), (30, '?'), (40, '1'), (50, '4'), (60, '1')],
    [(0, '1'), (20, '4'), (30, '?')],
    [(0, '1'), (20, '4')],
])
def test_corrected_extraction_survives_canonical_roundtrip(tmp_path, updates):
    from wodata.interruption_history import write_history
    race = extract_race(record(), source(updates))
    target = record(2026, 9)
    payload = {'schema_version': 1, 'semantic_role': 'historical_race_interruptions',
        'status': 'available', 'target': target, 'races': [race]}
    write_history(payload, tmp_path)
    assert load_history(2026, 9, tmp_path) == payload
    if updates == [(0, '1'), (20, '4')]:
        assert race['safety_car']['occurrences'][0]['end_status'] == 'race_end'
        assert 'end unavailable' not in history_rows(payload)[0]['safety_car']


def test_unknown_same_season_chronology_marks_preparation_selection_incomplete(tmp_path):
    def schedule(year, **kwargs):
        if year == 2026:
            return pd.DataFrame([
                {'RoundNumber': 1, 'Country': 'UK', 'Location': 'Silverstone',
                 'Session5DateUtc': pd.NaT, 'EventDate': pd.NaT},
                {'RoundNumber': 9, 'Country': 'UK', 'Location': 'Silverstone',
                 'Session5DateUtc': '2026-10-01T12:00:00Z'}])
        return pd.DataFrame([{'RoundNumber': 1, 'Country': 'UK', 'Location': 'Silverstone',
                             'Session5DateUtc': f'{year}-01-01T12:00:00Z'}])
    payload = prepare_history(2026, 9, data_root=tmp_path, schedule_loader=schedule,
                              source_loader=lambda *args: source(), earliest_season=2021)
    assert len(payload['races']) == 5
    assert all(race['season'] < 2026 for race in payload['races'])
    assert payload['status'] == 'incomplete'
    assert '2026-01 chronology unproven' in payload['selection']['warnings'][0]
    assert load_history(2026, 9, tmp_path) == payload



def test_missing_original_started_cannot_be_hidden_by_fastf1_first_started_candidate():
    data = source()
    data['session_status'] = pd.DataFrame([(40, 'Started'), (100, 'Finished')], columns=['Time', 'Status'])
    # FastF1 would derive session_start_time from this first remaining Started.
    data['official_start'] = 40
    race = extract_race(record(), data)
    assert race['coverage']['start_admission']['status'] == 'ambiguous'
    assert race['coverage']['official_start_seconds'] is None
    assert race['safety_car']['occurred'] is None


def prepare_review_case(root, data):
    """Exercise acquisition, extraction, canonical validation and local readback."""
    from types import SimpleNamespace
    from wodata.interruption_history import load_race_source

    def schedule(year, **kwargs):
        return pd.DataFrame([{'RoundNumber': 17 if year == 2025 else 1,
            'Country': 'UK', 'Location': 'Silverstone',
            'Session5DateUtc': f'{year}-01-01T12:00:00Z'}])

    def acquire(year, number):
        session = SimpleNamespace(track_status=data['track_status'],
            session_status=data['session_status'], laps=data['laps'],
            session_start_time=data['official_start'],
            race_control_messages=pd.DataFrame(), t0_date=None,
            timing_stream=data.get('timing_stream'))
        return load_race_source(year, number, data_root=root,
                                session_loader=lambda *args: session)

    payload = prepare_history(2025, 17, data_root=root, earliest_season=2024,
                              schedule_loader=schedule, source_loader=acquire)
    assert load_history(2025, 17, root) == payload
    return payload['races'][0]


@pytest.mark.parametrize('reverse_l1_rows', [False, True])
def test_prepare_surviving_restart_cannot_cancel_earlier_l1_completion(tmp_path, reverse_l1_rows):
    data = source([(0, '1'), (16, '4'), (20, '5'), (39, '1'), (50, '4'), (65, '1')])
    data['official_start'] = 40
    data['session_status'] = pd.DataFrame([(20, 'Aborted'), (40, 'Started'), (100, 'Finished')],
                                         columns=['Time', 'Status'])
    first_lap = [60, 15] if reverse_l1_rows else [15, 60]
    data['laps'] = pd.DataFrame({'LapNumber': [1, 1, 2], 'Time': first_lap + [70],
        'DriverNumber': (['backmarker', 'leader'] if reverse_l1_rows else ['leader', 'backmarker']) + ['leader'],
        'FastF1Generated': [False, False, False]})
    race = prepare_review_case(tmp_path, data)
    assert race['coverage']['start_admission']['status'] == 'ambiguous'
    assert race['coverage']['official_start_seconds'] is None
    assert 'Completed race laps precede' in race['coverage']['start_admission']['reason']
    assert race['safety_car']['occurred'] is None
    assert race['red_flag']['occurred'] is None


@pytest.mark.parametrize('missing_leader_row', [True, False])
def test_prepare_missing_leader_l1_does_not_use_remaining_backmarker(tmp_path, missing_leader_row):
    data = source([(0, '1'), (50, '4'), (65, '1')])
    rows = [(1, 60, 'backmarker', False, 1), (2, 70, 'leader', False, 1)]
    if not missing_leader_row:
        rows.insert(0, (1, None, 'leader', False, 2))
    # Position is FastF1's ranking of remaining completion rows, not authority.
    data['laps'] = pd.DataFrame(rows,
        columns=['LapNumber', 'Time', 'DriverNumber', 'FastF1Generated', 'Position'])
    data['timing_stream'] = pd.DataFrame([
        (0, 'leader', 1, 'LAP 1'), (40, 'leader', 1, 'LAP 2'),
        (70, 'leader', 1, 'LAP 3'), (0, 'backmarker', 2, '+0.0'),
        (60, 'backmarker', 2, '+20.0'),
    ], columns=['Time', 'Driver', 'Position', 'GapToLeader'])
    race = prepare_review_case(tmp_path, data)
    assert race['safety_car']['occurred'] is True
    assert race['safety_car']['observed_count'] == 1
    event = race['safety_car']['occurrences'][0]
    assert event['start_seconds'] == 50 and event['end_seconds'] == 65
    assert event['start_lap']['lap'] is None
    assert event['end_lap']['lap'] is None


def test_prepare_computed_position_alone_does_not_certify_leader_boundary(tmp_path):
    data = source([(0, '1'), (50, '4'), (65, '1')])
    data['laps'] = pd.DataFrame({'LapNumber': [1, 2], 'Time': [60, 70],
        'DriverNumber': ['backmarker', 'leader'], 'Position': [1, 1]})
    data['timing_stream'] = None
    race = prepare_review_case(tmp_path, data)
    assert race['safety_car']['observed_count'] == 1
    event = race['safety_car']['occurrences'][0]
    assert event['start_lap']['lap'] is None and event['end_lap']['lap'] is None


def test_prepare_independent_reported_leader_can_supply_boundary(tmp_path):
    data = source([(0, '1'), (50, '4'), (65, '1')])
    data['laps'] = pd.DataFrame({'LapNumber': [1, 2], 'Time': [60, 70],
        'DriverNumber': ['driver_a', 'driver_b'], 'Position': [1, 1]})
    # Eligibility follows independently reported leadership at the lap advance,
    # including a different driver for the next boundary.
    data['timing_stream'] = pd.DataFrame([
        (0, 'driver_a', 1, 'LAP 1'), (60, 'driver_a', 1, 'LAP 2'),
        (65, 'driver_b', 1, 'LAP 2'), (70, 'driver_b', 1, 'LAP 3'),
    ], columns=['Time', 'Driver', 'Position', 'GapToLeader'])
    race = prepare_review_case(tmp_path, data)
    event = race['safety_car']['occurrences'][0]
    assert event['start_lap']['lap'] == 1 and event['end_lap']['lap'] == 2


def test_prepare_later_leader_advance_cannot_repair_missing_original_boundary(tmp_path):
    data = source([(0, '1'), (50, '4'), (65, '1')])
    data['laps'] = pd.DataFrame({'LapNumber': [1, 2], 'Time': [60, 70],
        'DriverNumber': ['driver_b', 'driver_a']})
    data['timing_stream'] = pd.DataFrame([
        (0, 'driver_a', 1, 'LAP 1'), (40, 'driver_a', 1, 'LAP 2'),
        (70, 'driver_a', 1, 'LAP 3'),
        (45, 'driver_b', 1, 'LAP 1'), (60, 'driver_b', 1, 'LAP 2'),
    ], columns=['Time', 'Driver', 'Position', 'GapToLeader'])
    race = prepare_review_case(tmp_path, data)
    event = race['safety_car']['occurrences'][0]
    assert race['safety_car']['observed_count'] == 1
    assert event['start_lap']['lap'] is None and event['end_lap']['lap'] is None
