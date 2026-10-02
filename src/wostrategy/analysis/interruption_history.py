"""Deterministic historical interruption extraction from original TrackStatus."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real

import pandas as pd

from wodata.interruption_history import (
    ROLE, SCHEMA_VERSION, load_history, load_race_source, same_venue,
    schedule_races, write_history, unavailable_history,
)


CODES = {"1", "2", "4", "5", "6", "7"}


def _seconds(value):
    if value is None or pd.isna(value):
        return None
    try:
        result = float(value) if isinstance(value, Real) else pd.Timedelta(value).total_seconds()
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _timeline(frame):
    if frame is None or not {"Time", "Status"}.issubset(frame.columns):
        return [], True
    output, invalid = [], False
    for index, (_, row) in enumerate(frame.iterrows()):
        timestamp = _seconds(row["Time"])
        if timestamp is None:
            invalid = True
            continue
        output.append((timestamp, index, str(row["Status"])))
    return sorted(output), invalid


def _lap_boundaries(laps, official_start, timing_stream=None):
    """Require a reported leader lap advance and that driver's raw completion.

    FastF1 laps.Position is derived from available completion times, so it does
    not establish authority. Original timing-stream Position and successive
    leader GapToLeader labels (LAP N -> LAP N+1) identify the boundary's driver.
    Their publication times identify authority, never replace raw laps.Time.
    """
    if laps is None or not {"LapNumber", "Time", "DriverNumber"}.issubset(laps.columns):
        return {}
    if timing_stream is None or not {"Time", "Driver", "Position", "GapToLeader"}.issubset(timing_stream.columns):
        return {}
    stream = timing_stream.copy().reset_index(drop=True)
    stream["_seconds"] = stream["Time"].map(_seconds)
    stream = stream.dropna(subset=["_seconds", "Driver"]).sort_values("_seconds", kind="stable")
    previous = stream.groupby("Driver")[["Position", "GapToLeader"]].shift()
    current_lap = pd.to_numeric(
        stream["GapToLeader"].astype(str).str.extract(r"^LAP (\d+)$", expand=False), errors="coerce")
    previous_lap = pd.to_numeric(
        previous["GapToLeader"].astype(str).str.extract(r"^LAP (\d+)$", expand=False), errors="coerce")
    advances = stream.loc[stream["Position"].eq(1) & previous["Position"].eq(1)
                          & current_lap.eq(previous_lap + 1)]
    leaders = {}
    for index, row in advances.iterrows():
        number = int(current_lap.loc[index]) - 1
        if number >= 1:
            # Retain the first reported boundary driver even if its raw
            # completion is missing; a subsequent leader is not a repair.
            leaders.setdefault(number, str(row["Driver"]))
    boundaries = {number: None for number in leaders}
    observed = set()
    for _, row in laps.iterrows():
        generated = row.get("FastF1Generated", False)
        if pd.isna(generated) or bool(generated):
            continue
        number = row["LapNumber"]
        if pd.isna(number) or not isinstance(number, Real) or int(number) != number or number < 1:
            return {}
        end = _seconds(row["Time"])
        number = int(number)
        if str(row["DriverNumber"]) != leaders.get(number):
            continue
        if end is None:
            boundaries[number] = None
            observed.add(number)
            continue
        if number not in observed:
            boundaries[number] = end
            observed.add(number)
        elif boundaries[number] is not None:
            boundaries[number] = min(boundaries[number], end)
    previous = official_start
    for number in sorted(boundaries):
        end = boundaries[number]
        if end is not None:
            if end <= previous:
                return {}
            previous = end
    return boundaries


def _lap_context(timestamp, laps, official_start, timing_stream=None, *, interval_end=False):
    """Entry uses [start, end); SC withdrawal uses (start, end]."""
    unknown = {"lap": None, "status": "unavailable", "method": "raw_leader_lap_boundaries"}
    if timestamp is None:
        return unknown
    boundaries = _lap_boundaries(laps, official_start, timing_stream)
    for number in sorted(boundaries):
        start = official_start if number == 1 else boundaries.get(number - 1)
        end = boundaries[number]
        if start is None or end is None:
            continue
        inside = start < timestamp <= end if interval_end else start <= timestamp < end
        if inside:
            return {"lap": number, "status": "approximate", "method": "raw_leader_lap_boundaries"}
    return unknown


def _message_time(row, source):
    timestamp = _seconds(row.get("SessionTime"))
    if timestamp is None and source.get("t0_date") is not None and row.get("Time") is not None:
        try:
            timestamp = (pd.Timestamp(row["Time"]) - pd.Timestamp(source["t0_date"])).total_seconds()
        except (TypeError, ValueError):
            pass
    return timestamp


def _race_start(source, session, track):
    """Lock the first genuine Started interval; never replace it on restart.

    Started alone is FastF1's initial candidate, not proof against an aborted
    start. A non-generated completed race lap before suspension corroborates it.
    Before such corroboration, only a mapped explicit start-abort permits moving
    to a later Started. An unexplained early suspension remains ambiguous.
    """
    candidate = _seconds(source.get("official_start"))
    starts = sorted({t for t, _, code in session if code == "Started"})
    unknown = {"status": "ambiguous", "evidence": None, "reason": "Original Started evidence missing or conflicting"}
    if candidate is None or not starts or candidate != starts[0]:
        return None, unknown
    aborts = []
    messages = source.get("race_control_messages")
    if messages is not None and "Message" in messages:
        for _, row in messages.iterrows():
            if any(term in str(row["Message"]).upper() for term in ("START ABORTED", "ABORTED START")):
                timestamp = _message_time(row, source)
                if timestamp is None:
                    return None, {**unknown, "reason": "Start-abort evidence has no session-time mapping"}
                aborts.append(timestamp)
    completions = []
    laps = source.get("laps")
    if laps is not None and {"Time", "LapNumber"}.issubset(laps.columns):
        for _, row in laps.iterrows():
            generated = row.get("FastF1Generated", False)
            number = row["LapNumber"]
            timestamp = _seconds(row["Time"])
            if (not pd.isna(generated) and not bool(generated) and
                    isinstance(number, Real) and number >= 1 and int(number) == number and timestamp is not None):
                completions.append(timestamp)
    for index, candidate in enumerate(starts):
        if any(t <= candidate for t in completions):
            # Every reliable earlier completion is evidence of prior racing,
            # including before a surviving Aborted -> Started lifecycle. A
            # later completion of the same lap cannot cancel that conflict.
            return None, {**unknown, "reason": "Completed race laps precede candidate Started; original-start evidence missing"}
        boundary = min((t for t, _, code in session if t > candidate and code != "Started"), default=float("inf"))
        next_start = starts[index + 1] if index + 1 < len(starts) else float("inf")
        track_suspension = min((t for t, _, code in track if code == "5" and candidate <= t < next_start), default=float("inf"))
        limit = min(boundary, next_start, track_suspension)
        abort = min((t for t in aborts if candidate <= t < next_start), default=float("inf"))
        proof = min((t for t in completions if candidate < t < min(limit, abort)), default=float("inf"))
        if math.isfinite(proof):
            return candidate, {"status": "established", "evidence": "non_generated_race_lap_completion",
                               "corroborated_at_seconds": proof,
                               "reason": "Original Started corroborated before suspension; later aborts/restarts do not redefine it"}
        if math.isfinite(abort) and next_start < float("inf"):
            continue
        return None, {**unknown, "reason": "Started not corroborated before suspension; initial abort evidence missing"}
    return None, unknown


def extract_race(record, source=None, error=None):
    """One ordered extraction, with minimum groups across unknown fragments."""
    result = dict(record)
    source = source or {}
    track, invalid = _timeline(source.get("track_status"))
    session, session_invalid = _timeline(source.get("session_status"))
    start, admission = _race_start(source, session, track)
    if session_invalid:
        start = None
        admission = {"status": "ambiguous", "evidence": None, "reason": "Invalid/missing SessionStatus timing; initial lifecycle cannot be established"}
    finishes = [t for t, _, code in session if code in {"Finished", "Finalised", "Ends"}
                and start is not None and t > start]
    finish = min(finishes) if finishes else None
    integrity = source.get("track_status_integrity") or {"status": "unknown", "evidence": None}
    affirmative = (integrity.get("status") == "verified_complete" and
                   isinstance(integrity.get("evidence"), str) and bool(integrity["evidence"].strip()))
    reasons = [str(error)] if error else []
    if not affirmative:
        reasons.append("TrackStatus completeness/integrity not affirmatively established")
    if not track:
        reasons.append("Original TrackStatus unavailable")
    if start is None:
        reasons.append(admission["reason"])
    if finish is None:
        reasons.append("Race-end SessionStatus unavailable")
    if invalid or session_invalid or any(code not in CODES for _, _, code in track):
        reasons.append("Invalid or unknown source transitions")
    baseline = [row for row in track if start is not None and row[0] <= start]
    if not baseline:
        reasons.append("TrackStatus coverage does not include race start")
    if baseline and baseline[-1][0] < start and baseline[-1][2] in {"4", "5"}:
        reasons.append("Pre-start interruption still active at official start; post-start deployment unproven")
    if track and track[0][2] in {"4", "5"} and start is not None and track[0][0] >= start:
        reasons.append("Source begins during an interruption; deployment transition unavailable")
    if source.get("coverage_complete") is False or integrity.get("status") == "partial":
        reasons.append("Source explicitly marked partial")
    result["provenance"] = {
        "source": source.get("source", "fastf1_original_track_status"),
        "timeline_sha256": hashlib.sha256(json.dumps(track).encode()).hexdigest(),
        "lap_source": "raw session elapsed timing-line Time for the driver independently identified by original timing-stream leader lap advances; missing leader completions are not repaired",
        "layout_policy": "venue identity retained; no layout equivalence assertion",
        "source_availability_notes": source.get("source_availability_notes", []),
    }
    events = {"4": [], "5": []}
    previous = None
    active = None
    fragment = None
    sequence = 0
    if start is not None:
        for timestamp, index, code in track:
            if timestamp < start:
                previous = code
                continue
            if finish is not None and timestamp > finish:
                break
            if code == previous:
                continue
            if active is not None and code != active["track_status"]:
                if fragment is not None:
                    if code in CODES and previous in CODES:
                        fragment["end_seconds"] = timestamp
                        fragment["end_lap"] = _lap_context(timestamp, source.get("laps"), start, source.get("timing_stream"), interval_end=True)
                        fragment["end_status"] = "observed_transition"
                        fragment["end_track_status"] = code
                        if active["track_status"] == "4" and fragment["start_seconds"] == timestamp:
                            fragment["start_lap"] = _lap_context(None, None, start)
                            fragment["end_lap"] = _lap_context(None, None, start)
                            # The occurrence entry mirrors its first fragment.
                            if len(active["fragments"]) == 1:
                                active["start_lap"] = fragment["start_lap"]
                    else:
                        fragment["end_status"] = "ambiguous_transition"
                        fragment["uncertain_after_seconds"] = timestamp
                        active["boundary_ambiguous"] = True
                    active.update({key: fragment[key] for key in ("end_seconds", "end_lap", "end_status", "end_track_status")})
                    fragment = None
                if code in CODES:
                    # A trusted differing state proves an intervening exit,
                    # even if its exact time was lost in an unknown interval.
                    active["exit_proof"] = {"track_status": code, "observed_seconds": timestamp, "source_index": index}
                    active = None
            if code in events:
                continuing = active is not None and active["track_status"] == code
                fragment = {
                    "source_index": index, "first_observed_seconds": timestamp,
                    "start_seconds": timestamp if previous in CODES else None,
                    "start_lap": _lap_context(timestamp if previous in CODES else None, source.get("laps"), start, source.get("timing_stream")),
                    "start_status": "observed_transition" if previous in CODES else "start_transition_unavailable",
                    "end_seconds": None, "end_lap": _lap_context(None, None, start),
                    "end_status": "unavailable", "end_track_status": None,
                }
                if not continuing:
                    sequence += 1
                    active = {**fragment, "sequence": sequence, "track_status": code,
                              "boundary_ambiguous": previous not in CODES, "fragments": [], "exit_proof": None}
                    events[code].append(active)
                else:
                    active["boundary_ambiguous"] = True
                active["fragments"].append(fragment)
                active.update({key: fragment[key] for key in ("end_seconds", "end_lap", "end_status", "end_track_status")})
            previous = code
    complete = not reasons
    if complete and active is not None and fragment is not None and finish is not None:
        fragment.update({"end_seconds": finish, "end_status": "race_end",
                         "end_lap": _lap_context(finish, source.get("laps"), start, source.get("timing_stream"), interval_end=True)})
        active.update({key: fragment[key] for key in ("end_seconds", "end_lap", "end_status", "end_track_status")})
    if any(event["start_seconds"] is None or event["boundary_ambiguous"] for family in events.values() for event in family):
        complete = False
        reasons.append("Deployment boundaries or fragment continuity ambiguous")
    result["coverage"] = {
        "status": "complete" if complete else "incomplete" if track else "unavailable",
        "reasons": reasons, "official_start_seconds": start, "race_end_seconds": finish,
        "track_status_integrity": integrity, "start_admission": admission,
        "assessment": "Exact counts require affirmative TrackStatus integrity plus supported race admission and unambiguous deployment boundaries",
    }
    for name, code in (("safety_car", "4"), ("red_flag", "5")):
        occurrences = events[code]
        result[name] = {"occurred": True if occurrences else False if complete else None,
                        "count": len(occurrences) if complete else None,
                        "observed_count": len(occurrences), "occurrences": occurrences}
    return result


def _history_cutoff(target, as_of):
    cutoff = pd.Timestamp(as_of or datetime.now(timezone.utc))
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    target_time = pd.to_datetime(target.get("race_start"), utc=True, errors="coerce")
    return min(cutoff, target_time) if pd.notna(target_time) else cutoff


def _chronology(record, target, cutoff):
    if int(record["season"]) > cutoff.year:
        return "future"
    timestamp = pd.to_datetime(record.get("race_start"), utc=True, errors="coerce")
    if pd.notna(timestamp):
        if timestamp >= cutoff:
            return "future"
        if record.get("race_start_precision") == "date" and timestamp.normalize() == cutoff.normalize():
            return "unknown"
        return "prior"
    if int(record["season"]) in {cutoff.year, int(target["season"])}:
        return "unknown"
    return "prior"


def select_prior_races(target, records, limit=5, *, as_of=None):
    key = (int(target["season"]), int(target["round_number"]))
    unique = {}
    cutoff = _history_cutoff(target, as_of)
    for record in records:
        race_key = (int(record["season"]), int(record["round_number"]))
        if (race_key < key and same_venue(record.get("event_identity"), target["event_identity"])
                and _chronology(record, target, cutoff) == "prior"):
            unique[race_key] = record
    return [unique[key] for key in sorted(unique, reverse=True)[:limit]]


def prepare_history(season, round_number, *, data_root=None, schedule_loader=None, source_loader=None, earliest_season=1950):
    """Explicit acquisition stage, separate from both local-only consumers."""
    if schedule_loader is None:
        import fastf1
        from wodata import get_fastf1_raw_cache_dir
        cache = get_fastf1_raw_cache_dir(data_root)
        cache.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache))
        schedule_loader = fastf1.get_event_schedule
    as_of = datetime.now(timezone.utc)
    current = schedule_races(schedule_loader(int(season), include_testing=False), season)
    target = next(row for row in current if row["round_number"] == int(round_number))
    from wodata.event_operational import has_stable_event_identity
    if not has_stable_event_identity(target["event_identity"]):
        payload = unavailable_history(season, round_number, "Stable target venue identity unavailable")
        payload["target"] = target
        write_history(payload, data_root)
        return payload
    records = list(current)
    failures = []
    for year in range(int(season) - 1, int(earliest_season) - 1, -1):
        if len(select_prior_races(target, records, as_of=as_of)) >= 5:
            break
        try:
            records.extend(schedule_races(schedule_loader(year, include_testing=False), year))
        except Exception as exc:
            failures.append(f"{year} schedule unavailable: {type(exc).__name__}: {exc}")
    cutoff = _history_cutoff(target, as_of)
    for candidate in records:
        if ((candidate["season"], candidate["round_number"]) < (target["season"], target["round_number"])
                and same_venue(candidate["event_identity"], target["event_identity"])
                and _chronology(candidate, target, cutoff) == "unknown"):
            failures.append(f"{candidate['event_id']} chronology unproven; excluded from prior-race selection")
    races = []
    for record in select_prior_races(target, records, as_of=as_of):
        try:
            source = (source_loader(record["season"], record["round_number"]) if source_loader else
                      load_race_source(record["season"], record["round_number"], data_root=data_root))
            race = extract_race(record, source)
        except Exception as exc:
            race = extract_race(record, error=f"{type(exc).__name__}: {exc}")
        races.append(race)
    payload = {"schema_version": SCHEMA_VERSION, "semantic_role": ROLE, "target": target,
               "status": "available" if not failures else "incomplete",
               "selection": {"policy": "latest_five_prior_races_same_venue", "limit": 5,
                             "earliest_season": int(earliest_season), "as_of_utc": as_of.isoformat(), "warnings": failures,
                             "venue_policy": "wodata.event_operational._same_circuit"},
               "generated_at": as_of.isoformat(), "races": races}
    write_history(payload, data_root)
    return payload


def _event_text(family, sc=False):
    if family.get("occurred") is False:
        return "No SC" if sc else "No Red Flag"
    events = family.get("occurrences", [])
    if not events:
        return "No event observed; coverage unproven"
    count = family.get("count")
    label = str(count) if count is not None else f"at least {len(events)} (incomplete)"
    locations = []
    for event in events:
        pieces = []
        for fragment in event["fragments"]:
            start = fragment["start_lap"].get("lap")
            end = fragment["end_lap"].get("lap")
            if start is None:
                text = "entry lap unavailable"
            elif sc and end is not None:
                text = f"~L{start}" if start == end else f"~L{start}–{end}"
            else:
                text = f"~L{start}"
            if sc and fragment["end_status"] not in {"observed_transition", "race_end"}:
                text += " (end unavailable)"
            elif sc and end is None:
                text += " (end lap unavailable)"
            if sc and fragment["end_track_status"] == "5":
                text += " (to Red Flag)"
            pieces.append(text)
        text = "; ".join(pieces)
        if event["boundary_ambiguous"]:
            text += " (ambiguous boundaries/continuity; fragments may be separate deployments)"
        locations.append(text)
    return f"{label}: " + "; ".join(locations)


def history_rows(payload):
    """Shared descriptive presentation values; no consumer-side reconstruction."""
    return [{"edition": f"{race['season']} R{race['round_number']} — {race['event_identity'].get('event_name', race['event_id'])}",
             "safety_car": _event_text(race["safety_car"], True),
             "red_flag": _event_text(race["red_flag"]),
             "coverage": race["coverage"]["status"] +
                         (": " + "; ".join(race["coverage"]["reasons"]) if race["coverage"]["reasons"] else "")}
            for race in payload.get("races", [])]


def history_markdown(payload):
    lines = ["## Historical Safety Car / Red Flag context", "",
             "Latest five prior races at the same venue; pure VSC excluded. Descriptive context only. Lap values are approximate; the SC range covers laps with SC active for any portion; withdrawal exactly at a lap boundary excludes the next lap. Red Flags use their entry lap.", ""]
    rows = history_rows(payload)
    if not rows:
        lines.append(payload.get("reason", "No prior same-venue races found in the inspected schedules."))
    else:
        lines.extend(["| Edition | Safety Car periods / laps | Red Flags / laps | Source coverage |",
                      "| --- | --- | --- | --- |"])
        for row in rows:
            lines.append("| " + " | ".join(str(row[key]).replace("|", "\\|").replace("\n", " ") for key in ("edition", "safety_car", "red_flag", "coverage")) + " |")
    lines.extend(payload.get("selection", {}).get("warnings", []))
    return "\n".join(lines) + "\n"
