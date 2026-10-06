"""Reviewed source-data repairs, applied only when persisting clean laps.

This overlay does not participate in fitting, alter raw timing caches, or infer
physical tyre life from model age. Evidence and unresolved rows remain explicit.
"""
from __future__ import annotations

import hashlib
import json
import csv
import io
from pathlib import Path

import pandas as pd


REPAIR_DATA = Path(__file__).with_name("known_clean_lap_repairs.json")


def repair_clean_laps(frame: pd.DataFrame, *, year: int, race: int, session: str):
    """Return a copy and audit; preserve every existing non-null TyreLife."""
    raw = REPAIR_DATA.read_bytes()
    data = json.loads(raw)
    if data["schema_version"] != 1:
        raise ValueError("Unsupported known clean-lap repair schema")
    repaired = frame.copy(deep=True)
    audits = []
    for repair in data["repairs"]:
        event = repair["event"]
        if (int(year), int(race), str(session).upper()) != (
            event["year"], event["round"], event["session"]
        ):
            continue
        entries = []
        for row in repair["corrections"] + repair["unresolved"]:
            required = {"Driver", "LapNumber", "DriverNumber", "Stint", "Compound",
                        "EventName", "EventCountry", "Year", "Round", "SessionName", "TyreLife"}
            if not required.issubset(repaired.columns):
                status = "schema_mismatch"
                selected = repaired.iloc[0:0]
            else:
                mask = repaired.Driver.eq(row["driver"]) & repaired.LapNumber.eq(row["lap_number"])
                selected = repaired.loc[mask]
                status = "row_absent" if selected.empty else "identity_mismatch"
                if len(selected) == 1:
                    current = selected.iloc[0]
                    number = str(current.DriverNumber).removesuffix(".0")
                    matches = (number == row["driver_number"]
                               and current.Stint == row["stint"]
                               and current.Compound == row["compound"]
                               and current.EventName == event["event_name"]
                               and current.EventCountry == event["event_country"]
                               and current.Year == year and current.Round == race
                               and current.SessionName == event["session_name"])
                    if matches:
                        if pd.notna(current.TyreLife):
                            status = "existing_non_null_preserved"
                        elif "tyre_life" not in row:
                            status = "unresolved"
                        else:
                            repaired.loc[mask, "TyreLife"] = row["tyre_life"]
                            status = "applied"
            entries.append({**row, "status": status})
        audits.append(dict(repair_id=repair["repair_id"], event=event,
                           sources=repair["sources"], rows=entries))
    audit = dict(schema_version=1, repair_data_sha256=hashlib.sha256(raw).hexdigest(),
                 boundary="producer -> persisted clean laps; null-only physical TyreLife",
                 applied_rows=sum(row["status"] == "applied" for item in audits
                                  for row in item["rows"]), repairs=audits)
    return repaired, audit


def persist_clean_laps(frame: pd.DataFrame, path: Path, *, year: int, race: int,
                       session: str, cached: bool = False):
    """Persist repaired data plus source/derivation audit, without mutating input.

    Cached producer reuse materializes only reviewed missing source values;
    unrelated events and already repaired CSVs are not rewritten.
    """
    repaired, audit = repair_clean_laps(frame, year=year, race=race, session=session)
    path = Path(path)
    paths = []
    if not cached:
        repaired.to_csv(path, index=False)
    elif audit["applied_rows"]:
        # Keep existing persisted tokens rather than reserializing unrelated
        # floats through pandas. Only null TyreLife cells receive new tokens.
        records = list(csv.reader(io.StringIO(path.read_text())))
        column = records[0].index("TyreLife")
        driver_column = records[0].index("Driver")
        lap_column = records[0].index("LapNumber")
        changed = frame.TyreLife.isna() & repaired.TyreLife.notna()
        if len(records) - 1 != len(frame):
            raise ValueError("Cached clean-lap row count changed before persistence")
        for position in range(len(frame)):
            if changed.iloc[position]:
                raw = records[position + 1]
                if (raw[driver_column] != str(frame.iloc[position].Driver)
                        or float(raw[lap_column]) != frame.iloc[position].LapNumber):
                    raise ValueError("Cached clean-lap identity changed before persistence")
                if raw[column].strip():
                    raise ValueError("Refusing to overwrite non-empty cached TyreLife token")
                records[position + 1][column] = str(int(repaired.iloc[position].TyreLife))
        output = io.StringIO()
        csv.writer(output, lineterminator="\n").writerows(records)
        path.write_text(output.getvalue())
    if audit["repairs"]:
        audit_path = path.with_suffix(".repairs.json")
        # Keep the original applied audit on idempotent cache reads.
        if not cached or audit["applied_rows"] or not audit_path.exists():
            audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        paths.append(audit_path)
    return paths
