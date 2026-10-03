"""Explicit historical input preparation through the standard laps loader."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from wodata.circuit_traffic import (circuit_traffic_source_root,
    read_circuit_traffic_session, write_circuit_traffic_session)


class HistoricalTrafficSource:
    def __init__(self, *, circuit, data_root=None, seed_history=None):
        self.circuit=circuit
        self.data_root=data_root
        self.root=circuit_traffic_source_root(circuit=circuit,data_root=data_root)
        self.seed=Path(seed_history) if seed_history else None
        if self.seed:
            config=json.loads((self.seed/'parameters.json').read_text())
            if config['circuit'].strip().casefold()!=circuit.strip().casefold():
                raise ValueError('Seed history belongs to another circuit')
        self.seed_coverage=pd.read_csv(self.seed/'yearly_coverage.csv') if self.seed else pd.DataFrame()

    def __call__(self, year):
        # Session exports are reused without even loading a schedule or raw cache.
        saved=sorted((self.root/f'year={year}').glob('round=*/metadata.json'))
        if saved:
            return [read_circuit_traffic_session(p.parent) for p in saved]
        if self.seed is not None:
            rows=self.seed_coverage.loc[self.seed_coverage.year.eq(year)&self.seed_coverage.status.eq('available')]
            if len(rows):
                sessions=[]
                ids=json.loads((self.seed/'input_identities.json').read_text())
                import hashlib
                for row in rows.itertuples():
                    folder=self.seed/'sessions'/row.tag
                    for p in [folder/'laps.pkl',folder/'race_control.csv',folder/'metadata.json']:
                        if hashlib.sha256(p.read_bytes()).hexdigest()!=ids[str(p.resolve())]:
                            raise ValueError(f'Seed input hash mismatch: {p}')
                    metadata=dict(year=year,round=int(row.round),status='available',event=row.event,location=row.location,
                                  input_source='validated frozen circuit summary inputs',seed_path=str(folder.resolve()))
                    dest=write_circuit_traffic_session(circuit=self.circuit,year=year,round_number=int(row.round),
                        laps=pd.read_pickle(folder/'laps.pkl'),race_control=pd.read_csv(folder/'race_control.csv'),
                        metadata=metadata,data_root=self.data_root)
                    sessions.append(read_circuit_traffic_session(dest))
                return sessions
        import fastf1
        from fastf1 import _api
        from fastf1.utils import to_timedelta
        from wostrategy.core.session_loader import load_session_laps
        from wostrategy.core.session import Session
        cache=self.root/'raw_cache';cache.mkdir(parents=True,exist_ok=True)
        fastf1.Cache.enable_cache(str(cache))
        schedule=fastf1.get_event_schedule(year,include_testing=False,backend='fastf1')
        # Circuit identity is an exact Location match; never a country/event-name wildcard.
        matched=schedule.loc[schedule.Location.astype(str).str.strip().str.casefold().eq(self.circuit.strip().casefold())]
        sessions=[]
        for _,event in matched.iterrows():
            rnd=int(event.RoundNumber)
            metadata=dict(year=year,round=rnd,event=str(event.EventName),location=str(event.Location),status='available',
                fastf1_version=fastf1.__version__,session='R',telemetry=False,
                race_control_clock='original stream relative session timestamp; UTC preserved separately',
                input_source='standard load_session_laps; retained-record Session._add_stint_lap_number')
            try:
                captured=[]
                def factory(round_number,session_name):
                    session=fastf1.get_session(year,round_number,session_name,backend='fastf1')
                    session.load(laps=True,telemetry=False,weather=True,messages=True)
                    Session._add_stint_lap_number(SimpleNamespace(laps=session.laps))
                    captured.append(session)
                    return session
                laps=load_session_laps(year=year,rounds=[rnd],session_names=['R'],session_factory=factory)
                if laps.empty:raise ValueError('standard loader returned no laps')
                records=[]
                for frame in (_api.fetch_page(captured[0].api_path,'race_control_messages') or []):
                    messages=frame[1].get('Messages',{})
                    items=messages.items() if isinstance(messages,dict) else enumerate(messages)
                    for message_id,message in items:
                        records.append(dict(stream_message_id=str(message_id),session_time_s=to_timedelta(frame[0]).total_seconds(),**message))
                rc=pd.DataFrame(records,columns=None if records else ['session_time_s','Flag','RacingNumber','Message'])
                if len(rc):rc=rc.drop_duplicates('stream_message_id',keep='first')
                metadata['race_control_available']=not rc.empty
                dest=write_circuit_traffic_session(circuit=self.circuit,year=year,round_number=rnd,laps=laps,race_control=rc,
                    metadata=metadata,data_root=self.data_root)
                sessions.append(read_circuit_traffic_session(dest))
            except Exception as exc:
                sessions.append({'coverage':{**metadata,'status':'session_unavailable','reason':f'{type(exc).__name__}: {exc}'}})
        return sessions
