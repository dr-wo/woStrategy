"""Eight-year descriptive history using the fixed circuit-traffic calculation.

Preparation is explicit. Reports consume the persisted result without fitting.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from wodata.contracts import canonical_hash
from wodata.circuit_traffic import read_circuit_traffic, write_circuit_traffic
from . import _circuit_traffic_core as core

CALCULATION_VERSION = 'circuit-traffic-local-residual-v1'
GROUPING_VERSION = 'preceding-eight-calendar-years-v1'
COST_EDGES = [-np.inf,-10,-5,-3,-2,-1,-.5,0,.5,1,2,3,5,10,np.inf]
ADVANTAGE_EDGES = [-np.inf,-3,-2,-1.5,-1,-.5,0,.5,1,1.5,2,3,np.inf]
GROUP_LABELS = {'2022_2025': '2022–2025 within window', 'other_years': 'Other years within window', 'all': 'Complete eight-year window'}


def code_identity():
    files = [Path(__file__), Path(core.__file__), Path(__file__).with_name('circuit_traffic_source.py'),
             Path(__file__).parents[1] / 'core/session_loader.py',
             Path(__file__).parents[1] / 'core/session.py']
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def recipe_id():
    return canonical_hash({'code': code_identity(), 'calculation': CALCULATION_VERSION,
                           'grouping': GROUPING_VERSION, 'contract': core.CONTRACT, 'reference': core.REF_RULE})


def json_records(frame):
    # pandas' JSON conversion handles NaT, nullable scalars, numpy values and NaN.
    return json.loads(frame.to_json(orient='records', double_precision=15))


def _valid(row, kind, column):
    if row.get('kind') != kind or row.get('identifiable') is not True or row.get('unresolved_compound_interaction'):
        return False
    flag = 'advantage_usable' if column == 'advantage_s' else column.replace('_s','_usable')
    value = row.get(column)
    return row.get(flag) is True and value is not None and np.isfinite(value)


def _distribution(records, name, kind, column, title):
    candidates = [r for r in records if r.get('kind') == kind]
    usable = [r for r in candidates if _valid(r, kind, column)]
    values = [float(r[column]) for r in usable]
    edges = ADVANTAGE_EDGES if column == 'advantage_s' else COST_EDGES
    counts = np.histogram(values, bins=edges)[0].tolist()
    blues = [r for r in usable if r.get('evidenced_blue_flag') is True]
    exclusions = Counter()
    for r in candidates:
        if _valid(r, kind, column): continue
        reason = r.get('advantage_exclusion' if column == 'advantage_s' else column.replace('_s','_exclusion'))
        exclusions[reason or 'missing_or_ineligible_value'] += 1
    extra_keys = ['a_ref_age_extrapolated','b_ref_age_extrapolated'] if column=='advantage_s' else [column[0]+'_ref_age_extrapolated']
    result = dict(output=name, title=title, status='available' if values else 'unavailable', usable_N=len(values),
        mean_s=float(np.mean(values)) if values else None,
        identifiable_interactions=sum(r.get('identifiable') is True for r in candidates),
        candidate_pairings=len(candidates), excluded_N=len(candidates)-len(values), exclusions=dict(exclusions),
        identifiable_excluded_N=sum(r.get('identifiable') is True for r in candidates)-len(values),
        negative_N=sum(v<0 for v in values),
        age_extrapolated_N=sum(any(r.get(k) is True for k in extra_keys) for r in usable),
        histogram=dict(edges_s=[None if not np.isfinite(v) else v for v in edges],
            edge_semantics='null first/last = negative/positive infinity; left closed, right open; final bin right closed', counts=counts),
        usable_interaction_ids=[r['interaction_id'] for r in usable])
    if kind == 'lapping':
        result['blue_evidence'] = {
            'evidenced': {'N':len(blues), 'mean_s':float(np.mean([r[column] for r in blues])) if blues else None,
                          'counts':np.histogram([r[column] for r in blues],bins=edges)[0].tolist()},
            'inferred_only': {'N':len(usable)-len(blues), 'mean_s':float(np.mean([r[column] for r in usable if r not in blues])) if len(usable)>len(blues) else None}}
    if column == 'advantage_s':
        minimum = min(values) if values else None
        result.update(observed_successful_pass_minimum_advantage_s=minimum,
            observed_successful_pass_median_advantage_s=float(np.median(values)) if values else None,
            minimum_source_interaction_ids=[r['interaction_id'] for r in usable if r[column]==minimum],
            median_semantics='sorted record values; arithmetic mean of the two middle values for even N',
            interpretation='Observed timing-corroborated successful-pass sample; not a physical minimum or 50% passing-success threshold')
    return result


def summarize_records(*, circuit, target_season, records, coverage, provenance=None, reference_screen=None):
    years = list(range(int(target_season)-8, int(target_season)))
    # Reject, rather than silently incorporating or relabelling outside-window input.
    if any(int(r['year']) not in years for r in records + coverage):
        raise ValueError('Input outside the eight-year window (target year excluded)')
    if len({r['interaction_id'] for r in records}) != len(records):
        raise ValueError('Duplicate interaction IDs')
    coverage = list(coverage)
    for year in years:
        if not any(int(r['year'])==year for r in coverage):
            coverage.append(dict(year=year,status='unavailable',reason='no supplied session/data'))
    available = sorted({int(r['year']) for r in coverage if r['status']=='available'})
    available_set=set(available)
    if any(int(r['year']) not in available_set for r in records):
        raise ValueError('Interaction records lack available-year coverage')
    grouped_years={'2022_2025':[y for y in years if 2022<=y<=2025],
                   'other_years':[y for y in years if not 2022<=y<=2025], 'all':years}
    def section(selected):
        subset=[r for r in records if int(r['year']) in selected]
        outputs=[_distribution(subset,*spec) for spec in core.OUTPUTS]
        return dict(years=selected, available_years=[y for y in selected if y in available],
            unavailable_years=[y for y in selected if y not in available],
            coverage=[c for c in coverage if int(c['year']) in selected],
            outputs=outputs)
    groups=[dict(group=key,label=GROUP_LABELS[key],**section(selected)) for key,selected in grouped_years.items()]
    yearly=[dict(year=y,**section([y])) for y in years]
    has_output=any(o['usable_N'] for o in groups[-1]['outputs'])
    complete=all(o['usable_N'] for o in groups[-1]['outputs']) and len(available)==8 and all(c['status']=='available' for c in coverage)
    return dict(circuit=circuit,target_season=int(target_season),years=years,
        status='available' if complete else 'partial' if has_output else 'unavailable',
        available_years=available,unavailable_years=[y for y in years if y not in available],
        calculation_version=CALCULATION_VERSION,grouping_version=GROUPING_VERSION,recipe_id=recipe_id(),
        code_identity=code_identity(),contract=core.CONTRACT,reference_rule=core.REF_RULE,
        provenance=provenance or {},coverage=coverage,groups=groups,yearly=yearly,
        interactions=records,reference_source_screen=reference_screen or [],
        semantics={'pooled_means':'interaction weighted, calculated from individual records',
            'residual':'signed observed whole-lap time minus retrospective model reference',
            'reference':'team pooled; not driver-specific free-air truth; later evidence permitted',
            'successful_pass':'timing-corroborated order reversal, not independently verified competitive intent',
            'blue':'assigned lapped-car BLUE message context, not unique attribution to this lapper',
            'age':'StintLapNumber minus one; nonzero TyreLife offset unsupported; extrapolation retained',
            'causal':False,'operational_input':False})


def load_cached_history(*, circuit, target_season, data_root=None):
    return read_circuit_traffic(circuit=circuit,target_season=target_season,data_root=data_root,expected_recipe_id=recipe_id())


def prepare_history(*, circuit, target_season, data_root=None, source_loader, refresh=False):
    """Explicit preparation; valid saved analysis bypasses loading and fitting."""
    if not refresh:
        try:
            result,path=load_cached_history(circuit=circuit,target_season=target_season,data_root=data_root)
            for session in result['provenance'].get('input_sessions', []):
                for source, expected in session.get('files', {}).items():
                    if Path(source).exists() and hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
                        raise ValueError('Historical timing input changed; cached analysis is stale')
            return result,path,True
        except (FileNotFoundError,ValueError,KeyError):
            pass
    records=[];screens=[];coverage=[];inputs=[]
    for year in range(int(target_season)-8,int(target_season)):
        try:
            sessions=source_loader(year)
        except Exception as exc:
            coverage.append(dict(year=year,status='unavailable',reason=f'{type(exc).__name__}: {exc}'))
            continue
        if not sessions:
            coverage.append(dict(year=year,status='no_matching_race',reason='No matching circuit race in calendar'))
        for session in sessions:
            entry=dict(session['coverage'],year=year)
            inputs.append(session.get('identity',{}))
            if entry.get('status')!='available':
                coverage.append(entry);continue
            try:
                ledger,screen,lap_counts=core.analyse_session(session['laps'],session['race_control'],year=year,round_number=int(entry['round']))
                records.extend(json_records(ledger));screens.extend(json_records(screen))
                entry['lap_coverage']=lap_counts
            except Exception as exc:
                entry.update(status='analysis_unavailable',reason=f'{type(exc).__name__}: {exc}')
            coverage.append(entry)
    result=summarize_records(circuit=circuit,target_season=target_season,records=records,coverage=coverage,
        provenance={'input_sessions':inputs,'historical_as_of':False,'future_reference_evidence':True,
                    'numpy_version':np.__version__, 'pandas_version':pd.__version__},reference_screen=screens)
    path=write_circuit_traffic(result,data_root=data_root)
    return load_cached_history(circuit=circuit,target_season=target_season,data_root=data_root)[0],path,False
