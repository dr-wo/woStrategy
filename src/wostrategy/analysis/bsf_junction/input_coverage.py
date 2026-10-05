"""Provider input screening and reporting; outside the frozen numerical kernel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._transfer import COLS


def complete_clean_laps(clean):
    """Exclude incomplete rows and normalize valid numeric values in memory.

    Kernel row IDs are positions in the retained frame. Preserve their mapping
    to zero-based persisted file positions for score-term audits.
    Structural/kernel errors (roster, duplicates, noninteger ages) still fail.
    """
    required = COLS + ['LapTimeSeconds', 'FuelProxyLapsRemaining']
    missing_columns = sorted(set(required) - set(clean.columns))
    if missing_columns:
        raise ValueError('BSF clean-lap schema missing required covariates: '
                         + ', '.join(missing_columns))
    incomplete = clean[required].isna().copy()
    numeric = ['Stint', 'LapNumber', 'TyreLife', 'TyreAgeLaps',
               'LapTimeSeconds', 'FuelProxyLapsRemaining']
    parsed_numeric = {field: pd.to_numeric(clean[field], errors='coerce') for field in numeric}
    for field, values in parsed_numeric.items():
        incomplete[field] |= ~np.isfinite(values)
    for field in ['Team', 'Driver', 'Compound', 'FreshTyre']:
        incomplete[field] |= clean[field].astype(str).str.strip().eq('')
    clock = pd.to_timedelta(clean.LapStartTime, errors='coerce').dt.total_seconds()
    incomplete['LapStartTime'] |= ~np.isfinite(clock)
    excluded = incomplete.any(axis=1)
    counts = {field: int(n) for field, n in incomplete.sum().items() if n}
    # Reporting backlog (non-blocking): report the exact detected exclusion
    # category instead of listing missing, non-finite, or unparseable possibilities.
    coverage = dict(
        original_rows=len(clean), retained_rows=int((~excluded).sum()),
        excluded_rows=int(excluded.sum()),
        affected_drivers=sorted(clean.loc[excluded, 'Driver'].dropna().astype(str).unique()),
        incomplete_covariates=counts,
        reasons=[('missing, non-finite, or unparseable physical `TyreLife`' if field == 'TyreLife'
                  else f'missing, non-finite, or unparseable `{field}`') for field in counts],
        excluded_clean_laps=[dict(
            source_row=int(i), driver=row.Driver, team=row.Team, lap_number=row.LapNumber,
            incomplete_covariates=incomplete.columns[incomplete.loc[i]].tolist(),
        ) for i, row in clean.loc[excluded].iterrows()],
        retained_source_rows=clean.index[~excluded].tolist(),
        source_row_convention='zero-based data row in persisted clean-lap CSV',
        policy='exclude incomplete required covariates only; no inference or substitution',
    )
    if coverage['retained_rows'] == 0:
        raise ValueError(f'No usable canonical clean-lap inputs after excluding '
                         f'{coverage["excluded_rows"]}/{coverage["original_rows"]} '
                         'incomplete rows; BSF unavailable, no substitute')
    retained = clean.loc[~excluded].copy()
    for field, values in parsed_numeric.items():
        # Replace the column, not individual cells: a malformed CSV token can
        # make every valid value in the original column an object/string.
        # Keep fractional values intact for the frozen integer checks to reject.
        retained[field] = values.loc[~excluded]
    return retained.reset_index(drop=True), coverage


def graph_support_summary(table):
    return dict(
        provenance={flag: int(table.support_flag.eq(flag).sum()) for flag in ['P', 'F', 'N']},
        projected_teams=sorted(table.loc[table.support_flag.eq('P'), 'team'].tolist()),
        disconnected_components=int(table.graph_components.iloc[0]),
        junction_transfers=int(table.available_transfer_edge_count.sum()),
    )


def degraded_warning(round_number, coverage, support):
    flags = support['provenance']
    projected = support['projected_teams']
    projection = ('Only ' + ' and '.join(projected) + ' have projected support.'
                  if projected else 'No teams have projected support.')
    return (
        f'R{round_number}: DEGRADED INPUT — excluded {coverage["excluded_rows"]}/'
        f'{coverage["original_rows"]} clean-lap rows; retained {coverage["retained_rows"]}. '
        f'Affected drivers: {", ".join(coverage["affected_drivers"]) or "unknown"}. '
        f'Reason: {"; ".join(coverage["reasons"])}. '
        f'Resulting provenance: {flags["P"]} P / {flags["F"]} F / {flags["N"]} N; '
        f'{support["disconnected_components"]} graph components; '
        f'{support["junction_transfers"]} junction transfers. {projection} '
        'This event has reduced evidence coverage and should not be interpreted '
        'as a fully supported graph estimate.'
    )
