"""Read persisted race-review inputs; write separate deterministic BSF artifacts."""
from __future__ import annotations

from pathlib import Path
import gzip
import hashlib
import json
import io

import numpy as np
import pandas as pd
from wodata import get_data_root, get_race_performance_review_event_dir

from .constants import TEAMS, METHOD, VERSION, TITLE, METHODOLOGY, LIMITATION, CAPTION
from ._canonical import CanonicalBank
from .reference import calculate_bsf_reference, mercedes_relative_history
from .contract import (
    require_approved_kernel, validate_artifact_kernel,
    frozen_round_contract, validate_frozen_source_identity,
)
from .frozen_sources import resolve_frozen_sources, verify_frozen_sources
from .input_coverage import complete_clean_laps, graph_support_summary, degraded_warning


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_source_paths(*, season, round_number, data_root=None):
    root = get_race_performance_review_event_dir(
        year=season, round_number=round_number, session='R', data_root=data_root)
    prefix = f'race_performance_{season}_{round_number}_R'
    return {key: root / f'{prefix}_{suffix}' for key, suffix in {
        'clean': 'clean_laps.csv', 'parameters': 'sample_parameters.csv',
        'delta': 'compound_delta.csv', 'team_deg': 'team_compound_degradation.csv',
        'metadata': 'metadata.json',
    }.items()}


def canonical_reference_inputs(d, sources):
    """Exact frozen full-fit reference functional, without modifying its inputs."""
    bank = CanonicalBank(d, sources)
    fit = bank.fit()
    field = float(np.nanmean(fit.point_base))
    base = np.where(np.isfinite(fit.point_base), fit.point_base, field)
    prediction = fit.predict_rows(np.arange(len(d)))
    for j in np.flatnonzero(~np.isfinite(prediction)):
        x = d.iloc[j]
        effect = (fit.fuel * (x.FuelProxyLapsRemaining - bank.fuel_ref[0])
                  + fit.track * (x.LapNumber - bank.race_ref[0])
                  + fit.deg[(x.Team, x.Compound)] * (x.TyreAgeLaps - bank.age_ref[0])
                  + fit.delta[x.Compound])
        prediction[j] = field + effect
    return prediction, base, fit.counts


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _artifact_paths(season, round_number, data_root, artifact_root):
    base_root = (Path(artifact_root) if artifact_root else
                 get_data_root(data_root) / 'wostrategy/bsf_junction_reference/schema_v1')
    root = base_root / f'year={season}/round={round_number}/session=R'
    return (root, root / 'bsf_junction_reference.csv',
            root / 'bsf_junction_reference.metadata.json',
            root / 'bsf_junction_reference.score_terms.json.gz')


def read_bsf_event(*, season, round_number, data_root=None, artifact_root=None):
    """Read one completed event with its original identified source provenance.

    Closed history accepts only the approved kernel and original source identity.
    Reading a completed cache never requires current producer files.
    """
    require_approved_kernel()
    _, table_path, metadata_path, terms_path = _artifact_paths(
        season, round_number, data_root, artifact_root)
    meta = json.loads(metadata_path.read_text())
    validate_artifact_kernel(meta)
    validate_frozen_source_identity(meta, season, round_number)
    if (meta.get('table_sha256') != _sha(table_path)
            or meta.get('score_terms_sha256') != _sha(terms_path)):
        raise ValueError('Persisted BSF artifact identity/checksum mismatch')
    table = pd.read_csv(table_path, float_precision='round_trip')
    mercedes_relative_history(table)
    if not table['round'].eq(round_number).all() or not table['year'].eq(season).all():
        raise ValueError('Persisted BSF event identity mismatch')
    return table, meta


def get_bsf_event(*, season, round_number, data_root=None, artifact_root=None,
                  source_paths=None):
    numerical_kernel = require_approved_kernel()
    frozen = frozen_round_contract(season, round_number)
    root, table_path, metadata_path, terms_path = _artifact_paths(
        season, round_number, data_root, artifact_root)
    if frozen is not None:
        # Explicit replacement inputs must not bypass the pin through a cache hit.
        if source_paths is not None:
            verify_frozen_sources(source_paths, season=season, round_number=round_number)
        try:
            return read_bsf_event(season=season, round_number=round_number,
                                  data_root=data_root, artifact_root=artifact_root)
        except FileNotFoundError:
            pass
        # A missing closed cache can only use verified originals. Incompatible
        # completed artifacts above fail explicitly, never silently regenerate.
        sources = (resolve_frozen_sources(season=season, round_number=round_number,
                   data_root=data_root, artifact_root=artifact_root)
                   if source_paths is None else source_paths)
    else:
        sources = (canonical_source_paths(season=season, round_number=round_number, data_root=data_root)
                   if source_paths is None else source_paths)
    if set(sources) != {'clean', 'parameters', 'delta', 'team_deg', 'metadata'}:
        raise ValueError('BSF requires all five explicitly identified canonical producer input paths')
    sources = {key: Path(path) for key, path in sources.items()}
    missing = [str(p) for p in sources.values() if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError('BSF canonical producer inputs unavailable: ' + ', '.join(missing))
    kernel = Path(__file__).parent
    identity = {p.name: _sha(p) for p in sorted(kernel.glob('*.py'))}
    hashes = {str(p): _sha(p) for p in sources.values()}
    if metadata_path.exists():
        # A version/kernel mismatch is an error even when inputs have changed.
        meta = json.loads(metadata_path.read_text())
        validate_artifact_kernel(meta)
        validate_frozen_source_identity(meta, season, round_number)
        if all(p.exists() for p in [table_path, terms_path]):
            saved, meta = read_bsf_event(season=season, round_number=round_number,
                                       data_root=data_root, artifact_root=artifact_root)
            if meta.get('source_hashes') == hashes:
                return saved, meta
    d = pd.read_csv(sources['clean']).reset_index(drop=True)
    if not len(d):
        raise ValueError('No usable canonical clean-lap inputs; BSF unavailable, no substitute')
    d, coverage = complete_clean_laps(d)
    prediction, base, counts = canonical_reference_inputs(d, sources)
    table, terms = calculate_bsf_reference(
        d, prediction, base, counts, season=season, round_number=round_number)
    support = graph_support_summary(table)
    degraded = coverage['excluded_rows'] > 0
    root.mkdir(parents=True, exist_ok=True)
    temp = table_path.with_suffix('.csv.tmp')
    table.to_csv(temp, index=False, float_format='%.17g')
    temp.replace(table_path)
    with terms_path.with_suffix('.gz.tmp').open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8') as stream:
                json.dump(_jsonable(terms), stream, allow_nan=False, sort_keys=True)
    terms_path.with_suffix('.gz.tmp').replace(terms_path)
    meta = dict(
        status='degraded_input' if degraded else 'available',
        input_coverage=_jsonable(coverage), graph_support=support,
        warnings=[degraded_warning(round_number, coverage, support)] if degraded else [],
        year=season, round=round_number, method=METHOD,
        method_version=VERSION, acceptance='B; restricted interpretation', title=TITLE,
        methodology=METHODOLOGY, limitation=LIMITATION, report_caption=CAPTION,
        gauge='all_eleven_teams_arithmetic_mean_zero; lower_is_faster',
        team_order=TEAMS, support_classes={'P': 'projected graph support; not confidence',
                                         'F': 'full canonical fallback',
                                         'N': 'prior only; no usable clean evidence'},
        source_hashes=hashes, kernel_hashes=identity,
        numerical_kernel_fingerprint=numerical_kernel['fingerprint'],
        numerical_kernel_components=numerical_kernel['components'],
        approved_contract_sha256=_sha(kernel / 'approved_contracts.json'),
        canonical_reference='frozen saved-bank full-fit posterior mean; not producer summary median',
        table=str(table_path), score_terms=str(terms_path), table_sha256=_sha(table_path),
        score_terms_sha256=_sha(terms_path),
        source_selection=('approved_frozen_history' if frozen is not None else
                          'explicit_replay' if source_paths is not None else 'current_event_producer'),
    )
    if frozen is not None:
        meta.update(source_identity={role: _sha(p) for role, p in sources.items()},
                    frozen_source_contract_id=frozen['identity'])
    temp = metadata_path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    temp.replace(metadata_path)
    return table, meta


def generate_bsf_season(*, season, round_number, output_dir, figures_dir,
                        data_root=None, artifact_root=None, source_paths_by_round=None):
    """Persist current event, read/regenerate identified history, and render companion."""
    from wostrategy.plots.bsf_junction import plot_bsf_season

    explicit = source_paths_by_round or {}
    current, current_meta = get_bsf_event(
        season=season, round_number=round_number, data_root=data_root,
        artifact_root=artifact_root, source_paths=explicit.get(round_number))
    tables, provenance, unavailable = [], {}, {}
    for r in range(1, round_number + 1):
        try:
            if r == round_number:
                table, meta = current, current_meta
            elif r in explicit:
                table, meta = get_bsf_event(season=season, round_number=r,
                    data_root=data_root, artifact_root=artifact_root, source_paths=explicit[r])
            else:
                try:
                    table, meta = read_bsf_event(season=season, round_number=r,
                        data_root=data_root, artifact_root=artifact_root)
                except FileNotFoundError:
                    table, meta = get_bsf_event(season=season, round_number=r,
                        data_root=data_root, artifact_root=artifact_root)
            tables.append(table)
            provenance[str(r)] = meta
        except (ValueError, FileNotFoundError, KeyError, AssertionError) as exc:
            unavailable[str(r)] = f'{type(exc).__name__}: {exc}'
    history = mercedes_relative_history(pd.concat(tables, ignore_index=True))
    degraded_rounds = {r: meta for r, meta in provenance.items()
                       if meta.get('status') == 'degraded_input'}
    warnings = [warning for meta in degraded_rounds.values() for warning in meta['warnings']]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    current.to_csv(root / 'bsf_junction_reference.current.csv', index=False, float_format='%.17g')
    history.to_csv(root / 'bsf_junction_reference.season.csv', index=False, float_format='%.17g')
    figures = plot_bsf_season(history, figures_dir, season=season,
                              requested_rounds=list(range(1, round_number + 1)),
                              degraded_rounds=degraded_rounds)
    output = dict(
        status=('degraded_input' if degraded_rounds else
                'partial_history' if unavailable else 'available'),
        history_status='partial_history' if unavailable else 'complete',
        degraded_rounds=sorted(map(int, degraded_rounds)), warnings=warnings, method=METHOD,
        method_version=VERSION, acceptance='B; restricted interpretation', title=TITLE,
        methodology=METHODOLOGY, limitation=LIMITATION, report_caption=CAPTION,
        current_event=current_meta, requested_rounds=list(range(1, round_number + 1)),
        included_rounds=sorted(history['round'].unique().tolist()), unavailable_rounds=unavailable,
        team_order=TEAMS, reference_team='Mercedes', unit='s/lap',
        history_policy='reuse approved event identities; closed 2026 R1-R15 rebuilds only from verified original inputs',
        current_table=str(root / 'bsf_junction_reference.current.csv'),
        season_table=str(root / 'bsf_junction_reference.season.csv'),
        relative_team_performance=_jsonable(history.to_dict('records')),
        figures={k: str(p) for k, p in figures.items()}, event_provenance=provenance,
    )
    (root / 'bsf_junction_reference.season.json').write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return output
