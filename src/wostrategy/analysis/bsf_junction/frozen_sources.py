"""Resolve only hash-verified approved inputs for the closed 2026 R1–R15 set."""
from __future__ import annotations

import hashlib
from pathlib import Path

from wodata import get_data_root
from .contract import frozen_round_contract


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def verify_frozen_sources(sources, *, season, round_number):
    contract = frozen_round_contract(season, round_number)
    if contract is None:
        return
    if set(sources) != set(contract['sources']):
        raise ValueError('Frozen history requires all five approved source roles')
    for role, entry in contract['sources'].items():
        if not Path(sources[role]).is_file():
            raise FileNotFoundError('Approved frozen input unavailable: ' + str(sources[role]))
        if file_sha(sources[role]) != entry['sha256']:
            raise ValueError('Source is not approved for frozen history: ' + role)


def resolve_frozen_sources(*, season, round_number, data_root=None, artifact_root=None):
    contract = frozen_round_contract(season, round_number)
    if contract is None:
        raise ValueError('Event is not in the approved closed reference set')
    data = get_data_root(data_root)
    cache = (Path(artifact_root) if artifact_root else
             data / 'wostrategy/bsf_junction_reference/schema_v1')
    sources = {}
    for role, entry in contract['sources'].items():
        original = data / entry['path']
        if 'selection' not in entry:
            sources[role] = original
            continue
        # This is an exact byte extraction from the pinned original range bank,
        # never a candidate refit or fallback to the later per-event producer.
        target = cache / 'approved_inputs' / contract['identity'] / (role + '.csv')
        if target.is_file() and file_sha(target) == entry['sha256']:
            sources[role] = target
            continue
        if not original.is_file():
            raise FileNotFoundError('Approved original published bank unavailable: ' + str(original))
        if file_sha(original) != entry['archive_sha256']:
            raise ValueError('Original published bank identity mismatch: ' + str(original))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.csv.tmp')
        try:
            with original.open('rb') as src, temporary.open('wb') as dst:
                header = src.readline()
                if not header.rstrip().endswith(b',Round'):
                    raise ValueError('Approved published bank has unexpected round layout')
                dst.write(header)
                suffix = (',' + str(entry['selection']['Round'])).encode()
                for line in src:
                    if line.rstrip().endswith(suffix):
                        dst.write(line)
            if file_sha(temporary) != entry['sha256']:
                raise ValueError('Extracted original round bank identity mismatch')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        sources[role] = target
    verify_frozen_sources(sources, season=season, round_number=round_number)
    return sources
