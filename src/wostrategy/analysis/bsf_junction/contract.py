"""Approved v1 numerical kernel and closed-history identities; never self-approve."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path

from . import constants

NUMERICAL_MODULES = ('_support.py', '_transfer.py', '_graph.py', '_canonical.py', 'reference.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _ast_value(node):
    if isinstance(node, ast.AST):
        return {'node': type(node).__name__, **{
            key: _ast_value(value) for key, value in ast.iter_fields(node)
            if key not in {'type_comment', 'type_ignores'} and not (key == 'type_params' and not value)
        }}
    if isinstance(node, list):
        return [_ast_value(z) for z in node]
    return node


class _NumericalAST(ast.NodeTransformer):
    """Ignore documentation and the policy guard itself, never numerical statements."""
    def visit_ImportFrom(self, node):
        if node.module == 'contract' and [z.name for z in node.names] == ['require_approved_kernel']:
            return None
        return node

    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == 'require_approved_kernel' and not call.args and not call.keywords):
            return None
        return self.generic_visit(node)

    def generic_visit(self, node):
        node = super().generic_visit(node)
        body = getattr(node, 'body', None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            node.body = body[1:]
        return node


def runtime_kernel_identity():
    """Path-independent numerical AST plus live algorithm-defining configuration.

    Plot/report code and ordinary cache orchestration are deliberately excluded.
    The canonical reference function inside workflow.py remains pinned.
    """
    root = Path(__file__).parent
    components = {}
    for name in NUMERICAL_MODULES:
        tree = _NumericalAST().visit(ast.parse((root / name).read_text()))
        components[name] = digest(_ast_value(tree))
    tree = ast.parse((root / 'workflow.py').read_text())
    functional = next(z for z in tree.body if isinstance(z, ast.FunctionDef)
                      and z.name == 'canonical_reference_inputs')
    components['workflow.py:canonical_reference_inputs'] = digest(
        _ast_value(_NumericalAST().visit(functional)))
    support = importlib.import_module(__package__ + '._support')
    transfer = importlib.import_module(__package__ + '._transfer')
    configuration = json.loads(json.dumps(dict(
        TEAMS=constants.TEAMS, LADDER=support.LADDER,
        LABELS=support.LABELS, COLS=transfer.COLS)))
    identity = dict(components=components, configuration=configuration)
    return dict(fingerprint=digest(identity), **identity)


def approved_contract():
    registry = json.loads((Path(__file__).parent / 'approved_contracts.json').read_text())
    try:
        contract = registry['versions'][constants.VERSION]
    except KeyError:
        raise ValueError('Unapproved BSF method version; explicit version contract required') from None
    if contract['method'] != constants.METHOD:
        raise ValueError('BSF method does not match its approved version contract')
    return contract


def require_approved_kernel():
    actual = runtime_kernel_identity()
    if actual != approved_contract()['numerical_kernel']:
        raise ValueError('Unapproved numerical kernel for ' + constants.VERSION + '; explicit version bump required')
    return actual


def validate_artifact_kernel(metadata):
    require_approved_kernel()
    approved = approved_contract()
    if metadata.get('method_version') != constants.VERSION or metadata.get('method') != constants.METHOD:
        raise ValueError('Persisted BSF method/version identity mismatch')
    if 'numerical_kernel_fingerprint' in metadata:
        if (metadata['numerical_kernel_fingerprint'] != approved['numerical_kernel']['fingerprint']
                or metadata.get('numerical_kernel_components') != approved['numerical_kernel']['components']):
            raise ValueError('Persisted BSF numerical kernel is incompatible with v1')
    else:
        # Only the exact previously accepted implementation can migrate implicitly.
        old = metadata.get('kernel_hashes') or {}
        if any(old.get(name) != value for name, value in approved['legacy_forensic_hashes'].items()):
            raise ValueError('Persisted legacy BSF numerical kernel is incompatible with v1')


def frozen_round_contract(season, round_number):
    history = approved_contract()['closed_history']
    return (history['rounds'].get(str(round_number)) if int(season) == history['season'] else None)


def validate_frozen_source_identity(metadata, season, round_number):
    contract = frozen_round_contract(season, round_number)
    if contract is None:
        return
    approved = {role: z['sha256'] for role, z in contract['sources'].items()}
    identity = metadata.get('source_identity')
    if identity is None:
        # Legacy artifacts record path→hash; bind content to its original role.
        hashes = metadata.get('source_hashes') or {}
        names = {Path(path).name: value for path, value in hashes.items()}
        expected = {contract['legacy_source_names'][role]: value for role, value in approved.items()}
        valid = len(hashes) == len(approved) and names == expected
    else:
        valid = identity == approved and metadata.get('frozen_source_contract_id') == contract['identity']
        hashes = metadata.get('source_hashes') or {}
        valid = valid and len(hashes) == len(approved) and sorted(hashes.values()) == sorted(approved.values())
    if not valid:
        raise ValueError('Persisted BSF source identity is not the approved frozen history')
