"""E55: tamper-evident inventory of completed EDA run artifacts.

This is an output fingerprint, not a signature or proof that source data are authentic.
The inventory intentionally excludes itself to avoid self-reference.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

NAME = 'ARTIFACT_CHECKSUMS.json'


def _files(folder):
    root = Path(folder)
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Output cannot contain symbolic links')
        if path.is_file() and path.name != NAME:
            yield path


def _sha256_file(path):
    h = sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(folder):
    root = Path(folder)
    return {
        'schema_version': 1,
        'scope': 'completed_run_outputs_only',
        'policy': 'sha256 of file bytes, relative paths, excluding this inventory only; not evidence of input provenance',
        'artifacts': [
            {'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size, 'sha256': _sha256_file(p)}
            for p in _files(root)
        ],
    }


def write_inventory(folder):
    root = Path(folder)
    doc = inventory(root)
    (root / NAME).write_text(json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return doc


def verify_inventory(folder):
    root = Path(folder)
    p = root / NAME
    if not p.is_file():
        return {'passed': False, 'errors': ['artifact_inventory_missing']}
    try:
        saved = json.loads(p.read_text(encoding='utf-8'))
        if saved.get('schema_version') != 1 or not isinstance(saved.get('artifacts'), list):
            raise ValueError('Invalid schema')
        paths = [r['path'] for r in saved['artifacts']]
        if len(paths) != len(set(paths)) or any(not x or x.startswith('/') or '..' in Path(x).parts for x in paths):
            raise ValueError('Unsafe or duplicated paths')
        fresh = inventory(root)
        expected = {x['path']: (x['bytes'], x['sha256']) for x in saved['artifacts']}
        actual = {x['path']: (x['bytes'], x['sha256']) for x in fresh['artifacts']}
        errors = []
        for path in sorted(set(expected) - set(actual)):
            errors.append('missing:' + path)
        for path in sorted(set(actual) - set(expected)):
            errors.append('unexpected:' + path)
        for path in sorted(set(actual) & set(expected)):
            if actual[path] != expected[path]:
                errors.append('modified:' + path)
        return {'passed': not errors, 'errors': errors, 'files_checked': len(actual)}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {'passed': False, 'errors': ['invalid_inventory:' + type(exc).__name__]}
