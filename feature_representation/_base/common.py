import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

CATEGORIES = ('Electronics', 'Beauty_and_Personal_Care', 'Appliances', 'All_Beauty')
KEYS = ('category', 'parent_asin', 'split')
LEVELS = ('SYNTHETIC_FIXTURE_TEST', 'REAL_TRAIN_SUBSET_EXPERIMENT', 'FULL_DATA_EXECUTION')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def text_reader(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', newline='') as handle:
        yield handle


def csv_rows(path, fields):
    csv.field_size_limit(16 * 1024 * 1024)
    with text_reader(path) as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames) or set(reader.fieldnames) != set(fields):
            raise ValueError(f'SCHEMA_MISMATCH: {path}; required columns: {list(fields)}')
        for number, row in enumerate(reader, 1):
            if None in row or None in row.values():
                raise ValueError(f'INVALID_CSV_ROW: {path}, logical record {number}')
            yield number, row


def environment():
    versions = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'matplotlib', 'psutil', 'torch', 'sentence-transformers', 'transformers', 'huggingface-hub'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'processor': platform.processor(), 'cpu_count': os.cpu_count(), 'versions': versions}


def source_hashes():
    root = Path(__file__).resolve().parent
    files = sorted(root.glob('*.py')) + sorted((root.parent.parent / 'data_preparation' / '_base').glob('*.py'))
    return {str(p.relative_to(root.parent.parent)): digest(p) for p in files}
