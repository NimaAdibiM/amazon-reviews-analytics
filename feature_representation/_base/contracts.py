import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .common import CATEGORIES, KEYS, atomic_json, csv_rows, object_hash, read_json, text_reader
from .partner import features, split_scale


def validate_key(row):
    category, parent = row.get('category'), row.get('parent_asin')
    if category not in CATEGORIES or not isinstance(parent, str) or not parent.strip():
        raise ValueError('INVALID_COMPOSITE_KEY')
    parent.encode('utf-8')
    split = split_scale.assign_split(parent)
    if 'split' in row and row['split'] != split:
        raise ValueError('GLOBAL_PARENT_SPLIT_MISMATCH')
    return category, parent, split


def review_record(row):
    category, parent, split = validate_key(row)
    arrays = ('early_titles', 'early_bodies', 'early_timestamps', 'early_ratings')
    for name in arrays:
        if not isinstance(row.get(name), list) or len(row[name]) != 5:
            raise ValueError(f'FIVE_POSITIONS_REQUIRED: {name}')
    stamps, ratings = row['early_timestamps'], row['early_ratings']
    for stamp in stamps:
        if type(stamp) is not int:
            raise ValueError('INVALID_TIMESTAMP')
        try:
            datetime.fromtimestamp(stamp / 1000, timezone.utc)
        except (ValueError, OSError, OverflowError) as exc:
            raise ValueError('INVALID_TIMESTAMP') from exc
    if stamps != sorted(stamps):
        raise ValueError('COHORT_NOT_CHRONOLOGICAL')
    if any(type(x) is not int or x not in range(1, 6) for x in ratings):
        raise ValueError('INVALID_RATING')
    for name in arrays[:2]:
        if any(x is not None and not isinstance(x, str) for x in row[name]):
            raise ValueError('INVALID_NULLABLE_TEXT')
    titles = [features.normalize_review_text(x) for x in row['early_titles']]
    bodies = [features.normalize_review_text(x) for x in row['early_bodies']]
    order = sorted(range(5), key=lambda i: (stamps[i], titles[i], bodies[i], i))
    times = [stamps[i] for i in order]
    texts = ['\n'.join(x for x in (titles[i], bodies[i]) if x) for i in order]
    groups = [times.index(x) for x in times]
    return {'category': category, 'parent_asin': parent, 'split': split,
            'texts': texts, 'mask': [bool(x) for x in texts], 'timestamps': times,
            'tie_group': groups, 'has_tie': len(set(times)) != 5,
            'source_positions': [i + 1 for i in order], 'ratings': [ratings[i] for i in order],
            'text_document': '\n\n'.join(x for x in texts if x)}


def cohort_rows(path):
    with text_reader(path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f'BLANK_COHORT_RECORD: {number}')
            row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            if not isinstance(row, dict):
                raise ValueError('COHORT_OBJECT_REQUIRED')
            yield number, review_record(row)


def validate_numeric_contract(contract, scaler):
    if contract.get('key_columns') != list(KEYS) or contract.get('blocks') != {k: list(v) for k, v in features.FEATURE_GROUPS.items()}:
        raise ValueError('FEATURE_CONTRACT_MISMATCH')
    if contract.get('split_seed') != 42 or contract.get('numeric_pre_transforms_before_scaling') != features.FEATURE_PRE_TRANSFORMS:
        raise ValueError('FEATURE_PRETRANSFORM_OR_SPLIT_MISMATCH')
    if scaler.get('fit_split') != 'train' or scaler.get('split_seed') != 42:
        raise ValueError('SCALER_NOT_TRAIN_ONLY')
    if scaler.get('pre_transforms') != features.FEATURE_PRE_TRANSFORMS or set(scaler.get('features', {})) != set(features.SCALABLE_FIELDS):
        raise ValueError('SCALER_SCHEMA_MISMATCH')
    if scaler.get('split_hash_spec') != split_scale.SPLIT_HASH_SPEC:
        raise ValueError('SCALER_SPLIT_RULE_MISMATCH')
    for name, params in scaler['features'].items():
        if params.get('scale') is not None and (not math.isfinite(params['scale']) or params['scale'] <= 0 or not math.isfinite(params['mean'])):
            raise ValueError(f'INVALID_SCALER_PARAMETERS: {name}')


def numeric_values(row):
    result = []
    for name in features.FEATURE_VIEWS['full']:
        value = row[name]
        if not value.strip():
            result.append(None)
            continue
        try:
            value = float(value)
        except ValueError as exc:
            raise ValueError(f'INVALID_NUMERIC: {name}') from exc
        result.append(value if math.isfinite(value) else None)
    return result


def build_index(config, path):
    path = Path(path)
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    try:
        db.execute('CREATE TABLE products(category TEXT, parent TEXT, split TEXT, document TEXT, numeric TEXT, review TEXT, source_record INTEGER, PRIMARY KEY(category,parent))')
        cap = config['max_products']
        count = 0
        for number, row in csv_rows(config['documents'], KEYS + ('text_document',)):
            category, parent, split = validate_key(row)
            count += 1
            if count > cap:
                raise ValueError('PRODUCT_BUDGET_EXCEEDED')
            if config['level'] == 'REAL_TRAIN_SUBSET_EXPERIMENT' and split != 'train':
                raise ValueError('REAL_EXPERIMENT_REQUIRES_TRAIN_ONLY')
            if len(row['text_document'].encode('utf-8')) > config['max_document_bytes']:
                raise ValueError('DOCUMENT_BUDGET_EXCEEDED')
            try:
                db.execute('INSERT INTO products VALUES(?,?,?,?,NULL,NULL,?)', (category,parent,split,row['text_document'],number))
            except sqlite3.IntegrityError as exc:
                raise ValueError('DUPLICATE_KEY') from exc
        if not count:
            raise ValueError('EMPTY_INPUT')
        if config.get('numeric'):
            validate_numeric_contract(read_json(config['feature_contract']), read_json(config['scaler']))
            for number, row in csv_rows(config['numeric'], KEYS + features.FEATURE_VIEWS['full']):
                category, parent, split = validate_key(row)
                n = db.execute('UPDATE products SET numeric=? WHERE category=? AND parent=? AND split=? AND numeric IS NULL', (json.dumps(numeric_values(row)),category,parent,split)).rowcount
                if n != 1:
                    raise ValueError('NUMERIC_JOIN_EXTRA_OR_DUPLICATE')
            if db.execute('SELECT count(*) FROM products WHERE numeric IS NULL').fetchone()[0]:
                raise ValueError('NUMERIC_JOIN_MISSING')
        if config.get('cohorts'):
            for source in config['cohorts']:
                for number, row in cohort_rows(source):
                    category, parent, split = validate_key(row)
                    match = db.execute('SELECT document,review FROM products WHERE category=? AND parent=? AND split=?', (category,parent,split)).fetchone()
                    if match is None or match[1] is not None:
                        raise ValueError('COHORT_JOIN_EXTRA_OR_DUPLICATE')
                    if match[0] != row['text_document']:
                        raise ValueError('COHORT_DOCUMENT_MISMATCH')
                    row['source_file'] = Path(source).name
                    row['source_record'] = number
                    db.execute('UPDATE products SET review=? WHERE category=? AND parent=?', (json.dumps(row),category,parent))
            if db.execute('SELECT count(*) FROM products WHERE review IS NULL').fetchone()[0]:
                raise ValueError('COHORT_JOIN_MISSING')
        if config.get('models') and not config.get('cohorts'):
            raise ValueError('MISSING_COHORT_INPUT')
        db.commit()
        return count
    finally:
        db.close()


def index_rows(path):
    with sqlite3.connect(path) as db:
        for cat, parent, split, doc, numeric, review, source in db.execute('SELECT * FROM products ORDER BY category,parent'):
            yield {'category':cat, 'parent_asin':parent, 'split':split, 'text_document':doc,
                   'numeric':json.loads(numeric) if numeric else None,
                   'review':json.loads(review) if review else None, 'source_record':source}


def export_reviews(sources, output, max_products=4000):
    output = Path(output)
    temporary = output.with_name(output.name + '.partial')
    seen = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('OUTPUT_EXISTS')
    with temporary.open('w', encoding='utf-8') as handle:
        for source in sources:
            for number, row in cohort_rows(source):
                key = row['category'], row['parent_asin']
                if key in seen:
                    raise ValueError('DUPLICATE_KEY')
                seen.add(key)
                if len(seen) > max_products:
                    raise ValueError('PRODUCT_BUDGET_EXCEEDED')
                row['source_file'], row['source_record'] = Path(source).name, number
                handle.write(json.dumps(row, ensure_ascii=True) + '\n')
    temporary.replace(output)
    return len(seen)
