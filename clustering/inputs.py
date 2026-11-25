import json
import re
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
from scipy import sparse

from feature_representation.artifacts import load_stage_a_artifacts
from feature_representation._base.contracts import validate_key
from .common import Budget, LEVELS, ResourceLimit, digest, key_hash, key_tuple, matrix_bytes, read_json, require_train
from .sampling import add_key, choose_groups, create_index


def validate_auxiliary(keys,arrays):
    for family in ('bge','e5'):
        prefix='semantic_'+family
        mask=arrays.get(prefix+'_mask')
        if mask is None:
            if any(name.startswith(prefix+'_') for name in arrays):raise ValueError('SEMANTIC_MASK_REQUIRED')
            continue
        if mask.dtype!=bool or mask.shape!=(len(keys),5):raise ValueError('SEMANTIC_MASK_INVALID')
        empty=arrays.get(prefix+'_all_empty')
        if empty is None or empty.dtype!=bool or empty.shape!=(len(keys),) or not np.array_equal(empty,~mask.any(axis=1)):
            raise ValueError('SEMANTIC_ALL_EMPTY_MISMATCH')


def header(root,mode,limits):
    root=Path(root)
    if not (root/'manifest.json').is_file():
        raise FileNotFoundError('SKIPPED_NO_REAL_A_ARTIFACT')
    manifest=read_json(root/'manifest.json')
    if manifest.get('status')!='COMPLETE':
        raise ValueError('A_NOT_COMPLETE')
    total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if total>limits['max_input_bytes'] or manifest['n_products']>limits['max_scan_rows']:
        raise ResourceLimit('SKIPPED_RESOURCE_LIMIT: INPUT')
    meta=manifest['metadata']
    markers=json.dumps(meta).lower()
    synthetic=meta.get('synthetic') is True or 'synthetic_stub_' in markers or '"stub": true' in markers
    if mode=='real':
        if synthetic or meta.get('synthetic') is not False or meta.get('level') not in ('REAL_TRAIN_SUBSET_EXPERIMENT','FULL_DATA_EXECUTION'):
            raise ValueError('REAL_INPUT_REJECTED_SYNTHETIC_OR_UNKNOWN')
        scope=manifest.get('report',{}).get('fit_scope',{})
        if not isinstance(scope,dict) or scope.get('fit_split')!='train' or not scope.get('fit_keys'):
            raise ValueError('REAL_A_TRAIN_FIT_PROVENANCE_REQUIRED')
        for category,parent in scope['fit_keys']:
            validate_key({'category':category,'parent_asin':parent,'split':'train'})
        if len(scope['fit_keys'])!=scope.get('n_fit') or len(set(map(tuple,scope['fit_keys'])))!=scope['n_fit']:
            raise ValueError('REAL_A_FIT_KEYS_INVALID')
        for identity in meta.get('provenance',{}).get('models',{}).values():
            if identity.get('stub') or not re.fullmatch(r'[0-9a-f]{40}',identity.get('revision','')) or not identity.get('model_files_hash'):
                raise ValueError('REAL_MODEL_IDENTITY_REVISION_HASH_REQUIRED')
        if not meta.get('provenance',{}).get('inputs'):
            raise ValueError('REAL_INPUT_PROVENANCE_REQUIRED')
    elif mode!='synthetic' or not synthetic:
        raise ValueError('SYNTHETIC_MODE_REQUIRES_SYNTHETIC_ARTIFACT')
    return manifest,digest(root/'manifest.json')


def inspect_input(root,mode,limits):
    manifest,sha=header(root,mode,limits)
    budget=Budget(limits);counts={};spaces={};n=0
    for keys,arrays in load_stage_a_artifacts(root):
        validate_auxiliary(keys,arrays)
        budget.check(sum(matrix_bytes(x) for x in arrays.values()))
        for row in keys:
            tag=row['category']+'/'+row['split'];counts[tag]=counts.get(tag,0)+1
        for name,x in arrays.items():
            desc={'shape_tail':list(x.shape[1:]),'dtype':str(x.dtype),'sparse':sparse.issparse(x)}
            if name in spaces and desc!=spaces[name]:raise ValueError('SPACE_SCHEMA_CHANGED')
            spaces[name]=desc
        n+=len(keys)
    return {'status':'PASS','evidence_level':LEVELS[mode],'source_manifest_sha256':sha,'schema_version':manifest['schema_version'],'n_rows':n,'counts':counts,'spaces':spaces,'synthetic':mode=='synthetic','verification':'A public loader exhausted; all chunk arrays and checksums verified'}


def collect(root,mode,plan):
    limits=plan['resources'];manifest,sha=header(root,mode,limits);budget=Budget(limits)
    with tempfile.TemporaryDirectory(prefix='stage_b_index_') as temp:
        con=sqlite3.connect(str(Path(temp)/'keys.sqlite'));create_index(con)
        scanned=0
        for keys,arrays in load_stage_a_artifacts(root):
            validate_auxiliary(keys,arrays)
            budget.check(sum(matrix_bytes(x) for x in arrays.values()))
            for row in keys:
                if row['split']=='train':add_key(con,row,plan['sampling']['seed'])
            scanned+=len(keys)
            if scanned>limits['max_scan_rows']:raise ResourceLimit('SCAN_ROWS_LIMIT')
        con.commit()
        selected,details=choose_groups(con,plan['sampling']['strategy'],plan['sampling']['max_rows'],plan['sampling']['categories'])
        con.close()
    require_train(selected)
    wanted={key_tuple(r) for r in selected};found=[];parts={};total=0
    names={s['source'] for s in plan['spaces']}
    auxiliaries={'numeric_compact','numeric_missing'}
    for s in plan['spaces']:
        if s['source'].startswith('semantic_'):
            prefix='_'.join(s['source'].split('_')[:2]);auxiliaries.update({prefix+'_mask',prefix+'_all_empty',prefix+'_slope_defined',prefix+'_slope_defined_tie_group_mean'})
    for keys,arrays in load_stage_a_artifacts(root):
        validate_auxiliary(keys,arrays)
        ix=[i for i,r in enumerate(keys) if key_tuple(r) in wanted]
        if not ix:continue
        found.extend({k:keys[i][k] for k in ('category','parent_asin','split')} for i in ix)
        for name in names|auxiliaries:
            if name not in arrays:
                if name in names:raise ValueError('REQUESTED_SPACE_MISSING: '+name)
                continue
            x=arrays[name][ix];total+=matrix_bytes(x)
            if total>limits['max_sample_bytes']:raise ResourceLimit('SAMPLE_BYTES_LIMIT')
            parts.setdefault(name,[]).append(x)
        budget.check(total*2)
    if key_hash(found)!=key_hash(selected) or digest(Path(root)/'manifest.json')!=sha:
        raise ValueError('SOURCE_OR_SELECTED_KEYS_CHANGED')
    arrays={k:sparse.vstack(v,format='csr') if sparse.issparse(v[0]) else np.concatenate(v,axis=0) for k,v in parts.items()}
    details.update({'evidence_level':LEVELS[mode],'keys':selected,'key_hash':key_hash(selected),'source_manifest_sha256':sha,'seed':plan['sampling']['seed'],'n_rows':len(selected),'scanned_rows':scanned,'scope':'train_only','sample_bytes':total})
    return selected,arrays,details,manifest
