import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
from scipy import sparse

from feature_representation._base.contracts import validate_key
from feature_representation._base.common import CATEGORIES, atomic_json, digest, environment, now, object_hash, read_json


ROOT=Path(__file__).resolve().parents[1]
LEVELS={'synthetic':'SYNTHETIC_CONTRACT_TEST','real':'REAL_TRAIN_PILOT'}


class ResourceLimit(RuntimeError):
    pass


def code_hashes():
    return {p.relative_to(ROOT).as_posix():digest(p) for folder in ('clustering','feature_representation','data_preparation','analysis') for p in sorted((ROOT/folder).rglob('*.py'))}


def matrix_bytes(x):
    return int(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes) if sparse.issparse(x) else int(x.nbytes)


def matrix_hash(x):
    h=hashlib.sha256()
    h.update(str((x.shape,str(x.dtype),sparse.issparse(x))).encode())
    if sparse.issparse(x):
        x=x.tocsr(copy=True);x.sort_indices()
        for a in (x.data,x.indices,x.indptr):h.update(a.tobytes())
    else:
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def key_tuple(row):
    return row['category'],row['parent_asin']


def key_hash(rows):
    return object_hash([[r['category'],r['parent_asin'],r['split']] for r in rows])


def require_train(keys):
    if not keys or any(r.get('split')!='train' for r in keys):
        raise ValueError('TRAIN_ONLY_REQUIRED')
    for row in keys:validate_key(row)
    pairs=[key_tuple(r) for r in keys]
    if len(pairs)!=len(set(pairs)):
        raise ValueError('DUPLICATE_COMPOSITE_KEY')


def safe_path(root,name):
    p=(Path(root)/name).resolve()
    if not p.is_relative_to(Path(root).resolve()):
        raise ValueError('UNSAFE_PATH')
    return p


def file_hashes(root,exclude=()):
    root=Path(root)
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.relative_to(root).as_posix() not in exclude}


def verify_hashes(root,files):
    for name,sha in files.items():
        p=safe_path(root,name)
        if not p.is_file() or digest(p)!=sha:
            raise ValueError('CHECKSUM_MISMATCH: '+name)


def save_matrix(path,x):
    path=Path(path)
    if sparse.issparse(x):
        sparse.save_npz(path,x.tocsr())
    else:
        np.save(path,x,allow_pickle=False)


def load_matrix(path):
    return sparse.load_npz(path) if str(path).endswith('.npz') else np.load(path,allow_pickle=False)


class Budget:
    def __init__(self,limits):
        self.limits=limits
        self.started=time.perf_counter()
        self.peak=0

    def check(self,extra_bytes=0):
        rss=psutil.Process().memory_info().rss
        self.peak=max(self.peak,rss)
        if rss+extra_bytes>self.limits['max_memory_mb']*1024**2:
            raise ResourceLimit('SKIPPED_RESOURCE_LIMIT: RAM')
        if time.perf_counter()-self.started>self.limits['max_wall_seconds']:
            raise ResourceLimit('SKIPPED_RESOURCE_LIMIT: TIME')

    def pairwise(self,n,m=None):
        estimate=4*8*int(n)*int(n if m is None else m)
        if estimate>self.limits['max_pairwise_bytes']:
            raise ResourceLimit('SKIPPED_RESOURCE_LIMIT: PAIRWISE_ESTIMATE '+str(estimate))
        self.check(estimate)
        return estimate

    def matrix(self,x,copies=1):
        size=matrix_bytes(x)*copies
        if size>self.limits['max_sample_bytes']:
            raise ResourceLimit('SKIPPED_RESOURCE_LIMIT: SAMPLE_BYTES')
        self.check(size)
        return size
