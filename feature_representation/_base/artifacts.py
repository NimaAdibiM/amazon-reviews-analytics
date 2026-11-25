import hashlib
import io
import json
import sqlite3
from pathlib import Path

import numpy as np
from scipy import sparse

from .artifacts_v1 import load_stage_a_artifacts as load_v1, safe_path, verify_files
from .common import atomic_json, digest, object_hash, read_json
from .contracts import validate_key


SCHEMA='stage-a-2'
DATABASE='artifacts.sqlite'


def encode_array(values, n):
    if values.shape[0]!=n:
        raise ValueError('ARTIFACT_ROW_COUNT_MISMATCH')
    stream=io.BytesIO()
    if sparse.issparse(values):
        values=values.tocsr()
        if not np.isfinite(values.data).all():
            raise ValueError('NONFINITE_ARTIFACT')
        sparse.save_npz(stream,values)
        storage='csr_npz'
    else:
        values=np.asarray(values)
        if values.dtype.hasobject or not np.isfinite(values).all():
            raise ValueError('INVALID_ARRAY_VALUES')
        np.save(stream,values,allow_pickle=False)
        storage='npy'
    payload=stream.getvalue()
    return payload,{'shape':list(values.shape),'dtype':str(values.dtype),'storage':storage,
                    'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}


def open_readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)


def read_record(connection, index):
    found=connection.execute('SELECT record,sha256 FROM chunks WHERE id=?',(index,)).fetchone()
    if found is None:
        raise ValueError('MISSING_COMMITTED_CHUNK')
    record=json.loads(found[0])
    if object_hash(record)!=found[1] or record['index']!=index:
        raise ValueError('RECORD_CHECKSUM_MISMATCH')
    if len(record['keys'])!=record['n'] or object_hash(record['keys'])!=record['keys_hash']:
        raise ValueError('KEYS_HASH_OR_COUNT_MISMATCH')
    return record


def read_payload(connection, index, label, descriptor):
    found=connection.execute('SELECT payload FROM arrays WHERE chunk_id=? AND label=?',(index,label)).fetchone()
    if found is None or len(found[0])!=descriptor['bytes'] or hashlib.sha256(found[0]).hexdigest()!=descriptor['sha256']:
        raise ValueError('ARRAY_CHECKSUM_MISMATCH')
    return found[0]


def verify_database(connection, signature):
    state=connection.execute('SELECT signature,next_index,n_rows FROM state WHERE id=1').fetchone()
    if state is None or state[0]!=signature:
        raise ValueError('CHECKPOINT_SIGNATURE_MISMATCH')
    expected,rows,arrays=0,0,0
    for (index,) in connection.execute('SELECT id FROM chunks ORDER BY id'):
        if index!=expected:
            raise ValueError('CHUNK_ORDER_MISMATCH')
        record=read_record(connection,index)
        for label,descriptor in record['arrays'].items():
            read_payload(connection,index,label,descriptor)
            arrays+=1
        rows+=record['n']
        expected+=1
    if (expected,rows)!=(state[1],state[2]) or connection.execute('SELECT COUNT(*) FROM arrays').fetchone()[0]!=arrays:
        raise ValueError('DATABASE_COUNT_MISMATCH')
    return expected,rows


class ArtifactWriter:
    def __init__(self, root, signature, metadata):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.signature=signature
        self.metadata=metadata
        self.connection=None
        path=self.root/'checkpoint.json'
        if path.exists():
            header=read_json(path)
            if header.get('schema_version')!=SCHEMA:
                raise ValueError('LEGACY_CHECKPOINT_REQUIRES_ORIGINAL_WRITER_OR_NEW_OUTPUT')
            if header.get('signature')!=signature:
                raise ValueError('CHECKPOINT_SIGNATURE_MISMATCH')
            if header.get('database')!=DATABASE:
                raise ValueError('INVALID_CHECKPOINT_DATABASE')
        else:
            if any(self.root.iterdir()):
                raise ValueError('NONEMPTY_OUTPUT_WITHOUT_CHECKPOINT')
            atomic_json(path,{'schema_version':SCHEMA,'signature':signature,'database':DATABASE})
        database=self.root/DATABASE
        if not database.exists() and set(p.name for p in self.root.iterdir())!={'checkpoint.json'}:
            raise ValueError('MISSING_CHECKPOINT_DATABASE')
        connection=sqlite3.connect(database)
        self.connection=connection
        try:
            connection.execute('PRAGMA journal_mode=DELETE')
            connection.execute('PRAGMA synchronous=FULL')
            connection.execute('PRAGMA foreign_keys=ON')
            connection.execute('PRAGMA cache_size=-4096')
            tables=connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if not tables:
                if any(p.name not in {'checkpoint.json',DATABASE,DATABASE+'-journal'} for p in self.root.iterdir()):
                    raise ValueError('EMPTY_CHECKPOINT_DATABASE_WITH_EXISTING_OUTPUT')
                with connection:
                    connection.execute('BEGIN IMMEDIATE')
                    connection.execute('CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1),signature TEXT NOT NULL,next_index INTEGER NOT NULL,n_rows INTEGER NOT NULL)')
                    connection.execute('CREATE TABLE chunks (id INTEGER PRIMARY KEY,record TEXT NOT NULL,sha256 TEXT NOT NULL)')
                    connection.execute('CREATE TABLE arrays (chunk_id INTEGER NOT NULL REFERENCES chunks(id),label TEXT NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(chunk_id,label)) WITHOUT ROWID')
                    connection.execute('INSERT INTO state VALUES (1,?,0,0)',(signature,))
            self.chunk_count,self.row_count=verify_database(connection,signature)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection=None

    def get_record(self,index):
        return read_record(self.connection,index)

    def iter_metadata(self):
        for index in range(self.chunk_count):
            yield self.get_record(index)['metadata']

    def _insert_payload(self,index,label,payload):
        self.connection.execute('INSERT INTO arrays VALUES (?,?,?)',(index,label,payload))

    def write_chunk(self,index,rows,arrays,metadata=None):
        keys=[{**{k:r[k] for k in ('category','parent_asin','split')},'source_record':r['source_record']} for r in rows]
        key_hash=object_hash(keys)
        if index<self.chunk_count:
            old=self.get_record(index)
            if old['keys_hash']!=key_hash:
                raise ValueError('RESUME_KEY_ALIGNMENT_MISMATCH')
            return old
        if (self.root/'manifest.json').exists():
            raise ValueError('PUBLISHED_ARTIFACT_IMMUTABLE')
        if index!=self.chunk_count:
            raise ValueError('NONCONTIGUOUS_CHECKPOINT')
        record={'index':index,'n':len(rows),'keys_hash':key_hash,'keys':keys,'arrays':{},'metadata':metadata}
        with self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            current=self.connection.execute('SELECT next_index FROM state WHERE id=1').fetchone()[0]
            if current!=index:
                raise ValueError('CONCURRENT_WRITER_NOT_SUPPORTED')
            self.connection.execute('INSERT INTO chunks VALUES (?,?,?)',(index,'{}',''))
            for label,values in arrays.items():
                if not label.replace('_','').isalnum():
                    raise ValueError('INVALID_ARTIFACT_LABEL')
                payload,descriptor=encode_array(values,len(rows))
                self._insert_payload(index,label,payload)
                record['arrays'][label]=descriptor
                del payload
            self.connection.execute('UPDATE chunks SET record=?,sha256=? WHERE id=?',
                (json.dumps(record,sort_keys=True,ensure_ascii=False,allow_nan=False),object_hash(record),index))
            self.connection.execute('UPDATE state SET next_index=next_index+1,n_rows=n_rows+? WHERE id=1',(len(rows),))
        self.chunk_count+=1
        self.row_count+=len(rows)
        return record

    def publish(self,expected_rows,fitted_files,report):
        chunks,count=verify_database(self.connection,self.signature)
        if count!=expected_rows:
            raise ValueError('FINAL_ROW_COUNT_MISMATCH')
        verify_files(self.root,fitted_files)
        manifest={'schema_version':SCHEMA,'status':'COMPLETE','signature':self.signature,
                  'n_products':count,'n_chunks':chunks,'metadata':self.metadata,
                  'database':{'path':DATABASE,'sha256':digest(self.root/DATABASE)},
                  'fitted_files':fitted_files,'report':report}
        atomic_json(self.root/'manifest.json',manifest)
        return manifest


def load_stage_a_artifacts(root):
    root=Path(root)
    manifest=read_json(root/'manifest.json')
    if manifest.get('schema_version')=='stage-a-1':
        yield from load_v1(root)
        return
    if manifest.get('status')!='COMPLETE' or manifest.get('schema_version')!=SCHEMA:
        raise ValueError('UNPUBLISHED_OR_UNKNOWN_ARTIFACT')
    verify_files(root,manifest['fitted_files'])
    database=manifest['database']
    verify_files(root,{database['path']:database['sha256']})
    connection=open_readonly(safe_path(root,database['path']))
    try:
        state=connection.execute('SELECT signature,next_index,n_rows FROM state WHERE id=1').fetchone()
        if state!=(manifest['signature'],manifest['n_chunks'],manifest['n_products']):
            raise ValueError('MANIFEST_DATABASE_STATE_MISMATCH')
        total=0
        previous=None
        for index in range(manifest['n_chunks']):
            chunk=read_record(connection,index)
            keys=chunk['keys']
            for row in keys:
                category,parent,split=validate_key(row)
                key=(category,parent)
                if previous is not None and key<=previous:
                    raise ValueError('KEY_ORDER_OR_DUPLICATE')
                previous=key
            arrays={}
            for label,desc in chunk['arrays'].items():
                stream=io.BytesIO(read_payload(connection,index,label,desc))
                if desc['storage']=='csr_npz':
                    values=sparse.load_npz(stream)
                elif desc['storage']=='npy':
                    values=np.load(stream,allow_pickle=False)
                else:
                    raise ValueError('UNKNOWN_ARRAY_STORAGE')
                if list(values.shape)!=desc['shape'] or values.shape[0]!=len(keys) or str(values.dtype)!=desc['dtype']:
                    raise ValueError('LOADED_SHAPE_OR_DTYPE_MISMATCH')
                if not np.isfinite(values.data if sparse.issparse(values) else values).all():
                    raise ValueError('LOADED_NONFINITE')
                arrays[label]=values
            for label,values in arrays.items():
                if label.startswith('semantic_') and label.endswith('_vectors'):
                    mask=arrays.get(label[:-len('vectors')]+'mask')
                    if values.ndim!=3 or values.shape[1]!=5 or mask is None or mask.shape!=values.shape[:2] or mask.dtype!=bool:
                        raise ValueError('SEMANTIC_MASK_CONTRACT_MISMATCH')
                    if np.any(values[~mask]!=0):
                        raise ValueError('EMPTY_POSITION_NOT_ZERO')
            total+=len(keys)
            yield keys,arrays
        if total!=manifest['n_products']:
            raise ValueError('MANIFEST_ROW_COUNT_MISMATCH')
    finally:
        connection.close()
