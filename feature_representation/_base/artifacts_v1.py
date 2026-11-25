import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy import sparse

from .common import atomic_json, digest, object_hash, read_json
from .contracts import validate_key


def safe_path(root, relative):
    path = (Path(root)/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('UNSAFE_ARTIFACT_PATH')
    return path


def verify_files(root, hashes):
    for relative, sha in hashes.items():
        path = safe_path(root,relative)
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f'ARTIFACT_CHECKSUM_MISMATCH: {relative}')


class ArtifactWriter:
    def __init__(self, root, signature, metadata):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.signature=signature
        self.metadata=metadata
        path=self.root/'checkpoint.json'
        if path.exists():
            self.checkpoint=read_json(path)
            if self.checkpoint['signature'] != signature:
                raise ValueError('CHECKPOINT_SIGNATURE_MISMATCH')
            for item in self.checkpoint['chunks']:
                verify_files(self.root,item['files'])
        else:
            if any(self.root.iterdir()):
                raise ValueError('NONEMPTY_OUTPUT_WITHOUT_CHECKPOINT')
            self.checkpoint={'schema_version':'stage-a-1','signature':signature,'chunks':[]}
            atomic_json(path,self.checkpoint)

    def write_chunk(self, index, rows, arrays, metadata=None):
        keys=[{**{k:r[k] for k in ('category','parent_asin','split')},'source_record':r['source_record']} for r in rows]
        key_hash=object_hash(keys)
        if index < len(self.checkpoint['chunks']):
            old=self.checkpoint['chunks'][index]
            if old['keys_hash'] != key_hash:
                raise ValueError('RESUME_KEY_ALIGNMENT_MISMATCH')
            verify_files(self.root,old['files'])
            return old
        if index != len(self.checkpoint['chunks']):
            raise ValueError('NONCONTIGUOUS_CHECKPOINT')
        name=f'chunk_{index:06d}'
        temporary=self.root/(name+'.partial')
        if temporary.exists():
            shutil.rmtree(temporary)
        destination=self.root/name
        if destination.exists():
            shutil.rmtree(destination)
        temporary.mkdir()
        atomic_json(temporary/'keys.json',keys)
        if metadata is not None:
            atomic_json(temporary/'metadata.json',metadata)
        descriptors={}
        for label,values in arrays.items():
            if not label.replace('_','').isalnum():
                raise ValueError('INVALID_ARTIFACT_LABEL')
            if values.shape[0]!=len(rows):
                raise ValueError('ARTIFACT_ROW_COUNT_MISMATCH')
            if sparse.issparse(values):
                if not np.isfinite(values.data).all():
                    raise ValueError('NONFINITE_ARTIFACT')
                path=temporary/(label+'.npz')
                sparse.save_npz(path,values.tocsr())
                storage='csr_npz'
            else:
                values=np.asarray(values)
                if values.dtype.hasobject or not np.isfinite(values).all():
                    raise ValueError('INVALID_ARRAY_VALUES')
                path=temporary/(label+'.npy')
                np.save(path,values,allow_pickle=False)
                storage='npy'
            descriptors[label]={'path':name+'/'+path.name,'shape':list(values.shape),'dtype':str(values.dtype),'storage':storage}
        os.replace(temporary,destination)
        files={str(p.relative_to(self.root)):digest(p) for p in destination.iterdir() if p.is_file()}
        record={'index':index,'n':len(rows),'keys_hash':key_hash,'keys':name+'/keys.json',
                'arrays':descriptors,'files':files,'metadata':name+'/metadata.json' if metadata is not None else None}
        self.checkpoint['chunks'].append(record)
        atomic_json(self.root/'checkpoint.json',self.checkpoint)
        return record

    def publish(self, expected_rows, fitted_files, report):
        count=sum(c['n'] for c in self.checkpoint['chunks'])
        if count!=expected_rows:
            raise ValueError('FINAL_ROW_COUNT_MISMATCH')
        for chunk in self.checkpoint['chunks']:
            verify_files(self.root,chunk['files'])
        verify_files(self.root,fitted_files)
        manifest={'schema_version':'stage-a-1','status':'COMPLETE','signature':self.signature,
                  'n_products':count,'metadata':self.metadata,'chunks':self.checkpoint['chunks'],
                  'fitted_files':fitted_files,'report':report}
        atomic_json(self.root/'manifest.json',manifest)
        return manifest


def load_stage_a_artifacts(root):
    root=Path(root)
    manifest=read_json(root/'manifest.json')
    if manifest.get('status')!='COMPLETE' or manifest.get('schema_version')!='stage-a-1':
        raise ValueError('UNPUBLISHED_OR_UNKNOWN_ARTIFACT')
    verify_files(root,manifest['fitted_files'])
    total=0
    previous=None
    for expected_index,chunk in enumerate(manifest['chunks']):
        if chunk['index'] != expected_index:
            raise ValueError('CHUNK_ORDER_MISMATCH')
        verify_files(root,chunk['files'])
        keys=read_json(safe_path(root,chunk['keys']))
        if object_hash(keys)!=chunk['keys_hash'] or len(keys)!=chunk['n']:
            raise ValueError('KEYS_HASH_OR_COUNT_MISMATCH')
        for row in keys:
            category,parent,split=validate_key(row)
            key=(category,parent)
            if previous is not None and key<=previous:
                raise ValueError('KEY_ORDER_OR_DUPLICATE')
            previous=key
        arrays={}
        for label,desc in chunk['arrays'].items():
            if desc['path'] not in chunk['files']:
                raise ValueError('ARRAY_NOT_CHECKSUMMED')
            path=safe_path(root,desc['path'])
            values=sparse.load_npz(path) if desc['storage']=='csr_npz' else np.load(path,mmap_mode='r',allow_pickle=False)
            if list(values.shape)!=desc['shape'] or values.shape[0]!=len(keys) or str(values.dtype)!=desc['dtype']:
                raise ValueError('LOADED_SHAPE_OR_DTYPE_MISMATCH')
            if not np.isfinite(values.data if sparse.issparse(values) else values).all():
                raise ValueError('LOADED_NONFINITE')
            arrays[label]=values
        for label,values in arrays.items():
            if label.startswith('semantic_') and label.endswith('_vectors'):
                prefix=label[:-len('vectors')]
                mask=arrays.get(prefix+'mask')
                if values.ndim!=3 or values.shape[1]!=5 or mask is None or mask.shape!=values.shape[:2] or mask.dtype!=bool:
                    raise ValueError('SEMANTIC_MASK_CONTRACT_MISMATCH')
                if np.any(values[~mask]!=0):
                    raise ValueError('EMPTY_POSITION_NOT_ZERO')
        total+=len(keys)
        yield keys,arrays
    if total!=manifest['n_products']:
        raise ValueError('MANIFEST_ROW_COUNT_MISMATCH')
