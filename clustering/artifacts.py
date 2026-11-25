import csv
import os
from pathlib import Path
import shutil

from .common import atomic_json, digest, file_hashes, key_hash, now, read_json, require_train, verify_hashes


def write_assignments(path,keys,labels,reasons):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['category','parent_asin','split','cluster_label','status','reason']);w.writeheader()
        for row,label,reason in zip(keys,labels,reasons):
            w.writerow({**row,'cluster_label':'' if label is None else int(label),'status':'UNASSIGNED' if label is None else 'NOISE' if label==-1 else 'ASSIGNED','reason':reason})


def read_assignments(root):
    with (Path(root)/'assignments.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    keys=[{k:r[k] for k in ('category','parent_asin','split')} for r in rows]
    require_train(keys)
    if keys!=sorted(keys,key=lambda r:(r['category'],r['parent_asin'])):raise ValueError('ASSIGNMENT_ORDER')
    labels=[]
    for r in rows:
        label=None if r['cluster_label']=='' else int(r['cluster_label'])
        expected='UNASSIGNED' if label is None else 'NOISE' if label==-1 else 'ASSIGNED'
        if label is not None and label< -1 or r['status']!=expected:raise ValueError('ASSIGNMENT_LABEL_STATUS')
        labels.append(label)
    return keys,labels


def verify_trial(root,signature=None):
    root=Path(root);m=read_json(root/'trial_manifest.json')
    if m['status']!='COMPLETE':raise ValueError('TRIAL_NOT_COMPLETE')
    if signature is not None and m['signature']!=signature:raise ValueError('RESUME_SIGNATURE_MISMATCH')
    verify_hashes(root,m['files'])
    if set(file_hashes(root,('trial_manifest.json',)))!=set(m['files']):raise ValueError('UNREGISTERED_TRIAL_FILES')
    keys,labels=read_assignments(root)
    if key_hash(keys)!=m['sample_key_hash'] or len(labels)!=m['n_rows']:raise ValueError('ASSIGNMENT_KEY_HASH_MISMATCH')
    return m


def begin(root,trial_id,signature):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);final=root/trial_id;partial=root/(trial_id+'.partial')
    if final.exists():verify_trial(final,signature);return final,True
    if partial.exists():
        archive=root/'interrupted';archive.mkdir(exist_ok=True)
        target=archive/(trial_id+'_'+now().replace(':','').replace('.','_'));shutil.move(str(partial),str(target))
    partial.mkdir();return partial,False


def publish(partial,metadata):
    partial=Path(partial)
    for p in partial.rglob('*'):
        if p.is_file():
            with p.open('rb') as f:os.fsync(f.fileno())
    metadata={**metadata,'status':'COMPLETE','files':file_hashes(partial),'completed_at':now()}
    atomic_json(partial/'trial_manifest.json',metadata)
    final=partial.with_name(partial.name.removesuffix('.partial'));partial.rename(final)
    verify_trial(final,metadata['signature']);return final
