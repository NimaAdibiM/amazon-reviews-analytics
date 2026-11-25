import csv
import gzip
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from feature_representation.artifacts import load_stage_a_artifacts
from feature_representation.contracts import review_record, validate_numeric_contract
from feature_representation.partner import features, split_scale
from clustering.common import atomic_json, digest, key_hash, now, read_json


def read_csv(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def audit(base, output):
    base,output=Path(base),Path(output);output.mkdir(parents=True,exist_ok=True)
    sample=base/'samples/balanced';sm=read_json(sample/'sampling_manifest.json')
    checks={name:digest(sample/name)==sha for name,sha in sm['export_sha256'].items()}
    if not all(checks.values()):raise ValueError('SAMPLE_HASH_MISMATCH')
    docs=read_csv(sample/'text_documents.csv.gz');numeric=read_csv(sample/'features_scaled.csv.gz')
    raw=[json.loads(s) for s in (sample/'reviews_cohort.jsonl').read_text().splitlines()]
    recs=[review_record(r) for r in raw]
    key=lambda r:(r['category'],r['parent_asin'],r['split'])
    keys=[{k:r[k] for k in ('category','parent_asin','split')} for r in recs]
    tuples=list(map(key,keys))
    if tuples!=list(map(key,docs)) or tuples!=list(map(key,numeric)) or len(set(tuples))!=512 or any(k[2]!='train' for k in tuples):raise ValueError('MATCHED_512_TRAIN_REQUIRED')
    if Counter(k[0] for k in tuples)!=dict.fromkeys(sm['sample_category_counts'],128):raise ValueError('CATEGORY_BALANCE_MISMATCH')
    if any(r['text_document']!=d['text_document'] for r,d in zip(recs,docs)):raise ValueError('EARLY_TEXT_MISMATCH')
    contract=read_json(sample/'feature_contract.json');scaler=read_json(sample/'scaler_train.json')
    validate_numeric_contract(contract,scaler)
    names=[n for block in contract['blocks'].values() for n in block]
    values=np.array([[float(r[n]) for n in names] for r in numeric])
    if values.shape!=(512,40) or not np.isfinite(values).all():raise ValueError('NUMERIC_INVALID')
    verified_names=[n for n in names if n not in contract['blocks']['purchase']]
    errors={n:0.0 for n in verified_names}
    for r,vals in zip(raw,numeric):
        early=[{'rating':r['early_ratings'][i],'timestamp':r['early_timestamps'][i],'title':r['early_titles'][i],'text':r['early_bodies'][i]} for i in range(5)]
        rebuilt=features.make_features(early);transformed=split_scale.apply_scaler(rebuilt,scaler)
        for n in verified_names:errors[n]=max(errors[n],abs(float(vals[n])-float(transformed[n])))
    if max(errors.values())>1e-8:raise ValueError('RECOMPUTED_FEATURE_MISMATCH')
    spaces={};runs={};models={}
    for name in ('balanced','semantic'):
        root=base/'runs'/name;man=read_json(root/'manifest.json')
        with sqlite3.connect(root.joinpath('artifacts.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
            integrity=db.execute('PRAGMA integrity_check').fetchone()[0]
            n_arrays=db.execute('SELECT count(*) FROM arrays').fetchone()[0]
        if integrity!='ok':raise ValueError('SQLITE_INTEGRITY')
        chunks=list(load_stage_a_artifacts(root));rkeys=[k for ks,_ in chunks for k in ks]
        if list(map(key,rkeys))!=tuples:raise ValueError('ARTIFACT_KEYS_MISMATCH')
        if man['metadata']['synthetic'] or man['report']['fit_scope']['fit_split']!='train':raise ValueError('REAL_TRAIN_PROVENANCE_REQUIRED')
        fit=man['report']['fit_scope']['fit_keys']
        if set(map(tuple,fit))!={k[:2] for k in tuples}:raise ValueError('FIT_KEYS_MISMATCH')
        prov=man['metadata']['provenance']
        for path,desc in prov['inputs'].items():
            if digest(sample/Path(path).name)!=desc['sha256']:raise ValueError('PROVENANCE_HASH_MISMATCH')
        counts=sum(len(a) for _,a in chunks)
        if counts!=n_arrays:raise ValueError('EXTRA_OR_MISSING_ARRAYS')
        wanted=('tfidf_unigram','tfidf_svd_unigram','numeric_rating','numeric_compact','fusion_rating_text')
        if name=='semantic':wanted+=('semantic_bge_M_l2','semantic_e5_M_l2','semantic_bge_vectors','semantic_e5_vectors','semantic_bge_mask','semantic_e5_mask')
        assembled={n:sparse.vstack([a[n] for _,a in chunks]) if sparse.issparse(chunks[0][1][n]) else np.concatenate([a[n] for _,a in chunks]) for n in wanted}
        for family in ('bge','e5') if name=='semantic' else ():
            mean=normalize(assembled['semantic_'+family+'_vectors'].mean(axis=1))
            if not np.allclose(mean,assembled['semantic_'+family+'_M_l2'],atol=1e-6):raise ValueError('MEAN_AGGREGATION_MISMATCH')
        runs[name]={'status':'PASS','sqlite_integrity':integrity,'checked_arrays':counts,'chunks':len(chunks),'n_keys':len(rkeys),'manifest_sha256':digest(root/'manifest.json'),'fit_n':len(fit),'shapes':{n:list(a.shape) for n,a in assembled.items()},'source_input_hashes_verified':len(prov['inputs'])}
        spaces[name]=assembled;models[name]=prov['models']
    if not np.array_equal(spaces['balanced']['tfidf_svd_unigram'],spaces['semantic']['tfidf_svd_unigram']):raise ValueError('CLASSICAL_RUNS_DIFFER')
    profiles=[{k:r[k] for k in ('category','parent_asin','split','texts','ratings','timestamps','mask')} for r in recs]
    (output/'early_profiles.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in profiles))
    result={'status':'PASS','created_at':now(),'evidence_level':'SUPPLIED_REAL_BYTES_VERIFIED_NOT_RAW_REINFERENCE','sample_checks':checks,'runs':runs,'models':models,'n_products':512,'n_early_reviews':2560,'n_distinct_parents':len(set(k[1] for k in tuples)),'category_counts':dict(Counter(k[0] for k in tuples)),'sample_key_hash':key_hash(keys),'numeric_shape':list(values.shape),'numeric_features_recomputed':len(verified_names),'numeric_max_abs_errors':errors,'purchase_features':'NOT_RECOMPUTABLE_FROM_SAMPLE_COHORT; values and checksums verified; early_verified omitted upstream','full_population_status':'DOCUMENTARY_ONLY_FULL_DATABASE_ABSENT','raw_scan':False,'reinference':False,'population_train_counts_reported':sm['population_train_counts'],'scaler_fit':'upstream 362202 train; parameters verified but global refit unavailable','split_rule':'SHA256 seed42 parent; recomputed for all512','no_future_fields':all(not any(k in r for k in ('n_low_future','t15','label','target')) for r in raw),'input_files':{str(p.relative_to(base)):digest(p) for p in base.rglob('*') if p.is_file()}}
    atomic_json(output/'REAL_INPUT_AUDIT.json',result)
    atomic_json(output/'REPRESENTATION_ALIGNMENT.json',{'status':'PASS','keys':keys,'key_hash':key_hash(keys),'runs':runs,'same_order':True,'same_classical_arrays':True})
    return result
