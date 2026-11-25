import csv
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .artifacts import load_stage_a_artifacts
from .classical import diagnose_candidates, training_sample
from .common import atomic_json, read_json
from .contracts import index_rows, build_index


def language_audit(rows, output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for row in rows:
        if row['split']!='train' or row.get('review') is None:
            continue
        for position,text in enumerate(row['review']['texts'],1):
            letters=[ch for ch in text if ch.isalpha()]
            status='short' if len(letters)<20 else 'unknown'
            records.append({'category':row['category'],'parent_asin':row['parent_asin'],'position':position,
                'text':text,'automatic_status':status,'human_language':'','human_status':'','reviewer':'',
                'allowed_human_status':'known|unknown|mixed|short','script_is_not_language':True})
    with (output/'train_language_manual_audit.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        fields=list(records[0]) if records else ['category','parent_asin','position','text','human_language','human_status','reviewer']
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(records)
    atomic_json(output/'language_status.json',{'status':'MANUAL_LANGUAGE_AUDIT_PENDING','n_reviews':len(records),
        'automatic_status_counts':dict(Counter(r['automatic_status'] for r in records)),
        'language_detection_performed':False,'reason':'Do not infer English from Latin script; no validated detector/annotations supplied'})
    return records


def evaluate_pairs(artifacts, pairs_csv, output, seed=42, bootstrap=500):
    manifest=read_json(Path(artifacts)/'manifest.json')
    if manifest['metadata']['synthetic']:
        raise ValueError('SEMANTIC_QUALITY_REQUIRES_REAL_JUDGEMENTS_AND_REAL_EMBEDDINGS')
    vectors={}
    for keys,arrays in load_stage_a_artifacts(artifacts):
        for i,key in enumerate(keys):
            if key['split']!='train':
                continue
            for name in ('bge','e5'):
                label=f'semantic_{name}_vectors'
                if label not in arrays:
                    raise ValueError('BOTH_MODELS_REQUIRED_ON_SAME_SAMPLE')
                for j in range(5):
                    if arrays[f'semantic_{name}_mask'][i,j]:
                        vectors[(name,key['category'],key['parent_asin'],j+1)]=arrays[label][i,j].copy()
    scores={'bge':[],'e5':[]}; labels=[]; judged=[]
    with Path(pairs_csv).open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            if not row.get('reviewer','').strip():
                raise ValueError('HUMAN_REVIEWER_REQUIRED')
            label=float(row['human_similarity'])
            if not 0<=label<=1:
                raise ValueError('HUMAN_SIMILARITY_MUST_BE_0_TO_1')
            for name in scores:
                a=vectors[(name,row['left_category'],row['left_parent'],int(row['left_position']))]
                b=vectors[(name,row['right_category'],row['right_parent'],int(row['right_position']))]
                denominator=np.linalg.norm(a)*np.linalg.norm(b)
                if denominator==0:
                    raise ValueError('ZERO_VECTOR_PAIR')
                scores[name].append(float(a@b/denominator))
            labels.append(label); judged.append(row)
    if len(labels)<10 or len(set(labels))<2:
        raise ValueError('INSUFFICIENT_OR_CONSTANT_HUMAN_JUDGEMENTS')
    correlations={name:float(spearmanr(labels,values).statistic) for name,values in scores.items()}
    if not np.isfinite(list(correlations.values())).all():
        raise ValueError('DEGENERATE_SEMANTIC_SCORES')
    rng=np.random.default_rng(seed); delta=[]
    y=np.asarray(labels)
    for _ in range(bootstrap):
        take=rng.integers(0,len(y),len(y))
        if len(set(y[take]))<2:
            continue
        corr=[spearmanr(y[take],np.asarray(scores[n])[take]).statistic for n in ('e5','bge')]
        if np.isfinite(corr).all():
            delta.append(float(corr[0]-corr[1]))
    result={'status':'MODEL_SELECTION_PENDING','n_pairs':len(y),'spearman':correlations,
        'paired_e5_minus_bge_interval':np.quantile(delta,[.025,.975]).tolist() if delta else None,
        'bootstrap_unit':'annotated pair; exploratory only; shared products may invalidate independence',
        'seed':seed,'bootstrap':bootstrap,'warning':'Requires hub review of annotation quality, language coverage and dependence; never an automatic winner',
        'judgements':judged,'scores':scores}
    atomic_json(output,result)
    return result


def run_diagnostics(config, output):
    from .pipeline import validate_config, input_fingerprints
    validate_config(config)
    input_fingerprints(config)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    index=output/'diagnostic_index.sqlite'
    build_index(config,index)
    rows,scope=training_sample(index_rows(index),min(config['fit_max_products'],4000),config['fit_text_bytes'],config['seed'])
    results={c['name']:diagnose_candidates(rows,c,[2,4,8] if config['level']=='SYNTHETIC_FIXTURE_TEST' else [32,64,128,256],[config['seed'],config['seed']+1]) for c in config['classical']}
    atomic_json(output/'svd_candidate_diagnostics.json',{'level':config['level'],'fit_scope':scope,'candidates':results})
    language_audit(rows,output)
    index.unlink()
    return results
