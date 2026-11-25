import csv
from collections import Counter, defaultdict
from tempfile import TemporaryDirectory
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
        fields=['category','parent_asin','position','text','automatic_status','human_language','human_status','reviewer',
                'allowed_human_status','script_is_not_language']
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(records)
    atomic_json(output/'language_status.json',{'status':'MANUAL_LANGUAGE_AUDIT_PENDING','n_reviews':len(records),
        'automatic_status_counts':dict(Counter(r['automatic_status'] for r in records)),
        'language_detection_performed':False,'reason':'Do not infer English from Latin script; no validated detector/annotations supplied'})
    return records


def evaluate_pairs(artifacts, pairs_csv, output, seed=42, bootstrap=500):
    manifest=read_json(Path(artifacts)/'manifest.json')
    if manifest['metadata']['synthetic']:
        raise ValueError('SEMANTIC_QUALITY_REQUIRES_REAL_JUDGEMENTS_AND_REAL_EMBEDDINGS')
    if type(bootstrap) is not int or bootstrap<1:
        raise ValueError('INVALID_BOOTSTRAP_COUNT')
    ratings=defaultdict(dict)
    judged=[]
    with Path(pairs_csv).open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            reviewer=row.get('reviewer','').strip()
            if not reviewer:
                raise ValueError('HUMAN_REVIEWER_REQUIRED')
            label=float(row['human_similarity'])
            if not 0<=label<=1:
                raise ValueError('HUMAN_SIMILARITY_MUST_BE_0_TO_1')
            pair=tuple(sorted((row[side+'_category'],row[side+'_parent'],int(row[side+'_position']))
                              for side in ('left','right')))
            if pair[0]==pair[1] or any(key[2] not in range(1,6) for key in pair):
                raise ValueError('INVALID_REVIEW_PAIR')
            if reviewer in ratings[pair]:
                raise ValueError('DUPLICATE_JUDGEMENT')
            ratings[pair][reviewer]=label
            judged.append(row)
    pairs=list(ratings)
    labels=[float(np.mean(list(ratings[p].values()))) for p in pairs]
    if len(pairs)<10 or len(set(labels))<2:
        raise ValueError('INSUFFICIENT_OR_CONSTANT_HUMAN_JUDGEMENTS')
    needed={key for pair in pairs for key in pair}
    vectors={}
    for keys,arrays in load_stage_a_artifacts(artifacts):
        for name in ('bge','e5'):
            if 'semantic_'+name+'_vectors' not in arrays:
                raise ValueError('BOTH_MODELS_REQUIRED_ON_SAME_SAMPLE')
        for i,key in enumerate(keys):
            if key['split']!='train':
                continue
            for j in range(5):
                review=(key['category'],key['parent_asin'],j+1)
                if review in needed:
                    for name in ('bge','e5'):
                        if arrays[f'semantic_{name}_mask'][i,j]:
                            vectors[name,review]=arrays[f'semantic_{name}_vectors'][i,j].copy()
    scores={'bge':[],'e5':[]}
    for left,right in pairs:
        for name in scores:
            if (name,left) not in vectors or (name,right) not in vectors:
                raise ValueError('PAIR_REQUIRES_PRESENT_TRAIN_REVIEWS')
            a,b=vectors[name,left],vectors[name,right]
            denominator=np.linalg.norm(a)*np.linalg.norm(b)
            if denominator==0:
                raise ValueError('ZERO_VECTOR_PAIR')
            scores[name].append(float(a@b/denominator))
    correlations={name:float(spearmanr(labels,values).statistic) for name,values in scores.items()}
    if not np.isfinite(list(correlations.values())).all():
        raise ValueError('DEGENERATE_SEMANTIC_SCORES')
    rng=np.random.default_rng(seed); delta=[]
    y=np.asarray(labels)
                                                                                  
    parents={}
    def root(parent):
        parents.setdefault(parent,parent)
        while parents[parent]!=parent:
            parents[parent]=parents[parents[parent]]
            parent=parents[parent]
        return parent
    for left,right in pairs:
        parents[root(left[1])]=root(right[1])
    components=defaultdict(list)
    for index,(left,right) in enumerate(pairs):
        components[root(left[1])].append(index)
    blocks=list(components.values())
    for _ in range(bootstrap if len(blocks)>1 else 0):
        take=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
        if len(set(y[take]))<2:
            continue
        corr=[spearmanr(y[take],np.asarray(scores[n])[take]).statistic for n in ('e5','bge')]
        if np.isfinite(corr).all():
            delta.append(float(corr[0]-corr[1]))
    result={'status':'MODEL_SELECTION_PENDING','n_pairs':len(y),'spearman':correlations,
        'paired_e5_minus_bge_interval':np.quantile(delta,[.025,.975]).tolist() if delta else None,
        'bootstrap_unit':'connected components of pairs sharing a parent','n_independent_blocks':len(blocks),
        'valid_bootstrap_replicates':len(delta),'pair_order':pairs,'similarity_per_pair':labels,
        'seed':seed,'bootstrap':bootstrap,'warning':'Requires hub review of annotation quality, language coverage and dependence; never an automatic winner',
        'judgements':judged,'scores':scores}
    atomic_json(output,result)
    return result


def run_diagnostics(config, output, authorize_full=False):
    from .pipeline import validate_config, input_fingerprints
    validate_config(config,authorize_full)
    input_fingerprints(config)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=output) as work:
        index=Path(work)/'diagnostic_index.sqlite'
        build_index(config,index)
        rows,scope=training_sample(index_rows(index),min(config['fit_max_products'],4000),config['fit_text_bytes'],config['seed'])
    results={c['name']:diagnose_candidates(rows,c,[2,4,8] if config['level']=='SYNTHETIC_FIXTURE_TEST' else [32,64,128,256],[config['seed'],config['seed']+1]) for c in config['classical']}
    atomic_json(output/'svd_candidate_diagnostics.json',{'level':config['level'],'fit_scope':scope,'candidates':results})
    language_audit(rows,output)
    return results
