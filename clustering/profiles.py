from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from .common import digest, key_tuple, object_hash


FIELDS={'category','parent_asin','split','texts','ratings','timestamps','mask'}


def read_profiles(path,keys,limit,expected_sha=None):
    if path is None:return {},{'status':'SKIPPED_NO_EARLY_PROFILE_INPUT'}
    path=Path(path)
    if path.stat().st_size>limit:raise ValueError('PROFILE_INPUT_BYTE_LIMIT')
    sha=digest(path)
    if expected_sha is not None and sha!=expected_sha:raise ValueError('PROFILE_INPUT_HASH_MISMATCH')
    wanted={key_tuple(r) for r in keys};out={}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if set(row)!=FIELDS:raise ValueError('PROFILE_FIELDS_MUST_BE_EARLY_ONLY_ALLOWLIST')
            if row['split']!='train':raise ValueError('PROFILE_TRAIN_ONLY')
            if any(len(row[k])!=5 for k in ('texts','ratings','timestamps','mask')):raise ValueError('PROFILE_FIVE_REVIEWS_REQUIRED')
            if any(not isinstance(t,str) or len(t)>20000 for t in row['texts']):raise ValueError('PROFILE_TEXT_LIMIT')
            if any(not isinstance(v,(int,float)) or not np.isfinite(v) or not 1<=v<=5 for v in row['ratings']):raise ValueError('PROFILE_RATING_INVALID')
            if any(type(v) is not int for v in row['timestamps']) or row['timestamps']!=sorted(row['timestamps']):raise ValueError('PROFILE_TIMESTAMPS_INVALID')
            if any(type(v) is not bool for v in row['mask']):raise ValueError('PROFILE_MASK_INVALID')
            key=key_tuple(row)
            if key in out:raise ValueError('PROFILE_DUPLICATE_KEY')
            if key in wanted:out[key]=row
    return out,{'status':'PASS' if len(out)==len(keys) else 'PARTIAL_MISSING_KEYS','sha256':sha,'n_matched':len(out),'n_requested':len(keys)}


def redact(text):
    text=re.sub(r'https?://\S+|\b\S+@\S+\.\S+\b','[REDACTED]',text)
    text=re.sub(r'\b[+\d][\d ()-]{7,}\d\b','[REDACTED]',text)
    return text[:320]


def make_profiles(keys,x,labels,aux,arrays,mode,seed,allow_excerpts=False):
    cats=sorted({r['category'] for r in keys});totals=Counter(r['category'] for r in keys);clusters={};global_words=Counter();words={}
    for row in keys:
        rec=aux.get(key_tuple(row))
        if rec:
            terms=set(re.findall(r'[a-z]{3,}', ' '.join(rec['texts']).lower()))
            terms=sorted(terms,key=lambda t:object_hash(t))[:256]
            words[key_tuple(row)]=terms;global_words.update(terms)
    vocabulary=set(w for w,_ in sorted(global_words.items(),key=lambda item:(-item[1],item[0]))[:5000])
    for c in sorted(set(int(v) for v in labels)):
        ix=np.flatnonzero(labels==c);counts=Counter(keys[i]['category'] for i in ix);entry={'n':len(ix),'n_unique_parents':len({keys[i]['parent_asin'] for i in ix}),'fraction_of_selected_train_rows':len(ix)/len(keys),'category_counts':dict(counts),'category_fraction_within_cluster':{cat:counts[cat]/len(ix) for cat in cats},'fraction_of_category_in_cluster':{cat:counts[cat]/totals[cat] for cat in cats},'small_cluster_advisory_proposed_threshold_10':len(ix)<10}
        available=[aux[key_tuple(keys[i])] for i in ix if key_tuple(keys[i]) in aux]
        if available:
            ratings=np.asarray([r['ratings'] for r in available]);lengths=np.asarray([[len(t) for t in r['texts']] for r in available]);masks=np.asarray([r['mask'] for r in available])
            wc=Counter(w for i in ix for w in words.get(key_tuple(keys[i]),[]) if w in vocabulary)
            n_docs=len(available);rest=max(0,len(aux)-n_docs)
            terms=sorted(({'term':w,'cluster_document_count':v,'rest_document_count':global_words[w]-v,'smoothed_log_prevalence_ratio':float(np.log((v+1)/(n_docs+2)/((global_words[w]-v+1)/(rest+2))))} for w,v in wc.items() if v>=2),key=lambda r:(-r['smoothed_log_prevalence_ratio'],r['term']))[:15]
            entry.update({'early_profile_status':'AVAILABLE','n_early_profiles':n_docs,'early_rating_mean_by_position':ratings.mean(axis=0).tolist(),'early_rating_counts':{str(v):int(np.sum(ratings==v)) for v in range(1,6)},'early_low_star_share_mean':float(np.mean(ratings<=2)),'early_product_rating_sd_mean':float(np.std(ratings,axis=1,ddof=1).mean()),'early_rating_quantiles':np.quantile(ratings,[0,.25,.5,.75,1]).tolist(),'early_text_length_mean_by_position':lengths.mean(axis=0).tolist(),'empty_review_fraction':float((~masks).mean()),'distinctive_words':terms,'word_method':'bounded train document-frequency log prevalence; no labels or future fields'})
        else:entry['early_profile_status']='SKIPPED_NO_EARLY_PROFILE_INPUT'
        for name in ('numeric_compact','numeric_missing'):
            if name in arrays:entry[name+'_mean']=np.asarray(arrays[name][ix].mean(axis=0)).ravel().tolist()
        if c>=0:
            center=np.asarray(x[ix].mean(axis=0)).ravel();d=pairwise_distances(x[ix],center[None,:]).ravel();order=np.argsort(d,kind='stable');outside=np.flatnonzero((labels>=0)&(labels!=c))
            examples=[('representative',int(ix[order[0]])),('boundary_low_representativeness',int(ix[order[-1]]))]
            if len(outside):examples.append(('counterexample_nearest_external',int(outside[np.argmin(pairwise_distances(x[outside],center[None,:]).ravel())])))
            examples.append(('reproducible_random',int(sorted(ix,key=lambda i:object_hash([seed,*key_tuple(keys[i])]))[0])))
            entry['examples']=[]
            for role,i in examples:
                rec=aux.get(key_tuple(keys[i]));ex={'role':role,'key':keys[i],'distance_definition':'Euclidean distance to descriptive sample cluster mean, not necessarily fitted model center'}
                ex['early_excerpts']=[redact(t) for t in rec['texts'][:2]] if rec and (mode=='synthetic' or allow_excerpts) else 'WITHHELD_OR_UNAVAILABLE'
                entry['examples'].append(ex)
        clusters[str(c)]=entry
    from feature_representation._base.partner import features
    return {'status':'PASS','numeric_compact_columns':list(features.FEATURE_VIEWS['compact']),'scientific_level':'SYNTHETIC_FIXTURE_ONLY' if mode=='synthetic' else 'REAL_TRAIN_PILOT','clusters':clusters,'numeric_values':'A train-fitted transformed units; not raw rating units','excerpts_policy':'bounded early-only; regex redaction is not a privacy guarantee; real excerpts off by default','selection_bias':'profiles describe this selected sample only'}
