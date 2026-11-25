import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import time
from itertools import combinations

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'


def load(path):return json.loads(path.read_text())


def main(out=OUT):
    started=time.perf_counter();base=ROOT/'real_inputs/representation_audit/runs/semantic'
    con=sqlite3.connect((base/'artifacts.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    records=[json.loads(t[0]) for t in con.execute('SELECT record FROM chunks ORDER BY id')]
    keys=[(k['category'],k['parent_asin']) for r in records for k in r['keys']]
    cache={}
    for label in ['tfidf_svd_unigram','semantic_bge_M_l2','semantic_e5_M_l2','numeric_rating','fusion_rating_text']:
        chunks=[]
        for r in records:
            b=con.execute('SELECT payload FROM arrays WHERE chunk_id=? AND label=?',(r['index'],label)).fetchone()[0]
            assert hashlib.sha256(b).hexdigest()==r['arrays'][label]['sha256']
            chunks.append(np.load(io.BytesIO(b),allow_pickle=False))
        cache[label]=np.concatenate(chunks)
    con.close();checks=[];replays=[]
    exp=out/'official'
    for p in sorted((exp/'trials').glob('trial_*')):
        man=load(p/'trial_manifest.json') if (p/'trial_manifest.json').exists() else load(p/'manifest.json')
        spec=man['space'];cfg=man['trial']['config'];m=load(p/'metrics.json')
        with (p/'assignments.csv').open() as f:rows=list(csv.DictReader(f))
        assert keys==[(r['category'],r['parent_asin']) for r in rows]
        labels=np.array([int(r['cluster_label']) for r in rows]);x=cache[spec['source']]
        if spec['normalization']=='l2':x=normalize(x)
        keep=labels>=0;valid=1<len(set(labels[keep]))<int(keep.sum())
        sil=float(silhouette_score(x[keep],labels[keep],metric='cosine')) if valid else None
        actual=m['silhouette']['value']
        delta=abs(sil-actual) if sil is not None and actual is not None else None
        assert (sil is None)==(actual is None)
        if delta is not None:assert delta<1e-6
        if valid:
            stored=m['silhouette_samples']
            order=np.flatnonzero(keep)
            assert [row['index'] for row in stored]==order.tolist()
            np.testing.assert_allclose([row['value'] for row in stored],silhouette_samples(x[keep],labels[keep],metric='cosine'),rtol=0,atol=1e-6)
        checks.append({'trial':p.name,'silhouette_delta':delta,'status':'PASS','n':len(labels)})
        if cfg['algorithm']=='kmeans' and cfg['space'] in ('classical','bge','e5'):
            pred=KMeans(n_clusters=cfg['k'],n_init=10,max_iter=300,tol=.0001,random_state=man['trial']['seed'],algorithm='lloyd').fit_predict(x)
            ari=float(adjusted_rand_score(labels,pred));assert ari==1.0
            replays.append({'trial':p.name,'ari_replay':ari})
    summaries=load(out/'RESULTS_MATCHED.json')
    pre=load(ROOT/'documentation/REAL_INPUT_PREFLIGHT.json')
    agreement=[]
    for r in pre['results']:
        v=next(x for x in summaries if x['space']==r['space'] and x['algorithm']=='kmeans' and x['k']==r['k'])
        diffs={a:abs(v[a]-r[b]) for a,b in [('silhouette_median','silhouette_median'),('seed_ari_median','seed_ari_pairwise_median'),('subsample_ari_seed42_median','subsample_ari_median')]}
        agreement.append({'space':r['space'],'k':r['k'],'differences':diffs,'within_1e_6':all(d<1e-6 for d in diffs.values())})
    if not all(row['within_1e_6'] for row in agreement):
        raise ValueError('PREFLIGHT_COMPARISON_OUTSIDE_TOLERANCE')
    result={'status':'PASS','scope':'independent calculation path in same environment; not independent human reviewer','metric_checks':checks,'independent_primary_refits':replays,'hub_preflight_comparison':agreement,'elapsed_seconds':time.perf_counter()-started,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'no_raw_scan':True,'no_new_inference':True}
    (out/'INDEPENDENT_NUMERIC_CHECK.json').write_text(json.dumps(result,indent=2));print('PASS',len(checks),'metric checks;',len(replays),'primary refits')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    with threadpool_limits(limits=1):main(parser.parse_args().output)
