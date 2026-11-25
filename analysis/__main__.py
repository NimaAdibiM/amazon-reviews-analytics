import argparse
import csv
from itertools import combinations
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from clustering.artifacts import read_assignments, verify_trial
from clustering.assessment import PROPOSED
from clustering.common import atomic_json, code_hashes, digest, environment, now, object_hash, read_json
from clustering.policy import seal_policy
from clustering.runner import run, compare, report
from clustering.verification import verify_experiment
from .audit import audit


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'real_inputs/representation_audit'
OUT=ROOT/'results'


def policy(audit_result, out=OUT):
    frozen=read_json(ROOT/'config/analysis/CLUSTERING_ANALYSIS_PLAN.json')
    p=read_json(ROOT/'config/clustering/synthetic_policy.json')
    p.update({'status':'APPROVED_FOR_REAL_PILOT','proposal_only':False,'approved_by':'User supplied final hub decisions V4','approval_reference':'documentation/ANALYSIS_PROTOCOL.md','approved_at':now(),'source_manifest_sha256':digest(BASE/'runs/semantic/manifest.json'),'sample_key_hash':audit_result['sample_key_hash'],'code_sha256':object_hash(code_hashes()),'early_profiles_sha256':digest(out/'early_profiles.jsonl'),'research_contract':'V4_APPROVED_REAL512_EXPLORATORY_NO_FUTURE_PREDICTION','allow_reviewed_excerpts':False,'selection_policy':{'approved_thresholds':PROPOSED,'frozen_analysis':frozen,'selection_origin':'Approved after independent exploratory preflight; not preregistered before data','primary_representation':'e5','primary_k':3,'legacy_thresholds':'MB and Q1 are diagnostics only, not new scientific selection rules'}})
    plan=p['plan'];plan['resources'].update({'max_rows':512,'max_scan_rows':512,'max_wall_seconds':1800})
    plan['sampling'].update({'max_rows':512,'seed':42})
    plan['spaces']=[{'id':sid,'source':source,'normalization':'l2','metric':'euclidean'} for sid,source in frozen['primary_spaces'].items()]
    plan['spaces']+=[{'id':'numeric','source':'numeric_rating','normalization':'none','metric':'euclidean'},{'id':'numeric_l2','source':'numeric_rating','normalization':'l2','metric':'euclidean'},{'id':'fusion','source':'fusion_rating_text','normalization':'l2','metric':'euclidean','fusion_weights':{'text_space':'tfidf_svd_unigram','text_weight':.5,'numeric_weight':.5}},{'id':'e5_cosine','source':'semantic_e5_M_l2','normalization':'l2','metric':'cosine'}]
    plan['seeds']=frozen['seed_list'];plan['evaluation']={'max_rows':512,'seed':9001,'silhouette_metric':'cosine'}
    plan['stability']={'subsample_repeats':2,'fraction':.8,'min_overlap':20,'normalization_sensitivity':False,'random_seeds':[142,242],'fit_seed':42,'sampler':'numpy_parent_without_replacement_ceil'}
    km={'algorithm':'kmeans','n_init':10,'max_iter':300,'tol':.0001}
    plan['candidates']=[{**km,'space':s,'k_values':[3,5,8]} for s in frozen['primary_spaces']]
    plan['candidates'] +=[{**km,'space':s,'k_values':[3]} for s in ('numeric','numeric_l2','fusion')]
    plan['candidates'] +=[{'algorithm':'agglomerative','space':'e5_cosine','linkage':'average','k':3,'all_seeds':False},{'algorithm':'minibatch','space':'e5','k':3,'n_init':10,'max_iter':300,'tol':.0001,'batch_size':128,'init_size':384,'reassignment_ratio':.01,'max_no_improvement':20},{'algorithm':'dbscan','space':'e5_cosine','min_samples':10,'eps_quantile':.9,'all_seeds':False}]
    plan['common_eligible']=True;plan['figures']=False
    p['secondary_design']='Prespecified before official run: numeric raw train-zscore vs rowL2 scaling ablation; equal-weight classical/rating fusion; alternatives on E5; no outcome based retuning'
    return p


def summarize(out=OUT):
    exp=out/'official';verified=verify_experiment(exp);comp=compare(exp)
    profiles=[__import__('json').loads(s) for s in (out/'early_profiles.jsonl').read_text().splitlines()]
    profile_by_key={(r['category'],r['parent_asin']):r for r in profiles}
    if len(profile_by_key)!=len(profiles):raise ValueError('DUPLICATE_PROFILE_KEY')
    records=[];families={}
    for path in sorted((exp/'trials').glob('trial_*')):
        if not path.is_dir() or path.name.endswith('.partial'):continue
        man=verify_trial(path);m=read_json(path/'metrics.json');st=read_json(path/'stability.json');keys,labels=read_assignments(path);c=man['trial']['config'];seed=man['trial']['seed']
        row={'trial_id':path.name,'space':c['space'],'algorithm':c['algorithm'],'k':c.get('k'),'seed':seed,'n':len(keys),'silhouette_cosine':m['silhouette'].get('value'),'coverage':m['coverage'],'min_cluster':min(m['cluster_sizes'].values(),default=0),'inertia':m['cost'].get('inertia'),'seconds':m['cost']['elapsed_fit_seconds'],'rss_bytes':m['cost']['rss_after_fit_bytes']}
        records.append(row);fam=(c['space'],c['algorithm'],c.get('k'));families.setdefault(fam,[]).append((row,labels,st,path,keys))
    summaries=[]
    for (space,algo,k),group in families.items():
        aris=[adjusted_rand_score(a[1],b[1]) for a,b in combinations(group,2)]
        ref=next((v for v in group if v[0]['seed']==42),group[0]);labels=np.array([-1 if v is None else v for v in ref[1]]);keep=labels>=0
        ordered=[profile_by_key[(r['category'],r['parent_asin'])] for r in ref[4]]
        ratings=np.array([np.mean(r['ratings']) for r in ordered])[keep]
        cats=np.array([r['category'] for r in ordered])[keep]
        assigned=labels[keep]
        multiple_groups=len(set(assigned))>=2
        sub=[v['value'] for v in ref[2]['subsamples'] if v['status']=='DEFINED']
        sil=[v[0]['silhouette_cosine'] for v in group if v[0]['silhouette_cosine'] is not None]
        total=float(np.square(ratings-ratings.mean()).sum()) if ratings.size else 0.0
        between=sum(int((assigned==c).sum())*float(ratings[assigned==c].mean()-ratings.mean())**2 for c in set(assigned))
        row={'space':space,'algorithm':algo,'k':k,'n_seeds':len(group),'silhouette_median':float(np.median(sil)) if sil else None,'seed_ari_median':float(np.median(aris)) if aris else None,'subsample_ari_seed42_median':float(np.median(sub)) if sub else None,'min_cluster_over_seeds':min(v[0]['min_cluster'] for v in group),'fit_seconds_total':sum(v[0]['seconds'] for v in group),'category_nmi_seed42':float(normalized_mutual_info_score(cats,assigned)) if multiple_groups else None,'early_rating_eta2_seed42':between/total if multiple_groups and total else None,'reference_trial':ref[0]['trial_id'],'warning_cutoffs_met':bool(aris and sub and np.median(aris)>=.75 and np.median(sub)>=.60 and min(v[0]['min_cluster'] for v in group)>=10)}
        summaries.append(row)
    for name,rows in [('RESULTS_MATCHED.csv',summaries),('ALL_TRIALS.csv',records)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(out/'RESULTS_MATCHED.json',summaries)
    r=read_json(exp/'run_receipt.json');atomic_json(out/'REAL_EXECUTION_RECEIPT.json',{'status':'PASS','scope':'512 real balanced train products only','official_receipt':r,'verification':verified,'code':code_hashes(),'environment':environment(),'source_audit_sha256':digest(out/'REAL_INPUT_AUDIT.json'),'results_sha256':digest(out/'RESULTS_MATCHED.csv'),'raw_scan':False,'new_model_inference':False,'full_vectors_independently_verified':False})
    return summaries


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['audit','run','verify']);parser.add_argument('--output',type=Path);args=parser.parse_args()
    out=(args.output or (OUT if args.command=='verify' else ROOT/'evidence_phase2_review')).resolve()
    if args.command=='verify':
        print(verify_experiment(out/'official'));return
    if args.command=='audit':print(audit(BASE,out)['status']);return
    if args.command=='run':
        audited=audit(BASE,out)
        registered=out/'registered_policy.json'
        if registered.exists():p=read_json(registered);sealed=out/'policies'/f"policy_v1_{object_hash(p)}.json"
        else:
            p=policy(audited,out);atomic_json(registered,p);sealed=Path(seal_policy(registered,out/'policies')['policy_path'])
        atomic_json(out/'FINAL_SELECTION_POLICY.json',p['selection_policy']);atomic_json(out/'EXPERIMENT_PLAN.json',p['plan'])
        receipt=run(BASE/'runs/semantic',out/'official',p,'real',sealed,out/'early_profiles.jsonl')
        if receipt['status']!='PASS':raise RuntimeError('OFFICIAL_EXPERIMENT_INCOMPLETE')
    result=summarize(out);print(result);print(report(out/'official'))


if __name__=='__main__':main()
