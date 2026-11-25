from itertools import combinations
from pathlib import Path
import time
import sys

import numpy as np
from sklearn.preprocessing import normalize

from .verification import verify_experiment, verify_plan
from .assessment import assess
from .artifacts import begin, publish, read_assignments, verify_trial, write_assignments
from .clusterers import fit_cluster
from .common import Budget, LEVELS, ResourceLimit, atomic_json, code_hashes, environment, key_hash, key_tuple, now, object_hash, read_json
from .inputs import collect
from .metrics import aligned_ari, fixed_evaluation, internal_metrics
from .policy import check_sealed, validate_policy
from .profiles import make_profiles, read_profiles
from .sampling import subsample_indices, approved_subsample_indices
from .spaces import prepare


def expand(plan):
    trials=[]
    for candidate in plan['candidates']:
        seeds=plan['seeds'] if candidate.get('all_seeds',True) else plan['seeds'][:1]
        ks=candidate.get('k_values',[candidate.get('k')])
        scopes=['pooled'] if candidate.get('scope','pooled')=='pooled' else plan['sampling']['categories']
        for seed in seeds:
            for k in ks:
                for scope in scopes:
                    cfg={key:value for key,value in candidate.items() if key not in ('all_seeds','k_values','scope')}
                    if k is not None:cfg['k']=k
                    trials.append({'config':cfg,'seed':seed,'scope':scope})
    if len({object_hash(t) for t in trials})!=len(trials):raise ValueError('DUPLICATE_TRIAL_CONFIGURATION')
    fits=len(trials)*(1+plan['stability']['subsample_repeats']+int(plan['stability']['normalization_sensitivity']))
    if len(trials)>plan['resources']['max_trials'] or fits>plan['resources']['max_total_fits']:
        raise ResourceLimit('TRIAL_OR_FIT_COUNT_LIMIT')
    return trials


def stability(x,labels,keys,cfg,seed,plan,normalization):
    result={'subsamples':[],'normalization_sensitivity':{'status':'NOT_RUN'},'uncertainty':'descriptive repetitions; no independent-observation confidence interval','fit_count':0}
    for j in range(plan['stability']['subsample_repeats']):
        subseed=plan['stability'].get('random_seeds',[seed+1000+i for i in range(plan['stability']['subsample_repeats'])])[j]
        sampler=approved_subsample_indices if plan['stability'].get('sampler')=='numpy_parent_without_replacement_ceil' else subsample_indices
        ix=sampler(keys,plan['stability']['fraction'],subseed);subkeys=[keys[i] for i in ix]
        try:
            lab,_,_=fit_cluster(x[ix],cfg,plan['stability'].get('fit_seed',seed),plan['resources']);result['fit_count']+=1
            value=aligned_ari(keys,labels,subkeys,lab,plan['stability']['min_overlap'])
            value.update({'seed':subseed,'fit_seed':plan['stability'].get('fit_seed',seed),'sample_key_hash':key_hash(subkeys),'sampling_unit':'global parent_asin','fraction':plan['stability']['fraction']})
        except ValueError as e:value={'status':'INVALID','reason':str(e)}
        result['subsamples'].append(value)
    if plan['stability']['normalization_sensitivity']:
        if normalization=='none':
            alt=normalize(x);lab,_,_=fit_cluster(alt,cfg,seed,plan['resources']);result['fit_count']+=1
            result['normalization_sensitivity']={**aligned_ari(keys,labels,keys,lab,plan['stability']['min_overlap']),'change':'none to row L2; same train rows, algorithm, seed and parameters'}
        else:result['normalization_sensitivity']={'status':'NOT_RUN','reason':'Use paired explicit none/L2 spaces; inverse transform is not assumed'}
    return result


def run(root,output,policy,mode,policy_path=None,profiles_path=None):
    started=time.perf_counter();validate_policy(policy,mode)
    if mode=='real':check_sealed(policy,policy_path)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);plan=policy['plan'];budget=Budget(plan['resources'])
    keys,arrays,sample,manifest=collect(root,mode,plan)
    validate_policy(policy,mode,sample['source_manifest_sha256'],sample['key_hash'])
    if mode=='real' and (profiles_path is None or not policy.get('early_profiles_sha256')):raise ValueError('REAL_EARLY_PROFILE_HASH_BINDING_REQUIRED')
    aux,aux_report=read_profiles(profiles_path,keys,plan['resources']['max_profile_bytes'],policy.get('early_profiles_sha256'))
    if mode=='real' and aux_report['status']!='PASS':raise ValueError('REAL_PROFILE_COVERAGE_REQUIRED')
    code=code_hashes();signature=object_hash({'policy':policy,'source':sample['source_manifest_sha256'],'sample':sample['key_hash'],'code':code,'profiles':aux_report,'mode':mode})
    if (output/'experiment_plan.json').exists() and read_json(output/'experiment_plan.json')['signature']!=signature:raise ValueError('EXPERIMENT_RESUME_SIGNATURE_MISMATCH')
    trials=expand(plan)
    if not (output/'experiment_plan.json').exists():
        atomic_json(output/'experiment_plan.json',{'signature':signature,'created_at':now(),'policy':policy,'trials':trials,'code_sha256':code,'environment':environment(),'evidence_level':LEVELS[mode],'profile_input':aux_report,'command_argv':sys.argv})
    atomic_json(output/'sample_manifest.json',sample)
    verify_plan(output)
    plan_hash=object_hash(read_json(output/'experiment_plan.json'))
    prepared={s['id']:prepare(keys,arrays,s,manifest,plan['resources']) for s in plan['spaces']}
    if plan.get('common_eligible',False):
        common=np.logical_and.reduce([v[1] for v in prepared.values()])
        for sid,(x,eligible,reasons,trans) in prepared.items():
            reasons[eligible&~common]='COMMON_SPACE_ELIGIBILITY_EXCLUSION';eligible&=common
    outcomes=[]
    for trial in trials:
        budget.check();cfg=trial['config'];seed=trial['seed'];scope=trial['scope'];spec=next(s for s in plan['spaces'] if s['id']==cfg['space'])
        if cfg.get('metric',spec['metric'])!=spec['metric']:raise ValueError('ALGORITHM_SPACE_METRIC_MISMATCH')
        cfg={**cfg,'metric':spec['metric']};trial={**trial,'config':cfg};tid='trial_'+object_hash(trial)[:16];sig=object_hash([signature,trial]);partial,resumed=begin(output/'trials',tid,sig)
        if resumed:
            outcomes.append({'trial_id':tid,'status':'PASS','resumed':True});continue
        atomic_json(partial/'config_snapshot.json',{'trial':trial,'space':spec,'policy_sha256':object_hash(policy),'resources':plan['resources']})
        try:
            x,eligible,reasons,transform=prepared[cfg['space']];scope_mask=np.array([scope=='pooled' or r['category']==scope for r in keys]);ix=np.flatnonzero(eligible&scope_mask);local_keys=[keys[i] for i in ix];xx=x[ix]
            labels,model,cost=fit_cluster(xx,cfg,seed,plan['resources'])
            eval_ix=fixed_evaluation(local_keys,plan['evaluation']['max_rows'],plan['evaluation']['seed'])
            metrics=internal_metrics(xx,labels,local_keys,spec['metric'],eval_ix,plan['resources']);metrics['cost']=cost
            if plan['evaluation'].get('silhouette_metric')=='cosine' and spec['metric']!='cosine':
                extra=internal_metrics(xx,labels,local_keys,'cosine',eval_ix,plan['resources'])
                metrics['silhouette_euclidean']=metrics['silhouette']
                metrics['silhouette']=extra['silhouette']
                metrics['silhouette_samples']=extra['silhouette_samples']
                metrics['silhouette_metric']='cosine'
                metrics['ch_dbi_geometry']='euclidean'
            if hasattr(model,'cluster_centers_'):np.save(partial/'cluster_centers.npy',model.cluster_centers_,allow_pickle=False)
            stable=stability(xx,labels,local_keys,cfg,seed,plan,spec['normalization'])
            subset_arrays={k:v[ix] for k,v in arrays.items() if k in ('numeric_compact','numeric_missing')}
            local_aux={key_tuple(k):aux[key_tuple(k)] for k in local_keys if key_tuple(k) in aux}
            prof=make_profiles(local_keys,xx,labels,local_aux,subset_arrays,mode,seed,policy.get('allow_reviewed_excerpts',False))
            assignments=[None]*len(keys);why=reasons.copy();why[~scope_mask]='OUTSIDE_CATEGORY_SCOPE'
            for i,label in zip(ix,labels):assignments[i]=int(label)
            write_assignments(partial/'assignments.csv',keys,assignments,why)
            atomic_json(partial/'metrics.json',metrics);atomic_json(partial/'stability.json',stable);atomic_json(partial/'profiles.json',prof);atomic_json(partial/'transform.json',transform)
            atomic_json(partial/'decision_trace.json',{'status':'INCONCLUSIVE','reason':'Synthetic evidence cannot select scientific model' if mode=='synthetic' else 'Human multi-criteria review required; no automatic winner','policy_status':policy['status'],'policy_sha256':object_hash(policy),'research_contract':policy.get('research_contract','R01_R02_REMAIN_OPEN'),'selected':False})
            if plan.get('figures',True):
                from .plots import figures
                figures(partial/'figures',xx,labels,metrics,prof,model,'SYNTHETIC_FIXTURE_ONLY' if mode=='synthetic' else 'REAL_TRAIN_PILOT')
            budget.check()
            publish(partial,{'schema_version':'stage-b-trial-1','plan_sha256':plan_hash,'signature':sig,'trial_id':tid,'evidence_level':LEVELS[mode],'sample_key_hash':sample['key_hash'],'eligible_key_hash':key_hash(local_keys),'source_manifest_sha256':sample['source_manifest_sha256'],'n_rows':len(keys),'n_eligible':len(ix),'trial':trial,'space':spec,'profile_input':aux_report,'code_sha256':code,'mid_model_resume':False})
            outcomes.append({'trial_id':tid,'status':'PASS','resumed':False})
        except (ValueError,ResourceLimit) as exc:
            status='SKIPPED_RESOURCE_LIMIT' if isinstance(exc,ResourceLimit) else 'INVALID'
            atomic_json(partial/'failure.json',{'status':status,'reason':str(exc),'at':now(),'mid_model_resume':False});outcomes.append({'trial_id':tid,'status':status,'reason':str(exc),'resumed':False})
    receipt={'plan_sha256':plan_hash,'status':'PASS' if all(o['status']=='PASS' for o in outcomes) else 'INCOMPLETE','evidence_level':LEVELS[mode],'signature':signature,'outcomes':outcomes,'elapsed_seconds':time.perf_counter()-started,'peak_observed_rss_bytes':budget.peak,'raw_scan':False,'real_amazon_clustering_claim':mode=='real','claim_scope':'supplied selected train subset only' if mode=='real' else 'synthetic only','model_resume':'completed trial verification/reuse; interrupted trial restarted from beginning','completed_at':now()}
    atomic_json(output/'execution_receipts'/(now().replace(':','').replace('.','_')+'.json'),receipt)
    atomic_json(output/'run_receipt.json',receipt)
    return receipt


def compare(output):
    output=Path(output);verify_experiment(output);items=[]
    for p in sorted((output/'trials').glob('trial_*')):
        if p.name.endswith('.partial') or not p.is_dir():continue
        m=verify_trial(p);metrics=read_json(p/'metrics.json');keys,labels=read_assignments(p)
        keep=[i for i,v in enumerate(labels) if v is not None]
        items.append({'path':p,'manifest':m,'metrics':metrics,'keys':[keys[i] for i in keep],'labels':[labels[i] for i in keep]})
    policy=read_json(output/'experiment_plan.json')['policy'];assessment=assess(items,policy);thresholds=assessment['thresholds']
    seed_pairs=[];mb_pairs=[];rows=[]
    for item in items:
        m=item['manifest'];met=item['metrics'];rows.append({'trial_id':m['trial_id'],'trial':m['trial'],'space':m['space'],'sample_key_hash':m['sample_key_hash'],'eligible_key_hash':m['eligible_key_hash'],'silhouette':met['silhouette'],'inertia':met['cost'].get('inertia'),'coverage':met['coverage'],'cluster_sizes':met['cluster_sizes'],'cost':met['cost']})
    for a,b in combinations(items,2):
        ma,mb=a['manifest'],b['manifest'];ca,cb=ma['trial']['config'],mb['trial']['config'];common=ma['sample_key_hash']==mb['sample_key_hash'] and ma['eligible_key_hash']==mb['eligible_key_hash'] and ma['space']==mb['space'] and ma['trial']['scope']==mb['trial']['scope']
        if not common:continue
        if ca==cb and ma['trial']['seed']!=mb['trial']['seed']:
            seed_pairs.append({'a':ma['trial_id'],'b':mb['trial_id'],**aligned_ari(a['keys'],a['labels'],b['keys'],b['labels'])})
        if {ca['algorithm'],cb['algorithm']}=={'kmeans','minibatch'} and ca.get('k')==cb.get('k') and ma['trial']['seed']==mb['trial']['seed']:
            km,mini=(a,b) if ca['algorithm']=='kmeans' else (b,a);x,y=km['metrics'],mini['metrics'];fair=x['evaluation_retained_key_hash']==y['evaluation_retained_key_hash'];ratio=y['cost']['inertia']/x['cost']['inertia'] if x['cost']['inertia']>0 else None
            drop=x['silhouette']['value']-y['silhouette']['value'] if x['silhouette']['status']==y['silhouette']['status']=='DEFINED' else None
            mb_pairs.append({'kmeans':km['manifest']['trial_id'],'minibatch':mini['manifest']['trial_id'],'fair_fixed_evaluation':fair,'inertia_ratio':ratio,'silhouette_drop':drop,'proposed_threshold_diagnostic':bool(fair and ratio is not None and drop is not None and ratio<=thresholds['minibatch_max_inertia_ratio'] and drop<=thresholds['minibatch_max_silhouette_drop']),'status':'INCONCLUSIVE','reason':'Thresholds alone are insufficient; stability, profiles, scientific approval and comparable measured cost required'})
    values=[r['value'] for r in seed_pairs if r['status']=='DEFINED']
    result={'status':'PASS','evidence_level':read_json(output/'experiment_plan.json')['evidence_level'],'scientific_selection':'INCONCLUSIVE','rows':rows,'multi_criteria_assessment':assessment,'seed_stability':seed_pairs,'seed_ari_summary':{'median':float(np.median(values)) if values else None,'q1':float(np.quantile(values,.25)) if values else None,'pooling':'descriptive across registered candidate families; inspect pair family IDs before interpretation'},'minibatch_comparisons':mb_pairs,'cross_space_rule':'No global scalar ranking; raw inertia/CH/DBI are not comparable across distinct geometries','normalization_comparisons':[],'scope_comparisons':[]}
    for a,b in combinations(items,2):
        ma,mb=a['manifest'],b['manifest'];ca,cb=ma['trial']['config'],mb['trial']['config']
        same_algo={k:v for k,v in ca.items() if k!='space'}=={k:v for k,v in cb.items() if k!='space'} and ma['trial']['seed']==mb['trial']['seed']
        if same_algo and ma['space']['source']==mb['space']['source'] and ma['space']['normalization']!=mb['space']['normalization'] and ma['trial']['scope']==mb['trial']['scope']:
            result['normalization_comparisons'].append({'a':ma['trial_id'],'b':mb['trial_id'],**aligned_ari(a['keys'],a['labels'],b['keys'],b['labels'])})
        if ca==cb and ma['trial']['seed']==mb['trial']['seed'] and (ma['trial']['scope']=='pooled')!=(mb['trial']['scope']=='pooled'):
            result['scope_comparisons'].append({'a':ma['trial_id'],'b':mb['trial_id'],**aligned_ari(a['keys'],a['labels'],b['keys'],b['labels'])})
    atomic_json(output/'comparison.json',result);return result


def report(output):
    output=Path(output);result=compare(output);receipt=read_json(output/'run_receipt.json');level=receipt['evidence_level'];lines=['# Stage B experiment report', '','SYNTHETIC_FIXTURE_ONLY' if level=='SYNTHETIC_CONTRACT_TEST' else level,'','Scientific selection: INCONCLUSIVE. No Amazon population claim.','','|Trial|Algorithm|Space|k|Silhouette|Coverage|Fit seconds|','|---|---|---|---:|---:|---:|---:|']
    for r in result['rows']:
        c=r['trial']['config'];lines.append(f"|{r['trial_id']}|{c['algorithm']}|{r['space']['id']}|{c.get('k','—')}|{r['silhouette'].get('value')}|{r['coverage']:.4f}|{r['cost']['elapsed_fit_seconds']:.6f}|")
    lines+=['','Raw scores across different spaces are not a winner criterion. Inspect profiles, seed/subsample stability, exclusions, runtime and decision_trace.json.','', 'All images in a synthetic experiment are labeled SYNTHETIC_FIXTURE_ONLY.']
    from .plots import overview_figures
    overview_figures(output/'figures',result,'SYNTHETIC_FIXTURE_ONLY' if level=='SYNTHETIC_CONTRACT_TEST' else level)
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return {'status':'PASS','report':str(output/'report.md'),'n_trials':len(result['rows'])}
