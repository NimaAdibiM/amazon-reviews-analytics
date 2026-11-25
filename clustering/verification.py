from pathlib import Path

from .artifacts import read_assignments, verify_trial
from .common import key_hash, object_hash, read_json, require_train


def verify_plan(root):
    from .runner import expand
    root=Path(root);p=read_json(root/'experiment_plan.json');s=read_json(root/'sample_manifest.json')
    require_train(s['keys'])
    if key_hash(s['keys'])!=s['key_hash'] or len(s['keys'])!=s['n_rows']:raise ValueError('EXPERIMENT_SAMPLE_MISMATCH')
    modes={'SYNTHETIC_CONTRACT_TEST':'synthetic','REAL_TRAIN_PILOT':'real'}
    if p['evidence_level'] not in modes:raise ValueError('EXPERIMENT_LEVEL_INVALID')
    signature=object_hash({'policy':p['policy'],'source':s['source_manifest_sha256'],'sample':s['key_hash'],'code':p['code_sha256'],'profiles':p['profile_input'],'mode':modes[p['evidence_level']]})
    if signature!=p['signature']:raise ValueError('EXPERIMENT_PLAN_SIGNATURE_MISMATCH')
    if p['trials']!=expand(p['policy']['plan']) or not p['trials']:raise ValueError('EXPERIMENT_PLAN_TRIALS_MISMATCH')
    expected={}
    for trial in p['trials']:
        spec=next(v for v in p['policy']['plan']['spaces'] if v['id']==trial['config']['space'])
        if trial['config'].get('metric',spec['metric'])!=spec['metric']:raise ValueError('EXPERIMENT_METRIC_MISMATCH')
        t={**trial,'config':{**trial['config'],'metric':spec['metric']}};tid='trial_'+object_hash(t)[:16]
        if tid in expected:raise ValueError('EXPERIMENT_DUPLICATE_TRIAL_ID')
        expected[tid]=(t,spec)
    return p,s,expected


def verify_experiment(root):
    root=Path(root)
    try:
        p,s,expected=verify_plan(root);r=read_json(root/'run_receipt.json');ph=object_hash(p)
        if r['status']!='PASS':raise ValueError('EXPERIMENT_INCOMPLETE_RECEIPT')
        if r['signature']!=p['signature'] or r['evidence_level']!=p['evidence_level']:raise ValueError('EXPERIMENT_RECEIPT_SIGNATURE_MISMATCH')
        if 'plan_sha256' in r and r['plan_sha256']!=ph:raise ValueError('EXPERIMENT_PLAN_HASH_MISMATCH')
        outcomes=r['outcomes'];ids=[v['trial_id'] for v in outcomes]
        if len(ids)!=len(set(ids)) or set(ids)!=set(expected) or any(v['status']!='PASS' for v in outcomes):raise ValueError('EXPERIMENT_RECEIPT_OUTCOMES_MISMATCH')
        folder=root/'trials';present={v.name for v in folder.iterdir() if v.name!='interrupted'}
        if present!=set(expected):raise ValueError('EXPERIMENT_TRIAL_SET_MISMATCH: missing='+str(sorted(set(expected)-present))+' extra='+str(sorted(present-set(expected))))
        for tid,(trial,spec) in expected.items():
            m=verify_trial(folder/tid,object_hash([p['signature'],trial]))
            for name,value in {'trial_id':tid,'trial':trial,'space':spec,'code_sha256':p['code_sha256'],'profile_input':p['profile_input'],'sample_key_hash':s['key_hash'],'source_manifest_sha256':s['source_manifest_sha256'],'evidence_level':p['evidence_level']}.items():
                if m.get(name)!=value:raise ValueError('EXPERIMENT_TRIAL_BINDING_MISMATCH: '+name)
            if 'plan_sha256' in r and m.get('plan_sha256')!=ph:raise ValueError('EXPERIMENT_TRIAL_PLAN_HASH_MISMATCH')
            keys,labels=read_assignments(folder/tid);eligible=[k for k,label in zip(keys,labels) if label is not None]
            if key_hash(eligible)!=m['eligible_key_hash'] or len(eligible)!=m['n_eligible']:raise ValueError('EXPERIMENT_ELIGIBLE_KEYS_MISMATCH')
        return {'status':'PASS','evidence_level':p['evidence_level'],'scope':'complete_experiment','verified_trials':len(expected),'trial_ids':sorted(expected),'signature':p['signature'],'plan_sha256':ph,'binding':'full_plan_hash' if 'plan_sha256' in r else 'legacy_semantic_signature'}
    except (KeyError,TypeError,StopIteration,OSError) as exc:
        raise ValueError('EXPERIMENT_INCOMPLETE_OR_INVALID: '+str(exc)) from exc
