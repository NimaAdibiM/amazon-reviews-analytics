from pathlib import Path

from .common import atomic_json, code_hashes, digest, object_hash, read_json


SCHEMA='stage-b-policy-1'


def validate_policy(policy,mode,source_hash=None,sample_hash=None):
    if policy.get('schema_version')!=SCHEMA or type(policy.get('version')) is not int or policy['version']<1:
        raise ValueError('POLICY_SCHEMA_INVALID')
    if mode not in ('synthetic','real'):
        raise ValueError('FULL_REAL_RUN_NOT_AUTHORIZED')
    plan=policy['plan']
    if set(plan['sampling']['categories'])-set(('Electronics','All_Beauty','Appliances','Beauty_and_Personal_Care')):
        raise ValueError('UNKNOWN_CATEGORY')
    limits=plan['resources']
    for name in ('max_rows','max_scan_rows','max_input_bytes','max_sample_bytes','max_pairwise_bytes','max_memory_mb','max_wall_seconds','max_trials','max_total_fits','max_profile_bytes','threads'):
        if type(limits.get(name)) is not int or limits[name]<1:
            raise ValueError('INVALID_RESOURCE_LIMIT: '+name)
    if limits['max_rows']>4000 or limits['max_trials']>128 or limits['max_total_fits']>1024:
        raise ValueError('PILOT_CAP_EXCEEDED_REQUIRES_NEW_SCOPE')
    if plan['sampling']['max_rows']>limits['max_rows'] or plan['sampling']['strategy'] not in ('balanced','natural'):
        raise ValueError('INVALID_SAMPLING_POLICY')
    if not plan['spaces'] or len({s['id'] for s in plan['spaces']})!=len(plan['spaces']):
        raise ValueError('SPACE_IDS_REQUIRED_UNIQUE')
    for s in plan['spaces']:
        if not s['id'].replace('_','').isalnum():
            raise ValueError('INVALID_SPACE_ID')
        if s['normalization'] not in ('none','l2','standard_train') or s['metric'] not in ('euclidean','cosine'):
            raise ValueError('INVALID_GEOMETRY')
    if not plan['seeds'] or len(set(plan['seeds']))!=len(plan['seeds']) or any(type(s) is not int for s in plan['seeds']):
        raise ValueError('INVALID_SEEDS')
    if not 0<plan['stability']['fraction']<1 or not 0<=plan['stability']['subsample_repeats']<=10:
        raise ValueError('INVALID_STABILITY_PLAN')
    if policy.get('revision_kind')=='EXPLORATORY_REVISION':
        if not policy.get('previous_policy_sha256') or not policy.get('revision_reason') or policy['version']<2:
            raise ValueError('EXPLORATORY_REVISION_REQUIRES_HISTORY')
    if mode=='real':
        if policy.get('status')!='APPROVED_FOR_REAL_PILOT':
            raise ValueError('DRAFT_POLICY_REQUIRES_APPROVAL')
        for name in ('approved_by','approval_reference','approved_at','source_manifest_sha256','sample_key_hash','code_sha256'):
            if not policy.get(name):
                raise ValueError('MISSING_POLICY_APPROVAL_BINDING: '+name)
        if source_hash is not None and policy['source_manifest_sha256']!=source_hash:
            raise ValueError('POLICY_SOURCE_HASH_MISMATCH')
        if sample_hash is not None and policy['sample_key_hash']!=sample_hash:
            raise ValueError('POLICY_SAMPLE_HASH_MISMATCH')
        from .assessment import PROPOSED
        thresholds=policy.get('selection_policy',{}).get('approved_thresholds',{})
        if set(PROPOSED)-set(thresholds) or any(not isinstance(v,(int,float)) or not __import__('math').isfinite(v) for v in thresholds.values()):
            raise ValueError('APPROVED_SELECTION_THRESHOLDS_REQUIRED')
        if policy['code_sha256']!=object_hash(code_hashes()):
            raise ValueError('POLICY_CODE_HASH_MISMATCH')
    return policy


def seal_policy(source,destination):
    source,destination=Path(source),Path(destination)
    value=read_json(source)
    validate_policy(value,'real')
    destination.mkdir(parents=True,exist_ok=True)
    sha=object_hash(value)
    target=destination/f'policy_v{value["version"]}_{sha}.json'
    prior=list(destination.glob(f'policy_v{value["version"]}_*.json'))
    if prior and any(read_json(p)!=value for p in prior):
        raise ValueError('POLICY_VERSION_ALREADY_REGISTERED')
    if value.get('revision_kind')=='EXPLORATORY_REVISION':
        found=[p for p in destination.glob('policy_v*.json') if object_hash(read_json(p))==value['previous_policy_sha256']]
        if not found:
            raise ValueError('PREVIOUS_POLICY_NOT_REGISTERED')
    if not target.exists():
        atomic_json(target,value)
    return {'policy_path':str(target),'policy_sha256':sha,'file_sha256':digest(target),'status':'PASS','approval_origin':'User-supplied approval fields; registration does not create approval'}


def check_sealed(policy,path):
    sha=object_hash(policy)
    if Path(path).name!=f'policy_v{policy["version"]}_{sha}.json':
        raise ValueError('REAL_POLICY_MUST_BE_REGISTERED_HASHED_FILE')
