from collections import defaultdict

import numpy as np

from .common import object_hash, read_json


PROPOSED={'seed_ari_median_warning':.75,'seed_ari_q1_warning':.60,'min_cluster_warning':10,'minibatch_max_inertia_ratio':1.05,'minibatch_max_silhouette_drop':.02}


def assess(items,policy):
    thresholds=policy.get('selection_policy',{}).get('proposed_not_approved',PROPOSED)
    if policy['status']=='APPROVED_FOR_REAL_PILOT':
        thresholds=policy.get('selection_policy',{}).get('approved_thresholds',{})
        if set(PROPOSED)-set(thresholds):raise ValueError('APPROVED_SELECTION_THRESHOLDS_REQUIRED')
    families=defaultdict(list)
    for item in items:
        m=item['manifest'];trial=m['trial'];families[object_hash([trial['config'],trial['scope'],m['space'],m['eligible_key_hash']])].append(item)
    result=[]
    from .metrics import aligned_ari
    from itertools import combinations
    for identity,members in sorted(families.items()):
        seed_pairs=[aligned_ari(a['keys'],a['labels'],b['keys'],b['labels']) for a,b in combinations(members,2)]
        seed_values=[v['value'] for v in seed_pairs if v['status']=='DEFINED'];sub=[];issues=[];summaries=[]
        for item in members:
            m=item['metrics'];stable=read_json(item['path']/'stability.json');sub.extend(v['value'] for v in stable['subsamples'] if v.get('status')=='DEFINED')
            prof=read_json(item['path']/'profiles.json');has_profiles=all(v.get('early_profile_status')=='AVAILABLE' for v in prof['clusters'].values())
            summaries.append({'trial_id':item['manifest']['trial_id'],'silhouette':m['silhouette'],'calinski_harabasz':m['calinski_harabasz'],'davies_bouldin':m['davies_bouldin'],'coverage':m['coverage'],'sizes':m['cluster_sizes'],'cost':m['cost'],'early_profiles_available':has_profiles})
            if not has_profiles:issues.append('MISSING_EARLY_PROFILES')
            if m['silhouette']['status']!='DEFINED':issues.append('UNDEFINED_SILHOUETTE')
            if any(v<thresholds['min_cluster_warning'] for v in m['cluster_sizes'].values()):issues.append('PROPOSED_SMALL_CLUSTER_WARNING_NOT_AUTOMATIC_VETO')
        median=float(np.median(seed_values)) if seed_values else None;q1=float(np.quantile(seed_values,.25)) if seed_values else None
        if median is None:issues.append('INSUFFICIENT_SEED_REPETITIONS_OR_OVERLAP')
        elif median<thresholds['seed_ari_median_warning'] or q1<thresholds['seed_ari_q1_warning']:issues.append('STABILITY_WARNING')
        if not sub:issues.append('SUBSAMPLE_STABILITY_UNAVAILABLE')
        result.append({'family_hash':identity,'candidate':members[0]['manifest']['trial']['config'],'scope':members[0]['manifest']['trial']['scope'],'space':members[0]['manifest']['space'],'n_seeds':len(members),'seed_ari_median':median,'seed_ari_q1':q1,'subsample_ari_median':float(np.median(sub)) if sub else None,'warnings':sorted(set(issues)),'evidence':summaries,'selection_status':'INCONCLUSIVE','remaining_review':['semantic coherence and useful distinctions','coverage versus geometry and small groups','appropriate k or no supported selection','cost comparability and registered scientific policy'],'threshold_status':'APPROVED_IN_SUPPLIED_POLICY' if policy['status']=='APPROVED_FOR_REAL_PILOT' else 'PROPOSED_NOT_APPROVED'})
    return {'families':result,'thresholds':thresholds,'automatic_winner':False,'policy_sha256':object_hash(policy)}
