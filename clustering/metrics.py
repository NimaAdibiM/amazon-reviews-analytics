import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.metrics.pairwise import pairwise_distances

from .common import Budget, key_hash, key_tuple, object_hash


def undefined(reason):return {'status':'UNDEFINED','value':None,'reason':reason}


def defined(value):
    return {'status':'DEFINED','value':float(value)} if np.isfinite(value) else undefined('NONFINITE_RESULT')


def fixed_evaluation(keys,size,seed):
    return np.array(sorted(sorted(range(len(keys)),key=lambda i:object_hash([seed,*key_tuple(keys[i])]))[:size]),dtype=int)


def internal_metrics(x,labels,keys,metric,eval_indices,limits):
    budget=Budget(limits);labels=np.asarray(labels);assigned=labels>=0
    groups=np.unique(labels[assigned]);n=int(assigned.sum());k=len(groups)
    result={'n_rows':len(labels),'n_assigned':n,'coverage':n/len(labels) if len(labels) else 0,'noise_fraction':float(np.mean(labels==-1)),'n_clusters':k,'cluster_sizes':{str(c):int(np.sum(labels==c)) for c in groups},'evaluation_base_key_hash':key_hash([keys[i] for i in eval_indices]),'metric':metric,'small_cluster_advisory':bool(any(np.sum(labels==c)<10 for c in groups))}
    ix=np.asarray([i for i in eval_indices if labels[i]>=0],dtype=int);sub=labels[ix];uk=np.unique(sub)
    result['evaluation_retained_key_hash']=key_hash([keys[i] for i in ix]);result['n_evaluation_retained']=len(ix)
    scores=[]
    if 2<=len(uk)<len(ix):
        budget.pairwise(len(ix));values=silhouette_samples(x[ix],sub,metric=metric)
        result['silhouette']=defined(values.mean());scores=[{'index':int(i),'label':int(labels[i]),'value':float(v)} for i,v in zip(ix,values)]
    else:result['silhouette']=undefined('REQUIRES_2_TO_N_MINUS_1_NONNOISE_CLUSTERS_ON_FIXED_SAMPLE')
    result['silhouette_samples']=scores
    result['degeneracy']='ALL_NOISE' if n==0 else 'ONE_CLUSTER' if k==1 else 'SINGLETON_CLUSTERS' if k==n else 'NONE'
    if metric!='euclidean':
        result['calinski_harabasz']=undefined('NON_EUCLIDEAN_GEOMETRY');result['davies_bouldin']=undefined('NON_EUCLIDEAN_GEOMETRY')
    elif not 2<=k<n:
        result['calinski_harabasz']=undefined('INVALID_CLUSTER_COUNT');result['davies_bouldin']=undefined('INVALID_CLUSTER_COUNT')
    else:
        budget.check(8*k*x.shape[1]*3)
        means=np.stack([np.asarray(x[labels==c].mean(axis=0)).ravel() for c in groups]);overall=np.asarray(x[assigned].mean(axis=0)).ravel()
        counts=np.array([np.sum(labels==c) for c in groups]);within=0;spread=[]
        for c,mean in zip(groups,means):
            d=pairwise_distances(x[labels==c],mean[None,:],metric='euclidean').ravel();within+=float(d@d);spread.append(float(d.mean()))
        between=float(np.sum(counts[:,None]*(means-overall)**2))
        result['calinski_harabasz']=defined(between/(k-1)/(within/(n-k))) if within>1e-20 else undefined('ZERO_WITHIN_VARIANCE')
        budget.pairwise(k);dist=pairwise_distances(means);off=~np.eye(k,dtype=bool)
        if np.any(dist[off]<=1e-12):result['davies_bouldin']=undefined('COINCIDENT_CENTROIDS')
        else:
            np.fill_diagonal(dist,np.inf);ratios=(np.array(spread)[:,None]+np.array(spread)[None,:])/dist
            result['davies_bouldin']=defined(ratios.max(axis=1).mean())
    return result


def aligned_ari(keys_a,labels_a,keys_b,labels_b,min_overlap=10):
    a={key_tuple(r):int(v) for r,v in zip(keys_a,labels_a)};b={key_tuple(r):int(v) for r,v in zip(keys_b,labels_b)}
    overlap=sorted(a.keys()&b.keys());valid=[key for key in overlap if a[key]>=0 and b[key]>=0]
    out={'n_overlap':len(overlap),'n_compared_nonnoise':len(valid),'noise_agreement':sum((a[key]==-1)==(b[key]==-1) for key in overlap)/len(overlap) if overlap else None,'key_hash':object_hash(valid),'alignment':'composite-key intersection; labels are permutation invariant'}
    aa=[a[key] for key in valid];bb=[b[key] for key in valid]
    if len(valid)<min_overlap or not 2<=len(set(aa))<len(valid) or not 2<=len(set(bb))<len(valid):out.update({'status':'INSUFFICIENT_OVERLAP_OR_CLUSTERS','value':None})
    else:out.update({'status':'DEFINED','value':float(adjusted_rand_score(aa,bb))})
    return out
