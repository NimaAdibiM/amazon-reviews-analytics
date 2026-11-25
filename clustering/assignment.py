import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import normalize

from .artifacts import verify_trial
from .common import Budget, read_json


def assign_new_vectors(trial_root,x,limits,early_mask=None):
    from pathlib import Path
    root=Path(trial_root);m=verify_trial(root)
    if m['trial']['config']['algorithm'] not in ('kmeans','minibatch'):raise ValueError('TRANSDUCTIVE_OUT_OF_SAMPLE_UNSUPPORTED')
    t=read_json(root/'transform.json');Budget(limits).matrix(x,3)
    eligible=np.ones(x.shape[0],bool)
    if t['source'].startswith('semantic_'):
        if early_mask is None or early_mask.shape!=(x.shape[0],5) or early_mask.dtype!=bool:raise ValueError('SEMANTIC_MASK_REQUIRED')
        eligible&=early_mask.any(axis=1)
        if t['source'].split('_')[2]=='C' and m['space'].get('missing_policy')=='complete_reviews':eligible&=early_mask.all(axis=1)
    norms=np.asarray(x.multiply(x).sum(axis=1)).ravel() if sparse.issparse(x) else np.sum(x*x,axis=1)
    if not t['source'].startswith('numeric_') or t['normalization']=='l2':eligible&=norms>1e-24
    if not np.isfinite(x.data if sparse.issparse(x) else x).all():raise ValueError('NONFINITE_ASSIGNMENT_INPUT')
    if 'pca' in t:
        if sparse.issparse(x):raise ValueError('PCA_DENSE_REQUIRED')
        x=(x-np.asarray(t['pca']['mean']))@np.asarray(t['pca']['components']).T
    if t['normalization']=='l2':x=normalize(x)
    elif t['normalization']=='standard_train':
        if sparse.issparse(x):x=x.multiply(1/np.asarray(t['scale'])).tocsr()
        else:x=(x-np.asarray(t['mean']))/np.asarray(t['scale'])
    centers=np.load(root/'cluster_centers.npy',allow_pickle=False)
    if x.shape[1]!=centers.shape[1]:raise ValueError('ASSIGNMENT_DIMENSION_MISMATCH')
    Budget(limits).check(8*centers.size*3)
    labels=np.full(x.shape[0],None,dtype=object)
    if eligible.any():labels[eligible]=pairwise_distances_argmin(x[eligible],centers,metric='euclidean')
    return labels
