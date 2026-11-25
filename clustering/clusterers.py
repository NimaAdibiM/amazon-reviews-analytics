import time

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from .common import Budget


def fit_cluster(x,config,seed,limits):
    budget=Budget(limits);budget.matrix(x,4);n=len(x) if not sparse.issparse(x) else x.shape[0]
    name=config['algorithm'];metric=config.get('metric','euclidean');k=config.get('k')
    if n<3:raise ValueError('INSUFFICIENT_ELIGIBLE_ROWS')
    if k is not None and (type(k) is not int or not 2<=k<n):raise ValueError('INVALID_K')
    if name in ('kmeans','minibatch'):
        budget.check(8*x.shape[1]*(int(k or 0)+3)*3)
        if metric!='euclidean':raise ValueError('KMEANS_REQUIRES_EUCLIDEAN_OBJECTIVE')
        if not isinstance(config.get('n_init'),int) or config['n_init']<1:raise ValueError('EXPLICIT_N_INIT_REQUIRED')
        args={'n_clusters':k,'random_state':seed,'n_init':config['n_init'],'max_iter':config['max_iter'],'tol':config.get('tol',1e-4)}
        if name=='kmeans':model=KMeans(**args,algorithm='lloyd')
        else:
            if config['init_size']<k or config['batch_size']<1:raise ValueError('INVALID_MINIBATCH_PARAMETERS')
            model=MiniBatchKMeans(**args,batch_size=config['batch_size'],init_size=config['init_size'],reassignment_ratio=config['reassignment_ratio'],max_no_improvement=config['max_no_improvement'],compute_labels=True)
    elif name=='agglomerative':
        budget.pairwise(n)
        if sparse.issparse(x):raise ValueError('AGGLOMERATIVE_DENSE_REQUIRED_NO_IMPLICIT_DENSIFY')
        if config['linkage']=='ward' and metric!='euclidean':raise ValueError('WARD_REQUIRES_EUCLIDEAN')
        if config['linkage'] not in ('ward','average'):raise ValueError('UNAPPROVED_LINKAGE')
        threshold=config.get('distance_threshold')
        model=AgglomerativeClustering(n_clusters=None if threshold is not None else k,distance_threshold=threshold,linkage=config['linkage'],metric=metric,compute_distances=True)
    elif name=='dbscan':
        budget.pairwise(n)
        if type(config.get('min_samples')) is not int or not 2<=config['min_samples']<=n:raise ValueError('INVALID_MIN_SAMPLES')
        eps=config.get('eps')
        if eps is None:
            q=config.get('eps_quantile')
            if not isinstance(q,(float,int)) or not 0<q<1:raise ValueError('EPS_OR_TRAIN_QUANTILE_REQUIRED')
            with threadpool_limits(limits=limits['threads']):
                distances=NearestNeighbors(n_neighbors=config['min_samples'],metric=metric,n_jobs=1).fit(x).kneighbors(x,return_distance=True)[0][:,-1]
            eps=float(np.quantile(distances,q));eps=max(eps,np.finfo(float).eps)
        if not np.isfinite(eps) or eps<=0:raise ValueError('INVALID_EPS')
        model=DBSCAN(eps=eps,min_samples=config['min_samples'],metric=metric,n_jobs=1)
    else:raise ValueError('UNKNOWN_ALGORITHM')
    started=time.perf_counter()
    with threadpool_limits(limits=limits['threads']):labels=model.fit_predict(x)
    budget.check()
    diagnostics={'cluster_count_actual_excluding_noise':int(len(set(labels)-{-1})),'algorithm':name,'elapsed_fit_seconds':time.perf_counter()-started,'rss_after_fit_bytes':budget.peak,'n_rows':n,'geometry':metric,'inductive_assignment_available':name in ('kmeans','minibatch'),'mid_model_resume':False}
    if hasattr(model,'inertia_'):diagnostics.update({'inertia':float(model.inertia_),'iterations':int(model.n_iter_),'iteration_limit_reached':bool(model.n_iter_>=config['max_iter']),'objective':'Euclidean squared distance; L2 preprocessing does not make this spherical K-Means'})
    if name=='minibatch':diagnostics['n_init_semantics']='Initializations evaluated; one MiniBatch optimization per seed, independent seeds required'
    if name=='dbscan':
        diagnostics.update({'eps':float(eps),'min_samples_includes_self':True,'n_core_samples':int(len(model.core_sample_indices_)),'out_of_sample':'UNSUPPORTED_NO_PREDICT_CLAIM'})
        if config.get('eps') is None:diagnostics['train_k_distances_sorted']=np.sort(distances).tolist()
    return labels.astype(int),model,diagnostics
