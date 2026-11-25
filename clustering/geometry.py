import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances

from .common import Budget, key_hash
from .metrics import fixed_evaluation


def audit_geometry(x,keys,eligible,metric,limits):
    budget=Budget(limits);n,d=x.shape;budget.check(d*8*6)
    ix=np.flatnonzero(eligible)
    if len(ix)==0:return {'status':'UNDEFINED','reason':'NO_ELIGIBLE_ROWS'}
    use=ix[fixed_evaluation([keys[i] for i in ix],min(128,len(ix)),9001)];xx=x[use]
    norms=np.sqrt(np.asarray(xx.multiply(xx).sum(axis=1)).ravel()) if sparse.issparse(xx) else np.linalg.norm(xx,axis=1)
    mean=np.asarray(xx.mean(axis=0)).ravel();second=np.asarray(xx.multiply(xx).mean(axis=0)).ravel() if sparse.issparse(xx) else np.mean(xx*xx,axis=0)
    variance=np.maximum(0,second-mean**2);budget.pairwise(len(use));dist=pairwise_distances(xx,metric=metric);pairs=dist[np.triu_indices(len(use),1)]
    np.fill_diagonal(dist,np.inf);nearest=np.min(dist,axis=1) if len(use)>1 else np.array([])
    quant=lambda a:np.quantile(a,[0,.25,.5,.75,1]).tolist() if len(a) else None
    return {'status':'PASS','shape':[n,d],'dtype':str(x.dtype),'is_sparse':sparse.issparse(x),'sparsity':1-float(x.nnz)/(n*d) if sparse.issparse(x) else float(np.mean(x==0)),'audit_sample_n':len(use),'audit_sample_key_hash':key_hash([keys[i] for i in use]),'metric':metric,'row_norm_quantiles':quant(norms),'feature_variance_quantiles':quant(variance),'pair_distance_quantiles':quant(pairs),'distance_coefficient_of_variation':float(pairs.std()/pairs.mean()) if len(pairs) and pairs.mean()>0 else None,'sample_nearest_neighbor_quantiles':quant(nearest),'neighbor_scope':'nearest among bounded audit sample, not full-population nearest neighbor','seed':9001}
