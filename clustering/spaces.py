import re

import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler, normalize

from .common import Budget, matrix_hash, require_train


ALLOWED=re.compile(r'^(tfidf_[A-Za-z0-9_]+|semantic_(bge|e5)_(M|MT|C)(_l2|_pca|_tie_group_mean)?|numeric_(distance|compact|pca|rating)|fusion_[A-Za-z0-9_]+)$')


def prepare(keys,arrays,spec,manifest,limits):
    require_train(keys);name=spec['source']
    if not ALLOWED.fullmatch(name):raise ValueError('FORBIDDEN_FEATURE_SPACE')
    x=arrays[name]
    if x.ndim!=2 or x.shape[0]!=len(keys) or not np.isfinite(x.data if sparse.issparse(x) else x).all():
        raise ValueError('INVALID_FEATURE_MATRIX')
    budget=Budget(limits);budget.matrix(x,3);budget.check(8*x.shape[1]*6)
    eligible=np.ones(len(keys),dtype=bool);reasons=np.full(len(keys),'',dtype=object)
    if name.startswith('semantic_'):
        prefix='_'.join(name.split('_')[:2]);mask=arrays.get(prefix+'_mask')
        if mask is None or mask.shape!=(len(keys),5) or mask.dtype!=bool:raise ValueError('SEMANTIC_MASK_REQUIRED')
        absent=~mask.any(axis=1);eligible[absent]=False;reasons[absent]='ALL_EARLY_TEXT_EMPTY'
        if name.split('_')[2]=='C':
            policy=spec.get('missing_policy')
            if policy not in ('complete_reviews','zero_padding_diagnostic'):raise ValueError('C_MASK_POLICY_REQUIRED')
            if policy=='complete_reviews':
                missing=~mask.all(axis=1);eligible[missing]=False;reasons[missing]='C_INCOMPLETE_REVIEW_MASK'
    if name.startswith('fusion_'):
        name_config=name[len('fusion_'):]
        configs=manifest['metadata'].get('provenance',{}).get('config',{}).get('fusion',[])
        weights=next((v for v in configs if v['name']==name_config),None)
        if weights is None or spec.get('fusion_weights')!={k:weights[k] for k in ('text_space','text_weight','numeric_weight')}:
            raise ValueError('FUSION_WEIGHTS_NOT_EXPLICIT_OR_MISMATCH')
    norm=np.asarray(x.multiply(x).sum(axis=1)).ravel() if sparse.issparse(x) else np.sum(x*x,axis=1)
    if not name.startswith('numeric_') or spec['metric']=='cosine' or spec['normalization']=='l2':
        empty=norm<=1e-24;eligible[empty]=False;reasons[empty]='ZERO_VECTOR'
    out=x.copy()
    transform={'operation_order':'eligibility, optional train PCA, requested normalization, zero check','normalization':spec['normalization'],'fit_scope':'selected_train_only','already_scaled_numeric':name.startswith('numeric_'),'source':name}
    if name.startswith('semantic_') and name.split('_')[2]=='MT':
        slope_name=prefix+'_slope_defined'+('_tie_group_mean' if name.endswith('tie_group_mean') else '')
        slopes=arrays.get(slope_name)
        if slopes is None or slopes.dtype!=bool or slopes.shape!=(len(keys),):raise ValueError('MT_SLOPE_DEFINED_FLAG_REQUIRED')
        transform['n_slope_undefined']=int((~slopes).sum())
        transform['slope_policy']='A zero slope retained with explicit undefined count; not evidence of no temporal change'
    if spec.get('pca_components') is not None:
        from sklearn.decomposition import PCA
        k=spec['pca_components']
        if sparse.issparse(out):raise ValueError('PCA_REQUIRES_DENSE_NO_IMPLICIT_DENSIFY')
        if type(k) is not int or not 1<=k<min(int(eligible.sum()),out.shape[1]):raise ValueError('INVALID_PCA_COMPONENTS')
        pca=PCA(n_components=k,svd_solver='full');pca.fit(out[eligible]);out=pca.transform(out)
        transform['pca']={'n_components':k,'fit_scope':'selected eligible train only','explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'mean':pca.mean_.tolist(),'components':pca.components_.tolist()}
    if spec['normalization']=='l2':out=normalize(out,copy=False)
    elif spec['normalization']=='standard_train':
        if not eligible.any():raise ValueError('NO_ELIGIBLE_ROWS')
        scaler=StandardScaler(with_mean=not sparse.issparse(out));scaler.fit(out[eligible]);out=scaler.transform(out)
        transform['mean']=scaler.mean_.tolist();transform['scale']=scaler.scale_.tolist()
    if spec['metric']=='cosine':
        norms=np.asarray(out.multiply(out).sum(axis=1)).ravel() if sparse.issparse(out) else np.sum(out*out,axis=1)
        eligible[norms<=1e-24]=False;reasons[norms<=1e-24]='ZERO_AFTER_TRANSFORM'
    budget.matrix(out,3)
    transform.update({'matrix_hash':matrix_hash(out),'eligible_rows':int(eligible.sum()),'excluded_rows':int((~eligible).sum()),'shape':list(out.shape),'sparse':sparse.issparse(out)})
    from .geometry import audit_geometry
    transform['geometry_audit']=audit_geometry(out,keys,eligible,spec['metric'],limits)
    return out,eligible,reasons,transform
