import numpy as np
from sklearn.decomposition import PCA
from .partner import features, views


def numeric_matrix(rows):
    x = np.asarray([r['numeric'] for r in rows], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(views['full']):
        raise ValueError('NUMERIC_WIDTH_MISMATCH')
    if np.isinf(x).any():
        raise ValueError('NONFINITE_NUMERIC_INPUT')
    for i, name in enumerate(views['full']):
        if name not in features.SCALABLE_FIELDS and not np.isin(x[np.isfinite(x[:, i]), i], (0, 1)).all():
            raise ValueError(f'INVALID_BINARY: {name}')
    return x


def fit_numeric(rows, scaler, pca_k=None, tolerance=1e-10):
    if not rows or any(r['split'] != 'train' for r in rows):
        raise ValueError('FIT_REQUIRES_TRAIN_ONLY')
    full_names = list(views['full'])
    compact = list(views['compact'])
    indices = [full_names.index(n) for n in compact]
    x = numeric_matrix(rows)
    missing = ~np.isfinite(x)
    fill = np.zeros(40)
    policies = {}
    for i,name in enumerate(full_names):
        observed = x[~missing[:,i],i]
        if name in features.SCALABLE_FIELDS:
            if scaler['features'][name]['scale'] is None and len(observed):
                raise ValueError(f'UNFITTED_SCALER_HAS_VALUES: {name}')
            policies[name] = 'zero_in_existing_train_zscore_with_indicator'
        else:
            fill[i] = float(np.mean(observed) > .5) if len(observed) else 0
            policies[name] = 'train_mode_binary_with_indicator'
    clean = np.where(missing, fill, x)
    values = clean[:,indices]
    observed_counts = (~missing[:,indices]).sum(axis=0)
    variance = values.var(axis=0)
    keep = (observed_counts > 0) & (variance > tolerance)
    names = [n for n,k in zip(compact,keep) if k] + [n+'__missing' for n in compact]
    distance = np.concatenate((values[:,keep], missing[:,indices].astype(float)),axis=1)
    pca = None
    if pca_k is not None:
        if type(pca_k) is not int or not 1 <= pca_k < min(distance.shape) or distance.var(axis=0).sum() <= tolerance:
            raise ValueError('INVALID_PCA_K_OR_VARIANCE')
        pca = PCA(n_components=pca_k, svd_solver='full').fit(distance)
    unfitted = [i for i, name in enumerate(full_names) if name in features.SCALABLE_FIELDS and scaler['features'][name]['scale'] is None]
    return {'unfitted_indices':unfitted, 'fill':fill, 'indices':indices, 'keep':keep, 'pca':pca, 'policies':policies,
            'distance_feature_names':names, 'compact_feature_names':compact,
            'report':{'fit_split':'train','n_fit':len(rows), 'compact_variance':variance.tolist(),
                      'all_missing':[n for n,c in zip(compact,observed_counts) if c==0],
                      'excluded_from_distance':[n for n,k in zip(compact,keep) if not k],
                      'constant_tolerance':tolerance, 'rescaled':False,
                      'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist() if pca else None}}


def transform_numeric(fitted, rows):
    x = numeric_matrix(rows)
    missing = ~np.isfinite(x)
    for index in fitted.get('unfitted_indices', ()):
        if not missing[:, index].all():
            raise ValueError('UNFITTED_SCALER_HAS_VALUES')
    clean = np.where(missing,fitted['fill'],x).astype(np.float32)
    compact = clean[:,fitted['indices']]
    mask = missing[:,fitted['indices']]
    distance = np.concatenate((compact[:,fitted['keep']],mask.astype(np.float32)),axis=1)
    result = {'numeric_compact':compact,'numeric_missing':mask,'numeric_distance':distance,'numeric_full_imputed':clean}
    for group,names in features.FEATURE_GROUPS.items():
        index = [views['full'].index(n) for n in names]
        result['numeric_'+group] = clean[:,index]
        result['numeric_'+group+'_missing'] = missing[:,index]
    if fitted['pca'] is not None:
        result['numeric_pca'] = fitted['pca'].transform(distance).astype(np.float32)
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('NONFINITE_NUMERIC_OUTPUT')
    return result
