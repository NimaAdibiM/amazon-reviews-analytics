import numpy as np
from sklearn.preprocessing import normalize


def aggregate(vectors, mask, groups=None, tie_policy='canonical'):
    vectors = np.asarray(vectors, dtype=np.float32)
    mask = np.asarray(mask)
    if mask.dtype != bool:
        raise ValueError('BOOLEAN_REVIEW_MASK_REQUIRED')
    if vectors.ndim != 3 or vectors.shape[1] != 5 or mask.shape != vectors.shape[:2]:
        raise ValueError('FIVE_EMBEDDINGS_SHAPE_REQUIRED')
    if not np.isfinite(vectors).all():
        raise ValueError('NONFINITE_EMBEDDINGS')
    if tie_policy not in ('canonical','group_mean'):
        raise ValueError('UNKNOWN_TIE_POLICY')
    x = vectors.copy()
    x[~mask] = 0
    positions = np.broadcast_to(np.arange(1,6,dtype=np.float32), mask.shape).copy()
    if tie_policy == 'group_mean':
        if groups is None or np.shape(groups) != mask.shape:
            raise ValueError('TIE_GROUPS_REQUIRED')
        for row, labels in enumerate(groups):
            for label in set(labels):
                tied = np.asarray(labels) == label
                valid = tied & mask[row]
                positions[row,tied] = positions[row,tied].mean()
                if valid.any():
                    x[row,valid] = x[row,valid].mean(axis=0)
    count = mask.sum(axis=1)
    mean = x.sum(axis=1) / np.maximum(count,1)[:,None]
    centers = (positions * mask).sum(axis=1) / np.maximum(count,1)
    offsets = (positions-centers[:,None]) * mask
    denominator = np.square(offsets).sum(axis=1)
    defined = (count >= 2) & (denominator > 0)
    slope = np.einsum('ni,nid->nd', offsets, x) / np.where(denominator > 0, denominator, 1)[:,None]
    slope[~defined] = 0
    return {'M':mean.astype(np.float32), 'MT':np.concatenate((mean,slope),axis=1).astype(np.float32),
            'C':x.reshape(len(x),-1).astype(np.float32), 'slope_defined':defined, 'all_empty':count==0}


def fuse(text, numeric, text_weight, numeric_weight):
    from scipy import sparse
    if text.ndim != 2 or numeric.ndim != 2 or text.shape[0] != numeric.shape[0]:
        raise ValueError('INVALID_FUSION_BLOCKS_OR_WEIGHTS')
    if not np.isfinite([text_weight,numeric_weight]).all() or min(text_weight,numeric_weight)<0 or text_weight+numeric_weight==0:
        raise ValueError('INVALID_FUSION_BLOCKS_OR_WEIGHTS')
    blocks = [np.sqrt(text_weight)*normalize(text), np.sqrt(numeric_weight)*normalize(numeric)]
    if any(sparse.issparse(block) for block in blocks):
        return sparse.hstack(blocks,format='csr',dtype=np.float32)
    return np.concatenate(blocks,axis=1).astype(np.float32)
