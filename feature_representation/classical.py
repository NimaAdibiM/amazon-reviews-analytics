import heapq
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .common import object_hash


def training_sample(rows, limit, byte_limit, seed, balanced=False):
    heaps = {}
    counts = {}
    retained_bytes = 0
    for row in rows:
        if row['split'] != 'train':
            continue
        category = row['category']
        counts[category] = counts.get(category, 0) + 1
        bucket = category if balanced else 'all'
        rank = int(object_hash([seed, category, row['parent_asin']]), 16)
        heap = heaps.setdefault(bucket, [])
        item = (-rank, category, row['parent_asin'], row)
        if len(heap) < limit:
            heapq.heappush(heap, item)
            retained_bytes += len(row['text_document'].encode('utf-8'))
        elif item[:3] > heap[0][:3]:
            removed=heapq.heapreplace(heap, item)
            retained_bytes += len(row['text_document'].encode('utf-8'))-len(removed[3]['text_document'].encode('utf-8'))
        if retained_bytes > byte_limit:
            raise ValueError('FIT_TEXT_BUDGET_EXCEEDED: reservoir budget exceeded; reduce fit_max_products')
    selected = [item[3] for heap in heaps.values() for item in heap]
    selected.sort(key=lambda r: (r['category'], r['parent_asin']))
    if not selected:
        raise ValueError('NO_TRAIN_ROWS')
    size = sum(len(r['text_document'].encode('utf-8')) for r in selected)
    if size > byte_limit:
        raise ValueError('FIT_TEXT_BUDGET_EXCEEDED: reduce fit_max_products')
    return selected, {'fit_split':'train', 'fit_scope':'fit-on-train-subset' if len(selected) < sum(counts.values()) else 'all-available-train',
                      'n_fit':len(selected), 'population_train_counts':counts, 'text_bytes':size,
                      'selection':'deterministic bottom-hash stratified' if balanced else 'deterministic bottom-hash proportional',
                      'seed':seed, 'fit_keys':[[r['category'],r['parent_asin']] for r in selected]}


def fit_classical(rows, config, seed):
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('FIT_REQUIRES_TRAIN_ONLY')
    analyzer = config.get('analyzer', 'word')
    if analyzer not in ('word','char_wb'):
        raise ValueError('INVALID_ANALYZER')
    if analyzer == 'char_wb' and not config.get('char_sensitivity_reason'):
        raise ValueError('CHAR_SENSITIVITY_REQUIRES_REASON')
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=tuple(config['ngram_range']),
                               min_df=config['min_df'], max_df=config['max_df'], max_features=config['max_features'],
                               sublinear_tf=config['sublinear_tf'], dtype=np.float32, stop_words=None, lowercase=True)
    started = time.perf_counter()
    matrix = vectorizer.fit_transform(r['text_document'] for r in rows).tocsr()
    memory = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    if memory > config['max_sparse_bytes']:
        raise ValueError('SPARSE_BUDGET_EXCEEDED')
    k = config['k']
    if type(k) is not int or not 1 <= k < min(matrix.shape):
        raise ValueError(f'INVALID_SVD_K: k={k}, train_shape={matrix.shape}')
    estimated = (matrix.shape[0] + matrix.shape[1]) * (k + 10) * 8 * 4 + memory * 2
    if estimated > config['max_fit_work_bytes']:
        raise ValueError('SVD_WORK_BUDGET_EXCEEDED')
    reducer = TruncatedSVD(n_components=k, random_state=seed, n_iter=7)
    reducer.fit(matrix)
    if not np.isfinite(reducer.explained_variance_ratio_).all():
        raise ValueError('DEGENERATE_TFIDF_VARIANCE')
    sample = matrix[:min(256, matrix.shape[0])]
    projected = reducer.transform(sample)
    squared = float(sample.multiply(sample).sum())
    relative_error = max(0.0, 1.0 - float(np.square(projected).sum()) / squared) if squared else None
    report = {'train_shape':list(matrix.shape), 'nnz':matrix.nnz, 'sparsity':1-matrix.nnz/(matrix.shape[0]*matrix.shape[1]),
              'csr_bytes':memory, 'estimated_svd_work_bytes':estimated, 'seconds':time.perf_counter()-started,
              'vocabulary_size':len(vectorizer.vocabulary_), 'explained_variance_ratio':reducer.explained_variance_ratio_.tolist(),
              'sample_relative_squared_reconstruction_error':relative_error, 'reconstruction_n':sample.shape[0],
              'zero_train_documents':int(np.count_nonzero(matrix.getnnz(axis=1)==0)), 'k':k, 'seed':seed,
              'predictive_quality':'NOT_EVALUATED', 'normalization': 'row_l2_after_svd' if config['normalize'] else 'none_after_svd'}
    return {'vectorizer':vectorizer, 'reducer':reducer, 'config':config, 'report':report}


def transform_classical(fitted, documents):
    sparse = fitted['vectorizer'].transform(documents).tocsr().astype(np.float32)
    dense = fitted['reducer'].transform(sparse).astype(np.float32)
    if fitted['config']['normalize']:
        dense = normalize(dense).astype(np.float32)
    if not np.isfinite(dense).all():
        raise ValueError('NONFINITE_SVD_OUTPUT')
    return sparse, dense


def neighbor_overlap(left, right, neighbors=5):
    if left.shape[0] != right.shape[0] or left.shape[0] < 2:
        raise ValueError('NEIGHBOR_SAMPLE_INVALID')
    k = min(neighbors, left.shape[0] - 1)
    a, b = normalize(left), normalize(right)
    sa, sb = a @ a.T, b @ b.T
    np.fill_diagonal(sa, -np.inf)
    np.fill_diagonal(sb, -np.inf)
    ia, ib = np.argsort(-sa, axis=1)[:,:k], np.argsort(-sb, axis=1)[:,:k]
    return float(np.mean([len(set(x)&set(y))/k for x,y in zip(ia,ib)]))


def diagnose_candidates(rows, config, candidates, seeds):
    records = []
    for k in candidates:
        embeddings = []
        for seed in seeds:
            try:
                fitted = fit_classical(rows, {**config,'k':k}, seed)
                _, vectors = transform_classical(fitted, [r['text_document'] for r in rows[:256]])
                embeddings.append(vectors)
                records.append({'status':'PASS', **fitted['report']})
            except ValueError as exc:
                records.append({'status':'SKIPPED', 'k':k, 'seed':seed, 'reason':str(exc)})
        if len(embeddings) >= 2:
            records.append({'k':k,'diagnostic':'seed_neighbor_overlap','n':len(embeddings[0]),
                            'value':neighbor_overlap(embeddings[0],embeddings[1]), 'seeds':seeds[:2]})
    return {'selection':'K_SELECTION_PENDING', 'records':records}
