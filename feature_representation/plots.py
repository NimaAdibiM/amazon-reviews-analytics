from pathlib import Path
import heapq
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .artifacts import load_stage_a_artifacts
from .common import CATEGORIES, object_hash, read_json


def projection_sample(chunks, label, seed, limit=1000):
    """Balanced display, independent of database order. Scan all training keys."""
    cap = max(1, limit // len(CATEGORIES))
    groups = {c:[] for c in CATEGORIES}
    for keys,arrays in chunks:
        for i,key in enumerate(keys):
            if key['split']!='train':
                continue
            category, parent = key['category'], key['parent_asin']
            rank = int(object_hash([seed,category,parent]),16)
            heap = groups[category]
            if len(heap)<cap or -rank>heap[0][0]:
                item = (-rank,parent,arrays[label][i,:2].copy())
                if len(heap)<cap:
                    heapq.heappush(heap,item)
                else:
                    heapq.heapreplace(heap,item)
    return [(c,entry[2]) for c in CATEGORIES for entry in sorted(groups[c],key=lambda v:v[:2])]


def plot_run(artifacts, output):
    artifacts,output=Path(artifacts),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    manifest=read_json(artifacts/'manifest.json')
    report=manifest['report']
    label='SYNTHETIC FIXTURE' if manifest['metadata']['synthetic'] else 'REAL TRAIN SUBSET'
    if manifest['metadata']['level']=='FULL_DATA_EXECUTION':
        label='TRAIN DIAGNOSTICS FROM AUTHORIZED FULL RUN'
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    names=list(report['classical'])
    for name in names:
        r=report['classical'][name]
        axes[0,0].plot(np.arange(1,len(r['explained_variance_ratio'])+1),np.cumsum(r['explained_variance_ratio']),marker='o',label=name)
    axes[0,0].set(title='SVD cumulative explained variance (train fit)',xlabel='Components',ylabel='Fraction'); axes[0,0].legend()
    axes[0,1].bar(names,[report['classical'][n]['vocabulary_size'] for n in names]); axes[0,1].set(title='Train vocabulary',ylabel='Terms')
    axes[1,0].bar(names,[report['classical'][n]['sparsity'] for n in names]); axes[1,0].set(title='Train CSR sparsity',ylabel='Zero fraction')
    axes[1,1].bar(names,[report['classical'][n]['seconds'] for n in names]); axes[1,1].set(title='TF-IDF + SVD fit time',ylabel='Seconds')
    fig.suptitle(f'{label}; n_fit={report["fit_scope"]["n_fit"]}; seed={report["fit_scope"]["seed"]}\nGeometry and variance do not establish predictive value')
    fig.tight_layout(); fig.savefig(output/'representation_diagnostics.png',dpi=150); plt.close(fig)
    rows=projection_sample(load_stage_a_artifacts(artifacts),'tfidf_svd_'+names[0],report['fit_scope']['seed'])
    cats=[c for c,_ in rows]
    values=[v for _,v in rows]
    generated=['representation_diagnostics.png']
    if values and len(values[0])==2:
        x=np.asarray(values)
        fig,ax=plt.subplots(figsize=(8,6))
        for category in sorted(set(cats)):
            take=np.asarray(cats)==category; ax.scatter(x[take,0],x[take,1],label=f'{category} (n={int(take.sum())})',s=15,alpha=.65)
        ax.legend(fontsize=8); ax.set(title=f'{label}: first two SVD coordinates, category-balanced display, train n={len(x)}\nProjection only; no cluster claim',xlabel='SVD coordinate 1',ylabel='SVD coordinate 2')
        fig.tight_layout(); fig.savefig(output/'train_projection.png',dpi=150); plt.close(fig)
        generated.append('train_projection.png')
    if report.get('numeric') and report['numeric']['pca_explained_variance_ratio']:
        fig,ax=plt.subplots(figsize=(7,5)); ratios=report['numeric']['pca_explained_variance_ratio']
        ax.plot(np.arange(1,len(ratios)+1),np.cumsum(ratios),marker='o')
        ax.set(title=f'{label}: numeric PCA train fit',xlabel='Components',ylabel='Cumulative explained variance')
        fig.tight_layout(); fig.savefig(output/'numeric_pca.png',dpi=150); plt.close(fig)
        generated.append('numeric_pca.png')
    for name,stats in report['semantic'].items():
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        if 'train_token_summary' in stats:
            summary=stats['train_token_summary']
            for axis,key in zip(axes,('lengths','dropped')):
                counts=summary[key]['counts']
                axis.bar(range(len(counts)),counts)
                axis.set(title='Train token '+key,xlabel='Log2 bucket index (0 = zero)',ylabel='Exact count')
        else:
            axes[0].hist(stats['train_token_lengths'],bins=20); axes[0].set(title='Train token lengths',xlabel='Tokens including prefix/special tokens')
            axes[1].hist(stats['train_tokens_dropped'],bins=20); axes[1].set(title='Train dropped tokens',xlabel='Tokens dropped')
        fig.suptitle(f'{label} {name}; '+('STUB TOKENIZER - NOT MODEL RESULTS' if stats['stub'] else 'Model tokenizer'))
        fig.tight_layout(); fig.savefig(output/f'tokens_{name}.png',dpi=150); plt.close(fig)
        generated.append(f'tokens_{name}.png')
    return sorted(generated)
