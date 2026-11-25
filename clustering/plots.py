from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA, TruncatedSVD


def figures(root,x,labels,metrics,profiles,model,level):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    def save(name,title):
        plt.title(level+'\n'+title);plt.tight_layout();plt.savefig(root/(name+'.png'),dpi=110);plt.close()
    plt.figure(figsize=(6,4));ids,counts=np.unique(labels,return_counts=True);plt.bar(ids.astype(str),counts);plt.xlabel('Cluster (-1 = noise)');plt.ylabel('Sample rows');save('sizes','Cluster sizes')
    if 'train_k_distances_sorted' in metrics['cost']:
        plt.figure(figsize=(6,4));plt.plot(metrics['cost']['train_k_distances_sorted']);plt.axhline(metrics['cost']['eps'],color='red',linestyle='--',label='registered candidate eps');plt.legend();plt.xlabel('Sorted train row index');plt.ylabel('k-neighbor distance, self included');save('k_distances','Train k-distance diagnostic')
    rows=metrics['silhouette_samples']
    if rows:
        plt.figure(figsize=(6,4));plt.hist([r['value'] for r in rows],bins=20);plt.xlabel('Silhouette');save('silhouette','Fixed-sample silhouette distribution')
    if min(x.shape)>=2:
        z=TruncatedSVD(n_components=2,random_state=42).fit_transform(x) if sparse.issparse(x) else PCA(n_components=2,svd_solver='full').fit_transform(x)
        plt.figure(figsize=(6,4))
        for label in np.unique(labels):
            ix=labels==label;plt.scatter(z[ix,0],z[ix,1],s=10,label='noise' if label==-1 else str(label))
        plt.legend(title='Cluster',fontsize=7);save('projection','Train-only descriptive 2D projection, not clustering geometry')
    cs=profiles['clusters'];cats=sorted({cat for v in cs.values() for cat in v['category_counts']})
    plt.figure(figsize=(8,4));a=np.array([[v['category_fraction_within_cluster'][cat] for cat in cats] for v in cs.values()]);plt.imshow(a,aspect='auto',vmin=0,vmax=1);plt.xticks(range(len(cats)),cats,rotation=25,ha='right');plt.yticks(range(len(cs)),list(cs));plt.colorbar();save('categories','Category fractions within each cluster')
    if hasattr(model,'children_'):
        n=len(labels);counts=np.zeros(len(model.children_))
        for i,children in enumerate(model.children_):counts[i]=sum(1 if child<n else counts[child-n] for child in children)
        linkage=np.column_stack([model.children_,model.distances_,counts]);plt.figure(figsize=(9,4));dendrogram(linkage,truncate_mode='lastp',p=20);save('dendrogram','Sampled hierarchy; truncated display')


def overview_figures(root,comparison,level):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    groups={}
    for row in comparison['rows']:
        cfg=row['trial']['config']
        if cfg.get('k') is not None:groups.setdefault((row['space']['id'],cfg['algorithm'],row['trial']['scope']),[]).append(row)
    for group,rows in groups.items():
        if len({r['trial']['config']['k'] for r in rows})<2:continue
        fig,axs=plt.subplots(1,3,figsize=(12,3.5))
        for ax,name in zip(axs,('inertia','silhouette','seconds')):
            for row in rows:
                value=row['inertia'] if name=='inertia' else row['silhouette'].get('value') if name=='silhouette' else row['cost']['elapsed_fit_seconds']
                if value is not None:ax.scatter(row['trial']['config']['k'],value,s=18)
            ax.set_xlabel('k');ax.set_ylabel(name)
        fig.suptitle(level+'\n'+' / '.join(group)+'; separate spaces are not ranked');fig.tight_layout();fig.savefig(root/('_'.join(group)+'_k_curve.png'),dpi=110);plt.close(fig)
    families=comparison['multi_criteria_assessment']['families'];valid=[f for f in families if f['seed_ari_median'] is not None]
    if valid:
        fig,ax=plt.subplots(figsize=(8,4));ax.bar(range(len(valid)),[f['seed_ari_median'] for f in valid]);ax.set_xticks(range(len(valid)),[f['candidate']['algorithm']+' k='+str(f['candidate'].get('k')) for f in valid],rotation=30,ha='right');ax.set_ylabel('Median seed ARI on shared nonnoise keys');ax.set_ylim(-1,1);ax.set_title(level+'\nSeed stability per candidate family');fig.tight_layout();fig.savefig(root/'seed_stability.png',dpi=110);plt.close(fig)
