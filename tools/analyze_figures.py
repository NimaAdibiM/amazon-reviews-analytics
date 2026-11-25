import csv
import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from feature_representation.artifacts import load_stage_a_artifacts
from clustering.common import atomic_json, digest, read_json
from clustering.artifacts import read_assignments
from clustering.profiles import redact
from clustering.verification import verify_experiment

OUT=ROOT/'results';FIG=ROOT/'figures'
NAMES={'classical':'TF-IDF/SVD','bge':'BGE','e5':'E5'}
CATS=['All_Beauty','Appliances','Beauty_and_Personal_Care','Electronics']
SHORT=['All beauty','Appliances','Beauty & care','Electronics']


def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig)


def main():
    verify_experiment(OUT/'official');FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    summaries=read_json(OUT/'RESULTS_MATCHED.json');ref={r['space']:r for r in summaries if r['algorithm']=='kmeans' and r['k']==3}
    chunks=list(load_stage_a_artifacts(ROOT/'real_inputs/representation_audit/runs/semantic'))
    keys=[k for ks,a in chunks for k in ks];cats=np.array([k['category'] for k in keys]);allarrays={}
    for name,label in [('classical','tfidf_svd_unigram'),('bge','semantic_bge_M_l2'),('e5','semantic_e5_M_l2')]:allarrays[name]=normalize(np.concatenate([a[label] for _,a in chunks]))
    labels={s:np.array(read_assignments(OUT/'official/trials'/r['reference_trial'])[1]) for s,r in ref.items()}
    early=[json.loads(l) for l in (OUT/'early_profiles.jsonl').read_text().splitlines()];ratings=np.array([r['ratings'] for r in early]);lengths=np.array([sum(len(t) for t in r['texts']) for r in early]);days=np.array([(r['timestamps'][-1]-r['timestamps'][0])/86400000 for r in early])
    desc=[]
    for c in CATS:
        m=cats==c;desc.append({'category':c,'n':int(m.sum()),'rating_mean':float(ratings[m].mean()),'early_low_share':float((ratings[m]<=2).mean()),'text_chars_median':float(np.median(lengths[m])),'window_days_median':float(np.median(days[m])),'empty_reviews':sum(not t for i,r in enumerate(early) if m[i] for t in r['texts'])})
    atomic_json(OUT/'SAMPLE_DESCRIPTIVES.json',desc)
    pop=read_json(OUT/'REAL_INPUT_AUDIT.json')['population_train_counts_reported'];total=sum(pop.values())
    fig,ax=plt.subplots(figsize=(8,3.3));x=np.arange(4);ax.bar(x-.18,[100*pop[c]/total for c in CATS],.36,label='Reported train population');ax.bar(x+.18,[25]*4,.36,label='Verified comparison sample');ax.set(xticks=x,xticklabels=SHORT,ylabel='Percent',title='Balanced design changes category weights');ax.legend();save(fig,'01_sample_design')
    fig,axes=plt.subplots(2,3,figsize=(12,7));proj={}
    for j,s in enumerate(NAMES):
        model=PCA(n_components=2,svd_solver='full');z=model.fit_transform(allarrays[s]);proj[s]={'variance_ratio':model.explained_variance_ratio_.tolist()}
        for c in range(3):axes[0,j].scatter(*z[labels[s]==c].T,s=9,alpha=.6,label=f'C{c}')
        for c,short in zip(CATS,SHORT):axes[1,j].scatter(*z[cats==c].T,s=9,alpha=.6,label=short)
        axes[0,j].set_title(NAMES[s]+f" | first two PCs {100*model.explained_variance_ratio_.sum():.1f}%")
        for ax in axes[:,j]:ax.set(xlabel='PC1',ylabel='PC2')
    axes[0,2].legend(fontsize=8);axes[1,2].legend(fontsize=7);fig.suptitle('Real 512 train products | top: K-Means k=3; bottom: post-hoc categories\n2D views are not the clustering fit space');save(fig,'02_projections')
    fig,axes=plt.subplots(1,3,figsize=(11,3.3))
    for s in NAMES:
        rows=sorted([r for r in summaries if r['space']==s and r['algorithm']=='kmeans'],key=lambda r:r['k'])
        for ax,field,title in zip(axes,['silhouette_median','seed_ari_median','subsample_ari_seed42_median'],['Cosine silhouette','Seed stability (median ARI)','80% parent stability (median ARI)']):
            ax.plot([r['k'] for r in rows],[r[field] for r in rows],'o-',label=NAMES[s]);ax.set(title=title,xlabel='k',xticks=[3,5,8])
    axes[1].axhline(.75,color='grey',ls='--',lw=1);axes[2].axhline(.6,color='grey',ls='--',lw=1);axes[0].legend(fontsize=8);save(fig,'03_parameter_stability')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5));category_counts=np.array([[sum((labels['e5']==c)&(cats==cat)) for cat in CATS] for c in range(3)])
    im=axes[0].imshow(category_counts,cmap='Blues');axes[0].set(xticks=range(4),xticklabels=SHORT,yticks=range(3),yticklabels=['C0','C1','C2'],title='E5: post-hoc category counts');axes[0].tick_params(axis='x',rotation=20)
    for i in range(3):
        for j in range(4):axes[0].text(j,i,str(category_counts[i,j]),ha='center',va='center',color='black')
        axes[1].plot(range(1,6),ratings[labels['e5']==i].mean(axis=0),'o-',label=f'C{i} (n={sum(labels["e5"]==i)})')
    axes[1].set(title='Five early ratings only',xlabel='Observed review position',ylabel='Mean rating',ylim=(1,5.1));axes[1].legend(fontsize=8);save(fig,'04_e5_profiles')
    pairs=[{'left':a,'right':b,'ari':float(adjusted_rand_score(labels[a],labels[b]))} for a,b in combinations(ref,2)]
    atomic_json(OUT/'ABLATION_PARTITION_ARI.json',pairs)
    fig,axes=plt.subplots(1,2,figsize=(10,3.4));ss=['classical','e5','numeric','numeric_l2','fusion'];xx=np.arange(len(ss))
    axes[0].bar(xx,[ref[s]['silhouette_median'] for s in ss],color='#286b85');axes[0].set(xticks=xx,xticklabels=ss,title='Within-space cosine silhouette',ylim=(0,.75));axes[0].tick_params(axis='x',rotation=20)
    axes[1].bar(xx,[ref[s]['seed_ari_median'] for s in ss],color='#a96538');axes[1].set(xticks=xx,xticklabels=ss,title='Seed stability',ylim=(0,1.05));axes[1].tick_params(axis='x',rotation=20);save(fig,'05_ablation')
    counts=[];examples=[];lookup={(r['category'],r['parent_asin']):r for r in early}
    for s in NAMES:
        prof=read_json(OUT/'official/trials'/ref[s]['reference_trial']/'profiles.json')
        for c,v in prof['clusters'].items():
            counts.append({'space':s,'cluster':c,'n':v['n'],'early_rating_mean':float(np.mean(v['early_rating_mean_by_position'])),'early_low_share':v['early_low_star_share_mean'],**{cat:v['category_counts'].get(cat,0) for cat in CATS}})
            for ex in v['examples'][:2]:
                r=lookup[ex['key']['category'],ex['key']['parent_asin']]
                examples.append({'space':s,'cluster':c,'role':ex['role'],'category':r['category'],'early_ratings':r['ratings'],'excerpt':redact(next(t for t in r['texts'] if t)),'privacy':'regex redaction; public report excludes identifiers; excerpt reviewed by assistant, not independent human'})
    with (OUT/'CLUSTER_PROFILES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(counts[0]));w.writeheader();w.writerows(counts)
    atomic_json(OUT/'REPRESENTATIVE_EXAMPLES.json',examples);atomic_json(OUT/'PROJECTION_VARIANCE.json',proj)
    man=read_json(ROOT/'real_inputs/representation_audit/runs/semantic/manifest.json');variance=man['report']['classical']['unigram']['explained_variance_ratio']
    fig,ax=plt.subplots(figsize=(7,3.2));ax.plot(np.arange(1,len(variance)+1),np.cumsum(variance));ax.set(xlabel='SVD components',ylabel='Cumulative explained variance',title='Classical TF-IDF/SVD fitted on 512 train documents');save(fig,'06_svd_variance')
    atomic_json(OUT/'FIGURE_PROVENANCE.json',{'source_experiment_plan_sha256':digest(OUT/'official/experiment_plan.json'),'results_sha256':digest(OUT/'RESULTS_MATCHED.csv'),'scope':'REAL_512_TRAIN_ONLY','figures':{p.name:digest(p) for p in FIG.glob('*')}})
    print('Generated',len(list(FIG.glob('*.png'))),'figures')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
