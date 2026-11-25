import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.datasets import make_blobs

from feature_representation._base.artifacts import ArtifactWriter
from .common import CATEGORIES, atomic_json, object_hash


def default_policy():
    return {'schema_version':'stage-b-policy-1','version':1,'status':'DRAFT_REQUIRES_APPROVAL','revision_kind':'INITIAL','proposal_only':True,'plan':{'resources':{'max_rows':4000,'max_scan_rows':100000,'max_input_bytes':268435456,'max_sample_bytes':134217728,'max_pairwise_bytes':134217728,'max_memory_mb':2048,'max_wall_seconds':600,'max_trials':128,'max_total_fits':1024,'max_profile_bytes':33554432,'threads':1},'sampling':{'strategy':'balanced','max_rows':240,'categories':list(CATEGORIES),'seed':42},'spaces':[{'id':'text','source':'tfidf_svd_fixture','metric':'euclidean','normalization':'none'},{'id':'numeric','source':'numeric_compact','metric':'euclidean','normalization':'none'},{'id':'fusion','source':'fusion_fixture','metric':'euclidean','normalization':'none','fusion_weights':{'text_space':'tfidf_svd_fixture','text_weight':0.5,'numeric_weight':0.5}}],'seeds':[42,43,44,45,46],'evaluation':{'max_rows':160,'seed':9001},'stability':{'subsample_repeats':3,'fraction':0.8,'min_overlap':10,'normalization_sensitivity':True},'common_eligible':True,'figures':True,'candidates':[{'algorithm':'kmeans','space':'text','k_values':[2,3],'n_init':10,'max_iter':100},{'algorithm':'minibatch','space':'text','k_values':[2,3],'n_init':10,'max_iter':100,'batch_size':64,'init_size':120,'reassignment_ratio':0.01,'max_no_improvement':10},{'algorithm':'agglomerative','space':'text','k':3,'linkage':'ward','all_seeds':False},{'algorithm':'dbscan','space':'text','min_samples':5,'eps_quantile':0.8,'all_seeds':False},{'algorithm':'dbscan','space':'text','min_samples':5,'eps_quantile':0.95,'all_seeds':False},{'algorithm':'kmeans','space':'numeric','k':3,'n_init':10,'max_iter':100,'all_seeds':False},{'algorithm':'kmeans','space':'fusion','k':3,'n_init':10,'max_iter':100,'all_seeds':False},{'algorithm':'kmeans','space':'text','scope':'per_category','k':3,'n_init':10,'max_iter':100,'all_seeds':False}]}}


def make_fixture(output,n_parents=90,dimension=12,chunk_size=31):
    from feature_representation._base.partner import split_scale
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    parents=[];i=0
    while len(parents)<n_parents:
        p=f'SYNTHETIC_B_PARENT_{i:06d}'
        if split_scale.assign_split(p)=='train':parents.append(p)
        i+=1
    keys=sorted([{'category':cat,'parent_asin':p,'split':'train','source_record':j+1} for j,p in enumerate(parents) for cat in CATEGORIES],key=lambda r:(r['category'],r['parent_asin']))
    rng=np.random.default_rng(2026);centers=rng.normal(0,5,(3,dimension));targets=np.array([parents.index(r['parent_asin'])%3 for r in keys]);x=(centers[targets]+rng.normal(0,.3,(len(keys),dimension))).astype('float32')
    numeric=(x[:,:9]+rng.normal(0,.1,(len(keys),min(9,dimension)))).astype('float32');mask=np.ones((len(keys),5),bool);mask[0]=False;mask[1,2]=False
    vectors=np.repeat(x[:,None,:],5,axis=1);vectors[~mask]=0;m=vectors.sum(axis=1)/np.maximum(mask.sum(axis=1),1)[:,None];m=m.astype('float32')
    arrays={'tfidf_svd_fixture':x,'tfidf_fixture':sparse.csr_matrix(np.maximum(x,0)),'numeric_compact':numeric,'numeric_missing':np.zeros((len(keys),9),bool),'numeric_distance':numeric,'fusion_fixture':np.hstack((x*.5,numeric*.5))}
    for family in ('bge','e5'):
        prefix='semantic_'+family;arrays.update({prefix+'_vectors':vectors,prefix+'_mask':mask,prefix+'_all_empty':~mask.any(axis=1),prefix+'_slope_defined':mask.sum(axis=1)>=2,prefix+'_M':m,prefix+'_MT':np.hstack((m,np.zeros_like(m))),prefix+'_C':vectors.reshape(len(keys),-1)})
    metadata={'synthetic':True,'level':'SYNTHETIC_FIXTURE_TEST','provenance':{'models':{'bge':{'model_id':'SYNTHETIC_STUB_BGE','stub':True},'e5':{'model_id':'SYNTHETIC_STUB_E5','stub':True}},'config':{'fusion':[{'name':'fixture','text_space':'tfidf_svd_fixture','text_weight':.5,'numeric_weight':.5}]}}}
    with ArtifactWriter(output/'a_artifacts',object_hash(['B_fixture',n_parents,dimension,chunk_size]),metadata) as writer:
        for j,start in enumerate(range(0,len(keys),chunk_size)):
            end=start+chunk_size;writer.write_chunk(j,keys[start:end],{k:v[start:end] for k,v in arrays.items()})
        writer.publish(len(keys),{}, {'fixture_only':True,'fit_scope':'synthetic train fixture'})
    terms=['camera sensor lens','soap fragrance skin','oven heating kitchen']
    with (output/'early_profiles.jsonl').open('w',encoding='utf-8') as f:
        for index,(row,c) in enumerate(zip(keys,targets)):
            record={k:row[k] for k in ('category','parent_asin','split')};record.update({'texts':[f'Synthetic review {j} {terms[c]}' if mask[index,j] else '' for j in range(5)],'ratings':[int(c)+2]*5,'timestamps':list(range(5)),'mask':mask[index].tolist()});f.write(json.dumps(record)+'\n')
    atomic_json(output/'oracle_labels.json',{'status':'SYNTHETIC_ORACLE_ONLY_NOT_FEATURE_INPUT','labels':targets.tolist(),'keys':keys})
    policy=default_policy();atomic_json(output/'synthetic_policy.json',policy)
    return {'status':'PASS','rows':len(keys),'parents':len(parents),'evidence_level':'SYNTHETIC_CONTRACT_TEST'}
