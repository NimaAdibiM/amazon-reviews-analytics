import csv
import gzip
import json
import random
from pathlib import Path

from .common import CATEGORIES, KEYS, atomic_json
from .contracts import review_record
from .partner import features, split_scale


def write_csv(path,fields,rows):
    with gzip.open(path,'wt',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(destination, parents=36):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    texts=[]; numeric=[]; cohorts=[]
    words=['battery','screen','cream','fridge','charger','scent','washer','lotion','cable','oven','soap','sound']
    for c,category in enumerate(CATEGORIES):
        for i in range(parents):
            parent=f'SYNTHETIC_PARENT_{i:04d}'
            split=split_scale.assign_split(parent)
            keys=dict(category=category,parent_asin=parent,split=split)
            titles=[f'{words[(i+j+c)%len(words)]} {j}' for j in range(5)]
            bodies=[f'{words[(i*3+j+c)%len(words)]} not working' if j%2 else f'Works well &amp; useful<br>day {i+j}' for j in range(5)]
            if i%7==0:
                titles[2]=None; bodies[2]=None
            if i==0:
                titles=[None]*5; bodies=[None]*5
            if i%11==1:
                bodies[1]='عالی است اما قیمت زیاد است 中文 café "quoted"\nnew line'
            if i==2:
                bodies[3]='not working battery ' * 100
            if split!='train':
                bodies[0]=(bodies[0] or '')+' heldoutonlytoken'
            stamps=[1600000000000+j*(i+1)*86400000 for j in range(5)]
            if i%6==0:
                stamps[2]=stamps[1]
            ratings=[1+(i+j*2+c)%5 for j in range(5)]
            verified=[None]*5 if i%9==0 else [True,False,True,None,True]
            row={'category':category,'parent_asin':parent,'early_titles':titles,'early_bodies':bodies,
                 'early_timestamps':stamps,'early_ratings':ratings,'early_verified':verified,
                 'n_low_future':7,'t15':stamps[-1]+999999999,'t5':stamps[-1]}
            cohorts.append(row)
            early=[dict(title=t,text=b,timestamp=s,rating=r,verified_purchase=v) for t,b,s,r,v in zip(titles,bodies,stamps,ratings,verified)]
            value=features.make_features(early)
            doc=value.pop('text_document')
            assert doc==review_record(row)['text_document']
            texts.append({**keys,'text_document':doc})
            numeric.append({**keys,**value})
    rng=random.Random(42)
    rng.shuffle(texts); rng.shuffle(numeric); rng.shuffle(cohorts)
    write_csv(destination/'synthetic_text_documents.csv.gz',KEYS+('text_document',),texts)
    raw=destination/'synthetic_features.csv.gz'
    write_csv(raw,KEYS+features.FEATURE_VIEWS['full'],numeric)
    scaler=split_scale.fit_scaler(raw,destination/'synthetic_scaler_train.json',features.SCALABLE_FIELDS,
        pre_transforms=features.FEATURE_PRE_TRANSFORMS,passthrough_fields=[x for x in features.FEATURE_VIEWS['full'] if x not in features.SCALABLE_FIELDS])
    scaled=[split_scale.apply_scaler(row,scaler) for row in numeric]
    write_csv(destination/'synthetic_features_scaled.csv.gz',KEYS+features.FEATURE_VIEWS['full'],scaled)
    with (destination/'synthetic_reviews_cohort.jsonl').open('w',encoding='utf-8') as handle:
        for row in cohorts:
            handle.write(json.dumps(row,ensure_ascii=True)+'\n')
    atomic_json(destination/'synthetic_feature_contract.json',{'key_columns':list(KEYS),'blocks':{k:list(v) for k,v in features.FEATURE_GROUPS.items()},
        'split_seed':42,'numeric_pre_transforms_before_scaling':features.FEATURE_PRE_TRANSFORMS})
    atomic_json(destination/'SYNTHETIC_FIXTURE_NOTICE.json',{'level':'SYNTHETIC_FIXTURE_TEST','n_products':len(texts),'seed':42,
        'source':'Programmatically authored examples; not Amazon observations','shared_parent_ids_across_categories':True})
    return len(texts)


def default_config(directory, synthetic=True):
    p=Path(directory)
    prefix='synthetic_' if synthetic else ''
    classical={'analyzer':'word','min_df':1 if synthetic else 2,'max_df':1.0 if synthetic else .95,
               'max_features':50000,'sublinear_tf':True,'k':4 if synthetic else 32,'normalize':True,
               'max_sparse_bytes':512*1024**2,'max_fit_work_bytes':2*1024**3}
    return {'level':'SYNTHETIC_FIXTURE_TEST' if synthetic else 'REAL_TRAIN_SUBSET_EXPERIMENT','seed':42,
        'documents':str(p/(prefix+'text_documents.csv.gz')),'numeric':str(p/(prefix+'features_scaled.csv.gz')),
        'feature_contract':str(p/(prefix+'feature_contract.json')),'scaler':str(p/(prefix+'scaler_train.json')),
        'cohorts':[str(p/(prefix+'reviews_cohort.jsonl'))], 'models':{},'max_products':4000,
        'fit_max_products':4000,'fit_text_bytes':64*1024**2,'max_document_bytes':2*1024**2,
        'max_input_bytes':256*1024**2,'batch_products':16,'pca_k':3,'export_sparse':True,
        'classical':[{**classical,'name':'unigram','ngram_range':[1,1]},{**classical,'name':'bigram','ngram_range':[1,2]}],
        'fusion':[{'name':'balanced_example','text_space':'tfidf_svd_bigram','text_weight':.5,'numeric_weight':.5}]}
