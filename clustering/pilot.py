import csv
import gzip
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

from feature_representation._base.common import CATEGORIES, KEYS, csv_rows, text_reader
from feature_representation._base.contracts import numeric_values, review_record, validate_key, validate_numeric_contract
from feature_representation._base.fixtures import default_config, write_csv
from feature_representation._base.partner import features
from .common import atomic_json, digest, file_hashes, key_hash, key_tuple, read_json
from .sampling import add_key, choose_groups, create_index


def prepare_pilot(source,cohorts,output,max_rows=800,strategy='balanced',seed=42,max_scan_rows=1000000,max_input_bytes=1073741824):
    source,output=Path(source),Path(output)
    if output.exists():raise ValueError('PILOT_OUTPUT_EXISTS')
    if not 12<=max_rows<=4000 or max_scan_rows<max_rows or not 0<max_input_bytes<=4*1024**3:raise ValueError('INVALID_PILOT_PREPARATION_BUDGET')
    paths=[source/n for n in ('text_documents.csv.gz','features_scaled.csv.gz','scaler_train.json','feature_contract.json')]+[Path(p) for p in cohorts]
    for p in paths:
        if not p.is_file():raise FileNotFoundError('MISSING_PRECOMPUTED_INPUT: '+str(p))
    if sum(p.stat().st_size for p in paths)>max_input_bytes:raise ValueError('PRECOMPUTED_INPUT_BYTE_CAP')
    hashes={str(p.resolve()):digest(p) for p in paths};validate_numeric_contract(read_json(paths[3]),read_json(paths[2]))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pilot_index_',dir=output.parent) as tmp:
        con=sqlite3.connect(str(Path(tmp)/'index.sqlite'));create_index(con);con.execute('CREATE TABLE seen(category TEXT,parent TEXT,PRIMARY KEY(category,parent))');scanned=0
        for number,row in csv_rows(paths[0],KEYS+('text_document',)):
            if number>max_scan_rows:raise ValueError('PRECOMPUTED_SCAN_CAP_NO_PREFIX_SAMPLE_PUBLISHED')
            cat,parent,split=validate_key(row)
            try:con.execute('INSERT INTO seen VALUES(?,?)',key_tuple(row))
            except sqlite3.IntegrityError as exc:raise ValueError('DUPLICATE_DOCUMENT_KEY') from exc
            scanned+=1
            if split=='train':add_key(con,row,seed)
        con.commit();keys,details=choose_groups(con,strategy,max_rows,CATEGORIES);con.close()
        wanted={key_tuple(r) for r in keys};docs={};nums={};reviews={};profiles={};used=0
        if not keys:raise ValueError('NO_TRAIN_INPUTS')
        for number,row in csv_rows(paths[0],KEYS+('text_document',)):
            if number>max_scan_rows:raise ValueError('DOCUMENT_SECOND_PASS_SCAN_CAP')
            key=key_tuple(row)
            if key in wanted:
                if len(row['text_document'].encode())>2*1024**2:raise ValueError('DOCUMENT_TOO_LARGE')
                docs[key]={k:row[k] for k in KEYS+('text_document',)};used+=len(json.dumps(docs[key]).encode())
                if used>64*1024**2:raise ValueError('PILOT_OUTPUT_MEMORY_CAP')
        for number,row in csv_rows(paths[1],KEYS+features.FEATURE_VIEWS['full']):
            if number>max_scan_rows:raise ValueError('NUMERIC_SCAN_CAP')
            validate_key(row);key=key_tuple(row)
            if key in wanted:
                if key in nums:raise ValueError('DUPLICATE_NUMERIC_KEY')
                numeric_values(row);nums[key]={k:row[k] for k in KEYS+features.FEATURE_VIEWS['full']}
        scanned_cohorts=0
        for path in paths[4:]:
            with text_reader(path) as f:
                for line in f:
                    scanned_cohorts+=1
                    if scanned_cohorts>max_scan_rows:raise ValueError('COHORT_SCAN_CAP')
                    if len(line.encode())>4*1024**2:raise ValueError('COHORT_RECORD_BYTE_CAP')
                    raw=json.loads(line);key=validate_key(raw)[:2]
                    if key in wanted:
                        if key in reviews:raise ValueError('DUPLICATE_COHORT_KEY')
                        rec=review_record(raw)
                        if rec['text_document']!=docs[key]['text_document']:raise ValueError('EARLY_DOCUMENT_COHORT_MISMATCH')
                        reviews[key]={k:raw[k] for k in ('category','parent_asin','early_titles','early_bodies','early_timestamps','early_ratings')}
                        profiles[key]={k:rec[k] for k in ('category','parent_asin','split','texts','ratings','timestamps','mask')}
                        used+=len(json.dumps(reviews[key]).encode())+len(json.dumps(profiles[key]).encode())
                        if used>64*1024**2:raise ValueError('PILOT_OUTPUT_MEMORY_CAP')
        if wanted!=set(nums) or wanted!=set(reviews):raise ValueError('PILOT_JOIN_MISSING')
        if hashes!={str(p.resolve()):digest(p) for p in paths}:raise ValueError('PILOT_INPUT_CHANGED')
        partial=Path(tmp)/'publish';partial.mkdir()
        write_csv(partial/'text_documents.csv.gz',KEYS+('text_document',),[docs[key_tuple(r)] for r in keys])
        write_csv(partial/'features_scaled.csv.gz',KEYS+features.FEATURE_VIEWS['full'],[nums[key_tuple(r)] for r in keys])
        for name,data in (('reviews_cohort.jsonl',reviews),('early_profiles.jsonl',profiles)):
            with (partial/name).open('w',encoding='utf-8') as f:
                for row in keys:f.write(json.dumps(data[key_tuple(row)],ensure_ascii=False)+'\n')
        for name in ('scaler_train.json','feature_contract.json'):shutil.copy2(source/name,partial/name)
        config=default_config(output.resolve(),synthetic=False);config.update({'level':'REAL_TRAIN_SUBSET_EXPERIMENT','documents':str(output.resolve()/'text_documents.csv.gz'),'numeric':str(output.resolve()/'features_scaled.csv.gz'),'cohorts':[str(output.resolve()/'reviews_cohort.jsonl')],'feature_contract':str(output.resolve()/'feature_contract.json'),'scaler':str(output.resolve()/'scaler_train.json'),'models':{},'max_products':max_rows,'fit_max_products':max_rows,'pca_k':3})
        for c in config['classical']:c.update({'k':min(32,len(keys)-1),'min_df':2})
        config['fusion']=[]
        atomic_json(partial/'stage_a_pilot_proposed.json',config)
        receipt={**details,'status':'PASS','scope':'bounded complete scan of precomputed product outputs, no raw reviews scan','n_rows':len(keys),'keys':keys,'key_hash':key_hash(keys),'scanned_document_rows':scanned,'scanned_cohort_rows':scanned_cohorts,'seed':seed,'sources':hashes,'files':file_hashes(partial),'provenance_status':'USER_REAL_SOURCE_ATTESTATION_REQUIRED_BEFORE_A_EXECUTION','config_status':'PROPOSAL_NOT_EXECUTED_OR_SCIENTIFICALLY_APPROVED','scaler_scope':'Original upstream train; preserved, not refitted to pilot'}
        atomic_json(partial/'pilot_preparation_receipt.json',receipt);partial.rename(output)
    return receipt
