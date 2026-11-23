import heapq
import json
import shutil
from collections import Counter
from pathlib import Path

from .common import CATEGORIES, KEYS, atomic_json, csv_rows, digest, object_hash, text_reader
from .contracts import validate_key, review_record
from .fixtures import write_csv
from .partner import features


def extract_train_samples(source, cohorts, destination, per_category=200, seed=42, max_scan_records=1000000, max_output_bytes=50*1024**2, max_source_bytes=4*1024**3):
    source,destination=Path(source),Path(destination)
    if destination.exists():
        raise ValueError('SAMPLE_OUTPUT_EXISTS')
    if not 1<=per_category<=1000:
        raise ValueError('SAMPLE_QUOTA_OUT_OF_RANGE')
    required=[source/n for n in ('text_documents.csv.gz','features_scaled.csv.gz','scaler_train.json','feature_contract.json')]+[Path(p) for p in cohorts]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f'MISSING_PRECOMPUTED_INPUT: {p}')
    if sum(p.stat().st_size for p in required)>max_source_bytes:
        raise ValueError('SAMPLE_SOURCE_BYTE_BUDGET_EXCEEDED')
    heaps={c:[] for c in CATEGORIES}; natural=[]; counts=Counter(); seen=set()
    for number,row in csv_rows(source/'text_documents.csv.gz',KEYS+('text_document',)):
        if number>max_scan_records:
            raise ValueError('SAMPLE_SCAN_LIMIT_EXCEEDED: no prefix sample published')
        category,parent,split=validate_key(row)
        key=(category,parent)
        if key in seen:
            raise ValueError('DUPLICATE_KEY')
        seen.add(key)
        if split!='train':
            continue
        if len(row['text_document'].encode('utf-8'))>2*1024**2:
            raise ValueError('SAMPLE_DOCUMENT_TOO_LARGE')
        counts[category]+=1
        rank=int(object_hash([seed,category,parent]),16)
        item=(-rank,category,parent,row)
        for heap,cap in ((heaps[category],per_category),(natural,4*per_category)):
            if len(heap)<cap:
                heapq.heappush(heap,item)
            elif item[:3]>heap[0][:3]:
                heapq.heapreplace(heap,item)
    groups={'balanced':[x[3] for h in heaps.values() for x in h], 'natural':[x[3] for x in natural]}
    needed={(r['category'],r['parent_asin']) for rows in groups.values() for r in rows}
    numbers={}; reviews={}; used_bytes=sum(len(json.dumps(r).encode()) for rows in groups.values() for r in rows)
    for number,row in csv_rows(source/'features_scaled.csv.gz',KEYS+features.FEATURE_VIEWS['full']):
        if number>max_scan_records:
            raise ValueError('NUMERIC_SCAN_BUDGET_EXCEEDED')
        key=validate_key(row)[:2]
        if key in needed:
            if key in numbers:
                raise ValueError('SAMPLE_NUMERIC_DUPLICATE')
            numbers[key]=row
            used_bytes+=len(json.dumps(row).encode())
    for path in cohorts:
        with text_reader(path) as handle:
            for number,line in enumerate(handle,1):
                if number>max_scan_records:
                    raise ValueError('COHORT_SCAN_BUDGET_EXCEEDED')
                raw=json.loads(line)
                key=validate_key(raw)[:2]
                if key in needed:
                    if key in reviews:
                        raise ValueError('SAMPLE_COHORT_DUPLICATE')
                    review_record(raw)
                    safe={k:raw[k] for k in ('category','parent_asin','early_titles','early_bodies','early_timestamps','early_ratings')}
                    reviews[key]=safe
                    used_bytes+=len(json.dumps(safe).encode())
                if used_bytes>max_output_bytes:
                    raise ValueError('SAMPLE_OUTPUT_BYTE_BUDGET_EXCEEDED')
    if needed!=set(numbers) or needed!=set(reviews) or not needed:
        raise ValueError('SAMPLE_JOIN_MISSING')
    temporary=destination.with_name(destination.name+'.partial')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    provenance={'seed':seed,'population_train_counts':dict(counts),'sources':{str(p):digest(p) for p in required},
                'source_scan':'precomputed product outputs only; complete bounded scan, not raw reviews',
                'max_scan_records':max_scan_records,'max_output_bytes':max_output_bytes}
    for name,rows in groups.items():
        directory=temporary/name; directory.mkdir()
        rows.sort(key=lambda r:(r['category'],r['parent_asin']))
        write_csv(directory/'text_documents.csv.gz',KEYS+('text_document',),rows)
        write_csv(directory/'features_scaled.csv.gz',KEYS+features.FEATURE_VIEWS['full'],[numbers[(r['category'],r['parent_asin'])] for r in rows])
        with (directory/'reviews_cohort.jsonl').open('w',encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(reviews[(row['category'],row['parent_asin'])],ensure_ascii=True)+'\n')
        for filename in ('scaler_train.json','feature_contract.json'):
            shutil.copy2(source/filename,directory/filename)
        atomic_json(directory/'sampling_manifest.json',{**provenance,'sampling':name,'n':len(rows),
            'sample_category_counts':dict(Counter(r['category'] for r in rows)),
            'warning':'Scalers were fitted upstream on original train; do not treat as fit-on-this-subset scalers'})
    actual=sum(p.stat().st_size for p in temporary.rglob('*') if p.is_file())
    if actual>max_output_bytes:
        raise ValueError('SAMPLE_OUTPUT_BYTE_BUDGET_EXCEEDED')
    temporary.replace(destination)
    return {name:len(rows) for name,rows in groups.items()}
