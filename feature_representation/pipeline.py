import gc
import itertools
import os
import threading
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.preprocessing import normalize

from .aggregation import aggregate, fuse
from .artifacts import ArtifactWriter, load_stage_a_artifacts, verify_files
from .classical import fit_classical, training_sample, transform_classical
from .common import LEVELS, atomic_json, digest, environment, now, object_hash, read_json, source_hashes
from .contracts import build_index, check_scaled_source, index_rows
from .numeric import fit_numeric, transform_numeric
from .semantic import MODELS, SentenceBackend, encode_products, model_identity
from .token_stats import TokenSummary


def validate_config(config, authorize_full=False):
    if config['level'] not in LEVELS:
        raise ValueError('INVALID_EXECUTION_LEVEL')
    if config['level']=='FULL_DATA_EXECUTION' and not authorize_full:
        raise ValueError('FULL_RUN_REQUIRES_SEPARATE_AUTHORIZATION')
    if config['level']!='FULL_DATA_EXECUTION' and config['max_products']>4000:
        raise ValueError('LIMITED_RUN_MAXIMUM_4000_PRODUCTS')
    for name in ('batch_products','max_products','fit_max_products','fit_text_bytes','max_document_bytes','max_input_bytes'):
        if type(config[name]) is not int or config[name]<1:
            raise ValueError(f'INVALID_RESOURCE_BUDGET: {name}')
    if not config.get('classical'):
        raise ValueError('CLASSICAL_BASELINE_REQUIRED')
    if not any(c.get('analyzer','word')=='word' and c['ngram_range']==[1,1] for c in config['classical']):
        raise ValueError('UNIGRAM_BASELINE_REQUIRED')
    if len({c['name'] for c in config['classical']})!=len(config['classical']):
        raise ValueError('DUPLICATE_CLASSICAL_NAME')
    for c in config['classical']:
        if not c['name'].replace('_','').isalnum():
            raise ValueError('INVALID_CLASSICAL_NAME')
    if set(config.get('models',{}))-set(MODELS):
        raise ValueError('UNSUPPORTED_MODEL')
    fusion = config.get('fusion', [])
    names = [item['name'] for item in fusion]
    if len(names) != len(set(names)) or any(not n.replace('_','').isalnum() for n in names):
        raise ValueError('INVALID_OR_DUPLICATE_FUSION_NAME')
    for item in fusion:
        weights = (item['text_weight'],item['numeric_weight'])
        if not np.isfinite(weights).all() or min(weights)<0 or sum(weights)<=0:
            raise ValueError('INVALID_FUSION_WEIGHTS')
        if item.get('numeric_space','numeric_distance') not in ('numeric_distance','numeric_rating'):
            raise ValueError('INVALID_FUSION_NUMERIC_SPACE')
        classical_spaces = {'tfidf_svd_' + c['name'] for c in config['classical']}
        if config.get('export_sparse'):
            classical_spaces |= {'tfidf_' + c['name'] for c in config['classical']}
        semantic_spaces = {'semantic_' + name + '_' + form for name in config.get('models', {})
                           for form in ('M','MT','C','M_l2','MT_l2','C_l2')}
        if item['text_space'] not in classical_spaces | semantic_spaces:
            raise ValueError('FUSION_TEXT_SPACE_UNAVAILABLE')
    if config.get('models') and not config.get('cohorts'):
        raise ValueError('MISSING_COHORT_INPUT')
    for cfg in config.get('models',{}).values():
        if cfg.get('tie_policy','group_mean') not in ('group_mean','canonical'):
            raise ValueError('UNKNOWN_TIE_POLICY')
    if fusion and not config.get('numeric'):
        raise ValueError('FUSION_REQUIRES_NUMERIC')


def input_fingerprints(config):
    paths=[config['documents']]+config.get('cohorts',[])
    if config.get('numeric'):
        paths += [config['numeric'],config['scaler'],config['feature_contract']]
    missing=[str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError('MISSING_PHASE2_INPUT: '+', '.join(missing))
    if sum(Path(p).stat().st_size for p in paths)>config['max_input_bytes']:
        raise ValueError('INPUT_BYTE_BUDGET_EXCEEDED')
    hashes = {str(Path(p).resolve()):{'sha256':digest(p),'bytes':Path(p).stat().st_size} for p in paths}
    if config.get('numeric'):
        check_scaled_source(config, hashes)
    return hashes


class PeakMonitor:
    def __init__(self):
        self.stop=threading.Event()
        self.peak=0
        self.thread=threading.Thread(target=self.watch,daemon=True)

    def watch(self):
        process=psutil.Process()
        while not self.stop.is_set():
            self.peak=max(self.peak,process.memory_info().rss)
            self.stop.wait(.01)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set()
        self.thread.join()


def run_pipeline(config, output, authorize_full=False, backend_factory=None, interrupt_after=None):
    validate_config(config,authorize_full)
    if backend_factory is not None and config['level']!='SYNTHETIC_FIXTURE_TEST':
        raise ValueError('STUB_FORBIDDEN_OUTSIDE_SYNTHETIC_TEST')
    with PeakMonitor() as monitor:
        return _run(config,Path(output),backend_factory,interrupt_after,monitor)


def _run(config,output,backend_factory,interrupt_after,monitor):
    started=now(); timer=time.perf_counter()
    inputs=input_fingerprints(config)
    model_meta={}
    for name,cfg in config.get('models',{}).items():
        if backend_factory:
            model_meta[name]=backend_factory(name,cfg).identity
            if not model_meta[name].get('stub'):
                raise ValueError('INJECTED_BACKEND_MUST_BE_LABELLED_STUB')
        else:
            model_meta[name]=model_identity(name,cfg)
    env=environment()
    provenance={'config':config,'inputs':inputs,'models':model_meta,'code':source_hashes(),'environment':env}
    signature=object_hash(provenance)
    metadata={'level':config['level'],'synthetic':config['level']=='SYNTHETIC_FIXTURE_TEST',
              'model_selection':'MODEL_SELECTION_PENDING','research_contract':'REPRESENTATIONS_ONLY_NO_CLUSTER_OR_PREDICTIVE_EVALUATION',
              'provenance':provenance,'feature_dependencies':{'text':'early_titles/early_bodies positions 1..5 only',
              'numeric':'partner 40 allowed early-review features; no IDs/category/split/targets/snapshot in X'},
              'distance':{'tfidf_svd':'row L2 if configured; cosine excludes zero vectors',
                          'semantic':'M,MT,C raw aggregates; separate L2 variants; masks mandatory',
                          'numeric_compact':'9 partner-scaled features; zero imputation + separate mask; no second z-score',
                          'numeric_distance':'train-variable compact values + 9 missing indicators',
                          'fusion':'sqrt(weight) times unit-normalized blocks; numeric_rating for the main comparison, numeric_distance for extended analyses'},
              'tie_policy':{name:cfg.get('tie_policy','group_mean') for name,cfg in config.get('models',{}).items()},
              'tie_order':'documents use (timestamp,title,body,source position); primary semantic trend uses group midranks by default'}
    with ArtifactWriter(output,signature,metadata) as writer:
        resumed_chunks=writer.chunk_count
        resumed_rows=writer.row_count
        if (output/'manifest.json').exists():
            for _ in load_stage_a_artifacts(output):
                pass
            return read_json(output/'manifest.json')
        index=output/'work_index.sqlite'
        count=build_index(config,index)
        selected,fit_scope=training_sample(index_rows(index),config['fit_max_products'],config['fit_text_bytes'],config['seed'])
        fitted_receipt=output/'fitted_receipt.json'
        if fitted_receipt.exists():
            receipt=read_json(fitted_receipt)
            if receipt['signature']!=signature:
                raise ValueError('FITTED_SIGNATURE_MISMATCH')
            verify_files(output,receipt['files'])
            fitted=joblib.load(output/'fitted.joblib')
        else:
            fitted={'classical':{c['name']:fit_classical(selected,c,config['seed']) for c in config['classical']},
                    'numeric':fit_numeric(selected,read_json(config['scaler']),config.get('pca_k')) if config.get('numeric') else None}
            temporary=output/'fitted.joblib.partial'
            joblib.dump(fitted,temporary,compress=3)
            os.replace(temporary,output/'fitted.joblib')
            receipt={'signature':signature,'fit_scope':fit_scope,'files':{'fitted.joblib':digest(output/'fitted.joblib')}}
            atomic_json(fitted_receipt,receipt)
        atomic_json(output/'fit_scope.json',fit_scope)
        backends={}
        splits=Counter(); categories=Counter(); semantic_stats={}
        cursor=iter(index_rows(index))
        chunk_index=0
        while batch:=list(itertools.islice(cursor,config['batch_products'])):
            for row in batch:
                splits[row['split']]+=1; categories[row['category']]+=1
            if chunk_index<writer.chunk_count:
                writer.write_chunk(chunk_index,batch,{})
                chunk_index+=1
                continue
            arrays={}
            chunk_meta={'review_provenance':[],'semantic_metrics':{}}
            for name,value in fitted['classical'].items():
                sparse,dense=transform_classical(value,[r['text_document'] for r in batch])
                if config['export_sparse']:
                    arrays['tfidf_'+name]=sparse
                arrays['tfidf_svd_'+name]=dense
            if fitted['numeric'] is not None:
                arrays.update(transform_numeric(fitted['numeric'],batch))
            if config.get('models'):
                reviews=[r['review'] for r in batch]
                chunk_meta['review_provenance']=[{k:r[k] for k in ('source_positions','timestamps','tie_group','has_tie','ratings','source_file','source_record')} for r in reviews]
                for name,cfg in config['models'].items():
                    if name not in backends:
                        backends[name]=backend_factory(name,cfg) if backend_factory else SentenceBackend(name,cfg)
                    backend=backends[name]
                    elapsed=time.perf_counter()
                    encoded=encode_products(reviews,backend,cfg['chunking'],cfg.get('max_chunks_per_review',1024))
                    prefix='semantic_'+name+'_'
                    for label,value in encoded.items():
                        arrays[prefix+label]=value
                    groups=[r['tie_group'] for r in reviews]
                    policy = cfg.get('tie_policy','group_mean')
                    combined = aggregate(encoded['vectors'],encoded['mask'],groups,policy)
                    for label,value in combined.items():
                        arrays[prefix+label]=value
                        if label in ('M','MT','C'):
                            arrays[prefix+label+'_l2']=normalize(value).astype(np.float32)
                    if cfg.get('tie_sensitivity',True):
                        alternative = 'canonical' if policy=='group_mean' else 'group_mean'
                        other = aggregate(encoded['vectors'],encoded['mask'],groups,alternative)
                                                                                         
                        for method,values in ((policy,combined),(alternative,other)):
                            suffix = '_tie_group_mean' if method=='group_mean' else '_canonical'
                            for label in ('M','MT','C','slope_defined'):
                                arrays[prefix+label+suffix]=values[label]
                    train=np.asarray([r['split']=='train' for r in batch])
                    valid=encoded['mask'][train]
                    drops=encoded['tokens_dropped'][train][valid]
                    tokens=TokenSummary(config['seed'],name)
                    for row_index,row in enumerate(batch):
                        if row['split']=='train':
                            for position in np.flatnonzero(encoded['mask'][row_index]):
                                tokens.add([row['category'],row['parent_asin'],int(position)],
                                    encoded['token_lengths'][row_index,position],encoded['tokens_dropped'][row_index,position])
                    chunk_meta['semantic_metrics'][name]={'n_train_reviews':int(valid.sum()),
                        'train_token_summary':tokens.export(),
                        'train_truncated_reviews':int((drops>0).sum()),'seconds_all_chunk_splits':time.perf_counter()-elapsed,
                        'backend':backend.identity,'stub':bool(backend.identity.get('stub')),
                        'peak_cuda_allocated_bytes':backend.peak_cuda_bytes() if hasattr(backend,'peak_cuda_bytes') else None}
            for fusion in config.get('fusion',[]):
                label=fusion['text_space']
                if label not in arrays:
                    raise ValueError(f'FUSION_TEXT_SPACE_UNAVAILABLE: {label}')
                arrays['fusion_'+fusion['name']]=fuse(arrays[label],arrays[fusion.get('numeric_space','numeric_distance')],fusion['text_weight'],fusion['numeric_weight'])
            writer.write_chunk(chunk_index,batch,arrays,chunk_meta)
            chunk_index+=1
            if interrupt_after is not None and chunk_index>=interrupt_after:
                raise InterruptedError('INJECTED_TEST_INTERRUPT_AFTER_COMMITTED_BATCH')
        token_totals={}
        for m in writer.iter_metadata():
            for name,value in m['semantic_metrics'].items():
                target=semantic_stats.setdefault(name,{'n_train_reviews':0,'train_truncated_reviews':0,'seconds_all_chunk_splits':0,'backend':value['backend'],'stub':value['stub'],'peak_cuda_allocated_bytes':None})
                if value.get('peak_cuda_allocated_bytes') is not None:
                    target['peak_cuda_allocated_bytes']=max(target['peak_cuda_allocated_bytes'] or 0,value['peak_cuda_allocated_bytes'])
                for key in ('n_train_reviews','train_truncated_reviews','seconds_all_chunk_splits'):
                    target[key]+=value[key]
                token_totals.setdefault(name,TokenSummary(config['seed'],name)).merge(value['train_token_summary'])
        for name,tokens in token_totals.items():
            semantic_stats[name]['train_token_summary']=tokens.export()
        if input_fingerprints(config)!=inputs:
            raise ValueError('INPUT_CHANGED_DURING_RUN')
        fitted_files={p:digest(output/p) for p in ('fitted.joblib','fitted_receipt.json','fit_scope.json')}
        elapsed=time.perf_counter()-timer
        report={'start':started,'finish':now(),'n':count,'split_counts':dict(splits),'category_counts':dict(categories),
                'seconds_this_invocation':elapsed,'new_products_per_second_this_invocation':(count-resumed_rows)/elapsed,
                'resumed_chunks_at_start':resumed_chunks,'new_products_this_invocation':count-resumed_rows,
                'peak_rss_sampled_bytes':monitor.peak,'peak_rss_method':'psutil RSS sampled every 10ms; may miss transients',
                'artifact_bytes_before_manifest':sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
                'classical':{k:v['report'] for k,v in fitted['classical'].items()},'semantic':semantic_stats,
                'numeric':fitted['numeric']['report'] if fitted['numeric'] else None,'fit_scope':fit_scope}
        index.unlink()
        return writer.publish(count,fitted_files,report)
