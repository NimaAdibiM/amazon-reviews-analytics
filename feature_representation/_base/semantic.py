import re
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from .common import atomic_json, digest, environment, now, object_hash, read_json

MODELS = {'bge': 'BAAI/bge-small-en-v1.5', 'e5': 'intfloat/multilingual-e5-small'}


def prepare_model(name, destination, max_bytes=1500000000):
    if name not in MODELS:
        raise ValueError('UNSUPPORTED_MODEL')
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError('OPTIONAL_DEPENDENCY_MISSING: install requirements-stage-a-models.txt') from exc
    destination = Path(destination)
    if destination.exists():
        raise ValueError('MODEL_DESTINATION_EXISTS')
    info = HfApi(token=False).model_info(MODELS[name], files_metadata=True)
    if not re.fullmatch(r'[0-9a-f]{40}', info.sha):
        raise ValueError('UNRESOLVED_MODEL_REVISION')
    files = [x for x in info.siblings if x.rfilename.endswith(('.json','.txt','.model','.safetensors','.md')) and not x.rfilename.startswith(('onnx/','openvino/'))]
    if not any(x.rfilename.endswith('.safetensors') for x in files) or any(x.size is None for x in files):
        raise ValueError('MODEL_FILE_METADATA_MISSING')
    size = sum(x.size for x in files)
    if size > max_bytes:
        raise ValueError('MODEL_DOWNLOAD_BUDGET_EXCEEDED')
    snapshot_download(MODELS[name], revision=info.sha, local_dir=str(destination), allow_patterns=[x.rfilename for x in files], token=False)
    hashes = {x.rfilename:digest(destination/x.rfilename) for x in files}
    receipt = {'model_id':MODELS[name], 'revision':info.sha, 'source':'https://huggingface.co/'+MODELS[name],
               'license':getattr(info.card_data, 'license', None), 'expected_download_bytes':size,
               'actual_file_bytes':sum((destination/n).stat().st_size for n in hashes), 'files':hashes,'created_at':now()}
    atomic_json(destination/'STAGE_A_MODEL_SOURCE.json', receipt)
    return receipt


def model_identity(name, config):
    if name not in MODELS:
        raise ValueError('UNSUPPORTED_MODEL')
    source = Path(config['model_dir'])
    receipt_path = source/'STAGE_A_MODEL_SOURCE.json'
    if not receipt_path.is_file():
        raise FileNotFoundError(f'MODEL_NOT_AVAILABLE: {source}; run prepare-model explicitly')
    receipt = read_json(receipt_path)
    if receipt['model_id'] != MODELS[name] or not re.fullmatch(r'[0-9a-f]{40}',receipt['revision']):
        raise ValueError('MODEL_SOURCE_IDENTITY_MISMATCH')
    for relative, sha in receipt['files'].items():
        p = (source/relative).resolve()
        if not p.is_relative_to(source.resolve()) or not p.is_file() or digest(p) != sha:
            raise ValueError('MODEL_FILE_HASH_MISMATCH')
    return {'model_id':MODELS[name], 'revision':receipt['revision'], 'model_files_hash':object_hash(receipt['files']),
            'license':receipt['license'], 'model_file_bytes':receipt['actual_file_bytes'],
            'prefix':'query: ' if name=='e5' else '', 'trust_remote_code':False, 'local_files_only':True,
            'normalization':config['normalize'], 'device_requested':config['device'], 'stub':False}


class SentenceBackend:
    def __init__(self, name, config):
        self.identity = model_identity(name,config)
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError('OPTIONAL_DEPENDENCY_MISSING: torch/sentence-transformers') from exc
        torch.manual_seed(config.get('seed',42))
        device = config['device']
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(config['model_dir'], device=device, trust_remote_code=False,
                                         local_files_only=True, model_kwargs={'use_safetensors':True})
        self.tokenizer = self.model.tokenizer
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.max_length = int(self.model.max_seq_length)
        self.prefix = self.identity['prefix']
        self.normalize = config['normalize']
        self.batch_size = config['batch_size']
        self.identity.update({'dimension':self.dimension,'max_seq_length':self.max_length,'device_actual':device,
                              'tokenizer_class':type(self.tokenizer).__name__, 'versions':environment()['versions']})
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()

    def encode(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=self.normalize, prompt='').astype(np.float32)

    def peak_cuda_bytes(self):
        if not self.identity['device_actual'].startswith('cuda'):
            return None
        import torch
        return int(torch.cuda.max_memory_allocated())


def tokenize_chunks(text, backend):
    tokenizer = backend.tokenizer
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    overhead = len(tokenizer.encode(backend.prefix, add_special_tokens=True, truncation=False))
    width = backend.max_length - overhead - 2
    if width < 1:
        raise ValueError('TOKEN_BUDGET_TOO_SMALL')
    pending = [ids[i:i+width] for i in range(0,len(ids),width)]
    pieces = []
    while pending:
        part = pending.pop(0)
        decoded = tokenizer.decode(part, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        candidate = backend.prefix+decoded
        if len(tokenizer.encode(candidate, add_special_tokens=True, truncation=False)) <= backend.max_length:
            pieces.append((candidate,len(part)))
        elif len(part)>1:
            half=len(part)//2
            pending[0:0]=[part[:half],part[half:]]
        else:
            raise ValueError('TOKENIZER_CHUNK_CANNOT_FIT')
    return pieces


def encode_products(records, backend, chunking=False, max_chunks_per_review=1024):
    n,d = len(records),backend.dimension
    vectors = np.zeros((n,5,d),dtype=np.float32)
    mask = np.asarray([r['mask'] for r in records],dtype=bool)
    lengths = np.zeros((n,5),dtype=np.int64)
    dropped = np.zeros((n,5),dtype=np.int64)
    baseline_dropped = np.zeros((n,5),dtype=np.int64)
    chunk_counts = np.zeros((n,5),dtype=np.int32)
    pending, owners, weights = [],[],[]
    for i,row in enumerate(records):
        for j,text in enumerate(row['texts']):
            if not mask[i,j]:
                continue
            value=backend.prefix+text
            length=len(backend.tokenizer.encode(value,add_special_tokens=True,truncation=False))
            lengths[i,j]=length
            baseline_dropped[i,j]=max(0,length-backend.max_length)
            if chunking and length>backend.max_length:
                parts=tokenize_chunks(text,backend)
            else:
                parts=[(value,1)]
                dropped[i,j]=baseline_dropped[i,j]
            if len(parts)>max_chunks_per_review:
                raise ValueError('REVIEW_CHUNK_BUDGET_EXCEEDED')
            chunk_counts[i,j]=len(parts)
            total=sum(weight for _,weight in parts)
            for part,weight in parts:
                pending.append(part); owners.append((i,j)); weights.append(weight/total)
    for start in range(0,len(pending),backend.batch_size):
        end=start+backend.batch_size
        batch=backend.encode(pending[start:end])
        if batch.shape != (len(pending[start:end]),d) or not np.isfinite(batch).all():
            raise ValueError('BACKEND_EMBEDDING_INVALID')
        for v,(i,j),weight in zip(batch,owners[start:end],weights[start:end]):
            vectors[i,j]+=v*weight
    if backend.normalize and mask.any():
        vectors[mask]=normalize(vectors[mask]).astype(np.float32)
    return {'vectors':vectors, 'mask':mask, 'token_lengths':lengths,'tokens_dropped':dropped,
            'baseline_tokens_dropped':baseline_dropped,'chunk_counts':chunk_counts}
