import re
from tempfile import TemporaryDirectory
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix='.model-') as temporary:
        stage = Path(temporary) / 'snapshot'
        snapshot_download(MODELS[name], revision=info.sha, local_dir=str(stage),
                          allow_patterns=[x.rfilename for x in files], token=False)
        hashes = {}
        for entry in files:
            path = stage / entry.rfilename
            sha = digest(path)
            official = getattr(getattr(entry, 'lfs', None), 'sha256', None)
            if path.stat().st_size != entry.size or (official and sha != official):
                raise ValueError('INCOMPLETE_MODEL_DOWNLOAD')
            hashes[entry.rfilename] = sha
        receipt = {'model_id':MODELS[name], 'revision':info.sha, 'source':'https://huggingface.co/'+MODELS[name],
                   'license':getattr(info.card_data, 'license', None), 'expected_download_bytes':size,
                   'actual_file_bytes':size, 'files':hashes, 'created_at':now()}
        atomic_json(stage / 'STAGE_A_MODEL_SOURCE.json', receipt)
        if destination.exists():
            raise ValueError('MODEL_DESTINATION_EXISTS')
        stage.rename(destination)
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
    files = receipt['files']
    if 'config.json' not in files or not any(x.endswith('.safetensors') for x in files):
        raise ValueError('INCOMPLETE_MODEL_RECEIPT')
    for relative, sha in files.items():
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

    def encode_token_ids(self, content):
        """Encode exact tokenizer IDs; a piece starting mid-word stays intact."""
        import torch
        features = [{'input_ids':self.tokenizer.build_inputs_with_special_tokens(ids)} for ids in content]
        batch = self.tokenizer.pad(features, padding=True, return_tensors='pt').to(self.model.device)
        self.model.eval()
        with torch.inference_mode():
            vectors = self.model(dict(batch))['sentence_embedding']
            if self.normalize:
                vectors = torch.nn.functional.normalize(vectors, dim=1)
        return vectors.cpu().float().numpy()

    def peak_cuda_bytes(self):
        if not self.identity['device_actual'].startswith('cuda'):
            return None
        import torch
        return int(torch.cuda.max_memory_allocated())


def _id_pieces(text, backend, chunking, cap):
    tokenizer = backend.tokenizer
    ids = tokenizer.encode(backend.prefix + text, add_special_tokens=False, truncation=False)
    prefix = tokenizer.encode(backend.prefix.rstrip(), add_special_tokens=False, truncation=False) if backend.prefix else []
    special = len(tokenizer.encode('', add_special_tokens=True, truncation=False))
    width = backend.max_length - special - len(prefix)
    if width < 1 or ids[:len(prefix)] != prefix:
        raise ValueError('TOKEN_BUDGET_OR_PREFIX_MISMATCH')
    content = ids[len(prefix):]
    length = len(ids) + special
    count = max(1, (len(content)+width-1)//width) if chunking else 1
    if count > cap:
        raise ValueError('REVIEW_CHUNK_BUDGET_EXCEEDED')
    pieces = []
    for i in range(count):
        piece = content[i*width:(i+1)*width]
        pieces.append((prefix + piece, max(1,len(piece))))
    return length, pieces


def _lossless_text(ids, backend):
    tokenizer = backend.tokenizer
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if tokenizer.encode(text, add_special_tokens=False, truncation=False) != ids:
        raise ValueError('BACKEND_MUST_SUPPORT_EXACT_TOKEN_IDS')
    return text


def tokenize_chunks(text, backend):
    """Text-only adapter. Never silently modify tokens at a chunk boundary."""
    _, pieces = _id_pieces(text, backend, True, 2**31)
    return [(_lossless_text(ids, backend), weight) for ids, weight in pieces]


def encode_products(records, backend, chunking=False, max_chunks_per_review=1024):
    n, d = len(records), backend.dimension
    if type(max_chunks_per_review) is not int or max_chunks_per_review < 1 or backend.batch_size < 1:
        raise ValueError('INVALID_EMBEDDING_BUDGET')
    vectors = np.zeros((n,5,d),dtype=np.float32)
    mask = np.empty((n,5),dtype=bool)
    lengths = np.zeros((n,5),dtype=np.int64)
    dropped = np.zeros_like(lengths)
    baseline_dropped = np.zeros_like(lengths)
    chunk_counts = np.zeros((n,5),dtype=np.int32)
    exact = hasattr(backend,'encode_token_ids')
    pending, owners = [], []

    def flush():
        batch = backend.encode_token_ids(pending) if exact else backend.encode(pending)
        if batch.shape != (len(pending),d) or not np.isfinite(batch).all():
            raise ValueError('BACKEND_EMBEDDING_INVALID')
        for vector,(i,j,weight) in zip(batch,owners):
            vectors[i,j] += vector*weight
        pending.clear()
        owners.clear()

    for i,row in enumerate(records):
        texts = row['texts']
        observed = np.asarray(row['mask'])
        if (len(texts)!=5 or observed.shape!=(5,) or observed.dtype!=bool
                or any(not isinstance(t,str) for t in texts)
                or not np.array_equal(observed,[bool(t.strip()) for t in texts])):
            raise ValueError('INVALID_FIVE_TEXTS_OR_MASK')
        mask[i] = observed
        for j,text in enumerate(texts):
            if not observed[j]:
                continue
            length, pieces = _id_pieces(text, backend, chunking, max_chunks_per_review)
            lengths[i,j] = length
            baseline_dropped[i,j] = max(0,length-backend.max_length)
            dropped[i,j] = 0 if chunking else baseline_dropped[i,j]
            chunk_counts[i,j] = len(pieces)
            total = sum(weight for _,weight in pieces)
            for ids, weight in pieces:
                if exact:
                    value = ids
                elif len(pieces)==1:
                    value = backend.prefix+text
                else:
                    value = _lossless_text(ids,backend)
                pending.append(value)
                owners.append((i,j,weight/total))
                if len(pending) == backend.batch_size:
                    flush()
    if pending:
        flush()
    if backend.normalize and mask.any():
        vectors[mask]=normalize(vectors[mask]).astype(np.float32)
    return {'vectors':vectors, 'mask':mask, 'token_lengths':lengths,'tokens_dropped':dropped,
            'baseline_tokens_dropped':baseline_dropped,'chunk_counts':chunk_counts}
