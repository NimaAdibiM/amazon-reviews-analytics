"""Read each JSONL stream once and fingerprint the consumed input."""
from collections import Counter
import gzip
import hashlib
from itertools import islice
import json
from pathlib import Path

import quality_audit as quality
import structure_audit as structure


class HashingReader:
    def __init__(self, handle, digest):
        self.handle = handle
        self.digest = digest

    def read(self, size=-1):
        data = self.handle.read(size)
        self.digest.update(data)
        return data


def load_config(path):
    payload = Path(path).read_bytes()
    config = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Config must be an object.")
    root = structure.resolve(Path(path).resolve().parent, config.get("project_root", ".."))
    unit = config.get("timestamp_unit", "ms")
    if unit not in structure.TIMESTAMP_UNITS:
        raise ValueError("timestamp_unit must be s, ms, us or ns.")
    for field, default, minimum in (("batch_size", 25000, 1), ("progress_every_lines", 100000, 0)):
        value = config.setdefault(field, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid {field}.")
    return config, root, unit, hashlib.sha256(payload).hexdigest()


def scan(path, limit, consume, progress):
    before = path.stat()
    counts = Counter(lines=0, objects=0)
    content_hash = hashlib.sha256()
    file_hash = hashlib.sha256()
    compressed = path.name.lower().endswith(".gz")
    decoded_bytes = 0
    with path.open("rb") as stored:
        stream = gzip.GzipFile(fileobj=HashingReader(stored, file_hash)) if compressed else stored
        try:
            for line_no, line in enumerate(islice(stream, limit) if limit else stream, 1):
                decoded_bytes += len(line)
                content_hash.update(line)
                counts["lines"] += 1
                if not line.strip():
                    counts["blank"] += 1
                else:
                    try:
                        row = quality.parse_line(line, first_line=line_no == 1)
                    except (ValueError, RecursionError) as exc:
                        counts[type(exc).__name__] += 1
                    else:
                        if isinstance(row, dict):
                            counts["objects"] += 1
                            consume(row)
                        else:
                            counts["nonobject"] += 1
                progress(line_no)
            if compressed and not limit and stored.tell() != before.st_size:
                raise OSError("Gzip stream did not consume the expected stored file size.")
        finally:
            if compressed:
                stream.close()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev, before.st_ctime_ns) != (
        after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev, after.st_ctime_ns
    ):
        raise OSError("Input changed during the scan.")
    content_digest = content_hash.hexdigest()
    return dict(path=str(path), compressed=compressed, stored_bytes=before.st_size,
                mtime_ns=before.st_mtime_ns, scanned_decoded_bytes=decoded_bytes,
                scanned_content_sha256=content_digest,
                stored_file_sha256=(file_hash.hexdigest() if compressed else content_digest) if not limit else None,
                scan_limit=limit or None, eof_verified=not limit or counts["lines"] < limit,
                line_unit="physical lines in decompressed stream",
                excluded_from_parsed_objects=counts["lines"]-counts["objects"], counts=dict(counts))
