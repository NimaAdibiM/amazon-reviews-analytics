"""Build the observed 5+10 review cohort with a disk-backed, bounded index."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import mmap
import multiprocessing
import os
import shutil
import struct
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


REVIEW_TYPES = {
    "parent_asin": str,
    "timestamp": int,
    "rating": (int, float),
    "title": str,
    "text": str,
    "verified_purchase": bool,
    "user_id": str,
    "asin": str,
    "helpful_vote": int,
    "images": list,
}
REVIEW_FIELDS = tuple(REVIEW_TYPES)
STATES = ("absent", "null", "blank_string", "empty_array", "empty_object", "wrong_type", "present")
MISSING = object()
PRESENT = STATES.index("present")
MIN_TIMESTAMP_MS = -62_135_596_800_000
MAX_TIMESTAMP_MS = 253_402_300_799_999
RECORD = struct.Struct("<17sqQI B")
RECORD_DTYPE = np.dtype([
    ("parent", "V17"),
    ("timestamp", "<i8"),
    ("offset", "<u8"),
    ("length", "<u4"),
    ("rating", "u1"),
])
assert RECORD.size == RECORD_DTYPE.itemsize
AUDIT_VERSION = 1


class InvalidJSON(ValueError):
    pass


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise InvalidJSON("duplicate JSON object key")
    return result


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise InvalidJSON("nonfinite number")
    return number


def _parse(raw, offset):
    encoding = "utf-8-sig" if offset == 0 else "utf-8"
    return json.loads(raw.decode(encoding), object_pairs_hook=_unique_object,
                      parse_float=_finite_float, parse_constant=_finite_float)


def _field_state(value, expected_type):
    if value is MISSING:
        return 0
    if value is None:
        return 1
    actual_type = type(value)
    if actual_type is str and not value.strip():
        return 2
    if actual_type is list and not value:
        return 3
    if actual_type is dict and not value:
        return 4
    if actual_type == expected_type or (
        expected_type is REVIEW_TYPES["rating"] and actual_type in expected_type
    ):
        return PRESENT
    return 5


def _parent_key(parent):
    if type(parent) is not str or not parent.strip():
        return None
    try:
        encoded = parent.encode("utf-8")
    except UnicodeEncodeError:
        return None
    if len(encoded) == 10 and encoded.isalnum():
        return b"\x00" + encoded + b"\x00" * 6
    return b"\x01" + hashlib.blake2b(encoded, digest_size=16).digest()


def _scan_chunk(task):
    source, start, end, scratch, worker, shard_count = task
    field_counts = [[0] * len(STATES) for _ in REVIEW_FIELDS]
    core_invalid = [0] * 8
    counts = dict(physical_lines=0, blank_lines=0, rejected_json_lines=0, nonobject_lines=0,
                  parsed_reviews=0, valid_core_reviews=0)
    field_items = tuple(REVIEW_TYPES.items())
    buffers = [bytearray() for _ in range(shard_count)]
    destinations = [scratch / f"{worker}.{shard}.index" for shard in range(shard_count)]

    with source.open("rb") as stream:
        stream.seek(max(start - 1, 0))
        if start > 0 and stream.read(1) != b"\n":
            stream.readline()
        offset = stream.tell()
        while offset < end:
            line = stream.readline()
            if not line:
                break
            current = offset
            offset += len(line)
            counts["physical_lines"] += 1
            if not line.strip():
                counts["blank_lines"] += 1
                continue
            try:
                row = _parse(line, current)
            except (ValueError, UnicodeError, RecursionError):
                counts["rejected_json_lines"] += 1
                continue
            if not isinstance(row, dict):
                counts["nonobject_lines"] += 1
                continue
            counts["parsed_reviews"] += 1
            for (name, expected), census in zip(field_items, field_counts):
                census[_field_state(row.get(name, MISSING), expected)] += 1
            parent = _parent_key(row.get("parent_asin"))
            timestamp = row.get("timestamp")
            rating = row.get("rating")
            time_ok = type(timestamp) is int and MIN_TIMESTAMP_MS <= timestamp <= MAX_TIMESTAMP_MS
            rating_ok = type(rating) in (int, float) and rating in (1, 2, 3, 4, 5)
            mask = int(parent is None) + 2 * int(not time_ok) + 4 * int(not rating_ok)
            core_invalid[mask] += 1
            if mask:
                continue
            counts["valid_core_reviews"] += 1
                                                                             
            if parent[0] == 0:
                shard = int.from_bytes(parent[1:11], "little") % 2_147_483_647 % shard_count
            else:
                shard = int.from_bytes(parent[1:5], "little") % shard_count
            buffer = buffers[shard]
            buffer.extend(RECORD.pack(parent, timestamp, current, len(line), int(rating)))
            if len(buffer) >= 1024 * 1024:
                with destinations[shard].open("ab") as output:
                    output.write(buffer)
                buffer.clear()
    for buffer, destination in zip(buffers, destinations):
        if buffer:
            with destination.open("ab") as output:
                output.write(buffer)
    return counts, field_counts, core_invalid


def _source_row(source, record):
    offset = int(record["offset"])
    return _parse(source[offset:offset + int(record["length"])], offset)


def _canonical(row):
    return json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _parent_groups(source, records, order):
    parent = bytes(records["parent"][order[0]])
    if parent[0] == 0:
        yield parent[1:11].decode("ascii"), order
        return
                                                                         
    parents = {}
    for position in order:
        real_id = _source_row(source, records[position])["parent_asin"]
        parents.setdefault(real_id, []).append(position)
    for real_id, positions in parents.items():
        yield real_id, np.asarray(positions, dtype=order.dtype)


def _dedup_positions(source, records, positions, timestamps):
    """Only equal timestamps within a parent can contain identical full rows."""
    repeats = np.flatnonzero(timestamps[1:] == timestamps[:-1])
    if not repeats.size:
        return positions
    keep = np.ones(len(positions), dtype=np.bool_)
    previous = -2
    seen = set()
    for repeat in repeats:
        if repeat != previous + 1:
            seen = {_canonical(_source_row(source, records[positions[repeat]]))}
        current = _canonical(_source_row(source, records[positions[repeat + 1]]))
        if current in seen:
            keep[repeat + 1] = False
        else:
            seen.add(current)
        previous = repeat
    return positions[keep]


def _output_text(value):
    if type(value) is not str or not value.strip():
        return None
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return value


def _cohort_row(category, parent, source, records, positions):
    early = [_source_row(source, record) for record in records[positions[:5]]]
    row = {
        "category": category,
        "parent_asin": parent,
        "early_ratings": records["rating"][positions[:5]].tolist(),
        "early_timestamps": records["timestamp"][positions[:5]].tolist(),
        "early_titles": [_output_text(record.get("title")) for record in early],
        "early_bodies": [_output_text(record.get("text")) for record in early],
        "early_verified": [record["verified_purchase"] if type(record.get("verified_purchase")) is bool
                           else None for record in early],
        "n_low_future": int(np.count_nonzero(records["rating"][positions[5:15]] <= 2)),
        "t5": int(records["timestamp"][positions[4]]),
        "t15": int(records["timestamp"][positions[14]]),
    }
    for input_field, output_field in (("title", "titles"), ("text", "bodies"),
                                      ("verified_purchase", "verified")):
        row[f"early_{output_field}_states"] = [
            STATES[_field_state(record.get(input_field, MISSING), REVIEW_TYPES[input_field])]
            for record in early
        ]
    for value_field in ("titles", "bodies"):
        states = row[f"early_{value_field}_states"]
        for i, value in enumerate(row[f"early_{value_field}"]):
            if value is None and states[i] == "present":
                states[i] = "invalid_unicode"
    return row


def _process_shard(task):
    source_path, category, scratch, shard, worker_count = task
    parts = [scratch / f"{worker}.{shard}.index" for worker in range(worker_count)]
    parts = [path for path in parts if path.is_file()]
    output_path = scratch / f"{shard}.jsonl"
    counts = dict(products_with_valid_core=0, products_with_15_raw_core=0,
                  exact_duplicate_core_extras=0, distinct_valid_core_reviews=0,
                  products_with_5_distinct_core=0, products_with_15_distinct_core=0,
                  tied_5_6_products=0, tied_15_16_products=0, tied_at_both_boundaries=0,
                  excluded_tied_products=0, eligible_products=0)
    if not parts:
        return counts
    records = np.concatenate([np.fromfile(part, dtype=RECORD_DTYPE) for part in parts])
                                                                                 
    order = np.lexsort((records["offset"], records["timestamp"], records["parent"]))
    ordered_parents = records["parent"][order]
    breaks = np.flatnonzero(ordered_parents[1:] != ordered_parents[:-1]) + 1
    begins, ends = np.r_[0, breaks], np.r_[breaks, len(order)]
    with source_path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as source, \
            output_path.open("w", encoding="utf-8") as output:
        for start, end in zip(begins, ends):
            for parent, positions in _parent_groups(source, records, order[start:end]):
                n_raw = len(positions)
                counts["products_with_valid_core"] += 1
                counts["products_with_15_raw_core"] += n_raw >= 15
                if n_raw > 1:
                    positions = _dedup_positions(source, records, positions, records["timestamp"][positions])
                n_distinct = len(positions)
                counts["exact_duplicate_core_extras"] += n_raw - n_distinct
                counts["distinct_valid_core_reviews"] += n_distinct
                counts["products_with_5_distinct_core"] += n_distinct >= 5
                counts["products_with_15_distinct_core"] += n_distinct >= 15
                if n_distinct < 15:
                    continue
                ts = records["timestamp"]
                tie_first = bool(ts[positions[4]] == ts[positions[5]])
                tie_last = n_distinct > 15 and bool(ts[positions[14]] == ts[positions[15]])
                counts["tied_5_6_products"] += tie_first
                counts["tied_15_16_products"] += tie_last
                counts["tied_at_both_boundaries"] += tie_first and tie_last
                counts["excluded_tied_products"] += tie_first or tie_last
                if tie_first or tie_last:
                    continue
                counts["eligible_products"] += 1
                output.write(json.dumps(_cohort_row(category, parent, source, records, positions),
                                        ensure_ascii=False, separators=(",", ":")) + "\n")
    for part in parts:
        part.unlink()
    return counts


def _add_counts(target, observed):
    for name, count in observed.items():
        target[name] = target.get(name, 0) + count


def _fingerprint(path):
    result = path.stat()
    return {
        "path": str(path),
        "bytes": result.st_size,
        "mtime_ns": result.st_mtime_ns,
        "inode": result.st_ino,
        "device": result.st_dev,
    }


def _run(executor, function, jobs):
    return map(function, jobs) if executor is None else executor.map(function, jobs)


def ingest_category(reviews_path, category, output_dir, *, workers=4, shards=64, force=False):
    """Publish aggregate review quality and product examples for one category.

    Each worker indexes a disjoint source byte interval. Equal parent IDs always
    land in the same shard. Product examples retain the five original review
    positions and expose absent/unusable text as null values with explicit states.
    """
    if workers < 1 or shards < 1:
        raise ValueError("workers and shards must be positive")
    source = Path(reviews_path).resolve(strict=True)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cohort_path = out / "reviews_cohort.jsonl"
    audit_path = out / "reviews_audit.json"
    source_identity = _fingerprint(source)
    config = {"audit_version": AUDIT_VERSION, "source": source_identity, "category": category,
              "shards": shards}
    if not force and audit_path.is_file() and cohort_path.is_file():
        existing = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing["config"] == config and cohort_path.stat().st_size == existing["cohort_bytes"]:
            return existing

    with tempfile.TemporaryDirectory(prefix="review-index-", dir=out) as temp:
        scratch = Path(temp)
        local_source = source
        if source.name.lower().endswith(".gz"):
            local_source = scratch / "reviews.jsonl"
            with gzip.open(source, "rb") as raw, local_source.open("wb") as destination:
                shutil.copyfileobj(raw, destination, length=8 * 1024 * 1024)

        size = local_source.stat().st_size
        print(f"[{category}] indexing {size:,} review bytes", flush=True)
        jobs = [
            (local_source, size * worker // workers, size * (worker + 1) // workers,
             scratch, worker, shards)
            for worker in range(workers)
        ]
        field_totals = {name: dict.fromkeys(STATES, 0) for name in REVIEW_FIELDS}
        counts = {}
        invalid = [0] * 8
        executor = (ProcessPoolExecutor(max_workers=workers,
                                        mp_context=multiprocessing.get_context("spawn"))
                    if workers > 1 else None)
        try:
            for part, census, masks in _run(executor, _scan_chunk, jobs):
                _add_counts(counts, part)
                invalid = [a + b for a, b in zip(invalid, masks)]
                for state_counts, name in zip(census, REVIEW_FIELDS):
                    _add_counts(field_totals[name], dict(zip(STATES, state_counts)))
            print(f"[{category}] grouping {counts['valid_core_reviews']:,} valid reviews",
                  flush=True)
            tasks = [(local_source, category, scratch, shard, workers) for shard in range(shards)]
            for result in _run(executor, _process_shard, tasks):
                _add_counts(counts, result)
        finally:
            if executor is not None:
                executor.shutdown(cancel_futures=True)

        if _fingerprint(source) != source_identity:
            raise RuntimeError(f"Review input changed during scan: {source}")
        if any(sum(states.values()) != counts["parsed_reviews"] for states in field_totals.values()):
            raise RuntimeError("Review field-state counts do not reconcile")
        if sum(invalid) != counts["parsed_reviews"] or invalid[0] != counts["valid_core_reviews"]:
            raise RuntimeError("Review validity counts do not reconcile")
        if counts["valid_core_reviews"] != (counts["distinct_valid_core_reviews"] +
                                             counts["exact_duplicate_core_extras"]):
            raise RuntimeError("Review deduplication counts do not reconcile")
        if counts["eligible_products"] != (counts["products_with_15_distinct_core"] -
                                            counts["excluded_tied_products"]):
            raise RuntimeError("Product eligibility counts do not reconcile")

        combined = scratch / "combined.jsonl"
        with combined.open("wb") as destination:
            for shard in range(shards):
                path = scratch / f"{shard}.jsonl"
                if path.exists():
                    with path.open("rb") as staged:
                        shutil.copyfileobj(staged, destination, length=8 * 1024 * 1024)
        audit = {
            "config": config,
            "counts": counts,
            "review_field_states": field_totals,
            "core_validity_patterns": {
                "+".join(name for index, name in enumerate(("parent_asin", "timestamp", "rating"))
                         if mask & (1 << index)) or "all_valid": count
                for mask, count in enumerate(invalid)
            },
            "cohort_path": str(cohort_path),
            "cohort_bytes": combined.stat().st_size,
        }
        staged_audit = scratch / "audit.json"
        staged_audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
        os.replace(combined, cohort_path)
        os.replace(staged_audit, audit_path)
    return audit
