from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import csv
import gzip
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import platform
import sqlite3
import sys
import time
from uuid import uuid4

VERSION = "1.26.0-s25-s26"
# These are the four categories explicitly selected for this project.
PROJECT_CATEGORIES = (
    "Electronics", "Beauty_and_Personal_Care", "Appliances", "All_Beauty"
)
TABLES = ("review_products", "review_users", "review_asins", "metadata_products")
# Published Amazon Reviews'23 relationship: both sides use parent_asin.
# This is a description of the intended join, not a claim of key uniqueness.
PARENT_JOIN_FIELD = "parent_asin"
# Review timestamp is Unix time in the publisher's field table.  A published
# sample uses a millisecond-scale value under the key sort_timestamp, not
# timestamp: the unit in each LOCAL file must still be verified (S18/S19).
REVIEW_TIMESTAMP_FIELD = "timestamp"
TIMESTAMP_UNITS = {"s": 1, "ms": 1000, "us": 1_000_000, "ns": 1_000_000_000}
DEFAULT_TIMESTAMP_UNIT = "ms"
# The publisher's May 1996--September 2023 range is a corpus-level
# plausibility reference, not a strict validity constraint for local records.
DEFAULT_REFERENCE_START = "1996-05-01T00:00:00+00:00"
DEFAULT_REFERENCE_END_EXCLUSIVE = "2023-10-01T00:00:00+00:00"
# A single mapping defines which identifier belongs to which input role.
# This is also used to label the CSV/JSON outputs, so that internal SQLite
# table names are not mistaken for original Amazon field names.
IDENTIFIER_FIELDS = {
    "reviews": {
        "user_id": "review_users",
        "asin": "review_asins",
        PARENT_JOIN_FIELD: "review_products",
    },
    "metadata": {
        PARENT_JOIN_FIELD: "metadata_products",
    },
}
# Reference field names published by McAuley Lab for Amazon Reviews'23.
# Unexpected fields are reported, never silently discarded or treated as errors.
# Field meanings and published non-null types depend on the FILE ROLE.
# Values are expected *raw* Python JSON-parser types, NOT inferred pandas dtypes.
# This reference is descriptive: mismatches are counted, not automatically fixed.
FIELD_SPECS = {
    "reviews": {
        "rating": (("integer", "number"), "ordinal_rating", "Star rating attached to a review"),
        "title": (("string",), "text", "Headline of a user's review"),
        "text": (("string",), "text", "Full text of a user's review"),
        "images": (("array",), "nested", "Image descriptors provided with the review"),
        "asin": (("string",), "identifier", "Identifier of the reviewed product variant"),
        "parent_asin": (("string",), "join_key", "Product-parent ID used to join to metadata"),
        "user_id": (("string",), "identifier", "Reviewer identifier; not a real-world identity"),
        "timestamp": (("integer",), "temporal", "Review timestamp in Unix epoch units; confirm units separately"),
        "helpful_vote": (("integer",), "count", "Cumulative helpful-vote count; event-time availability unverified"),
        "verified_purchase": (("boolean",), "binary", "Whether the purchase is marked verified"),
    },
    "metadata": {
        "main_category": (("string",), "categorical", "Product category shown in metadata, not the filename category"),
        "title": (("string",), "text", "Product title, NOT a review headline"),
        "average_rating": (("integer", "number"), "numeric_aggregate", "Crawl-time product mean rating; not historical"),
        "rating_number": (("integer",), "count_aggregate", "Crawl-time product rating count; not historical"),
        "features": (("array",), "nested", "Bullet-point product features"),
        "description": (("array",), "nested", "Product description fragments"),
        "price": (("integer", "number"), "numeric_snapshot", "Price in USD at crawl time, not historical price"),
        "images": (("array",), "nested", "Product image descriptors, NOT user review images"),
        "videos": (("array",), "nested", "Product video descriptors"),
        "store": (("string",), "categorical", "Store name or label"),
        "categories": (("array",), "nested", "Product category hierarchy"),
        "details": (("object",), "nested", "Variable product attributes"),
        "parent_asin": (("string",), "candidate_primary_key", "Parent product ID; uniqueness checked separately"),
        "bought_together": (("array",), "nested", "Website-recommended product bundles, if present"),
    },
}
DOCUMENTED_FIELDS = {role: frozenset(specs) for role, specs in FIELD_SPECS.items()}
# S21: a review is the sample of interest. Product metadata lives in a
# DIFFERENT file and can only be attached to a review through parent_asin.
# Image fields contain descriptors/URLs, not downloaded or verified image bytes.
# Groups are a map of possible information, not claims of row-level usability.
# Row-level modality availability is measured separately in S22.
MODALITY_MAP = (
    ("review_text", "reviews", ("title", "text"),
     "Review headline/body; one or both can be missing, null or blank."),
    ("review_rating", "reviews", ("rating",),
     "The rating of this review; not the product-page average rating."),
    ("review_images", "reviews", ("images",),
     "Descriptors and URLs of reviewer-posted images; actual media is not validated."),
    ("review_time", "reviews", (REVIEW_TIMESTAMP_FIELD,),
     "Review event time; conversion and unit assumptions are audited separately."),
    ("user_item_interaction", "reviews",
     ("user_id", "asin", PARENT_JOIN_FIELD, "verified_purchase", "helpful_vote"),
     "A collected review is a user-item interaction; this is not a clickstream "
     "or a complete history of purchases or votes."),
    ("product_metadata", "metadata",
     ("main_category", "title", "features", "description", "store", "categories",
      "details", "bought_together", "videos", "average_rating", "rating_number"),
     "Product-page/crawl-time metadata. average_rating and rating_number are "
     "current aggregates and must NOT be assumed available at earlier review times."),
    ("product_images", "metadata", ("images",),
     "Product-page image descriptors and URLs; different from review images."),
    ("product_price", "metadata", ("price",),
     "Product-page crawl-time price, not the price at each review timestamp."),
)
# S22: per-record content tests. A "usable" modality here means a NONEMPTY
# raw descriptor of the expected shape, NOT validated media/downloadability,
# correctness of a claim, or historical feature availability.
PRODUCT_DESCRIPTION_FIELDS = (
    "main_category", "title", "features", "description", "videos",
    "store", "categories", "details", "bought_together",
)
META_COVERAGE_FIELDS = ("product_metadata", "product_images", "product_price")
REVIEW_COVERAGE_FIELDS = (
    "review_text", "review_rating", "review_images", "review_time", "user_item_interaction"
)


def has_nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def has_nonempty_content(value):
    """A structural content signal: find a nonblank scalar even inside containers.

    Unlike bool(list_or_dict), a container made only of nulls/blank strings
    is not counted as informative. This does NOT validate the factual meaning
    or historical availability of product metadata.
    """
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            if item.strip():
                return True
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, bool) or isinstance(item, int):
            return True  # 0 and False can be informative product attributes.
        elif isinstance(item, float) and math.isfinite(item):
            return True
    return False


def has_image_descriptor(value):
    """Local usable URL/descriptor, not proof an external image is reachable."""
    return isinstance(value, list) and any(
        isinstance(item, dict) and any(
            (key == "url" or key.endswith("_url") or key in ("thumb", "large", "hi_res"))
            and has_nonempty_text(url)
            for key, url in item.items()
        ) for item in value
    )


def review_modality_flags(row, id_states):
    value = row.get("rating")
    rating_ok = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and 1 <= value <= 5 and (isinstance(value, int) or math.isfinite(value)))
    return {
        "review_text": has_nonempty_text(row.get("title")) or has_nonempty_text(row.get("text")),
        "review_text_body": has_nonempty_text(row.get("text")),  # separately useful for S23
        "review_rating": rating_ok,
        "review_images": has_image_descriptor(row.get("images")),
        "user_item_interaction": (
            id_states["user_id"] == "indexable" and id_states["asin"] == "indexable"
        ),
    }


def metadata_modality_flags(row):
    return (
        int(any(has_nonempty_content(row.get(field)) for field in PRODUCT_DESCRIPTION_FIELDS)),
        int(has_image_descriptor(row.get("images"))),
        int(not isinstance(row.get("price"), bool)
            and isinstance(row.get("price"), (int, float))
            and row["price"] >= 0
            and (isinstance(row["price"], int) or math.isfinite(row["price"]))),
    )


# The free-form details dictionary can contain many different keys.
# Cap the number of DISTINCT nested child names held in memory per field/context.
# Overflow is made explicit in the outputs, never hidden.
MAX_NESTED_CHILD_KEYS = 5000
# For the research question, 5 observed reviews define input and the next
# 10 define target. Ties across positions 5/6 or 15/16 may change membership.
INPUT_REVIEW_COUNT = 5
TARGET_REVIEW_COUNT = 10
# A fixed-width TEXT key preserves order even for valid nanosecond Unix
# timestamps that exceed SQLite's signed 64-bit INTEGER range.
TIMESTAMP_ORDER_OFFSET = 10 ** 21


def sortable_timestamp_key(raw_timestamp):
    """Preserve *exact raw epoch integer* equality/order in all supported units."""
    shifted = TIMESTAMP_ORDER_OFFSET + raw_timestamp
    if not (0 <= shifted < 10 ** 22):
        raise ValueError("Convertible timestamp outside supported ordering-key range")
    return f"{shifted:022d}"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def type_name(value):
    # bool must precede int: Python considers bool a subclass of int.
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(type(value).__name__)


def indexable_id(value):
    """Keep identifiers verbatim. Do not coerce, trim, case-fold or repair."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return False
    return True


class NonStandardJSONConstant(ValueError):
    """A non-standard NaN/Infinity literal rejected during JSON decoding."""


def reject_constant(value):
    raise NonStandardJSONConstant("Non-standard JSON constant: " + value)


def identifier_state(record, field):
    """Classify every object row for one ID; do not modify original values."""
    if field not in record:
        return "absent_key"
    value = record[field]
    if value is None:
        return "explicit_null"
    if not isinstance(value, str):
        return "wrong_type"
    if not value.strip():
        return "blank_string"
    if not indexable_id(value):
        # Can occur for a valid JSON string containing an isolated surrogate.
        return "invalid_utf8"
    return "indexable"


def parse_error_reason(error):
    """Group parsing problems without storing potentially private review text."""
    if isinstance(error, UnicodeDecodeError):
        return "invalid_utf8"
    if isinstance(error, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(error, NonStandardJSONConstant):
        return "non_standard_json_constant"
    if isinstance(error, RecursionError):
        return "excessive_nesting"
    return "invalid_json_value"



def explicit_utc_datetime(value, setting_name):
    """Require an explicit timezone, normalize to UTC, and reject invalid settings."""
    if not isinstance(value, str):
        raise ValueError(f"{setting_name} must be an ISO 8601 date/time string with a timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid {setting_name}: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{setting_name} requires an explicit timezone (+00:00 or Z)")
    return parsed.astimezone(timezone.utc)


def epoch_to_utc(value, unit):
    """Convert a strict integer Unix timestamp to UTC without float rounding.

    Return (datetime, None) or (None, reason); never modify source values.
    Millisecond interpretation is configured, not inferred from digit count.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None, "wrong_raw_type"
    whole_seconds, remainder = divmod(value, TIMESTAMP_UNITS[unit])
    microseconds = remainder * 1_000_000 // TIMESTAMP_UNITS[unit]
    try:
        return (datetime(1970, 1, 1, tzinfo=timezone.utc)
                + timedelta(seconds=whole_seconds, microseconds=microseconds)), None
    except (OverflowError, ValueError):
        return None, "outside_datetime_range"

def connect_index(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-65536")  # approximately 64 MiB
    db.execute("PRAGMA temp_store=FILE")
    for table in TABLES:
        db.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID")
    db.execute("CREATE TABLE variant_links (asin TEXT, parent TEXT, n INTEGER NOT NULL, PRIMARY KEY(asin,parent)) WITHOUT ROWID")
    # Exact timestamp frequency per parent on disk, not 40m Python objects in RAM.
    db.execute("CREATE TABLE review_time_groups ("
               "parent TEXT NOT NULL, time_key TEXT NOT NULL, n INTEGER NOT NULL, "
               "PRIMARY KEY(parent,time_key)) WITHOUT ROWID")
    # Per-parent aggregated metadata flags; never materialize the raw join.
    db.execute("CREATE TABLE metadata_modality_index ("
               "parent TEXT PRIMARY KEY, n INTEGER NOT NULL, "
               "description_n INTEGER NOT NULL, image_n INTEGER NOT NULL, "
               "price_n INTEGER NOT NULL) WITHOUT ROWID")
    db.commit()
    return db


def flush(db, buffers, links, review_times=None, metadata_modalities=None):
    with db:
        for table, counter in buffers.items():
            db.executemany(
                f"INSERT INTO {table}(id,n) VALUES (?,?) "
                "ON CONFLICT(id) DO UPDATE SET n=n+excluded.n", counter.items()
            )
        db.executemany(
            "INSERT INTO variant_links(asin,parent,n) VALUES (?,?,?) "
            "ON CONFLICT(asin,parent) DO UPDATE SET n=n+excluded.n",
            ((a, p, n) for (a, p), n in links.items()),
        )
        if review_times is not None:
            db.executemany(
                "INSERT INTO review_time_groups(parent,time_key,n) VALUES (?,?,?) "
                "ON CONFLICT(parent,time_key) DO UPDATE SET n=n+excluded.n",
                ((parent, timestamp, n) for (parent, timestamp), n in review_times.items()),
            )
        if metadata_modalities is not None:
            db.executemany(
                "INSERT INTO metadata_modality_index"
                "(parent,n,description_n,image_n,price_n) VALUES (?,?,?,?,?) "
                "ON CONFLICT(parent) DO UPDATE SET "
                "n=n+excluded.n, description_n=description_n+excluded.description_n, "
                "image_n=image_n+excluded.image_n, price_n=price_n+excluded.price_n",
                ((parent, *values) for parent, values in metadata_modalities.items()),
            )
    for counter in buffers.values():
        counter.clear()
    links.clear()
    if review_times is not None:
        review_times.clear()
    if metadata_modalities is not None:
        metadata_modalities.clear()


def scan(path, role, db, batch_size, progress_every, preview_lines, timestamp_unit=DEFAULT_TIMESTAMP_UNIT,
         reference_start=DEFAULT_REFERENCE_START,
         reference_end_exclusive=DEFAULT_REFERENCE_END_EXCLUSIVE,
         as_of_utc=None):
    if timestamp_unit not in TIMESTAMP_UNITS:
        raise ValueError(f"Unsupported timestamp unit: {timestamp_unit!r}")
    reference_low = explicit_utc_datetime(reference_start, "reference_start")
    reference_high = explicit_utc_datetime(reference_end_exclusive, "reference_end_exclusive")
    if reference_low >= reference_high:
        raise ValueError("reference_start must precede reference_end_exclusive")
    if as_of_utc is None:
        as_of = datetime.now(timezone.utc)
    elif isinstance(as_of_utc, datetime):
        if as_of_utc.tzinfo is None or as_of_utc.utcoffset() is None:
            raise ValueError("as_of_utc must have a timezone")
        as_of = as_of_utc.astimezone(timezone.utc)
    else:
        as_of = explicit_utc_datetime(as_of_utc, "as_of_utc")
    stat_before = path.stat()
    result = {
        "path": str(path), "stored_bytes": stat_before.st_size,
        "compression": "gzip" if path.suffix.lower() == ".gz" else "none",
        "physical_lines_scanned": 0, "object_records": 0,
        "blank_lines": 0, "parse_errors": 0, "non_object_records": 0,
        "error_line_examples": [], "indexed_identifier_rows": {},
        "unindexed_identifier_rows": {},
    }
    fields = defaultdict(Counter)
    # Count present-but-empty values separately; absent keys are derived from
    # the number of parsed JSON objects minus present occurrences.
    empty_values = defaultdict(Counter)
    members = defaultdict(Counter)
    nested_stats = defaultdict(Counter)  # (top-level field, container type) -> counts
    nested_min_size = {}
    child_types = defaultdict(Counter)  # (top-level field, location, child key) -> raw types
    tracked_child_keys = Counter()
    untracked_child_occurrences = Counter()  # explicit when distinct-key cap is exceeded
    keys_per_record = Counter()  # Counts top-level keys on EACH valid JSON object.
    parse_reasons = Counter()  # Separate causes; never store the raw record content.
    time_states = Counter()  # Exactly one conversion state per parsed review object.
    time_plausibility = Counter()  # Flags among convertible times; not errors or filters.
    time_min = time_max = None  # UTC min/max calculated during the same scan.
    timed_review_rows_with_parent = 0
    whole_second_aligned_rows = 0
    time_group_buffer = Counter()
    metadata_modality_buffer = {}  # one compact entry per parent in current batch
    modality_counts = Counter()  # simple per-input-file coverage counters
    buffers = {table: Counter() for table in TABLES}
    links = Counter()
    indexed = Counter()
    unindexed = Counter()
    invalid_id_reasons = defaultdict(Counter)
    indexed_id_surrounding_whitespace = Counter()
    digest = hashlib.sha256()
    since_flush = 0
    paired_review_rows = 0  # Reviews with two usable IDs in the same record.
    started = time.monotonic()
    opener = gzip.open if result["compression"] == "gzip" else open
    with opener(path, "rb") as stream:
        lines = islice(stream, preview_lines) if preview_lines else stream
        for line_no, raw in enumerate(lines, 1):
            result["physical_lines_scanned"] += 1
            digest.update(raw)
            if raw.strip():
                try:
                    encoding = "utf-8-sig" if line_no == 1 else "utf-8"
                    row = json.loads(raw.decode(encoding), parse_constant=reject_constant)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
                    result["parse_errors"] += 1
                    reason = parse_error_reason(error)
                    parse_reasons[reason] += 1
                    if len(result["error_line_examples"]) < 10:
                        result["error_line_examples"].append({
                            "line": line_no, "reason": reason,
                            "error_type": type(error).__name__,
                        })
                else:
                    if not isinstance(row, dict):
                        result["non_object_records"] += 1
                    else:
                        result["object_records"] += 1
                        keys_per_record[len(row)] += 1
                        for field, value in row.items():
                            fields[field][type_name(value)] += 1
                            if isinstance(value, str) and not value.strip():
                                empty_values[field]["blank_string"] += 1
                            elif isinstance(value, list) and not value:
                                empty_values[field]["empty_array"] += 1
                            elif isinstance(value, dict) and not value:
                                empty_values[field]["empty_object"] += 1
                            if isinstance(value, (list, dict)):
                                container_kind = "array" if isinstance(value, list) else "object"
                                container_id = (field, container_kind)
                                stats = nested_stats[container_id]
                                size = len(value)
                                stats["container_rows"] += 1
                                stats["empty_rows"] += int(size == 0)
                                stats["total_items_or_keys"] += size
                                stats["max_size"] = max(stats["max_size"], size)
                                nested_min_size[container_id] = min(
                                    nested_min_size.get(container_id, size), size
                                )

                                if isinstance(value, list):
                                    for member in value:
                                        members[(field, "array_element")][type_name(member)] += 1
                                        if isinstance(member, dict):
                                            for child_key, child in member.items():
                                                location = "array_object"
                                                child_id = (field, location, child_key)
                                                group = (field, location)
                                                if (child_id in child_types or
                                                        tracked_child_keys[group] < MAX_NESTED_CHILD_KEYS):
                                                    if child_id not in child_types:
                                                        tracked_child_keys[group] += 1
                                                    child_types[child_id][type_name(child)] += 1
                                                else:
                                                    untracked_child_occurrences[group] += 1
                                else:
                                    for child_key, child in value.items():
                                        members[(field, "object_value")][type_name(child)] += 1
                                        location = "object"
                                        child_id = (field, location, child_key)
                                        group = (field, location)
                                        if (child_id in child_types or
                                                tracked_child_keys[group] < MAX_NESTED_CHILD_KEYS):
                                            if child_id not in child_types:
                                                tracked_child_keys[group] += 1
                                            child_types[child_id][type_name(child)] += 1
                                        else:
                                            untracked_child_occurrences[group] += 1
                        id_states = {}
                        for field, table in IDENTIFIER_FIELDS[role].items():
                            state = identifier_state(row, field)
                            id_states[field] = state
                            if state == "indexable":
                                value = row[field]
                                buffers[table][value] += 1
                                indexed[field] += 1
                                # Still usable as-is, but whitespace can prevent joins.
                                if value != value.strip():
                                    indexed_id_surrounding_whitespace[field] += 1
                            else:
                                invalid_id_reasons[field][state] += 1
                                unindexed[field] += 1
                        if role == "reviews":
                            for modality, available in review_modality_flags(row, id_states).items():
                                modality_counts[modality] += int(available)
                        else:
                            flags = metadata_modality_flags(row)
                            for modality, available in zip(META_COVERAGE_FIELDS, flags):
                                modality_counts[modality] += available
                            if id_states[PARENT_JOIN_FIELD] == "indexable":
                                parent = row[PARENT_JOIN_FIELD]
                                values = metadata_modality_buffer.setdefault(parent, [0, 0, 0, 0])
                                values[0] += 1
                                for index, flag in enumerate(flags, 1):
                                    values[index] += flag
                        if (role == "reviews" and id_states["asin"] == "indexable"
                                and id_states["parent_asin"] == "indexable"):
                            links[(row["asin"], row["parent_asin"])] += 1
                            paired_review_rows += 1
                        # S17: the documented review timestamp, not a metadata
                        # crawl time.  No silent aliasing of sort_timestamp.
                        if role == "reviews":
                            if REVIEW_TIMESTAMP_FIELD not in row:
                                time_states["absent_key"] += 1
                            elif row[REVIEW_TIMESTAMP_FIELD] is None:
                                time_states["explicit_null"] += 1
                            else:
                                instant, reason = epoch_to_utc(
                                    row[REVIEW_TIMESTAMP_FIELD], timestamp_unit)
                                if reason:
                                    time_states[reason] += 1
                                else:
                                    time_states["convertible"] += 1
                                    raw_timestamp = row[REVIEW_TIMESTAMP_FIELD]
                                    if raw_timestamp % TIMESTAMP_UNITS[timestamp_unit] == 0:
                                        whole_second_aligned_rows += 1
                                    if id_states[PARENT_JOIN_FIELD] == "indexable":
                                        # Ties are measured on the EXACT raw integer,
                                        # never a rounded Python datetime.
                                        time_key = sortable_timestamp_key(raw_timestamp)
                                        time_group_buffer[(row[PARENT_JOIN_FIELD], time_key)] += 1
                                        timed_review_rows_with_parent += 1
                                    time_min = instant if time_min is None else min(time_min, instant)
                                    time_max = instant if time_max is None else max(time_max, instant)
                                    if instant < reference_low:
                                        time_plausibility["before_reference_start"] += 1
                                        time_plausibility["outside_reference_window"] += 1
                                    elif instant >= reference_high:
                                        time_plausibility["at_or_after_reference_end"] += 1
                                        time_plausibility["outside_reference_window"] += 1
                                    else:
                                        time_plausibility["within_reference_window"] += 1
                                    # Independent flag: future relative to audit reference date.
                                    if instant > as_of:
                                        time_plausibility["after_as_of"] += 1
                        since_flush += 1
                        if since_flush >= batch_size:
                            flush(db, buffers, links, time_group_buffer, metadata_modality_buffer)
                            since_flush = 0
            else:
                result["blank_lines"] += 1
            if progress_every and line_no % progress_every == 0:
                elapsed = time.monotonic() - started
                print(f"[{role}] {line_no:,} lines, {elapsed / 60:.1f} minutes", flush=True)
        # A preview may happen to include the entire file. Check without
        # counting or processing the extra line; full scans already reach EOF.
        reached_eof = preview_lines == 0 or not bool(stream.readline())

    # Each physical line must belong to exactly one of these four groups.
    counted_lines = sum(result[key] for key in (
        "object_records", "non_object_records", "blank_lines", "parse_errors"
    ))
    if counted_lines != result["physical_lines_scanned"]:
        raise RuntimeError(
            f"Line accounting mismatch in {path}: "
            f"{counted_lines} classified vs {result['physical_lines_scanned']} read"
        )
    if sum(parse_reasons.values()) != result["parse_errors"]:
        raise RuntimeError(f"Parse-error accounting mismatch in {path}")
    result["parse_error_reasons"] = dict(sorted(parse_reasons.items()))
    flush(db, buffers, links, time_group_buffer, metadata_modality_buffer)
    stat_after = path.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
        raise RuntimeError("Input changed while being read: " + str(path))
    documented = DOCUMENTED_FIELDS[role]
    observed = set(fields)
    column_inventory = []
    for field in sorted(documented | observed):
        type_counts = fields.get(field, {})
        present = sum(type_counts.values())
        total = result["object_records"]
        explicit_null = type_counts.get("null", 0)
        blank_string = empty_values[field]["blank_string"]
        empty_array = empty_values[field]["empty_array"]
        empty_object = empty_values[field]["empty_object"]
        nonempty = present - explicit_null - blank_string - empty_array - empty_object
        if nonempty < 0:
            raise RuntimeError(f"Field-value accounting mismatch in {path}: {field}")
        spec = FIELD_SPECS[role].get(field)
        expected_types, logical_role, description = spec if spec else (
            (), "undocumented", "Observed extra field; meaning not verified by reference")
        # Exclude explicit null: null/presence are audited separately (S07).
        # An undocumented field has no reference type against which to compare.
        type_deviation = (
            sum(count for kind, count in type_counts.items()
                if kind != "null" and kind not in expected_types)
            if spec else None
        )
        column_inventory.append({
            "field": field,
            "documented_in_reference": field in documented,
            "observed_in_scanned_scope": field in observed,
            "present_rows": present,
            "absent_rows": total - present,
            "explicit_null_rows": explicit_null,
            "blank_string_rows": blank_string,
            "empty_array_rows": empty_array,
            "empty_object_rows": empty_object,
            "nonempty_value_rows": nonempty,
            "absent_or_null_percent": round((total - present + explicit_null) * 100 / total, 6) if total else None,
            "present_percent_of_object_records": round(present * 100 / total, 6) if total else None,
            "logical_role": logical_role,
            "meaning": description,
            "expected_nonnull_raw_types": list(expected_types),
            "observed_raw_type_counts": dict(sorted(type_counts.items())),
            "reference_type_deviation_nonnull_rows": type_deviation,
        })
    # Each valid JSON object contributes to exactly one key-count bucket.
    if sum(keys_per_record.values()) != result["object_records"]:
        raise RuntimeError(f"Key-count accounting mismatch in {path}")
    for item in column_inventory:
        if item["present_rows"] + item["absent_rows"] != result["object_records"]:
            raise RuntimeError(f"Column presence accounting mismatch in {path}: {item['field']}")
        if (item["explicit_null_rows"] + item["blank_string_rows"] +
            item["empty_array_rows"] + item["empty_object_rows"] +
            item["nonempty_value_rows"] != item["present_rows"]):
            raise RuntimeError(f"Field-value accounting mismatch in {path}: {item['field']}")
    # A parsed object has exactly one ID state for each applicable field.
    for field in IDENTIFIER_FIELDS[role]:
        bad = sum(invalid_id_reasons[field].values())
        good = indexed[field]
        if good + bad != result["object_records"] or bad != unindexed[field]:
            raise RuntimeError(f"Identifier-state accounting mismatch in {path}: {field}")
        if indexed_id_surrounding_whitespace[field] > good:
            raise RuntimeError(f"Identifier whitespace count exceeds indexed rows: {field}")
    if role == "reviews":
        if sum(time_states.values()) != result["object_records"]:
            raise RuntimeError(f"Temporal state accounting mismatch in {path}")
        convertible = time_states["convertible"]
        if (time_plausibility["within_reference_window"] +
                time_plausibility["outside_reference_window"] != convertible):
            raise RuntimeError(f"Temporal reference-window accounting mismatch in {path}")
        if (time_plausibility["before_reference_start"] +
                time_plausibility["at_or_after_reference_end"] !=
                time_plausibility["outside_reference_window"]):
            raise RuntimeError(f"Temporal out-of-window accounting mismatch in {path}")
        if time_plausibility["after_as_of"] > convertible:
            raise RuntimeError(f"Temporal future-time count exceeds convertible times in {path}")
        if not (0 <= timed_review_rows_with_parent <= convertible
                and 0 <= whole_second_aligned_rows <= convertible):
            raise RuntimeError(f"S20 temporal tie-index accounting mismatch in {path}")
        temporal = {
            "status": "audited_review_timestamp",
            "field": REVIEW_TIMESTAMP_FIELD,
            "meaning": "Publisher-documented time of a review; not purchase time",
            "unit": timestamp_unit,
            "unit_status": "configured_assumption_to_verify_on_local_data",
            "timezone": "UTC",
            "counts": {key: time_states[key] for key in (
                "absent_key", "explicit_null", "wrong_raw_type",
                "outside_datetime_range", "convertible")},
            "earliest_convertible_utc": time_min.isoformat() if time_min else None,
            "latest_convertible_utc": time_max.isoformat() if time_max else None,
            # S18: this is the span within the OBSERVED local records, not the
            # duration of the publishing or crawling process.  A single valid
            # timestamp yields zero; no valid timestamps yields null.
            "observed_span_days": (
                round((time_max - time_min).total_seconds() / 86400, 9)
                if time_min is not None else None
            ),
            "reference_full_corpus_period": "May 1996 through September 2023",
            "reference_period_scope": "publisher's entire Amazon Reviews'23 dataset, NOT this category",
            "reference_start_inclusive_utc": reference_low.isoformat(),
            "reference_end_exclusive_utc": reference_high.isoformat(),
            "audit_as_of_utc": as_of.isoformat(),
            "plausibility_flags": {name: time_plausibility[name] for name in (
                "within_reference_window", "outside_reference_window",
                "before_reference_start", "at_or_after_reference_end", "after_as_of")},
            "observed_range_scope": "only successfully parsed records in this scan",
            "convertible_rows_with_indexable_parent": timed_review_rows_with_parent,
            "whole_second_aligned_convertible_rows": whole_second_aligned_rows,
            "whole_second_alignment_note": "Divisibility alone does not prove timestamp measurement precision",
            "sort_timestamp_observed_rows": sum(fields.get("sort_timestamp", {}).values()),
            "note": "The publisher's schema documents timestamp, but its public sample uses "
                    "sort_timestamp. This audit does not silently treat them as synonyms. "
                    "The publisher's global date range is a plausibility reference, not a "
                    "row filter or a proof that an out-of-window record is wrong. "
                    "Future-time flags are relative to the recorded audit_as_of_utc.",
        }
    else:
        temporal = {
            "status": "not_applicable_no_documented_review_timestamp",
            "field": None, "unit": None,
            "note": "Product metadata is a crawl-time snapshot; no review timestamp is expected.",
        }
    if role == "reviews":
        modality_counts["review_time"] = time_states["convertible"]
    for modality in (REVIEW_COVERAGE_FIELDS if role == "reviews" else META_COVERAGE_FIELDS):
        if not 0 <= modality_counts[modality] <= result["object_records"]:
            raise RuntimeError(f"S22 invalid source coverage count: {modality}")
    result.update({
        "modality_available_counts": dict(modality_counts),
        "temporal": temporal,
        "indexed_identifier_rows": dict(indexed), "unindexed_identifier_rows": dict(unindexed),
        # Not identical to either standalone ID-availability count: BOTH
        # must occur in the SAME review for its observed variant-parent edge.
        "review_rows_with_usable_asin_parent_pair": (paired_review_rows if role == "reviews" else None),
        "unindexed_identifier_reasons": {
            field: {key: invalid_id_reasons[field][key] for key in (
                "absent_key", "explicit_null", "wrong_type", "blank_string", "invalid_utf8")}
            for field in IDENTIFIER_FIELDS[role]
        },
        "indexed_identifier_surrounding_whitespace_rows": {
            field: indexed_id_surrounding_whitespace[field] for field in IDENTIFIER_FIELDS[role]
        },
        "union_top_level_field_count": len(fields),
        "observed_field_names": sorted(observed),
        "documented_field_names": sorted(documented),
        "documented_fields_not_observed_in_scan": sorted(documented - observed),
        "additional_observed_field_names": sorted(observed - documented),
        "keys_per_record_distribution": dict(sorted(keys_per_record.items())),
        "column_inventory": column_inventory,
        "field_types": {key: dict(value) for key, value in sorted(fields.items())},
        "container_member_types": [
            {"field": key[0], "member_kind": key[1], "types": dict(value)}
            for key, value in sorted(members.items())
        ],
        "nested_containers": [
            {"field": field, "container_type": kind, **dict(counts),
             "min_size": nested_min_size[(field, kind)],
             "mean_size": counts["total_items_or_keys"] / counts["container_rows"]}
            for (field, kind), counts in sorted(nested_stats.items())
        ],
        "nested_child_types": [
            {"field": field, "location": location, "child_key": child_key,
             "raw_type_occurrences": dict(sorted(counts.items()))}
            for (field, location, child_key), counts in sorted(child_types.items())
        ],
        "nested_child_key_limit_per_field_and_location": MAX_NESTED_CHILD_KEYS,
        "nested_child_key_tracking_overflow": [
            {"field": field, "location": location, "untracked_occurrences": n,
             "note": "Only the first distinct child names up to the limit are itemized; container totals remain exact."}
            for (field, location), n in sorted(untracked_child_occurrences.items())
        ],
        "readable_json_values": result["object_records"] + result["non_object_records"],
        "excluded_from_object_records": (
            result["blank_lines"] + result["parse_errors"] + result["non_object_records"]
        ),
        "line_accounting_valid": True,
        "reached_eof": reached_eof,
        "has_structural_input_issues": any(result[key] for key in (
            "non_object_records", "blank_lines", "parse_errors"
        )),
        "scanned_content_sha256": digest.hexdigest(),
        "hash_scope": "Scanned decompressed bytes, including original line endings and BOM if present.",
        "scan_mode": "head_preview" if preview_lines else "entire_supplied_file",
        "elapsed_seconds": round(time.monotonic() - started, 3),
    })
    print(f"[{role}] finished: {result['object_records']:,} object records", flush=True)
    return result


def scalar(db, query):
    return db.execute(query).fetchone()[0]


def summarize_temporal_ties(db, review_file):
    """Exactly measure same-parent equal timestamps, including boundary ties.

    The on-disk index has one row for each (parent, exact timestamp). The
    cumulative-count window counts tied GROUPS, not arbitrary tie-break orders.
    """
    total_groups, indexed_rows, tied_groups, tied_rows, extra_tied_rows, tied_parents = db.execute("""
        SELECT COUNT(*), COALESCE(SUM(n),0),
               COALESCE(SUM(n>1),0),
               COALESCE(SUM(CASE WHEN n>1 THEN n ELSE 0 END),0),
               COALESCE(SUM(CASE WHEN n>1 THEN n-1 ELSE 0 END),0),
               COUNT(DISTINCT CASE WHEN n>1 THEN parent END)
        FROM review_time_groups
    """).fetchone()
    expected = review_file["temporal"]["convertible_rows_with_indexable_parent"]
    if indexed_rows != expected:
        raise RuntimeError(f"S20 timestamp index mismatch: SQLite={indexed_rows}, scan={expected}")
    boundary5 = INPUT_REVIEW_COUNT
    boundary15 = INPUT_REVIEW_COUNT + TARGET_REVIEW_COUNT
    tie_input, tie_target = db.execute("""
        WITH cumulative AS (
            SELECT n,
                   SUM(n) OVER (
                       PARTITION BY parent ORDER BY time_key
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS last_position
            FROM review_time_groups
        )
        SELECT COALESCE(SUM(last_position - n < ? AND last_position > ?),0),
               COALESCE(SUM(last_position - n < ? AND last_position > ?),0)
        FROM cumulative
    """, (boundary5, boundary5, boundary15, boundary15)).fetchone()
    parents_with_15 = db.execute("""
        SELECT COUNT(*) FROM (
            SELECT parent FROM review_time_groups GROUP BY parent
            HAVING SUM(n)>=?
        )
    """, (boundary15,)).fetchone()[0]
    if not (0 <= extra_tied_rows <= tied_rows <= indexed_rows and
            0 <= tied_groups <= total_groups and
            0 <= tie_input <= tied_parents and 0 <= tie_target <= tied_parents and
            0 <= tie_target <= parents_with_15):
        raise RuntimeError("S20 impossible timestamp-tie summary")
    return {
        "time_field": REVIEW_TIMESTAMP_FIELD,
        "grouping_field": PARENT_JOIN_FIELD,
        "ordering": "exact_raw_epoch_integer_within_each_parent",
        "timestamp_groups": total_groups,
        "review_rows_with_indexable_parent_and_convertible_timestamp": indexed_rows,
        "parents_with_duplicate_timestamps": tied_parents,
        "timestamp_groups_with_multiple_reviews": tied_groups,
        "review_rows_in_tied_timestamp_groups": tied_rows,
        "extra_timestamp_occurrences_after_first": extra_tied_rows,
        "parents_with_at_least_15_timed_reviews_not_final_eligibility": parents_with_15,
        "parents_with_tie_across_positions_5_and_6": tie_input,
        "parents_with_tie_across_positions_15_and_16": tie_target,
        "all_convertible_review_times_aligned_to_whole_seconds": (
            review_file["temporal"]["whole_second_aligned_convertible_rows"]
            == review_file["temporal"]["counts"]["convertible"]),
        "whole_second_aligned_convertible_rows":
            review_file["temporal"]["whole_second_aligned_convertible_rows"],
        "total_convertible_review_rows": review_file["temporal"]["counts"]["convertible"],
        "scan_scope": review_file["scan_mode"],
        "note": "Ties are based on identical raw timestamps and exact nonblank parent IDs. "
                "Only successfully parsed review objects with a convertible timestamp "
                "and usable parent are eligible for tie counting. These counts do not "
                "establish which reviews qualify for the modeling task; no sorting or "
                "tie-breaking of individual reviews is performed.",
    }


def summarize_index(db, files):
    # Fail clearly if a future refactor accidentally changes the key mapping.
    if (IDENTIFIER_FIELDS["reviews"].get(PARENT_JOIN_FIELD) != "review_products"
            or IDENTIFIER_FIELDS["metadata"].get(PARENT_JOIN_FIELD) != "metadata_products"):
        raise RuntimeError("Review/metadata index must use parent_asin on both sides")
    ids = {}
    for file_role, mapping in IDENTIFIER_FIELDS.items():
        info = files[file_role]
        for field, table in mapping.items():
            distinct, occurrences, repeated_values = db.execute(
                f"SELECT COUNT(*), COALESCE(SUM(n),0), "
                f"COALESCE(SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END),0) "
                f"FROM {table}"
            ).fetchone()
            # SQLite and the streaming scan must agree on the number of indexed IDs.
            # This catches lost/double-counted batches, not just bad SQL summaries.
            if occurrences != info["indexed_identifier_rows"].get(field, 0):
                raise RuntimeError(
                    f"Identifier-index mismatch for {file_role}.{field}: "
                    f"SQLite={occurrences}; scan={info['indexed_identifier_rows'].get(field, 0)}"
                )
            if not (0 <= repeated_values <= distinct <= occurrences):
                raise RuntimeError(f"Impossible identifier counts in {table}")
            reasons = info["unindexed_identifier_reasons"][field]
            invalid = sum(reasons.values())
            rows = info["object_records"]
            if occurrences + invalid != rows:
                raise RuntimeError(f"ID coverage mismatch for {file_role}.{field}")
            ids[table] = {
                "distinct_indexable_values": distinct,
                "indexed_row_occurrences": occurrences,
                "object_rows_denominator": rows,
                "rows_without_indexable_id": invalid,
                "absent_key_rows": reasons["absent_key"],
                "explicit_null_rows": reasons["explicit_null"],
                "wrong_type_rows": reasons["wrong_type"],
                "blank_string_rows": reasons["blank_string"],
                "invalid_utf8_rows": reasons["invalid_utf8"],
                "indexed_with_surrounding_whitespace_rows": (
                    info["indexed_identifier_surrounding_whitespace_rows"][field]
                ),
                "indexable_percent_of_object_rows": (
                    round(100 * occurrences / rows, 6) if rows else None
                ),
                # A value occurring 3 times counts ONCE here, not three times.
                "values_occurring_more_than_once": repeated_values,
                # The same value occurring 3 times contributes TWO here.
                "repeated_occurrences_after_first": occurrences - distinct,
            }
    # Counts are derived from observed (asin, parent_asin) pairs, not a
    # presumed universal catalog mapping. A parent having multiple ASINs is
    # expected; one ASIN mapped to multiple parents merits investigation.
    pair_count, pair_rows, asins_with_parent, parents_with_asin, repeated_pairs = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(n),0), COUNT(DISTINCT asin), "
        "COUNT(DISTINCT parent), "
        "COALESCE(SUM(CASE WHEN n>1 THEN 1 ELSE 0 END),0) "
        "FROM variant_links"
    ).fetchone()
    review_pair_rows = files["reviews"]["review_rows_with_usable_asin_parent_pair"]
    if pair_rows != review_pair_rows:
        raise RuntimeError(
            f"Variant-parent pair index mismatch: SQLite={pair_rows}, scan={review_pair_rows}"
        )
    if not (0 <= pair_count <= pair_rows <= min(
            ids["review_asins"]["indexed_row_occurrences"],
            ids["review_products"]["indexed_row_occurrences"])):
        raise RuntimeError("Impossible variant-parent pair counts")
    if asins_with_parent > ids["review_asins"]["distinct_indexable_values"]:
        raise RuntimeError("Mapped ASIN count exceeds indexable ASINs")
    if parents_with_asin > ids["review_products"]["distinct_indexable_values"]:
        raise RuntimeError("Mapped parent count exceeds indexable parents")
    relations = {
        # Explicitly record the source-field join, not only internal SQLite IDs.
        "review_join_field": PARENT_JOIN_FIELD,
        "metadata_join_field": PARENT_JOIN_FIELD,
        "join_comparison": "exact_string_equality_no_normalization",
        "shared_product_ids": scalar(db, "SELECT COUNT(*) FROM review_products r JOIN metadata_products m ON r.id=m.id"),
        "review_product_ids_without_metadata": scalar(db, "SELECT COUNT(*) FROM review_products r WHERE NOT EXISTS (SELECT 1 FROM metadata_products m WHERE m.id=r.id)"),
        "metadata_product_ids_without_reviews": scalar(db, "SELECT COUNT(*) FROM metadata_products m WHERE NOT EXISTS (SELECT 1 FROM review_products r WHERE r.id=m.id)"),
        "review_rows_with_matching_metadata_id": scalar(db, "SELECT COALESCE(SUM(r.n),0) FROM review_products r JOIN metadata_products m ON r.id=m.id"),
        "metadata_ids_with_multiple_rows": scalar(db, "SELECT COUNT(*) FROM metadata_products WHERE n>1"),
        "max_metadata_rows_per_product_id": scalar(db, "SELECT COALESCE(MAX(n),0) FROM metadata_products"),
        "distinct_asin_parent_pairs": pair_count,
        "review_rows_with_usable_asin_parent_pair": pair_rows,
        "review_rows_without_usable_asin_parent_pair": (
            files["reviews"]["object_records"] - pair_rows
        ),
        "distinct_asins_with_parent_link": asins_with_parent,
        "distinct_parents_with_asin_link": parents_with_asin,
        "asin_parent_pairs_occurring_more_than_once": repeated_pairs,
        "asins_linked_to_multiple_parents": scalar(db, "SELECT COUNT(*) FROM (SELECT asin FROM variant_links GROUP BY asin HAVING COUNT(*)>1)"),
        "parents_linked_to_multiple_asins": scalar(db, "SELECT COUNT(*) FROM (SELECT parent FROM variant_links GROUP BY parent HAVING COUNT(*)>1)"),
    }
    # Joined row count detects row multiplication, without materializing the join.
    relations["potential_inner_join_rows"] = scalar(db, "SELECT COALESCE(SUM(r.n*m.n),0) FROM review_products r JOIN metadata_products m ON r.id=m.id")
    denom = ids["review_products"]["indexed_row_occurrences"]
    relations["matched_share_of_reviews_with_indexable_product_id"] = (
        relations["review_rows_with_matching_metadata_id"] / denom if denom else None
    )
    # S14: metadata key repetition can multiply raw review rows in a join.
    # Report uniqueness and non-null / usable coverage SEPARATELY; a key can
    # be unique among observed valid IDs while some metadata records lack IDs.
    meta_id_stats = ids["metadata_products"]
    meta_row_count = files["metadata"]["object_records"]
    missing_meta_keys = meta_id_stats["rows_without_indexable_id"]
    relations["metadata_key_unique_among_indexable_ids"] = (
        relations["metadata_ids_with_multiple_rows"] == 0
    )
    relations["metadata_key_complete_in_scanned_scope"] = (
        meta_row_count > 0 and missing_meta_keys == 0
    )
    relations["metadata_parent_key_unique_and_complete_in_scanned_scope"] = (
        relations["metadata_key_unique_among_indexable_ids"]
        and relations["metadata_key_complete_in_scanned_scope"]
    )
    relations["metadata_rows_without_usable_parent_key"] = missing_meta_keys
    relations["metadata_extra_rows_after_first_per_parent"] = (
        meta_id_stats["indexed_row_occurrences"]
        - meta_id_stats["distinct_indexable_values"]
    )
    # SUM(r.n) counts review rows subject to >1 metadata match, NOT the
    # multiplied joined rows. No raw review/metadata join is materialized.
    relations["review_rows_matching_multiple_metadata_rows"] = scalar(
        db, "SELECT COALESCE(SUM(r.n),0) FROM review_products r "
            "JOIN metadata_products m ON r.id=m.id WHERE m.n>1"
    )
    matched = relations["review_rows_with_matching_metadata_id"]
    inner = relations["potential_inner_join_rows"]
    relations["extra_inner_join_rows_from_metadata_repetition"] = inner - matched
    # A SQL LEFT JOIN retains one row for every unmatched / non-indexable
    # review parent, while matched rows are multiplied by metadata frequency.
    relations["potential_left_join_rows"] = inner + (
        files["reviews"]["object_records"] - matched
    )
    # S15: coverage is a proportion of ORIGINAL reviewed objects or unique
    # parent identifiers, never of duplicated JOIN rows. Keep denominators
    # explicit: valid IDs and all JSON-object review rows are different sets.
    total_review_rows = files["reviews"]["object_records"]
    usable_review_rows = ids["review_products"]["indexed_row_occurrences"]
    unusable_review_rows = ids["review_products"]["rows_without_indexable_id"]
    distinct_review_parents = ids["review_products"]["distinct_indexable_values"]
    distinct_metadata_parents = ids["metadata_products"]["distinct_indexable_values"]
    matched_distinct_parents = relations["shared_product_ids"]
    valid_but_unmatched_rows = usable_review_rows - matched
    all_unmatched_rows = total_review_rows - matched
    if not (0 <= matched <= usable_review_rows <= total_review_rows
            and usable_review_rows + unusable_review_rows == total_review_rows
            and 0 <= matched_distinct_parents <= distinct_review_parents
            and matched_distinct_parents + relations["review_product_ids_without_metadata"]
                == distinct_review_parents
            and matched_distinct_parents + relations["metadata_product_ids_without_reviews"]
                == distinct_metadata_parents
            and valid_but_unmatched_rows + unusable_review_rows == all_unmatched_rows):
        raise RuntimeError("Review/metadata coverage accounting mismatch")

    def coverage_percent(numerator, denominator):
        return round(100 * numerator / denominator, 6) if denominator else None

    relations.update({
        "review_object_rows_denominator": total_review_rows,
        "review_rows_with_usable_parent_id": usable_review_rows,
        "review_rows_without_usable_parent_id": unusable_review_rows,
        "review_rows_with_usable_parent_but_no_metadata": valid_but_unmatched_rows,
        "review_rows_without_metadata_match_total": all_unmatched_rows,
        "review_distinct_usable_parent_ids_denominator": distinct_review_parents,
        "metadata_distinct_usable_parent_ids_denominator": distinct_metadata_parents,
        "matched_percent_of_all_review_object_rows": coverage_percent(
            matched, total_review_rows),
        "matched_percent_of_review_rows_with_usable_parent_id": coverage_percent(
            matched, usable_review_rows),
        "matched_percent_of_distinct_review_parent_ids": coverage_percent(
            matched_distinct_parents, distinct_review_parents),
        "both_input_files_reached_eof": (
            files["reviews"]["reached_eof"] and files["metadata"]["reached_eof"]
        ),
        "coverage_scope_note": (
            "Coverage applies only to parsed JSON-object rows in these two scanned inputs. "
            "Preview or independently sampled files cannot establish full-dataset coverage. "
            "Numerators count original review rows, before any join multiplication; "
            "unusable parent IDs are not matched or normalized."
        ),
    })
    if not (0 <= relations["review_rows_matching_multiple_metadata_rows"] <= matched <= inner):
        raise RuntimeError("Invalid review/metadata join multiplicities")
    if relations["potential_left_join_rows"] < files["reviews"]["object_records"]:
        raise RuntimeError("LEFT JOIN cannot have fewer rows than original reviews")
    relations["interpretation"] = (
        "All results refer to scanned objects only; preview mode does not establish "
        "full-file key uniqueness or completeness. Multiple metadata rows for "
        "one parent may multiply joined review rows, but do not prove full-record "
        "duplicates. Join row counts are projected without materializing a join."
    )
    return ids, relations


def summarize_modalities(files, relations):
    """S21: identify source fields, join conditions and evidence of observation.

    This is a field-level INVENTORY, not a per-review modality coverage rate.
    S22 will separately define valid content and count usable reviews.
    """
    result = []
    joined_reviews = relations["review_rows_with_matching_metadata_id"]
    for modality, role, field_names, meaning in MODALITY_MAP:
        info = files[role]
        observed = set(info["observed_field_names"])
        present_fields = [field for field in field_names if field in observed]
        absent_fields = [field for field in field_names if field not in observed]
        needs_join = role == "metadata"
        item = {
            "modality": modality,
            "source_file_role": role,
            "source_unit": "review_record" if role == "reviews" else "metadata_product_record",
            "source_fields": list(field_names),
            "observed_fields_in_scan": present_fields,
            "not_observed_in_scan": absent_fields,
            "source_field_status": (
                "all_fields_observed_in_scan" if not absent_fields else
                "some_fields_observed_in_scan" if present_fields else
                "no_fields_observed_in_scan"
            ),
            "source_object_rows": info["object_records"],
            "requires_review_metadata_join": needs_join,
            "join_field": PARENT_JOIN_FIELD if needs_join else None,
            "review_rows_with_any_metadata_match": joined_reviews if needs_join else None,
            "meaning_and_limitations": meaning,
            "availability_note": (
                "A source field being observed even once is NOT proof of availability "
                "for every sample. Metadata joins alone do not guarantee the "
                "specific product field is usable. S22 will count modality availability."
            ),
        }
        result.append(item)
    return result



def summarize_modality_coverage(db, files, relationships):
    """S22: content-level availability with explicit source and review denominators.

    Metadata record counts refer to metadata rows, whereas attached counts
    refer to ORIGINAL reviews, never multiplied joined output rows.
    """
    review_total = files["reviews"]["object_records"]
    metadata_total = files["metadata"]["object_records"]
    reviews = files["reviews"]["modality_available_counts"]
    metadata = files["metadata"]["modality_available_counts"]
    rows = []

    def add(modality, level, available, total, rule, unambiguous=None):
        if not 0 <= available <= total:
            raise RuntimeError(f"S22 impossible availability {modality}: {available}/{total}")
        if unambiguous is not None and not 0 <= unambiguous <= available:
            raise RuntimeError(f"S26 invalid unambiguous coverage for {modality}")
        rows.append({
            "modality": modality, "measurement_level": level,
            "available_records": available, "denominator_records": total,
            "unavailable_records": total - available,
            "available_percent": round(100 * available / total, 6) if total else None,
            "unambiguous_available_records": unambiguous,
            "unambiguous_available_percent": (
                round(100 * unambiguous / total, 6)
                if unambiguous is not None and total else None
            ),
            "operational_definition": rule,
            "scan_scope": files["reviews"]["scan_mode"],
        })

    review_rules = {
        "review_text": "nonblank review title OR body",
        "review_rating": "finite numeric rating between 1 and 5 (not boolean)",
        "review_images": "at least one image descriptor with a nonblank URL reference",
        "review_time": "timestamp successfully converted using configured epoch unit",
        "user_item_interaction": "both user_id and asin are usable nonblank strings",
        "review_text_body": "nonblank review body only; supplementary diagnostic",
    }
    for modality in (*REVIEW_COVERAGE_FIELDS, "review_text_body"):
        add(modality, "review_record", reviews.get(modality, 0), review_total,
            review_rules[modality])
    metadata_rules = {
        "product_metadata": "at least one nonempty descriptive product field; excludes price/rating aggregates",
        "product_images": "at least one product-image descriptor with a nonblank URL reference",
        "product_price": "finite nonnegative numeric crawl-time product price (not boolean)",
    }
    for modality in META_COVERAGE_FIELDS:
        add(modality, "metadata_record", metadata.get(modality, 0), metadata_total,
            metadata_rules[modality])

    indexed_rows = scalar(db, "SELECT COALESCE(SUM(n),0) FROM metadata_modality_index")
    if indexed_rows != files["metadata"]["indexed_identifier_rows"].get(PARENT_JOIN_FIELD, 0):
        raise RuntimeError("S22 metadata coverage index does not match scanned metadata IDs")

    # Each metadata parent contributes at most ONCE to a review's flag,
    # even if the same parent has multiple metadata rows.
    # SQL CASE MAX over the parent means "any matched metadata row has content".
    for modality, column in (
        ("product_metadata", "description_n"),
        ("product_images", "image_n"),
        ("product_price", "price_n"),
    ):
        available, unambiguous = db.execute(
            "SELECT COALESCE(SUM(r.n),0), "
            "COALESCE(SUM(CASE WHEN m.n=1 THEN r.n ELSE 0 END),0) "
            "FROM review_products r JOIN metadata_modality_index m "
            "ON r.id=m.parent WHERE m." + column + ">0"
        ).fetchone()
        add(modality, "review_record_after_parent_join", available, review_total,
            "exact parent_asin match AND " + metadata_rules[modality] +
            "; available means AT LEAST ONE matching metadata row; "
            "unique-parent-match coverage excludes parents with repeated metadata rows",
            unambiguous=unambiguous)
        if available > relationships["review_rows_with_matching_metadata_id"]:
            raise RuntimeError("S22 modality count exceeds matched review count")
        if unambiguous > (relationships["review_rows_with_matching_metadata_id"] -
                          relationships["review_rows_matching_multiple_metadata_rows"]):
            raise RuntimeError("S26 unique metadata coverage exceeds unambiguous join rows")
    return rows


# S23: compare identical S22 operational definitions across file categories.
# The category is the selected FILE category, not metadata.main_category.
# Product prices/images are evaluated both within metadata and at review level
# after the same exact parent_asin matching rule as S22.
CATEGORY_COMPARISON_MEASURES = (
    ("review_text_body", "review_record"),
    ("review_text", "review_record"),
    ("review_images", "review_record"),
    ("product_images", "review_record_after_parent_join"),
    ("product_price", "review_record_after_parent_join"),
    ("product_images", "metadata_record"),
    ("product_price", "metadata_record"),
)


def summarize_category_comparison(coverage_by_category):
    """S23: exact count/percentage comparison without rereading raw records.

    Percentages with a zero denominator are undefined (None), NOT zero.
    Between-category spread is descriptive, NOT a test of missingness mechanism.
    """
    rows = []
    spreads = []
    if set(coverage_by_category) != set(PROJECT_CATEGORIES):
        raise ValueError("S23 requires results for all four project categories")

    for modality, level in CATEGORY_COMPARISON_MEASURES:
        valid = []
        for category in PROJECT_CATEGORIES:
            matches = [entry for entry in coverage_by_category[category]
                       if entry["modality"] == modality
                       and entry["measurement_level"] == level]
            if len(matches) != 1:
                raise RuntimeError(f"S23 expected exactly one {category}/{modality}/{level} entry")
            item = matches[0]
            available = item["available_records"]
            total = item["denominator_records"]
            missing = item["unavailable_records"]
            if not (0 <= available <= total and missing == total - available):
                raise RuntimeError(f"S23 inconsistent S22 counts: {category}/{modality}")
            observed_percent = round(100 * available / total, 6) if total else None
            if item["available_percent"] != observed_percent:
                raise RuntimeError(f"S23 inconsistent S22 percent: {category}/{modality}")
            missing_percent = round(100 * missing / total, 6) if total else None
            rows.append({
                "category": category, "modality": modality, "measurement_level": level,
                "available_records": available, "unavailable_records": missing,
                "denominator_records": total, "available_percent": observed_percent,
                "unavailable_percent": missing_percent,
                "operational_definition": item["operational_definition"],
            })
            if observed_percent is not None:
                valid.append((category, observed_percent))
        # A spread is descriptive; groups with zero rows are excluded and noted.
        smallest = min(valid, key=lambda x: x[1]) if valid else None
        largest = max(valid, key=lambda x: x[1]) if valid else None
        spreads.append({
            "modality": modality, "measurement_level": level,
            "categories_with_defined_percent": len(valid),
            "lowest_category": smallest[0] if smallest else None,
            "lowest_available_percent": smallest[1] if smallest else None,
            "highest_category": largest[0] if largest else None,
            "highest_available_percent": largest[1] if largest else None,
            "availability_range_percentage_points": (
                round(largest[1] - smallest[1], 6) if len(valid) >= 2 else None
            ),
        })
    return rows, spreads



def summarize_missingness_association(comparison_rows):
    """S24: DESCRIPTIVE association between input-file category and missingness.

    The rows are exact per-category counts from S22/S23. Cramer's V here
    summarizes an r-by-2 contingency table (category x available/missing).
    This does not test MCAR/MAR/MNAR, establish causation, or supply valid
    inferential p-values for possibly clustered/overlapping review records.
    """
    results = []
    for modality, level in CATEGORY_COMPARISON_MEASURES:
        group_rows = [r for r in comparison_rows
                      if r["modality"] == modality and r["measurement_level"] == level]
        if len(group_rows) != len(PROJECT_CATEGORIES):
            raise RuntimeError("S24 expected a row for each category and modality")
        if {r["category"] for r in group_rows} != set(PROJECT_CATEGORIES):
            raise RuntimeError("S24 category labels do not match the project configuration")
        nonempty = [r for r in group_rows if r["denominator_records"] > 0]
        total = sum(r["denominator_records"] for r in nonempty)
        available = sum(r["available_records"] for r in nonempty)
        missing = sum(r["unavailable_records"] for r in nonempty)
        if available + missing != total:
            raise RuntimeError("S24 category contingency counts are inconsistent")
        rates = [r["unavailable_records"] / r["denominator_records"] for r in nonempty]
        gap = 100 * (max(rates) - min(rates)) if rates else None
        v = None
        if len(nonempty) < 2:
            status = "fewer_than_two_nonempty_categories"
        elif available == 0 or missing == 0:
            status = "no_variation_in_availability_indicator"
        else:
            pooled_avail = available / total
            # Exact formula for Cramer's V in an r x 2 table, using counts.
            weighted_variance = sum(
                r["denominator_records"] *
                ((r["available_records"] / r["denominator_records"]) - pooled_avail) ** 2
                for r in nonempty
            )
            v_squared = weighted_variance / (total * pooled_avail * (1 - pooled_avail))
            v = round(math.sqrt(min(1.0, max(0.0, v_squared))), 6)
            status = ("different_observed_category_rates" if gap > 0
                      else "equal_observed_category_rates")
        results.append({
            "modality": modality, "measurement_level": level,
            "nonempty_categories": len(nonempty),
            "categories_with_zero_records": [r["category"] for r in group_rows
                                             if r["denominator_records"] == 0],
            "total_records_across_files": total,
            "total_available_records": available,
            "total_unavailable_records": missing,
            "pooled_unavailable_percent": round(100 * missing / total, 6) if total else None,
            "min_unavailable_percent": round(100 * min(rates), 6) if rates else None,
            "max_unavailable_percent": round(100 * max(rates), 6) if rates else None,
            "missingness_gap_percentage_points": round(gap, 6) if gap is not None else None,
            "cramers_v_descriptive": v,
            "interpretation_status": status,
        })
    return results



def export_category_comparison(folder, coverage_by_category, scope):
    rows, spreads = summarize_category_comparison(coverage_by_category)
    associations = summarize_missingness_association(rows)
    write_csv(folder / "category_modality_comparison.csv", (
        "category", "modality", "measurement_level", "available_records",
        "unavailable_records", "denominator_records", "available_percent",
        "unavailable_percent", "operational_definition"), (
        (r["category"], r["modality"], r["measurement_level"],
         r["available_records"], r["unavailable_records"],
         r["denominator_records"], r["available_percent"],
         r["unavailable_percent"], r["operational_definition"])
        for r in rows
    ))
    write_csv(folder / "category_modality_spread.csv", (
        "modality", "measurement_level", "categories_with_defined_percent",
        "lowest_category", "lowest_available_percent", "highest_category",
        "highest_available_percent", "availability_range_percentage_points"), (
        tuple(r[key] for key in (
            "modality", "measurement_level", "categories_with_defined_percent",
            "lowest_category", "lowest_available_percent", "highest_category",
            "highest_available_percent", "availability_range_percentage_points"))
        for r in spreads
    ))
    write_json(folder / "category_modality_comparison.json", {
        "scope": scope,
        "category_definition": "Category determined by input filenames/configuration, not metadata.main_category",
        "interpretation": (
            "Descriptive missingness comparison, not proof of randomness or a causal category effect; "
            "preview samples cannot establish category-level population rates."
        ),
        "coverage_rows": rows, "between_category_spread": spreads,
    })
    # S24: association from the already-counted four-category table; no new scan.
    columns = (
        "modality", "measurement_level", "nonempty_categories",
        "total_records_across_files", "total_available_records",
        "total_unavailable_records", "pooled_unavailable_percent",
        "min_unavailable_percent", "max_unavailable_percent",
        "missingness_gap_percentage_points", "cramers_v_descriptive",
        "interpretation_status", "categories_with_zero_records",
    )
    write_csv(folder / "missingness_category_association.csv", columns, (
        (json.dumps(r[key]) if isinstance(r[key], list) else r[key] for key in columns)
        for r in associations
    ))
    write_json(folder / "missingness_category_association.json", {
        "scope": scope,
        "category_variable": "input file category, not metadata.main_category",
        "method": "category by binary availability: descriptive rate gap and Cramer's V",
        "inferential_p_values_computed": False,
        "why_no_p_value": (
            "Reviews within a product/user may be dependent; categories can overlap; "
            "review collection is observational and not a random sample of purchases. "
            "Independent-row chi-square p-values would not be justified here."
        ),
        "missingness_conclusion": (
            "An observed association with category is evidence against a simple "
            "category-independent missingness description, not proof of an "
            "MCAR/MAR/MNAR mechanism. Equal rates do not prove MCAR. "
            "Metadata join availability is measured on review rows."
        ),
        "results": associations,
    })
    report_lines = [
        "# S24 — Association between category and information unavailability", "",
        "Data scope: " + scope,
        "Descriptive association only; no p-value and no claim of MCAR/MAR/MNAR or causality.",
        "Rows are drawn from the same S22/S23 counts; no extra raw-file pass.",
        "Cramer's V is between 0 and 1 when defined; it measures association, not causation.",
        "Non-independent reviews and possible overlap between category files limit inference.", "",
        "| Modality | Level | Missing rate gap (pp) | Cramer's V | Status |",
        "|---|---|---:|---:|---|",
    ]
    for r in associations:
        gap_value = (f"{r['missingness_gap_percentage_points']:.2f}"
                     if r["missingness_gap_percentage_points"] is not None else "N/A")
        v_value = (f"{r['cramers_v_descriptive']:.4f}"
                   if r["cramers_v_descriptive"] is not None else "N/A")
        report_lines.append(
            f"| {r['modality']} | {r['measurement_level']} | {gap_value} | "
            f"{v_value} | {r['interpretation_status']} |"
        )
    (folder / "missingness_category_association.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8")

    percent_lookup = {(r["category"], r["modality"], r["measurement_level"]): r
                      for r in rows}
    display_measures = CATEGORY_COMPARISON_MEASURES[:5]
    lines = [
        "# S23 — Comparison of information availability across Amazon categories",
        "",
        "Each cell shows percent available (available / readable JSON-object review rows).",
        "The category is the input-file label, not metadata.main_category.",
        "Image availability means a nonblank URL descriptor, not confirmed image access.",
        "Product image and price availability require a parent_asin metadata match.",
        "Price refers to metadata at crawl time, not the historical review-time price.",
        f"Scan scope: {scope}. Head previews are NOT representative estimates.",
        "Differences are descriptive only; S24 will examine association with category.", "",
        "| Category | " + " | ".join(k.replace("_", " ") for k, _ in display_measures) + " |",
        "|---|" + "---:|" * len(display_measures),
    ]
    for category in PROJECT_CATEGORIES:
        vals = []
        for modality, level in display_measures:
            v = percent_lookup[(category, modality, level)]
            vals.append((f"{v['available_percent']:.2f}%"
                         if v["available_percent"] is not None else "N/A (0 rows)"))
        lines.append("| " + category + " | " + " | ".join(vals) + " |")
    lines.extend(["", "See category_modality_comparison.csv for counts, missing percentages, "
                  "metadata-record comparisons and exact denominators.",
                  "See category_modality_spread.csv for the descriptive gap in percentage points.", ""])
    (folder / "category_modality_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return rows, spreads


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def export_report(folder, summary):
    write_json(folder / "summary.json", summary)
    schema = []
    containers = []
    nested_shapes = []
    nested_child_rows = []
    for role, info in summary["files"].items():
        field_descriptions = {item["field"]: item for item in info["column_inventory"]}
        for field, types in info["field_types"].items():
            reference = field_descriptions[field]
            for kind, count in sorted(types.items()):
                schema.append((
                    role, field, kind, count, sum(types.values()), info["object_records"],
                    reference["meaning"], reference["logical_role"],
                    "|".join(reference["expected_nonnull_raw_types"]),
                    reference["reference_type_deviation_nonnull_rows"],
                ))
        for item in info["container_member_types"]:
            for kind, count in sorted(item["types"].items()):
                containers.append((role, item["field"], item["member_kind"], kind, count))
        for item in info["nested_containers"]:
            nested_shapes.append((role, item["field"], item["container_type"],
                                  item["container_rows"], item["empty_rows"],
                                  item["total_items_or_keys"], item["min_size"],
                                  item["max_size"], item["mean_size"]))
        for item in info["nested_child_types"]:
            for kind, count in item["raw_type_occurrences"].items():
                nested_child_rows.append((role, item["field"], item["location"],
                                          item["child_key"], kind, count))
    write_csv(folder / "field_schema.csv", (
        "file_role", "field", "json_type", "type_occurrences", "field_present_rows",
        "object_rows", "meaning", "logical_role", "expected_nonnull_raw_types",
        "reference_type_deviation_nonnull_rows"), schema)
    inventory = [
        (role, entry["field"], entry["documented_in_reference"],
         entry["observed_in_scanned_scope"], entry["present_rows"], entry["absent_rows"],
         entry["explicit_null_rows"], entry["blank_string_rows"],
         entry["empty_array_rows"], entry["empty_object_rows"],
         entry["nonempty_value_rows"], entry["absent_or_null_percent"],
         entry["present_percent_of_object_records"], entry["logical_role"],
         entry["meaning"], "|".join(entry["expected_nonnull_raw_types"]),
         json.dumps(entry["observed_raw_type_counts"], sort_keys=True),
         entry["reference_type_deviation_nonnull_rows"])
        for role, info in summary["files"].items()
        for entry in info["column_inventory"]
    ]
    write_csv(folder / "column_inventory.csv", (
        "file_role", "field", "documented_in_reference", "observed_in_scanned_scope",
        "present_rows", "absent_rows", "explicit_null_rows", "blank_string_rows",
        "empty_array_rows", "empty_object_rows", "nonempty_value_rows",
        "absent_or_null_percent", "present_percent_of_object_records",
        "logical_role", "meaning", "expected_nonnull_raw_types",
        "observed_raw_type_counts", "reference_type_deviation_nonnull_rows"), inventory)
    write_csv(folder / "container_schema.csv", ("file_role", "field", "member_kind", "json_type", "occurrences"), containers)
    write_csv(folder / "nested_containers.csv", (
        "file_role", "field", "container_type", "container_rows", "empty_rows",
        "total_items_or_keys", "min_size", "max_size", "mean_size"), nested_shapes)
    write_csv(folder / "nested_child_types.csv", (
        "file_role", "field", "location", "child_key", "json_type", "occurrences"),
        nested_child_rows)
    id_columns = (
        "distinct_indexable_values", "indexed_row_occurrences",
        "values_occurring_more_than_once", "repeated_occurrences_after_first",
        "object_rows_denominator", "rows_without_indexable_id",
        "absent_key_rows", "explicit_null_rows", "wrong_type_rows",
        "blank_string_rows", "invalid_utf8_rows",
        "indexed_with_surrounding_whitespace_rows", "indexable_percent_of_object_rows",
    )
    write_csv(folder / "identifiers.csv",
        ("file_role", "source_field", "analytical_role", "meaning", "index_table") + id_columns,
        ((item["file_role"], item["source_field"], item["analytical_role"],
          item["meaning"], item["index_table"]) +
         tuple(summary["identifiers"][item["index_table"]][name] for name in id_columns)
         for item in summary["identifier_definitions"]))
    write_csv(folder / "modality_inventory.csv", (
        "modality", "source_file_role", "source_unit", "source_fields",
        "observed_fields_in_scan", "not_observed_in_scan", "source_field_status",
        "source_object_rows", "requires_review_metadata_join", "join_field",
        "review_rows_with_any_metadata_match", "meaning_and_limitations", "availability_note"), (
        (item["modality"], item["source_file_role"], item["source_unit"],
         "|".join(item["source_fields"]), "|".join(item["observed_fields_in_scan"]),
         "|".join(item["not_observed_in_scan"]), item["source_field_status"],
         item["source_object_rows"], item["requires_review_metadata_join"],
         item["join_field"], item["review_rows_with_any_metadata_match"],
         item["meaning_and_limitations"], item["availability_note"])
        for item in summary["modality_inventory"]
    ))
    write_csv(folder / "modality_coverage.csv", (
        "modality", "measurement_level", "available_records", "denominator_records",
        "unavailable_records", "available_percent", "operational_definition", "scan_scope",
        "unambiguous_available_records", "unambiguous_available_percent"), (
        (row["modality"], row["measurement_level"], row["available_records"],
         row["denominator_records"], row["unavailable_records"],
         row["available_percent"], row["operational_definition"], row["scan_scope"],
         row["unambiguous_available_records"], row["unambiguous_available_percent"])
        for row in summary["modality_coverage"]
    ))
    write_csv(folder / "relationships.csv", ("metric", "value"), summary["relationships"].items())
    write_csv(folder / "temporal_ties.csv", ("metric", "value"), summary["temporal_ordering"].items())
    write_csv(folder / "temporal_variables.csv", (
        "file_role", "time_field", "epoch_unit", "timezone", "unit_status",
        "convertible_rows", "absent_key_rows", "explicit_null_rows",
        "wrong_raw_type_rows", "outside_datetime_range_rows",
        "earliest_convertible_utc", "latest_convertible_utc", "observed_span_days",
        "reference_full_corpus_period", "reference_period_scope", "observed_range_scope",
        "reference_start_inclusive_utc", "reference_end_exclusive_utc", "audit_as_of_utc",
        "within_reference_window_rows", "outside_reference_window_rows",
        "before_reference_start_rows", "at_or_after_reference_end_rows", "after_as_of_rows",
        "sort_timestamp_observed_rows"), (
        (role, item["temporal"].get("field"), item["temporal"].get("unit"),
         item["temporal"].get("timezone"), item["temporal"].get("unit_status"),
         item["temporal"].get("counts", {}).get("convertible"),
         item["temporal"].get("counts", {}).get("absent_key"),
         item["temporal"].get("counts", {}).get("explicit_null"),
         item["temporal"].get("counts", {}).get("wrong_raw_type"),
         item["temporal"].get("counts", {}).get("outside_datetime_range"),
         item["temporal"].get("earliest_convertible_utc"),
         item["temporal"].get("latest_convertible_utc"),
         item["temporal"].get("observed_span_days"),
         item["temporal"].get("reference_full_corpus_period"),
         item["temporal"].get("reference_period_scope"),
         item["temporal"].get("observed_range_scope"),
         item["temporal"].get("reference_start_inclusive_utc"),
         item["temporal"].get("reference_end_exclusive_utc"),
         item["temporal"].get("audit_as_of_utc"),
         item["temporal"].get("plausibility_flags", {}).get("within_reference_window"),
         item["temporal"].get("plausibility_flags", {}).get("outside_reference_window"),
         item["temporal"].get("plausibility_flags", {}).get("before_reference_start"),
         item["temporal"].get("plausibility_flags", {}).get("at_or_after_reference_end"),
         item["temporal"].get("plausibility_flags", {}).get("after_as_of"),
         item["temporal"].get("sort_timestamp_observed_rows"))
        for role, item in summary["files"].items()
    ))
    lines = ["# Structural data audit", "", f"Run: {summary['run_id']}",
             f"Scope: {summary['scope']}", f"Category: {summary['category']}", "", "## Files", "",
             "| File | Lines read | JSON objects | Blank | Parse errors | Non-object JSON | EOF reached | Columns |",
             "|---|---:|---:|---:|---:|---:|---|---:|"]
    for role, info in summary["files"].items():
        lines.append(
            f"| {role} | {info['physical_lines_scanned']:,} | {info['object_records']:,} "
            f"| {info['blank_lines']:,} | {info['parse_errors']:,} "
            f"| {info['non_object_records']:,} | {info['reached_eof']} "
            f"| {info['union_top_level_field_count']} |"
        )
    lines += ["", "Line counts are internally checked: JSON objects + blank + parse errors + non-object JSON = lines read.",
              "A completed scan does not mean the input has no invalid or non-object lines.", ""]
    lines += ["## Parse issue reasons", "",
              "Only error categories and up to ten line-number examples per file are saved; raw review content is not included."]
    for role, info in summary["files"].items():
        lines.append(f"- {role}: {json.dumps(info['parse_error_reasons'], sort_keys=True)}")
        lines.append(f"  First examples: {json.dumps(info['error_line_examples'], sort_keys=True)}")
    lines.append("")
    lines += ["", "## Top-level column inventory", "",
              "Column count means the UNION of field names across readable JSON objects, not the number of keys in the first row.",
              "Reference names come from the dataset publisher. An extra field is not necessarily an error; a reference field not seen in a preview is not proven absent from the full file.", ""]
    for role, info in summary["files"].items():
        lines += [f"### {role}",
                  f"Observed field names: {', '.join(info['observed_field_names']) or '(none)'}",
                  f"Reference fields not observed in scanned scope: {', '.join(info['documented_fields_not_observed_in_scan']) or '(none)'}",
                  f"Additional observed field names: {', '.join(info['additional_observed_field_names']) or '(none)'}",
                  f"Keys per record distribution: {json.dumps(info['keys_per_record_distribution'], sort_keys=True)}",
                  f"Detailed per-field presence: column_inventory.csv (denominator: {info['object_records']} JSON objects)", ""]
    lines += ["", "## Field presence, null and empty values", "",
              "All counts use the parsed JSON object-row count as their denominator.",
              "Absent key != explicit null != blank/whitespace-only string != [] != {}.",
              "Blank strings are detected without changing the original data.",
              "Nonempty here means not one of those empty cases; it does NOT certify semantic usability or expected type.",
              "The detailed counts for each field are in column_inventory.csv.", ""]
    lines += ["", "## Field types and meanings", "",
              "Raw JSON value types are counted separately from analytical meaning.",
              "Reference expectations are role-specific (review title != product title).",
              "Non-null type deviations are flags for investigation, not automatic proof of bad data.",
              "Explicit null values are counted by raw type, but are not treated as type deviations.",
              "Reference types and all observed type counts: column_inventory.csv / field_schema.csv.", ""]
    for role, info in summary["files"].items():
        flagged = [f"{entry['field']}: {entry['reference_type_deviation_nonnull_rows']}"
                   for entry in info["column_inventory"]
                   if entry["reference_type_deviation_nonnull_rows"]]
        lines.append(f"{role} — non-null deviations from publisher reference: "
                     + (", ".join(flagged) if flagged else "none observed"))
    lines += ["", "## Nested field structure", "",
              "Top-level arrays and objects remain ONE field each; elements and keys are not extra top-level rows.",
              "nested_containers.csv: container frequency, empty count, total elements/keys, min/max/mean size.",
              "container_schema.csv: types of direct array members or object values.",
              "nested_child_types.csv: types of keys inside array-of-object elements and direct object keys.",
              "Nested child keys are examined two structural levels deep; deeper dictionaries are not recursively expanded.",
              f"For memory safety, at most {MAX_NESTED_CHILD_KEYS} distinct child names per field/context are itemized; any overflow is reported in summary.json.", ""]
    for role, info in summary["files"].items():
        overflow = info["nested_child_key_tracking_overflow"]
        lines.append(f"{role}: {len(info['nested_containers'])} array/object shapes; "
                     f"{len(info['nested_child_types'])} tracked child-key paths; "
                     f"overflow={json.dumps(overflow, ensure_ascii=True)}")
    lines += ["", "## Identifier roles and counts", "",
              "Amazon review user_id identifies a reviewer, asin identifies an item/variant,",
              "and parent_asin groups product variants. Join review and metadata by parent_asin.",
              "These are dataset identifiers, not verified real-world identities or guaranteed unique review IDs.", ""]
    for item in summary["identifier_definitions"]:
        value = summary["identifiers"][item["index_table"]]
        lines.append(
            f"- {item['file_role']}.{item['source_field']} ({item['analytical_role']}): "
            f"{item['meaning']}; {value['distinct_indexable_values']:,} distinct IDs; "
            f"{value['indexed_row_occurrences']:,} occurrences."
        )
    lines += ["", "## Identifier usability (S11)", "",
              "Counts use parseable JSON-object rows, not physical lines, as denominators.",
              "Non-indexable ID reasons are exclusive: absent, null, wrong type, blank string, or invalid UTF-8.",
              "IDs with surrounding whitespace remain indexable but are flagged because joins are exact.",
              "No ID strings are trimmed, converted, or repaired. See identifiers.csv.", ""]
    for item in summary["identifier_definitions"]:
        v = summary["identifiers"][item["index_table"]]
        lines.append(
            f"- {item['file_role']}.{item['source_field']}: "
            f"indexable={v['indexed_row_occurrences']:,}/{v['object_rows_denominator']:,}, "
            f"unindexable={v['rows_without_indexable_id']:,} "
            f"(absent={v['absent_key_rows']:,}; null={v['explicit_null_rows']:,}; "
            f"wrong_type={v['wrong_type_rows']:,}; blank={v['blank_string_rows']:,}; "
            f"invalid_utf8={v['invalid_utf8_rows']:,}); "
            f"indexable_with_surrounding_whitespace={v['indexed_with_surrounding_whitespace_rows']:,}."
        )
    lines += ["", "## Variant ASIN to parent mapping (S12)", "",
              "Each usable pair is taken from the SAME review record; absent/non-string/blank IDs are excluded.",
              "A parent with multiple ASINs is normal. An ASIN linked to multiple parents is flagged, not automatically corrected.",
              "Repeated observations of the same pair are not distinct variant-parent relationships.",
              "Pair counts refer only to scanned files; no global product-catalog consistency is claimed.",
              "See relationships.csv for the counts, including excluded rows and coverage.", ""]
    for key in (
        "distinct_asin_parent_pairs", "review_rows_with_usable_asin_parent_pair",
        "review_rows_without_usable_asin_parent_pair", "distinct_asins_with_parent_link",
        "distinct_parents_with_asin_link", "asin_parent_pairs_occurring_more_than_once",
        "asins_linked_to_multiple_parents", "parents_linked_to_multiple_asins",
    ):
        lines.append(f"- {key}: {summary['relationships'][key]}")
    lines += ["", "## File relationship (S13)", "",
              f"Join key: reviews.{PARENT_JOIN_FIELD} = metadata.{PARENT_JOIN_FIELD}.",
              "Do NOT join reviews.asin to metadata.parent_asin; asin is a variant ID.",
              "A match compares original ID strings exactly, without trimming or casting.",
              "This audit counts possible matches without creating a new merged dataset.",
              "Each review also refers to a user_id and a reviewed asin/variant.",
              "Metadata key uniqueness is measured, not assumed. Missing or non-indexable identifiers are counted separately.", ""]
    for key, value in summary["relationships"].items():
        lines.append(f"- {key}: {value}")
    r = summary["relationships"]
    lines += ["", "## Metadata parent key and join multiplicity (S14)", "",
              "A reliable one-row-per-review join needs metadata.parent_asin "
              "to be present and unique in the scanned metadata records.",
              "Uniqueness among usable keys is not the same as a complete key; "
              "empty metadata cannot establish both.",
              f"Metadata key complete (scanned scope): {r['metadata_key_complete_in_scanned_scope']}",
              f"Metadata parent key unique and complete (scanned scope): "
              f"{r['metadata_parent_key_unique_and_complete_in_scanned_scope']}",
              f"Review rows with multiple metadata matches: "
              f"{r['review_rows_matching_multiple_metadata_rows']}",
              f"Additional inner-join rows from multiple metadata matches: "
              f"{r['extra_inner_join_rows_from_metadata_repetition']}",
              f"Projected left-join row count: {r['potential_left_join_rows']}",
              "Repeated parent IDs do NOT necessarily mean duplicate full metadata records.",
              "All projected counts assume exact original-string equality and no normalization.",
              "In preview mode these conclusions apply ONLY to the scanned prefixes.", ""]
    lines += ["", "## Review-to-metadata coverage (S15)", "",
              "The denominator of each percentage is explicit; these are original review rows, not rows after joining.",
              f"Review rows with metadata: {r['review_rows_with_matching_metadata_id']:,} / "
              f"{r['review_object_rows_denominator']:,} "
              f"({r['matched_percent_of_all_review_object_rows']}%).",
              f"Review rows with usable parent ID and matching metadata: "
              f"{r['review_rows_with_matching_metadata_id']:,} / "
              f"{r['review_rows_with_usable_parent_id']:,} "
              f"({r['matched_percent_of_review_rows_with_usable_parent_id']}%).",
              f"Distinct review parent IDs with metadata: "
              f"{r['shared_product_ids']:,} / "
              f"{r['review_distinct_usable_parent_ids_denominator']:,} "
              f"({r['matched_percent_of_distinct_review_parent_ids']}%).",
              f"Reviews with a usable parent but no metadata: "
              f"{r['review_rows_with_usable_parent_but_no_metadata']:,}.",
              f"Reviews without a usable parent: {r['review_rows_without_usable_parent_id']:,}.",
              f"All reviews without a metadata match: "
              f"{r['review_rows_without_metadata_match_total']:,}.",
              f"Metadata-only parent IDs: {r['metadata_product_ids_without_reviews']:,}.",
              "Undefined percentages have null values in the JSON/CSV outputs when their denominator is zero.",
              "A match counts one original review once, even if metadata contains multiple rows for that parent.",
              "Preview mode, sampled files, and malformed JSON limit what coverage means; "
              "do not generalize these values to every Amazon review.", ""]
    lines += ["", "## Observed review date range (S18)", "",
              "The publisher reports May 1996 through September 2023 for the ENTIRE corpus;",
              "this is not the guaranteed range of any selected category or local sample.",
              "All range statistics below use only convertible timestamps in scanned review objects.",
              "No timestamps are filtered or changed; date-range plausibility is examined in S19.", ""]
    for role, info in summary["files"].items():
        if role != "reviews":
            continue
        t = info["temporal"]
        count = t["counts"]["convertible"]
        lines.append(f"- Review times: {count:,} convertible records; "
                     f"earliest={t['earliest_convertible_utc']}; "
                     f"latest={t['latest_convertible_utc']}; "
                     f"span_days={t['observed_span_days']}.")
        lines.append(f"- Scope: {info['scan_mode']}; reached_eof={info['reached_eof']}. "
                     "These are observed review dates, not purchase dates or collection dates.")
        if not count:
            lines.append("- No convertible review timestamp was observed; the range is undefined.")
    lines.append("")
    lines += ["", "## Temporal plausibility and invalid timestamps (S19)", "",
              "Technical failures (missing, null, wrong type, out of Python datetime range) are counted separately.",
              "A convertible timestamp outside the published corpus-wide time window is only flagged for review.",
              "No local timestamp is filtered or corrected; flags do not prove the configured epoch unit.",
              "Within/outside the reference window partitions convertible timestamps; after_as_of may overlap either group.",
              "Reference bounds and the audit as-of time are recorded in temporal_variables.csv and summary.json.", ""]
    for role, info in summary["files"].items():
        if role == "reviews":
            t = info["temporal"]
            lines.append(f"- {role}: conversion_states={json.dumps(t['counts'], sort_keys=True)}, "
                         f"plausibility_flags={json.dumps(t['plausibility_flags'], sort_keys=True)}")
    ties = summary["temporal_ordering"]
    lines += ["", "## Same-product timestamp ties and order precision (S20)", "",
              "Only equal raw epoch values within the same usable parent_asin are ties.",
              "An equality across positions 5/6 or 15/16 leaves the review-set membership ambiguous until a deterministic tie-break rule is defined.",
              "Products with >=15 convertible timestamps are NOT yet proven to have 15 valid ratings/texts for the final prediction task.",
              "Whole-second alignment is reported but does not prove original timestamp precision.",
              "See temporal_ties.csv and temporal_ordering in summary.json.",
              f"- Parents with timestamp collisions: {ties['parents_with_duplicate_timestamps']:,}.",
              f"- Parents tied across positions 5/6: {ties['parents_with_tie_across_positions_5_and_6']:,}.",
              f"- Parents tied across positions 15/16: {ties['parents_with_tie_across_positions_15_and_16']:,}.", ""]
    lines += ["", "## Temporal variable (S17)", "",
              "Review timestamps are Unix epoch values interpreted with an explicitly configured unit; all displayed dates are UTC.",
              "The default is milliseconds, provisionally based on the publisher's published millisecond-scale example; the local timestamp unit needs confirmation.",
              "The published schema calls the field timestamp but a sample calls it sort_timestamp; neither is silently substituted for the other.",
              "The metadata file has no documented review timestamp; do not interpret product attributes as historical snapshots.",
              "See temporal_variables.csv and files.reviews.temporal in summary.json.", ""]
    for role, item in summary["files"].items():
        lines.append(f"- {role}: {json.dumps(item['temporal'], ensure_ascii=True, sort_keys=True)}")
    lines += ["", "## Identifier repetition", "",
              "Repeated identifier values do not by themselves mean duplicate review records.",
              "'values_occurring_more_than_once' counts distinct identifiers with frequency > 1; "
              "'repeated_occurrences_after_first' counts additional occurrences beyond the first.",
              "See identifiers.csv for all four identifier roles.", ""]
    lines += ["", "## Multimodal field and source inventory (S21)", "",
              "A review is the sample. Product metadata is a separate observation "
              "attached only through a parent_asin match.",
              "Observed source fields indicate presence somewhere in the scanned "
              "scope, NOT per-review nonempty content or downloadable image media.",
              "Review images and product images are different; product-page rating "
              "aggregates and prices are not assumed historically available.",
              "See modality_inventory.csv and modality_inventory in summary.json.", ""]
    for item in summary["modality_inventory"]:
        lines.append(f"- {item['modality']} ({item['source_file_role']}): "
                     f"{item['source_field_status']}; "
                     f"observed={','.join(item['observed_fields_in_scan']) or '(none)'}; "
                     f"requires_join={item['requires_review_metadata_join']}")
    lines += ["", "## Content availability by modality (S22)", "",
              "Each denominator and operational definition is explicit; an image URL is not proof of a downloadable image.",
              "Metadata values describe crawl-time pages, not necessarily historical review-time features.",
              "Coverage in review_record_after_parent_join counts original review rows only; duplicate metadata rows do not inflate it.",
              "For a repeated metadata parent, 'available' means SOME candidate row has content; "
              "unambiguous_available counts only reviews linked to exactly ONE metadata row.",
              "Nonempty product-description containers must have at least one nonblank/finite scalar "
              "at any nesting depth; this is structural availability, not validated usefulness.",
              "See modality_coverage.csv and modality_coverage in summary.json.", ""]
    for item in summary["modality_coverage"]:
        detail = (f"; unambiguous={item['unambiguous_available_records']} "
                  f"({item['unambiguous_available_percent']}%)"
                  if item['unambiguous_available_records'] is not None else "")
        lines.append(f"- {item['modality']} [{item['measurement_level']}]: "
                     f"{item['available_records']}/{item['denominator_records']} "
                     f"({item['available_percent']}%){detail}.")
    lines += ["", "## Interpretation and limits", "",
              "- This step inventories structure. It does not deduplicate, filter for 15 reviews, clean, build targets or split data.",
              "- A JSONL file is a sequence of objects, not a fixed-width table. Field count means the union of observed top-level keys.",
              "- A present null field is distinct from an absent field. A container is kept as a nested field, not flattened into invented rows.",
              "- Identifier counts include all parseable object records with indexable IDs, irrespective of rating/time validity. Eligible-product counts belong to later work.",
              "- ID indexing requires nonblank UTF-8-encodable strings; identifiers are not otherwise normalized.",
              "- Relationship coverage from two independent head samples must not be generalized to the full dataset.",
              "- Malformed, blank and non-object lines are structural diagnostics. Full quality quantification belongs to step 2.",
              "- Current metadata aggregates and current product attributes are not introduced as historical model inputs.",
              "- Runtime and SQLite versions, effective settings and content fingerprints are recorded in summary.json.", ""]
    (folder / "structure_report.md").write_text("\n".join(lines), encoding="utf-8")


def resolve(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def collect_dataset_paths(config, root):
    """Check all four input pairs before processing any raw data.

    Paths come from the config, not from a computer-specific hard-coded folder.
    One physical file may not accidentally be assigned to two datasets/roles.
    """
    datasets = config.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Config must contain 'datasets' with four category entries.")
    missing = set(PROJECT_CATEGORIES) - set(datasets)
    unexpected = set(datasets) - set(PROJECT_CATEGORIES)
    if missing or unexpected:
        raise ValueError(
            f"Dataset categories mismatch. Missing: {sorted(missing)}; "
            f"unexpected: {sorted(unexpected)}"
        )

    paths_by_category = {}
    used_paths = {}
    for category in PROJECT_CATEGORIES:
        entry = datasets[category]
        if not isinstance(entry, dict):
            raise ValueError(f"{category}: expected a mapping of file paths")
        paths = {}
        for role, setting in (("reviews", "reviews_path"),
                              ("metadata", "metadata_path")):
            value = entry.get(setting)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{category}/{role}: missing '{setting}'")
            path = resolve(root, value)
            if not (path.name.lower().endswith(".jsonl")
                    or path.name.lower().endswith(".jsonl.gz")):
                raise ValueError(f"{category}/{role}: expected a JSONL or JSONL.gz file: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"{category}/{role}: input file not found: {path}")
            if path in used_paths:
                raise ValueError(
                    f"File reused for {category}/{role} and {used_paths[path]}: {path}"
                )
            used_paths[path] = f"{category}/{role}"
            paths[role] = path
        paths_by_category[category] = paths
    return paths_by_category


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit the four Amazon Reviews'23 category pairs")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/structure_all_categories.json"
    )
    parser.add_argument(
        "--preview-lines", type=int, default=0,
        help="First N physical lines per input; zero scans entire supplied files."
    )
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    root = resolve(config_path.parent, config.get("project_root", ".."))
    batch_size = int(config.get("batch_size", 25000))
    progress_every = int(config.get("progress_every_lines", 100000))
    if batch_size <= 0 or progress_every < 0 or args.preview_lines < 0:
        parser.error("Batch size must be positive and line limits/intervals nonnegative.")
    timestamp_unit = config.get("timestamp_unit", DEFAULT_TIMESTAMP_UNIT)
    if timestamp_unit not in TIMESTAMP_UNITS:
        raise ValueError(f"timestamp_unit must be one of: {', '.join(TIMESTAMP_UNITS)}")
    reference_start = config.get("reference_start", DEFAULT_REFERENCE_START)
    reference_end_exclusive = config.get("reference_end_exclusive", DEFAULT_REFERENCE_END_EXCLUSIVE)
    reference_low = explicit_utc_datetime(reference_start, "reference_start")
    reference_high = explicit_utc_datetime(reference_end_exclusive, "reference_end_exclusive")
    if reference_low >= reference_high:
        raise ValueError("reference_start must precede reference_end_exclusive")
    # One reference instant for all eight files, rather than a moving 'now'.
    audit_as_of = (explicit_utc_datetime(config["as_of_utc"], "as_of_utc")
                   if "as_of_utc" in config else datetime.now(timezone.utc))
    input_role = config.get("input_role", "dataset")
    if input_role not in ("dataset", "sample"):
        raise ValueError("input_role must be dataset or sample.")

    # Preflight: validate every selected category, file and role before scanning.
    datasets = collect_dataset_paths(config, root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    scope = "head_preview" if args.preview_lines else (
        "sample_files" if input_role == "sample" else "entire_supplied_dataset_files"
    )
    run_folder = resolve(root, config.get("output_directory", "reports/phase1/structure")) / run_id
    run_index_dir = resolve(root, config.get("index_directory", "data/derived/structure")) / run_id
    run_folder.mkdir(parents=True, exist_ok=False)
    run_index_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_folder / "RUN_STATUS.json"
    status = {"status": "running", "run_id": run_id, "scope": scope,
              "started_utc": utc_now(), "audit_as_of_utc": audit_as_of.isoformat(),
              "categories": list(PROJECT_CATEGORIES)}
    write_json(status_path, status)
    print(f"Run: {run_id}\nScope: {scope}\nOutputs: {run_folder}", flush=True)
    summaries = {}
    input_issues_by_file = {}
    coverage_by_category = {}  # tiny S22 summaries, not raw review rows
    try:
        for category, paths in datasets.items():
            print(f"\n[{category}] starting: reviews={paths['reviews']} metadata={paths['metadata']}", flush=True)
            folder = run_folder / category
            index_dir = run_index_dir / category
            folder.mkdir(parents=True, exist_ok=False)
            index_dir.mkdir(parents=True, exist_ok=False)
            db_path = index_dir / "identifiers.sqlite"

            # Crucial: each category has its OWN SQLite database and counts.
            db = connect_index(db_path)
            try:
                files = {}
                for role, path in paths.items():
                    # If a file cannot be read (e.g., truncated gzip), retain
                    # which input was active in RUN_STATUS.json.
                    status["active_input"] = {
                        "category": category, "role": role, "path": str(path)
                    }
                    write_json(status_path, status)
                    files[role] = scan(path, role, db, batch_size,
                                       progress_every, args.preview_lines, timestamp_unit,
                                       reference_start, reference_end_exclusive, audit_as_of)
                    status.pop("active_input", None)
                    write_json(status_path, status)
                identifiers, relationships = summarize_index(db, files)
                temporal_ordering = summarize_temporal_ties(db, files["reviews"])
                modality_inventory = summarize_modalities(files, relationships)
                modality_coverage = summarize_modality_coverage(db, files, relationships)
            finally:
                db.close()

            summary = {
                "audit_version": VERSION, "run_id": run_id, "scope": scope,
                "category": category, "input_role": input_role,
                "started_utc": status["started_utc"], "completed_utc": utc_now(),
                "runtime": {"python": platform.python_version(),
                            "sqlite": sqlite3.sqlite_version, "platform": platform.platform()},
                "effective_settings": {"config_path": str(config_path),
                                       "batch_size": batch_size,
                                       "progress_every_lines": progress_every,
                                       "preview_lines": args.preview_lines,
                                       "timestamp_unit": timestamp_unit,
                                       "reference_start": reference_low.isoformat(),
                                       "reference_end_exclusive": reference_high.isoformat(),
                                       "audit_as_of_utc": audit_as_of.isoformat()},
                "files": files, "identifiers": identifiers,
                "identifier_definitions": [
                    {"file_role": role, "source_field": field,
                     "analytical_role": FIELD_SPECS[role][field][1],
                     "meaning": FIELD_SPECS[role][field][2], "index_table": table}
                    for role, mapping in IDENTIFIER_FIELDS.items()
                    for field, table in mapping.items()
                ],
                "relationships": relationships,
                "temporal_ordering": temporal_ordering,
                "modality_inventory": modality_inventory,
                "modality_coverage": modality_coverage,
                "index_path": str(db_path), "output_path": str(folder),
                "source_provenance": config.get("source_provenance", {}),
            }
            export_report(folder, summary)
            coverage_by_category[category] = modality_coverage
            for role, info in files.items():
                if info["has_structural_input_issues"]:
                    input_issues_by_file[f"{category}/{role}"] = {
                        key: info[key] for key in
                        ("blank_lines", "parse_errors", "non_object_records",
                         "parse_error_reasons")
                    }
            summaries[category] = str(folder / "summary.json")
            print(f"[{category}] completed: {summaries[category]}", flush=True)

        # S23/S24: compare coverage and describe its association with category.
        # Do not load the raw files a second time.
        export_category_comparison(run_folder, coverage_by_category, scope)

        # A small run-wide index makes it impossible to overlook a category.
        write_json(run_folder / "datasets_manifest.json", {
            "run_id": run_id, "scope": scope,
            "categories": list(PROJECT_CATEGORIES),
            "summary_paths": summaries,
            "category_comparison": str(run_folder / "category_modality_comparison.csv"),
            "category_comparison_json": str(run_folder / "category_modality_comparison.json"),
            "category_comparison_readable": str(run_folder / "category_modality_comparison.md"),
            "missingness_association": str(run_folder / "missingness_category_association.csv"),
            "missingness_association_json": str(run_folder / "missingness_category_association.json"),
            "missingness_association_readable": str(run_folder / "missingness_category_association.md"),
            "input_paths": {
                category: {role: str(path) for role, path in paths.items()}
                for category, paths in datasets.items()
            },
        })
        # A completed run can still reveal malformed input. Keep these distinct.
        status.update(status="completed", completed_utc=utc_now(),
                      data_issues_detected=bool(input_issues_by_file),
                      data_issues_by_file=input_issues_by_file)
        write_json(status_path, status)
        print(
            f"Completed all four datasets; input issues detected: {bool(input_issues_by_file)}. "
            f"Manifest: {run_folder / 'datasets_manifest.json'}", flush=True
        )
        return 0
    except BaseException as error:
        if "active_input" in status:
            status["failed_input"] = status.pop("active_input")
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error), ended_utc=utc_now())
        write_json(status_path, status)
        print(f"Run did not complete; do not treat it as a successful four-category audit: {status_path}",
              file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
