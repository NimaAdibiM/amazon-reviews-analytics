from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import platform
import re
import sqlite3
import sys
import time
from uuid import uuid4
from urllib.parse import urlsplit

import structure_audit as common

VERSION = "1.0.26-q01-through-q25-structure-integration"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
EXPECTED = {
    "reviews": {
        "rating": {"number", "integer"}, "title": {"string"},
        "text": {"string"}, "images": {"array"}, "asin": {"string"},
        "parent_asin": {"string"}, "user_id": {"string"},
        "timestamp": {"integer"}, "helpful_vote": {"integer"},
        "verified_purchase": {"boolean"},
    },
    "metadata": {
        "main_category": {"string"}, "title": {"string"},
        "average_rating": {"number", "integer"}, "rating_number": {"integer"},
        "features": {"array"}, "description": {"array"},
        "price": {"number", "integer"}, "images": {"array"},
        "videos": {"array"}, "store": {"string"}, "categories": {"array"},
        "details": {"object"}, "parent_asin": {"string"}, "bought_together": {"array"},
    },
}
IMAGE_KEYS = {
    "reviews": ("small_image_url", "medium_image_url", "large_image_url"),
    "metadata": ("hi_res", "thumb", "large"),
}
MISSING_STATES = ("absent", "null", "blank_string", "empty_array", "empty_object", "present_nonempty")

# Q01: Raw missing states must remain independent of whether nested content is usable.
# These definitions are SCREENING RULES for known Amazon fields, not an assertion
# that an image exists online or that text is semantically meaningful.
NESTED_CONTENT_FIELDS = {
    "reviews": {"images": "review_image_http_url"},
    "metadata": {
        "features": "nonblank_string_element",
        "description": "nonblank_string_element",
        "categories": "nonblank_string_element",
        "bought_together": "nonblank_string_element",
        "images": "product_image_http_url",
        "videos": "product_video_http_url",
        "details": "nonblank_detail_key_with_nonempty_value",
    },
}



class DuplicateJSONKey(ValueError):
    pass


class NonFiniteJSON(ValueError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKey(key)
        result[key] = value
    return result


def finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise NonFiniteJSON(text)
    return value


def no_constant(text):
    raise NonFiniteJSON(text)


def parse_line(raw, first_line=False):
    return json.loads(raw.decode("utf-8-sig" if first_line else "utf-8"),
                      object_pairs_hook=unique_object, parse_float=finite_float,
                      parse_constant=no_constant)


def canonical(row):
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def missing_state(value):
    if value is None:
        return "null"
    if isinstance(value, str) and not value.strip():
        return "blank_string"
    if isinstance(value, list) and not value:
        return "empty_array"
    if isinstance(value, dict) and not value:
        return "empty_object"
    return "present_nonempty"


def usable_http_url(value):
    """Syntax-only HTTP(S) URL screening, without any network request."""
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
        return False
    try:
        value.encode("utf-8", "strict")
        url = urlsplit(value)
        return url.scheme.lower() in ("http", "https") and bool(url.hostname)
    except (ValueError, UnicodeError):
        return False


def has_nonempty_detail_value(value):
    """A nonempty scalar or nested value, not proof that it is domain-valid.

    JSON booleans and numeric zero are legitimate scalar content.
    Iterative traversal avoids recursion limits on heterogeneous metadata.
    """
    pending = [value]
    while pending:
        item = pending.pop()
        if item is None:
            continue
        if isinstance(item, str):
            if item.strip():
                return True
        elif isinstance(item, dict):
            pending.extend(v for k, v in item.items() if isinstance(k, str) and k.strip())
        elif isinstance(item, list):
            pending.extend(item)
        else:
            # Accept scalars 0 and False as present content. Their *validity*
            # belongs to a separate audit rule, not Q01 missingness.
            return True
    return False


def nested_has_usable_content(value, rule):
    """Field-specific content check, separate from raw missing-state counts."""
    if rule == "nonblank_string_element":
        return isinstance(value, list) and any(nonblank(item) for item in value)
    if rule in ("review_image_http_url", "product_image_http_url", "product_video_http_url"):
        if not isinstance(value, list):
            return False
        keys = ("small_image_url", "medium_image_url", "large_image_url") if rule == "review_image_http_url" else ("hi_res", "thumb", "large") if rule == "product_image_http_url" else ("url",)
        return any(isinstance(item, dict) and any(usable_http_url(item.get(key)) for key in keys)
                   for item in value)
    if rule == "nonblank_detail_key_with_nonempty_value":
        return isinstance(value, dict) and any(
            isinstance(key, str) and key.strip() and has_nonempty_detail_value(item)
            for key, item in value.items()
        )
    raise ValueError("Unknown Q01 nested content rule: " + rule)


def q02_field_missingness(states, object_records):
    """Summarize six exclusive Q01 states with a fixed, explicit denominator.

    A missing field or explicit JSON null is *raw missing*. Blank strings and
    empty containers are also reported in the broader *missing-or-empty*
    measure. Neither classification asserts semantic usability or validity.
    Duplicate records remain in the census because this is a RAW file audit.
    """
    if not isinstance(object_records, int) or isinstance(object_records, bool) or object_records < 0:
        raise ValueError("Q02 denominator must be a nonnegative integer")
    counts = {state: states.get(state, 0) for state in MISSING_STATES}
    if (any(not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts.values()) or sum(counts.values()) != object_records):
        raise AssertionError("Q02 field missingness count/denominator mismatch")
    absent, null, blank, arrays, objects, nonempty = (counts[k] for k in MISSING_STATES)
    missing = absent + null
    empty_present = blank + arrays + objects
    missing_or_empty = missing + empty_present

    def share(count):
        return count / object_records if object_records else None

    out = {
        "object_records_denominator": object_records,
        "present_rows": object_records - absent,
        "present_nonnull_rows": object_records - absent - null,
        "raw_missing_rows": missing,
        "empty_present_rows": empty_present,
        "raw_missing_or_empty_rows": missing_or_empty,
        "raw_present_nonempty_rows": nonempty,
        "raw_missing_share": share(missing),
        "raw_missing_percent": 100 * share(missing) if object_records else None,
        "raw_missing_or_empty_share": share(missing_or_empty),
        "raw_missing_or_empty_percent": 100 * share(missing_or_empty) if object_records else None,
        "raw_present_nonempty_share": share(nonempty),
        "raw_present_nonempty_percent": 100 * share(nonempty) if object_records else None,
    }
    for state, count in counts.items():
        out[state + "_rows"] = count
        out[state + "_percent"] = 100 * share(count) if object_records else None
    if missing_or_empty + nonempty != object_records:
        raise AssertionError("Q02 presence/missingness conservation failed")
    return out



# Q07: Strict raw JSON types for review fields only. The existing structural
# type vocabulary is preserved (integer, number, boolean, string, ...).
# This is a syntactic diagnostic, NEVER an implicit conversion.
Q07_NUMERIC_FIELDS = frozenset(('rating', 'timestamp', 'helpful_vote'))
Q07_NUMERIC_LIKE = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z', re.ASCII)
Q07_INTEGER_LIKE = re.compile(r'[+-]?[0-9]+\Z', re.ASCII)


def q07_numeric_string_candidate(field, value):
    """Screen a finite-length ASCII numeric spelling, without parsing/coercing.

    A candidate is still a WRONG-TYPE string. Numeric range/semantic validity
    is a separate check (Q11/Q12/Q13). Very long strings are not inspected.
    """
    if field not in Q07_NUMERIC_FIELDS or not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate or len(candidate) > 128:
        return False
    pattern = Q07_NUMERIC_LIKE if field == 'rating' else Q07_INTEGER_LIKE
    return pattern.fullmatch(candidate) is not None


def q07_review_type_summary(types, missing, wrong_counts, numeric_candidates, examples, n):
    """One immutable diagnostic row per expected/observed review field.

    Invariants: observed types cover all present objects, null is missing and
    never wrong type, and the old A wrong-type rules agree with the new report.
    Counts are FIELD OCCURRENCES, not distinct bad reviews across fields.
    """
    rows = []
    for field in sorted(set(types) | set(EXPECTED['reviews'])):
        observed = types.get(field, {})
        statuses = missing[field]
        absent = statuses['absent']
        explicit_null = statuses['null']
        present_nonnull = n - absent - explicit_null
        if (sum(observed.values()) != n-absent or
            observed.get('null', 0) != explicit_null or present_nonnull < 0):
            raise AssertionError('Q07 observed-type census mismatch: '+field)
        expected = EXPECTED['reviews'].get(field)
        incompatible = ({kind: count for kind, count in observed.items()
                         if kind != 'null' and kind not in expected}
                        if expected is not None else {})
        wrong = sum(incompatible.values())
        compatible = present_nonnull - wrong
        if compatible < 0 or (expected is not None and
                              wrong != wrong_counts.get(field+'_wrong_type', 0)):
            raise AssertionError('Q07 wrong-type flag mismatch: '+field)
        if expected is None and wrong_counts.get(field+'_wrong_type', 0):
            raise AssertionError('Q07 unexpected wrong-type flag on unknown field: '+field)
        candidates = numeric_candidates.get(field, 0)
        if candidates < 0 or candidates > incompatible.get('string', 0):
            raise AssertionError('Q07 numeric-string candidate is not a mismatch: '+field)
        rows.append({
            'field': field,
            'documented_field': expected is not None,
            'expected_nonnull_types': sorted(expected) if expected is not None else [],
            'observed_types': dict(sorted(observed.items())),
            'object_records_denominator': n,
            'absent_rows': absent,
            'explicit_null_rows': explicit_null,
            'present_nonnull_rows': present_nonnull,
            'compatible_nonnull_rows': compatible if expected is not None else None,
            'incompatible_nonnull_rows': wrong if expected is not None else None,
            'incompatible_percent_of_present_nonnull': (100*wrong/present_nonnull if present_nonnull and expected is not None else None),
            'incompatible_percent_of_all_objects': (100*wrong/n if n and expected is not None else None),
            'incompatible_observed_types': dict(sorted(incompatible.items())),
            'numeric_like_string_candidates': candidates,
            'wrong_type_example_physical_lines': examples.get(field+'_wrong_type', []),
            'assessment': ('not_in_documented_schema_profile_only' if expected is None else
                           'field_never_observed' if absent == n else
                           'no_nonnull_values' if present_nonnull == 0 else
                           'incompatible_observations' if wrong else 'types_compatible'),
            'initial_decision': ('Profile only; extra field is not automatically invalid.' if expected is None else
                                 'Preserve raw values; quarantine mismatched field values for later use, '
                                 'check candidate conversions against both type and domain rules; do not coerce automatically.'),
        })
    expected_rows = [row for row in rows if row['documented_field']]
    return dict(file_role='reviews', unit='parsed review object / field occurrence',
                denominator_parsed_review_objects=n,
                expected_field_count=len(EXPECTED['reviews']),
                observed_extra_field_count=sum(not row['documented_field'] for row in rows),
                fields_with_incompatible_observations=sum(bool(row['incompatible_nonnull_rows']) for row in expected_rows),
                incompatible_field_occurrences=sum(row['incompatible_nonnull_rows'] for row in expected_rows),
                numeric_like_string_candidate_occurrences=sum(row['numeric_like_string_candidates'] for row in expected_rows),
                fields=rows,
                interpretation=('Null/absent are missingness, not type mismatch. Boolean is not integer. '
                                'Numeric-looking strings remain wrong-type strings. Fields can overlap on the same review, '
                                'so summed field occurrences are NOT distinct defective reviews. '
                                'Original JSON values and raw files are unchanged.'))


# Q08: Metadata type compatibility.  Top-level contracts are the documented
# Amazon columns above.  Nested contracts are diagnostic assumptions for the
# known list/media fields, NOT a schema claim for arbitrary details values.
Q08_NUMERIC_FIELDS = frozenset(('price', 'average_rating', 'rating_number'))
Q08_LIST_ITEM_TYPES = {
    'features': 'string', 'description': 'string',
    'categories': 'string', 'bought_together': 'string',
    'images': 'object', 'videos': 'object',
}
Q08_MEDIA_ATTRIBUTE_TYPES = {
    'images': ('hi_res', 'thumb', 'large', 'variant'),
    'videos': ('url', 'title', 'user_id'),
}


def q08_numeric_string_candidate(field, value):
    """Lexical hint only; this does not coerce or validate numbers."""
    if field not in Q08_NUMERIC_FIELDS or not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate or len(candidate) > 128:
        return False
    return (Q07_INTEGER_LIKE if field == 'rating_number' else Q07_NUMERIC_LIKE).fullmatch(candidate) is not None


class Q08MetadataNested:
    """Bounded-memory counters for immediate children of known nested fields.

    A JSON null *inside* a list or attribute is reported as a null child,
    not a wrong non-null type.  No keys/values or full records are retained.
    `details.*` profiles immediate heterogeneous values with no assumed type.
    """
    def __init__(self, example_limit):
        self.types = defaultdict(Counter)
        self.containers = Counter()
        self.parent_affected = Counter()
        self.example_lines = defaultdict(list)
        self.example_limit = example_limit
        self.paths = {}
        for field, expected in Q08_LIST_ITEM_TYPES.items():
            self.paths[field + '[]'] = ('list_element', field, expected)
        for field, keys in Q08_MEDIA_ATTRIBUTE_TYPES.items():
            for key in keys:
                self.paths[f'{field}[].{key}'] = ('recognized_media_attribute', field, 'string')
        self.paths['details.*'] = ('immediate_details_value', 'details', None)

    def add(self, row, line):
        bad_paths = set()
        for field, expected in Q08_LIST_ITEM_TYPES.items():
            items = row.get(field)
            if not isinstance(items, list):
                continue  # Q08 top-level mismatch already tracked independently.
            self.containers[field] += 1
            path = field + '[]'
            for item in items:
                kind = 'null' if item is None else ('boolean' if isinstance(item,bool) else
                       'integer' if isinstance(item,int) else 'number' if isinstance(item,float) else
                       'string' if isinstance(item,str) else 'array' if isinstance(item,list) else
                       'object' if isinstance(item,dict) else 'unsupported')
                self.types[path][kind] += 1
                if kind not in (expected, 'null'):
                    bad_paths.add(path)
                if field not in Q08_MEDIA_ATTRIBUTE_TYPES or not isinstance(item, dict):
                    continue
                for attribute in Q08_MEDIA_ATTRIBUTE_TYPES[field]:
                    if attribute not in item:
                        continue
                    child = item[attribute]
                    child_kind = ('null' if child is None else 'boolean' if isinstance(child,bool) else
                         'integer' if isinstance(child,int) else 'number' if isinstance(child,float) else
                         'string' if isinstance(child,str) else 'array' if isinstance(child,list) else
                         'object' if isinstance(child,dict) else 'unsupported')
                    childpath = f'{field}[].{attribute}'
                    self.types[childpath][child_kind] += 1
                    if child_kind not in ('null','string'):
                        bad_paths.add(childpath)
        details = row.get('details')
        if isinstance(details, dict):
            self.containers['details'] += 1
            for v in details.values():
                kind = ('null' if v is None else 'boolean' if isinstance(v,bool) else
                        'integer' if isinstance(v,int) else 'number' if isinstance(v,float) else
                        'string' if isinstance(v,str) else 'array' if isinstance(v,list) else 'object')
                self.types['details.*'][kind] += 1
        for path in bad_paths:
            self.parent_affected[path] += 1
            if len(self.example_lines[path]) < self.example_limit:
                self.example_lines[path].append(line)
        return bool(bad_paths)

    def result(self, n):
        rows = []
        for path, (unit, field, expected) in sorted(self.paths.items()):
            counts = dict(sorted(self.types[path].items()))
            observed = sum(counts.values())
            nulls = counts.get('null', 0)
            nonnull = observed-nulls
            wrong = (sum(v for kind,v in counts.items() if kind not in (expected,'null'))
                     if expected is not None else None)
            compatible = (nonnull-wrong) if expected is not None else None
            affected = self.parent_affected[path]
            containers = self.containers[field]
            if affected > containers or (wrong is not None and (wrong > nonnull or affected > wrong)):
                raise AssertionError('Q08 nested count consistency failed: '+path)
            rows.append(dict(path=path, parent_field=field, unit=unit,
                    expected_nonnull_type=expected, observed_type_counts=counts,
                    parsed_metadata_objects=n, valid_parent_containers=containers,
                    observed_child_occurrences=observed, null_child_occurrences=nulls,
                    present_nonnull_child_occurrences=nonnull,
                    compatible_nonnull_child_occurrences=compatible,
                    incompatible_nonnull_child_occurrences=wrong,
                    incompatible_percent_of_nonnull_child_occurrences=(100*wrong/nonnull if nonnull and wrong is not None else None),
                    affected_metadata_records=affected if expected is not None else None,
                    affected_percent_of_valid_parent_containers=(100*affected/containers if containers and expected is not None else None),
                    example_physical_lines=self.example_lines.get(path,[]),
                    assessment=('heterogeneous_values_profile_only' if expected is None else
                                'no_child_occurrences' if not observed else
                                'only_null_children' if not nonnull else
                                'mismatched_nonnull_children' if wrong else 'compatible_nonnull_children'),
                    initial_decision=('Profile only; details values have no single guaranteed type.' if expected is None else
                                      'Keep source unchanged; investigate malformed nonnull nested members. Null child is missing, not wrong type.'),
                    ))
        return dict(denominator_parsed_metadata_objects=n, paths=rows,
                    interpretation=('Null child entries are shown separately and are not wrong non-null types. '
                                    'Nested list elements and recognized media attributes are screened under an adopted '
                                    'field-specific contract; other media keys and arbitrary details values are not '
                                    'declared erroneous. The nested denominator is occurrence count, not product rows.'))


def q08_metadata_type_summary(types, missing, wrong_counts, candidates, examples, n):
    """Strict top-level metadata census, independently reconciled with A flags."""
    rows = []
    for field in sorted(set(types) | set(EXPECTED['metadata'])):
        observed = types.get(field, {})
        statuses = missing[field]
        absent, nulls = statuses['absent'], statuses['null']
        present_nonnull = n - absent - nulls
        if (sum(observed.values()) != n-absent or
            observed.get('null', 0) != nulls or present_nonnull < 0):
            raise AssertionError('Q08 metadata observed-type census mismatch: '+field)
        expected = EXPECTED['metadata'].get(field)
        incompatible = ({kind: cnt for kind,cnt in observed.items()
                         if kind != 'null' and kind not in expected}
                        if expected is not None else {})
        wrong = sum(incompatible.values())
        compatible = present_nonnull - wrong
        if expected is not None and (wrong != wrong_counts.get(field+'_wrong_type',0) or compatible < 0):
            raise AssertionError('Q08 metadata wrong-type flag mismatch: '+field)
        if expected is None and wrong_counts.get(field+'_wrong_type',0):
            raise AssertionError('Q08 undocumented-field wrong-type flag: '+field)
        numeric_candidates = candidates.get(field, 0)
        if numeric_candidates > incompatible.get('string',0):
            raise AssertionError('Q08 candidate is not a wrong-type string: '+field)
        rows.append(dict(field=field, documented_field=expected is not None,
                  expected_nonnull_types=sorted(expected) if expected is not None else [],
                  observed_types=dict(sorted(observed.items())), object_records_denominator=n,
                  absent_rows=absent, explicit_null_rows=nulls, present_nonnull_rows=present_nonnull,
                  compatible_nonnull_rows=compatible if expected is not None else None,
                  incompatible_nonnull_rows=wrong if expected is not None else None,
                  incompatible_percent_of_present_nonnull=(100*wrong/present_nonnull if present_nonnull and expected is not None else None),
                  incompatible_percent_of_all_objects=(100*wrong/n if n and expected is not None else None),
                  incompatible_observed_types=dict(sorted(incompatible.items())),
                  numeric_like_string_candidates=numeric_candidates,
                  wrong_type_example_physical_lines=examples.get(field+'_wrong_type',[]),
                  assessment=('not_in_documented_schema_profile_only' if expected is None else
                              'field_never_observed' if absent == n else
                              'no_nonnull_values' if present_nonnull == 0 else
                              'incompatible_observations' if wrong else 'types_compatible'),
                  initial_decision=('Profile only; extra fields are not automatically errors.' if expected is None else
                                    'Keep raw value unchanged. Inspect type-specific handling and verify domain separately; no implicit coercion.')))
    expected_rows = [r for r in rows if r['documented_field']]
    return dict(file_role='metadata', denominator_parsed_metadata_objects=n,
                expected_field_count=len(EXPECTED['metadata']),
                observed_extra_field_count=sum(not r['documented_field'] for r in rows),
                fields_with_incompatible_observations=sum(bool(r['incompatible_nonnull_rows']) for r in expected_rows),
                incompatible_field_occurrences=sum(r['incompatible_nonnull_rows'] for r in expected_rows),
                numeric_like_string_candidate_occurrences=sum(r['numeric_like_string_candidates'] for r in expected_rows),
                fields=rows, interpretation=('Field occurrences may overlap on the same metadata record. '
                  'Absent/null values are missingness, not type errors; False is not an integer. '
                  'An apparently numeric string is still string-typed and cannot be silently converted.'))

# Q09: Lexical format diagnostics; these do not normalize/repair source values.
# Free-form review bodies and names are NOT compared case-insensitively.
Q09_FIELDS = {
    'reviews': ('parent_asin', 'asin', 'user_id', 'title'),
    'metadata': ('parent_asin', 'main_category', 'store', 'title'),
}
Q09_ASIN_FIELDS = frozenset(('parent_asin', 'asin'))
Q09_SPACES = re.compile(r'\s{2,}', re.UNICODE)
Q09_ASIN_CONVENTION = re.compile(r'[A-Z0-9]{10}\Z', re.ASCII)
Q09_CATEGORY_MAX_CHARS = 256


def q09_normalized_category(text):
    """Candidate equivalence for category LABELS only, not a silent recoding."""
    return ' '.join(text.split()).casefold()


class Q09Formats:
    """Fixed-size counters, plus SQLite-backed exact category spelling groups.

    The raw category string is retained in the temporary SQLite index only if
    short and Unicode-encodable. The result reports bounded example spellings.
    No product IDs, review text or user IDs are stored in these groups.
    """
    def __init__(self, role, db, example_limit, batch_size):
        self.role, self.db = role, db
        self.counts = Counter()
        self.examples = defaultdict(list)
        self.buffer = Counter()
        self.buffer_lines = {}
        self.example_limit = example_limit
        self.batch_size = batch_size

    def _count(self, field, rule, line):
        if rule != "nonblank_strings_inspected":
            self._row_problem = True
        key = (field, rule)
        self.counts[key] += 1
        if len(self.examples[key]) < self.example_limit:
            self.examples[key].append(line)

    def _category(self, field, value, line):
        if not isinstance(value, str) or not value.strip():
            return
        if len(value) > Q09_CATEGORY_MAX_CHARS:
            self._count(field, 'category_label_too_long_for_variant_index', line)
            return
        try:
            value.encode('utf-8', 'strict')
        except UnicodeError:
            self._count(field, 'category_label_unicode_unsupported_for_variant_index', line)
            return
        key = (self.role, field, q09_normalized_category(value), value)
        self.buffer[key] += 1
        self.buffer_lines.setdefault(key, line)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def _inspect_text(self, field, value, line, category=False, asin=False, title=False):
        if not isinstance(value,str) or not value.strip():
            return
        self._count(field, 'nonblank_strings_inspected', line)
        if value != value.strip():
            self._count(field, 'surrounding_whitespace', line)
        if (category or title) and Q09_SPACES.search(value):
            self._count(field, 'multiple_consecutive_whitespace', line)
        if asin:
            if any('a' <= ch <= 'z' for ch in value):
                self._count(field, 'lowercase_ascii_in_asin', line)
            if Q09_ASIN_CONVENTION.fullmatch(value) is None:
                self._count(field, 'asin_convention_mismatch_screen_only', line)
        if category:
            self._category(field, value, line)

    def add(self, row, line):
        self._row_problem = False
        for field in Q09_FIELDS[self.role]:
            self._inspect_text(field, row.get(field), line,
                category=field=='main_category', asin=field in Q09_ASIN_FIELDS,
                title=field=='title')
        if self.role != 'metadata':
            return self._row_problem
        categories = row.get('categories')
        if isinstance(categories,list):
            for item in categories:
                self._inspect_text('categories[]',item,line,category=True)
        details = row.get('details')
        if isinstance(details,dict):
            normalized_keys = set()
            any_bad = False
            for key in details:
                if not isinstance(key,str):
                    continue
                if not key.strip():
                    self._count('details.keys', 'blank_key',line)
                if key != key.strip() or Q09_SPACES.search(key):
                    self._count('details.keys', 'whitespace_in_key',line)
                norm = q09_normalized_category(key)
                if norm in normalized_keys:
                    any_bad = True
                normalized_keys.add(norm)
            if any_bad:
                self._count('details.keys', 'normalized_key_collision_record',line)
        return self._row_problem

    def flush(self):
        if not self.buffer: return
        self.db.executemany("""INSERT INTO q09_category_forms VALUES (?,?,?,?,?,?)
            ON CONFLICT(role,field,normalized,raw) DO UPDATE SET
            n=n+excluded.n, first_line=MIN(first_line,excluded.first_line)""",
            ((role,field,norm,raw,n,self.buffer_lines[(role,field,norm,raw)])
             for (role,field,norm,raw),n in self.buffer.items()))
        self.buffer.clear(); self.buffer_lines.clear()

    def result(self, n):
        self.flush()
        rows=[]
        for field in (*Q09_FIELDS[self.role], *(['categories[]','details.keys'] if self.role=='metadata' else [])):
            for rule in ('nonblank_strings_inspected','surrounding_whitespace',
                         'multiple_consecutive_whitespace','lowercase_ascii_in_asin',
                         'asin_convention_mismatch_screen_only',
                         'category_label_too_long_for_variant_index',
                         'category_label_unicode_unsupported_for_variant_index',
                         'blank_key','whitespace_in_key','normalized_key_collision_record'):
                # Only emit relevant meaningful field/rule combinations.
                if rule in ('lowercase_ascii_in_asin','asin_convention_mismatch_screen_only') and field not in Q09_ASIN_FIELDS: continue
                if rule in ('multiple_consecutive_whitespace',) and field not in ('main_category','categories[]','title'): continue
                if rule in ('category_label_too_long_for_variant_index','category_label_unicode_unsupported_for_variant_index') and field not in ('main_category','categories[]'): continue
                if rule in ('blank_key','whitespace_in_key','normalized_key_collision_record') and field != 'details.keys': continue
                if field=='details.keys' and rule not in ('blank_key','whitespace_in_key','normalized_key_collision_record'):continue
                if field in ('parent_asin','asin','user_id','store') and rule not in ('nonblank_strings_inspected','surrounding_whitespace','lowercase_ascii_in_asin','asin_convention_mismatch_screen_only'):continue
                if field=='categories[]' and rule in ('lowercase_ascii_in_asin','asin_convention_mismatch_screen_only'):continue
                count=self.counts[(field,rule)]
                unit = 'category list elements' if field=='categories[]' else 'details occurrences' if field=='details.keys' else 'parsed object records'
                den = self.counts[(field,'nonblank_strings_inspected')] if field=='categories[]' else n
                # Details keys: a record may contain multiple whitespace/blank keys;
                # percentages are intentionally omitted for key occurrences.
                percent=(100*count/den if den and field!='details.keys' else None)
                rows.append(dict(field=field,rule=rule,count=count,denominator=den if field!='details.keys' else None,
                                 denominator_unit=unit,percent=percent,
                                 example_physical_lines=self.examples.get((field,rule),[])))
        variants=[]
        sql="""SELECT field,normalized,COUNT(*) spellings,SUM(n) occurrences
               FROM q09_category_forms WHERE role=? GROUP BY field,normalized
               HAVING COUNT(*)>1 ORDER BY field,normalized"""
        for field,norm,spellings,occurrences in self.db.execute(sql,(self.role,)):
            examples=self.db.execute("""SELECT raw,n,first_line FROM q09_category_forms
                         WHERE role=? AND field=? AND normalized=? ORDER BY n DESC,raw LIMIT ?""",
                         (self.role,field,norm,self.example_limit)).fetchall()
            variants.append(dict(field=field, normalized_label=norm, distinct_raw_spellings=spellings,
                                 observed_label_occurrences=occurrences,
                                 example_raw_spellings=[{'raw':raw,'count':n,'first_line':line} for raw,n,line in examples]))
        for group in variants:
            if group['distinct_raw_spellings']<2 or group['observed_label_occurrences']<group['distinct_raw_spellings']:
                raise AssertionError('Q09 category grouping invariant failed')
        return dict(denominator_parsed_objects=n,format_rules=rows,category_variant_groups=variants,
                    category_label_max_indexed_chars=Q09_CATEGORY_MAX_CHARS,
                    category_variant_index_exclusion='Too-long or invalid-Unicode labels excluded only from variant pairing, not from raw data.',
                    interpretation=('Format diagnostics are screening hints, never automatic normalization. '
                      'Case folding applies ONLY to category labels; raw product/user IDs remain exact. '
                      'Metadata main_category and categories[] are grouped separately. '
                      'Text body/description are free-form and deliberately exempt. '
                      'Timestamp is numeric epoch; time interpretation belongs to S17/Q13. '
                      'Numeric-looking strings remain reported by Q07/Q08; no unit or currency inferred.'))


# Q10: Within-record, evidence-bounded cross-field diagnostics.
# The professor's promotion/discount example is NOT an Amazon schema rule.
# These outcomes are descriptive/suspicious candidates, not ground-truth repairs.
class Q10CrossFieldConsistency:
    def __init__(self, role, example_limit):
        self.role = role
        self.example_limit = example_limit
        self.counts = Counter()
        self.examples = defaultdict(list)
        self.rules = (
            ('asin_vs_parent_asin', (
                'both_usable_equal', 'both_usable_different', 'not_assessable'))
            if role == 'reviews' else
            ('rating_number_vs_average_rating', (
                'zero_count_zero_average', 'zero_count_positive_average',
                'positive_count_zero_average', 'positive_count_positive_average',
                'not_assessable'))
        )
        self.relationship, self.outcomes = self.rules

    def add(self, row, source_line):
        if self.role == 'reviews':
            child = row.get('asin')
            parent = row.get('parent_asin')
            if not (common.indexable_id(child) and common.indexable_id(parent)):
                outcome = 'not_assessable'
            else:
                outcome = 'both_usable_equal' if child == parent else 'both_usable_different'
        else:
            count = row.get('rating_number')
            average = row.get('average_rating')
            # A zero average is retained as a possible source sentinel solely
            # to detect a cross-field inconsistency candidate. Domain checks
            # for each individual field belong to Q11-Q14, not to Q10.
            count_ok = integer(count) and count >= 0
            average_ok = number(average) and 0 <= average <= 5
            if not (count_ok and average_ok):
                outcome = 'not_assessable'
            else:
                outcome = ('zero_count_' if count == 0 else 'positive_count_') + (
                    'zero_average' if average == 0 else 'positive_average')
        self.counts[outcome] += 1
        if len(self.examples[outcome]) < self.example_limit:
            self.examples[outcome].append(source_line)
        return outcome in ('zero_count_positive_average','positive_count_zero_average')

    def result(self, parsed_objects):
        if sum(self.counts.values()) != parsed_objects:
            raise AssertionError('Q10 one cross-field outcome per parsed object expected')
        candidates = {'zero_count_positive_average', 'positive_count_zero_average'}
        rows=[]
        for outcome in self.outcomes:
            count=self.counts[outcome]
            rows.append(dict(relationship=self.relationship, outcome=outcome, count=count,
                denominator_parsed_objects=parsed_objects,
                percent_of_parsed_objects=100*count/parsed_objects if parsed_objects else None,
                example_physical_lines=self.examples.get(outcome, []),
                assessment=('suspicious_not_proven_invalid' if outcome in candidates else
                    'not_assessable_due_to_missing_type_or_domain' if outcome=='not_assessable' else
                    'descriptive_not_a_validation_failure'),
                initial_decision=('Inspect snapshot metadata and its source semantics; do not modify or infer from review subset.'
                    if outcome in candidates else 'Keep raw values; no automatic correction.')))
        if sum(x['count'] for x in rows) != parsed_objects:
            raise AssertionError('Q10 reported outcomes do not conserve objects')
        return dict(role=self.role, denominator_parsed_objects=parsed_objects,
            relationship=self.relationship, results=rows,
            suspicious_candidate_rows=sum(self.counts[k] for k in candidates),
            interpretation=('Exact raw equality of asin and parent_asin is descriptive; distinct valid IDs can designate variants. '
                'No relationship between verified_purchase, helpful_vote and other review fields is assumed.'
                if self.role == 'reviews' else
                'rating_number and average_rating are snapshot metadata, and the zero-average sentinel is uncertain. '
                'Zero/nonzero disagreement is only a candidate for investigation, not proof of error. '
                'Do not compare aggregate product rating_number with review rows in this download: coverage and crawl times differ. '
                'No assumption that main_category must appear in categories[].'))


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def nonblank(value):
    return isinstance(value, str) and bool(value.strip())


# Use the FINAL structure audit's epoch conversion contract. No digit-count
# heuristics and no floating-point conversion of large Unix timestamps.
TIME_UNITS = {'seconds':'s','milliseconds':'ms','microseconds':'us','nanoseconds':'ns'}

def interpretable_time(value, unit):
    if unit not in TIME_UNITS:
        raise ValueError('Unsupported timestamp unit: '+repr(unit))
    return common.epoch_to_utc(value, TIME_UNITS[unit])[1] is None


# Q13: audit representability separately from temporal plausibility.  The
# configured release window is expressed in the configured timestamp unit.
# Its bounds are NOT a hard filter on usable review chronology.
Q13_STATES = (
    "missing_or_empty", "wrong_nonnull_type", "unrepresentable_in_configured_unit",
    "representable_before_reference_window", "representable_inside_reference_window",
    "representable_at_or_after_reference_end",
)


def q13_datetime(value, unit):
    if not integer(value):
        return None
    if unit not in TIME_UNITS:
        raise ValueError('Unsupported timestamp unit: '+repr(unit))
    instant, error = common.epoch_to_utc(value, TIME_UNITS[unit])
    return instant if error is None else None


def q13_utc_reference(value):
    if not isinstance(value, str):
        raise ValueError('q13_audit_as_of_utc must be an ISO timestamp with timezone')
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('Invalid q13_audit_as_of_utc ISO timestamp') from error
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError('q13_audit_as_of_utc must include a timezone')
    return dt.astimezone(timezone.utc)


class Q13TimestampAudit:
    """Mutually exclusive raw timestamp states and overlapping suspicion flags."""
    def __init__(self, settings):
        self.unit = settings['timestamp_unit']
        self.lower = settings['review_timestamp_min']
        self.upper = settings['review_timestamp_max_exclusive']
        self.as_of = q13_utc_reference(settings['q13_audit_as_of_utc'])
        self.limit = settings['example_lines_per_rule']
        self.counts = Counter()
        self.examples = defaultdict(list)
        self.suspicions = Counter()
        self.suspicion_examples = defaultdict(list)
        self.earliest = None
        self.latest = None
        self.earliest_in_window = None
        self.latest_in_window = None

    def add(self, row, line):
        raw = row.get('timestamp')
        if 'timestamp' not in row or missing_state(raw) != 'present_nonempty':
            state = 'missing_or_empty'
        elif not integer(raw):
            state = 'wrong_nonnull_type'
        else:
            date = q13_datetime(raw, self.unit)
            if date is None:
                state = 'unrepresentable_in_configured_unit'
            else:
                self.earliest = date if self.earliest is None else min(date, self.earliest)
                self.latest = date if self.latest is None else max(date, self.latest)
                if raw < self.lower:
                    state = 'representable_before_reference_window'
                elif raw >= self.upper:
                    state = 'representable_at_or_after_reference_end'
                else:
                    state = 'representable_inside_reference_window'
                    self.earliest_in_window = date if self.earliest_in_window is None else min(date,self.earliest_in_window)
                    self.latest_in_window = date if self.latest_in_window is None else max(date,self.latest_in_window)
                if date > self.as_of:
                    flag = 'representable_after_audit_as_of'
                    self.suspicions[flag] += 1
                    if len(self.suspicion_examples[flag]) < self.limit:
                        self.suspicion_examples[flag].append(line)
        self.counts[state] += 1
        if len(self.examples[state]) < self.limit:
            self.examples[state].append(line)

    def result(self, n, states, types, rules, available, q03):
        if sum(self.counts.values()) != n:
            raise AssertionError('Q13 status partition does not equal parsed review records')
        representable = sum(self.counts[k] for k in Q13_STATES if k.startswith('representable_'))
        if representable != available['interpretable_timestamp_in_configured_unit']:
            raise AssertionError('Q13 representability disagrees with original A count')
        if representable != q03['field_outcomes']['timestamp']['usable']['rows']:
            raise AssertionError('Q13 representability disagrees with Q03')
        if self.counts['missing_or_empty'] != sum(states.get(k, 0) for k in
                ('absent','null','blank_string','empty_array','empty_object')):
            raise AssertionError('Q13 missing timestamps disagree with Q01')
        wrong_nonnull_type = sum(value for typ, value in types.items() if typ not in ('integer','null'))
        if wrong_nonnull_type != rules.get('timestamp_wrong_type', 0):
            raise AssertionError('Q13 wrong types disagree with original A/Q07')
        if (self.counts['wrong_nonnull_type'] +
            sum(states.get(k,0) for k in ('blank_string','empty_array','empty_object'))
            != wrong_nonnull_type):
            raise AssertionError('Q13 missing and wrong type categories inconsistent')
        expected_old_uninterpretable = (n - states.get('absent',0) - states.get('null',0) - representable)
        if rules.get('timestamp_uninterpretable', 0) != expected_old_uninterpretable:
            raise AssertionError('Q13 unrepresentable/old rule mismatch')
        before = self.counts['representable_before_reference_window']
        after = self.counts['representable_at_or_after_reference_end']
        if rules.get('timestamp_outside_declared_coverage', 0) != before + after:
            raise AssertionError('Q13 outside reference window disagrees with original A')
        if self.suspicions['representable_after_audit_as_of'] > representable:
            raise AssertionError('Q13 audit-future timestamps exceed representable population')
        def iso(dt): return dt.isoformat().replace('+00:00','Z') if dt is not None else None
        def row(status):
            c = self.counts[status]
            return dict(status=status,count=c,denominator_parsed_review_objects=n,
                percent_of_parsed_reviews=100*c/n if n else None,
                example_physical_lines=self.examples[status],
                assessment=('technically_unusable' if status in ('missing_or_empty','wrong_nonnull_type','unrepresentable_in_configured_unit')
                            else 'plausibility_warning' if status!='representable_inside_reference_window'
                            else 'representable_in_reference_window'))
        return dict(timestamp_field='timestamp',configured_unit=self.unit,timezone='UTC',
            reference_start_in_timestamp_units=self.lower,reference_end_exclusive_in_timestamp_units=self.upper,
            reference_start_utc=iso(q13_datetime(self.lower,self.unit)),
            reference_end_exclusive_utc=iso(q13_datetime(self.upper,self.unit)),
            audit_as_of_utc=iso(self.as_of),denominator_parsed_review_objects=n,
            statuses=[row(status) for status in Q13_STATES],
            representable_rows=representable,technically_unusable_rows=n-representable,
            representable_after_audit_as_of_rows=self.suspicions['representable_after_audit_as_of'],
            representable_after_audit_as_of_percent_of_parsed_reviews=(100*self.suspicions['representable_after_audit_as_of']/n if n else None),
            representable_after_audit_as_of_example_physical_lines=self.suspicion_examples['representable_after_audit_as_of'],
            earliest_representable_utc=iso(self.earliest),latest_representable_utc=iso(self.latest),
            earliest_in_reference_window_utc=iso(self.earliest_in_window),
            latest_in_reference_window_utc=iso(self.latest_in_window),
            interpretation=('Technical representability is distinct from a release-window or audit-as-of warning. '
                'Out-of-window representable values remain counted in the ORIGINAL Q03 core-usable rows; '
                'that is not final historical eligibility. Do not infer a 5+10 prediction cohort from these counts. '
                'The scan is read-only; no unit autodetection, date repair, filtering or deletion is performed.'))


def decision(rule, role=None):
    if rule == "full_record_duplicate_extra":
        if role == "metadata":
            return ("Retain raw metadata unchanged. For a future derived metadata table, remove only "
                    "VERIFIED identical extra copies; separately assess distinct records sharing a parent_asin "
                    "before deciding on any join. Never treat identical copies as distinct product versions.")
        return "Later retain one verified canonical copy and log the removed copies; do not count duplicates as separate reviews. Raw files remain unchanged."
    if rule.startswith("core_"):
        return "Count as unsuitable for the main sequence until resolved; do not delete from raw data. Empty text alone is allowed."
    if rule in ("rating_out_of_domain", "timestamp_uninterpretable") or rule.startswith("parent_asin_"):
        return "Investigate; unresolved invalid product ID, rating or time cannot define the main ordered sequence."
    if rule.startswith("timestamp_outside"):
        return "Check timestamp unit and source coverage in the temporal audit; no automatic rescaling or deletion."
    if "price" in rule:
        return "Keep missing/invalid price explicit; do not impute blindly. Current price is not a main historical feature."
    if "text" in rule or "title" in rule:
        return "Retain the review's position; keep a text-availability flag. Inspect unusual content without automatic deletion."
    if rule.startswith("event_"):
        return "Possible event grouping only; shared user/product/time does not prove that records describe one event."
    return "Inspect examples and document a later processing rule; no automatic repair or deletion in this step."


class ExactDuplicates:
    """Hash index plus full canonical comparison, retaining first source offsets.

    The gzip input is simultaneously spooled to disk for inexpensive verification.
    For plain JSONL, the original file is used via a separate read handle.
    A digest collision creates a separate entry rather than a false duplicate.
    """
    def __init__(self, db, role, backing, writer=None, example_limit=5):
        self.db, self.role = db, role
        self.reader = open(backing, "rb")
        self.writer = writer
        # One stored canonical representative per distinct record. We persist
        # only source offsets and counts, never a second copy of every JSON row.
        self.example_limit = example_limit
        self.collision_distinct_records = 0
        db.execute(f"CREATE TABLE duplicates_{role} (digest BLOB, offset INTEGER, first_line INTEGER, first_duplicate_line INTEGER, n INTEGER, PRIMARY KEY(digest,offset)) WITHOUT ROWID")

    def add(self, payload, offset, line_no):
        digest = hashlib.sha256(payload).digest()
        rows = self.db.execute(f"SELECT offset, first_line FROM duplicates_{self.role} WHERE digest=?", (digest,)).fetchall()
        for earlier_offset, first_line in rows:
            if self.writer:
                self.writer.flush()
            self.reader.seek(earlier_offset)
            earlier = canonical(parse_line(self.reader.readline(), earlier_offset == 0))
            if earlier == payload:
                self.db.execute(
                    f"UPDATE duplicates_{self.role} SET n=n+1, "
                    f"first_duplicate_line=COALESCE(first_duplicate_line,?) WHERE digest=? AND offset=?",
                    (line_no, digest, earlier_offset))
                return digest, first_line, earlier_offset
        if rows:
            # Same SHA-256 digest but a DIFFERENT full parsed JSON record:
            # treat it as a separate canonical group, not as a duplicate.
            self.collision_distinct_records += 1
        self.db.execute(f"INSERT INTO duplicates_{self.role} VALUES (?,?,?,NULL,1)", (digest, offset, line_no))
        return digest, None, offset

    def summary(self):
        row = self.db.execute(f"SELECT COUNT(*), COALESCE(SUM(n),0), COALESCE(SUM(CASE WHEN n>1 THEN 1 ELSE 0 END),0), COALESCE(SUM(CASE WHEN n>1 THEN n ELSE 0 END),0), COALESCE(SUM(n-1),0) FROM duplicates_{self.role}").fetchone()
        return dict(zip(("unique_canonical_records", "compared_records", "duplicate_groups", "rows_in_duplicate_groups", "duplicate_extra_rows"), row))

    def q04_examples(self):
        """Bounded, reproducible source-line evidence; no review contents exposed."""
        pairs = self.db.execute(
            f"SELECT first_line,first_duplicate_line,n FROM duplicates_{self.role} "
            f"WHERE n>1 ORDER BY first_line LIMIT ?", (self.example_limit,)
        ).fetchall()
        return {
            "example_first_and_duplicate_lines": [
                {"first_source_line": a, "first_duplicate_source_line": b,
                 "copies_in_group": copies} for a, b, copies in pairs],
            "distinct_records_sharing_a_digest_with_another_record": self.collision_distinct_records,
        }

    def close(self):
        self.reader.close()


def q04_review_exact_duplicate_summary(base, examples, object_records,
                                        canonicalization_failures, duplicate_rule_count):
    """Q04 exact duplicate audit, explicitly scoped to parsed review objects.

    Overlapping row defects do not change duplicate identity. The sample is
    only for source-line traceability, not an estimate of duplicate frequency.
    """
    unique = base["unique_canonical_records"]
    compared = base["compared_records"]
    groups = base["duplicate_groups"]
    involved = base["rows_in_duplicate_groups"]
    extras = base["duplicate_extra_rows"]
    untested = canonicalization_failures
    if not (compared + untested == object_records and unique + extras == compared
            and involved - extras == groups and 0 <= groups <= unique
            and 0 <= extras <= involved <= compared
            and duplicate_rule_count == extras):
        raise AssertionError("Q04 duplicate census conservation failure")

    def percent(count, denominator):
        return 100.0 * count / denominator if denominator else None

    return {
        "definition": "Equality of FULL strictly parsed JSON objects after deterministic key ordering; "
                      "JSON whitespace/object key order ignored, array order and numeric types retained. "
                      "An SHA-256 match is always verified against canonical source bytes.",
        "unit": "parsed review object (not product ID, user ID or review-event key)",
        "denominator_parsed_review_objects": object_records,
        "compared_canonical_review_objects": compared,
        "unable_to_compare_canonicalization_failures": untested,
        "distinct_canonical_review_records": unique,
        "duplicate_groups": groups,
        "review_rows_in_duplicate_groups": involved,
        "extra_duplicate_review_rows": extras,
        "extra_duplicate_percent_of_compared": percent(extras, compared),
        "extra_duplicate_percent_of_parsed": percent(extras, object_records),
        "rows_in_duplicate_groups_percent_of_compared": percent(involved, compared),
        "status": "complete_for_scanned_parsed_objects" if untested == 0
                  else "incomplete_due_to_uncanonicalizable_parsed_objects",
        "examples": examples["example_first_and_duplicate_lines"],
        "distinct_records_sharing_a_digest_with_another_record":
            examples["distinct_records_sharing_a_digest_with_another_record"],
        "initial_decision": "Do not alter raw files. In a future derived review table, retain the earliest "
                            "verified canonical copy and document removed extras; reevaluate "
                            "5+10 eligibility after deduplication.",
        "caveat": "All rates concern the scanned part of this input; blank/invalid/nonobject "
                  "lines cannot be compared as parsed review objects. A head preview is NOT a full-file estimate. "
                  "Duplicate checks across DIFFERENT category files are not included here and need a separate cross-file audit. Same product/user/timestamp or similar review text alone does NOT prove a full-record duplicate.",
    }


def q05_metadata_exact_duplicate_summary(base, examples, object_records,
                                          canonicalization_failures, duplicate_rule_count):
    """Q05 metadata *full-record* equality, not repeated parent_asin equality.

    Checks the same exact duplicate conservation identities as Q04. A malformed
    JSON line is not a parsed metadata object and is counted at the file level.
    """
    unique = base["unique_canonical_records"]
    compared = base["compared_records"]
    groups = base["duplicate_groups"]
    involved = base["rows_in_duplicate_groups"]
    extras = base["duplicate_extra_rows"]
    untested = canonicalization_failures
    if not (compared + untested == object_records and unique + extras == compared
            and involved - extras == groups and 0 <= groups <= unique
            and 0 <= extras <= involved <= compared
            and duplicate_rule_count == extras):
        raise AssertionError("Q05 metadata duplicate census conservation failure")

    def pct(count, denominator):
        return 100.0 * count / denominator if denominator else None

    return {
        "definition": "Exact equality of complete strictly parsed JSON metadata objects after "
                      "canonical key ordering; JSON whitespace and object key order ignored, "
                      "array order and numeric types retained. All SHA-256 matches are verified "
                      "by rereading and comparing the entire canonical source record.",
        "unit": "parsed metadata object (NOT distinct parent_asin or joined review)",
        "denominator_parsed_metadata_objects": object_records,
        "compared_canonical_metadata_objects": compared,
        "unable_to_compare_canonicalization_failures": untested,
        "distinct_canonical_metadata_records": unique,
        "duplicate_groups": groups,
        "metadata_rows_in_duplicate_groups": involved,
        "extra_duplicate_metadata_rows": extras,
        "extra_duplicate_percent_of_compared": pct(extras, compared),
        "extra_duplicate_percent_of_parsed": pct(extras, object_records),
        "rows_in_duplicate_groups_percent_of_compared": pct(involved, compared),
        "status": "complete_for_scanned_parsed_objects" if untested == 0
                  else "incomplete_due_to_uncanonicalizable_parsed_objects",
        "examples": examples["example_first_and_duplicate_lines"],
        "distinct_records_sharing_a_digest_with_another_record":
            examples["distinct_records_sharing_a_digest_with_another_record"],
        "initial_decision": "Preserve raw metadata. Only verified extra copies of the SAME complete "
                            "metadata object are candidates for removal in a later derived view. "
                            "Do not collapse DIFFERENT records with a shared parent_asin; "
                            "flag those for separate key/version reconciliation before joins.",
        "caveat": "Scanned-input only: a head preview cannot establish full-file duplicate counts. "
                  "Malformed, non-object and empty lines are separate parsing diagnostics, not "
                  "metadata object duplicates. A repeated parent_asin with DIFFERENT data is "
                  "not an exact duplicate (Q06/S14). A full-record duplicate in one file "
                  "says nothing about overlap with another category file. No records are removed.",
    }


def init_db(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA cache_size=-65536")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("CREATE TABLE numeric_values (metric TEXT, value REAL, n INTEGER, PRIMARY KEY(metric,value)) WITHOUT ROWID")
    db.execute("CREATE TABLE event_candidates (event_key BLOB PRIMARY KEY, first_payload BLOB, n INTEGER, varied INTEGER) WITHOUT ROWID")
    # Q03: product-level counts are exact and disk-backed (no in-memory product census).
    # Raw identifier JSON is serialized as ASCII bytes to preserve exact identity,
    # including strings that are unsafe for SQLite's Unicode TEXT encoder.
    db.execute("CREATE TABLE q03_review_products (parent_key BLOB PRIMARY KEY, raw_rows INTEGER NOT NULL, core_usable_rows INTEGER NOT NULL, core_with_body_rows INTEGER NOT NULL) WITHOUT ROWID")
    # Q06 exact RAW identifier multiplicity. Key is a canonical JSON string as
    # ASCII bytes; no trimming, case-folding or guessed identity merging.
    db.execute("CREATE TABLE q06_identifier_counts (role TEXT NOT NULL, field TEXT NOT NULL, key BLOB NOT NULL, raw_rows INTEGER NOT NULL, canonicalized_rows INTEGER NOT NULL, PRIMARY KEY(role,field,key)) WITHOUT ROWID")
    # One reference offset per *verified canonical metadata record* under a key;
    # unlike digest-only grouping, this is safe even with digest collisions.
    db.execute("CREATE TABLE q06_metadata_versions (parent_key BLOB NOT NULL, representative_offset INTEGER NOT NULL, copies INTEGER NOT NULL, PRIMARY KEY(parent_key,representative_offset)) WITHOUT ROWID")
    db.execute("CREATE TABLE q09_category_forms (role TEXT, field TEXT, normalized TEXT, raw TEXT, n INTEGER NOT NULL, first_line INTEGER NOT NULL, PRIMARY KEY(role,field,normalized,raw)) WITHOUT ROWID")
    return db



# Q11: Accepted review stars must be a WHOLE NUMBER in the closed range 1..5.
# Finite JSON numeric 5.0 (or 5e0) is a valid representation of five stars;
# 4.5 is not. Do not round, cast numeric-looking strings or coerce bools.
Q11_STATES = (
    'valid_integral_star', 'missing_or_empty', 'wrong_nonnull_type',
    'numeric_outside_1_to_5', 'numeric_fractional_within_1_to_5',
)


class Q11ReviewRatingValidity:
    """Exclusive per-review partition, plus orthogonal numeric anomaly counts.

    Counts are PRE-dedup and use parsed review objects as denominator.
    JSON parse failures are recorded elsewhere and never used as rating rows.
    """
    def __init__(self, example_limit):
        self.counts = Counter()
        self.valid_stars = Counter()
        self.examples = defaultdict(list)
        self.numeric_fractional_all = 0
        self.fractional_outside_intersection = 0
        self.valid_float_integral = 0
        self.example_limit = example_limit

    def add(self, record, physical_line):
        if 'rating' not in record or missing_state(record['rating']) != 'present_nonempty':
            status = 'missing_or_empty'
        else:
            value = record['rating']
            if not number(value):  # rejects bool and numeric-looking strings
                status = 'wrong_nonnull_type'
            else:
                fractional = value % 1 != 0
                if fractional:
                    self.numeric_fractional_all += 1
                if not 1 <= value <= 5:
                    status = 'numeric_outside_1_to_5'
                    if fractional:
                        self.fractional_outside_intersection += 1
                elif fractional:
                    status = 'numeric_fractional_within_1_to_5'
                else:
                    status = 'valid_integral_star'
                    self.valid_stars[int(value)] += 1
                    self.valid_float_integral += type(value) is float
        self.counts[status] += 1
        if len(self.examples[status]) < self.example_limit:
            self.examples[status].append(physical_line)

    def result(self, parsed_objects, prior_valid_rating, prior_invalid_rule, missing_statuses):
        all_counts = {key: self.counts[key] for key in Q11_STATES}
        valid = all_counts['valid_integral_star']
        total = sum(all_counts.values())
        absent_or_null = missing_statuses['absent'] + missing_statuses['null']
        # Prior A flags only PRESENT and invalid values. Missing null/absent
        # are excluded from rating_out_of_domain, but blank strings are flagged.
        if (total != parsed_objects or valid != prior_valid_rating or
            sum(self.valid_stars.values()) != valid or
            self.numeric_fractional_all != self.fractional_outside_intersection + all_counts['numeric_fractional_within_1_to_5'] or
            parsed_objects - valid - absent_or_null != prior_invalid_rule or
            self.valid_float_integral > valid):
            raise AssertionError('Q11 counts disagree with Q03/base rating validity or raw missingness')
        def pct(count):
            return 100.0 * count / parsed_objects if parsed_objects else None
        rows = []
        for name in Q11_STATES:
            rows.append({
                'status': name, 'count': all_counts[name], 'denominator_parsed_review_objects': parsed_objects,
                'percent_of_parsed_review_objects': pct(all_counts[name]),
                'example_physical_lines': self.examples[name],
                'initial_decision': ('Retain as valid five-level rating; no conversion needed.' if name == 'valid_integral_star' else
                    'Keep original position and mark rating missing; do not impute target stars.' if name == 'missing_or_empty' else
                    'Mark rating unusable for the main sequence; inspect type, never coerce automatically.' if name == 'wrong_nonnull_type' else
                    'Mark rating unusable for the main sequence; inspect source, do not clip or round.'),
            })
        return {
            'field': 'rating', 'unit': 'parsed raw review objects before duplicate removal',
            'denominator_parsed_review_objects': parsed_objects,
            'valid_integral_stars_rows': valid,
            'invalid_or_missing_rows': parsed_objects - valid,
            'missing_absent_or_explicit_null_rows': absent_or_null,
            'numeric_fractional_all_rows': self.numeric_fractional_all,
            'numeric_fractional_outside_range_intersection_rows': self.fractional_outside_intersection,
            'valid_integral_float_representation_rows': self.valid_float_integral,
            'results': rows,
            'valid_star_counts': {str(i): self.valid_stars[i] for i in range(1,6)},
            'valid_star_percent_of_all_parsed': {str(i): pct(self.valid_stars[i]) for i in range(1,6)},
            'interpretation': ('Valid rating means a finite JSON int/float representing exactly one of the '
                'integers 1..5; 5.0 is valid, 4.5 is invalid. No conversion or rounding. '
                'Missing/empty vs wrong type vs numeric out of range vs fractional in range are exclusive. '
                'The additional fractional-all metric overlaps numeric out of range by design. '
                'Corrupt/nonobject JSON lines are not in the denominator. No deduplication or data repair.'),
        }



# Q12: Value/domain checks for review helpful_vote and verified_purchase.
# Q07 handles raw field types; Q12 must not silently coerce integer-like
# floats or numeric strings. All partitions are mutually exclusive.
Q12_FIELDS = {
    'helpful_vote': ('missing_or_empty', 'valid_zero_integer',
                     'valid_positive_integer', 'negative_integer',
                     'fractional_float', 'integral_float_wrong_type',
                     'other_wrong_nonnull_type'),
    'verified_purchase': ('missing_or_empty', 'valid_true', 'valid_false',
                          'wrong_nonnull_type'),
}


class Q12ReviewValueValidity:
    """Streaming validity census of review votes and verified purchase.

    Every parsed review contributes one and only one status to each field.
    Extra numeric flags are non-exclusive and explicitly labelled as such.
    """
    def __init__(self, example_limit):
        self.counts = defaultdict(Counter)
        self.examples = defaultdict(list)
        self.numeric_flags = Counter()
        self.example_limit = example_limit

    def add(self, record, line):
        for field in Q12_FIELDS:
            if field not in record or missing_state(record[field]) != 'present_nonempty':
                status = 'missing_or_empty'
            else:
                value = record[field]
                if field == 'helpful_vote':
                    if type(value) is int:
                        status = ('negative_integer' if value < 0 else
                                  'valid_zero_integer' if value == 0 else
                                  'valid_positive_integer')
                    elif type(value) is float:
                        if value < 0:
                            self.numeric_flags['negative_float_values'] += 1
                        if not value.is_integer():
                            status = 'fractional_float'
                            self.numeric_flags['fractional_float_values'] += 1
                        else:
                            status = 'integral_float_wrong_type'
                            self.numeric_flags['integral_float_wrong_type_values'] += 1
                    else:
                        status = 'other_wrong_nonnull_type'
                else:
                    status = ('valid_true' if value is True else
                              'valid_false' if value is False else
                              'wrong_nonnull_type')
            self.counts[field][status] += 1
            if len(self.examples[(field,status)]) < self.example_limit:
                self.examples[(field,status)].append(line)

    def result(self, parsed_objects, missing_states, observed_types, rule_counts):
        outputs = {}
        for field, statuses in Q12_FIELDS.items():
            counts = {status: self.counts[field][status] for status in statuses}
            if sum(counts.values()) != parsed_objects:
                raise AssertionError('Q12 partition does not sum: '+field)
            raw = missing_states[field]
            missing_or_empty = sum(raw[state] for state in
                ('absent','null','blank_string','empty_array','empty_object'))
            if counts['missing_or_empty'] != missing_or_empty:
                raise AssertionError('Q12 missing states disagree: '+field)
            types = observed_types.get(field,{})
            wrong_q07 = rule_counts.get(field+'_wrong_type',0)
            if field == 'helpful_vote':
                valid = counts['valid_zero_integer'] + counts['valid_positive_integer']
                invalid_present = parsed_objects - valid - raw['absent'] - raw['null']
                if rule_counts.get('helpful_vote_invalid',0) != invalid_present:
                    raise AssertionError('Q12 helpful_vote disagrees with A validation')
                if types.get('integer',0) != valid+counts['negative_integer']:
                    raise AssertionError('Q12 helpful_vote integer type mismatch')
                if types.get('number',0) != counts['fractional_float']+counts['integral_float_wrong_type']:
                    raise AssertionError('Q12 helpful_vote float type mismatch')
                if wrong_q07 != (counts['fractional_float'] + counts['integral_float_wrong_type'] +
                        counts['other_wrong_nonnull_type'] + raw['blank_string'] +
                        raw['empty_array'] + raw['empty_object']):
                    raise AssertionError('Q12 helpful_vote disagrees with Q07 wrong types')
            else:
                valid = counts['valid_true'] + counts['valid_false']
                if valid != types.get('boolean',0):
                    raise AssertionError('Q12 verified_purchase boolean mismatch')
                if wrong_q07 != counts['wrong_nonnull_type'] + raw['blank_string'] + raw['empty_array'] + raw['empty_object']:
                    raise AssertionError('Q12 verified_purchase disagrees with Q07 wrong types')
            rows=[]
            for status in statuses:
                ok = status in (('valid_zero_integer','valid_positive_integer') if field=='helpful_vote'
                                else ('valid_true','valid_false'))
                rows.append(dict(field=field,status=status,count=counts[status],
                    denominator_parsed_review_objects=parsed_objects,
                    percent_of_parsed_reviews=100*counts[status]/parsed_objects if parsed_objects else None,
                    example_physical_lines=self.examples[(field,status)],
                    initial_decision=(
                        'Retain valid value with its original representation; do not modify raw review.'
                        if ok else
                        'Retain missing/empty as unavailable, without imputation or inventing a purchase flag.'
                        if status=='missing_or_empty' else
                        'Flag for source inspection; do not convert, round, clamp, or delete automatically.')))
            outputs[field] = dict(field=field,unit='parsed raw review object before deduplication',
                denominator_parsed_review_objects=parsed_objects,
                valid_rows=valid,missing_or_empty_rows=counts['missing_or_empty'],
                invalid_nonmissing_rows=parsed_objects-valid-counts['missing_or_empty'],
                rows=rows)
        return dict(fields=outputs,numeric_overlap_flags={key:self.numeric_flags[key] for key in
                    ('negative_float_values','fractional_float_values','integral_float_wrong_type_values')},
            interpretation=('Helpful votes must be raw JSON integers >=0; negative integers, '
                'fractional floats, integral floats, and other wrong types are distinguished. '
                'For verified_purchase only raw JSON boolean true/false is valid. '
                'Integer 0/1, strings true/false and missing/empty values are not coerced. '
                'Numeric overlap flags are descriptive, not extra excluded rows. '
                'Existing Q07 wrong-type flags and original A helpful_vote_invalid are cross-checked. '
                'No unsupported relationship between votes and verified_purchase is presumed; '
                'no rows are changed or deleted.'))




# Q14: Metadata numeric domain audit. This is independent of the raw JSON
# type audit (Q08) and of snapshot-aggregate consistency candidates (Q10).
# A source zero sentinel is a WARNING, not proof of a corrupt product record.
Q14_FIELDS = {
    'price': ('missing_or_empty','wrong_nonnull_type','negative_number',
              'zero_requires_inspection','positive_number'),
    'average_rating': ('missing_or_empty','wrong_nonnull_type','negative_number',
                       'zero_possible_source_sentinel','numeric_above_five',
                       'numeric_within_1_to_5'),
    'rating_number': ('missing_or_empty','valid_zero_integer','valid_positive_integer',
                      'negative_integer','fractional_float',
                      'integral_float_wrong_type','other_wrong_nonnull_type'),
}


class Q14MetadataValueValidity:
    """Exclusive domain-status counts for each parsed metadata object.

    Retains conservative A flags; distinguishes provisional sentinel warnings,
    invalid values and raw type errors. Never modifies metadata.
    """
    def __init__(self, example_limit):
        self.counts = defaultdict(Counter)
        self.examples = defaultdict(list)
        self.flags = Counter()
        self.example_limit = example_limit

    def add(self, record, line):
        for field in Q14_FIELDS:
            if field not in record or missing_state(record[field]) != 'present_nonempty':
                status = 'missing_or_empty'
            else:
                value = record[field]
                if field == 'rating_number':
                    if type(value) is int:
                        status = ('negative_integer' if value < 0 else
                                  'valid_zero_integer' if value == 0 else
                                  'valid_positive_integer')
                    elif type(value) is float:
                        if value < 0:
                            self.flags['rating_number_negative_float'] += 1
                        if not value.is_integer():
                            status = 'fractional_float'
                            self.flags['rating_number_fractional_float'] += 1
                        else:
                            status = 'integral_float_wrong_type'
                            self.flags['rating_number_integral_float'] += 1
                    else:
                        status = 'other_wrong_nonnull_type'
                elif not number(value):
                    status = 'wrong_nonnull_type'
                elif field == 'price':
                    status = ('negative_number' if value < 0 else
                              'zero_requires_inspection' if value == 0 else
                              'positive_number')
                else:
                    status = ('negative_number' if value < 0 else
                              'zero_possible_source_sentinel' if value == 0 else
                              'numeric_above_five' if value > 5 else
                              'numeric_within_1_to_5')
            self.counts[field][status] += 1
            if len(self.examples[(field,status)]) < self.example_limit:
                self.examples[(field,status)].append(line)

    def result(self, parsed_objects, missing_states, observed_types, rule_counts, available):
        outputs = {}
        for field, statuses in Q14_FIELDS.items():
            counts = {status:self.counts[field][status] for status in statuses}
            if sum(counts.values()) != parsed_objects:
                raise AssertionError('Q14 partition does not sum: '+field)
            raw = missing_states[field]
            missing = sum(raw[state] for state in (
                'absent','null','blank_string','empty_array','empty_object'))
            if missing != counts['missing_or_empty']:
                raise AssertionError('Q14 missing-state mismatch: '+field)
            if field == 'price':
                valid = counts['positive_number'] + counts['zero_requires_inspection']
                suspect = counts['zero_requires_inspection']
                if available['valid_current_price'] != valid:
                    raise AssertionError('Q14 price availability disagrees with A')
                if rule_counts.get('price_zero_requires_inspection',0) != suspect:
                    raise AssertionError('Q14 zero price disagrees with A')
                incompatible = counts['wrong_nonnull_type']
            elif field == 'average_rating':
                valid = counts['numeric_within_1_to_5']
                suspect = counts['zero_possible_source_sentinel']
                incompatible = counts['wrong_nonnull_type']
            else:
                valid = counts['valid_zero_integer'] + counts['valid_positive_integer']
                suspect = 0
                incompatible = (counts['fractional_float'] +
                    counts['integral_float_wrong_type'] + counts['other_wrong_nonnull_type'])
                if observed_types.get(field, {}).get('integer',0) != (
                        valid + counts['negative_integer']):
                    raise AssertionError('Q14 raw rating_number integer mismatch')
                if observed_types.get(field, {}).get('number',0) != (
                        counts['fractional_float'] + counts['integral_float_wrong_type']):
                    raise AssertionError('Q14 raw rating_number float mismatch')
                if self.flags['rating_number_integral_float'] != counts['integral_float_wrong_type'] or self.flags['rating_number_fractional_float'] != counts['fractional_float']:
                    raise AssertionError('Q14 rating_number supplementary flags mismatch')
            # The original A flags every present, non-null, domain-invalid
            # value. Blank or empty containers are among those legacy flags.
            legacy_invalid = parsed_objects - valid - raw['absent'] - raw['null']
            if rule_counts.get(field+'_invalid',0) != legacy_invalid:
                raise AssertionError('Q14 domain validity disagrees with A: '+field)
            expected_type_mismatches = (incompatible + raw['blank_string'] +
                raw['empty_array'] + raw['empty_object'])
            if rule_counts.get(field+'_wrong_type',0) != expected_type_mismatches:
                raise AssertionError('Q14 raw type mismatch disagrees with Q08: '+field)
            rows=[]
            for status in statuses:
                good = status in (
                    ('positive_number','zero_requires_inspection') if field=='price' else
                    ('numeric_within_1_to_5',) if field=='average_rating' else
                    ('valid_zero_integer','valid_positive_integer'))
                warning = status in ('zero_requires_inspection','zero_possible_source_sentinel')
                decision = ('Preserve value; inspect source convention or potential sentinel, do not infer a free item or invalid record.' if warning else
                    'Retain observed valid field; do not use crawl-time aggregate as a historical predictor.' if good else
                    'Keep unavailable; do not use zero to impute missing metadata.' if status=='missing_or_empty' else
                    'Flag invalid field for inspection; do not coerce, clip, round or delete the product.')
                rows.append(dict(field=field,status=status,count=counts[status],
                    denominator_parsed_metadata_objects=parsed_objects,
                    percent_of_parsed_metadata=100*counts[status]/parsed_objects if parsed_objects else None,
                    example_physical_lines=self.examples[(field,status)],
                    initial_decision=decision))
            outputs[field]=dict(field=field,
                denominator_parsed_metadata_objects=parsed_objects,
                valid_under_A_domain_rows=valid,
                missing_or_empty_rows=missing,
                invalid_nonmissing_under_A_rows=parsed_objects-valid-missing,
                warning_zero_rows=suspect,
                statuses=rows)
        return dict(fields=outputs,
            supplementary_flags={k:self.flags[k] for k in (
                'rating_number_negative_float','rating_number_fractional_float',
                'rating_number_integral_float')},
            interpretation=('Price: finite JSON numeric >=0 is type/domain-usable, but zero is a diagnostic warning. '
                'Average rating: 1..5 is the adopted valid numeric range; zero is a possible '
                'source-specific unrated sentinel, and is deliberately still counted by the '
                'legacy average_rating_invalid flag. Rating count: raw nonnegative integer only; '
                'integral floats and numeric-looking strings are not silently converted. '
                'Status categories are exclusive for each field, while supplementary flags overlap. '
                'Values are crawl-time metadata, not guaranteed historically available predictors. '
                'Invalid and missing fields do not imply deleting the product record. '
                'No source data are cleaned, imputed or altered.'))

def flush_metrics(db, histogram):
    db.executemany("INSERT INTO numeric_values VALUES (?,?,?) ON CONFLICT(metric,value) DO UPDATE SET n=n+excluded.n",
                   ((metric, value, count) for (metric, value), count in histogram.items()))
    histogram.clear()
    db.commit()



# Q03 status covers only the THREE required fields for forming a dated,
# product-level review/rating sequence. Text, images and metadata are optional
# at this step, not reasons to remove the observation.
Q03_CORE_FIELDS = ("parent_asin", "rating", "timestamp")


def q03_field_outcome(row, field, valid):
    if valid:
        return "usable"
    if field not in row or missing_state(row[field]) != "present_nonempty":
        return "missing_or_empty"
    return "present_but_unusable"


def q03_review_readiness(mask_counts, field_statuses, text_counts, products, n, threshold):
    """Exclusive review classifications + non-exclusive failure causes.

    n counts *parsed raw JSON objects* before deduplication. Product counts
    are NOT a definitive prediction-cohort size: timestamps, tie handling,
    exact duplicates and held-out splits still need their own decisions.
    """
    if sum(mask_counts.values()) != n:
        raise AssertionError("Q03 core-mask counts do not sum to parsed review objects")
    for field in Q03_CORE_FIELDS:
        if sum(field_statuses[field].values()) != n:
            raise AssertionError("Q03 field-outcome counts do not sum: " + field)
    if text_counts["body_usable"] + text_counts["body_unusable"] != n:
        raise AssertionError("Q03 body text counts do not sum")
    if text_counts["any_text_usable"] + text_counts["any_text_unusable"] != n:
        raise AssertionError("Q03 title-or-body counts do not sum")
    if text_counts["core_with_body"] > mask_counts[0] or text_counts["core_with_any_text"] > mask_counts[0]:
        raise AssertionError("Q03 text/core intersection invalid")

    def rate(count):
        return 100.0 * count / n if n else None

    failures = {"none": mask_counts[0]}
    for mask in range(1,8):
        label = "+".join(field for i,field in enumerate(Q03_CORE_FIELDS) if mask & (1 << i))
        failures[label] = mask_counts[mask]
    invalid_union = n - mask_counts[0]
    if invalid_union != sum(v for k,v in failures.items() if k != "none"):
        raise AssertionError("Q03 exclusive failure patterns inconsistent")
    product_row = products.execute("""
        SELECT COUNT(*), COALESCE(SUM(raw_rows),0),
          COALESCE(SUM(core_usable_rows),0),
          COALESCE(SUM(core_with_body_rows),0),
          COALESCE(SUM(raw_rows >= ?),0),
          COALESCE(SUM(core_usable_rows >= ?),0),
          COALESCE(SUM(raw_rows >= ? AND core_usable_rows < ?),0),
          COALESCE(SUM(core_usable_rows = 0),0),
          COALESCE(SUM(raw_rows > core_usable_rows),0)
        FROM q03_review_products
    """, (threshold,threshold,threshold,threshold)).fetchone()
    keys = ("products_with_usable_parent_key", "raw_rows_with_usable_parent_key",
            "core_usable_rows_with_usable_parent_key", "core_with_body_rows_with_usable_parent_key",
            "products_at_least_threshold_raw", "products_at_least_threshold_core_before_dedup",
            "products_falling_below_threshold_due_to_unusable_core_rows",
            "products_with_zero_core_usable_rows", "products_with_some_unusable_core_rows")
    product_counts = dict(zip(keys, product_row))
    if product_counts["core_usable_rows_with_usable_parent_key"] != mask_counts[0]:
        raise AssertionError("Q03 product/core row counts mismatch")
    if product_counts["core_with_body_rows_with_usable_parent_key"] != text_counts["core_with_body"]:
        raise AssertionError("Q03 product/core/text counts mismatch")
    if product_counts["raw_rows_with_usable_parent_key"] != field_statuses["parent_asin"]["usable"]:
        raise AssertionError("Q03 parent-key population count mismatch")
    if product_counts["products_at_least_threshold_core_before_dedup"] > product_counts["products_at_least_threshold_raw"]:
        raise AssertionError("Q03 core eligibility cannot exceed raw")
    return {
        "unit": "parsed review JSON object before deduplication",
        "denominator_review_objects": n,
        "required_for_ordered_rating_sequence": list(Q03_CORE_FIELDS),
        "field_outcomes": {
            field: {kind: {"rows": field_statuses[field][kind], "percent_of_review_objects": rate(field_statuses[field][kind])}
                    for kind in ("usable", "missing_or_empty", "present_but_unusable")}
            for field in Q03_CORE_FIELDS
        },
        "core_usable_review_rows": mask_counts[0],
        "core_unusable_review_rows_union": invalid_union,
        "core_usable_percent": rate(mask_counts[0]),
        "core_unusable_percent": rate(invalid_union),
        "exclusive_failure_patterns": {key: {"rows": val,"percent_of_review_objects":rate(val)}
                                       for key,val in failures.items()},
        "text_optional": {
            key: {"rows": text_counts[key], "percent_of_review_objects":rate(text_counts[key])}
            for key in ("body_usable", "body_unusable", "any_text_usable", "any_text_unusable",
                        "core_with_body", "core_without_body", "core_with_any_text", "core_without_any_text")
        },
        "minimum_observed_core_reviews_per_product": threshold,
        "product_counts_before_dedup": product_counts,
        "product_count_interpretation": "These are screening counts based on raw records. At least 15 apparently usable observations is necessary but NOT sufficient for final 5+10 eligibility; exact duplicates, same-time ordering, coverage and splitting have not been resolved.",
        "metadata_interpretation": "No metadata field or review text is required for the basic observed rating/time sequence. Metadata feature coverage after join belongs to S26; historical crawl-time snapshot values cannot be assumed available at the prediction cutoff.",
    }



# Q06 counts observed identifiers, NOT duplicate events or distinct products
# established in the full S14 structural audit. Repeated review keys are normal.
Q06_FIELDS = {"reviews": ("parent_asin", "asin", "user_id"), "metadata": ("parent_asin",)}


def flush_q06(db, key_buffer, version_buffer):
    if key_buffer:
        db.executemany("""INSERT INTO q06_identifier_counts VALUES (?,?,?,?,?)
            ON CONFLICT(role,field,key) DO UPDATE SET
                raw_rows=raw_rows+excluded.raw_rows,
                canonicalized_rows=canonicalized_rows+excluded.canonicalized_rows""",
            ((role,field,key,counts[0],counts[1]) for (role,field,key),counts in key_buffer.items()))
        key_buffer.clear()
    if version_buffer:
        db.executemany("""INSERT INTO q06_metadata_versions VALUES (?,?,?)
            ON CONFLICT(parent_key,representative_offset) DO UPDATE SET
                copies=copies+excluded.copies""",
            ((key,offset,count) for (key,offset),count in version_buffer.items()))
        version_buffer.clear()
    db.commit()


def q06_identifier_summary(db, role, object_records, canonical_failures, full_duplicate_extras):
    """Counts raw exact-ID repeats separately from full-record duplicates."""
    out=[]
    for field in Q06_FIELDS[role]:
        row=db.execute("""SELECT COUNT(*),COALESCE(SUM(raw_rows),0),
                COALESCE(SUM(canonicalized_rows),0),
                COALESCE(SUM(raw_rows>1),0),
                COALESCE(SUM(CASE WHEN raw_rows>1 THEN raw_rows ELSE 0 END),0),
                COALESCE(SUM(CASE WHEN raw_rows>1 THEN raw_rows-1 ELSE 0 END),0)
            FROM q06_identifier_counts WHERE role=? AND field=?""",(role,field)).fetchone()
        distinct, valid_rows, canonical_rows, groups, involved, extras=row
        if not (0<=canonical_rows<=valid_rows<=object_records and
                groups<=distinct<=valid_rows and involved-extras==groups and
                distinct+extras==valid_rows and valid_rows-canonical_rows<=canonical_failures):
            raise AssertionError('Q06 identifier conservation failure: '+role+'.'+field)
        out.append(dict(file_role=role,field=field,unit='exact raw nonblank string identifier',
                        denominator_parsed_objects=object_records,
                        rows_with_usable_raw_id=valid_rows,
                        rows_without_usable_raw_id=object_records-valid_rows,
                        distinct_raw_ids=distinct,repeated_id_groups=groups,
                        rows_in_repeated_id_groups=involved,rows_beyond_first_per_id=extras,
                        repeated_id_rows_percent_of_parsed_objects=100*involved/object_records if object_records else None,
                        repeated_id_rows_percent_of_valid_ids=100*involved/valid_rows if valid_rows else None,
                        rows_with_unclassifiable_canonical_record=valid_rows-canonical_rows,
                        interpretation=('Review identifiers may legitimately repeat across observations; '
                                        'the repeated-key count is NOT a duplicate-review count.' if role=='reviews'
                                        else 'Shared metadata parent keys require distinguishing identical copies '
                                             'from genuinely different product metadata records.'),
                        first_stage_duplicate_extras_all_ids=full_duplicate_extras))
    metadata=None
    if role=='metadata':
        # One outcome per parent group: only-identical copies; conflicting
        # verified versions; or unclassified because canonicalization failed.
        classified=db.execute("""SELECT
            COALESCE(SUM(CASE WHEN k.raw_rows>1 AND k.raw_rows=k.canonicalized_rows AND v.versions=1 THEN 1 ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN k.raw_rows>1 AND k.raw_rows=k.canonicalized_rows AND v.versions>1 THEN 1 ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN k.raw_rows>1 AND k.raw_rows>k.canonicalized_rows THEN 1 ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN k.raw_rows>1 AND k.raw_rows=k.canonicalized_rows AND v.versions=1 THEN k.raw_rows ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN k.raw_rows>1 AND k.raw_rows=k.canonicalized_rows AND v.versions>1 THEN k.raw_rows ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN k.raw_rows>1 AND k.raw_rows>k.canonicalized_rows THEN k.raw_rows ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN k.raw_rows=k.canonicalized_rows THEN k.raw_rows-v.versions ELSE 0 END),0)
        FROM q06_identifier_counts k
        LEFT JOIN (SELECT parent_key,COUNT(*) versions,SUM(copies) copies
                  FROM q06_metadata_versions GROUP BY parent_key) v ON k.key=v.parent_key
        WHERE k.role='metadata' AND k.field='parent_asin'""").fetchone()
        only_identical, conflicting, unknown, only_rows, conflict_rows, unknown_rows, id_duplicate_extras=classified
        repeated=out[0]['repeated_id_groups']
        if (only_identical+conflicting+unknown!=repeated or
            only_rows+conflict_rows+unknown_rows!=out[0]['rows_in_repeated_id_groups'] or
            not 0<=id_duplicate_extras<=full_duplicate_extras):
            raise AssertionError('Q06 metadata-version partition mismatch')
        metadata=dict(repeated_parent_groups_only_identical_records=only_identical,
                      repeated_parent_groups_with_distinct_verified_versions=conflicting,
                      repeated_parent_groups_indeterminate=unknown,
                      rows_in_identical_only_parent_groups=only_rows,
                      rows_in_conflicting_parent_groups=conflict_rows,
                      rows_in_indeterminate_parent_groups=unknown_rows,
                      verified_extra_full_record_copies_among_complete_valid_parent_groups=id_duplicate_extras,
                      full_record_duplicate_extras_in_whole_metadata_file=full_duplicate_extras,
                      caution='Conflicting versions are NOT full-record duplicates. '
                              'Do not arbitrarily select one metadata row for joins. '
                              'If canonicalization fails, distinct-version status is unknown; '
                              'S14 join multiplicity and cross-category comparisons remain separate.')
    return dict(role=role,scope='single scanned file; raw identifier equality without normalization',
                identifier_fields=out,metadata_parent_version_classification=metadata,
                full_record_duplicate_extras_in_same_file=full_duplicate_extras,
                caveat='Repeated identifier is not proof of a duplicate row or a repeated real-world event. '
                       'Use Q04/Q05 for verified full-record duplicates; structural S14 for joins. '
                       'Preview results must not be interpreted as whole-file results.')


# Q20: fixed, explicit per-row family masks, not sums of independent rule totals.
# Every row is counted ONCE in exactly one signature.  The union is therefore
# exact for these defined families, not for all possible defects and not for
# Q15/Q16 post-scan statistical outliers (which need per-row membership).
Q20_FAMILIES = (
    'core_or_parent_id_unusable',
    'expected_field_absent_or_null',
    'expected_field_wrong_type',
    'invalid_numeric_or_timestamp',
    'exact_full_record_duplicate_extra',
    'nested_type_anomaly',
    'format_or_text_screen',
    'source_or_cross_field_warning',
)
Q20_STRONG_FAMILIES = frozenset((
    'core_or_parent_id_unusable', 'expected_field_wrong_type',
    'invalid_numeric_or_timestamp', 'exact_full_record_duplicate_extra',
    'nested_type_anomaly',
))
Q20_SCOPE_NOTE = (
    'Counts overlap among eight explicitly defined families of parsed JSON object records BEFORE dedup. '
    'Absent/null EXPECTED fields (including optional metadata) are descriptive missingness, NOT a proof of unusability. '
    'Formatting and source warnings are provisional; union(any family) must NOT be called erroneous rows. '
    'Q06 repeated identifiers and post-scan Q15/Q16 statistical outlier memberships are NOT in this union. '
    'No claim is made about the union of ALL possible quality issues; review core readiness remains Q03.'
)

class Q20RowOverlap:
    def __init__(self, role, example_limit):
        self.role, self.example_limit = role, example_limit
        self.masks = Counter()
        self.examples = defaultdict(list)

    def add(self, row, line, row_rules, *, core_unusable,
            nested_type_anomaly, format_issue, cross_field_warning):
        families = set()
        if core_unusable:
            families.add('core_or_parent_id_unusable')
        if any(f not in row or row[f] is None for f in EXPECTED[self.role]):
            families.add('expected_field_absent_or_null')
        if any(v is not None and f in EXPECTED[self.role] and
               common.type_name(v) not in EXPECTED[self.role][f]
               for f,v in row.items()):
            families.add('expected_field_wrong_type')
        invalid_rules = ('rating_out_of_domain','timestamp_uninterpretable',
                         'helpful_vote_invalid','price_invalid',
                         'average_rating_invalid','rating_number_invalid')
        if any(f in row_rules for f in invalid_rules):
            families.add('invalid_numeric_or_timestamp')
        if 'full_record_duplicate_extra' in row_rules:
            families.add('exact_full_record_duplicate_extra')
        if nested_type_anomaly or any(x in row_rules for x in (
                'images_nonobject_element','features_nonstring_element',
                'description_nonstring_element','categories_nonstring_element')):
            families.add('nested_type_anomaly')
        if format_issue or any(rule.endswith(('_surrounding_whitespace',
                '_possible_missing_marker','_unpaired_unicode_surrogate',
                '_replacement_character','_html_like_markup','_control_character'))
                or rule in ('text_very_short_heuristic','text_very_long_heuristic')
                for rule in row_rules):
            families.add('format_or_text_screen')
        if cross_field_warning or 'timestamp_outside_declared_coverage' in row_rules or \
                'price_zero_requires_inspection' in row_rules or \
                (self.role == 'metadata' and number(row.get('average_rating')) and
                 row['average_rating'] == 0):
            families.add('source_or_cross_field_warning')
        mask = sum(1 << index for index,name in enumerate(Q20_FAMILIES) if name in families)
        self.masks[mask] += 1
        if len(self.examples[mask]) < self.example_limit:
            self.examples[mask].append(line)

    def result(self, n):
        if sum(self.masks.values()) != n:
            raise AssertionError('Q20 record signature partition mismatch: '+self.role)
        family_counts = {name:sum(count for mask,count in self.masks.items() if mask & (1<<i))
                         for i,name in enumerate(Q20_FAMILIES)}
        pairs = []
        for i,name1 in enumerate(Q20_FAMILIES):
            for j in range(i+1,len(Q20_FAMILIES)):
                name2=Q20_FAMILIES[j]
                intersection=sum(count for mask,count in self.masks.items()
                                 if mask & (1<<i) and mask & (1<<j))
                if not 0 <= intersection <= min(family_counts[name1],family_counts[name2]):
                    raise AssertionError('Q20 pair intersection invariant')
                pairs.append(dict(first_family=name1,second_family=name2,
                                  count=intersection,denominator_parsed_objects=n,
                                  percent_of_parsed_objects=100*intersection/n if n else None))
        strong_mask=sum(1<<i for i,name in enumerate(Q20_FAMILIES) if name in Q20_STRONG_FAMILIES)
        union_counts = {
            'any_defined_family':sum(c for mask,c in self.masks.items() if mask),
            'strong_technical_or_structural_concern':sum(c for mask,c in self.masks.items() if mask & strong_mask),
            'only_descriptive_or_provisional_screens':sum(c for mask,c in self.masks.items() if mask and not mask & strong_mask),
            'no_defined_family':self.masks[0],
            'at_least_two_distinct_families':sum(c for mask,c in self.masks.items() if mask.bit_count()>=2),
        }
        if union_counts['any_defined_family']+union_counts['no_defined_family']!=n or \
                union_counts['strong_technical_or_structural_concern']+union_counts['only_descriptive_or_provisional_screens']!=union_counts['any_defined_family']:
            raise AssertionError('Q20 union identity mismatch')
        signatures=[]
        for mask,count in sorted(self.masks.items()):
            families=[name for i,name in enumerate(Q20_FAMILIES) if mask & (1<<i)]
            signatures.append(dict(signature=mask,active_families=families,count=count,
                                   denominator_parsed_objects=n,
                                   percent_of_parsed_objects=100*count/n if n else None,
                                   example_physical_lines=self.examples[mask]))
        return dict(file_role=self.role,denominator_parsed_objects=n,
                    family_counts=family_counts,pair_intersections=pairs,
                    exact_signatures=signatures,union_counts=union_counts,
                    included_families=list(Q20_FAMILIES),
                    excluded_analyses=('Q06 repeated identifier groups','Q15/Q16 statistical outliers',
                                       'post-join cross-file discrepancies'),
                    scope_note=Q20_SCOPE_NOTE)


def analyze_file(path, role, db, work, settings, preview_lines):
    before = path.stat()
    states, types = defaultdict(Counter), defaultdict(Counter)
    # Q01: keep the raw six-state census and nested content assessment separate.
    nested_content_counts = {field: Counter() for field in NESTED_CONTENT_FIELDS[role]}
    rules, examples, available = Counter(), defaultdict(list), Counter()
    availability_names = (["core_sequence_fields_usable_before_dedup", "valid_rating",
        "interpretable_timestamp_in_configured_unit", "nonblank_review_title_or_body",
        "nonblank_review_body", "nonblank_review_title"] if role == "reviews" else
        ["valid_current_price", "nonblank_features_string_element",
         "nonblank_description_string_element", "nonblank_categories_string_element",
         "nonblank_product_title"])
    available.update({name: 0 for name in availability_names + ["image_http_url_reference"]})
    histogram = Counter()
    # Q07: small, in-memory counters only; no raw strings or entire rows are saved.
    q07_numeric_candidates = Counter()
    # Q08: bounded-memory diagnostics for metadata type compatibility.
    q08_numeric_candidates = Counter()
    q08_nested = Q08MetadataNested(settings['example_lines_per_rule']) if role == 'metadata' else None
    q09_formats = Q09Formats(role, db, settings['example_lines_per_rule'], settings['batch_size'])
    q10_relations = Q10CrossFieldConsistency(role, settings['example_lines_per_rule'])
    q11_rating = Q11ReviewRatingValidity(settings['example_lines_per_rule']) if role == 'reviews' else None
    q12_values = Q12ReviewValueValidity(settings['example_lines_per_rule']) if role == 'reviews' else None
    q13_times = Q13TimestampAudit(settings) if role == 'reviews' else None
    q14_values = Q14MetadataValueValidity(settings['example_lines_per_rule']) if role == 'metadata' else None
    q20_overlap = Q20RowOverlap(role, settings['example_lines_per_rule'])
    q24_text = Q24ReviewUnicodeAudit(settings['example_lines_per_rule']) if role == 'reviews' else None
    q25_text_rating = Q25RatingTextAssociation(settings) if role == 'reviews' else None
    current_q20_rules = set()
    # Q03: bounded-memory counters; keyed product counts are flushed to SQLite.
    q03_field_status = defaultdict(Counter)
    q03_masks = Counter()
    q03_text = Counter()
    q03_product_buffer = defaultdict(lambda: [0,0,0])
    q06_key_buffer = defaultdict(lambda: [0,0])
    q06_version_buffer = Counter()
    info = dict(path=str(path), stored_bytes=before.st_size, physical_lines=0,
                object_records=0, blank_lines=0, parse_errors=0,
                duplicate_json_key_lines=0, nonfinite_json_lines=0,
                nonobject_lines=0, canonicalization_failures=0)
    digest = hashlib.sha256()
    unit = settings["timestamp_unit"]
    min_time, max_time = settings["review_timestamp_min"], settings["review_timestamp_max_exclusive"]
    parse_error_examples = []
    offset = 0
    started = time.monotonic()
    gzip_input = path.suffix.lower() == ".gz"
    backing = work / (role + "_decompressed.jsonl") if gzip_input else path
    spool = open(backing, "wb") if gzip_input else None
    detector = ExactDuplicates(db, role, backing, spool,
                               example_limit=settings["example_lines_per_rule"])

    def flag(rule, line):
        current_q20_rules.add(rule)
        rules[rule] += 1
        if len(examples[rule]) < settings["example_lines_per_rule"]:
            examples[rule].append(line)

    def flush_q03_products():
        if not q03_product_buffer:
            return
        db.executemany(
            "INSERT INTO q03_review_products VALUES (?,?,?,?) "
            "ON CONFLICT(parent_key) DO UPDATE SET "
            "raw_rows=raw_rows+excluded.raw_rows, "
            "core_usable_rows=core_usable_rows+excluded.core_usable_rows, "
            "core_with_body_rows=core_with_body_rows+excluded.core_with_body_rows",
            ((key, *counts) for key, counts in q03_product_buffer.items()))
        q03_product_buffer.clear()

    def metric(name, value, line):
        # Keep histogram representations exact for integer values stored as REAL.
        if integer(value) and abs(value) > 2**53:
            flag(name + "_too_large_for_numeric_summary", line)
            return
        histogram[(role + "." + name, float(value))] += 1

    opener = gzip.open if gzip_input else open
    try:
        with opener(path, "rb") as stream:
            lines = islice(stream, preview_lines) if preview_lines else stream
            for line_no, raw in enumerate(lines, 1):
                current_offset = offset
                offset += len(raw)
                info["physical_lines"] += 1
                digest.update(raw)
                if spool:
                    spool.write(raw)
                if not raw.strip():
                    info["blank_lines"] += 1
                else:
                    try:
                        row = parse_line(raw, line_no == 1)
                    except (UnicodeDecodeError, ValueError, RecursionError) as error:
                        info["parse_errors"] += 1
                        info["duplicate_json_key_lines"] += isinstance(error, DuplicateJSONKey)
                        info["nonfinite_json_lines"] += isinstance(error, NonFiniteJSON)
                        if len(parse_error_examples) < settings["example_lines_per_rule"]:
                            parse_error_examples.append({"line": line_no, "type": type(error).__name__})
                    else:
                        if not isinstance(row, dict):
                            info["nonobject_lines"] += 1
                        else:
                            info["object_records"] += 1
                            current_q20_rules.clear()
                            for field, value in row.items():
                                raw_state = missing_state(value)
                                states[field][raw_state] += 1
                                if field in nested_content_counts:
                                    counters = nested_content_counts[field]
                                    if raw_state == "present_nonempty":
                                        counters["raw_present_nonempty_rows"] += 1
                                        if nested_has_usable_content(value, NESTED_CONTENT_FIELDS[role][field]):
                                            counters["usable_content_rows"] += 1
                                        else:
                                            counters["raw_nonempty_but_unusable_rows"] += 1
                                kind = common.type_name(value)
                                types[field][kind] += 1
                                if role == "reviews" and q07_numeric_string_candidate(field, value):
                                    q07_numeric_candidates[field] += 1
                                if role == "metadata" and q08_numeric_string_candidate(field, value):
                                    q08_numeric_candidates[field] += 1
                                if value is not None and field in EXPECTED[role] and kind not in EXPECTED[role][field]:
                                    flag(field + "_wrong_type", line_no)
                                if isinstance(value, str) and value.strip().casefold() in settings["potential_missing_strings"]:
                                    flag(field + "_possible_missing_marker", line_no)
                                if isinstance(value, str):
                                    try:
                                        value.encode("utf-8", errors="strict")
                                    except UnicodeEncodeError:
                                        flag(field + "_unpaired_unicode_surrogate", line_no)
                            q20_nested_issue = q08_nested.add(row, line_no) if q08_nested is not None else False
                            q20_format_issue = q09_formats.add(row, line_no)
                            q20_cross_warning = q10_relations.add(row, line_no)
                            for field in ("parent_asin", "asin", "user_id") if role == "reviews" else ("parent_asin",):
                                if not common.indexable_id(row.get(field)):
                                    flag(field + "_unusable_id", line_no)
                                elif row[field] != row[field].strip():
                                    flag(field + "_surrounding_whitespace", line_no)
                            try:
                                payload = canonical(row)
                            except (ValueError, RecursionError):
                                info["canonicalization_failures"] += 1
                                payload = None
                            canonical_offset = None
                            if payload is not None:
                                payload_digest, duplicate_of, canonical_offset = detector.add(payload, current_offset, line_no)
                                if duplicate_of is not None:
                                    flag("full_record_duplicate_extra", line_no)
                            # Raw key equality is deliberately NOT normalization.
                            # Store only bounded counters and canonical representative offsets.
                            for id_field in Q06_FIELDS[role]:
                                id_value = row.get(id_field)
                                if common.indexable_id(id_value):
                                    key = canonical(id_value)
                                    counts = q06_key_buffer[(role,id_field,key)]
                                    counts[0] += 1
                                    counts[1] += int(canonical_offset is not None)
                                    if role == "metadata" and canonical_offset is not None:
                                        q06_version_buffer[(key,canonical_offset)] += 1
                            if role == "reviews":
                                q11_rating.add(row, line_no)
                                q12_values.add(row, line_no)
                                q13_times.add(row, line_no)
                                rating = row.get("rating")
                                rating_ok = number(rating) and rating in (1, 2, 3, 4, 5)
                                if rating is not None and not rating_ok:
                                    flag("rating_out_of_domain", line_no)
                                timestamp = row.get("timestamp")
                                time_ok = interpretable_time(timestamp, unit)
                                if timestamp is not None and not time_ok:
                                    flag("timestamp_uninterpretable", line_no)
                                if time_ok and not min_time <= timestamp < max_time:
                                    flag("timestamp_outside_declared_coverage", line_no)
                                core_ok = common.indexable_id(row.get("parent_asin")) and rating_ok and time_ok
                                if not core_ok:
                                    flag("core_sequence_fields_unusable_before_dedup", line_no)
                                else:
                                    available["core_sequence_fields_usable_before_dedup"] += 1
                                available["valid_rating"] += rating_ok
                                available["interpretable_timestamp_in_configured_unit"] += time_ok
                                # Q03: separate missing/empty from present-but-invalid,
                                # and count nonexclusive causes via an EXCLUSIVE bitmask.
                                valid_core = (common.indexable_id(row.get("parent_asin")),rating_ok,time_ok)
                                q03_mask = 0
                                for bit, (field, valid) in enumerate(zip(Q03_CORE_FIELDS, valid_core)):
                                    q03_field_status[field][q03_field_outcome(row,field,valid)] += 1
                                    if not valid:
                                        q03_mask |= (1 << bit)
                                q03_masks[q03_mask] += 1
                                # Do not make body text, title or any metadata a core
                                # inclusion requirement: first-five positions stay fixed.
                                body_usable = nonblank(row.get("text"))
                                any_text_usable = body_usable or nonblank(row.get("title"))
                                q03_text["body_usable" if body_usable else "body_unusable"] += 1
                                q03_text["any_text_usable" if any_text_usable else "any_text_unusable"] += 1
                                if core_ok:
                                    q03_text["core_with_body" if body_usable else "core_without_body"] += 1
                                    q03_text["core_with_any_text" if any_text_usable else "core_without_any_text"] += 1
                                if valid_core[0]:
                                    parent_key = canonical(row["parent_asin"])
                                    bucket = q03_product_buffer[parent_key]
                                    bucket[0] += 1
                                    if core_ok:
                                        bucket[1] += 1
                                        bucket[2] += int(body_usable)
                                # Q24 scans only the raw review BODY; title and metadata are separate.
                                q24_text.add(row.get("text"), line_no)
                                q25_text_rating.add(row)
                                for field in ("text", "title"):
                                    value = row.get(field)
                                    if isinstance(value, str):
                                        metric(field + "_characters", len(value), line_no)
                                        if field == "text":
                                            words = len(value.split())
                                            metric("text_whitespace_words", words, line_no)
                                            if value.strip() and words <= settings["very_short_text_words"]:
                                                flag("text_very_short_heuristic", line_no)
                                            if len(value) > settings["very_long_text_characters"]:
                                                flag("text_very_long_heuristic", line_no)
                                        if "\ufffd" in value:
                                            flag(field + "_replacement_character", line_no)
                                        if re.search(r"<[^>]+>", value):
                                            flag(field + "_html_like_markup", line_no)
                                        if any(ord(c) < 32 and c not in "\t\n\r" for c in value):
                                            flag(field + "_control_character", line_no)
                                has_text = nonblank(row.get("text")) or nonblank(row.get("title"))
                                available["nonblank_review_title_or_body"] += has_text
                                available["nonblank_review_body"] += nonblank(row.get("text"))
                                available["nonblank_review_title"] += nonblank(row.get("title"))
                                votes = row.get("helpful_vote")
                                if votes is not None:
                                    if integer(votes) and votes >= 0:
                                        metric("helpful_vote", votes, line_no)
                                    else:
                                        flag("helpful_vote_invalid", line_no)
                                if payload is not None and common.indexable_id(row.get("user_id")) and common.indexable_id(row.get("parent_asin")) and time_ok:
                                    event_key = hashlib.sha256(canonical([row["user_id"], row["parent_asin"], timestamp])).digest()
                                    db.execute("INSERT INTO event_candidates VALUES (?,?,1,0) ON CONFLICT(event_key) DO UPDATE SET n=n+1, varied=MAX(varied,first_payload!=excluded.first_payload)", (event_key, payload_digest))
                            else:
                                q14_values.add(row, line_no)
                                for field in ("price", "average_rating", "rating_number"):
                                    value = row.get(field)
                                    if value is None:
                                        continue
                                    valid = (number(value) and value >= 0) if field == "price" else ((number(value) and 1 <= value <= 5) if field == "average_rating" else (integer(value) and value >= 0))
                                    if valid:
                                        if field != "average_rating":
                                            metric(field, value, line_no)
                                    else:
                                        flag(field + "_invalid", line_no)
                                available["valid_current_price"] += number(row.get("price")) and row["price"] >= 0
                                for field in ("features", "description", "categories"):
                                    value = row.get(field)
                                    if isinstance(value, list):
                                        if any(not isinstance(v, str) for v in value):
                                            flag(field + "_nonstring_element", line_no)
                                        available["nonblank_" + field + "_string_element"] += any(nonblank(v) for v in value)
                                available["nonblank_product_title"] += nonblank(row.get("title"))
                                if row.get("price") == 0 and number(row.get("price")):
                                    flag("price_zero_requires_inspection", line_no)
                            images = row.get("images")
                            has_url = False
                            if isinstance(images, list):
                                if any(not isinstance(image, dict) for image in images):
                                    flag("images_nonobject_element", line_no)
                                has_url = any(isinstance(image, dict) and any(isinstance(image.get(key), str) and image[key].startswith(("https://", "http://")) for key in IMAGE_KEYS[role]) for image in images)
                            available["image_http_url_reference"] += has_url
                            q20_overlap.add(row,line_no,current_q20_rules,
                                core_unusable=(not core_ok if role=="reviews" else
                                    not common.indexable_id(row.get("parent_asin"))),
                                nested_type_anomaly=q20_nested_issue,
                                format_issue=q20_format_issue,
                                cross_field_warning=q20_cross_warning)
                if line_no % settings["batch_size"] == 0:
                    if role == "reviews":
                        flush_q03_products()
                    flush_q06(db, q06_key_buffer, q06_version_buffer)
                    flush_metrics(db, histogram)
                if settings["progress_every_lines"] and line_no % settings["progress_every_lines"] == 0:
                    print(f"[{role}] {line_no:,} lines, {(time.monotonic()-started)/60:.1f} minutes", flush=True)
        if role == "reviews":
            flush_q03_products()
        flush_q06(db, q06_key_buffer, q06_version_buffer)
        q09_formats.flush()
        flush_metrics(db, histogram)
        info["full_record_duplicates"] = detector.summary()
        # Q04 and Q05 both reuse A's verified full-record duplicate detector.
        # Only the reporting unit/interpretation differs by file role.
        if role == "reviews":
            info["q04_review_exact_duplicates"] = q04_review_exact_duplicate_summary(
                info["full_record_duplicates"], detector.q04_examples(),
                info["object_records"], info["canonicalization_failures"],
                rules["full_record_duplicate_extra"])
        else:
            info["q05_metadata_exact_duplicates"] = q05_metadata_exact_duplicate_summary(
                info["full_record_duplicates"], detector.q04_examples(),
                info["object_records"], info["canonicalization_failures"],
                rules["full_record_duplicate_extra"])
        info["q06_identifier_multiplicity"] = q06_identifier_summary(
            db, role, info["object_records"], info["canonicalization_failures"],
            info["full_record_duplicates"]["duplicate_extra_rows"])
    finally:
        detector.close()
        if spool:
            spool.close()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Input changed during read: " + str(path))
    n = info["object_records"]
    missing = {}
    for field in sorted(set(states) | set(EXPECTED[role])):
        counts = dict(states[field])
        counts["absent"] = n - sum(counts.values())
        missing[field] = {state: counts.get(state, 0) for state in MISSING_STATES}
        # Q01 conservation check: each object must occupy exactly one state.
        if any(value < 0 for value in missing[field].values()) or sum(missing[field].values()) != n:
            raise AssertionError(f"Q01 missing-state census mismatch: {role}.{field}")
    # Q02: rate denominators are successfully parsed JSON object records only,
    # not physical lines, parse failures, or a post-deduplication population.
    q02_fields = {field: q02_field_missingness(counts, n)
                  for field, counts in missing.items()}
    nested_content = {}
    for field, rule in NESTED_CONTENT_FIELDS[role].items():
        counts = nested_content_counts[field]
        raw_nonempty = missing[field]["present_nonempty"]
        assert counts["raw_present_nonempty_rows"] == raw_nonempty, (role, field)
        usable = counts["usable_content_rows"]
        not_usable = counts["raw_nonempty_but_unusable_rows"]
        if usable + not_usable != raw_nonempty:
            raise AssertionError(f"Q01 nested content mismatch: {role}.{field}")
        nested_content[field] = {
            "content_rule": rule,
            "denominator_object_records": n,
            "raw_present_nonempty_rows": raw_nonempty,
            "usable_content_rows": usable,
            "raw_nonempty_but_unusable_rows": not_usable,
            "usable_share_of_all_objects": usable / n if n else None,
            "usable_share_of_raw_nonempty": usable / raw_nonempty if raw_nonempty else None,
        }
    # Q03 reporting, using the same core validity contract as A: no new
    # cleaning, no future filtering, no deduplication or cohort construction.
    q03 = (q03_review_readiness(q03_masks,q03_field_status,q03_text,db,n,
                               settings["q03_min_reviews_for_5_plus_10"])
           if role == "reviews" else None)
    info.update(missing_states=missing,
                q03_review_readiness=q03,
                q03_metadata_optional_only=("Metadata fields are feature availability diagnostics, not prerequisites for the main rating/time sequence. Product-joined impact is deferred to S26."
                                           if role == "metadata" else None),
                q01_raw_states_validated=True,
                q01_nested_content=nested_content,
                q02_field_missingness=q02_fields,
                q02_denominator_definition="Accepted parsed JSON object records in the scanned input; raw copies included; physical non-object/blank/invalid lines excluded.",
                q07_review_type_compatibility=(q07_review_type_summary(types, missing, rules,
                    q07_numeric_candidates, examples, n) if role == "reviews" else None),
                q08_metadata_type_compatibility=(q08_metadata_type_summary(types, missing, rules,
                    q08_numeric_candidates, examples, n) if role == 'metadata' else None),
                q08_metadata_nested_types=(q08_nested.result(n) if q08_nested is not None else None),
                q09_format_audit=q09_formats.result(n),
                q10_cross_field_consistency=q10_relations.result(n),
                q11_review_rating_validity=(q11_rating.result(n, available["valid_rating"],
                    rules.get("rating_out_of_domain", 0), missing["rating"]) if q11_rating is not None else None),
                q12_review_value_validity=(q12_values.result(n, missing, types, rules)
                    if q12_values is not None else None),
                q13_review_timestamp_validity=(q13_times.result(n, missing['timestamp'],
                    types.get('timestamp', {}), rules, available, q03)
                    if q13_times is not None else None),
                q14_metadata_value_validity=(q14_values.result(n,missing,types,rules,available)
                    if q14_values is not None else None),
                q20_row_issue_overlaps=q20_overlap.result(n),
                q24_review_unicode_audit=(q24_text.result(n,types.get("text",{}).get("string",0),rules) if q24_text is not None else None),
                q25_rating_text_accumulator=(q25_text_rating if q25_text_rating is not None else None),
                types={key: dict(value) for key, value in sorted(types.items())},
                rule_counts=dict(sorted(rules.items())),
                rule_example_line_numbers=dict(examples),
                modality_availability={key: {"rows": count, "denominator_object_records": n, "share": count/n if n else None} for key, count in sorted(available.items())},
                parse_error_examples=parse_error_examples,
                scanned_content_sha256=digest.hexdigest(),
                hash_scope="Exactly the scanned decompressed bytes; no BOM or line-ending normalization.",
                scan_mode="head_preview" if preview_lines else "entire_supplied_file",
                elapsed_seconds=round(time.monotonic()-started, 3))
    print(f"[{role}] finished: {n:,} object records", flush=True)
    return info


# Q22: body-text length distribution, from the existing disk-backed SQLite
# frequency table. No extra raw-file pass and no per-review text retention.
# Python len(str) measures Unicode code points (not bytes or grapheme clusters).
# str.split() uses whitespace-separated units (not language-aware tokens).
Q22_METRICS = {
    'unicode_codepoints': 'reviews.text_characters',
    'whitespace_words': 'reviews.text_whitespace_words',
}
Q22_BUCKETS = {
    'unicode_codepoints': ((0,0), (1,9), (10,49), (50,99), (100,199),
                           (200,499), (500,999), (1000,1999),
                           (2000,4999), (5000,None)),
    'whitespace_words': ((0,0), (1,1), (2,3), (4,9), (10,19),
                         (20,49), (50,99), (100,199),
                         (200,499), (500,None)),
}
Q22_QUANTILES = (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def q22_review_text_lengths(db, review_info):
    """Report two exact, fixed-population length distributions before dedup.

    Existing A histograms count every string-valued text field, including
    empty/whitespace-only strings. Missing, null, and wrong-type values have
    NO assigned length; zero-length is a legitimate string value, not missing.
    """
    n = review_info['object_records']
    if type(n) is not int or n < 0:
        raise AssertionError('Q22 invalid parsed review denominator')
    missing = review_info['missing_states']['text']
    absent, null = missing['absent'], missing['null']
    strings = review_info['types'].get('text', {}).get('string', 0)
    usable = review_info['q03_review_readiness']['text_optional']['body_usable']['rows']
    if not all(type(x) is int and x >= 0 for x in (absent,null,strings,usable)):
        raise AssertionError('Q22 invalid raw state counts')
    whitespace_only = strings - usable
    wrong = n - absent - null - strings
    if min(whitespace_only,wrong) < 0 or absent + null + whitespace_only + usable + wrong != n:
        raise AssertionError('Q22 raw body-text population partition mismatch')
    states = {
        'absent': absent,
        'explicit_null': null,
        'empty_or_whitespace_only_string': whitespace_only,
        'nonblank_string': usable,
        'wrong_nonnull_type': wrong,
    }
    distributions = {}
    for unit, metric in Q22_METRICS.items():
        observed, qs = q15_weighted_quantiles(db, metric, Q22_QUANTILES)
        if observed != strings:
            raise AssertionError('Q22 histogram string count does not match type census: '+metric)
        bins = Q22_BUCKETS[unit]
        totals = [0] * len(bins)
        distinct = 0
        minimum = maximum = None
        mean = 0.0
        cumulative = 0
        for value, freq in db.execute(
                'SELECT value,n FROM numeric_values WHERE metric=? ORDER BY value', (metric,)):
            if not math.isfinite(value) or value < 0 or not value.is_integer() or type(freq) is not int or freq <= 0:
                raise AssertionError('Q22 histogram has an invalid length or frequency')
            length = int(value)
            if minimum is None:
                minimum = length
            maximum = length
            mean += (length - mean) * freq / (cumulative + freq)
            cumulative += freq
            distinct += 1
            for i, (lower, upper) in enumerate(bins):
                if length >= lower and (upper is None or length <= upper):
                    totals[i] += freq
                    break
            else:
                raise AssertionError('Q22 uncovered text length: '+str(length))
        if cumulative != observed or sum(totals) != strings:
            raise AssertionError('Q22 weighted histogram partition mismatch')
        distributions[unit] = {
            'metric': metric,
            'unit': unit,
            'string_rows': strings,
            'denominator_parsed_review_objects': n,
            'distinct_observed_lengths': distinct,
            'minimum': minimum,
            'maximum': maximum,
            'mean': mean if strings else None,
            'quantiles': {'p'+str(round(p*100)): qs[p] for p in Q22_QUANTILES},
            'quantile_method': 'Frequency-weighted linear interpolation at (n-1)*p; all string-valued body texts, including whitespace-only.',
            'buckets': [dict(lower_inclusive=lower, upper_inclusive=upper,
                             count=count,
                             denominator_string_values=strings,
                             percent_of_string_values=100*count/strings if strings else None,
                             denominator_parsed_review_objects=n,
                             percent_of_parsed_review_objects=100*count/n if n else None)
                        for (lower,upper),count in zip(bins,totals)]
        }
    # A split() length of zero is exactly an empty/whitespace-only string.
    if distributions['whitespace_words']['buckets'][0]['count'] != whitespace_only:
        raise AssertionError('Q22 whitespace-only body count differs from Q03')
    return {
        'version':'Q22', 'file_role':'reviews', 'scope':'raw parsed review objects before deduplication',
        'denominator_parsed_review_objects':n,
        'string_rows':strings,
        'states':states,
        'distributions':distributions,
        'interpretation':(
            'Lengths describe the review BODY (text), not title, title+body, metadata description, '
            'or 5+10 eligible products. Empty/whitespace-only strings remain in the string histograms '
            'and are also reported separately, not double counted in the five exclusive field states. '
            'Unicode code points are neither UTF-8 bytes nor visible grapheme clusters. '
            'Whitespace split counts are not linguistic word-token counts for all languages. '
            'A missing/null/nonstring body has no imputed length. '
            'Raw duplicate copies are included; preview/sample distributions cannot represent full input. '
            'Very short/long threshold reporting is Q23, language/unusual characters Q24, '
            'and rating/text relationship Q25; no screening changes the chronological review positions.'),
    }




# Q25: descriptive association of observed body text with the raw review star.
# A field-level exploratory audit, not a supervised experiment or sentiment model.
# No token vocabulary or raw bodies are retained; all samples are counted pre-dedup.
Q25_STATES=('absent','null','blank_string','nonblank_string','wrong_nonnull_type')
Q25_FEATURES=('very_short_nonblank','very_long_nonblank','has_exclamation','has_question')


class Q25RatingTextAssociation:
    def __init__(self, settings):
        self.short=settings['very_short_text_words']
        self.long=settings['very_long_text_characters']
        self.stars=Counter()
        self.states={star:Counter() for star in range(1,6)}
        self.features={star:Counter() for star in range(1,6)}
        self.lengths={star:dict(chars=0,words=0,min_chars=None,max_chars=None,
                                min_words=None,max_words=None) for star in range(1,6)}
        self.invalid_rating_rows=0

    def add(self,row):
        rating=row.get('rating')
        if not (number(rating) and rating in (1,2,3,4,5)):
            self.invalid_rating_rows+=1
            return
        star=int(rating)
        self.stars[star]+=1
        if 'text' not in row:
            state='absent'
        else:
            body=row['text']
            if body is None: state='null'
            elif isinstance(body,str): state='nonblank_string' if body.strip() else 'blank_string'
            else: state='wrong_nonnull_type'
        self.states[star][state]+=1
        if state!='nonblank_string':
            return
        body=row['text']
        chars=len(body)
        words=len(body.split())
        lengths=self.lengths[star]
        lengths['chars']+=chars
        lengths['words']+=words
        for name,value in (('chars',chars),('words',words)):
            lo,hi='min_'+name,'max_'+name
            lengths[lo]=value if lengths[lo] is None else min(lengths[lo],value)
            lengths[hi]=value if lengths[hi] is None else max(lengths[hi],value)
        self.features[star]['very_short_nonblank']+=int(words<=self.short)
        self.features[star]['very_long_nonblank']+=int(chars>self.long)
        self.features[star]['has_exclamation']+=int('!' in body or '！' in body)
        self.features[star]['has_question']+=int('?' in body or '؟' in body or '？' in body)

    def result(self,review_rows,q11_valid_star_distribution):
        if sum(self.stars.values())+self.invalid_rating_rows!=review_rows:
            raise AssertionError('Q25 invalid/valid rating partition mismatch')
        expected={int(k):v for k,v in q11_valid_star_distribution.items()}
        if {star:self.stars[star] for star in range(1,6)} != {star:expected.get(star,0) for star in range(1,6)}:
            raise AssertionError('Q25 rating strata must match Q11 valid-star distribution')
        totals=[]
        valid_total=sum(self.stars.values())
        for star in range(1,6):
            if sum(self.states[star].values())!=self.stars[star]:
                raise AssertionError('Q25 per-star text state partition mismatch')
            count=self.states[star]['nonblank_string']
            if any(self.features[star][f]>count for f in Q25_FEATURES):
                raise AssertionError('Q25 textual feature cannot exceed nonblank sample')
            lengths=self.lengths[star]
            totals.append(dict(star=star,valid_rating_rows=self.stars[star],
                text_states={state:self.states[star][state] for state in Q25_STATES},
                features={f:self.features[star][f] for f in Q25_FEATURES},
                nonblank_string_bodies=count,
                sum_unicode_codepoints=lengths['chars'],
                sum_whitespace_words=lengths['words'],
                mean_unicode_codepoints=lengths['chars']/count if count else None,
                mean_whitespace_words=lengths['words']/count if count else None,
                min_unicode_codepoints=lengths['min_chars'],max_unicode_codepoints=lengths['max_chars'],
                min_whitespace_words=lengths['min_words'],max_whitespace_words=lengths['max_words'],
                nonblank_body_percent_of_rating_rows=100*count/self.stars[star] if self.stars[star] else None))
        def combine(stars):
            n=sum(self.stars[x] for x in stars)
            text=sum(self.states[x]['nonblank_string'] for x in stars)
            words=sum(self.lengths[x]['words'] for x in stars)
            return dict(valid_rating_rows=n,nonblank_string_bodies=text,
                        nonblank_percent_of_valid_ratings=100*text/n if n else None,
                        mean_whitespace_words_of_nonblank=words/text if text else None)
        return dict(version='Q25',scope='raw_parsed_review_objects_pre_dedup',
            denominator_all_parsed_review_objects=review_rows,
            denominator_valid_rating_review_objects=valid_total,
            invalid_or_missing_rating_rows=self.invalid_rating_rows,
            per_star=totals,
            groups={'low_rating_1_2':combine((1,2)),'neutral_rating_3':combine((3,)),
                    'high_rating_4_5':combine((4,5))},
            length_definition='Unicode codepoints len(text); whitespace groups len(text.split()); only nonblank string BODY values',
            short_word_threshold=self.short,long_character_threshold=self.long,
            feature_definition='Presence per nonblank text; punctuation is NOT sentiment; features can overlap',
            limitations=['Only valid one-to-five stars; malformed rating rows are excluded from conditional strata.',
                         'Raw observed reviews include exact duplicate copies and products with fewer than fifteen reviews.',
                         'Missing or malformed body text is a state, not a reason to remove a rating or reorder time.',
                         'Length/punctuation are descriptive text features, not calibrated language/sentiment signals.',
                         'No causal, predictive, statistical significance or out-of-sample claims are made.',
                         'Snapshot-level association may reflect product, category, time, or missingness confounding.'])


# Q24 — review BODY script inventory and unusual Unicode signals; no model-based
# language inference. Only strings are scanned. Character scans run inside the
# existing raw-file pass; no text, script-by-text maps, or raw examples are kept.
Q24_SCRIPTS = ('Latin','Arabic','Cyrillic','Greek','Hebrew','Han','Hiragana',
               'Katakana','Hangul','Devanagari','Thai','Other_letter_script')
Q24_FLAGS = (
    'replacement_character', 'unpaired_surrogate', 'c0_control_except_tab_cr_lf',
    'c1_control', 'bidi_embedding_or_isolate_control', 'unicode_noncharacter',
    'private_use_codepoint')
Q24_PATTERNS = ('no_letters','one_letter_script','multiple_letter_scripts')
# Field-specific preliminary actions; labels are warnings, not edits.
Q24_ACTIONS = {
    'replacement_character': 'Inspect original encoding/source; preserve the raw string and do not reconstruct lost text automatically.',
    'unpaired_surrogate': 'Flag possible downstream UTF-8 encoding failure; inspect the source and keep raw JSON evidence.',
    'c0_control_except_tab_cr_lf': 'Inspect control use in context; do not strip without an audited preprocessing policy.',
    'c1_control': 'Inspect Unicode C1 control occurrence; it may be intentional or an encoding artifact.',
    'bidi_embedding_or_isolate_control': 'Inspect visual direction handling; right-to-left typography may use such markers legitimately.',
    'unicode_noncharacter': 'Inspect source encoding; do not infer language or drop the entire review automatically.',
    'private_use_codepoint': 'Inspect source-specific font/markup conventions; private-use text is not automatically corrupt.',
}


def q24_unicode_script(cp):
    # A deliberately broad Unicode BLOCK proxy, not normative Unicode Script
    # property classification (e.g. shared inherited characters are excluded
    # by the isalpha() filter; some historic/extended scripts land in Other).
    if (0x0041 <= cp <= 0x024F or 0x1E00 <= cp <= 0x1EFF or
            0x2C60 <= cp <= 0x2C7F or 0xA720 <= cp <= 0xA7FF or
            0xAB30 <= cp <= 0xAB6F):
        return 'Latin'
    if (0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or
            0x08A0 <= cp <= 0x08FF or 0xFB50 <= cp <= 0xFDFF or
            0xFE70 <= cp <= 0xFEFF or 0x1EE00 <= cp <= 0x1EEFF):
        return 'Arabic'
    if 0x0400 <= cp <= 0x052F or 0x2DE0 <= cp <= 0x2DFF or 0xA640 <= cp <= 0xA69F:
        return 'Cyrillic'
    if 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF: return 'Greek'
    if 0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F: return 'Hebrew'
    if (0x3400 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or
            0x20000 <= cp <= 0x323AF): return 'Han'
    if 0x3040 <= cp <= 0x309F: return 'Hiragana'
    if 0x30A0 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF: return 'Katakana'
    if (0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F or
            0xA960 <= cp <= 0xA97F or 0xAC00 <= cp <= 0xD7AF or
            0xD7B0 <= cp <= 0xD7FF): return 'Hangul'
    if 0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF: return 'Devanagari'
    if 0x0E00 <= cp <= 0x0E7F: return 'Thai'
    return 'Other_letter_script'


class Q24ReviewUnicodeAudit:
    def __init__(self, sample_limit):
        self.string_rows=0
        self.letter_occurrences=Counter()
        self.rows_with_script=Counter()
        self.dominant80=Counter()
        self.patterns=Counter()
        self.flags=Counter()
        self.examples=defaultdict(list)
        self.limit=sample_limit

    def add(self, body, line):
        if not isinstance(body,str):
            return
        self.string_rows+=1
        scripts=Counter()
        problems=set()
        for ch in body:
            cp=ord(ch)
            if ch.isalpha():
                scripts[q24_unicode_script(cp)]+=1
            if cp==0xFFFD: problems.add('replacement_character')
            if 0xD800<=cp<=0xDFFF: problems.add('unpaired_surrogate')
            if cp<0x20 and cp not in (9,10,13): problems.add('c0_control_except_tab_cr_lf')
            if 0x7F<=cp<=0x9F: problems.add('c1_control')
            if 0x202A<=cp<=0x202E or 0x2066<=cp<=0x2069:
                problems.add('bidi_embedding_or_isolate_control')
            if 0xFDD0<=cp<=0xFDEF or (cp&0xFFFF) in (0xFFFE,0xFFFF):
                problems.add('unicode_noncharacter')
            if (0xE000<=cp<=0xF8FF or 0xF0000<=cp<=0xFFFFD or
                    0x100000<=cp<=0x10FFFD): problems.add('private_use_codepoint')
        self.letter_occurrences.update(scripts)
        self.rows_with_script.update(scripts.keys())
        pattern=('no_letters' if not scripts else
                 'one_letter_script' if len(scripts)==1 else 'multiple_letter_scripts')
        self.patterns[pattern]+=1
        if scripts:
            length=sum(scripts.values())
            high=max(scripts.values())
            # If tied, neither script uniquely dominates. Threshold descriptive.
            if high*5>=length*4 and list(scripts.values()).count(high)==1:
                leading=next(k for k,v in scripts.items() if v==high)
                self.dominant80[leading]+=1
            else:
                self.dominant80['no_unique_80_percent_dominant']+=1
        else:
            self.dominant80['no_letters']+=1
        for rule in problems:
            self.flags[rule]+=1
            if len(self.examples[rule])<self.limit:
                self.examples[rule].append(line)

    def result(self, n, string_count_from_types, existing_rules):
        if self.string_rows!=string_count_from_types or self.string_rows>n:
            raise AssertionError('Q24 text-string population mismatches Q22 raw type census')
        if sum(self.patterns.values())!=self.string_rows or sum(self.dominant80.values())!=self.string_rows:
            raise AssertionError('Q24 script-pattern partition mismatch')
        letters=sum(self.letter_occurrences.values())
        for name,legacy in (('replacement_character','text_replacement_character'),
                            ('unpaired_surrogate','text_unpaired_unicode_surrogate'),
                            ('c0_control_except_tab_cr_lf','text_control_character')):
            if self.flags[name]!=existing_rules.get(legacy,0):
                raise AssertionError('Q24 legacy unicode flag mismatch: '+name)
        scripts=[]
        for script in Q24_SCRIPTS:
            c=self.letter_occurrences[script]
            row=self.rows_with_script[script]
            scripts.append(dict(script=script,letter_occurrences=c,
                denominator_all_letters=letters,
                percent_of_all_letters=100*c/letters if letters else None,
                reviews_containing_script=row,
                denominator_string_bodies=self.string_rows,
                percent_of_string_bodies=100*row/self.string_rows if self.string_rows else None,
                dominant80_body_rows=self.dominant80[script]))
        return dict(version='Q24',file_role='reviews',
            scope='raw parsed review BODY strings before deduplication',
            denominator_parsed_review_objects=n, string_valued_body_rows=self.string_rows,
            total_letter_occurrences=letters,script_rows=scripts,
            pattern_counts={k:self.patterns[k] for k in Q24_PATTERNS},
            dominant80_counts={k:self.dominant80[k] for k in Q24_SCRIPTS+('no_letters','no_unique_80_percent_dominant')},
            flag_counts={k:self.flags[k] for k in Q24_FLAGS},
            flag_initial_decisions=dict(Q24_ACTIONS),
            example_physical_lines={k:self.examples[k] for k in Q24_FLAGS},
            interpretation=(
                'Unicode BLOCK proxies for alphabetic characters, NOT detected languages. '
                'Arabic script includes multiple languages; Latin likewise. Han/Kana can coexist naturally. '
                'Multiple scripts, emoji, combining marks, Persian ZWNJ and ordinary tabs/newlines '
                'are not quality failures. Replacement characters and malformed code points warrant inspection; '
                'bidi formatting controls may be legitimate, so no automatic deletion or normalization. '
                'All rate denominators are explicit. Script-presence flags can overlap. '
                'No actual language classifier or sample-based language accuracy validation has been run. '
                'Do not change historical 5+10 positions or claim input language from these descriptive proxies.'))


# Q23: Exploratory short/long review BODY thresholds from Q22 disk-backed
# frequency tables. Do not re-read source records or infer short/long overlap
# from separate one-dimensional marginals. Primary thresholds preserve the
# pre-existing A screening rule: nonblank <=K whitespace words, and >L chars.
Q23_SHORT_SENSITIVITY_WORDS = (1, 2, 3, 5, 10)
Q23_LONG_SENSITIVITY_CHARACTERS = (1000, 2000, 5000, 10000, 20000)


def q23_short_long_review_text(db, info, q22, settings):
    """Quantify primary and sensitivity thresholds; do not classify validity.

    Each count is frequency-weighted from the EXISTING SQLite histograms.
    For short texts, a length of zero (empty/whitespace only) is separately
    reported, NOT called a short nonblank review. Long includes any string,
    even whitespace-only strings, matching the pre-existing A flag semantics.
    """
    n = info['object_records']
    strings = q22['string_rows']
    blank = q22['states']['empty_or_whitespace_only_string']
    nonblank = q22['states']['nonblank_string']
    if blank + nonblank != strings or any(type(v) is not int or v < 0 for v in (n, strings, blank, nonblank)):
        raise AssertionError('Q23 inconsistent Q22 text populations')
    short_main = settings['very_short_text_words']
    long_main = settings['very_long_text_characters']
    for label, bound in (('very_short_text_words', short_main),
                         ('very_long_text_characters', long_main)):
        if type(bound) is not int or bound < 0:
            raise ValueError('Q23 requires a nonnegative integer '+label)
    word_bounds = tuple(sorted(set(Q23_SHORT_SENSITIVITY_WORDS + (short_main,))))
    char_bounds = tuple(sorted(set(Q23_LONG_SENSITIVITY_CHARACTERS + (long_main,))))
    counts = {'words': {k: 0 for k in word_bounds},
              'characters': {k: 0 for k in char_bounds}}
    histogram_total = {}
    for unit, metric, bounds in (
        ('words', Q22_METRICS['whitespace_words'], word_bounds),
        ('characters', Q22_METRICS['unicode_codepoints'], char_bounds),
    ):
        seen = 0
        for value, freq in db.execute(
                'SELECT value,n FROM numeric_values WHERE metric=? ORDER BY value', (metric,)):
            if (not math.isfinite(value) or value < 0 or not value.is_integer()
                    or type(freq) is not int or freq <= 0):
                raise AssertionError('Q23 invalid histogram length/frequency: '+metric)
            length = int(value)
            seen += freq
            if unit == 'words':
                if length > 0:  # matches A: text.strip() and <= K words
                    for bound in bounds:
                        if length <= bound:
                            counts[unit][bound] += freq
            else:
                for bound in bounds:
                    if length > bound:  # matches A: len(text) > L
                        counts[unit][bound] += freq
        if seen != strings:
            raise AssertionError('Q23 histogram count mismatch: '+metric)
        histogram_total[unit] = seen
    # Reconcile primary sensitivity counts to the original A row-level flags.
    primary_short = counts['words'][short_main]
    primary_long = counts['characters'][long_main]
    rules = info['rule_counts']
    if primary_short != rules.get('text_very_short_heuristic', 0):
        raise AssertionError('Q23 short count differs from existing A rule')
    if primary_long != rules.get('text_very_long_heuristic', 0):
        raise AssertionError('Q23 long count differs from existing A rule')
    if counts['words'].get(0, 0) != 0:
        raise AssertionError('Q23 zero-word string incorrectly classified as short')
    if nonblank == 0 and primary_short:
        raise AssertionError('Q23 short count without nonblank texts')

    def row(kind, bound, count, *, primary):
        # All rows have two explicitly typed denominators. No zero-imputation
        # is used when the relevant denominator is zero.
        reference = nonblank if kind == 'short_nonblank_whitespace_words' else strings
        return dict(
            kind=kind,
            unit='whitespace_words' if kind == 'short_nonblank_whitespace_words' else 'unicode_codepoints',
            comparison='nonblank_words_at_most' if kind == 'short_nonblank_whitespace_words' else 'characters_strictly_more_than',
            threshold=bound,
            is_primary=primary,
            count=count,
            denominator_reference_strings=reference,
            percent_of_reference_strings=100*count/reference if reference else None,
            denominator_parsed_review_objects=n,
            percent_of_parsed_review_objects=100*count/n if n else None,
            example_physical_lines=(info['rule_example_line_numbers'].get(
                'text_very_short_heuristic' if kind == 'short_nonblank_whitespace_words'
                else 'text_very_long_heuristic', []) if primary else []),
        )

    table = [row('short_nonblank_whitespace_words', k, counts['words'][k], primary=k==short_main)
             for k in word_bounds]
    table.extend(row('long_unicode_codepoints', k, counts['characters'][k], primary=k==long_main)
                 for k in char_bounds)
    return dict(
        version='Q23', scope='raw parsed review objects before deduplication',
        denominator_parsed_review_objects=n,
        string_valued_body_rows=strings,
        nonblank_body_rows=nonblank,
        empty_or_whitespace_only_body_rows=blank,
        absent_null_or_wrong_type_rows=n-strings,
        primary_short_word_threshold_inclusive=short_main,
        primary_long_character_threshold_exclusive=long_main,
        primary_short_nonblank_rows=primary_short,
        primary_long_string_rows=primary_long,
        rows=table,
        count_warning=('Short and long counts are marginal; separate histograms do not identify '
                       'their intersection/union. Never add rates to estimate unique affected reviews.'),
        interpretation=(
            'Short uses 1..K whitespace-delimited units and excludes 0-word empty/whitespace-only strings. '
            'Long uses >L Unicode code points and includes all string-valued bodies, even whitespace-only; '
            'this matches the earlier row-level A flag. Missing, null, and nonstring bodies have no length. '
            'Primary thresholds are configurable A screening heuristics, not scientific universal cutoffs. '
            'Sensitivity thresholds are descriptive; no automatic drop/trim, model preprocessing, '
            'deduplication, language normalization or modification to 5+10 chronological positions. '
            'Preview and raw repeated copies remain in the scanned population.'),
    )


# Q15: zero-aware diagnostic of review counts. The ORIGINAL numeric_outliers.csv
# remains unchanged for backwards compatibility. Positive counts are screened
# separately on log1p scale; zero values are never silently dropped from the
# population/denominator. This is an exploratory screen, NOT an invalidity rule.
def q15_weighted_quantiles(db, metric, probabilities, *, positive_only=False,
                           transform=None):
    """Exact frequency-weighted linear-interpolated quantiles (O(1) RAM).

    Sorting occurs in the existing SQLite frequency table, NOT an in-memory
    sorted list of records. A single cursor walk obtains every selected rank.
    """
    predicate = " AND value>0" if positive_only else ""
    n = db.execute(
        "SELECT COALESCE(SUM(n),0) FROM numeric_values WHERE metric=?" + predicate,
        (metric,)).fetchone()[0]
    if n == 0:
        return 0, {q: None for q in probabilities}
    targets = {q: ((n - 1) * q, math.floor((n - 1) * q),
                   math.ceil((n - 1) * q)) for q in probabilities}
    ranks = {r for _, lo, hi in targets.values() for r in (lo, hi)}
    seen = {}
    cumulative = 0
    cursor = db.execute(
        "SELECT value,n FROM numeric_values WHERE metric=?" + predicate +
        " ORDER BY value", (metric,))
    for value, freq in cursor:
        for rank in ranks:
            if rank not in seen and cumulative <= rank < cumulative + freq:
                seen[rank] = transform(value) if transform else value
        cumulative += freq
        if len(seen) == len(ranks):
            break
    if len(seen) != len(ranks):
        raise AssertionError("Q15 weighted rank resolution failed")
    return n, {q: seen[lo] + (position - lo) * (seen[hi] - seen[lo])
               for q, (position, lo, hi) in targets.items()}


def q15_review_numeric_outliers(db, info):
    """Review helpful_vote: raw integer >=0, within exact-float capacity.

    Missing, malformed, negative, fractional, integral-looking floats and very
    large valid integers do NOT enter the statistical reference distribution.
    Q12 is the source of truth about how many review rows are domain-valid.
    """
    metric = "reviews.helpful_vote"
    n, quantiles = q15_weighted_quantiles(db, metric, (0.95, 0.99))
    positive_n, log_q = q15_weighted_quantiles(
        db, metric, (0.25, 0.75), positive_only=True, transform=math.log1p)
    zero_n = n - positive_n
    if zero_n < 0:
        raise AssertionError("Q15 negative number of zero votes")
    q12_valid = info["q12_review_value_validity"]["fields"]["helpful_vote"]["valid_rows"]
    skipped_large = info["rule_counts"].get(
        "helpful_vote_too_large_for_numeric_summary", 0)
    if n + skipped_large != q12_valid:
        raise AssertionError("Q15 valid votes do not match Q12 and histogram exclusions")

    p99 = quantiles[0.99]
    above_p99 = (db.execute(
        "SELECT COALESCE(SUM(n),0) FROM numeric_values "
        "WHERE metric=? AND value>?", (metric, p99)).fetchone()[0]
        if n else None)
    result = dict(
        metric=metric, reference_unit="raw parsed review objects, before deduplication",
        denominator_parsed_review_objects=info["object_records"],
        valid_nonnegative_raw_integer_rows=q12_valid,
        screened_valid_integer_rows=n,
        valid_large_integer_rows_not_screened=skipped_large,
        zero_vote_rows=zero_n, positive_vote_rows=positive_n,
        zero_percent_of_screened=100 * zero_n/n if n else None,
        p95_all_screened=quantiles[0.95], p99_all_screened=p99,
        strictly_above_p99_rows=above_p99,
        strictly_above_p99_percent=100*above_p99/n if n else None,
        positive_log_q1=log_q[0.25], positive_log_q3=log_q[0.75],
        positive_log_upper_fence=None, positive_upper_fence_raw=None,
        positive_upper_outlier_rows=None,
        positive_upper_outlier_percent_of_positive=None,
        positive_upper_outlier_percent_of_screened=None,
        status="unclassified", method="Positive-only log1p Tukey upper fence: Q3 + 3*IQR",
        quantile_method="Frequency-weighted linear interpolation at (n-1)*q; SQLite frequency table",
        initial_decision=("Retain all values; investigate unusually high positive vote counts. "
                          "Never mark them as impossible or remove reviews from the time sequence."),
        caveat=("Descriptive, global within-file screening only. Statistical flags are not bad-data "
                "proof. Counts >2**53 are separately flagged and omitted from float-based diagnostics. "
                "Never fit such thresholds before train/test splitting for predictive preprocessing."))
    if positive_n == 0:
        result["status"] = "no_positive_reference_values"
    elif positive_n < 4:
        result["status"] = "fewer_than_four_positive_values"
    else:
        q1, q3 = log_q[0.25], log_q[0.75]
        if q3 <= q1:
            result["status"] = "zero_positive_log_iqr_not_classified"
        else:
            upper = q3 + 3 * (q3 - q1)
            if not math.isfinite(upper):
                result["status"] = "nonfinite_log_upper_fence_not_classified"
            else:
                result["positive_log_upper_fence"] = upper
                result["positive_upper_fence_raw"] = math.expm1(upper)
                flagged = 0
                for value, freq in db.execute(
                    "SELECT value,n FROM numeric_values WHERE metric=? AND value>0 "
                    "ORDER BY value", (metric,)):
                    lv = math.log1p(value)
                    # Avoid classifying a mathematically equal boundary as an
                    # outlier because of a few floating-point rounding ULPs.
                    if lv - upper > 8 * max(math.ulp(lv), math.ulp(upper)):
                        flagged += freq
                result.update(status="computed", positive_upper_outlier_rows=flagged,
                    positive_upper_outlier_percent_of_positive=100*flagged/positive_n,
                    positive_upper_outlier_percent_of_screened=100*flagged/n)
    return result



# Q16: metadata numeric outliers. Reuses the SAME disk-backed frequency
# table and exact-frequency quantiles as Q15. Domain validity is established
# by Q14, not by any statistical fence. Results are descriptive only.
def q16_metadata_numeric_outliers(db, info):
    """Two independent distributions, one per field: price and rating count.

    Prices: raw two-sided Tukey 1.5*IQR is the primary screen; a positive-only
    log1p 3*IQR upper-tail screen is an additional sensitivity analysis.
    Rating counts: positive-only log1p 3*IQR is primary; raw Tukey is a baseline.
    All counts and excluded records are made explicit in each denominator.
    """
    q14 = info['q14_metadata_value_validity']['fields']
    results = {}
    for field in ('price', 'rating_number'):
        metric = 'metadata.' + field
        n, quantiles = q15_weighted_quantiles(db, metric, (0.25, 0.75, 0.95, 0.99))
        positive_n, logs = q15_weighted_quantiles(
            db, metric, (0.25, 0.75), positive_only=True, transform=math.log1p)
        zero_n = n - positive_n
        if zero_n < 0:
            raise AssertionError('Q16 negative number of zero values: '+field)
        valid = q14[field]['valid_under_A_domain_rows']
        skipped = info['rule_counts'].get(field+'_too_large_for_numeric_summary', 0)
        if n + skipped != valid:
            raise AssertionError('Q16 histogram/Q14 valid count mismatch: '+field)
        if field == 'price' and zero_n != q14[field]['warning_zero_rows']:
            raise AssertionError('Q16 zero price disagrees with Q14 warnings')
        if field == 'rating_number':
            zero_q14 = next(s['count'] for s in q14[field]['statuses']
                            if s['status']=='valid_zero_integer')
            if zero_n != zero_q14:
                raise AssertionError('Q16 zero rating count disagrees with Q14')
        p99 = quantiles[0.99]
        above99 = (db.execute('SELECT COALESCE(SUM(n),0) FROM numeric_values '
                 'WHERE metric=? AND value>?', (metric,p99)).fetchone()[0] if n else None)
        raw_status = 'insufficient_reference_values'
        raw_lower = raw_upper = raw_flagged = None
        if n >= 4:
            q1, q3 = quantiles[0.25], quantiles[0.75]
            if q3 <= q1:
                raw_status = 'zero_raw_iqr_not_classified'
            else:
                iqr = q3-q1
                lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
                if not (math.isfinite(lower) and math.isfinite(upper)):
                    raw_status='nonfinite_raw_fence_not_classified'
                else:
                    raw_status='computed'; raw_lower,raw_upper=lower,upper
                    raw_flagged = db.execute(
                        'SELECT COALESCE(SUM(n),0) FROM numeric_values '
                        'WHERE metric=? AND (value<? OR value>?)',
                        (metric,lower,upper)).fetchone()[0]
        log_status = 'no_positive_reference_values' if not positive_n else (
                     'fewer_than_four_positive_values' if positive_n<4 else 'unclassified')
        log_upper = log_upper_raw = log_flagged = None
        if positive_n >= 4:
            q1, q3 = logs[0.25], logs[0.75]
            if q3 <= q1:
                log_status='zero_positive_log_iqr_not_classified'
            else:
                upper = q3+3*(q3-q1)
                if not math.isfinite(upper):
                    log_status='nonfinite_log_fence_not_classified'
                else:
                    log_status='computed'; log_upper=upper
                    # The log-space fence remains usable even if its inverse
                    # overflows the representable float range.
                    if upper <= math.log(sys.float_info.max):
                        log_upper_raw=math.expm1(upper)
                    log_flagged=0
                    for value,freq in db.execute(
                            'SELECT value,n FROM numeric_values WHERE metric=? AND value>0 '
                            'ORDER BY value',(metric,)):
                        lv=math.log1p(value)
                        if lv-upper > 8*max(math.ulp(lv),math.ulp(upper)):
                            log_flagged+=freq
        primary='raw_two_sided_1_5_iqr' if field=='price' else 'positive_only_log1p_upper_3_iqr'
        primary_status=raw_status if field=='price' else log_status
        primary_rows=raw_flagged if field=='price' else log_flagged
        results[field]=dict(
            metric=metric,
            reference_unit='parsed metadata objects before exact deduplication; within supplied file',
            denominator_parsed_metadata_objects=info['object_records'],
            domain_valid_rows=valid, valid_large_integer_rows_not_screened=skipped,
            screened_valid_rows=n, zero_rows=zero_n, positive_rows=positive_n,
            zero_percent_of_screened=(100*zero_n/n if n else None),
            positive_percent_of_screened=(100*positive_n/n if n else None),
            p95_all_screened=quantiles[0.95],p99_all_screened=p99,
            strictly_above_p99_rows=above99,
            strictly_above_p99_percent=(100*above99/n if n else None),
            raw_q1=quantiles[0.25],raw_q3=quantiles[0.75],
            raw_lower_fence=raw_lower,raw_upper_fence=raw_upper,
            raw_two_sided_outlier_rows=raw_flagged,
            raw_outlier_percent_of_screened=(100*raw_flagged/n if raw_flagged is not None else None),
            raw_status=raw_status,
            positive_log_q1=logs[0.25],positive_log_q3=logs[0.75],
            positive_log_upper_fence=log_upper,positive_upper_fence_raw=log_upper_raw,
            positive_log_upper_outlier_rows=log_flagged,
            positive_log_outlier_percent_of_positive=(100*log_flagged/positive_n if log_flagged is not None else None),
            positive_log_outlier_percent_of_screened=(100*log_flagged/n if log_flagged is not None else None),
            positive_log_status=log_status,
            primary_screen=primary,primary_status=primary_status,
            primary_flagged_rows=primary_rows,
            primary_percent_of_screened=(100*primary_rows/n if primary_rows is not None else None),
            quantile_method='SQLite frequency weighted linear interpolation at (n-1)*q; no row materialization',
            initial_decision=('Retain product and value; inspect price scale, currency, category and source context. '
                'Zero price is a Q14 warning, not automatically an outlier or free product.' if field=='price'
                else 'Retain product and rating count; investigate very high counts as plausible popularity, not corruption.'),
            caveat=('Diagnostic only, no automatic fixing/deletion. Raw and log fences may disagree; neither is proof of error. '
                'The reference contains raw records including exact duplicates; category-specific screens are deferred '
                'to Q17. Metadata aggregates and current price are not valid historical prediction features by default. '
                'Invalid/missing values and exact integers beyond 2**53 are excluded from float-based reference. '
                'Never reuse global thresholds as preprocessing trained before the predictive split.'))
    return results



# Q18: Quality errors and statistical extremes are DIFFERENT concepts.
# Synthesize Q11-Q16 + Q13 without a new raw-file pass. Exclusive source-field
# categories sum to parsed objects; an outlier is counted only among domain-
# accepted values, not as an additional invalid row. Some warnings are valid
# (zero price), while other source conventions remain unresolved (mean=0).
def q18_outlier_vs_invalid(summary):
    files = summary['files']
    q15 = summary['q15_review_numeric_outliers']
    q16 = summary['q16_metadata_numeric_outliers']
    q11 = files['reviews']['q11_review_rating_validity']
    q12 = files['reviews']['q12_review_value_validity']['fields']['helpful_vote']
    q13 = files['reviews']['q13_review_timestamp_validity']
    q14 = files['metadata']['q14_metadata_value_validity']['fields']
    def rowcounts(items, status_key='status'):
        return {r[status_key]:r['count'] for r in items}
    def total(d, *keys):
        return sum(d[k] for k in keys)
    lines = []
    def put(role, field, *, missing, wrong, invalid, accepted, warning=0,
            warning_accepted=False, screening=None, method=None, status=None,
            screened=None, valid_unscreened=None, flagged=None, precision_note=None,
            extra_warning=None, warning_basis=None):
        n = files[role]['object_records']
        categories = (missing,wrong,invalid,accepted,warning)
        if any(type(x) is not int or x<0 for x in categories):
            raise AssertionError('Q18 invalid exclusive partition count: '+role+'.'+field)
        if sum(categories)!=n:
            raise AssertionError('Q18 exclusive categories do not sum to objects: '+role+'.'+field)
        domain_usable = accepted + (warning if warning_accepted else 0)
        if screening is None:
            if any(x is not None for x in (screened,valid_unscreened,flagged)):
                raise AssertionError('Q18 no screening but populated screen statistics')
            if method is None: method='not_applicable_bounded_or_temporal_field'
            status='not_applicable'
        else:
            if not all(type(x) is int and x>=0 for x in (screened,valid_unscreened)):
                raise AssertionError('Q18 invalid screened/unscreened values')
            if screened+valid_unscreened != domain_usable:
                raise AssertionError('Q18 statistical reference/domain mismatch: '+role+'.'+field)
            if status == 'computed':
                if type(flagged) is not int or not 0<=flagged<=screened:
                    raise AssertionError('Q18 statistical flagged count out of range')
            elif flagged is not None:
                raise AssertionError('Q18 undefined statistical screen cannot have a flagged count')
        return dict(file_role=role, field=field, scope=summary['scope'],
            input_path=files[role]['path'], denominator_parsed_object_rows=n,
            missing_or_empty_rows=missing, wrong_nonempty_type_rows=wrong,
            numeric_domain_or_technical_invalid_rows=invalid,
            domain_usable_without_warning_rows=accepted,
            source_warning_only_rows=warning,
            source_warning_counted_as_domain_usable=warning_accepted,
            domain_usable_rows=domain_usable,
            domain_unusable_rows=n-domain_usable,
            exclusive_partition_total=sum(categories),
            warning_basis=warning_basis or 'none',
            supplementary_overlapping_warning_rows=extra_warning,
            statistical_method=method,
            statistical_status=status,
            screened_domain_usable_rows=screened,
            valid_rows_not_screened_for_numeric_precision=valid_unscreened,
            statistical_flagged_valid_rows=flagged,
            statistical_flagged_percent_of_screened=(100*flagged/screened if flagged is not None and screened else None),
            statistical_flagged_percent_of_all_objects=(100*flagged/n if flagged is not None and n else None),
            precision_or_scope_note=precision_note or 'Raw records included; not a deduplicated population.',
            decision=('Inspect true invalid/missing values by Q11-Q14/Q13 rules. '
                      'Preserve valid but statistically flagged values for context review. '
                      'Never remove rows or interpret statistical extremes as impossible values.'))

    rc=rowcounts(q11['results'])
    lines.append(put('reviews','rating',missing=rc['missing_or_empty'],
        wrong=rc['wrong_nonnull_type'],
        invalid=total(rc,'numeric_outside_1_to_5','numeric_fractional_within_1_to_5'),
        accepted=rc['valid_integral_star'],
        precision_note='Finite discrete 1..5 stars; domain validation only, no meaningful numerical outlier fence.'))
    vc=rowcounts(q12['rows'])
    lines.append(put('reviews','helpful_vote',missing=vc['missing_or_empty'],
        wrong=total(vc,'fractional_float','integral_float_wrong_type','other_wrong_nonnull_type'),
        invalid=vc['negative_integer'],
        accepted=total(vc,'valid_zero_integer','valid_positive_integer'),
        screening='screened', method=q15['method'],status=q15['status'],
        screened=q15['screened_valid_integer_rows'],
        valid_unscreened=q15['valid_large_integer_rows_not_screened'],
        flagged=q15['positive_upper_outlier_rows'],
        precision_note='Statistical extreme concerns only valid nonnegative raw integers <= 2**53; skipped huge valid integers are reported.' ))
    tc=rowcounts(q13['statuses'])
    lines.append(put('reviews','timestamp',missing=tc['missing_or_empty'],
        wrong=tc['wrong_nonnull_type'],invalid=tc['unrepresentable_in_configured_unit'],
        accepted=tc['representable_inside_reference_window'],
        warning=total(tc,'representable_before_reference_window','representable_at_or_after_reference_end'),
        warning_accepted=True,extra_warning=q13['representable_after_audit_as_of_rows'],
        warning_basis='Technically representable outside configured reference interval; future-as-of flag may overlap.',
        precision_note='Representable outside window is not proven invalid; not a statistical outlier analysis.'))
    for field in ('price','rating_number','average_rating'):
        c=rowcounts(q14[field]['statuses'])
        if field=='price':
            kwargs=dict(missing=c['missing_or_empty'], wrong=c['wrong_nonnull_type'],
                invalid=c['negative_number'],accepted=c['positive_number'],
                warning=c['zero_requires_inspection'],warning_accepted=True,
                warning_basis='Zero price may be a genuine free price or a source sentinel; preserve pending provenance check.')
        elif field=='rating_number':
            kwargs=dict(missing=c['missing_or_empty'],
                wrong=total(c,'fractional_float','integral_float_wrong_type','other_wrong_nonnull_type'),
                invalid=c['negative_integer'],
                accepted=total(c,'valid_zero_integer','valid_positive_integer'))
        else:
            kwargs=dict(missing=c['missing_or_empty'],wrong=c['wrong_nonnull_type'],
                invalid=total(c,'negative_number','numeric_above_five'),
                accepted=c['numeric_within_1_to_5'],
                warning=c['zero_possible_source_sentinel'],warning_accepted=False,
                warning_basis='Zero mean rating: source sentinel hypothesis only; not accepted in Q14 adopted 1..5 domain.')
        if field in ('price','rating_number'):
            stat=q16[field]
            kwargs.update(screening='screened',method=stat['primary_screen'],
                status=stat['primary_status'],
                screened=stat['screened_valid_rows'],
                valid_unscreened=stat['valid_large_integer_rows_not_screened'],
                flagged=stat['primary_flagged_rows'],
                precision_note=('Q16 primary statistical screen only; alternate raw/log sensitivity screen '
                                'remains in Q16. Huge exact integer values are excluded only from float diagnostics.'))
        else:
            kwargs.update(precision_note='Snapshot mean rating is bounded; investigate invalid ranges and sentinel conventions, not a statistical tail fence.')
        lines.append(put('metadata',field,**kwargs))
    return dict(rows=lines,unit='parsed JSON object row, within each role/file before deduplication',
                interpretation=(
                    'Each field has exactly one exclusive status per object: missing/empty; wrong type; '
                    'numeric/technical invalid; domain-usable without warning; or source warning. '
                    'Source warnings are not proof of errors; zero price is provisionally usable, '
                    'zero average_rating is not under Q14 baseline, and representable unusual dates remain technically usable. '
                    'Statistical outliers are a subset of screened domain-usable values, not another category '
                    'to add to invalid totals. A null flagged value means screening was unavailable, never zero. '
                    'Percent denominators and large-number exclusions are explicit. '
                    'No inference about record-level union of bad fields is made from marginal counts. '
                    'No cleaning, deletion, or predictive selection is performed.'))

def numeric_summary(db, metric):
    n = db.execute("SELECT COALESCE(SUM(n),0) FROM numeric_values WHERE metric=?", (metric,)).fetchone()[0]
    if not n:
        return {"n": 0, "outlier_status": "no_observations"}
    targets = {q: ((n-1)*q, math.floor((n-1)*q), math.ceil((n-1)*q)) for q in (0.25, 0.5, 0.75)}
    wanted = {rank for _, lo, hi in targets.values() for rank in (lo, hi)}
    found = {}
    cumulative, mean = 0, 0.0
    minimum = maximum = None
    for value, count in db.execute("SELECT value,n FROM numeric_values WHERE metric=? ORDER BY value", (metric,)):
        if minimum is None:
            minimum = value
        maximum = value
        for rank in wanted:
            if cumulative <= rank < cumulative+count:
                found[rank] = value
        mean += (value-mean) * (count/(cumulative+count))
        cumulative += count
    quantiles = {q: found[lo] + (position-lo)*(found[hi]-found[lo]) for q, (position,lo,hi) in targets.items()}
    q1, median, q3 = (quantiles[q] for q in (0.25,0.5,0.75))
    result = dict(n=n, minimum=minimum, maximum=maximum, mean=mean, q1=q1, median=median, q3=q3,
                  quantile_method="Linear interpolation at (n-1)*q over the complete observed frequency table.")
    if q3 == q1:
        result.update(outlier_status="zero_iqr_not_classified", lower_fence=None, upper_fence=None, flagged_rows=None)
    else:
        lower, upper = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
        flagged = db.execute("SELECT COALESCE(SUM(n),0) FROM numeric_values WHERE metric=? AND (value<? OR value>?)", (metric,lower,upper)).fetchone()[0]
        result.update(outlier_status="diagnostic_only", lower_fence=lower if math.isfinite(lower) else None,
                      upper_fence=upper if math.isfinite(upper) else None, flagged_rows=flagged,
                      flagged_share=flagged/n)
    return result




# Q19: Explicit, testable rate/denominator ledger across the existing audit.
# This is a REPORT-ONLY consolidation: stream existing small CSV summaries,
# never rescan raw review/metadata files and never change Q01-Q18 counters.
# Every existing *rate-bearing* CSV column is mapped to an explicit numerator,
# denominator and unit. Percent (0..100) and share (0..1) are NOT conflated.
Q19_RATE_SPEC = {
    # tuple: (numerator column, denominator column, rate column, denominator unit, relation)
    'missingness.csv': [('count','object_records','share','parsed_objects','exclusive_by_field')],
    'modality_availability.csv': [('available_rows','object_records','share','parsed_objects','overlapping_flags')],
    'q01_nested_content.csv': [
        ('usable_content_rows','object_records','usable_share_of_all_objects','parsed_objects','subset'),
        ('usable_content_rows','raw_present_nonempty_rows','usable_share_of_raw_nonempty','raw_nonempty_field_values','subset')],
    'q03_review_readiness.csv': [('rows','denominator_review_objects','percent_of_review_objects','parsed_review_objects','mixed_exclusive_and_overlapping_by_section')],
    'q06_identifier_multiplicity.csv': [
        ('rows_in_repeated_id_groups','parsed_object_denominator','repeated_rows_percent_of_parsed_objects','parsed_objects','overlapping_across_ids'),
        ('rows_in_repeated_id_groups','rows_with_usable_raw_id','repeated_rows_percent_of_valid_ids','rows_with_usable_raw_identifier','subset')],
    'q07_review_type_compatibility.csv': [
        ('incompatible_nonnull_rows','present_nonnull_rows','incompatible_percent_of_present_nonnull','non_null_field_occurrences','subset'),
        ('incompatible_nonnull_rows','parsed_review_objects','incompatible_percent_of_all_objects','parsed_review_objects','subset')],
    'q08_metadata_type_compatibility.csv': [
        ('incompatible_nonnull_rows','present_nonnull_rows','incompatible_percent_of_present_nonnull','non_null_field_occurrences','subset'),
        ('incompatible_nonnull_rows','parsed_metadata_objects','incompatible_percent_of_all_objects','parsed_metadata_objects','subset')],
    'q08_metadata_nested_types.csv': [
        ('incompatible_nonnull_child_occurrences','present_nonnull_child_occurrences','incompatible_percent_of_nonnull_child_occurrences','non_null_child_occurrences','subset'),
        ('affected_metadata_records','valid_parent_containers','affected_percent_of_valid_parent_containers','metadata_records_with_valid_parent_container','subset')],
    'q09_format_flags.csv': [('count','denominator','percent','specified_in_denominator_unit_column','potentially_overlapping_format_flags')],
    'q10_cross_field_consistency.csv': [('count','denominator_parsed_objects','percent_of_parsed_objects','parsed_objects','exclusive_per_relationship')],
    'q11_review_rating_validity.csv': [('count','denominator_parsed_review_objects','percent_of_parsed_review_objects','parsed_review_objects','exclusive')],
    'q11_review_star_distribution.csv': [
        ('count','denominator_parsed_review_objects','percent_of_all_parsed','parsed_review_objects','exclusive_valid_stars'),
        ('count','denominator_valid_ratings','percent_of_valid_ratings','valid_rated_review_objects','exclusive_valid_stars')],
    'q12_review_value_validity.csv': [('count','denominator_parsed_review_objects','percent_of_parsed_reviews','parsed_review_objects','exclusive_per_field')],
    'q12_review_numeric_flags.csv': [('count','denominator_parsed_review_objects','percent_of_parsed_reviews','parsed_review_objects','overlapping_flags')],
    'q13_review_timestamp_validity.csv': [('count','denominator_parsed_review_objects','percent_of_parsed_reviews','parsed_review_objects','exclusive')],
    'q13_timestamp_suspicions.csv': [('count','denominator_parsed_review_objects','percent_of_parsed_reviews','parsed_review_objects','overlapping_warning')],
    'q14_metadata_value_validity.csv': [('count','denominator_parsed_metadata_objects','percent_of_parsed_metadata','parsed_metadata_objects','exclusive_per_field')],
    'q14_metadata_numeric_flags.csv': [('count','denominator_parsed_metadata_objects','percent_of_parsed_metadata','parsed_metadata_objects','overlapping_flags')],
    'q15_review_numeric_outliers.csv': [
        ('zero_vote_rows','screened_valid_integer_rows','zero_percent_of_screened','screened_valid_votes','subset'),
        ('strictly_above_p99_rows','screened_valid_integer_rows','strictly_above_p99_percent','screened_valid_votes','subset_when_computed'),
        ('positive_upper_outlier_rows','positive_vote_rows','positive_upper_outlier_percent_of_positive','screened_positive_votes','subset_when_computed'),
        ('positive_upper_outlier_rows','screened_valid_integer_rows','positive_upper_outlier_percent_of_screened','screened_valid_votes','subset_when_computed')],
    'q16_metadata_numeric_outliers.csv': [
        ('zero_rows','screened_valid_rows','zero_percent_of_screened','screened_valid_metadata_values','subset'),
        ('positive_rows','screened_valid_rows','positive_percent_of_screened','screened_valid_metadata_values','subset'),
        ('strictly_above_p99_rows','screened_valid_rows','strictly_above_p99_percent','screened_valid_metadata_values','subset_when_computed'),
        ('raw_two_sided_outlier_rows','screened_valid_rows','raw_outlier_percent_of_screened','screened_valid_metadata_values','subset_when_computed'),
        ('positive_log_upper_outlier_rows','positive_rows','positive_log_outlier_percent_of_positive','screened_positive_metadata_values','subset_when_computed'),
        ('positive_log_upper_outlier_rows','screened_valid_rows','positive_log_outlier_percent_of_screened','screened_valid_metadata_values','subset_when_computed'),
        ('primary_flagged_rows','screened_valid_rows','primary_percent_of_screened','screened_valid_metadata_values','subset_when_computed')],
    'q18_outlier_vs_invalid.csv': [
        ('statistical_flagged_valid_rows','screened_domain_usable_rows','statistical_flagged_percent_of_screened','screened_domain_usable_rows','subset_when_computed'),
        ('statistical_flagged_valid_rows','denominator_parsed_object_rows','statistical_flagged_percent_of_all_objects','parsed_objects','subset_when_computed')],
    'quality_rules.csv': [('affected_rows','object_records','share','parsed_objects','overlapping_rules')],
}
# Q02 has multiple status and composite percentage/share columns.
Q19_RATE_SPEC['q02_field_missingness.csv'] = (
    [(name+'_rows','object_records_denominator',name+'_percent','parsed_objects','exclusive_field_state')
     for name in MISSING_STATES] +
    [(name+'_rows','object_records_denominator',name+'_share','parsed_objects','composite_or_subset')
     for name in ('raw_missing','raw_missing_or_empty','raw_present_nonempty')] +
    [(name+'_rows','object_records_denominator',name+'_percent','parsed_objects','composite_or_subset')
     for name in ('raw_missing','raw_missing_or_empty','raw_present_nonempty')]
)

# Q20 extends Q19's fail-closed rate registry; no older rate is modified.
for _q20_filename in ('q20_issue_families.csv','q20_issue_pairs.csv',
                      'q20_issue_signatures.csv','q20_issue_union.csv'):
    Q19_RATE_SPEC[_q20_filename] = [
        ('count','denominator_parsed_objects','percent_of_parsed_objects',
         'parsed_objects','overlapping_families_or_exclusive_signature_as_labeled')]

# Q22 extends the Q19 denominator contracts without altering prior rates.
Q19_RATE_SPEC['q22_text_presence.csv'] = [
    ('count','denominator_parsed_review_objects','percent_of_parsed_review_objects',
     'parsed_review_objects','exclusive_body_text_presence_states')]
Q19_RATE_SPEC['q22_text_length_buckets.csv'] = [
    ('count','denominator_string_values','percent_of_string_values',
     'string_valued_body_texts','exclusive_length_buckets_per_metric'),
    ('count','denominator_parsed_review_objects','percent_of_parsed_review_objects',
     'parsed_review_objects','subset_of_parsed_reviews')]

# Q23 sensitivity rates: reference strings differ by screen, and all-object rates
# use the same parsed-review-object count. No additional raw scan.
Q19_RATE_SPEC['q23_short_long_thresholds.csv'] = [
    ('count','denominator_reference_strings','percent_of_reference_strings',
     'short_nonblank_or_all_string_bodies_as_labeled','subset_of_reference_strings'),
    ('count','denominator_parsed_review_objects','percent_of_parsed_review_objects',
     'parsed_review_objects','subset_of_parsed_reviews')]

# Q24: script names are descriptive proxies, and script-presence rows overlap.
Q19_RATE_SPEC['q24_unicode_script_profile.csv'] = [
    ('letter_occurrences','denominator_all_letters','percent_of_all_letters',
     'unicode_alphabetic_codepoints','exclusive_letter_script_partition'),
    ('reviews_containing_script','denominator_string_bodies','percent_of_string_bodies',
     'string_valued_review_bodies','overlapping_script_presence')]
Q19_RATE_SPEC['q24_script_patterns.csv'] = [
    ('count','denominator_string_bodies','percent_of_string_bodies',
     'string_valued_review_bodies','exclusive_script_pattern')]
Q19_RATE_SPEC['q24_unicode_flags.csv'] = [
    ('count','denominator_string_bodies','percent_of_string_bodies',
     'string_valued_review_bodies','overlapping_unicode_warnings'),
    ('count','denominator_parsed_review_objects','percent_of_parsed_review_objects',
     'parsed_review_objects','overlapping_unicode_warnings')]

# Q25: per-star contingency rows partition each rating stratum;
# feature flags can overlap and use only nonblank bodies for each star.
Q19_RATE_SPEC['q25_rating_text_presence.csv'] = [
    ('count','denominator_rating_rows','percent_of_rating_rows','valid_rating_rows_in_star','exclusive_per_star'),
    ('count','denominator_all_valid_rating_rows','percent_of_all_valid_ratings','all_valid_star_reviews','exclusive_across_star_and_text_state')]
Q19_RATE_SPEC['q25_rating_text_features.csv'] = [
    ('count','denominator_nonblank_bodies_in_star','percent_of_nonblank_bodies_in_star','nonblank_valid_star_review_bodies','overlapping_text_features')]
Q19_RATE_SPEC['q25_rating_text_length.csv'] = [
    ('nonblank_string_bodies','valid_rating_rows','nonblank_body_percent_of_rating_rows','valid_rating_rows_in_star','subset')]

Q19_COLUMNS = ('scope','source_file','source_row','file_role','measure','numerator_field',
               'numerator','denominator_field','denominator','denominator_unit',
               'reported_rate','computed_rate','rate_scale','assessment','counting_relation')


def q19_validate_rate(num_text, den_text, rate_text, scale, label):
    """Check an existing reported rate against exact integer census counts.

    Missing screen statistics and zero denominators must have undefined rates.
    Never interpret them as zero. Reject inf/nan, negative or oversized counts.
    """
    def as_count(v):
        if v is None or str(v).strip() == '':
            return None
        value = int(str(v))
        if str(value) != str(v).strip() or value < 0:
            raise AssertionError('Q19 noncanonical/nonnegative count: '+label)
        return value
    try:
        numerator, denominator = as_count(num_text), as_count(den_text)
    except (TypeError, ValueError) as exc:
        raise AssertionError('Q19 invalid integer census: '+label) from exc
    if denominator is None:
        raise AssertionError('Q19 missing denominator: '+label)
    if numerator is not None and numerator > denominator:
        raise AssertionError('Q19 numerator exceeds denominator: '+label)
    defined = numerator is not None and denominator > 0
    expected = numerator * scale / denominator if defined else None
    if rate_text is None or str(rate_text).strip() == '':
        if defined:
            raise AssertionError('Q19 missing defined rate: '+label)
        return numerator, denominator, None, 'undefined_zero_denominator' if denominator == 0 else 'undefined_uncomputed'
    try:
        observed = float(rate_text)
    except (ValueError, TypeError) as exc:
        raise AssertionError('Q19 nonnumeric rate: '+label) from exc
    if not math.isfinite(observed) or not defined:
        raise AssertionError('Q19 reported rate has undefined denominator: '+label)
    if not math.isclose(expected, observed, rel_tol=1e-10, abs_tol=1e-8):
        raise AssertionError('Q19 rate/count/denominator mismatch: '+label)
    return numerator, denominator, expected, 'verified'


def q19_rate_denominator_audit(folder, summary):
    """Reconcile all currently published CSV ratio columns, without raw scans.

    A schema coverage assertion makes adding a new rate column a fail-closed
    change: Q19 must be told its correct numerator and denominator explicitly.
    A percentage's denominator is a typed population, never the nearest total.
    """
    rows = []
    counts = Counter()
    scope = summary['scope']
    for role, info in summary['files'].items():
        physical = info['physical_lines']
        partition = {k: info[k] for k in ('object_records','parse_errors','blank_lines','nonobject_lines')}
        if any(type(v) is not int or v < 0 for v in partition.values()) or sum(partition.values()) != physical:
            raise AssertionError('Q19 physical-line partition mismatch: '+role)
        for name, value in partition.items():
            rows.append((scope,'quality_summary.json','file_census',role,name,name,value,
                         'physical_lines',physical,'scanned_physical_lines',None,
                         100*value/physical if physical else None,'percent_derived',
                         'verified_derived' if physical else 'undefined_zero_denominator', 'exclusive_file_line_census'))
            counts['file_census_metrics'] += 1
    # Fail closed if a future version starts reporting a new rate-bearing
    # CSV without a consciously chosen Q19 denominator contract.
    for candidate in folder.glob('*.csv'):
        if candidate.name.startswith('q19_'):
            continue
        with candidate.open('r',newline='',encoding='utf-8-sig') as f:
            header = next(csv.reader(f), [])
        if any('percent' in name or 'share' in name for name in header) and candidate.name not in Q19_RATE_SPEC:
            raise AssertionError('Q19 missing registered rate file: '+candidate.name)
    for filename, specs in Q19_RATE_SPEC.items():
        path = folder/filename
        if not path.is_file():
            raise AssertionError('Q19 missing previous CSV rate source: '+filename)
        with path.open('r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise AssertionError('Q19 empty CSV without header: '+filename)
            rate_names = {h for h in reader.fieldnames if 'percent' in h or 'share' in h}
            expected_names = {spec[2] for spec in specs}
            if rate_names != expected_names:
                raise AssertionError('Q19 undocumented rate columns: '+filename+' '+str(sorted(rate_names ^ expected_names)))
            for num, den, rate, _, _ in specs:
                if num not in reader.fieldnames or den not in reader.fieldnames:
                    raise AssertionError('Q19 missing census column in '+filename+': '+num+' / '+den)
            for index, row in enumerate(reader, 2):
                if None in row:
                    raise AssertionError('Q19 malformed extra CSV columns: '+filename)
                role = row.get('file_role') or ('reviews' if filename.startswith(('q03_','q07_','q11_','q12_','q13_','q15_')) else
                                            'metadata' if filename.startswith(('q08_','q14_','q16_')) else None)
                if role is None and filename=='q18_outlier_vs_invalid.csv':
                    role = row['file_role']
                measure = row.get('field') or row.get('metric') or row.get('relationship') or row.get('rule') or row.get('availability_definition') or row.get('path') or 'dataset'
                sub = row.get('status') or row.get('outcome') or row.get('state') or row.get('numeric_flag') or row.get('star') or row.get('rule')
                if sub is not None: measure += ':'+str(sub)
                for num, den, rate, unit, relation in specs:
                    label = filename+':'+str(index)+':'+rate
                    scale = 1 if 'share' in rate else 100
                    if filename == 'q09_format_flags.csv' and row.get('field') == 'details.keys':
                        # Q09 explicitly withholds a denominator because one
                        # metadata object can have arbitrarily many detail keys.
                        # Inventing one would turn an occurrence count into a
                        # misleading record-level percentage.
                        numerator = int(row[num]); denominator = None
                        if row[rate] not in ('',None) or row[den] not in ('',None):
                            raise AssertionError('Q19 details-key rate must remain undefined: '+label)
                        computed, outcome = None, 'not_reported_no_reference_denominator'
                    elif filename == 'q18_outlier_vs_invalid.csv' and row.get('statistical_status') == 'not_applicable':
                        numerator = None
                        denominator = int(row[den]) if row[den] not in ('',None) else None
                        if row[num] not in ('',None) or row[rate] not in ('',None):
                            raise AssertionError('Q19 no statistical screen but populated rate: '+label)
                        computed, outcome = None, 'not_applicable_no_statistical_screen'
                    else:
                        numerator, denominator, computed, outcome = q19_validate_rate(row[num], row[den], row[rate], scale,label)
                    if role in ('reviews','metadata') and unit in ('parsed_objects','parsed_review_objects','parsed_metadata_objects'):
                        expected_n = summary['files'][role]['object_records']
                        if denominator != expected_n:
                            raise AssertionError('Q19 wrong file-role denominator: '+label)
                    actual_unit = row.get('denominator_unit') if unit == 'specified_in_denominator_unit_column' else unit
                    rows.append((scope,filename,index,role,measure,num,numerator,den,denominator,
                                 actual_unit,float(row[rate]) if row[rate] not in ('',None) else None,
                                 computed,'share_0_to_1' if scale==1 else 'percent_0_to_100',
                                 outcome,relation))
                    counts[outcome] += 1
                    counts['rate_rows'] += 1
        counts['rate_source_files'] += 1
    # Duplicate percentages in the older A/Q04/Q05 summary are output as
    # measure/value rows, not an explicit ratio column; preserve their semantics.
    for role, name in (('reviews','q04_review_exact_duplicates'),('metadata','q05_metadata_exact_duplicates')):
        detail = summary['files'][role][name]
        for rate_key, num_key, den_key in (
            ('extra_duplicate_percent_of_compared', 'extra_duplicate_'+('review' if role=='reviews' else 'metadata')+'_rows','compared_canonical_'+('review' if role=='reviews' else 'metadata')+'_objects'),
            ('extra_duplicate_percent_of_parsed','extra_duplicate_'+('review' if role=='reviews' else 'metadata')+'_rows','denominator_parsed_'+('review' if role=='reviews' else 'metadata')+'_objects'),
            ('rows_in_duplicate_groups_percent_of_compared',('review' if role=='reviews' else 'metadata')+'_rows_in_duplicate_groups','compared_canonical_'+('review' if role=='reviews' else 'metadata')+'_objects'),
        ):
            label=name+':'+rate_key
            numerator, denominator, expected, outcome = q19_validate_rate(detail[num_key],detail[den_key],detail[rate_key],100,label)
            rows.append((scope,name+'.csv','summary_measure',role,rate_key,num_key,numerator,den_key,denominator,
                         'canonical_records' if 'compared' in rate_key else 'parsed_objects',
                         detail[rate_key],expected,'percent_0_to_100',outcome,'duplicate_subset'))
            counts[outcome] += 1; counts['rate_rows'] += 1
    common.write_csv(folder/'q19_rate_denominator_ledger.csv',Q19_COLUMNS,
                     (tuple('' if val is None else val for val in row) for row in rows))
    output = {'scope':scope,'audit_status':'verified','source_rate_csv_files':counts['rate_source_files'],
              'checked_rate_rows':counts['rate_rows'],'verified_rate_rows':counts['verified'],
              'undefined_zero_denominator_rows':counts['undefined_zero_denominator'],
              'undefined_uncomputed_rows':counts['undefined_uncomputed'],
              'unreported_no_reference_denominator_rows':counts['not_reported_no_reference_denominator'],
              'not_applicable_statistical_rate_rows':counts['not_applicable_no_statistical_screen'],
              'file_census_metrics':counts['file_census_metrics'],
              'ledger_rows':len(rows),'output':'q19_rate_denominator_ledger.csv',
              'definitions':{'share_0_to_1':'fraction in [0,1]',
                             'percent_0_to_100':'percentage in [0,100]',
                             'undefined':'blank, never numeric zero',
                             'reconciliation':'numerator / denominator, using exact integer counts; raw duplicate copies included; within-file only'},
              'limitations':['Cross-field flags can overlap; ledger percentages must not be summed unless documented as an exclusive partition.',
                             'Nested-child denominator is a number of nested elements, not a number of products.',
                             'Positive-only statistical screening is a different reference subset from all parsed objects.',
                             'A full supplied-file scan cannot be inferred from a preview/sample scope.',
                             'Q19 does not assert a unique union of affected records; that is Q20.',
                             'Only rate-bearing outputs with explicit numeric numerator and denominator are audited; raw numeric quantiles, counts of unique IDs/products, and Q17 comparison mode are separate.']}
    common.write_json(folder/'q19_rate_integrity.json',output)
    return output


# Q21: Evidence-linked, conservative, read-only initial decision register.
# This is a PLAN for later preprocessing, NOT an operation that cleans input.
# Every count comes from existing Q01-Q20 summaries; overlapping rows NEVER
# form a new union. Unknown observed rules are explicitly left for triage.
Q21_POLICIES = {
    'parse_or_record_structure': (
        'confirmed_technical_readability_issue','quarantine_from_derived_object_table',
        'derived ingest, prior to product grouping',
        'Non-object, ambiguous, or malformed JSON cannot be interpreted as an ordinary review/product record. Preserve raw bytes and line evidence.'),
    'canonicalization_unresolved': (
        'technical_uncertainty','manual_investigation',
        'exact-duplicate verification',
        'A canonicalization failure prevents certainty about record identity; do not claim that the row is unique or automatically remove it.'),
    'review_core_unusable': (
        'task_specific_blocker','exclude_unresolved_from_ordered_rating_sequence',
        'deduplicated, temporally ordered review sequence construction',
        'A valid product parent, integral star 1..5, and interpretable timestamp are essential for the defined first-five/next-ten target. Never delete raw rows.'),
    'metadata_join_key_unusable': (
        'task_specific_join_blocker','retain_metadata_but_do_not_join_on_missing_key',
        'product-to-metadata join, if used',
        'Absent or unusable parent_asin cannot establish a defensible raw-key product join; the core reviews-only target is not affected.'),
    'optional_text_missing_or_unusable': (
        'feature_availability','preserve_review_position_and_missing_text_indicator',
        'text-feature construction after historical cutoff',
        'Empty/nonstring title or text does not remove a chronological rating; decide a text-coverage policy before comparing against the rating-only baseline.'),
    'optional_field_missing': (
        'feature_availability','keep_missingness_explicit',
        'feature availability / descriptive reporting',
        'Optional fields missing are not evidence of a bad record. Do not infer that absence is random or blindly impute.'),
    'noncore_wrong_type': (
        'technical_field_issue','flag_without_coercion',
        'field-level parsing/feature construction',
        'An unexpected raw JSON type does not justify automatically converting numeric strings, booleans or nested data.'),
    'invalid_noncore_value': (
        'technical_field_issue','flag_without_auto_repair',
        'field-level validation and, if used, feature extraction',
        'Impossible or technically invalid field values cannot be used as valid measurements, but need not invalidate a whole review/product.'),
    'verified_exact_duplicate_extra': (
        'verified_record_redundancy','deduplicate_only_in_derived_table',
        'BEFORE ordering reviews and deciding five-plus-ten eligibility',
        'Full canonical payloads have been verified equal. Extra copies should not become extra observations; keep raw bytes, original line references and an audit trail.'),
    'repeated_identifier_not_duplicate': (
        'identity_ambiguity','preserve_records_and_check_join_multiplicity',
        'identifier grouping / S14 join design',
        'Repeated user_id, asin or parent_asin is not proof of a duplicate review. For metadata parent_asin, distinguish identical copies, distinct versions and indeterminate groups.'),
    'nested_wrong_type': (
        'technical_optional_structure_issue','isolate_invalid_nested_elements_if_feature_needed',
        'multimodal feature construction only',
        'A malformed image/list element does not by itself invalidate a review rating or timestamp; null elements are separate from wrong nonnull types.'),
    'format_candidate': (
        'unverified_format_warning','inspect_examples_without_normalizing_ids',
        'schema/source convention review',
        'Whitespace, unusual characters and category spelling candidates are not proof of error. Never casefold product or user IDs or rewrite historical texts by default.'),
    'source_or_cross_field_warning': (
        'source_dependent_warning','verify_source_semantics_before_intervention',
        'source documentation / provenance check',
        'Zero price, zero product average and aggregate mismatches may be source conventions; representable timestamps outside coverage are not automatically invalid.'),
    'statistical_outlier_candidate': (
        'descriptive_statistical_warning','retain_and_investigate_context',
        'exploratory quality report; fit any model preprocessing only on training split',
        'A high valid price or count is not an impossible value. Null statistical classifications mean the fence was not estimable, not zero outliers.'),
    'large_integer_precision': (
        'diagnostic_precision_limitation','keep_exact_integer_and_omit_from_float_screen',
        'numeric diagnostic only',
        'Exact valid integers >2**53 are intentionally excluded from float-based quantiles without declaring them invalid.'),
    'unknown_rule_triage': (
        'unclassified_rule','human_review_required_before_action',
        'quality policy review before derived-data cleaning',
        'This observed rule has no approved Q21 mapping. Do not guess a repair or silently treat it as a confirmed error.'),
}
Q21_COLUMNS = (
    'scope','file_role','issue_source','field','issue','observed_count','count_unit',
    'denominator','denominator_unit','example_physical_lines','policy_id',
    'certainty','initial_action','action_stage','reason','source_reference',
    'counting_caveat','raw_data_modified')

def q21_policy_for(role, source, field, issue):
    if source == 'physical_line_census':
        return 'parse_or_record_structure'
    if source == 'canonicalization':
        return 'canonicalization_unresolved'
    if source == 'exact_duplicate':
        return 'verified_exact_duplicate_extra'
    if source == 'repeated_identifier_group':
        return 'repeated_identifier_not_duplicate'
    if source == 'statistical_screen':
        return 'statistical_outlier_candidate'
    if source == 'q09_format':
        return 'format_candidate'
    if source == 'q10_relation':
        return 'source_or_cross_field_warning'
    if source == 'nested_structure':
        return 'nested_wrong_type'
    if source == 'q03_core':
        return 'review_core_unusable'
    if source == 'raw_missingness':
        if role == 'reviews' and field in ('parent_asin','rating','timestamp'):
            return 'review_core_unusable'
        if role == 'metadata' and field == 'parent_asin':
            return 'metadata_join_key_unusable'
        if role == 'reviews' and field in ('text','title'):
            return 'optional_text_missing_or_unusable'
        return 'optional_field_missing'
    if source == 'rule_count':
        if issue == 'full_record_duplicate_extra':
            return 'verified_exact_duplicate_extra'
        if issue in ('core_sequence_fields_unusable_before_dedup','rating_out_of_domain','timestamp_uninterpretable'):
            return 'review_core_unusable'
        if issue == 'parent_asin_unusable_id':
            return 'review_core_unusable' if role == 'reviews' else 'metadata_join_key_unusable'
        if issue.endswith('_too_large_for_numeric_summary'):
            return 'large_integer_precision'
        if issue.endswith('_wrong_type'):
            if role=='reviews' and field in ('rating','timestamp','parent_asin'):
                return 'review_core_unusable'
            if role=='reviews' and field in ('text','title'):
                return 'optional_text_missing_or_unusable'
            return 'noncore_wrong_type'
        if issue.endswith('_invalid'):
            return 'invalid_noncore_value'
        if issue.endswith(('_nonstring_element','_nonobject_element')):
            return 'nested_wrong_type'
        if issue.endswith(('_possible_missing_marker','_surrounding_whitespace',
                '_unpaired_unicode_surrogate','_replacement_character',
                '_html_like_markup','_control_character')) or issue in (
                'text_very_short_heuristic','text_very_long_heuristic'):
            return 'format_candidate'
        if issue in ('timestamp_outside_declared_coverage','price_zero_requires_inspection'):
            return 'source_or_cross_field_warning'
        if issue in ('user_id_unusable_id','asin_unusable_id'):
            return 'optional_field_missing'
        if issue.startswith('event_'):
            return 'repeated_identifier_not_duplicate'
    return 'unknown_rule_triage'


def q21_initial_decisions(summary):
    """Materialize only existing observations. Not a union or eligibility census."""
    entries = []
    unknown = set()
    def add(role, source, field, issue, count, unit, den, den_unit, reference,
            examples=(), policy=None, caveat='Counts may overlap other rows; do not sum.'):
        if type(count) is not int or count < 0:
            raise AssertionError('Q21 noninteger or negative count: '+str((source,issue,count)))
        if den is not None and (type(den) is not int or den < 0 or count > den):
            raise AssertionError('Q21 invalid denominator: '+str((source,issue,count,den)))
        pid=policy or q21_policy_for(role,source,field,issue)
        if pid not in Q21_POLICIES:
            raise AssertionError('Q21 unknown policy: '+pid)
        if pid == 'unknown_rule_triage':
            unknown.add(role+':'+issue)
        certainty, action, stage, reason=Q21_POLICIES[pid]
        entries.append(dict(scope=summary['scope'],file_role=role,issue_source=source,
            field=field,issue=issue,observed_count=count,count_unit=unit,
            denominator=den,denominator_unit=den_unit,
            example_physical_lines=list(examples),policy_id=pid,certainty=certainty,
            initial_action=action,action_stage=stage,reason=reason,
            source_reference=reference,counting_caveat=caveat,raw_data_modified=False))

    for role, info in summary['files'].items():
        n = info['object_records']
        lines = info['physical_lines']
        for key in ('blank_lines','nonobject_lines','parse_errors'):
            count=info[key]
            if count:
                add(role,'physical_line_census','',key,count,'physical_lines',lines,'scanned_physical_lines',
                    'quality_summary.json:files.'+role+'.'+key,
                    info.get('parse_error_examples',[]) if key=='parse_errors' else ())
        if info['canonicalization_failures']:
            add(role,'canonicalization','', 'canonicalization_failures',
                info['canonicalization_failures'],'parsed_object_records',n,'parsed_object_records',
                'quality_summary.json:files.'+role+'.canonicalization_failures')
        for field, states in sorted(info['missing_states'].items()):
            # Only schema fields; extra free-form keys aren't asserted mandatory.
            if field not in EXPECTED[role]:
                continue
            missing = sum(states[k] for k in ('absent','null','blank_string','empty_array','empty_object'))
            if missing:
                add(role,'raw_missingness',field,'missing_or_empty_raw_field',missing,
                    'parsed_object_records',n,'parsed_object_records',
                    'quality_summary.json:files.'+role+'.missing_states.'+field,
                    caveat='Absent/null/blank/empty components overlap with row-level rules and each other across fields; no row lines available for this aggregate.')
        for rule,count in sorted(info['rule_counts'].items()):
            if not count: continue
            field = rule.split('_wrong_type')[0] if rule.endswith('_wrong_type') else (
                rule.split('_unusable_id')[0] if rule.endswith('_unusable_id') else
                next((name for name in EXPECTED[role] if rule.startswith(name+'_')
                      and (rule.endswith('_invalid') or rule.endswith('_too_large_for_numeric_summary'))), '') )
            add(role,'rule_count',field,rule,count,'rule_occurrences',None,'not_asserted_row_unit',
                'quality_summary.json:files.'+role+'.rule_counts.'+rule,
                info['rule_example_line_numbers'].get(rule,[]),
                caveat='Existing named-rule counter; may overlap other rules; a rule occurrence is not added to a unique affected-row union.')
        exact = info['full_record_duplicates']['duplicate_extra_rows']
        if exact:
            add(role,'exact_duplicate','', 'verified_extra_identical_record_copies',exact,
                'parsed_object_records',n,'parsed_object_records',
                'quality_summary.json:files.'+role+'.full_record_duplicates',
                policy='verified_exact_duplicate_extra',
                caveat='Same phenomenon as full_record_duplicate_extra; do not add the two counts.')
        for item in info['q06_identifier_multiplicity']['identifier_fields']:
            if item['repeated_id_groups']:
                add(role,'repeated_identifier_group',item['field'],'repeated_raw_id_groups',
                    item['repeated_id_groups'],'distinct_raw_identifier_groups',
                    item['distinct_raw_ids'],'distinct_raw_identifier_groups',
                    'quality_summary.json:files.'+role+'.q06_identifier_multiplicity',
                    caveat='Group count, not row count. Repeated raw IDs do not establish duplicate review events.')
        for f in info['q09_format_audit']['format_rules']:
            if f['count']:
                # Details key counts may exceed number of objects: no fabricated denominator.
                den = f['denominator'] if f['denominator_unit'] != 'details occurrences' else None
                add(role,'q09_format',f['field'],f['rule'],f['count'],f['denominator_unit'],
                    den,f['denominator_unit'] if den is not None else 'not_available',
                    'quality_summary.json:files.'+role+'.q09_format_audit',f['example_physical_lines'])
        for item in info['q10_cross_field_consistency']['results']:
            if item['count'] and item['assessment'] not in (
                'descriptive_equal','descriptive_different','descriptive_nonassessable',
                'consistent_positive','consistent_zero','not_assessable'):
                # Explicit source warning; descriptive inequality of two IDs not actionable.
                if 'warning' in item['assessment'] or 'suspic' in item['assessment']:
                    add(role,'q10_relation',item['relationship'],item['outcome'],item['count'],
                        'parsed_object_records',n,'parsed_object_records',
                        'quality_summary.json:files.'+role+'.q10_cross_field_consistency',
                        item['example_physical_lines'])
        if role == 'metadata':
            for p in info['q08_metadata_nested_types']['paths']:
                if p['affected_metadata_records']:
                    add(role,'nested_structure',p['path'],'wrong_nonnull_nested_type',
                        p['affected_metadata_records'],'affected_parent_records',
                        n,'parsed_metadata_objects',
                        'quality_summary.json:files.metadata.q08_metadata_nested_types',
                        p['example_physical_lines'])
            version=info['q06_identifier_multiplicity']['metadata_parent_version_classification']
            for k in ('repeated_parent_groups_with_distinct_verified_versions',
                      'repeated_parent_groups_indeterminate'):
                if version[k]:
                    add(role,'repeated_identifier_group','parent_asin',k,version[k],
                        'repeated_parent_groups',
                        info['q06_identifier_multiplicity']['identifier_fields'][0]['repeated_id_groups'],
                        'repeated_parent_groups',
                        'quality_summary.json:files.metadata.q06_identifier_multiplicity',
                        policy='repeated_identifier_not_duplicate')
    review=summary['files']['reviews']
    core=review['q03_review_readiness']['core_unusable_review_rows_union']
    if core:
        add('reviews','q03_core','parent_asin+rating+timestamp','core_unusable_union_before_dedup',
            core,'parsed_review_objects',review['object_records'],'parsed_review_objects',
            'quality_summary.json:files.reviews.q03_review_readiness',
            caveat='EXACT core-unusable union, BEFORE dedup; do not add to overlapping field/rule counts or interpret as final cohort.')
    # Issue-level decisions that legacy rules alone cannot distinguish.
    q03_text=review['q03_review_readiness']['text_optional']
    for key in ('core_without_body','core_without_any_text'):
        count=q03_text[key]['rows']
        if count:
            add('reviews','raw_missingness','text' if key=='core_without_body' else 'title+text',
                key,count,'parsed_review_objects',review['object_records'],
                'parsed_review_objects',
                'quality_summary.json:files.reviews.q03_review_readiness.text_optional',
                policy='optional_text_missing_or_unusable',
                caveat='Subset of core-usable rows before dedup; two text coverage indicators overlap; never add to core-unusable counts.')
    future=review['q13_review_timestamp_validity']['representable_after_audit_as_of_rows']
    if future:
        add('reviews','q10_relation','timestamp','representable_after_audit_as_of',
            future,'parsed_review_objects',review['object_records'],
            'parsed_review_objects',
            'quality_summary.json:files.reviews.q13_review_timestamp_validity',
            review['q13_review_timestamp_validity']['representable_after_audit_as_of_example_physical_lines'],
            policy='source_or_cross_field_warning',
            caveat='Overlapping plausibility warning, not a technical failure; timestamps remain representable.')
    metadata=summary['files']['metadata']
    for field in ('price','average_rating'):
        flagged=metadata['q14_metadata_value_validity']['fields'][field]['warning_zero_rows']
        if flagged:
            add('metadata','q10_relation',field,'zero_value_source_convention_requires_check',
                flagged,'parsed_metadata_objects',metadata['object_records'],
                'parsed_metadata_objects',
                'quality_summary.json:files.metadata.q14_metadata_value_validity',
                policy='source_or_cross_field_warning',
                caveat='Not evidence of a corrupted product; average_rating zero is separately outside Q14 adopted numeric range. Double counting with legacy value flags is prohibited.')
    for field in ('main_category','categories[]'):
        groups=sum(1 for item in metadata['q09_format_audit']['category_variant_groups']
                   if item['field']==field)
        if groups:
            add('metadata','q09_format',field,'potentially_equivalent_category_label_groups',
                groups,'category_label_groups',None,'not_defined',
                'quality_summary.json:files.metadata.q09_format_audit.category_variant_groups',
                caveat='Grouped label spellings, not per-record count. Never normalize raw IDs or assert group equivalence without review.')
    for role, results in (('reviews',{'helpful_vote':summary['q15_review_numeric_outliers']}),
                           ('metadata',summary['q16_metadata_numeric_outliers'])):
        for field, data in results.items():
            if role=='reviews':
                status=data['status']; flag=data['positive_upper_outlier_rows']
                examined=data['screened_valid_integer_rows']
            else:
                status=data['primary_status']; flag=data['primary_flagged_rows']
                examined=data['screened_valid_rows']
            if flag is None:
                add(role,'statistical_screen',field,'outlier_screen_not_computable:'+str(status),
                    0,'screened_values',examined,'screened_valid_values',
                    'quality_summary.json:'+('q15_review_numeric_outliers' if role=='reviews' else 'q16_metadata_numeric_outliers'),
                    caveat='Zero here means no reported outlier count; the statistical result is UNDEFINED, not zero.')
            elif flag:
                add(role,'statistical_screen',field,'statistically_flagged_valid_values',
                    flag,'screened_valid_values',examined,'screened_valid_values',
                    'quality_summary.json:'+('q15_review_numeric_outliers' if role=='reviews' else 'q16_metadata_numeric_outliers'),
                    caveat='A subset of DOMAIN-VALID values; never add to missing/invalid counts.')

    return dict(scope=summary['scope'],version='Q21',
                catalog={pid:dict(certainty=v[0],initial_action=v[1],action_stage=v[2],
                                  rationale=v[3]) for pid,v in Q21_POLICIES.items()},
                issues=entries,issue_rows=len(entries),unclassified_rule_names=sorted(unknown),
                raw_data_modified=False,
                interpretation=('Initial decisions are task/field-specific and provisional; all evidence counts reuse Q01-Q20. '
                'Issues from different sources and fields overlap, and some count groups rather than rows; NEVER SUM issue counts. '
                'No 5+10 cohort, source-level product-version resolution, text treatment, or feature cleaning has been executed. '
                'Full-run conclusions cannot be drawn from preview scope. '
                'Historical metadata aggregates/price must not be used as predictors without cutoff-time evidence. '
                'The source files and existing audit results are unchanged.'))


def export(folder, summary):
    common.write_json(folder / "quality_summary.json", summary)
    missing_rows, type_rows, rule_rows, availability_rows = [], [], [], []
    q01_nested_rows = []
    q02_field_rows = []
    q03_review_rows = []
    q03_product_rows = []
    q04_review_rows = []
    q04_example_rows = []
    q05_metadata_rows = []
    q05_example_rows = []
    q06_identifier_rows = []
    q06_version_rows = []
    q07_type_rows = []
    q08_type_rows = []
    q08_nested_rows = []
    q09_format_rows = []
    q09_variant_rows = []
    q10_consistency_rows = []
    q11_rating_rows = []
    q11_star_rows = []
    q12_value_rows = []
    q12_overlap_rows = []
    q13_state_rows = []
    q13_future_rows = []
    q14_value_rows = []
    q14_numeric_flags = []
    q18_separation_rows = []
    q02_columns = (
        "scope", "file_role", "input_path", "field", "documented_field",
        "object_records_denominator", "present_rows", "present_nonnull_rows",
        "absent_rows", "null_rows", "blank_string_rows", "empty_array_rows",
        "empty_object_rows", "present_nonempty_rows",
        "raw_missing_rows", "raw_missing_share", "raw_missing_percent",
        "empty_present_rows", "raw_missing_or_empty_rows",
        "raw_missing_or_empty_share", "raw_missing_or_empty_percent",
        "raw_present_nonempty_rows", "raw_present_nonempty_share",
        "raw_present_nonempty_percent",
        "absent_percent", "null_percent", "blank_string_percent",
        "empty_array_percent", "empty_object_percent", "present_nonempty_percent",
    )
    q18_cols = (
        'scope','file_role','input_path','field','denominator_parsed_object_rows',
        'missing_or_empty_rows','wrong_nonempty_type_rows',
        'numeric_domain_or_technical_invalid_rows','domain_usable_without_warning_rows',
        'source_warning_only_rows','source_warning_counted_as_domain_usable',
        'domain_usable_rows','domain_unusable_rows','exclusive_partition_total',
        'warning_basis','supplementary_overlapping_warning_rows',
        'statistical_method','statistical_status','screened_domain_usable_rows',
        'valid_rows_not_screened_for_numeric_precision','statistical_flagged_valid_rows',
        'statistical_flagged_percent_of_screened','statistical_flagged_percent_of_all_objects',
        'precision_or_scope_note','decision')
    for item in summary['q18_outlier_vs_invalid']['rows']:
        q18_separation_rows.append(tuple(item.get(k) for k in q18_cols))
    for role, info in summary["files"].items():
        n = info["object_records"]
        for field, states in info["missing_states"].items():
            for state, count in states.items():
                missing_rows.append((role,field,state,count,n,count/n if n else ""))
        if role == 'reviews':
            q11 = info['q11_review_rating_validity']
            for item in q11['results']:
                q11_rating_rows.append((summary['scope'],role,info['path'],item['status'],
                    item['count'],item['denominator_parsed_review_objects'],
                    item['percent_of_parsed_review_objects'],
                    json.dumps(item['example_physical_lines']),item['initial_decision']))
            for star in range(1, 6):
                q11_star_rows.append((summary['scope'],role,info['path'],star,
                    q11['valid_star_counts'][str(star)],q11['denominator_parsed_review_objects'],
                    q11['valid_star_percent_of_all_parsed'][str(star)],
                    q11['valid_integral_stars_rows'],
                    100 * q11['valid_star_counts'][str(star)] / q11['valid_integral_stars_rows']
                    if q11['valid_integral_stars_rows'] else None))
        if role == 'metadata':
            q14 = info['q14_metadata_value_validity']
            for field, detail in q14['fields'].items():
                for item in detail['statuses']:
                    q14_value_rows.append((summary['scope'], role, info['path'], field,
                        item['status'],item['count'],item['denominator_parsed_metadata_objects'],
                        item['percent_of_parsed_metadata'],
                        json.dumps(item['example_physical_lines']),item['initial_decision']))
            for flag, count in sorted(q14['supplementary_flags'].items()):
                q14_numeric_flags.append((summary['scope'],role,info['path'],'rating_number',flag,
                    count,n,100*count/n if n else None))
        if role == 'reviews':
            q13 = info['q13_review_timestamp_validity']
            for item in q13['statuses']:
                q13_state_rows.append((summary['scope'],role,info['path'],q13['configured_unit'],
                    q13['timezone'],q13['reference_start_utc'],q13['reference_end_exclusive_utc'],
                    q13['audit_as_of_utc'],item['status'],item['count'],
                    item['denominator_parsed_review_objects'],item['percent_of_parsed_reviews'],
                    json.dumps(item['example_physical_lines']),item['assessment']))
            q13_future_rows.append((summary['scope'],role,info['path'],
                'representable_after_audit_as_of',q13['representable_after_audit_as_of_rows'],
                q13['denominator_parsed_review_objects'],
                q13['representable_after_audit_as_of_percent_of_parsed_reviews'],
                json.dumps(q13['representable_after_audit_as_of_example_physical_lines']),
                q13['audit_as_of_utc'],q13['earliest_representable_utc'],
                q13['latest_representable_utc'],q13['earliest_in_reference_window_utc'],
                q13['latest_in_reference_window_utc']))
        if role == 'reviews':
            q12 = info['q12_review_value_validity']
            for field, detail in q12['fields'].items():
                for item in detail['rows']:
                    q12_value_rows.append((summary['scope'],role,info['path'],field,
                        item['status'],item['count'],item['denominator_parsed_review_objects'],
                        item['percent_of_parsed_reviews'],
                        json.dumps(item['example_physical_lines']),item['initial_decision']))
            for key, count in sorted(q12['numeric_overlap_flags'].items()):
                q12_overlap_rows.append((summary['scope'],role,info['path'],'helpful_vote',key,
                    count,n,100*count/n if n else None))
        q10 = info['q10_cross_field_consistency']
        for item in q10['results']:
            q10_consistency_rows.append((summary['scope'],role,info['path'],
                item['relationship'],item['outcome'],item['count'],
                item['denominator_parsed_objects'],item['percent_of_parsed_objects'],
                item['assessment'],json.dumps(item['example_physical_lines']),
                item['initial_decision']))
        q09 = info['q09_format_audit']
        for item in q09['format_rules']:
            q09_format_rows.append((summary['scope'],role,info['path'],item['field'],item['rule'],
                item['count'],item['denominator'],item['denominator_unit'],item['percent'],
                json.dumps(item['example_physical_lines'],ensure_ascii=True)))
        for item in q09['category_variant_groups']:
            q09_variant_rows.append((summary['scope'],role,info['path'],item['field'],
                item['normalized_label'],item['distinct_raw_spellings'],
                item['observed_label_occurrences'],json.dumps(item['example_raw_spellings'],ensure_ascii=False)))
        q06 = info["q06_identifier_multiplicity"]
        for field in q06["identifier_fields"]:
            q06_identifier_rows.append((summary["scope"], role, info["path"],field["field"],
                field["denominator_parsed_objects"],field["rows_with_usable_raw_id"],
                field["rows_without_usable_raw_id"],field["distinct_raw_ids"],
                field["repeated_id_groups"],field["rows_in_repeated_id_groups"],
                field["rows_beyond_first_per_id"],field["repeated_id_rows_percent_of_parsed_objects"],
                field["repeated_id_rows_percent_of_valid_ids"],
                field["rows_with_unclassifiable_canonical_record"],field["interpretation"]))
        if role == "metadata":
            q08 = info['q08_metadata_type_compatibility']
            for item in q08['fields']:
                q08_type_rows.append((summary['scope'],info['path'],item['field'],
                    item['documented_field'],'|'.join(item['expected_nonnull_types']),
                    json.dumps(item['observed_types'],sort_keys=True),
                    item['object_records_denominator'],item['absent_rows'],item['explicit_null_rows'],
                    item['present_nonnull_rows'],item['compatible_nonnull_rows'],
                    item['incompatible_nonnull_rows'],item['incompatible_percent_of_present_nonnull'],
                    item['incompatible_percent_of_all_objects'],
                    json.dumps(item['incompatible_observed_types'],sort_keys=True),
                    item['numeric_like_string_candidates'],
                    json.dumps(item['wrong_type_example_physical_lines']),
                    item['assessment'],item['initial_decision']))
            for item in info['q08_metadata_nested_types']['paths']:
                q08_nested_rows.append((summary['scope'],info['path'],item['path'],item['parent_field'],
                    item['unit'],item['expected_nonnull_type'],
                    json.dumps(item['observed_type_counts'],sort_keys=True),
                    item['parsed_metadata_objects'],item['valid_parent_containers'],
                    item['observed_child_occurrences'],item['null_child_occurrences'],
                    item['present_nonnull_child_occurrences'],item['compatible_nonnull_child_occurrences'],
                    item['incompatible_nonnull_child_occurrences'],
                    item['incompatible_percent_of_nonnull_child_occurrences'],
                    item['affected_metadata_records'],item['affected_percent_of_valid_parent_containers'],
                    json.dumps(item['example_physical_lines']),item['assessment'],item['initial_decision']))
            for key,value in q06["metadata_parent_version_classification"].items():
                q06_version_rows.append((summary["scope"],info["path"],key,
                                         value if value is not None else ""))
            q05 = info["q05_metadata_exact_duplicates"]
            for key in (
                "denominator_parsed_metadata_objects", "compared_canonical_metadata_objects",
                "unable_to_compare_canonicalization_failures", "distinct_canonical_metadata_records",
                "duplicate_groups", "metadata_rows_in_duplicate_groups", "extra_duplicate_metadata_rows",
                "extra_duplicate_percent_of_compared", "extra_duplicate_percent_of_parsed",
                "rows_in_duplicate_groups_percent_of_compared",
                "distinct_records_sharing_a_digest_with_another_record", "status",
            ):
                q05_metadata_rows.append((summary["scope"], role, info["path"], key,
                                          "" if q05[key] is None else q05[key],
                                          q05["compared_canonical_metadata_objects"],
                                          q05["denominator_parsed_metadata_objects"]))
            for pair in q05["examples"]:
                q05_example_rows.append((summary["scope"], info["path"],
                                         pair["first_source_line"],
                                         pair["first_duplicate_source_line"],
                                         pair["copies_in_group"]))
        if role == "reviews":
            q07 = info["q07_review_type_compatibility"]
            for item in q07['fields']:
                q07_type_rows.append((summary['scope'],info['path'],item['field'],
                    item['documented_field'], '|'.join(item['expected_nonnull_types']),
                    json.dumps(item['observed_types'],sort_keys=True),
                    item['object_records_denominator'],item['absent_rows'],item['explicit_null_rows'],
                    item['present_nonnull_rows'],item['compatible_nonnull_rows'],
                    item['incompatible_nonnull_rows'],item['incompatible_percent_of_present_nonnull'],
                    item['incompatible_percent_of_all_objects'],
                    json.dumps(item['incompatible_observed_types'],sort_keys=True),
                    item['numeric_like_string_candidates'],
                    json.dumps(item['wrong_type_example_physical_lines']),
                    item['assessment'],item['initial_decision']))
            q04 = info["q04_review_exact_duplicates"]
            for key in (
                "denominator_parsed_review_objects", "compared_canonical_review_objects",
                "unable_to_compare_canonicalization_failures", "distinct_canonical_review_records",
                "duplicate_groups", "review_rows_in_duplicate_groups", "extra_duplicate_review_rows",
                "extra_duplicate_percent_of_compared", "extra_duplicate_percent_of_parsed",
                "rows_in_duplicate_groups_percent_of_compared",
                "distinct_records_sharing_a_digest_with_another_record", "status",
            ):
                q04_review_rows.append((summary["scope"],role,info["path"],key,
                                        "" if q04[key] is None else q04[key],
                                        q04["compared_canonical_review_objects"],
                                        q04["denominator_parsed_review_objects"]))
            for pair in q04["examples"]:
                q04_example_rows.append((summary["scope"], info["path"],
                                         pair["first_source_line"],
                                         pair["first_duplicate_source_line"],
                                         pair["copies_in_group"]))
            readiness = info["q03_review_readiness"]
            for field, statuses in readiness["field_outcomes"].items():
                for status, entry in statuses.items():
                    q03_review_rows.append((summary["scope"], role, info["path"],
                                            "core_field_status", field, status, entry["rows"], n,
                                            entry["percent_of_review_objects"]))
            for pattern, entry in readiness["exclusive_failure_patterns"].items():
                q03_review_rows.append((summary["scope"], role, info["path"],
                                        "exclusive_core_failure_pattern", "parent_asin+rating+timestamp",pattern,
                                        entry["rows"],n,entry["percent_of_review_objects"]))
            for measure in ("core_usable_review_rows", "core_unusable_review_rows_union"):
                value = readiness[measure]
                q03_review_rows.append((summary["scope"],role,info["path"],"core_union",measure,"all",value,n,
                                        (100*value/n if n else None)))
            for key,entry in readiness["text_optional"].items():
                q03_review_rows.append((summary["scope"],role,info["path"],"optional_text",key,"all",
                                        entry["rows"],n,entry["percent_of_review_objects"]))
            for key,value in readiness["product_counts_before_dedup"].items():
                q03_product_rows.append((summary["scope"],info["path"],key,value,
                                         readiness["minimum_observed_core_reviews_per_product"],
                                         "raw review objects before dedup; not final product eligibility"))
        for field, values in sorted(info["q02_field_missingness"].items()):
            context = {"scope": summary["scope"], "file_role": role,
                       "input_path": info["path"], "field": field,
                       "documented_field": field in EXPECTED[role]}
            record = {**context, **values}
            q02_field_rows.append(tuple(record.get(column) if record.get(column) is not None else ""
                                        for column in q02_columns))
        for field, entry in info["q01_nested_content"].items():
            q01_nested_rows.append((
                role, field, entry["content_rule"], entry["denominator_object_records"],
                entry["raw_present_nonempty_rows"], entry["usable_content_rows"],
                entry["raw_nonempty_but_unusable_rows"],
                entry["usable_share_of_all_objects"], entry["usable_share_of_raw_nonempty"],
            ))
        for field in sorted(set(info["types"]) | set(EXPECTED[role])):
            for kind, count in info["types"].get(field, {}).items():
                type_rows.append((role,field,kind,count,"|".join(sorted(EXPECTED[role].get(field, []))), "not_specified" if field not in EXPECTED[role] else ("missing" if kind == "null" else ("compatible" if kind in EXPECTED[role][field] else "mismatch"))))
        for rule, count in info["rule_counts"].items():
            rule_rows.append((role,rule,count,n,count/n if n else "",json.dumps(info["rule_example_line_numbers"].get(rule,[])),decision(rule, role)))
        for name, data in info["modality_availability"].items():
            availability_rows.append((role,name,data["rows"],n,data["share"]))
    # Q20: each family/pair/signature/union rate has the parsed-object census
    # for its OWN input role as the denominator; all are checked by Q19.
    q20_family_rows, q20_pair_rows, q20_signature_rows, q20_union_rows = [], [], [], []
    for role, info in summary['files'].items():
        overlap=info['q20_row_issue_overlaps']
        n=overlap['denominator_parsed_objects']
        for family,count in overlap['family_counts'].items():
            q20_family_rows.append((summary['scope'],role,info['path'],family,
                count,n,100*count/n if n else None))
        for pair in overlap['pair_intersections']:
            q20_pair_rows.append((summary['scope'],role,info['path'],
                pair['first_family'],pair['second_family'],pair['count'],n,
                pair['percent_of_parsed_objects']))
        for signature in overlap['exact_signatures']:
            q20_signature_rows.append((summary['scope'],role,info['path'],
                signature['signature'],'|'.join(signature['active_families']),
                signature['count'],n,signature['percent_of_parsed_objects'],
                json.dumps(signature['example_physical_lines'])))
        for name,count in overlap['union_counts'].items():
            q20_union_rows.append((summary['scope'],role,info['path'],name,
                count,n,100*count/n if n else None))
    common.write_csv(folder/'q20_issue_families.csv',
        ('scope','file_role','input_path','family','count','denominator_parsed_objects','percent_of_parsed_objects'),
        q20_family_rows)
    common.write_csv(folder/'q20_issue_pairs.csv',
        ('scope','file_role','input_path','first_family','second_family','count','denominator_parsed_objects','percent_of_parsed_objects'),
        q20_pair_rows)
    common.write_csv(folder/'q20_issue_signatures.csv',
        ('scope','file_role','input_path','signature','active_families','count','denominator_parsed_objects','percent_of_parsed_objects','example_physical_lines'),
        q20_signature_rows)
    common.write_csv(folder/'q20_issue_union.csv',
        ('scope','file_role','input_path','union_definition','count','denominator_parsed_objects','percent_of_parsed_objects'),
        q20_union_rows)
    common.write_csv(folder / 'q18_outlier_vs_invalid.csv', q18_cols, q18_separation_rows)
    common.write_csv(folder / "missingness.csv", ("file_role","field","state","count","object_records","share"),missing_rows)
    common.write_csv(folder / "q02_field_missingness.csv", q02_columns, q02_field_rows)
    common.write_csv(folder / "q03_review_readiness.csv",
                     ("scope","file_role","input_path","section","field_or_measure","status_or_pattern",
                      "rows","denominator_review_objects","percent_of_review_objects"),
                     ((*(v if v is not None else "" for v in row),) for row in q03_review_rows))
    common.write_csv(folder / "q03_product_impact.csv",
                     ("scope","reviews_input_path","measure","value","minimum_core_reviews","interpretation"),
                     q03_product_rows)
    common.write_csv(folder / "q04_review_exact_duplicates.csv",
                     ("scope","file_role","input_path","measure","value",
                      "compared_canonical_review_objects","parsed_review_objects"), q04_review_rows)
    common.write_csv(folder / "q04_duplicate_example_lines.csv",
                     ("scope","reviews_input_path","first_source_line",
                      "first_duplicate_source_line","copies_in_group"), q04_example_rows)
    common.write_csv(folder / "q05_metadata_exact_duplicates.csv",
                     ("scope","file_role","input_path","measure","value",
                      "compared_canonical_metadata_objects","parsed_metadata_objects"), q05_metadata_rows)
    common.write_csv(folder / "q05_metadata_duplicate_example_lines.csv",
                     ("scope","metadata_input_path","first_source_line",
                      "first_duplicate_source_line","copies_in_group"), q05_example_rows)
    common.write_csv(folder / "q06_identifier_multiplicity.csv",
                     ("scope","file_role","input_path","field","parsed_object_denominator",
                      "rows_with_usable_raw_id","rows_without_usable_raw_id","distinct_raw_ids",
                      "repeated_id_groups","rows_in_repeated_id_groups","rows_beyond_first_per_id",
                      "repeated_rows_percent_of_parsed_objects","repeated_rows_percent_of_valid_ids",
                      "rows_with_uncanonicalizable_records","interpretation"),q06_identifier_rows)
    common.write_csv(folder / "q06_metadata_parent_versions.csv",
                     ("scope","metadata_input_path","measure","value"),q06_version_rows)
    common.write_csv(folder / "q01_nested_content.csv",
                     ("file_role", "field", "content_rule", "object_records",
                      "raw_present_nonempty_rows", "usable_content_rows", "raw_nonempty_but_unusable_rows",
                      "usable_share_of_all_objects", "usable_share_of_raw_nonempty"),
                     q01_nested_rows)
    common.write_csv(folder / "q07_review_type_compatibility.csv",
                     ("scope","reviews_input_path","field","documented_field",
                      "expected_nonnull_types","observed_raw_types_json","parsed_review_objects",
                      "absent_rows","explicit_null_rows","present_nonnull_rows",
                      "compatible_nonnull_rows","incompatible_nonnull_rows",
                      "incompatible_percent_of_present_nonnull","incompatible_percent_of_all_objects",
                      "incompatible_observed_types_json","numeric_like_string_candidates",
                      "wrong_type_example_physical_lines_json","assessment","initial_decision"),
                     (("" if v is None else v for v in row) for row in q07_type_rows))
    common.write_csv(folder / 'q08_metadata_type_compatibility.csv',
                     ('scope','metadata_input_path','field','documented_field',
                      'expected_nonnull_types','observed_raw_types_json','parsed_metadata_objects',
                      'absent_rows','explicit_null_rows','present_nonnull_rows',
                      'compatible_nonnull_rows','incompatible_nonnull_rows',
                      'incompatible_percent_of_present_nonnull','incompatible_percent_of_all_objects',
                      'incompatible_observed_types_json','numeric_like_string_candidates',
                      'wrong_type_example_physical_lines_json','assessment','initial_decision'),
                     (tuple('' if v is None else v for v in row) for row in q08_type_rows))
    common.write_csv(folder / 'q08_metadata_nested_types.csv',
                     ('scope','metadata_input_path','path','parent_field','unit',
                      'expected_nonnull_type','observed_types_json','parsed_metadata_objects',
                      'valid_parent_containers','child_occurrences','null_child_occurrences',
                      'present_nonnull_child_occurrences','compatible_nonnull_child_occurrences',
                      'incompatible_nonnull_child_occurrences',
                      'incompatible_percent_of_nonnull_child_occurrences',
                      'affected_metadata_records','affected_percent_of_valid_parent_containers',
                      'example_physical_lines_json','assessment','initial_decision'),
                     (tuple('' if v is None else v for v in row) for row in q08_nested_rows))
    common.write_csv(folder / 'q09_format_flags.csv',
                     ('scope','file_role','input_path','field','rule','count',
                      'denominator','denominator_unit','percent','example_physical_lines_json'),
                     (tuple('' if v is None else v for v in row) for row in q09_format_rows))
    common.write_csv(folder / 'q09_category_variant_groups.csv',
                     ('scope','file_role','input_path','field','normalized_label',
                      'distinct_raw_spellings','observed_label_occurrences',
                      'example_raw_spellings_json'),q09_variant_rows)
    common.write_csv(folder / 'q11_review_rating_validity.csv',
                     ('scope','file_role','input_path','status','count','denominator_parsed_review_objects',
                      'percent_of_parsed_review_objects','example_physical_lines_json','initial_decision'),
                     (tuple('' if v is None else v for v in row) for row in q11_rating_rows))
    common.write_csv(folder / 'q11_review_star_distribution.csv',
                     ('scope','file_role','input_path','star','count','denominator_parsed_review_objects',
                      'percent_of_all_parsed','denominator_valid_ratings','percent_of_valid_ratings'),
                     (tuple('' if v is None else v for v in row) for row in q11_star_rows))
    common.write_csv(folder / 'q12_review_value_validity.csv',
                     ('scope','file_role','input_path','field','status','count',
                      'denominator_parsed_review_objects','percent_of_parsed_reviews',
                      'example_physical_lines_json','initial_decision'),
                     (tuple('' if v is None else v for v in row) for row in q12_value_rows))
    common.write_csv(folder / 'q12_review_numeric_flags.csv',
                     ('scope','file_role','input_path','field','numeric_flag','count',
                      'denominator_parsed_review_objects','percent_of_parsed_reviews'),
                     (tuple('' if v is None else v for v in row) for row in q12_overlap_rows))
    common.write_csv(folder / 'q13_review_timestamp_validity.csv',
                     ('scope','file_role','input_path','configured_unit','timezone_utc',
                      'reference_start_utc','reference_end_exclusive_utc','audit_as_of_utc',
                      'status','count','denominator_parsed_review_objects','percent_of_parsed_reviews',
                      'example_physical_lines_json','assessment'),
                     (tuple('' if v is None else v for v in row) for row in q13_state_rows))
    common.write_csv(folder / 'q13_timestamp_suspicions.csv',
                     ('scope','file_role','input_path','suspicion','count',
                      'denominator_parsed_review_objects','percent_of_parsed_reviews',
                      'example_physical_lines_json','audit_as_of_utc','earliest_representable_utc',
                      'latest_representable_utc','earliest_in_reference_window_utc',
                      'latest_in_reference_window_utc'),
                     (tuple('' if v is None else v for v in row) for row in q13_future_rows))
    common.write_csv(folder / 'q14_metadata_value_validity.csv',
                     ('scope','file_role','input_path','field','status','count',
                      'denominator_parsed_metadata_objects','percent_of_parsed_metadata',
                      'example_physical_lines_json','initial_decision'),
                     (tuple('' if v is None else v for v in row) for row in q14_value_rows))
    common.write_csv(folder / 'q14_metadata_numeric_flags.csv',
                     ('scope','file_role','input_path','field','numeric_flag','count',
                      'denominator_parsed_metadata_objects','percent_of_parsed_metadata'),
                     (tuple('' if v is None else v for v in row) for row in q14_numeric_flags))
    common.write_csv(folder / 'q10_cross_field_consistency.csv',
                     ('scope','file_role','input_path','relationship','outcome','count',
                      'denominator_parsed_objects','percent_of_parsed_objects','assessment',
                      'example_physical_lines_json','initial_decision'),
                     (tuple('' if v is None else v for v in row) for row in q10_consistency_rows))
    common.write_csv(folder / "types.csv", ("file_role","field","observed_type","count","expected_nonnull_types","assessment"),type_rows)
    common.write_csv(folder / "quality_rules.csv", ("file_role","rule","affected_rows","object_records","share","example_physical_lines","initial_decision"),rule_rows)
    common.write_csv(folder / "modality_availability.csv", ("file_role","availability_definition","available_rows","object_records","share"),availability_rows)
    common.write_csv(folder / "numeric_outliers.csv", ("metric","n","minimum","maximum","mean","q1","median","q3","lower_fence","upper_fence","flagged_rows","outlier_status"),
                     ((name, *[data.get(key) for key in ("n","minimum","maximum","mean","q1","median","q3","lower_fence","upper_fence","flagged_rows","outlier_status")]) for name,data in summary["numeric_diagnostics"].items()))
    q15_columns = (
        "metric", "reference_unit", "denominator_parsed_review_objects",
        "valid_nonnegative_raw_integer_rows", "screened_valid_integer_rows",
        "valid_large_integer_rows_not_screened", "zero_vote_rows", "positive_vote_rows",
        "zero_percent_of_screened", "p95_all_screened", "p99_all_screened",
        "strictly_above_p99_rows", "strictly_above_p99_percent",
        "positive_log_q1", "positive_log_q3", "positive_log_upper_fence",
        "positive_upper_fence_raw", "positive_upper_outlier_rows",
        "positive_upper_outlier_percent_of_positive",
        "positive_upper_outlier_percent_of_screened", "status", "method",
        "quantile_method", "initial_decision", "caveat")
    q15 = summary["q15_review_numeric_outliers"]
    common.write_csv(folder / "q15_review_numeric_outliers.csv", q15_columns,
                     [tuple(q15.get(key) for key in q15_columns)])
    q16_columns = (
        'metric','reference_unit','denominator_parsed_metadata_objects',
        'domain_valid_rows','valid_large_integer_rows_not_screened',
        'screened_valid_rows','zero_rows','positive_rows','zero_percent_of_screened',
        'positive_percent_of_screened','p95_all_screened','p99_all_screened',
        'strictly_above_p99_rows','strictly_above_p99_percent',
        'raw_q1','raw_q3','raw_lower_fence','raw_upper_fence',
        'raw_two_sided_outlier_rows','raw_outlier_percent_of_screened','raw_status',
        'positive_log_q1','positive_log_q3','positive_log_upper_fence',
        'positive_upper_fence_raw','positive_log_upper_outlier_rows',
        'positive_log_outlier_percent_of_positive','positive_log_outlier_percent_of_screened',
        'positive_log_status','primary_screen','primary_status','primary_flagged_rows',
        'primary_percent_of_screened','quantile_method','initial_decision','caveat')
    q16 = summary['q16_metadata_numeric_outliers']
    common.write_csv(folder/'q16_metadata_numeric_outliers.csv',q16_columns,
                     [tuple(q16[field].get(k) for k in q16_columns)
                      for field in ('price','rating_number')])
    lines = ["# Diagnostic data quality audit", "", f"Scope: {summary['scope']}",
             "", "Raw data is unchanged. No record is deleted or imputed; no prediction target is created.",
             "", "## Input and parsing", ""]
    for role, info in summary["files"].items():
        lines += [f"### {role}", "",f"- Physical lines: {info['physical_lines']:,}",
                  f"- Accepted unambiguous JSON objects: {info['object_records']:,}",
                  f"- Parse errors (including duplicate JSON keys and nonfinite numbers): {info['parse_errors']:,}",
                  f"- Blank lines: {info['blank_lines']:,}; nonobject lines: {info['nonobject_lines']:,}",
                  f"- Verified full-record duplicate extra rows: {info['full_record_duplicates']['duplicate_extra_rows']:,}", ""]
    lines += ["## Counts and initial decisions", "", "| File | Rule | Affected rows | Initial decision |", "|---|---|---:|---|"]
    lines += [f"| {row[0]} | {row[1]} | {row[2]:,} | {row[-1]} |" for row in rule_rows]
    lines += ["", "## Missing fields (Q01)", "", "missingness.csv reports the six exclusive raw states for each field. The sum of states is asserted to equal all parsed object records for that file. Neither numeric zero nor boolean False is missing. Suspicious sentinel words remain literal strings and are flagged separately.",
              "", "q01_nested_content.csv separately reports field-specific *syntactic* content usability inside known nonempty nested fields. Its raw_nonempty_but_unusable_rows are a subset of present_nonempty, not a seventh missing state. Valid-looking media URLs are not downloaded; nonblank strings may still be semantically uninformative.",
              "", "## Field missingness rates (Q02)", "", "q02_field_missingness.csv reports one row for every expected or observed field in both review and metadata inputs. The denominator is the count of successfully parsed JSON objects in THAT FILE and THAT scan (not physical lines or valid-after-dedup rows). Original copies are counted; errors/blank/nonobjects remain in file-level diagnostics.",
              "", "raw_missing = absent + null. raw_missing_or_empty = raw_missing + blank_string + empty_array + empty_object. These are two distinct definitions; neither includes nonempty content which may be wrong type or semantically unusable. No source records are modified. State percentages and both combined rates are given explicitly in 0-100 percent units; original missingness.csv shares remain 0-1.",
              "", "A missing field across the entire file is still reported when documented in EXPECTED. If a scan has zero accepted objects, rates are undefined (null / blank in CSV), NOT zero percent. Scope (full supplied file, sample, or head preview) must be checked before interpreting a rate as a dataset property.",
              "", "Missing text alone does not remove a review from the ordered sequence. Missing current price is retained explicitly; no blind imputation is used.",
              "", "## Q03: effect of missing/invalid core fields on analysis readiness", "",
              "q03_review_readiness.csv reports missing-or-empty and present-but-unusable values separately for parent_asin, rating and timestamp. Overlapping failures are counted exactly once in the union via mutually exclusive bit masks. The denominators are parsed raw review objects before exact deduplication; title and body text are OPTIONAL for review positions and are reported separately.",
              "q03_product_impact.csv counts distinct products with an indexable raw parent_asin and compares product counts with at least the configured minimum of 15 raw reviews versus 15 core-usable reviews. This uses an on-disk SQLite aggregation and does not load all product identifiers into memory. Any product count of 15+ is a preliminary screening upper bound ONLY, not definitive 5+10 cohort membership. Deduplication, temporal tie-breaking, joins and splits are separate work. Missing current price, metadata and images never automatically remove a review from the main rating/time sequence.",
              "", "## Q04: full-record exact duplicates in reviews", "",
              "q04_review_exact_duplicates.csv distinguishes unique canonical records, groups with repeats, all rows in those groups, and extra copies beyond the first. Percentages use explicit compared/parsed review-object denominators, never product or user keys. q04_duplicate_example_lines.csv contains bounded physical-line pairs (first copy, first repeated copy), without exposing review text.",
              "SHA-256 serves only as an index: any digest match is rechecked against the COMPLETE canonical JSON record from disk. Whitespace and object key order do not change parsed identity; array order and int/float types remain significant. Distinct canonical rows under a digest collision are counted separately. A canonicalization failure is counted as not compared, not assumed unique.",
              "This result is only for the scanned input (head_preview is not an exhaustive audit). No duplicates are removed. Duplicate keys or malformed JSON are handled by parsing checks and excluded from the parsed-review denominator. Shared copies across different category files require a separate cross-file check. Same user/product/time or similar text are not automatically exact duplicates. Product eligibility in Q03 is still BEFORE deduplication.",
              "", "## Q05: full-record exact duplicates in metadata", "",
              "q05_metadata_exact_duplicates.csv gives the exact same verified canonical-JSON census for metadata objects, including unique records, duplicate groups, rows in groups, extra copies, rates with explicit parsed/compared denominators and collision diagnostics. q05_metadata_duplicate_example_lines.csv records bounded physical source-line pairs without metadata contents.",
              "A repeated parent_asin is NOT automatically a duplicate metadata record: different records sharing one key must be handled in the separate Q06/S14 key-multiplicity audit before joining to reviews. Verified identical extra copies are only PROPOSED for later removal in a derived view, and source files are never altered. A preview/partial scan proves nothing about the entire dataset; between-category exact duplicates require a separate cross-file analysis.",
              "", "## Q06: identifier multiplicity versus full-record duplicates", "",
              "q06_identifier_multiplicity.csv counts exact RAW nonblank identifier keys separately from Q04/Q05 verified complete-record duplicates. Review parent_asin, asin and user_id are allowed to repeat; repeated keys alone never justify removing reviews. Percentages are of scanned parsed objects or key-bearing objects as labeled.",
              "q06_metadata_parent_versions.csv splits repeated metadata parent_asin groups into exact-copy-only, verified different records (conflicting versions), and indeterminate when canonicalization could not be completed. Metadata with different records under the same parent key must NOT be arbitrarily joined; this is a diagnostic complement to structural S14, not a replacement for its join tests.",
              "No raw IDs or text are exposed in these summary CSVs. No deduplication is performed. Cross-category overlaps are not covered. With head_preview only the scanned prefix is counted.",
              "", "## Q07: review field type compatibility", "",
              "q07_review_type_compatibility.csv profiles all documented and additionally observed REVIEW fields, including expected non-null types, observed raw JSON types, explicit-null/absent counts, compatible and incompatible non-null values, and mismatch percentages with both denominators. Numeric-looking string counts are optional conversion candidates only: they remain WRONG TYPE until a separate, justified transformation. No conversion is performed here.",
              "Missing/null values are not type errors. Booleans are not integers. Extra fields are profiled but do not fail an unspecified schema. Wrong-type occurrences are per FIELD and may overlap on the same record: the total is not a distinct-record count. Zero-observation and null-only fields are preserved; their rates are undefined where the non-null denominator is zero. Q08 separately audits metadata field types. All Q01-Q06 outputs are retained.",
              "", "## Q08: metadata top-level and nested field types", "",
              "q08_metadata_type_compatibility.csv reports raw JSON type compatibility for all documented and observed METADATA fields, with explicit absent/null counts, denominators, and numeric-looking string candidates (not automatically converted). Integer-only rating_number is distinct from float-permitted price and average_rating. Domain validity is checked separately in Q14.",
              "q08_metadata_nested_types.csv separately screens items in known string and media-object lists, and recognized string-valued image/video descriptor attributes. Child JSON nulls are counted separately and are NOT wrong non-null types. Nested percentages use CHILD OCCURRENCES, whereas affected-metadata-record percentages use the number of records containing a correctly typed parent container. Details object values are profiled without an invented fixed value schema; unknown nested media keys are not automatically errors.",
              "No raw values, IDs or text are exported. These are screening assumptions for known fields, not claims that all source versions always use the same nested schema. No records are changed. Q01-Q07 outputs are preserved.",
              "", "## Q09: lexical format and categorical spelling consistency", "",
              "q09_format_flags.csv reports nonblank inspected strings, surrounding/repeated whitespace, ASIN uppercase-ten-character convention warnings, and normalized detail-key collisions. These are diagnostic candidates and NEVER validation failures or automated repairs. Review body and heterogeneous details values are excluded from case-based matching.",
              "q09_category_variant_groups.csv reports distinct short, valid-Unicode category label SPELLINGS sharing an identical whitespace-collapsed/casefolded representation; main_category and categories[] are independent groups. Only category labels are casefolded for this screening. No identifiers or review contents are normalized, joined, removed or recoded. Raw category samples are included in this targeted, bounded example report to support manual inspection.",
              "A 256-character cap applies ONLY to category-label variant grouping to bound disk use; all excluded labels are counted in q09_format_flags.csv. No guarantee that differently spelled category labels have the same meaning; even casefold matches need inspection. Counts can overlap; totals are not counts of bad records. Numeric-like strings and timestamps are already checked in Q07/Q08 and S17/Q13; the Amazon data do not supply a universal unit/currency normalization contract. No transformations occur.",
              "", "## Q11: review-star rating domain and integer validity", "",
              "q11_review_rating_validity.csv partitions every parsed review object into exactly one of five statuses: valid integral 1..5 (including 5.0), missing/empty, wrong nonnull type (including boolean), numeric outside [1,5], and fractional numeric within [1,5]. No automatic clipping, rounding or parsing of numeric-looking strings occurs. Numeric fractions outside [1,5] are counted as outside-range in the exclusive partition, but are also covered in a separately named overlapping fractional-all summary. Rates use parsed raw review objects, NOT physical lines or deduplicated records. All zero statuses and undefined empty-file rates are reported explicitly.",
              "q11_review_star_distribution.csv reports valid 1..5 stars both as a proportion of all parsed reviews and of valid rated reviews. The existing A/Q03 valid_rating and rating_out_of_domain counts are independently cross-checked. This is a diagnostic and not a prediction-label construction. For invalid/missing ratings the raw review is retained; the record remains unusable for the core rating sequence until resolved. No source files are changed.",
              "", "## Q12: review helpful_vote and verified_purchase domain checks", "",
              "q12_review_value_validity.csv partitions EACH parsed review into one status for helpful_vote and one for verified_purchase. Helpful_vote must be a raw JSON nonnegative integer (not boolean or integral-looking float). Verified_purchase must be a raw JSON boolean; numeric 0/1 and strings are not coerced. Explicit null and absent/empty field values are reported as missing/empty and excluded from the valid count. Both fields use the same denominator of all parsed review objects, including raw duplicate copies but excluding malformed or nonobject lines.",
              "q12_review_numeric_flags.csv provides descriptive overlapping diagnostics for float-valued helpful_vote, including negative and fractional candidates. These flags are NOT additional bad rows and must not be summed with exclusive invalid statuses. Original helpful_vote_invalid and Q07 wrong-type counts are cross-checked to detect regressions. A missing/invalid helpful_vote or verified_purchase is a field-level limitation, not evidence that the entire review should be removed or that rating/chronological position is invalid. No automatic repair, deletion or transformation is performed.",
              "", "## Q13: review timestamp technical validity and plausibility", "",
              "q13_review_timestamp_validity.csv partitions all parsed review objects into six mutually exclusive statuses: missing/empty; wrong nonempty type; unrepresentable integer; representable before release-window start; representable within window; and representable at/after window end. All statuses have explicit denominator and sample physical-line positions. Counts cross-check Q01, Q03, Q07 and original temporal flags. This is an audit of review creation timestamp, NOT metadata crawl time.",
              "q13_timestamp_suspicions.csv separately reports a potentially overlapping warning for REPRESENTABLE timestamps later than the fixed as-of UTC time, and exact observed earliest/latest datetimes. Reference bounds are in the configured timestamp unit (seconds or milliseconds), shown both as numeric epochs and UTC dates. The documented dataset window is a plausibility reference, not evidence of an invalid observation. Technical representability does not guarantee a usable historical sequence, and Q03 original core counts remain unchanged; future timestamps require a downstream eligibility policy. Neither input nor timestamps are modified. For deterministic replication, set quality_settings.q13_audit_as_of_utc to a timezone-aware ISO timestamp.",
              "", "## Q14: metadata price, average_rating and rating_number validity", "",
              "q14_metadata_value_validity.csv partitions every parsed METADATA object into one exclusive status for each of price, average_rating and rating_number, with explicit denominators, example source lines and initial decisions. Price must be a JSON numeric >=0; zero is a plausible free-item/sentinel warning, not proof of error. Product average_rating values 1..5 are domain-valid, whereas zero is separately screened as a possible unrated source sentinel; original A treats zero as outside its adopted rating range and its legacy counters remain unchanged. Product rating_number must be a nonnegative raw JSON integer; negative, fractional and integral-looking floats are separated and never rounded or converted. Missing/empty and raw wrong types remain distinguishable.",
              "q14_metadata_numeric_flags.csv contains explicitly overlapping metadata rating_number flags, which must not be added to the exclusive counts. The Q14 output cross-checks Q08 raw-type counts, Q10 snapshot context and original A domain rules, but it does not infer contradictions by equating rating_number to the count of downloaded reviews. Very high valid numeric values are examined further as statistical outliers in Q16. Current metadata aggregates and current price are NOT historically available predictors merely because the file contains them. Source files stay unchanged.",
              "", "## Q10: within-record cross-field consistency candidates", "",
              "q10_cross_field_consistency.csv partitions EACH parsed object into one outcome with explicit denominator, including zero-count outcomes and nonassessable cases. Review asin / parent_asin equality is purely descriptive: different valid IDs are not an error. Metadata zero/nonzero disagreement between rating_number and average_rating is a suspicious snapshot-aggregate candidate only, not a logically proven contradiction. Source-specific zero sentinel behavior and crawl timing must be checked before any decision. No invented hard rule for verified_purchase, helpful_vote, main_category/categories, or full-review-vs-metadata rating totals is imposed. No raw data is changed; no pair is inferred from separate unmatched records.",
              "", "## Possible review-event groups", "", json.dumps(summary["candidate_event_groups"], ensure_ascii=True),
              "", "## Statistical outliers", "", "Full scanned-data frequency tables define diagnostic quartiles and Tukey fences. No outlier is automatically deleted. With zero IQR the count is undefined, not zero: inspect the discrete/zero-inflated distribution. Short/long text thresholds are configurable heuristics, not proof of bad data.",
              "", "## Q15: review numeric outliers (helpful votes)", "",
              "q15_review_numeric_outliers.csv screens only domain-valid raw nonnegative integer helpful_vote counts. This is the meaningful free-ranging raw numeric review attribute; star ratings have a finite discrete domain (Q11), and timestamps are audited for temporal validity in Q13. Text lengths have their own dedicated text audit (Q22-Q23).",
              "For zero-inflated counts, the optional upper-tail screen is based on positive counts only: log1p transform, frequency-weighted Q1/Q3, upper fence Q3+3*IQR, and strictly-above-fence positive observations. Positive-only is a statistical reference subset, NOT removal of zero-vote reviews. Report both the screened-all denominator and the positive-only denominator explicitly. If there are fewer than four positive values or the positive-log IQR is zero, the statistical outlier count is UNDEFINED, never zero.",
              "The existing numeric_outliers.csv stays unchanged as a descriptive raw-scale 1.5-IQR baseline. The new 95th/99th percentiles refer to ALL screened valid vote counts, with a separate strict >p99 descriptive tail count. Domain-invalid votes and valid integers above 2**53 are excluded from the floating-point reference and documented through Q12 and explicit omission counts. No record is removed, no imputation occurs, and a high vote count is not an impossible value. Thresholds derived here must not be reused as fitted predictors before a leakage-safe split.",
              "", "## Q16: metadata numeric outliers (price and rating counts)", "",
              "q16_metadata_numeric_outliers.csv reports exactly two independent distributions: metadata.price and metadata.rating_number. Both first use Q14-validated values; invalid, missing and raw type-mismatched values are not silently coerced. Valid raw integer values above 2**53 are reported as excluded from floating-point diagnostic summaries, not ignored. Reference denominator is parsed metadata objects; screened and positive-only denominators are stated separately.",
              "For price, the primary descriptive screen is the original two-sided raw-scale Tukey 1.5*IQR rule, with positive-only log1p 3*IQR as a sensitivity analysis. For nonnegative integer rating counts, the primary descriptive screen uses only positive values to form a log1p upper 3*IQR reference and also reports the raw-scale baseline. Zero values remain in the census; a zero price is a warning from Q14, not missingness or a proven error.",
              "The 95th and 99th percentiles and strictly-above-p99 counts are for ALL screened values. If either reference has fewer than four values, zero IQR or unusable fences, its outlier count is undefined (null), not zero. The entire available file forms this provisional distribution; category-specific comparisons are Q17 and cannot be inferred from this single-file screen. No values or records are changed, and metadata snapshot fields must not be used to predict historical reviews without leakage checks.",
              "", "## Q18: separate data errors and statistical extremes", "",
              "q18_outlier_vs_invalid.csv combines prior validation (Q11, Q12, Q13, Q14) and statistical screening (Q15, Q16). Each field is partitioned into mutually exclusive missing, wrong-type, domain/technical-invalid, accepted, and source-warning statuses. The exclusive categories sum exactly to parsed objects. Some warnings are technically usable (zero price and representable timestamps outside a plausibility window); a zero product average is an unresolved sentinel warning and remains unusable under Q14's adopted baseline.",
              "A statistically flagged outlier is already an accepted field value: NEVER add it to missing/invalid counts. Screening is not applied to bounded star ratings, average_rating or timestamp. Null statistical flags mean no defensible fence was computed; they do NOT mean zero outliers. Very large integers deliberately omitted from float-based diagnostics remain valid where Q12/Q14 accept them. Domain-usability is a FIELD-level classification, not permission to use historical metadata as predictive features and not a count of eligible 5+10 products.",
              "", "## Limits", "", "- Rule counts can overlap; their sum is not a count of unique bad records.",
              "- Duplicate counts compare complete canonical JSON records after strict parsing; formatting and object key order are ignored, array order and numeric type representation are preserved.",
              "- Possible event groups use user/product/time, but are not asserted to be one real review event. No event versions are resolved in this step.",
              "- Availability of an HTTP image URL does not prove that a downloadable image exists. No images or URLs are fetched.",
              "- Modality rates have within-file denominators. Cross-file product joins and missingness mechanisms require later analysis; absence cannot be declared random here.",
              "- Timestamp units are configured explicitly. Coverage flags are suspicions for follow-up, not proof of invalid dates.",
              "- Numeric-string values are reported as type mismatches, not silently converted.",
              "- Global quartiles are descriptive only; do not reuse them as fitted preprocessing thresholds for model evaluation.", ""]
    (folder / "quality_report.md").write_text("\n".join(lines),encoding="utf-8")
    with (folder/'quality_report.md').open('a',encoding='utf-8') as report:
        report.write('\n## Q20 — exact within-row overlaps among defined audit families\n'
                     'q20_issue_families.csv contains nonexclusive family counts; q20_issue_pairs.csv contains exact pair intersections; '
                     'q20_issue_signatures.csv partitions each parsed object into one exact bitmask; '
                     'q20_issue_union.csv gives exact broad and narrower unions. '
                     +Q20_SCOPE_NOTE+'\n')
    # Q21 is a pure report aggregation, independent of original rule decisions.
    # No rate/share column is introduced: an issue register is NOT an additive census.
    q21 = summary['q21_initial_decisions']
    common.write_csv(folder/'q21_initial_decisions.csv', Q21_COLUMNS,
        [tuple(json.dumps(item.get(k),ensure_ascii=False) if k=='example_physical_lines'
               else item.get(k) for k in Q21_COLUMNS) for item in q21['issues']])
    common.write_json(folder/'q21_policy_catalog.json',
        dict(scope=summary['scope'],version=q21['version'],catalog=q21['catalog'],
             unclassified_rule_names=q21['unclassified_rule_names'],
             interpretation=q21['interpretation']))
    # Q22: preserve the existing Q01-Q21 CSVs. Length histograms are made
    # from the existing disk-backed numeric frequency table, NOT from raw data.
    q22 = summary['q22_review_text_lengths']
    q22_population_cols = ('scope','file_role','state','count',
                           'denominator_parsed_review_objects','percent_of_parsed_review_objects')
    q22_states = [tuple((summary['scope'],'reviews',state,count,
                          q22['denominator_parsed_review_objects'],
                          100*count/q22['denominator_parsed_review_objects']
                          if q22['denominator_parsed_review_objects'] else None))
                  for state,count in q22['states'].items()]
    common.write_csv(folder/'q22_text_presence.csv', q22_population_cols, q22_states)
    q22_dist_cols = ('scope','file_role','unit','metric','string_rows',
                     'denominator_parsed_review_objects','distinct_observed_lengths',
                     'minimum','maximum','mean','p25','p50','p75','p90','p95','p99',
                     'quantile_method')
    common.write_csv(folder/'q22_text_length_distribution.csv', q22_dist_cols,
        [(summary['scope'],'reviews',unit,d['metric'],d['string_rows'],
          d['denominator_parsed_review_objects'],d['distinct_observed_lengths'],
          d['minimum'],d['maximum'],d['mean'],
          *(d['quantiles']['p'+str(p)] for p in (25,50,75,90,95,99)),
          d['quantile_method']) for unit,d in q22['distributions'].items()])
    q22_bin_cols = ('scope','file_role','unit','metric','lower_inclusive','upper_inclusive',
                    'count','denominator_string_values','percent_of_string_values',
                    'denominator_parsed_review_objects','percent_of_parsed_review_objects')
    common.write_csv(folder/'q22_text_length_buckets.csv',q22_bin_cols,
        [(summary['scope'],'reviews',unit,d['metric'],b['lower_inclusive'],
          b['upper_inclusive'],b['count'],b['denominator_string_values'],
          b['percent_of_string_values'],b['denominator_parsed_review_objects'],
          b['percent_of_parsed_review_objects'])
         for unit,d in q22['distributions'].items() for b in d['buckets']])
    # Q23: thresholds and sensitivity variants are derived only from
    # the Q22 histograms and prior A flags; all older CSVs stay intact.
    q23 = summary['q23_review_short_long_text']
    q23_cols = ('scope','file_role','kind','unit','comparison','threshold',
                'is_primary','count','denominator_reference_strings',
                'percent_of_reference_strings','denominator_parsed_review_objects',
                'percent_of_parsed_review_objects','example_physical_lines')
    common.write_csv(folder/'q23_short_long_thresholds.csv', q23_cols,
        [(summary['scope'],'reviews',*(json.dumps(r[k],ensure_ascii=False) if k=='example_physical_lines' else r[k]
                                       for k in q23_cols[2:])) for r in q23['rows']])
    # Q24: descriptive Unicode/script diagnostics from the existing raw scan.
    q24=summary['q24_review_unicode_audit']
    q24_script_cols=('scope','file_role','script','letter_occurrences',
                     'denominator_all_letters','percent_of_all_letters',
                     'reviews_containing_script','denominator_string_bodies',
                     'percent_of_string_bodies','dominant80_body_rows')
    common.write_csv(folder/'q24_unicode_script_profile.csv',q24_script_cols,
        [(summary['scope'],'reviews',*(r[k] for k in q24_script_cols[2:]))
         for r in q24['script_rows']])
    q24_pattern_cols=('scope','file_role','pattern','count','denominator_string_bodies',
                      'percent_of_string_bodies')
    common.write_csv(folder/'q24_script_patterns.csv',q24_pattern_cols,
        [(summary['scope'],'reviews',k,c,q24['string_valued_body_rows'],
          100*c/q24['string_valued_body_rows'] if q24['string_valued_body_rows'] else None)
         for k,c in q24['pattern_counts'].items()])
    q24_flag_cols=('scope','file_role','flag','count','denominator_string_bodies',
                   'percent_of_string_bodies','denominator_parsed_review_objects',
                   'percent_of_parsed_review_objects','example_physical_lines_json','initial_decision')
    common.write_csv(folder/'q24_unicode_flags.csv',q24_flag_cols,
        [(summary['scope'],'reviews',k,c,q24['string_valued_body_rows'],
          100*c/q24['string_valued_body_rows'] if q24['string_valued_body_rows'] else None,
          q24['denominator_parsed_review_objects'],
          100*c/q24['denominator_parsed_review_objects'] if q24['denominator_parsed_review_objects'] else None,
          json.dumps(q24['example_physical_lines'][k],ensure_ascii=True),
          q24['flag_initial_decisions'][k])
         for k,c in q24['flag_counts'].items()])
    # Q25 — per-star body presence, length, and transparent surface cues.
    q25=summary['q25_rating_text_association']
    q25_presence_cols=('scope','star','text_state','count','denominator_rating_rows',
        'percent_of_rating_rows','denominator_all_valid_rating_rows','percent_of_all_valid_ratings')
    common.write_csv(folder/'q25_rating_text_presence.csv',q25_presence_cols,
        [(summary['scope'],r['star'],state,r['text_states'][state],r['valid_rating_rows'],
          100*r['text_states'][state]/r['valid_rating_rows'] if r['valid_rating_rows'] else None,
          q25['denominator_valid_rating_review_objects'],
          100*r['text_states'][state]/q25['denominator_valid_rating_review_objects']
          if q25['denominator_valid_rating_review_objects'] else None)
         for r in q25['per_star'] for state in Q25_STATES])
    q25_features_cols=('scope','star','feature','count','denominator_nonblank_bodies_in_star',
        'percent_of_nonblank_bodies_in_star','feature_definition')
    common.write_csv(folder/'q25_rating_text_features.csv',q25_features_cols,
        [(summary['scope'],r['star'],feat,r['features'][feat],r['nonblank_string_bodies'],
          100*r['features'][feat]/r['nonblank_string_bodies'] if r['nonblank_string_bodies'] else None,
          q25['feature_definition'])
         for r in q25['per_star'] for feat in Q25_FEATURES])
    q25_length_cols=('scope','star','valid_rating_rows','nonblank_string_bodies',
        'nonblank_body_percent_of_rating_rows','sum_unicode_codepoints','sum_whitespace_words',
        'mean_unicode_codepoints','mean_whitespace_words',
        'min_unicode_codepoints','max_unicode_codepoints',
        'min_whitespace_words','max_whitespace_words')
    common.write_csv(folder/'q25_rating_text_length.csv',q25_length_cols,
        [(summary['scope'],*(r[k] for k in q25_length_cols[1:])) for r in q25['per_star']])

    summary['q19_rate_integrity'] = q19_rate_denominator_audit(folder, summary)
    common.write_json(folder/'quality_summary.json',summary)
    with (folder/'quality_report.md').open('a',encoding='utf-8') as report:
        report.write('\n\n## Q19: rate denominators and arithmetic reconciliation\n\n'
                     'q19_rate_denominator_ledger.csv gives each rate its numerator, denominator, '
                     'denominator unit, original rate, recomputed rate, scope, and relation (exclusive, '
                     'subset, or overlapping). Every rate-bearing column in the registered Q01-Q18 CSV '
                     'outputs has an explicit contract; missing contracts, mismatched rates, invalid '
                     'count totals, and incorrect parsed-object denominators stop the audit. '
                     'Shares are in [0,1] while percentages are in [0,100]. '
                     'Undefined rates remain blank, never zero. Non-object, malformed and blank '
                     'physical lines are counted in a separate physical-line census.\n'
                     'q19_rate_integrity.json gives the reconciliation totals and limitations. '
                     'No raw records are revisited and no Q01-Q18 output is changed.\n')
        report.write('\n## Q22 — review body-text length distributions\n'
                     'q22_text_presence.csv partitions parsed review objects into absent, null, '
                     'blank-string, nonblank-string, and wrong-type body fields. '
                     'q22_text_length_distribution.csv gives Unicode code-point and whitespace-separated '
                     'word counts with six weighted quantiles, minimum, maximum, and mean. '
                     'q22_text_length_buckets.csv contains fixed, exhaustive and disjoint bins. '
                     'Percentages use explicit string or all-parsed-review denominators and are checked by Q19. '
                     'Length zero means a present empty string; no length is assigned to missing or nonstring values. '
                     'Distributions count all raw parsed review objects before dedup, including repeated copies. '
                     'This audit does not select 5+10 cohorts or treat missing body text as an invalid rating.\n')
        report.write('\n## Q25 — preliminary relationship between review BODY text and rating\n'
                     'q25_rating_text_presence.csv cross-tabulates mutually exclusive text states by valid star rating (1..5). '
                     'q25_rating_text_length.csv reports character and whitespace-word means and extrema, CONDITIONED on nonblank string bodies. '
                     'q25_rating_text_features.csv reports nonexclusive short, long, exclamation and question presence within nonblank bodies. '
                     'All output rates are Q19-validated. Invalid rating rows are excluded from per-star strata but counted separately. '
                     'Only raw review BODY is assessed; title and metadata text are not substituted. '
                     'This is a descriptive association on raw pre-dedup records, not a sentiment analysis, a significance test, '
                     'a leakage-safe held-out model result, or a causal finding. Missing text never changes a chronological review position.\n')
        report.write('\n## Q24 — Unicode script and character diagnostics for review body\n'
                     'q24_unicode_script_profile.csv counts alphabetic Unicode code points and review bodies '
                     'with each approximate Unicode-block script label; such labels are not detected languages. '
                     'q24_script_patterns.csv partitions all string bodies into no letters, a single script, '
                     'or multiple scripts. q24_unicode_flags.csv counts possible encoding/control warnings '
                     'per BODY, with overlap allowed; all denominators are registered and validated by Q19. '
                     'The pre-existing text replacement/surrogate/C0-control flags are reconciled exactly. '
                     'Emoji, natural multilingual writing and Persian zero-width non-joiners are not flagged as errors. '
                     'BiDi controls can be legitimate; a warning is not proof of corruption. '
                     'Neither language classification nor actual character-level repair is performed.\n')
        report.write('\n## Q23 — very short and very long review body texts\n'
                     'q23_short_long_thresholds.csv reports short nonblank bodies (1..K whitespace words) '
                     'and long string bodies (>L Unicode code points), using the existing Q22 histograms. '
                     'The two pre-existing A thresholds (configurable) are labeled PRIMARY; '
                     'additional thresholds are sensitivity comparisons, not universal data-quality cutoffs. '
                     'Whitespace-only body strings are not counted short; however extremely long whitespace-only '
                     'strings can meet the long threshold, as the original A rule does. Missing, null and '
                     'wrong-type bodies have no assigned length. Rates are shown both against the applicable '
                     'reference string set and against all parsed review objects; Q19 verifies every rate. '
                     'The short/long intersection cannot be reconstructed from separate 1D histograms, '
                     'so no claim of their unique union is made. Raw inputs, 5+10 review positions, '
                     'and all older Q01-Q22 reports remain unchanged.\n')
        report.write('\n## Q21 — evidence-linked initial quality decisions\n' 
                     'q21_initial_decisions.csv lists observed issues, count units, evidence '
                     'sources, task-specific actions, stages and rationale. '
                     'q21_policy_catalog.json lists all action contracts, including policies '
                     'not observed in this scan. Unknown rules require manual triage; '
                     'no observed count is a directive to delete source rows. '
                     'Issue counts overlap, group-based counts cannot be added to row counts, '
                     'and Q15/Q16 statistical flags are subsets of valid measurements. '
                     'Q03 readiness counts are before dedup and are not final 5+10 cohorts. '
                     'No cleaning, feature fitting, joining or model training occurs here.\n')



# Q17: CATEGORY-SPECIFIC diagnostic comparison of prior FULL per-category
# Q15/Q16 reports. This is a distinct, metadata-only action: no raw JSONL
# files are rescanned, categories are not pooled, and no inference is made
# that four category downloads contain independent products or reviews.
Q17_CATEGORIES = ('Electronics', 'Beauty_and_Personal_Care', 'Appliances', 'All_Beauty')
Q17_METRICS = ('reviews.helpful_vote', 'metadata.price', 'metadata.rating_number')


def q17_compare_summaries(named_paths, folder):
    """Compare 2-4 separately generated quality_summary.json files.

    The four download categories are *not* assumed disjoint. Global thresholds
    and global outlier percentages are deliberately NOT recalculated here.
    Q15/Q16 have already fitted descriptive screens within each input file.
    """
    if not 2 <= len(named_paths) <= 4:
        raise ValueError('Q17 requires 2-4 named category summaries.')
    sources = {}
    for category, filepath in named_paths:
        if category not in Q17_CATEGORIES:
            raise ValueError('Unknown Q17 category: ' + category)
        if category in sources:
            raise ValueError('Repeated Q17 category: ' + category)
        source = Path(filepath).expanduser().resolve()
        data = json.loads(source.read_text(encoding='utf-8-sig'), parse_constant=no_constant)
        if not isinstance(data, dict) or not all(k in data for k in
                ('files','scope','q15_review_numeric_outliers','q16_metadata_numeric_outliers')):
            raise ValueError('Not a Q15/Q16 quality_summary.json: ' + str(source))
        # Native S25/S26 config runs stamp the selected input category. Refuse
        # to silently swap two category labels during Q17 comparisons.
        if data.get('source_category') is not None and data['source_category'] != category:
            raise ValueError('Q17 category label disagrees with recorded source_category: '+str(source))
        for role in ('reviews','metadata'):
            if not isinstance(data['files'].get(role), dict) or not all(
                    k in data['files'][role] for k in ('path','object_records','scan_mode','scanned_content_sha256')):
                raise ValueError('Missing Q17 provenance for '+role+': '+str(source))
        sources[category] = dict(path=source, data=data)
    order = [c for c in Q17_CATEGORIES if c in sources]
    missing_categories = [c for c in Q17_CATEGORIES if c not in sources]
    duplicate_inputs = []
    # Explicitly mark, but never assume, shared review/product membership.
    for role in ('reviews','metadata'):
        for i, a in enumerate(order):
            for b in order[i+1:]:
                left, right = (sources[c]['data']['files'][role] for c in (a,b))
                if (Path(left['path']).resolve() == Path(right['path']).resolve() or
                    left['scanned_content_sha256'] == right['scanned_content_sha256']):
                    duplicate_inputs.append({'role':role,'category_a':a,'category_b':b,
                                             'reason':'identical input path or scanned-byte SHA-256'})
    rows = []
    for category in order:
        entry=sources[category]
        data=entry['data']
        full=(data['scope']=='entire_supplied_dataset_files' and
              all(data['files'][r]['scan_mode']=='entire_supplied_file' for r in ('reviews','metadata')))
        q15=data['q15_review_numeric_outliers']
        q16=data['q16_metadata_numeric_outliers']
        if not all(k in q16 for k in ('price','rating_number')):
            raise ValueError('Incomplete Q16 summary in '+str(entry['path']))
        for metric, data_row, role in (
            ('reviews.helpful_vote',q15,'reviews'),
            ('metadata.price',q16['price'],'metadata'),
            ('metadata.rating_number',q16['rating_number'],'metadata')):
            if data_row.get('metric') != metric:
                raise ValueError('Metric mismatch in '+str(entry['path'])+': '+metric)
            n=(data_row.get('screened_valid_integer_rows') if role=='reviews' else
               data_row.get('screened_valid_rows'))
            flagged=(data_row.get('positive_upper_outlier_rows') if role=='reviews' else
                     data_row.get('primary_flagged_rows'))
            status=(data_row.get('status') if role=='reviews' else data_row.get('primary_status'))
            method=(data_row.get('method') if role=='reviews' else data_row.get('primary_screen'))
            valid=(data_row.get('valid_nonnegative_raw_integer_rows') if role=='reviews' else
                   data_row.get('domain_valid_rows'))
            zero=(data_row.get('zero_vote_rows') if role=='reviews' else data_row.get('zero_rows'))
            positive=(data_row.get('positive_vote_rows') if role=='reviews' else data_row.get('positive_rows'))
            if not all(type(v) is int and v >= 0 for v in (n,valid,zero,positive)) or n != zero+positive or n>valid:
                raise ValueError('Inconsistent Q15/Q16 denominators for '+category+' '+metric)
            if flagged is not None and (type(flagged) is not int or not 0 <= flagged <= n):
                raise ValueError('Invalid flagged count for '+category+' '+metric)
            if status == 'computed' and (flagged is None or not n):
                raise ValueError('Computed screen lacks usable count: '+category+' '+metric)
            if status != 'computed' and flagged is not None:
                raise ValueError('Unclassified screen has a flag count: '+category+' '+metric)
            outlier_percent=100*flagged/n if flagged is not None and n else None
            row=dict(category=category,metric=metric,
                source_summary_json=str(entry['path']), run_id=data.get('run_id'),
                input_role=role,input_path=data['files'][role]['path'],
                scanned_sha256=data['files'][role]['scanned_content_sha256'],
                scope=data['scope'],complete_file_scan=full,
                denominator_parsed_objects=data['files'][role]['object_records'],
                valid_under_domain=valid,screened_rows=n,unscreened_valid_rows=valid-n,
                zero_rows=zero,positive_rows=positive,
                zero_percent_of_screened=100*zero/n if n else None,
                p99_all_screened=data_row.get('p99_all_screened'),
                primary_method=method,primary_status=status,
                primary_flagged_rows=flagged,primary_flagged_percent_of_screened=outlier_percent,
                raw_lower_fence=data_row.get('raw_lower_fence'),
                raw_upper_fence=data_row.get('raw_upper_fence'),
                positive_upper_fence_raw=data_row.get('positive_upper_fence_raw'),
                warning='Descriptive per-file screening; percentages are not evidence of data corruption.')
            rows.append(row)
    summaries = []
    for metric in Q17_METRICS:
        group=[row for row in rows if row['metric']==metric]
        same_method=(len({row['primary_method'] for row in group})==1)
        all_full=(len(group)==4 and all(row['complete_file_scan'] for row in group))
        all_defined=all(row['primary_status']=='computed' for row in group)
        strictly_comparable=(all_full and all_defined and same_method and not duplicate_inputs)
        rates=[row['primary_flagged_percent_of_screened'] for row in group
               if row['primary_flagged_percent_of_screened'] is not None]
        summaries.append(dict(metric=metric,category_count=len(group),
            all_four_present=(len(group)==4),full_file_scans=all_full,
            common_screen=group[0]['primary_method'] if same_method else None,
            all_screens_computed=all_defined,duplicate_input_evidence=bool(duplicate_inputs),
            comparison_status=('descriptive_four_category_comparison' if strictly_comparable else
                'partial_or_noncomparable_do_not_claim_four_category_result'),
            percentage_point_spread=(max(rates)-min(rates) if strictly_comparable else None),
            minimum_category=(min(group,key=lambda r:r['primary_flagged_percent_of_screened'])['category']
                              if strictly_comparable else None),
            maximum_category=(max(group,key=lambda r:r['primary_flagged_percent_of_screened'])['category']
                              if strictly_comparable else None)))
    output = dict(q17_version=VERSION,comparison_type='Between-download descriptive only; NOT pooled or independent statistical inference',
        provided_categories=order,missing_categories=missing_categories,
        duplicate_input_evidence=duplicate_inputs,
        overlap_warning=('The Amazon download categories can share products/reviews. Category-level rates are descriptive; '
                         'shared rows/products have NOT been deduplicated across category files. '
                         'Do not apply independent-sample tests or make causal claims.'),
        thresholds_warning=('Each category keeps its OWN full-file threshold from Q15/Q16. '
                            'Comparing outlier rates does NOT mean the same numeric cutoff was used. '
                            'Global/snapshot thresholds must not be reused for training-time preprocessing.'),
        coverage_warning=('Only a complete full-file audit of all four categories supports a four-category description. '
                          'Partial scans, samples and undefined fences are individually visible and block spread claims.'),
        rows=rows,metric_comparisons=summaries)
    out=Path(folder).expanduser().resolve()
    if out.exists():
        raise FileExistsError('Q17 output directory already exists; choose a new directory: '+str(out))
    out.mkdir(parents=True,exist_ok=False)
    common.write_json(out/'q17_category_outlier_comparison.json',output)
    columns=('category','metric','scope','complete_file_scan','denominator_parsed_objects',
             'valid_under_domain','screened_rows','unscreened_valid_rows','zero_rows','positive_rows',
             'zero_percent_of_screened','p99_all_screened','primary_method','primary_status',
             'primary_flagged_rows','primary_flagged_percent_of_screened','raw_lower_fence',
             'raw_upper_fence','positive_upper_fence_raw','input_path','scanned_sha256','source_summary_json')
    common.write_csv(out/'q17_category_outlier_comparison.csv',columns,
                     [tuple(r.get(k) for k in columns) for r in rows])
    text=['# Q17: Between-category numeric outlier diagnostics','',
          'This is a postprocessing comparison of per-category Q15 and Q16 reports; no raw data are rescanned.','',
          'Provided categories: '+', '.join(order),
          'Missing categories: '+(', '.join(missing_categories) if missing_categories else 'none'),'',
          'Each row preserves its own reference denominator, method, percentile, and category-specific threshold.','',
          '| Metric | Category | Screened | Flagged | % of screened | Screen status | Full input |',
          '|---|---|---:|---:|---:|---|---|']
    for row in rows:
        f=lambda v: 'undefined' if v is None else str(round(v,5)) if isinstance(v,float) else str(v)
        text.append('| '+ ' | '.join([row['metric'],row['category'],f(row['screened_rows']),
            f(row['primary_flagged_rows']),f(row['primary_flagged_percent_of_screened']),
            str(row['primary_status']),str(row['complete_file_scan'])])+' |')
    text.extend(['','## Interpretation',''])
    for m in summaries:
        text.append('- '+m['metric']+': '+m['comparison_status']+
                    ('; rate spread = '+str(round(m['percentage_point_spread'],5))+' percentage points.'
                     if m['percentage_point_spread'] is not None else '; spread intentionally unavailable.'))
    text.extend(['',output['overlap_warning'], '',output['thresholds_warning'],'',output['coverage_warning'],
                 '', 'No raw files are modified. No new outliers are labeled in this step.'])
    (out/'Q17_CATEGORY_COMPARISON_REPORT_EN.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return output


def q17_cli_pairs(entries):
    result=[]
    for item in entries:
        category,sep,path=item.partition('=')
        if not sep or not path.strip():
            raise ValueError('Q17 input must be Category=/path/to/quality_summary.json: '+item)
        result.append((category,Path(path)))
    return result



# Final S25/S26 structure-audit configuration compatibility. A structure config
# contains four input pairs and uses s/ms/us/ns, unlike the legacy quality
# single-pair config. Keep the quality audit per-category, with an explicit
# category; do not silently scan all datasets or silently choose Electronics.
_NATIVE_UNIT_TO_QUALITY = {'s':'seconds','ms':'milliseconds',
                           'us':'microseconds','ns':'nanoseconds'}
_QUALITY_TO_NATIVE = {v:k for k,v in _NATIVE_UNIT_TO_QUALITY.items()}


def q26_epoch_boundary(utc_dt, unit):
    if utc_dt.tzinfo is None or utc_dt.utcoffset() is None:
        raise ValueError('Timestamp boundary must have an explicit timezone')
    duration = utc_dt.astimezone(timezone.utc) - EPOCH
    micros = (duration.days * 86400 + duration.seconds) * 1_000_000 + duration.microseconds
    factor = common.TIMESTAMP_UNITS[_QUALITY_TO_NATIVE[unit]]
    value = micros * factor // 1_000_000
    if micros * factor % 1_000_000:
        raise ValueError('Reference boundary is not exactly expressible in '+unit)
    return value


def q26_prepare_config(config, root, category=None):
    """Translate FINAL four-category structure config to existing quality inputs.

    Do not touch user files. Reject conflicts between structure and quality
    time contracts. The legacy flat quality config remains supported.
    """
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a JSON object')
    translated = dict(config)
    if 'datasets' in config:
        if category is None:
            raise ValueError('Four-category structure config requires --category; no default is safe')
        # Validate all categories and roles just as the final structure audit does.
        all_paths = common.collect_dataset_paths(config, root)
        if category not in all_paths:
            raise ValueError('Unsupported category: '+str(category))
        if 'reviews_path' in config or 'metadata_path' in config:
            raise ValueError('Ambiguous mixed flat and four-category configuration')
        translated['reviews_path'] = str(all_paths[category]['reviews'])
        translated['metadata_path'] = str(all_paths[category]['metadata'])
        translated['output_directory'] = config.get('quality_output_directory', 'reports/phase1/quality')
        translated['index_directory'] = config.get('quality_index_directory', 'data/derived/quality')
        translated['quality_input_category'] = category
        original = config.get('quality_settings', {})
        if not isinstance(original, dict):
            raise ValueError('quality_settings must be an object')
        qsettings = dict(original)
        native = config.get('timestamp_unit', common.DEFAULT_TIMESTAMP_UNIT)
        if native not in _NATIVE_UNIT_TO_QUALITY:
            raise ValueError('Unsupported final structure timestamp_unit: '+repr(native))
        expected = _NATIVE_UNIT_TO_QUALITY[native]
        if 'timestamp_unit' in qsettings:
            override = _NATIVE_UNIT_TO_QUALITY.get(qsettings['timestamp_unit'], qsettings['timestamp_unit'])
            if override != expected:
                raise ValueError('Quality timestamp_unit conflicts with structure timestamp_unit')
        qsettings['timestamp_unit'] = expected
        start = common.explicit_utc_datetime(config.get('reference_start',common.DEFAULT_REFERENCE_START),'reference_start')
        end = common.explicit_utc_datetime(config.get('reference_end_exclusive',common.DEFAULT_REFERENCE_END_EXCLUSIVE),'reference_end_exclusive')
        if start >= end:
            raise ValueError('reference_start must precede reference_end_exclusive')
        for key, date in (('review_timestamp_min',start),('review_timestamp_max_exclusive',end)):
            computed = q26_epoch_boundary(date,expected)
            if key in qsettings and qsettings[key] != computed:
                raise ValueError('Quality '+key+' conflicts with structure reference dates')
            qsettings[key] = computed
        if 'as_of_utc' in config:
            as_of = common.explicit_utc_datetime(config['as_of_utc'],'as_of_utc').isoformat()
            if 'q13_audit_as_of_utc' in qsettings and common.explicit_utc_datetime(qsettings['q13_audit_as_of_utc'],'q13_audit_as_of_utc') != common.explicit_utc_datetime(as_of,'as_of_utc'):
                raise ValueError('Quality as-of conflicts with structure as_of_utc')
            qsettings['q13_audit_as_of_utc'] = as_of
        for old_key, new_key in (('batch_size','batch_size'),('progress_every_lines','progress_every_lines')):
            if old_key in config and new_key not in qsettings:
                qsettings[new_key] = config[old_key]
        translated['quality_settings'] = qsettings
        return translated
    if category is not None:
        raise ValueError('--category requires the final structure config with datasets')
    return translated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[1]/"configs/quality_electronics.json")
    parser.add_argument("--preview-lines",type=int,default=0)
    parser.add_argument("--category",choices=common.PROJECT_CATEGORIES,
                        help='Required with final structure four-category config (datasets).')
    parser.add_argument("--q17-compare",nargs='+',metavar='CATEGORY=SUMMARY_JSON',
                        help='Compare 2-4 existing per-category Q15/Q16 summaries without rescanning raw data.')
    parser.add_argument("--q17-output",type=Path,
                        help='New, nonexistent directory for Q17 comparison reports.')
    args = parser.parse_args(argv)
    if args.q17_compare is not None:
        if args.preview_lines or args.q17_output is None:
            parser.error('Q17 comparison requires --q17-output and cannot use --preview-lines.')
        result=q17_compare_summaries(q17_cli_pairs(args.q17_compare),args.q17_output)
        print('Q17 reports written to:',args.q17_output.resolve())
        for metric in result['metric_comparisons']:
            print(metric['metric'],metric['comparison_status'])
        return 0
    if args.q17_output is not None:
        parser.error('--q17-output is only available with --q17-compare')
    path = args.config.resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    root = common.resolve(path.parent,config.get("project_root",".."))
    config = q26_prepare_config(config, root, args.category)
    settings = {
        "batch_size":25000,"progress_every_lines":100000,"example_lines_per_rule":5,
        "timestamp_unit":"milliseconds", "review_timestamp_min":830908800000,
        "review_timestamp_max_exclusive":1696118400000,
        "very_short_text_words":3,"very_long_text_characters":10000,
        "q03_min_reviews_for_5_plus_10":15,
        "potential_missing_strings":["none","null","n/a","nan","unknown"],
    }
    settings.update(config.get("quality_settings",{}))
    # Defaults are epoch milliseconds. Convert omitted bounds to the configured
    # time unit exactly; native structure configs supply matching bounds above.
    explicit_quality = config.get('quality_settings', {})
    if settings['timestamp_unit'] not in TIME_UNITS:
        raise ValueError('timestamp_unit must be seconds/milliseconds/microseconds/nanoseconds')
    factor = common.TIMESTAMP_UNITS[TIME_UNITS[settings['timestamp_unit']]]
    for key in ('review_timestamp_min','review_timestamp_max_exclusive'):
        if key not in explicit_quality:
            default_ms = settings[key]
            settings[key] = default_ms * factor // 1000
    # Freeze as-of once per execution; the user may supply a reproducible UTC
    # reference. Naive datetimes and invalid reference bounds are rejected.
    if settings.get('q13_audit_as_of_utc') is None:
        settings['q13_audit_as_of_utc'] = datetime.now(timezone.utc).isoformat()
    settings['q13_audit_as_of_utc'] = q13_utc_reference(settings['q13_audit_as_of_utc']).isoformat()
    for key in ('review_timestamp_min','review_timestamp_max_exclusive'):
        if not integer(settings[key]) or q13_datetime(settings[key],settings['timestamp_unit']) is None:
            raise ValueError(key+' must be a representable integer epoch in timestamp_unit')
    for name in ("very_short_text_words", "very_long_text_characters"):
        if type(settings[name]) is not int or settings[name] < 0:
            raise ValueError("Q23 requires a nonnegative integer "+name)
    if args.preview_lines < 0 or settings["batch_size"] <= 0 or settings["progress_every_lines"] < 0 or settings["example_lines_per_rule"] < 0:
        raise ValueError("Invalid negative limit or nonpositive batch size.")
    threshold = settings["q03_min_reviews_for_5_plus_10"]
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
        raise ValueError("q03_min_reviews_for_5_plus_10 must be a positive integer")
    if settings["review_timestamp_min"] >= settings["review_timestamp_max_exclusive"]:
        raise ValueError("Timestamp coverage limits are not ordered.")
    if config.get("input_role","dataset") not in ("sample","dataset"):
        raise ValueError("input_role must be sample or dataset.")
    inputs = {role:common.resolve(root,config[key]) for role,key in (("reviews","reviews_path"),("metadata","metadata_path"))}
    for source in inputs.values():
        if not source.is_file():
            raise FileNotFoundError(source)
        if not (source.name.lower().endswith('.jsonl') or source.name.lower().endswith('.jsonl.gz')):
            raise ValueError('Expected JSONL/JSONL.gz input: '+str(source))
    if inputs["reviews"] == inputs["metadata"]:
        raise ValueError("Review and metadata files must differ.")
    # Metadata from a final structure config records the selected source file
    # category, avoiding confusion with metadata.main_category.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8]
    folder=common.resolve(root,config.get("output_directory","reports/phase1/quality"))/run_id
    work=common.resolve(root,config.get("index_directory","data/derived/quality"))/run_id
    folder.mkdir(parents=True,exist_ok=False)
    work.mkdir(parents=True,exist_ok=False)
    scope="head_preview" if args.preview_lines else ("sample_files" if config.get("input_role")=="sample" else "entire_supplied_dataset_files")
    status=dict(status="running",run_id=run_id,scope=scope,started_utc=common.utc_now())
    common.write_json(folder/"RUN_STATUS.json",status)
    db=None
    print(f"Quality audit: {run_id}\nScope: {scope}\nOutputs: {folder}",flush=True)
    try:
        db=init_db(work/"quality_index.sqlite")
        files={role:analyze_file(source,role,db,work,settings,args.preview_lines) for role,source in inputs.items()}
        print("Computing exact duplicate counts and numeric diagnostics...",flush=True)
        q25_accumulator=files["reviews"].pop("q25_rating_text_accumulator")
        files["metadata"].pop("q25_rating_text_accumulator")
        q24_result=files["reviews"].pop("q24_review_unicode_audit")
        files["metadata"].pop("q24_review_unicode_audit")
        metrics=[row[0] for row in db.execute("SELECT DISTINCT metric FROM numeric_values ORDER BY metric")]
        event=db.execute("SELECT COUNT(*),COALESCE(SUM(CASE WHEN n>1 THEN 1 ELSE 0 END),0),COALESCE(SUM(CASE WHEN varied=1 THEN 1 ELSE 0 END),0) FROM event_candidates").fetchone()
        summary=dict(audit_version=VERSION,run_id=run_id,scope=scope,started_utc=status["started_utc"],completed_utc=common.utc_now(),
                     runtime=dict(python=platform.python_version(),sqlite=sqlite3.sqlite_version,platform=platform.platform()),
                     effective_settings=dict(config=config,quality_settings=settings,preview_lines=args.preview_lines),
                     files=files,numeric_diagnostics={name:numeric_summary(db,name) for name in metrics},
                     q15_review_numeric_outliers=q15_review_numeric_outliers(db,files["reviews"]),
                     q16_metadata_numeric_outliers=q16_metadata_numeric_outliers(db,files["metadata"]),
                     candidate_event_groups=dict(groups=event[0],repeated_groups=event[1],groups_with_different_record_fingerprints=event[2],
                         interpretation="SHA-256 grouping of exact user_id, parent_asin and timestamp. Candidate events only, not proven duplicates; hashes do not resolve event identity."),
                     index_path=str(work/"quality_index.sqlite"),output_path=str(folder),
                     source_provenance=config.get("source_provenance",{}),
                     source_category=config.get('quality_input_category'),
                     structure_audit_integration=dict(
                         module_version=common.VERSION,
                         timestamp_unit=TIME_UNITS[settings['timestamp_unit']],
                         reference_start=q13_datetime(settings['review_timestamp_min'],settings['timestamp_unit']).isoformat(),
                         reference_end_exclusive=q13_datetime(settings['review_timestamp_max_exclusive'],settings['timestamp_unit']).isoformat(),
                         note='Quality uses stricter JSON decoding than structure: duplicate object keys and nonfinite floating-point literals are rejected. Raw input remains unchanged.'))
        summary['q22_review_text_lengths'] = q22_review_text_lengths(db, files['reviews'])
        summary['q24_review_unicode_audit'] = q24_result
        summary['q25_rating_text_association'] = q25_accumulator.result(
            files['reviews']['object_records'],files['reviews']['q11_review_rating_validity']['valid_star_counts'])
        summary['q23_review_short_long_text'] = q23_short_long_review_text(
            db, files['reviews'], summary['q22_review_text_lengths'], settings)
        summary['q18_outlier_vs_invalid'] = q18_outlier_vs_invalid(summary)
        summary['q21_initial_decisions'] = q21_initial_decisions(summary)
        export(folder,summary)
        db.close()
        db=None
        # Spooling is needed only to verify candidates while streaming gzip.
        for role,source in inputs.items():
            if source.suffix.lower()==".gz":
                (work/(role+"_decompressed.jsonl")).unlink()
        status.update(status="completed",completed_utc=common.utc_now())
        common.write_json(folder/"RUN_STATUS.json",status)
        print(f"Completed. Send this file:\n{folder/'quality_summary.json'}",flush=True)
        return 0
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",error_type=type(error).__name__,error=str(error),ended_utc=common.utc_now())
        common.write_json(folder/"RUN_STATUS.json",status)
        raise
    finally:
        if db is not None:
            db.close()


if __name__=="__main__":
    raise SystemExit(main())
