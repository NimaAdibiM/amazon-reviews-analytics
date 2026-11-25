"""Build phase-two product features, isolated labels, and data-quality evidence."""

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import numpy as np

from features import EARLY_REVIEW_COUNT, FEATURE_GROUPS, FEATURE_PRE_TRANSFORMS, SCALABLE_FIELDS, make_features
from ingest import STATES, ingest_category
from metadata import CATEGORIES, read_for_products, scan_metadata
from split_scale import SPLIT_FRACTIONS, SPLIT_HASH_SPEC, SPLIT_SEED, apply_scaler, assign_split, fit_scaler

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "data" / "raw"
KEYS = ("category", "parent_asin", "split")
NUMERIC = tuple(name for group in FEATURE_GROUPS.values() for name in group)
EARLY_RAW_FIELDS = ("titles", "bodies", "verified")
EARLY_FIELD_STATES = {
    "titles": (*STATES, "invalid_unicode"),
    "bodies": (*STATES, "invalid_unicode"),
    "verified": STATES,
    "titles_after_cleaning": ("present", "empty"),
    "bodies_after_cleaning": ("present", "empty"),
    "documents": ("present", "empty"),
    "products_with_missing_body": ("yes", "no"),
}
EARLY_PRODUCT_FIELDS = {"documents", "products_with_missing_body"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def cohorts(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def writer(stack, directory, filename, fields):
    stream = stack.enter_context(gzip.open(directory / filename, "wt", encoding="utf-8", newline="", compresslevel=3))
    result = csv.DictWriter(stream, fieldnames=fields)
    result.writeheader()
    return result


def audit_early(output, early_states, splits):
    for category, fields in early_states.items():
        products = sum(splits[category].values())
        for field, counts in fields.items():
            total = products if field in EARLY_PRODUCT_FIELDS else EARLY_REVIEW_COUNT * products
            if sum(counts.values()) != total:
                raise ValueError(f"{category}/early_{field}: denominator does not reconcile")
            for state, count in counts.items():
                output.writerow((category, "early_reviews", field, state, count, total, count / total if total else ""))


def audit_fields(directory, reports, early_states, splits):
    with (directory / "missingness_by_field.csv").open("w", encoding="utf-8", newline="") as handle:
        output = csv.writer(handle)
        output.writerow(("category", "role", "field", "state", "rows", "parsed_denominator", "share"))
        for category, stages in reports.items():
            for role, fields, total in (
                ("reviews", stages["reviews"]["review_field_states"], stages["reviews"]["counts"]["parsed_reviews"]),
                ("metadata", stages["metadata"]["field_states"], stages["metadata"]["counts"]["objects"]),
            ):
                for field, counts in fields.items():
                    if sum(counts.values()) != total:
                        raise ValueError(f"{category}/{role}/{field}: missingness denominator does not reconcile")
                    for state, count in counts.items():
                        output.writerow((category, role, field, state, count, total, count / total if total else ""))
        audit_early(output, early_states, splits)


def publish(directory, reviews, metadata, reports):
    splits = {category: Counter() for category in CATEGORIES}
    early_states = {
        category: {field: Counter(dict.fromkeys(states, 0)) for field, states in EARLY_FIELD_STATES.items()}
        for category in CATEGORIES
    }
    snapshots = {category: Counter() for category in CATEGORIES}
    parent_categories = {}
    labels = {category: Counter() for category in CATEGORIES}
    with ExitStack() as stack:
        numeric = writer(stack, directory, "features.csv.gz", KEYS + NUMERIC)
        text = writer(stack, directory, "text_documents.csv.gz", KEYS + ("text_document",))
        target = writer(stack, directory, "labels.csv.gz",
                        KEYS + ("t5_utc_ms", "t15_utc_ms", "future_low_count", "future_low_share"))
        snapshot = writer(stack, directory, "metadata_snapshot_audit_only.csv.gz",
                          KEYS + ("metadata_status", "price_snapshot", "price_state",
                                  "rating_average_snapshot", "rating_count_snapshot"))
        for category in CATEGORIES:
            path = reviews / category / "reviews_cohort.jsonl"
            parents = {row["parent_asin"] for row in cohorts(path)}
            side = read_for_products(metadata / f"{category}.csv.gz", parents)
            for row in cohorts(path):
                parent = row["parent_asin"]
                if parent not in parents:
                    raise ValueError(f"{category}: repeated or unexpected parent")
                parents.remove(parent)
                split = assign_split(parent)
                parent_categories[parent] = parent_categories.get(parent, 0) + 1
                keys = dict(category=category, parent_asin=parent, split=split)
                early = [
                    dict(rating=rating, timestamp=stamp, title=title, text=body, verified_purchase=verified)
                    for rating, stamp, title, body, verified in zip(
                        row["early_ratings"], row["early_timestamps"], row["early_titles"],
                        row["early_bodies"], row["early_verified"], strict=True)
                ]
                values = make_features(early)
                document = values.pop("text_document")
                numeric.writerow({**keys, **values})
                text.writerow({**keys, "text_document": document})
                count = row["n_low_future"]
                if not (0 <= count <= 10 and row["t5"] < row["t15"]):
                    raise ValueError(f"{category}: invalid product target")
                target.writerow({**keys, "t5_utc_ms": row["t5"], "t15_utc_ms": row["t15"],
                                 "future_low_count": count, "future_low_share": count / 10})
                labels[category][count] += 1
                states = early_states[category]
                for field in EARLY_RAW_FIELDS:
                    states[field].update(row[f"early_{field}_states"])
                for field, missing in (("titles", values["title_missing_count"]),
                                       ("bodies", values["body_missing_count"])):
                    states[f"{field}_after_cleaning"].update(empty=missing, present=EARLY_REVIEW_COUNT - missing)
                states["documents"]["present" if document else "empty"] += 1
                states["products_with_missing_body"]["yes" if values["body_missing_count"] else "no"] += 1
                item = side.get(parent)
                if item is None:
                    status = "ambiguous" if parent in side else "no_metadata"
                    snapshot.writerow({**keys, "metadata_status": status, "price_state": "not_available"})
                    snapshots[category][status] += 1
                else:
                    snapshot.writerow({**keys, **{k: v for k, v in item.items() if k not in ("category", "parent_asin")}})
                    snapshots[category]["unique_metadata"] += 1
                    snapshots[category]["price_" + item["price_state"]] += 1
                splits[category][split] += 1
            expected = reports[category]["reviews"]["counts"]["eligible_products"]
            if parents or sum(splits[category].values()) != expected:
                raise ValueError(f"{category}: exported row count differs from cohort")
            print(f"[{category}] exported {expected:,} rows", flush=True)
    scaler = fit_scaler(directory / "features.csv.gz", directory / "scaler_train.json", SCALABLE_FIELDS,
                        pre_transforms=FEATURE_PRE_TRANSFORMS,
                        passthrough_fields=(field for field in NUMERIC if field not in SCALABLE_FIELDS))
    with ExitStack() as stack:
        source = stack.enter_context(gzip.open(directory / "features.csv.gz", "rt", encoding="utf-8", newline=""))
        scaled = writer(stack, directory, "features_scaled.csv.gz", KEYS + NUMERIC)
        for row in csv.DictReader(source):
            scaled.writerow(apply_scaler(row, scaler))
    audit_fields(directory, reports, early_states, splits)
    write_json(directory / "feature_contract.json", {
        "key_columns": KEYS, "blocks": FEATURE_GROUPS, "scalable_fields": SCALABLE_FIELDS,
        "numeric_pre_transforms_before_scaling": FEATURE_PRE_TRANSFORMS,
        "text_normalization": "Decode HTML entities, replace HTML br tags with spaces, trim, replace lone surrogates.",
        "main_rating_only_features": FEATURE_GROUPS["rating"],
        "main_combined": {"numeric": FEATURE_GROUPS["rating"], "text": "text_document"},
        "optional_blocks": ["timing", "purchase", "text"],
        "split_seed": SPLIT_SEED, "split_rule": SPLIT_HASH_SPEC, "split_proportions": SPLIT_FRACTIONS,
        "targets_and_dates": "labels.csv.gz", "snapshot_only": "metadata_snapshot_audit_only.csv.gz",
    })
    return {"category_splits": splits, "early_missingness": early_states, "metadata_join": snapshots,
            "future_low_count_histogram": labels, "unique_parents": len(parent_categories),
            "parents_with_multiple_categories": sum(n > 1 for n in parent_categories.values()),
            "split_totals": scaler["split_row_counts"],
            "training_feature_diagnostics": scaler["training_feature_diagnostics"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "output")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="Rebuild review-stage checkpoints")
    parser.add_argument("--reuse-stages", action="store_true",
                        help="Reuse complete stage artifacts after checking their input identities")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        parser.error("--out must be inside the phase 2 directory")
    out.mkdir(parents=True, exist_ok=True)
    reviews = out / "review_stage"
    metadata = out / "metadata_inventory"
    reports = {}
    for category in CATEGORIES:
        raw_reviews = args.data / f"{category}.jsonl"
        review_audit = ingest_category(raw_reviews, category, reviews / category,
                                       workers=args.workers, force=args.force)
        raw_meta = (args.data / f"meta_{category}.jsonl").resolve()
        meta_path = metadata / f"{category}.json"
        if args.reuse_stages and meta_path.is_file() and (metadata / f"{category}.csv.gz").is_file():
            meta_audit = json.loads(meta_path.read_text())
            with raw_meta.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != meta_audit["source_sha256"]:
                raise ValueError(f"{category}: metadata input hash changed")
        else:
            meta_audit = scan_metadata(raw_meta, category, metadata)
        with raw_reviews.open("rb") as source:
            review_digest = hashlib.file_digest(source, "sha256").hexdigest()
        reports[category] = {"reviews": {**review_audit, "source_sha256": review_digest},
                             "metadata": meta_audit}
    with TemporaryDirectory(prefix="export-", dir=out) as staging:
        stage = Path(staging)
        summary = publish(stage, reviews, metadata, reports)
        write_json(stage / "run_summary.json", {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "entire_supplied_dataset_files",
            "python": platform.python_version(), "numpy": np.__version__,
            "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                            for name in ("prepare.py", "ingest.py", "metadata.py", "features.py", "split_scale.py")},
            "inputs": reports, **summary,
        })
        files = {}
        for path in sorted(stage.iterdir()):
            with path.open("rb") as source:
                files[path.name] = {"bytes": path.stat().st_size,
                                    "sha256": hashlib.file_digest(source, "sha256").hexdigest()}
        write_json(stage / "manifest.json", files)
        (out / "manifest.json").unlink(missing_ok=True)
        for path in sorted(stage.iterdir(), key=lambda p: p.name == "manifest.json"):
            path.replace(out / path.name)
    print(f"Completed four categories. Manifest: {out / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
