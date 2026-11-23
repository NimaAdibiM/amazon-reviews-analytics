                                                                                 
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import gzip
import hashlib
from itertools import zip_longest
import json
import math
import sys
from pathlib import Path

from split_scale import apply_scaler, assign_split

ROOT = Path(__file__).resolve().parent


def verify(output, phase_one=None):
    csv.field_size_limit(sys.maxsize)
    manifest = json.loads((output / "manifest.json").read_text())
    for name, item in manifest.items():
        path = output / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Output integrity failure: {name}")
    summary = json.loads((output / "run_summary.json").read_text())
    contract = json.loads((output / "feature_contract.json").read_text())
    scaler = json.loads((output / "scaler_train.json").read_text())
    if scaler["fit_split"] != "train":
        raise ValueError("Scaler was not fitted on train")
    if scaler.get("pre_transforms", {}) != contract.get("numeric_pre_transforms_before_scaling", {}):
        raise ValueError("Scaler pre-transforms differ from the feature contract")
    for field, params in scaler["features"].items():
        if params["variance"] == 0 and params["scale"] != 1.0:
            raise ValueError(f"{field}: a constant feature must use scale 1")
    predictor_fields = {field for fields in contract["blocks"].values() for field in fields}
    feature_schema = {*contract["key_columns"], *predictor_fields}
    seen = set()
    counts = Counter()
    targets = Counter()
    observed_early = Counter()
    names = ("features", "features_scaled", "text_documents", "labels", "metadata_snapshot_audit_only")
    with ExitStack() as stack:
        readers = [csv.DictReader(stack.enter_context(
            gzip.open(output / f"{name}.csv.gz", "rt", encoding="utf-8", newline=""))) for name in names]
        if any(set(reader.fieldnames or []) != feature_schema for reader in readers[:2]):
            raise ValueError("Numeric predictor schema differs from feature contract")
        for i, rowset in enumerate(zip_longest(*readers)):
            if any(row is None for row in rowset):
                raise ValueError("Output tables have different row counts")
            keys = [tuple(row[k] for k in contract["key_columns"]) for row in rowset]
            category, parent, split = keys[0]
            if any(key != keys[0] for key in keys) or keys[0][:2] in seen or assign_split(parent) != split:
                raise ValueError("Output key alignment or split assignment failed")
            seen.add(keys[0][:2])
            counts[category, split] += 1
            label = rowset[3]
            low = int(label["future_low_count"])
            if not (0 <= low <= 10 and float(label["future_low_share"]) == low / 10
                    and int(label["t5_utc_ms"]) < int(label["t15_utc_ms"])):
                raise ValueError("Invalid label or label maturity")
            targets[category, low] += 1
            raw, scaled, document, _, _ = rowset
            for field in predictor_fields:
                if bool(raw[field]) != bool(scaled[field]) or (
                    scaled[field] and not math.isfinite(float(scaled[field]))
                ):
                    raise ValueError(f"{field}: missingness changed or scaled value is not finite")
            observed_early[category, "titles_after_cleaning", "empty"] += int(raw["title_missing_count"])
            missing_bodies = int(raw["body_missing_count"])
            observed_early[category, "bodies_after_cleaning", "empty"] += missing_bodies
            observed_early[category, "products_with_missing_body", "yes" if missing_bodies else "no"] += 1
            observed_early[category, "documents", "present" if document["text_document"] else "empty"] += 1
            if i % 997 == 0:
                transformed = apply_scaler(rowset[0], scaler)
                if any(str(value) != rowset[1][field] for field, value in transformed.items()):
                    raise ValueError("Scaled values differ from the frozen transformation")
    prior = json.loads((phase_one / "manifest.json").read_text()) if phase_one else None
    early_csv = {}
    with (output / "missingness_by_field.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["role"] != "early_reviews":
                continue
            key = (row["category"], row["field"], row["state"])
            total, n = int(row["parsed_denominator"]), int(row["rows"])
            if key in early_csv or not 0 <= n <= total or (total and float(row["share"]) != n / total):
                raise ValueError(f"Invalid early missingness row: {key}")
            early_csv[key] = (n, total)
    expected_early = {}
    for category, splits in summary["category_splits"].items():
        if any(counts[category, split] != n for split, n in splits.items()):
            raise ValueError(f"{category}: split counts do not match summary")
        products = sum(splits.values())
        early = summary["early_missingness"][category]
        for field, states in early.items():
            total = products if field in {"documents", "products_with_missing_body"} else 5 * products
            if sum(states.values()) != total:
                raise ValueError(f"{category}/{field}: early field counts do not reconcile")
            for state, n in states.items():
                expected_early[category, field, state] = (n, total)
        for field in ("titles_after_cleaning", "bodies_after_cleaning", "documents", "products_with_missing_body"):
            for state, n in early[field].items():
                actual = observed_early[category, field, state]
                if field.endswith("_after_cleaning") and state == "present":
                    actual = 5 * products - observed_early[category, field, "empty"]
                if n != actual:
                    raise ValueError(f"{category}/{field}/{state}: early available text does not reconcile")
        if prior:
            for role in ("reviews", "metadata"):
                if summary["inputs"][category][role]["source_sha256"] != prior["inputs"][category][role]["scanned_content_sha256"]:
                    raise ValueError(f"{category}/{role}: source hash differs from phase one")
            cohort = json.loads((phase_one / category / "cohort.json").read_text())
            expected = cohort["policies"]["retain"]["future_low_share"]["bins"]
            for entry in expected:
                low = round(entry["lower"] * 10)
                if targets[category, low] != entry["numerator"]:
                    raise ValueError(f"{category}/{low}: target count differs from phase-one retain policy")
    if len(seen) != sum(summary["split_totals"].values()):
        raise ValueError("Sample count differs from run summary")
    if early_csv != expected_early:
        raise ValueError("Early CSV missingness differs from run summary")
    return {"output_files_verified": len(manifest), "rows_verified": len(seen),
            "early_missingness_rows_verified": len(early_csv),
            "numeric_values_finite_and_missingness_preserved": True,
            "phase1_source_hash_and_retain_histogram_matched": bool(prior)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output")
    parser.add_argument("--phase-one", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.out, args.phase_one), indent=2))
