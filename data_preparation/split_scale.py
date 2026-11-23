                                                                            

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile


SPLIT_SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_HASH_SPEC = (
    "SHA-256 of the ASCII seed, a NUL byte, and the unmodified UTF-8 parent_asin. "
    "Interpret the first 8 bytes as an unsigned big-endian integer. "
    "Integers below floor(0.70 * 2^64) are train, those below "
    "floor(0.85 * 2^64) are validation, and the rest are test."
)
_HASH_SPACE = 1 << 64
_TRAIN_END = 70 * _HASH_SPACE // 100
_VALIDATION_END = 85 * _HASH_SPACE // 100
_SEED_PREFIX = str(SPLIT_SEED).encode("ascii") + b"\0"
ZERO_DOMINANCE_THRESHOLD = 0.99


def assign_split(parent_asin: str) -> str:
                                                                      
    if not isinstance(parent_asin, str) or not parent_asin.strip():
        raise ValueError("parent_asin must be a nonblank string")
    digest = hashlib.sha256(_SEED_PREFIX + parent_asin.encode("utf-8")).digest()
    rank = int.from_bytes(digest[:8], "big")
    if rank < _TRAIN_END:
        return "train"
    if rank < _VALIDATION_END:
        return "validation"
    return "test"


@dataclass(slots=True)
class _Moments:
    count: int = 0
    zeros: int = 0
    mean: float = 0.0
    squared_deviations: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.zeros += value == 0
        deviation = value - self.mean
        self.mean += deviation / self.count
        self.squared_deviations += deviation * (value - self.mean)

    def summarize(self, training_rows: int) -> dict:
        summary = {
            "observed_training_rows": self.count,
            "missing_training_rows": training_rows - self.count,
            "zero_training_rows": self.zeros,
        }
        if self.count == 0:
            return {**summary, "mean": None, "variance": None, "scale": None}
        variance = max(0.0, self.squared_deviations / self.count)
        if not math.isfinite(self.mean) or not math.isfinite(self.squared_deviations):
            raise ValueError("Training values exceed the scaler's numeric range")
        return {
            **summary,
            "mean": self.mean,
            "variance": variance,
            "scale": math.sqrt(variance) if variance else 1.0,
        }


def _number(value: str | float | int | None, field: str) -> float | None:
    if value is None or isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{field}: expected a finite number or a blank") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field}: expected a finite number or a blank")
    return number


def _transform(number: float, method: str) -> float:
    if method == "identity":
        return number
    if method == "signed_log1p":
        return math.copysign(math.log1p(abs(number)), number)
    if method == "log1p" and number >= 0:
        return math.log1p(number)
    raise ValueError(f"Cannot apply {method} to {number}. log1p requires a nonnegative value.")


def fit_scaler(features_csv: Path, output: Path, scalable_fields: Iterable[str], *,
               pre_transforms: Mapping[str, str] | None = None,
               passthrough_fields: Iterable[str] = ()) -> dict:
                                                                                        
    features_csv = Path(features_csv)
    output = Path(output)
    if features_csv.resolve() == output.resolve():
        raise ValueError("Scaler output must differ from the feature input")
    fields = tuple(scalable_fields)
    passthrough = tuple(passthrough_fields)
    audited = fields + passthrough
    if len(set(audited)) != len(audited) or {"category", "split", "parent_asin"}.intersection(audited):
        raise ValueError("Select unique numeric features and omit split and parent_asin")
    transforms = dict(pre_transforms or {})
    if set(transforms) - set(fields) or set(transforms.values()) - {"log1p", "signed_log1p"}:
        raise ValueError("Pre-transforms must be log1p or signed_log1p on scaled fields")

    moments = {field: _Moments() for field in audited}
    counts = dict.fromkeys(SPLIT_FRACTIONS, 0)
    opener = gzip.open if features_csv.suffix == ".gz" else open
    with opener(features_csv, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        required = {"split", *audited}
        if not required.issubset(headers) or len(set(headers)) != len(headers):
            raise ValueError("Feature CSV must have unique columns and include split and all selected features")
        verify_groups = "parent_asin" in headers
        for row in reader:
            split = row["split"]
            if split not in counts or None in row or None in row.values():
                raise ValueError(f"Invalid feature row ending on CSV line {reader.line_num}")
            if verify_groups and assign_split(row["parent_asin"]) != split:
                raise ValueError(f"Parent split does not match the frozen rule at CSV line {reader.line_num}")
            counts[split] += 1
            if split == "train":
                for field, accum in moments.items():
                    number = _number(row[field], field)
                    if number is not None:
                        accum.add(_transform(number, transforms.get(field, "identity")))

    rows = sum(counts.values())
    profile = {field: accum.summarize(counts["train"]) for field, accum in moments.items()}
    artifact = {
        "schema_version": 2,
        "feature_file": features_csv.name,
        "fit_split": "train",
        "method": "Apply any fixed pre-transform, then z-score using training population variance (ddof=0).",
        "pre_transforms": transforms,
        "missing_policy": "Ignore blank training values for fitting and preserve blanks when transforming.",
        "constant_policy": "Use scale 1 for zero training variance.",
        "unfitted_policy": "A feature with no observed training value has null parameters.",
        "split_seed": SPLIT_SEED,
        "split_hash_spec": SPLIT_HASH_SPEC,
        "expected_split_fractions": SPLIT_FRACTIONS,
        "split_row_counts": counts,
        "observed_split_row_fractions": {
            split: count / rows if rows else None for split, count in counts.items()
        },
        "verified_group_assignment_rows": rows if verify_groups else None,
        "features": {field: profile[field] for field in fields},
        "passthrough_feature_profiles": {field: profile[field] for field in passthrough},
        "training_feature_diagnostics": {
            "constant_features": [field for field, stats in profile.items() if stats["variance"] == 0],
            "all_missing_features": [field for field, stats in profile.items() if stats["variance"] is None],
            "missing_counts": {field: stats["missing_training_rows"] for field, stats in profile.items()
                               if stats["missing_training_rows"]},
            "near_constant_zero_share_threshold": ZERO_DOMINANCE_THRESHOLD,
            "near_constant_zero_shares": {
                field: stats["zero_training_rows"] / stats["observed_training_rows"]
                for field, stats in profile.items() if stats["variance"] and
                stats["zero_training_rows"] / stats["observed_training_rows"] >= ZERO_DOMINANCE_THRESHOLD
            },
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(artifact, indent=2, allow_nan=False) + "\n"
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", dir=output.parent, encoding="utf-8", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return artifact


def apply_scaler(row: dict, artifact: dict) -> dict:
                                                                                 
    transformed = row.copy()
    transforms = artifact.get("pre_transforms", {})
    for field, params in artifact["features"].items():
        number = _number(row[field], field)
        if number is None:
            continue
        if params["scale"] is None:
            raise ValueError(f"{field}: no observed training data to fit scaling parameters")
        prepared = _transform(number, transforms.get(field, "identity"))
        scaled = (prepared - params["mean"]) / params["scale"]
        if not math.isfinite(scaled):
            raise ValueError(f"{field}: scaling produced a non-finite result")
        transformed[field] = scaled
    return transformed
