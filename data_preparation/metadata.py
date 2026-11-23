                                                                                    

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

CATEGORIES = ("All_Beauty", "Appliances", "Beauty_and_Personal_Care", "Electronics")
FIELDS = {
    "parent_asin": str, "main_category": str, "title": str, "subtitle": str,
    "author": (str, dict), "average_rating": (int, float), "rating_number": int,
    "price": (int, float), "images": list, "videos": list, "features": list,
    "description": list, "categories": list, "details": dict, "store": str,
    "bought_together": (list, str),
}
SNAPSHOT_FIELDS = ("category", "parent_asin", "price_snapshot", "price_state",
                   "rating_average_snapshot", "rating_count_snapshot", "metadata_status")
ABSENT = object()


def _state(value, kind):
    if value is ABSENT:
        return "absent"
    if value is None:
        return "null"
    if type(value) not in (kind if isinstance(kind, tuple) else (kind,)):
        return "wrong_type"
    if isinstance(value, str) and not value.strip():
        return "blank"
    if isinstance(value, (list, dict)) and not value:
        return "empty_container"
    return "present"


def _number(value, lower=0, upper=None):
    if type(value) is int:
        return value if value >= lower and (upper is None or value <= upper) else None
    if type(value) is float and math.isfinite(value) and value >= lower and (upper is None or value <= upper):
        return value
    return None


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant {value}")


def scan_metadata(path, category, output):
                                                                                          
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = output / f"{category}.csv.gz"
    temporary = table.with_suffix(".tmp")
    states = defaultdict(Counter)
    totals = Counter()
    digest = hashlib.sha256()
    seen = set()
    repeated = set()
    with Path(path).open("rb") as source, gzip.open(temporary, "wt", newline="", encoding="utf-8", compresslevel=3) as handle:
        writer = csv.writer(handle)
        writer.writerow(SNAPSHOT_FIELDS)
        for line in source:
            digest.update(line)
            totals["physical_lines"] += 1
            try:
                item = json.loads(line, parse_constant=_reject_constant)
            except (ValueError, RecursionError):
                totals["unparsed"] += 1
                continue
            if not isinstance(item, dict):
                totals["nonobject"] += 1
                continue
            totals["objects"] += 1
            for key, kind in FIELDS.items():
                states[key][_state(item.get(key, ABSENT), kind)] += 1
            parent = item.get("parent_asin")
            if not isinstance(parent, str) or not parent.strip() or any(0xD800 <= ord(c) <= 0xDFFF for c in parent):
                totals["invalid_parent"] += 1
                continue
            if parent in seen:
                repeated.add(parent)
                totals["additional_parent_rows"] += 1
            seen.add(parent)
            price = _number(item.get("price"))
            price_state = _state(item.get("price", ABSENT), (int, float))
            if price_state == "present":
                price_state = "valid" if price is not None else "invalid"
            totals["price_" + price_state] += 1
            if price is not None:
                totals["price_zero"] += price == 0
            count = item.get("rating_number")
            count = count if type(count) is int and count >= 0 else None
            writer.writerow((category, parent, price, price_state, _number(item.get("average_rating"), 1, 5),
                             count, "snapshot"))
    temporary.replace(table)
    report = {"category": category, "source": str(path), "source_sha256": digest.hexdigest(),
              "counts": dict(totals), "field_states": states,
              "unique_parent_keys": len(seen), "repeated_parent_keys": len(repeated)}
    (output / f"{category}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def read_for_products(path, parents):
                                                                           
    result = {}
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parent = row["parent_asin"]
            if parent in parents:
                result[parent] = None if parent in result else row
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    for name in CATEGORIES:
        result = scan_metadata(arguments.data / f"meta_{name}.jsonl", name, arguments.out)
        print(name, result["counts"], flush=True)
