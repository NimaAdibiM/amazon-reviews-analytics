"""Extract independently checkable summary evidence, never claim raw-file verification.

Usage: python -m audit_lab.quality_reference --archive /path/to/QUALITY_SEND.zip --output file.json
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

CATS = ("All_Beauty", "Appliances", "Beauty_and_Personal_Care", "Electronics")


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def summarize_archive(archive, root):
    archive = Path(archive)
    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Quality archive CRC failure")
        run = json.loads(z.read("RUN_MANIFEST.json"))
        if run.get("status") != "completed" or set(run.get("full", {})) != set(CATS):
            raise ValueError("Full-reference manifest incomplete")
        code_hash = sha256((root / "quality_audit.py").read_bytes())
        struct_hash = sha256((root / "structure_audit.py").read_bytes())
        if run.get("code_sha256") != code_hash or run.get("structure_sha256") != struct_hash:
            raise ValueError("Reference code hashes do not match archive manifest")
        summaries = {}
        for category in CATS:
            run_info = run["full"][category]
            if run_info["status"] != "completed" or run_info["scope"] != "entire_supplied_dataset_files":
                raise ValueError("Only complete full-scope evidence is accepted")
            paths = [p for p in z.namelist() if p.startswith(f"outputs/full/{category}/") and p.endswith("/quality_summary.json")]
            if len(paths) != 1:
                raise ValueError(f"Missing/ambiguous quality summary for {category}")
            p = paths[0]
            summary = json.loads(z.read(p))
            if summary.get("source_category") != category or summary.get("scope") != run_info["scope"]:
                raise ValueError(f"Category/scope mismatch: {category}")
            roles = summary["files"]
            counts = {role: int(roles[role]["object_records"]) for role in ("reviews", "metadata")}
            if counts["reviews"] != run_info["review_objects"] or counts["metadata"] != run_info["metadata_objects"]:
                raise ValueError(f"Counts mismatch: {category}")
            extra = int(roles["reviews"]["full_record_duplicates"]["duplicate_extra_rows"])
            if not (0 <= extra <= counts["reviews"]):
                raise ValueError("Impossible duplicate count")
            summaries[category] = {
                "reviews": counts["reviews"], "metadata": counts["metadata"],
                "reviews_exact_duplicate_extra": extra,
                "quality_summary_sha256": sha256(z.read(p)),
                "quality_scanned_content_sha256": {role: roles[role].get("scanned_content_sha256") for role in ("reviews", "metadata")},
                "quality_scan_mode": {role: roles[role].get("scan_mode") for role in ("reviews", "metadata")},
                "raw_bytes_available_in_this_archive": False,
                "path_in_archive": p,
            }
        return {
            "status": "verified_archive_internal_consistency_only",
            "source_run_id": run["run_id"],
            "source_as_of_utc": run["audit_as_of_utc"],
            "archive_file_sha256": sha256(archive.read_bytes()),
            "reference_code_hashes_matched": True,
            "categories": summaries,
            "totals": {"reviews": sum(v["reviews"] for v in summaries.values()),
                       "metadata": sum(v["metadata"] for v in summaries.values()),
                       "reviews_exact_duplicate_extra_within_categories": sum(v["reviews_exact_duplicate_extra"] for v in summaries.values())},
            "limitations": [
                "No access to eight raw input files in this environment: source hashes cannot be freshly recomputed.",
                "Duplicate totals are within-category extra canonical records; cross-category dedup is a different quantity.",
                "The review counts describe raw parsed records, not eligible 5+10 products or independent purchases.",
            ]
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = summarize_archive(args.archive, root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Quality archive internally verified: {result['totals']}")


if __name__ == "__main__":
    main()
