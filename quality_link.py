"""E03/E11: conditional, read-only reconciliation with Q01–Q25 evidence.

Nothing is imported from historical quality data unless every current source
content hash, scan scope, and definition checks match.  Archive-internal
consistency is NOT the same as verification of the raw source bytes.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile, BadZipFile

import quality_audit as quality
import structure_audit as structure

CATEGORIES = ("All_Beauty", "Appliances", "Beauty_and_Personal_Care", "Electronics")


def _file_bytes_hash(path):
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_archive(path):
    """Read only named JSON members; never extract archive paths onto disk."""
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Quality archive CRC error")
        run = json.loads(archive.read("RUN_MANIFEST.json"))
        if run.get("status") != "completed" or set(run.get("full", {})) != set(CATEGORIES):
            raise ValueError("Quality archive does not contain four completed category runs")
        if (run.get("code_sha256") != _file_bytes_hash(quality.__file__)
                or run.get("structure_sha256") != _file_bytes_hash(structure.__file__)):
            raise ValueError("Quality archive was produced by different reference audit code")
        output = {}
        for cat in CATEGORIES:
            entry = run["full"][cat]
            if entry.get("status") != "completed" or entry.get("scope") != "entire_supplied_dataset_files":
                raise ValueError("Quality archive includes a non-full/incomplete result")
            prefix = f"outputs/full/{cat}/"
            matches = [member for member in archive.namelist()
                       if member.startswith(prefix) and member.endswith("/quality_summary.json")]
            if len(matches) != 1:
                raise ValueError(f"Missing or multiple quality summaries for {cat}")
            raw = archive.read(matches[0])
            item = json.loads(raw)
            for role, expected in (("reviews", "review_objects"), ("metadata", "metadata_objects")):
                if item["files"][role]["object_records"] != entry[expected]:
                    raise ValueError(f"Quality archive count mismatch for {cat}/{role}")
            output[cat] = (item, sha256(raw).hexdigest())
    return output


def _extract_directory(path):
    output = {}
    for cat in CATEGORIES:
        file = Path(path) / cat / "quality_summary.json"
        if file.exists():
            raw = file.read_bytes()
            output[cat] = (json.loads(raw), sha256(raw).hexdigest())
    return output


def _number_at(payload, *keys):
    result = payload
    for key in keys:
        result = result[key]
    if type(result) is not int or result < 0:
        raise ValueError("Quality metric must be a nonnegative integer")
    return result


def _metric_checks(q, raw):
    """Compare exactly equivalent definitions, never reinterpret snapshot rates."""
    checks = {}
    review = q["files"]["reviews"]
    meta = q["files"]["metadata"]
    rv = raw["reviews"]
    md = raw["metadata"]

    def check(name, expected, actual):
        if type(expected) is not int or type(actual) is not int:
            raise ValueError(f"Noninteger Q/EDA check: {name}")
        checks[name] = {"quality": expected, "eda": actual, "match": expected == actual}

    check("Q11_valid_rating", _number_at(review, "q11_review_rating_validity", "valid_integral_stars_rows"),
          rv["numeric"]["rating"]["valid"])
    valid_star_counts = review["q11_review_rating_validity"]["valid_star_counts"]
    actual_stars = {int(item["lower"]): item["numerator"] for item in rv["numeric"]["rating"]["bins"]}
    for star in range(1, 6):
        check(f"Q11_star_{star}", _number_at(valid_star_counts, str(star)), actual_stars.get(star, 0))
    check("Q13_valid_timestamp", _number_at(review, "q13_review_timestamp_validity", "representable_rows"),
          rv["modalities"]["timestamp"]["numerator"])
    qm = review["modality_availability"]
    for name, qname in (("body", "nonblank_review_body"),
                        ("title", "nonblank_review_title"),
                        ("any_text", "nonblank_review_title_or_body"),
                        ("review_image_url", "image_http_url_reference")):
        check(f"Q22_{name}", _number_at(qm, qname, "rows"), rv["modalities"][name]["numerator"])
    mm = meta["modality_availability"]
    check("Q14_valid_price", _number_at(mm, "valid_current_price", "rows"),
          md["modalities"]["valid_price"]["numerator"])
    check("Q22_product_image", _number_at(mm, "image_http_url_reference", "rows"),
          md["modalities"]["product_image_url"]["numerator"])
    # The complete Q22/Q23 dictionaries exist in actual Q01-Q25 exports.
    # The small synthetic compatibility fixture intentionally provides only
    # the earlier core checkpoints; do not fabricate these absent metrics.
    if "q22_review_text_lengths" in q and "q23_review_short_long_text" in q:
        q22=q["q22_review_text_lengths"]
        q23=q["q23_review_short_long_text"]
        if q23["primary_short_word_threshold_inclusive"] != 3 or q23["primary_long_character_threshold_exclusive"] != 10000:
            raise ValueError("Q23 threshold contract differs from EDA")
        check("Q22_body_all_strings",_number_at(q22,"string_rows"),
              rv["numeric"]["body_chars_all_strings"]["valid"])
        check("Q22_body_nonblank",_number_at(q22,"states","nonblank_string"),
              rv["numeric"]["body_chars"]["valid"])
        check("Q23_short_nonblank",_number_at(q23,"primary_short_nonblank_rows"),
              rv["text_thresholds_q23"]["short_nonblank_words_at_most_3"]["numerator"])
        check("Q23_long_all_strings",_number_at(q23,"primary_long_string_rows"),
              rv["text_thresholds_q23"]["long_all_string_codepoints_more_than_10000"]["numerator"])
    return checks


def reconcile_quality(inputs, raw, scope, unit, directory=None, archive=None):
    """Return an explicit abstention on unavailable/nonmatching evidence.

    Full category matching is a precondition for publishing *any* imported Q
    metric, preventing a mixture of verified and unverified categories.
    """
    if directory is not None and archive is not None:
        raise ValueError("Specify either quality directory or quality archive, not both")
    if directory is None and archive is None:
        return {"status": "not_provided", "reused_metrics": []}
    source = Path(archive or directory)
    output = {"status": "unverified", "format": "zip" if archive else "directory",
              "source": str(source.resolve()), "source_sha256": _file_bytes_hash(source) if archive else None,
              "reused_metrics": [], "by_category": {}, "limitations": [
                  "A matched quality summary only proves consistency with the scanned local input, not authenticity of upstream download.",
                  "Snapshot fields cannot be asserted to have existed at the fifth review."]}
    try:
        summaries = _extract_archive(source) if archive else _extract_directory(source)
    except (ValueError, KeyError, TypeError, OSError, BadZipFile, json.JSONDecodeError) as exc:
        output["reason"] = "invalid_or_incompatible_archive_or_directory"
        output["details"] = type(exc).__name__ + ": " + str(exc)
        return output
    for cat in CATEGORIES:
        failures = []
        payload, digest = summaries.get(cat, (None, None))
        if payload is None:
            output["by_category"][cat] = {"status": "missing"}
            continue
        if payload.get("audit_version") != quality.VERSION:
            failures.append("quality_version")
        if payload.get("structure_audit_integration", {}).get("module_version") != structure.VERSION:
            failures.append("structure_version")
        if payload.get("source_category") != cat:
            failures.append("category")
        if payload.get("scope") != scope or scope != "entire_supplied_dataset_files":
            failures.append("scope")
        if payload.get("structure_audit_integration", {}).get("timestamp_unit") != unit:
            failures.append("timestamp_unit")
        for role in ("reviews", "metadata"):
            try:
                observed = inputs[cat][role]
                expected = payload["files"][role]
                if expected.get("scanned_content_sha256") != observed["scanned_content_sha256"]:
                    failures.append(f"{role}_content_sha256")
                if expected.get("object_records") != observed["counts"]["objects"]:
                    failures.append(f"{role}_objects")
                if expected.get("physical_lines") != observed["counts"]["lines"]:
                    failures.append(f"{role}_physical_lines")
            except (KeyError, TypeError):
                failures.append(f"{role}_missing_evidence")
        if failures:
            output["by_category"][cat] = {"status": "mismatch", "failures": failures,
                                          "quality_summary_sha256": digest}
        else:
            try:
                checks = _metric_checks(payload, raw[cat])
            except (KeyError, TypeError, ValueError) as exc:
                output["by_category"][cat] = {
                    "status": "incompatible_metric_definitions", "failures": [type(exc).__name__],
                    "quality_summary_sha256": digest}
                continue
            failures.extend(k for k, v in checks.items() if not v["match"])
            output["by_category"][cat] = {
                "status": "checked" if not failures else "metric_mismatch",
                "failures": failures, "quality_summary_sha256": digest,
                "checked_metric_count": len(checks), "checks": checks}
    if all(row["status"] == "checked" for row in output["by_category"].values()):
        output["status"] = "matched_all_inputs_and_metric_definitions"
        output["note"] = "Q summary fields were verified against the identical complete local source bytes; not an independent raw scan."
        output["reused_metrics"] = ["Q11", "Q13", "Q14", "Q22"]
    else:
        output["status"] = "rejected_incomplete_or_mismatched_quality_evidence"
        output["reused_metrics"] = []
    return output
