"""Regression checks for the reviewed phase-two preprocessing rules."""

import csv
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from features import make_features, normalize_review_text
from ingest import ingest_category
from prepare import audit_early
from split_scale import apply_scaler, fit_scaler


class PhaseTwoRevisionTests(unittest.TestCase):
    def test_html_normalization_drives_text_and_length_features(self):
        self.assertEqual(normalize_review_text("&lt;BR /&gt;A&amp;B<br/>C\ud800"), "A&B C\ufffd")
        early = [{"rating": 3, "timestamp": time, "title": "Title"} for time in range(5)]
        early[0]["text"] = "A<br/>B &amp; C"
        early[1]["text"] = "<BR />&nbsp;"
        values = make_features(early)
        self.assertIn("Title\nA B & C", values["text_document"])
        self.assertEqual(values["body_words_mean"], 4 / 5)
        self.assertEqual(values["body_chars_max"], 7)
        self.assertEqual(values["body_missing_count"], 4)

    def test_early_audit_denominators(self):
        handle = io.StringIO()
        states = {"All_Beauty": {"bodies": {"present": 9, "blank_string": 1},
                                "documents": {"present": 2},
                                "products_with_missing_body": {"yes": 1, "no": 1}}}
        splits = {"All_Beauty": {"train": 1, "test": 1}}
        audit_early(csv.writer(handle), states, splits)
        counts = {(row[2], row[3]): row[4:] for row in csv.reader(io.StringIO(handle.getvalue()))}
        self.assertEqual(counts["bodies", "blank_string"], ["1", "10", "0.1"])
        self.assertEqual(counts["products_with_missing_body", "yes"], ["1", "2", "0.5"])
        states["All_Beauty"]["bodies"]["blank_string"] = 0
        with self.assertRaisesRegex(ValueError, "denominator does not reconcile"):
            audit_early(csv.writer(handle), states, splits)

    def test_scaling_uses_transformed_train_only_and_keeps_constants_finite(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "features.csv"
            data = ("split,gap,slope,constant,flag\ntrain,0,-3,0,0\ntrain,3,3,0,0\n"
                    "validation,30,1000,0,0\ntest,9999,-9999,,0\n")
            source.write_text(data)
            scaler = fit_scaler(source, Path(folder) / "scaler.json", ("gap", "slope", "constant"),
                                pre_transforms={"gap": "log1p", "slope": "signed_log1p"},
                                passthrough_fields=("flag",))
            self.assertAlmostEqual(scaler["features"]["gap"]["mean"], math.log(4) / 2)
            self.assertAlmostEqual(scaler["features"]["slope"]["mean"], 0.0)
            self.assertAlmostEqual(scaler["features"]["slope"]["scale"], math.log(4))
            self.assertEqual(scaler["features"]["constant"]["scale"], 1.0)
            self.assertEqual(scaler["training_feature_diagnostics"]["constant_features"], ["constant", "flag"])
            scaled = apply_scaler({"gap": "3", "slope": "-3", "constant": "0", "flag": "0"}, scaler)
            self.assertEqual(scaled, {"gap": 1.0, "slope": -1.0, "constant": 0.0, "flag": "0"})
            self.assertEqual(apply_scaler({**scaled, "constant": ""}, scaler)["constant"], "")
            self.assertEqual(source.read_text(), data)

    def test_boundary_ties_exclude_parent_while_internal_ties_are_retained(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "reviews.jsonl"
            with source.open("w", encoding="utf-8") as handle:
                for parent, times in (
                    ("tie5", [*range(1, 6), 5, *range(7, 17)]),
                    ("tie15", [*range(1, 16), 15]),
                    ("internal", [1, 1, *range(3, 17)]),
                ):
                    for i, timestamp in reversed(list(enumerate(times))):
                        handle.write(json.dumps({"parent_asin": parent, "timestamp": timestamp,
                                                 "rating": 1 if i % 2 else 5, "title": str(i)}) + "\n")
            audit = ingest_category(source, "All_Beauty", root / "out", workers=1, shards=2)
            self.assertEqual(audit["counts"]["tied_5_6_products"], 1)
            self.assertEqual(audit["counts"]["tied_15_16_products"], 1)
            rows = [json.loads(line) for line in Path(audit["cohort_path"]).read_text().splitlines()]
            self.assertEqual([row["parent_asin"] for row in rows], ["internal"])


if __name__ == "__main__":
    unittest.main()
