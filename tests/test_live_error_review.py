"""Offline regression tests for integrity, output safety and conservative hints."""
import csv
import hashlib
import importlib.util
import io
import json
import random
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "33_review_live_errors.py"
spec = importlib.util.spec_from_file_location("step33", SCRIPT)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True), encoding="utf-8")


class ReviewTests(unittest.TestCase):
    def setUp(self):
        # Use workspace-local temporary files on the user's restricted Windows setup.
        base = SCRIPT.parent / ".test-work"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name)
        self.run = self.root / "project" / "reports" / "saved_run"
        self.run.mkdir(parents=True)
        self.output = self.root / "project" / "reports" / "review"
        fields = review.FIELDS
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for n in range(1, 9):
            writer.writerow({"url": f"https://example{n}.test/", "label": "benign" if n <= 4 else "phish",
                             "label_source": "=untrusted independent evidence", "label_checked_at": "2026-10-11"})
        raw = stream.getvalue().encode("utf-8")
        (self.run / "frozen_input.csv").write_bytes(raw)
        self.rows = review.load_frozen(raw)
        self.model = {"threshold": .5, "model_trained": False, "model_bundle_sha256": "a" * 64,
                      "frozen_specification_sha256": "b" * 64, "feature_count": 106, "model_name": "lightgbm_context_all_features",
                      "best_iteration": 1269, "numeric_type": "float32", "runtime_version": "1.0.0"}
        order = list(range(1, 9))
        random.Random(42).shuffle(order)
        identity = {"schema_version": 1, "evaluator_version": "1.0.0", "input_csv_sha256": hashlib.sha256(raw).hexdigest(),
                    "normalized_rows_sha256": review.object_sha(self.rows), "rows": self.rows, "model": self.model,
                    "capture": {"basis": "http_source", "javascript_executed": False, "attempts_per_url": 1,
                                "max_html_bytes": 32 * 1024 * 1024, "timeout_seconds": 45, "max_redirects": 5},
                    "shuffle_seed": 42, "request_order_ordinals": order}
        self.manifest = {"started_at_utc": "2026-10-11T05:00:00+00:00", "identity": identity,
                         "identity_sha256": review.object_sha(identity)}
        dump(self.run / "evaluation_protocol.json", self.manifest)
        states = {1: ("prediction_complete", .1), 2: ("prediction_complete", .5),
                  3: ("inconclusive_capture", .9), 4: ("capture_failed", None),
                  5: ("prediction_complete", .2), 6: ("prediction_complete", .8),
                  7: ("capture_failed", None), 8: ("interrupted_capture", None)}
        self.records = {}
        connection = sqlite3.connect(self.run / "checkpoint.sqlite3")
        connection.execute("CREATE TABLE pages (ordinal INTEGER PRIMARY KEY, record_json TEXT NOT NULL, record_sha256 TEXT NOT NULL, html_zlib BLOB)")
        for row in self.rows:
            n = row["ordinal"]
            status, score = states[n]
            record = {"ordinal": n, "url": row["normalized_url"], "attempt_id": f"{n:032x}", "phase": "complete", "status": status,
                      "attempt_started_at_utc": "2026-10-11T05:00:01+00:00", "capture_completed_at_utc": "2026-10-11T05:00:02+00:00"}
            archive = None
            if score is None:
                record["error"] = "HTTP 403 forbidden" if n == 4 else "Capture attempt interrupted" if n == 8 else "Could not resolve the website hostname."
            else:
                html = "<html><head><title>Sample page</title></head><body><p>Ordinary source text</p></body></html>"
                html_sha = hashlib.sha256(html.encode()).hexdigest()
                archive = zlib.compress(html.encode())
                record.update(html_sha256=html_sha, prediction_completed_at_utc="2026-10-11T05:00:03+00:00",
                              capture={"requested_url": row["normalized_url"], "final_url": row["normalized_url"],
                                       "capture_basis": "http_source", "capture_complete": True, "status_code": 200,
                                       "javascript_executed": False, "external_resources_requested": False},
                              result={"status": status, "model": self.model, "capture_basis": "http_source", "capture_basis_verified": True,
                                      "page_url": row["normalized_url"], "html_sha256": html_sha,
                                      "javascript_executed_by_detector": False, "external_javascript_downloaded_by_detector": False,
                                      "phishing_score": score, "static_model_prediction": "phish" if score >= .5 else "benign",
                                      "active_quality_flags": ["unavailable"] if n == 3 else [], "features": {"form_count": 0}})
            encoded = review.canonical_json(record)
            connection.execute("INSERT INTO pages VALUES(?,?,?,?)", (n, encoded, hashlib.sha256(encoded.encode()).hexdigest(), archive))
            self.records[n] = record
        connection.commit()
        connection.close()
        self.report = review.recompute_report(self.rows, self.records, self.manifest)
        self.report["generated_at_utc"] = "2026-10-11T05:00:04+00:00"
        dump(self.run / "live_evaluation_report.json", self.report)
        self.write_prediction_csv()
        self.hashes = {name: review.file_sha(self.run / name) for name in review.SOURCE_FILES}

    def tearDown(self):
        self.temp.cleanup()

    def write_prediction_csv(self):
        with (self.run / "live_evaluation_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, review.CSV_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in review.prediction_rows(self.rows, self.records):
                writer.writerow({key: review.csv_safe(value) for key, value in row.items()})

    def mutate_record(self, n, change):
        record = json.loads(review.canonical_json(self.records[n]))
        change(record)
        encoded = review.canonical_json(record)
        connection = sqlite3.connect(self.run / "checkpoint.sqlite3")
        connection.execute("UPDATE pages SET record_json=?,record_sha256=? WHERE ordinal=?",
                           (encoded, hashlib.sha256(encoded.encode()).hexdigest(), n))
        connection.commit()
        connection.close()

    def reject(self, pattern=None):
        with self.assertRaisesRegex(review.ReviewError, pattern or ".+"):
            review.load_source(self.run)

    def test_valid_review_preserves_scores_denominators_and_inputs(self):
        source = review.load_source(self.run)
        result = review.save_review(source, self.output)
        metrics = result["original_report_retained"]["classification_metrics_on_scored_pages"]
        self.assertEqual(metrics["confusion_matrix"], {"TN": 1, "FP": 1, "FN": 1, "TP": 1})
        self.assertEqual(metrics["accuracy"], .5)
        self.assertEqual(result["original_report_retained"]["scored_coverage_all_selected_pages"], .5)
        self.assertEqual(result["error_count"], 2)
        self.assertEqual(result["all_rows"][1]["prediction"], "phish")  # threshold equality
        self.assertEqual(result["all_rows"][2]["diagnostic_prediction"], "phish")
        self.assertNotIn("prediction", result["all_rows"][2])
        self.assertNotIn("prediction", result["all_rows"][3])
        self.assertEqual({name: review.file_sha(self.run / name) for name in review.SOURCE_FILES}, self.hashes)
        self.assertEqual(sorted(path.name for path in self.run.iterdir()), sorted(review.SOURCE_FILES))
        self.assertTrue(all(path.name.endswith(".html.txt") for path in (self.output / "source_copies").iterdir()))
        self.assertTrue(all(item["complete"] for item in result["source_copy_manifest"]))
        with (self.output / "review_index.csv").open(encoding="utf-8-sig", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        for key in ("markup_scan_cut", "markup_depth_cut", "visible_text_scan_cut", "inline_script_scan_cut"):
            self.assertEqual(csv_rows[0][key], "False")
            self.assertEqual(csv_rows[3][key], "")  # A failed capture has no source observation.
        self.assertGreater(int(csv_rows[0]["html_character_count"]), 0)
        self.assertGreater(int(csv_rows[0]["visible_source_text_characters_observed"]), 0)

    def test_record_hash_mismatch(self):
        connection = sqlite3.connect(self.run / "checkpoint.sqlite3")
        connection.execute("UPDATE pages SET record_sha256='bad' WHERE ordinal=1")
        connection.commit()
        connection.close()
        self.reject("record 1 checksum")

    def test_html_corruption_truncation_trailing_size_and_utf8(self):
        for blob in (b"not-zlib", zlib.compress(b"abc")[:-1], zlib.compress(b"abc") + b"extra", zlib.compress(b"\xff")):
            with self.subTest(blob=blob), self.assertRaises(review.ReviewError):
                review.archive_html(blob)
        with self.assertRaisesRegex(review.ReviewError, "oversized"):
            review.archive_html(zlib.compress(b"x" * 9), limit=8)

    def test_archived_html_hash_mismatch(self):
        connection = sqlite3.connect(self.run / "checkpoint.sqlite3")
        connection.execute("UPDATE pages SET html_zlib=? WHERE ordinal=1", (zlib.compress(b"different source"),))
        connection.commit()
        connection.close()
        self.reject("HTML checksum")

    def test_frozen_bytes_mismatch(self):
        with (self.run / "frozen_input.csv").open("ab") as handle:
            handle.write(b"\n")
        self.reject("Frozen input bytes")

    def test_protocol_hash_mismatch(self):
        self.manifest["identity"]["model"]["threshold"] = .4
        dump(self.run / "evaluation_protocol.json", self.manifest)
        self.reject("Protocol identity checksum")

    def test_protocol_rows_mismatch_even_with_rehashed_identity(self):
        self.manifest["identity"]["rows"][0]["label"] = "phish"
        self.manifest["identity_sha256"] = review.object_sha(self.manifest["identity"])
        dump(self.run / "evaluation_protocol.json", self.manifest)
        self.reject("normalized rows mismatch")

    def test_forged_metrics(self):
        self.report["classification_metrics_on_scored_pages"]["accuracy"] = 1.0
        dump(self.run / "live_evaluation_report.json", self.report)
        self.reject("report disagrees")

    def test_prediction_csv_mismatch(self):
        path = self.run / "live_evaluation_predictions.csv"
        text = path.read_text(encoding="utf-8-sig").replace(",0.5,phish,False,", ",0.5,phish,True,")
        path.write_text(text, encoding="utf-8-sig")
        self.reject("Prediction CSV disagrees")

    def test_partial_checkpoint_rejected(self):
        self.mutate_record(1, lambda record: record.update(phase="captured", status="captured_awaiting_prediction"))
        self.reject("partial")

    def test_partial_report_rejected(self):
        self.report["status"] = "live_evaluation_partial"
        dump(self.run / "live_evaluation_report.json", self.report)
        self.reject("incomplete")

    def test_result_status_and_model_provenance_mismatch(self):
        self.mutate_record(1, lambda record: record["result"].update(status="inconclusive_capture"))
        self.reject("provenance")

    def test_failed_capture_cannot_have_prediction(self):
        self.mutate_record(4, lambda record: record.update(result={"phishing_score": .1}))
        self.reject("Failed/interrupted")

    def test_score_threshold_mismatch(self):
        self.mutate_record(2, lambda record: record["result"].update(static_model_prediction="benign"))
        self.reject("score/threshold")

    def test_wrong_url_and_missing_record(self):
        self.mutate_record(1, lambda record: record.update(url="https://different.test/"))
        self.reject("identity mismatch")

    def test_active_sidecar_rejected(self):
        (self.run / "checkpoint.sqlite3-wal").write_bytes(b"active")
        self.reject("sidecar")

    def test_output_overlap_model_folder_and_nonempty_protection(self):
        for target in (self.run, self.run / "review", self.run.parent, self.run.parent.parent / "detector_runtime" / "review"):
            with self.subTest(target=target), self.assertRaises(review.ReviewError):
                review.validate_output(self.run, target)
        self.output.mkdir()
        (self.output / "unrelated.txt").write_text("keep")
        with self.assertRaisesRegex(review.ReviewError, "not empty"):
            review.save_review(review.load_source(self.run), self.output)
        self.assertEqual((self.output / "unrelated.txt").read_text(), "keep")

    def test_source_only_hints_and_benign_controls(self):
        plain = review.inspect_html('<body><p>' + 'Ordinary article text. ' * 20 + '</p><script src="site.js"></script></body>')
        self.assertEqual(plain["review_hints"], [])
        self.assertEqual(plain["external_script_count"], 1)
        self.assertEqual(plain["form_count"], 0)
        page = review.inspect_html('<body><script>location.replace("https://example.test/")</script><p hidden>Continue shopping</p><style>Continue shopping</style><template>Continue shopping</template></body>')
        self.assertIn("tiny_script_navigation_shell_hint", page["review_hints"])
        self.assertNotIn("continue_shopping_interstitial_hint", page["review_hints"])
        contact = review.inspect_html('<body><form><textarea></textarea><button>Contact us</button></form><script>location.href="/ok"</script></body>')
        self.assertEqual(contact["review_hints"], [])
        self.assertEqual(contact["textarea_count"], 1)
        shell = review.inspect_html('<head><title>Sign in</title></head><body><div id="app"></div><script src="app.js"></script></body>')
        self.assertIn("low_text_script_source_hint", shell["review_hints"])
        holding = review.inspect_html('<body><p>Preparando sua navegação</p><script>window.location.replace("/next")</script></body>')
        self.assertIn("redirect_holding_page_hint", holding["review_hints"])
        amazon = review.inspect_html('<head><title>Amazon.com</title></head><body>Click the button below to continue shopping.</body>')
        self.assertIn("continue_shopping_interstitial_hint", amazon["review_hints"])

    def test_csv_formula_escape_and_markdown_no_url_autolink(self):
        for text in ("=SUM(1,2)", "+cmd", "-1", "@cmd", "\tcmd", "\rcmd"):
            self.assertEqual(review.csv_safe(text), "'" + text)
        source = review.load_source(self.run)
        report = review.build_review(source)
        md = review.make_markdown(report)
        self.assertNotIn("https://example2.test/", md)
        self.assertIn("https&#58;//example2.test/", md)
        self.assertEqual(source["prediction_rows"][0]["label_source"], "=untrusted independent evidence")

    def test_markup_depth_and_prefix_are_bounded_and_disclosed(self):
        nested = '<div>' * (review.MAX_MARKUP_DEPTH + 10) + '<script>location.href="/next"</script>'
        audit = review.inspect_html(nested)
        self.assertTrue(audit["markup_depth_cut"])
        self.assertTrue(audit["markup_scan_cut"])
        self.assertEqual(audit["review_hints"], [])
        audit = review.inspect_html('<body>' + 'x' * review.MARKUP_SCAN_CHARACTERS + '</body>')
        self.assertTrue(audit["markup_scan_cut"])

    def test_copy_cap_keeps_all_error_rows_and_reports_omissions(self):
        original_cap = review.MAX_SOURCE_COPIES
        try:
            review.MAX_SOURCE_COPIES = 1
            source = review.load_source(self.run)
            result = review.build_review(source)
        finally:
            review.MAX_SOURCE_COPIES = original_cap
        self.assertEqual(result["error_count"], 2)
        self.assertEqual(result["reviewed_rows"], 8)
        self.assertEqual(len(result["source_copy_manifest"]), 1)
        self.assertEqual(len(source["source_copies"]), 1)
        self.assertEqual(result["source_copy_limits"]["omitted_error_copies"], 1)

    def test_asymmetric_metrics_independent_constants(self):
        metrics = review.classification_metrics([("benign", "benign"), ("benign", "benign"), ("benign", "phish"),
                                                 ("phish", "phish"), ("phish", "phish"), ("phish", "benign"), ("phish", "benign")])
        self.assertEqual(metrics["confusion_matrix"], {"TN": 2, "FP": 1, "FN": 2, "TP": 2})
        self.assertAlmostEqual(metrics["accuracy"], 4 / 7)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 1 / 2)
        self.assertAlmostEqual(metrics["false_positive_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["false_negative_rate"], 1 / 2)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 7 / 12)
        self.assertAlmostEqual(metrics["f1"], 4 / 7)


if __name__ == "__main__":
    unittest.main()
