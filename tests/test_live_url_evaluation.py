"""Offline evaluation-plumbing tests. Stub scores do not measure model accuracy."""
import csv
import hashlib
import http.server
import io
import json
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from live_evaluation.engine import Checkpoint, RunLock, run_evaluation, validate_prediction
from live_evaluation.protocol import FIELDS, EvaluationError, classification_metrics, load_url_list, wilson_interval
from live_evaluation.runtime import import_runtime

RUNTIME_ROOT = Path(os.environ.get("LIVE_EVAL_RUNTIME_ROOT", str(ROOT))).resolve()
capture = import_runtime(RUNTIME_ROOT)


def csv_bytes(entries):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(entries)
    return text.getvalue().encode("utf-8")


def entries():
    return [{"url": f"https://{label}{number}.example/", "label": label,
             "label_source": "Controlled unit-test fixture (not a real evaluation URL)",
             "label_checked_at": date.today().isoformat()}
            for label in ("benign", "phish") for number in (1, 2)]


class StubDetector:
    """Fixed scores chosen to verify bookkeeping, with no fitted classifier."""
    def __init__(self, fail_once=False):
        self.fail_once = fail_once
        self.inference_calls = 0

    def describe(self):
        return {"model_name": "TEST_STUB_NOT_A_CLASSIFIER", "threshold": 0.5,
                "feature_count": 106, "model_trained": False, "testing_stub": True}

    def predict_html(self, html, page_url, capture_basis):
        self.inference_calls += 1
        if self.fail_once:
            self.fail_once = False
            raise MemoryError("Injected inference interruption")
        score = float(html.split("SCORE=")[1].split(";")[0])
        inconclusive = "INCONCLUSIVE" in html
        return {"status": "inconclusive_capture" if inconclusive else "prediction_complete",
                "model": self.describe(), "phishing_score": score,
                "static_model_prediction": "phish" if score >= 0.5 else "benign",
                "page_url": page_url, "capture_basis": capture_basis,
                "html_sha256": hashlib.sha256(html.encode()).hexdigest(),
                "active_quality_flags": ["test_challenge"] if inconclusive else []}


class StubCapture:
    CaptureError = capture.CaptureError
    MAX_HTML_BYTES = capture.MAX_HTML_BYTES
    TOTAL_TIMEOUT = capture.TOTAL_TIMEOUT
    MAX_REDIRECTS = capture.MAX_REDIRECTS

    def __init__(self, pages=None, interrupt_at=None):
        self.pages = pages or {}
        self.interrupt_at = interrupt_at
        self.requests = []

    def capture_url(self, url):
        self.requests.append(url)
        if self.interrupt_at == len(self.requests):
            raise KeyboardInterrupt()
        value = self.pages.get(url, "<html>SCORE=0.1;</html>")
        if isinstance(value, Exception):
            raise value
        return capture.Capture(value, {
            "requested_url": url, "final_url": url, "capture_basis": "http_source",
            "capture_complete": True, "status_code": 200, "javascript_executed": False,
            "external_resources_requested": False,
            "response_body_sha256": hashlib.sha256(value.encode()).hexdigest(),
        })


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="live-url-eval-test-")
        self.folder = Path(self.temporary.name)
        self.csv_path = self.folder / "urls.csv"
        self.output = self.folder / "reports"
        self.raw = csv_bytes(entries())
        self.csv_path.write_bytes(self.raw)
        self.rows, _ = load_url_list(self.csv_path, capture.normalize_url, per_class=2)

    def tearDown(self):
        self.temporary.cleanup()

    def run_fixture(self, fetcher, detector=None, rows=None, raw=None, checksums=None):
        return run_evaluation(rows or self.rows, self.raw if raw is None else raw, self.output,
                              detector or StubDetector(), fetcher,
                              checksums or {"test": "unit-test-program-v1"})

    def predictions(self):
        with (self.output / "live_evaluation_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_complete_balanced_pilot_and_threshold_boundary(self):
        scores = (0.1, 0.9, 0.5, 0.2)
        fetcher = StubCapture({row["normalized_url"]: f"<html>SCORE={score};</html>" for row, score in zip(self.rows, scores)})
        report = self.run_fixture(fetcher)
        metrics = report["classification_metrics_on_scored_pages"]
        self.assertEqual(metrics["confusion_matrix"], {"TN": 1, "FP": 1, "FN": 1, "TP": 1})
        for name in ("accuracy", "precision", "recall", "false_positive_rate", "f1", "balanced_accuracy"):
            self.assertEqual(metrics[name], 0.5)
        self.assertTrue(report["scored_subset_balanced"])
        self.assertEqual(report["status"], "live_evaluation_complete")
        self.assertEqual(self.predictions()[2]["prediction"], "phish")
        self.assertEqual((self.output / "frozen_input.csv").read_bytes(), self.raw)
        self.assertEqual(len(fetcher.requests), 4)

    def test_completed_run_resumes_without_new_requests(self):
        first = StubCapture()
        self.run_fixture(first)
        second = StubCapture()
        self.run_fixture(second)
        self.assertEqual(second.requests, [])

    def test_capture_failure_is_unscored_not_benign(self):
        failed = self.rows[0]["normalized_url"]
        report = self.run_fixture(StubCapture({failed: capture.CaptureError("Fixture HTTP 403")}))
        self.assertEqual(report["scored_class_counts"], {"benign": 1, "phish": 2})
        self.assertFalse(report["scored_subset_balanced"])
        self.assertEqual(report["coverage_by_class"]["benign"]["capture_failed"], 1)
        self.assertEqual(report["classification_metrics_on_scored_pages"]["scored_pages"], 3)
        first = self.predictions()[0]
        self.assertEqual(first["status"], "capture_failed")
        self.assertEqual(first["prediction"], "")
        self.assertEqual(first["correct"], "")
        self.assertEqual(first["phishing_score"], "")

    def test_challenge_score_is_diagnostic_only(self):
        blocked = self.rows[0]["normalized_url"]
        report = self.run_fixture(StubCapture({blocked: "<html>INCONCLUSIVE SCORE=0.9;</html>"}))
        self.assertEqual(report["coverage_by_class"]["benign"]["inconclusive_capture"], 1)
        self.assertEqual(report["classification_metrics_on_scored_pages"]["confusion_matrix"]["FP"], 0)
        first = self.predictions()[0]
        self.assertEqual(first["prediction"], "")
        self.assertEqual(first["correct"], "")
        self.assertEqual(first["diagnostic_prediction"], "phish")

    def test_inference_interruption_uses_saved_html_on_resume(self):
        fetcher = StubCapture()
        with self.assertRaises(MemoryError):
            self.run_fixture(fetcher, StubDetector(fail_once=True))
        self.assertEqual(len(fetcher.requests), 1)
        self.assertEqual(json.loads((self.output / "live_evaluation_report.json").read_text())["status"], "live_evaluation_partial")
        checkpoint = Checkpoint(self.output / "checkpoint.sqlite3")
        try:
            self.assertEqual(next(iter(checkpoint.records().values()))["phase"], "captured")
        finally:
            checkpoint.close()
        resumed = StubCapture()
        report = self.run_fixture(resumed)
        self.assertEqual(len(resumed.requests), 3)
        self.assertNotIn(fetcher.requests[0], resumed.requests)
        self.assertEqual(report["completed_attempts"], 4)

    def test_capture_interruption_retained_and_not_retried(self):
        fetcher = StubCapture(interrupt_at=2)
        with self.assertRaises(KeyboardInterrupt):
            self.run_fixture(fetcher)
        resumed = StubCapture()
        report = self.run_fixture(resumed)
        self.assertEqual(len(resumed.requests), 2)
        self.assertEqual(len(set(fetcher.requests + resumed.requests)), 4)
        self.assertEqual(sum(x["interrupted_capture"] for x in report["coverage_by_class"].values()), 1)
        self.assertEqual(report["classification_metrics_on_scored_pages"]["scored_pages"], 3)

    def test_changed_input_labels_and_code_refuse_resume(self):
        self.run_fixture(StubCapture())
        for changes in ({"raw": self.raw + b"\n"}, {"checksums": {"test": "changed"}}):
            fetcher = StubCapture()
            with self.assertRaisesRegex(EvaluationError, "Resume refused"):
                self.run_fixture(fetcher, **changes)
            self.assertEqual(fetcher.requests, [])
        changed_rows = [dict(row) for row in self.rows]
        changed_rows[0]["label"], changed_rows[2]["label"] = "phish", "benign"
        with self.assertRaisesRegex(EvaluationError, "Resume refused"):
            self.run_fixture(StubCapture(), rows=changed_rows)

    def test_frozen_csv_and_checkpoint_integrity(self):
        self.run_fixture(StubCapture())
        original = (self.output / "frozen_input.csv").read_bytes()
        (self.output / "frozen_input.csv").write_bytes(original + b"\n")
        with self.assertRaisesRegex(EvaluationError, "Frozen input CSV"):
            self.run_fixture(StubCapture())
        (self.output / "frozen_input.csv").write_bytes(original)
        connection = sqlite3.connect(self.output / "checkpoint.sqlite3")
        connection.execute("UPDATE pages SET record_json='{}' WHERE ordinal=1")
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(EvaluationError, "integrity check"):
            self.run_fixture(StubCapture())

    def test_changed_archive_refuses_resume(self):
        self.run_fixture(StubCapture())
        original_report = (self.output / "live_evaluation_report.json").read_bytes()
        connection = sqlite3.connect(self.output / "checkpoint.sqlite3")
        connection.execute("UPDATE pages SET html_zlib=? WHERE ordinal=1", (b"corrupt",))
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(EvaluationError, "archive is corrupt"):
            self.run_fixture(StubCapture())
        self.assertEqual((self.output / "live_evaluation_report.json").read_bytes(), original_report)

    def test_all_failed_captures_have_no_accuracy(self):
        report = self.run_fixture(StubCapture({row["normalized_url"]: capture.CaptureError("Fixture unreachable") for row in self.rows}))
        metrics = report["classification_metrics_on_scored_pages"]
        self.assertEqual(report["scored_coverage_all_selected_pages"], 0)
        self.assertIsNone(metrics["accuracy"])
        self.assertIsNone(metrics["recall"])
        self.assertIsNone(metrics["false_positive_rate"])
        self.assertIsNone(metrics["balanced_accuracy"])

    def test_missing_source_date_duplicates_and_balance_rejected(self):
        variants = []
        source = entries()
        source[0]["label_source"] = ""
        variants.append((source, "label_source"))
        source = entries()
        source[0]["label_checked_at"] = "2026/10/09"
        variants.append((source, "YYYY-MM-DD"))
        source = entries()
        source[0]["label_checked_at"] = (date.today() + timedelta(days=1)).isoformat()
        variants.append((source, "future"))
        source = entries()
        source[0]["url"] = "https://same.example:443/path#first"
        source[1]["url"] = "https://SAME.example/path#second"
        variants.append((source, "duplicate URL"))
        source = entries()
        source[0]["label"] = "phish"
        variants.append((source, "Expected 2 benign"))
        for candidate, message in variants:
            self.csv_path.write_bytes(csv_bytes(candidate))
            with self.subTest(message=message), self.assertRaisesRegex(EvaluationError, message):
                load_url_list(self.csv_path, capture.normalize_url, 2)

    def test_offline_preflight_does_not_load_model_or_write_reports(self):
        result = subprocess.run([sys.executable, str(ROOT / "32_evaluate_live_urls.py"),
                                 "--urls", str(self.csv_path), "--project-dir", str(RUNTIME_ROOT),
                                 "--per-class", "2", "--check-list", "--output-dir", str(self.output)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("No websites requested", result.stdout)
        self.assertFalse(self.output.exists())

    def test_zero_false_positives_has_nonzero_uncertainty(self):
        lower, upper = wilson_interval(0, 50)
        self.assertAlmostEqual(lower, 0)
        self.assertAlmostEqual(upper, 0.0713475991, places=8)
        self.assertIsNone(classification_metrics([])["accuracy"])

    def test_mismatched_threshold_prediction_rejected(self):
        detector = StubDetector()
        html = "<html>SCORE=0.5;</html>"
        result = detector.predict_html(html, "https://fixture.example/", "http_source")
        result["static_model_prediction"] = "benign"
        with self.assertRaisesRegex(EvaluationError, "fixed 0.50 threshold"):
            validate_prediction(result, detector.describe(), result["html_sha256"], result["page_url"])

    def test_shared_final_url_and_html_retained_and_reported(self):
        report = self.run_fixture(StubCapture())
        groups = report["duplicate_capture_groups"]["html_sha256"]
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]["conflicting_labels"])
        self.assertEqual(report["classification_metrics_on_scored_pages"]["scored_pages"], 4)

    def test_process_lock_releases_and_prevents_concurrent_run(self):
        path = self.folder / "run.lock"
        with RunLock(path):
            with self.assertRaisesRegex(EvaluationError, "Another evaluation process"):
                with RunLock(path):
                    pass
        with RunLock(path):
            pass

    def test_real_http_capture_integrates_and_partial_response_is_failed(self):
        requests = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                if self.path == "/missing":
                    self.send_error(404)
                    return
                score = "0.9" if self.path == "/phish" else "0.1"
                body = (f'<html><script src="/external.js"></script>SCORE={score};</html>').encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body) + (100 if self.path == "/short" else 0)))
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True
            def log_message(self, *unused):
                pass
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            source = entries()
            for row, path in zip(source, ("/benign", "/missing", "/phish", "/short")):
                row["url"] = f"http://public-test.example:{server.server_port}{path}"
            raw = csv_bytes(source)
            self.csv_path.write_bytes(raw)
            rows, _ = load_url_list(self.csv_path, capture.normalize_url, 2)
            # Loopback permission exists only inside this controlled test fixture.
            with patch.object(capture, "_public_addresses", return_value=["127.0.0.1"]):
                report = self.run_fixture(capture, rows=rows, raw=raw)
            self.assertEqual(report["classification_metrics_on_scored_pages"]["confusion_matrix"], {"TN": 1, "FP": 0, "FN": 0, "TP": 1})
            self.assertEqual(report["classification_metrics_on_scored_pages"]["scored_pages"], 2)
            self.assertEqual(sum(x["capture_failed"] for x in report["coverage_by_class"].values()), 2)
            self.assertNotIn("/external.js", requests)
            self.assertTrue(all(row["prediction"] == "" for row in self.predictions() if row["status"] == "capture_failed"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
