"""Regression tests with local HTTP fixtures and a temporary synthetic model.

No real dataset/model is trained, changed or used to select parameters here.
Run with the project's original dependencies, from the extracted project root.
"""
import copy
import gzip
import http.client
import json
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from detector_runtime import capture
from detector_runtime import model as runtime
from detector_runtime.ui import make_server


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class FixtureHandler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = b"<!doctype html><html><body>Test page</body></html>"
                if self.path == "/private-redirect":
                    self.send_response(302)
                    self.send_header("Location", "http://127.0.0.2/inside")
                    self.end_headers()
                    return
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/html")
                    self.end_headers()
                    return
                if self.path == "/gzip":
                    body = gzip.compress(body)
                if self.path == "/broken-gzip":
                    body = gzip.compress(body)[:-8]
                if self.path == "/large-gzip":
                    body = gzip.compress(b"A" * 10000)
                if self.path == "/invalid-utf8":
                    body += b"\xff"
                self.send_response(403 if self.path == "/forbidden" else 200)
                self.send_header("Content-Type", "application/json" if self.path == "/json" else "text/html; charset=utf-8")
                if "gzip" in self.path:
                    self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body) + (30 if self.path == "/short" else 0)))
                self.end_headers()
                self.wfile.write(body)

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://capture.example:{cls.server.server_port}"
        cls.original_resolver = staticmethod(capture._public_addresses)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def local_resolver(self, host, port):
        # Only this test fixture gets a loopback address. Production resolver
        # and redirect validation remain exercised for every other host.
        if host == "capture.example":
            return ["127.0.0.1"]
        return self.original_resolver(host, port)

    def test_source_and_gzip_capture_are_complete(self):
        with patch.object(capture, "_public_addresses", self.local_resolver):
            plain = capture.capture_url(self.url + "/html")
            gz = capture.capture_url(self.url + "/gzip")
        self.assertEqual(plain.html, gz.html)
        self.assertEqual(plain.metadata["capture_basis"], "http_source")
        self.assertTrue(plain.metadata["capture_complete"])
        self.assertFalse(plain.metadata["javascript_executed"])

    def test_redirect_records_final_url(self):
        with patch.object(capture, "_public_addresses", self.local_resolver):
            result = capture.capture_url(self.url + "/redirect")
        self.assertEqual(result.metadata["final_url"], self.url + "/html")
        self.assertEqual(len(result.metadata["redirects"]), 1)

    def test_private_redirect_is_blocked(self):
        with patch.object(capture, "_public_addresses", self.local_resolver):
            with self.assertRaises(capture.CaptureError):
                capture.capture_url(self.url + "/private-redirect")

    def test_invalid_captures_have_no_prediction(self):
        for path in ["/forbidden", "/json", "/short", "/broken-gzip", "/invalid-utf8"]:
            with self.subTest(path=path), patch.object(capture, "_public_addresses", self.local_resolver):
                with self.assertRaises(capture.CaptureError):
                    capture.capture_url(self.url + path)

    def test_bounded_decompression_and_source_size(self):
        with patch.object(capture, "_public_addresses", self.local_resolver):
            with self.assertRaises(capture.CaptureError):
                capture.capture_url(self.url + "/large-gzip", max_bytes=512)
            with self.assertRaises(capture.CaptureError):
                capture.capture_url(self.url + "/html", max_bytes=12)

    def test_private_and_mixed_dns_answers_are_blocked(self):
        for address in ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "::ffff:127.0.0.1", "224.0.0.1"]:
            rows = [(2, 1, 6, "", ("8.8.8.8", 80)), (2, 1, 6, "", (address, 80))]
            with self.subTest(address=address), patch.object(capture.socket, "getaddrinfo", return_value=rows):
                with self.assertRaises(capture.CaptureError):
                    capture._public_addresses("dns.example", 80)

    def test_invalid_urls(self):
        for url in ["file:///tmp/a", "javascript:alert(1)", "https://user:secret@example.com/", "https://example.com:99999/", "https://example.com/\r\nX:header", "https://bad..example/"]:
            with self.subTest(url=url), self.assertRaises(capture.CaptureError):
                capture.normalize_url(url)

    def test_bom_and_meta_encoding(self):
        html = "<html><p>你好</p></html>"
        self.assertEqual(capture.decode_html(html.encode("utf-16"))[0], html)
        text = '<html><meta charset="windows-1252"><p>café</p></html>'
        self.assertEqual(capture.decode_html(text.encode("cp1252"))[0], text)


class ModelAndUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import lightgbm as lgb
        import numpy as np
        import joblib
        from importlib.metadata import version
        cls.tmp = tempfile.TemporaryDirectory(prefix="synthetic_detector_qa_")
        cls.root = Path(cls.tmp.name)
        package_root = Path(__file__).resolve().parents[1]
        # Extract the ZIP into the existing project root before running tests.
        for name in [runtime.BASE_FILE, runtime.CONTEXT_FILE]:
            shutil.copyfile(package_root / name, cls.root / name)
        original = json.loads((package_root / runtime.FREEZE_REL).read_text(encoding="utf-8-sig"))
        spec = copy.deepcopy(original["specification"])
        cls.adapter = runtime.FeatureAdapter(cls.root)
        cls.adapter.verify_contract(spec["feature_extraction_contract"])
        names = cls.adapter.names
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 8, (600, 106)).astype(np.float32)
        y = (x[:, 0] > 4).astype(np.int32)
        hp = {**spec["training_hyperparameters"], "n_estimators": 30, "min_child_samples": 8, "n_jobs": 1}
        cls.toy = lgb.LGBMClassifier(**hp)
        cls.toy.fit(x[:480], y[:480], eval_set=[(x[480:], y[480:])], eval_metric="binary_logloss",
                    feature_name=names, callbacks=[lgb.early_stopping(5, verbose=False)])
        spec["training_hyperparameters"] = hp
        spec["best_iteration"] = cls.toy.best_iteration_
        spec["training_package_versions"] = {k: version(k) for k in spec["training_package_versions"]}
        spec["extraction_package_versions"] = {k: version(k) for k in spec["extraction_package_versions"]}
        spec["development_class_counts"]["train"] = {"benign": int(sum(y[:480] == 0)), "phish": int(sum(y[:480] == 1))}
        old_root = runtime.PureWindowsPath(spec["input_files"]["trainer"]["path"]).parent
        refs = {}
        for key in ["base_extractor", "context_extractor", "context_schema", "run_config", "model_bundle"]:
            relative = runtime.PureWindowsPath(spec["input_files"][key]["path"]).relative_to(old_root)
            refs[key] = cls.root.joinpath(*relative.parts)
            refs[key].parent.mkdir(parents=True, exist_ok=True)
        refs["context_schema"].write_text(json.dumps(spec["feature_extraction_contract"]), encoding="utf-8")
        refs["run_config"].write_text('{"synthetic_qa":true}\n', encoding="utf-8")
        cls.bundle = {"bundle_version": "form_context_1.0", "model_name": spec["model_name"], "estimator": cls.toy,
            "feature_names": names, "label_mapping": {"benign": 0, "phish": 1}, "threshold": 0.5,
            "feature_extraction_contract": spec["feature_extraction_contract"], "base_extractor_code_sha256": runtime.BASE_SHA,
            "extractor_code_sha256": runtime.CONTEXT_SHA, "run_config_sha256": runtime.digest(refs["run_config"]),
            "package_versions": spec["training_package_versions"], "training_label_counts": spec["development_class_counts"]["train"]}
        joblib.dump(cls.bundle, refs["model_bundle"], compress=3)
        for key, path in refs.items():
            spec["input_files"][key]["sha256"] = runtime.digest(path)
        spec["model_bundle_sha256"] = runtime.digest(refs["model_bundle"])
        cls.spec = spec
        cls.spec_sha = runtime.object_sha(spec)
        frozen = {"specification": spec, "specification_sha256": cls.spec_sha}
        freeze = cls.root / runtime.FREEZE_REL
        freeze.parent.mkdir(parents=True)
        freeze.write_text(json.dumps(frozen), encoding="utf-8")
        # Test-only patch to load a synthetic model. Production CLI has no
        # bypass flag and always requires the user's exact frozen spec hash.
        with patch.object(runtime, "FROZEN_SPEC_SHA", cls.spec_sha):
            cls.detector = runtime.FrozenDetector(cls.root, threads=1)
        cls.example = (package_root / "examples/example_page.html").read_text()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_feature_and_prediction_parity(self):
        import numpy as np
        url = "https://example.com/login"
        expected, quality = self.adapter.context.extract_enhanced_static_features(
            self.example, url, self.adapter.base, **self.adapter.parameters, full_html_characters=len(self.example))
        result = self.detector.predict_html(self.example, url, "browser_rendered")
        self.assertEqual(result["features"], expected)
        self.assertEqual(result["quality_flags"], quality)
        x = np.asarray([[expected[n] for n in self.adapter.names]], dtype=np.float32)
        score = float(self.toy.booster_.predict(x, num_iteration=self.spec["best_iteration"], num_threads=1)[0])
        self.assertEqual(result["phishing_score"], score)
        self.assertEqual(result["static_model_prediction"], "phish" if score >= 0.5 else "benign")
        self.assertEqual(result["model"]["feature_count"], 106)
        self.assertFalse(result["model"]["model_trained"])

    def test_complete_url_capture_reaches_frozen_inference(self):
        from detector_runtime.reports import analyze_url
        CaptureTests.setUpClass()
        try:
            with patch.object(capture, "_public_addresses", return_value=["127.0.0.1"]):
                result = analyze_url(self.detector, CaptureTests.url + "/redirect")
                failed = analyze_url(self.detector, CaptureTests.url + "/forbidden")
            self.assertEqual(result["status"], "prediction_complete")
            self.assertEqual(result["capture"]["final_url"], CaptureTests.url + "/html")
            self.assertTrue(result["capture_basis_verified"])
            self.assertIsInstance(result["phishing_score"], float)
            self.assertEqual(failed["status"], "capture_failed")
            self.assertIsNone(failed["phishing_score"])
            self.assertIsNone(failed["static_model_prediction"])
        finally:
            CaptureTests.tearDownClass()

    def test_cli_check_saves_completed_offline_report(self):
        import contextlib
        import io
        import runpy
        import sys
        script = Path(__file__).resolve().parents[1] / "29_run_webpage_detector.py"
        output = io.StringIO()
        with patch.object(runtime, "FROZEN_SPEC_SHA", self.spec_sha), patch.object(sys, "argv", [str(script), "--project-dir", str(self.root), "--check"]), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as stopped:
                runpy.run_path(str(script), run_name="__main__")
            self.assertEqual(stopped.exception.code, 0)
        self.assertIn("Frozen runtime verified: 106 features", output.getvalue())
        reports = [json.loads(p.read_text()) for p in (self.root / "reports/webpage_analysis_v1").glob("*.json")]
        self.assertTrue(any(r["status"] == "frozen_runtime_check_complete" and not r["websites_requested"] for r in reports))

    def test_full_html_length_preserved_before_cap(self):
        html = self.example + " " * 4_000_001
        features, quality = self.adapter.extract(html, "https://example.com/")
        self.assertEqual(features["html_character_count"], len(html))
        self.assertEqual(quality["html_analysis_truncated"], 1)

    def test_challenge_is_inconclusive(self):
        result = self.detector.predict_html("<html><body>Verify you are human</body></html>", "https://example.com/")
        self.assertEqual(result["status"], "inconclusive_capture")
        self.assertEqual(result["display_result"], "Inconclusive capture")

    def test_changed_model_hash_is_rejected_before_deserialization(self):
        path = self.detector.refs["model_bundle"]
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"changed")
            with patch.object(runtime, "FROZEN_SPEC_SHA", self.spec_sha), patch("joblib.load") as loader:
                with self.assertRaises(ValueError):
                    runtime.FrozenDetector(self.root)
                loader.assert_not_called()
        finally:
            path.write_bytes(original)

    def test_feature_order_and_threshold_changes_rejected(self):
        import lightgbm as lgb
        for field, value in [("feature_names", list(reversed(self.adapter.names))), ("threshold", 0.9)]:
            invalid = {**self.bundle, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                runtime._validate_bundle(invalid, self.spec, self.bundle["run_config_sha256"], lgb)

    def test_interface_valid_html_and_csrf_rejection(self):
        server = make_server(self.detector, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
            conn.request("GET", "/")
            response = conn.getresponse()
            html = response.read().decode()
            self.assertEqual(response.status, 200)
            import re
            token = re.search(r'name="detector-token" content="([^"]+)"', html).group(1)
            payload = json.dumps({"mode": "html", "page_url": "https://example.com/", "html": self.example, "basis": "browser_rendered"})
            headers = {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{server.server_port}", "X-Detector-Token": token}
            conn.request("POST", "/api/analyze", payload, headers)
            response = conn.getresponse()
            result = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(result["status"], "prediction_complete")
            self.assertTrue((self.root / "reports/webpage_analysis_v1" / result["saved_report_filename"]).is_file())
            conn.request("POST", "/api/analyze", payload, {**headers, "Origin": "https://outside.example"})
            response = conn.getresponse()
            response.read()
            self.assertEqual(response.status, 403)
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
