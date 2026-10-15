"""Loopback interface for source and integrated browser analysis."""
import hmac
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .capture import MAX_HTML_BYTES
from .reports import analyze_url, save_report

WEB_DIR = Path(__file__).resolve().parent / "web"


def make_server(detector, port=8765):
    token = secrets.token_urlsafe(32)
    gate = threading.BoundedSemaphore(1)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Avoid logging URL query strings or pasted page contents.
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def valid_host(self):
            p = self.server.server_port
            return self.headers.get("Host") in {f"127.0.0.1:{p}", f"localhost:{p}"}

        def send_bytes(self, data, mime, status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, value, status=200):
            self.send_bytes(json.dumps(value, ensure_ascii=True, allow_nan=False).encode(), "application/json; charset=utf-8", status)

        def do_GET(self):
            if not self.valid_host():
                self.send_json({"error": "Open this interface through its local address."}, 403)
                return
            files = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            if self.path == "/api/state":
                self.send_json({"ready": True, "model_name": detector.spec["model_name"],
                                "feature_count": 106, "threshold": 0.5})
            elif self.path in files:
                filename, mime = files[self.path]
                data = (WEB_DIR / filename).read_bytes()
                if filename == "index.html":
                    data = data.replace(b"__CSRF_TOKEN__", token.encode())
                self.send_bytes(data, mime)
            else:
                self.send_json({"error": "Not found."}, 404)

        def do_POST(self):
            origins = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            supplied_token = self.headers.get("X-Detector-Token", "")
            if (not self.valid_host() or self.headers.get("Origin") not in origins
                    or self.headers.get("Sec-Fetch-Site") == "cross-site"
                    or not hmac.compare_digest(supplied_token, token)):
                self.close_connection = True
                self.send_json({"error": "Request did not come from this local interface. Refresh the page."}, 403)
                return
            if self.path != "/api/analyze":
                self.close_connection = True
                self.send_json({"error": "Not found."}, 404)
                return
            if self.headers.get_content_type() != "application/json" or self.headers.get("Transfer-Encoding"):
                self.close_connection = True
                self.send_json({"error": "Use a JSON request with a content length."}, 400)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_HTML_BYTES + 64 * 1024:
                    raise ValueError("Input is empty or exceeds 32 MiB.")
                if not gate.acquire(blocking=False):
                    self.close_connection = True
                    self.send_json({"error": "An analysis is already running. Wait for it to finish."}, 429)
                    return
                try:
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ValueError("Incomplete request.")
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        raise ValueError("Expected an analysis request.")
                    mode = payload.get("mode")
                    if mode == "url":
                        result = analyze_url(detector, payload.get("url"))
                    elif mode == "browser":
                        from .behavior import analyze_browser_url
                        result = analyze_browser_url(detector, payload.get("url"))
                    elif mode == "demo":
                        from .behavior_examples import analyze_demo
                        result = analyze_demo(detector, payload.get("case", "delayed_login"))
                    elif mode == "html":
                        html = payload.get("html")
                        if not isinstance(html, str) or len(html.encode()) > MAX_HTML_BYTES:
                            raise ValueError("HTML exceeds 32 MiB or is not text.")
                        result = detector.predict_html(html, payload.get("page_url"), payload.get("basis", "user_supplied_unknown"))
                        result["capture"] = {"input": "pasted_html", "capture_complete": "not_independently_verified"}
                    else:
                        raise ValueError("Choose browser, URL source, demo or saved HTML analysis.")
                    path = save_report(detector.root, result)
                    result["saved_report_filename"] = path.name
                    self.send_json(result)
                finally:
                    gate.release()
            except (OSError, ValueError, KeyError, TypeError) as e:
                self.close_connection = True
                self.send_json({"status": "analysis_failed", "display_result": "Analysis could not complete", "error": str(e),
                                "model_prediction_made": False, "phishing_score": None}, 400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
