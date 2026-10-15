"""Validate a fresh capture collector on owned, controlled HTTP webpages.

Keep the verified 21_validate_browser_behaviour.py beside this file. Reuse
playwright==1.63.0 and its installed Chromium in this project's .venv.
Run: python 23_validate_resource_capture.py

The built-in pages are served on two ephemeral 127.0.0.1 HTTP ports. Chromium
really receives the HTML, CSS, external JavaScript and dummy responses from
these owned servers. Every other page request is aborted. No real website or
PhreshPhish record is opened. Synthetic scenario roles are NOT dataset labels.

Each JSON evidence bundle saves initial response HTML, final DOM HTML, full
response bodies as Base64 plus SHA-256, DOM snapshots, events and requests.
Checks compare the received resource bytes to the owned fixture definitions.
This validates acquisition, not real-world accuracy or a labelled dataset.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import platform
import re
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

VERSION = "1.0"
PILOT_SHA = "35333800b2f2c5f40cee5c25518beba09f53475c724b6ad51f5d9b20cba3048b"
REPORT = "resource_capture_report.json"
MAX_RESPONSES = 150
MAX_BODY_BYTES = 1_048_576
MAX_TOTAL_BODY_BYTES = 8_388_608
CSS = b"body{font-family:sans-serif;padding:16px}label{display:block}"
SCENARIOS = {
    "normal_same_origin_login": "ordinary_control",
    "ordinary_cross_origin_login": "ordinary_control",
    "ordinary_password_visibility_toggle": "ordinary_control",
    "delayed_contact_form": "ordinary_control",
    "mocked_external_script_form": "behaviour_scenario",
    "encoded_delayed_form": "behaviour_scenario",
    "japanese_label_xhr": "behaviour_scenario",
    "delayed_dummy_submission": "behaviour_scenario",
    "unknown_request_containment": "network_boundary_control",
}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def save_json(path, value):
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_pilot(root):
    path = root / "21_validate_browser_behaviour.py"
    if not path.is_file() or sha_bytes(path.read_bytes()) != PILOT_SHA:
        raise ValueError("Keep the original verified 21_validate_browser_behaviour.py beside script 23.")
    spec = importlib.util.spec_from_file_location("verified_pilot21", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixtures(pilot, page_origin, sink_origin):
    # Rebuild BEFORE Base64 encoding so encoded fixture URLs also point locally.
    pilot.PAGE_ORIGIN, pilot.SINK_ORIGIN = page_origin, sink_origin
    by_id = {row["id"]: row for row in pilot.cases()}
    rows = []
    for case_id, role in SCENARIOS.items():
        row = dict(by_id[case_id])
        source = row["html"]
        scripts = re.findall(r"<script>(.*?)</script>", source, re.S)
        script = row.get("asset_js", "\n".join(scripts) or "/* Owned collector control: no page action. */")
        script_path = "/cases/" + case_id + ".js"
        if scripts:
            source = re.sub(r"<script>.*?</script>", "<script src='" + script_path + "'></script>", source, flags=re.S)
        elif "asset_js" in row:
            source = source.replace("src='/fixture.js'", "src='" + script_path + "'")
        else:
            source = source.replace("</body>", "<script src='" + script_path + "'></script></body>")
        source = source.replace("</head>", "<link rel='stylesheet' href='/fixture.css'></head>")
        row.update(html=source, external_js=script, script_path=script_path,
                   scenario_role_not_a_dataset_label=role)
        rows.append(row)
    return rows


class ScriptReferences(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sources = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag.lower() == "script" and attrs.get("src"):
            self.sources.append(attrs["src"])


def start_servers(pilot):
    resources, received, lock = {}, [], threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def respond(self, status, content_type, body, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def handle_owned_request(self):
            path = urlsplit(self.path).path
            size = int(self.headers.get("Content-Length", "0"))
            if size > 65_536 or size < 0:
                self.respond(413, "text/plain", b"Owned request limit")
                return
            body = self.rfile.read(size) if size else b""
            with lock:
                received.append({"port": self.server.server_port, "path": path,
                                 "method": self.command, "body": body.decode("utf-8", errors="replace")})
            if path == "/collect" and self.command in ("POST", "OPTIONS"):
                self.respond(204, "text/plain", b"", {
                    "Access-Control-Allow-Origin": page_origin,
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "content-type"})
            elif path == "/favicon.ico" and self.command == "GET":
                self.respond(204, "image/x-icon", b"")
            elif self.command == "GET" and path in resources:
                content_type, content = resources[path]
                self.respond(200, content_type, content)
            else:
                self.respond(404, "text/plain", b"Unknown owned fixture route")

        do_GET = handle_owned_request
        do_POST = handle_owned_request
        do_OPTIONS = handle_owned_request

    servers = []
    threads = []
    try:
        for _ in range(2):
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.daemon_threads = True
            servers.append(server)
        page_origin = "http://127.0.0.1:" + str(servers[0].server_port)
        sink_origin = "http://127.0.0.1:" + str(servers[1].server_port)
        rows = fixtures(pilot, page_origin, sink_origin)
        resources["/fixture.css"] = ("text/css; charset=utf-8", CSS)
        for row in rows:
            resources[urlsplit(row["url"]).path] = ("text/html; charset=utf-8", row["html"].encode("utf-8"))
            resources[row["script_path"]] = ("application/javascript; charset=utf-8", row["external_js"].encode("utf-8"))
        for server in servers:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            threads.append(thread)
        return rows, resources, received, lock, servers, threads, page_origin, sink_origin
    except BaseException:
        for server in servers:
            server.server_close()
        raise


def capture_case(browser, row, pilot, resources, received, lock, origins, observe_ms, action_ms):
    started = time.monotonic()
    events, requests, responses, errors, navigations = [], [], [], [], []
    counters = Counter()
    context = browser.new_context(service_workers="block", accept_downloads=False)
    context.set_default_timeout(6000)
    context.set_default_navigation_timeout(6000)
    with lock:
        server_start = len(received)

    def record_event(source, event):
        if len(events) >= pilot.MAX_EVENTS:
            counters["trace_events_dropped"] += 1
            return
        event = dict(event)
        event["case_elapsed_ms"] = (time.monotonic() - started) * 1000
        event["phase"] = "dummy_probe" if counters["probe_started"] else "passive"
        event["is_main_frame"] = source["frame"] == source["page"].main_frame
        if "snapshot" in event:
            event["snapshot"] = pilot.enrich_snapshot(event["snapshot"])
        events.append(event)

    def route_request(route):
        request = route.request
        parsed = urlsplit(request.url)
        request_origin = pilot.origin(request.url)
        allowed_gets = {urlsplit(row["url"]).path, row["script_path"], "/fixture.css", "/favicon.ico"}
        allowed = request_origin in origins and not parsed.query and (
            request.method == "GET" and parsed.path in allowed_gets or
            request.method in ("POST", "OPTIONS") and parsed.path == "/collect")
        body = request.post_data or ""
        decoded = unquote(request.url + "\n" + body)
        item = {"url": request.url, "method": request.method, "resource_type": request.resource_type,
                "navigation": request.is_navigation_request(),
                "initial_fixture_request": request.url == row["url"] and request.method == "GET",
                "case_elapsed_ms": (time.monotonic() - started) * 1000,
                "phase": "dummy_probe" if counters["probe_started"] else "passive",
                "cross_origin_from_fixture": request_origin != pilot.origin(row["url"]),
                "contains_dummy_identity": pilot.USER_MARKER in decoded,
                "contains_dummy_password": pilot.PASSWORD_MARKER in decoded,
                "post_body": body, "page_request_sent_to_live_network": False,
                "sent_to_owned_loopback_server": bool(allowed),
                "route_action": "continued_owned_loopback" if allowed else "aborted_unknown_request"}
        if len(requests) < pilot.MAX_REQUESTS:
            requests.append(item)
        else:
            counters["request_events_dropped"] += 1
        if allowed:
            route.continue_()
        else:
            route.abort()

    def retain_response(request):
        try:
            response = request.response()
            if response is None:
                return
            if len(responses) >= MAX_RESPONSES:
                counters["response_events_dropped"] += 1
                return
            body = b"" if response.status in (204, 304) else response.body()
            if len(body) > MAX_BODY_BYTES or counters["response_body_bytes"] + len(body) > MAX_TOTAL_BODY_BYTES:
                counters["response_body_budget_exceeded"] += 1
                return
            counters["response_body_bytes"] += len(body)
            responses.append({"url": response.url, "method": request.method,
                              "resource_type": request.resource_type, "status": response.status,
                              "headers": response.all_headers(), "body_bytes": len(body),
                              "body_sha256": sha_bytes(body),
                              "body_base64": base64.b64encode(body).decode("ascii"),
                              "body_encoding": "base64_of_playwright_decoded_response_body",
                              "body_complete": True,
                              "case_elapsed_ms": (time.monotonic() - started) * 1000})
        except Exception as error:
            counters["response_body_errors"] += 1
            errors.append("Response capture: " + str(error)[:1000])

    try:
        context.expose_binding("__pilot_record21", record_event)
        context.add_init_script(script=pilot.INSTRUMENT_JS)
        context.route("**/*", route_request)
        context.route_web_socket("**/*", lambda ws: ws.close())
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)[:1500]))
        context.on("requestfinished", retain_response)
        page.on("framenavigated", lambda frame: navigations.append({"url": frame.url,
                "case_elapsed_ms": (time.monotonic() - started) * 1000})
                if frame == page.main_frame and frame.url != "about:blank" else None)
        page.goto(row["url"], wait_until="domcontentloaded")
        page.wait_for_function("ms => window.__pilot21 && performance.now()-window.__pilot21.started >= ms",
                               arg=observe_ms, timeout=observe_ms + 6000)
        page.evaluate("window.__pilot21.flush()")
        initial_events = [e for e in events if e["kind"] == "dom_content_loaded" and
                          e["is_main_frame"] and e["document_url"] == row["url"]]
        if len(initial_events) != 1:
            raise ValueError("Expected exactly one initial document snapshot.")
        initial = initial_events[0]["snapshot"]
        before = pilot.enrich_snapshot(page.evaluate("window.__pilot21.snapshot()"))
        if row["action"] in ("submit", "contact", "toggle"):
            counters["probe_started"] = 1
            for field in before["fields"]:
                if field["disabled"] or not field["rendered_visible"]:
                    continue
                marker = pilot.PASSWORD_MARKER if field["passwordlike"] else pilot.USER_MARKER if field["identitylike"] else None
                if marker:
                    page.locator("input,textarea").nth(field["dom_index"]).fill(marker)
            page.locator("#toggle" if row["action"] == "toggle" else "#go").click(no_wait_after=True)
        page.evaluate("window.__pilot21.actionObservationStarted=performance.now()")
        page.wait_for_function("ms => window.__pilot21 && performance.now()-window.__pilot21.actionObservationStarted >= ms",
                               arg=action_ms, timeout=action_ms + 6000)
        page.evaluate("window.__pilot21.flush()")
        final = pilot.enrich_snapshot(page.evaluate("window.__pilot21.snapshot()"))
        final_html = page.content()
        snapshots = [initial, before, final] + [e["snapshot"] for e in events if "snapshot" in e]
        quality = {"snapshot_storage_cut": int(any(s["storage_cut"] for s in snapshots)),
                   "trace_events_dropped": counters["trace_events_dropped"] + max(s["trace_events_dropped"] for s in snapshots),
                   "request_events_dropped": counters["request_events_dropped"],
                   "response_events_dropped": counters["response_events_dropped"],
                   "response_body_errors": counters["response_body_errors"],
                   "response_body_budget_exceeded": counters["response_body_budget_exceeded"],
                   "page_script_error_count": len(errors)}
        features = pilot.summarize_result(events, requests, navigations, initial, final)
        # That earlier metric refers to mocked route.fulfill; these scripts were
        # received through real owned HTTP, so give their count a distinct name.
        features.pop("locally_fulfilled_fixture_script_requests")
        script_responses = [r for r in responses if r["resource_type"] == "script" and r["status"] == 200]
        features["captured_external_script_response_count"] = len(script_responses)
        checks = pilot.check_case({**row, "expected": {k: v for k, v in row["expected"].items()
                                 if k != "external_fixture_scripts"}}, features, quality)
        html_responses = [r for r in responses if r["url"] == row["url"] and r["method"] == "GET"]
        received_html = base64.b64decode(html_responses[0]["body_base64"]).decode("utf-8") if len(html_responses) == 1 else None
        checks.append({"property": "initial_html_response_exact", "passed": received_html == row["html"]})
        resource_bytes_valid = True
        for response in responses:
            body = base64.b64decode(response["body_base64"], validate=True)
            path = urlsplit(response["url"]).path
            expected = resources[path][1] if response["method"] == "GET" and path in resources else b""
            resource_bytes_valid &= body == expected and len(body) == response["body_bytes"] and sha_bytes(body) == response["body_sha256"]
        checks.append({"property": "all_received_response_bodies_exact_and_hashed", "passed": resource_bytes_valid})
        parser = ScriptReferences()
        parser.feed(received_html or "")
        refs = [urljoin(row["url"], value) for value in parser.sources]
        checks.append({"property": "every_referenced_external_script_body_retained",
                       "passed": bool(refs) and all(any(r["url"] == url for r in script_responses) for url in refs)})
        checks.append({"property": "css_body_retained", "passed": any(r["resource_type"] == "stylesheet" and
                       base64.b64decode(r["body_base64"]) == CSS for r in responses)})
        with lock:
            server_requests = list(received[server_start:])
        server_markers = Counter()
        for request in server_requests:
            decoded = unquote(request["body"])
            server_markers["identity"] += pilot.USER_MARKER in decoded
            server_markers["password"] += pilot.PASSWORD_MARKER in decoded
        checks.append({"property": "server_received_expected_dummy_identity_and_password_requests",
                       "passed": server_markers["password"] == features["requests_containing_dummy_password"] and
                       server_markers["identity"] == features["requests_containing_dummy_identity"]})
        checks.append({"property": "final_dom_html_retained", "passed": isinstance(final_html, str) and "<html" in final_html})
        checks.append({"property": "page_requests_only_owned_loopback_or_aborted",
                       "passed": all(not r["page_request_sent_to_live_network"] and
                        (r["route_action"] == "aborted_unknown_request" or pilot.origin(r["url"]) in origins) for r in requests)})
        return {"case_id": row["id"], "scenario_role_not_a_dataset_label": row["scenario_role_not_a_dataset_label"],
                "fixture_url": row["url"], "status": "passed" if all(c["passed"] for c in checks) else "failed",
                "captured_utc": datetime.now(timezone.utc).isoformat(), "checks": checks, "quality": quality,
                "initial_html": received_html, "final_html": final_html,
                "initial_html_utf8_sha256": sha_bytes((received_html or "").encode("utf-8")),
                "final_html_utf8_sha256": sha_bytes(final_html.encode("utf-8")),
                "initial_snapshot": initial, "before_dummy_probe_snapshot": before, "final_snapshot": final,
                "behaviour_features": features, "events": events, "requests": requests, "responses": responses,
                "owned_server_requests": server_requests, "page_script_errors": errors,
                "dummy_probe_action": row["action"], "main_frame_navigations": navigations,
                "case_elapsed_seconds": time.monotonic() - started, "training_eligible": False}
    except Exception as error:
        return {"case_id": row["id"], "scenario_role_not_a_dataset_label": row["scenario_role_not_a_dataset_label"],
                "status": "failed", "error": str(error)[:4000], "events": events,
                "requests": requests, "responses": responses, "page_script_errors": errors,
                "checks": [{"property": "capture_completed", "passed": False}], "training_eligible": False}
    finally:
        context.close()


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "reports/fresh_capture_pilot_v1")
    parser.add_argument("--observe-ms", type=int, default=1000)
    parser.add_argument("--action-ms", type=int, default=800)
    args = parser.parse_args()
    if not 700 <= args.observe_ms <= 5000 or not 500 <= args.action_ms <= 5000:
        parser.error("Use --observe-ms 700..5000 and --action-ms 500..5000.")
    pilot = load_pilot(root)
    output = args.output_dir.resolve()
    reserved = [root / "reports" / name for name in (
        "features", "features_context_v2", "balanced", "baseline_training_v1_1",
        "form_context_training_v2", "capture_quality_v1", "browser_pilot_v1", "archived_browser_pilot_v1")]
    reserved.append(Path(r"D:\new dataset\phreshphish_dataset"))
    if root.is_relative_to(output) or any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in reserved):
        raise ValueError("Use a separate output folder outside datasets and earlier reports.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("This output folder already has files. For a new run use --output-dir .\\reports\\fresh_capture_pilot_v1_retry")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise SystemExit("Install in .venv: python -m pip install playwright==1.63.0") from error
    if version("playwright") != "1.63.0":
        raise ValueError("Use the existing validated Playwright version: python -m pip install playwright==1.63.0")
    canonical = fixtures(pilot, "https://page.pilot.test", "https://sink.pilot.test")
    fixture_sha = pilot.object_digest(canonical)
    servers = threads = []
    try:
        rows, resources, received, lock, servers, threads, page_origin, sink_origin = start_servers(pilot)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, chromium_sandbox=True,
                args=["--disable-background-networking", "--disable-component-update", "--no-first-run"])
            try:
                output.mkdir(parents=True, exist_ok=True)
                bundles = output / "evidence"
                bundles.mkdir()
                config = {"script_version": VERSION, "script_sha256": sha_bytes(Path(__file__).read_bytes()),
                          "verified_script21_sha256": PILOT_SHA, "fixture_definitions_sha256": fixture_sha,
                          "python_version": platform.python_version(), "playwright_version": version("playwright"),
                          "chromium_version": browser.version, "chromium_sandbox_requested": True,
                          "observe_ms": args.observe_ms, "action_ms": args.action_ms,
                          "owned_loopback_origins": [page_origin, sink_origin],
                          "page_network_policy": "exact_owned_loopback_routes_only_all_other_requests_aborted",
                          "max_response_body_bytes": MAX_BODY_BYTES, "max_total_response_body_bytes_per_case": MAX_TOTAL_BODY_BYTES,
                          "max_response_records_per_case": MAX_RESPONSES}
                print(f"Resource capture pilot v{VERSION}: {len(rows)} controlled cases; real owned HTTP responses.", flush=True)
                print("4 ordinary controls + 4 behaviour scenarios + 1 network-boundary check; none are training labels.", flush=True)
                summaries = []
                for n, row in enumerate(rows, 1):
                    result = capture_case(browser, row, pilot, resources, received, lock,
                                          {page_origin, sink_origin}, args.observe_ms, args.action_ms)
                    path = bundles / (row["id"] + ".json")
                    save_json(path, result)
                    summary = {k: result[k] for k in ("case_id", "scenario_role_not_a_dataset_label", "status", "checks")}
                    summary.update(evidence_file="evidence/" + path.name, evidence_sha256=sha_bytes(path.read_bytes()),
                                   behaviour_features=result.get("behaviour_features"), quality=result.get("quality"),
                                   response_count=len(result.get("responses", [])), error=result.get("error"))
                    summaries.append(summary)
                    print(f"  {n}/{len(rows)} {row['id']}: {result['status'].upper()} (evidence saved)", flush=True)
                passed = sum(r["status"] == "passed" for r in summaries)
                report = {"status": "resource_capture_pilot_complete" if passed == len(rows) else "resource_capture_pilot_checks_failed",
                          "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                          "controlled_cases": len(rows), "passed_cases": passed, "failed_cases": len(rows) - passed,
                          "scenario_counts_not_dataset_labels": dict(Counter(row["scenario_role_not_a_dataset_label"] for row in rows)),
                          "cases": summaries, "live_websites_requested": False, "real_dataset_loaded": False,
                          "model_trained": False, "phishing_accuracy_measured": False,
                          "official_test_evaluated": False, "training_eligible": False,
                          "previous_model_and_dataset_changed": False,
                          "limitations": [
                              "Owned synthetic scenarios verify capture and observation, not phishing accuracy or real labelled data.",
                              "Four ordinary controls and four behaviour scenarios are paired software tests, not balanced real classes. The ninth case checks request containment.",
                              "Cross-origin credential submission and dynamic forms occur in ordinary controls and do not establish phishing.",
                              "Body bytes come from Playwright response.body and are decoded HTTP response payloads, not a byte-exact wire capture.",
                              "Bounded windows, selected dummy interactions and capped main-document instrumentation can miss other behaviour.",
                              "Service workers are blocked; CSS and owned external JS are available here. General hostile or live URL capture is not enabled.",
                              "This does not recover the original missing external JavaScript from historical PhreshPhish records.",
                              "Real collection needs independently evidenced labels, provenance, capture timestamps and separate collection/split planning before model training."]}
                save_json(output / REPORT, report)
                print(f"Saved report: {(output / REPORT).resolve()}", flush=True)
                if passed != len(rows):
                    print("Capture checks failed. Upload the report before continuing.", flush=True)
                    return 1
                print(f"Resource capture pilot complete: {passed}/{len(rows)} controlled cases passed. No model trained or phishing accuracy measured.", flush=True)
                return 0
            finally:
                browser.close()
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Controlled capture interrupted. Preserve its evidence and use a new --output-dir to rerun.")
    except Exception as error:
        raise SystemExit("Resource capture stopped: " + str(error))
