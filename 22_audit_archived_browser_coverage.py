"""Audit passive browser coverage on a balanced TRAINING-only archive sample.

Keep scripts 09 and 21 beside this file. Install playwright==1.63.0 and
psutil==7.2.2 in the existing .venv, then run `python -m playwright install
chromium`. Existing duckdb and tldextract packages are reused.

Default: 50 benign + 50 phish, ranked reproducibly by seed 42 and record ID.
Selected pages are retained even when replay fails or exceeds a resource cap.
No buttons are clicked, credentials entered, live resources fetched, model
trained, or validation/test HTML opened. Inline JS DOES execute in Chromium.
This is limited offline replay, not a faithful replay of the original website.
"""

import argparse
import csv
import hashlib
import heapq
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

VERSION = "1.0"
ROOT = Path(__file__).resolve().parent
BASE_SHA = "4c845262a0cb27c085a2fc80c4d37f427addad5405b7319ba20438f7e18ab7ef"
PILOT_SHA = "35333800b2f2c5f40cee5c25518beba09f53475c724b6ad51f5d9b20cba3048b"
LABELS = ("benign", "phish")
MAX_HTML = 4_000_000
MAX_EVENTS = 200
STATE_NAME = "archived_browser_checkpoint.json"
REPORT_NAME = "archived_browser_coverage_report.json"
CSP = ("default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' data: blob:; "
       "style-src 'unsafe-inline' data:; img-src data:; font-src data:; "
       "connect-src 'none'; worker-src 'none'; frame-src 'none'; object-src 'none'; "
       "media-src 'none'; form-action 'none'; base-uri http: https:")
FEATURES = {
    "initial_passwordlike_controls": "Credential-cued controls in the first archived main-document DOMContentLoaded snapshot, within the 150-control cap.",
    "maximum_passwordlike_controls": "Maximum credential-cued controls across retained main-document snapshots, within the control cap.",
    "final_passwordlike_controls": "Credential-cued controls in the final archived main-document snapshot, within the control cap.",
    "final_native_password_controls": "Final browser type=password controls, within the control cap.",
    "forms_added_after_dom_ready": "Observed forms inserted after DOMContentLoaded; reinsertion can contribute.",
    "controls_added_after_dom_ready": "Observed input/textarea insertions after DOMContentLoaded.",
    "observed_attribute_changes": "Changes to the instrumented field/form attributes after DOMContentLoaded.",
    "observed_submit_events": "Submit events firing without any deliberate user interaction in this pilot.",
    "observed_submit_method_calls": "Instrumented direct form.submit calls during passive observation.",
    "observed_fetch_calls": "Instrumented fetch calls, including calls blocked by replay policy.",
    "observed_xhr_send_calls": "Instrumented XHR.send calls, including calls blocked by replay policy.",
    "observed_beacon_calls": "Instrumented beacon calls, including calls blocked by replay policy.",
    "observed_timer_callbacks": "Instrumented function-based setTimeout callbacks that fired.",
    "navigation_attempts_after_initial_load": "Intercepted navigation requests after the single archived source response; these are blocked.",
    "committed_main_frame_navigations_after_initial": "Committed main-frame navigation count after the initial response; aborted navigation can lead to an error document.",
    "initial_form_count": "Browser forms in the initial archived DOM snapshot.",
    "final_form_count": "Browser forms in the final archived DOM snapshot.",
    "initial_entry_control_count": "Initial input/textarea user-entry controls; browser count before the storage cap.",
    "final_entry_control_count": "Final input/textarea user-entry controls; browser count before the storage cap.",
}


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_module(name, path, expected_hash):
    if not path.is_file() or digest_file(path) != expected_hash:
        raise ValueError(f"Missing or changed verified dependency: {path.name}.")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def named_hash(mapping, name):
    hits = [value for key, value in mapping.items() if str(key).replace('\\', '/').split('/')[-1] == name]
    if len(hits) != 1:
        raise ValueError(f"Missing or ambiguous provenance hash for {name}.")
    return hits[0]


def select_training(manifest, balance, per_class, seed):
    heaps = {label: [] for label in LABELS}
    counts = {split: Counter() for split in ("train", "validation")}
    seen = set()
    groups = {split: set() for split in counts}
    required = {"sha256", "label", "split", "site_group", "source_file", "html_characters", "raw_html_sha256",
                "normalized_html_sha256", "sampled_similarity_candidate", "in_validation_sensitivity_subset"}
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Use the balanced manifest produced by script 08.")
        for row in reader:
            split, label, page_id = row["split"], row["label"], row["sha256"]
            if split not in counts or label not in LABELS or page_id in seen:
                raise ValueError("Unexpected or repeated balanced-manifest row.")
            if not all(re.fullmatch(r"[0-9a-f]{64}", row[k]) for k in ("sha256", "raw_html_sha256", "normalized_html_sha256")):
                raise ValueError("Invalid manifest fingerprint.")
            if not re.fullmatch(r"train-\d+\.parquet", row["source_file"]) or not row["site_group"] or int(row["html_characters"]) <= 0:
                raise ValueError("Invalid manifest source or HTML size.")
            if row["sampled_similarity_candidate"] not in ("0", "1") or row["in_validation_sensitivity_subset"] not in ("0", "1"):
                raise ValueError("Invalid manifest flag.")
            if split == "train" and (row["sampled_similarity_candidate"] != "0" or row["in_validation_sensitivity_subset"] != "0"):
                raise ValueError("Unexpected flag on a training row.")
            seen.add(page_id)
            counts[split][label] += 1
            groups[split].add(row["site_group"])
            if split != "train":
                continue
            meta = {"record_id": page_id, "label": label, "split": split, "site_group": row["site_group"],
                    "source_file": row["source_file"], "html_characters": int(row["html_characters"]),
                    "raw_html_sha256": row["raw_html_sha256"]}
            rank = int(hashlib.sha256(f"{seed}|{page_id}".encode()).hexdigest(), 16)
            item = (-rank, page_id, meta)
            heapq.heappush(heaps[label], item)
            if len(heaps[label]) > per_class:
                heapq.heappop(heaps[label])
    if {s: dict(c) for s, c in counts.items()} != balance["balanced_counts"] or groups["train"] & groups["validation"]:
        raise ValueError("Manifest counts or group separation differ from the balanced audit.")
    if any(len(heaps[label]) != per_class for label in LABELS):
        raise ValueError("Insufficient balanced training rows for the requested sample.")
    return sorted((item[2] for heap in heaps.values() for item in heap), key=lambda x: (x["source_file"], x["record_id"]))


GUARD_JS = r"""
(() => {
  const blocked = name => function() {
    window.__pilot_record22_policy?.(name).catch(()=>{});
    throw new DOMException('Disabled by offline archive replay policy','SecurityError');
  };
  for (const name of ['RTCPeerConnection','webkitRTCPeerConnection','WebTransport','Worker','SharedWorker']) {
    try { Object.defineProperty(window,name,{value:blocked(name),writable:false,configurable:false}); } catch {}
  }
  window.open = function() { window.__pilot_record22_policy?.('window.open').catch(()=>{}); return null; };
  document.addEventListener('securitypolicyviolation',e=>{
    window.__pilot_record22_policy?.('csp:'+e.effectiveDirective).catch(()=>{});
  });
})();
"""


def observe_archive(payload, pilot, playwright):
    from playwright.sync_api import Error as PlaywrightError
    browser = playwright.chromium.launch(headless=True, chromium_sandbox=True, timeout=10000,
        args=["--disable-background-networking", "--disable-component-update", "--no-first-run",
              "--host-resolver-rules=MAP * ~NOTFOUND", "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"])
    if payload["kind"] == "browser_check":
        result = {"status": "browser_ready", "chromium_version": browser.version, "chromium_sandbox_requested": True}
        browser.close()
        return result
    events, requests, navigations, errors = [], [], [], []
    quality = Counter()
    page = None
    initial_url = None
    initial = final = None
    observation_complete = False
    observed_start = time.monotonic()
    context = browser.new_context(service_workers="block", accept_downloads=False, permissions=[])
    context.set_default_timeout(5000)

    def event_callback(source, event):
        if source["frame"] != source["page"].main_frame:
            quality["non_main_frame_events_ignored"] += 1
            return
        if len(events) >= MAX_EVENTS:
            quality["trace_events_dropped"] += 1
            return
        try:
            if len(json.dumps(event, allow_nan=False)) > 128000 or not isinstance(event, dict):
                quality["invalid_or_large_trace_events"] += 1
                return
            if "snapshot" in event:
                event["snapshot"] = pilot.enrich_snapshot(event["snapshot"])
            events.append(event)
        except (ValueError, TypeError, KeyError, AttributeError):
            quality["invalid_or_large_trace_events"] += 1

    def policy_callback(source, name):
        quality["csp_violations" if str(name).startswith("csp:") else "blocked_api_calls"] += 1

    def route_callback(route):
        nonlocal initial_url
        request = route.request
        first = initial_url is None and request.is_navigation_request() and request.frame == page.main_frame
        item = {"navigation": request.is_navigation_request(), "initial_fixture_request": first,
                "contains_dummy_identity": False, "contains_dummy_password": False, "cross_origin_from_fixture": False,
                "route_action": "fulfilled_embedded_fixture" if first else "aborted_unknown_request"}
        if first:
            initial_url = request.url
            route.fulfill(status=200, content_type="text/html; charset=utf-8", body=payload["html"],
                          headers={"Content-Security-Policy": CSP, "X-DNS-Prefetch-Control": "off"})
        else:
            quality["blocked_requests"] += 1
            quality["blocked_external_script_requests"] += int(request.resource_type == "script")
            quality["blocked_navigation_requests"] += int(request.is_navigation_request())
            route.abort()
        if len(requests) < MAX_EVENTS:
            requests.append(item)
        else:
            quality["request_events_dropped"] += 1

    try:
        context.expose_binding("__pilot_record21", event_callback)
        context.expose_binding("__pilot_record22_policy", policy_callback)
        context.add_init_script(script=GUARD_JS + "\n" + pilot.INSTRUMENT_JS)
        context.route("**/*", route_callback)
        context.route_web_socket("**/*", lambda ws: (quality.update({"blocked_websockets": 1}), ws.close()))
        page = context.new_page()
        page.on("dialog", lambda dialog: dialog.dismiss())
        context.on("page", lambda other: (quality.update({"blocked_popups": 1}), other.close()))
        page.on("pageerror", lambda error: (quality.update({"page_script_errors": 1}), errors.append(str(error)[:500]) if len(errors) < 3 else None))
        page.on("framenavigated", lambda frame: navigations.append({"url": frame.url[:1000]})
                if frame == page.main_frame and frame.url != "about:blank" else None)
        page.goto(payload["page_url"], wait_until="domcontentloaded", timeout=5000)
        page.evaluate("window.__pilot21.flush()")
        candidates = [e["snapshot"] for e in events if e.get("kind") == "dom_content_loaded" and e.get("document_url") == initial_url]
        if len(candidates) == 1:
            initial = candidates[0]
        page.evaluate("window.__pilot21.observeStart=performance.now()")
        page.wait_for_function("ms => window.__pilot21 && performance.now()-window.__pilot21.observeStart >= ms",
                               arg=payload["observe_ms"], timeout=payload["observe_ms"]+5000)
        observation_complete = True
        page.evaluate("window.__pilot21.flush()")
        if not quality["blocked_navigation_requests"]:
            final = pilot.enrich_snapshot(page.evaluate("window.__pilot21.snapshot()"))
    except PlaywrightError as error:
        quality["browser_observation_errors"] += 1
        if len(errors) < 3:
            errors.append(str(error)[:500])
    finally:
        if initial is None:
            candidates = [e["snapshot"] for e in events if e.get("kind") == "dom_content_loaded" and e.get("document_url") == initial_url]
            if len(candidates) == 1:
                initial = candidates[0]
    try:
        snapshots = [s for s in [initial, final] + [e["snapshot"] for e in events if "snapshot" in e] if s]
        quality["snapshot_storage_cut"] = int(any(s["storage_cut"] for s in snapshots))
        quality["trace_events_dropped"] += max([s["trace_events_dropped"] for s in snapshots] or [0])
        quality["initial_snapshot_missing"] = int(initial is None)
        quality["final_snapshot_missing"] = int(final is None)
        quality["observation_window_incomplete"] = int(not observation_complete)
        features = {name: None for name in FEATURES}
        # Unavailable final observations stay missing, rather than becoming zero.
        if initial and final:
            summary = pilot.summarize_result(events, requests, navigations, initial, final)
            features.update({name: summary[name] for name in FEATURES if name in summary})
        else:
            counts = Counter(e.get("kind") for e in events)
            for feature, kind in [("observed_submit_events", "submit_event"), ("observed_submit_method_calls", "form_submit_method_call"),
                                  ("observed_fetch_calls", "fetch_call"), ("observed_xhr_send_calls", "xhr_send_call"),
                                  ("observed_beacon_calls", "send_beacon_call"), ("observed_timer_callbacks", "timer_callback_fired")]:
                features[feature] = counts[kind]
            features["forms_added_after_dom_ready"] = sum(e.get("forms_added", 0) for e in events)
            features["controls_added_after_dom_ready"] = sum(e.get("controls_added", 0) for e in events)
            features["observed_attribute_changes"] = sum(e.get("attributes_changed", 0) for e in events)
            features["navigation_attempts_after_initial_load"] = sum(r["navigation"] and not r["initial_fixture_request"] for r in requests)
            features["committed_main_frame_navigations_after_initial"] = max(0, len(navigations)-1)
            if snapshots:
                features["maximum_passwordlike_controls"] = max(s["passwordlike_count"] for s in snapshots)
        for prefix, snapshot in [("initial", initial), ("final", final)]:
            if snapshot:
                features[prefix+"_passwordlike_controls"] = snapshot["passwordlike_count"]
                features[prefix+"_form_count"] = snapshot["form_count"]
                features[prefix+"_entry_control_count"] = snapshot["entry_control_count"]
        if final:
            features["final_native_password_controls"] = final["native_password_count"]
        if any(v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0) for v in features.values()):
            raise ValueError("Invalid browser observation values.")
        completed = bool(initial and final and observation_complete)
        status = "observed" if completed else "partial_observation"
        return {"status": status, "browser_observation_completed": completed, "behaviour_features": features,
                "quality": dict(quality), "browser_error_examples": errors,
                "observed_event_count": len(events), "intercepted_request_count": len(requests),
                "observation_seconds": time.monotonic()-observed_start, "chromium_version": browser.version,
                "chromium_sandbox_requested": True, "page_request_forwarding_used": False,
                "user_interaction_performed": False}
    finally:
        context.close()
        browser.close()


def worker_main():
    payload = json.loads(sys.stdin.buffer.read())
    pilot = load_module("pilot21_worker", ROOT/"21_validate_browser_behaviour.py", PILOT_SHA)
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as p:
            result = observe_archive(payload, pilot, p)
    except Exception as error:
        result = {"status": "browser_failed", "error": str(error)[:1800], "browser_observation_completed": False}
    print(json.dumps(result, allow_nan=False), flush=True)


def stop_owned_processes(process, known):
    import psutil
    try:
        known.extend(psutil.Process(process.pid).children(recursive=True))
    except psutil.NoSuchProcess:
        pass
    for child in reversed(known):
        try:
            child.kill()  # psutil retains creation identity to guard PID reuse.
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()


def run_worker(payload, timeout_seconds, memory_mib):
    import psutil
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    known = []
    start, peak, stopped = time.monotonic(), 0, None
    input_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    stdout, stderr = b"", b""
    try:
        while True:
            try:
                parent = psutil.Process(process.pid)
                family = [parent, *parent.children(recursive=True)]
                known = family[1:]
                rss = 0
                for member in family:
                    try:
                        rss += member.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                if rss > memory_mib*1024**2:
                    stopped = "memory_budget_stop"
            except psutil.NoSuchProcess:
                pass
            if time.monotonic()-start > timeout_seconds:
                stopped = "worker_timeout"
            if stopped:
                stop_owned_processes(process, known)
                process.communicate(timeout=5)
                break
            try:
                # communicate() retains partially sent input/output after a
                # timeout. Short retries let this same thread sample RAM and
                # deadlines while safely draining both process pipes.
                stdout, stderr = process.communicate(input=input_bytes, timeout=.1)
                break
            except subprocess.TimeoutExpired:
                input_bytes = None
                if peak == 0 and time.monotonic()-start > 1:
                    stopped = "worker_process_monitor_unavailable"
        if stopped:
            result = {"status": stopped, "browser_observation_completed": False, "behaviour_features": {k: None for k in FEATURES}}
        else:
            if process.returncode != 0:
                raise ValueError("Archive worker failed: " + stderr.decode("utf-8", "replace")[:1000])
            if peak == 0:
                raise ValueError("The private worker's memory usage could not be monitored.")
            result = json.loads(stdout)
        result["worker_peak_summed_rss_mib"] = round(peak/1024**2, 2)
        result["worker_elapsed_seconds"] = time.monotonic()-start
        return result
    finally:
        if process.poll() is None:
            stop_owned_processes(process, known)
            process.communicate(timeout=5)


def write_csv(path, rows, columns):
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def prepare(args):
    common = load_module("static09_archive", ROOT/"09_extract_static_features.py", BASE_SHA)
    pilot = load_module("pilot21_archive", ROOT/"21_validate_browser_behaviour.py", PILOT_SHA)
    paths = {"manifest": args.manifest, "balance_report": args.balance_report, "extraction_report": args.extraction_report,
             "controlled_pilot_report": args.pilot_report, "static_script": ROOT/"09_extract_static_features.py",
             "controlled_script": ROOT/"21_validate_browser_behaviour.py"}
    hashes = {key: digest_file(path) for key, path in paths.items()}
    balance, base, controlled = read_json(args.balance_report), read_json(args.extraction_report), read_json(args.pilot_report)
    if balance["status"] != "balanced_experimental_manifests_complete" or base["status"] != "static_feature_extraction_complete":
        raise ValueError("Balanced manifests and original feature extraction must be complete.")
    if (hashes["manifest"] != balance["outputs"][args.manifest.name]["sha256"]
            or hashes["manifest"] != named_hash(base["inputs"]["inputs_sha256"], args.manifest.name)
            or hashes["balance_report"] != named_hash(base["inputs"]["inputs_sha256"], args.balance_report.name)
            or base["inputs"]["code_sha256"] != BASE_SHA):
        raise ValueError("The existing balanced-manifest/feature provenance differs.")
    if (controlled["status"] != "browser_behaviour_pilot_complete" or controlled["passed_cases"] != 17
            or controlled["failed_cases"] != 0 or controlled["config"]["script_sha256"] != PILOT_SHA
            or controlled["config"]["fixture_definitions_sha256"] != object_digest(pilot.cases())
            or controlled["credential_cue_checks"] != pilot.cue_checks()):
        raise ValueError("Complete script 21's original controlled pilot before this archive pilot.")
    selected = select_training(args.manifest, balance, args.per_class, args.seed)
    expected = {row["name"]: row for row in base["inputs"]["source_files"]}
    stats = []
    for name in sorted({row["source_file"] for row in selected}):
        p = args.data_dir/name
        stat = p.stat()
        row = {"name": name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if row != expected.get(name):
            raise ValueError(f"Training archive metadata differs from the original extraction: {name}.")
        stats.append(row)
    reserved = [args.data_dir, *[p.parent for p in paths.values() if p.parent != ROOT]]
    output = args.output_dir.resolve()
    if ROOT.is_relative_to(output) or any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in reserved):
        raise ValueError("Use a separate archive-pilot output folder outside the dataset and existing result folders.")
    return common, paths, hashes, selected, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT/"reports/balanced/balanced_training_validation_manifest.csv")
    parser.add_argument("--balance-report", type=Path, default=ROOT/"reports/balanced/balance_report.json")
    parser.add_argument("--extraction-report", type=Path, default=ROOT/"reports/features/feature_extraction_report.json")
    parser.add_argument("--pilot-report", type=Path, default=ROOT/"reports/browser_pilot_v1/browser_pilot_report.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"reports/archived_browser_pilot_v1")
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-limit", default="4GB", help="DuckDB source-decoding budget; separate from browser-worker RAM.")
    parser.add_argument("--observe-ms", type=int, default=3000)
    parser.add_argument("--worker-timeout", type=int, default=25)
    parser.add_argument("--worker-memory-mib", type=int, default=1536, help="Conservative summed RSS of each worker and descendants; shared memory can be counted more than once.")
    args = parser.parse_args()
    if not 1 <= args.per_class <= 250 or not 700 <= args.observe_ms <= 10000:
        parser.error("Use --per-class 1..250 and --observe-ms 700..10000.")
    if not max(20, args.observe_ms//1000+15) <= args.worker_timeout <= 60 or not 768 <= args.worker_memory_mib <= 4096:
        parser.error("Use --worker-timeout 20..60 (at least observation seconds + 15) and --worker-memory-mib 768..4096.")
    for dependency in ("playwright", "psutil"):
        version(dependency)
    if version("playwright") != "1.63.0":
        raise ValueError("Use .\\.venv\\Scripts\\python.exe -m pip install playwright==1.63.0 psutil==7.2.2, then .\\.venv\\Scripts\\python.exe -m playwright install chromium.")
    common, paths, hashes, selected, stats = prepare(args)
    print(f"Archive browser coverage pilot v{VERSION}: {args.per_class} benign + {args.per_class} phish, TRAINING only.", flush=True)
    print("Passive offline replay: inline JS executes; external resources and user interactions are disabled.", flush=True)
    health = run_worker({"kind": "browser_check"}, 20, args.worker_memory_mib)
    if health["status"] != "browser_ready":
        raise ValueError("Browser preflight failed: " + health.get("error", health["status"]) +
                         ". Chromium sandboxing and worker memory monitoring must remain available. "
                         "For browser-installation errors, run the .venv Chromium install command. Use an ordinary non-administrator terminal and paste this output if the check still fails.")
    config = {"script_version": VERSION, "script_sha256": digest_file(Path(__file__)), "input_hashes": hashes,
              "selected_pages": selected, "selected_source_files": stats, "data_dir": str(args.data_dir.resolve()),
              "python_version": platform.python_version(), "duckdb_version": common.duckdb.__version__,
              "tldextract_version": version("tldextract"), "playwright_version": version("playwright"),
              "psutil_version": version("psutil"), "chromium_version": health["chromium_version"],
              "per_class": args.per_class, "seed": args.seed, "observe_ms": args.observe_ms,
              "worker_timeout_seconds": args.worker_timeout, "worker_summed_rss_limit_mib": args.worker_memory_mib,
              "max_html_characters": MAX_HTML, "chromium_sandbox_requested": True, "csp": CSP,
              "page_network_policy": "single_archived_response_all_other_page_requests_blocked", "interaction_policy": "passive_no_clicks_or_credentials"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir/STATE_NAME
    if state_path.exists():
        state = read_json(state_path)
        if state["config"] != config:
            raise ValueError("Code, inputs, browser, selection or settings differ from this checkpoint. Choose a new --output-dir.")
    else:
        if any(args.output_dir.iterdir()):
            raise ValueError("Use an empty pilot output folder or its matching checkpoint.")
        state = {"config": config, "pages": {}}
        save_json(state_path, state)
    by_id = {row["record_id"]: row for row in selected}
    if not set(state["pages"]).issubset(by_id):
        raise ValueError("Unknown page in archive-pilot checkpoint.")
    for page_id, saved in state["pages"].items():
        if object_digest(saved["result"]) != saved["result_digest"] or saved["result"]["metadata"] != by_id[page_id]:
            raise ValueError("A checkpointed archive result or its provenance changed.")
    resumed = len(state["pages"])
    print(f"Chromium {health['chromium_version']}; {resumed} selected pages checkpointed; DuckDB {args.memory_limit}, one scan thread.", flush=True)
    resolver = common.ReferenceResolver()
    shards = defaultdict(list)
    for meta in selected:
        if meta["record_id"] not in state["pages"]:
            shards[meta["source_file"]].append(meta["record_id"])
    for index, (name, ids) in enumerate(sorted(shards.items()), 1):
        print(f"Archive {index}/{len(shards)}: {name}; {len(ids)} pending pages.", flush=True)
        con = common.duckdb.connect()
        seen = set()
        try:
            con.execute("SET threads=1")
            con.execute("SET preserve_insertion_order=false")
            con.execute("SET memory_limit=?", [args.memory_limit])
            path = args.data_dir/name
            slots = ",".join("?" for _ in ids)
            counts = dict(con.execute(f"SELECT sha256,count(*) FROM read_parquet(?) WHERE sha256 IN ({slots}) GROUP BY sha256", [str(path), *ids]).fetchall())
            if counts != {page_id: 1 for page_id in ids}:
                raise ValueError(f"Missing or duplicated selected records: {name}.")
            query = f"SELECT sha256,label,url,length(html),sha256(html),substr(html,1,{MAX_HTML}) FROM read_parquet(?) WHERE sha256 IN ({slots})"
            cursor = con.sql(query, params=[str(path), *ids])
            while (record := cursor.fetchone()) is not None:
                page_id, label, url, length, raw_hash, source = record
                meta = by_id[page_id]
                if page_id in seen or page_id not in ids or label != meta["label"] or length != meta["html_characters"] or raw_hash != meta["raw_html_sha256"]:
                    raise ValueError(f"Selected HTML/label fingerprint mismatch: {page_id}.")
                seen.add(page_id)
                page_url = common.canonical_page_url(url)
                if resolver.resolve(page_url, page_url)[1] != meta["site_group"]:
                    raise ValueError(f"Selected URL site group mismatch: {page_id}.")
                if length > MAX_HTML:
                    result = {"status": "html_oversize_not_executed", "browser_observation_completed": False,
                              "behaviour_features": {k: None for k in FEATURES}}
                else:
                    if hashlib.sha256(source.encode("utf-8")).hexdigest() != raw_hash:
                        raise ValueError("Python HTML bytes differ from the audited full source.")
                    static, static_quality = common.extract_static_features(source, page_url, resolver)
                    result = run_worker({"kind": "archive", "page_url": page_url, "html": source, "observe_ms": args.observe_ms},
                                        args.worker_timeout, args.worker_memory_mib)
                    result.setdefault("behaviour_features", {k: None for k in FEATURES})
                    result.setdefault("quality", {})
                    result["quality"].update({"external_script_references_without_bodies": static["script_external_count"],
                                               "static_html_parse_error": static_quality["html_parse_error"]})
                    if result["status"] == "observed" and any(result["quality"].values()):
                        result["status"] = "observed_with_gaps"
                result["metadata"] = meta
                result["selected_full_html_hash_verified"] = True
                result["label_provided_to_browser_worker"] = False
                state["pages"][page_id] = {"result": result, "result_digest": object_digest(result)}
                save_json(state_path, state)
                print(f"  Saved {len(state['pages'])}/{len(selected)}: {label}; {result['status']}.", flush=True)
            if seen != set(ids):
                raise ValueError(f"Selected source read incomplete: {name}.")
        except common.duckdb.OutOfMemoryException as error:
            raise MemoryError(f"DuckDB could not decode {name} within {args.memory_limit}.") from error
        finally:
            con.close()
    if hashes != {key: digest_file(path) for key, path in paths.items()}:
        raise ValueError("An input manifest, report or dependency script changed during the pilot.")
    for item in stats:
        stat = (args.data_dir/item["name"]).stat()
        if {"name": item["name"], "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != item:
            raise ValueError("A training archive changed during the pilot.")
    if set(state["pages"]) != set(by_id):
        raise ValueError("Not all selected pages have a saved coverage outcome.")
    pages = [state["pages"][row["record_id"]]["result"] for row in selected]
    schema = {"analysis_mode": "passive_limited_offline_replay", "feature_definitions": FEATURES,
              "feature_names": list(FEATURES), "metadata_columns": ["record_id", "label", "split", "site_group", "source_file", "status"],
              "missing_value_policy": "CSV blank/JSON null when an observation is unavailable; never substitute zero.",
              "quality_and_labels_are_classifier_inputs": False, "model_training_ready": False}
    save_json(args.output_dir/"passive_browser_feature_schema.json", schema)
    rows = [{**{k: p["metadata"][k] for k in schema["metadata_columns"] if k != "status"}, "status": p["status"], **p["behaviour_features"]} for p in pages]
    write_csv(args.output_dir/"training_pilot_browser_features.csv", rows, [*schema["metadata_columns"], *FEATURES])
    status_counts = {label: dict(Counter(p["status"] for p in pages if p["metadata"]["label"] == label)) for label in LABELS}
    coverage = {label: {"selected_pages": args.per_class,
                       "completed_observation_windows": sum(bool(p["browser_observation_completed"]) for p in pages if p["metadata"]["label"] == label),
                       "pages_with_missing_external_script_bodies": sum(p.get("quality", {}).get("external_script_references_without_bodies", 0) > 0 for p in pages if p["metadata"]["label"] == label),
                       "pages_with_observed_dom_form_additions": sum((p["behaviour_features"].get("forms_added_after_dom_ready") or 0) > 0 for p in pages if p["metadata"]["label"] == label),
                       "site_groups": len({p["metadata"]["site_group"] for p in pages if p["metadata"]["label"] == label}),
                       "missing_feature_values": sum(sum(v is None for v in p["behaviour_features"].values()) for p in pages if p["metadata"]["label"] == label)} for label in LABELS}
    from datetime import datetime, timezone
    report = {"status": "archived_browser_coverage_audit_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "selected_training_counts": {label: args.per_class for label in LABELS},
              "pages_resumed_from_checkpoint": resumed, "status_counts_by_label": status_counts, "coverage_by_label": coverage,
              "behaviour_feature_definitions": FEATURES, "pages": pages,
              "outputs": {name: {"sha256": digest_file(args.output_dir/name)} for name in ["passive_browser_feature_schema.json", "training_pilot_browser_features.csv"]},
              "model_trained": False, "phishing_accuracy_measured": False, "official_test_evaluated": False,
              "validation_html_opened": False, "input_files_unchanged": True, "raw_source_copies_written": False,
              "balanced_selected_sample_preserved": True, "failed_pages_replaced": False,
              "limitations": [
                  "Completion means every selected page has a coverage outcome; it does not mean every replay was successful or complete.",
                  "The selected sample is balanced. Successful/clean observations can be unbalanced and must be reported by label; failures are retained without replacement.",
                  "Sampling is deterministic and uniform by record within label, not uniform by site group. A small pilot is not an accuracy benchmark.",
                  "This is passive limited offline replay. CSP, missing external JS/CSS/images, blocked workers/frames, and blocked navigation change page behaviour.",
                  "Inline scripts execute, but no clicks or dummy credential interactions are performed. Zero observed activity is not evidence that a page is benign or inactive under other conditions.",
                  "Snapshots store at most 150 entry controls and 50 forms; trace events are capped. Quality flags expose truncation and execution gaps.",
                  "Only chosen bounded observation windows are examined. Later, interaction-triggered, frame, shadow-DOM or evasive behaviour may be missed.",
                  "Chromium sandboxing, request/CSP restrictions and per-page process budgets are requested. This remains a research prototype, not a hardened hostile-content analysis system.",
                  "The worker RAM limit uses summed process RSS, which can count shared pages more than once. Oversize HTML is skipped, not truncated and executed.",
                  "Raw HTML hashes are verified for pending selected records. Resume checks saved-result digests and source metadata; complete Parquet bytes are not re-hashed on resume.",
                  "Record IDs, labels, source shards, site groups, error/coverage flags and annotations are metadata, not classifier features. The existing model remains unchanged."]}
    save_json(args.output_dir/REPORT_NAME, report)
    print("Coverage by label:", json.dumps(coverage, indent=2), flush=True)
    print(f"Saved report: {(args.output_dir/REPORT_NAME).resolve()}", flush=True)
    print("Archived browser coverage audit complete. No model trained or accuracy measured.", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker_main()
    else:
        try:
            main()
        except KeyboardInterrupt:
            raise SystemExit("Interrupted. Rerun the same command to resume saved coverage outcomes.")
        except MemoryError as error:
            raise SystemExit(f"Archive decoding needs more available RAM: {error}. Close memory-heavy apps and rerun; checkpoints are preserved.")
        except Exception as error:
            raise SystemExit(f"Archived browser coverage audit stopped: {error}. Keep the checkpoint and paste this output. A report is complete only after the final completion message.")
