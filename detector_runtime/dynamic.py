"""Controlled dynamic observation and rendered/static feature diagnostics.

Only the trusted built-in fixtures are accepted. No arbitrary URL/HTML input.
"""
import base64
import json
import time

from . import controlled_fixtures as pilot
from .model import FeatureAdapter, digest


def render_comparisons(browser, adapter, error_class):
    form = pilot.login_form("p a s s w o r d", "text")
    delayed = "setTimeout(()=>{document.querySelector('#mount').innerHTML=" + json.dumps(form) + ";},400);"
    rows = [
        {"id": "delayed_form", "html": pilot.document("<main id='mount'>Loading</main>", delayed)},
        {"id": "encoded_delayed_form", "html": pilot.document("<main id='mount'>Loading</main>",
            "new Function(atob(" + json.dumps(base64.b64encode(delayed.encode()).decode()) + "))();")},
        {"id": "mock_external_form", "html": pilot.document("<main id='mount'>Loading</main><script src='/fixture.js'></script>"), "script": delayed},
    ]
    results = []
    for row in rows:
        url = pilot.PAGE_ORIGIN + "/runtime_diagnostics/" + row["id"]
        context = browser.new_context(service_workers="block", accept_downloads=False, permissions=[])
        routes = []

        def handle(route):
            r = route.request
            if r.url == url and r.method == "GET":
                action = "fulfilled_fixture_html"
                route.fulfill(status=200, content_type="text/html; charset=utf-8", body=row["html"])
            elif r.url == pilot.PAGE_ORIGIN + "/fixture.js" and r.method == "GET" and "script" in row:
                action = "fulfilled_mock_script"
                route.fulfill(status=200, content_type="text/javascript", body=row["script"])
            else:
                action = "aborted"
                route.abort()
            routes.append({"url": r.url, "method": r.method, "action": action})

        try:
            context.route("**/*", handle)
            context.route_web_socket("**/*", lambda ws: ws.close())
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=6000)
            page.wait_for_function("document.querySelector('#secret') !== null", timeout=6000)
            rendered = page.content()
            source_features, source_quality = adapter.extract(row["html"], url)
            rendered_features, rendered_quality = adapter.extract(rendered, page.url)
            tracked = ["form_count", "input_count", "password_input_count",
                       "context_passwordlike_control_count", "context_plaintext_passwordlike_control_count",
                       "context_identity_password_pair_form_count"]
            checks = {
                "source_has_no_form": source_features["form_count"] == 0,
                "rendered_has_one_form": rendered_features["form_count"] == 1,
                "rendered_has_two_inputs": rendered_features["input_count"] == 2,
                "rendered_plaintext_password_cue_found": rendered_features["context_plaintext_passwordlike_control_count"] == 1,
                "rendered_identity_password_pair_found": rendered_features["context_identity_password_pair_form_count"] == 1,
                "classifier_feature_count_still_106": len(source_features) == len(rendered_features) == 106,
            }
            results.append({"case_id": row["id"], "status": "passed" if all(checks.values()) else "failed",
                "checks": checks, "source_features": {n: source_features[n] for n in tracked},
                "rendered_features": {n: rendered_features[n] for n in tracked},
                "source_quality": source_quality, "rendered_quality": rendered_quality,
                "routes": routes, "classifier_used": False})
        except (error_class, ValueError) as e:
            results.append({"case_id": row["id"], "status": "failed", "error": str(e)})
        finally:
            context.close()
    return results


def run_lab(project_dir, headed=False):
    from playwright.sync_api import sync_playwright, Error as PlaywrightError
    from importlib.metadata import version
    from pathlib import Path
    adapter = FeatureAdapter(project_dir)
    cue_checks = pilot.cue_checks()
    if not all(c["passed"] for c in cue_checks):
        raise ValueError("A controlled credential-cue check failed.")
    started = time.monotonic()
    with sync_playwright() as p:
        # These are known local fixtures, not hostile webpages. The Linux
        # container QA environment runs as root; on Windows request sandboxing.
        import sys
        browser = p.chromium.launch(headless=not headed, chromium_sandbox=sys.platform == "win32",
            args=["--disable-background-networking", "--disable-component-update", "--no-first-run",
                  "--host-resolver-rules=MAP * ~NOTFOUND"])
        try:
            results = []
            rows = pilot.cases()
            for n, row in enumerate(rows, 1):
                result = pilot.run_case(browser, row, 900, 650, PlaywrightError)
                results.append(result)
                print(f"  Behavior {n}/{len(rows)} {row['id']}: {result['status'].upper()}", flush=True)
            comparisons = render_comparisons(browser, adapter, PlaywrightError)
            for row in comparisons:
                print(f"  Rendered-feature comparison {row['case_id']}: {row['status'].upper()}", flush=True)
            passed = sum(r["status"] == "passed" for r in results + comparisons)
            count = len(results) + len(comparisons)
            return {"status": "controlled_dynamic_runtime_complete" if passed == count else "controlled_dynamic_runtime_checks_failed",
                "elapsed_seconds": time.monotonic() - started,
                "playwright_version": version("playwright"), "chromium_version": browser.version,
                "fixture_implementation_sha256": digest(Path(pilot.__file__)),
                "feature_extractor_sha256": {"base": digest(Path(project_dir) / "09_extract_static_features.py"),
                    "context": digest(Path(project_dir) / "15_extract_form_context.py")},
                "observation_windows_ms": {"initial": 900, "after_synthetic_interaction": 650},
                "controlled_cases": count, "passed_cases": passed, "failed_cases": count - passed,
                "credential_cue_checks": cue_checks, "behavior_cases": results,
                "rendered_feature_comparisons": comparisons,
                "model_trained": False, "classifier_used": False, "phishing_accuracy_measured": False,
                "live_page_requests_continued": False, "synthetic_values_only": True,
                "limitations": [
                    "These are instrumentation tests, not a balanced phishing training or evaluation dataset.",
                    "Ordinary cross-origin login is intentionally a control: credential transfer does not prove malicious intent.",
                    "Observed dynamic features are separate from the unchanged 106-feature classifier.",
                    "Requests from the known fixture pages are fulfilled from memory or aborted; this is not a hardened hostile-page sandbox.",
                    "Finite waits and chosen interactions can miss later/untriggered behavior. One encoded fixture does not cover all obfuscation.",
                    "The fixture credential cues are broader than the frozen static cues; rendering does not erase all cue-coverage limitations.",
                    "Real dynamic phishing accuracy needs independently labelled captures and a new holdout."]}
        finally:
            browser.close()
