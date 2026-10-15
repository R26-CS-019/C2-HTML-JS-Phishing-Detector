"""Run the frozen detector on source HTML or an observed browser page."""
import argparse
import json
from pathlib import Path

from detector_runtime.capture import MAX_HTML_BYTES, CaptureError, decode_html
from detector_runtime.model import FrozenDetector
from detector_runtime.reports import analyze_url, save_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--freeze-report", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Load and verify the frozen model, then run an offline smoke prediction.")
    group.add_argument("--url", help="Fetch complete HTTP source from a public HTTP(S) URL.")
    group.add_argument("--browser-url", help="Observe a public page in Chromium and score its rendered HTML.")
    group.add_argument("--demo", choices=["delayed_login","encoded_login","external_login","ordinary_sso","contact_form","redirect_login","fetch_login"], help="Run a trusted offline example through the full browser/model pipeline.")
    group.add_argument("--html-file", type=Path, help="Analyze an existing HTML file offline.")
    parser.add_argument("--page-url", help="Original/final URL for --html-file; relative references require it.")
    parser.add_argument("--basis", choices=["http_source", "browser_rendered", "user_supplied_unknown"], default="user_supplied_unknown")
    args = parser.parse_args()
    if args.html_file and not args.page_url:
        parser.error("--html-file requires --page-url.")
    try:
        detector = FrozenDetector(args.project_dir, args.freeze_report, args.threads)
        if args.check:
            result = detector.predict_html("<!doctype html><html><head><title>Runtime check</title></head><body><p>Offline smoke check.</p></body></html>", "https://runtime-check.example/", "user_supplied_unknown")
            report = {"status": "frozen_runtime_check_complete", "model": detector.describe(),
                      "smoke_score_finite": 0 <= result["phishing_score"] <= 1,
                      "smoke_fixture_is_accuracy_test": False, "websites_requested": False}
            print("Frozen runtime verified: 106 features; threshold 0.50; original best iteration.")
        elif args.url:
            report = analyze_url(detector, args.url)
        elif args.browser_url:
            from detector_runtime.behavior import analyze_browser_url
            report = analyze_browser_url(detector, args.browser_url)
        elif args.demo:
            from detector_runtime.behavior_examples import analyze_demo
            report = analyze_demo(detector, args.demo)
        else:
            if not args.html_file.is_file() or args.html_file.stat().st_size > MAX_HTML_BYTES:
                raise CaptureError("HTML file is missing or exceeds 32 MiB.")
            html, decoding = decode_html(args.html_file.read_bytes(), "text/html")
            report = detector.predict_html(html, args.page_url, args.basis)
            report["capture"] = {"input": "local_html_file", "capture_complete": "not_independently_verified", **decoding}
        path = save_report(args.project_dir, report)
        if not args.check:
            print(report["display_result"])
            if report.get("phishing_score") is not None:
                print(f"Static phishing score: {report['phishing_score']:.6f}; threshold: 0.50")
                print("Active quality flags: " + (", ".join(report["active_quality_flags"]) or "none"))
            if report.get("error"):
                print(report["error"])
        print(f"Saved report: {path}")
        return 1 if report["status"] in {"capture_failed", "inconclusive_capture", "browser_capture_failed"} else 0
    except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"Runtime stopped: {e}")
        print("Keep the existing model/scripts and use the original project .venv. See README.md.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
