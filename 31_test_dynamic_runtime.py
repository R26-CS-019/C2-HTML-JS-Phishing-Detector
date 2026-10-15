"""Run trusted dynamic fixtures and compare source/rendered feature extraction."""
import argparse
from pathlib import Path

from detector_runtime.reports import save_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    try:
        from detector_runtime.dynamic import run_lab
        print("Controlled dynamic runtime: trusted fixtures, synthetic values, intercepted page requests.", flush=True)
        report = run_lab(args.project_dir, args.headed)
        path = save_report(args.project_dir, report, "dynamic_runtime_v1")
        print(f"Saved report: {path}", flush=True)
        if report["failed_cases"]:
            print("Controlled checks failed. See the report; no accuracy claim is valid.")
            return 1
        print(f"Dynamic runtime checks complete: {report['passed_cases']}/{report['controlled_cases']} passed. No phishing accuracy measured.")
        return 0
    except ImportError as e:
        print(f"Dynamic runtime stopped: {e}")
        print("Install the additions in requirements_runtime.txt, then run python -m playwright install chromium.")
        return 1
    except Exception as e:
        # Browser/OS policy errors do not become a false successful report.
        print(f"Dynamic runtime stopped: {e}")
        print("If Chromium is missing, run .\\.venv\\Scripts\\python.exe -m playwright install chromium.")
        print("A report is complete only after the final completion message.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
