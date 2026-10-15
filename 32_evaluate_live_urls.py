"""Evaluate an independently labelled, balanced URL pilot with the frozen model."""
import argparse
import json
from pathlib import Path

from live_evaluation.engine import run_evaluation
from live_evaluation.protocol import LABELS, file_sha, load_url_list
from live_evaluation.runtime import RUNTIME_FILES, import_runtime, load_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls", type=Path, required=True, help="Filled CSV UTF-8 with url,label,label_source,label_checked_at.")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, help="Default: PROJECT/reports/live_evaluation_v1. Same folder resumes the same run.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--per-class", type=int, default=50, help="Required number of each class; default 50 (100 total).")
    parser.add_argument("--check-list", action="store_true", help="Check inputs offline; no model load, website requests or report writes.")
    args = parser.parse_args()
    output_dir = args.output_dir or args.project_dir / "reports/live_evaluation_v1"
    try:
        capture = import_runtime(args.project_dir)
        rows, raw = load_url_list(args.urls, capture.normalize_url, args.per_class)
        print(f"URL list validated: {args.per_class} benign + {args.per_class} phish; sources and dates supplied.")
        print("Label truth is supplied by you, not inferred from the model.")
        if args.check_list:
            print("Offline list check complete. No websites requested; no model predictions.")
            return 0
        detector = load_detector(args.project_dir, args.threads)
        print("Frozen model verified: 106 features; threshold 0.50; best iteration 1269. No retraining.")
        code_paths = [Path(__file__).resolve(), *sorted((Path(__file__).resolve().parent / "live_evaluation").glob("*.py"))]
        checksums = {str(path.relative_to(Path(__file__).resolve().parent)): file_sha(path) for path in code_paths}
        checksums.update(RUNTIME_FILES)
        def progress(done, total, csv_row, status):
            print(f"Completed {done}/{total}; CSV row {csv_row}: {status}", flush=True)
        report = run_evaluation(rows, raw, output_dir, detector, capture, checksums, progress)
        print("Live evaluation complete.")
        print("Scored class counts: " + json.dumps(report["scored_class_counts"]))
        metrics = report["classification_metrics_on_scored_pages"]
        for key in ("accuracy", "precision", "recall", "false_positive_rate", "f1"):
            value = metrics[key]
            print(f"  {key}: {value:.4%}" if value is not None else f"  {key}: unavailable (no denominator)")
        print(f"Scored coverage: {report['scored_coverage_all_selected_pages']:.1%} of selected URLs.")
        for warning in report["warnings"]:
            print("Note: " + warning)
        print(f"Saved report: {(output_dir / 'live_evaluation_report.json').resolve()}")
        print(f"Saved predictions: {(output_dir / 'live_evaluation_predictions.csv').resolve()}")
        return 0
    except KeyboardInterrupt:
        print("Evaluation interrupted. Keep the CSV and output folder; rerun the same command to resume.")
        print("A capture interrupted mid-request stays recorded as interrupted; completed captures are retained.")
        return 130
    except Exception as error:
        print(f"Live evaluation stopped: {type(error).__name__}: {error}")
        print("Fix the reported cause and rerun the same command. Keep the original model, runtime and .venv.")
        print("A report is complete only after 'Live evaluation complete'.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
