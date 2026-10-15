"""Audit saved validation predictions and explore decision thresholds.

Requirements: NumPy. No models are fitted or changed, and no official test
data or websites are opened. Thresholds are development diagnostics, not an
operational policy. Sensitivity subsets use thresholds chosen on main
validation; they never choose a threshold independently.

By default, reads the two LightGBM prediction files from script 10 v1.1.
Use --prediction-files for uploaded/renamed copies; SHA-256 identifies each
model against the saved baseline report rather than guessing from filenames.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VERSION = "1.0"
MODELS = ("lightgbm_all_features", "lightgbm_without_size_features")
METADATA = ("record_id", "label", "site_group", "source_file",
            "sampled_similarity_candidate", "in_validation_sensitivity_subset")
HEADER = (*METADATA, "phishing_score", "prediction_at_0_50", "error_type")
FPR_LIMITS = (0.01, 0.005, 0.001)


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_predictions(path, expected_counts):
    rows, y, scores, seen = [], [], [], set()
    pattern = re.compile(r"[0-9a-fA-F]{64}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(HEADER):
            raise ValueError(f"Unexpected prediction columns in {path.name}.")
        for row in reader:
            if None in row or any(row.get(name) is None for name in HEADER):
                raise ValueError(f"Malformed row at line {reader.line_num}.")
            identity = row["record_id"]
            if not pattern.fullmatch(identity) or identity in seen:
                raise ValueError("Missing, malformed or repeated record ID.")
            seen.add(identity)
            if row["label"] not in ("benign", "phish"):
                raise ValueError("Unexpected label.")
            if not row["site_group"] or not row["source_file"]:
                raise ValueError("Missing grouping metadata.")
            for name in METADATA[-2:]:
                if row[name] not in ("0", "1"):
                    raise ValueError("Invalid validation subset flag.")
            if row[METADATA[-2]] == row[METADATA[-1]] == "1":
                raise ValueError("Similarity candidate present in the sensitivity subset.")
            label, score = int(row["label"] == "phish"), float(row["phishing_score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Prediction scores must be finite and between 0 and 1.")
            prediction = int(score >= 0.5)
            error = "false_positive" if not label and prediction else "false_negative" if label and not prediction else "correct"
            if row["prediction_at_0_50"] != ("phish" if prediction else "benign") or row["error_type"] != error:
                raise ValueError("Saved prediction/error columns disagree with score >= 0.50.")
            rows.append({name: row[name] for name in METADATA})
            y.append(label)
            scores.append(score)
    y, scores = np.asarray(y, dtype=np.int8), np.asarray(scores, dtype=np.float64)
    counts = {"benign": int(np.count_nonzero(y == 0)), "phish": int(np.count_nonzero(y == 1))}
    if counts != expected_counts or not len(y) or min(counts.values()) == 0:
        raise ValueError("Prediction label counts differ from the saved report.")
    return rows, y, scores


def threshold_curve(y, scores):
    """Exact decision states: equal scores move together with score >= t."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_y = scores[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(y) - 1]
    cumulative_tp = np.cumsum(sorted_y, dtype=np.int64)
    tp = np.r_[0, cumulative_tp[ends]]
    fp = np.r_[0, ends + 1 - cumulative_tp[ends]]
    # The first state rejects every page, including a score of exactly 1.
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), sorted_scores[ends]]
    positives, negatives = int(np.count_nonzero(y)), int(np.count_nonzero(y == 0))
    fn, tn = positives - tp, negatives - fp
    accuracy = (tp + tn) / len(y)
    recall, fpr = tp / positives, fp / negatives
    precision = np.divide(tp, tp + fp, out=np.zeros(len(tp), dtype=float), where=tp + fp > 0)
    average_precision = float(np.sum(np.diff(recall) * precision[1:]))
    roc_auc = float(np.sum(np.diff(fpr) * (recall[1:] + recall[:-1]) / 2))
    return {"threshold": thresholds, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": accuracy, "recall": recall, "fpr": fpr,
            "average_precision": average_precision, "roc_auc": roc_auc}


def metrics_at(y, scores, threshold):
    predicted = scores >= threshold
    tp = int(np.count_nonzero((y == 1) & predicted))
    fp = int(np.count_nonzero((y == 0) & predicted))
    tn = int(np.count_nonzero((y == 0) & ~predicted))
    fn = int(np.count_nonzero((y == 1) & ~predicted))
    n, p = tn + fp, tp + fn
    return {"threshold": float(threshold), "rows": len(y), "benign_rows": n, "phishing_rows": p,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "accuracy": (tn + tp) / len(y) if len(y) else None,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / p if p else None,
            "false_positive_rate": fp / n if n else None,
            "false_negative_rate": fn / p if p else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def select_diagnostic_indices(curve):
    # Arrays start at the highest threshold. Last tie-break keeps that state.
    correct = curve["tp"] + curve["tn"]
    choices = np.flatnonzero(correct == np.max(correct))
    choices = choices[curve["fp"][choices] == np.min(curve["fp"][choices])]
    selected = [("maximum_main_validation_accuracy", int(choices[0]), None)]
    negatives = int(curve["tn"][0])
    for limit in FPR_LIMITS:
        # Use integer error budgets to avoid a floating-point boundary issue.
        budget = math.floor(limit * negatives + 1e-12)
        choices = np.flatnonzero(curve["fp"] <= budget)
        choices = choices[curve["tp"][choices] == np.max(curve["tp"][choices])]
        choices = choices[curve["fp"][choices] == np.min(curve["fp"][choices])]
        selected.append((f"maximum_recall_with_empirical_fpr_at_most_{limit:g}", int(choices[0]), limit))
    return selected


def error_examples(rows, y, scores, count=10):
    result = {}
    for name, mask, descending in (("highest_score_false_positives", (y == 0) & (scores >= 0.5), True),
                                   ("lowest_score_false_negatives", (y == 1) & (scores < 0.5), False)):
        indices = np.flatnonzero(mask)
        indices = sorted(indices, key=lambda i: (-scores[i] if descending else scores[i], rows[i]["record_id"]))[:count]
        result[name] = [{**rows[i], "phishing_score": float(scores[i])} for i in indices]
    return result


def group_error_summary(rows, y, scores):
    groups = {}
    for row, label, score in zip(rows, y, scores):
        g = groups.setdefault(row["site_group"], {"site_group": row["site_group"], "benign_rows": 0, "phishing_rows": 0,
                                                   "false_positives": 0, "false_negatives": 0})
        g["benign_rows" if label == 0 else "phishing_rows"] += 1
        if label == 0 and score >= 0.5:
            g["false_positives"] += 1
        if label == 1 and score < 0.5:
            g["false_negatives"] += 1
    values = list(groups.values())
    for g in values:
        g["false_positive_rate"] = g["false_positives"] / g["benign_rows"] if g["benign_rows"] else None
        g["false_negative_rate"] = g["false_negatives"] / g["phishing_rows"] if g["phishing_rows"] else None
    return {"unique_site_groups": len(values),
            "top_groups_by_false_positive_count_at_0_50": sorted(values, key=lambda g: (-g["false_positives"], g["site_group"]))[:10],
            "highest_fpr_groups_with_at_least_20_benign_pages": sorted(
                [g for g in values if g["benign_rows"] >= 20],
                key=lambda g: (-g["false_positive_rate"], -g["benign_rows"], g["site_group"]))[:10]}


def summary_markdown(report):
    count = sum(report["main_validation_counts"].values())
    lines = ["# LightGBM validation threshold analysis", "",
             "Both uploaded prediction files match their saved model-report SHA-256 hashes. "
             f"They cover the same {count:,} pages; no model or dataset was changed.", "",
             "## Main validation", "",
             "| Model | Diagnostic threshold | Accuracy | Phishing recall | False-positive rate | FP | FN |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for result in report["model_results"]:
        points = [("initial 0.50", result["initial_0_50"]["main_validation"]),
                  ("max accuracy", result["diagnostic_operating_points"][0]["main_validation"]),
                  ("FPR ≤ 1%", result["diagnostic_operating_points"][1]["main_validation"])]
        name = result["model_name"].replace("lightgbm_", "").replace("_", " ")
        for label, metrics in points:
            cm = metrics["confusion_matrix"]
            lines.append(f"| {name} ({label}) | {metrics['threshold']:.6f} | {metrics['accuracy']:.2%} | {metrics['recall']:.2%} | {metrics['false_positive_rate']:.2%} | {cm['fp']:,} | {cm['fn']:,} |")
    lines += ["", "Thresholds in this table are rounded for reading. The JSON stores full precision; "
              "the decision rule is `phishing_score >= threshold`.", "",
              "The maximum-accuracy thresholds increase false positives. At a 1% empirical false-positive limit, "
              "many more phishing pages are missed. These are validation trade-offs, not guaranteed deployment rates. "
              "No operational threshold was selected.", "", "## Error overlap at 0.50", ""]
    overlap = report["error_overlap_at_0_50"]
    for error in ("false_positives", "false_negatives"):
        item = overlap[error]
        lines.append(f"- {error.replace('_', ' ').capitalize()}: {item['both_models']:,} shared; "
                     f"{item['only_first_model']:,} only in {report['model_results'][0]['model_name']}; "
                     f"{item['only_second_model']:,} only in {report['model_results'][1]['model_name']}.")
    lines += ["", "## Groups contributing the most false positives", "",
              "These are saved site-group values, not model input features or claims about a live website.", "",
              "| Model | Largest FP contributor at 0.50 | Benign pages | False positives | Group FPR | Share of all FP |",
              "|---|---|---:|---:|---:|---:|"]
    for result in report["model_results"]:
        group = result["site_group_error_summary_at_0_50"]["top_groups_by_false_positive_count_at_0_50"][0]
        total_fp = result["initial_0_50"]["main_validation"]["confusion_matrix"]["fp"]
        if total_fp:
            lines.append(f"| {result['model_name']} | `{group['site_group']}` | {group['benign_rows']:,} | "
                         f"{group['false_positives']:,} | {group['false_positive_rate']:.2%} | {group['false_positives'] / total_fp:.2%} |")
    lines += ["", "Review saved HTML and extracted features for these records before changing the feature pipeline. "
              "The grouping concentration suggests a useful review target; it does not establish the cause of the errors."]
    lines += ["", "## Analysis limits", "",
              "Main validation was already used for early stopping and model comparison. Threshold optimization "
              "on it is further development work; it is not an independent final evaluation. Sensitivity results "
              "in the JSON apply the same main-selected thresholds without separate optimization.", "",
              "The prediction files contain IDs, grouping metadata and scores. They do not contain HTML, "
              "feature values or extraction-cap flags, so they cannot establish why a page was misclassified "
              "or whether its label is wrong. Review the highest-score false positives by ID using saved data.", "",
              "The original baseline report does show worse aggregate performance among pages reaching an "
              "HTML or JavaScript analysis cap. That association deserves investigation, but does not prove "
              "the cap caused the errors. The JSON copies the cap-subset metrics from that report.", "",
              "The official test remains unevaluated. Balanced-validation precision does not directly represent "
              "precision at a lower real-world phishing prevalence.", "",
              "Next: compare CatBoost and XGBoost on the identical feature files and splits. Keep the ranking "
              "criterion explicit, then agree a false-positive target before choosing the deployment threshold.", ""]
    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, default=root / "reports/baseline_training_v1_1/baseline_training_report.json")
    parser.add_argument("--prediction-files", type=Path, nargs=2,
                        help="Two files in any order; verified hashes determine the LightGBM model names.")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/lightgbm_threshold_analysis")
    args = parser.parse_args()
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8-sig"))
    if baseline.get("status") != "static_baseline_training_complete":
        raise ValueError("Use a completed script 10 baseline report.")
    expected = {r["model_name"]: r for r in baseline["model_results"] if r["model_name"] in MODELS}
    if set(expected) != set(MODELS):
        raise ValueError("Baseline report must include both LightGBM models.")
    paths = args.prediction_files or [args.baseline_report.parent / name / "validation_predictions.csv" for name in MODELS]
    path_by_model = {}
    input_hashes = {str(args.baseline_report.resolve()): digest_file(args.baseline_report)}
    for path in paths:
        sha = digest_file(path)
        matches = [name for name, result in expected.items() if result["outputs"]["validation_predictions.csv"]["sha256"] == sha]
        if len(matches) != 1 or matches[0] in path_by_model:
            raise ValueError(f"Missing, repeated or changed model predictions: {path.name}.")
        path_by_model[matches[0]] = path
        input_hashes[str(path.resolve())] = sha
    all_data, results = [], []
    for name in MODELS:
        rows, y, scores = read_predictions(path_by_model[name], baseline["main_validation_counts"])
        if all_data and rows != all_data[0][0]:
            raise ValueError("Models do not share identical validation metadata and row order.")
        sensitivity = np.asarray([row["in_validation_sensitivity_subset"] == "1" for row in rows])
        counts = {"benign": int(np.count_nonzero((y == 0) & sensitivity)), "phish": int(np.count_nonzero((y == 1) & sensitivity))}
        if counts != baseline["validation_sensitivity_counts"] or min(counts.values()) == 0:
            raise ValueError("Sensitivity counts disagree with baseline report.")
        curve = threshold_curve(y, scores)
        original = metrics_at(y, scores, 0.5)
        original_sensitivity = metrics_at(y[sensitivity], scores[sensitivity], 0.5)
        for key, observed in (("main_validation", original), ("validation_sensitivity", original_sensitivity)):
            saved = expected[name]["metrics"][key]
            if observed["confusion_matrix"] != saved["confusion_matrix"] or saved["threshold"] != 0.5:
                raise ValueError("Prediction confusion counts disagree with saved metrics.")
        for key in ("average_precision", "roc_auc"):
            if not math.isclose(curve[key], expected[name]["metrics"]["main_validation"][key], abs_tol=1e-12):
                raise ValueError(f"Recomputed {key} differs from the saved report.")
        points = []
        for purpose, index, target in select_diagnostic_indices(curve):
            threshold = float(curve["threshold"][index])
            points.append({"purpose": purpose, "empirical_fpr_limit": target,
                           "main_validation": metrics_at(y, scores, threshold),
                           "validation_sensitivity_same_threshold": metrics_at(y[sensitivity], scores[sensitivity], threshold)})
        results.append({"model_name": name, "prediction_sha256": digest_file(path_by_model[name]),
                        "score_min": float(scores.min()), "score_max": float(scores.max()),
                        "distinct_scores": len(curve["threshold"]) - 1,
                        "average_precision_recomputed": curve["average_precision"], "roc_auc_recomputed": curve["roc_auc"],
                        "initial_0_50": {"main_validation": original, "validation_sensitivity": original_sensitivity},
                        "diagnostic_operating_points": points,
                        "analysis_cap_metrics_from_baseline_report": {k: expected[name]["metrics"][k] for k in
                            ("validation_reached_an_analysis_cap", "validation_did_not_reach_analysis_caps")},
                        "error_examples_at_0_50": error_examples(rows, y, scores),
                        "site_group_error_summary_at_0_50": group_error_summary(rows, y, scores)})
        all_data.append((rows, y, scores))
        print(f"Verified {name}: {len(y):,} balanced validation rows; {len(curve['threshold']) - 1:,} distinct scores.", flush=True)
    y, first, second = all_data[0][1], all_data[0][2], all_data[1][2]
    overlap = {}
    for error, a, b in (("false_positives", (y == 0) & (first >= 0.5), (y == 0) & (second >= 0.5)),
                         ("false_negatives", (y == 1) & (first < 0.5), (y == 1) & (second < 0.5))):
        overlap[error] = {"both_models": int(np.count_nonzero(a & b)), "only_first_model": int(np.count_nonzero(a & ~b)),
                          "only_second_model": int(np.count_nonzero(~a & b))}
    report = {"status": "validation_threshold_analysis_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "script_version": VERSION, "analysis_code_sha256": digest_file(Path(__file__)),
              "input_sha256": input_hashes, "python_version": platform.python_version(), "numpy_version": np.__version__,
              "main_validation_counts": baseline["main_validation_counts"],
              "validation_sensitivity_counts": baseline["validation_sensitivity_counts"],
              "decision_rule": "phishing_score >= threshold", "equal_scores_policy": "move every tied score together",
              "accuracy_tie_break": "fewest false positives, then highest threshold",
              "fpr_target_tie_break": "most true positives within the integer FP budget, then fewest FP, then highest threshold",
              "illustrative_empirical_fpr_limits": list(FPR_LIMITS),
              "model_results": results, "error_overlap_at_0_50": overlap,
              "operational_threshold_selected": False, "official_test_evaluated": False,
              "limitations": ["Main validation is development data already used for early stopping; selected threshold performance is optimistic for deployment.",
                              "Sensitivity validation is nested and only reports thresholds selected on main validation.",
                              "FPR limits are illustrative empirical constraints, not agreed deployment targets or statistical guarantees.",
                              "Per-row cap flags, feature values and HTML are not included in these prediction files; causes and label validity cannot be established here.",
                              "Site groups contain repeated/correlated pages; no independent-page confidence interval is claimed."]}
    for path, sha in input_hashes.items():
        if digest_file(Path(path)) != sha:
            raise ValueError("An input changed during analysis.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "threshold_analysis_report.json"
    save_json(report_path, report)
    summary = args.output_dir / "threshold_analysis_summary.md"
    summary.write_text(summary_markdown(report), encoding="utf-8")
    print(f"Saved report: {report_path.resolve()}", flush=True)
    print(f"Saved summary: {summary.resolve()}", flush=True)
    print("Threshold analysis complete. Operational threshold selection and final test evaluation remain pending.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"Threshold analysis stopped: {error}")
