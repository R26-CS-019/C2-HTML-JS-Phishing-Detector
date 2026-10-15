"""Audit the four saved script 16 validation predictions and threshold trade-offs.

Requirements: NumPy only, already installed in the project's virtual environment.
Save beside script 16 and run after its final completion message. Reads saved
prediction/feature CSVs and JSON reports; no fitting, model loading, raw Parquet,
web requests, JavaScript execution or official test evaluation is performed.

Thresholds are descriptive development diagnostics. None is installed into a
model. Exact tied scores move together under phishing_score >= threshold.
Main validation selects each illustrative threshold; sensitivity, quality and
site-group subsets reuse that threshold without optimizing their own settings.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VERSION = "1.0"
MODEL_PAIRS = (("lightgbm_context_all_features", "lightgbm_all_features"),
               ("lightgbm_context_without_size_features", "lightgbm_without_size_features"))
MODELS = ("lightgbm_all_features", "lightgbm_without_size_features",
          "lightgbm_context_all_features", "lightgbm_context_without_size_features")
METADATA = ("record_id", "label", "site_group", "source_file",
            "sampled_similarity_candidate", "in_validation_sensitivity_subset")
FPR_LIMITS = (0.01, 0.005, 0.001)
EXTRA_SIGNALS = ("password_input_count", "context_passwordlike_control_count")


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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
        "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / p if p else None,
        "false_positive_rate": fp / n if n else None, "false_negative_rate": fn / p if p else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def threshold_curve(y, scores):
    if not len(y) or not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("The main threshold curve needs both classes.")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_y = scores[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(y) - 1]
    cumulative_tp = np.cumsum(sorted_y, dtype=np.int64)
    tp = np.r_[0, cumulative_tp[ends]]
    fp = np.r_[0, ends + 1 - cumulative_tp[ends]]
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), sorted_scores[ends]]
    p, n = int(np.count_nonzero(y == 1)), int(np.count_nonzero(y == 0))
    fn, tn = p - tp, n - fp
    recall, fpr = tp / p, fp / n
    precision = np.divide(tp, tp + fp, out=np.zeros(len(tp), dtype=float), where=tp + fp > 0)
    return {"threshold": thresholds, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "average_precision": float(np.sum(np.diff(recall) * precision[1:])),
        "roc_auc": float(np.sum(np.diff(fpr) * (recall[1:] + recall[:-1]) / 2))}


def diagnostic_indices(curve):
    correct = curve["tp"] + curve["tn"]
    choices = np.flatnonzero(correct == np.max(correct))
    choices = choices[curve["fp"][choices] == np.min(curve["fp"][choices])]
    selected = [("maximum_main_validation_accuracy", int(choices[0]), None)]
    for limit in FPR_LIMITS:
        budget = math.floor(limit * int(curve["tn"][0]) + 1e-12)
        choices = np.flatnonzero(curve["fp"] <= budget)
        choices = choices[curve["tp"][choices] == np.max(curve["tp"][choices])]
        choices = choices[curve["fp"][choices] == np.min(curve["fp"][choices])]
        selected.append((f"maximum_recall_with_empirical_fpr_at_most_{limit:g}", int(choices[0]), limit))
    return selected


def read_reference(path, schema, feature_report):
    quality = schema["quality_columns_not_classifier_inputs"]
    columns = [*schema["metadata_columns"], *quality, *schema["feature_names"]]
    if schema["metadata_columns"] != list(METADATA) or not set(EXTRA_SIGNALS).issubset(schema["feature_names"]):
        raise ValueError("Unexpected script 15 metadata/feature schema.")
    rows, flags, signals = [], {k: [] for k in quality}, {k: [] for k in EXTRA_SIGNALS}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            raise ValueError("Validation feature CSV columns differ from the saved schema.")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed validation feature row.")
            rows.append({k: row[k] for k in METADATA})
            for k in quality:
                value = int(row[k])
                if value < 0:
                    raise ValueError("Negative quality flag/count.")
                flags[k].append(value)
            for k in EXTRA_SIGNALS:
                value = float(row[k])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid field-presence signal.")
                signals[k].append(value)
    y = np.asarray([int(r["label"] == "phish") for r in rows], dtype=np.int8)
    if {"benign": int(np.count_nonzero(y == 0)), "phish": int(np.count_nonzero(y == 1))} != feature_report["extracted_label_counts"]["validation"]:
        raise ValueError("Validation feature row counts differ from the extraction report.")
    for label, value in (("benign", 0), ("phish", 1)):
        summary = feature_report["quality_summary_by_split_and_label"]["validation"][label]
        for k, values in flags.items():
            if int(np.count_nonzero((np.asarray(values) > 0) & (y == value))) != summary[k + "_rows"]:
                raise ValueError(f"Quality counts differ from the feature report: {label}/{k}.")
    return rows, {k: np.asarray(v) for k, v in {**flags, **signals}.items()}


def read_predictions(path, reference, expected_counts):
    header = [*METADATA, "phishing_score", "prediction_at_0_50", "error_type"]
    y, scores, seen = [], [], set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != header:
            raise ValueError("Unexpected prediction CSV columns.")
        for row in reader:
            i = len(y)
            if i >= len(reference) or None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed/extra prediction row.")
            if any(row[k] != reference[i][k] for k in METADATA):
                raise ValueError("Prediction IDs, labels or subset metadata differ between model/feature files.")
            identity = row["record_id"]
            if len(identity) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in identity) or identity in seen:
                raise ValueError("Malformed or repeated record ID.")
            seen.add(identity)
            if row["label"] not in ("benign", "phish") or not row["site_group"] or not row["source_file"]:
                raise ValueError("Invalid label/group metadata.")
            if any(row[k] not in ("0", "1") for k in METADATA[-2:]) or row[METADATA[-2]] == row[METADATA[-1]] == "1":
                raise ValueError("Invalid sensitivity/similarity flags.")
            label, score = int(row["label"] == "phish"), float(row["phishing_score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Phishing scores must be finite and in [0,1].")
            predicted = score >= 0.5
            error = "false_positive" if not label and predicted else "false_negative" if label and not predicted else "correct"
            if row["prediction_at_0_50"] != ("phish" if predicted else "benign") or row["error_type"] != error:
                raise ValueError("Saved prediction/error flags disagree with threshold 0.50.")
            y.append(label); scores.append(score)
    y, scores = np.asarray(y, dtype=np.int8), np.asarray(scores, dtype=np.float64)
    counts = {"benign": int(np.count_nonzero(y == 0)), "phish": int(np.count_nonzero(y == 1))}
    if len(y) != len(reference) or counts != expected_counts or not len(y) or counts["benign"] != counts["phish"]:
        raise ValueError("Prediction rows must match the nonempty balanced validation split.")
    return y, scores


def verify_metrics(actual, saved):
    for k, v in actual.items():
        other = saved.get(k)
        if isinstance(v, float):
            if not isinstance(other, (int, float)) or not math.isclose(v, other, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Recomputed metrics differ from the training report: {k}.")
        elif v != other:
            raise ValueError(f"Recomputed metrics differ from the training report: {k}.")


def group_summary(rows, y, scores):
    groups = {}
    for row, label, score in zip(rows, y, scores):
        g = groups.setdefault(row["site_group"], {"site_group": row["site_group"], "benign_rows": 0,
            "phishing_rows": 0, "false_positives": 0, "false_negatives": 0})
        g["benign_rows" if label == 0 else "phishing_rows"] += 1
        g["false_positives"] += int(label == 0 and score >= 0.5)
        g["false_negatives"] += int(label == 1 and score < 0.5)
    for g in groups.values():
        g["false_positive_rate"] = g["false_positives"] / g["benign_rows"] if g["benign_rows"] else None
        g["false_negative_rate"] = g["false_negatives"] / g["phishing_rows"] if g["phishing_rows"] else None
    values = list(groups.values())
    return {"unique_site_groups": len(values),
        "top_groups_by_false_positive_count_at_0_50": sorted(values, key=lambda g: (-g["false_positives"], g["site_group"]))[:15],
        "top_groups_by_false_negative_count_at_0_50": sorted(values, key=lambda g: (-g["false_negatives"], g["site_group"]))[:15],
        "highest_fpr_groups_with_at_least_20_benign_pages": sorted([g for g in values if g["benign_rows"] >= 20], key=lambda g: (-g["false_positive_rate"], -g["benign_rows"], g["site_group"]))[:15]}


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=root / "reports/form_context_training_v2")
    parser.add_argument("--baseline-dir", type=Path, default=root / "reports/baseline_training_v1_1")
    parser.add_argument("--features-dir", type=Path, default=root / "reports/features_context_v2")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/context_threshold_analysis_v2")
    args = parser.parse_args()
    for directory in (args.training_dir, args.baseline_dir, args.features_dir):
        if args.output_dir.resolve().is_relative_to(directory.resolve()) or directory.resolve().is_relative_to(args.output_dir.resolve()):
            parser.error("Use a separate analysis output folder outside the training and feature folders.")
    paths = {"training_report": args.training_dir / "form_context_training_report.json",
        "training_config": args.training_dir / "run_config.json",
        "baseline_report": args.baseline_dir / "baseline_training_report.json",
        "feature_report": args.features_dir / "form_context_feature_report.json",
        "schema": args.features_dir / "static_feature_schema.json",
        "validation_features": args.features_dir / "validation_static_features.csv"}
    for name in MODELS:
        directory = args.training_dir if name.startswith("lightgbm_context_") else args.baseline_dir
        paths[name] = directory / name / "validation_predictions.csv"
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Missing input: {path}. Complete script 16 and keep its existing reports/predictions.")
    hashes = {k: digest_file(p) for k, p in paths.items()}
    report, config, baseline, feature_report, schema = (load_json(paths[k]) for k in
        ("training_report", "training_config", "baseline_report", "feature_report", "schema"))
    if report.get("status") != "form_context_model_training_complete" or feature_report.get("status") != "form_context_feature_extraction_complete":
        raise ValueError("Complete scripts 15 and 16 before analysis.")
    if report["run_config"] != config or hashes["training_config"] != report["run_config_sha256"]:
        raise ValueError("The saved training report/configuration differ.")
    for key, expected in (("labels_changed", False), ("balanced_splits_changed", False), ("official_test_evaluated", False),
                           ("operational_threshold_selected", False), ("original_features_unchanged", True)):
        if report.get(key) is not expected:
            raise ValueError(f"Unexpected training report policy: {key}.")
    if config["threshold"] != 0.5 or report["feature_count"] != 106 or report["context_feature_count"] != 20:
        raise ValueError("Unexpected script 16 threshold/feature contract.")
    for key, input_key in (("feature_report", "context_report"), ("schema", "context_schema"),
                           ("validation_features", "context_validation"), ("baseline_report", "baseline_report")):
        if hashes[key] != config["input_hashes"][input_key]:
            raise ValueError(f"An input changed after script 16: {paths[key]}.")
    for key in ("schema", "validation_features"):
        if hashes[key] != feature_report["outputs"][paths[key].name]["sha256"]:
            raise ValueError("Validation features/schema differ from script 15's report.")
    counts = report["main_validation_counts"]
    if counts != feature_report["extracted_label_counts"]["validation"] or report["validation_sensitivity_counts"] != feature_report["validation_sensitivity_counts"]:
        raise ValueError("The validation split counts changed.")
    if report["shared_train_validation_site_groups"] != 0 or report["shared_train_validation_record_ids"] != 0:
        raise ValueError("The training report records overlapping train/validation data.")
    saved_models = {r["model_name"]: r for r in report["model_results"]}
    if set(saved_models) != set(MODELS) or len(saved_models) != len(report["model_results"]):
        raise ValueError("Expected four unique LightGBM comparisons from script 16.")
    old_models = {r["model_name"]: r for r in baseline["model_results"]}
    for name in MODELS:
        saved = saved_models[name]
        expected_config = report["run_config_sha256"] if name.startswith("lightgbm_context_") else config["baseline_run_config_sha256"]
        if saved["run_config_sha256"] != expected_config or hashes[name] != saved["outputs"]["validation_predictions.csv"]["sha256"]:
            raise ValueError(f"Model configuration or predictions changed: {name}.")
        if name in old_models and hashes[name] != old_models[name]["outputs"]["validation_predictions.csv"]["sha256"]:
            raise ValueError("Original baseline prediction provenance differs.")
    print("Checking saved prediction files, balanced IDs and quality metadata...", flush=True)
    rows, quality = read_reference(paths["validation_features"], schema, feature_report)
    sensitivity = np.asarray([r["in_validation_sensitivity_subset"] == "1" for r in rows])
    native = quality["password_input_count"] > 0
    context = quality["context_passwordlike_control_count"] > 0
    capped = (quality["html_analysis_truncated"] > 0) | (quality["js_analysis_truncated"] > 0)
    masks = {"validation_sensitivity": sensitivity, "html_or_js_analysis_cap_reached": capped,
        "html_and_js_analysis_caps_not_reached": ~capped,
        "password_cue_present": context, "password_cue_without_native_password_input": context & ~native,
        "no_password_cue_present": ~context,
        "human_verification_hint_present": quality["context_human_verification_hint_present"] > 0,
        "unavailable_page_hint_present": quality["context_unavailable_page_hint_present"] > 0,
        "context_association_limited": (quality["context_parse_error"] > 0) | (quality["context_association_storage_cut"] > 0)}
    site_groups = config["diagnostic_groups_not_classifier_inputs"]
    site_masks = {k: np.asarray([r["site_group"] == group for r in rows]) for k, group in site_groups.items()}
    results = []
    for name in MODELS:
        y, scores = read_predictions(paths[name], rows, counts)
        sensitivity_counts = {"benign": int(np.count_nonzero((y == 0) & sensitivity)), "phish": int(np.count_nonzero((y == 1) & sensitivity))}
        if sensitivity_counts != report["validation_sensitivity_counts"]:
            raise ValueError("Sensitivity membership differs from the training report.")
        curve = threshold_curve(y, scores)
        initial = metrics_at(y, scores, 0.5)
        verify_metrics(initial, saved_models[name]["metrics"]["main_validation"])
        verify_metrics(metrics_at(y[sensitivity], scores[sensitivity], 0.5), saved_models[name]["metrics"]["validation_sensitivity"])
        for key in ("average_precision", "roc_auc"):
            if not math.isclose(curve[key], saved_models[name]["metrics"]["main_validation"][key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Recomputed {key} differs from saved scores: {name}.")
        def point(label, threshold, limit=None):
            return {"label": label, "illustrative_empirical_fpr_limit": limit,
                "integer_main_validation_false_positive_budget": math.floor(limit * counts["benign"] + 1e-12) if limit is not None else None,
                "main_validation": metrics_at(y, scores, threshold),
                "subsets_same_main_selected_threshold": {k: metrics_at(y[mask], scores[mask], threshold) for k, mask in masks.items()},
                "previously_inspected_groups_same_main_selected_threshold": {k: {"site_group": site_groups[k], **metrics_at(y[mask], scores[mask], threshold)} for k, mask in site_masks.items()}}
        points = [point("initial_0_50", 0.5)]
        points.extend(point(label, float(curve["threshold"][index]), limit) for label, index, limit in diagnostic_indices(curve))
        results.append({"model_name": name, "feature_count": saved_models[name]["feature_count"],
            "average_precision": curve["average_precision"], "roc_auc": curve["roc_auc"],
            "distinct_scores": len(curve["threshold"]) - 1, "operating_points": points,
            "remaining_error_groups_at_0_50": group_summary(rows, y, scores)})
        for p in points[:3]:
            m = p["main_validation"]; cm = m["confusion_matrix"]
            print(f"  {name} | {p['label']} | threshold={m['threshold']:.6f} | accuracy={100*m['accuracy']:.2f}% | recall={100*m['recall']:.2f}% | FPR={100*m['false_positive_rate']:.2f}% | FP={cm['fp']:,} | FN={cm['fn']:,}", flush=True)
    comparisons = []
    by_name = {r["model_name"]: r for r in results}
    for new, old in MODEL_PAIRS:
        for limit in FPR_LIMITS:
            def find(name):
                return next(p["main_validation"] for p in by_name[name]["operating_points"] if p["illustrative_empirical_fpr_limit"] == limit)
            a, b = find(old), find(new)
            comparisons.append({"new_model": new, "compared_with": old, "illustrative_empirical_fpr_limit": limit,
                "integer_false_positive_budget": math.floor(limit * counts["benign"] + 1e-12),
                "old_main_selected_threshold": a["threshold"], "new_main_selected_threshold": b["threshold"],
                "recall_change_percentage_points": 100 * (b["recall"] - a["recall"]),
                "accuracy_change_percentage_points": 100 * (b["accuracy"] - a["accuracy"]),
                "old_fp": a["confusion_matrix"]["fp"], "new_fp": b["confusion_matrix"]["fp"],
                "old_fn": a["confusion_matrix"]["fn"], "new_fn": b["confusion_matrix"]["fn"]})
    for key, path in paths.items():
        if digest_file(path) != hashes[key]:
            raise ValueError("An input changed during analysis. No completed report was written.")
    analysis_config = {"script_version": VERSION, "analysis_code_sha256": digest_file(Path(__file__)),
        "input_hashes": hashes, "models": list(MODELS), "illustrative_empirical_fpr_limits": list(FPR_LIMITS),
        "python_version": platform.python_version(), "numpy_version": np.__version__}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "analysis_config.json"
    if config_path.exists() and load_json(config_path) != analysis_config:
        raise ValueError("This output folder contains another analysis configuration. Use a new --output-dir.")
    if not config_path.exists():
        if any(args.output_dir.iterdir()):
            raise ValueError("Use an empty output folder or its matching analysis_config.json.")
        save_json(config_path, analysis_config)
    result = {"status": "context_threshold_analysis_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": VERSION, "analysis_config": analysis_config, "main_validation_counts": counts,
        "validation_sensitivity_counts": report["validation_sensitivity_counts"], "model_results": results,
        "comparisons_at_same_empirical_fpr_budget": comparisons, "operational_threshold_selected": False,
        "official_test_evaluated": False, "model_retrained": False, "model_thresholds_changed": False,
        "labels_changed": False, "balanced_splits_changed": False, "websites_requested": False, "javascript_executed": False,
        "decision_rule": "phishing_score >= threshold; every equal score moves together",
        "maximum_accuracy_tie_break": "fewest false positives, then highest threshold",
        "fpr_budget_tie_break": "most true positives within the integer FP budget, then fewest FP, then highest threshold",
        "limitations": [
            "Main validation was previously used for feature design, early stopping and model selection. Threshold diagnostics are optimistic development estimates, not final independent performance.",
            "The 1%, 0.5% and 0.1% FPR limits are illustrative empirical budgets, not agreed deployment targets or guarantees on future sites.",
            "Sensitivity, quality and site-group subsets reuse the main-selected threshold. They never choose thresholds independently and do not alter model scores.",
            "Raising a threshold reduces or preserves false positives but also reduces or preserves phishing recall. Scores are not proven calibrated probabilities.",
            "Balanced validation does not establish operational precision at a low phishing prevalence. Single-class or small site groups cannot establish broad reliability.",
            "Quality hints can occur in benign and phishing-labelled captures. No labels are changed and no CAPTCHA, brand or domain is automatically exempted.",
            "These remain static HTML/JS features. Runtime behavior, external script contents and final redirect URLs are not added by this analysis."]}
    destination = args.output_dir / "context_threshold_report.json"
    save_json(destination, result)
    print(f"Saved report: {destination.resolve()}", flush=True)
    print("Context threshold analysis complete. No operational threshold selected; official test remains unevaluated.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"Context threshold analysis stopped: {error}. A completed report is valid after the final completion message.")
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Rerun the same command; original predictions/features are unchanged.")
