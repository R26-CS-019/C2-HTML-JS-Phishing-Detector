"""Explain saved LightGBM validation errors without retraining or crawling.

Uses the completed 86-feature script 10 v1.1 model, its original validation
features/predictions and its gain importance file. Reuses script 10's checked
NumPy feature reader. No new packages beyond the working baseline environment
are needed; LightGBM supplies feature contributions without the shap package.

Outputs: a JSON report with contributions plus a CSV of selected feature rows.
Only trusted model bundles from your own baseline run should be loaded.
Diagnostics do not prove label correctness, causation or runtime behavior.
No official test data, raw HTML, external JavaScript or websites are opened.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

VERSION = "1.0"
MODEL = "lightgbm_all_features"
SEED = 42
THRESHOLD = 0.50
SIZE_FEATURES = {"html_character_count", "tag_count", "text_character_count",
                 "title_character_count", "js_analysed_character_count"}


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def checked_reader(path, sha):
    if digest_file(path) != sha:
        raise ValueError("Keep the script 10 version recorded in your completed baseline report.")
    spec = importlib.util.spec_from_file_location("baseline_error_review_reader", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.VERSION != "1.1" or module.THRESHOLD != THRESHOLD:
        raise ValueError("This review requires script 10 v1.1 and its initial threshold 0.50.")
    return module


def read_importance(path, names):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["feature", "split_gain", "split_count"]:
            raise ValueError("Unexpected LightGBM importance columns.")
        rows = [{"feature": r["feature"], "split_gain": float(r["split_gain"]), "split_count": int(r["split_count"])} for r in reader]
    if len(rows) != len(names) or {r["feature"] for r in rows} != set(names):
        raise ValueError("Importance features differ from the saved schema.")
    if any(not math.isfinite(r["split_gain"]) or r["split_gain"] < 0 or r["split_count"] < 0 for r in rows):
        raise ValueError("Invalid feature importance value.")
    total = sum(r["split_gain"] for r in rows)
    if total <= 0:
        raise ValueError("The model has no positive split gain to review.")
    for row in rows:
        row["share_of_training_split_gain"] = row["split_gain"] / total
    return sorted(rows, key=lambda r: (-r["split_gain"], r["feature"]))


def read_scores(path, table, schema, np):
    metadata = schema["metadata_columns"]
    header = [*metadata, "phishing_score", "prediction_at_0_50", "error_type"]
    result = np.empty(len(table), dtype=np.float64)
    loaded = 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != header:
            raise ValueError("Unexpected prediction CSV columns.")
        for i, row in enumerate(reader):
            if i >= len(table) or None in row or any(row.get(name) is None for name in header):
                raise ValueError("Unexpected prediction row count or width.")
            if any(row[name] != str(table.metadata[name][i]) for name in metadata):
                raise ValueError("Prediction rows differ from validation record IDs, labels or metadata.")
            score = float(row["phishing_score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid saved phishing score.")
            pred = int(score >= THRESHOLD)
            label = int(table.y[i])
            error = "false_positive" if not label and pred else "false_negative" if label and not pred else "correct"
            if row["prediction_at_0_50"] != ("phish" if pred else "benign") or row["error_type"] != error:
                raise ValueError("Saved prediction and error columns disagree with the score.")
            result[i] = score
            loaded += 1
    if loaded != len(table):
        raise ValueError("Missing saved prediction rows.")
    return result


def describe(values, mask, np):
    x = values[mask]
    if not len(x):
        return {"rows": 0}
    q = np.quantile(x, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {"rows": len(x), "mean": float(np.mean(x, dtype=np.float64)),
            "zero_fraction": float(np.mean(x == 0)),
            **{name: float(v) for name, v in zip(("min", "p10", "p25", "median", "p75", "p90", "max"), q)}}


def group_summary(groups, y, predicted, np):
    totals, benign, phish = Counter(groups), Counter(groups[y == 0]), Counter(groups[y == 1])
    fp = Counter(groups[(y == 0) & predicted])
    fn = Counter(groups[(y == 1) & ~predicted])
    records = [{"site_group": name, "rows": totals[name], "benign_rows": benign[name], "phishing_rows": phish[name],
                "false_positives": fp[name], "false_negatives": fn[name],
                "false_positive_rate": fp[name] / benign[name] if benign[name] else None,
                "false_negative_rate": fn[name] / phish[name] if phish[name] else None} for name in totals]
    return {"unique_site_groups": len(records),
            "top_groups_by_false_positive_count": sorted(records, key=lambda r: (-r["false_positives"], r["site_group"]))[:20],
            "top_groups_by_false_negative_count": sorted(records, key=lambda r: (-r["false_negatives"], r["site_group"]))[:20]}


def pick_examples(mask, scores, ids, count, category, descending, np):
    candidates = np.flatnonzero(mask)
    candidates = sorted(candidates, key=lambda i: (-scores[i] if descending else scores[i], ids[i]))
    n = min(count, len(candidates))
    extreme_n = (n + 1) // 2
    selected = list(candidates[:extreme_n])
    remainder = candidates[extreme_n:]
    salt = int.from_bytes(hashlib.sha256(category.encode("utf-8")).digest()[:4], "big")
    rng = np.random.default_rng(SEED + salt)
    if n > extreme_n:
        selected += sorted(int(i) for i in rng.choice(remainder, size=n - extreme_n, replace=False))
    return selected


def write_sample_csv(path, table, schema, scores, selected, categories):
    meta, quality, features = schema["metadata_columns"], schema["quality_columns_not_classifier_inputs"], schema["feature_names"]
    tmp = path.with_suffix(".csv.tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["review_category", "phishing_score", *meta, *quality, *features])
            for i in selected:
                w.writerow([categories[i], format(float(scores[i]), ".17g"),
                            *(table.metadata[n][i] for n in meta), *(table.quality[n][i] for n in quality), *table.values[i]])
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=root / "reports/features")
    parser.add_argument("--baseline-report", type=Path, default=root / "reports/baseline_training_v1_1/baseline_training_report.json")
    parser.add_argument("--baseline-script", type=Path, default=root / "10_train_static_baselines.py")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/error_review_v1")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--samples-per-category", type=int, default=8, help="Up to 48 diagnostic pages by default; not training data.")
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.samples_per_category <= 50:
        parser.error("Use positive threads and 1 to 50 samples per category.")
    baseline = load_json(args.baseline_report)
    if baseline.get("status") != "static_baseline_training_complete":
        raise ValueError("Use a completed script 10 v1.1 report.")
    result = next((r for r in baseline["model_results"] if r["model_name"] == MODEL), None)
    if result is None:
        raise ValueError("The completed report must contain lightgbm_all_features.")
    model_dir = args.baseline_report.parent / MODEL
    paths = {"baseline_report": args.baseline_report, "baseline_script": args.baseline_script,
             "extraction_report": args.features_dir / "feature_extraction_report.json",
             "schema": args.features_dir / "static_feature_schema.json",
             "validation": args.features_dir / "validation_static_features.csv",
             "bundle": model_dir / "model_bundle.joblib", "predictions": model_dir / "validation_predictions.csv",
             "importance": model_dir / "feature_importance.csv"}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Missing input: {path}.")
    hashes = {name: digest_file(path) for name, path in paths.items()}
    for name in ("extraction_report", "schema", "validation"):
        if hashes[name] != baseline["run_config"]["input_hashes"][name]:
            raise ValueError(f"Input differs from the completed baseline: {paths[name].name}.")
    for name in ("bundle", "predictions", "importance"):
        if hashes[name] != result["outputs"][paths[name].name]["sha256"]:
            raise ValueError(f"Saved model output changed: {paths[name].name}.")
    if args.output_dir.resolve() in (args.features_dir.resolve(), model_dir.resolve(), args.baseline_report.parent.resolve()):
        raise ValueError("Review outputs must use a separate folder.")
    common = checked_reader(args.baseline_script, baseline["run_config"]["training_code_sha256"])
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits

    schema, extraction = load_json(paths["schema"]), load_json(paths["extraction_report"])
    names = schema["feature_names"]
    if extraction.get("status") != "static_feature_extraction_complete" or names != result["feature_names"] or names != extraction["inputs"]["feature_names"]:
        raise ValueError("Feature schema differs from the trained model.")
    if set(names).intersection(schema["metadata_columns"] + schema["quality_columns_not_classifier_inputs"]):
        raise ValueError("Metadata must remain outside classifier inputs.")
    print("Loading the original validation features and saved predictions...", flush=True)
    table = common.load_features(paths["validation"], schema, baseline["main_validation_counts"], "validation", extraction)
    sensitivity = table.metadata["in_validation_sensitivity_subset"] == 1
    if common.counts(table, sensitivity) != baseline["validation_sensitivity_counts"]:
        raise ValueError("Sensitivity subset differs from the completed run.")
    importance = read_importance(paths["importance"], names)
    scores = read_scores(paths["predictions"], table, schema, np)
    # Hashes identify our own saved baseline bundle before deserialization.
    bundle = joblib.load(paths["bundle"])
    if bundle["model_name"] != MODEL or bundle["feature_names"] != names or bundle["label_mapping"] != {"benign": 0, "phish": 1}:
        raise ValueError("Unexpected bundle model, feature order or label mapping.")
    if bundle["run_config_sha256"] != result["run_config_sha256"] or bundle["feature_extraction_contract"] != schema or bundle["threshold"] != THRESHOLD:
        raise ValueError("Bundle differs from its recorded extraction/training contract.")
    model = bundle["estimator"]
    if model.objective_ != "binary" or list(model.classes_) != [0, 1] or model.booster_.feature_name() != names:
        raise ValueError("Unexpected LightGBM objective, class mapping or feature names.")
    gain = model.booster_.feature_importance(importance_type="gain")
    by_name = {r["feature"]: r["split_gain"] for r in importance}
    if not np.allclose(gain, [by_name[n] for n in names], rtol=1e-10, atol=1e-8):
        raise ValueError("Importance values do not match the saved best model.")
    with threadpool_limits(limits=args.threads):
        reproduced = model.booster_.predict(table.values, num_iteration=model.best_iteration_, num_threads=args.threads)
    max_probability_difference = float(np.max(np.abs(reproduced - scores)))
    if max_probability_difference > 1e-9:
        raise ValueError("Reloaded model does not reproduce the saved validation scores.")
    metrics = common.binary_metrics(table.y, scores)
    if metrics["confusion_matrix"] != result["metrics"]["main_validation"]["confusion_matrix"]:
        raise ValueError("Validation confusion counts differ from the completed baseline.")
    pred = scores >= THRESHOLD
    masks = {"benign": table.y == 0, "phish": table.y == 1,
             "false_positive": (table.y == 0) & pred, "true_benign": (table.y == 0) & ~pred,
             "false_negative": (table.y == 1) & ~pred, "true_phish": (table.y == 1) & pred}
    groups = table.metadata["site_group"]
    summary = group_summary(groups, table.y, pred, np)
    fp_group = summary["top_groups_by_false_positive_count"][0]["site_group"] if masks["false_positive"].any() else None
    fn_group = summary["top_groups_by_false_negative_count"][0]["site_group"] if masks["false_negative"].any() else None
    fp_match = groups == fp_group if fp_group else np.zeros(len(table), dtype=bool)
    fn_match = groups == fn_group if fn_group else np.zeros(len(table), dtype=bool)
    masks["benign_in_largest_fp_group"] = masks["benign"] & fp_match
    masks["fp_in_largest_fp_group"] = masks["false_positive"] & fp_match
    # These descriptive cohorts are selected using validation errors; they
    # are review targets and must not be treated as fresh performance tests.
    focus = [r["feature"] for r in importance[:15]]
    focus += [n for n in ("password_input_count", "password_form_cross_site_action_count", "js_fetch_call_count",
                         "js_form_submit_call_count", "form_cross_site_action_count") if n not in focus]
    distributions = [{"feature": n, "share_of_training_split_gain": by_name[n] / sum(by_name.values()),
                      "cohorts": {name: describe(table.values[:, names.index(n)], mask, np) for name, mask in masks.items()}} for n in focus]
    capped = (table.quality["html_analysis_truncated"] > 0) | (table.quality["js_analysis_truncated"] > 0)
    quality = {"reached_analysis_cap": common.binary_metrics(table.y[capped], scores[capped]),
               "did_not_reach_analysis_caps": common.binary_metrics(table.y[~capped], scores[~capped]),
               "quality_flag_rows_by_error_cohort": {name: {q: int(np.count_nonzero((table.quality[q] > 0) & mask))
                   for q in schema["quality_columns_not_classifier_inputs"]} for name, mask in masks.items()}}
    selection = [
        ("false_positive_largest_group", masks["false_positive"] & fp_match, True),
        ("false_positive_other_groups", masks["false_positive"] & ~fp_match, True),
        ("false_negative_largest_group", masks["false_negative"] & fn_match, False),
        ("false_negative_other_groups", masks["false_negative"] & ~fn_match, False),
        ("true_benign_controls", masks["true_benign"] & fp_match if (masks["true_benign"] & fp_match).any() else masks["true_benign"], False),
        ("true_phish_controls", masks["true_phish"] & fn_match if (masks["true_phish"] & fn_match).any() else masks["true_phish"], True)]
    selected, categories, selection_report = [], {}, []
    for category, mask, descending in selection:
        indices = pick_examples(mask, scores, table.metadata["record_id"], args.samples_per_category, category, descending, np)
        selected += indices
        categories.update({i: category for i in indices})
        selection_report.append({"category": category, "eligible_rows": int(mask.sum()), "selected_rows": len(indices)})
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Invalid diagnostic sample selection.")
    print(f"Saved scores reproduced. Explaining {len(selected)} selected errors and controls...", flush=True)
    x = np.ascontiguousarray(table.values[selected])
    with threadpool_limits(limits=args.threads):
        raw = model.booster_.predict(x, raw_score=True, num_iteration=model.best_iteration_, num_threads=args.threads)
        contributions = np.asarray(model.booster_.predict(x, pred_contrib=True, num_iteration=model.best_iteration_, num_threads=args.threads))
    if contributions.shape != (len(selected), len(names) + 1) or not np.isfinite(contributions).all():
        raise ValueError("Unexpected binary feature-contribution output.")
    reconstruction_error = float(np.max(np.abs(contributions.sum(axis=1) - raw)))
    reconstructed_scores = 1 / (1 + np.exp(-np.clip(raw, -700, 700)))
    if reconstruction_error > 1e-7 or not np.allclose(reconstructed_scores, scores[selected], rtol=0, atol=1e-9):
        raise ValueError("Feature contributions do not reconstruct the saved scores.")
    examples = []
    for j, i in enumerate(selected):
        feature_values = {n: float(table.values[i, k]) for k, n in enumerate(names)}
        def explanation(indices):
            return [{"feature": names[k], "value": feature_values[names[k]], "raw_score_contribution": float(contributions[j, k])} for k in indices]
        positive = sorted([k for k in range(len(names)) if contributions[j, k] > 0], key=lambda k: (-contributions[j, k], names[k]))[:8]
        negative = sorted([k for k in range(len(names)) if contributions[j, k] < 0], key=lambda k: (contributions[j, k], names[k]))[:8]
        examples.append({"review_category": categories[i],
                         "metadata": {n: str(table.metadata[n][i]) for n in schema["metadata_columns"]},
                         "phishing_score": float(scores[i]), "raw_score": float(raw[j]),
                         "expected_raw_score": float(contributions[j, -1]),
                         "quality": {n: int(table.quality[n][i]) for n in schema["quality_columns_not_classifier_inputs"]},
                         "feature_values": feature_values, "strongest_push_toward_phish": explanation(positive),
                         "strongest_push_toward_benign": explanation(negative)})
    for name, path in paths.items():
        if digest_file(path) != hashes[name]:
            raise ValueError("An input changed while reviewing; no complete report was written.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "review_sample_features.csv"
    write_sample_csv(sample_path, table, schema, scores, selected, categories)
    report = {"status": "static_error_review_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "script_version": VERSION, "review_code_sha256": digest_file(Path(__file__)), "input_hashes": hashes,
              "model_name": MODEL, "threshold": THRESHOLD, "seed": SEED, "threads": args.threads,
              "python_version": platform.python_version(),
              "package_versions": {n: version(n) for n in ("numpy", "lightgbm", "scikit-learn", "joblib", "threadpoolctl")},
              "main_validation_counts": common.counts(table), "validation_metrics_recomputed": metrics,
              "largest_false_positive_group": fp_group, "largest_false_negative_group": fn_group,
              "group_summary": summary, "global_gain_importance": importance,
              "gain_share_by_named_group": {
                  "anchor_features": sum(r["share_of_training_split_gain"] for r in importance if r["feature"].startswith("anchor_")),
                  "five_size_features": sum(r["share_of_training_split_gain"] for r in importance if r["feature"] in SIZE_FEATURES)},
              "focus_feature_distributions": distributions, "quality_diagnostics": quality,
              "review_sample_selection": selection_report, "review_examples": examples,
              "sample_selection_policy": "Half the selected rows have the most extreme error/correct scores; the remaining rows are sampled reproducibly within each category. Controls prefer the corresponding largest-error group when available. Review data are not used for training.",
              "contribution_definition": "Native LightGBM pred_contrib; positive values increase the raw binary score toward phish, negative values decrease it. The expected value plus all feature contributions reconstructs the raw score; sigmoid reconstructs the saved phishing score. Values are not probabilities or causal effects.",
              "verification": {"max_saved_probability_difference": max_probability_difference,
                               "max_contribution_raw_score_reconstruction_error": reconstruction_error,
                               "inputs_unchanged": True},
              "sample_csv_sha256": digest_file(sample_path), "model_retrained": False, "threshold_changed": False,
              "training_data_loaded": False, "official_test_evaluated": False,
              "limitations": ["Global gain importance measures training split improvements, not error-specific causes or direction.",
                              "Correlated features may share or redistribute attribution; low gain does not prove a feature is useless.",
                              "Selected error cohorts are development diagnostics, not independent evaluation datasets or estimates of deployment accuracy.",
                              "The report contains features and model associations only; HTML content, label correctness and actual executed behavior have not been checked.",
                              "No domain allowlist, relabeling, automatic feature deletion or threshold change is performed."]}
    report_path = args.output_dir / "error_review_report.json"
    save_json(report_path, report)
    print(f"Saved report: {report_path.resolve()}", flush=True)
    print(f"Saved selected feature rows: {sample_path.resolve()}", flush=True)
    print("Static error review complete. No model trained or changed; official test untouched.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ImportError) as error:
        raise SystemExit(f"Error review stopped: {error}. A completed report is valid only after the final completion message.")
    except MemoryError:
        raise SystemExit("Not enough available RAM. Close memory-heavy apps and rerun. Model and feature files were not changed.")
