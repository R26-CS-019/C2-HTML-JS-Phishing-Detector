"""Compare two form-context LightGBM models with the verified script 10 baselines.

Save beside unchanged scripts 09, 10 v1.1 and 15. Uses the already installed
NumPy, scikit-learn, LightGBM, joblib and threadpoolctl; no Pandas CSV reader.
Only the feature CSVs and existing baseline predictions are read. No dataset
Parquet, websites, JavaScript execution or official test data is opened.

The two new models use all 106 features and 101 features without the original
five size measurements. Training settings, package versions, seed, balanced
rows, domain-separated splits and threshold 0.50 match the completed baseline.
Existing baseline predictions are verified and reused, without retraining the
86/81-feature baselines. Main validation remains development data.

Completed models can be reused on rerun when configuration/output hashes match.
An interrupted fit restarts that model; previously completed models are kept.
"""

import argparse
import copy
import csv
import hashlib
import importlib.util
import inspect
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

VERSION = "1.0"
THRESHOLD = 0.50
SEED = 42
BASELINE_SCRIPT_SHA = "e3b1b8f3c8400b55437246881f337ea57621aa63ee3b5e1e42f2740a736da6fb"
SIZE_FEATURES = ("html_character_count", "tag_count", "text_character_count",
                 "title_character_count", "js_analysed_character_count")
CONTEXT_NAMES = (
    "context_user_entry_control_count", "context_passwordlike_control_count",
    "context_plaintext_passwordlike_control_count", "context_label_only_passwordlike_control_count",
    "context_normalized_password_cue_control_count", "context_identitylike_control_count",
    "context_passwordlike_textarea_count", "context_passwordlike_unowned_control_count",
    "context_passwordlike_form_count", "context_identity_password_pair_form_count",
    "context_passwordlike_post_form_count", "context_passwordlike_get_form_count",
    "context_plaintext_passwordlike_post_form_count", "context_passwordlike_cross_site_form_count",
    "context_passwordlike_https_to_http_form_count", "context_passwordlike_handler_form_count",
    "context_passwordlike_control_ratio", "context_identitylike_control_ratio",
    "context_plaintext_passwordlike_fraction", "context_identity_password_pair_form_ratio")
CONTEXT_QUALITY = (
    "context_parse_error", "context_association_storage_cut", "context_duplicate_element_id_seen",
    "context_human_verification_hint_present", "context_unavailable_page_hint_present",
    "context_text_hint_scan_cut")
MODEL_PAIRS = (
    ("lightgbm_context_all_features", "lightgbm_all_features"),
    ("lightgbm_context_without_size_features", "lightgbm_without_size_features"))
PACKAGES = ("numpy", "scikit-learn", "lightgbm", "joblib", "threadpoolctl")
PREDICTION_COLUMNS = ["record_id", "label", "site_group", "source_file",
    "sampled_similarity_candidate", "in_validation_sensitivity_subset",
    "phishing_score", "prediction_at_0_50", "error_type"]


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


def load_reader(path, expected_sha):
    if digest_file(path) != expected_sha or expected_sha != BASELINE_SCRIPT_SHA:
        raise ValueError("Keep the unchanged script 10 v1.1 used for your completed baseline.")
    spec = importlib.util.spec_from_file_location("form_context_baseline_utilities", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.VERSION != "1.1" or module.SEED != SEED or module.THRESHOLD != THRESHOLD or tuple(module.SIZE_FEATURES) != SIZE_FEATURES:
        raise ValueError("Unexpected baseline feature reader or comparison contract.")
    return module


def checked_inputs(args):
    paths = {"base_report": args.base_features_dir / "feature_extraction_report.json",
        "base_schema": args.base_features_dir / "static_feature_schema.json",
        "context_report": args.features_dir / "form_context_feature_report.json",
        "context_schema": args.features_dir / "static_feature_schema.json",
        "baseline_report": args.baseline_dir / "baseline_training_report.json",
        "baseline_config": args.baseline_dir / "run_config.json",
        "baseline_script": args.baseline_script, "base_extractor": args.base_extractor_script,
        "context_extractor": args.context_extractor_script}
    csv_names = {"train": "train_static_features.csv", "validation": "validation_static_features.csv",
                 "sensitivity": "validation_sensitivity_static_features.csv"}
    for key, filename in csv_names.items():
        paths["base_" + key] = args.base_features_dir / filename
        paths["context_" + key] = args.features_dir / filename
    for _, old_name in MODEL_PAIRS:
        paths[old_name + "_predictions"] = args.baseline_dir / old_name / "validation_predictions.csv"
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Missing input: {path}. Complete scripts 10 and 15, or specify their existing report folders.")
    hashes = {key: digest_file(path) for key, path in paths.items()}
    base, base_schema, context, schema, baseline = (load_json(paths[key]) for key in
        ("base_report", "base_schema", "context_report", "context_schema", "baseline_report"))
    if base.get("status") != "static_feature_extraction_complete" or context.get("status") != "form_context_feature_extraction_complete" or baseline.get("status") != "static_baseline_training_complete":
        raise ValueError("All input reports must record completed extraction/training.")
    if baseline.get("official_test_evaluated") is not False or baseline.get("operational_threshold_selected") is not False:
        raise ValueError("This experiment requires the existing development baseline at threshold 0.50.")
    for key, expected in (("model_trained", False), ("official_test_evaluated", False),
            ("balanced_splits_changed", False), ("labels_changed", False), ("original_features_unchanged", True)):
        if context.get(key) is not expected:
            raise ValueError(f"Unexpected form-context report policy: {key}.")
    for report, prefix in ((base, "base_"), (context, "context_")):
        for key in ("schema", "train", "validation", "sensitivity"):
            if hashes[prefix + key] != report["outputs"][paths[prefix + key].name]["sha256"]:
                raise ValueError(f"Feature output differs from its report: {paths[prefix + key]}.")
        for key, split in (("train", "train"), ("validation", "validation")):
            if report["outputs"][csv_names[key]]["rows"] != sum(report["extracted_label_counts"][split].values()):
                raise ValueError("Feature report row counts are inconsistent.")
        if report["outputs"][csv_names["sensitivity"]]["rows"] != sum(report["validation_sensitivity_counts"].values()):
            raise ValueError("Sensitivity feature report row counts are inconsistent.")
    ci, bi, bc = context["inputs"], base["inputs"], baseline["run_config"]
    for key in ("schema", "train", "validation", "sensitivity"):
        if ci["input_hashes"][key] != hashes["base_" + key] or bc["input_hashes"][key] != hashes["base_" + key]:
            raise ValueError("New features and the baseline do not use the same original feature files.")
    if ci["input_hashes"]["extraction_report"] != hashes["base_report"] or bc["input_hashes"]["extraction_report"] != hashes["base_report"]:
        raise ValueError("Original extraction report provenance differs between experiments.")
    if any(value != hashes["base_extractor"] for value in
            (bi["code_sha256"], base_schema["implementation_code_sha256"], ci["base_implementation_code_sha256"],
             ci["input_hashes"]["extractor_script"], schema["base_implementation_code_sha256"],
             bc["extractor_code_sha256"], bc["input_hashes"]["extractor"])):
        raise ValueError("Script 09 differs from the original extraction implementation.")
    if ci["code_sha256"] != hashes["context_extractor"] or schema["implementation_code_sha256"] != hashes["context_extractor"]:
        raise ValueError("Script 15 differs from the implementation that created the new features.")
    if bc["training_code_sha256"] != hashes["baseline_script"] or hashes["baseline_script"] != BASELINE_SCRIPT_SHA:
        raise ValueError("Baseline training code does not match unchanged script 10 v1.1.")
    if load_json(paths["baseline_config"]) != bc or hashes["baseline_config"] != baseline["run_config_sha256"]:
        raise ValueError("Baseline training report and saved run configuration differ.")
    if bc["seed"] != SEED or bc["threshold"] != THRESHOLD or bc["threads"] != args.threads:
        raise ValueError(f"Use --threads {bc['threads']} to match the existing seed/threshold/thread settings.")
    names = schema["feature_names"]
    if base["feature_count"] != 86 or base_schema["feature_names"] != bi["feature_names"] or len(base_schema["feature_names"]) != 86:
        raise ValueError("Expected the original 86-feature schema.")
    if (context["base_feature_count"], context["context_feature_count"], context["feature_count"], schema["feature_count"]) != (86, 20, 106, 106):
        raise ValueError("Expected 86 original features plus 20 form-context features.")
    if names != base_schema["feature_names"] + list(CONTEXT_NAMES) or ci["context_feature_names"] != list(CONTEXT_NAMES) or len(names) != len(set(names)):
        raise ValueError("Unexpected form-context feature names or order.")
    if schema["metadata_columns"] != base_schema["metadata_columns"] or schema["quality_columns_not_classifier_inputs"] != base_schema["quality_columns_not_classifier_inputs"] + list(CONTEXT_QUALITY):
        raise ValueError("Metadata/quality column contracts have changed.")
    if set(names).intersection(schema["metadata_columns"] + schema["quality_columns_not_classifier_inputs"]) or not set(SIZE_FEATURES).issubset(names):
        raise ValueError("Labels, IDs, site names and quality flags must remain outside classifier features.")
    if schema["parameters"] != ci["parameters"] or ci["parameters"] != bi["parameters"] or bi["parameters"] != base_schema["parameters"]:
        raise ValueError("HTML/JavaScript analysis limits differ between experiments.")
    if schema["association_limits"] != ci["association_limits"] or schema["external_js_policy"] != "references_only_no_download" or ci["external_js_policy"] != schema["external_js_policy"]:
        raise ValueError("Unexpected form-context extraction contract.")
    if context["extracted_label_counts"] != base["extracted_label_counts"] or context["validation_sensitivity_counts"] != base["validation_sensitivity_counts"]:
        raise ValueError("Balanced split counts have changed.")
    if baseline["training_counts"] != base["extracted_label_counts"]["train"] or baseline["main_validation_counts"] != base["extracted_label_counts"]["validation"] or baseline["validation_sensitivity_counts"] != base["validation_sensitivity_counts"]:
        raise ValueError("Baseline and form-context balanced splits differ.")
    old_results = {r["model_name"]: r for r in baseline["model_results"]}
    if len(old_results) != len(baseline["model_results"]):
        raise ValueError("Repeated model results in the baseline report.")
    for _, old_name in MODEL_PAIRS:
        result = old_results[old_name]
        expected_names = [n for n in base_schema["feature_names"] if old_name != "lightgbm_without_size_features" or n not in SIZE_FEATURES]
        if result["run_config_sha256"] != baseline["run_config_sha256"] or result["feature_names"] != expected_names or result["feature_count"] != len(expected_names) or not result["eligible_for_validation_ranking"]:
            raise ValueError("Unexpected baseline model configuration or feature list.")
        if hashes[old_name + "_predictions"] != result["outputs"]["validation_predictions.csv"]["sha256"]:
            raise ValueError(f"Saved baseline predictions changed: {old_name}.")
    return paths, hashes, base_schema, context, schema, baseline, old_results


def check_csv_against_table(path, csv_schema, table, indices, common):
    """Compare every old value and metadata field, streaming to save RAM."""
    np = common.np
    metadata = csv_schema["metadata_columns"]
    quality = csv_schema["quality_columns_not_classifier_inputs"]
    names = csv_schema["feature_names"]
    feature_indices = [table.feature_names.index(n) for n in names]
    expected_header = [*metadata, *quality, *names]
    rows = 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        if next(reader, None) != expected_header:
            raise ValueError(f"Unexpected columns in {path.name}.")
        for row in reader:
            if rows >= len(indices) or len(row) != len(expected_header):
                raise ValueError(f"Unexpected comparison row count/width in {path.name}.")
            i = int(indices[rows])
            if any(row[j] != str(table.metadata[n][i]) for j, n in enumerate(metadata)):
                raise ValueError(f"Original row metadata or subset membership changed in {path.name}, row {rows + 1}.")
            if any(int(row[len(metadata) + j]) != int(table.quality[n][i]) for j, n in enumerate(quality)):
                raise ValueError(f"Original quality metadata changed in {path.name}, row {rows + 1}.")
            with np.errstate(over="ignore", invalid="ignore"):
                numeric = np.asarray(row[len(metadata) + len(quality):], dtype=np.float32)
            if not np.array_equal(numeric, table.values[i, feature_indices]):
                raise ValueError(f"Feature values changed in {path.name}, row {rows + 1}.")
            rows += 1
    if rows != len(indices):
        raise ValueError(f"Comparison file has missing rows: {path.name}.")


def load_predictions(path, validation, common):
    np = common.np
    scores = np.empty(len(validation), dtype=np.float64)
    rows = 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != PREDICTION_COLUMNS:
            raise ValueError(f"Unexpected prediction CSV header: {path}.")
        for row in reader:
            if rows >= len(validation) or None in row or any(v is None for v in row.values()):
                raise ValueError(f"Unexpected prediction row count/width: {path}.")
            if any(row[n] != str(validation.metadata[n][rows]) for n in PREDICTION_COLUMNS[:6]):
                raise ValueError(f"Prediction IDs, labels or split metadata differ: {path}, row {rows + 1}.")
            score = float(row["phishing_score"])
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid saved phishing score.")
            predicted = int(score >= THRESHOLD)
            error = "false_positive" if validation.y[rows] == 0 and predicted else "false_negative" if validation.y[rows] == 1 and not predicted else "correct"
            if row["prediction_at_0_50"] != ("phish" if predicted else "benign") or row["error_type"] != error:
                raise ValueError("Saved prediction/error flags do not match threshold 0.50.")
            scores[rows] = score
            rows += 1
    if rows != len(validation):
        raise ValueError("Saved prediction CSV has missing rows.")
    return scores


def assert_metrics_match(actual, expected, np):
    for key, value in actual.items():
        other = expected.get(key)
        if isinstance(value, float):
            if not isinstance(other, (int, float)) or not np.isclose(value, other, rtol=0, atol=1e-12):
                raise ValueError(f"Saved/recomputed metrics differ: {key}.")
        elif value != other:
            raise ValueError(f"Saved/recomputed metrics differ: {key}.")


def evaluation_masks(validation, common):
    capped = (validation.quality["html_analysis_truncated"] > 0) | (validation.quality["js_analysis_truncated"] > 0)
    context_limited = (validation.quality["context_association_storage_cut"] > 0) | (validation.quality["context_parse_error"] > 0)
    return {"main_validation": common.np.ones(len(validation), dtype=bool),
        "validation_sensitivity": validation.metadata["in_validation_sensitivity_subset"] == 1,
        "validation_flagged_similarity_candidates": validation.metadata["sampled_similarity_candidate"] == 1,
        "validation_reached_an_analysis_cap": capped, "validation_did_not_reach_analysis_caps": ~capped,
        "validation_context_association_limited": context_limited,
        "validation_context_association_not_limited": ~context_limited}


def train_one(name, feature_names, args, schema, train, validation, config, config_hash, common):
    model_dir = args.output_dir / name
    model_dir.mkdir(exist_ok=True)
    result_path = model_dir / "model_result.json"
    output_names = ("model_bundle.joblib", "validation_predictions.csv", "feature_importance.csv", "learning_curve.json")
    if result_path.exists():
        cached = load_json(result_path)
        if cached.get("run_config_sha256") != config_hash or cached.get("feature_names") != feature_names or cached.get("model_name") != name:
            raise ValueError("A saved model uses another configuration. Choose a new --output-dir.")
        if set(cached["outputs"]) == set(output_names) and all((model_dir / filename).is_file() and digest_file(model_dir / filename) == cached["outputs"][filename]["sha256"] for filename in output_names):
            print(f"Reusing completed model: {name}", flush=True)
            scores = load_predictions(model_dir / "validation_predictions.csv", validation, common)
            for key, mask in evaluation_masks(validation, common).items():
                assert_metrics_match(common.binary_metrics(validation.y[mask], scores[mask]), cached["metrics"][key], common.np)
            return cached, scores
        print(f"Incomplete model output: restarting {name}.", flush=True)
    x_train, x_validation = train.feature_matrix(feature_names), validation.feature_matrix(feature_names)
    model = common.lgb.LGBMClassifier(**config["lightgbm_hyperparameters"])
    history = {}
    print(f"Training {name}: {len(train):,} rows, {len(feature_names)} features, CPU threads={args.threads}.", flush=True)
    started = time.perf_counter()
    with common.threadpool_limits(limits=args.threads):
        eval_args = {"eval_X": x_validation, "eval_y": validation.y} if "eval_X" in inspect.signature(model.fit).parameters else {"eval_set": [(x_validation, validation.y)]}
        model.fit(x_train, train.y, **eval_args, feature_name=feature_names, eval_names=["main_validation"],
            eval_metric="binary_logloss", callbacks=[
                common.lgb.early_stopping(config["early_stopping"]["rounds"], first_metric_only=True, verbose=True),
                common.lgb.log_evaluation(100), common.lgb.record_evaluation(history)])
        fit_seconds = time.perf_counter() - started
        scores = common.positive_probabilities(model, x_validation)
        train_scores = common.positive_probabilities(model, x_train)
    metrics = {"training": common.binary_metrics(train.y, train_scores)}
    metrics.update({key: common.binary_metrics(validation.y[mask], scores[mask]) for key, mask in evaluation_masks(validation, common).items()})
    bundle = {"bundle_version": "form_context_" + VERSION, "model_name": name, "estimator": model,
        "feature_names": feature_names, "label_mapping": common.LABELS, "threshold": THRESHOLD,
        "feature_extraction_contract": schema, "extractor_code_sha256": config["context_extractor_code_sha256"],
        "base_extractor_code_sha256": config["base_extractor_code_sha256"], "run_config_sha256": config_hash,
        "package_versions": config["package_versions"], "training_label_counts": common.counts(train),
        "development_stage": "form_context_development_not_final_tested_or_calibrated"}
    model_path = model_dir / "model_bundle.joblib"
    temporary = model_path.with_suffix(".joblib.tmp")
    try:
        common.joblib.dump(bundle, temporary, compress=3)
        temporary.replace(model_path)
    finally:
        temporary.unlink(missing_ok=True)
    common.write_predictions(model_dir / "validation_predictions.csv", validation, validation.y, scores)
    common.write_importance(model_dir / "feature_importance.csv", model, feature_names, name)
    save_json(model_dir / "learning_curve.json", history)
    result = {"model_name": name, "run_config_sha256": config_hash, "feature_names": feature_names,
        "feature_count": len(feature_names), "fit_seconds": fit_seconds,
        "best_iteration_or_solver_iterations": int(model.best_iteration_ or model.n_estimators_),
        "solver_converged": True, "eligible_for_validation_ranking": True, "fit_warnings": [],
        "metrics": metrics, "reused_from_baseline": False,
        "outputs": {filename: {"sha256": digest_file(model_dir / filename)} for filename in output_names}}
    save_json(result_path, result)
    m = metrics["main_validation"]
    print(f"  Main validation: accuracy={m['accuracy']:.4f}; recall={m['recall']:.4f}; FPR={m['false_positive_rate']:.4f}; AP={m['average_precision']:.4f}.", flush=True)
    return result, scores


def paired_changes(y, old_scores, new_scores, common):
    np = common.np
    old, new = old_scores >= THRESHOLD, new_scores >= THRESHOLD
    benign, phish = y == 0, y == 1
    old_m = common.binary_metrics(y, old_scores)
    new_m = common.binary_metrics(y, new_scores)
    keys = ("accuracy", "recall", "precision", "f1", "false_positive_rate")
    return {"rows": len(y), "threshold": THRESHOLD,
        "change_percentage_points": {k: 100 * (new_m[k] - old_m[k]) if new_m.get(k) is not None and old_m.get(k) is not None else None for k in keys},
        "average_precision_change": new_m["average_precision"] - old_m["average_precision"] if len(y) and new_m.get("average_precision") is not None else None,
        "false_positives_removed": int(np.count_nonzero(benign & old & ~new)),
        "new_false_positives": int(np.count_nonzero(benign & ~old & new)),
        "false_negatives_removed": int(np.count_nonzero(phish & ~old & new)),
        "new_false_negatives": int(np.count_nonzero(phish & old & ~new))}


def diagnostics_groups(context):
    """Previously inspected groups, used for reporting only and never in X."""
    groups = {}
    for category in ("false_positive_largest_group", "false_negative_largest_group"):
        names = {r["metadata"]["site_group"] for r in context["selected_review_pages_context"] if r["review_category"] == category}
        if len(names) > 1:
            raise ValueError("The inspected largest-group examples disagree on their site group.")
        if names:
            groups[category] = next(iter(names))
    return groups


def write_comparison(path, results):
    fields = ["model", "reused_from_baseline", "evaluation_set", "rows", "feature_count", "accuracy",
              "precision", "recall", "f1", "false_positive_rate", "roc_auc", "average_precision", "tn", "fp", "fn", "tp"]
    temporary = path.with_suffix(".csv.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for result in results:
                for evaluation_set in ("main_validation", "validation_sensitivity"):
                    metrics = result["metrics"][evaluation_set]
                    writer.writerow({"model": result["model_name"], "reused_from_baseline": result["reused_from_baseline"],
                        "evaluation_set": evaluation_set, "feature_count": result["feature_count"],
                        **{key: metrics[key] for key in fields[3:4] + fields[5:12]}, **metrics["confusion_matrix"]})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=root / "reports/features_context_v2")
    parser.add_argument("--base-features-dir", type=Path, default=root / "reports/features")
    parser.add_argument("--baseline-dir", type=Path, default=root / "reports/baseline_training_v1_1")
    parser.add_argument("--baseline-script", type=Path, default=root / "10_train_static_baselines.py")
    parser.add_argument("--base-extractor-script", type=Path, default=root / "09_extract_static_features.py")
    parser.add_argument("--context-extractor-script", type=Path, default=root / "15_extract_form_context.py")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/form_context_training_v2")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive and match the completed baseline.")
    for directory in (args.features_dir, args.base_features_dir, args.baseline_dir):
        if args.output_dir.resolve().is_relative_to(directory.resolve()) or directory.resolve().is_relative_to(args.output_dir.resolve()):
            parser.error("Use a separate training output folder outside the feature and baseline folders.")
    print(f"Form-context trainer v{VERSION}; verifying extraction and baseline provenance...", flush=True)
    paths, hashes, base_schema, context, schema, baseline, old_results = checked_inputs(args)
    common = load_reader(args.baseline_script, baseline["run_config"]["training_code_sha256"])
    np = common.np
    packages = {name: version(name) for name in PACKAGES}
    bc = baseline["run_config"]
    if packages != bc["package_versions"] or platform.python_version() != bc["python_version"]:
        raise ValueError("Package/Python versions changed since the baseline. Keep the original working environment for this controlled comparison.")
    fit_args = SimpleNamespace(trees=bc["lightgbm_hyperparameters"]["n_estimators"], threads=args.threads)
    parameters = common.lgb_parameters(fit_args)
    if parameters != bc["lightgbm_hyperparameters"] or bc["early_stopping"]["metric"] != "binary_logloss" or bc["early_stopping"]["set"] != "main_validation":
        raise ValueError("Unexpected baseline hyperparameters or early-stopping settings.")
    if fit_args.trees < 2 or bc["early_stopping"]["rounds"] < 1:
        raise ValueError("Invalid baseline iteration settings.")
    print("Loading 106-feature training and validation files...", flush=True)
    train = common.load_features(paths["context_train"], schema, context["extracted_label_counts"]["train"], "train", context)
    validation = common.load_features(paths["context_validation"], schema, context["extracted_label_counts"]["validation"], "validation", context)
    if set(train.metadata["site_group"]) & set(validation.metadata["site_group"]) or set(train.metadata["record_id"]) & set(validation.metadata["record_id"]):
        raise ValueError("Training and validation share a record ID or site group.")
    sensitivity = validation.metadata["in_validation_sensitivity_subset"] == 1
    if common.counts(validation, sensitivity) != context["validation_sensitivity_counts"]:
        raise ValueError("Validation sensitivity membership differs from the report.")
    if any(c["benign"] != c["phish"] or c["benign"] == 0 for c in (common.counts(train), common.counts(validation), common.counts(validation, sensitivity))):
        raise ValueError("Every training/validation subset must be nonempty and 1:1 balanced.")
    print("Checking original 86 feature values, row metadata and sensitivity membership...", flush=True)
    check_csv_against_table(paths["base_train"], base_schema, train, np.arange(len(train)), common)
    check_csv_against_table(paths["base_validation"], base_schema, validation, np.arange(len(validation)), common)
    check_csv_against_table(paths["context_sensitivity"], schema, validation, np.flatnonzero(sensitivity), common)
    check_csv_against_table(paths["base_sensitivity"], base_schema, validation, np.flatnonzero(sensitivity), common)
    groups = diagnostics_groups(context)
    masks = evaluation_masks(validation, common)
    all_results, score_vectors = [], {}
    for _, old_name in MODEL_PAIRS:
        scores = load_predictions(paths[old_name + "_predictions"], validation, common)
        old = copy.deepcopy(old_results[old_name])
        for key, mask in masks.items():
            metrics = common.binary_metrics(validation.y[mask], scores[mask])
            if key in old["metrics"]:
                assert_metrics_match(metrics, old["metrics"][key], np)
            old["metrics"][key] = metrics
        old["reused_from_baseline"] = True
        old["source_model_directory"] = str((args.baseline_dir / old_name).resolve())
        all_results.append(old)
        score_vectors[old_name] = scores
        print(f"Verified previous model: {old_name} ({old['feature_count']} features).", flush=True)
    model_features = {name: [f for f in schema["feature_names"] if name != "lightgbm_context_without_size_features" or f not in SIZE_FEATURES] for name, _ in MODEL_PAIRS}
    config = {"script_version": VERSION, "training_code_sha256": digest_file(Path(__file__)), "input_hashes": hashes,
        "baseline_run_config_sha256": baseline["run_config_sha256"],
        "base_extractor_code_sha256": hashes["base_extractor"], "context_extractor_code_sha256": hashes["context_extractor"],
        "seed": SEED, "threshold": THRESHOLD, "threads": args.threads,
        "lightgbm_hyperparameters": parameters, "early_stopping": bc["early_stopping"],
        "model_features": model_features, "size_features_omitted_in_comparison": list(SIZE_FEATURES),
        "diagnostic_groups_not_classifier_inputs": groups,
        "candidate_ranking": bc["candidate_ranking"], "python_version": platform.python_version(),
        "package_versions": packages, "input_reader": "python_csv_and_numpy",
        "method": "Same balanced rows, old numeric model inputs, packages, hyperparameters and threshold; add 20 form-context features. Reuse verified baseline predictions."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    if config_path.exists():
        if load_json(config_path) != config:
            raise ValueError("The output folder belongs to another configuration. Choose a new --output-dir.")
    elif any(args.output_dir.iterdir()):
        raise ValueError("Use an empty training output folder or the existing folder with its matching run_config.json.")
    else:
        save_json(config_path, config)
    config_hash = digest_file(config_path)
    for name, _ in MODEL_PAIRS:
        result, scores = train_one(name, model_features[name], args, schema, train, validation, config, config_hash, common)
        all_results.append(result)
        score_vectors[name] = scores
    paired = {}
    for name, old_name in MODEL_PAIRS:
        paired[name] = {"compared_with": old_name,
            **{key: paired_changes(validation.y[mask], score_vectors[old_name][mask], score_vectors[name][mask], common)
               for key, mask in masks.items() if key in ("main_validation", "validation_sensitivity")}}
    group_diagnostics = {}
    for category, group in groups.items():
        mask = validation.metadata["site_group"] == group
        group_diagnostics[category] = {"site_group": group, "rows": int(np.count_nonzero(mask)),
            "model_metrics": {name: common.binary_metrics(validation.y[mask], scores[mask]) for name, scores in score_vectors.items()},
            "paired_changes": {name: paired_changes(validation.y[mask], score_vectors[old][mask], score_vectors[name][mask], common) for name, old in MODEL_PAIRS}}
    for key, path in paths.items():
        if digest_file(path) != hashes[key]:
            raise ValueError(f"An input changed during training: {path}. No new complete report was written.")
    if digest_file(config_path) != config_hash:
        raise ValueError("Run configuration changed during training.")
    ranked = sorted(all_results, key=common.ranking_key)
    comparison_path = args.output_dir / "model_comparison.csv"
    write_comparison(comparison_path, all_results)
    report = {"status": "form_context_model_training_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": VERSION, "run_config": config, "run_config_sha256": config_hash,
        "feature_count": 106, "context_feature_count": 20, "training_counts": common.counts(train),
        "main_validation_counts": common.counts(validation), "validation_sensitivity_counts": common.counts(validation, sensitivity),
        "shared_train_validation_site_groups": 0, "shared_train_validation_record_ids": 0,
        "original_model_inputs_verified_at_training_precision": True,
        "original_features_unchanged": True, "balanced_splits_changed": False, "labels_changed": False,
        "model_results": all_results, "paired_validation_changes": paired,
        "previously_inspected_group_diagnostics": group_diagnostics,
        "training_constant_features": [n for n, same in zip(schema["feature_names"], np.min(train.values, axis=0) == np.max(train.values, axis=0)) if same],
        "context_quality_summary_not_classifier_inputs": context["quality_summary_by_split_and_label"],
        "candidate_ranking_by_main_validation": [r["model_name"] for r in ranked],
        "leading_validation_candidate": ranked[0]["model_name"],
        "highest_accuracy_candidate_at_0_50": max(all_results, key=lambda r: r["metrics"]["main_validation"]["accuracy"])["model_name"],
        "comparison_csv_sha256": digest_file(comparison_path), "official_test_evaluated": False,
        "operational_threshold_selected": False, "websites_requested": False, "javascript_executed": False,
        "limitations": [
            "Main validation was used for earlier error review, feature design, early stopping and candidate selection. These are development results, not independent final accuracy.",
            "Adding features changes the sampled columns and fitted trees, even with the same seed/hyperparameters. This is a feature-set comparison, not causal attribution to any single new feature.",
            "Threshold 0.50 is fixed for this comparison, not calibrated or selected for deployment. Higher accuracy can still accompany more false positives.",
            "Balanced validation does not estimate precision at a low real-world phishing prevalence. Compare FPR, recall, AP and confusion counts as well as accuracy.",
            "The nested sensitivity subset and previously inspected groups are diagnostics only; they do not select the candidate or become classifier inputs.",
            "Removing five size measurements does not remove all page-complexity or collection signals in remaining counts/ratios.",
            "Generic field cues can occur on benign login, contact, or document-viewing pages. No single feature forces a phishing decision; page labels are retained.",
            "HTML/JS caps and static association limits remain. The features do not observe CSS visibility, runtime JavaScript, final redirects or external script contents.",
            "Baseline validation predictions are independently checked; baseline training metrics are retained from its verified report without reloading/retraining those models.",
            "Completed model outputs are reused only with matching configuration and hashes. An interrupted fit restarts that model."]}
    report_path = args.output_dir / "form_context_training_report.json"
    save_json(report_path, report)
    print("Comparison at threshold 0.50:", flush=True)
    for result in all_results:
        m = result["metrics"]["main_validation"]
        print(f"  {result['model_name']} ({result['feature_count']} features): accuracy={100*m['accuracy']:.2f}%; recall={100*m['recall']:.2f}%; FPR={100*m['false_positive_rate']:.2f}%; AP={m['average_precision']:.4f}; FP={m['confusion_matrix']['fp']:,}; FN={m['confusion_matrix']['fn']:,}.", flush=True)
    print(f"Saved comparison report: {report_path.resolve()}", flush=True)
    print(f"Leading candidate by main validation ranking: {report['leading_validation_candidate']}", flush=True)
    print("Form-context model training complete. Official test not evaluated; operational threshold still pending.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ImportError) as error:
        raise SystemExit(f"Form-context training stopped: {error}. A completed report is valid after the final completion message.")
    except MemoryError:
        raise SystemExit("Not enough RAM. Close memory-heavy apps and rerun the same command; completed models can be reused.")
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Rerun the same command; completed models can be reused and an unfinished fit restarts.")
