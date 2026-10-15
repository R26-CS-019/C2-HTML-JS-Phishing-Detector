"""Train and compare three static baselines from script 09 feature CSVs.

Requirements: lightgbm, scikit-learn, numpy, joblib, threadpoolctl.
Feature CSVs are read with Python's csv module into NumPy arrays; Pandas is
not required. A blocked optional Pandas installation can be uninstalled from
this project's virtual environment without changing the feature files.
Training and validation remain separate. No Parquet, websites, browser,
official test data or external JavaScript is loaded. Validation is development
data, not an unbiased final test. Completed model runs can be reused.
"""
import argparse
import csv
import hashlib
import inspect
import json
import platform
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

VERSION = "1.1"
THRESHOLD = 0.50
SEED = 42
LABELS = {"benign": 0, "phish": 1}
SIZE_FEATURES = ("html_character_count", "tag_count", "text_character_count",
                 "title_character_count", "js_analysed_character_count")
MODEL_NAMES = ("lightgbm_all_features", "logistic_regression", "lightgbm_without_size_features")


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


@dataclass
class FeatureTable:
    """Aligned numeric features and separate metadata, without Pandas."""

    feature_names: list[str]
    values: np.ndarray
    metadata: dict[str, np.ndarray]
    quality: dict[str, np.ndarray]
    y: np.ndarray

    def __len__(self):
        return len(self.y)

    def feature_matrix(self, names):
        if list(names) == self.feature_names:
            return self.values
        indices = [self.feature_names.index(name) for name in names]
        return np.ascontiguousarray(self.values[:, indices])


def counts(table, mask=None):
    labels = table.y if mask is None else table.y[mask]
    return {label: int(np.count_nonzero(labels == value)) for label, value in LABELS.items()}


def load_features(path, schema, expected_counts, split, extraction):
    features = schema["feature_names"]
    metadata = schema["metadata_columns"]
    quality = schema["quality_columns_not_classifier_inputs"]
    columns = [*metadata, *quality, *features]
    required_metadata = {"record_id", "label", "site_group", "source_file", "sampled_similarity_candidate", "in_validation_sensitivity_subset"}
    if not required_metadata.issubset(metadata):
        raise ValueError("Use the script 09 schema and CSV files.")
    if set(expected_counts) != set(LABELS) or any(type(n) is not int or n < 0 for n in expected_counts.values()):
        raise ValueError("Invalid label counts in the extraction report.")
    row_count = sum(expected_counts.values())
    flags = {"sampled_similarity_candidate", "in_validation_sensitivity_subset"}
    metadata_values = {name: np.empty(row_count, dtype=np.int8 if name in flags else object) for name in metadata}
    quality_values = {name: np.empty(row_count, dtype=np.int32) for name in quality}
    values = np.empty((row_count, len(features)), dtype=np.float32)
    labels = np.empty(row_count, dtype=np.int8)
    id_pattern = re.compile(r"[0-9a-fA-F]{64}")
    seen_ids = set()
    loaded = 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        if next(reader, None) != columns:
            raise ValueError(f"Unexpected CSV columns or order in {path.name}.")
        for row in reader:
            if loaded >= row_count or len(row) != len(columns):
                raise ValueError(f"Unexpected row count or CSV width in {path.name}, line {reader.line_num}.")
            for index, name in enumerate(metadata):
                value = row[index]
                if name in flags:
                    if value not in ("0", "1"):
                        raise ValueError(f"Invalid subset flag in {path.name}, line {reader.line_num}.")
                    metadata_values[name][loaded] = int(value)
                else:
                    metadata_values[name][loaded] = value
            label = str(metadata_values["label"][loaded])
            if label not in LABELS:
                raise ValueError(f"Unexpected label in {path.name}, line {reader.line_num}.")
            labels[loaded] = LABELS[label]
            record_id = str(metadata_values["record_id"][loaded])
            if not id_pattern.fullmatch(record_id) or record_id in seen_ids:
                raise ValueError(f"Missing, repeated or malformed record ID in {path.name}, line {reader.line_num}.")
            seen_ids.add(record_id)
            if any(metadata_values[name][loaded] == "" for name in ("site_group", "source_file")):
                raise ValueError(f"Missing grouping metadata in {path.name}, line {reader.line_num}.")
            for index, name in enumerate(quality, start=len(metadata)):
                try:
                    value = int(row[index])
                except ValueError as error:
                    raise ValueError(f"Invalid quality metadata in {path.name}, line {reader.line_num}.") from error
                if not 0 <= value <= np.iinfo(np.int32).max:
                    raise ValueError(f"Invalid quality metadata in {path.name}, line {reader.line_num}.")
                quality_values[name][loaded] = value
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    values[loaded] = np.asarray(row[len(metadata) + len(quality):], dtype=np.float32)
            except ValueError as error:
                raise ValueError(f"Nonnumeric feature in {path.name}, line {reader.line_num}.") from error
            loaded += 1
            if loaded % 50_000 == 0:
                print(f"  {split}: loaded {loaded:,}/{row_count:,} rows.", flush=True)
    table = FeatureTable(list(features), values, metadata_values, quality_values, labels)
    if loaded != row_count or counts(table) != expected_counts:
        raise ValueError(f"Label counts differ from the extraction report in {path.name}.")
    if split == "train" and any(metadata_values[name].any() for name in flags):
        raise ValueError("Validation flags unexpectedly present in training.")
    if ((metadata_values["in_validation_sensitivity_subset"] == 1) & (metadata_values["sampled_similarity_candidate"] == 1)).any():
        raise ValueError("A sensitivity record has a sampled similarity flag.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Features must be finite and nonnegative. Missing values are not imputed silently.")
    for label, value in LABELS.items():
        mask = labels == value
        reported = extraction["quality_summary_by_split_and_label"][split][label]
        for name in quality:
            if int(np.count_nonzero((quality_values[name] > 0) & mask)) != reported[name + "_rows"]:
                raise ValueError(f"Quality counts differ from the extraction report: {split}/{label}/{name}.")
    return table


def binary_metrics(y, probabilities, threshold=THRESHOLD):
    y = np.asarray(y, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(y) != len(probabilities) or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid probability vector.")
    if not len(y):
        return {"rows": 0, "status": "empty_subset", "threshold": threshold}
    predicted = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    both_labels = len(np.unique(y)) == 2
    return {
        "rows": len(y), "threshold": threshold, "positive_class": "phish",
        "benign_rows": tn + fp, "phishing_rows": tp + fn,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if both_labels else None,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "false_positive_rate": fp / (tn + fp) if tn + fp else None,
        "false_negative_rate": fn / (tp + fn) if tp + fn else None,
        "mcc": float(matthews_corrcoef(y, predicted)) if both_labels else None,
        "roc_auc": float(roc_auc_score(y, probabilities)) if both_labels else None,
        "average_precision": float(average_precision_score(y, probabilities)) if both_labels else None,
        "log_loss": float(log_loss(y, np.column_stack((1 - probabilities, probabilities)), labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "precision_zero_denominator_policy": "0 if no phishing predictions",
    }


def positive_probabilities(estimator, x):
    classes = list(estimator.classes_)
    if classes != [0, 1]:
        raise ValueError("Model class mapping differs from benign=0/phish=1.")
    if isinstance(estimator, lgb.LGBMClassifier) and isinstance(x, np.ndarray):
        if estimator.objective_ != "binary":
            raise ValueError("Expected a binary LightGBM objective.")
        # FeatureTable.feature_matrix supplies the saved feature order. The
        # native binary predictor accepts NumPy directly and uses the same
        # best iteration as predict_proba, without a DataFrame-name warning.
        return np.asarray(estimator.booster_.predict(x, num_iteration=estimator.best_iteration_ or None), dtype=np.float64)
    return estimator.predict_proba(x)[:, 1]


def lgb_parameters(args):
    return {"objective": "binary", "boosting_type": "gbdt", "n_estimators": args.trees,
            "learning_rate": 0.03, "num_leaves": 31, "max_depth": -1, "min_child_samples": 100,
            "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": SEED,
            "n_jobs": args.threads, "deterministic": True, "force_col_wise": True,
            "verbosity": -1, "importance_type": "gain", "device_type": "cpu"}


def make_model(name, args):
    if name == "logistic_regression":
        # Every transform is fitted on training only and stored in this bundle.
        return Pipeline([
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=args.logistic_max_iter,
                                               tol=1e-4, random_state=SEED)),
        ])
    return lgb.LGBMClassifier(**lgb_parameters(args))


def write_predictions(path, validation, y, probabilities):
    temporary = path.with_suffix(".csv.tmp")
    prediction = (probabilities >= THRESHOLD).astype(np.int8)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["record_id", "label", "site_group", "source_file", "sampled_similarity_candidate",
                             "in_validation_sensitivity_subset", "phishing_score", "prediction_at_0_50", "error_type"])
            metadata_names = ("record_id", "label", "site_group", "source_file", "sampled_similarity_candidate", "in_validation_sensitivity_subset")
            for i in range(len(validation)):
                row = [validation.metadata[name][i] for name in metadata_names]
                error = "false_positive" if y[i] == 0 and prediction[i] == 1 else "false_negative" if y[i] == 1 and prediction[i] == 0 else "correct"
                writer.writerow([*row, format(float(probabilities[i]), ".17g"), "phish" if prediction[i] else "benign", error])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_importance(path, estimator, names, model_name):
    if model_name == "logistic_regression":
        weights = estimator.named_steps["classifier"].coef_[0]
        rows = [(name, float(abs(weight)), float(weight)) for name, weight in zip(names, weights)]
        header = ("feature", "absolute_standardised_log1p_coefficient", "signed_coefficient")
    else:
        gain = estimator.booster_.feature_importance(importance_type="gain")
        splits = estimator.booster_.feature_importance(importance_type="split")
        rows = [(name, float(g), int(s)) for name, g, s in zip(names, gain, splits)]
        header = ("feature", "split_gain", "split_count")
    temporary = path.with_suffix(".csv.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(sorted(rows, key=lambda row: (-row[1], row[0])))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train_one(name, args, schema, train, validation, y_train, y_validation, config, config_hash):
    model_dir = args.output_dir / name
    model_dir.mkdir(exist_ok=True)
    result_path = model_dir / "model_result.json"
    if result_path.exists():
        cached = load_json(result_path)
        if cached.get("run_config_sha256") != config_hash:
            raise ValueError("Model result belongs to a different training configuration. Use a new --output-dir.")
        if all((model_dir / filename).is_file() and digest_file(model_dir / filename) == info["sha256"]
               for filename, info in cached["outputs"].items()):
            print(f"Reusing completed model: {name}", flush=True)
            return cached
        print(f"Incomplete or changed model output: retraining {name}.", flush=True)
    names = [n for n in schema["feature_names"] if name != "lightgbm_without_size_features" or n not in SIZE_FEATURES]
    x_train, x_validation = train.feature_matrix(names), validation.feature_matrix(names)
    model = make_model(name, args)
    print(f"Training {name}: {len(train):,} rows, {len(names)} features, CPU threads={args.threads}.", flush=True)
    started = time.perf_counter()
    history = {}
    fit_warnings = []
    with threadpool_limits(limits=args.threads):
        if name == "logistic_regression":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x_train, y_train)
            fit_warnings = [{"category": warning.category.__name__, "message": str(warning.message)} for warning in caught]
            converged = not any(issubclass(warning.category, ConvergenceWarning) for warning in caught)
            iterations = int(np.max(model.named_steps["classifier"].n_iter_))
        else:
            # LightGBM 4.7 introduced separate eval_X/eval_y arguments. Keep
            # older installations compatible without hiding library warnings.
            validation_args = {"eval_X": x_validation, "eval_y": y_validation} if "eval_X" in inspect.signature(model.fit).parameters else {"eval_set": [(x_validation, y_validation)]}
            model.fit(x_train, y_train, **validation_args, feature_name=names, eval_names=["main_validation"],
                      eval_metric="binary_logloss", callbacks=[
                          lgb.early_stopping(args.early_stopping_rounds, first_metric_only=True, verbose=True),
                          lgb.log_evaluation(100), lgb.record_evaluation(history)])
            converged = True
            iterations = int(model.best_iteration_ or model.n_estimators_)
        fit_seconds = time.perf_counter() - started
        p_validation = positive_probabilities(model, x_validation)
        p_train = positive_probabilities(model, x_train)
    sensitivity = validation.metadata["in_validation_sensitivity_subset"] == 1
    flagged = validation.metadata["sampled_similarity_candidate"] == 1
    capped = (validation.quality["html_analysis_truncated"] > 0) | (validation.quality["js_analysis_truncated"] > 0)
    metrics = {"training": binary_metrics(y_train, p_train),
               "main_validation": binary_metrics(y_validation, p_validation),
               "validation_sensitivity": binary_metrics(y_validation[sensitivity], p_validation[sensitivity]),
               "validation_flagged_similarity_candidates": binary_metrics(y_validation[flagged], p_validation[flagged]),
               "validation_reached_an_analysis_cap": binary_metrics(y_validation[capped], p_validation[capped]),
               "validation_did_not_reach_analysis_caps": binary_metrics(y_validation[~capped], p_validation[~capped])}
    bundle = {"bundle_version": VERSION, "model_name": name, "estimator": model,
              "feature_names": names, "label_mapping": LABELS, "threshold": THRESHOLD,
              "feature_extraction_contract": schema, "extractor_code_sha256": config["extractor_code_sha256"],
              "run_config_sha256": config_hash, "package_versions": config["package_versions"],
              "training_label_counts": counts(train), "development_stage": "baseline_not_final_tested_or_calibrated"}
    model_path = model_dir / "model_bundle.joblib"
    temporary = model_path.with_suffix(".joblib.tmp")
    try:
        joblib.dump(bundle, temporary, compress=3)
        temporary.replace(model_path)
    finally:
        temporary.unlink(missing_ok=True)
    predictions_path = model_dir / "validation_predictions.csv"
    importance_path = model_dir / "feature_importance.csv"
    curve_path = model_dir / "learning_curve.json"
    write_predictions(predictions_path, validation, y_validation, p_validation)
    write_importance(importance_path, model, names, name)
    save_json(curve_path, history)
    result = {"model_name": name, "run_config_sha256": config_hash,
              "feature_names": names, "feature_count": len(names), "fit_seconds": fit_seconds,
              "best_iteration_or_solver_iterations": iterations, "solver_converged": converged,
              "eligible_for_validation_ranking": converged, "fit_warnings": fit_warnings,
              "metrics": metrics,
              "outputs": {p.name: {"sha256": digest_file(p)} for p in (model_path, predictions_path, importance_path, curve_path)}}
    save_json(result_path, result)
    main_metrics = metrics["main_validation"]
    print(f"  Main validation: accuracy={main_metrics['accuracy']:.4f}; F1={main_metrics['f1']:.4f}; FPR={main_metrics['false_positive_rate']:.4f}; AP={main_metrics['average_precision']:.4f}.", flush=True)
    if not converged:
        print("  Logistic Regression did not converge; metrics are diagnostic and this run is excluded from candidate ranking.", flush=True)
    return result


def ranking_key(result):
    m = result["metrics"]["main_validation"]
    # Use ONLY main validation. Sensitivity/cap strata never select a winner.
    return (-m["average_precision"], -m["roc_auc"], m["false_positive_rate"], result["model_name"])


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=root / "reports/features")
    parser.add_argument("--extractor-script", type=Path, default=root / "09_extract_static_features.py")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/baseline_training")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trees", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--logistic-max-iter", type=int, default=1000)
    args = parser.parse_args()
    if args.threads < 1 or args.trees < 2 or args.early_stopping_rounds < 1 or args.logistic_max_iter < 1:
        parser.error("Threads, iteration limits and early stopping must be positive.")
    paths = {"extraction_report": args.features_dir / "feature_extraction_report.json",
             "schema": args.features_dir / "static_feature_schema.json",
             "train": args.features_dir / "train_static_features.csv",
             "validation": args.features_dir / "validation_static_features.csv",
             "sensitivity": args.features_dir / "validation_sensitivity_static_features.csv",
             "extractor": args.extractor_script}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Missing input: {path}. Complete script 09 first.")
    hashes = {name: digest_file(path) for name, path in paths.items()}
    extraction, schema = load_json(paths["extraction_report"]), load_json(paths["schema"])
    if extraction["status"] != "static_feature_extraction_complete":
        raise ValueError("Feature extraction must be complete.")
    for name in ("schema", "train", "validation", "sensitivity"):
        if hashes[name] != extraction["outputs"][paths[name].name]["sha256"]:
            raise ValueError(f"Feature output changed after extraction: {paths[name].name}.")
    if hashes["extractor"] != extraction["inputs"]["code_sha256"] or schema["implementation_code_sha256"] != hashes["extractor"]:
        raise ValueError("The extractor script differs from the code that produced the feature files. Keep the matching script for reproducible inference.")
    names = schema["feature_names"]
    if len(names) != len(set(names)) or names != extraction["inputs"]["feature_names"] or len(names) != extraction["feature_count"]:
        raise ValueError("Feature schema differs from the extraction report.")
    if set(names).intersection(schema["metadata_columns"] + schema["quality_columns_not_classifier_inputs"]):
        raise ValueError("Metadata or labels cannot become classifier features.")
    if not set(SIZE_FEATURES).issubset(names):
        raise ValueError("The size comparison requires the complete script 09 feature schema.")
    if schema["parameters"] != extraction["inputs"]["parameters"] or schema["external_js_policy"] != "references_only_no_download":
        raise ValueError("Unexpected extraction contract.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir.resolve() == args.features_dir.resolve():
        raise ValueError("Training outputs must use a separate folder from feature inputs.")
    config = {"script_version": VERSION, "training_code_sha256": digest_file(Path(__file__)), "input_hashes": hashes,
              "extractor_code_sha256": hashes["extractor"], "seed": SEED, "threshold": THRESHOLD,
              "threads": args.threads, "model_names": list(MODEL_NAMES),
              "lightgbm_hyperparameters": lgb_parameters(args),
              "early_stopping": {"rounds": args.early_stopping_rounds, "metric": "binary_logloss", "set": "main_validation"},
              "logistic_regression_hyperparameters": {"C": 1.0, "solver": "lbfgs", "max_iter": args.logistic_max_iter, "tol": 1e-4,
                                                    "random_state": SEED, "class_weight": None, "preprocessing": "log1p then training-fitted StandardScaler"},
              "size_features_omitted_in_comparison": list(SIZE_FEATURES),
              "candidate_ranking": "Main validation average precision descending; tie ROC AUC descending, FPR at 0.50 ascending, model name. Sensitivity sets are not used for selection.",
              "python_version": platform.python_version(),
              "input_reader": "python_csv_and_numpy",
              "package_versions": {name: version(name) for name in ("numpy", "scikit-learn", "lightgbm", "joblib", "threadpoolctl")}}
    config_path = args.output_dir / "run_config.json"
    if config_path.exists() and load_json(config_path) != config:
        raise ValueError("This output folder contains another configuration. Use a new --output-dir for a separate experiment.")
    if not config_path.exists():
        save_json(config_path, config)
    config_hash = digest_file(config_path)
    print(f"Static baseline trainer v{VERSION}; CSV reader: Python csv + NumPy.", flush=True)
    print("Loading and checking training/validation feature files...", flush=True)
    train = load_features(paths["train"], schema, extraction["extracted_label_counts"]["train"], "train", extraction)
    validation = load_features(paths["validation"], schema, extraction["extracted_label_counts"]["validation"], "validation", extraction)
    if set(train.metadata["site_group"]) & set(validation.metadata["site_group"]) or set(train.metadata["record_id"]) & set(validation.metadata["record_id"]):
        raise ValueError("Training and validation share a record ID or site group.")
    sensitivity_mask = validation.metadata["in_validation_sensitivity_subset"] == 1
    sensitivity_counts = counts(validation, sensitivity_mask)
    if sensitivity_counts != extraction["validation_sensitivity_counts"]:
        raise ValueError("Sensitivity validation counts differ from the feature report.")
    if any(c["benign"] != c["phish"] or min(c.values()) == 0 for c in (counts(train), counts(validation), sensitivity_counts)):
        raise ValueError("All training and validation subsets must be nonempty and 1:1 balanced.")
    y_train, y_validation = train.y, validation.y
    results = []
    for name in MODEL_NAMES:
        results.append(train_one(name, args, schema, train, validation, y_train, y_validation, config, config_hash))
    for name, path in paths.items():
        if digest_file(path) != hashes[name]:
            raise ValueError("An input changed while training. No new complete comparison report was written.")
    ranked = sorted([r for r in results if r["eligible_for_validation_ranking"]], key=ranking_key)
    comparison_path = args.output_dir / "model_comparison.csv"
    temporary = comparison_path.with_suffix(".csv.tmp")
    fields = ["model", "evaluation_set", "rows", "feature_count", "accuracy", "precision", "recall", "f1", "false_positive_rate",
              "roc_auc", "average_precision", "tn", "fp", "fn", "tp"]
    try:
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for result in results:
                for evaluation_set in ("main_validation", "validation_sensitivity"):
                    metrics = result["metrics"][evaluation_set]
                    writer.writerow({"model": result["model_name"], "evaluation_set": evaluation_set,
                                     "rows": metrics["rows"], "feature_count": result["feature_count"],
                                     **{key: metrics[key] for key in fields[4:11]}, **metrics["confusion_matrix"]})
        temporary.replace(comparison_path)
    finally:
        temporary.unlink(missing_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "static_baseline_training_complete",
              "script_version": VERSION, "run_config": config, "run_config_sha256": config_hash,
              "training_counts": counts(train), "main_validation_counts": counts(validation), "validation_sensitivity_counts": sensitivity_counts,
              "shared_train_validation_site_groups": 0, "feature_count": len(names),
              "training_constant_features": [n for n, same in zip(names, np.min(train.values, axis=0) == np.max(train.values, axis=0)) if same],
              "model_results": results, "candidate_ranking_by_main_validation": [r["model_name"] for r in ranked],
              "leading_validation_candidate": ranked[0]["model_name"] if ranked else None,
              "operational_threshold_selected": False, "official_test_evaluated": False,
              "balanced_accuracy_reference": 0.5,
              "comparison_csv_sha256": digest_file(comparison_path),
              "limitations": [
                  "These are development-validation metrics. Main validation is used for early stopping and candidate ranking; neither validation set is an independent final test.",
                  "Threshold 0.50 is an initial comparison setting, not a chosen operational threshold or a calibrated probability of phishing.",
                  "Balanced validation precision differs from precision at low real-world phishing prevalence. Report FPR, recall and a full official test evaluation later.",
                  "The secondary validation is nested and has changed template composition. Report both sets rather than selecting the higher score.",
                  "Five size measurements are omitted in one comparison, but remaining counts/ratios may still encode size or collection artefacts; this does not remove all dataset bias.",
                  "Feature importance and coefficients show model associations, not causal proof of harvesting or malicious behavior.",
                  "Models use static code indicators and referenced external-script properties, not executed JavaScript behavior.",
                  "Early stopping and fixed starting hyperparameters are not an exhaustive optimization. Additional tuning must continue to preserve the untouched final test.",
                  "Completed model checkpoints are reused only when configuration and output hashes match. A partially fitted model restarts after interruption."
              ]}
    report_path = args.output_dir / "baseline_training_report.json"
    save_json(report_path, report)
    print(f"Saved comparison report: {report_path.resolve()}", flush=True)
    print(f"Leading candidate by main validation ranking: {report['leading_validation_candidate']}", flush=True)
    print("Static baseline training complete. Official test not evaluated; operational threshold selection still pending.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError) as error:
        raise SystemExit(f"Training stopped: {error}. A completed report is valid only after the final completion message.")
    except MemoryError:
        raise SystemExit("Not enough available RAM. Close memory-heavy applications and rerun the same command. Completed model outputs can be reused; the interrupted model restarts.")
