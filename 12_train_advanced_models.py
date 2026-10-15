"""Compare CatBoost and XGBoost with the completed script 10 baselines.

Run beside the unchanged scripts 09 and 10 in HTML_JS_PhishingDetector.
Requires NumPy, scikit-learn, LightGBM, joblib, threadpoolctl, CatBoost >=1.2
and XGBoost >=2.0. CatBoost also requires Pandas through its Python package.
The feature reader uses Python csv + NumPy, reusing script 10 v1.1's checked
reader and metrics. No feature extraction, websites, JS execution or official
test evaluation is performed. CPU training is the default and only mode.

Four comparisons: each model family with all features and with the five size
features omitted. Fixed starting parameters are not a hyperparameter search.
Completed models are reused only with matching inputs/configuration/outputs;
an interrupted fit restarts. Validation is development data, not a final test.
"""

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

VERSION = "1.0"
SEED = 42
THRESHOLD = 0.50
FAMILIES = ("xgboost", "catboost")
SIZE_FEATURES = ("html_character_count", "tag_count", "text_character_count",
                 "title_character_count", "js_analysed_character_count")


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


def load_checked_reader(path, expected_sha):
    if digest_file(path) != expected_sha:
        raise ValueError("Script 10 differs from the code recorded in your baseline report. Keep the matching v1.1 script.")
    spec = importlib.util.spec_from_file_location("static_baseline_reader_v1_1", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.VERSION != "1.1" or tuple(module.SIZE_FEATURES) != SIZE_FEATURES or module.THRESHOLD != THRESHOLD:
        raise ValueError("This comparison requires the unchanged script 10 v1.1 feature contract.")
    return module


def xgb_parameters(args):
    return {"objective": "binary:logistic", "eval_metric": "logloss", "booster": "gbtree",
            "tree_method": "hist", "device": "cpu", "eta": 0.03, "max_depth": 6,
            "min_child_weight": 10.0, "subsample": 0.8, "colsample_bytree": 0.8,
            "alpha": 0.1, "lambda": 1.0, "max_bin": 256, "seed": SEED,
            "nthread": args.threads, "scale_pos_weight": 1.0, "verbosity": 1}


def cat_parameters(args):
    return {"loss_function": "Logloss", "eval_metric": "Logloss", "iterations": args.trees,
            "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 3.0,
            "boosting_type": "Plain", "bootstrap_type": "Bernoulli", "subsample": 0.8,
            "rsm": 0.8, "border_count": 128, "random_seed": SEED,
            "thread_count": args.threads, "task_type": "CPU", "use_best_model": True,
            "allow_writing_files": False}


def preflight(families):
    packages = {}
    for family in families:
        try:
            packages[family] = importlib.import_module(family)
        except (ImportError, OSError) as error:
            extra = " CatBoost's Python package also needs Pandas." if family == "catboost" else ""
            raise ValueError(f"Cannot import {family}: {error}.{extra} "
                             "Use packages allowed in your working environment. If Windows reports Application Control, "
                             "keep the error and use an approved environment; do not change the security policy. "
                             "You can run XGBoost separately with --families xgboost and a separate --output-dir.") from error
        numbers = version(family).split(".")
        major, minor = int(numbers[0]), int(numbers[1])
        if (family == "xgboost" and major < 2) or (family == "catboost" and (major, minor) < (1, 2)):
            raise ValueError(f"Update {family} to a supported version (XGBoost >=2.0, CatBoost >=1.2).")
    return packages


def fit_family(family, packages, names, x_train, y_train, x_validation, y_validation, args):
    history = {}
    if family == "xgboost":
        xgb = packages[family]
        # Quantization borders come from training; validation uses that matrix
        # as its reference, never independently defines the feature bins.
        d_train = xgb.QuantileDMatrix(x_train, label=y_train, feature_names=names,
                                      max_bin=256, nthread=args.threads)
        d_validation = xgb.QuantileDMatrix(x_validation, label=y_validation, feature_names=names,
                                           max_bin=256, ref=d_train, nthread=args.threads)
        fitted = xgb.train(xgb_parameters(args), d_train, num_boost_round=args.trees,
                           evals=[(d_validation, "main_validation")], evals_result=history,
                           early_stopping_rounds=args.early_stopping_rounds, verbose_eval=100)
        best_zero_based = int(fitted.best_iteration)
        rounds_run = int(fitted.num_boosted_rounds())
        # Native xgb.train retains the last model. Slice explicitly so saved
        # predictions and feature importance use only the chosen best trees.
        model = fitted[:best_zero_based + 1]
        p_train = model.predict(d_train, validate_features=True)
        p_validation = model.predict(d_validation, validate_features=True)
    else:
        cat = packages[family]
        pool_train = cat.Pool(x_train, label=y_train, feature_names=names)
        pool_validation = cat.Pool(x_validation, label=y_validation, feature_names=names)
        model = cat.CatBoostClassifier(**cat_parameters(args))
        model.fit(pool_train, eval_set=pool_validation,
                  early_stopping_rounds=args.early_stopping_rounds, verbose=100)
        if list(model.classes_) != [0, 1] or list(model.feature_names_) != names:
            raise ValueError("CatBoost label mapping or feature order differs from the saved contract.")
        history = model.get_evals_result()
        best_zero_based = int(model.get_best_iteration())
        rounds_run = len(history["validation"]["Logloss"])
        if best_zero_based < 0 or int(model.tree_count_) != best_zero_based + 1:
            raise ValueError("CatBoost did not retain the chosen best validation iteration.")
        p_train = model.predict_proba(pool_train, thread_count=args.threads)[:, 1]
        p_validation = model.predict_proba(pool_validation, thread_count=args.threads)[:, 1]
    return model, p_train, p_validation, history, best_zero_based + 1, rounds_run


def positive_scores(model, family, x, names, threads, packages):
    """Reload/inference helper: supply exactly the bundle's saved feature order."""
    if family == "xgboost":
        matrix = packages[family].DMatrix(x, feature_names=names, nthread=threads)
        return model.predict(matrix, validate_features=True)
    pool = packages[family].Pool(x, feature_names=names)
    return model.predict_proba(pool, thread_count=threads)[:, 1]


def write_importance(path, model, family, names):
    if family == "xgboost":
        gain, weight = model.get_score(importance_type="total_gain"), model.get_score(importance_type="weight")
        header = ["feature", "total_split_gain", "split_count"]
        rows = [(name, float(gain.get(name, 0.0)), int(weight.get(name, 0))) for name in names]
    else:
        header = ["feature", "prediction_values_change_importance"]
        values = model.get_feature_importance(type="PredictionValuesChange")
        if len(values) != len(names):
            raise ValueError("Unexpected CatBoost feature importance width.")
        rows = [(name, float(value)) for name, value in zip(names, values)]
    temporary = path.with_suffix(".csv.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(sorted(rows, key=lambda r: (-r[1], r[0])))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train_one(name, family, args, packages, common, schema, train, validation, config, config_hash):
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits

    model_dir = args.output_dir / name
    model_dir.mkdir(exist_ok=True)
    result_path = model_dir / "model_result.json"
    if result_path.exists():
        cached = load_json(result_path)
        if cached.get("run_config_sha256") != config_hash:
            raise ValueError("Model result belongs to another configuration. Use a new --output-dir.")
        if cached.get("outputs") and all((model_dir / file).is_file() and digest_file(model_dir / file) == info["sha256"]
                                        for file, info in cached["outputs"].items()):
            print(f"Reusing completed model: {name}", flush=True)
            return cached
        print(f"Incomplete or changed outputs: retraining {name}.", flush=True)
    names = [f for f in schema["feature_names"] if not name.endswith("without_size_features") or f not in SIZE_FEATURES]
    x_train, x_validation = train.feature_matrix(names), validation.feature_matrix(names)
    print(f"Training {name}: {len(train):,} rows, {len(names)} features, CPU threads={args.threads}.", flush=True)
    started = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        model, p_train, p_validation, history, best_rounds, rounds_run = fit_family(
            family, packages, names, x_train, train.y, x_validation, validation.y, args)
    fit_and_score_seconds = time.perf_counter() - started
    p_train, p_validation = np.asarray(p_train, dtype=np.float64), np.asarray(p_validation, dtype=np.float64)
    sensitivity = validation.metadata["in_validation_sensitivity_subset"] == 1
    flagged = validation.metadata["sampled_similarity_candidate"] == 1
    capped = (validation.quality["html_analysis_truncated"] > 0) | (validation.quality["js_analysis_truncated"] > 0)
    metrics = {"training": common.binary_metrics(train.y, p_train),
               "main_validation": common.binary_metrics(validation.y, p_validation),
               "validation_sensitivity": common.binary_metrics(validation.y[sensitivity], p_validation[sensitivity]),
               "validation_flagged_similarity_candidates": common.binary_metrics(validation.y[flagged], p_validation[flagged]),
               "validation_reached_an_analysis_cap": common.binary_metrics(validation.y[capped], p_validation[capped]),
               "validation_did_not_reach_analysis_caps": common.binary_metrics(validation.y[~capped], p_validation[~capped])}
    bundle = {"bundle_version": VERSION, "model_name": name, "model_family": family, "estimator": model,
              "feature_names": names, "label_mapping": {"benign": 0, "phish": 1}, "threshold": THRESHOLD,
              "feature_extraction_contract": schema, "extractor_code_sha256": config["extractor_code_sha256"],
              "run_config_sha256": config_hash, "package_versions": config["package_versions"],
              "best_boosting_rounds": best_rounds, "prediction_policy": "saved model contains only the best validation trees",
              "training_label_counts": common.counts(train), "development_stage": "not_final_tested_or_calibrated"}
    model_path = model_dir / "model_bundle.joblib"
    temporary = model_path.with_suffix(".joblib.tmp")
    try:
        joblib.dump(bundle, temporary, compress=3)
        temporary.replace(model_path)
    finally:
        temporary.unlink(missing_ok=True)
    native_path = model_dir / ("model.ubj" if family == "xgboost" else "model.cbm")
    temporary_native = native_path.with_name("temporary" + native_path.suffix)
    try:
        model.save_model(str(temporary_native))
        temporary_native.replace(native_path)
    finally:
        temporary_native.unlink(missing_ok=True)
    predictions_path, importance_path, curve_path = (model_dir / filename for filename in
                                                    ("validation_predictions.csv", "feature_importance.csv", "learning_curve.json"))
    common.write_predictions(predictions_path, validation, validation.y, p_validation)
    write_importance(importance_path, model, family, names)
    save_json(curve_path, history)
    result = {"model_name": name, "model_family": family, "run_config_sha256": config_hash,
              "feature_count": len(names), "feature_names": names, "best_boosting_rounds": best_rounds,
              "rounds_run_before_best_model_selection": rounds_run,
              "fit_and_score_seconds": fit_and_score_seconds,
              "timing_note": "Includes data-matrix/pool construction, fit and training/validation prediction; not comparable with script 10 fit_seconds.",
              "eligible_for_validation_ranking": True, "metrics": metrics,
              "feature_importance_note": "Model-specific association measure; not causal evidence or directly comparable between families.",
              "outputs": {p.name: {"sha256": digest_file(p)} for p in
                          (model_path, native_path, predictions_path, importance_path, curve_path)}}
    save_json(result_path, result)
    m = metrics["main_validation"]
    print(f"  Main validation: accuracy={m['accuracy']:.4f}; F1={m['f1']:.4f}; "
          f"FPR={m['false_positive_rate']:.4f}; AP={m['average_precision']:.4f}.", flush=True)
    return result


def check_inputs(args):
    paths = {"extraction_report": args.features_dir / "feature_extraction_report.json",
             "schema": args.features_dir / "static_feature_schema.json",
             "train": args.features_dir / "train_static_features.csv",
             "validation": args.features_dir / "validation_static_features.csv",
             "sensitivity": args.features_dir / "validation_sensitivity_static_features.csv",
             "extractor": args.extractor_script, "baseline_script": args.baseline_script,
             "baseline_report": args.baseline_report}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Missing input: {path}. Keep the completed scripts 09/10 and their reports.")
    hashes = {name: digest_file(path) for name, path in paths.items()}
    baseline, extraction, schema = (load_json(paths[name]) for name in ("baseline_report", "extraction_report", "schema"))
    if baseline.get("status") != "static_baseline_training_complete" or baseline.get("script_version") != "1.1":
        raise ValueError("Use your completed script 10 v1.1 baseline report.")
    if extraction.get("status") != "static_feature_extraction_complete":
        raise ValueError("Feature extraction must be complete.")
    for key in ("extraction_report", "schema", "train", "validation", "sensitivity", "extractor"):
        if hashes[key] != baseline["run_config"]["input_hashes"][key]:
            raise ValueError(f"Input differs from the LightGBM experiment: {paths[key].name}.")
    if hashes["baseline_script"] != baseline["run_config"]["training_code_sha256"]:
        raise ValueError("Script 10 differs from the code recorded in your baseline report.")
    for key in ("schema", "train", "validation", "sensitivity"):
        if hashes[key] != extraction["outputs"][paths[key].name]["sha256"]:
            raise ValueError(f"Feature output changed after extraction: {paths[key].name}.")
    if hashes["extractor"] != extraction["inputs"]["code_sha256"] or schema["implementation_code_sha256"] != hashes["extractor"]:
        raise ValueError("Keep the matching script 09 code that produced your features.")
    names = schema["feature_names"]
    if len(names) != len(set(names)) or names != extraction["inputs"]["feature_names"] or len(names) != extraction["feature_count"]:
        raise ValueError("Feature names or order differ from the extraction report.")
    if len(names) != baseline["feature_count"] or set(names).intersection(schema["metadata_columns"] + schema["quality_columns_not_classifier_inputs"]):
        raise ValueError("Unexpected feature schema; metadata and labels must stay outside classifier inputs.")
    if not set(SIZE_FEATURES).issubset(names) or schema["parameters"] != extraction["inputs"]["parameters"] or schema["external_js_policy"] != "references_only_no_download":
        raise ValueError("Unexpected script 09 extraction contract.")
    if baseline["run_config"]["seed"] != SEED or baseline["run_config"]["threshold"] != THRESHOLD:
        raise ValueError("Baseline seed or initial comparison threshold differs.")
    return paths, hashes, baseline, extraction, schema


def write_comparison(path, results):
    fields = ["model", "evaluation_set", "rows", "feature_count", "accuracy", "precision", "recall", "f1",
              "false_positive_rate", "roc_auc", "average_precision", "tn", "fp", "fn", "tp"]
    temporary = path.with_suffix(".csv.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for result in results:
                for evaluation_set in ("main_validation", "validation_sensitivity"):
                    m = result["metrics"][evaluation_set]
                    writer.writerow({"model": result["model_name"], "evaluation_set": evaluation_set,
                                     "rows": m["rows"], "feature_count": result["feature_count"],
                                     **{k: m[k] for k in fields[4:11]}, **m["confusion_matrix"]})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=root / "reports/features")
    parser.add_argument("--baseline-report", type=Path, default=root / "reports/baseline_training_v1_1/baseline_training_report.json")
    parser.add_argument("--baseline-script", type=Path, default=root / "10_train_static_baselines.py")
    parser.add_argument("--extractor-script", type=Path, default=root / "09_extract_static_features.py")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/advanced_training_v1")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trees", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES),
                        help="Default: both. Choose xgboost alone if CatBoost dependencies are unavailable.")
    args = parser.parse_args()
    if args.threads < 1 or args.trees < 2 or args.early_stopping_rounds < 1 or len(args.families) != len(set(args.families)):
        parser.error("Use positive limits and unique model families.")
    # Stable order regardless of command-line family order.
    args.families = [family for family in FAMILIES if family in args.families]
    paths, hashes, baseline, extraction, schema = check_inputs(args)
    if args.output_dir.resolve() in (args.features_dir.resolve(), args.baseline_report.parent.resolve()):
        raise ValueError("Use a new output folder separate from features and baseline results.")
    common = load_checked_reader(args.baseline_script, hashes["baseline_script"])
    packages = preflight(args.families)
    import numpy as np
    model_names = [f"{family}_{suffix}" for family in args.families for suffix in ("all_features", "without_size_features")]
    dependencies = ["numpy", "scikit-learn", "lightgbm", "joblib", "threadpoolctl", *args.families]
    if "catboost" in args.families:
        dependencies.append("pandas")
    config = {"script_version": VERSION, "training_code_sha256": digest_file(Path(__file__)),
              "input_hashes": hashes, "extractor_code_sha256": hashes["extractor"], "seed": SEED,
              "threshold": THRESHOLD, "threads": args.threads, "model_names": model_names,
              "families": args.families, "device": "cpu", "trees_limit": args.trees,
              "xgboost_hyperparameters": xgb_parameters(args) if "xgboost" in args.families else None,
              "catboost_hyperparameters": cat_parameters(args) if "catboost" in args.families else None,
              "early_stopping": {"rounds": args.early_stopping_rounds, "metric": "logloss", "set": "main_validation"},
              "candidate_ranking": baseline["run_config"]["candidate_ranking"],
              "size_features_omitted_in_comparison": list(SIZE_FEATURES),
              "input_reader": "unchanged_script_10_v1_1_python_csv_numpy",
              "python_version": platform.python_version(), "platform": platform.platform(),
              "package_versions": {name: version(name) for name in dependencies}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    if config_path.exists() and load_json(config_path) != config:
        raise ValueError("This output folder contains another experiment. Use a new --output-dir.")
    if not config_path.exists():
        save_json(config_path, config)
    config_hash = digest_file(config_path)
    print(f"Advanced static trainer v{VERSION}; {len(model_names)} CPU comparisons; unchanged balanced feature files.", flush=True)
    print("Loading and checking training/validation features...", flush=True)
    train = common.load_features(paths["train"], schema, extraction["extracted_label_counts"]["train"], "train", extraction)
    validation = common.load_features(paths["validation"], schema, extraction["extracted_label_counts"]["validation"], "validation", extraction)
    if set(train.metadata["record_id"]) & set(validation.metadata["record_id"]) or set(train.metadata["site_group"]) & set(validation.metadata["site_group"]):
        raise ValueError("Training and validation share a record or site group.")
    sensitivity = validation.metadata["in_validation_sensitivity_subset"] == 1
    counts = {"training": common.counts(train), "validation": common.counts(validation),
              "sensitivity": common.counts(validation, sensitivity)}
    expected_counts = {"training": baseline["training_counts"], "validation": baseline["main_validation_counts"],
                       "sensitivity": baseline["validation_sensitivity_counts"]}
    if counts != expected_counts or counts["sensitivity"] != extraction["validation_sensitivity_counts"]:
        raise ValueError("Balanced split counts differ from the completed LightGBM experiment.")
    if any(c["benign"] != c["phish"] or min(c.values()) == 0 for c in counts.values()):
        raise ValueError("Training and validation subsets must remain nonempty and 1:1 balanced.")
    for result in baseline["model_results"]:
        if result["metrics"]["main_validation"]["rows"] != len(validation) or result["metrics"]["main_validation"]["threshold"] != THRESHOLD:
            raise ValueError("Baseline metrics differ from the comparison set or threshold.")
        expected_names = [name for name in schema["feature_names"] if result["model_name"] != "lightgbm_without_size_features" or name not in SIZE_FEATURES]
        if result["feature_names"] != expected_names:
            raise ValueError("Baseline feature order differs from the checked schema.")
    advanced = []
    for family in args.families:
        for suffix in ("all_features", "without_size_features"):
            name = f"{family}_{suffix}"
            advanced.append(train_one(name, family, args, packages, common, schema, train, validation, config, config_hash))
    for name, path in paths.items():
        if digest_file(path) != hashes[name]:
            raise ValueError("An input changed while training; no new completed comparison report was written.")
    combined = [{**r, "result_source": "completed_script_10_report"} for r in baseline["model_results"]]
    combined += [{**r, "result_source": "this_advanced_comparison"} for r in advanced]
    # Keep the already declared ranking: main AP, then ROC AUC, FPR at 0.50,
    # model name. Nested sensitivity/cap results never select the winner.
    ranked = sorted([r for r in combined if r["eligible_for_validation_ranking"]], key=common.ranking_key)
    comparison_path = args.output_dir / "model_comparison.csv"
    write_comparison(comparison_path, combined)
    report = {"status": "advanced_static_training_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "script_version": VERSION, "run_config": config, "run_config_sha256": config_hash,
              "training_counts": counts["training"], "main_validation_counts": counts["validation"],
              "validation_sensitivity_counts": counts["sensitivity"], "shared_train_validation_site_groups": 0,
              "feature_count": len(schema["feature_names"]),
              "training_constant_features": [name for name, same in zip(schema["feature_names"],
                  np.min(train.values, axis=0) == np.max(train.values, axis=0)) if same],
              "model_results": combined, "advanced_model_names": model_names,
              "candidate_ranking_by_main_validation": [r["model_name"] for r in ranked],
              "leading_validation_candidate": ranked[0]["model_name"] if ranked else None,
              "official_test_evaluated": False, "operational_threshold_selected": False,
              "comparison_csv_sha256": digest_file(comparison_path),
              "limitations": [
                  "Development validation was used for early stopping and candidate ranking; it is not an independent final test.",
                  "These fixed starting hyperparameters are not optimized, and advanced models are not guaranteed to outperform LightGBM.",
                  "The 0.50 threshold is for comparison; calibration and an agreed deployment false-positive target remain pending.",
                  "Balanced-validation precision does not directly represent precision at a lower deployment phishing prevalence.",
                  "Sensitivity is nested validation with different template composition; report it alongside the main set.",
                  "All models use the same static HTML/JS features; they do not execute scripts or observe runtime behavior.",
                  "Removing five size measurements does not remove all collection bias; other counts may retain size information.",
                  "Importance measures show model associations, not causal proof of malicious behavior.",
                  "Native model files require the accompanying feature order and extraction contract for inference; use the model bundle/metadata.",
                  "Joblib bundles should be loaded only from your own trusted runs.",
                  "Completed model outputs can be reused; interrupted fits restart. CPU mode was selected for this comparison."
              ]}
    report_path = args.output_dir / "advanced_training_report.json"
    save_json(report_path, report)
    print(f"Saved comparison report: {report_path.resolve()}", flush=True)
    print(f"Leading candidate by the declared main-validation ranking: {report['leading_validation_candidate']}", flush=True)
    print("Advanced static training complete. Official test not evaluated; operational threshold selection still pending.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ImportError) as error:
        raise SystemExit(f"Training stopped: {error}. A completed report is valid only after the final completion message.")
    except MemoryError:
        raise SystemExit("Not enough available RAM. Close memory-heavy applications and rerun the same command. Completed model outputs can be reused; the interrupted model restarts.")
