"""Compare two fixed experiments with the completed 106-feature model.

Run after scripts 16 and 18, using the same virtual environment. Script 17
supplies the checked threshold mathematics. No new packages are required.

Experiment A removes 16 page-size/general-count features and keeps 90 features.
Experiment B keeps all 106 features but changes num_leaves from 31 to 15 and
min_child_samples from 100 to 500. Other training parameters stay unchanged.
All rows, labels, domain splits, extraction limits and the 0.50 comparison
threshold stay unchanged. Reuses the current model's verified predictions.
Fits only the two new models, in a separate output folder. A descriptive 1% FPR
operating point is also reported; it is not installed into any model.

No raw Parquet, live websites, browser JavaScript or official test is accessed.
"""

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

VERSION = "1.0"
SEED = 42
THRESHOLD = 0.50
CURRENT = "lightgbm_context_all_features"
REDUCED = "lightgbm_context_without_structure_features"
REGULARIZED = "lightgbm_context_smaller_trees"
TRAINER_SHA = "d685b0edf2ba248045ea39dfe9821ef6f9f477750468ed93fb2696a289096524"
THRESHOLD_HELPER_SHA = "e88859af65d1d95e9d1400c0f40ad8abd276259d73b7ef06d2852d743747aabe"
REMOVED_FEATURES = (
    "html_character_count", "tag_count", "text_character_count", "title_character_count",
    "js_analysed_character_count", "anchor_count", "anchor_http_count",
    "anchor_fragment_or_empty_count", "anchor_mailto_count", "image_count",
    "script_count", "script_external_count", "script_inline_executable_count",
    "script_non_executable_data_count", "script_async_count", "script_defer_count")


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


def load_helper(path, expected_sha, name):
    if not path.is_file() or digest_file(path) != expected_sha:
        raise ValueError(f"Keep unchanged {path.name} beside this script, or specify its script-path option.")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def one_result(report, name):
    found = [r for r in report["model_results"] if r["model_name"] == name]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one result for {name}.")
    return found[0]


def checked_current(args, trainer, paths, hashes, context, schema, baseline):
    extra = {"current_training_report": args.training_dir / "form_context_training_report.json",
        "current_training_config": args.training_dir / "run_config.json",
        "current_predictions": args.training_dir / CURRENT / "validation_predictions.csv",
        "current_bundle": args.training_dir / CURRENT / "model_bundle.joblib",
        "error_review": args.review_dir / "context_error_review_report.json",
        "trainer_helper": args.trainer_script, "threshold_helper": args.threshold_script,
        "review_script": args.review_script}
    for key, path in extra.items():
        if not path.is_file():
            raise ValueError(f"Missing input: {path}. Complete scripts 16 and 18 first.")
        paths[key] = path
        hashes[key] = digest_file(path)
    run, config, review = (load_json(extra[k]) for k in
        ("current_training_report", "current_training_config", "error_review"))
    if (run.get("status") != "form_context_model_training_complete"
            or review.get("status") != "context_error_review_complete"):
        raise ValueError("Use completed script 16 and 18 reports.")
    if config != run["run_config"] or hashes["current_training_config"] != run["run_config_sha256"]:
        raise ValueError("The current training configuration differs from its report.")
    if config["training_code_sha256"] != hashes["trainer_helper"] or review["review_code_sha256"] != hashes["review_script"]:
        raise ValueError("The existing training/review code changed after its completed run.")
    for key, value in config["input_hashes"].items():
        if hashes.get(key) != value:
            raise ValueError(f"An original input changed after script 16: {key}.")
    for report in (run, review):
        for key in ("official_test_evaluated", "labels_changed", "balanced_splits_changed", "websites_requested", "javascript_executed"):
            if report.get(key) is not False:
                raise ValueError(f"Unexpected development report policy: {key}.")
    if (run.get("original_features_unchanged") is not True
            or run.get("original_model_inputs_verified_at_training_precision") is not True
            or run.get("operational_threshold_selected") is not False
            or review.get("threshold_changed") is not False or review["threshold"] != THRESHOLD
            or run["shared_train_validation_record_ids"] != 0 or run["shared_train_validation_site_groups"] != 0):
        raise ValueError("Unexpected feature, threshold or split-preservation contract.")
    result = one_result(run, CURRENT)
    if (result["feature_names"] != schema["feature_names"] or result["feature_count"] != 106
            or result["run_config_sha256"] != run["run_config_sha256"]
            or result["outputs"]["validation_predictions.csv"]["sha256"] != hashes["current_predictions"]
            or result["outputs"]["model_bundle.joblib"]["sha256"] != hashes["current_bundle"]):
        raise ValueError("The existing reference model/predictions differ from script 16's report.")
    for key, source_key in (("training_report", "current_training_report"), ("training_config", "current_training_config"),
            ("feature_report", "context_report"), ("schema", "context_schema"), ("validation", "context_validation"),
            ("new_scores", "current_predictions"), ("new_bundle", "current_bundle")):
        if review["input_hashes"][key] != hashes[source_key]:
            raise ValueError("Script 18 reviewed different files/models from the current experiment.")
    if review["new_validation_metrics"] != result["metrics"]["main_validation"]:
        raise ValueError("The error review and reference training metrics differ.")
    if (run["training_counts"] != context["extracted_label_counts"]["train"]
            or run["main_validation_counts"] != context["extracted_label_counts"]["validation"]
            or run["validation_sensitivity_counts"] != context["validation_sensitivity_counts"]):
        raise ValueError("The current balanced row counts differ from extraction.")
    if (config["lightgbm_hyperparameters"] != baseline["run_config"]["lightgbm_hyperparameters"]
            or config["early_stopping"] != baseline["run_config"]["early_stopping"]
            or config["threshold"] != THRESHOLD or config["seed"] != SEED):
        raise ValueError("Unexpected original training/early-stopping settings.")
    return run, review, result


def write_config(path, config):
    if path.exists():
        if load_json(path) != config:
            raise ValueError("An existing output uses another configuration. Choose a new --output-dir.")
    else:
        save_json(path, config)
    return digest_file(path)


def experiment_plan(schema, parameters):
    names = schema["feature_names"]
    if len(REMOVED_FEATURES) != 16 or not set(REMOVED_FEATURES).issubset(names):
        raise ValueError("The expected 16 structure features are missing.")
    reduced = [name for name in names if name not in REMOVED_FEATURES]
    if len(reduced) != 90 or len([name for name in reduced if name.startswith("context_")]) != 20:
        raise ValueError("The reduced model must retain 90 features including all 20 context features.")
    if parameters["num_leaves"] != 31 or parameters["min_child_samples"] != 100:
        raise ValueError("Expected the current 31-leaf / 100-sample starting parameters.")
    return {REDUCED: {"feature_names": reduced, "parameter_changes": {},
                     "removed_features": list(REMOVED_FEATURES)},
        REGULARIZED: {"feature_names": names, "parameter_changes": {"num_leaves": 15, "min_child_samples": 500},
                      "removed_features": []}}


def operating_points(y, scores, sensitivity, common, thresholds):
    np = common.np
    curve = thresholds.threshold_curve(y, scores)
    choice = next(item for item in thresholds.diagnostic_indices(curve) if item[2] == .01)
    threshold = float(curve["threshold"][choice[1]])
    budget = int(np.floor(.01 * np.count_nonzero(y == 0) + 1e-12))
    main = common.binary_metrics(y, scores, threshold=threshold)
    if main["confusion_matrix"]["fp"] > budget:
        raise ValueError("The descriptive main-validation 1% FPR budget was exceeded.")
    return {"initial_threshold_0_50": {"main_validation": common.binary_metrics(y, scores),
                "validation_sensitivity": common.binary_metrics(y[sensitivity], scores[sensitivity])},
        "illustrative_main_validation_fpr_at_most_1_percent": {"threshold": threshold,
            "main_validation_false_positive_budget": budget, "main_validation": main,
            "validation_sensitivity_same_main_selected_threshold": common.binary_metrics(y[sensitivity], scores[sensitivity], threshold=threshold),
            "installed_into_model": False}}


def write_comparison(path, results, points):
    columns = ["model", "features", "accuracy_at_0_50", "recall_at_0_50", "fpr_at_0_50", "fp_at_0_50", "fn_at_0_50",
               "average_precision", "roc_auc", "best_iteration", "illustrative_1_percent_fpr_threshold",
               "recall_at_illustrative_1_percent_fpr", "fp_at_illustrative_1_percent_fpr"]
    temporary = path.with_suffix(".csv.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in results:
                m = r["metrics"]["main_validation"]
                p = points[r["model_name"]]["illustrative_main_validation_fpr_at_most_1_percent"]
                writer.writerow({"model": r["model_name"], "features": r["feature_count"],
                    "accuracy_at_0_50": m["accuracy"], "recall_at_0_50": m["recall"], "fpr_at_0_50": m["false_positive_rate"],
                    "fp_at_0_50": m["confusion_matrix"]["fp"], "fn_at_0_50": m["confusion_matrix"]["fn"],
                    "average_precision": m["average_precision"], "roc_auc": m["roc_auc"],
                    "best_iteration": r["best_iteration_or_solver_iterations"],
                    "illustrative_1_percent_fpr_threshold": p["threshold"],
                    "recall_at_illustrative_1_percent_fpr": p["main_validation"]["recall"],
                    "fp_at_illustrative_1_percent_fpr": p["main_validation"]["confusion_matrix"]["fp"]})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    options = {"features-dir": "reports/features_context_v2", "base-features-dir": "reports/features",
        "baseline-dir": "reports/baseline_training_v1_1", "training-dir": "reports/form_context_training_v2",
        "review-dir": "reports/context_error_review_v2", "output-dir": "reports/structure_comparisons_v3",
        "baseline-script": "10_train_static_baselines.py", "base-extractor-script": "09_extract_static_features.py",
        "context-extractor-script": "15_extract_form_context.py", "trainer-script": "16_train_form_context_models.py",
        "threshold-script": "17_analyse_context_thresholds.py", "review-script": "18_review_context_errors.py"}
    for option, default in options.items():
        parser.add_argument("--" + option, type=Path, default=root / default)
    parser.add_argument("--threads", type=int, default=4, help="Match the completed baseline, normally 4.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive and match the completed baseline.")
    for folder in (args.features_dir, args.base_features_dir, args.baseline_dir, args.training_dir, args.review_dir):
        if args.output_dir.resolve().is_relative_to(folder.resolve()) or folder.resolve().is_relative_to(args.output_dir.resolve()):
            parser.error("Use a separate experiment folder outside all existing feature/model/review folders.")
    print(f"Structure comparison trainer v{VERSION}; checking the existing experiment...", flush=True)
    trainer = load_helper(args.trainer_script, TRAINER_SHA, "structure_comparison_trainer16")
    thresholds = load_helper(args.threshold_script, THRESHOLD_HELPER_SHA, "structure_comparison_threshold17")
    paths, hashes, base_schema, context, schema, baseline, _ = trainer.checked_inputs(args)
    run, review, reference = checked_current(args, trainer, paths, hashes, context, schema, baseline)
    common = trainer.load_reader(args.baseline_script, baseline["run_config"]["training_code_sha256"])
    np = common.np
    packages = {name: version(name) for name in trainer.PACKAGES}
    if packages != run["run_config"]["package_versions"] or platform.python_version() != run["run_config"]["python_version"]:
        raise ValueError("Use the unchanged virtual environment that trained script 16's models.")
    plan = experiment_plan(schema, run["run_config"]["lightgbm_hyperparameters"])
    config = {"script_version": VERSION, "training_code_sha256": digest_file(Path(__file__)),
        "input_hashes": hashes, "reference_model": CURRENT, "reference_run_config_sha256": run["run_config_sha256"],
        "seed": SEED, "threshold": THRESHOLD, "threads": args.threads,
        "reference_lightgbm_hyperparameters": run["run_config"]["lightgbm_hyperparameters"],
        "early_stopping": run["run_config"]["early_stopping"], "experiments": plan,
        "package_versions": packages, "python_version": platform.python_version(),
        "candidate_ranking": baseline["run_config"]["candidate_ranking"],
        "method": "Two prespecified development experiments: jointly remove 16 structure features; or reduce num_leaves and increase min_child_samples. Reuse verified reference predictions. Same balanced rows, extraction, seed and comparison threshold."}
    config_path = args.output_dir / "run_config.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not config_path.is_file():
        raise ValueError("Use an empty experiment folder or the existing folder with its matching run_config.json.")
    if config_path.is_file() and load_json(config_path) != config:
        raise ValueError("The experiment folder belongs to other inputs/settings. Choose a new --output-dir.")
    print("Loading the existing 106-feature training and validation CSVs...", flush=True)
    train = common.load_features(paths["context_train"], schema, context["extracted_label_counts"]["train"], "train", context)
    validation = common.load_features(paths["context_validation"], schema, context["extracted_label_counts"]["validation"], "validation", context)
    sensitivity = validation.metadata["in_validation_sensitivity_subset"] == 1
    if set(train.metadata["site_group"]) & set(validation.metadata["site_group"]) or set(train.metadata["record_id"]) & set(validation.metadata["record_id"]):
        raise ValueError("Training and validation share record IDs or site groups.")
    if common.counts(validation, sensitivity) != run["validation_sensitivity_counts"]:
        raise ValueError("The sensitivity subset changed.")
    if any(c["benign"] == 0 or c["benign"] != c["phish"] for c in
           (common.counts(train), common.counts(validation), common.counts(validation, sensitivity))):
        raise ValueError("Every train/validation subset must remain nonempty and 1:1 balanced.")
    trainer.check_csv_against_table(paths["context_sensitivity"], schema, validation, np.flatnonzero(sensitivity), common)
    scores = trainer.load_predictions(paths["current_predictions"], validation, common)
    for key, mask in trainer.evaluation_masks(validation, common).items():
        trainer.assert_metrics_match(common.binary_metrics(validation.y[mask], scores[mask]), reference["metrics"][key], np)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = write_config(config_path, config)
    results, score_vectors, per_model_configs = [], {CURRENT: scores}, {}
    current = copy.deepcopy(reference)
    current["reused_existing_model"] = True
    current["source_model_directory"] = str((args.training_dir / CURRENT).resolve())
    results.append(current)
    for name, experiment in plan.items():
        model_dir = args.output_dir / name
        model_dir.mkdir(exist_ok=True)
        model_config = {"script_version": VERSION, "training_code_sha256": config["training_code_sha256"],
            "parent_run_config_sha256": config_hash, "input_hashes": hashes, "model_name": name,
            "feature_names": experiment["feature_names"], "removed_features": experiment["removed_features"],
            "seed": SEED, "threshold": THRESHOLD, "threads": args.threads,
            "lightgbm_hyperparameters": {**config["reference_lightgbm_hyperparameters"], **experiment["parameter_changes"]},
            "early_stopping": config["early_stopping"], "package_versions": packages,
            "context_extractor_code_sha256": run["run_config"]["context_extractor_code_sha256"],
            "base_extractor_code_sha256": run["run_config"]["base_extractor_code_sha256"],
            "development_method": config["method"]}
        model_config_path = model_dir / "run_config.json"
        model_config_hash = write_config(model_config_path, model_config)
        per_model_configs[name] = {"config": model_config, "sha256": model_config_hash}
        result, score = trainer.train_one(name, experiment["feature_names"], args, schema, train, validation,
                                         model_config, model_config_hash, common)
        result = copy.deepcopy(result)
        result["reused_existing_model"] = False
        result["source_model_directory"] = str(model_dir.resolve())
        results.append(result)
        score_vectors[name] = score
    points = {name: operating_points(validation.y, score, sensitivity, common, thresholds)
              for name, score in score_vectors.items()}
    paired = {name: {"main_validation": trainer.paired_changes(validation.y, scores, score_vectors[name], common),
        "validation_sensitivity": trainer.paired_changes(validation.y[sensitivity], scores[sensitivity], score_vectors[name][sensitivity], common)} for name in plan}
    # These groups were chosen in development error review. They report diagnostics
    # at shared thresholds and are not used as features or for model ranking.
    group_names = sorted({r["site_group"] for kind in ("top_groups_by_remaining_fp", "top_groups_by_remaining_fn")
                          for r in review["paired_group_summary"][kind][:5]})
    diagnostics = {}
    for group in group_names:
        mask = validation.metadata["site_group"] == group
        diagnostics[group] = {"rows": int(mask.sum()), "model_metrics_at_0_50": {
            name: common.binary_metrics(validation.y[mask], score[mask]) for name, score in score_vectors.items()},
            "model_metrics_at_same_main_selected_1_percent_threshold": {
            name: common.binary_metrics(validation.y[mask], score[mask],
                threshold=points[name]["illustrative_main_validation_fpr_at_most_1_percent"]["threshold"])
            for name, score in score_vectors.items()}}
    for key, path in paths.items():
        if digest_file(path) != hashes[key]:
            raise ValueError(f"An input changed during training: {path}. No new complete report was written.")
    if digest_file(config_path) != config_hash:
        raise ValueError("The experiment configuration changed during training.")
    for name, recorded in per_model_configs.items():
        if digest_file(args.output_dir / name / "run_config.json") != recorded["sha256"]:
            raise ValueError("A model configuration changed during training.")
    comparison = args.output_dir / "model_comparison.csv"
    write_comparison(comparison, results, points)
    ranked = sorted(results, key=common.ranking_key)
    report = {"status": "structure_model_comparison_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": VERSION, "run_config": config, "run_config_sha256": config_hash,
        "per_model_configurations": per_model_configs, "model_results": results,
        "training_counts": common.counts(train), "main_validation_counts": common.counts(validation),
        "validation_sensitivity_counts": common.counts(validation, sensitivity),
        "paired_changes_against_current_106_feature_model": paired, "operating_points": points,
        "previously_reviewed_group_diagnostics": diagnostics,
        "candidate_ranking_by_main_validation": [r["model_name"] for r in ranked],
        "leading_validation_candidate": ranked[0]["model_name"],
        "highest_accuracy_candidate_at_0_50": max(results, key=lambda r: r["metrics"]["main_validation"]["accuracy"])["model_name"],
        "lowest_false_positive_rate_candidate_at_0_50": min(results, key=lambda r: r["metrics"]["main_validation"]["false_positive_rate"])["model_name"],
        "comparison_csv_sha256": digest_file(comparison), "inputs_unchanged": True,
        "original_model_retrained": False, "original_features_unchanged": True, "labels_changed": False,
        "balanced_splits_changed": False, "shared_train_validation_record_ids": 0, "shared_train_validation_site_groups": 0,
        "official_test_evaluated": False, "operational_threshold_selected": False,
        "websites_requested": False, "javascript_executed": False,
        "limitations": ["Validation was already used for early stopping, error review and feature design. These are development experiments, not independent final performance estimates.",
            "The ablation jointly removes 16 features and changes fitted trees/column sampling. It does not isolate the causal importance of an individual feature.",
            "Remaining features still contain page-structure information; the reduced model is not a purely behavioral detector.",
            "The smaller-tree experiment changes two parameters together. Either experiment may underfit or perform worse than the current model.",
            "The 1% FPR thresholds are selected using main validation scores. Their empirical limit is not guaranteed on sensitivity subsets, particular domains or future data.",
            "Main validation ranking follows the existing AP/ROC/FPR rule. The report separately identifies highest accuracy and lowest FPR at 0.50; no model/threshold is installed automatically.",
            "Labels, unavailable-page captures and difficult domains are retained. A static score or error review cannot prove a label is wrong.",
            "HTML/JS analysis caps, external-script reference-only policy and static parsing limits remain. Browser behavior and final redirected URLs are not observed.",
            "Completed models are reused only with matching configurations and output hashes; an interrupted fit restarts that model."]}
    path = args.output_dir / "structure_comparison_report.json"
    save_json(path, report)
    print("Comparison at threshold 0.50:", flush=True)
    for result in results:
        m = result["metrics"]["main_validation"]
        low = points[result["model_name"]]["illustrative_main_validation_fpr_at_most_1_percent"]["main_validation"]
        print(f"  {result['model_name']} ({result['feature_count']} features): accuracy={100*m['accuracy']:.2f}%; recall={100*m['recall']:.2f}%; FPR={100*m['false_positive_rate']:.2f}%; FP={m['confusion_matrix']['fp']:,}; FN={m['confusion_matrix']['fn']:,}; recall at illustrative 1% FPR={100*low['recall']:.2f}%.", flush=True)
    print(f"Saved report: {path.resolve()}", flush=True)
    print(f"Leading candidate by the existing validation ranking: {report['leading_validation_candidate']}", flush=True)
    print("Structure model comparison complete. Official test untouched; operational threshold still pending.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ImportError) as error:
        raise SystemExit(f"Structure comparison stopped: {error}. A report is complete only after the final completion message.")
    except MemoryError:
        raise SystemExit("Insufficient RAM. Close memory-heavy apps and rerun; completed experiment models can be reused.")
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Rerun the same command; completed models can be reused and an unfinished fit restarts.")
