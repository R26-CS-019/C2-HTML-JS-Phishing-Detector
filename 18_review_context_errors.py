"""Explain residual errors in the completed 106-feature form-context experiment.

Save beside scripts 10 and 16. Uses the existing virtual environment. Reads
validation features and the project's own, hash-verified 86/106-feature model
bundles. Reproduces their predictions before explaining selected pages with
native LightGBM contributions. No additional SHAP package is required.

Keeps the 0.50 development threshold. This is a diagnostic review, not training
or independent evaluation. No training rows, official test, raw HTML, live
websites or JavaScript are loaded or executed. Outputs use a separate folder.
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
MODEL = "lightgbm_context_all_features"
OLD_MODEL = "lightgbm_all_features"
THRESHOLD = 0.50
SEED = 42
BASELINE_SCRIPT_SHA = "e3b1b8f3c8400b55437246881f337ea57621aa63ee3b5e1e42f2740a736da6fb"
PACKAGES = ("numpy", "lightgbm", "scikit-learn", "joblib", "threadpoolctl")
SIZE_FEATURES = {"html_character_count", "tag_count", "text_character_count",
                 "title_character_count", "js_analysed_character_count"}
METADATA = ["record_id", "label", "site_group", "source_file",
            "sampled_similarity_candidate", "in_validation_sensitivity_subset"]


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


def load_reader(path):
    if digest_file(path) != BASELINE_SCRIPT_SHA:
        raise ValueError("Keep unchanged script 10 v1.1 beside this script, or specify --baseline-script.")
    spec = importlib.util.spec_from_file_location("context_review_baseline_reader", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.VERSION != "1.1" or module.SEED != SEED or module.THRESHOLD != THRESHOLD:
        raise ValueError("Unexpected baseline reader contract.")
    return module


def model_result(report, name):
    candidates = [r for r in report["model_results"] if r["model_name"] == name]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one completed result for {name}.")
    return candidates[0]


def checked_inputs(args):
    paths = {"training_report": args.training_dir / "form_context_training_report.json",
             "training_config": args.training_dir / "run_config.json",
             "baseline_report": args.baseline_dir / "baseline_training_report.json",
             "baseline_config": args.baseline_dir / "run_config.json",
             "feature_report": args.features_dir / "form_context_feature_report.json",
             "schema": args.features_dir / "static_feature_schema.json",
             "validation": args.features_dir / "validation_static_features.csv",
             "base_schema": args.base_features_dir / "static_feature_schema.json",
             "baseline_script": args.baseline_script, "training_script": args.training_script}
    for prefix, directory, name in (("new", args.training_dir, MODEL), ("old", args.baseline_dir, OLD_MODEL)):
        for key, filename in (("bundle", "model_bundle.joblib"), ("scores", "validation_predictions.csv"),
                              ("importance", "feature_importance.csv")):
            paths[prefix + "_" + key] = directory / name / filename
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Missing input: {path}. Keep the completed script 10, 15 and 16 outputs.")
    hashes = {key: digest_file(path) for key, path in paths.items()}
    run, config, baseline, features, schema, base_schema = (load_json(paths[k]) for k in
        ("training_report", "training_config", "baseline_report", "feature_report", "schema", "base_schema"))
    if run.get("status") != "form_context_model_training_complete" or features.get("status") != "form_context_feature_extraction_complete" or baseline.get("status") != "static_baseline_training_complete":
        raise ValueError("Use completed extraction and training reports.")
    if config != run["run_config"] or hashes["training_config"] != run["run_config_sha256"]:
        raise ValueError("The script 16 report and saved configuration differ.")
    if hashes["training_script"] != config["training_code_sha256"] or hashes["baseline_script"] != BASELINE_SCRIPT_SHA:
        raise ValueError("The existing training scripts differ from their recorded versions.")
    bc = baseline["run_config"]
    if (load_json(paths["baseline_config"]) != bc or hashes["baseline_config"] != baseline["run_config_sha256"]
            or config["baseline_run_config_sha256"] != baseline["run_config_sha256"]
            or bc["training_code_sha256"] != BASELINE_SCRIPT_SHA):
        raise ValueError("Baseline configuration provenance differs between experiments.")
    for key, input_key in (("feature_report", "context_report"), ("schema", "context_schema"),
                           ("validation", "context_validation"), ("baseline_report", "baseline_report"),
                           ("baseline_script", "baseline_script"), ("base_schema", "base_schema")):
        if hashes[key] != config["input_hashes"][input_key]:
            raise ValueError(f"An input changed after script 16: {paths[key]}.")
    for key in ("schema", "validation"):
        if hashes[key] != features["outputs"][paths[key].name]["sha256"]:
            raise ValueError("The feature files differ from script 15's report.")
    for report in (run, features):
        for key, expected in (("labels_changed", False), ("balanced_splits_changed", False),
                              ("official_test_evaluated", False), ("original_features_unchanged", True)):
            if report.get(key) is not expected:
                raise ValueError(f"Unexpected development report policy: {key}.")
    if (run.get("operational_threshold_selected") is not False
            or run.get("original_model_inputs_verified_at_training_precision") is not True
            or baseline.get("official_test_evaluated") is not False
            or config["threshold"] != THRESHOLD or config["seed"] != SEED
            or run["shared_train_validation_record_ids"] != 0 or run["shared_train_validation_site_groups"] != 0):
        raise ValueError("Unexpected development threshold, feature preservation or split contract.")
    names = schema["feature_names"]
    context_names = features["inputs"]["context_feature_names"]
    if (len(names) != 106 or len(set(names)) != 106 or len(base_schema["feature_names"]) != 86
            or len(context_names) != 20 or names != base_schema["feature_names"] + context_names
            or schema["metadata_columns"] != METADATA
            or schema["metadata_columns"] != base_schema["metadata_columns"]
            or set(names).intersection(METADATA + schema["quality_columns_not_classifier_inputs"])):
        raise ValueError("Expected the unchanged 86 features plus 20 form-context features, with metadata excluded.")
    new, old = model_result(run, MODEL), model_result(baseline, OLD_MODEL)
    if model_result(run, OLD_MODEL)["outputs"] != old["outputs"]:
        raise ValueError("The saved baseline model provenance differs between reports.")
    for prefix, result, expected_names, config_hash in (("new", new, names, run["run_config_sha256"]),
            ("old", old, base_schema["feature_names"], baseline["run_config_sha256"])):
        if result["feature_names"] != expected_names or result["feature_count"] != len(expected_names) or result["run_config_sha256"] != config_hash:
            raise ValueError(f"Unexpected {prefix} model feature/configuration contract.")
        for key in ("bundle", "scores", "importance"):
            p = paths[prefix + "_" + key]
            if hashes[prefix + "_" + key] != result["outputs"][p.name]["sha256"]:
                raise ValueError(f"Saved model output changed: {p}.")
    if (run["main_validation_counts"] != features["extracted_label_counts"]["validation"]
            or run["main_validation_counts"] != baseline["main_validation_counts"]
            or run["validation_sensitivity_counts"] != features["validation_sensitivity_counts"]):
        raise ValueError("Validation counts differ between experiments.")
    return paths, hashes, run, features, schema, base_schema, new, old


def read_scores(path, table, np):
    scores = np.empty(len(table), dtype=np.float64)
    loaded = 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != METADATA + ["phishing_score", "prediction_at_0_50", "error_type"]:
            raise ValueError("Unexpected prediction CSV columns.")
        for i, row in enumerate(reader):
            if i >= len(table) or None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed or extra prediction row.")
            if any(row[k] != str(table.metadata[k][i]) for k in METADATA):
                raise ValueError("Prediction IDs, labels or subset metadata differ from the feature file.")
            score = float(row["phishing_score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid saved phishing score.")
            predicted, label = score >= THRESHOLD, int(table.y[i])
            error = "false_positive" if not label and predicted else "false_negative" if label and not predicted else "correct"
            if row["prediction_at_0_50"] != ("phish" if predicted else "benign") or row["error_type"] != error:
                raise ValueError("Saved prediction/error columns disagree with the score.")
            scores[i] = score
            loaded += 1
    if loaded != len(table):
        raise ValueError("Missing prediction rows.")
    return scores


def read_importance(path, names):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["feature", "split_gain", "split_count"]:
            raise ValueError("Unexpected importance CSV columns.")
        rows = [{"feature": r["feature"], "split_gain": float(r["split_gain"]),
                 "split_count": int(r["split_count"])} for r in reader]
    if len(rows) != len(names) or {r["feature"] for r in rows} != set(names):
        raise ValueError("Importance features differ from the saved model.")
    if any(not math.isfinite(r["split_gain"]) or r["split_gain"] < 0 or r["split_count"] < 0 for r in rows):
        raise ValueError("Invalid feature importance value.")
    total = sum(r["split_gain"] for r in rows)
    if total <= 0:
        raise ValueError("The model has no positive split gain.")
    for row in rows:
        row["share_of_training_split_gain"] = row["split_gain"] / total
    return sorted(rows, key=lambda r: (-r["split_gain"], r["feature"]))


def verify_metrics(actual, expected):
    for key, value in actual.items():
        other = expected.get(key)
        if isinstance(value, float):
            if not isinstance(other, (int, float)) or not math.isclose(value, other, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Recomputed validation metrics differ: {key}.")
        elif value != other:
            raise ValueError(f"Recomputed validation metrics differ: {key}.")


def load_verified_model(path, result, schema, config_hash, common, table, scores, importance, threads):
    # checked_inputs binds these local project bundles to their completed reports.
    bundle = common.joblib.load(path)
    names = schema["feature_names"]
    if (bundle["model_name"] != result["model_name"] or bundle["feature_names"] != names
            or bundle["label_mapping"] != {"benign": 0, "phish": 1}
            or bundle["threshold"] != THRESHOLD or bundle["run_config_sha256"] != config_hash
            or bundle["feature_extraction_contract"] != schema):
        raise ValueError("A saved model bundle differs from its extraction/training contract.")
    model = bundle["estimator"]
    if not isinstance(model, common.lgb.LGBMClassifier) or model.objective_ != "binary" or list(model.classes_) != [0, 1] or model.booster_.feature_name() != names:
        raise ValueError("Unexpected binary LightGBM model/class/feature mapping.")
    if int(model.best_iteration_ or model.n_estimators_) != result["best_iteration_or_solver_iterations"]:
        raise ValueError("The saved best iteration differs from the training report.")
    np = common.np
    by_name = {r["feature"]: r for r in importance}
    for kind, field in (("gain", "split_gain"), ("split", "split_count")):
        actual = model.booster_.feature_importance(importance_type=kind, iteration=model.best_iteration_ or None)
        if not np.allclose(actual, [by_name[n][field] for n in names], rtol=1e-10, atol=1e-8):
            raise ValueError("Importance values do not match the saved model.")
    with common.threadpool_limits(limits=threads):
        reproduced = model.booster_.predict(table.feature_matrix(names), num_iteration=model.best_iteration_ or None, num_threads=threads)
    difference = float(np.max(np.abs(reproduced - scores)))
    if difference > 1e-9:
        raise ValueError("The saved model does not reproduce its validation predictions.")
    return model, difference


def describe(values, mask, np):
    x = values[mask]
    if not len(x):
        return {"rows": 0}
    q = np.quantile(x, [0, .1, .25, .5, .75, .9, 1])
    return {"rows": len(x), "mean": float(np.mean(x, dtype=np.float64)),
            "zero_fraction": float(np.mean(x == 0)),
            **{k: float(v) for k, v in zip(("min", "p10", "p25", "median", "p75", "p90", "max"), q)}}


def paired_group_summary(groups, y, old_scores, scores):
    records = {}
    transitions = Counter()
    for group, label, before, after in zip(groups, y, old_scores >= THRESHOLD, scores >= THRESHOLD):
        old_error = "fp" if not label and before else "fn" if label and not before else "correct"
        new_error = "fp" if not label and after else "fn" if label and not after else "correct"
        transitions[old_error + "_to_" + new_error] += 1
        r = records.setdefault(str(group), {"site_group": str(group), "benign_rows": 0, "phishing_rows": 0,
            "old_fp": 0, "new_fp": 0, "old_fn": 0, "new_fn": 0,
            "fp_fixed": 0, "fp_introduced": 0, "fn_fixed": 0, "fn_introduced": 0})
        r["benign_rows" if not label else "phishing_rows"] += 1
        for k, condition in (("old_fp", old_error == "fp"), ("new_fp", new_error == "fp"),
                ("old_fn", old_error == "fn"), ("new_fn", new_error == "fn"),
                ("fp_fixed", old_error == "fp" and new_error == "correct"),
                ("fp_introduced", old_error == "correct" and new_error == "fp"),
                ("fn_fixed", old_error == "fn" and new_error == "correct"),
                ("fn_introduced", old_error == "correct" and new_error == "fn")):
            r[k] += int(condition)
    rows = list(records.values())
    for r in rows:
        r["new_false_positive_rate"] = r["new_fp"] / r["benign_rows"] if r["benign_rows"] else None
        r["new_recall"] = 1 - r["new_fn"] / r["phishing_rows"] if r["phishing_rows"] else None
        r["net_fn_reduction"] = r["old_fn"] - r["new_fn"]
    return {"paired_transitions": dict(transitions), "unique_site_groups": len(rows),
        "top_groups_by_remaining_fp": sorted(rows, key=lambda r: (-r["new_fp"], r["site_group"]))[:20],
        "top_groups_by_remaining_fn": sorted(rows, key=lambda r: (-r["new_fn"], r["site_group"]))[:20],
        "top_groups_by_net_fn_reduction": sorted(rows, key=lambda r: (-r["net_fn_reduction"], r["site_group"]))[:20]}


def pick_examples(mask, scores, ids, groups, count, category, descending, diverse, np):
    candidates = np.flatnonzero(mask).tolist()
    candidates.sort(key=lambda i: ((-scores[i] if descending else scores[i]), str(ids[i])))
    count = min(count, len(candidates))
    extremes = (count + 1) // 2
    selected, seen_groups = [], set()
    def add(order, limit):
        for i in order:
            if len(selected) >= limit:
                break
            if i not in selected and (not diverse or str(groups[i]) not in seen_groups):
                selected.append(i)
                seen_groups.add(str(groups[i]))
    add(candidates, extremes)
    salt = int.from_bytes(hashlib.sha256(category.encode()).digest()[:4], "big")
    random_order = np.random.default_rng(SEED + salt).permutation(candidates).tolist()
    add(random_order, count)
    # If too few distinct groups remain, fill from the same seeded ordering.
    for i in random_order:
        if len(selected) == count:
            break
        if i not in selected:
            selected.append(i)
    return selected


def historical_cases(features, table, old_scores, names, np):
    lookup = {str(identity): i for i, identity in enumerate(table.metadata["record_id"])}
    selected, categories = [], {}
    for page in features["selected_review_pages_context"]:
        rid = page["metadata"]["record_id"]
        if rid not in lookup or lookup[rid] in categories:
            raise ValueError("A previously inspected page is missing or repeated in validation.")
        i = lookup[rid]
        if any(str(page["metadata"][k]) != str(table.metadata[k][i]) for k in METADATA):
            raise ValueError("Previously inspected page metadata changed.")
        if not math.isclose(float(page["original_baseline_phishing_score"]), float(old_scores[i]), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Previously inspected page baseline scores changed.")
        if (set(page["context_features"]) != set(names[86:])
                or set(page["context_quality"]) != {k for k in table.quality if k.startswith("context_")}
                or np.float32(page["original_password_input_count"]) != table.values[i, names.index("password_input_count")]):
            raise ValueError("Previously inspected page feature/quality contract changed.")
        for name, value in page["context_features"].items():
            if np.float32(value) != table.values[i, names.index(name)]:
                raise ValueError("Previously inspected form-context features changed.")
        for name, value in page["context_quality"].items():
            if value != int(table.quality[name][i]):
                raise ValueError("Previously inspected context quality metadata changed.")
        selected.append(i)
        categories[i] = "previously_inspected_" + page["review_category"]
    return selected, categories


def explain(model, x, names, scores, threads, common):
    np = common.np
    with common.threadpool_limits(limits=threads):
        raw = np.asarray(model.booster_.predict(x, raw_score=True,
            num_iteration=model.best_iteration_ or None, num_threads=threads))
        contrib = np.asarray(model.booster_.predict(x, pred_contrib=True,
            num_iteration=model.best_iteration_ or None, num_threads=threads))
    if contrib.shape != (len(x), len(names) + 1) or not np.isfinite(contrib).all() or not np.isfinite(raw).all():
        raise ValueError("Unexpected binary model-contribution output.")
    error = float(np.max(np.abs(contrib.sum(axis=1) - raw)))
    if error > 1e-7 or not np.allclose(1 / (1 + np.exp(-np.clip(raw, -700, 700))), scores, rtol=0, atol=1e-9):
        raise ValueError("Feature contributions do not reconstruct the saved scores.")
    return raw, contrib, error


def write_sample(path, table, schema, old_scores, scores, selected, categories):
    quality, names = schema["quality_columns_not_classifier_inputs"], schema["feature_names"]
    columns = ["review_category", *METADATA, "old_phishing_score", "phishing_score", *quality, *names]
    temporary = path.with_suffix(".csv.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for i in selected:
                writer.writerow({"review_category": categories[i],
                    **{k: str(table.metadata[k][i]) for k in METADATA},
                    "old_phishing_score": float(old_scores[i]), "phishing_score": float(scores[i]),
                    **{k: int(table.quality[k][i]) for k in quality},
                    **{n: float(table.values[i, k]) for k, n in enumerate(names)}})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    for option, default in (("training-dir", "reports/form_context_training_v2"),
            ("baseline-dir", "reports/baseline_training_v1_1"), ("features-dir", "reports/features_context_v2"),
            ("base-features-dir", "reports/features"), ("baseline-script", "10_train_static_baselines.py"),
            ("training-script", "16_train_form_context_models.py"), ("output-dir", "reports/context_error_review_v2")):
        parser.add_argument("--" + option, type=Path, default=root / default)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--samples-per-category", type=int, default=8, help="Eight fresh rows in each of ten categories, plus previously inspected pages.")
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.samples_per_category <= 20:
        parser.error("Use positive threads and 1 to 20 samples per category.")
    for directory in (args.training_dir, args.baseline_dir, args.features_dir, args.base_features_dir):
        if args.output_dir.resolve().is_relative_to(directory.resolve()) or directory.resolve().is_relative_to(args.output_dir.resolve()):
            parser.error("Use a separate review output folder outside the feature and model folders.")
    print(f"Context error reviewer v{VERSION}; verifying completed files...", flush=True)
    paths, hashes, run, features, schema, base_schema, result, old_result = checked_inputs(args)
    common = load_reader(args.baseline_script)
    np = common.np
    packages = {name: version(name) for name in PACKAGES}
    if packages != run["run_config"]["package_versions"] or platform.python_version() != run["run_config"]["python_version"]:
        raise ValueError("Use the same working virtual environment that produced script 16's models.")
    report_path = args.output_dir / "context_error_review_report.json"
    if report_path.exists():
        previous = load_json(report_path)
        if previous.get("input_hashes") != hashes or previous.get("review_code_sha256") != digest_file(Path(__file__)):
            raise ValueError("This review folder belongs to different inputs/code. Choose a new --output-dir.")
    if args.output_dir.exists() and any(p.name not in {"context_error_review_report.json", "context_review_sample_features.csv"} for p in args.output_dir.iterdir()):
        raise ValueError("Use an empty review folder or this script's existing matching output folder.")
    print("Loading validation features only...", flush=True)
    table = common.load_features(paths["validation"], schema, run["main_validation_counts"], "validation", features)
    sensitivity = table.metadata["in_validation_sensitivity_subset"] == 1
    counts = common.counts(table)
    if (counts["benign"] == 0 or counts["benign"] != counts["phish"]
            or common.counts(table, sensitivity) != run["validation_sensitivity_counts"]):
        raise ValueError("Expected the same nonempty balanced validation and sensitivity subsets.")
    scores = read_scores(paths["new_scores"], table, np)
    old_scores = read_scores(paths["old_scores"], table, np)
    names, old_names = schema["feature_names"], base_schema["feature_names"]
    importance, old_importance = read_importance(paths["new_importance"], names), read_importance(paths["old_importance"], old_names)
    metrics, old_metrics = common.binary_metrics(table.y, scores), common.binary_metrics(table.y, old_scores)
    verify_metrics(metrics, result["metrics"]["main_validation"])
    verify_metrics(old_metrics, old_result["metrics"]["main_validation"])
    verify_metrics(common.binary_metrics(table.y[sensitivity], scores[sensitivity]), result["metrics"]["validation_sensitivity"])
    print("Reproducing all saved validation scores from both verified models...", flush=True)
    model, difference = load_verified_model(paths["new_bundle"], result, schema, run["run_config_sha256"], common, table, scores, importance, args.threads)
    old_model, old_difference = load_verified_model(paths["old_bundle"], old_result, base_schema, old_result["run_config_sha256"], common, table, old_scores, old_importance, args.threads)
    pred, before, y = scores >= THRESHOLD, old_scores >= THRESHOLD, table.y
    groups = table.metadata["site_group"]
    fp, fn = (y == 0) & pred, (y == 1) & ~pred
    summary = paired_group_summary(groups, y, old_scores, scores)
    fp_group = summary["top_groups_by_remaining_fp"][0]["site_group"] if fp.any() else None
    fn_group = summary["top_groups_by_remaining_fn"][0]["site_group"] if fn.any() else None
    fp_match, fn_match = groups == fp_group, groups == fn_group
    native = table.values[:, names.index("password_input_count")] > 0
    context = table.values[:, names.index("context_passwordlike_control_count")] > 0
    capped = (table.quality["html_analysis_truncated"] > 0) | (table.quality["js_analysis_truncated"] > 0)
    masks = {"all_false_positives": fp, "all_false_negatives": fn,
        "all_true_benign": (y == 0) & ~pred, "all_true_phish": (y == 1) & pred,
        "largest_fp_group_false_positives": fp & fp_match,
        "largest_fp_group_true_benign": (y == 0) & ~pred & fp_match,
        "largest_fn_group_false_negatives": fn & fn_match,
        "largest_fn_group_true_phish": (y == 1) & pred & fn_match,
        "password_cue_without_native_input_false_negatives": fn & context & ~native,
        "password_cue_without_native_input_true_phish": (y == 1) & pred & context & ~native}
    quality_masks = {"analysis_cap_reached": capped, "analysis_caps_not_reached": ~capped,
        "password_cue_present": context, "password_cue_without_native_input": context & ~native,
        "no_recognized_password_cue": ~context,
        "human_verification_hint_present": table.quality["context_human_verification_hint_present"] > 0,
        "unavailable_page_hint_present": table.quality["context_unavailable_page_hint_present"] > 0,
        "association_limited": (table.quality["context_parse_error"] > 0) | (table.quality["context_association_storage_cut"] > 0)}
    selected, categories = historical_cases(features, table, old_scores, names, np)
    historical_count = len(selected)
    selection_report = [{"category": "previously_inspected_pages", "selected_rows": historical_count}]
    selections = [("persistent_fp_largest_group", fp & before & fp_match, True, False),
        ("persistent_fp_other_groups", fp & before & ~fp_match, True, True),
        ("new_false_positives", fp & ~before, True, True),
        ("fixed_false_positive_controls", (y == 0) & ~pred & before, False, True),
        ("persistent_fn_largest_group", fn & ~before & fn_match, False, False),
        ("persistent_fn_other_groups", fn & ~before & ~fn_match, False, True),
        ("new_false_negatives", fn & before, False, True),
        ("fixed_false_negative_controls", (y == 1) & pred & ~before, True, True),
        ("true_benign_largest_fp_group_controls", (y == 0) & ~pred & ~before & fp_match, False, False),
        ("true_phish_largest_fn_group_controls", (y == 1) & pred & before & fn_match, True, False)]
    for category, mask, descending, diverse in selections:
        available = mask.copy()
        available[selected] = False
        indices = pick_examples(available, scores, table.metadata["record_id"], groups, args.samples_per_category, category, descending, diverse, np)
        selected.extend(indices)
        categories.update({i: category for i in indices})
        selection_report.append({"category": category, "eligible_rows": int(mask.sum()),
            "available_after_previous_selections": int(available.sum()), "selected_rows": len(indices),
            "selected_site_groups": len({str(groups[i]) for i in indices}), "prefer_distinct_site_groups": diverse})
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Invalid diagnostic sample selection.")
    print(f"Explaining {len(selected)} selected pages with both models...", flush=True)
    x = np.ascontiguousarray(table.values[selected])
    raw, contributions, reconstruction_error = explain(model, x, names, scores[selected], args.threads, common)
    old_raw, old_contributions, old_reconstruction_error = explain(old_model, x[:, :86].copy(), old_names, old_scores[selected], args.threads, common)
    examples = []
    def prediction_error(label, score):
        return "false_positive" if label == 0 and score >= THRESHOLD else "false_negative" if label == 1 and score < THRESHOLD else "correct"
    for j, i in enumerate(selected):
        values = {n: float(table.values[i, k]) for k, n in enumerate(names)}
        def details(ns, cs, margin, probability):
            positive = sorted((k for k in range(len(ns)) if cs[k] > 0), key=lambda k: (-cs[k], ns[k]))[:8]
            negative = sorted((k for k in range(len(ns)) if cs[k] < 0), key=lambda k: (cs[k], ns[k]))[:8]
            def ranked(indices):
                return [{"feature": ns[k], "value": values[ns[k]], "raw_score_contribution": float(cs[k])} for k in indices]
            return {"phishing_score": float(probability), "error_at_0_50": prediction_error(int(y[i]), probability),
                "raw_score": float(margin), "expected_raw_score": float(cs[-1]),
                "feature_contributions": {n: float(cs[k]) for k, n in enumerate(ns)},
                "strongest_push_toward_phish": ranked(positive), "strongest_push_toward_benign": ranked(negative)}
        old_exp = details(old_names, old_contributions[j], old_raw[j], old_scores[i])
        new_exp = details(names, contributions[j], raw[j], scores[i])
        examples.append({"review_category": categories[i], "metadata": {k: str(table.metadata[k][i]) for k in METADATA},
            "quality": {k: int(table.quality[k][i]) for k in schema["quality_columns_not_classifier_inputs"]},
            "feature_values": values, "old_model": old_exp, "new_model": new_exp,
            "new_context_feature_contribution_sum": float(contributions[j, 86:-1].sum()),
            "raw_score_change": float(raw[j] - old_raw[j])})
    focus = list(dict.fromkeys([r["feature"] for r in importance[:15]] + sorted(SIZE_FEATURES)
                              + ["password_input_count"] + names[86:]))
    distributions = [{"feature": n, "cohorts": {k: describe(table.values[:, names.index(n)], m, np)
                     for k, m in masks.items()}} for n in focus]
    for key, path in paths.items():
        if digest_file(path) != hashes[key]:
            raise ValueError("An input changed during review; no completed report was written.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = args.output_dir / "context_review_sample_features.csv"
    write_sample(sample, table, schema, old_scores, scores, selected, categories)
    report = {"status": "context_error_review_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": VERSION, "review_code_sha256": digest_file(Path(__file__)), "input_hashes": hashes,
        "model_name": MODEL, "old_model_name": OLD_MODEL, "threshold": THRESHOLD, "seed": SEED,
        "threads": args.threads, "python_version": platform.python_version(), "package_versions": packages,
        "main_validation_counts": counts, "validation_sensitivity_counts": common.counts(table, sensitivity),
        "old_validation_metrics": old_metrics, "new_validation_metrics": metrics,
        "paired_group_summary": summary, "largest_false_positive_group": fp_group, "largest_false_negative_group": fn_group,
        "global_gain_importance": importance, "gain_share_by_named_group": {
            "anchor_features": sum(r["share_of_training_split_gain"] for r in importance if r["feature"].startswith("anchor_")),
            "five_size_features": sum(r["share_of_training_split_gain"] for r in importance if r["feature"] in SIZE_FEATURES),
            "twenty_context_features": sum(r["share_of_training_split_gain"] for r in importance if r["feature"] in names[86:])},
        "quality_cohorts": {k: {"old_model": common.binary_metrics(y[m], old_scores[m]),
            "new_model": common.binary_metrics(y[m], scores[m])} for k, m in quality_masks.items()},
        "focus_feature_distributions": distributions, "review_sample_selection": selection_report, "review_examples": examples,
        "sample_selection_policy": "Include all previously inspected script 15 pages first. Fresh categories select half extreme-score cases and half seeded random cases. Across-domain categories prefer distinct site groups; fill from seeded candidates if there are too few groups. No duplicate record IDs. These selected pages are diagnostic examples, not a representative accuracy sample.",
        "contribution_definition": "Native LightGBM pred_contrib in binary raw-score units: positive values push toward phish, negative toward benign. The expected value plus every feature contribution reconstructs the raw score; sigmoid reconstructs the score. Contributions describe model associations, not causes, calibrated risk, or proof of malicious behavior. Changes between two separately fitted models also reflect changes to the other features and trees.",
        "verification": {"new_max_saved_probability_difference": difference, "old_max_saved_probability_difference": old_difference,
            "new_max_raw_score_reconstruction_error": reconstruction_error,
            "old_max_raw_score_reconstruction_error": old_reconstruction_error, "inputs_unchanged": True},
        "sample_csv_sha256": digest_file(sample), "model_retrained": False, "threshold_changed": False,
        "labels_changed": False, "balanced_splits_changed": False, "training_data_loaded": False,
        "official_test_evaluated": False, "websites_requested": False, "javascript_executed": False,
        "limitations": ["Global split gain is a training statistic, not an error-specific contribution or an accuracy percentage.",
            "Correlated features can redistribute importance; low gain is not evidence that a feature should be deleted.",
            "Validation was used for early stopping, feature design and these diagnostics, so this is not an independent performance test.",
            "Quality hints and absent password cues are imperfect static observations; they cannot establish label correctness or page safety.",
            "This script does not inspect archived HTML content or execute browser behavior."]}
    save_json(report_path, report)
    print(f"Saved report: {report_path.resolve()}", flush=True)
    print(f"Saved selected feature rows: {sample.resolve()}", flush=True)
    print("Context error review complete. Models and threshold unchanged; official test untouched.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ImportError) as error:
        raise SystemExit(f"Context error review stopped: {error}. A report is complete only after the final completion message.")
    except MemoryError:
        raise SystemExit("Context error review stopped: insufficient RAM. Close memory-heavy applications and rerun.")
