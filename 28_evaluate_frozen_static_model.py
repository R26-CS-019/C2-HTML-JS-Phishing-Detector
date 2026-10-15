"""Evaluate the frozen static LightGBM baseline on script-27 test features.

Run in the original working .venv beside unchanged scripts 09, 15, 16, 25,
26 and 27. Uses existing NumPy, scikit-learn, LightGBM, joblib, threadpoolctl.
No training, calibration, feature/threshold search, HTML decoding, website
requests or JS execution occurs. The same eligible-page predictions supply
both cohorts. Saved validation predictions must reproduce before test scoring.
"""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import platform
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

VERSION = "1.0"
THRESHOLD = 0.50
SEED = 42
LABELS = {"benign": 0, "phish": 1}
CODE_SHA = {
    "25_freeze_static_baseline.py": "08ce9defa54b85ffa4ad5a0cbc2b951761d7fb4a4b0269820d2be34b8c4308c7",
    "26_prepare_static_test_cohorts.py": "fe5d87805a0f8b5c7102b560747550f33820373c17a92cb4221132bb9021de5e",
    "27_extract_static_test_features.py": "d49a768a44b90f9c79de83097f89fabcf96aa4117cd1ea218b6d72bc45cd9156",
}
EXTRA_METADATA = ["source_row_number", "in_balanced_test_cohort"]
HEX64 = re.compile(r"[0-9a-fA-F]{64}\Z")
OUTPUT_NAMES = {"static_test_evaluation_report.json", "evaluation_attempt.json",
    "eligible_test_predictions.csv", "balanced_test_predictions.csv", "test_metrics_summary.csv"}
FEATURE_OUTPUTS = {"eligible_test_static_features.csv", "balanced_test_static_features.csv",
    "static_feature_schema.json", "test_feature_index.sqlite"}
PREPARATION_OUTPUTS = {"eligible_test_manifest.csv", "balanced_test_manifest.csv",
    "test_selection_decisions.jsonl", "test_preparation_index.sqlite"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8"))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_helper(path, wanted):
    require(path.is_file() and digest(path) == wanted, f"Keep the unchanged original helper: {path.name}.")
    spec = importlib.util.spec_from_file_location("test28_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def verify_outputs(directory, report, expected):
    require(set(report["outputs"]) == expected, "Completed report is missing expected outputs.")
    for name in sorted(expected):
        path = directory / name
        require(path.is_file() and digest(path) == report["outputs"][name]["sha256"],
                f"Completed input output changed or is missing: {path}.")
    return {directory / name: report["outputs"][name]["sha256"] for name in expected}


def checked_inputs(args):
    freeze_module = load_helper(args.project_dir / "25_freeze_static_baseline.py", CODE_SHA["25_freeze_static_baseline.py"])
    preparer = load_helper(args.project_dir / "26_prepare_static_test_cohorts.py", CODE_SHA["26_prepare_static_test_cohorts.py"])
    require(digest(args.project_dir / "27_extract_static_test_features.py") == CODE_SHA["27_extract_static_test_features.py"],
            "Keep unchanged script 27, which produced the completed test features.")
    freeze = read_json(args.freeze_report)
    spec = freeze["specification"]
    require(freeze.get("status") == "static_baseline_frozen_test_not_evaluated"
            and freeze.get("official_test_evaluated") is False
            and freeze_module.object_sha(spec) == freeze["specification_sha256"]
            and spec["script_sha256"] == CODE_SHA["25_freeze_static_baseline.py"]
            and spec["test_protocol"] == freeze_module.test_protocol()
            and spec["model_name"] == freeze_module.MODEL_NAME
            and spec["threshold"] == THRESHOLD and spec["seed"] == SEED
            and spec["feature_count"] == len(spec["feature_names"]) == 106,
            "Use the completed unchanged script-25 frozen baseline.")
    print("Verifying frozen model/development artifacts and completed test outputs...", flush=True)
    refs = preparer.frozen_reference_paths(args, spec)
    protected = {path: spec["input_files"][name]["sha256"] for name, path in refs.items()}
    protected[args.freeze_report] = digest(args.freeze_report)
    feature_path = args.features_dir / "test_feature_extraction_report.json"
    preparation_path = args.test_dir / "test_preparation_report.json"
    extraction, prepared = read_json(feature_path), read_json(preparation_path)
    ei, pi = extraction["inputs"], prepared["inputs"]
    require(extraction.get("status") == "static_test_feature_extraction_complete_not_evaluated"
            and extraction.get("official_test_evaluated") is False
            and extraction.get("model_loaded") is False
            and extraction.get("model_trained") is False
            and extraction.get("quality_based_exclusions") == 0
            and extraction.get("cohort_membership_changed") is False
            and extraction.get("labels_changed") is False
            and prepared.get("status") == "static_test_cohorts_prepared_not_evaluated"
            and prepared.get("official_test_evaluated") is False
            and prepared.get("balanced_test_cohort_available") is True,
            "Finish scripts 26 and 27 before test evaluation.")
    require(ei["code_sha256"] == CODE_SHA["27_extract_static_test_features.py"]
            and pi["code_sha256"] == ei["preparer_code_sha256"] == CODE_SHA["26_prepare_static_test_cohorts.py"]
            and ei["freeze_report_sha256"] == pi["freeze_report_sha256"] == protected[args.freeze_report]
            and ei["frozen_specification_sha256"] == pi["frozen_specification_sha256"] == freeze["specification_sha256"]
            and ei["test_preparation_report_sha256"] == digest(preparation_path)
            and ei["test_preparation_output_hashes"] == prepared["outputs"]
            and ei["source_files"] == pi["source_files"]
            and ei["source_file_snapshots"] == prepared["source_file_snapshots"], "Test-feature provenance differs from the frozen preparation.")
    schema = read_json(args.features_dir / "static_feature_schema.json")
    require(schema == spec["feature_extraction_contract"] == read_json(refs["context_schema"])
            and ei["feature_names"] == schema["feature_names"] == spec["feature_names"]
            and ei["parameters"] == schema["parameters"]
            and ei["association_limits"] == schema["association_limits"]
            and ei["quality_columns_not_classifier_inputs"] == schema["quality_columns_not_classifier_inputs"]
            and ei["metadata_columns"] == schema["metadata_columns"] + EXTRA_METADATA
            and ei["base_extractor_sha256"] == spec["input_files"]["base_extractor"]["sha256"]
            and ei["context_extractor_sha256"] == spec["input_files"]["context_extractor"]["sha256"],
            "Test features differ from the frozen extraction schema/order.")
    require(extraction["feature_count"] == ei["feature_count"] == len(set(schema["feature_names"])) == 106
            and not set(schema["feature_names"]).intersection(ei["metadata_columns"] + ei["quality_columns_not_classifier_inputs"]),
            "Labels/metadata/quality must remain outside the exact 106 features.")
    expected_counts = {"eligible": prepared["eligible_unbalanced_counts"], "balanced": prepared["balanced_test_counts"]}
    require(extraction["extracted_label_counts"] == expected_counts
            and ei["eligible_test_counts"] == expected_counts["eligible"]
            and ei["balanced_test_counts"] == expected_counts["balanced"]
            and expected_counts["balanced"]["benign"] == expected_counts["balanced"]["phish"] > 0,
            "Prepared/extracted test class counts differ.")
    protected.update(verify_outputs(args.features_dir, extraction, FEATURE_OUTPUTS))
    protected.update(verify_outputs(args.test_dir, prepared, PREPARATION_OUTPUTS))
    protected.update({feature_path: digest(feature_path), preparation_path: digest(preparation_path)})
    require(extraction["outputs"]["eligible_test_static_features.csv"]["rows"] == sum(expected_counts["eligible"].values())
            == extraction["outputs"]["test_feature_index.sqlite"]["rows"]
            and extraction["outputs"]["balanced_test_static_features.csv"]["rows"] == sum(expected_counts["balanced"].values()),
            "Test feature output row counts differ.")
    installed = {}
    for package, wanted in spec["training_package_versions"].items():
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = None
        require(installed[package] == wanted,
                f"Use the original working .venv: {package} requires {wanted}, found {installed[package]}. Do not upgrade before frozen-model evaluation.")
    require(platform.python_version() == spec["training_python_version"],
            f"Use the original Python {spec['training_python_version']} .venv; current Python is {platform.python_version()}.")
    config = {
        "script_version": VERSION, "code_sha256": digest(Path(__file__)), "helper_code_sha256": CODE_SHA,
        "freeze_report_sha256": protected[args.freeze_report], "frozen_specification_sha256": freeze["specification_sha256"],
        "test_preparation_report_sha256": protected[preparation_path], "test_feature_extraction_report_sha256": protected[feature_path],
        "test_feature_output_hashes": extraction["outputs"], "test_preparation_output_hashes": prepared["outputs"],
        "model_name": spec["model_name"], "model_bundle_sha256": spec["model_bundle_sha256"],
        "best_iteration": spec["best_iteration"], "threshold": THRESHOLD, "seed": SEED,
        "positive_class": "phish", "label_mapping": LABELS, "model_input_dtype": "float32",
        "feature_names": spec["feature_names"], "feature_count": 106,
        "package_versions": installed, "python_version": platform.python_version(), "threads": args.threads,
        "eligible_test_counts": expected_counts["eligible"], "balanced_test_counts": expected_counts["balanced"],
        "validation_reproduction_policy": "All original validation IDs/decisions; saved scores rtol=1e-8, atol=1e-10; no model/threshold selection.",
        "test_scoring_policy": "Predict each eligible page once; balanced scores are the unchanged script-26 subset.",
        "test_guided_tuning": False, "model_refit": False,
    }
    return config, freeze, refs, extraction, prepared, protected, freeze_module, preparer


def dependencies():
    import numpy as np
    import joblib
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
    from threadpoolctl import threadpool_limits
    return SimpleNamespace(np=np, joblib=joblib, lgb=lgb, roc_auc_score=roc_auc_score,
        average_precision_score=average_precision_score, log_loss=log_loss, threadpool_limits=threadpool_limits)


def check_output_scope(args, refs):
    output = args.output_dir.resolve()
    protected_dirs = [args.features_dir.resolve(), args.test_dir.resolve(), args.freeze_report.parent.resolve(),
        *(p.parent.resolve() for name, p in refs.items() if name not in ("base_extractor", "context_extractor", "trainer"))]
    for parent in protected_dirs:
        require(not output.is_relative_to(parent) and not parent.is_relative_to(output), "Use a separate evaluation output folder outside existing report folders.")
    require(not args.project_dir.resolve().is_relative_to(output), "Evaluation folder must not contain the project.")
    if output.exists():
        require(output.is_dir(), "Evaluation output path is not a folder.")
        allowed = OUTPUT_NAMES | {n + ".tmp" for n in OUTPUT_NAMES}
        require(all(p.name in allowed for p in output.iterdir()), "Evaluation output folder contains unrelated files; choose a new --output-dir.")
        require(not any(output.iterdir()) or (output / "evaluation_attempt.json").is_file(),
                "Existing evaluation outputs have no matching attempt record; keep them and choose a new --output-dir.")
    else:
        output.mkdir(parents=True)


def reuse_complete(args, config):
    attempt_path = args.output_dir / "evaluation_attempt.json"
    if attempt_path.exists():
        require(read_json(attempt_path)["inputs"] == config,
                "Existing evaluation attempt uses different code/inputs. Keep it; changes after test exposure require a new independent holdout.")
    path = args.output_dir / "static_test_evaluation_report.json"
    if not path.exists():
        return None
    report = read_json(path)
    require(report.get("status") == "frozen_static_test_evaluation_complete" and report["inputs"] == config
            and report["official_test_evaluated"] is True and report["model_trained"] is False
            and report["threshold"] == THRESHOLD,
            "Existing completed evaluation differs. Keep it; use a new independent holdout for further model changes.")
    verify_outputs(args.output_dir, report, OUTPUT_NAMES - {"static_test_evaluation_report.json"})
    for cohort, counts in (("eligible", config["eligible_test_counts"]), ("balanced", config["balanced_test_counts"])):
        require(report["metrics"][cohort]["rows"] == sum(counts.values())
                and report["metrics"][cohort]["benign_rows"] == counts["benign"]
                and report["metrics"][cohort]["phishing_rows"] == counts["phish"], "Cached evaluation cohort counts differ.")
    return report


def open_feature_index(args, extraction, prepared, preparer):
    for directory, name in ((args.features_dir, "test_feature_index.sqlite"), (args.test_dir, "test_preparation_index.sqlite")):
        for suffix in ("-wal", "-journal"):
            sidecar = Path(str(directory / name) + suffix)
            require(not sidecar.exists() or sidecar.stat().st_size == 0, "Finish and close checkpoint writers before evaluation.")
    db = sqlite3.connect((args.features_dir / "test_feature_index.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.execute("PRAGMA cache_size=-32768")
        db.execute("ATTACH DATABASE ? AS audit", [(args.test_dir / "test_preparation_index.sqlite").resolve().as_uri() + "?mode=ro"])
        require(json.loads(db.execute("SELECT value FROM state WHERE key='config'").fetchone()[0]) == extraction["inputs"]
                and json.loads(db.execute("SELECT value FROM audit.state WHERE key='config'").fetchone()[0]) == prepared["inputs"],
                "Checkpoint configuration differs from its completed report.")
        require(db.execute("SELECT count(*) FROM pages").fetchone()[0] == sum(prepared["eligible_unbalanced_counts"].values())
                and not db.execute("SELECT 1 FROM pages WHERE processed!=1 OR feature_values IS NULL OR quality_values IS NULL LIMIT 1").fetchone(),
                "Test feature checkpoint is incomplete.")
        require(not db.execute("""SELECT 1 FROM pages p
            LEFT JOIN audit.test_pages t ON p.id=t.record_id
            LEFT JOIN audit.decisions d ON t.row_key=d.row_key
            WHERE t.record_id IS NULL OR d.decision IS NOT 'eligible'
              OR p.label IS NOT t.label OR p.site_group IS NOT t.site_group OR p.source_file IS NOT t.source_file
              OR p.source_row_number IS NOT t.source_row_number OR p.page_url IS NOT t.url
              OR p.html_characters IS NOT t.html_characters OR p.raw_sha IS NOT t.raw_html_sha256
              OR p.normalized_sha IS NOT t.normalized_html_sha256 OR p.balanced IS NOT d.in_balanced_test_cohort
            LIMIT 1""").fetchone(), "Feature checkpoint membership differs from the prepared test cohort.")
        # Check the original audited decisions on a separate read-only handle;
        # this function uses neither model scores nor current HTML.
        audit = sqlite3.connect((args.test_dir / "test_preparation_index.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
        try:
            chosen = preparer.checked_selection(audit, prepared["eligible_unbalanced_counts"], prepared["per_class_balanced_target"])
            require(chosen == prepared["balanced_test_counts"], "Prepared balanced membership differs.")
        finally:
            audit.close()
        return db
    except BaseException:
        db.close()
        raise


def parse_vector(row, metadata_size, qualities, feature_names, deps):
    np = deps.np
    flags = []
    for value in row[metadata_size:metadata_size+len(qualities)]:
        require(re.fullmatch(r"[0-9]+", value) is not None, "Invalid nonnegative integer quality value.")
        flags.append(int(value))
    values = np.asarray(row[metadata_size+len(qualities):], dtype=np.float32)
    require(values.shape == (len(feature_names),) and np.isfinite(values).all() and (values >= 0).all(),
            "Invalid/non-finite/negative float32 model input.")
    return values, flags


def load_features(path, schema, metadata_columns, expected_counts, deps, expected_rows=None, validation_metadata=None):
    np = deps.np
    n = sum(expected_counts.values())
    names, qualities = schema["feature_names"], schema["quality_columns_not_classifier_inputs"]
    header = [*metadata_columns, *qualities, *names]
    values = np.empty((n, len(names)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    quality = {name: np.empty(n, dtype=np.int64) for name in qualities}
    metadata, counts, seen, ordinals = [], Counter(), set(), set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        require(next(reader, None) == header, f"Unexpected exact feature order/header: {path.name}.")
        if expected_rows is None:
            paired = ((row, None) for row in reader)
        else:
            paired = itertools.zip_longest(reader, expected_rows)
        loaded = 0
        for row, checkpoint in paired:
            require(row is not None and len(row) == len(header) and loaded < n, f"Malformed/extra feature row in {path.name}.")
            meta = tuple(row[:len(metadata_columns)])
            rid, label, group, source, similarity, sensitivity = meta[:6]
            require(HEX64.fullmatch(rid) is not None and rid not in seen and label in LABELS and group and source
                    and similarity in ("0", "1") and sensitivity in ("0", "1"), f"Invalid/duplicate feature metadata in {path.name}.")
            numeric, flags = parse_vector(row, len(metadata_columns), qualities, names, deps)
            if expected_rows is not None:
                require(checkpoint is not None, "CSV has more rows than its checkpoint.")
                pid, plabel, pgroup, psource, ordinal, balanced, feature_json, quality_json = checkpoint
                require(meta == (pid, plabel, pgroup, psource, "0", "0", str(ordinal), str(balanced)),
                        "Eligible test metadata differs from its checkpoint.")
                require((source, ordinal) not in ordinals and ordinal >= 0 and balanced in (0, 1), "Repeated/invalid physical source row.")
                ordinals.add((source, ordinal))
                old_features, old_quality = json.loads(feature_json), json.loads(quality_json)
                require(len(old_features) == len(names) and len(old_quality) == len(qualities)
                        and np.array_equal(numeric, np.asarray(old_features, dtype=np.float32)) and flags == old_quality,
                        "Test feature/quality CSV values differ from the completed checkpoint.")
            if validation_metadata is not None:
                require(rid in validation_metadata and all(meta[j] == validation_metadata[rid][name]
                        for j, name in enumerate(metadata_columns)), "Validation feature metadata differs from the frozen manifest.")
            values[loaded] = numeric; y[loaded] = LABELS[label]
            for name, value in zip(qualities, flags):
                quality[name][loaded] = value
            metadata.append(meta); counts[label] += 1; seen.add(rid); loaded += 1
            if loaded % 25000 == 0:
                print(f"  {path.name}: verified {loaded:,}/{n:,} rows.", flush=True)
    require(loaded == n and dict(counts) == expected_counts, f"Feature coverage/class counts differ: {path.name}.")
    if validation_metadata is not None:
        require(seen == set(validation_metadata), "Validation IDs differ from the frozen manifest.")
    return SimpleNamespace(values=values, y=y, quality=quality, metadata=metadata, metadata_columns=metadata_columns)


def verify_balanced_csv(path, table, schema, counts, deps):
    np = deps.np
    mask = np.asarray([m[-1] == "1" for m in table.metadata], dtype=bool)
    indices = np.flatnonzero(mask)
    require({l: int(np.count_nonzero(mask & (table.y == n))) for l, n in LABELS.items()} == counts, "Balanced feature mask differs.")
    header = [*table.metadata_columns, *schema["quality_columns_not_classifier_inputs"], *schema["feature_names"]]
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file); require(next(reader, None) == header, "Balanced test CSV header differs.")
        for row, index in itertools.zip_longest(reader, indices):
            require(row is not None and index is not None and len(row) == len(header), "Balanced test CSV coverage differs.")
            require(tuple(row[:len(table.metadata_columns)]) == table.metadata[index], "Balanced test CSV membership/metadata differs.")
            values, flags = parse_vector(row, len(table.metadata_columns), schema["quality_columns_not_classifier_inputs"], schema["feature_names"], deps)
            require(np.array_equal(values, table.values[index])
                    and flags == [int(table.quality[name][index]) for name in schema["quality_columns_not_classifier_inputs"]],
                    "Balanced CSV features/quality differ from the eligible-row subset.")
    return mask


def verify_quality_summary(table, balanced, extraction, deps):
    np = deps.np
    names = extraction["inputs"]["feature_names"]
    tracked = {names.index("password_input_count"): "password_input_present_rows",
               names.index("script_inline_executable_count"): "inline_executable_script_present_rows"}
    tracked.update({j: name + "_present_rows" for j, name in enumerate(names) if name.startswith("context_")})
    for cohort, membership in (("eligible", np.ones(len(table.y), dtype=bool)), ("balanced", balanced)):
        for label, encoded in LABELS.items():
            mask = membership & (table.y == encoded)
            actual = {"rows": int(np.count_nonzero(mask))}
            actual.update({name + "_rows": int(np.count_nonzero((values > 0) & mask)) for name, values in table.quality.items()})
            actual.update({key: int(np.count_nonzero((table.values[:, index] > 0) & mask)) for index, key in tracked.items()})
            require(actual == extraction["quality_summary_by_cohort_and_label"][cohort][label], "Recomputed test quality/presence summaries differ.")


def checked_model(path, spec, refs, deps):
    # The artifact is the user's own frozen training output. Hash verification
    # precedes deserialization; no arbitrary replacement pickle is accepted.
    require(digest(path) == spec["model_bundle_sha256"], "Frozen model bytes changed before loading.")
    bundle = deps.joblib.load(path)
    require(isinstance(bundle, dict) and bundle.get("bundle_version") == "form_context_1.0"
            and bundle.get("model_name") == spec["model_name"]
            and bundle.get("feature_names") == spec["feature_names"]
            and bundle.get("label_mapping") == LABELS and bundle.get("threshold") == THRESHOLD
            and bundle.get("feature_extraction_contract") == spec["feature_extraction_contract"]
            and bundle.get("base_extractor_code_sha256") == spec["input_files"]["base_extractor"]["sha256"]
            and bundle.get("extractor_code_sha256") == spec["input_files"]["context_extractor"]["sha256"]
            and bundle.get("run_config_sha256") == digest(refs["run_config"])
            and bundle.get("package_versions") == spec["training_package_versions"]
            and bundle.get("training_label_counts") == spec["development_class_counts"]["train"],
            "Loaded model bundle differs from its frozen training provenance.")
    model = bundle["estimator"]
    require(isinstance(model, deps.lgb.LGBMClassifier) and list(model.classes_) == [0, 1]
            and model.objective_ == "binary" and model.n_features_in_ == 106
            and model.booster_.num_feature() == 106 and model.booster_.feature_name() == spec["feature_names"]
            and model.best_iteration_ == spec["best_iteration"] > 0
            and model.booster_.current_iteration() >= spec["best_iteration"],
            "Loaded estimator type, label mapping, feature order or best iteration differs.")
    params = model.get_params(deep=False)
    require(all(params.get(name) == value for name, value in spec["training_hyperparameters"].items()),
            "Loaded estimator hyperparameters differ from the frozen model.")
    require(digest(path) == spec["model_bundle_sha256"], "Model bytes changed while loading.")
    return model


def predict(model, table, spec, args, deps, progress=None):
    np = deps.np
    require(table.values.dtype == np.float32 and table.values.shape[1] == 106, "Predict with the frozen float32 feature matrix.")
    scores = np.empty(len(table.y), dtype=np.float64)
    with deps.threadpool_limits(limits=args.threads):
        for start in range(0, len(scores), 5000):
            end = min(start + 5000, len(scores))
            batch = np.asarray(model.booster_.predict(table.values[start:end], num_iteration=spec["best_iteration"],
                raw_score=False, num_threads=args.threads), dtype=np.float64)
            require(batch.shape == (end-start,) and np.isfinite(batch).all() and ((batch >= 0) & (batch <= 1)).all(), "Invalid model-score vector.")
            scores[start:end] = batch
            if progress:
                progress(end)
    return scores


def metrics(y, scores, deps):
    np = deps.np
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    require(y.shape == scores.shape and np.isin(y, [0, 1]).all() and np.isfinite(scores).all()
            and ((scores >= 0) & (scores <= 1)).all(), "Invalid metric inputs.")
    if not len(y):
        return {"rows": 0, "status": "empty_subset", "threshold": THRESHOLD, "positive_class": "phish"}
    predicted = scores >= THRESHOLD
    tn = int(np.count_nonzero((y == 0) & ~predicted)); fp = int(np.count_nonzero((y == 0) & predicted))
    fn = int(np.count_nonzero((y == 1) & ~predicted)); tp = int(np.count_nonzero((y == 1) & predicted))
    benign, phish = tn + fp, tp + fn
    both = benign > 0 and phish > 0
    recall = tp / phish if phish else None
    return {"rows": len(y), "threshold": THRESHOLD, "positive_class": "phish", "benign_rows": benign, "phishing_rows": phish,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": (tn+tp)/len(y), "balanced_accuracy": (tn/benign+recall)/2 if both else None,
        "precision": tp/(tp+fp) if tp+fp else 0.0, "recall": recall,
        "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
        "false_positive_rate": fp/benign if benign else None, "false_negative_rate": fn/phish if phish else None,
        "roc_auc": float(deps.roc_auc_score(y, scores)) if both else None,
        "average_precision": float(deps.average_precision_score(y, scores)) if both else None,
        "log_loss": float(deps.log_loss(y, np.column_stack((1-scores, scores)), labels=[0, 1])),
        "brier_score": float(np.mean((scores-y)**2)),
        "precision_zero_denominator_policy": "0 if no phishing predictions",
        "undefined_metric_policy": "null for class-denominator/ranking metrics lacking a required class; F1=0 for zero F1 denominator",
    }


def reproduce_validation(model, refs, spec, freeze, freeze_module, args, deps):
    print("Reproducing all saved validation scores before test prediction...", flush=True)
    _, _, valmeta = freeze_module.checked_manifest(refs["balanced_manifest"], spec["development_class_counts"])
    table = load_features(refs["context_validation"], spec["feature_extraction_contract"],
        spec["feature_extraction_contract"]["metadata_columns"], spec["development_class_counts"]["validation"], deps,
        validation_metadata=valmeta)
    expected = {}
    with refs["validation_predictions"].open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file); require(reader.fieldnames == freeze_module.PREDICTION_COLUMNS, "Unexpected frozen validation prediction columns.")
        for row in reader:
            rid = row["record_id"]
            require(rid not in expected and rid in valmeta
                    and all(row[name] == valmeta[rid][name] for name in freeze_module.PREDICTION_COLUMNS[:6]),
                    "Saved validation prediction metadata differs.")
            expected[rid] = row
    require(set(expected) == set(valmeta), "Saved validation prediction coverage differs.")
    scores = predict(model, table, spec, args, deps)
    old_scores = deps.np.asarray([float(expected[m[0]]["phishing_score"]) for m in table.metadata], dtype=deps.np.float64)
    require(deps.np.isfinite(old_scores).all() and deps.np.allclose(scores, old_scores, rtol=1e-8, atol=1e-10),
            "Loaded model does not reproduce saved validation scores. No test predictions made.")
    for meta, score, encoded in zip(table.metadata, scores, table.y):
        predicted = "phish" if score >= THRESHOLD else "benign"
        error = "false_positive" if encoded == 0 and predicted == "phish" else "false_negative" if encoded == 1 and predicted == "benign" else "correct"
        require(expected[meta[0]]["prediction_at_0_50"] == predicted and expected[meta[0]]["error_type"] == error,
                "Reproduced validation decisions differ. No test predictions made.")
    actual = metrics(table.y, scores, deps)
    prior = freeze["verified_validation_metrics_at_0_50"]
    require(actual["confusion_matrix"] == prior["confusion_matrix"] and actual["rows"] == prior["rows"], "Reproduced validation confusion counts differ.")
    for name in freeze_module.METRICS_CHECKED:
        require(math.isclose(actual[name], prior[name], rel_tol=1e-9, abs_tol=1e-10), f"Reproduced validation metric differs: {name}.")
    return {"all_scores_reproduced": True, "all_decisions_reproduced": True, "rows": len(scores),
        "maximum_absolute_score_difference": float(deps.np.max(deps.np.abs(scores-old_scores))),
        "metrics": actual, "used_for_model_or_threshold_selection": False}


def quality_metrics(table, balanced, scores, deps):
    np = deps.np
    result = {}
    for cohort, member in (("eligible", np.ones(len(scores), dtype=bool)), ("balanced", balanced)):
        groups = {}
        for name, flag in table.quality.items():
            groups[name] = {"present": metrics(table.y[member & (flag > 0)], scores[member & (flag > 0)], deps),
                            "absent": metrics(table.y[member & (flag == 0)], scores[member & (flag == 0)], deps)}
        caps = (table.quality["html_analysis_truncated"] > 0) | (table.quality["js_analysis_truncated"] > 0)
        context = (table.quality["context_parse_error"] > 0) | (table.quality["context_association_storage_cut"] > 0)
        for name, mask in (("html_or_js_cap_reached", caps), ("html_js_caps_not_reached", ~caps),
                           ("context_parse_or_association_limit", context), ("context_parse_and_association_not_limited", ~context)):
            groups[name] = metrics(table.y[member & mask], scores[member & mask], deps)
        result[cohort] = groups
    return result


def export_predictions(directory, table, balanced, scores, quality_names):
    outputs = {}
    header = [*table.metadata_columns, *quality_names, "phishing_score", "prediction_at_0_50", "error_type"]
    for name, selected in (("eligible_test_predictions.csv", None), ("balanced_test_predictions.csv", balanced)):
        path = directory / name; temporary = path.with_suffix(path.suffix + ".tmp")
        count = 0
        try:
            with temporary.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file); writer.writerow(header)
                for index, (metadata, encoded, score) in enumerate(zip(table.metadata, table.y, scores)):
                    if selected is not None and not selected[index]:
                        continue
                    predicted = "phish" if score >= THRESHOLD else "benign"
                    error = "false_positive" if encoded == 0 and predicted == "phish" else "false_negative" if encoded == 1 and predicted == "benign" else "correct"
                    writer.writerow([*metadata, *[int(table.quality[q][index]) for q in quality_names], float(score), predicted, error]); count += 1
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        outputs[name] = {"rows": count, "sha256": digest(path)}
    return outputs


def export_summary(directory, overall):
    fields = ["cohort", "rows", "benign_rows", "phishing_rows", "threshold", "accuracy", "balanced_accuracy",
        "precision", "recall", "f1", "false_positive_rate", "false_negative_rate", "roc_auc", "average_precision", "tn", "fp", "fn", "tp"]
    path = directory / "test_metrics_summary.csv"; temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fields, extrasaction="ignore"); writer.writeheader()
            for cohort in ("balanced", "eligible"):
                writer.writerow({"cohort": cohort, **overall[cohort], **overall[cohort]["confusion_matrix"]})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"rows": 2, "sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--freeze-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    require(1 <= args.threads <= 64, "Use a positive --threads value up to 64.")
    for name, relative in {"features_dir": "reports/test_features_v1", "test_dir": "reports/test_preparation_v1",
        "freeze_report": "reports/static_baseline_freeze_v1/static_baseline_freeze_report.json", "output_dir": "reports/static_test_evaluation_v1"}.items():
        if getattr(args, name) is None:
            setattr(args, name, args.project_dir / relative)
    print(f"Frozen static model evaluation v{VERSION}; fixed threshold 0.50; no fitting/tuning.", flush=True)
    config, freeze, refs, extraction, prepared, protected, freeze_module, preparer = checked_inputs(args)
    check_output_scope(args, refs)
    report = reuse_complete(args, config)
    if report:
        print("Verified and reused completed evaluation; no new model predictions.", flush=True)
    else:
        deps = dependencies()
        spec = freeze["specification"]; schema = spec["feature_extraction_contract"]
        print("Checking eligible features against checkpoints and balanced-subset membership...", flush=True)
        db = open_feature_index(args, extraction, prepared, preparer)
        try:
            selected = db.execute("SELECT id,label,site_group,source_file,source_row_number,balanced,feature_values,quality_values FROM pages ORDER BY id")
            table = load_features(args.features_dir / "eligible_test_static_features.csv", schema, extraction["inputs"]["metadata_columns"],
                prepared["eligible_unbalanced_counts"], deps, expected_rows=selected)
        finally:
            db.close()
        balanced = verify_balanced_csv(args.features_dir / "balanced_test_static_features.csv", table, schema, prepared["balanced_test_counts"], deps)
        verify_quality_summary(table, balanced, extraction, deps)
        print("Loading the verified frozen model bundle...", flush=True)
        model = checked_model(refs["model_bundle"], spec, refs, deps)
        reproduction = reproduce_validation(model, refs, spec, freeze, freeze_module, args, deps)
        for path, wanted in protected.items():
            require(digest(path) == wanted, f"Input changed before test scoring: {path.name}.")
        attempt_path = args.output_dir / "evaluation_attempt.json"
        previous = read_json(attempt_path) if attempt_path.exists() else None
        attempt = {"status": "test_prediction_started", "inputs": config,
            "first_started_utc": previous["first_started_utc"] if previous else utc_now(),
            "latest_started_utc": utc_now(), "run_count": previous.get("run_count", 0)+1 if previous else 1,
            "validation_reproduction_complete": True, "eligible_rows_predicted_this_run": 0,
            "test_prediction_started": True, "model_refit": False, "test_guided_tuning": False}
        write_json(attempt_path, attempt)
        def progress(rows):
            attempt["eligible_rows_predicted_this_run"] = rows
            write_json(attempt_path, attempt)
            print(f"  Predicted {rows:,}/{len(table.y):,} eligible test pages.", flush=True)
        start = time.perf_counter()
        scores = predict(model, table, spec, args, deps, progress=progress)
        seconds = time.perf_counter()-start
        overall = {"eligible": metrics(table.y, scores, deps), "balanced": metrics(table.y[balanced], scores[balanced], deps)}
        diagnostics = quality_metrics(table, balanced, scores, deps)
        outputs = export_predictions(args.output_dir, table, balanced, scores, schema["quality_columns_not_classifier_inputs"])
        outputs["test_metrics_summary.csv"] = export_summary(args.output_dir, overall)
        for name, cohort in (("eligible_test_predictions.csv", "eligible"), ("balanced_test_predictions.csv", "balanced")):
            require(outputs[name]["rows"] == overall[cohort]["rows"], "Prediction-output row counts differ.")
        for path, wanted in protected.items():
            require(digest(path) == wanted, f"Input changed during evaluation: {path.name}. Do not treat partial results as final.")
        attempt.update(status="test_prediction_complete", completed_utc=utc_now())
        write_json(attempt_path, attempt)
        outputs[attempt_path.name] = {"sha256": digest(attempt_path)}
        report = {
            "status": "frozen_static_test_evaluation_complete", "script_version": VERSION, "created_utc": utc_now(),
            "inputs": config, "outputs": outputs, "model_name": spec["model_name"], "feature_count": 106,
            "threshold": THRESHOLD, "positive_class": "phish", "best_iteration": spec["best_iteration"],
            "metrics": overall, "quality_subset_metrics": diagnostics, "quality_subsets_are_overlapping": True,
            "validation_runtime_reproduction": reproduction, "test_prediction_seconds": seconds,
            "raw_release_test_rows": prepared["raw_test_rows"], "raw_release_class_counts": prepared["raw_test_label_counts"],
            "preparation_exclusive_decision_counts": prepared["exclusive_decision_counts"],
            "preparation_overlap_flag_counts": prepared["overlapping_audit_flag_counts_by_valid_label"],
            "coverage_percent_of_all_raw_rows": prepared["coverage_percent_of_all_raw_rows"],
            "quality_summary_by_cohort_and_label": extraction["quality_summary_by_cohort_and_label"],
            "cohort_membership_changed": False, "labels_changed": False, "quality_based_exclusions": 0,
            "model_loaded": True, "model_trained": False, "model_threshold_changed": False, "test_guided_tuning": False,
            "official_test_evaluated": True, "official_release_full_benchmark_evaluated": False,
            "test_evaluation_scope": "Frozen static baseline on predeclared filtered balanced and eligible unbalanced test cohorts.",
            "existing_dataset_or_model_modified": False, "websites_requested": False, "javascript_executed": False,
            "operational_alert_policy_selected": False, "dynamic_component_accuracy_evaluated": False,
            "raw_html_decoded_this_step": False,
            "limitations": [
                "These scores describe the filtered script-26 cohorts, not the complete official-release benchmark. Report raw counts, exclusions and coverage.",
                "The balanced and eligible cohorts overlap; they share predictions and are not two independent experiments.",
                "Balanced precision/accuracy and this filtered class ratio do not establish real-world deployment precision or prevalence.",
                "Group/content exclusions reduce known development overlap but do not prove campaign/template independence or independent page-level sampling.",
                "HTML/JS caps and static association limits remain. Quality-subset diagnostics overlap; absent parser errors do not prove capture completeness.",
                "Static archived features do not measure live crawling, external script contents, delayed execution or resistance to dynamic obfuscation.",
                "The fixed 0.50 comparison threshold does not guarantee any false-positive target or calibrated probability. Deployment alert policy remains separate work.",
                "Validation reproduction verifies runtime compatibility; validation was previously used for development and is not a new independent score.",
                "Report the test results as measured. Subsequent feature/model/threshold changes require a new independent holdout for final evaluation.",
                "Input hashes are checked before and after prediction; files are not held under a filesystem lock. Interrupted scoring restarts the unchanged frozen run, with an attempt record.",
            ],
        }
        write_json(args.output_dir / "static_test_evaluation_report.json", report)
    for cohort in ("balanced", "eligible"):
        m = report["metrics"][cohort]
        print(f"{cohort.capitalize()} test: rows={m['rows']:,}; accuracy={m['accuracy']:.4f}; precision={m['precision']:.4f}; recall={m['recall']:.4f}; F1={m['f1']:.4f}; FPR={m['false_positive_rate']:.4f}.", flush=True)
        print("  Confusion counts: " + json.dumps(m["confusion_matrix"], sort_keys=True), flush=True)
    print(f"Saved report: {(args.output_dir / 'static_test_evaluation_report.json').resolve()}", flush=True)
    print("Frozen static test evaluation complete. Model not retrained.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Stopped. Keep the attempt record; rerun the same command with unchanged code/model/features. A result is final only after the completion message.") from None
    except Exception as error:
        raise SystemExit("Frozen static evaluation stopped: " + str(error) + " A result is final only after the completion message.") from None
