"""Record the existing 106-feature static baseline before held-out evaluation.

Run beside scripts 09, 15 and 16, using the existing reports folders.
Requires only the Python standard library. Reads saved development artifacts;
does not import an estimator, load Parquet, train, execute JS or visit websites.

This freezes a STATIC BASELINE comparison at its existing threshold 0.50.
It does not finalize the dynamic-behaviour component or select a deployment
alert policy. Test extraction and inference are separate, later steps.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PureWindowsPath

VERSION = "1.0"
MODEL_NAME = "lightgbm_context_all_features"
THRESHOLD = 0.50
SEED = 42
CODE_SHA256 = {
    "base_extractor": "4c845262a0cb27c085a2fc80c4d37f427addad5405b7319ba20438f7e18ab7ef",
    "context_extractor": "62d468f83d124346cce3866894f85ae4bec4cdcd3163267b23625b61764d16eb",
    "trainer": "d685b0edf2ba248045ea39dfe9821ef6f9f477750468ed93fb2696a289096524",
}
PREDICTION_COLUMNS = [
    "record_id", "label", "site_group", "source_file",
    "sampled_similarity_candidate", "in_validation_sensitivity_subset",
    "phishing_score", "prediction_at_0_50", "error_type",
]
METRICS_CHECKED = (
    "accuracy", "balanced_accuracy", "precision", "recall", "f1",
    "false_positive_rate", "false_negative_rate",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_bytes(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")


def object_sha(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def expected_digest(report, filename):
    return report["outputs"][filename]["sha256"]


def original_input_digest(report, filename):
    # Earlier Windows reports store absolute paths; allow moving the project.
    values = [value for path, value in report["inputs"]["inputs_sha256"].items()
              if PureWindowsPath(path).name == filename]
    require(len(values) == 1, f"Missing/ambiguous original input digest: {filename}.")
    return values[0]


def class_counts(value, description):
    require(isinstance(value, dict) and set(value) == {"benign", "phish"},
            f"Unexpected classes in {description}.")
    require(all(type(n) is int and n > 0 for n in value.values()),
            f"Invalid counts in {description}.")
    require(value["benign"] == value["phish"], f"Classes are not balanced in {description}.")
    return value


def check_header(path, expected):
    with path.open(newline="", encoding="utf-8-sig") as file:
        require(next(csv.reader(file), None) == expected, f"Unexpected CSV columns: {path.name}.")


def checked_manifest(path, expected_counts):
    counts = {split: Counter() for split in ("train", "validation")}
    groups = {split: set() for split in counts}
    ids = {split: set() for split in counts}
    validation_metadata = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames))
                and {"sha256", "label", "site_group", "split", "source_file",
                     "sampled_similarity_candidate", "in_validation_sensitivity_subset"}.issubset(reader.fieldnames),
                "Balanced manifest must contain unique metadata columns.")
        for number, row in enumerate(reader, 1):
            require(None not in row and all(v is not None for v in row.values()),
                    f"Malformed manifest row {number}.")
            split, label, record_id, group = (row[k] for k in ("split", "label", "sha256", "site_group"))
            require(split in counts and label in ("benign", "phish") and record_id and group,
                    f"Invalid manifest metadata, row {number}.")
            require(record_id not in ids[split], f"Repeated record ID in {split}, row {number}.")
            ids[split].add(record_id)
            groups[split].add(group)
            counts[split][label] += 1
            require(row["source_file"] and row["sampled_similarity_candidate"] in ("0", "1")
                    and row["in_validation_sensitivity_subset"] in ("0", "1"),
                    f"Invalid manifest source or flags, row {number}.")
            if split == "validation":
                validation_metadata[record_id] = {
                    "record_id": record_id,
                    **{name: row[name] for name in PREDICTION_COLUMNS[1:6]},
                }
    require({split: dict(c) for split, c in counts.items()} == expected_counts,
            "Balanced manifest label counts differ from the reports.")
    require(not ids["train"].intersection(ids["validation"]), "Record IDs cross train/validation.")
    require(not groups["train"].intersection(groups["validation"]), "Site groups cross train/validation.")
    return ids, groups, validation_metadata


def threshold_metrics(matrix):
    tn, fp, fn, tp = (matrix[k] for k in ("tn", "fp", "fn", "tp"))
    benign, phish = tn + fp, tp + fn
    require(benign > 0 and phish > 0, "Both validation classes are required.")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / phish
    return {
        "accuracy": (tp + tn) / (benign + phish),
        "balanced_accuracy": (recall + tn / benign) / 2,
        "precision": precision, "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "false_positive_rate": fp / benign,
        "false_negative_rate": fn / phish,
    }


def checked_predictions(path, expected, validation_metadata):
    validation_ids = set(validation_metadata)
    matrix = Counter({"tn": 0, "fp": 0, "fn": 0, "tp": 0})
    labels, ids = Counter(), set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        require(reader.fieldnames == PREDICTION_COLUMNS, "Unexpected validation prediction columns.")
        for number, row in enumerate(reader, 1):
            require(None not in row and all(v is not None for v in row.values()),
                    f"Malformed validation prediction, row {number}.")
            record_id, label = row["record_id"], row["label"]
            require(record_id in validation_ids and record_id not in ids
                    and label in ("benign", "phish"),
                    f"Invalid/repeated validation prediction metadata, row {number}.")
            require(all(row[name] == validation_metadata[record_id][name] for name in PREDICTION_COLUMNS[:6]),
                    f"Validation prediction metadata differs from its manifest, row {number}.")
            require(row["sampled_similarity_candidate"] in ("0", "1")
                    and row["in_validation_sensitivity_subset"] in ("0", "1"),
                    f"Invalid validation subset flags, row {number}.")
            score = float(row["phishing_score"])
            require(math.isfinite(score) and 0 <= score <= 1,
                    f"Invalid phishing score, row {number}.")
            predicted = score >= THRESHOLD
            error = ("false_positive" if label == "benign" and predicted else
                     "false_negative" if label == "phish" and not predicted else "correct")
            require(row["prediction_at_0_50"] == ("phish" if predicted else "benign")
                    and row["error_type"] == error, f"Inconsistent decision flags, row {number}.")
            matrix["fp" if label == "benign" and predicted else
                   "tn" if label == "benign" else "tp" if predicted else "fn"] += 1
            labels[label] += 1
            ids.add(record_id)
    require(ids == validation_ids, "Validation predictions do not cover the exact balanced validation IDs.")
    require(labels == {"benign": expected["benign_rows"], "phish": expected["phishing_rows"]}
            and sum(labels.values()) == expected["rows"], "Validation prediction counts differ.")
    require(dict(matrix) == expected["confusion_matrix"] and expected["threshold"] == THRESHOLD,
            "Validation confusion matrix or threshold differs from the training report.")
    actual = threshold_metrics(matrix)
    for name, value in actual.items():
        require(math.isclose(value, expected[name], rel_tol=1e-10, abs_tol=1e-12),
                f"Validation metric differs: {name}.")
    return {"rows": len(ids), "confusion_matrix": dict(matrix), "threshold": THRESHOLD, **actual}


def test_protocol():
    return {
        "scope": "Static archived HTML/inline-JS baseline only; no live websites or dynamic-accuracy claim.",
        "positive_class": "phish", "decision_rule": "phishing_score >= 0.50", "seed": SEED,
        "model_refit": False, "test_guided_threshold_or_feature_selection": False,
        "official_split_only": "Use the released test-*.parquet split; never move development rows into test.",
        "selection_before_predictions": True,
        "report_population": "Record raw release counts and every exclusion reason before scoring.",
        "archive_eligibility": [
            "Valid supplied label and parseable HTTP(S) URL/site group under the existing grouping policy; no DNS lookup needed for archived analysis.",
            "Non-null, non-whitespace archived HTML; no current resources fetched to replace missing historical JS.",
            "Quarantine conflicting exact/whitespace-normalized content-label groups; do not relabel.",
            "Deterministically retain one same-label exact copy, ranked by seed and record ID.",
            "Keep parse/truncation/association-limit quality flags; do not exclude pages based on model scores.",
        ],
        "development_overlap_audit": [
            "Audit record IDs, private-suffix-aware site groups, exact HTML and whitespace-normalized HTML against development data.",
            "Primary clean cohort excludes test/development record, group and exact/normalized content overlaps.",
            "Keep exclusions and overlap counts; these filters do not prove campaign/template independence.",
        ],
        "primary_balanced_cohort": {
            "population": "Eligible development-disjoint test archives after the stated filters.",
            "per_class_target": "min(eligible benign count, eligible phish count); fail if either is zero.",
            "selection": "Within each label, sort by SHA256(UTF8(str(seed) + '|' + record_id)), then record_id; take the target count.",
            "replacement_or_oversampling": False,
            "interpretation": "A derived balanced, filtered cohort; do not call it the complete official-release benchmark.",
        },
        "secondary_unbalanced_cohort": {
            "population": "All eligible development-disjoint archives before class balancing.",
            "interpretation": "Preserves this filtered cohort's class ratio; it does not establish real deployment prevalence.",
        },
        "metrics": ["accuracy", "balanced_accuracy", "precision", "recall", "F1", "FPR",
                    "FNR", "TP/TN/FP/FN", "ROC AUC", "average precision"],
        "quality_reporting": "Report exclusions, coverage and class counts, plus metrics by capture-quality flags.",
        "after_test": "Report the results as measured. Further model changes require a new independent holdout.",
    }


def prepare(args):
    model_dir = args.training_dir / MODEL_NAME
    paths = {
        "training_report": args.training_dir / "form_context_training_report.json",
        "run_config": args.training_dir / "run_config.json",
        "model_result": model_dir / "model_result.json",
        "model_bundle": model_dir / "model_bundle.joblib",
        "validation_predictions": model_dir / "validation_predictions.csv",
        "context_report": args.features_dir / "form_context_feature_report.json",
        "context_schema": args.features_dir / "static_feature_schema.json",
        "context_train": args.features_dir / "train_static_features.csv",
        "context_validation": args.features_dir / "validation_static_features.csv",
        "context_sensitivity": args.features_dir / "validation_sensitivity_static_features.csv",
        "base_report": args.base_features_dir / "feature_extraction_report.json",
        "base_schema": args.base_features_dir / "static_feature_schema.json",
        "balance_report": args.balance_dir / "balance_report.json",
        "balanced_manifest": args.balance_dir / "balanced_training_validation_manifest.csv",
        "base_extractor": args.project_dir / "09_extract_static_features.py",
        "context_extractor": args.project_dir / "15_extract_form_context.py",
        "trainer": args.project_dir / "16_train_form_context_models.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    require(not missing, "Missing saved inputs:\n  " + "\n  ".join(missing))
    for directory in (args.training_dir, args.features_dir, args.base_features_dir, args.balance_dir):
        require(not args.output_dir.resolve().is_relative_to(directory.resolve())
                and not directory.resolve().is_relative_to(args.output_dir.resolve()),
                "Use a separate freeze output folder outside existing report folders.")
    require(not args.project_dir.resolve().is_relative_to(args.output_dir.resolve()),
            "The freeze output folder must not contain the project.")
    print("Checking saved model, feature files and provenance...", flush=True)
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in CODE_SHA256.items():
        require(hashes[name] == expected, f"Keep the exact original script: {paths[name].name}.")
    training = read_json(paths["training_report"])
    config = read_json(paths["run_config"])
    result = read_json(paths["model_result"])
    context = read_json(paths["context_report"])
    schema = read_json(paths["context_schema"])
    base = read_json(paths["base_report"])
    base_schema = read_json(paths["base_schema"])
    balance = read_json(paths["balance_report"])
    for report, status in ((training, "form_context_model_training_complete"),
                          (context, "form_context_feature_extraction_complete"),
                          (base, "static_feature_extraction_complete"),
                          (balance, "balanced_experimental_manifests_complete")):
        require(report.get("status") == status, f"Expected completed report: {status}.")
    require(training.get("official_test_evaluated") is False
            and context.get("official_test_evaluated") is False
            and training.get("websites_requested") is False
            and training.get("javascript_executed") is False,
            "This step expects the completed static development experiment, with test not evaluated.")
    require(training.get("shared_train_validation_record_ids") == 0
            and training.get("shared_train_validation_site_groups") == 0
            and balance.get("shared_train_validation_site_groups") == 0,
            "Development reports contain record/site-group overlaps.")
    require(training["run_config"] == config
            and hashes["run_config"] == training["run_config_sha256"],
            "Saved training configuration differs from the training report.")
    require(config["training_code_sha256"] == hashes["trainer"]
            and config["base_extractor_code_sha256"] == hashes["base_extractor"]
            and config["context_extractor_code_sha256"] == hashes["context_extractor"]
            and config["threshold"] == THRESHOLD and config["seed"] == SEED,
            "Unexpected original code, seed or threshold.")
    require(training["leading_validation_candidate"] == MODEL_NAME,
            "The preselected static baseline is not the reported leading validation candidate.")
    results = [r for r in training["model_results"] if r["model_name"] == MODEL_NAME]
    require(len(results) == 1 and results[0] == result, "Model result differs from its training report.")
    require(result["eligible_for_validation_ranking"] is True
            and result["reused_from_baseline"] is False
            and result["run_config_sha256"] == hashes["run_config"], "Unexpected selected model provenance.")
    require(result["outputs"]["model_bundle.joblib"]["sha256"] == hashes["model_bundle"]
            and result["outputs"]["validation_predictions.csv"]["sha256"] == hashes["validation_predictions"],
            "Saved model or validation predictions changed after training.")
    names = schema["feature_names"]
    require(len(names) == len(set(names)) == schema["feature_count"] == result["feature_count"] == 106
            and names == result["feature_names"] == config["model_features"][MODEL_NAME]
            and names == base_schema["feature_names"] + context["inputs"]["context_feature_names"],
            "The exact ordered 106-feature schema does not match the selected model.")
    require(len(base_schema["feature_names"]) == 86 and context["context_feature_count"] == 20
            and not set(names).intersection(schema["metadata_columns"] + schema["quality_columns_not_classifier_inputs"]),
            "Labels, IDs, groups and quality flags must remain outside the classifier features.")
    require(schema["implementation_code_sha256"] == hashes["context_extractor"]
            and schema["base_implementation_code_sha256"] == hashes["base_extractor"]
            and base_schema["implementation_code_sha256"] == hashes["base_extractor"]
            and schema["external_js_policy"] == "references_only_no_download"
            and schema["parameters"] == context["inputs"]["parameters"] == base["inputs"]["parameters"]
            and schema["association_limits"] == context["inputs"]["association_limits"],
            "Feature extraction limits or implementation contract differ.")
    extraction_versions = {"duckdb": context["inputs"]["duckdb_version"],
                           "tldextract": context["inputs"]["tldextract_version"]}
    require(extraction_versions == {"duckdb": base["inputs"]["duckdb_version"],
                                    "tldextract": base["inputs"]["tldextract_version"]},
            "Original and context extraction package versions differ.")
    for name in ("base_report", "base_schema", "context_report", "context_schema",
                 "context_train", "context_validation", "context_sensitivity"):
        require(config["input_hashes"][name] == hashes[name], f"Training input changed: {paths[name].name}.")
    for name in ("context_schema", "context_train", "context_validation", "context_sensitivity"):
        require(expected_digest(context, paths[name].name) == hashes[name],
                f"Feature output digest differs from extraction report: {paths[name].name}.")
    require(expected_digest(base, "static_feature_schema.json") == hashes["base_schema"]
            and expected_digest(balance, paths["balanced_manifest"].name) == hashes["balanced_manifest"]
            and context["inputs"]["input_hashes"]["manifest"] == hashes["balanced_manifest"]
            and original_input_digest(base, "balance_report.json") == hashes["balance_report"]
            and original_input_digest(base, paths["balanced_manifest"].name) == hashes["balanced_manifest"],
            "Balanced manifest/report provenance differs from extraction.")
    counts = {"train": class_counts(training["training_counts"], "training"),
              "validation": class_counts(training["main_validation_counts"], "validation")}
    require(counts == balance["balanced_counts"] == context["extracted_label_counts"] == base["extracted_label_counts"],
            "Development class counts differ between reports.")
    sensitivity = class_counts(training["validation_sensitivity_counts"], "validation sensitivity")
    require(sensitivity == balance["validation_sensitivity_counts"] == context["validation_sensitivity_counts"]
            == base["validation_sensitivity_counts"], "Sensitivity class counts differ between reports.")
    header = schema["metadata_columns"] + schema["quality_columns_not_classifier_inputs"] + names
    for name, split in (("context_train", "train"), ("context_validation", "validation"),
                        ("context_sensitivity", "sensitivity")):
        check_header(paths[name], header)
        expected_rows = sum((sensitivity if split == "sensitivity" else counts[split]).values())
        require(context["outputs"][paths[name].name]["rows"] == expected_rows,
                f"Reported feature rows differ: {paths[name].name}.")
    ids, groups, validation_metadata = checked_manifest(paths["balanced_manifest"], counts)
    print("Rechecking saved validation decisions at threshold 0.50...", flush=True)
    metrics = checked_predictions(paths["validation_predictions"], result["metrics"]["main_validation"],
                                  validation_metadata)
    installed = {}
    expected_packages = {**config["package_versions"], **extraction_versions}
    for package in expected_packages:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = None
    warnings = [f"{name}: trained with {wanted}; current environment reports {installed[name]}."
                for name, wanted in expected_packages.items() if installed[name] != wanted]
    specification = {
        "script_version": VERSION, "script_sha256": sha256_file(Path(__file__)),
        "model_name": MODEL_NAME, "model_bundle_sha256": hashes["model_bundle"],
        "best_iteration": result["best_iteration_or_solver_iterations"],
        "feature_names": names, "feature_count": len(names),
        "model_input_numeric_type": "float32",
        "feature_extraction_contract": schema, "threshold": THRESHOLD,
        "threshold_role": "Fixed static-baseline comparison; deployment alert policy is not selected.",
        "positive_class": "phish", "seed": SEED,
        "training_hyperparameters": config["lightgbm_hyperparameters"],
        "early_stopping": config["early_stopping"],
        "training_package_versions": config["package_versions"],
        "extraction_package_versions": extraction_versions,
        "training_python_version": config["python_version"],
        "input_files": {name: {"path": str(path.resolve()), "sha256": hashes[name]}
                        for name, path in paths.items()},
        "development_class_counts": counts, "validation_sensitivity_counts": sensitivity,
        "test_protocol": test_protocol(),
    }
    return specification, metrics, installed, warnings


def save_report(output_dir, report):
    path = output_dir / "static_baseline_freeze_report.json"
    # Never overwrite an earlier specification, partial output or unknown files.
    if output_dir.exists():
        require(output_dir.is_dir(), "The freeze output path is not a folder.")
        contents = list(output_dir.iterdir())
        if contents:
            require(contents == [path] and path.is_file(),
                    "Use an empty freeze folder; existing/partial outputs were left in place.")
            prior = read_json(path)
            comparison_keys = set(report) - {
                "created_utc", "current_python_version", "current_package_versions", "environment_version_warnings",
            }
            require(prior.get("status") == "static_baseline_frozen_test_not_evaluated"
                    and prior.get("specification_sha256") == object_sha(prior["specification"])
                    and all(prior.get(key) == report[key] for key in comparison_keys),
                    "An existing freeze differs. Keep it; use a new --output-dir before any test evaluation.")
            print("Reusing matching static-baseline specification.", flush=True)
            return path
    else:
        output_dir.mkdir(parents=True)
    # Exclusive creation makes simultaneous attempts fail without clobbering.
    with path.open("xb") as file:
        file.write(json_bytes(report))
    return path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=root)
    parser.add_argument("--training-dir", type=Path, default=root / "reports/form_context_training_v2")
    parser.add_argument("--features-dir", type=Path, default=root / "reports/features_context_v2")
    parser.add_argument("--base-features-dir", type=Path, default=root / "reports/features")
    parser.add_argument("--balance-dir", type=Path, default=root / "reports/balanced")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/static_baseline_freeze_v1")
    args = parser.parse_args()
    print(f"Static-baseline freeze v{VERSION}; standard library only.", flush=True)
    specification, metrics, installed, warnings = prepare(args)
    report = {
        "status": "static_baseline_frozen_test_not_evaluated",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "specification": specification, "specification_sha256": object_sha(specification),
        "verified_validation_metrics_at_0_50": metrics,
        "independently_recomputed_metrics": list(METRICS_CHECKED) + ["confusion_matrix", "class_counts"],
        "current_python_version": platform.python_version(),
        "current_package_versions": installed, "environment_version_warnings": warnings,
        "model_bundle_loaded": False, "model_runtime_inference_verified": False,
        "model_trained": False, "existing_model_modified": False,
        "official_test_evaluated": False, "official_test_files_opened": False,
        "websites_requested": False, "javascript_executed": False,
        "operational_alert_policy_selected": False,
        "dynamic_component_finalized": False,
        "limitations": [
            "This records an artifact specification and verifies reported file hashes; estimator loading and inference still need verification in the later evaluator.",
            "Threshold-dependent validation metrics are recomputed; AP/ROC AUC and training scores are not recomputed here.",
            "Validation was used repeatedly for development and early stopping; its scores are not final generalization claims.",
            "The test protocol creates a derived balanced cohort with exclusions. Coverage and unbalanced-cohort results must be reported too.",
            "Archived static features cannot establish resistance to dynamically loaded or delayed JavaScript on real websites.",
            "The 0.50 threshold remains the existing comparison baseline; it does not guarantee any FPR target or calibrated deployment probability.",
            "Specifying a protocol does not apply its test filters or prove the actual test is development-disjoint; the later audit must measure overlaps.",
        ],
    }
    path = save_report(args.output_dir, report)
    for message in warnings:
        print("Environment note: " + message, flush=True)
    print(f"Model: {MODEL_NAME}; features: 106; threshold: 0.50.", flush=True)
    print(f"Verified validation: accuracy={100*metrics['accuracy']:.2f}%; "
          f"recall={100*metrics['recall']:.2f}%; FPR={100*metrics['false_positive_rate']:.2f}%.", flush=True)
    print(f"Saved report: {path.resolve()}", flush=True)
    print("Static baseline freeze complete. Official test not evaluated.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        raise SystemExit("Static baseline freeze stopped: " + str(error)) from None
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Rerun after checking any partial output folder.") from None
