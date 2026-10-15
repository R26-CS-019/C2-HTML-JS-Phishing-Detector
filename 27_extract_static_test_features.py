"""Extract the frozen 106 static features for the cohorts prepared by script 26.

Run beside unchanged scripts 02, 05, 09, 15, 25 and 26, using the existing
project virtual environment. This step never imports/loads an estimator,
predicts, fits, changes cohort membership, downloads resources or executes JS.
It reads archived HTML as text, transfers only the original capped prefix to
Python, and checkpoints every 100 pages. Rerun the same command to resume.
"""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import platform
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

VERSION = "1.0"
PREPARER_SHA = "fe5d87805a0f8b5c7102b560747550f33820373c17a92cb4221132bb9021de5e"
LABELS = ("benign", "phish")
CHECKPOINT_EVERY = 100
EXTRA_METADATA = ["source_row_number", "in_balanced_test_cohort"]
OUTPUT_NAMES = {
    "eligible_test_static_features.csv", "balanced_test_static_features.csv",
    "static_feature_schema.json", "test_feature_extraction_report.json",
    "test_feature_index.sqlite", "duckdb_temp",
}
# SQL identifiers are fixed; paths, ordinals and the prefix cap are parameters.
# Normalization is identical to script 05 and the completed script-26 audit.
SCAN_SQL = r"""
SELECT p.file_row_number,p.sha256,p.url,p.label,length(p.html),sha256(p.html),
       sha256(trim(regexp_replace(p.html,'[[:space:]]+',' ','g'))),
       substr(p.html,1,?)
FROM read_parquet(?,file_row_number=true) p
JOIN wanted w ON p.file_row_number=w.ordinal
"""
PAGE_COLUMNS = (
    "id", "label", "site_group", "source_file", "source_row_number",
    "page_url", "html_characters", "raw_sha", "normalized_sha", "balanced",
)
PAGE_QUERY = """SELECT t.record_id,t.label,t.site_group,t.source_file,t.source_row_number,
    t.url,t.html_characters,t.raw_html_sha256,t.normalized_html_sha256,d.in_balanced_test_cohort
    FROM audit.test_pages t JOIN audit.decisions d ON t.row_key=d.row_key
    WHERE d.decision='eligible'"""


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


def load_preparer(path):
    require(path.is_file() and digest(path) == PREPARER_SHA,
            "Use the unchanged original 26_prepare_static_test_cohorts.py that produced the completed audit.")
    spec = importlib.util.spec_from_file_location("test27_preparer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def checked_inputs(args, duckdb, tldextract):
    preparer = load_preparer(args.project_dir / "26_prepare_static_test_cohorts.py")
    prior_args = SimpleNamespace(project_dir=args.project_dir, data_dir=args.data_dir,
        freeze_report=args.freeze_report, fingerprint_report=args.fingerprint_report,
        fingerprint_index=args.fingerprint_index, development_manifest=args.development_manifest,
        output_dir=args.test_dir, memory_limit=args.memory_limit)
    # Reuse the exact original provenance checks. They hash model bytes without
    # unpickling, recompute metadata summaries, and check all completed outputs.
    prior_config, files, references, _, _ = preparer.checked_inputs(prior_args, duckdb, tldextract)
    print("Verifying test-preparation outputs and test-file SHA-256 checksums; this can take time...", flush=True)
    report = preparer.reuse_complete(prior_args, prior_config)
    require(report and report["status"] == "static_test_cohorts_prepared_not_evaluated"
            and report["balanced_test_cohort_available"] is True,
            "Finish script 26 successfully before test feature extraction.")
    freeze = read_json(args.freeze_report)
    specification = freeze["specification"]
    schema = specification["feature_extraction_contract"]
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        common = preparer.load_helper(references["base_extractor"], specification["input_files"]["base_extractor"]["sha256"])
        context = preparer.load_helper(references["context_extractor"], specification["input_files"]["context_extractor"]["sha256"])
    finally:
        sys.dont_write_bytecode = previous
    names = [*common.FEATURE_NAMES, *context.CONTEXT_NAMES]
    quality = [*common.QUALITY_COLUMNS, *context.CONTEXT_QUALITY]
    require(names == schema["feature_names"] == specification["feature_names"]
            and len(names) == len(set(names)) == 106,
            "Original extractors differ from the frozen ordered 106-feature list.")
    require(quality == schema["quality_columns_not_classifier_inputs"]
            and list(common.METADATA_COLUMNS) == schema["metadata_columns"]
            and context.LIMITS == schema["association_limits"]
            and {**common.FEATURE_DEFINITIONS, **context.CONTEXT_DEFINITIONS} == schema["feature_definitions"]
            and schema["external_js_policy"] == "references_only_no_download",
            "Original feature definitions, quality columns or extraction limits differ.")
    require(not set(names).intersection([*schema["metadata_columns"], *quality, *EXTRA_METADATA]),
            "Metadata and quality columns must remain outside classifier features.")
    require(all(type(schema["parameters"][name]) is int and schema["parameters"][name] > 0
                for name in ("max_html_characters", "max_js_characters")), "Invalid frozen extraction caps.")
    protected_files = {args.freeze_report: digest(args.freeze_report),
        args.test_dir / "test_preparation_report.json": digest(args.test_dir / "test_preparation_report.json"),
        args.test_dir / "test_preparation_index.sqlite": report["outputs"]["test_preparation_index.sqlite"]["sha256"],
        args.test_dir / "eligible_test_manifest.csv": report["outputs"]["eligible_test_manifest.csv"]["sha256"],
        args.test_dir / "balanced_test_manifest.csv": report["outputs"]["balanced_test_manifest.csv"]["sha256"],
        args.fingerprint_index: prior_config["fingerprint_index_sha256"],
    }
    config = {
        "script_version": VERSION, "code_sha256": digest(Path(__file__)),
        "preparer_code_sha256": PREPARER_SHA,
        "freeze_report_sha256": digest(args.freeze_report),
        "frozen_specification_sha256": freeze["specification_sha256"],
        "test_preparation_report_sha256": digest(args.test_dir / "test_preparation_report.json"),
        "test_preparation_output_hashes": report["outputs"],
        "data_dir": str(args.data_dir.resolve()), "source_files": prior_config["source_files"],
        "source_file_snapshots": report["source_file_snapshots"],
        "duckdb_version": prior_config["duckdb_version"], "tldextract_version": prior_config["tldextract_version"],
        "base_extractor_sha256": specification["input_files"]["base_extractor"]["sha256"],
        "context_extractor_sha256": specification["input_files"]["context_extractor"]["sha256"],
        "feature_names": names, "feature_count": len(names),
        "quality_columns_not_classifier_inputs": quality,
        "metadata_columns": [*schema["metadata_columns"], *EXTRA_METADATA],
        "parameters": schema["parameters"], "association_limits": schema["association_limits"],
        "external_js_policy": schema["external_js_policy"],
        "numeric_csv_policy": "Preserve extractor numeric values; the later frozen-model evaluator must load X as float32.",
        "scan_sql_sha256": hashlib.sha256(SCAN_SQL.encode()).hexdigest(),
        "checkpoint_every_pages": CHECKPOINT_EVERY,
        "eligible_test_counts": report["eligible_unbalanced_counts"], "balanced_test_counts": report["balanced_test_counts"],
        "cohort_membership_policy": "Reuse script-26 membership; quality flags never remove or rebalance pages.",
    }
    return config, files, references, schema, report, common, context, protected_files


def check_output_scope(args, references):
    output = args.output_dir.resolve()
    for path in (args.data_dir.resolve(), args.test_dir.resolve(), args.freeze_report.parent.resolve(),
                 args.fingerprint_index.parent.resolve(),
                 *(p.parent.resolve() for name, p in references.items()
                   if name not in ("base_extractor", "context_extractor", "trainer"))):
        require(not output.is_relative_to(path) and not path.is_relative_to(output),
                "Use a separate test-feature output folder outside existing datasets/report folders.")
    require(not args.project_dir.resolve().is_relative_to(output), "Output folder must not contain the project.")
    if output.exists():
        require(output.is_dir(), "Output path is not a folder.")
        allowed = OUTPUT_NAMES | {name + ".tmp" for name in OUTPUT_NAMES} | {
            "test_feature_index.sqlite-journal", "test_feature_index.sqlite-wal", "test_feature_index.sqlite-shm"}
        require(all(p.name in allowed for p in output.iterdir()), "Output folder contains unrelated files; choose a new --output-dir.")
        require(not any(output.iterdir()) or (output / "test_feature_index.sqlite").is_file(),
                "Existing outputs have no matching checkpoint; choose a new --output-dir.")
    else:
        output.mkdir(parents=True)


def verify_manifests(db, args, preparer_columns):
    columns = "t.record_id,t.label,t.site_group,t.source_file,t.source_row_number,t.url,t.hostname,t.domain_status,t.html_characters,t.raw_html_sha256,t.normalized_html_sha256,t.selection_rank_sha256,d.in_balanced_test_cohort"
    for filename, condition in (("eligible_test_manifest.csv", "d.decision='eligible'"),
                                ("balanced_test_manifest.csv", "d.in_balanced_test_cohort=1")):
        query = f"SELECT {columns} FROM audit.test_pages t JOIN audit.decisions d ON t.row_key=d.row_key WHERE {condition} ORDER BY t.record_id"
        with (args.test_dir / filename).open(newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            require(next(reader, None) == preparer_columns, f"Unexpected header: {filename}.")
            for actual, expected in itertools.zip_longest(reader, db.execute(query)):
                require(actual is not None and expected is not None
                        and actual == ["" if v is None else str(v) for v in expected],
                        f"Manifest membership/metadata differs from the completed test index: {filename}.")


def checkpoint_init(db, args, config):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS pages(
            id TEXT PRIMARY KEY,label TEXT NOT NULL,site_group TEXT NOT NULL,
            source_file TEXT NOT NULL,source_row_number INTEGER NOT NULL,page_url TEXT NOT NULL,
            html_characters INTEGER NOT NULL,raw_sha TEXT NOT NULL,normalized_sha TEXT NOT NULL,
            balanced INTEGER NOT NULL CHECK(balanced IN (0,1)),
            feature_values TEXT,quality_values TEXT,processed INTEGER NOT NULL DEFAULT 0 CHECK(processed IN (0,1)),
            UNIQUE(source_file,source_row_number));
        CREATE INDEX IF NOT EXISTS pages_source_index ON pages(source_file,processed);
    """)
    saved = db.execute("SELECT value FROM state WHERE key='config'").fetchone()
    if saved:
        require(json.loads(saved[0]) == config, "Checkpoint code or inputs changed. Keep it; use a new --output-dir.")
    else:
        require(not db.execute("SELECT 1 FROM pages LIMIT 1").fetchone(), "Unrecognized partial index; choose a new --output-dir.")
        require(shutil.disk_usage(args.output_dir).free >= 1024**3, "Less than 1 GiB free for feature outputs and temporary files.")
        db.execute("BEGIN")
        try:
            db.execute("INSERT INTO pages(" + ",".join(PAGE_COLUMNS) + ") " + PAGE_QUERY)
            db.execute("INSERT INTO state VALUES ('config',?)", [json.dumps(config, sort_keys=True)])
            db.commit()
        except BaseException:
            db.rollback()
            raise
    expected = "(" + PAGE_QUERY + ")"
    comparisons = " OR ".join("p." + name + " IS NOT e." + source for name, source in zip(PAGE_COLUMNS,
        ("record_id", "label", "site_group", "source_file", "source_row_number", "url",
         "html_characters", "raw_html_sha256", "normalized_html_sha256", "in_balanced_test_cohort")))
    require(not db.execute(f"SELECT 1 FROM pages p LEFT JOIN {expected} e ON p.id=e.record_id WHERE e.record_id IS NULL OR {comparisons} LIMIT 1").fetchone(),
            "Checkpoint membership/metadata differs from the frozen test cohort.")
    require(db.execute("SELECT count(*) FROM pages").fetchone()[0] == sum(config["eligible_test_counts"].values()),
            "Checkpoint selected-row coverage differs.")
    require(not db.execute("SELECT 1 FROM pages WHERE processed=1 AND (feature_values IS NULL OR quality_values IS NULL) LIMIT 1").fetchone(),
            "Completed checkpoint row is missing feature/quality values.")
    require(class_counts(db, "1") == config["eligible_test_counts"]
            and class_counts(db, "balanced=1") == config["balanced_test_counts"], "Checkpoint class counts differ.")


def class_counts(db, condition):
    return {label: db.execute(f"SELECT count(*) FROM pages WHERE label=? AND ({condition})", [label]).fetchone()[0] for label in LABELS}


def checked_values(features, quality, names, quality_names):
    require(list(features) == names and list(quality) == quality_names, "Extracted feature/quality names or order differ.")
    values = [features[n] for n in names]
    flags = [quality[n] for n in quality_names]
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values),
            "Non-finite, negative or nonnumeric feature value.")
    require(all(type(v) is int and v >= 0 for v in flags), "Invalid capture-quality value.")
    return values, flags


def scan_source(db, path, args, schema, common, context, resolver):
    rows = db.execute("""SELECT source_row_number,id,label,site_group,page_url,html_characters,raw_sha,normalized_sha
        FROM pages WHERE source_file=? AND processed=0""", [path.name]).fetchall()
    pending = {row[0]: tuple(row[1:]) for row in rows}
    if not pending:
        print("  Already checkpointed, or no eligible pages in this file.", flush=True)
        return
    print(f"  Pending eligible pages: {len(pending):,}; checkpoint every {CHECKPOINT_EVERY}.", flush=True)
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    con = common.duckdb.connect()
    seen = set()
    caps = schema["parameters"]
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET memory_limit=?", [args.memory_limit])
        con.execute("SET temp_directory=?", [str((args.output_dir / "duckdb_temp").resolve())])
        con.execute("CREATE TEMP TABLE wanted(ordinal BIGINT PRIMARY KEY)")
        con.executemany("INSERT INTO wanted VALUES (?)", [(ordinal,) for ordinal in pending])
        cursor = con.execute(SCAN_SQL, [caps["max_html_characters"], str(path.resolve())])
        db.execute("BEGIN")
        while (row := cursor.fetchone()) is not None:
            ordinal, record_id, url, label, length, raw, normalized, prefix = row
            require(ordinal in pending and ordinal not in seen, "Unexpected/repeated selected source-row number.")
            canonical = common.canonical_page_url(url)
            actual_group = resolver.resolve(canonical, canonical)[1]
            require((record_id, label, actual_group, canonical, length, raw, normalized) == pending[ordinal]
                    and isinstance(prefix, str),
                    f"Archived ID, label, URL or HTML fingerprints differ from the test audit: {path.name}:{ordinal}.")
            features, quality = context.extract_enhanced_static_features(prefix, canonical, common, resolver,
                max_html_characters=caps["max_html_characters"], max_js_characters=caps["max_js_characters"],
                full_html_characters=length)
            values, flags = checked_values(features, quality, schema["feature_names"], schema["quality_columns_not_classifier_inputs"])
            changed = db.execute("""UPDATE pages SET feature_values=?,quality_values=?,processed=1
                WHERE id=? AND source_file=? AND source_row_number=? AND processed=0""",
                [json.dumps(values, separators=(",", ":"), allow_nan=False),
                 json.dumps(flags, separators=(",", ":")), record_id, path.name, ordinal]).rowcount
            require(changed == 1, "Unexpected checkpoint update count.")
            seen.add(ordinal)
            if len(seen) % CHECKPOINT_EVERY == 0:
                require(before == (path.stat().st_size, path.stat().st_mtime_ns), "Active test file metadata changed.")
                db.commit()
                db.execute("BEGIN")
            if len(seen) % 500 == 0:
                print(f"  Extracted {len(seen):,}/{len(pending):,} pending pages.", flush=True)
        require(seen == set(pending), f"Selected test rows are missing from {path.name}.")
        require(before == (path.stat().st_size, path.stat().st_mtime_ns), "Active test file metadata changed.")
        db.commit()
        print(f"  Saved features for {len(seen):,} pages.", flush=True)
    except BaseException:
        db.rollback()
        raise
    finally:
        con.close()


def export_features(db, directory, schema):
    names = schema["feature_names"]
    qualities = schema["quality_columns_not_classifier_inputs"]
    header = [*schema["metadata_columns"], *EXTRA_METADATA, *qualities, *names]
    outputs, summaries = {}, {}
    tracked = {names.index("password_input_count"): "password_input_present_rows",
               names.index("script_inline_executable_count"): "inline_executable_script_present_rows"}
    tracked.update({i: n + "_present_rows" for i, n in enumerate(names) if n.startswith("context_")})
    for filename, cohort, condition in (("eligible_test_static_features.csv", "eligible", "1"),
                                       ("balanced_test_static_features.csv", "balanced", "balanced=1")):
        path = directory / filename; temporary = path.with_suffix(path.suffix + ".tmp")
        summaries[cohort] = {label: Counter() for label in LABELS}
        count = 0
        try:
            with temporary.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file); writer.writerow(header)
                for rid, label, group, source, ordinal, balanced, fv, qv in db.execute(f"""SELECT id,label,site_group,source_file,source_row_number,balanced,feature_values,quality_values
                    FROM pages WHERE processed=1 AND ({condition}) ORDER BY id"""):
                    values, flags = json.loads(fv), json.loads(qv)
                    checked_values(dict(zip(names, values)), dict(zip(qualities, flags)), names, qualities)
                    require(len(values) == len(names) and len(flags) == len(qualities), "Invalid stored feature/quality vector length.")
                    # The two development-only diagnostic memberships are zero
                    # for test, while the separate balanced-test marker is kept.
                    writer.writerow([rid, label, group, source, 0, 0, ordinal, balanced, *flags, *values])
                    summary = summaries[cohort][label]; summary["rows"] += 1
                    for n, value in zip(qualities, flags):
                        summary[n + "_rows"] += int(value > 0)
                    for index, key in tracked.items():
                        summary[key] += int(values[index] > 0)
                    count += 1
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        outputs[filename] = {"rows": count, "sha256": digest(path)}
    path = directory / "static_feature_schema.json"
    write_json(path, schema)  # Exact frozen schema, not a newly designed schema.
    outputs[path.name] = {"sha256": digest(path)}
    return outputs, summaries


def reuse_complete(args, config):
    path = args.output_dir / "test_feature_extraction_report.json"
    if not path.exists():
        return None
    report = read_json(path)
    require(report.get("status") == "static_test_feature_extraction_complete_not_evaluated"
            and report.get("inputs") == config,
            "Existing completed report differs. Keep it; use a new --output-dir.")
    require(report.get("official_test_evaluated") is False and report.get("model_loaded") is False
            and report.get("model_trained") is False and report.get("feature_count") == 106,
            "Unexpected evaluation state in the existing extraction report.")
    expected_names = OUTPUT_NAMES - {"test_feature_extraction_report.json", "duckdb_temp"}
    require(set(report["outputs"]) == expected_names, "Existing report is missing required output entries.")
    for name, entry in report["outputs"].items():
        require((args.output_dir / name).is_file() and digest(args.output_dir / name) == entry["sha256"],
                f"Completed feature output changed or is missing: {name}. Keep it; use a new --output-dir.")
    require(report["extracted_label_counts"] == {"eligible": config["eligible_test_counts"], "balanced": config["balanced_test_counts"]},
            "Existing extraction report has inconsistent class counts.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\new dataset\phreshphish_dataset\data"))
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--freeze-report", type=Path)
    parser.add_argument("--fingerprint-report", type=Path)
    parser.add_argument("--fingerprint-index", type=Path)
    parser.add_argument("--development-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--memory-limit", default="4GB")
    args = parser.parse_args()
    defaults = {
        "test_dir": "reports/test_preparation_v1", "freeze_report": "reports/static_baseline_freeze_v1/static_baseline_freeze_report.json",
        "fingerprint_report": "reports/html_fingerprints/html_fingerprint_report.json",
        "fingerprint_index": "reports/html_fingerprints/html_fingerprint_index.sqlite",
        "development_manifest": "reports/split/training_validation_manifest.csv", "output_dir": "reports/test_features_v1",
    }
    for name, relative in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, args.project_dir / relative)
    try:
        import duckdb
        import tldextract
    except ImportError as error:
        raise ValueError("Run with the existing project .venv Python containing DuckDB and tldextract.") from error
    print(f"Frozen static test-feature extraction v{VERSION}; 106 features; no model predictions.", flush=True)
    config, files, references, schema, prepared, common, context, protected_files = checked_inputs(args, duckdb, tldextract)
    check_output_scope(args, references)
    report = reuse_complete(args, config)
    if report:
        print("Verified and reused completed feature outputs; no HTML decoded again.", flush=True)
    else:
        print(f"DuckDB memory limit: {args.memory_limit}; threads: 1; HTML cap: {schema['parameters']['max_html_characters']:,}; JS cap: {schema['parameters']['max_js_characters']:,}.", flush=True)
        path = args.output_dir / "test_feature_index.sqlite"
        db = sqlite3.connect(path.resolve().as_uri(), uri=True)
        db.execute("PRAGMA cache_size=-32768")
        db.execute("PRAGMA temp_store=FILE")
        try:
            db.execute("ATTACH DATABASE ? AS audit", [(args.test_dir / "test_preparation_index.sqlite").resolve().as_uri() + "?mode=ro"])
            preparer = sys.modules["test27_preparer"]
            verify_manifests(db, args, preparer.MANIFEST_COLUMNS)
            checkpoint_init(db, args, config)
            resolver = common.ReferenceResolver()
            for number, source in enumerate(files, 1):
                print(f"Test HTML/JS features {number}/{len(files)}: {source.name}", flush=True)
                scan_source(db, source, args, schema, common, context, resolver)
            require(not db.execute("SELECT 1 FROM pages WHERE processed!=1 LIMIT 1").fetchone(), "Test features remain incomplete.")
            counts = {"eligible": class_counts(db, "processed=1"), "balanced": class_counts(db, "processed=1 AND balanced=1")}
            require(counts == {"eligible": config["eligible_test_counts"], "balanced": config["balanced_test_counts"]}, "Extracted class counts differ.")
            outputs, summaries = export_features(db, args.output_dir, schema)
            for filename, cohort in (("eligible_test_static_features.csv", "eligible"), ("balanced_test_static_features.csv", "balanced")):
                require(outputs[filename]["rows"] == sum(counts[cohort].values())
                        and {l: summaries[cohort][l]["rows"] for l in LABELS} == counts[cohort], "Export coverage/counts differ.")
            report = {
                "status": "static_test_feature_extraction_complete_not_evaluated", "script_version": VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(), "inputs": config, "outputs": outputs,
                "python_version": platform.python_version(), "last_run_memory_limit": args.memory_limit, "threads": 1,
                "feature_count": 106, "base_feature_count": 86, "context_feature_count": 20,
                "extracted_label_counts": counts, "quality_summary_by_cohort_and_label": summaries,
                "frozen_feature_schema_reused": True, "cohort_membership_changed": False, "labels_changed": False,
                "quality_based_exclusions": 0, "model_loaded": False, "model_trained": False,
                "official_test_files_opened": True, "official_test_features_extracted": True,
                "official_test_evaluated": False, "test_accuracy": None, "model_threshold_changed": False,
                "existing_dataset_or_model_modified": False, "websites_requested": False, "javascript_executed": False,
                "coverage_percent_of_all_raw_rows": prepared["coverage_percent_of_all_raw_rows"],
                "limitations": [
                    "These are feature outputs, not predictions or measured test accuracy.",
                    "The balanced and unbalanced cohorts are the filtered script-26 populations; report their exclusions and coverage with evaluation results.",
                    "The unbalanced cohort preserves this filtered population's class ratio, not real-world deployment prevalence.",
                    "Static HTML and inline-JS lexical cues do not establish live/delayed/obfuscated runtime behaviour. External JS remains references only.",
                    "Original HTML/JS caps and association limits remain. Quality flags are preserved and never used to remove or rebalance pages.",
                    "Metadata, labels, quality flags and cohort membership are outside the exact 106 classifier features. The later evaluator must use the frozen feature order and float32 X.",
                    "Full test-file checksums are verified before scanning; selected archive fingerprints/metadata are checked during extraction. Files are not held under a filesystem lock.",
                    "An interruption rolls back the active group of at most 100 pages; earlier committed groups resume with the unchanged code and inputs.",
                ],
            }
            db.commit()
            db.execute("DETACH DATABASE audit")
        finally:
            db.close()
        for protected, wanted in protected_files.items():
            require(digest(protected) == wanted, f"Preparation/frozen input changed during extraction: {protected.name}.")
        after = [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sorted(args.data_dir.glob("test-*.parquet"))]
        require(after == config["source_files"], "Test source metadata changed during extraction.")
        report["outputs"][path.name] = {"rows": sum(config["eligible_test_counts"].values()), "sha256": digest(path)}
        write_json(args.output_dir / "test_feature_extraction_report.json", report)
    print("Feature counts: " + json.dumps(report["extracted_label_counts"], sort_keys=True), flush=True)
    print(f"Saved report: {(args.output_dir / 'test_feature_extraction_report.json').resolve()}", flush=True)
    print("Static test feature extraction complete. No model evaluated.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Stopped. Rerun the same command; completed groups of 100 pages are checkpointed.") from None
    except Exception as error:
        if type(error).__name__ == "OutOfMemoryException":
            raise SystemExit("Not enough RAM to decode the active test file. Close memory-heavy apps and rerun the same command. Earlier page checkpoints are retained. A report is complete only after the final completion message.") from None
        raise SystemExit("Static test feature extraction stopped: " + str(error) + " A report is complete only after the final completion message.") from None
