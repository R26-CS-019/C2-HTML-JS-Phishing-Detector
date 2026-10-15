"""Audit official test archives and prepare the predeclared static test cohorts.

Run after script 25, beside unchanged scripts 02, 05, 09 and 25.
Uses existing DuckDB and tldextract packages. No estimator is imported/loaded;
no scores, fitting, feature extraction, websites, DNS or JS execution occur.

Development references cover ALL rows in the completed script-05 fingerprint
index, including rows later withheld/quarantined. Test HTML is hashed inside
DuckDB, using exactly script 05's normalization, not transferred to Python.
Each test shard is a transaction/checkpoint; an interrupted shard restarts.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path, PureWindowsPath

VERSION = "1.0"
SEED = 42
LABELS = ("benign", "phish")
HELPER_SHA = {
    "25_freeze_static_baseline.py": "08ce9defa54b85ffa4ad5a0cbc2b951761d7fb4a4b0269820d2be34b8c4308c7",
    "02_audit_domains.py": "ec8ef827783f6677c954de95cb64f26e3427f8c596376170b57016eaa38b18c8",
    "05_audit_html_fingerprints.py": "9b6c5cd37ddf8d7612c94c143120a156505622db16b757aefbcd7d438716be06",
}
SQL = r"""
WITH normalized AS (
    SELECT file_row_number, sha256, url, label, html,
           trim(regexp_replace(html, '[[:space:]]+', ' ', 'g')) AS normalized_html
    FROM read_parquet(?, file_row_number=true)
)
SELECT file_row_number, sha256, url, label, length(html), sha256(html),
       sha256(normalized_html), html IS NULL,
       (normalized_html IS NULL OR normalized_html = '')
FROM normalized
"""
FLAGS = (
    "invalid_label", "invalid_record_id", "unresolved_url", "null_or_empty_html",
    "repeated_test_record_id", "conflicting_exact_labels", "conflicting_normalized_labels",
    "development_record_id_overlap", "development_site_group_overlap",
    "development_exact_html_overlap", "development_normalized_html_overlap",
    "same_label_exact_duplicate",
)
MANIFEST_COLUMNS = [
    "record_id", "label", "site_group", "source_file", "source_row_number",
    "url", "hostname", "domain_status", "html_characters",
    "raw_html_sha256", "normalized_html_sha256", "selection_rank_sha256",
    "in_balanced_test_cohort",
]
OUTPUT_NAMES = (
    "balanced_test_manifest.csv", "eligible_test_manifest.csv",
    "test_selection_decisions.jsonl", "test_preparation_report.json",
    "test_preparation_index.sqlite", "duckdb_temp",
)
HEX64 = re.compile(r"[0-9a-fA-F]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path):
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


def load_helper(path, expected_sha):
    require(path.is_file(), f"Missing original helper script: {path}.")
    require(digest_file(path) == expected_sha, f"Keep the unchanged original helper: {path.name}.")
    spec = importlib.util.spec_from_file_location("test26_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def frozen_reference_paths(args, specification):
    # Relocate references beneath the same project without changing the frozen
    # specification itself. PureWindowsPath also handles our POSIX test fixtures.
    original_root = PureWindowsPath(specification["input_files"]["trainer"]["path"]).parent
    result = {}
    for name, item in specification["input_files"].items():
        try:
            relative = PureWindowsPath(item["path"]).relative_to(original_root)
            require(".." not in relative.parts, "Invalid frozen relative input path.")
            path = args.project_dir.joinpath(*relative.parts)
        except ValueError:
            path = Path(item["path"])
        require(path.is_file(), f"Missing frozen development input: {path}.")
        require(digest_file(path) == item["sha256"], f"Frozen development input changed: {path}.")
        result[name] = path
    return result


def checked_inputs(args, duckdb, tldextract):
    freeze_module = load_helper(args.project_dir / "25_freeze_static_baseline.py",
                                HELPER_SHA["25_freeze_static_baseline.py"])
    freeze = read_json(args.freeze_report)
    specification = freeze["specification"]
    require(freeze.get("status") == "static_baseline_frozen_test_not_evaluated"
            and freeze.get("official_test_evaluated") is False
            and freeze_module.object_sha(specification) == freeze["specification_sha256"],
            "Use the completed, unchanged script-25 freeze report.")
    require(specification["script_sha256"] == HELPER_SHA["25_freeze_static_baseline.py"]
            and specification["test_protocol"] == freeze_module.test_protocol()
            and specification["model_name"] == freeze_module.MODEL_NAME
            and specification["feature_count"] == 106
            and specification["threshold"] == 0.50 and specification["seed"] == SEED,
            "Unexpected frozen static baseline or test-selection protocol.")
    require(duckdb.__version__ == specification["extraction_package_versions"]["duckdb"]
            and version("tldextract") == specification["extraction_package_versions"]["tldextract"],
            "Use the original DuckDB/tldextract versions recorded by script 25; do not upgrade before this audit.")
    print("Verifying frozen development artifacts...", flush=True)
    references = frozen_reference_paths(args, specification)
    base = read_json(references["base_report"])
    require(read_json(references["context_schema"]) == specification["feature_extraction_contract"],
            "Saved feature schema differs from the frozen specification.")
    expected_fp = freeze_module.original_input_digest(base, "html_fingerprint_report.json")
    require(args.fingerprint_report.is_file() and digest_file(args.fingerprint_report) == expected_fp,
            "The development fingerprint report differs from the one used for frozen feature extraction.")
    fp_report = read_json(args.fingerprint_report)
    require(fp_report.get("status") == "complete_exact_and_whitespace_fingerprint_audit",
            "The development fingerprint audit is not complete.")
    require(args.fingerprint_index.is_file(), "Missing script-05 html_fingerprint_index.sqlite.")
    require(args.development_manifest.is_file()
            and digest_file(args.development_manifest) == fp_report["inputs"]["manifest_sha256"],
            "Original development split manifest differs from its fingerprint report.")
    group_module = load_helper(args.project_dir / "02_audit_domains.py", HELPER_SHA["02_audit_domains.py"])
    fp_module = load_helper(args.project_dir / "05_audit_html_fingerprints.py", HELPER_SHA["05_audit_html_fingerprints.py"])
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(args.fingerprint_index) + suffix)
        require(not sidecar.exists() or sidecar.stat().st_size == 0,
                "Close development-index writers and finish their checkpoint before this audit.")
    index_sha = digest_file(args.fingerprint_index)
    development = sqlite3.connect(args.fingerprint_index.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        development.execute("PRAGMA cache_size=-32768")
        saved = development.execute("SELECT value FROM state WHERE key='config'").fetchone()
        require(saved and json.loads(saved[0]) == fp_report["inputs"], "Development index/report inputs differ.")
        total, incomplete, missing = development.execute("""SELECT count(*),
            sum(processed!=1),sum(html_empty=0 AND (raw_html_sha256 IS NULL OR normalized_html_sha256 IS NULL))
            FROM pages""").fetchone()
        require(total == fp_report["total_records_scanned"] and not incomplete and not missing,
                "Development fingerprint index is incomplete.")
        require(not development.execute("SELECT 1 FROM pages WHERE label IS NULL OR label NOT IN ('benign','phish') LIMIT 1").fetchone(),
                "Unexpected development-index label.")
        for column, key in (("raw_html_sha256", "raw_html_sha256_audit"),
                            ("normalized_html_sha256", "whitespace_normalized_html_sha256_audit")):
            require(fp_module.fingerprint_summary(development, column) == fp_report[key],
                    f"Development fingerprint summary differs: {column}.")
    finally:
        development.close()
    require(digest_file(args.fingerprint_index) == index_sha, "Development index changed during verification.")
    files = sorted(args.data_dir.glob("test-*.parquet"))
    require(files, f"No test-*.parquet files in {args.data_dir}.")
    stats = [{"name": path.name, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
             for path in files]
    config = {
        "script_version": VERSION, "code_sha256": digest_file(Path(__file__)),
        "freeze_report_sha256": digest_file(args.freeze_report),
        "frozen_specification_sha256": freeze["specification_sha256"],
        "fingerprint_report_sha256": expected_fp, "fingerprint_index_sha256": index_sha,
        "development_manifest_sha256": digest_file(args.development_manifest),
        "fingerprint_index_path": str(args.fingerprint_index.resolve()),
        "data_dir": str(args.data_dir.resolve()), "source_files": stats,
        "duckdb_version": duckdb.__version__, "tldextract_version": version("tldextract"),
        "grouping_policy": "Original script 02; offline bundled PSL with private suffixes; no DNS.",
        "normalization": fp_report["normalization"], "fingerprint_sql_sha256": hashlib.sha256(SQL.encode()).hexdigest(),
        "seed": SEED, "test_protocol": specification["test_protocol"],
        "development_scope": "All rows in the completed original development fingerprint index, including later withheld/quarantined rows.",
        "additional_structural_checks": "Missing/malformed supplied IDs and repeated supplied IDs are quarantined; ambiguous records are not assigned fabricated IDs.",
    }
    return config, files, references, fp_report, group_module


def check_output_scope(args, references):
    output = args.output_dir.resolve()
    data = args.data_dir.resolve()
    require(not output.is_relative_to(data) and not data.is_relative_to(output),
            "Use a separate output folder outside the dataset.")
    require(output != args.project_dir.resolve() and not args.project_dir.resolve().is_relative_to(output),
            "Output folder must not contain the project.")
    protected = {p.parent.resolve() for name, p in references.items()
                 if name not in ("base_extractor", "context_extractor", "trainer")}
    protected.update((args.freeze_report.parent.resolve(), args.fingerprint_index.parent.resolve()))
    require(not any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected),
            "Use a new test-preparation folder outside existing report/index folders.")
    if output.exists():
        require(output.is_dir(), "Output path is not a folder.")
        unknown = [p.name for p in output.iterdir() if p.name not in OUTPUT_NAMES
                   and p.name not in {name + ".tmp" for name in OUTPUT_NAMES}
                   and p.name not in {"test_preparation_index.sqlite-journal", "test_preparation_index.sqlite-wal", "test_preparation_index.sqlite-shm"}]
        require(not unknown, "Output folder contains unrelated files; choose a new --output-dir.")
        require(not any(output.iterdir()) or (output / "test_preparation_index.sqlite").is_file(),
                "Existing outputs have no matching checkpoint. Keep them; choose a new --output-dir.")
    else:
        output.mkdir(parents=True)


def init_index(db, args, config, references, fp_report):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sources(name TEXT PRIMARY KEY,sha256 TEXT NOT NULL,rows INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS test_pages(
            row_key TEXT PRIMARY KEY,source_file TEXT NOT NULL,source_row_number INTEGER NOT NULL,
            record_id TEXT,label TEXT,supplied_url TEXT,url TEXT,hostname TEXT,site_group TEXT,domain_status TEXT,
            html_characters INTEGER,raw_html_sha256 TEXT,normalized_html_sha256 TEXT,
            html_null INTEGER NOT NULL,html_empty INTEGER NOT NULL,selection_rank_sha256 TEXT NOT NULL,
            UNIQUE(source_file,source_row_number));
    """)
    saved = db.execute("SELECT value FROM state WHERE key='config'").fetchone()
    if saved:
        require(json.loads(saved[0]) == config,
                "Checkpoint inputs, code or source file metadata differ. Keep it; use a new --output-dir.")
        return
    require(not db.execute("SELECT 1 FROM sources LIMIT 1").fetchone()
            and not db.execute("SELECT 1 FROM test_pages LIMIT 1").fetchone(),
            "Unrecognized partial index; use a new --output-dir.")
    require(shutil.disk_usage(args.output_dir).free >= 2 * 1024**3,
            "Less than 2 GiB free for the audit index, manifests and temporary storage.")
    print("Preparing read-only references to all development fingerprints...", flush=True)
    uri = args.fingerprint_index.resolve().as_uri() + "?mode=ro"
    db.execute("ATTACH DATABASE ? AS development_source", [uri])
    try:
        db.execute("BEGIN")
        db.execute("CREATE TABLE development_ids(record_id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO development_ids SELECT sha256 FROM development_source.pages")
        db.execute("CREATE TABLE development_sites(site_group TEXT PRIMARY KEY)")
        db.execute("INSERT INTO development_sites SELECT DISTINCT site_group FROM development_source.pages WHERE site_group IS NOT NULL AND site_group!=''")
        for target, column in (("development_raw", "raw_html_sha256"),
                               ("development_normalized", "normalized_html_sha256")):
            db.execute(f"CREATE TABLE {target}(fingerprint TEXT PRIMARY KEY,labels INTEGER NOT NULL)")
            db.execute(f"""INSERT INTO {target}
                SELECT {column},sum(DISTINCT CASE label WHEN 'benign' THEN 1 ELSE 2 END)
                FROM development_source.pages WHERE html_empty=0 AND {column} IS NOT NULL
                GROUP BY {column}""")
        db.execute("CREATE TEMP TABLE frozen_selected(record_id TEXT PRIMARY KEY,label TEXT,split TEXT,site_group TEXT,source_file TEXT,raw TEXT,normalized TEXT)")
        fields = ("sha256", "label", "split", "site_group", "source_file", "raw_html_sha256", "normalized_html_sha256")
        count, batch = 0, []
        with references["balanced_manifest"].open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            require(reader.fieldnames and set(fields).issubset(reader.fieldnames), "Unexpected frozen balanced manifest columns.")
            for row in reader:
                batch.append(tuple(row[name] for name in fields)); count += 1
                if len(batch) >= 5000:
                    db.executemany("INSERT INTO frozen_selected VALUES (?,?,?,?,?,?,?)", batch); batch.clear()
            if batch:
                db.executemany("INSERT INTO frozen_selected VALUES (?,?,?,?,?,?,?)", batch)
        frozen = read_json(args.freeze_report)["specification"]
        require(count == sum(sum(v.values()) for v in frozen["development_class_counts"].values()),
                "Unexpected frozen development manifest row count.")
        mismatches = db.execute("""SELECT count(*) FROM frozen_selected s
            LEFT JOIN development_source.pages p ON s.record_id=p.sha256
            WHERE p.sha256 IS NULL OR s.label IS NOT p.label OR s.split IS NOT p.split OR s.site_group IS NOT p.site_group
            OR s.source_file IS NOT p.source_file OR s.raw IS NOT p.raw_html_sha256 OR s.normalized IS NOT p.normalized_html_sha256""").fetchone()[0]
        require(mismatches == 0, "Frozen selected records differ from the full development fingerprint index.")
        require(db.execute("SELECT count(*) FROM development_ids").fetchone()[0] == fp_report["total_records_scanned"],
                "Development-reference row count differs.")
        db.execute("INSERT INTO state VALUES ('config',?)", [json.dumps(config, sort_keys=True)])
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.execute("DETACH DATABASE development_source")


def scan_file(db, path, args, grouping, extractor, duckdb):
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    current_sha = digest_file(path)
    saved = db.execute("SELECT sha256,rows FROM sources WHERE name=?", [path.name]).fetchone()
    if saved:
        require(saved[0] == current_sha, f"Completed test shard bytes changed: {path.name}.")
        n = db.execute("SELECT count(*) FROM test_pages WHERE source_file=?", [path.name]).fetchone()[0]
        require(n == saved[1], f"Checkpoint rows differ for {path.name}.")
        print(f"  Verified checkpoint: {n:,} rows; skipping HTML decoding.", flush=True)
        return
    require(not db.execute("SELECT 1 FROM test_pages WHERE source_file=? LIMIT 1", [path.name]).fetchone(),
            f"Unexpected partial source rows in {path.name}.")
    con = duckdb.connect()
    count = 0
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET memory_limit=?", [args.memory_limit])
        con.execute("SET temp_directory=?", [str((args.output_dir / "duckdb_temp").resolve())])
        expected = con.execute("SELECT count(*) FROM read_parquet(?)", [str(path.resolve())]).fetchone()[0]
        cursor = con.execute(SQL, [str(path.resolve())])
        db.execute("BEGIN")
        while True:
            rows = cursor.fetchmany(256)
            if not rows:
                break
            batch = []
            for ordinal, record_id, supplied_url, label, length, raw, normalized, null, empty in rows:
                require(type(ordinal) is int and ordinal >= 0, "Invalid Parquet file-row number.")
                host, group, status = grouping.normalize_domain(supplied_url, extractor)
                value = str(supplied_url or "").strip()
                canonical = "https:" + value if value.startswith("//") else value if "://" in value else "https://" + value
                require(bool(null) or (raw is not None and normalized is not None and length is not None),
                        f"Missing non-null HTML fingerprints in {path.name}.")
                rid = str(record_id) if record_id is not None else None
                label = str(label) if label is not None else None
                rank = hashlib.sha256(f"{SEED}|{rid or ''}".encode("utf-8")).hexdigest()
                batch.append((f"{path.name}:{ordinal}", path.name, ordinal, rid, label, supplied_url,
                              canonical, host, group, status, length, raw, normalized, int(null), int(empty), rank))
            db.executemany("INSERT INTO test_pages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
            count += len(rows)
        coverage = db.execute("SELECT min(source_row_number),max(source_row_number),count(*) FROM test_pages WHERE source_file=?", [path.name]).fetchone()
        require(count == expected and (count == 0 or coverage == (0, count-1, count)),
                f"Test shard row coverage differs: {path.name}.")
        require(before == (path.stat().st_size, path.stat().st_mtime_ns), f"Test shard changed during scanning: {path.name}.")
        db.execute("INSERT INTO sources VALUES (?,?,?)", [path.name, current_sha, count])
        db.commit()
        print(f"  Saved shard checkpoint: {count:,} rows.", flush=True)
    except BaseException:
        db.rollback()
        raise
    finally:
        con.close()


def prepare_decisions(db):
    db.create_function("valid_id", 1, lambda value: int(isinstance(value, str) and HEX64.fullmatch(value) is not None), deterministic=True)
    db.executescript("""
        CREATE INDEX IF NOT EXISTS test_id_index ON test_pages(record_id);
        CREATE INDEX IF NOT EXISTS test_group_index ON test_pages(site_group);
        CREATE INDEX IF NOT EXISTS test_raw_index ON test_pages(raw_html_sha256);
        CREATE INDEX IF NOT EXISTS test_norm_index ON test_pages(normalized_html_sha256);
        DROP TABLE IF EXISTS decisions;
        DROP TABLE IF EXISTS repeated_ids;
        DROP TABLE IF EXISTS test_raw_labels;
        DROP TABLE IF EXISTS test_normalized_labels;
        CREATE TABLE repeated_ids AS SELECT record_id FROM test_pages
            WHERE record_id IS NOT NULL GROUP BY record_id HAVING count(*)>1;
        CREATE UNIQUE INDEX repeated_ids_index ON repeated_ids(record_id);
        CREATE TABLE test_raw_labels AS SELECT raw_html_sha256 AS fingerprint,
            sum(DISTINCT CASE label WHEN 'benign' THEN 1 ELSE 2 END) AS labels
            FROM test_pages WHERE label IN ('benign','phish') AND html_empty=0 GROUP BY raw_html_sha256;
        CREATE UNIQUE INDEX test_raw_labels_index ON test_raw_labels(fingerprint);
        CREATE TABLE test_normalized_labels AS SELECT normalized_html_sha256 AS fingerprint,
            sum(DISTINCT CASE label WHEN 'benign' THEN 1 ELSE 2 END) AS labels
            FROM test_pages WHERE label IN ('benign','phish') AND html_empty=0 GROUP BY normalized_html_sha256;
        CREATE UNIQUE INDEX test_norm_labels_index ON test_normalized_labels(fingerprint);
        CREATE TABLE decisions AS SELECT t.row_key,
            (t.label IS NULL OR t.label NOT IN ('benign','phish')) AS invalid_label,
            (valid_id(t.record_id)=0) AS invalid_record_id,
            (t.site_group IS NULL OR t.site_group='') AS unresolved_url,
            (t.html_null=1 OR t.html_empty=1) AS null_or_empty_html,
            (repeat.record_id IS NOT NULL) AS repeated_test_record_id,
            CASE WHEN t.html_empty=0 AND (tr.labels | coalesce(dr.labels,0))=3 THEN 1 ELSE 0 END AS conflicting_exact_labels,
            CASE WHEN t.html_empty=0 AND (tn.labels | coalesce(dn.labels,0))=3 THEN 1 ELSE 0 END AS conflicting_normalized_labels,
            (di.record_id IS NOT NULL) AS development_record_id_overlap,
            (ds.site_group IS NOT NULL) AS development_site_group_overlap,
            (dr.fingerprint IS NOT NULL) AS development_exact_html_overlap,
            (dn.fingerprint IS NOT NULL) AS development_normalized_html_overlap,
            0 AS same_label_exact_duplicate, '' AS decision, 0 AS in_balanced_test_cohort
        FROM test_pages t
        LEFT JOIN repeated_ids repeat ON t.record_id=repeat.record_id
        LEFT JOIN development_ids di ON t.record_id=di.record_id
        LEFT JOIN development_sites ds ON t.site_group=ds.site_group
        LEFT JOIN development_raw dr ON t.raw_html_sha256=dr.fingerprint
        LEFT JOIN development_normalized dn ON t.normalized_html_sha256=dn.fingerprint
        LEFT JOIN test_raw_labels tr ON t.raw_html_sha256=tr.fingerprint
        LEFT JOIN test_normalized_labels tn ON t.normalized_html_sha256=tn.fingerprint;
        CREATE UNIQUE INDEX decisions_key_index ON decisions(row_key);
        UPDATE decisions SET decision=CASE
            WHEN invalid_label THEN 'exclude_invalid_label'
            WHEN invalid_record_id THEN 'quarantine_invalid_record_id'
            WHEN unresolved_url THEN 'quarantine_unresolved_url'
            WHEN null_or_empty_html THEN 'exclude_null_or_empty_html'
            WHEN repeated_test_record_id THEN 'quarantine_repeated_record_id'
            WHEN conflicting_exact_labels THEN 'quarantine_conflicting_exact_labels'
            WHEN conflicting_normalized_labels THEN 'quarantine_conflicting_normalized_labels'
            ELSE 'archive_eligible' END;
        WITH ranked AS (
            SELECT t.row_key,row_number() OVER (PARTITION BY t.raw_html_sha256,t.label
                ORDER BY t.selection_rank_sha256,t.record_id,t.row_key) AS copy_rank
            FROM test_pages t JOIN decisions d ON t.row_key=d.row_key WHERE d.decision='archive_eligible'
        ) UPDATE decisions SET same_label_exact_duplicate=1,decision='exclude_same_label_exact_duplicate'
          WHERE row_key IN (SELECT row_key FROM ranked WHERE copy_rank>1);
        UPDATE decisions SET decision=CASE
            WHEN development_record_id_overlap THEN 'exclude_development_record_id_overlap'
            WHEN development_site_group_overlap THEN 'exclude_development_site_group_overlap'
            WHEN development_exact_html_overlap THEN 'exclude_development_exact_html_overlap'
            WHEN development_normalized_html_overlap THEN 'exclude_development_normalized_html_overlap'
            ELSE 'eligible' END WHERE decision='archive_eligible';
    """)
    eligible = {label: 0 for label in LABELS}
    for label, count in db.execute("SELECT t.label,count(*) FROM test_pages t JOIN decisions d ON t.row_key=d.row_key WHERE d.decision='eligible' GROUP BY t.label"):
        eligible[label] = count
    target = min(eligible.values())
    if target:
        for label in LABELS:
            db.execute("""UPDATE decisions SET in_balanced_test_cohort=1 WHERE row_key IN (
                SELECT t.row_key FROM test_pages t JOIN decisions d ON t.row_key=d.row_key
                WHERE d.decision='eligible' AND t.label=?
                ORDER BY t.selection_rank_sha256,t.record_id LIMIT ?)""", [label, target])
    db.commit()
    return eligible, target


def query_counts(db, condition):
    return {label: db.execute(f"""SELECT count(*) FROM test_pages t JOIN decisions d ON t.row_key=d.row_key
        WHERE t.label=? AND ({condition})""", [label]).fetchone()[0] for label in LABELS}


def checked_selection(db, eligible, target):
    chosen = query_counts(db, "d.in_balanced_test_cohort=1")
    require(chosen == {label: target for label in LABELS}, "Balanced selection counts failed.")
    require(not db.execute("""SELECT 1 FROM test_pages t JOIN decisions d ON t.row_key=d.row_key
        WHERE d.decision='eligible' AND (d.invalid_label OR d.invalid_record_id OR d.unresolved_url
        OR d.null_or_empty_html OR d.repeated_test_record_id OR d.conflicting_exact_labels
        OR d.conflicting_normalized_labels OR d.development_record_id_overlap OR d.development_site_group_overlap
        OR d.development_exact_html_overlap OR d.development_normalized_html_overlap OR d.same_label_exact_duplicate) LIMIT 1""").fetchone(),
        "An ineligible/overlapping test row remained selected.")
    require(query_counts(db, "d.decision='eligible'") == eligible, "Eligible class counts changed.")
    repeated = db.execute("""SELECT count(*) FROM (SELECT t.record_id FROM test_pages t
        JOIN decisions d ON t.row_key=d.row_key WHERE d.decision='eligible'
        GROUP BY t.record_id HAVING count(*)>1)""").fetchone()[0]
    require(repeated == 0, "Repeated record IDs remain in the eligible cohort.")
    copies = db.execute("""SELECT count(*) FROM (SELECT t.raw_html_sha256 FROM test_pages t
        JOIN decisions d ON t.row_key=d.row_key WHERE d.decision='eligible'
        GROUP BY t.raw_html_sha256 HAVING count(*)>1)""").fetchone()[0]
    require(copies == 0, "Exact HTML copies remain in the eligible cohort.")
    return chosen


def export_outputs(db, directory):
    outputs = {}
    columns = "t.record_id,t.label,t.site_group,t.source_file,t.source_row_number,t.url,t.hostname,t.domain_status,t.html_characters,t.raw_html_sha256,t.normalized_html_sha256,t.selection_rank_sha256,d.in_balanced_test_cohort"
    for filename, condition in (("balanced_test_manifest.csv", "d.in_balanced_test_cohort=1"),
                                ("eligible_test_manifest.csv", "d.decision='eligible'")):
        path = directory / filename; temporary = path.with_suffix(path.suffix + ".tmp")
        count = 0
        try:
            with temporary.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file); writer.writerow(MANIFEST_COLUMNS)
                for row in db.execute(f"SELECT {columns} FROM test_pages t JOIN decisions d ON t.row_key=d.row_key WHERE {condition} ORDER BY t.record_id"):
                    writer.writerow(row); count += 1
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        outputs[filename] = {"rows": count, "sha256": digest_file(path)}
    path = directory / "test_selection_decisions.jsonl"; temporary = path.with_suffix(path.suffix + ".tmp")
    fields = ["source_record_key", *MANIFEST_COLUMNS, "supplied_url", "decision", *FLAGS]
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            for row in db.execute(f"""SELECT t.row_key,{columns},t.supplied_url,d.decision,
                {','.join('d.'+name for name in FLAGS)} FROM test_pages t JOIN decisions d ON t.row_key=d.row_key
                ORDER BY t.source_file,t.source_row_number"""):
                file.write(json.dumps(dict(zip(fields, row)), ensure_ascii=True, allow_nan=False) + "\n"); count += 1
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    outputs[path.name] = {"rows": count, "sha256": digest_file(path)}
    return outputs


def build_report(db, args, config, fp_report, eligible, target, chosen, outputs):
    counts = dict(db.execute("SELECT coalesce(label,'<missing>'),count(*) FROM test_pages GROUP BY coalesce(label,'<missing>')"))
    reasons = dict(db.execute("SELECT decision,count(*) FROM decisions GROUP BY decision ORDER BY decision"))
    flags = {name: query_counts(db, "d." + name + "=1") for name in FLAGS}
    detail = [{"label": label, "decision": decision, "rows": count}
              for label, decision, count in db.execute("""SELECT t.label,d.decision,count(*)
                  FROM test_pages t JOIN decisions d ON t.row_key=d.row_key GROUP BY t.label,d.decision ORDER BY t.label,d.decision""")]
    total = db.execute("SELECT count(*) FROM test_pages").fetchone()[0]
    require(sum(counts.values()) == sum(reasons.values()) == total == outputs["test_selection_decisions.jsonl"]["rows"],
            "Raw-label/decision/output counts are inconsistent.")
    require(outputs["balanced_test_manifest.csv"]["rows"] == sum(chosen.values())
            and outputs["eligible_test_manifest.csv"]["rows"] == sum(eligible.values()), "Manifest counts differ.")
    groups = {"eligible": {}, "balanced": {}}
    for name, condition in (("eligible", "d.decision='eligible'"), ("balanced", "d.in_balanced_test_cohort=1")):
        for label in LABELS:
            groups[name][label] = db.execute(f"""SELECT count(DISTINCT t.site_group) FROM test_pages t
                JOIN decisions d ON t.row_key=d.row_key WHERE t.label=? AND ({condition})""", [label]).fetchone()[0]
    normalized = db.execute("""SELECT count(*) FROM (SELECT t.normalized_html_sha256
        FROM test_pages t JOIN decisions d ON t.row_key=d.row_key WHERE d.decision='eligible'
        GROUP BY t.normalized_html_sha256 HAVING count(*)>1)""").fetchone()[0]
    report = {
        "status": "static_test_cohorts_prepared_not_evaluated" if target else "static_test_audit_complete_no_balanced_cohort",
        "script_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": config, "source_file_snapshots": [{"name": n, "sha256": h, "rows": c} for n, h, c in db.execute("SELECT name,sha256,rows FROM sources ORDER BY name")],
        "last_run_memory_limit": args.memory_limit, "threads": 1,
        "python_version": platform.python_version(), "development_reference_rows": fp_report["total_records_scanned"],
        "raw_test_rows": total, "raw_test_label_counts": dict(counts),
        "exclusive_decision_counts": reasons, "decisions_by_label": detail,
        "overlapping_audit_flag_counts_by_valid_label": flags,
        "audit_flag_counts_are_not_exclusive": True,
        "eligible_unbalanced_counts": eligible, "balanced_test_counts": chosen,
        "per_class_balanced_target": target, "balanced_test_cohort_available": target > 0,
        "records_withheld_for_class_balance": {label: eligible[label]-chosen[label] for label in LABELS},
        "coverage_percent_of_all_raw_rows": {
            "eligible": 100 * sum(eligible.values()) / total if total else 0.0,
            "balanced": 100 * sum(chosen.values()) / total if total else 0.0,
        },
        "unique_site_groups_by_cohort_and_label": groups,
        "same_label_normalized_matching_groups_remaining_in_eligible_cohort": normalized,
        "outputs": outputs, "cohorts_selected_before_model_predictions": True,
        "model_loaded": False, "model_trained": False, "features_extracted": False,
        "official_test_files_opened": True, "official_test_labels_and_html_read": True,
        "official_test_evaluated": False, "model_threshold_changed": False,
        "existing_dataset_or_model_modified": False, "websites_requested": False,
        "javascript_executed": False, "test_accuracy": None,
        "limitations": [
            "This is metadata/fingerprint selection, not model evaluation. No test accuracy has been measured.",
            "The primary balanced cohort is filtered and development-disjoint under recorded checks; it is not the complete official-release benchmark.",
            "All development-index rows supply overlap references, including withheld/quarantined rows. This is more conservative than comparing only fitted training rows.",
            "Class balance is by pages, not by site groups, brands, languages or template families.",
            "URL eligibility reuses the original schemeless-URL and hostname-fallback grouping heuristics; it is not browser-navigation validation.",
            "Same-label whitespace-normalized variants remain within test. Exact/normalized equality does not detect general near-duplicates or establish campaign independence.",
            "Whitespace normalization can change meaningful text/JS strings; matches are conservatively excluded or quarantined, never relabelled.",
            "Flags can overlap; exclusive decisions use the recorded precedence. Sum exclusive decisions, not flags, to account for raw rows.",
            "Exact-copy representatives are chosen before development-overlap exclusion, according to the frozen protocol; an excluded representative is not replaced by a different copy.",
            "Archived content is analyzed offline. External JavaScript remains historical references, not recovered resources or observed live behaviour.",
            "Test-shard SHA-256 snapshots record the files actually scanned. During a new scan, size/mtime changes are checked; this is not a locked filesystem snapshot.",
        ],
    }
    return report


def reuse_complete(args, config):
    path = args.output_dir / "test_preparation_report.json"
    if not path.exists():
        return None
    report = read_json(path)
    require(report.get("inputs") == config and report.get("status") in (
        "static_test_cohorts_prepared_not_evaluated", "static_test_audit_complete_no_balanced_cohort"),
        "Existing completed report belongs to another configuration. Keep it; use a new --output-dir.")
    require(report.get("official_test_evaluated") is False and report.get("model_loaded") is False,
            "Unexpected model-evaluation state in the existing preparation report.")
    require(set(report["outputs"]) == {"balanced_test_manifest.csv", "eligible_test_manifest.csv",
            "test_selection_decisions.jsonl", "test_preparation_index.sqlite"},
            "Existing completed report does not list every required output.")
    require(report["balanced_test_counts"] == {label: report["per_class_balanced_target"] for label in LABELS}
            and report["balanced_test_cohort_available"] == (report["per_class_balanced_target"] > 0),
            "Existing completed report has inconsistent cohort counts.")
    for name, entry in report["outputs"].items():
        require(name in OUTPUT_NAMES and (args.output_dir / name).is_file()
                and digest_file(args.output_dir / name) == entry["sha256"],
                f"Completed output changed or is missing: {name}. Keep it; use a new --output-dir.")
    for source in report["source_file_snapshots"]:
        require(digest_file(args.data_dir / source["name"]) == source["sha256"],
                f"Completed source bytes changed: {source['name']}.")
    require({s["name"] for s in report["source_file_snapshots"]} == {s["name"] for s in config["source_files"]},
            "Completed source list differs.")
    return report


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=root)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\new dataset\phreshphish_dataset\data"))
    parser.add_argument("--freeze-report", type=Path, default=root / "reports/static_baseline_freeze_v1/static_baseline_freeze_report.json")
    parser.add_argument("--fingerprint-report", type=Path, default=root / "reports/html_fingerprints/html_fingerprint_report.json")
    parser.add_argument("--fingerprint-index", type=Path, default=root / "reports/html_fingerprints/html_fingerprint_index.sqlite")
    parser.add_argument("--development-manifest", type=Path, default=root / "reports/split/training_validation_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/test_preparation_v1")
    parser.add_argument("--memory-limit", default="4GB")
    args = parser.parse_args()
    try:
        import duckdb
        import tldextract
    except ImportError as error:
        raise ValueError("Run with the existing project .venv Python, which already contains DuckDB and tldextract.") from error
    print(f"Static test-cohort preparation v{VERSION}; no model predictions.", flush=True)
    config, files, references, fp_report, grouping = checked_inputs(args, duckdb, tldextract)
    check_output_scope(args, references)
    reused = reuse_complete(args, config)
    if reused:
        print("Verified and reused completed test preparation; no HTML decoded again.", flush=True)
        report = reused
    else:
        print(f"DuckDB memory limit: {args.memory_limit}; threads: 1; {len(files)} test files.", flush=True)
        path = args.output_dir / "test_preparation_index.sqlite"
        db = sqlite3.connect(path.resolve().as_uri(), uri=True)
        db.execute("PRAGMA cache_size=-32768")
        db.execute("PRAGMA temp_store=FILE")
        try:
            init_index(db, args, config, references, fp_report)
            extractor = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
            for number, source in enumerate(files, 1):
                print(f"Test fingerprints {number}/{len(files)}: {source.name}", flush=True)
                scan_file(db, source, args, grouping, extractor, duckdb)
            require(db.execute("SELECT count(*) FROM sources").fetchone()[0] == len(files), "Incomplete test-source checkpoints.")
            print("Checking conflicts, copies and development overlap; selecting cohorts...", flush=True)
            eligible, target = prepare_decisions(db)
            chosen = checked_selection(db, eligible, target)
            outputs = export_outputs(db, args.output_dir)
            report = build_report(db, args, config, fp_report, eligible, target, chosen, outputs)
            db.commit()
        finally:
            db.close()
        require(digest_file(args.fingerprint_index) == config["fingerprint_index_sha256"], "Development reference index changed during this run.")
        require(digest_file(args.freeze_report) == config["freeze_report_sha256"], "Frozen specification changed during this run.")
        after = [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sorted(args.data_dir.glob("test-*.parquet"))]
        require(after == config["source_files"], "Test source metadata changed during this run.")
        report["outputs"][path.name] = {"rows": report["raw_test_rows"], "sha256": digest_file(path)}
        write_json(args.output_dir / "test_preparation_report.json", report)
    print("Raw test counts: " + json.dumps(report["raw_test_label_counts"], sort_keys=True), flush=True)
    print("Eligible cohort: " + json.dumps(report["eligible_unbalanced_counts"], sort_keys=True), flush=True)
    print("Balanced cohort: " + json.dumps(report["balanced_test_counts"], sort_keys=True), flush=True)
    print(f"Saved report: {(args.output_dir / 'test_preparation_report.json').resolve()}", flush=True)
    if not report["balanced_test_cohort_available"]:
        raise SystemExit("Test audit complete, but a balanced cohort is unavailable. Review the exclusion report; no model evaluated.")
    print("Static test preparation complete. No model evaluated.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Stopped. Rerun the same command: completed shard checkpoints are retained.") from None
    except Exception as error:
        if type(error).__name__ == "OutOfMemoryException":
            raise SystemExit("Not enough RAM to decode the active test shard. Close memory-heavy apps and rerun the same command. Completed shards remain checkpointed. A report is complete only after the final completion message.") from None
        raise SystemExit("Static test preparation stopped: " + str(error) + " A report is complete only after the final completion message.") from None
