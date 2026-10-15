"""Audit exact and whitespace-normalized HTML fingerprints, training data only.

Requires duckdb plus Python's standard library. DuckDB hashes HTML inside the
query so Python receives only small fingerprint records. Completed source files
are checkpointed in SQLite for resuming. This is NOT a general near-duplicate
detector, a browser, feature extraction, balancing, filtering, or training.
"""
import argparse
import csv
import hashlib
import json
import platform
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb

VERSION = "1.0"
SQL = r"""
WITH normalized AS (
    SELECT sha256, label, html,
           trim(regexp_replace(html, '[[:space:]]+', ' ', 'g')) AS normalized_html
    FROM read_parquet(?)
)
SELECT sha256, label, length(html), sha256(html), sha256(normalized_html),
       html IS NULL,
       (normalized_html IS NULL OR normalized_html = ''),
       left(regexp_extract(html, '(?is)<title[^>]*>(.*?)</title>', 1), 240)
FROM normalized
"""


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def init_index(db, manifest, split_report, config):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS pages (
            sha256 TEXT PRIMARY KEY, label TEXT NOT NULL, split TEXT NOT NULL,
            site_group TEXT NOT NULL, source_file TEXT NOT NULL, hostname TEXT,
            html_characters INTEGER, raw_html_sha256 TEXT, normalized_html_sha256 TEXT,
            html_null INTEGER, html_empty INTEGER, title_excerpt TEXT,
            processed INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS source_file_index ON pages(source_file);
    """)
    saved = db.execute("SELECT value FROM state WHERE key='config'").fetchone()
    if saved:
        if json.loads(saved[0]) != config:
            raise ValueError("Inputs differ from the existing fingerprint checkpoint. Use a new --output-dir to start a separate audit.")
        return
    if db.execute("SELECT count(*) FROM pages").fetchone()[0]:
        raise ValueError("Incomplete checkpoint metadata. Use a new --output-dir.")
    counts = {name: Counter() for name in ("train", "validation", "quarantine")}
    required = {"sha256", "label", "split", "site_group", "source_file", "hostname"}
    try:
        db.execute("BEGIN")
        with manifest.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("Input manifest is missing the required split metadata columns.")
            batch = []
            for row in reader:
                split, label = row["split"], row["label"]
                if split not in counts or label not in ("benign", "phish"):
                    raise ValueError("Unexpected split or label in the input manifest.")
                if not row["sha256"] or (split != "quarantine" and not row["site_group"]):
                    raise ValueError("Missing supplied hash or usable site group in the manifest.")
                counts[split][label] += 1
                batch.append(tuple(row[key] for key in (
                    "sha256", "label", "split", "site_group", "source_file", "hostname")))
                if len(batch) >= 5000:
                    db.executemany("INSERT INTO pages(sha256,label,split,site_group,source_file,hostname) VALUES (?,?,?,?,?,?)", batch)
                    batch.clear()
            if batch:
                db.executemany("INSERT INTO pages(sha256,label,split,site_group,source_file,hostname) VALUES (?,?,?,?,?,?)", batch)
        actual = {name: {label: counts[name][label] for label in ("benign", "phish")} for name in counts}
        if actual != split_report["label_counts_by_split"]:
            raise ValueError("Manifest counts differ from the split report.")
        shared = db.execute("""SELECT count(*) FROM (
            SELECT site_group FROM pages WHERE split IN ('train','validation')
            GROUP BY site_group HAVING count(DISTINCT split)>1)
        """).fetchone()[0]
        if shared:
            raise ValueError("The input manifest has website groups crossing train/validation.")
        expected_groups = dict(db.execute("SELECT split,count(DISTINCT site_group) FROM pages WHERE split IN ('train','validation') GROUP BY split"))
        if expected_groups != split_report["group_counts_by_split"]:
            raise ValueError("Manifest group counts differ from the split report.")
        db.execute("INSERT INTO state(key,value) VALUES ('config',?)", [json.dumps(config, sort_keys=True)])
        db.commit()
    except BaseException:
        db.rollback()
        raise


def scan_file(db, path, memory_limit):
    expected, completed = db.execute("SELECT count(*),coalesce(sum(processed),0) FROM pages WHERE source_file=?", [path.name]).fetchone()
    if completed == expected and expected:
        print(f"  Checkpoint found: skipping {expected:,} completed rows.", flush=True)
        return
    if not expected:
        raise ValueError(f"Dataset source file not represented in manifest: {path.name}")
    if completed:
        raise ValueError(f"Unexpected partial checkpoint for {path.name}; use a new --output-dir.")
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET memory_limit=?", [memory_limit])
        cursor = con.execute(SQL, [str(path)])
        db.execute("BEGIN")
        count = 0
        while True:
            batch = cursor.fetchmany(256)
            if not batch:
                break
            updates = [(chars, raw, normalized, int(null), int(empty), title,
                        sha, path.name, label)
                       for sha, label, chars, raw, normalized, null, empty, title in batch]
            before = db.total_changes
            db.executemany("""UPDATE pages SET html_characters=?,raw_html_sha256=?,
                normalized_html_sha256=?,html_null=?,html_empty=?,title_excerpt=?,processed=1
                WHERE sha256=? AND source_file=? AND label=? AND processed=0""", updates)
            if db.total_changes - before != len(batch):
                raise ValueError(f"Supplied identifiers/labels differ from manifest, or are repeated, in {path.name}.")
            count += len(batch)
        if count != expected:
            raise ValueError(f"Expected {expected:,} rows in {path.name}, read {count:,}.")
        db.commit()
        print(f"  Checkpoint saved: {count:,} rows.", flush=True)
    except BaseException:
        db.rollback()
        raise
    finally:
        con.close()


def fingerprint_summary(db, column):
    # Column names are internal constants, never supplied by the command line.
    if column not in ("raw_html_sha256", "normalized_html_sha256"):
        raise ValueError("Unexpected fingerprint column")
    grouped = f"""SELECT {column} AS fingerprint,count(*) AS n,
        sum(split='train') AS train_rows,sum(split='validation') AS validation_rows,
        count(DISTINCT label) AS labels,count(DISTINCT site_group) AS sites
        FROM pages WHERE split IN ('train','validation') AND html_empty=0
        GROUP BY {column} HAVING count(*)>1"""
    values = db.execute(f"""SELECT count(*),coalesce(sum(n),0),coalesce(sum(n-1),0),
        coalesce(sum(train_rows>0 AND validation_rows>0),0),
        coalesce(sum(CASE WHEN train_rows>0 AND validation_rows>0 THEN train_rows ELSE 0 END),0),
        coalesce(sum(CASE WHEN train_rows>0 AND validation_rows>0 THEN validation_rows ELSE 0 END),0),
        coalesce(sum(labels>1),0),coalesce(sum(sites>1),0) FROM ({grouped})""").fetchone()
    keys = ["duplicate_groups", "rows_in_duplicate_groups", "extra_rows_with_same_fingerprint",
            "cross_split_groups", "training_rows_in_cross_split_groups",
            "validation_rows_in_cross_split_groups", "conflicting_label_groups", "cross_site_groups"]
    summary = dict(zip(keys, values))
    # Prioritize cross-split matches and conflicting labels for review.
    candidates = db.execute(f"""SELECT * FROM ({grouped})
        ORDER BY (train_rows>0 AND validation_rows>0) DESC,(labels>1) DESC,n DESC,fingerprint
        LIMIT 10""").fetchall()
    summary["example_groups"] = []
    fields = ["supplied_sha256", "label", "split", "site_group", "source_file", "hostname", "html_characters", "title_excerpt"]
    for fp, n, train_rows, val_rows, labels, sites in candidates:
        examples = []
        for split in ("train", "validation"):
            members = db.execute(f"""SELECT sha256,label,split,site_group,source_file,hostname,html_characters,title_excerpt
                FROM pages WHERE {column}=? AND split=? AND html_empty=0 ORDER BY sha256 LIMIT 3""", [fp, split]).fetchall()
            examples.extend(dict(zip(fields, member)) for member in members)
        summary["example_groups"].append({"fingerprint": fp,"rows": n,
            "training_rows": train_rows,"validation_rows": val_rows,
            "distinct_labels": labels,"distinct_site_groups": sites,"examples": examples})
    return summary


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=root / "reports" / "split" / "training_validation_manifest.csv")
    parser.add_argument("--split-report", type=Path,
                        default=root / "reports" / "split" / "split_report.json")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "reports" / "html_fingerprints")
    parser.add_argument("--memory-limit", default="4GB")
    args = parser.parse_args()
    for path in (args.manifest, args.split_report):
        if not path.is_file():
            parser.error(f"Missing input: {path}. Run the corrected script 04 first.")
    split_report = json.loads(args.split_report.read_text(encoding="utf-8-sig"))
    if split_report.get("script_version") != "1.1":
        parser.error("Use the revised script 04 (version 1.1) split report.")
    files = sorted(args.data_dir.glob("train-*.parquet"))
    if not files:
        parser.error(f"No train-*.parquet files in {args.data_dir}")
    stats = [{"name": path.name,"bytes": path.stat().st_size,"mtime_ns": path.stat().st_mtime_ns}
             for path in files]
    config = {"algorithm_version": VERSION,"data_dir": str(args.data_dir.resolve()),
              "manifest_sha256": digest_file(args.manifest),
              "split_report_sha256": digest_file(args.split_report),"source_files": stats}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.output_dir / "html_fingerprint_index.sqlite"
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA cache_size=-32768")
    print(f"DuckDB memory limit: {args.memory_limit}; threads: 1", flush=True)
    try:
        print("Checking split manifest and fingerprint checkpoint...", flush=True)
        init_index(db, args.manifest, split_report, config)
        indexed_files = {row[0] for row in db.execute("SELECT DISTINCT source_file FROM pages")}
        if indexed_files != {path.name for path in files}:
            raise ValueError("Dataset training file names differ from the split manifest.")
        for index, path in enumerate(files, 1):
            print(f"HTML fingerprints {index}/{len(files)}: {path.name}", flush=True)
            scan_file(db, path, args.memory_limit)
        if digest_file(args.manifest) != config["manifest_sha256"] or digest_file(args.split_report) != config["split_report_sha256"]:
            raise ValueError("Split input changed during this run. A completed report was not written.")
        after_stats = [{"name": path.name,"bytes": path.stat().st_size,"mtime_ns": path.stat().st_mtime_ns}
                       for path in files]
        if after_stats != stats:
            raise ValueError("Dataset file metadata changed during this run. A completed report was not written.")
        total, processed = db.execute("SELECT count(*),sum(processed) FROM pages").fetchone()
        if total != processed or total != split_report["input_training_rows"]:
            raise ValueError("Incomplete scan or unexpected total record count.")
        print("Counting matching HTML fingerprints...", flush=True)
        db.execute("CREATE INDEX IF NOT EXISTS raw_hash_index ON pages(raw_html_sha256)")
        db.execute("CREATE INDEX IF NOT EXISTS normalized_hash_index ON pages(normalized_html_sha256)")
        db.commit()
        count_rows = db.execute("""SELECT split,label,count(*),sum(html_null),sum(html_empty),
            coalesce(sum(sha256=raw_html_sha256),0) FROM pages GROUP BY split,label ORDER BY split,label""").fetchall()
        fields = ["split","label","rows_scanned","null_html_rows","null_or_whitespace_only_html_rows",
                  "supplied_identifier_matches_raw_html_sha256_rows"]
        report = {"created_utc": datetime.now(timezone.utc).isoformat(),"script_version": VERSION,
                  "status": "complete_exact_and_whitespace_fingerprint_audit",
                  "python_version": platform.python_version(),"duckdb_version": duckdb.__version__,
                  "last_run_duckdb_memory_limit": args.memory_limit,
                  "inputs": config,"training_files_scanned": len(files),"total_records_scanned": total,
                  "counts_by_split_and_label": [dict(zip(fields,row)) for row in count_rows],
                  "raw_html_sha256_audit": fingerprint_summary(db,"raw_html_sha256"),
                  "whitespace_normalized_html_sha256_audit": fingerprint_summary(db,"normalized_html_sha256"),
                  "normalization": "Replace every run of ASCII whitespace with one space and trim edge spaces. Preserve case, tags, attributes, text and script content otherwise.",
                  "balanced_subsets_created": False,
                  "limitations": [
                      "All training HTML was fingerprinted, including the quarantined row. Match statistics use only non-empty train/validation records.",
                      "Only exact HTML and whitespace-normalized equality are tested; general near-duplicate/template similarity remains unchecked.",
                      "Whitespace normalization can alter meaningful text or JavaScript strings, so normalized matches require review rather than automatic relabelling.",
                      "SHA-256 equality is used as a practical content-equality check; matching payloads are not separately byte-compared.",
                      "Title excerpts are regex diagnostics, not independently verified page labels or browser parsing results.",
                      "No rows were removed or relabelled and no balanced dataset or model was created.",
                      "Official test contents were not read. No webpages were opened and no JavaScript executed.",
                      "Resume requires matching split input hashes and dataset path/file size/modification times; these metadata checks do not prove underlying file bytes are unchanged.",
                  ]}
        report_path = args.output_dir / "html_fingerprint_report.json"
        write_json_atomic(report_path, report)
        db.execute("INSERT OR REPLACE INTO state(key,value) VALUES ('last_complete_report_utc',?)", [report["created_utc"]])
        db.commit()
        print("Exact HTML cross-split matching groups:", report["raw_html_sha256_audit"]["cross_split_groups"], flush=True)
        print("Whitespace-normalized cross-split matching groups:", report["whitespace_normalized_html_sha256_audit"]["cross_split_groups"], flush=True)
        print(f"Saved report: {report_path.resolve()}", flush=True)
        print(f"Saved checkpoint index: {db_path.resolve()}", flush=True)
        print("HTML fingerprint audit complete. General near-duplicate check and 1:1 balancing still pending.", flush=True)
    except duckdb.OutOfMemoryException:
        raise SystemExit("Not enough memory for HTML decoding. Close memory-heavy applications and rerun the same command. Completed file checkpoints are retained; use a larger --memory-limit only if spare RAM permits. No new completed report was written.")
    except (ValueError, sqlite3.Error, duckdb.Error) as error:
        raise SystemExit(str(error) + " No new completed report was written.")
    except KeyboardInterrupt:
        raise SystemExit("Stopped. Rerun the same command to reuse completed file checkpoints. The active file will be scanned again.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
