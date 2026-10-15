"""Audit PhreshPhish training metadata and a reproducible HTML sample.

Requires only duckdb and Python's standard library. No JavaScript execution,
web requests, model training, or test HTML inspection is performed.
"""
import argparse
import ctypes
import csv
import json
import os
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import duckdb


def choose_memory_limit():
    """Budget half currently available RAM, capped at 4 GiB; no extra package."""
    available = None
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in (
                    "total_physical", "available_physical", "total_pagefile",
                    "available_pagefile", "total_virtual", "available_virtual",
                    "available_extended_virtual")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            available = status.available_physical
    elif hasattr(os, "sysconf"):
        try:
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            pass
    if available is None:
        return "2GB"
    megabytes = max(128, min(4096, int(available / (2 * 1024**2))))
    return f"{megabytes}MiB"


def open_database(memory_limit):
    con = duckdb.connect()
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit=?", [memory_limit])
    return con


class StructureParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.forms = 0
        self.passwords = 0
        self.external_scripts = 0
        self.inline_scripts = 0
        self.event_attributes = 0
        self.in_inline_script = False
        self.inline_has_text = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.event_attributes += sum(name.startswith("on") for name in attrs)
        if tag == "form":
            self.forms += 1
        if tag == "input" and (attrs.get("type") or "").lower() == "password":
            self.passwords += 1
        if tag == "script":
            src = (attrs.get("src") or "").strip()
            kind = (attrs.get("type") or "").strip().lower()
            is_js = kind in ("", "module", "text/javascript", "application/javascript",
                             "text/ecmascript", "application/ecmascript")
            if src and is_js:
                self.external_scripts += 1
            self.in_inline_script = is_js and not src
            self.inline_has_text = False

    def handle_data(self, data):
        if self.in_inline_script and data.strip():
            self.inline_has_text = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.finish_script()

    def finish_script(self):
        if self.in_inline_script and self.inline_has_text:
            self.inline_scripts += 1
        self.in_inline_script = False
        self.inline_has_text = False


def inspect_html(html):
    html = html or ""
    parser = StructureParser()
    error = ""
    try:
        parser.feed(html)
        parser.close()
        parser.finish_script()
    except Exception as exc:
        error = type(exc).__name__
    return {
        "html_characters": len(html),
        "empty_html": int(not html.strip()),
        "parse_error": error,
        "form_count": parser.forms,
        "password_input_count": parser.passwords,
        "inline_js_block_count": parser.inline_scripts,
        "external_js_reference_count": parser.external_scripts,
        "event_attribute_count": parser.event_attributes,
    }


def main():
    args_parser = argparse.ArgumentParser(description=__doc__)
    args_parser.add_argument("--data-dir", type=Path,
                             default=Path(r"D:\new dataset\phreshphish_dataset\data"))
    args_parser.add_argument("--output-dir", type=Path,
                             default=Path(__file__).resolve().parent / "reports" / "audit")
    args_parser.add_argument("--per-class", type=int, default=250)
    args_parser.add_argument("--seed", type=int, default=42)
    args_parser.add_argument("--memory-limit", default=None,
                             help="Override DuckDB memory budget, e.g. 2GB or 4GB; default: half available RAM, capped at 4GiB")
    args = args_parser.parse_args()
    if args.per_class < 1:
        args_parser.error("--per-class must be positive")
    data_dir = args.data_dir.resolve()
    train_files = sorted(data_dir.glob("train-*.parquet"))
    test_files = sorted(data_dir.glob("test-*.parquet"))
    if not train_files:
        raise SystemExit(f"No train-*.parquet files found in {data_dir}. Check --data-dir.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_limit = args.memory_limit or choose_memory_limit()
    con = open_database(memory_limit)
    print(f"DuckDB memory limit: {memory_limit}; threads: 1", flush=True)
    files = [str(path) for path in train_files]
    required = {"sha256", "url", "label", "date", "html"}
    for path in train_files:
        names = {row[0] for row in con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()}
        if not required.issubset(names):
            raise SystemExit(f"Missing columns in {path.name}: {sorted(required - names)}")
    print(f"Found {len(train_files)} training files and {len(test_files)} test files.", flush=True)
    print("Counting training labels and checking metadata...", flush=True)
    overview = con.execute("""
        SELECT label, count(*) AS rows,
               count(*) FILTER (WHERE sha256 IS NULL OR trim(sha256)='') AS missing_hash,
               count(*) FILTER (WHERE url IS NULL OR trim(url)='') AS missing_url,
               min(date), max(date)
        FROM read_parquet(?) GROUP BY label ORDER BY label
    """, [files]).fetchall()
    metadata = [{"label": row[0], "rows": row[1], "missing_hash": row[2],
                 "missing_url": row[3], "earliest_date": row[4], "latest_date": row[5]}
                for row in overview]
    print("Class counts:", {row[0]: row[1] for row in overview}, flush=True)
    duplicate_info = con.execute("""
        WITH groups AS (
            SELECT sha256, count(*) AS n, count(DISTINCT label) AS labels
            FROM read_parquet(?)
            WHERE sha256 IS NOT NULL AND trim(sha256)<>''
            GROUP BY sha256
        )
        SELECT count(*) FILTER (WHERE n>1),
               coalesce(sum(n-1) FILTER (WHERE n>1),0),
               count(*) FILTER (WHERE labels>1) FROM groups
    """, [files]).fetchone()
    print("Selecting a reproducible HTML sample (training only)...", flush=True)
    selected = con.execute("""
        WITH candidates AS (
            SELECT sha256, label, min(filename) AS source_file
            FROM read_parquet(?, filename=true)
            WHERE label IN ('benign','phish')
              AND sha256 IS NOT NULL AND trim(sha256)<>''
            GROUP BY sha256,label
        )
        SELECT sha256,label,source_file FROM candidates
        QUALIFY row_number() OVER (
            PARTITION BY label ORDER BY md5(sha256 || ?),sha256
        ) <= ?
        ORDER BY source_file,label,sha256
    """, [files, str(args.seed), args.per_class]).fetchall()
    selected_set = set(selected)
    grouped = defaultdict(list)
    for sha, label, source in selected:
        grouped[source].append(sha)
    fields = ["source_file", "sha256", "label", "html_characters", "empty_html",
              "parse_error", "form_count", "password_input_count",
              "inline_js_block_count", "external_js_reference_count", "event_attribute_count"]
    totals = defaultdict(Counter)
    lengths = defaultdict(list)
    csv_path = output_dir / "training_html_sample_audit.csv"
    seen = set()
    # Release metadata-phase buffers before the larger HTML decoding phase.
    con.close()
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for i, (source, hashes) in enumerate(grouped.items(), 1):
            print(f"HTML sample file {i}/{len(grouped)}: {Path(source).name}", flush=True)
            # A fresh reader per shard releases cached Parquet buffers between files.
            con = open_database(memory_limit)
            placeholders = ",".join("?" for _ in hashes)
            try:
                cursor = con.execute(
                    "SELECT sha256,label,html FROM read_parquet(?) "
                    f"WHERE sha256 IN ({placeholders}) AND label IN ('benign','phish')",
                    [source, *hashes])
            except duckdb.OutOfMemoryException:
                con.close()
                raise SystemExit(
                    f"Not enough memory to decode {Path(source).name} with budget {memory_limit}. "
                    "Close memory-heavy apps and rerun. If spare RAM permits, override with "
                    "--memory-limit 4GB. Dataset files have not been modified. "
                    "A completed JSON report is only valid after the final 'Audit complete' message.")
            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                sha, label, html = row
                if (sha, label) in seen or (sha,label,source) not in selected_set:
                    continue
                seen.add((sha,label))
                measures = inspect_html(html)
                writer.writerow({"source_file": Path(source).name, "sha256": sha,
                                 "label": label, **measures})
                output.flush()
                stats = totals[label]
                stats["samples"] += 1
                for key in ("empty_html", "parse_error", "form_count", "password_input_count",
                            "inline_js_block_count", "external_js_reference_count", "event_attribute_count"):
                    stats[key + "_present"] += int(bool(measures[key]))
                lengths[label].append(measures["html_characters"])
                del html, row
            con.close()
    sample_summary = {}
    for label, stats in totals.items():
        n = stats["samples"]
        sample_summary[label] = {"samples": n, "page_counts": dict(stats),
                                "page_percentages": {key: round(100*value/n, 2)
                                                     for key,value in stats.items() if key!="samples"},
                                "mean_html_characters": round(sum(lengths[label])/n, 1)}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(), "duckdb_version": duckdb.__version__,
        "script_version": "1.1", "duckdb_memory_limit": memory_limit, "duckdb_threads": 1,
        "data_dir": str(data_dir), "training_files": len(train_files),
        "test_file_names_count_only": len(test_files), "training_metadata": metadata,
        "supplied_hash_audit": {"repeated_hash_groups": duplicate_info[0],
                                "extra_rows_with_repeated_hash": duplicate_info[1],
                                "hash_groups_with_conflicting_labels": duplicate_info[2]},
        "html_sample_method": "Lowest MD5(sha256 || seed) per class, unique supplied hash/label pairs; one source file per pair.",
        "seed": args.seed, "requested_samples_per_class": args.per_class,
        "selected_pairs": len(selected), "inspected_pairs": len(seen),
        "training_html_sample": sample_summary,
        "limitations": [
            "Metadata counts cover all training rows; HTML measurements cover the sample only.",
            "The balanced sample does not represent natural phishing prevalence.",
            "Supplied hashes were not recomputed; near-duplicates and domain overlap are not audited here.",
            "HTML parser indicators are structural observations, not phishing labels or verified runtime behavior.",
            "Script references do not establish availability of external JavaScript content.",
            "No links were visited and no HTML or JavaScript was executed. Test contents were not read.",
            "Results are a data audit, not model performance or proof of content quality.",
        ],
    }
    report_path = output_dir / "training_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("HTML sample summary:", json.dumps(sample_summary, indent=2), flush=True)
    print(f"Saved report: {report_path}\nSaved sample audit: {csv_path}\nAudit complete.")


if __name__ == "__main__":
    main()
