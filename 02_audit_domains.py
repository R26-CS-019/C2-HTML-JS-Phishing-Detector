"""Index training URLs and audit train/test metadata overlap.

Requirements: duckdb, tldextract. HTML, JavaScript, and test labels are not read.
No websites or remote Public Suffix Lists are requested. This creates manifests,
not a training split or a classifier. Run in the same project as the first audit.
"""
import argparse
import csv
import ipaddress
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

import duckdb
import tldextract


def normalize_domain(url, extractor):
    if not isinstance(url, str) or not url.strip():
        return "", "", "missing_url"
    value = url.strip()
    try:
        parsed = urlsplit(value if "://" in value or value.startswith("//") else "//" + value)
        if parsed.scheme and parsed.scheme.lower() not in ("http", "https"):
            return "", "", "unsupported_scheme"
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if not hostname or any(char.isspace() for char in hostname):
            return "", "", "invalid_hostname"
        try:
            address = ipaddress.ip_address(hostname)
            hostname = address.compressed
            return hostname, "ip:" + hostname, "ip_address"
        except ValueError:
            pass
        hostname = hostname.encode("idna").decode("ascii").lower()
        if len(hostname) > 253 or any(not part or len(part) > 63 for part in hostname.split(".")):
            return "", "", "invalid_hostname"
        result = extractor(hostname)
        if result.domain and result.suffix:
            domain = result.top_domain_under_public_suffix
            status = "private_suffix_site" if result.is_private else "registered_domain"
            return hostname, "domain:" + domain, status
        return hostname, "host:" + hostname, "hostname_fallback"
    except (ValueError, UnicodeError):
        return "", "", "invalid_url"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(r"D:\new dataset\phreshphish_dataset\data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "reports" / "domains")
    args = parser.parse_args()
    train = sorted(args.data_dir.glob("train-*.parquet"))
    test = sorted(args.data_dir.glob("test-*.parquet"))
    if not train:
        raise SystemExit(f"No training Parquet files found in {args.data_dir}.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    extractor = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None,
                                    include_psl_private_domains=True)
    con = duckdb.connect()
    con.execute("SET threads=1")
    con.execute("SET memory_limit='1GB'")
    con.execute("SET preserve_insertion_order=false")
    train_labels = Counter()
    train_status = Counter()
    train_groups = defaultdict(Counter)
    train_hashes = set()
    missing_hashes = 0
    repeated_hash_rows = 0
    manifest = output / "training_domain_manifest.csv"
    fields = ["sha256", "label", "date", "source_file", "hostname", "site_group", "domain_status"]
    # Per-shard queries bound metadata buffering and do not project HTML.
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, path in enumerate(train, 1):
            print(f"Training metadata {index}/{len(train)}: {path.name}", flush=True)
            cursor = con.execute("SELECT sha256,url,label,date FROM read_parquet(?)", [str(path)])
            while True:
                batch = cursor.fetchmany(1000)
                if not batch:
                    break
                for sha, url, label, date in batch:
                    host, group, status = normalize_domain(url, extractor)
                    label = label if label is not None else "<missing>"
                    train_labels[label] += 1
                    train_status[status] += 1
                    if group:
                        train_groups[group][label] += 1
                    if sha:
                        repeated_hash_rows += int(sha in train_hashes)
                        train_hashes.add(sha)
                    else:
                        missing_hashes += 1
                    writer.writerow({"sha256": sha or "", "label": label,
                                     "date": date.isoformat() if date else "",
                                     "source_file": path.name, "hostname": host,
                                     "site_group": group, "domain_status": status})
            file.flush()
    test_rows = 0
    test_status = Counter()
    test_groups = set()
    overlapping_groups = set()
    overlapping_hashes = set()
    test_domain_overlap_rows = 0
    test_hash_overlap_rows = 0
    for index, path in enumerate(test, 1):
        print(f"Test URL/hash integrity check {index}/{len(test)}: {path.name}", flush=True)
        cursor = con.execute("SELECT sha256,url FROM read_parquet(?)", [str(path)])
        while True:
            batch = cursor.fetchmany(1000)
            if not batch:
                break
            for sha, url in batch:
                test_rows += 1
                _, group, status = normalize_domain(url, extractor)
                test_status[status] += 1
                if group:
                    test_groups.add(group)
                    if group in train_groups:
                        test_domain_overlap_rows += 1
                        overlapping_groups.add(group)
                if sha and sha in train_hashes:
                    test_hash_overlap_rows += 1
                    overlapping_hashes.add(sha)
    counts_file = output / "training_domain_counts.csv"
    with counts_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["site_group", "total_rows", "benign_rows", "phish_rows", "other_label_rows"])
        for group in sorted(train_groups):
            counts = train_groups[group]
            n = sum(counts.values())
            writer.writerow([group, n, counts["benign"], counts["phish"],
                             n-counts["benign"]-counts["phish"]])
    top = sorted(train_groups.items(), key=lambda item: (-sum(item[1].values()), item[0]))[:10]
    overlap_train_rows = sum(sum(train_groups[group].values()) for group in overlapping_groups)
    total_train_rows = sum(train_labels.values())
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "script_version": "1.0",
        "python_version": platform.python_version(), "duckdb_version": duckdb.__version__,
        "tldextract_version": version("tldextract"), "data_dir": str(args.data_dir.resolve()),
        "training_file_count": len(train), "test_file_count": len(test),
        "training_rows": total_train_rows, "training_label_counts": dict(train_labels),
        "training_domain_status_counts": dict(train_status),
        "training_unique_site_groups": len(train_groups),
        "training_groups_with_multiple_rows": sum(sum(c.values())>1 for c in train_groups.values()),
        "training_groups_with_both_labels": sum(c["benign"]>0 and c["phish"]>0 for c in train_groups.values()),
        "training_missing_hash_rows": missing_hashes,
        "training_repeated_supplied_hash_rows": repeated_hash_rows,
        "largest_training_groups": [{"site_group": group, "rows": sum(c.values()),
                                     "labels": dict(c)} for group,c in top],
        "test_metadata_rows": test_rows, "test_domain_status_counts": dict(test_status),
        "test_unique_site_groups": len(test_groups),
        "train_test_metadata_overlap": {
            "check_performed": bool(test),
            "shared_site_groups": len(overlapping_groups),
            "training_rows_in_shared_site_groups": overlap_train_rows,
            "test_rows_in_shared_site_groups": test_domain_overlap_rows,
            "test_rows_in_shared_site_groups_percent": round(100*test_domain_overlap_rows/test_rows,2) if test_rows else None,
            "shared_supplied_hashes": len(overlapping_hashes),
            "test_rows_with_training_hash": test_hash_overlap_rows,
        },
        "grouping_method": "Normalized IDNA hostname; Public Suffix List including private suffixes; exact IP or hostname fallback.",
        "psl_source": "Bundled snapshot; cache and HTTP updates disabled. tldextract version recorded.",
        "limitations": [
            "Test URL/hash metadata were used only to inspect evaluation overlap. Test labels and HTML were not read.",
            "Shared domains do not automatically mean duplicate pages or invalid labels.",
            "A group with both labels can be legitimate, e.g. compromised or multi-tenant hosting.",
            "URL/domain groups do not identify identical owners, campaigns, or templates on different domains.",
            "Supplied hashes are compared as provided; HTML hashes were not recomputed.",
            "Unknown suffixes use the exact hostname and need review before splitting.",
            "No train/validation split, filtering decision, feature extraction, or model training is done here.",
            "This is not an audit of near-duplicate HTML or runtime JavaScript behavior.",
        ],
    }
    report_file = output / "domain_audit_report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    con.close()
    print("Training labels:", dict(train_labels), flush=True)
    print("Training site groups:", len(train_groups), flush=True)
    print("Train/test overlap:", json.dumps(report["train_test_metadata_overlap"], indent=2), flush=True)
    print(f"Saved report: {report_file}\nSaved manifest: {manifest}\nSaved counts: {counts_file}\nDomain audit complete.")


if __name__ == "__main__":
    main()
