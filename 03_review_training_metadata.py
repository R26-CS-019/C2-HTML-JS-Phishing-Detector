"""Inspect training metadata before deciding on exclusions or splitting.

Requires the already-installed duckdb and tldextract packages. Reads training
URL/label/target/date/hash columns only; never opens websites or reads HTML,
test data, or remote suffix lists. Writes one small JSON report.
"""
import argparse
import hashlib
import ipaddress
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import duckdb
import tldextract


def normalize_domain(url, extractor):
    # Same grouping rule as 02_audit_domains.py, version 1.0.
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
            status = "private_suffix_site" if result.is_private else "registered_domain"
            return hostname, "domain:" + result.top_domain_under_public_suffix, status
        return hostname, "host:" + hostname, "hostname_fallback"
    except (ValueError, UnicodeError):
        return "", "", "invalid_url"


def bounded_sample(samples, bucket, row, limit, seed):
    # Lowest seeded hashes give the same sample regardless of scan order.
    identity = json.dumps(row, sort_keys=True, ensure_ascii=True)
    rank = hashlib.sha256((str(seed) + "|" + identity).encode()).hexdigest()
    items = samples[bucket]
    if len(items) < limit or (rank, identity) < items[-1][:2]:
        items.append((rank, identity, row))
        items.sort(key=lambda item: item[:2])
        del items[limit:]


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--domain-report", type=Path,
                        default=root / "reports" / "domains" / "domain_audit_report.json")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "reports" / "metadata_review")
    parser.add_argument("--samples-per-label", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.samples_per_label <= 50:
        parser.error("--samples-per-label must be between 1 and 50")
    if not args.domain_report.is_file():
        parser.error(f"Missing domain report: {args.domain_report}. Run script 02 first.")
    audit = json.loads(args.domain_report.read_text(encoding="utf-8-sig"))
    expected_labels = audit["training_label_counts"]
    groups = sorted({entry["site_group"] for entry in audit["largest_training_groups"]})
    if not groups:
        parser.error("The domain report contains no groups to review.")
    files = sorted(args.data_dir.glob("train-*.parquet"))
    if not files:
        parser.error(f"No train-*.parquet files in {args.data_dir}")
    extractor = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None,
                                    include_psl_private_domains=True)
    selected = set(groups)
    samples = defaultdict(list)
    labels = Counter()
    statuses = Counter()
    group_labels = defaultdict(Counter)
    host_counts = defaultdict(Counter)
    path_counts = defaultdict(Counter)
    target_counts = defaultdict(Counter)
    invalid_counts = Counter()
    fallback_labels = Counter()
    con = duckdb.connect()
    con.execute("SET threads=1")
    con.execute("SET memory_limit='1GB'")
    con.execute("SET preserve_insertion_order=false")
    try:
        for index, path in enumerate(files, 1):
            print(f"Training metadata {index}/{len(files)}: {path.name}", flush=True)
            cursor = con.execute("SELECT sha256,url,label,target,date FROM read_parquet(?)", [str(path)])
            while True:
                batch = cursor.fetchmany(1000)
                if not batch:
                    break
                for sha, url, label, target, date in batch:
                    label = label if label is not None else "<missing>"
                    labels[label] += 1
                    host, group, status = normalize_domain(url, extractor)
                    statuses[status] += 1
                    row = {"sha256": sha, "url": url, "label": label,
                           "target": target, "date": date.isoformat() if date else None,
                           "source_file": path.name, "hostname": host,
                           "site_group": group, "domain_status": status}
                    if not group:
                        invalid_counts[status] += 1
                        bounded_sample(samples, ("unresolved", status), row, 50, args.seed)
                    elif status == "hostname_fallback":
                        fallback_labels[label] += 1
                        bounded_sample(samples, ("fallback", label), row,
                                       args.samples_per_label, args.seed)
                    if group not in selected:
                        continue
                    group_labels[group][label] += 1
                    host_counts[(group, label)][host] += 1
                    target_counts[(group, label)][str(target) if target is not None else "<missing>"] += 1
                    # Paths are summarized only for google.com; full URLs stay in samples.
                    if group == "domain:google.com":
                        value = url.strip()
                        parsed = urlsplit(value if "://" in value or value.startswith("//") else "//" + value)
                        path_counts[(group, label)][parsed.path or "/"] += 1
                        row["query_parameter_names"] = sorted({key for key, _ in parse_qsl(parsed.query)})
                    bounded_sample(samples, (group, label), row,
                                   args.samples_per_label, args.seed)
    finally:
        con.close()
    if dict(labels) != expected_labels or len(files) != audit["training_file_count"]:
        raise SystemExit("Training counts differ from script 02. No completed report written. Check --data-dir.")
    if dict(statuses) != audit["training_domain_status_counts"]:
        raise SystemExit("Domain grouping differs from script 02. No completed report written. Check package versions/data.")
    def top(counter):
        return [{"value": value, "rows": count}
                for value, count in sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:10]]
    reviews = []
    for group in groups:
        entry = {"site_group": group, "label_counts": dict(group_labels[group]), "by_label": {}}
        for label in sorted(group_labels[group]):
            entry["by_label"][label] = {
                "top_hostnames": top(host_counts[(group, label)]),
                "top_targets": top(target_counts[(group, label)]),
                "samples": [item[2] for item in samples[(group, label)]],
            }
            if group == "domain:google.com":
                entry["by_label"][label]["top_url_paths"] = top(path_counts[(group, label)])
        reviews.append(entry)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "script_version": "1.0",
        "python_version": platform.python_version(), "duckdb_version": duckdb.__version__,
        "tldextract_version": version("tldextract"), "seed": args.seed,
        "samples_per_group_and_label": args.samples_per_label,
        "training_file_count": len(files), "training_label_counts": dict(labels),
        "domain_audit_counts_match": True, "training_domain_status_counts": dict(statuses),
        "selected_group_reviews": reviews,
        "unresolved_url_counts": dict(invalid_counts),
        "unresolved_url_samples": [item[2] for key in sorted(samples) if key[0] == "unresolved"
                                   for item in samples[key]],
        "hostname_fallback_label_counts": dict(fallback_labels),
        "hostname_fallback_samples": [item[2] for key in sorted(samples) if key[0] == "fallback"
                                      for item in samples[key]],
        "limitations": [
            "Metadata review only: no HTML, JavaScript, test data, or websites were read.",
            "Targets and labels are supplied annotations, not independently verified findings.",
            "URL samples and path summaries cannot establish redirect destinations or page behavior.",
            "No rows were excluded, relabelled, split, or modified; no model was trained.",
            "Groups were selected from the earlier largest-group list, not to estimate dataset-wide error rates.",
            "Unresolved URL samples are capped at 50 per status; hostname fallback samples are bounded per label.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "training_metadata_review.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(output)
    print(f"Saved report: {output.resolve()}", flush=True)
    print("Metadata review complete.", flush=True)


if __name__ == "__main__":
    main()
