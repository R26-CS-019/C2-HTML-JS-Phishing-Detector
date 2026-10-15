"""Prepare a provisional manifest after exact/whitespace HTML overlap checks.

Standard library only. Reads script 05's SQLite index in read-only mode; does
not reread Parquet, alter dataset/index files, train a model, or balance classes.
Writes retained and excluded metadata manifests with an auditable reason for
each exclusion. General near-duplicate checks and 1:1 balancing remain pending.
"""
import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERSION = "1.0"
LABELS = ("benign", "phish")


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def match_counts(db, column, retained_only=False):
    if column not in ("raw_html_sha256", "normalized_html_sha256"):
        raise ValueError("Unexpected hash column")
    join = "JOIN decisions d ON p.sha256=d.sha256" if retained_only else ""
    eligible = "AND d.decision='eligible'" if retained_only else ""
    return db.execute(f"""SELECT count(*),coalesce(sum(labels>1),0),
        coalesce(sum(splits>1),0) FROM (
        SELECT p.{column},count(DISTINCT p.label) AS labels,count(DISTINCT p.split) AS splits
        FROM pages p {join} WHERE p.split IN ('train','validation') AND p.html_empty=0 {eligible}
        GROUP BY p.{column} HAVING count(*)>1)""").fetchone()


def prepare_decisions(db, seed):
    db.create_function("stable_rank", 1,
                       lambda value: hashlib.sha256(f"{seed}|exact-representative|{value}".encode()).hexdigest(),
                       deterministic=True)
    db.executescript("""
        CREATE TEMP TABLE raw_conflicts AS
        SELECT raw_html_sha256 AS fingerprint FROM pages
        WHERE split IN ('train','validation') AND html_empty=0
        GROUP BY raw_html_sha256 HAVING count(DISTINCT label)>1;
        CREATE INDEX raw_conflicts_index ON raw_conflicts(fingerprint);
        CREATE TEMP TABLE normalized_conflicts AS
        SELECT normalized_html_sha256 AS fingerprint FROM pages
        WHERE split IN ('train','validation') AND html_empty=0
        GROUP BY normalized_html_sha256 HAVING count(DISTINCT label)>1;
        CREATE INDEX normalized_conflicts_index ON normalized_conflicts(fingerprint);
        CREATE TEMP TABLE decisions (sha256 TEXT PRIMARY KEY,decision TEXT NOT NULL);
        INSERT INTO decisions
        SELECT p.sha256,CASE
            WHEN p.split='quarantine' THEN 'quarantine_unresolved_url'
            WHEN p.html_empty=1 OR p.html_null=1 THEN 'exclude_null_or_empty_html'
            WHEN r.fingerprint IS NOT NULL THEN 'quarantine_conflicting_exact_html_labels'
            WHEN n.fingerprint IS NOT NULL THEN 'quarantine_conflicting_normalized_html_labels'
            ELSE 'eligible' END
        FROM pages p
        LEFT JOIN raw_conflicts r ON p.raw_html_sha256=r.fingerprint
        LEFT JOIN normalized_conflicts n ON p.normalized_html_sha256=n.fingerprint;
        WITH ranked AS (
            SELECT p.sha256,row_number() OVER (
                PARTITION BY p.raw_html_sha256
                ORDER BY CASE p.split WHEN 'train' THEN 0 ELSE 1 END,stable_rank(p.sha256),p.sha256
            ) AS copy_rank
            FROM pages p JOIN decisions d ON p.sha256=d.sha256 WHERE d.decision='eligible'
        )
        UPDATE decisions SET decision='exclude_exact_duplicate_html'
        WHERE sha256 IN (SELECT sha256 FROM ranked WHERE copy_rank>1);
        CREATE TEMP TABLE training_normalized_fingerprints AS
        SELECT DISTINCT p.normalized_html_sha256 AS fingerprint
        FROM pages p JOIN decisions d ON p.sha256=d.sha256
        WHERE d.decision='eligible' AND p.split='train';
        CREATE INDEX training_normalized_index ON training_normalized_fingerprints(fingerprint);
        UPDATE decisions SET decision='exclude_normalized_validation_overlap'
        WHERE decision='eligible' AND sha256 IN (
            SELECT p.sha256 FROM pages p JOIN training_normalized_fingerprints t
            ON p.normalized_html_sha256=t.fingerprint WHERE p.split='validation'
        );
    """)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=root / "reports" / "split" / "training_validation_manifest.csv")
    parser.add_argument("--fingerprint-report", type=Path,
                        default=root / "reports" / "html_fingerprints" / "html_fingerprint_report.json")
    parser.add_argument("--fingerprint-index", type=Path,
                        default=root / "reports" / "html_fingerprints" / "html_fingerprint_index.sqlite")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "reports" / "prepared")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for path in (args.manifest,args.fingerprint_report,args.fingerprint_index):
        if not path.is_file():
            parser.error(f"Missing input: {path}. Finish scripts 04 and 05 first.")
    fingerprint_report = json.loads(args.fingerprint_report.read_text(encoding="utf-8-sig"))
    if fingerprint_report.get("status") != "complete_exact_and_whitespace_fingerprint_audit":
        parser.error("The fingerprint report is not a completed script 05 audit.")
    manifest_digest = digest_file(args.manifest)
    if manifest_digest != fingerprint_report["inputs"]["manifest_sha256"]:
        parser.error("Manifest changed after the fingerprint audit. Use the matching inputs.")
    report_digest = digest_file(args.fingerprint_report)
    # read-only main DB; all decision tables exist only in SQLite temporary storage.
    db = sqlite3.connect(args.fingerprint_index.resolve().as_uri() + "?mode=ro",uri=True)
    db.execute("PRAGMA cache_size=-32768")
    temporary_paths = []
    try:
        state = db.execute("SELECT value FROM state WHERE key='config'").fetchone()
        if not state or json.loads(state[0]) != fingerprint_report["inputs"]:
            raise ValueError("Fingerprint index and report have different inputs.")
        total,scanned,hash_missing = db.execute("""SELECT count(*),coalesce(sum(processed),0),
            sum(html_empty=0 AND (raw_html_sha256 IS NULL OR normalized_html_sha256 IS NULL)) FROM pages""").fetchone()
        if total != scanned or total != fingerprint_report["total_records_scanned"] or hash_missing:
            raise ValueError("Fingerprint index is incomplete or has missing hashes.")
        for column,key in (("raw_html_sha256","raw_html_sha256_audit"),
                           ("normalized_html_sha256","whitespace_normalized_html_sha256_audit")):
            actual = match_counts(db,column)
            expected = fingerprint_report[key]
            if actual != (expected["duplicate_groups"],expected["conflicting_label_groups"],expected["cross_split_groups"]):
                raise ValueError("Fingerprint index counts differ from the completed report.")
        print("Resolving exact copies, conflicting labels and normalized overlap...",flush=True)
        prepare_decisions(db,args.seed)
        raw_after = match_counts(db,"raw_html_sha256",True)
        norm_after = match_counts(db,"normalized_html_sha256",True)
        if any(raw_after) or norm_after[1] or norm_after[2]:
            raise ValueError("Fingerprint consistency checks failed after preparation.")
        shared_sites = db.execute("""SELECT count(*) FROM (
            SELECT p.site_group FROM pages p JOIN decisions d ON p.sha256=d.sha256
            WHERE d.decision='eligible' GROUP BY p.site_group HAVING count(DISTINCT p.split)>1)
        """).fetchone()[0]
        if shared_sites:
            raise ValueError("Website groups cross training/validation after preparation.")
        kept_counts = {split:{label:0 for label in LABELS} for split in ("train","validation")}
        for split,label,n in db.execute("""SELECT p.split,p.label,count(*) FROM pages p
            JOIN decisions d ON p.sha256=d.sha256 WHERE d.decision='eligible' GROUP BY p.split,p.label"""):
            kept_counts[split][label] = n
        group_counts = dict(db.execute("""SELECT p.split,count(DISTINCT p.site_group) FROM pages p
            JOIN decisions d ON p.sha256=d.sha256 WHERE d.decision='eligible' GROUP BY p.split"""))
        reasons = dict(db.execute("SELECT decision,count(*) FROM decisions GROUP BY decision"))
        reason_detail = [{"original_split":split,"label":label,"decision":decision,"rows":n}
                        for split,label,decision,n in db.execute("""SELECT p.split,p.label,d.decision,count(*)
                            FROM pages p JOIN decisions d ON p.sha256=d.sha256
                            GROUP BY p.split,p.label,d.decision ORDER BY p.split,p.label,d.decision""")]
        retained_total = sum(sum(counts.values()) for counts in kept_counts.values())
        args.output_dir.mkdir(parents=True,exist_ok=True)
        retained_path = args.output_dir / "clean_training_validation_manifest.csv"
        excluded_path = args.output_dir / "excluded_records_manifest.csv"
        if any(path.resolve() in (args.manifest.resolve(),args.fingerprint_report.resolve(),args.fingerprint_index.resolve())
               for path in (retained_path,excluded_path)):
            raise ValueError("Output paths must not overwrite inputs.")
        retained_tmp = retained_path.with_suffix(".csv.tmp")
        excluded_tmp = excluded_path.with_suffix(".csv.tmp")
        temporary_paths = [retained_tmp,excluded_tmp]
        written = Counter()
        lookup = db.cursor()
        with args.manifest.open(newline="",encoding="utf-8-sig") as source, \
                retained_tmp.open("w",newline="",encoding="utf-8") as retained, \
                excluded_tmp.open("w",newline="",encoding="utf-8") as excluded:
            reader = csv.DictReader(source)
            extra_fields = ["raw_html_sha256","normalized_html_sha256","html_characters","preparation_decision"]
            if any(field in reader.fieldnames for field in extra_fields):
                raise ValueError("Use the original script 04 manifest, not an already prepared manifest.")
            fields = reader.fieldnames + extra_fields
            retained_writer = csv.DictWriter(retained,fieldnames=fields)
            excluded_writer = csv.DictWriter(excluded,fieldnames=fields)
            retained_writer.writeheader();excluded_writer.writeheader()
            for row in reader:
                match = lookup.execute("""SELECT d.decision,p.raw_html_sha256,p.normalized_html_sha256,p.html_characters,
                    p.label,p.split,p.site_group,p.source_file FROM pages p JOIN decisions d ON p.sha256=d.sha256
                    WHERE p.sha256=?""",[row["sha256"]]).fetchone()
                if not match or tuple(row[key] for key in ("label","split","site_group","source_file")) != match[4:]:
                    raise ValueError("Input manifest row differs from the fingerprint index.")
                decision,raw,normalized,length = match[:4]
                out = dict(row,raw_html_sha256=raw,normalized_html_sha256=normalized,
                           html_characters=length,preparation_decision=decision)
                if decision == "eligible":
                    retained_writer.writerow(out);written["retained"] += 1
                else:
                    excluded_writer.writerow(out);written["excluded"] += 1
        if written["retained"] != retained_total or sum(written.values()) != total:
            raise ValueError("Output row counts do not match the fingerprint index.")
        if digest_file(args.manifest) != manifest_digest or digest_file(args.fingerprint_report) != report_digest:
            raise ValueError("An input file changed during preparation.")
        retained_tmp.replace(retained_path);excluded_tmp.replace(excluded_path)
        balanced_targets = {split:{"benign":min(counts.values()),"phish":min(counts.values())}
                            for split,counts in kept_counts.items()}
        report = {
            "created_utc":datetime.now(timezone.utc).isoformat(),"script_version":VERSION,"seed":args.seed,
            "status":"exact_filters_complete_general_near_duplicate_check_and_balancing_pending",
            "input_manifest_sha256":manifest_digest,"input_fingerprint_report_sha256":report_digest,
            "output_retained_manifest_sha256":digest_file(retained_path),
            "output_excluded_manifest_sha256":digest_file(excluded_path),
            "input_rows":total,"retained_rows":retained_total,"excluded_rows":written["excluded"],
            "retained_label_counts_by_split":kept_counts,"retained_group_counts_by_split":group_counts,
            "decisions_row_counts":reasons,"decisions_by_original_split_and_label":reason_detail,
            "remaining_exact_duplicate_groups":raw_after[0],
            "remaining_cross_split_raw_html_groups":raw_after[2],
            "remaining_cross_split_normalized_html_groups":norm_after[2],
            "remaining_normalized_matching_groups_within_splits":norm_after[0],
            "remaining_conflicting_exact_or_normalized_label_groups":raw_after[1]+norm_after[1],
            "shared_train_validation_site_groups":shared_sites,
            "balanced_subsets_created":False,"provisional_balanced_targets_if_no_more_exclusions":balanced_targets,
            "policy":[
                "Original unresolved-URL quarantine is retained.",
                "Null or whitespace-only captures are excluded from this HTML-focused experiment.",
                "All rows in exact or whitespace-normalized groups with both labels are set aside for manual review, without relabelling.",
                "For same-label exact copies, retain one row globally: prefer the original training split, then lowest seeded SHA-256 rank of the supplied identifier.",
                "A validation record whose normalized HTML matches retained training is excluded conservatively; normalized variants within one split remain for later similarity checks.",
                "No records move between website groups or from one original split to another.",
            ],
            "limitations":[
                "This is a provisional experimental subset, not a corrected release of the source dataset.",
                "Identical HTML with different labels can reflect different URL contexts; these records are not declared mislabelled.",
                "Whitespace can be meaningful, including inside JavaScript strings; normalized filtering is conservative and does not establish semantic equivalence.",
                "Training-preferred deduplication prunes validation, changing its distribution; report exclusions and evaluate the untouched official test separately.",
                "No title-based quality filtering was performed; title excerpts alone cannot verify capture failures.",
                "General near-duplicate/template leakage is still pending. Zero fingerprint matches is not proof of zero leakage.",
                "Class balancing is not performed yet. Final balanced counts can be smaller after subsequent similarity checks.",
                "Original Parquet files and the fingerprint index are not modified; no test content is read and no model is trained.",
            ],
        }
        report_path = args.output_dir / "preparation_report.json"
        report_tmp = report_path.with_suffix(".json.tmp")
        temporary_paths.append(report_tmp)
        report_tmp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
        report_tmp.replace(report_path)
        print(json.dumps(kept_counts,indent=2),flush=True)
        print(f"Retained: {retained_total:,}; set aside: {written['excluded']:,}",flush=True)
        print("Cross-split exact HTML matches: 0; normalized HTML matches: 0; website groups: 0",flush=True)
        print(f"Saved report: {report_path.resolve()}",flush=True)
        print(f"Saved retained manifest: {retained_path.resolve()}",flush=True)
        print(f"Saved exclusion manifest: {excluded_path.resolve()}",flush=True)
        print("HTML manifest preparation complete. General similarity checks and 1:1 balancing still pending.",flush=True)
    except (ValueError,sqlite3.Error) as error:
        raise SystemExit(str(error) + " No new completed preparation report was written.")
    finally:
        db.close()
        for path in temporary_paths:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
