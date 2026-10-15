"""Build reproducible 1:1 experimental subsets using script 06/07 outputs.

Standard library only. Does not read Parquet, execute HTML/JS, train a model,
relabel pages, or touch official test data. The second validation manifest is
a sensitivity analysis: sampled similarity candidates are not proven copies.
"""
import argparse
import csv
import hashlib
import json
import math
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERSION = "1.0"
LABELS = ("benign", "phish")
SPLITS = ("train", "validation")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_rank(seed, purpose, page_id):
    return hashlib.sha256(f"{seed}|{purpose}|{page_id}".encode()).hexdigest()


def counts(db, condition="1"):
    result = {s: {label: 0 for label in LABELS} for s in SPLITS}
    for split, label, n in db.execute(
            f"SELECT split,label,count(*) FROM pages WHERE {condition} GROUP BY split,label"):
        result[split][label] = n
    return result


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "reports/prepared/clean_training_validation_manifest.csv")
    parser.add_argument("--preparation-report", type=Path, default=root / "reports/prepared/preparation_report.json")
    parser.add_argument("--similarity-report", type=Path, default=root / "reports/similarity/html_similarity_report.json")
    parser.add_argument("--candidates", type=Path, default=root / "reports/similarity/validation_similarity_candidates.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/balanced")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    inputs = [args.manifest, args.preparation_report, args.similarity_report, args.candidates]
    for path in inputs:
        if not path.is_file():
            parser.error(f"Missing input: {path}. Complete scripts 06 and 07 first.")
    hashes = {str(p.resolve()): digest(p) for p in inputs}
    preparation = json.loads(args.preparation_report.read_text(encoding="utf-8-sig"))
    similarity = json.loads(args.similarity_report.read_text(encoding="utf-8-sig"))
    manifest_hash = hashes[str(args.manifest.resolve())]
    if manifest_hash != preparation["output_retained_manifest_sha256"]:
        raise ValueError("Prepared manifest differs from script 06 report.")
    if similarity["status"] != "sampled_similarity_candidate_audit_complete_not_exhaustive":
        raise ValueError("Script 07 must have completed successfully.")
    if similarity["inputs"]["manifest_sha256"] != manifest_hash or similarity["inputs"]["preparation_report_sha256"] != hashes[str(args.preparation_report.resolve())]:
        raise ValueError("Similarity report belongs to different prepared inputs.")
    if similarity["candidate_manifest_sha256"] != hashes[str(args.candidates.resolve())]:
        raise ValueError("Candidate CSV differs from the completed similarity report.")
    names = ["balanced_training_validation_manifest.csv", "balanced_validation_without_flagged_candidates.csv",
             "balance_decisions.csv", "balance_report.json"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / name for name in names]
    if any(p.resolve() in {i.resolve() for i in inputs} for p in outputs):
        raise ValueError("Outputs must not overwrite inputs.")
    # A report is valid only when the current run prints its completion message.
    temporary_outputs = []
    try:
        with tempfile.TemporaryDirectory(prefix="balance-", dir=args.output_dir) as temp_dir:
            db = sqlite3.connect(str(Path(temp_dir) / "metadata.sqlite"))
            try:
                db.execute("PRAGMA cache_size=-32768")
                db.execute("PRAGMA temp_store=FILE")
                db.executescript("""
                    CREATE TABLE pages(id TEXT PRIMARY KEY,label TEXT,split TEXT,site_group TEXT,
                        raw TEXT UNIQUE,norm TEXT,source_file TEXT,html_chars INTEGER,rank TEXT,
                        primary_selected INTEGER DEFAULT 0,sensitivity_selected INTEGER DEFAULT 0);
                    CREATE TABLE candidates(validation_id TEXT PRIMARY KEY,training_id TEXT,
                        validation_label TEXT,training_label TEXT,score REAL);
                """)
                print("Checking prepared metadata and similarity candidates...", flush=True)
                with args.manifest.open(newline="", encoding="utf-8-sig") as source:
                    reader = csv.DictReader(source)
                    fields = reader.fieldnames
                    required = {"sha256", "label", "split", "site_group", "source_file", "html_characters",
                                "raw_html_sha256", "normalized_html_sha256", "preparation_decision"}
                    if not fields or not required.issubset(fields):
                        raise ValueError("Use the retained manifest from script 06.")
                    for row in reader:
                        if row["split"] not in SPLITS or row["label"] not in LABELS or row["preparation_decision"] != "eligible":
                            raise ValueError("Unexpected label, split or decision in prepared manifest.")
                        if any(not row[k] for k in required) or int(row["html_characters"]) <= 0:
                            raise ValueError("Missing metadata or empty capture in prepared manifest.")
                        db.execute("INSERT INTO pages(id,label,split,site_group,raw,norm,source_file,html_chars,rank) VALUES (?,?,?,?,?,?,?,?,?)",
                                   (row["sha256"], row["label"], row["split"], row["site_group"], row["raw_html_sha256"],
                                    row["normalized_html_sha256"], row["source_file"], int(row["html_characters"]),
                                    stable_rank(args.seed, "primary-balance", row["sha256"])))
                input_counts = counts(db)
                total = sum(sum(c.values()) for c in input_counts.values())
                if input_counts != preparation["retained_label_counts_by_split"] or total != similarity["records_sketched"]:
                    raise ValueError("Prepared counts do not match audit reports.")
                for col in ("site_group", "norm"):
                    if db.execute(f"SELECT 1 FROM pages GROUP BY {col} HAVING count(DISTINCT split)>1 LIMIT 1").fetchone():
                        raise ValueError(f"Training/validation overlap in {col}.")
                if db.execute("SELECT 1 FROM pages GROUP BY norm HAVING count(DISTINCT label)>1 LIMIT 1").fetchone():
                    raise ValueError("Conflicting normalized HTML labels remain.")
                with args.candidates.open(newline="", encoding="utf-8-sig") as source:
                    for row in csv.DictReader(source):
                        v = db.execute("SELECT label,split,site_group,source_file,html_chars FROM pages WHERE id=?", (row["validation_sha256"],)).fetchone()
                        t = db.execute("SELECT label,split,site_group,source_file,html_chars FROM pages WHERE id=?", (row["training_sha256"],)).fetchone()
                        if v != (row["validation_label"], "validation", row["validation_site_group"], row["validation_source_file"], int(row["validation_html_characters"])) or t != (row["training_label"], "train", row["training_site_group"], row["training_source_file"], int(row["training_html_characters"])):
                            raise ValueError("Candidate metadata differs from prepared manifest.")
                        score = float(row["estimated_jaccard"])
                        if not math.isfinite(score) or not similarity["inputs"]["parameters"]["threshold"] <= score <= 1:
                            raise ValueError("Invalid candidate score.")
                        db.execute("INSERT INTO candidates VALUES (?,?,?,?,?)", (row["validation_sha256"], row["training_sha256"], v[0], t[0], score))
                candidate_counts = dict(db.execute("SELECT validation_label,count(*) FROM candidates GROUP BY validation_label"))
                if candidate_counts != similarity["validation_matches_by_label"] or sum(candidate_counts.values()) != similarity["validation_rows_with_candidate_match"]:
                    raise ValueError("Candidate counts differ from similarity report.")
                different_labels = db.execute("SELECT count(*) FROM candidates WHERE validation_label<>training_label").fetchone()[0]
                if different_labels != similarity["recorded_pairs_with_different_labels"]:
                    raise ValueError("Different-label candidate count differs from report.")
                db.execute("CREATE INDEX balance_rank ON pages(split,label,rank,id)")
                targets = {split: min(c.values()) for split, c in input_counts.items()}
                if min(targets.values()) <= 0:
                    raise ValueError("Both labels must be present in each split.")
                for split in SPLITS:
                    for label in LABELS:
                        db.execute("""UPDATE pages SET primary_selected=1 WHERE id IN
                            (SELECT id FROM pages WHERE split=? AND label=? ORDER BY rank,id LIMIT ?)""", (split, label, targets[split]))
                eligible = counts(db, "primary_selected=1 AND id NOT IN (SELECT validation_id FROM candidates)")["validation"]
                sensitivity_target = min(eligible.values())
                if sensitivity_target <= 0:
                    raise ValueError("No balanced validation subset remains after withholding candidates.")
                db.create_function("sensitivity_rank", 1, lambda value: stable_rank(args.seed, "sensitivity-balance", value), deterministic=True)
                for label in LABELS:
                    db.execute("""UPDATE pages SET sensitivity_selected=1 WHERE id IN
                        (SELECT id FROM pages WHERE split='validation' AND label=? AND primary_selected=1
                         AND id NOT IN (SELECT validation_id FROM candidates)
                         ORDER BY sensitivity_rank(id),id LIMIT ?)""", (label, sensitivity_target))
                db.commit()
                selected_counts = counts(db, "primary_selected=1")
                sensitivity_counts = counts(db, "sensitivity_selected=1")["validation"]
                if any(len(set(c.values())) != 1 for c in selected_counts.values()) or len(set(sensitivity_counts.values())) != 1:
                    raise ValueError("Balancing failed.")
                extras = ["sampled_similarity_candidate", "sampled_similarity_training_sha256", "sampled_similarity_estimate",
                          "sampled_similarity_training_label", "similarity_training_example_in_balanced_train",
                          "balance_decision", "in_validation_sensitivity_subset"]
                if set(extras).intersection(fields):
                    raise ValueError("Use the prepared manifest, not an already balanced output.")
                temporary_outputs = [p.with_suffix(p.suffix + ".tmp") for p in outputs]
                primary_candidate_counts = Counter()
                written = [0, 0, 0]
                with args.manifest.open(newline="", encoding="utf-8-sig") as source, \
                        temporary_outputs[0].open("w", newline="", encoding="utf-8") as primary, \
                        temporary_outputs[1].open("w", newline="", encoding="utf-8") as secondary, \
                        temporary_outputs[2].open("w", newline="", encoding="utf-8") as decisions:
                    writers = [csv.DictWriter(f, fieldnames=fields + extras) for f in (primary, secondary, decisions)]
                    for writer in writers:
                        writer.writeheader()
                    for row in csv.DictReader(source):
                        selected, sensitivity_selected = db.execute("SELECT primary_selected,sensitivity_selected FROM pages WHERE id=?", (row["sha256"],)).fetchone()
                        candidate = db.execute("""SELECT c.training_id,c.score,c.training_label,p.primary_selected
                            FROM candidates c JOIN pages p ON c.training_id=p.id WHERE c.validation_id=?""", (row["sha256"],)).fetchone()
                        row.update(dict(zip(extras, [int(candidate is not None), candidate[0] if candidate else "",
                            candidate[1] if candidate else "", candidate[2] if candidate else "", candidate[3] if candidate else "",
                            "selected_primary" if selected else "withheld_for_class_balance", sensitivity_selected])))
                        writers[2].writerow(row); written[2] += 1
                        if selected:
                            writers[0].writerow(row); written[0] += 1
                            if candidate:
                                primary_candidate_counts[row["label"]] += 1
                        if sensitivity_selected:
                            writers[1].writerow(row); written[1] += 1
                if written != [sum(sum(c.values()) for c in selected_counts.values()), 2 * sensitivity_target, total]:
                    raise ValueError("Output accounting failed.")
                group_counts = dict(db.execute("SELECT split,count(DISTINCT site_group) FROM pages WHERE primary_selected=1 GROUP BY split"))
                report = {
                    "created_utc": datetime.now(timezone.utc).isoformat(), "script_version": VERSION, "seed": args.seed,
                    "status": "balanced_experimental_manifests_complete", "balanced_subsets_created": True,
                    "inputs_sha256": hashes, "input_counts": input_counts, "balanced_counts": selected_counts,
                    "balanced_group_counts": group_counts, "primary_records": written[0],
                    "records_withheld_for_class_balance": total - written[0],
                    "validation_sensitivity_counts": sensitivity_counts, "validation_sensitivity_records": written[1],
                    "similarity_candidates_in_primary_validation": {l: primary_candidate_counts[l] for l in LABELS},
                    "similarity_candidates_in_sensitivity_validation": 0,
                    "shared_train_validation_site_groups": 0, "cross_split_raw_or_normalized_html_groups": 0,
                    "policy": [
                        "Primary training and validation are separately 1:1 balanced by seeded SHA-256 ranking, undersampling the majority label without replacement.",
                        "Original splits and labels are preserved; similarity candidates remain in primary validation and are explicitly flagged.",
                        "Secondary validation is nested within primary validation, with every script 07 flagged page withheld, then rebalanced 1:1. Use the SAME balanced training model for both evaluations.",
                        "Candidate withholding uses the pre-balance training audit, even when a matched benign training example was not selected. It is conservative.",
                        "No original data is deleted, relabelled, duplicated or overwritten. Test selection is deferred until final evaluation."
                    ],
                    "limitations": [
                        "The sensitivity subset can differ in difficulty and template distribution. Report both validation results and their counts; do not select whichever score looks better.",
                        "Sampled similarity hits are not confirmed duplicate pages or label errors. No-hit pages are not proven free of near copies.",
                        "Balanced evaluation does not represent real-world phishing prevalence; also report false positive rate, precision, recall and the full official test result later.",
                        "Undersampling discards some benign examples from this training experiment; original files and the decision CSV retain their identities.",
                        "Manifest files contain metadata references, not copied HTML or model features. No model is trained and no accuracy is measured."
                    ],
                    "outputs": {outputs[i].name: {"rows": written[i], "sha256": digest(temporary_outputs[i])} for i in range(3)}
                }
                for path in inputs:
                    if digest(path) != hashes[str(path.resolve())]:
                        raise ValueError("Input changed during balancing; outputs were not committed.")
                temporary_outputs[3].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
                for temporary, final in zip(temporary_outputs, outputs):
                    temporary.replace(final)
                print("Balanced primary counts:", json.dumps(selected_counts, indent=2), flush=True)
                print("Balanced validation sensitivity counts:", json.dumps(sensitivity_counts), flush=True)
                print(f"Saved report: {outputs[3].resolve()}", flush=True)
                print("Balanced manifest preparation complete. No model trained yet.", flush=True)
            finally:
                db.close()
    finally:
        for path in temporary_outputs:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, sqlite3.Error) as error:
        raise SystemExit(f"Balancing stopped: {error}. A report is valid only after the final completion message.")
