"""Approximate training/validation HTML similarity candidate audit.

Requires duckdb and the standard library. Uses case-folded lexical 3-token
shingles from the complete HTML when <=24,576 characters, otherwise 8,192
characters each from the beginning, middle and end. Bottom-k hashed sketches
estimate Jaccard similarity. Shared low-hash anchors retrieve candidates.
This heuristic search is NOT exhaustive and does not establish semantic or
runtime equivalence. Only candidates are recorded; no records are removed,
balanced, relabelled, moved, or used to train a model. Test files are not read.

Method background: https://arxiv.org/abs/1303.5479 (bottom-k estimator).
Window sampling, anchor retrieval, length filtering and thresholds below are
implementation choices, not a reproduction of the PhreshPhish paper pipeline.
"""
import argparse
import csv
import hashlib
import heapq
import json
import re
import shutil
import sqlite3
import struct
import zlib
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import duckdb

VERSION = "1.0"
TOKENS = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_sketch(windows, size, seed):
    key = hashlib.sha256(str(seed).encode()).digest()
    values = set()
    for window in windows:
        words = TOKENS.findall((window or "").casefold())
        # Window boundaries do not create artificial adjacent token sequences.
        for index in range(len(words) - 2):
            shingle = "\x1f".join(words[index:index + 3]).encode("utf-8")
            value = int.from_bytes(hashlib.blake2b(shingle, digest_size=8, key=key).digest(), "little")
            values.add(value)
    sketch = tuple(heapq.nsmallest(size, values))
    packed = struct.pack(f"<{len(sketch)}Q", *sketch)
    return sketch, len(values), zlib.compress(packed, level=1)


@lru_cache(maxsize=256)
def unpack_sketch(blob):
    packed = zlib.decompress(blob)
    if len(packed) % 8:
        raise ValueError("Invalid sketch byte count in checkpoint.")
    return struct.unpack(f"<{len(packed)//8}Q", packed)


def bottom_k_jaccard(a, b, size):
    # Bottom-k(A union B) is obtainable from bottom-k(A) and bottom-k(B).
    # Count common elements among the k smallest distinct union elements.
    i = j = common = count = 0
    while count < size and (i < len(a) or j < len(b)):
        if i == len(a):
            j += 1
        elif j == len(b):
            i += 1
        elif a[i] == b[j]:
            common += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
        count += 1
    return (common / count if count else 0.0), common, count


def init_checkpoint(db, manifest, preparation, config):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS pages (
            page_id INTEGER PRIMARY KEY,sha256 TEXT NOT NULL UNIQUE,label TEXT NOT NULL,
            split TEXT NOT NULL,site_group TEXT NOT NULL,source_file TEXT NOT NULL,
            html_characters INTEGER NOT NULL,sketch BLOB,unique_shingles INTEGER,
            sampled_characters INTEGER,processed INTEGER NOT NULL DEFAULT 0,
            search_done INTEGER NOT NULL DEFAULT 0,candidates_checked INTEGER,
            candidates_truncated INTEGER NOT NULL DEFAULT 0,search_skipped INTEGER NOT NULL DEFAULT 0,
            search_skipped_common_anchors INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS source_file_index ON pages(source_file);
        CREATE TABLE IF NOT EXISTS anchors (
            anchor BLOB NOT NULL,page_id INTEGER NOT NULL,PRIMARY KEY(anchor,page_id)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS matches (
            validation_id INTEGER PRIMARY KEY,training_id INTEGER NOT NULL,
            estimated_jaccard REAL NOT NULL,common_sampled_hashes INTEGER NOT NULL,
            union_sample_size INTEGER NOT NULL,shared_anchors INTEGER NOT NULL
        );
    """)
    saved = db.execute("SELECT value FROM state WHERE key='config'").fetchone()
    if saved:
        if json.loads(saved[0]) != config:
            raise ValueError("Inputs or similarity parameters differ from this checkpoint. Use a new --output-dir for a separate run.")
        return
    if db.execute("SELECT count(*) FROM pages").fetchone()[0]:
        raise ValueError("Checkpoint metadata is incomplete. Use a new --output-dir.")
    counts = {split:Counter() for split in ("train","validation")}
    try:
        db.execute("BEGIN")
        with manifest.open(newline="",encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            required = {"sha256","label","split","site_group","source_file","html_characters","preparation_decision"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("Use the retained manifest from script 06.")
            batch = []
            for row in reader:
                split,label = row["split"],row["label"]
                if split not in counts or label not in ("benign","phish") or row["preparation_decision"] != "eligible":
                    raise ValueError("Unexpected record in retained manifest.")
                if not row["site_group"] or int(row["html_characters"]) <= 0:
                    raise ValueError("Missing site group or empty capture in retained manifest.")
                counts[split][label] += 1
                batch.append((row["sha256"],label,split,row["site_group"],row["source_file"],int(row["html_characters"])))
                if len(batch) >= 5000:
                    db.executemany("INSERT INTO pages(sha256,label,split,site_group,source_file,html_characters) VALUES (?,?,?,?,?,?)",batch)
                    batch.clear()
            if batch:
                db.executemany("INSERT INTO pages(sha256,label,split,site_group,source_file,html_characters) VALUES (?,?,?,?,?,?)",batch)
        actual = {split:{label:counts[split][label] for label in ("benign","phish")} for split in counts}
        if actual != preparation["retained_label_counts_by_split"]:
            raise ValueError("Retained manifest counts differ from the preparation report.")
        shared = db.execute("""SELECT count(*) FROM (
            SELECT site_group FROM pages GROUP BY site_group HAVING count(DISTINCT split)>1)
        """).fetchone()[0]
        if shared:
            raise ValueError("Website groups overlap in the input manifest.")
        db.execute("INSERT INTO state VALUES ('config',?)",[json.dumps(config,sort_keys=True)])
        db.commit()
    except BaseException:
        db.rollback()
        raise


def scan_file(db, path, args):
    records = db.execute("""SELECT page_id,sha256,label,split,html_characters,processed
        FROM pages WHERE source_file=? ORDER BY sha256""",[path.name]).fetchall()
    if not records:
        print("  No retained records in this source file.",flush=True)
        return
    if all(row[5] for row in records):
        print(f"  Checkpoint found: skipping {len(records):,} completed records.",flush=True)
        return
    if any(row[5] for row in records):
        raise ValueError(f"Unexpected partial source checkpoint: {path.name}")
    expected = {row[1]:row[:5] for row in records}
    window = args.window_characters
    limit = 3 * window
    placeholders = ",".join("?" for _ in records)
    # Only bounded window strings cross into Python. The full HTML column still
    # needs Parquet decoding, which can require substantial memory per row group.
    sql = f"""SELECT sha256,label,length(html),
        CASE WHEN length(html)<={limit} THEN html ELSE substr(html,1,{window}) END,
        CASE WHEN length(html)>{limit} THEN substr(html,
            greatest(1,cast(floor((length(html)-{window})/2) AS BIGINT)+1),{window}) ELSE '' END,
        CASE WHEN length(html)>{limit} THEN substr(html,-{window}) ELSE '' END
        FROM read_parquet(?) WHERE sha256 IN ({placeholders})"""
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET memory_limit=?",[args.memory_limit])
        cursor = con.execute(sql,[str(path),*[row[1] for row in records]])
        db.execute("BEGIN")
        seen = set()
        while True:
            batch = cursor.fetchmany(16)
            if not batch:
                break
            for sha,label,length,first,middle,last in batch:
                info = expected.get(sha)
                if not info or sha in seen or label != info[2] or length != info[4]:
                    raise ValueError(f"Record metadata differs from the prepared manifest in {path.name}.")
                seen.add(sha)
                sketch,unique_count,blob = make_sketch((first,middle,last),args.sketch_size,args.seed)
                sampled = sum(len(value or "") for value in (first,middle,last))
                page_id,_,_,split,_ = info
                db.execute("""UPDATE pages SET sketch=?,unique_shingles=?,sampled_characters=?,processed=1
                    WHERE page_id=?""",[blob,unique_count,sampled,page_id])
                if split == "train" and unique_count >= args.minimum_shingles:
                    db.executemany("INSERT INTO anchors(anchor,page_id) VALUES (?,?)",
                                   [(struct.pack("<Q",value),page_id) for value in sketch[:args.anchor_count]])
        if len(seen) != len(expected):
            raise ValueError(f"Missing retained records in {path.name}.")
        db.commit()
        print(f"  Saved sketches: {len(seen):,} records.",flush=True)
    except BaseException:
        db.rollback()
        raise
    finally:
        con.close()


def prepare_anchor_frequencies(db):
    if db.execute("SELECT 1 FROM state WHERE key='anchor_frequencies_ready'").fetchone():
        return
    print("Counting training anchor frequencies...",flush=True)
    try:
        db.execute("BEGIN")
        db.execute("CREATE TABLE IF NOT EXISTS anchor_frequencies (anchor BLOB PRIMARY KEY,documents INTEGER NOT NULL) WITHOUT ROWID")
        db.execute("DELETE FROM anchor_frequencies")
        db.execute("INSERT INTO anchor_frequencies SELECT anchor,count(*) FROM anchors GROUP BY anchor")
        db.execute("INSERT INTO state VALUES ('anchor_frequencies_ready','true')")
        db.commit()
    except BaseException:
        db.rollback()
        raise


def search_validation(db,args):
    total = db.execute("SELECT count(*) FROM pages WHERE split='validation'").fetchone()[0]
    done = db.execute("SELECT count(*) FROM pages WHERE split='validation' AND search_done=1").fetchone()[0]
    print(f"Validation candidate search: {done:,}/{total:,} already checkpointed.",flush=True)
    last_id = 0
    while True:
        batch = db.execute("""SELECT page_id,html_characters,sketch,unique_shingles FROM pages
            WHERE split='validation' AND search_done=0 AND page_id>? ORDER BY page_id LIMIT 100""",[last_id]).fetchall()
        if not batch:
            break
        try:
            db.execute("BEGIN")
            for page_id,length,blob,unique_count in batch:
                last_id = page_id
                sketch = unpack_sketch(blob)
                if unique_count < args.minimum_shingles:
                    db.execute("UPDATE pages SET search_done=1,search_skipped=1,candidates_checked=0 WHERE page_id=?",[page_id])
                    done += 1
                    continue
                anchors = [struct.pack("<Q",value) for value in sketch[:args.anchor_count]]
                placeholders = ",".join("?" for _ in anchors)
                # Very common lexical anchors can otherwise create huge joins
                # among unrelated pages. This speed limit reduces search recall
                # and is reported explicitly; it never marks a skipped page clean.
                anchors = [row[0] for row in db.execute(f"""SELECT anchor FROM anchor_frequencies
                    WHERE anchor IN ({placeholders}) AND documents<=? ORDER BY anchor""",
                    [*anchors,args.max_anchor_frequency])]
                if not anchors:
                    db.execute("""UPDATE pages SET search_done=1,search_skipped_common_anchors=1,
                        candidates_checked=0 WHERE page_id=?""",[page_id])
                    done += 1
                    continue
                placeholders = ",".join("?" for _ in anchors)
                candidates = db.execute(f"""SELECT p.page_id,p.sketch,p.unique_shingles,count(*) AS common_anchors
                    FROM anchors a JOIN pages p ON a.page_id=p.page_id
                    WHERE a.anchor IN ({placeholders}) AND p.html_characters>=? AND p.html_characters<=?
                    GROUP BY p.page_id ORDER BY common_anchors DESC,p.sha256 LIMIT ?""",
                    [*anchors,length*args.minimum_length_ratio,length/args.minimum_length_ratio,args.max_candidates+1]).fetchall()
                truncated = int(len(candidates)>args.max_candidates)
                checked = 0
                for train_id,train_blob,train_unique,shared_anchors in candidates[:args.max_candidates]:
                    checked += 1
                    estimate,common,sample_n = bottom_k_jaccard(sketch,unpack_sketch(train_blob),args.sketch_size)
                    if estimate >= args.threshold:
                        db.execute("INSERT INTO matches VALUES (?,?,?,?,?,?)",
                                   [page_id,train_id,estimate,common,sample_n,shared_anchors])
                        break
                db.execute("""UPDATE pages SET search_done=1,candidates_checked=?,
                    candidates_truncated=? WHERE page_id=?""",[checked,truncated,page_id])
                done += 1
            db.commit()
        except BaseException:
            db.rollback()
            raise
        if done % 500 < 100 or done == total:
            print(f"Validation comparisons: {done:,}/{total:,}",flush=True)
    if db.execute("SELECT count(*) FROM pages WHERE split='validation' AND search_done=0").fetchone()[0]:
        raise ValueError("Validation search is incomplete.")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,default=root/"reports"/"prepared"/"clean_training_validation_manifest.csv")
    parser.add_argument("--preparation-report",type=Path,default=root/"reports"/"prepared"/"preparation_report.json")
    parser.add_argument("--fingerprint-report",type=Path,default=root/"reports"/"html_fingerprints"/"html_fingerprint_report.json")
    parser.add_argument("--output-dir",type=Path,default=root/"reports"/"similarity")
    parser.add_argument("--memory-limit",default="4GB")
    parser.add_argument("--window-characters",type=int,default=8192)
    parser.add_argument("--sketch-size",type=int,default=512)
    parser.add_argument("--anchor-count",type=int,default=16)
    parser.add_argument("--minimum-shingles",type=int,default=128)
    parser.add_argument("--minimum-length-ratio",type=float,default=0.90)
    parser.add_argument("--threshold",type=float,default=0.98)
    parser.add_argument("--max-candidates",type=int,default=512)
    parser.add_argument("--max-anchor-frequency",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=42)
    args = parser.parse_args()
    if not 1024 <= args.window_characters <= 65536:
        parser.error("--window-characters must be between 1024 and 65536")
    if not 128 <= args.sketch_size <= 2048 or not 1 <= args.anchor_count <= min(64,args.sketch_size):
        parser.error("Use sketch size 128..2048 and anchor count 1..64 not exceeding sketch size")
    if not args.anchor_count <= args.minimum_shingles <= args.sketch_size:
        parser.error("--minimum-shingles must lie between anchor count and sketch size")
    if not 0.80 <= args.threshold <= 1 or not 0.50 <= args.minimum_length_ratio <= 1:
        parser.error("Threshold must be 0.80..1 and minimum length ratio 0.50..1")
    if not 1 <= args.max_candidates <= 4096:
        parser.error("--max-candidates must be 1..4096")
    if not 1 <= args.max_anchor_frequency <= 20000:
        parser.error("--max-anchor-frequency must be 1..20000")
    for path in (args.manifest,args.preparation_report,args.fingerprint_report):
        if not path.is_file():
            parser.error(f"Missing input: {path}. Complete scripts 05 and 06 first.")
    preparation = json.loads(args.preparation_report.read_text(encoding="utf-8-sig"))
    fingerprint = json.loads(args.fingerprint_report.read_text(encoding="utf-8-sig"))
    manifest_digest = digest_file(args.manifest)
    if manifest_digest != preparation["output_retained_manifest_sha256"]:
        parser.error("Retained manifest differs from the preparation report")
    if digest_file(args.fingerprint_report) != preparation["input_fingerprint_report_sha256"]:
        parser.error("Fingerprint report differs from the preparation report")
    if any(preparation[key] for key in ("remaining_cross_split_raw_html_groups",
                                       "remaining_cross_split_normalized_html_groups","shared_train_validation_site_groups")):
        parser.error("Exact/normalized/site overlap checks must pass before this audit")
    files = sorted(args.data_dir.glob("train-*.parquet"))
    stats = [{"name":path.name,"bytes":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns} for path in files]
    if stats != fingerprint["inputs"]["source_files"]:
        parser.error("Dataset file names/sizes/modification times differ from the completed fingerprint audit")
    parameters = {key:getattr(args,key) for key in (
        "window_characters","sketch_size","anchor_count","minimum_shingles",
        "minimum_length_ratio","threshold","max_candidates","max_anchor_frequency","seed")}
    config = {"algorithm_version":VERSION,"manifest_sha256":manifest_digest,
              "preparation_report_sha256":digest_file(args.preparation_report),
              "fingerprint_report_sha256":digest_file(args.fingerprint_report),
              "data_dir":str(args.data_dir.resolve()),"source_files":stats,"parameters":parameters}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    index_path = args.output_dir/"html_similarity_index.sqlite"
    if not index_path.exists():
        estimated_bytes = preparation["retained_rows"] * (args.sketch_size*8 + args.anchor_count*48 + 1024)
        if shutil.disk_usage(args.output_dir).free < estimated_bytes:
            parser.error(f"Not enough free space for the similarity checkpoint estimate ({estimated_bytes/1024**3:.1f} GiB). Use --output-dir on a drive with enough space.")
    db = sqlite3.connect(index_path)
    db.execute("PRAGMA cache_size=-32768")
    temporary_output = None
    try:
        print(f"DuckDB memory limit: {args.memory_limit}; threads: 1",flush=True)
        print("Checking retained manifest and similarity checkpoint...",flush=True)
        init_checkpoint(db,args.manifest,preparation,config)
        for index,path in enumerate(files,1):
            print(f"HTML sketches {index}/{len(files)}: {path.name}",flush=True)
            scan_file(db,path,args)
        total,processed = db.execute("SELECT count(*),sum(processed) FROM pages").fetchone()
        if total != processed or total != preparation["retained_rows"]:
            raise ValueError("Incomplete sketches or unexpected record count")
        prepare_anchor_frequencies(db)
        search_validation(db,args)
        if digest_file(args.manifest) != manifest_digest or digest_file(args.preparation_report) != config["preparation_report_sha256"]:
            raise ValueError("Inputs changed during the audit")
        after_stats = [{"name":path.name,"bytes":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns} for path in files]
        if after_stats != stats:
            raise ValueError("Dataset metadata changed during the audit")
        match_query = """SELECT v.sha256,t.sha256,v.label,t.label,v.site_group,t.site_group,
            m.estimated_jaccard,m.common_sampled_hashes,m.union_sample_size,m.shared_anchors,
            v.html_characters,t.html_characters,v.sampled_characters,t.sampled_characters,
            v.source_file,t.source_file FROM matches m
            JOIN pages v ON m.validation_id=v.page_id JOIN pages t ON m.training_id=t.page_id
            ORDER BY v.sha256"""
        fields = ["validation_sha256","training_sha256","validation_label","training_label",
                  "validation_site_group","training_site_group","estimated_jaccard",
                  "common_sampled_hashes","union_sample_size","shared_anchors",
                  "validation_html_characters","training_html_characters",
                  "validation_sampled_characters","training_sampled_characters",
                  "validation_source_file","training_source_file"]
        matches_path = args.output_dir/"validation_similarity_candidates.csv"
        temporary_output = matches_path.with_suffix(".csv.tmp")
        examples = []
        with temporary_output.open("w",newline="",encoding="utf-8") as file:
            writer = csv.writer(file);writer.writerow(fields)
            for row in db.execute(match_query):
                writer.writerow(row)
                if len(examples)<10:
                    examples.append(dict(zip(fields,row)))
        temporary_output.replace(matches_path)
        match_counts = dict(db.execute("""SELECT v.label,count(*) FROM matches m
            JOIN pages v ON m.validation_id=v.page_id GROUP BY v.label"""))
        different_label_matches = db.execute("""SELECT count(*) FROM matches m JOIN pages v
            ON m.validation_id=v.page_id JOIN pages t ON m.training_id=t.page_id WHERE v.label<>t.label""").fetchone()[0]
        count_fields = ["split","label","rows","complete_html_sampled_rows","below_minimum_shingles_rows",
                        "mean_sampled_character_fraction"]
        coverage = [dict(zip(count_fields,row)) for row in db.execute("""SELECT split,label,count(*),
            sum(sampled_characters=html_characters),sum(unique_shingles<?),
            avg(cast(sampled_characters AS REAL)/html_characters) FROM pages GROUP BY split,label ORDER BY split,label""",
            [args.minimum_shingles])]
        skipped,common_skipped,truncated,without_candidates,checked = db.execute("""SELECT sum(search_skipped),
            sum(search_skipped_common_anchors),sum(candidates_truncated),
            sum(search_skipped=0 AND search_skipped_common_anchors=0 AND candidates_checked=0),sum(candidates_checked)
            FROM pages WHERE split='validation'""").fetchone()
        report = {
            "created_utc":datetime.now(timezone.utc).isoformat(),"script_version":VERSION,
            "status":"sampled_similarity_candidate_audit_complete_not_exhaustive",
            "duckdb_version":duckdb.__version__,"inputs":config,"records_sketched":total,
            "window_sampling":"Whole HTML up to 3 times the window size; otherwise equal character windows from beginning, middle, and end.",
            "token_representation":"Case-folded Unicode word tokens and individual non-whitespace punctuation tokens; sets of contiguous 3-token shingles within each window.",
            "sketch_method":"512 bottom-ranked 64-bit keyed BLAKE2b shingle hashes by default; Jaccard estimated from bottom-k of the two-sketch union.",
            "candidate_retrieval":"Shared lowest-hash anchors after excluding anchors exceeding the frequency cap, length-ratio filter, ranked by shared anchors; bounded candidate comparisons, stopping at first threshold match.",
            "method_background":"https://arxiv.org/abs/1303.5479",
            "coverage_by_split_and_label":coverage,
            "validation_rows_with_candidate_match":sum(match_counts.values()),
            "validation_matches_by_label":{label:match_counts.get(label,0) for label in ("benign","phish")},
            "recorded_pairs_with_different_labels":different_label_matches,
            "validation_search_skipped_low_shingle_rows":skipped,
            "validation_search_skipped_no_usable_anchor_rows":common_skipped,
            "validation_candidate_lists_truncated_rows":truncated,
            "validation_rows_without_retrieved_candidates":without_candidates,
            "candidate_pairs_compared":checked,"candidate_manifest_sha256":digest_file(matches_path),
            "example_candidate_pairs":examples,"records_removed":0,"balanced_subsets_created":False,
            "limitations":[
                "Scores concern sampled lexical HTML/JavaScript content, not the whole document for large pages and not executed behavior.",
                "A score of 0.98 is an estimated token-set similarity; it is not model accuracy or a probability of phishing.",
                "Shared boilerplate can produce high scores between semantically different pages; hits are candidates, not verified duplicate pages.",
                "Anchor retrieval, common-anchor exclusions, length filtering, short-document skipping and candidate caps can miss near copies; no absence-of-leakage guarantee is made.",
                "One qualifying training example is recorded per validation record, not all matching pairs or necessarily the nearest match.",
                "Keyed 64-bit hash collisions are possible; sketch estimates also have sampling error. Case-folding/punctuation tokenization can change distinctions important to JavaScript.",
                "Within-training and within-validation near copies are not audited here; exact copies were handled in script 06.",
                "No capture-quality or label correctness decisions are made. Different labels in candidate pairs do not prove mislabelling.",
                "No dataset files are modified, no official test contents read, and no model or balanced subset created.",
                "Dataset file metadata checks do not prove underlying bytes are unchanged; checkpoints require the same input hashes, parameters and file metadata.",
            ],
        }
        report_path = args.output_dir/"html_similarity_report.json"
        report_tmp = report_path.with_suffix(".json.tmp")
        report_tmp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
        report_tmp.replace(report_path)
        print("Validation records with sampled similarity candidates:",sum(match_counts.values()),flush=True)
        print("Candidate counts by validation label:",report["validation_matches_by_label"],flush=True)
        print(f"Saved report: {report_path.resolve()}",flush=True)
        print(f"Saved candidate list: {matches_path.resolve()}",flush=True)
        print("HTML similarity candidate audit complete. Review candidates before finalizing balanced subsets.",flush=True)
    except duckdb.OutOfMemoryException:
        raise SystemExit("Not enough memory for Parquet decoding. Close memory-heavy applications and rerun the same command. Completed file checkpoints are retained. No new complete similarity report was written.")
    except (ValueError,sqlite3.Error,duckdb.Error) as error:
        raise SystemExit(str(error)+" No new complete similarity report was written.")
    except KeyboardInterrupt:
        raise SystemExit("Stopped. Rerun the same command to reuse completed sketch files and validation-search checkpoints.")
    finally:
        db.close()
        if temporary_output:
            temporary_output.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
