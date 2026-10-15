"""One capture attempt per frozen URL, with transactional per-page checkpoints."""
import csv
import hashlib
import io
import json
import math
import os
import random
import sqlite3
import tempfile
import uuid
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from . import VERSION
from .protocol import LABELS, EvaluationError, canonical_json, classification_metrics, object_sha, require, safe_ratio

SCHEMA_VERSION = 1
ARCHIVE_UTF8_LIMIT = 128 * 1024 * 1024
COMPLETE_STATUSES = {"prediction_complete", "inconclusive_capture", "capture_failed", "interrupted_capture"}
CSV_COLUMNS = [
    "csv_row", "url", "normalized_url", "label", "label_source", "label_checked_at",
    "attempt_started_at_utc", "capture_completed_at_utc", "prediction_completed_at_utc",
    "final_url", "status", "phishing_score", "prediction", "correct",
    "diagnostic_score", "diagnostic_prediction", "active_quality_flags", "error", "html_sha256",
]


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".live-eval-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_write(path, (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8"))


class RunLock:
    """OS lock releases after Ctrl+C/process exit, including on Windows."""

    def __init__(self, path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.handle = open(self.path, "a+b")
        if self.handle.seek(0, 2) == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            raise EvaluationError("Another evaluation process is using this output folder. Keep one running copy.") from error
        return self

    def __exit__(self, *unused):
        if self.handle is not None:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def archive_html(blob):
    require(blob is not None, "Checkpoint is missing a captured page archive.")
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(blob, ARCHIVE_UTF8_LIMIT + 1)
        require(len(raw) <= ARCHIVE_UTF8_LIMIT and decoder.eof and not decoder.unused_data
                and not decoder.unconsumed_tail, "Checkpoint archive is oversized or incomplete.")
        return raw.decode("utf-8")
    except (zlib.error, UnicodeError) as error:
        raise EvaluationError("Checkpoint page archive is corrupt.") from error


class Checkpoint:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS pages "
                                "(ordinal INTEGER PRIMARY KEY, record_json TEXT NOT NULL, "
                                "record_sha256 TEXT NOT NULL, html_zlib BLOB)")
        self.connection.commit()

    def close(self):
        self.connection.close()

    def save(self, record, html=None, keep_archive=False):
        encoded = canonical_json(record)
        old = self.connection.execute("SELECT html_zlib FROM pages WHERE ordinal=?", (record["ordinal"],)).fetchone() if keep_archive else None
        archive = old[0] if old is not None else (zlib.compress(html.encode("utf-8"), 6) if html is not None else None)
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO pages VALUES(?,?,?,?)", (
                record["ordinal"], encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest(), archive))

    def records(self):
        result = {}
        for ordinal, encoded, expected in self.connection.execute("SELECT ordinal,record_json,record_sha256 FROM pages ORDER BY ordinal"):
            require(hashlib.sha256(encoded.encode("utf-8")).hexdigest() == expected,
                    f"Checkpoint record {ordinal} failed its integrity check.")
            record = json.loads(encoded)
            require(record["ordinal"] == ordinal, "Checkpoint row identity changed.")
            result[ordinal] = record
        return result

    def html(self, ordinal):
        row = self.connection.execute("SELECT html_zlib FROM pages WHERE ordinal=?", (ordinal,)).fetchone()
        require(row is not None, "Checkpoint page is missing.")
        return archive_html(row[0])

    def verify(self, rows, descriptor):
        require(self.connection.execute("PRAGMA quick_check").fetchone()[0] == "ok", "SQLite checkpoint integrity check failed.")
        inputs = {row["ordinal"]: row for row in rows}
        for ordinal, record in self.records().items():
            require(ordinal in inputs and record["url"] == inputs[ordinal]["normalized_url"], "Checkpoint URL identity changed.")
            phase, status = record["phase"], record["status"]
            require((phase == "started" and status == "capture_started")
                    or (phase == "captured" and status == "captured_awaiting_prediction")
                    or (phase == "complete" and status in COMPLETE_STATUSES), "Unknown checkpoint state.")
            if phase == "captured" or status in {"prediction_complete", "inconclusive_capture"}:
                html = self.html(ordinal)
                require(hashlib.sha256(html.encode("utf-8")).hexdigest() == record["html_sha256"], "Captured HTML checksum changed.")
                validate_capture(record["capture"], record["url"])
                if phase == "complete":
                    validate_prediction(record["result"], descriptor, record["html_sha256"], record["capture"]["final_url"])
            else:
                require("result" not in record, "A failed capture must not contain a model prediction.")


def validate_capture(metadata, url):
    require(metadata.get("requested_url") == url and metadata.get("capture_basis") == "http_source"
            and metadata.get("capture_complete") is True and metadata.get("status_code") == 200
            and isinstance(metadata.get("final_url"), str) and metadata["final_url"]
            and metadata.get("javascript_executed") is False and metadata.get("external_resources_requested") is False,
            "Capture metadata differs from the static HTTP-source protocol.")


def validate_prediction(result, descriptor, html_sha, final_url):
    score = result.get("phishing_score")
    require(result.get("status") in {"prediction_complete", "inconclusive_capture"}
            and result.get("model") == descriptor and result.get("capture_basis") == "http_source"
            and result.get("page_url") == final_url and result.get("html_sha256") == html_sha,
            "Prediction provenance differs from the captured page or frozen model.")
    require(isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)
            and 0 <= score <= 1 and result.get("static_model_prediction") == ("phish" if score >= 0.5 else "benign"),
            "Prediction score/label differs from the fixed 0.50 threshold.")


def csv_safe(value):
    """Keep evidence explanations from becoming formulas when opened in Excel."""
    if isinstance(value, str) and value[:1] in {"=", "+", "-", "@", "\t", "\r"}:
        return "'" + value
    return value


def duplicate_groups(rows, records, field):
    groups = defaultdict(list)
    for row in rows:
        record = records.get(row["ordinal"], {})
        if record.get("phase") not in {"captured", "complete"} or "capture" not in record:
            continue
        key = record.get("html_sha256") if field == "html_sha256" else record["capture"].get("final_url")
        if key:
            groups[key].append({"csv_row": row["csv_row"], "label": row["label"]})
    return [{field: key, "pages": pages, "conflicting_labels": len({x["label"] for x in pages}) > 1}
            for key, pages in sorted(groups.items()) if len(pages) > 1]


def make_summary(rows, records, manifest):
    coverage = {label: {"selected": 0, "prediction_complete": 0, "inconclusive_capture": 0,
                        "capture_failed": 0, "interrupted_capture": 0, "pending": 0} for label in LABELS}
    pairs, output_rows = [], []
    for row in rows:
        record = records.get(row["ordinal"], {})
        status = record.get("status", "not_attempted")
        coverage[row["label"]]["selected"] += 1
        coverage[row["label"]][status if status in COMPLETE_STATUSES else "pending"] += 1
        output = {key: row[key] for key in ("csv_row", "url", "normalized_url", "label", "label_source", "label_checked_at")}
        output.update({key: record.get(key, "") for key in (
            "attempt_started_at_utc", "capture_completed_at_utc", "prediction_completed_at_utc", "error", "html_sha256")})
        output.update({"status": status, "final_url": record.get("capture", {}).get("final_url", "")})
        result = record.get("result", {})
        output["active_quality_flags"] = ";".join(result.get("active_quality_flags", []))
        if status == "prediction_complete":
            predicted = result["static_model_prediction"]
            pairs.append((row["label"], predicted))
            output.update({"phishing_score": result["phishing_score"], "prediction": predicted,
                           "correct": predicted == row["label"]})
        elif status == "inconclusive_capture":
            output.update({"diagnostic_score": result["phishing_score"],
                           "diagnostic_prediction": result["static_model_prediction"]})
        output_rows.append(output)
    for counts in coverage.values():
        counts["scored_coverage"] = safe_ratio(counts["prediction_complete"], counts["selected"])
        counts["complete_http_capture_coverage"] = safe_ratio(counts["prediction_complete"] + counts["inconclusive_capture"], counts["selected"])
    scored_counts = {label: coverage[label]["prediction_complete"] for label in LABELS}
    completed = sum(sum(counts[status] for status in COMPLETE_STATUSES) for counts in coverage.values())
    distinct_hosts = {label: len({urlsplit(row["normalized_url"]).hostname for row in rows if row["label"] == label}) for label in LABELS}
    metrics = classification_metrics(pairs)
    warnings = []
    if completed < len(rows):
        warnings.append("This is a partial report. Wait for 'Live evaluation complete'.")
    if metrics["scored_pages"] < len(rows):
        warnings.append("Accuracy/FPR use scored pages only. Missing and inconclusive pages remain in coverage denominators.")
    if scored_counts["benign"] != scored_counts["phish"]:
        warnings.append("The input list is balanced; the scored subset is not balanced. No successful pages were dropped to rebalance it.")
    if not all(scored_counts.values()):
        warnings.append("Both classes need scored pages to estimate balanced accuracy and both class error rates.")
    duplicates = {field: duplicate_groups(rows, records, field) for field in ("final_url", "html_sha256")}
    if any(duplicates.values()):
        warnings.append("Some URLs reached the same final URL or HTML. Review duplicate groups; rows and frozen labels were retained.")
    summary = {
        "status": "live_evaluation_complete" if completed == len(rows) else "live_evaluation_partial",
        "evaluator_version": VERSION, "generated_at_utc": now_utc(),
        "protocol_sha256": manifest["identity_sha256"], "started_at_utc": manifest["started_at_utc"],
        "model": manifest["identity"]["model"], "threshold": 0.5,
        "selected_pages": len(rows), "completed_attempts": completed,
        "input_class_counts": {label: coverage[label]["selected"] for label in LABELS},
        "input_list_balanced": True, "scored_class_counts": scored_counts,
        "scored_subset_balanced": scored_counts["benign"] == scored_counts["phish"] and scored_counts["benign"] > 0,
        "coverage_by_class": coverage, "distinct_input_hostnames_by_class": distinct_hosts,
        "scored_coverage_all_selected_pages": safe_ratio(metrics["scored_pages"], len(rows)),
        "classification_metrics_on_scored_pages": metrics,
        "correct_scored_predictions_divided_by_all_selected_pages": safe_ratio(
            metrics["confusion_matrix"]["TP"] + metrics["confusion_matrix"]["TN"], len(rows)),
        "all_selected_fraction_is_not_classification_accuracy": True,
        "duplicate_capture_groups": duplicates,
        "labels_verified_by_evaluator": False, "model_trained_or_tuned": False,
        "javascript_executed": False, "external_javascript_downloaded": False,
        "sampling_design": "user-selected balanced live pilot; no probability sampling assumed",
        "warnings": warnings,
        "limitations": [
            "Labels come from the supplied evidence/date; this evaluator cannot establish ground truth or whether a page later changed.",
            "Do not use model predictions to choose or relabel evaluation URLs.",
            "Metrics describe this live HTTP-source pilot, not population accuracy or a new official PhreshPhish benchmark.",
            "Training/test archives used browser-rendered HTML. This runtime captures HTTP source without running JavaScript.",
            "Conditional accuracy excludes capture failures and recognized challenge/unavailable-page responses; report coverage alongside it.",
            "Wilson intervals describe binomial sampling uncertainty only. Site selection and correlated pages can add further uncertainty.",
        ],
        "files": {"predictions": "live_evaluation_predictions.csv", "frozen_labels": "frozen_input.csv",
                  "protocol": "evaluation_protocol.json", "captured_html_and_full_results": "checkpoint.sqlite3"},
    }
    return summary, output_rows


def write_reports(output_dir, rows, checkpoint, manifest):
    summary, records = make_summary(rows, checkpoint.records(), manifest)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: csv_safe(value) for key, value in record.items()} for record in records)
    atomic_write(output_dir / "live_evaluation_predictions.csv", stream.getvalue().encode("utf-8-sig"))
    atomic_json(output_dir / "live_evaluation_report.json", summary)
    return summary


def run_evaluation(rows, raw, output_dir, detector, capture_module, code_checksums, progress=None):
    """Detector/capture injection supports offline plumbing tests, not CLI model substitution."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    order = [row["ordinal"] for row in rows]
    random.Random(42).shuffle(order)
    identity = {
        "schema_version": SCHEMA_VERSION, "evaluator_version": VERSION,
        "input_csv_sha256": hashlib.sha256(raw).hexdigest(), "normalized_rows_sha256": object_sha(rows),
        "rows": rows, "code_sha256": code_checksums, "model": detector.describe(),
        "capture": {"basis": "http_source", "max_html_bytes": capture_module.MAX_HTML_BYTES,
                    "timeout_seconds": capture_module.TOTAL_TIMEOUT, "max_redirects": capture_module.MAX_REDIRECTS,
                    "attempts_per_url": 1, "javascript_executed": False},
        "shuffle_seed": 42, "request_order_ordinals": order,
        "label_policy": "independent evidence and check date supplied before predictions; evaluator does not verify label truth",
        "scoring_policy": "all prediction_complete pages; no replacement or rebalancing of successful captures",
    }
    require(identity["model"]["threshold"] == 0.5, "Threshold must stay 0.50.")
    with RunLock(output_dir / ".run.lock"):
        protocol_path = output_dir / "evaluation_protocol.json"
        new_run = not protocol_path.is_file()
        if new_run:
            require(not any(path.name != ".run.lock" for path in output_dir.iterdir()),
                    "Output folder is not empty and has no complete protocol. Use a new --output-dir.")
            manifest = {"started_at_utc": now_utc(), "identity": identity, "identity_sha256": object_sha(identity)}
        else:
            manifest = json.loads(protocol_path.read_text(encoding="utf-8"))
            require(manifest.get("identity_sha256") == object_sha(manifest["identity"])
                    == object_sha(identity),
                    "Resume refused: URL list, labels, code, environment, model or capture settings changed. Keep the original inputs and command.")
            frozen_csv = output_dir / "frozen_input.csv"
            require(frozen_csv.is_file() and hashlib.sha256(frozen_csv.read_bytes()).hexdigest() == identity["input_csv_sha256"],
                    "Frozen input CSV is missing or changed.")
            require((output_dir / "checkpoint.sqlite3").is_file(), "Checkpoint is missing. Do not restart the same run by deleting checkpoints.")
        checkpoint = Checkpoint(output_dir / "checkpoint.sqlite3")
        checkpoint_validated = False
        try:
            if new_run:
                atomic_write(output_dir / "frozen_input.csv", raw)
                atomic_json(protocol_path, manifest)
            checkpoint.verify(rows, identity["model"])
            checkpoint_validated = True
            records = checkpoint.records()
            # A process lost mid-request has an unknown capture outcome. Retain it
            # as interrupted, rather than making an unrecorded second request.
            for record in records.values():
                if record["phase"] == "started":
                    record.update({"phase": "complete", "status": "interrupted_capture", "capture_completed_at_utc": now_utc(),
                                   "error": "The previous capture attempt was interrupted; it was not silently retried."})
                    checkpoint.save(record)
            inputs = {row["ordinal"]: row for row in rows}
            for ordinal in order:
                record = checkpoint.records().get(ordinal)
                if record is not None and record["phase"] == "complete":
                    continue
                row = inputs[ordinal]
                if record is None:
                    record = {"ordinal": ordinal, "url": row["normalized_url"], "attempt_id": uuid.uuid4().hex,
                              "attempt_started_at_utc": now_utc(), "phase": "started", "status": "capture_started"}
                    checkpoint.save(record)
                    try:
                        captured = capture_module.capture_url(row["normalized_url"])
                    except capture_module.CaptureError as error:
                        record.update({"phase": "complete", "status": "capture_failed", "capture_completed_at_utc": now_utc(), "error": str(error)})
                        checkpoint.save(record)
                    else:
                        validate_capture(captured.metadata, record["url"])
                        html = captured.html
                        require(isinstance(html, str) and html.strip()
                                and len(html.encode("utf-8")) <= ARCHIVE_UTF8_LIMIT, "Captured HTML is empty or exceeds the archive limit.")
                        record.update({"phase": "captured", "status": "captured_awaiting_prediction",
                                       "capture_completed_at_utc": now_utc(), "capture": captured.metadata,
                                       "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest()})
                        checkpoint.save(record, html=html)
                if record["phase"] == "captured":
                    html = checkpoint.html(ordinal)
                    result = detector.predict_html(html, record["capture"]["final_url"], "http_source")
                    validate_prediction(result, identity["model"], record["html_sha256"], record["capture"]["final_url"])
                    result["capture_basis_verified"] = True
                    record.update({"phase": "complete", "status": result["status"], "result": result,
                                   "prediction_completed_at_utc": now_utc()})
                    checkpoint.save(record, keep_archive=True)
                summary = write_reports(output_dir, rows, checkpoint, manifest)
                if progress is not None:
                    progress(summary["completed_attempts"], len(rows), row["csv_row"], record["status"])
            return write_reports(output_dir, rows, checkpoint, manifest)
        except BaseException:
            # Preserve transactionally committed captures before a model/import/
            # memory/interrupt failure. A later inference retry uses the archive.
            if checkpoint_validated:
                write_reports(output_dir, rows, checkpoint, manifest)
            raise
        finally:
            checkpoint.close()
