"""Step 33: read-only, offline review of a completed step-32 evaluation.

Only the Python standard library is used. Saved labels, scores and metrics stay
unchanged. Source observations are hints for manual review, not executed behavior.
"""
import argparse
import csv
import hashlib
import io
import ipaddress
import json
import math
import os
import random
import re
import shutil
import sqlite3
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

VERSION = "1.0.0"
ARCHIVE_UTF8_LIMIT = 128 * 1024 * 1024
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_ROWS = 2000
SOURCE_COPY_BYTES = 2 * 1024 * 1024
MAX_SOURCE_COPIES = 24
MARKUP_SCAN_CHARACTERS = 2 * 1024 * 1024
MAX_MARKUP_DEPTH = 512
FIELDS = ["url", "label", "label_source", "label_checked_at"]
LABELS = ("benign", "phish")
STATUSES = {"prediction_complete", "inconclusive_capture", "capture_failed", "interrupted_capture"}
SOURCE_FILES = ("evaluation_protocol.json", "frozen_input.csv", "checkpoint.sqlite3",
                "live_evaluation_report.json", "live_evaluation_predictions.csv")
CSV_COLUMNS = ["csv_row", "url", "normalized_url", "label", "label_source", "label_checked_at",
               "attempt_started_at_utc", "capture_completed_at_utc", "prediction_completed_at_utc",
               "final_url", "status", "phishing_score", "prediction", "correct", "diagnostic_score",
               "diagnostic_prediction", "active_quality_flags", "error", "html_sha256"]


class ReviewError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ReviewError(message)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def object_sha(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded_read(path, limit):
    require(path.is_file(), f"Required source file is missing: {path}")
    require(path.stat().st_size <= limit, f"Source file exceeds its read limit: {path.name}")
    with path.open("rb") as handle:
        value = handle.read(limit + 1)
    require(len(value) <= limit, f"Source file grew beyond its read limit: {path.name}")
    return value


def strict_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, f"Duplicate JSON field: {key}")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda token: (_ for _ in ()).throw(ReviewError(f"Invalid JSON number: {token}")))


def timestamp(value, name):
    require(isinstance(value, str), f"Missing {name} timestamp.")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ReviewError(f"Invalid {name} timestamp.") from error
    require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0,
            f"{name} timestamp must be UTC.")
    return result


def canonical_url(value):
    """Step-32 syntax normalization only; no DNS or network access."""
    require(isinstance(value, str) and value.strip(), "Empty URL in frozen labels.")
    value = value.strip()
    require(len(value) <= 8192 and not any(ord(c) <= 32 or ord(c) == 127 for c in value), "Invalid URL characters.")
    p = urlsplit(value)
    require(p.scheme.lower() in {"http", "https"} and p.hostname and p.username is None and p.password is None,
            "Frozen URL must be HTTP(S) without embedded credentials.")
    host = p.hostname.rstrip(".").lower()
    require(host and "%" not in host, "Invalid hostname.")
    try:
        address = ipaddress.ip_address(host)
        host = address.compressed
        authority = f"[{host}]" if address.version == 6 else host
    except ValueError:
        host = host.encode("idna").decode("ascii")
        require(len(host) <= 253 and re.fullmatch(r"[a-z0-9.-]+", host) and ".." not in host,
                "Invalid hostname.")
        require(all(part and len(part) <= 63 and part[0] != "-" and part[-1] != "-" for part in host.split(".")),
                "Invalid hostname label.")
        authority = host
    port = p.port
    require(port is None or 1 <= port <= 65535, "Invalid URL port.")
    if port is not None and port != (443 if p.scheme.lower() == "https" else 80):
        authority += f":{port}"
    return urlunsplit((p.scheme.lower(), authority, quote(p.path or "/", safe="/%:@!$&'()*+,;=-._~"),
                       quote(p.query, safe="/%?:@!$&'()*+,;=-._~[]"), ""))


def load_frozen(raw):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    require(reader.fieldnames is not None and len(reader.fieldnames) == 4 and set(reader.fieldnames) == set(FIELDS),
            "Frozen input must have exactly the four step-32 fields.")
    rows, seen = [], set()
    for entry in reader:
        require(None not in entry and all(value is not None for value in entry.values()), "Malformed frozen CSV row.")
        entry = {key: value.strip() for key, value in entry.items()}
        if not any(entry.values()):
            continue
        require(all(entry.values()) and entry["label"] in LABELS, "Missing or invalid frozen label fields.")
        require(len(entry["label_source"]) <= 4096 and all(ord(c) >= 32 and ord(c) != 127 for c in entry["label_source"]),
                "Invalid frozen label source.")
        require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry["label_checked_at"]), "Invalid frozen label date.")
        date.fromisoformat(entry["label_checked_at"])
        normalized = canonical_url(entry["url"])
        require(normalized not in seen, "Duplicate normalized frozen URL.")
        seen.add(normalized)
        rows.append({"ordinal": len(rows) + 1, "csv_row": reader.line_num, **entry, "normalized_url": normalized})
        require(len(rows) <= MAX_ROWS, "Frozen input exceeds 2000 rows.")
    counts = Counter(row["label"] for row in rows)
    require(rows and counts["benign"] == counts["phish"], "Frozen input must have equal nonzero class counts.")
    return rows


def archive_html(blob, limit=ARCHIVE_UTF8_LIMIT):
    require(isinstance(blob, bytes) and len(blob) <= ARCHIVE_UTF8_LIMIT + 1024 * 1024,
            "Missing or oversized compressed HTML archive.")
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(blob, limit + 1)
        require(len(raw) <= limit and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
                "HTML archive is oversized, truncated or has trailing compressed data.")
        return raw.decode("utf-8")
    except (zlib.error, UnicodeError) as error:
        raise ReviewError("HTML archive is corrupt or is not valid UTF-8.") from error


def safe_ratio(n, d):
    return n / d if d else None


def wilson(n, d):
    if not d:
        return None
    z = 1.959963984540054
    p, divisor = n / d, 1 + z * z / d
    center = (p + z * z / (2 * d)) / divisor
    margin = z * math.sqrt(p * (1 - p) / d + z * z / (4 * d * d)) / divisor
    return [max(0.0, center - margin), min(1.0, center + margin)]


def classification_metrics(pairs):
    counts = Counter(pairs)
    tn, fp, fn, tp = (counts[p] for p in (("benign", "benign"), ("benign", "phish"), ("phish", "benign"), ("phish", "phish")))
    n = tn + fp + fn + tp
    recall, specificity = safe_ratio(tp, tp + fn), safe_ratio(tn, tn + fp)
    return {"positive_class": "phish", "scored_pages": n,
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
            "accuracy": safe_ratio(tp + tn, n),
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "precision": safe_ratio(tp, tp + fp), "recall": recall,
            "false_positive_rate": safe_ratio(fp, fp + tn), "false_negative_rate": safe_ratio(fn, fn + tp),
            "specificity": specificity, "f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
            "wilson_95_intervals": {"accuracy": wilson(tp + tn, n), "precision": wilson(tp, tp + fp),
                                    "recall": wilson(tp, tp + fn), "false_positive_rate": wilson(fp, fp + tn)},
            "rates_are_fractions_not_percentages": True,
            "denominator": "prediction_complete pages only; failures and inconclusive captures are excluded and reported separately"}


def csv_safe(value):
    return "'" + value if isinstance(value, str) and value[:1] in {"=", "+", "-", "@", "\t", "\r"} else value


def csv_text(value):
    return "" if value is None else str(csv_safe(value))


def prediction_rows(rows, records):
    outputs = []
    for row in rows:
        record = records[row["ordinal"]]
        item = {key: row[key] for key in ("csv_row", *FIELDS, "normalized_url")}
        item.update({key: record.get(key, "") for key in ("attempt_started_at_utc", "capture_completed_at_utc",
                    "prediction_completed_at_utc", "error", "html_sha256")})
        item.update(status=record["status"], final_url=record.get("capture", {}).get("final_url", ""))
        result = record.get("result", {})
        item["active_quality_flags"] = ";".join(result.get("active_quality_flags", []))
        if record["status"] == "prediction_complete":
            item.update(phishing_score=result["phishing_score"], prediction=result["static_model_prediction"],
                        correct=row["label"] == result["static_model_prediction"])
        elif record["status"] == "inconclusive_capture":
            item.update(diagnostic_score=result["phishing_score"], diagnostic_prediction=result["static_model_prediction"])
        outputs.append(item)
    return outputs


def duplicate_groups(rows, records, field):
    groups = defaultdict(list)
    for row in rows:
        record = records[row["ordinal"]]
        if "capture" not in record:
            continue
        key = record.get("html_sha256") if field == "html_sha256" else record["capture"]["final_url"]
        groups[key].append({"csv_row": row["csv_row"], "label": row["label"]})
    return [{field: key, "pages": pages, "conflicting_labels": len({x["label"] for x in pages}) > 1}
            for key, pages in sorted(groups.items()) if len(pages) > 1]


def recompute_report(rows, records, manifest):
    coverage = {label: {"selected": 0, "prediction_complete": 0, "inconclusive_capture": 0,
                        "capture_failed": 0, "interrupted_capture": 0, "pending": 0} for label in LABELS}
    pairs = []
    for row in rows:
        record = records[row["ordinal"]]
        coverage[row["label"]]["selected"] += 1
        coverage[row["label"]][record["status"]] += 1
        if record["status"] == "prediction_complete":
            pairs.append((row["label"], record["result"]["static_model_prediction"]))
    for counts in coverage.values():
        counts["scored_coverage"] = safe_ratio(counts["prediction_complete"], counts["selected"])
        counts["complete_http_capture_coverage"] = safe_ratio(counts["prediction_complete"] + counts["inconclusive_capture"], counts["selected"])
    scored = {label: coverage[label]["prediction_complete"] for label in LABELS}
    metrics = classification_metrics(pairs)
    return {"status": "live_evaluation_complete", "evaluator_version": manifest["identity"]["evaluator_version"],
            "protocol_sha256": manifest["identity_sha256"], "started_at_utc": manifest["started_at_utc"],
            "model": manifest["identity"]["model"], "threshold": 0.5,
            "selected_pages": len(rows), "completed_attempts": len(rows),
            "input_class_counts": {label: coverage[label]["selected"] for label in LABELS},
            "input_list_balanced": True, "scored_class_counts": scored,
            "scored_subset_balanced": scored["benign"] == scored["phish"] and scored["benign"] > 0,
            "coverage_by_class": coverage,
            "distinct_input_hostnames_by_class": {label: len({urlsplit(row["normalized_url"]).hostname for row in rows if row["label"] == label}) for label in LABELS},
            "scored_coverage_all_selected_pages": safe_ratio(metrics["scored_pages"], len(rows)),
            "classification_metrics_on_scored_pages": metrics,
            "correct_scored_predictions_divided_by_all_selected_pages": safe_ratio(metrics["confusion_matrix"]["TP"] + metrics["confusion_matrix"]["TN"], len(rows)),
            "all_selected_fraction_is_not_classification_accuracy": True,
            "duplicate_capture_groups": {field: duplicate_groups(rows, records, field) for field in ("final_url", "html_sha256")},
            "labels_verified_by_evaluator": False, "model_trained_or_tuned": False,
            "javascript_executed": False, "external_javascript_downloaded": False,
            "sampling_design": "user-selected balanced live pilot; no probability sampling assumed"}


class SourceParser(HTMLParser):
    """Observe markup and bounded lexical hints; never resolve a reference."""
    TEXT_LIMIT = 100000
    SCRIPT_LIMIT = 1024 * 1024

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.counts = Counter()
        self.stack = []
        self.text, self.title, self.scripts = [], [], []
        self.text_chars = self.script_chars = 0
        self.text_cut = self.script_cut = False
        self.event_navigation_count = 0
        self.depth_cut = False

    def handle_starttag(self, tag, attributes):
        if self.depth_cut:
            return
        attrs = dict(attributes)
        self.counts["tag_count"] += 1
        if tag in {"form", "script", "textarea", "input", "button", "select"}:
            self.counts[tag + "_count"] += 1
        if tag == "script" and attrs.get("src"):
            self.counts["external_script_count"] += 1
        if tag == "input":
            typ = (attrs.get("type") or "text").lower()
            if typ == "password":
                self.counts["password_input_count"] += 1
            if typ not in {"hidden", "submit", "button", "reset", "image"}:
                self.counts["static_user_entry_input_count"] += 1
        if tag == "meta" and (attrs.get("http-equiv") or "").lower() == "refresh":
            self.counts["meta_refresh_count"] += 1
        for key, value in attributes:
            if key.startswith("on") and value and navigation_hint(value):
                self.event_navigation_count += 1
        # HTML void elements have no stack entry; hidden/style hints are source observations only.
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            if len(self.stack) >= MAX_MARKUP_DEPTH:
                self.depth_cut = True
                return
            parent_hidden = self.stack[-1][1] if self.stack else False
            style = re.sub(r"\s+", "", attrs.get("style") or "").lower()
            hidden = parent_hidden or tag in {"script", "style", "template", "head"} or "hidden" in attrs or "display:none" in style or "visibility:hidden" in style
            self.stack.append((tag, hidden))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if self.depth_cut:
            return
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if self.depth_cut:
            return
        tags = {tag for tag, _ in self.stack}
        if "title" in tags and "script" not in tags and "style" not in tags:
            remaining = 512 - sum(map(len, self.title))
            if remaining > 0:
                self.title.append(data[:remaining])
        if "script" in tags:
            remaining = self.SCRIPT_LIMIT - self.script_chars
            if remaining > 0:
                self.scripts.append(data[:remaining])
                self.script_chars += min(len(data), remaining)
            self.script_cut |= len(data) > remaining
        elif not self.stack or not self.stack[-1][1]:
            remaining = self.TEXT_LIMIT - self.text_chars
            if remaining > 0:
                self.text.append(data[:remaining])
                self.text_chars += min(len(data), remaining)
            self.text_cut |= len(data) > remaining


def navigation_hint(text):
    return bool(re.search(r"\b(?:window\s*\.\s*|document\s*\.\s*)?location\s*(?:\.\s*(?:href\s*=|replace\s*\(|assign\s*\()|=)", text, re.I))


def inspect_html(html):
    parser = SourceParser()
    parser.feed(html[:MARKUP_SCAN_CHARACTERS])
    parser.close()
    visible = re.sub(r"\s+", " ", " ".join(parser.text)).strip()
    script = "\n".join(parser.scripts)
    navigation = navigation_hint(script) or parser.event_navigation_count > 0
    title = re.sub(r"\s+", " ", "".join(parser.title)).strip()
    flags = []
    lower = (title + " " + visible).lower()
    holding = any(phrase in lower for phrase in ("preparando sua navega", "redirecionando", "redirecting you", "you are being redirected", "please wait while we redirect", "redirecting to", "redirecting..."))
    markup_cut = len(html) > MARKUP_SCAN_CHARACTERS or parser.depth_cut
    if holding and (navigation or parser.counts["meta_refresh_count"]) and not markup_cut:
        flags.append("redirect_holding_page_hint")
    if "continue shopping" in lower and ("click the button below" in lower or "amazon.com" in lower) and len(visible) < 2000 and not markup_cut:
        flags.append("continue_shopping_interstitial_hint")
    controls = parser.counts["static_user_entry_input_count"] + parser.counts["textarea_count"] + parser.counts["select_count"]
    if len(visible) < 200 and controls == 0 and navigation and len(html) < 10000 and not markup_cut:
        flags.append("tiny_script_navigation_shell_hint")
    if len(visible) < 100 and controls == 0 and parser.counts["form_count"] == 0 and parser.counts["script_count"] > 0 and not markup_cut:
        flags.append("low_text_script_source_hint")
    counts = {key: parser.counts[key] for key in ("tag_count", "form_count", "script_count", "external_script_count", "input_count", "password_input_count", "textarea_count", "select_count", "button_count", "static_user_entry_input_count", "meta_refresh_count")}
    return {"title": title, "html_character_count": len(html), "visible_source_text_characters_observed": len(visible),
            "visible_source_text_sha256": hashlib.sha256(visible.encode("utf-8")).hexdigest(),
            "visible_source_text_excerpt": visible[:600], **counts,
            "inline_javascript_navigation_lexical_hint": navigation,
            "inline_script_scan_cut": parser.script_cut, "visible_text_scan_cut": parser.text_cut,
            "markup_scan_cut": markup_cut, "markup_depth_cut": parser.depth_cut,
            "review_hints": flags, "source_inspection_only": True,
            "hints_do_not_change_capture_status_or_label": True}


def failure_category(error, status):
    if status == "interrupted_capture":
        return "interrupted_capture"
    match = re.search(r"\bHTTP\s+(\d{3})\b", error, re.I)
    if match:
        return "HTTP_" + match.group(1)
    lower = error.lower()
    if "resolve" in lower or "hostname" in lower and "dns" in lower:
        return "hostname_resolution"
    if "timeout" in lower or "timed out" in lower:
        return "timeout"
    if "incomplete" in lower or "connection" in lower or "ssl" in lower or "tls" in lower:
        return "incomplete_or_transport_error"
    return "other_capture_error"


def ensure_inactive(run_dir):
    for suffix in ("-wal", "-journal"):
        sidecar = run_dir / ("checkpoint.sqlite3" + suffix)
        require(not sidecar.exists() or sidecar.stat().st_size == 0,
                f"Checkpoint has an active/recovery sidecar ({sidecar.name}). Finish step 32 first.")


def load_source(run_dir):
    run_dir = Path(run_dir).resolve()
    require(run_dir.is_dir(), f"Run folder is missing: {run_dir}")
    ensure_inactive(run_dir)
    for name in SOURCE_FILES:
        require((run_dir / name).is_file() and not (run_dir / name).is_symlink(), f"Missing or linked source artifact: {name}")
    hashes = {name: file_sha(run_dir / name) for name in SOURCE_FILES}
    try:
        manifest = strict_json(bounded_read(run_dir / SOURCE_FILES[0], MAX_JSON_BYTES))
        identity = manifest["identity"]
        require(manifest["identity_sha256"] == object_sha(identity), "Protocol identity checksum mismatch.")
        require(identity["schema_version"] == 1, "Unsupported checkpoint protocol schema.")
        require(identity["evaluator_version"] == "1.0.0", "Unsupported evaluator version.")
        raw = bounded_read(run_dir / "frozen_input.csv", MAX_INPUT_BYTES)
        require(hashlib.sha256(raw).hexdigest() == identity["input_csv_sha256"], "Frozen input bytes checksum mismatch.")
        rows = load_frozen(raw)
        require(rows == identity["rows"] and object_sha(rows) == identity["normalized_rows_sha256"], "Frozen input/protocol normalized rows mismatch.")
        model = identity["model"]
        require(model["threshold"] == 0.5 and model.get("model_trained") is False, "Protocol model/threshold provenance mismatch.")
        require(model.get("model_name") == "lightgbm_context_all_features" and model.get("feature_count") == 106
                and model.get("best_iteration") == 1269 and model.get("numeric_type") == "float32"
                and model.get("runtime_version") == "1.0.0", "Unsupported frozen model descriptor.")
        for field in ("model_bundle_sha256", "frozen_specification_sha256"):
            require(isinstance(model.get(field), str) and re.fullmatch(r"[0-9a-f]{64}", model[field]), "Invalid model checksum descriptor.")
        capture = identity["capture"]
        require(capture["basis"] == "http_source" and capture["javascript_executed"] is False and capture["attempts_per_url"] == 1
                and capture["max_html_bytes"] == 32 * 1024 * 1024 and capture["timeout_seconds"] == 45 and capture["max_redirects"] == 5,
                "Protocol capture settings differ from step 32.")
        order = [row["ordinal"] for row in rows]
        random.Random(42).shuffle(order)
        require(identity["shuffle_seed"] == 42 and identity["request_order_ordinals"] == order, "Protocol request order mismatch.")
        start = timestamp(manifest["started_at_utc"], "run start")
        report = strict_json(bounded_read(run_dir / "live_evaluation_report.json", MAX_JSON_BYTES))
        end = timestamp(report["generated_at_utc"], "report generation")
        require(start <= end and report.get("status") == "live_evaluation_complete", "Source evaluation is incomplete or has invalid times.")
        records, audits, source_copies = {}, {}, {}
        copy_candidates = copy_error_candidates = 0
        # Never instantiate the evaluator's writable Checkpoint class.
        connection = sqlite3.connect((run_dir / "checkpoint.sqlite3").as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            require(connection.execute("PRAGMA query_only").fetchone()[0] == 1, "SQLite query-only mode was not enabled.")
            require(connection.execute("PRAGMA quick_check").fetchall() == [("ok",)], "SQLite quick_check failed.")
            schema = connection.execute("PRAGMA table_info(pages)").fetchall()
            require([(r[1], r[2], r[3], r[5]) for r in schema] == [("ordinal", "INTEGER", 0, 1), ("record_json", "TEXT", 1, 0), ("record_sha256", "TEXT", 1, 0), ("html_zlib", "BLOB", 0, 0)], "Unexpected checkpoint pages schema.")
            require(connection.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == len(rows), "Checkpoint row count differs from the complete frozen input.")
            lengths = connection.execute("SELECT MAX(length(CAST(record_json AS BLOB))), MAX(length(html_zlib)) FROM pages").fetchone()
            require(lengths[0] is not None and lengths[0] <= MAX_JSON_BYTES
                    and (lengths[1] is None or lengths[1] <= ARCHIVE_UTF8_LIMIT + 1024 * 1024),
                    "Checkpoint contains an oversized JSON record or compressed archive.")
            for ordinal, encoded, stored_sha, blob in connection.execute("SELECT ordinal,record_json,record_sha256,html_zlib FROM pages ORDER BY ordinal"):
                require(isinstance(ordinal, int) and 1 <= ordinal <= len(rows), "Unexpected checkpoint ordinal.")
                require(isinstance(encoded, str) and len(encoded.encode("utf-8")) <= MAX_JSON_BYTES, "Oversized checkpoint record.")
                require(hashlib.sha256(encoded.encode("utf-8")).hexdigest() == stored_sha, f"Checkpoint record {ordinal} checksum mismatch.")
                record = strict_json(encoded)
                row = rows[ordinal - 1]
                require(record["ordinal"] == ordinal and record["url"] == row["normalized_url"], "Checkpoint row/URL identity mismatch.")
                require(record["phase"] == "complete" and record["status"] in STATUSES, "Checkpoint contains a partial or unknown capture state.")
                attempted = timestamp(record["attempt_started_at_utc"], "attempt start")
                captured = timestamp(record["capture_completed_at_utc"], "capture completion")
                require(start <= attempted <= captured <= end, "Capture timestamps differ from run/report timing.")
                require(isinstance(record.get("attempt_id"), str) and re.fullmatch(r"[0-9a-f]{32}", record["attempt_id"]), "Invalid capture attempt identity.")
                if record["status"] in {"prediction_complete", "inconclusive_capture"}:
                    metadata, result = record["capture"], record["result"]
                    html = archive_html(blob)
                    html_sha = hashlib.sha256(html.encode("utf-8")).hexdigest()
                    require(html.strip() and html_sha == record["html_sha256"], "Captured HTML checksum mismatch or empty archive.")
                    require(metadata.get("requested_url") == record["url"] and metadata.get("capture_basis") == "http_source"
                            and metadata.get("capture_complete") is True and metadata.get("status_code") == 200
                            and isinstance(metadata.get("final_url"), str) and metadata["final_url"]
                            and metadata.get("javascript_executed") is False and metadata.get("external_resources_requested") is False,
                            "Capture metadata differs from the saved HTTP-source protocol.")
                    canonical_url(metadata["final_url"])
                    require(result.get("status") == record["status"] and result.get("model") == model
                            and result.get("capture_basis") == "http_source" and result.get("capture_basis_verified") is True
                            and result.get("page_url") == metadata["final_url"] and result.get("html_sha256") == html_sha
                            and result.get("javascript_executed_by_detector") is False
                            and result.get("external_javascript_downloaded_by_detector") is False,
                            "Prediction capture/model provenance mismatch.")
                    score = result.get("phishing_score")
                    require(isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) and 0 <= score <= 1
                            and result.get("static_model_prediction") == ("phish" if score >= 0.5 else "benign"), "Prediction score/threshold mismatch.")
                    flags = result.get("active_quality_flags")
                    require(isinstance(flags, list) and all(isinstance(v, str) for v in flags), "Invalid original quality flags.")
                    predicted = timestamp(record["prediction_completed_at_utc"], "prediction completion")
                    require(captured <= predicted <= end, "Prediction timing mismatch.")
                    audits[ordinal] = inspect_html(html)
                    wrong = record["status"] == "prediction_complete" and result["static_model_prediction"] != row["label"]
                    if wrong or audits[ordinal]["review_hints"]:
                        copy_candidates += 1
                        copy_error_candidates += int(wrong)
                        if wrong and len(source_copies) >= MAX_SOURCE_COPIES:
                            optional = [key for key, value in source_copies.items() if not value["wrong"]]
                            if optional:
                                del source_copies[optional[-1]]
                        if len(source_copies) < MAX_SOURCE_COPIES:
                            raw_html = html.encode("utf-8")
                            source_copies[ordinal] = {"raw": raw_html[:SOURCE_COPY_BYTES], "original_bytes": len(raw_html), "wrong": wrong}
                else:
                    require("result" not in record and "capture" not in record and "html_sha256" not in record and blob is None
                            and "prediction_completed_at_utc" not in record, "Failed/interrupted capture contains prediction or HTML artifacts.")
                    require(isinstance(record.get("error"), str) and record["error"], "Failed capture has no recorded error.")
                records[ordinal] = record
        finally:
            connection.close()
        require(len({record["attempt_id"] for record in records.values()}) == len(records), "Duplicate capture attempt identities.")
        recomputed = recompute_report(rows, records, manifest)
        for key, value in recomputed.items():
            require(report.get(key) == value, f"Original report disagrees with checkpoint: {key}.")
        expected_csv = prediction_rows(rows, records)
        reader = csv.DictReader(io.StringIO(bounded_read(run_dir / "live_evaluation_predictions.csv", MAX_JSON_BYTES).decode("utf-8-sig"), newline=""))
        require(reader.fieldnames == CSV_COLUMNS, "Prediction CSV headers differ from step 32.")
        csv_rows = []
        for actual in reader:
            require(None not in actual and all(value is not None for value in actual.values()), "Malformed prediction CSV row.")
            csv_rows.append(actual)
            require(len(csv_rows) <= MAX_ROWS, "Prediction CSV exceeds 2000 rows.")
        require(len(csv_rows) == len(expected_csv), "Prediction CSV row count mismatch.")
        for actual, expected in zip(csv_rows, expected_csv):
            require(actual == {key: csv_text(expected.get(key, "")) for key in CSV_COLUMNS}, f"Prediction CSV disagrees with checkpoint at row {expected['csv_row']}.")
        return {"run_dir": str(run_dir), "input_hashes": hashes, "manifest": manifest, "report": report,
                "rows": rows, "records": records, "audits": audits, "source_copies": source_copies,
                "copy_candidates": copy_candidates, "copy_error_candidates": copy_error_candidates,
                "prediction_rows": expected_csv}
    finally:
        ensure_inactive(run_dir)
        require({name: file_sha(run_dir / name) for name in SOURCE_FILES} == hashes, "Source artifacts changed during review. No review can be trusted.")


def inside(path, parent):
    return path == parent or parent in path.parents


def validate_output(run_dir, output_dir):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    require(not inside(output_dir, run_dir) and not inside(run_dir, output_dir), "Output folder must be separate from the source run folder.")
    # When the source is PROJECT/reports/RUN, protect the project's code/model folders.
    if run_dir.parent.name.lower() == "reports":
        project = run_dir.parent.parent
        if inside(output_dir, project):
            require(output_dir.parent == project / "reports" and output_dir != project / "reports",
                    "Within the project, use a separate direct subfolder of reports for review output.")
    require(not output_dir.exists() or output_dir.is_dir() and not any(output_dir.iterdir()),
            "Review output folder is not empty. Keep it and choose a new --output-dir.")
    return output_dir


def markdown_text(value):
    # HTML escapes prevent links/autolinks and code/markup supplied by a captured site.
    value = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*_{}\[\]()!|#])", r"\\\1", value).replace("\r", " ").replace("\n", " ")


def percent(value):
    return "unavailable" if value is None else f"{100 * value:.2f}%"


def build_review(source):
    index, errors, failure_counts, hint_counts = [], [], Counter(), Counter()
    for row, original in zip(source["rows"], source["prediction_rows"]):
        ordinal = row["ordinal"]
        record = source["records"][ordinal]
        audit = source["audits"].get(ordinal)
        item = {"ordinal": ordinal, **original, "error_type": "", "source_observations": audit}
        if original["status"] == "prediction_complete" and not original["correct"]:
            item["error_type"] = "false_positive" if row["label"] == "benign" else "false_negative"
            item["stored_model_features"] = record["result"].get("features", {})
            item["feature_values_are_diagnostic_not_causal_attributions"] = True
            errors.append(item)
        if original["status"] in {"capture_failed", "interrupted_capture"}:
            item["failure_category"] = failure_category(record["error"], record["status"])
            failure_counts[item["failure_category"]] += 1
        if audit:
            hint_counts.update(audit["review_hints"])
        index.append(item)
    copies = sorted(source["source_copies"].items(), key=lambda pair: (not pair[1]["wrong"], pair[0]))[:MAX_SOURCE_COPIES]
    copy_metadata = []
    for ordinal, copy in copies:
        copied = len(copy["raw"])
        copy_metadata.append({"ordinal": ordinal, "csv_row": source["rows"][ordinal - 1]["csv_row"],
                              "path": f"source_copies/row_{source['rows'][ordinal - 1]['csv_row']:04d}.html.txt",
                              "copied_utf8_bytes": copied, "original_utf8_bytes": copy["original_bytes"],
                              "complete": copied == copy["original_bytes"],
                              "copy_sha256": hashlib.sha256(copy["raw"]).hexdigest(),
                              "purpose": "plain text source inspection; never execute or open as a webpage"})
    text_groups = defaultdict(list)
    for ordinal, audit in source["audits"].items():
        if not audit["visible_text_scan_cut"] and not audit["markup_scan_cut"] and audit["visible_source_text_characters_observed"] > 0:
            text_groups[audit["visible_source_text_sha256"]].append(source["rows"][ordinal - 1]["csv_row"])
    repeated_text = [{"visible_source_text_sha256": key, "csv_rows": values,
                      "diagnostic_only": True, "is_not_exact_html_duplication": True}
                     for key, values in sorted(text_groups.items()) if len(values) > 1]
    return {"review_version": VERSION, "review_generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_run_dir": source["run_dir"], "source_inputs_sha256": source["input_hashes"],
            "source_inputs_unchanged_after_validation": True,
            "integrity_checks_detect_consistency_not_label_truth_or_authenticity": True,
            "original_protocol_sha256": source["manifest"]["identity_sha256"],
            "original_report_retained": source["report"],
            "reviewed_rows": len(index), "error_count": len(errors), "errors": errors,
            "failure_category_counts": dict(sorted(failure_counts.items())),
            "source_hint_counts": dict(sorted(hint_counts.items())),
            "duplicate_capture_groups": source["report"]["duplicate_capture_groups"],
            "repeated_visible_source_text_groups": repeated_text,
            "all_rows": index, "source_copy_manifest": copy_metadata,
            "source_copy_limits": {"max_copies": MAX_SOURCE_COPIES, "max_bytes_each": SOURCE_COPY_BYTES,
                                   "candidate_copies": source["copy_candidates"], "written_copies": len(copies),
                                   "omitted_candidate_copies": source["copy_candidates"] - len(copies),
                                   "omitted_error_copies": source["copy_error_candidates"] - sum(copy["wrong"] for _, copy in copies)},
            "source_inspection_limits": {"max_markup_characters": MARKUP_SCAN_CHARACTERS, "max_markup_depth": MAX_MARKUP_DEPTH,
                                         "max_visible_text_characters": SourceParser.TEXT_LIMIT, "max_inline_script_characters": SourceParser.SCRIPT_LIMIT},
            "changes_to_original_evaluation": {"labels": False, "scores": False, "statuses": False, "metrics": False, "excluded_rows": 0},
            "limitations": ["This is offline inspection of saved HTTP source, not browser behavior validation.",
                            "No network requests, model loading, training, JavaScript execution, external resources or form submission occur.",
                            "Visible-source excerpts are approximate: CSS files and JavaScript were not evaluated. Script/style/template/hidden inline text is omitted.",
                            "Markup inspection is limited to the first 2097152 characters and 512 nested elements; scan-cut flags disclose incomplete observations.",
                            "Navigation/interstitial/shell hints may be benign and need manual review. External scripts or no form alone are not quality failures.",
                            "Stored feature values describe the prior classifier input, not which features caused a prediction.",
                            "Duplicate captures and source hints retain every original row and denominator.",
                            "Checksums check artifact consistency; they do not authenticate source evidence or establish phishing ground truth."]}


def make_markdown(review):
    report = review["original_report_retained"]
    metrics = report["classification_metrics_on_scored_pages"]
    matrix = metrics["confusion_matrix"]
    lines = ["# Step 33 — offline live error review", "", "The original evaluation is retained without changes. Source hints are observations for manual review.", "",
             f"Reviewed {review['reviewed_rows']} selected URLs; {metrics['scored_pages']} scored, {review['error_count']} incorrect predictions.",
             f"Original accuracy: **{percent(metrics['accuracy'])}**. Coverage: **{percent(report['scored_coverage_all_selected_pages'])}**.",
             f"Original confusion matrix: TN={matrix['TN']}, FP={matrix['FP']}, FN={matrix['FN']}, TP={matrix['TP']}.", "",
             "## Incorrect predictions", "", "URLs below are plain evidence text. Use the saved text copies to inspect source.", "",
             "| CSV row | Type | Label | Prediction | Original score | URL | Source hints |", "|---|---|---|---|---:|---|---|"]
    for item in review["errors"]:
        hint = "; ".join(item["source_observations"]["review_hints"]) or "none"
        values = [markdown_text(v) for v in (item["csv_row"], item["error_type"], item["label"], item["prediction"], f"{item['phishing_score']:.6f}", item["url"], hint)]
        # A literal colon entity avoids Markdown/GFM URL autolinking.
        values[5] = values[5].replace(":", "&#58;")
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## Capture failures", "", "Failures have no prediction and remain outside the scored denominator.", ""]
    lines += [f"- {markdown_text(key)}: {count}" for key, count in review["failure_category_counts"].items()]
    lines += ["", "## Source observations and duplicates", ""]
    lines += [f"- {markdown_text(key)}: {count} saved captures" for key, count in review["source_hint_counts"].items()]
    for kind, groups in review["duplicate_capture_groups"].items():
        lines.append(f"- Exact duplicate groups by {markdown_text(kind)}: {len(groups)}; all original rows retained.")
        for group in groups:
            lines.append("  - CSV rows " + ", ".join(str(p["csv_row"]) for p in group["pages"]) + (" (conflicting labels)" if group["conflicting_labels"] else ""))
    lines.append(f"- Repeated visible-source text groups: {len(review['repeated_visible_source_text_groups'])}; diagnostic template clues only.")
    for group in review["repeated_visible_source_text_groups"]:
        lines.append("  - CSV rows " + ", ".join(str(value) for value in group["csv_rows"]))
    lines += ["", "## Evidence files", "", "- `review_index.csv`: every selected row, original prediction/status and bounded source observations.",
              "- `live_error_review_report.json`: original report, all rows, stored error feature values, integrity checks and copy manifest.",
              "- `source_copies/*.html.txt`: plain UTF-8 source copies, limited to 2 MiB each and 24 copies. Check the manifest for truncation.",
              "", "## Limits", ""]
    lines += ["- " + markdown_text(text) for text in review["limitations"]]
    lines += ["", "## Input checksums", "", "All five source artifacts were hashed before and after inspection and before publishing this review.", ""]
    lines += [f"- `{name}`: `{value}`" for name, value in review["source_inputs_sha256"].items()]
    return "\n".join(lines) + "\n"


def save_review(source, output_dir):
    output_dir = validate_output(source["run_dir"], output_dir)
    review = build_review(source)
    require({name: file_sha(Path(source["run_dir"]) / name) for name in SOURCE_FILES} == source["input_hashes"], "Inputs changed before review output.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".step33-", dir=output_dir.parent))
    try:
        (stage / "live_error_review_report.json").write_text(json.dumps(review, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
        (stage / "live_error_review_summary.md").write_text(make_markdown(review), encoding="utf-8")
        columns = ["ordinal", *CSV_COLUMNS, "error_type", "failure_category", "title", "html_character_count", "form_count", "password_input_count", "textarea_count", "script_count", "external_script_count", "static_user_entry_input_count", "inline_javascript_navigation_lexical_hint", "review_hints", "markup_scan_cut", "markup_depth_cut", "visible_text_scan_cut", "inline_script_scan_cut", "visible_source_text_characters_observed", "visible_source_text_excerpt"]
        with (stage / "review_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, columns, lineterminator="\n")
            writer.writeheader()
            for item in review["all_rows"]:
                audit = item["source_observations"] or {}
                flat = {**item, **audit}
                flat["review_hints"] = ";".join(audit.get("review_hints", []))
                writer.writerow({key: csv_safe(flat.get(key, "")) for key in columns})
        if review["source_copy_manifest"]:
            (stage / "source_copies").mkdir()
        for item in review["source_copy_manifest"]:
            raw = source["source_copies"][item["ordinal"]]["raw"]
            # Trim only an incomplete final UTF-8 codepoint; never append HTML markup.
            text = raw.decode("utf-8", errors="ignore")
            data = text.encode("utf-8")
            item["copied_utf8_bytes"] = len(data)
            item["complete"] = len(data) == item["original_utf8_bytes"]
            item["copy_sha256"] = hashlib.sha256(data).hexdigest()
            (stage / item["path"]).write_bytes(data)
        (stage / "live_error_review_report.json").write_text(json.dumps(review, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
        ensure_inactive(Path(source["run_dir"]))
        require({name: file_sha(Path(source["run_dir"]) / name) for name in SOURCE_FILES} == source["input_hashes"], "Inputs changed while preparing review output.")
        # Recheck after staging: never replace a nonempty folder or unrelated file.
        validate_output(source["run_dir"], output_dir)
        if output_dir.exists():
            output_dir.rmdir()
        os.rename(stage, output_dir)
        return review
    finally:
        if stage.exists():
            resolved_stage = stage.resolve()
            require(resolved_stage.parent == output_dir.parent.resolve() and resolved_stage.name.startswith(".step33-")
                    and resolved_stage != output_dir and not stage.is_symlink(), "Refusing cleanup outside the verified staging folder.")
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline review of completed step-32 evidence; no model or network access.")
    project = Path(__file__).resolve().parent
    parser.add_argument("--run-dir", type=Path, default=project / "reports" / "live_evaluation_local_20261011")
    parser.add_argument("--output-dir", type=Path, default=project / "reports" / "live_error_review_v1")
    parser.add_argument("--check-only", action="store_true", help="Verify the saved run and print counts without writing anything.")
    args = parser.parse_args(argv)
    try:
        if not args.check_only:
            validate_output(args.run_dir, args.output_dir)
        source = load_source(args.run_dir)
        if args.check_only:
            review = build_review(source)
            print("Saved evaluation integrity check passed. No output files written.")
        else:
            review = save_review(source, args.output_dir)
            print("Live error review complete.")
        metrics = source["report"]["classification_metrics_on_scored_pages"]
        print(f"Selected: {len(source['rows'])}; scored: {metrics['scored_pages']}; incorrect predictions: {review['error_count']}; capture failures/interruption: {sum(review['failure_category_counts'].values())}")
        print(f"Original accuracy: {percent(metrics['accuracy'])}; original coverage: {percent(source['report']['scored_coverage_all_selected_pages'])}.")
        if not args.check_only:
            for name in ("live_error_review_report.json", "live_error_review_summary.md", "review_index.csv"):
                print(str(args.output_dir.resolve() / name))
        return 0
    except (ReviewError, KeyError, ValueError, TypeError, UnicodeError, OSError, sqlite3.Error, csv.Error, RecursionError, OverflowError) as error:
        print(f"Live error review stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
