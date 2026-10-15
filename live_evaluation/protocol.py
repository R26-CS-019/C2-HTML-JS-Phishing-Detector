"""Read labels before prediction; preserve input bytes and explicit denominators."""
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

FIELDS = ["url", "label", "label_source", "label_checked_at"]
LABELS = ("benign", "phish")
MAX_INPUT_BYTES = 32 * 1024 * 1024


class EvaluationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvaluationError(message)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def object_sha(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_url(value, normalizer):
    """Runtime syntax normalization plus removal of an explicit default port."""
    normalized = normalizer(value)
    parsed = urlsplit(normalized)
    if parsed.port == (443 if parsed.scheme == "https" else 80):
        host = parsed.hostname
        authority = f"[{host}]" if ":" in host else host
        return urlunsplit((parsed.scheme, authority, parsed.path, parsed.query, ""))
    return normalized


def load_url_list(path, normalizer, per_class=50, today=None):
    require(isinstance(per_class, int) and 1 <= per_class <= 1000, "--per-class must be 1..1000.")
    path = Path(path)
    require(path.is_file(), f"URL CSV is missing: {path}")
    require(path.stat().st_size <= MAX_INPUT_BYTES, "URL CSV exceeds 32 MiB.")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError as error:
        raise EvaluationError("Save the URL list as CSV UTF-8, not an Excel .xlsx file.") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    require(reader.fieldnames is not None and len(reader.fieldnames) == 4
            and set(reader.fieldnames) == set(FIELDS),
            "CSV header must contain exactly: " + ",".join(FIELDS))
    today = today or date.today()
    rows, seen, errors = [], {}, []
    try:
        for entry in reader:
            line = reader.line_num
            if None in entry or any(value is None for value in entry.values()):
                errors.append(f"CSV row {line}: expected four fields.")
                continue
            entry = {key: value.strip() for key, value in entry.items()}
            if not any(entry.values()):
                continue
            try:
                missing = [key for key in FIELDS if not entry[key]]
                require(not missing, "fill " + ", ".join(missing) + ".")
                require(entry["label"] in LABELS, "label must be benign or phish.")
                source = entry["label_source"]
                require(len(source) <= 4096 and all(ord(c) >= 32 and ord(c) != 127 for c in source),
                        "label_source must be a readable evidence URL or explanation, at most 4096 characters.")
                checked = entry["label_checked_at"]
                require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", checked), "label_checked_at must be YYYY-MM-DD.")
                checked_date = date.fromisoformat(checked)
                require(checked_date <= today, "label_checked_at cannot be in the future.")
                normalized = canonical_url(entry["url"], normalizer)
                require(normalized not in seen,
                        f"duplicate URL after normalization; first used in CSV row {seen.get(normalized)}.")
                seen[normalized] = line
                rows.append({"ordinal": len(rows) + 1, "csv_row": line,
                             **entry, "normalized_url": normalized})
            except ValueError as error:
                errors.append(f"CSV row {line}: {error}")
            if len(rows) > 2000 or len(errors) >= 15:
                break
    except csv.Error as error:
        errors.append(f"CSV parsing error: {error}")
    require(not errors, "URL list needs attention:\n  " + "\n  ".join(errors))
    counts = Counter(row["label"] for row in rows)
    require(counts == {label: per_class for label in LABELS},
            f"Expected {per_class} benign + {per_class} phish URLs; found "
            f"{counts.get('benign', 0)} benign + {counts.get('phish', 0)} phish.")
    return rows, raw


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def wilson_interval(successes, trials):
    """Two-sided approximate 95% Wilson interval; missing denominator is null."""
    if not trials:
        return None
    z = 1.959963984540054
    p = successes / trials
    divisor = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / divisor
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / divisor
    return [max(0.0, center - margin), min(1.0, center + margin)]


def classification_metrics(scored):
    """scored contains (independent true label, model prediction) pairs only."""
    counts = Counter(scored)
    tn = counts[("benign", "benign")]
    fp = counts[("benign", "phish")]
    fn = counts[("phish", "benign")]
    tp = counts[("phish", "phish")]
    n = tp + tn + fp + fn
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    return {
        "positive_class": "phish", "scored_pages": n,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "accuracy": safe_ratio(tp + tn, n),
        "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "precision": safe_ratio(tp, tp + fp), "recall": recall,
        "false_positive_rate": safe_ratio(fp, fp + tn),
        "false_negative_rate": safe_ratio(fn, fn + tp),
        "specificity": specificity, "f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "wilson_95_intervals": {
            "accuracy": wilson_interval(tp + tn, n),
            "precision": wilson_interval(tp, tp + fp),
            "recall": wilson_interval(tp, tp + fn),
            "false_positive_rate": wilson_interval(fp, fp + tn),
        },
        "rates_are_fractions_not_percentages": True,
        "denominator": "prediction_complete pages only; failures and inconclusive captures are excluded and reported separately",
    }
