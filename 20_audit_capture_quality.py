"""Inspect archived captures for the selected script 18 errors and controls.

Run in the same project and virtual environment. Uses existing scripts 09, 14,
15 and their duckdb/tldextract dependencies. Audits the 128 selected pages by
default, not the whole dataset. Checks raw HTML fingerprints, validation
membership, all 106 float32 features and all 11 quality values.

Reads only selected validation IDs from the original TRAIN-release Parquets.
Writes a separate report, index and optional capped .html.txt source copies.
Use --no-source-copies if writing those extra copies fails; HTML is still read,
verified and analysed in memory. Resumes verified v1.0 checkpoints after each
page. No fitting, relabelling, filtering, live requests or JS execution.
Hints describe archived text; they do not establish page safety or a bad label.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import re
import struct
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

VERSION = "1.1"
PREVIOUS_VERSION_SHA = "f29704e994991402150c9f5eb62cc0343e799f972548599d182726f6427c0b6d"
MODEL = "lightgbm_context_all_features"
REPORT_NAME = "capture_quality_report.json"
CHECKPOINT_NAME = "capture_quality_checkpoint.json"
PINS = {
    "extractor_script": "4c845262a0cb27c085a2fc80c4d37f427addad5405b7319ba20438f7e18ab7ef",
    "inspection_script": "dbc550cdebbfc1e36c94bcc27f34c05fdea966898abbe1e3cffa1d6f1e73ea52",
    "context_script": "62d468f83d124346cce3866894f85ae4bec4cdcd3163267b23625b61764d16eb",
    "review_script": "581792b4451133a4c27569f559ca540aebfddbac3a39db83837b5fce70e49633",
}
JS_REQUIRED = re.compile(r"(?:enable|turn on)\s+javascript|javascript\s+(?:is\s+)?(?:required|disabled)|does(?:n't| not) work[^.]{0,60}without javascript", re.I)
LEXICAL_TITLE = re.compile(r"<title\b[^>]{0,2000}>", re.I)
FLAGS = {
    "zero_parsed_title_characters": "The existing title_character_count is zero in the analysed prefix.",
    "lexical_title_with_zero_parsed_title": "A title-like opening string exists but the original parser counted zero title characters; strings in comments/scripts also match.",
    "human_verification_text_hint": "A bounded text scan matched a human-verification phrase; not proof that the whole page is a challenge.",
    "access_restriction_text_hint": "A bounded text scan matched an access-restriction phrase; not an observed HTTP status.",
    "unavailable_page_text_hint": "A bounded text scan matched an unavailable-page phrase; not confirmation of a removed page.",
    "javascript_required_text_hint": "A bounded text scan matched a request to enable JavaScript.",
    "declared_meta_refresh": "At least one parsed meta tag declares refresh; no navigation was performed.",
    "javascript_navigation_lexical_hint": "Existing capped JS scan matched location assignment/navigation syntax; execution and destination are unknown.",
    "cross_site_canonical_reference": "A parsed canonical link resolves to another site group; not evidence of an HTTP redirect.",
    "external_script_reference_present": "External script references exist, but their bodies were not captured by this feature pipeline.",
    "low_parsed_text_with_scripts": "At most 200 non-script/style text characters and at least one script; a diagnostic heuristic, not proof of an incomplete capture.",
    "no_static_password_cue": "Neither native password inputs nor script 15 password-cued controls were found under the existing caps.",
    "html_prefix_limited": "The original HTML analysis cap was reached.",
    "javascript_scan_limited": "The original inline/event JavaScript analysis cap was reached.",
    "parser_or_association_limit": "An original parse, unclosed raw-text or context association-limit flag is present.",
    "text_hint_scan_limited": "At least one inspection/context text-hint scan reached its storage limit.",
}


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_helper(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def error_type(label, score):
    return "false_positive" if label == "benign" and score >= .5 else "false_negative" if label == "phish" and score < .5 else "correct"


def validate_examples(review, common, context):
    examples = review["review_examples"]
    if not 1 <= len(examples) <= 300:
        raise ValueError("Expected 1 to 300 selected examples from script 18.")
    ids = set()
    features = set(common.FEATURE_NAMES) | set(context.CONTEXT_NAMES)
    quality = set(common.QUALITY_COLUMNS) | set(context.CONTEXT_QUALITY)
    for example in examples:
        meta = example["metadata"]
        page_id = meta["record_id"]
        if not re.fullmatch(r"[0-9a-f]{64}", page_id) or page_id in ids:
            raise ValueError("Invalid or repeated review record ID.")
        ids.add(page_id)
        if not re.fullmatch(r"train-\d+\.parquet", meta["source_file"]):
            raise ValueError("Only original TRAIN-release shards are permitted.")
        if meta["label"] not in ("benign", "phish") or any(meta[k] not in ("0", "1") for k in
                ("sampled_similarity_candidate", "in_validation_sensitivity_subset")):
            raise ValueError("Invalid review label or subset membership.")
        if set(example["feature_values"]) != features or set(example["quality"]) != quality:
            raise ValueError("Review does not contain the exact 106 features and 11 quality values.")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
               for v in example["feature_values"].values()):
            raise ValueError("Non-finite or negative saved feature value.")
        if any(type(v) is not int or v < 0 for v in example["quality"].values()):
            raise ValueError("Invalid saved quality value.")
        for key in ("new_model", "old_model"):
            score = example[key]["phishing_score"]
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid saved model score.")
            if example[key]["error_at_0_50"] != error_type(meta["label"], score):
                raise ValueError("Saved error category and score disagree.")
    return examples


def check_saved_scores(path, examples):
    selected = {e["metadata"]["record_id"]: e for e in examples}
    found = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            page_id = row["record_id"]
            if page_id not in selected:
                continue
            e = selected[page_id]
            if page_id in found or any(row[k] != v for k, v in e["metadata"].items()):
                raise ValueError("Selected prediction metadata differs from the review.")
            if (float(row["phishing_score"]) != e["new_model"]["phishing_score"]
                    or row["error_type"] != e["new_model"]["error_at_0_50"]):
                raise ValueError("Selected score/error differs from the saved model predictions.")
            found.add(page_id)
    if found != set(selected):
        raise ValueError("A selected example is missing from the saved predictions.")


def prepare(args):
    root = Path(__file__).resolve().parent
    paths = {
        "review_report": args.review_report,
        "feature_report": args.features_dir / "form_context_feature_report.json",
        "base_report": args.base_features_dir / "feature_extraction_report.json",
        "schema": args.features_dir / "static_feature_schema.json",
        "base_schema": args.base_features_dir / "static_feature_schema.json",
        "training_report": args.training_dir / "form_context_training_report.json",
        "new_scores": args.training_dir / MODEL / "validation_predictions.csv",
        "manifest": args.manifest,
        "extractor_script": root / "09_extract_static_features.py",
        "inspection_script": root / "14_inspect_saved_error_pages.py",
        "context_script": root / "15_extract_form_context.py",
        "review_script": root / "18_review_context_errors.py",
    }
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Missing input: {path}. Keep the earlier completed project files.")
    hashes = {key: digest_file(path) for key, path in paths.items()}
    for key, expected in PINS.items():
        if hashes[key] != expected:
            raise ValueError(f"Keep the existing verified version of {paths[key].name}.")
    review, features, base, schema, base_schema, training = (read_json(paths[k]) for k in
        ("review_report", "feature_report", "base_report", "schema", "base_schema", "training_report"))
    statuses = [(review, "context_error_review_complete"), (features, "form_context_feature_extraction_complete"),
                (base, "static_feature_extraction_complete"), (training, "form_context_model_training_complete")]
    if any(r.get("status") != s for r, s in statuses):
        raise ValueError("Use the completed script 09, 15, 16 and 18 reports.")
    if (review["model_name"] != MODEL or review["threshold"] != .5
            or review["review_code_sha256"] != hashes["review_script"]):
        raise ValueError("Use the current 106-feature model's script 18 review at threshold 0.50.")
    for r in (review, features, training):
        for key in ("labels_changed", "balanced_splits_changed", "official_test_evaluated"):
            if r.get(key) is not False:
                raise ValueError(f"Unexpected development report policy: {key}.")
    for key in ("feature_report", "schema", "base_schema", "training_report", "new_scores"):
        if hashes[key] != review["input_hashes"][key]:
            raise ValueError(f"Input differs from the completed context review: {paths[key].name}.")
    for key, recorded in (("base_report", "extraction_report"), ("base_schema", "schema"),
                          ("manifest", "manifest"), ("extractor_script", "extractor_script")):
        if hashes[key] != features["inputs"]["input_hashes"][recorded]:
            raise ValueError(f"Input differs from completed context extraction: {paths[key].name}.")
    inspection = load_helper(paths["inspection_script"], "capture_inspection14")
    common = load_helper(paths["extractor_script"], "capture_features09")
    context = load_helper(paths["context_script"], "capture_context15")
    caps = base["inputs"]["parameters"]
    if (set(caps) != {"max_html_characters", "max_js_characters"}
            or any(type(v) is not int or v < 1000 for v in caps.values())
            or caps != schema["parameters"] or caps != base_schema["parameters"]
            or caps != features["inputs"]["parameters"]):
        raise ValueError("Unexpected or inconsistent original extraction limits.")
    if (schema["feature_names"] != [*common.FEATURE_NAMES, *context.CONTEXT_NAMES]
            or schema["quality_columns_not_classifier_inputs"] != [*common.QUALITY_COLUMNS, *context.CONTEXT_QUALITY]
            or base_schema["feature_names"] != list(common.FEATURE_NAMES)
            or len(schema["feature_names"]) != 106
            or schema["association_limits"] != context.LIMITS
            or features["inputs"]["association_limits"] != context.LIMITS):
        raise ValueError("Feature schema or context limits differ from the recorded implementation.")
    if (schema["implementation_code_sha256"] != hashes["context_script"]
            or schema["base_implementation_code_sha256"] != hashes["extractor_script"]
            or base_schema["implementation_code_sha256"] != hashes["extractor_script"]
            or base["inputs"]["code_sha256"] != hashes["extractor_script"]
            or features["inputs"]["code_sha256"] != hashes["context_script"]
            or hashes["schema"] != features["outputs"][paths["schema"].name]["sha256"]
            or hashes["base_schema"] != base["outputs"][paths["base_schema"].name]["sha256"]
            or hashes["manifest"] != inspection.input_hash_named(base["inputs"]["inputs_sha256"], args.manifest.name)):
        raise ValueError("The original feature/manifest provenance differs.")
    examples = validate_examples(review, common, context)
    audited = inspection.read_selected_manifest(args.manifest, examples)
    check_saved_scores(paths["new_scores"], examples)
    expected = {s["name"]: s for s in base["inputs"]["source_files"]}
    if features["inputs"]["source_files"] != base["inputs"]["source_files"]:
        raise ValueError("The two extraction reports used different original source files.")
    source_stats = []
    for name in sorted({e["metadata"]["source_file"] for e in examples}):
        path = args.data_dir / name
        if not path.is_file() or not path.resolve().is_relative_to(args.data_dir.resolve()):
            raise ValueError(f"Missing original TRAIN-release source: {path}.")
        stat = path.stat()
        item = {"name": name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if item != expected.get(name):
            raise ValueError(f"Source metadata changed since extraction: {name}. Use the original dataset files.")
        source_stats.append(item)
    output = args.output_dir.resolve()
    reserved = [args.data_dir, args.features_dir, args.base_features_dir, args.training_dir,
                args.review_report.parent, args.manifest.parent]
    if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in reserved):
        raise ValueError("Use a separate audit output folder outside the dataset and prior result folders.")
    if any(p.resolve().is_relative_to(output) for p in paths.values()):
        raise ValueError("The audit output folder must not contain any input file.")
    return paths, hashes, review, examples, audited, caps, source_stats, common, context, inspection


def capture_parser_type(common, inspection):
    class CaptureParser(inspection.inspection_parser_type(common)):
        def __init__(self, url, resolver, max_js):
            super().__init__(url, resolver, max_js)
            self.tag_structure = Counter()
            self.title_events = []
            self.metadata_refs = []
            self.metadata_ref_count = 0
            self.app_markers = []

        def handle_starttag(self, tag, attrs):
            a = {k: v or "" for k, v in attrs}
            if tag in ("html", "head", "body", "title", "noscript"):
                self.tag_structure[tag + "_open"] += 1
            if tag == "title" and len(self.title_events) < 8:
                self.title_events.append({"line": self.getpos()[0], "column": self.getpos()[1],
                    "script_active_before_tag": self.script_active, "style_active_before_tag": self.style_active})
            kind, raw = None, ""
            if tag == "link" and "canonical" in a.get("rel", "").lower().split():
                kind, raw = "canonical", a.get("href", "")
            elif tag == "meta" and a.get("http-equiv", "").strip().lower() == "refresh":
                kind, raw = "meta_refresh", a.get("content", "")
            if kind:
                self.metadata_ref_count += 1
                if len(self.metadata_refs) < 24:
                    self.metadata_refs.append((kind, raw))
            markers = [k for k in ("data-reactroot", "ng-app") if k in a]
            if tag in ("div", "main", "section") and a.get("id", "").lower() in ("root", "app", "__next", "__nuxt"):
                markers.append("common_app_root_id")
            if markers and len(self.app_markers) < 12:
                self.app_markers.append({"tag": tag, "id": inspection.brief(a.get("id"), 100), "markers": markers})
            super().handle_starttag(tag, attrs)

        def handle_endtag(self, tag):
            if tag in ("html", "head", "body", "title", "noscript"):
                self.tag_structure[tag + "_close"] += 1
            super().handle_endtag(tag)

        def notes(self, prefix_was_cut):
            notes = super().notes(prefix_was_cut)
            # Resolve with the full first base, not the shortened display URL.
            base = self.page_url
            if self.first_base is not None:
                resolved, _, _, kind = self.resolver.resolve(self.first_base, self.page_url)
                if kind == "http":
                    base = resolved
            references = []
            for kind, raw in self.metadata_refs:
                reference = raw
                if kind == "meta_refresh":
                    match = common.REFRESH_URL.search(raw)
                    reference = match.group(1).strip().strip("\"'") if match else ""
                resolved, group, scheme, category = self.resolver.resolve(reference, base) if reference.strip() else ("", "", "", "absent")
                references.append({"kind": kind, "raw_value": inspection.brief(raw),
                    "resolved_reference": inspection.brief(resolved, 1500), "target_site_group": group,
                    "scheme": scheme, "reference_type": category,
                    "cross_site": category == "http" and bool(self.page_group) and group != self.page_group})
            text = inspection.normalized_text(" ".join(self.text_parts))
            js_hints = []
            for match in JS_REQUIRED.finditer(text):
                js_hints.append(text[max(0, match.start()-80):match.end()+120])
                if len(js_hints) == 3:
                    break
            notes["capture_structure"] = {
                "parsed_tag_counts": dict(self.tag_structure), "title_open_events": self.title_events,
                "metadata_references": references, "metadata_reference_notes_cut": self.metadata_ref_count > len(references),
                "common_app_marker_examples": self.app_markers, "javascript_required_text_excerpts": js_hints,
                "observation_type": "Static parser and bounded text notes; no browser DOM or runtime observation."}
            return notes
    return CaptureParser


def inspect_static_page(record, extra, example, audited, common, inspection,
                        resolver, parser_type, caps, output, save_source_copies):
    """Script 14's verified parsing policy, with raw-source export optional.

    Keep the original script 14 file intact: its hash belongs to earlier reports.
    This adapter only separates the optional copy from the inspection result.
    """
    page_id, label, url, full_length, raw_hash, prefix, *extra_values = record
    if (page_id != audited["sha256"] or label != audited["label"]
            or full_length != int(audited["html_characters"]) or raw_hash != audited["raw_html_sha256"]):
        raise ValueError(f"Selected HTML, supplied ID or label differs from its audited content: {page_id}.")
    page_url = common.canonical_page_url(url)
    if resolver.resolve(page_url, page_url)[1] != audited["site_group"]:
        raise ValueError(f"Archived URL site group differs from the audited manifest: {page_id}.")
    features, quality = common.extract_static_features(prefix, url, resolver, **caps, full_html_characters=full_length)
    differences = [name for name in common.FEATURE_NAMES
                   if inspection.as_float32(features[name]) != example["feature_values"][name]]
    if differences or quality != example["quality"]:
        raise ValueError(f"Archived HTML does not reproduce the saved baseline features/quality: {page_id}; {differences[:6]}.")
    p = parser_type(url, resolver, caps["max_js_characters"])
    inspection_parse_error = 0
    try:
        p.feed(prefix)
        p.close()
    except (AssertionError, ValueError):
        inspection_parse_error = 1
    p.finish()
    if p.js_scanned != features["js_analysed_character_count"] or inspection_parse_error != quality["html_parse_error"]:
        raise ValueError("Inspection parser differs from the original parsing policy.")
    prefix_hash = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    source_file = source_hash = None
    if save_source_copies:
        source_path = output / "page_source_text" / f"{page_id}.html.txt"
        operation = "Writing the optional raw-source text copy"
        try:
            source_path.write_text(prefix, encoding="utf-8", newline="")
            operation = "Reading back the optional raw-source text copy"
            source_hash = digest_file(source_path)
        except OSError as error:
            error.add_note(f"{operation} failed for record {page_id}. Rerun with --no-source-copies to analyse all pages without exporting new raw-source copies.")
            raise
        if source_hash != prefix_hash:
            raise ValueError(f"Written source copy differs from its in-memory HTML prefix: {page_id}.")
        source_file = str(source_path.relative_to(output))
    annotations = {key: (value.isoformat() if hasattr(value, "isoformat") else value)
                   for key, value in zip(extra, extra_values)}
    return {"review_category": example["review_category"], "metadata": example["metadata"],
        "phishing_score": example["phishing_score"], "archived_page_url": url,
        "dataset_annotations_for_inspection_only": annotations,
        "html_content_sha256": raw_hash, "full_html_characters": full_length,
        "analysed_source_prefix_characters": len(prefix),
        "stored_source_prefix_characters": len(prefix) if save_source_copies else 0,
        "source_text_file": source_file, "source_text_file_sha256": source_hash,
        "source_prefix_utf8_sha256": prefix_hash,
        "source_copy_status": "saved" if save_source_copies else "not_requested",
        "verification": {"raw_html_hash_matches_audited_manifest": True,
                         "all_baseline_float32_features_reproduced": True, "quality_flags_reproduced": True},
        "baseline_feature_values": example["feature_values"], "baseline_quality_flags": quality,
        "inspection": p.notes(bool(quality["html_analysis_truncated"]))}


def inspect_record(record, extra, example, audited, common, context, inspection, resolver, parser_type, caps, output,
                   save_source_copies=True):
    prefix, url = record[5], record[2]
    context_values, context_quality = context.extract_form_context(prefix, url, common, resolver, caps["max_html_characters"])
    differences = [k for k in context.CONTEXT_NAMES if inspection.as_float32(context_values[k]) != example["feature_values"][k]]
    if differences or context_quality != {k: example["quality"][k] for k in context.CONTEXT_QUALITY}:
        raise ValueError(f"Archived HTML does not reproduce saved context values: {record[0]}; {differences[:6]}.")
    old_example = {"review_category": example["review_category"], "metadata": example["metadata"],
        "phishing_score": example["new_model"]["phishing_score"],
        "feature_values": {k: example["feature_values"][k] for k in common.FEATURE_NAMES},
        "quality": {k: example["quality"][k] for k in common.QUALITY_COLUMNS}}
    page = inspect_static_page(record, extra, old_example, audited, common, inspection,
                               resolver, parser_type, caps, output, save_source_copies)
    page["feature_values"] = example["feature_values"]
    page["quality"] = example["quality"]
    page.pop("baseline_feature_values")
    page.pop("baseline_quality_flags")
    page["current_error_at_0_50"] = example["new_model"]["error_at_0_50"]
    page["old_model_error_at_0_50"] = example["old_model"]["error_at_0_50"]
    page["old_model_phishing_score"] = example["old_model"]["phishing_score"]
    page["verification"].update(all_106_float32_features_reproduced=True, all_11_quality_values_reproduced=True)
    page["capture_http_status"] = None
    page["observed_final_redirect_url"] = None
    page["capture_completeness_verdict"] = "undetermined_from_static_archive"
    page["manual_review_status"] = "not_reviewed"
    f, q, notes = page["feature_values"], page["quality"], page["inspection"]
    lexical = []
    lexical_count = 0
    for match in LEXICAL_TITLE.finditer(prefix):
        lexical_count += 1
        if len(lexical) < 4:
            lexical.append({"character_offset": match.start(),
                "source_excerpt": inspection.brief(prefix[max(0, match.start()-80):match.end()+220], 500)})
    page["title_diagnostic"] = {
        "baseline_title_character_count": f["title_character_count"],
        "lexical_title_open_string_count_in_prefix": lexical_count,
        "lexical_examples": lexical, "lexical_examples_cut": lexical_count > len(lexical),
        "prefix_is_full_archived_html": not bool(q["html_analysis_truncated"]),
        "interpretation": "Lexical matches may be comments/script strings. An absent or empty parsed title alone proves neither corruption nor a parser bug."}
    hints = {h["hint"] for h in notes["text_hint_matches"]}
    structure = notes["capture_structure"]
    flags = {
        "zero_parsed_title_characters": f["title_character_count"] == 0,
        "lexical_title_with_zero_parsed_title": lexical_count > 0 and f["title_character_count"] == 0,
        "human_verification_text_hint": bool(q["context_human_verification_hint_present"] or "human_verification_phrase" in hints),
        "access_restriction_text_hint": "access_restriction_phrase" in hints,
        "unavailable_page_text_hint": bool(q["context_unavailable_page_hint_present"] or "unavailable_page_phrase" in hints),
        "javascript_required_text_hint": bool(structure["javascript_required_text_excerpts"]),
        "declared_meta_refresh": f["meta_refresh_count"] > 0,
        "javascript_navigation_lexical_hint": f["js_location_assignment_count"] + f["js_location_navigation_call_count"] > 0,
        "cross_site_canonical_reference": any(r["kind"] == "canonical" and r["cross_site"] for r in structure["metadata_references"]),
        "external_script_reference_present": f["script_external_count"] > 0,
        "low_parsed_text_with_scripts": f["text_character_count"] <= 200 and f["script_count"] > 0,
        "no_static_password_cue": f["password_input_count"] == 0 and f["context_passwordlike_control_count"] == 0,
        "html_prefix_limited": bool(q["html_analysis_truncated"]),
        "javascript_scan_limited": bool(q["js_analysis_truncated"]),
        "parser_or_association_limit": any(q[k] for k in ("html_parse_error", "unclosed_rawtext_element", "context_parse_error", "context_association_storage_cut")),
        "text_hint_scan_limited": bool(q["context_text_hint_scan_cut"] or notes["text_hint_scan_cut"]),
    }
    page["diagnostic_flags"] = flags
    return page


def summarize(pages, grouping):
    groups = defaultdict(list)
    for p in pages:
        key = p["metadata"]["site_group"] if grouping == "site_group" else p[grouping]
        groups[key].append(p)
    return {key: {"selected_pages": len(rows),
        "label_counts": dict(Counter(p["metadata"]["label"] for p in rows)),
        "current_error_counts": dict(Counter(p["current_error_at_0_50"] for p in rows)),
        "diagnostic_flag_page_counts": {flag: sum(p["diagnostic_flags"][flag] for p in rows) for flag in FLAGS}}
        for key, rows in sorted(groups.items())}


def write_index(path, pages):
    columns = ["record_id", "label", "site_group", "source_file", "review_category", "current_error_at_0_50",
        "phishing_score", "old_model_error_at_0_50", "full_html_characters", "title_text", "source_text_file", "source_copy_status", *FLAGS]
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for p in pages:
                row = {k: p[k] for k in columns if k in p}
                row.update({k: p["metadata"][k] for k in ("record_id", "label", "site_group", "source_file")})
                row["title_text"] = p["inspection"]["title_text"]
                row.update({k: int(v) for k, v in p["diagnostic_flags"].items()})
                # Preserve untrusted page titles as text when opening in Excel.
                writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r", "\n")) else v for k, v in row.items()})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(state_path, config, by_id, audited, output, save_source_copies):
    """Migrate only the known v1.0 code; keep every data/provenance check."""
    if not state_path.exists():
        if any(output.iterdir()):
            raise ValueError("Use an empty audit output folder or its matching checkpoint.")
        state = {"config": config, "pages": {}, "checkpoint_migrations": []}
        save_json(state_path, state)
        return state, []
    state = read_json(state_path)
    original = state["config"]
    migrate = original != config
    if migrate:
        version_keys = {"script_version", "audit_code_sha256", "source_copy_policy"}
        same_inputs = {k: v for k, v in original.items() if k not in version_keys} == {
            k: v for k, v in config.items() if k not in version_keys}
        known_previous = original.get("script_version") == "1.0" and original.get("audit_code_sha256") == PREVIOUS_VERSION_SHA
        same_current = (original.get("script_version") == VERSION
                        and original.get("audit_code_sha256") == config["audit_code_sha256"])
        if not same_inputs or not (known_previous or same_current):
            raise ValueError("Checkpoint inputs or code differ. Use a new --output-dir for a different run.")
    if not set(state["pages"]).issubset(by_id):
        raise ValueError("Unexpected page IDs in the checkpoint.")
    recheck = []
    normalized = {}
    for page_id, saved in state["pages"].items():
        p, e, m = saved["page"], by_id[page_id], audited[page_id]
        if (object_digest(p) != saved["page_digest"] or p["metadata"] != e["metadata"]
                or p["feature_values"] != e["feature_values"] or p["quality"] != e["quality"]
                or p["phishing_score"] != e["new_model"]["phishing_score"]
                or p["current_error_at_0_50"] != e["new_model"]["error_at_0_50"]
                or p["html_content_sha256"] != m["raw_html_sha256"]
                or p["full_html_characters"] != int(m["html_characters"])):
            raise ValueError("A checkpointed page or its audited provenance changed. Use a new --output-dir.")
        expected_length = min(p["full_html_characters"], config["parameters"]["max_html_characters"])
        source = output / "page_source_text" / f"{page_id}.html.txt"
        if p.get("source_text_file") is not None:
            if p["source_text_file"].replace("\\", "/") != f"page_source_text/{page_id}.html.txt":
                raise ValueError("Unexpected source path in a checkpointed page.")
            try:
                actual_hash = digest_file(source)
            except OSError as error:
                if not save_source_copies:
                    # Never silently trust a vanished/unreadable export. Re-read
                    # and verify this page from its original Parquet instead.
                    recheck.append({"record_id": page_id, "reason": "saved_source_copy_unavailable",
                                    "errno": error.errno, "winerror": getattr(error, "winerror", None)})
                    continue
                error.add_note("A saved raw-source copy is unavailable. Rerun with --no-source-copies to re-audit that page from the original dataset.")
                raise
            if actual_hash != p["source_text_file_sha256"] or p["stored_source_prefix_characters"] != expected_length:
                raise ValueError("A checkpointed page or source text changed. Use a new --output-dir.")
            if p.get("source_prefix_utf8_sha256", actual_hash) != actual_hash:
                raise ValueError("A checkpointed source prefix hash changed.")
            if expected_length == p["full_html_characters"] and actual_hash != p["html_content_sha256"]:
                raise ValueError("A full saved source copy differs from the audited HTML fingerprint.")
            if original.get("script_version") == "1.0":
                p = {**p, "analysed_source_prefix_characters": expected_length,
                     "source_prefix_utf8_sha256": actual_hash, "source_copy_status": "saved"}
        else:
            if (p.get("source_copy_status") != "not_requested" or p.get("source_text_file_sha256") is not None
                    or p.get("stored_source_prefix_characters") != 0
                    or p.get("analysed_source_prefix_characters") != expected_length
                    or not re.fullmatch(r"[0-9a-f]{64}", p.get("source_prefix_utf8_sha256", ""))):
                raise ValueError("Invalid checkpoint entry for an in-memory source inspection.")
            if expected_length == p["full_html_characters"] and p["source_prefix_utf8_sha256"] != m["raw_html_sha256"]:
                raise ValueError("Checkpointed in-memory HTML hash differs from its audited fingerprint.")
        normalized[page_id] = {"page": p, "page_digest": object_digest(p)}
    if migrate or recheck:
        backup = state_path.with_name(state_path.stem + ".before_resume_" + digest_file(state_path)[:12] + ".json")
        if backup.exists():
            if read_json(backup) != state:
                raise ValueError("Existing checkpoint backup has different content.")
        else:
            save_json(backup, state)
        state.setdefault("checkpoint_migrations", []).append({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "previous_script_version": original["script_version"], "previous_code_sha256": original["audit_code_sha256"],
            "current_script_version": VERSION, "current_code_sha256": config["audit_code_sha256"],
            "source_copy_policy": config["source_copy_policy"], "backup_file": backup.name,
            "reused_verified_pages": len(normalized), "pages_to_reaudit": recheck})
        state["config"] = config
        state["pages"] = normalized
        save_json(state_path, state)
        print(f"Checkpoint resumed safely: {len(normalized)} verified pages retained; {len(recheck)} source copies require re-audit.", flush=True)
    return state, recheck


def audit(args):
    paths, hashes, review, examples, audited, caps, stats, common, context, inspection = prepare(args)
    by_id = {e["metadata"]["record_id"]: e for e in examples}
    save_source_copies = not getattr(args, "no_source_copies", False)
    config = {"script_version": VERSION, "audit_code_sha256": digest_file(Path(__file__)),
        "input_hashes": hashes, "selected_source_files": stats, "parameters": caps,
        "data_dir": str(args.data_dir.resolve()), "selected_validation_ids": list(by_id),
        "python_version": platform.python_version(), "duckdb_version": common.duckdb.__version__,
        "tldextract_version": version("tldextract"), "model_name": MODEL, "threshold": .5,
        "source_copy_policy": "save_new_raw_source_copies" if save_source_copies else "analyse_without_new_raw_source_copies"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / CHECKPOINT_NAME
    state, recheck = load_checkpoint(state_path, config, by_id, audited, args.output_dir, save_source_copies)
    resumed = len(state["pages"])
    if save_source_copies:
        (args.output_dir / "page_source_text").mkdir(exist_ok=True)
    print(f"Capture-quality audit v{VERSION}: {len(examples)} selected validation pages; {resumed} checkpointed.", flush=True)
    print(f"DuckDB memory limit: {args.memory_limit}; threads: 1. Archived HTML is parsed as text.", flush=True)
    if not save_source_copies:
        print("New raw-source copies disabled. Every pending page is still read, fingerprint-checked and analysed.", flush=True)
    resolver = common.ReferenceResolver()
    parser_type = capture_parser_type(common, inspection)
    shards = defaultdict(list)
    for page_id, e in by_id.items():
        shards[e["metadata"]["source_file"]].append(page_id)
    for shard_index, (name, all_ids) in enumerate(sorted(shards.items()), 1):
        ids = [page_id for page_id in all_ids if page_id not in state["pages"]]
        if not ids:
            continue
        path = args.data_dir / name
        print(f"Archive {shard_index}/{len(shards)}: {name}; {len(ids)} pending pages.", flush=True)
        con = common.duckdb.connect()
        seen = set()
        try:
            con.execute("SET threads=1")
            con.execute("SET preserve_insertion_order=false")
            con.execute("SET memory_limit=?", [args.memory_limit])
            columns = {r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()}
            if not {"sha256", "label", "url", "html"}.issubset(columns):
                raise ValueError(f"Unexpected source columns: {name}.")
            slots = ",".join("?" for _ in all_ids)
            counts = dict(con.execute(f"SELECT sha256,count(*) FROM read_parquet(?) WHERE sha256 IN ({slots}) GROUP BY sha256", [str(path), *all_ids]).fetchall())
            if counts != {page_id: 1 for page_id in all_ids}:
                raise ValueError(f"Missing or duplicated selected records in {name}.")
            extra = [k for k in ("date", "target", "lang", "lang_score") if k in columns]
            slots = ",".join("?" for _ in ids)
            extra_sql = "".join(", " + k for k in extra)
            # Full hash is evaluated in SQL; Python receives a capped prefix.
            # Parquet page decoding still requires working RAM despite filtering.
            query = f"SELECT sha256,label,url,length(html),sha256(html),substr(html,1,{caps['max_html_characters']}){extra_sql} FROM read_parquet(?) WHERE sha256 IN ({slots})"
            cursor = con.sql(query, params=[str(path), *ids])
            while (record := cursor.fetchone()) is not None:
                page_id = record[0]
                if page_id not in ids or page_id in seen:
                    raise ValueError(f"Unexpected or repeated selected row: {name}.")
                seen.add(page_id)
                p = inspect_record(record, extra, by_id[page_id], audited[page_id], common, context,
                                   inspection, resolver, parser_type, caps, args.output_dir, save_source_copies)
                state["pages"][page_id] = {"page": p, "page_digest": object_digest(p)}
                save_json(state_path, state)
                print(f"  Verified and saved {len(state['pages'])}/{len(examples)} pages.", flush=True)
            if seen != set(ids):
                raise ValueError(f"Selected records were not all read from {name}.")
        except common.duckdb.OutOfMemoryException as error:
            raise MemoryError(f"DuckDB could not decode {name} within {args.memory_limit}.") from error
        except common.duckdb.Error as error:
            raise ValueError(f"Could not read {name}: {error}") from error
        finally:
            con.close()
    if {key: digest_file(path) for key, path in paths.items()} != hashes:
        raise ValueError("An input report, script, schema, prediction file or manifest changed during the audit.")
    for item in stats:
        stat = (args.data_dir / item["name"]).stat()
        if {"name": item["name"], "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != item:
            raise ValueError("Source file metadata changed during the audit.")
    if set(state["pages"]) != set(by_id):
        raise ValueError("The audit is incomplete.")
    pages = [state["pages"][page_id]["page"] for page_id in by_id]
    index = args.output_dir / "capture_quality_index.csv"
    write_index(index, pages)
    report = {"status": "capture_quality_audit_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "selected_validation_pages": len(pages), "pages_resumed_from_checkpoint": resumed,
        "checkpoint_migrations": state.get("checkpoint_migrations", []),
        "source_copy_availability": dict(Counter(p["source_copy_status"] for p in pages)),
        "duckdb_memory_limit": args.memory_limit, "duckdb_threads": 1,
        "sample_selection_policy": review["sample_selection_policy"], "diagnostic_flag_definitions": FLAGS,
        "summary_by_current_error": summarize(pages, "current_error_at_0_50"),
        "summary_by_site_group": summarize(pages, "site_group"),
        "summary_by_selected_category": summarize(pages, "review_category"),
        "selected_zero_title_record_ids": [p["metadata"]["record_id"] for p in pages if p["diagnostic_flags"]["zero_parsed_title_characters"]],
        "pages": pages, "index_csv_sha256": digest_file(index),
        "model_retrained": False, "labels_changed": False, "balanced_splits_changed": False,
        "threshold_changed": False, "official_test_evaluated": False, "websites_requested": False,
        "javascript_executed": False, "input_reports_scripts_manifest_and_predictions_unchanged": True,
        "limitations": [
            "These deliberately selected development errors/controls are not a representative quality survey or an independent accuracy evaluation.",
            "Flags overlap and are diagnostic hints, not capture-quality labels, deletion rules or reasons to relabel a page.",
            "No HTTP status, final redirect chain, live website or executed JavaScript was observed. Canonical URLs and refresh syntax are declarations only.",
            "A lack of static password cues does not establish a benign page; phishing includes non-credential scams and content absent from this capture.",
            "Parser text may include hidden content. Missing titles and small text counts can occur on legitimate pages and do not by themselves prove broken HTML.",
            "HTML/JS caps and bounded inspection lists apply. External JS bodies, CSS layout, browser DOM repair and dynamically inserted content are unobserved.",
            "The original dataset URL remains the reference base. Dataset annotations are inspection context only, never classifier inputs.",
            "Raw-source exports are optional. Pages without an exported copy are still analysed in memory, checked against the full audited HTML hash, and recorded with a prefix hash. No selected page is skipped because its copy was not requested.",
            "Full selected HTML hashes are verified against the audited manifest on first inspection. Resume checks checkpoint digests, audited provenance, available source copies and source metadata; complete Parquet shard bytes are not re-hashed. Unavailable source copies in --no-source-copies mode trigger re-audit from the original Parquet.",
            "The audit preserves all model inputs, class counts, validation membership and original labels. Human review and additional evidence are required to judge capture/label consistency."]}
    output = args.output_dir / REPORT_NAME
    save_json(output, report)
    print("Selected case counts:", json.dumps({k: v["selected_pages"] for k, v in report["summary_by_current_error"].items()}), flush=True)
    print(f"Saved report: {output.resolve()}", flush=True)
    print("Capture-quality audit complete. Upload capture_quality_report.json for review.", flush=True)
    return report


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    for option, default in (("review-report", "reports/context_error_review_v2/context_error_review_report.json"),
            ("features-dir", "reports/features_context_v2"), ("base-features-dir", "reports/features"),
            ("training-dir", "reports/form_context_training_v2"),
            ("manifest", "reports/balanced/balanced_training_validation_manifest.csv"),
            ("output-dir", "reports/capture_quality_v1")):
        parser.add_argument("--" + option, type=Path, default=root / default)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\new dataset\phreshphish_dataset\data"))
    parser.add_argument("--memory-limit", default="4GB", help="DuckDB budget; one scan thread, as in the completed extraction.")
    parser.add_argument("--no-source-copies", action="store_true",
                        help="Analyse all pages and save reports/checkpoints without exporting new .html.txt copies; safely resumes v1.0 checkpoints.")
    audit(parser.parse_args())


if __name__ == "__main__":
    try:
        main()
    except OSError as error:
        traceback.print_exception(error)
        raise SystemExit(f"Capture-quality audit stopped during a file operation (errno={error.errno}, winerror={getattr(error, 'winerror', None)}). Keep the existing checkpoint and paste this full error output. A report is complete only after the final completion message.")
    except (ValueError, KeyError, ImportError, TypeError, struct.error) as error:
        raise SystemExit(f"Capture-quality audit stopped: {error} A report is complete only after the final completion message.")
    except MemoryError:
        raise SystemExit("Not enough available RAM. Close memory-heavy apps and rerun the same command; completed pages are checkpointed.")
    except KeyboardInterrupt:
        raise SystemExit("Audit interrupted. Rerun the same command to continue from verified page checkpoints.")
