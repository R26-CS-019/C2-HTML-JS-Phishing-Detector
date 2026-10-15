"""Inspect the archived HTML behind script 13's selected validation errors.

Requirements: the existing duckdb and tldextract packages from script 09.
Run this beside scripts 09 and 13. It reads the original TRAIN-release
Parquet shards, selecting only the validation IDs chosen by script 13.
The audited HTML hashes, labels, site groups and all 86 baseline features
are checked before a page is accepted. No model is trained or changed.

Output: reports/html_error_review_v1/html_error_review_report.json, a
resumable checkpoint, and bounded archived source copies with .html.txt
extensions. HTML and JavaScript are parsed as text, never executed.
Titles, text and regex excerpts are inspection evidence, not new labels,
proof of malicious behavior, or independent evaluation results.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

VERSION = "1.0"
REPORT_NAME = "html_error_review_report.json"
CHECKPOINT_NAME = "html_inspection_checkpoint.json"
CATEGORIES = {
    "false_positive_largest_group", "false_positive_other_groups",
    "false_negative_largest_group", "false_negative_other_groups",
    "true_benign_controls", "true_phish_controls",
}
JS_CONTEXT_FEATURES = (
    "js_fetch_call_count", "js_xhr_constructor_count", "js_sendbeacon_call_count",
    "js_form_submit_call_count", "js_location_assignment_count",
    "js_location_navigation_call_count", "js_password_hint_count",
    "js_eval_call_count", "js_atob_call_count", "js_document_write_call_count",
)
TEXT_HINT_PATTERNS = {
    "human_verification_phrase": r"verify (?:that )?you are (?:a )?human|are you a robot|robot check|unusual traffic|checking your browser|complete the security check",
    "access_restriction_phrase": r"access denied|request (?:was )?blocked|temporarily blocked|enable javascript and cookies|attention required",
    "unavailable_page_phrase": r"page (?:was )?not found|this (?:site|page|website) is (?:no longer available|unavailable)|website has been removed|account (?:has been )?suspended",
    "credential_request_phrase": r"enter your (?:password|email|credentials)|confirm your (?:password|identity|account)|verify your account|sign in|log in|login",
}
TEXT_HINTS = {k: re.compile(v, re.I) for k, v in TEXT_HINT_PATTERNS.items()}


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def as_float32(value):
    # Script 10 trained on float32 feature matrices; compare that exact form.
    return struct.unpack("f", struct.pack("f", value))[0]


def brief(value, limit=1000):
    return str(value or "")[:limit]


def normalized_text(value):
    return " ".join(value.split())


def input_hash_named(hashes, name):
    matches = [v for k, v in hashes.items() if k.replace("\\", "/").rsplit("/", 1)[-1] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one recorded input hash for {name}.")
    return matches[0]


def checked_extractor(path, expected_hash):
    if digest_file(path) != expected_hash:
        raise ValueError("Keep the script 09 implementation recorded in your original extraction report.")
    spec = importlib.util.spec_from_file_location("archived_error_page_extractor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def read_selected_manifest(path, examples):
    selected = {e["metadata"]["record_id"]: e for e in examples}
    found = {}
    required = {"sha256", "label", "split", "site_group", "source_file", "html_characters",
                "raw_html_sha256", "sampled_similarity_candidate", "in_validation_sensitivity_subset"}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Use script 08's balanced primary manifest.")
        for row in reader:
            page_id = row["sha256"]
            if page_id not in selected:
                continue
            if page_id in found:
                raise ValueError("Repeated selected ID in the balanced manifest.")
            meta = selected[page_id]["metadata"]
            if row["split"] != "validation" or any(row[k] != meta[k] for k in
                    ("label", "site_group", "source_file", "sampled_similarity_candidate", "in_validation_sensitivity_subset")):
                raise ValueError("Selected sample differs from the original validation manifest.")
            if int(row["html_characters"]) <= 0 or not re.fullmatch(r"[0-9a-f]{64}", row["raw_html_sha256"]):
                raise ValueError("Missing audited HTML fingerprint or length.")
            found[page_id] = row
    if set(found) != set(selected):
        raise ValueError("A selected validation ID is missing from the balanced manifest.")
    return found


def inspection_parser_type(common):
    class InspectionParser(common.StaticParser):
        """Add bounded inspection notes while preserving the original scan order."""

        def __init__(self, url, resolver, max_js):
            super().__init__(url, resolver, max_js)
            self.title_parts = []
            self.text_parts = []
            self.text_kept = 0
            self.text_seen = 0
            self.form_notes = []
            self.field_notes = []
            self.resource_notes = []
            self.resource_seen = Counter()
            self.js_contexts = defaultdict(list)
            self.inline_script_notes = []
            self.current_script_note = None
            self.current_code_source = "inline_script"

        def handle_starttag(self, tag, attrs):
            a = {k: v or "" for k, v in attrs}
            if tag == "form" and len(self.form_notes) < 40:
                self.form_notes.append({"index": len(self.forms), "id": brief(a.get("id"), 200),
                                        "name": brief(a.get("name"), 200)})
            elif tag == "input" and len(self.field_notes) < 160:
                owner = ("id", a["form"]) if a.get("form") else (
                    "index", self.form_stack[-1] if self.form_stack else None)
                self.field_notes.append({"owner": owner, "type": brief(a.get("type", "text").strip().lower() or "text", 100),
                                         **{k: brief(a.get(k), 300) for k in
                                            ("name", "id", "placeholder", "autocomplete", "aria-label")},
                                         "required": "required" in a,
                                         "formaction": brief(a.get("formaction")),
                                         "has_formaction_override": "formaction" in a})
            if tag in ("script", "iframe") and a.get("src"):
                self.resource_seen[tag] += 1
                if len(self.resource_notes) < 40:
                    self.resource_notes.append({"kind": tag, "raw_reference": brief(a["src"]),
                                                "script_type": brief(a.get("type"), 100)})
            if tag == "script":
                self.current_script_note = {"index": int(self.c["script_count"]), "characters": 0,
                    "type": brief(a.get("type"), 100), "has_src": "src" in a,
                    "inline_body_executable_by_baseline": "src" not in a and
                        a.get("type", "").lower().split(";", 1)[0].strip() in common.EXECUTABLE_TYPES,
                    "js_budget_remaining_at_open": max(0, self.max_js - self.js_scanned)}
                self.inline_script_notes.append(self.current_script_note)
            self.current_code_source = "event_attribute_or_javascript_reference"
            super().handle_starttag(tag, attrs)
            self.current_code_source = "inline_script"

        def handle_endtag(self, tag):
            super().handle_endtag(tag)
            if tag == "script":
                self.current_script_note = None

        def handle_data(self, data):
            if self.script_active and self.current_script_note is not None:
                self.current_script_note["characters"] += len(data)
            if not self.script_active and not self.style_active:
                self.text_seen += len(data)
                if self.text_kept < 50_000:
                    fragment = data[:50_000 - self.text_kept]
                    self.text_parts.append(fragment)
                    self.text_kept += len(fragment)
                if self.title_active and sum(map(len, self.title_parts)) < 1000:
                    self.title_parts.append(data[:1000])
            super().handle_data(data)

        def add_js(self, code):
            scanned = code[:max(0, self.max_js - self.js_scanned)]
            for name in JS_CONTEXT_FEATURES:
                if len(self.js_contexts[name]) >= 3:
                    continue
                for match in common.COMPILED_JS[name].finditer(scanned):
                    self.js_contexts[name].append({"code_source": self.current_code_source,
                        "excerpt": brief(normalized_text(scanned[max(0, match.start()-85):match.end()+160]), 300)})
                    if len(self.js_contexts[name]) >= 3:
                        break
            super().add_js(code)

        def finish(self):
            if self.script_active and self.current_script_note is not None:
                self.current_script_note["unterminated_in_inspected_prefix"] = True
                if self.rawdata:
                    self.current_script_note["characters"] += len(self.rawdata)
            return super().finish()

        def notes(self, prefix_was_cut):
            # finish() has already resolved form ownership and recorded JS seen.
            base = self.page_url
            if self.first_base is not None:
                resolved, _, _, category = self.resolver.resolve(self.first_base, self.page_url)
                if category == "http":
                    base = resolved

            def reference_note(raw, empty_uses_document=False):
                actual = self.page_url if empty_uses_document and not raw.strip() else raw
                resolved, group, scheme, category = self.resolver.resolve(actual, base)
                return {"raw_reference": brief(raw), "resolved_reference": brief(resolved, 1500),
                        "reference_type": category, "scheme": scheme, "target_site_group": group,
                        "cross_site_by_baseline": category == "http" and bool(self.page_group) and group != self.page_group}

            fields_by_form = defaultdict(list)
            unowned = []
            for field in self.field_notes:
                field = dict(field)
                kind, owner = field.pop("owner")
                idx = self.form_ids.get(owner) if kind == "id" else owner
                field["resolved_form_index"] = idx
                field["credential_hint_by_text"] = bool(common.CREDENTIAL_HINT.search(
                    " ".join(field[k] for k in ("name", "id", "placeholder", "autocomplete", "aria-label"))))
                if field["has_formaction_override"]:
                    field["formaction_reference"] = reference_note(field["formaction"], True)
                (unowned if idx is None else fields_by_form[idx]).append(field)
            forms = []
            for note in self.form_notes:
                form = self.forms[note["index"]]
                forms.append({**note, "method": form["method"], "action": reference_note(form["action"], True),
                              "parsed_password_inputs": form["passwords"],
                              "inspected_input_fields": fields_by_form[note["index"]]})
            resources = [{**note, "reference": reference_note(note["raw_reference"])} for note in self.resource_notes]
            text = normalized_text(" ".join(self.text_parts))
            title = normalized_text(" ".join(self.title_parts))[:1000]
            hints = []
            for name, pattern in TEXT_HINTS.items():
                matches = []
                for match in pattern.finditer(text):
                    matches.append(text[max(0, match.start()-80):match.end()+120])
                    if len(matches) == 3:
                        break
                if matches:
                    hints.append({"hint": name, "text_excerpts": matches})
            scripts = sorted(self.inline_script_notes, key=lambda n: (-n["characters"], n["index"]))[:8]
            return {"title_text": title, "parsed_text_excerpt": text[:6000],
                "parsed_text_definition": "Text nodes outside script/style, including hidden content; not browser-visible text.",
                "text_hint_scan_characters": self.text_kept, "text_hint_scan_cut": self.text_seen > self.text_kept,
                "text_hint_matches": hints, "base_href": brief(self.first_base), "resolved_base_url": brief(base, 1500),
                "forms": forms, "form_notes_cut": len(self.forms) > len(self.form_notes),
                "input_fields_without_resolved_form": unowned,
                "input_notes_cut": int(self.c["input_count"]) > len(self.field_notes),
                "resource_references": resources,
                "resource_notes_cut": sum(self.resource_seen.values()) > len(self.resource_notes),
                "inline_event_js_characters_seen_in_html_prefix": self.js_seen,
                "inline_event_js_characters_scanned": self.js_scanned,
                "js_character_total_is_lower_bound": prefix_was_cut,
                "largest_script_bodies_in_html_prefix": scripts,
                "js_lexical_contexts": {name: notes for name, notes in self.js_contexts.items() if notes}}

    return InspectionParser


def check_examples(review, common):
    examples = review.get("review_examples", [])
    if not 1 <= len(examples) <= 300:
        raise ValueError("Expected 1 to 300 selected diagnostic examples from script 13.")
    seen = set()
    for e in examples:
        meta, score, category = e["metadata"], e["phishing_score"], e["review_category"]
        page_id = meta["record_id"]
        if not re.fullmatch(r"[0-9a-f]{64}", page_id) or page_id in seen:
            raise ValueError("Invalid or repeated selected record ID.")
        seen.add(page_id)
        if not re.fullmatch(r"train-\d+\.parquet", meta["source_file"]) or category not in CATEGORIES:
            raise ValueError("Only the original training-release shards and script 13 review categories are allowed.")
        if meta["label"] not in ("benign", "phish") or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid selected label or score.")
        expect_label = "benign" if category.startswith(("false_positive", "true_benign")) else "phish"
        expect_pred = category.startswith(("false_positive", "true_phish"))
        if meta["label"] != expect_label or (score >= review["threshold"]) != expect_pred:
            raise ValueError("Selected category differs from its saved label or prediction.")
        if set(e["feature_values"]) != set(common.FEATURE_NAMES) or set(e["quality"]) != set(common.QUALITY_COLUMNS):
            raise ValueError("Selected feature names or quality flags differ from script 09.")
        if any(not math.isfinite(v) or v < 0 for v in e["feature_values"].values()):
            raise ValueError("Invalid saved feature value.")
    return examples


def inspect_page(record, extra_names, example, audited, common, resolver, parser_type, caps, output_dir):
    page_id, label, url, full_length, raw_hash, prefix, *extra_values = record
    if page_id != audited["sha256"] or label != audited["label"] or full_length != int(audited["html_characters"]) or raw_hash != audited["raw_html_sha256"]:
        raise ValueError(f"Selected HTML, supplied ID or label differs from its audited content: {page_id}.")
    page_url = common.canonical_page_url(url)
    if resolver.resolve(page_url, page_url)[1] != audited["site_group"]:
        raise ValueError(f"Archived URL site group differs from the audited manifest: {page_id}.")
    features, quality = common.extract_static_features(prefix, url, resolver, **caps, full_html_characters=full_length)
    differences = [name for name in common.FEATURE_NAMES if as_float32(features[name]) != example["feature_values"][name]]
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
    source_path = output_dir / "page_source_text" / f"{page_id}.html.txt"
    # The extension makes these archived copies open as source text.
    source_path.write_text(prefix, encoding="utf-8", newline="")
    annotations = {key: (value.isoformat() if hasattr(value, "isoformat") else value)
                   for key, value in zip(extra_names, extra_values)}
    return {"review_category": example["review_category"], "metadata": example["metadata"],
        "phishing_score": example["phishing_score"], "archived_page_url": url,
        "dataset_annotations_for_inspection_only": annotations,
        "html_content_sha256": raw_hash, "full_html_characters": full_length,
        "stored_source_prefix_characters": len(prefix), "source_text_file": str(source_path.relative_to(output_dir)),
        "source_text_file_sha256": digest_file(source_path),
        "verification": {"raw_html_hash_matches_audited_manifest": True,
                         "all_baseline_float32_features_reproduced": True, "quality_flags_reproduced": True},
        "baseline_feature_values": example["feature_values"], "baseline_quality_flags": quality,
        "inspection": p.notes(bool(quality["html_analysis_truncated"]))}


def make_summary(pages):
    categories = {}
    for category in sorted(CATEGORIES):
        cohort = [p for p in pages if p["review_category"] == category]
        hints = Counter(h["hint"] for p in cohort for h in p["inspection"]["text_hint_matches"])
        categories[category] = {"selected_pages": len(cohort), "text_hint_page_counts": dict(hints),
            "pages_with_html_cap": sum(p["baseline_quality_flags"]["html_analysis_truncated"] for p in cohort),
            "pages_with_js_cap": sum(p["baseline_quality_flags"]["js_analysis_truncated"] for p in cohort),
            "pages_with_parsed_password_input": sum(p["baseline_feature_values"]["password_input_count"] > 0 for p in cohort)}
    return categories


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\new dataset\phreshphish_dataset\data"))
    parser.add_argument("--review-report", type=Path, default=root / "reports/error_review_v1/error_review_report.json")
    parser.add_argument("--features-dir", type=Path, default=root / "reports/features")
    parser.add_argument("--baseline-report", type=Path, default=root / "reports/baseline_training_v1_1/baseline_training_report.json")
    parser.add_argument("--manifest", type=Path, default=root / "reports/balanced/balanced_training_validation_manifest.csv")
    parser.add_argument("--extractor-script", type=Path, default=root / "09_extract_static_features.py")
    parser.add_argument("--review-script", type=Path, default=root / "13_review_static_errors.py")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/html_error_review_v1")
    parser.add_argument("--memory-limit", default="4GB", help="DuckDB budget, as in your completed extraction; one scan thread.")
    args = parser.parse_args()
    paths = {"review_report": args.review_report, "extraction_report": args.features_dir / "feature_extraction_report.json",
             "schema": args.features_dir / "static_feature_schema.json", "baseline_report": args.baseline_report,
             "manifest": args.manifest, "extractor_script": args.extractor_script, "review_script": args.review_script}
    for path in paths.values():
        if not path.is_file():
            parser.error(f"Missing input: {path}.")
    hashes = {name: digest_file(path) for name, path in paths.items()}
    review, extraction, schema = (load_json(paths[k]) for k in ("review_report", "extraction_report", "schema"))
    if review.get("status") != "static_error_review_complete" or review.get("model_name") != "lightgbm_all_features" or review.get("threshold") != 0.5:
        raise ValueError("Use the completed script 13 report for the original 86-feature model at threshold 0.50.")
    if extraction.get("status") != "static_feature_extraction_complete":
        raise ValueError("The original feature extraction must be complete.")
    for key in ("extraction_report", "schema", "baseline_report"):
        if hashes[key] != review["input_hashes"][key]:
            raise ValueError(f"Input differs from the completed error review: {paths[key].name}.")
    if hashes["review_script"] != review["review_code_sha256"] or hashes["extractor_script"] != extraction["inputs"]["code_sha256"]:
        raise ValueError("Keep the scripts 09 and 13 versions recorded in your completed reports.")
    if hashes["manifest"] != input_hash_named(extraction["inputs"]["inputs_sha256"], args.manifest.name):
        raise ValueError("Balanced manifest differs from the one used for feature extraction.")
    caps = extraction["inputs"]["parameters"]
    if schema["parameters"] != caps or schema["implementation_code_sha256"] != hashes["extractor_script"]:
        raise ValueError("Original extraction caps or implementation differ from the feature schema.")
    if set(caps) != {"max_html_characters", "max_js_characters"} or any(type(v) is not int or v < 1000 for v in caps.values()):
        raise ValueError("Unexpected original extraction caps.")
    common = checked_extractor(args.extractor_script, hashes["extractor_script"])
    if list(common.FEATURE_NAMES) != schema["feature_names"] or list(common.QUALITY_COLUMNS) != schema["quality_columns_not_classifier_inputs"]:
        raise ValueError("Original extractor and saved schema differ.")
    examples = check_examples(review, common)
    audited = read_selected_manifest(args.manifest, examples)
    by_id = {e["metadata"]["record_id"]: e for e in examples}
    expected_stats = {s["name"]: s for s in extraction["inputs"]["source_files"]}
    source_stats = []
    for name in sorted({e["metadata"]["source_file"] for e in examples}):
        path = args.data_dir / name
        if not path.is_file() or not path.resolve().is_relative_to(args.data_dir.resolve()):
            raise ValueError(f"Missing original source shard: {path}.")
        stat = path.stat()
        metadata = {"name": name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if metadata != expected_stats.get(name):
            raise ValueError(f"Source shard metadata changed since extraction: {name}. Use the original dataset files.")
        source_stats.append(metadata)
    if args.output_dir.resolve().is_relative_to(args.data_dir.resolve()) or args.output_dir.resolve() in (
            args.features_dir.resolve(), args.baseline_report.parent.resolve(), args.review_report.parent.resolve()):
        raise ValueError("Use a separate inspection output folder outside the raw dataset and prior result folders.")
    config = {"script_version": VERSION, "inspection_code_sha256": digest_file(Path(__file__)),
              "input_hashes": hashes, "selected_source_files": source_stats, "parameters": caps,
              "data_dir": str(args.data_dir.resolve()), "selected_validation_ids": list(by_id),
              "duckdb_version": common.duckdb.__version__, "tldextract_version": version("tldextract")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / CHECKPOINT_NAME
    if state_path.exists():
        state = load_json(state_path)
        if state["config"] != config:
            raise ValueError("Existing inspection checkpoint belongs to different inputs or code. Choose a new --output-dir.")
    else:
        if any(args.output_dir.iterdir()):
            raise ValueError("Use an empty inspection output folder or its matching checkpoint.")
        state = {"config": config, "pages": {}}
        save_json(state_path, state)
    if not set(state["pages"]).issubset(by_id):
        raise ValueError("Unexpected page IDs in the inspection checkpoint.")
    for page_id, page in state["pages"].items():
        source_path = args.output_dir / "page_source_text" / f"{page_id}.html.txt"
        if not source_path.is_file() or digest_file(source_path) != page["source_text_file_sha256"]:
            raise ValueError("A checkpointed source text file is missing or changed. Choose a new --output-dir.")
    (args.output_dir / "page_source_text").mkdir(exist_ok=True)
    resumed = len(state["pages"])
    print(f"Archived HTML inspection v{VERSION}: {len(examples)} selected validation pages; {resumed} checkpointed.", flush=True)
    print(f"DuckDB memory limit: {args.memory_limit}; threads: 1. Sources are parsed as text.", flush=True)
    resolver = common.ReferenceResolver()
    parser_type = inspection_parser_type(common)
    pending = defaultdict(list)
    for page_id, e in by_id.items():
        if page_id not in state["pages"]:
            pending[e["metadata"]["source_file"]].append(page_id)
    for index, (name, ids) in enumerate(sorted(pending.items()), 1):
        path = args.data_dir / name
        print(f"Source {index}/{len(pending)}: {name}; selected pages: {len(ids)}.", flush=True)
        con = common.duckdb.connect()
        seen = set()
        try:
            con.execute("SET threads=1")
            con.execute("SET preserve_insertion_order=false")
            con.execute("SET memory_limit=?", [args.memory_limit])
            columns = {row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchall()}
            if not {"sha256", "label", "url", "html"}.issubset(columns):
                raise ValueError(f"Unexpected source columns in {name}.")
            extra = [n for n in ("date", "target", "lang", "lang_score") if n in columns]
            extra_sql = "".join(", " + n for n in extra)
            placeholders = ",".join("?" for _ in ids)
            # SQL hashes the full archived content. Python receives a bounded
            # prefix one row at a time; Parquet decoding still needs working RAM.
            query = f"SELECT sha256,label,url,length(html),sha256(html),substr(html,1,{caps['max_html_characters']}){extra_sql} FROM read_parquet(?) WHERE sha256 IN ({placeholders})"
            cursor = con.sql(query, params=[str(path), *ids])
            while (record := cursor.fetchone()) is not None:
                page_id = record[0]
                if page_id not in ids or page_id in seen:
                    raise ValueError(f"Unexpected or repeated selected record in {name}.")
                seen.add(page_id)
                page = inspect_page(record, extra, by_id[page_id], audited[page_id], common, resolver, parser_type, caps, args.output_dir)
                state["pages"][page_id] = page
                save_json(state_path, state)
                print(f"  Verified and checkpointed {len(state['pages'])}/{len(examples)} pages.", flush=True)
            if seen != set(ids):
                raise ValueError(f"Missing selected validation record in {name}.")
        except common.duckdb.OutOfMemoryException as error:
            raise MemoryError(f"DuckDB could not decode {name} within {args.memory_limit}.") from error
        except common.duckdb.Error as error:
            raise ValueError(f"Could not read original source shard {name}: {error}") from error
        finally:
            con.close()
    if {key: digest_file(path) for key, path in paths.items()} != hashes:
        raise ValueError("An input script, report or manifest changed during inspection.")
    for metadata in source_stats:
        stat = (args.data_dir / metadata["name"]).stat()
        if {"name": metadata["name"], "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != metadata:
            raise ValueError("Source file metadata changed during inspection.")
    if set(state["pages"]) != set(by_id):
        raise ValueError("Inspection is incomplete.")
    pages = [state["pages"][page_id] for page_id in by_id]
    report = {"status": "archived_html_error_inspection_complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": VERSION, "python_version": platform.python_version(), "config": config,
        "duckdb_memory_limit": args.memory_limit, "duckdb_threads": 1,
        "selected_validation_pages": len(pages), "pages_resumed_from_checkpoint": resumed,
        "summary_by_selected_category": make_summary(pages), "pages": pages,
        "model_retrained": False, "labels_changed": False, "balanced_splits_changed": False,
        "threshold_changed": False, "official_test_evaluated": False, "websites_requested": False,
        "javascript_executed": False, "input_reports_scripts_and_manifest_unchanged": True,
        "limitations": [
            "These deliberately selected error/control cases are development diagnostics, not representative prevalence or independent accuracy estimates.",
            "Phrase hints, static form targets and regex excerpts do not establish malicious intent, executed network requests or label correctness.",
            "Parsed text includes hidden text and does not reflect browser rendering; stored HTML can contain challenge, error or unavailable-page captures.",
            "HTML prefixes and the original inline/event/JS-reference scan cap still apply. External script bodies are not available under this policy.",
            "Form ownership follows the original tolerant parser, not browser DOM repair. Lists and text excerpts have explicit storage limits.",
            "Dataset target/language/date annotations are inspection context only and must not become classifier inputs.",
            "Selected HTML hashes are checked against the audited manifest when first inspected. Resumed pages use checked source text and matching source-file metadata; entire Parquet shard bytes are not re-hashed.",
            "No current live website, final redirect destination or dynamic behavior was observed. Original site labels remain unchanged."]}
    output_path = args.output_dir / REPORT_NAME
    save_json(output_path, report)
    print(f"Saved report: {output_path.resolve()}", flush=True)
    print("Archived HTML inspection complete. Balanced data and saved models were not changed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ImportError, struct.error) as error:
        raise SystemExit(f"Archived HTML inspection stopped: {error}. A completed report is valid after the final completion message.")
    except MemoryError:
        raise SystemExit("Not enough available RAM. Close memory-heavy apps and rerun the same command; inspected pages were checkpointed.")
    except KeyboardInterrupt:
        raise SystemExit("Inspection interrupted. Rerun the same command to continue from verified page checkpoints.")
