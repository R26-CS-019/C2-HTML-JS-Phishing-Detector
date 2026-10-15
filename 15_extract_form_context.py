"""Add 20 form-context features to the existing balanced static baseline.

Uses script 09's original feature CSVs, audited manifest and read-only SQLite
checkpoint. Preserves every row, split, sensitivity flag and original feature.
Writes SEPARATE outputs in reports/features_context_v2, with checkpoints every
100 pages. Requirements: the existing duckdb and tldextract packages.

No browser, website requests, external JS downloads or script execution.
No model trained. These static cues do not prove credential harvesting.
Challenge/unavailable-page hints remain quality metadata, not model features.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import re
import shutil
import sqlite3
import struct
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

VERSION = "1.0"
CONTEXT_DEFINITIONS = {
    "context_user_entry_control_count": "Parsed input text/password/email/tel/url/search/number controls and textareas; hidden/button controls excluded.",
    "context_passwordlike_control_count": "User-entry controls with native type=password or password cues in associated labels/attributes.",
    "context_plaintext_passwordlike_control_count": "Password-cued controls whose declared type is not password; static declaration only.",
    "context_label_only_passwordlike_control_count": "Non-password-type controls with a password cue only in associated label/aria-labelledby text.",
    "context_normalized_password_cue_control_count": "Controls with a password cue recognized only after normalization/common substitutions; not proof of obfuscation.",
    "context_identitylike_control_count": "User-entry controls with type=email or identity/email/username cues in labels/attributes.",
    "context_passwordlike_textarea_count": "Password-cued textarea controls.",
    "context_passwordlike_unowned_control_count": "Password-cued controls without a resolved form in this tolerant parser.",
    "context_passwordlike_form_count": "Forms owning at least one password-cued control.",
    "context_identity_password_pair_form_count": "Forms owning password-cued and identity-cued controls; a combined label can cue both on one control.",
    "context_passwordlike_post_form_count": "Password-cued forms declaring method=post.",
    "context_passwordlike_get_form_count": "Password-cued forms with missing/empty method or method=get.",
    "context_plaintext_passwordlike_post_form_count": "POST forms with a password-cued control without native type=password.",
    "context_passwordlike_cross_site_form_count": "Password-cued forms with a resolved HTTP(S) action outside the page site group.",
    "context_passwordlike_https_to_http_form_count": "Password-cued forms on HTTPS pages whose action resolves to HTTP.",
    "context_passwordlike_handler_form_count": "Password-cued forms declaring onsubmit; presence is not observed execution.",
    "context_passwordlike_control_ratio": "Password-cued controls divided by max(1,user-entry controls).",
    "context_identitylike_control_ratio": "Identity-cued controls divided by max(1,user-entry controls).",
    "context_plaintext_passwordlike_fraction": "Non-password-type password-cued controls divided by max(1,password-cued controls).",
    "context_identity_password_pair_form_ratio": "Password/identity-pair forms divided by max(1,parsed forms).",
}
CONTEXT_NAMES = tuple(CONTEXT_DEFINITIONS)
CONTEXT_QUALITY = (
    "context_parse_error", "context_association_storage_cut", "context_duplicate_element_id_seen",
    "context_human_verification_hint_present", "context_unavailable_page_hint_present", "context_text_hint_scan_cut",
)
LIMITS = {"max_open_elements": 512, "max_association_items": 20000,
          "max_controls": 10000, "max_text_per_association": 1024, "max_text_hint_characters": 50000}
ENTRY_TYPES = {"text", "password", "email", "tel", "url", "search", "number"}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
PASSWORD_WORDS = re.compile(r"\b(?:password|passwd|pwd|passphrase|passwort|contrasena|senha|mot\s+de\s+passe)\b", re.I)
PASSWORD_VARIANT = re.compile(r"\bp[\W_]*a[\W_]*[s5][\W_]*[s5][\W_]*(?:w|v[\W_]*v)[\W_]*[o0][\W_]*r[\W_]*d\b", re.I)
IDENTITY_WORDS = re.compile(r"\b(?:e\s*mail|email|username|user\s*id|login\s*id|account\s*id|benutzername|utilisateur)\b", re.I)
HUMAN_PROMPT = re.compile(r"not a robot|are you (?:a )?(?:robot|human)|verify (?:that )?you are (?:a )?human|checking your browser|unusual traffic|complete the security check|enter the characters you see", re.I)
UNAVAILABLE_PROMPT = re.compile(r"page (?:was )?not found|this (?:page|site|website) is (?:no longer available|unavailable)|website has been removed|account (?:has been )?suspended", re.I)


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def words(value, normalize=False):
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    if normalize:
        value = "".join(c for c in unicodedata.normalize("NFKD", value)
                        if not unicodedata.combining(c) and unicodedata.category(c) != "Cf")
    return " ".join(re.sub(r"[^\w]+|_", " ", value.lower()).split())


def password_cue(value):
    """Return (recognized, normalization_needed); do not inspect field values."""
    raw = words(value)
    if PASSWORD_WORDS.search(raw) or raw in {"pass", "pass required"}:
        return True, False
    normalized = words(value, True)
    if PASSWORD_WORDS.search(normalized) or normalized in {"pass", "pass required"}:
        return True, True
    disguised = bool(PASSWORD_VARIANT.search(normalized))
    return disguised, disguised


def identity_cue(value):
    return bool(IDENTITY_WORDS.search(words(value, True)))


class AssociationLimit(Exception):
    pass


class TextBuffer:
    def __init__(self):
        self.parts, self.characters, self.cut = [], 0, False

    def add(self, text):
        room = max(0, LIMITS["max_text_per_association"] - self.characters)
        if len(text) > room:
            self.cut = True
        if room:
            self.parts.append(text[:room])
            self.characters += len(text[:room])

    def text(self):
        return " ".join(self.parts)


class FormContextParser(HTMLParser):
    def __init__(self, page_url, resolver, canonicalize):
        super().__init__(convert_charrefs=True)
        self.url = canonicalize(page_url)
        self.resolver = resolver
        self.page_group = resolver.resolve(self.url, self.url)[1]
        self.frames, self.fields, self.forms, self.labels = [], [], [], []
        self.element_ids, self.id_text, self.id_fields, self.id_forms = {}, {}, {}, {}
        self.first_base = None
        self.script_active = self.style_active = False
        self.storage_cut = self.duplicate_ids = False
        self.text_parts, self.text_kept, self.text_seen = [], 0, 0

    def handle_starttag(self, tag, attrs):
        a = {k: v or "" for k, v in attrs}
        if len(self.frames) >= LIMITS["max_open_elements"]:
            raise AssociationLimit()
        if tag == "script":
            self.script_active = True
        elif tag == "style":
            self.style_active = True
        elif tag == "base" and self.first_base is None and "href" in a:
            self.first_base = a["href"]
        form_idx = label_idx = field_idx = None
        if tag == "form":
            if len(self.forms) >= LIMITS["max_association_items"]:
                raise AssociationLimit()
            form_idx = len(self.forms)
            self.forms.append({"method": a.get("method", "get").strip().lower() or "get",
                               "action": a.get("action", ""), "handler": "onsubmit" in a})
        if tag == "label":
            if len(self.labels) >= LIMITS["max_association_items"]:
                raise AssociationLimit()
            label_idx = len(self.labels)
            self.labels.append({"explicit": "for" in a, "target": a.get("for", ""), "first_field": None, "text": TextBuffer()})
        if tag in {"input", "textarea", "select", "button"}:
            if len(self.fields) >= LIMITS["max_controls"]:
                raise AssociationLimit()
            kind = "textarea" if tag == "textarea" else a.get("type", "text").strip().lower() or "text"
            eligible = tag == "textarea" or tag == "input" and kind in ENTRY_TYPES
            current_form = next((f[3] for f in reversed(self.frames) if f[3] is not None), None)
            owner = ("id", a["form"]) if a.get("form") else ("index", current_form)
            field_idx = len(self.fields)
            self.fields.append({"type": kind, "eligible": eligible, "owner": owner,
                "attribute_text": " ".join(a.get(k, "") for k in ("id", "name", "placeholder", "autocomplete", "aria-label", "title")),
                "attribute_values": [a.get(k, "") for k in ("id", "name", "placeholder", "autocomplete", "aria-label", "title")],
                "aria_ids": a.get("aria-labelledby", "").split()[:16], "label_text": []})
            if kind != "hidden":
                for frame in self.frames:
                    if frame[2] is not None:
                        label = self.labels[frame[2]]
                        if label["first_field"] is None:
                            label["first_field"] = field_idx
        element_id = a.get("id", "")
        text_id = None
        if element_id:
            if element_id in self.element_ids:
                self.duplicate_ids = True
            elif len(self.element_ids) >= LIMITS["max_association_items"]:
                self.storage_cut = True
            else:
                self.element_ids[element_id] = tag
                self.id_text[element_id] = TextBuffer()
                text_id = element_id
                if field_idx is not None and self.fields[field_idx]["type"] != "hidden":
                    self.id_fields[element_id] = field_idx
                if form_idx is not None:
                    self.id_forms[element_id] = form_idx
        if tag not in VOID:
            self.frames.append((tag, text_id, label_idx, form_idx))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "script":
            self.script_active = False
        elif tag == "style":
            self.style_active = False
        for idx in range(len(self.frames)-1, -1, -1):
            if self.frames[idx][0] == tag:
                del self.frames[idx:]
                break

    def handle_data(self, data):
        if self.script_active or self.style_active:
            return
        self.text_seen += len(data)
        room = max(0, LIMITS["max_text_hint_characters"] - self.text_kept)
        if room:
            self.text_parts.append(data[:room])
            self.text_kept += len(data[:room])
        for _, text_id, label_idx, _ in self.frames:
            if text_id is not None:
                self.id_text[text_id].add(data)
            if label_idx is not None:
                self.labels[label_idx]["text"].add(data)

    def finish(self, parse_error):
        for label in self.labels:
            idx = self.id_fields.get(label["target"]) if label["explicit"] else label["first_field"]
            if idx is not None:
                self.fields[idx]["label_text"].append(label["text"].text())
                self.storage_cut |= label["text"].cut
        base = self.url
        if self.first_base is not None:
            candidate, _, _, category = self.resolver.resolve(self.first_base, self.url)
            if category == "http":
                base = candidate
        per_form = [Counter() for _ in self.forms]
        c = Counter()
        for field in self.fields:
            if not field["eligible"]:
                continue
            c["context_user_entry_control_count"] += 1
            associated = list(field["label_text"])
            for element_id in field["aria_ids"]:
                if element_id in self.id_text:
                    associated.append(self.id_text[element_id].text())
                    self.storage_cut |= self.id_text[element_id].cut
            label_text = " ".join(associated)
            attr_signals = [password_cue(v) for v in field["attribute_values"]]
            label_signals = [password_cue(v) for v in associated]
            attr_pw = any(s[0] for s in attr_signals)
            label_pw = any(s[0] for s in label_signals)
            attr_normalized = attr_pw and not any(s[0] and not s[1] for s in attr_signals)
            label_normalized = label_pw and not any(s[0] and not s[1] for s in label_signals)
            native = field["type"] == "password"
            password = native or attr_pw or label_pw
            identity = field["type"] == "email" or identity_cue(field["attribute_text"] + " " + label_text)
            plain = password and not native
            c["context_passwordlike_control_count"] += int(password)
            c["context_plaintext_passwordlike_control_count"] += int(plain)
            c["context_label_only_passwordlike_control_count"] += int(not native and label_pw and not attr_pw)
            c["context_normalized_password_cue_control_count"] += int(password and not native and
                (attr_normalized or label_normalized) and not ((attr_pw and not attr_normalized) or (label_pw and not label_normalized)))
            c["context_identitylike_control_count"] += int(identity)
            c["context_passwordlike_textarea_count"] += int(password and field["type"] == "textarea")
            owner_type, owner = field["owner"]
            idx = self.id_forms.get(owner) if owner_type == "id" else owner
            if idx is None:
                c["context_passwordlike_unowned_control_count"] += int(password)
            else:
                per_form[idx]["password"] += int(password)
                per_form[idx]["identity"] += int(identity)
                per_form[idx]["plain"] += int(plain)
        for form, signals in zip(self.forms, per_form):
            if not signals["password"]:
                continue
            ref = form["action"] if form["action"].strip() else self.url
            _, group, scheme, category = self.resolver.resolve(ref, base)
            c["context_passwordlike_form_count"] += 1
            c["context_identity_password_pair_form_count"] += int(signals["identity"] > 0)
            c["context_passwordlike_post_form_count"] += int(form["method"] == "post")
            c["context_passwordlike_get_form_count"] += int(form["method"] == "get")
            c["context_plaintext_passwordlike_post_form_count"] += int(form["method"] == "post" and signals["plain"] > 0)
            c["context_passwordlike_cross_site_form_count"] += int(category == "http" and bool(self.page_group) and group != self.page_group)
            c["context_passwordlike_https_to_http_form_count"] += int(category == "http" and urlsplit(self.url).scheme == "https" and scheme == "http")
            c["context_passwordlike_handler_form_count"] += int(form["handler"])
        c["context_passwordlike_control_ratio"] = c["context_passwordlike_control_count"] / max(1, c["context_user_entry_control_count"])
        c["context_identitylike_control_ratio"] = c["context_identitylike_control_count"] / max(1, c["context_user_entry_control_count"])
        c["context_plaintext_passwordlike_fraction"] = c["context_plaintext_passwordlike_control_count"] / max(1, c["context_passwordlike_control_count"])
        c["context_identity_password_pair_form_ratio"] = c["context_identity_password_pair_form_count"] / max(1, len(self.forms))
        features = {name: c[name] for name in CONTEXT_NAMES}
        if any(not math.isfinite(v) or v < 0 for v in features.values()):
            raise ValueError("Invalid form-context feature value.")
        text = " ".join(self.text_parts)
        quality = {"context_parse_error": int(parse_error), "context_association_storage_cut": int(self.storage_cut),
            "context_duplicate_element_id_seen": int(self.duplicate_ids),
            "context_human_verification_hint_present": int(bool(HUMAN_PROMPT.search(text))),
            "context_unavailable_page_hint_present": int(bool(UNAVAILABLE_PROMPT.search(text))),
            "context_text_hint_scan_cut": int(self.text_seen > self.text_kept)}
        return features, quality


def extract_form_context(html, page_url, common, resolver=None, max_html_characters=4000000):
    """No labels, scores, dataset annotations or field values enter the cues."""
    if not isinstance(html, str) or max_html_characters <= 0:
        raise ValueError("HTML must be text and the HTML cap positive.")
    p = FormContextParser(page_url, resolver or common.ReferenceResolver(), common.canonical_page_url)
    parse_error = 0
    try:
        p.feed(html[:max_html_characters])
        p.close()
    except AssociationLimit:
        p.storage_cut = True
    except (AssertionError, ValueError):
        parse_error = 1
    return p.finish(parse_error)


def extract_enhanced_static_features(html, page_url, common, resolver=None,
                                     max_html_characters=4000000, max_js_characters=500000,
                                     full_html_characters=None):
    """Inference adapter: original 86 features plus these exact 20 context features."""
    resolver = resolver or common.ReferenceResolver()
    features, quality = common.extract_static_features(html, page_url, resolver, max_html_characters,
                                                       max_js_characters, full_html_characters)
    context, context_quality = extract_form_context(html, page_url, common, resolver, max_html_characters)
    return {**features, **context}, {**quality, **context_quality}


def checked_extractor(path, expected_sha):
    if digest_file(path) != expected_sha:
        raise ValueError("Keep the exact script 09 implementation used for your completed original features.")
    spec = importlib.util.spec_from_file_location("original_context_feature_extractor", path)
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(common)
    finally:
        sys.dont_write_bytecode = previous
    return common


def label_counts(db, condition="1"):
    counts = {s: {label: 0 for label in ("benign", "phish")} for s in ("train", "validation")}
    for split, label, n in db.execute(f"SELECT split,label,count(*) FROM pages WHERE {condition} GROUP BY split,label"):
        if split not in counts or label not in counts[split]:
            raise ValueError("Unexpected original split or label.")
        counts[split][label] = n
    return counts


def verify_original_checkpoint(db, extraction, common, paths):
    saved = db.execute("SELECT value FROM state WHERE key='config'").fetchone()
    if not saved or json.loads(saved[0]) != extraction["inputs"]:
        raise ValueError("Original SQLite checkpoint differs from the completed extraction report.")
    if db.execute("SELECT 1 FROM pages WHERE processed!=1 OR raw_sha IS NULL OR html_characters<=0 LIMIT 1").fetchone():
        raise ValueError("Original feature checkpoint is incomplete.")
    if label_counts(db) != extraction["extracted_label_counts"] or label_counts(db, "sensitivity=1")["validation"] != extraction["validation_sensitivity_counts"]:
        raise ValueError("Original checkpoint class counts or sensitivity membership differ.")
    if db.execute("SELECT 1 FROM pages GROUP BY site_group HAVING count(DISTINCT split)>1 LIMIT 1").fetchone():
        raise ValueError("Original train/validation site groups overlap.")
    header = [*common.METADATA_COLUMNS, *common.QUALITY_COLUMNS, *common.FEATURE_NAMES]
    for key, split, condition in (("train", "train", "split='train'"),
                                  ("validation", "validation", "split='validation'"),
                                  ("sensitivity", "validation", "sensitivity=1")):
        expected_n = db.execute(f"SELECT count(*) FROM pages WHERE {condition}").fetchone()[0]
        previous_id, seen = "", 0
        with paths[key].open(newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            if next(reader, None) != header:
                raise ValueError(f"Unexpected original feature columns: {paths[key].name}.")
            for row in reader:
                if len(row) != len(header) or row[0] <= previous_id:
                    raise ValueError("Original feature IDs must be unique and sorted as exported by script 09.")
                previous_id = row[0]
                saved_row = db.execute("SELECT id,label,site_group,source_file,similarity_candidate,sensitivity,quality_values,feature_values,split FROM pages WHERE id=?", (row[0],)).fetchone()
                if not saved_row or [str(v) for v in saved_row[:6]] != row[:6] or saved_row[8] != split:
                    raise ValueError("Original feature metadata differs from its checkpoint.")
                if key == "sensitivity" and saved_row[5] != 1:
                    raise ValueError("Unexpected sensitivity row.")
                if json.loads(saved_row[6]) != [int(v) for v in row[6:6+len(common.QUALITY_COLUMNS)]] or json.loads(saved_row[7]) != [float(v) for v in row[6+len(common.QUALITY_COLUMNS):]]:
                    raise ValueError("Original feature/quality values differ from the byte-hashed CSV.")
                seen += 1
        if seen != expected_n:
            raise ValueError("Original feature CSV membership differs from its checkpoint.")
        print(f"  Original {key}: verified {seen:,} rows.", flush=True)
    audited_n = 0
    with paths["manifest"].open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"sha256", "label", "split", "site_group", "source_file", "html_characters",
                    "raw_html_sha256", "sampled_similarity_candidate", "in_validation_sensitivity_subset"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Use script 08's balanced primary manifest.")
        seen_ids = set()
        for row in reader:
            page_id = row["sha256"]
            if page_id in seen_ids:
                raise ValueError("Repeated ID in the original balanced manifest.")
            seen_ids.add(page_id)
            expected = (row["label"], row["split"], row["site_group"], row["source_file"],
                        int(row["html_characters"]), row["raw_html_sha256"],
                        int(row["sampled_similarity_candidate"]), int(row["in_validation_sensitivity_subset"]))
            saved_row = db.execute("SELECT label,split,site_group,source_file,html_characters,raw_sha,similarity_candidate,sensitivity FROM pages WHERE id=?", (page_id,)).fetchone()
            if saved_row != expected:
                raise ValueError("Original checkpoint differs from the audited balanced manifest.")
            audited_n += 1
    if audited_n != db.execute("SELECT count(*) FROM pages").fetchone()[0]:
        raise ValueError("Original checkpoint membership differs from the balanced manifest.")


def checkpoint_init(db, config):
    db.executescript("""CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS contexts(id TEXT PRIMARY KEY,source_file TEXT NOT NULL,features TEXT NOT NULL,quality TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS contexts_source ON contexts(source_file);""")
    saved = db.execute("SELECT value FROM state WHERE key='config'").fetchone()
    if saved:
        if json.loads(saved[0]) != config:
            raise ValueError("Existing context checkpoint belongs to different inputs/code. Choose a new --output-dir.")
    elif db.execute("SELECT count(*) FROM contexts").fetchone()[0]:
        raise ValueError("Context checkpoint has rows without its configuration.")
    else:
        db.execute("INSERT INTO state VALUES('config',?)", (json.dumps(config, sort_keys=True),))
        db.commit()


def scan_source(original, output, path, common, args, resolver):
    completed = {r[0] for r in output.execute("SELECT id FROM contexts WHERE source_file=?", (path.name,))}
    rows = original.execute("SELECT id,label,site_group,html_characters,raw_sha FROM pages WHERE source_file=?", (path.name,)).fetchall()
    if not completed.issubset({r[0] for r in rows}):
        raise ValueError("Unexpected checkpoint ID for a source shard.")
    expected = {r[0]: r[1:] for r in rows if r[0] not in completed}
    if not expected:
        print("  Already checkpointed, or no balanced records in this shard.", flush=True)
        return
    print(f"  Pending balanced pages: {len(expected):,}.", flush=True)
    con = common.duckdb.connect()
    seen = set()
    caps = args.original_caps
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET memory_limit=?", [args.memory_limit])
        placeholders = ",".join("?" for _ in expected)
        query = f"SELECT sha256,label,url,length(html),sha256(html),substr(html,1,{caps['max_html_characters']}) FROM read_parquet(?) WHERE sha256 IN ({placeholders})"
        cursor = con.sql(query, params=[str(path), *expected])
        output.execute("BEGIN")
        while (row := cursor.fetchone()) is not None:
            page_id, label, url, length, raw_sha, prefix = row
            if page_id not in expected or page_id in seen:
                raise ValueError("Unexpected or repeated selected source ID.")
            actual_group = resolver.resolve(common.canonical_page_url(url), common.canonical_page_url(url))[1]
            if (label, actual_group, length, raw_sha) != tuple(expected[page_id]):
                raise ValueError(f"HTML, label or URL site group differs from the original audit: {page_id}.")
            features, quality = extract_form_context(prefix, url, common, resolver, caps["max_html_characters"])
            output.execute("INSERT INTO contexts VALUES(?,?,?,?)", (page_id, path.name,
                json.dumps([features[n] for n in CONTEXT_NAMES], separators=(",", ":")),
                json.dumps([quality[n] for n in CONTEXT_QUALITY], separators=(",", ":"))))
            seen.add(page_id)
            if len(seen) % 100 == 0:
                output.commit()
                output.execute("BEGIN")
            if len(seen) % 500 == 0:
                print(f"  Extracted {len(seen):,}/{len(expected):,}; checkpoint every 100 pages.", flush=True)
        if seen != set(expected):
            raise ValueError(f"Missing balanced source records in {path.name}.")
        output.commit()
        print(f"  Saved context for {len(seen):,} pages.", flush=True)
    except common.duckdb.OutOfMemoryException as error:
        output.rollback()
        raise MemoryError(f"DuckDB could not decode {path.name} within {args.memory_limit}.") from error
    except BaseException:
        output.rollback()
        raise
    finally:
        con.close()


def export_features(db, common, output_dir):
    names = [*common.FEATURE_NAMES, *CONTEXT_NAMES]
    quality_names = [*common.QUALITY_COLUMNS, *CONTEXT_QUALITY]
    summaries = {s: {label: Counter() for label in ("benign", "phish")} for s in ("train", "validation")}
    outputs = {}
    tracked = {common.FEATURE_NAMES.index("password_input_count"): "password_input_present_rows",
               common.FEATURE_NAMES.index("script_inline_executable_count"): "inline_executable_script_present_rows"}
    tracked.update({len(common.FEATURE_NAMES)+i: n+"_present_rows" for i, n in enumerate(CONTEXT_NAMES)})
    for name, condition in (("train_static_features.csv", "p.split='train'"),
                            ("validation_static_features.csv", "p.split='validation'"),
                            ("validation_sensitivity_static_features.csv", "p.sensitivity=1")):
        path = output_dir / name
        tmp = path.with_suffix(".csv.tmp")
        written = 0
        try:
            with tmp.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([*common.METADATA_COLUMNS, *quality_names, *names])
                query = f"SELECT p.id,p.label,p.site_group,p.source_file,p.similarity_candidate,p.sensitivity,p.quality_values,p.feature_values,c.quality,c.features,p.split FROM base.pages p JOIN contexts c ON p.id=c.id WHERE {condition} ORDER BY p.id"
                for row in db.execute(query):
                    quality = json.loads(row[6]) + json.loads(row[8])
                    values = json.loads(row[7]) + json.loads(row[9])
                    if len(quality) != len(quality_names) or len(values) != len(names) or any(not math.isfinite(v) or v < 0 for v in values) or any(type(v) is not int or v < 0 for v in quality):
                        raise ValueError("Invalid merged checkpoint values.")
                    writer.writerow([*row[:6], *quality, *values])
                    written += 1
                    if name != "validation_sensitivity_static_features.csv":
                        summary = summaries[row[10]][row[1]]
                        summary["rows"] += 1
                        for flag, value in zip(quality_names, quality):
                            summary[flag + "_rows"] += int(value > 0)
                        for idx, key in tracked.items():
                            summary[key] += int(values[idx] > 0)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        outputs[name] = {"rows": written, "sha256": digest_file(path)}
    return outputs, summaries


def selected_review_summary(db, review, common):
    result = []
    for page in review["pages"]:
        meta = page["metadata"]
        row = db.execute("SELECT p.label,p.split,p.site_group,p.source_file,p.feature_values,c.features,c.quality FROM base.pages p JOIN contexts c ON p.id=c.id WHERE p.id=?", (meta["record_id"],)).fetchone()
        if not row or row[:4] != (meta["label"], "validation", meta["site_group"], meta["source_file"]):
            raise ValueError("HTML review page differs from original validation membership.")
        base = {n: struct.unpack("f", struct.pack("f", v))[0] for n, v in zip(common.FEATURE_NAMES, json.loads(row[4]))}
        if base != page["baseline_feature_values"]:
            raise ValueError("Original baseline features differ from the HTML review.")
        result.append({"metadata": meta, "review_category": page["review_category"],
            "original_baseline_phishing_score": page["phishing_score"],
            "original_password_input_count": base["password_input_count"],
            "context_features": dict(zip(CONTEXT_NAMES, json.loads(row[5]))),
            "context_quality": dict(zip(CONTEXT_QUALITY, json.loads(row[6])))})
    return result


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\new dataset\phreshphish_dataset\data"))
    parser.add_argument("--features-dir", type=Path, default=root / "reports/features")
    parser.add_argument("--manifest", type=Path, default=root / "reports/balanced/balanced_training_validation_manifest.csv")
    parser.add_argument("--extractor-script", type=Path, default=root / "09_extract_static_features.py")
    parser.add_argument("--html-review-report", type=Path, default=root / "reports/html_error_review_v1/html_error_review_report.json")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/features_context_v2")
    parser.add_argument("--memory-limit", default="4GB")
    args = parser.parse_args()
    paths = {"extraction_report": args.features_dir / "feature_extraction_report.json",
        "schema": args.features_dir / "static_feature_schema.json", "original_checkpoint": args.features_dir / "static_features_checkpoint.sqlite",
        "train": args.features_dir / "train_static_features.csv", "validation": args.features_dir / "validation_static_features.csv",
        "sensitivity": args.features_dir / "validation_sensitivity_static_features.csv", "manifest": args.manifest,
        "extractor_script": args.extractor_script, "html_review_report": args.html_review_report}
    for path in paths.values():
        if not path.is_file():
            parser.error(f"Missing original input: {path}.")
    hashes = {name: digest_file(path) for name, path in paths.items()}
    extraction, schema, review = (load_json(paths[n]) for n in ("extraction_report", "schema", "html_review_report"))
    if extraction.get("status") != "static_feature_extraction_complete" or review.get("status") != "archived_html_error_inspection_complete":
        raise ValueError("Complete the original feature extraction and script 14 inspection first.")
    for key in ("schema", "train", "validation", "sensitivity"):
        if hashes[key] != extraction["outputs"][paths[key].name]["sha256"]:
            raise ValueError(f"Original feature output changed: {paths[key].name}.")
    for key in ("schema", "extraction_report", "extractor_script"):
        if hashes[key] != review["config"]["input_hashes"][key]:
            raise ValueError("HTML inspection and original feature provenance differ.")
    recorded_manifest = [v for k, v in extraction["inputs"]["inputs_sha256"].items()
                         if k.replace("\\", "/").rsplit("/", 1)[-1] == args.manifest.name]
    if recorded_manifest != [hashes["manifest"]]:
        raise ValueError("Balanced manifest differs from the one used for original features.")
    if args.output_dir.resolve().is_relative_to(args.data_dir.resolve()) or args.output_dir.resolve() in (args.features_dir.resolve(), args.html_review_report.parent.resolve()):
        raise ValueError("Use a separate output folder outside the dataset and original reports.")
    common = checked_extractor(args.extractor_script, extraction["inputs"]["code_sha256"])
    if list(common.FEATURE_NAMES) != schema["feature_names"] or len(common.FEATURE_NAMES) != 86:
        raise ValueError("This extension requires the original 86-feature schema.")
    args.original_caps = extraction["inputs"]["parameters"]
    if schema["parameters"] != args.original_caps or schema["implementation_code_sha256"] != hashes["extractor_script"]:
        raise ValueError("Original schema and extraction implementation/caps differ.")
    source_paths = sorted(args.data_dir.glob("train-*.parquet"))
    source_stats = [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in source_paths]
    if source_stats != extraction["inputs"]["source_files"]:
        raise ValueError("Original dataset shard metadata changed. Use the original downloaded dataset.")
    config = {"script_version": VERSION, "code_sha256": digest_file(Path(__file__)), "input_hashes": hashes,
        "base_implementation_code_sha256": hashes["extractor_script"], "source_files": source_stats,
        "data_dir": str(args.data_dir.resolve()), "parameters": args.original_caps, "association_limits": LIMITS,
        "context_feature_names": list(CONTEXT_NAMES), "external_js_policy": "references_only_no_download",
        "duckdb_version": common.duckdb.__version__, "tldextract_version": version("tldextract"),
        "python_version": platform.python_version()}
    original_uri = paths["original_checkpoint"].resolve().as_uri() + "?mode=ro"
    original = sqlite3.connect(original_uri, uri=True)
    output = None
    try:
        print("Checking original balanced features and read-only checkpoint...", flush=True)
        verify_original_checkpoint(original, extraction, common, paths)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = args.output_dir / "form_context_checkpoint.sqlite"
        if not checkpoint.exists() and any(args.output_dir.iterdir()):
            raise ValueError("Use an empty context output folder or its matching checkpoint.")
        if not checkpoint.exists() and shutil.disk_usage(args.output_dir).free < 2 * 1024 ** 3:
            raise ValueError("Keep at least 2 GiB free for the new checkpoint and feature CSVs.")
        output = sqlite3.connect(checkpoint.resolve().as_uri(), uri=True)
        checkpoint_init(output, config)
        output.execute("ATTACH DATABASE ? AS base", (original_uri,))
        print(f"Features: 86 original + {len(CONTEXT_NAMES)} context; memory limit {args.memory_limit}; scan threads: 1.", flush=True)
        print("Original rows, labels and splits are preserved; no model is trained.", flush=True)
        resolver = common.ReferenceResolver()
        for i, path in enumerate(source_paths, 1):
            print(f"Form context {i}/{len(source_paths)}: {path.name}", flush=True)
            scan_source(original, output, path, common, args, resolver)
        if output.execute("SELECT count(*) FROM contexts").fetchone()[0] != original.execute("SELECT count(*) FROM pages").fetchone()[0]:
            raise ValueError("New feature extraction is incomplete.")
        if output.execute("SELECT 1 FROM contexts c LEFT JOIN base.pages p ON c.id=p.id WHERE p.id IS NULL OR c.source_file!=p.source_file LIMIT 1").fetchone():
            raise ValueError("Context checkpoint membership differs from the original balanced rows.")
        for key, path in paths.items():
            if digest_file(path) != hashes[key]:
                raise ValueError(f"Original input changed during extraction: {path.name}.")
        if [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in source_paths] != source_stats:
            raise ValueError("Source shard metadata changed during extraction.")
        outputs, summaries = export_features(output, common, args.output_dir)
        selected = selected_review_summary(output, review, common)
        names = [*common.FEATURE_NAMES, *CONTEXT_NAMES]
        quality_names = [*common.QUALITY_COLUMNS, *CONTEXT_QUALITY]
        new_schema = {"schema_version": "2.0", "feature_count": len(names), "feature_names": names,
            "feature_definitions": {**common.FEATURE_DEFINITIONS, **CONTEXT_DEFINITIONS},
            "metadata_columns": list(common.METADATA_COLUMNS), "quality_columns_not_classifier_inputs": quality_names,
            "parameters": args.original_caps, "association_limits": LIMITS, "implementation_code_sha256": config["code_sha256"],
            "base_implementation_code_sha256": hashes["extractor_script"], "external_js_policy": "references_only_no_download",
            "inference_contract": "Use extract_enhanced_static_features() from this exact script 15 with original verified script 09 and these exact HTML/JS caps. Labels, site names and quality flags remain outside X.",
            "method_references": ["https://html.spec.whatwg.org/multipage/forms.html#the-label-element", "https://docs.python.org/3.12/library/unicodedata.html"]}
        schema_path = args.output_dir / "static_feature_schema.json"
        write_json(schema_path, new_schema)
        outputs[schema_path.name] = {"sha256": digest_file(schema_path)}
        report = {"status": "form_context_feature_extraction_complete", "script_version": VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(), "inputs": config, "outputs": outputs,
            "base_feature_count": 86, "context_feature_count": len(CONTEXT_NAMES), "feature_count": len(names),
            "extracted_label_counts": extraction["extracted_label_counts"],
            "validation_sensitivity_counts": extraction["validation_sensitivity_counts"],
            "quality_summary_by_split_and_label": summaries, "selected_review_pages_context": selected,
            "model_trained": False, "labels_changed": False, "balanced_splits_changed": False,
            "official_test_evaluated": False, "original_features_unchanged": True,
            "policy": ["All original 86 feature values are copied from the verified original checkpoint without modification.",
                "Balanced row IDs, domain-separated splits and sensitivity membership are retained.",
                "Full archived HTML hashes and labels are checked when new context is extracted.",
                "Only field-associated text supplies credential cues; a sign-in link or password article elsewhere is not a credential-field cue.",
                "Challenge/unavailable-page hints and association limits are quality flags, not classifier features or new labels.",
                "No brand target, dataset annotation, label, record ID, site name or source file becomes a classifier feature."],
            "limitations": ["This is a static extension, not observed credential harvesting, dynamic JS execution or the completed crawling/behavior component.",
                "Credential dictionaries cover selected English/German/French/Spanish/Portuguese terms and common variants, not every language or obfuscation method.",
                "Tolerant form/label ownership approximates HTML association; browser DOM repair, CSS visibility, Shadow DOM and dynamic labels are not observed.",
                "HTML and association caps can leave partial signals. All pages are retained and quality flags record these limits.",
                "Password cues and same-site/cross-site POST forms occur on legitimate sites; no individual feature determines a phishing label.",
                "Development validation errors motivated the features. Improvement must be measured; final claims require an untouched test set.",
                "Original dataset URLs remain the reference base. Final redirect URLs and capture-label consistency cannot be reconstructed here.",
                "Completed context checkpoints use matching configuration and source metadata; complete Parquet shard bytes are not re-hashed on resume."]}
        report_path = args.output_dir / "form_context_feature_report.json"
        write_json(report_path, report)
        print("Preserved class counts:", json.dumps(report["extracted_label_counts"], indent=2), flush=True)
        print(f"Saved report: {report_path.resolve()}", flush=True)
        print("Form-context feature extraction complete. No model trained; original features and official test unchanged.", flush=True)
    finally:
        if output is not None:
            output.close()
        original.close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ImportError, sqlite3.Error) as error:
        raise SystemExit(f"Form-context extraction stopped: {error}. A completed report is valid after the final completion message.")
    except MemoryError:
        raise SystemExit("Not enough RAM for source decoding. Close memory-heavy apps and rerun the same command to continue from checkpoints.")
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Rerun the same command to continue; completed checkpoints and original inputs are preserved.")
