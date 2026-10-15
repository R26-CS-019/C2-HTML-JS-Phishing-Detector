"""Extract a static HTML/inline-JavaScript baseline from balanced PhreshPhish.

Requirements: duckdb, tldextract (already used by earlier scripts).
Runs offline: no browser, website requests, script execution or external JS
downloads. Keep this file: extract_static_features() is reusable for inference.
"""
import argparse
import csv
import hashlib
import ipaddress
import json
import math
import platform
import re
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from html.parser import HTMLParser
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import duckdb
import tldextract

VERSION = "1.0"
DEFAULT_MAX_HTML_CHARACTERS = 4_000_000
DEFAULT_MAX_JS_CHARACTERS = 500_000
LABELS = ("benign", "phish")
SPLITS = ("train", "validation")

# These names are the ONLY classifier input columns. Labels, IDs, site names,
# similarity flags, capture date and source shard must never become features.
HTML_DEFINITIONS = {
    "html_character_count": "Full archived HTML character count, before the analysis cap.",
    "tag_count": "Opening tags encountered in the inspected HTML prefix.",
    "text_character_count": "Text node characters outside script/style; not browser-visible text.",
    "title_character_count": "Characters in parsed title text.",
    "form_count": "Parsed form opening tags.",
    "form_post_count": "Forms declaring method=post.",
    "form_get_count": "Forms with missing/empty method or method=get.",
    "form_empty_action_count": "Forms with missing or empty action.",
    "form_cross_site_action_count": "Forms whose resolved HTTP(S) action has a different site group.",
    "form_https_to_http_action_count": "Forms on HTTPS pages with resolved HTTP actions.",
    "form_non_http_action_count": "Forms declaring a non-HTTP(S) action scheme.",
    "form_invalid_action_count": "Forms whose action cannot be interpreted as a valid HTTP(S) reference.",
    "form_onsubmit_count": "Forms declaring an onsubmit attribute.",
    "form_autocomplete_off_count": "Forms declaring autocomplete=off.",
    "password_form_count": "Forms associated with at least one parsed password input.",
    "password_form_cross_site_action_count": "Password forms with a cross-site HTTP(S) action.",
    "password_form_post_count": "Password forms declaring method=post.",
    "input_count": "Parsed input tags.",
    "password_input_count": "Inputs declaring type=password.",
    "email_input_count": "Inputs declaring type=email.",
    "hidden_input_count": "Inputs declaring type=hidden.",
    "text_input_count": "Inputs with missing/empty type or type=text.",
    "submit_input_count": "Inputs declaring type=submit or type=image.",
    "required_input_count": "Inputs declaring required.",
    "credential_hint_input_count": "Inputs with user/email/password/login/OTP hints in name/id/autocomplete; heuristic only.",
    "inputs_without_resolved_form_count": "Inputs not associated with a parsed form, including unresolved form attributes.",
    "formaction_override_count": "Input/button elements declaring a formaction override.",
    "formaction_cross_site_count": "Parsed input/button formaction overrides resolving to a different site group.",
    "button_count": "Parsed button tags.",
    "anchor_count": "Parsed anchor tags, including anchors with no href.",
    "anchor_http_count": "Anchors with non-fragment references resolving to HTTP(S).",
    "anchor_cross_site_count": "HTTP(S) anchors whose destination site group differs from the page.",
    "anchor_fragment_or_empty_count": "Anchors with empty/missing href or fragment-only href.",
    "anchor_javascript_count": "Anchors with a javascript: href.",
    "anchor_mailto_count": "Anchors with a mailto: href.",
    "anchor_invalid_count": "Other anchor references that cannot be resolved to HTTP(S).",
    "image_count": "Parsed img tags.",
    "iframe_count": "Parsed iframe tags.",
    "iframe_cross_site_count": "Iframes with cross-site HTTP(S) src references.",
    "iframe_hidden_hint_count": "Iframes with hidden/display:none/visibility:hidden or explicit zero width/height hints.",
    "meta_refresh_count": "Meta tags declaring http-equiv=refresh.",
    "meta_refresh_cross_site_count": "Refresh tags with a parsed URL resolving to another site group.",
    "script_count": "All parsed script opening tags.",
    "script_external_count": "Script tags with a nonempty src; code contents are not downloaded.",
    "script_cross_site_count": "Nonempty script src references resolving to another site group.",
    "script_inline_executable_count": "Inline script tags with empty/default, module, or recognised JS MIME type.",
    "script_non_executable_data_count": "Script tags with other types, e.g. application/ld+json; not scanned as JS.",
    "script_async_count": "Script tags declaring async.",
    "script_defer_count": "Script tags declaring defer.",
    "event_attribute_count": "Attributes whose names begin with on; lexical event hints.",
    "onsubmit_attribute_count": "All parsed onsubmit attributes.",
    "onclick_attribute_count": "All parsed onclick attributes.",
    "onload_attribute_count": "All parsed onload attributes.",
    "hidden_style_element_count": "Elements declaring hidden or direct inline display:none/visibility:hidden.",
}
RATIO_DEFINITIONS = {
    "password_input_ratio": "password_input_count / max(1,input_count).",
    "hidden_input_ratio": "hidden_input_count / max(1,input_count).",
    "form_post_ratio": "form_post_count / max(1,form_count).",
    "form_cross_site_ratio": "form_cross_site_action_count / max(1,form_count).",
    "anchor_cross_site_ratio": "anchor_cross_site_count / max(1,anchor_http_count).",
    "script_cross_site_ratio": "script_cross_site_count / max(1,script_external_count).",
    "event_attributes_per_tag": "event_attribute_count / max(1,tag_count).",
}
# Lexical matches can occur in comments/strings or benign libraries. They do
# NOT prove execution, harvesting, malicious intent or effective obfuscation.
JS_PATTERNS = {
    "js_eval_call_count": r"\beval\s*\(",
    "js_function_constructor_count": r"\b(?:new\s+)?Function\s*\(",
    "js_atob_call_count": r"\batob\s*\(",
    "js_decode_call_count": r"\b(?:unescape|decodeURI|decodeURIComponent)\s*\(",
    "js_char_code_call_count": r"\b(?:fromCharCode|fromCodePoint)\s*\(",
    "js_hex_unicode_escape_count": r"\\(?:x[0-9a-f]{2}|u[0-9a-f]{4}|u\{[0-9a-f]+\})",
    "js_settimeout_call_count": r"\bsetTimeout\s*\(",
    "js_setinterval_call_count": r"\bsetInterval\s*\(",
    "js_location_reference_count": r"\blocation\b",
    "js_location_assignment_count": r"\blocation\s*(?:\.\s*(?:href|host|hostname|pathname|search|hash)\s*)?=(?!=|>)",
    "js_location_navigation_call_count": r"\blocation\s*\.\s*(?:replace|assign|reload)\s*\(",
    "js_window_open_call_count": r"\bwindow\s*\.\s*open\s*\(",
    "js_fetch_call_count": r"\bfetch\s*\(",
    "js_xhr_constructor_count": r"\bXMLHttpRequest\s*\(",
    "js_sendbeacon_call_count": r"\bsendBeacon\s*\(",
    "js_document_write_call_count": r"\bdocument\s*\.\s*(?:write|writeln)\s*\(",
    "js_html_assignment_count": r"\b(?:innerHTML|outerHTML)\s*=(?!=|>)",
    "js_create_element_call_count": r"\bcreateElement\s*\(",
    "js_event_listener_call_count": r"\baddEventListener\s*\(",
    "js_prevent_default_call_count": r"\bpreventDefault\s*\(",
    "js_form_submit_call_count": r"\.\s*(?:submit|requestSubmit)\s*\(",
    "js_storage_reference_count": r"\b(?:localStorage|sessionStorage)\b",
    "js_cookie_reference_count": r"\bdocument\s*\.\s*cookie\b",
    "js_password_hint_count": r"\b(?:password|passwd|pwd)\b",
}
COMPILED_JS = {name: re.compile(pattern, re.I) for name, pattern in JS_PATTERNS.items()}
JS_DEFINITIONS = {name: "Lexical matches of regex " + pattern + " in analysed inline/event/JS-URL code."
                  for name, pattern in JS_PATTERNS.items()}
JS_DEFINITIONS["js_analysed_character_count"] = "JS characters actually scanned, capped globally per page; excludes external files and non-executable script types."
FEATURE_DEFINITIONS = dict(HTML_DEFINITIONS, **RATIO_DEFINITIONS, **JS_DEFINITIONS)
FEATURE_NAMES = tuple(FEATURE_DEFINITIONS)
METADATA_COLUMNS = ("record_id", "label", "site_group", "source_file", "sampled_similarity_candidate", "in_validation_sensitivity_subset")
QUALITY_COLUMNS = ("html_analysis_truncated", "js_analysis_truncated", "html_parse_error",
                   "unclosed_rawtext_element", "invalid_reference_count")
EXECUTABLE_TYPES = {"", "module", "text/javascript", "application/javascript", "application/ecmascript",
                    "text/ecmascript", "text/jscript", "text/livescript", "application/x-javascript"}
HIDDEN_STYLE = re.compile(r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden)\s*(?:!important\s*)?(?:;|$)", re.I)
CREDENTIAL_HINT = re.compile(r"(?:pass(?:word|wd)?|pwd|user(?:name)?|e[-_]?mail|login|otp|one[-_ ]?time)", re.I)
REFRESH_URL = re.compile(r"(?:^|;)\s*url\s*=\s*(.*?)\s*$", re.I)


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class ReferenceResolver:
    """Offline site grouping, consistent with script 02's private-suffix policy."""
    def __init__(self):
        self.extractor = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)

    @lru_cache(maxsize=32768)
    def site_group(self, host):
        try:
            host = host.rstrip(".").lower()
            if not host or any(c.isspace() for c in host):
                return ""
            try:
                return "ip:" + ipaddress.ip_address(host).compressed
            except ValueError:
                pass
            host = host.encode("idna").decode("ascii").lower()
            if len(host) > 253 or any(not p or len(p) > 63 for p in host.split(".")):
                return ""
            ext = self.extractor(host)
            return "domain:" + ext.top_domain_under_public_suffix if ext.domain and ext.suffix else "host:" + host
        except (ValueError, UnicodeError):
            return ""

    def resolve(self, reference, base_url):
        try:
            resolved = urljoin(base_url, reference.strip())
            parsed = urlsplit(resolved)
            if parsed.scheme.lower() not in ("http", "https"):
                return resolved, "", parsed.scheme.lower(), "non_http"
            group = self.site_group(parsed.hostname or "")
            return resolved, group, parsed.scheme.lower(), "http" if group else "invalid"
        except (ValueError, UnicodeError):
            return "", "", "", "invalid"


def canonical_page_url(url):
    value = str(url or "").strip()
    return "https:" + value if value.startswith("//") else value if "://" in value else "https://" + value


class StaticParser(HTMLParser):
    def __init__(self, page_url, resolver, max_js_characters):
        super().__init__(convert_charrefs=True)
        self.page_url = canonical_page_url(page_url)
        self.resolver = resolver
        self.page_group = resolver.resolve(self.page_url, self.page_url)[1]
        self.page_scheme = urlsplit(self.page_url).scheme.lower()
        self.c = Counter()
        self.max_js = max_js_characters
        self.js_seen = 0
        self.js_scanned = 0
        self.references = []
        self.first_base = None
        self.forms = []
        self.form_stack = []
        self.form_ids = {}
        self.controls = []
        self.script_active = False
        self.scan_script = False
        self.style_active = False
        self.title_active = False
        self.invalid_refs = 0

    def add_js(self, code):
        self.js_seen += len(code)
        remaining = max(0, self.max_js - self.js_scanned)
        sample = code[:remaining]
        self.js_scanned += len(sample)
        if sample:
            for name, pattern in COMPILED_JS.items():
                self.c[name] += sum(1 for _ in pattern.finditer(sample))

    def handle_starttag(self, tag, attrs):
        a = {name: value if value is not None else "" for name, value in attrs}
        self.c["tag_count"] += 1
        hidden = "hidden" in a or bool(HIDDEN_STYLE.search(a.get("style", "")))
        self.c["hidden_style_element_count"] += int(hidden)
        for name, value in attrs:
            if name.startswith("on"):
                self.c["event_attribute_count"] += 1
                if name in ("onsubmit", "onclick", "onload"):
                    self.c[name + "_attribute_count"] += 1
                self.add_js(value or "")
        if tag == "base" and self.first_base is None and "href" in a:
            self.first_base = a["href"]
        elif tag == "form":
            idx = len(self.forms)
            form = {"action": a.get("action", ""), "method": a.get("method", "get").strip().lower() or "get", "passwords": 0}
            self.forms.append(form)
            self.form_stack.append(idx)
            if a.get("id") and a["id"] not in self.form_ids:
                self.form_ids[a["id"]] = idx
            self.c["form_count"] += 1
            self.c["form_post_count"] += int(form["method"] == "post")
            self.c["form_get_count"] += int(form["method"] == "get")
            self.c["form_empty_action_count"] += int(not form["action"].strip())
            self.c["form_onsubmit_count"] += int("onsubmit" in a)
            self.c["form_autocomplete_off_count"] += int(a.get("autocomplete", "").lower() == "off")
        elif tag == "input":
            kind = a.get("type", "text").strip().lower() or "text"
            self.c["input_count"] += 1
            for t in ("password", "email", "hidden", "text"):
                self.c[t + "_input_count"] += int(kind == t)
            self.c["submit_input_count"] += int(kind in ("submit", "image"))
            self.c["required_input_count"] += int("required" in a)
            self.c["credential_hint_input_count"] += int(bool(CREDENTIAL_HINT.search(" ".join(a.get(k, "") for k in ("name", "id", "autocomplete")))))
            owner = ("id", a["form"]) if "form" in a else ("index", self.form_stack[-1] if self.form_stack else None)
            self.controls.append((owner, kind == "password"))
        elif tag == "button":
            self.c["button_count"] += 1
        elif tag == "a":
            self.c["anchor_count"] += 1
            href = a.get("href", "").strip()
            if not href or href.startswith("#"):
                self.c["anchor_fragment_or_empty_count"] += 1
            elif href.lower().startswith("javascript:"):
                self.c["anchor_javascript_count"] += 1
                self.add_js(href[len("javascript:"):])
            elif href.lower().startswith("mailto:"):
                self.c["anchor_mailto_count"] += 1
            else:
                self.references.append(("anchor", href))
        elif tag == "img":
            self.c["image_count"] += 1
        elif tag == "iframe":
            self.c["iframe_count"] += 1
            zero_size = any(a.get(k, "").strip().lower() in ("0", "0px", "0%") for k in ("width", "height"))
            self.c["iframe_hidden_hint_count"] += int(hidden or zero_size)
            if a.get("src", "").strip():
                self.references.append(("iframe", a["src"]))
        elif tag == "meta" and a.get("http-equiv", "").strip().lower() == "refresh":
            self.c["meta_refresh_count"] += 1
            match = REFRESH_URL.search(a.get("content", ""))
            if match:
                self.references.append(("meta_refresh", match.group(1).strip().strip("\"'")))
        elif tag == "script":
            self.script_active = True
            script_type = a.get("type", "").split(";", 1)[0].strip().lower()
            executable = script_type in EXECUTABLE_TYPES or script_type.startswith("text/javascript1.")
            external = bool(a.get("src", "").strip())
            self.scan_script = executable and not external
            self.c["script_count"] += 1
            self.c["script_external_count"] += int(external)
            self.c["script_inline_executable_count"] += int(self.scan_script)
            self.c["script_non_executable_data_count"] += int(not executable)
            self.c["script_async_count"] += int("async" in a)
            self.c["script_defer_count"] += int("defer" in a)
            if external:
                self.references.append(("script", a["src"]))
        elif tag == "style":
            self.style_active = True
        elif tag == "title":
            self.title_active = True
        if tag in ("input", "button") and "formaction" in a:
            self.c["formaction_override_count"] += 1
            self.references.append(("formaction", a["formaction"]))

    def handle_endtag(self, tag):
        if tag == "form" and self.form_stack:
            self.form_stack.pop()
        elif tag == "script":
            self.script_active = False
            self.scan_script = False
        elif tag == "style":
            self.style_active = False
        elif tag == "title":
            self.title_active = False

    def handle_data(self, data):
        if self.script_active:
            if self.scan_script:
                self.add_js(data)
        elif not self.style_active:
            self.c["text_character_count"] += len(data)
            if self.title_active:
                self.c["title_character_count"] += len(data)

    def finish(self):
        # HTMLParser can leave unterminated raw script/style text buffered.
        unclosed = int(self.script_active or self.style_active)
        if self.script_active and self.scan_script and self.rawdata:
            self.add_js(self.rawdata)
        base_url = self.page_url
        if self.first_base is not None:
            candidate, _, _, category = self.resolver.resolve(self.first_base, self.page_url)
            if category == "http":
                base_url = candidate
        for (owner_type, owner), password in self.controls:
            idx = self.form_ids.get(owner) if owner_type == "id" else owner
            if idx is None:
                self.c["inputs_without_resolved_form_count"] += 1
            elif password:
                self.forms[idx]["passwords"] += 1
        for form in self.forms:
            # Empty form action targets the document URL, not <base href>.
            ref = form["action"] if form["action"].strip() else self.page_url
            _, group, scheme, category = self.resolver.resolve(ref, base_url)
            cross = category == "http" and bool(self.page_group) and group != self.page_group
            self.c["form_cross_site_action_count"] += int(cross)
            self.c["form_https_to_http_action_count"] += int(category == "http" and self.page_scheme == "https" and scheme == "http")
            self.c["form_non_http_action_count"] += int(category == "non_http")
            self.c["form_invalid_action_count"] += int(category == "invalid")
            self.invalid_refs += int(category == "invalid")
            if form["passwords"]:
                self.c["password_form_count"] += 1
                self.c["password_form_cross_site_action_count"] += int(cross)
                self.c["password_form_post_count"] += int(form["method"] == "post")
        for kind, ref in self.references:
            # Empty formaction also refers to the current document.
            ref = ref if ref.strip() else self.page_url
            _, group, _, category = self.resolver.resolve(ref, base_url)
            cross = category == "http" and bool(self.page_group) and group != self.page_group
            if kind == "anchor":
                self.c["anchor_http_count"] += int(category == "http")
                self.c["anchor_cross_site_count"] += int(cross)
                self.c["anchor_invalid_count"] += int(category != "http")
            elif kind == "formaction":
                self.c["formaction_cross_site_count"] += int(cross)
            else:
                self.c[kind + "_cross_site_count"] += int(cross)
            self.invalid_refs += int(category == "invalid")
        self.c["js_analysed_character_count"] = self.js_scanned
        return unclosed


def extract_static_features(html, page_url, resolver=None, max_html_characters=DEFAULT_MAX_HTML_CHARACTERS,
                            max_js_characters=DEFAULT_MAX_JS_CHARACTERS, full_html_characters=None):
    """Return (named_numeric_features, quality_flags), without using a label.

    For inference pass saved/live HTML and the original/final HTTP page URL.
    Use the SAME schema, caps and inline-only JS policy as training. The CLI
    already receives a capped prefix from DuckDB; full_html_characters records
    its original length. No function here makes network requests.
    """
    if not isinstance(html, str) or max_html_characters <= 0 or max_js_characters <= 0:
        raise ValueError("HTML must be text and analysis caps must be positive.")
    original_length = len(html) if full_html_characters is None else full_html_characters
    if original_length < len(html):
        raise ValueError("Full HTML length cannot be smaller than the supplied prefix.")
    p = StaticParser(page_url, resolver or ReferenceResolver(), max_js_characters)
    parse_error = 0
    try:
        p.feed(html[:max_html_characters])
        p.close()
    except (AssertionError, ValueError):
        # Preserve partial counts and flag the record; do not silently drop a
        # class or call tolerant parsing proof of a valid webpage capture.
        parse_error = 1
    unclosed = p.finish()
    p.c["html_character_count"] = original_length
    ratios = {
        "password_input_ratio": ("password_input_count", "input_count"),
        "hidden_input_ratio": ("hidden_input_count", "input_count"),
        "form_post_ratio": ("form_post_count", "form_count"),
        "form_cross_site_ratio": ("form_cross_site_action_count", "form_count"),
        "anchor_cross_site_ratio": ("anchor_cross_site_count", "anchor_http_count"),
        "script_cross_site_ratio": ("script_cross_site_count", "script_external_count"),
        "event_attributes_per_tag": ("event_attribute_count", "tag_count"),
    }
    for name, (numerator, denominator) in ratios.items():
        p.c[name] = p.c[numerator] / max(1, p.c[denominator])
    features = {name: p.c[name] for name in FEATURE_NAMES}
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in features.values()):
        raise ValueError("Non-finite or negative feature value.")
    quality = {"html_analysis_truncated": int(original_length > max_html_characters),
               "js_analysis_truncated": int(p.js_seen > max_js_characters), "html_parse_error": parse_error,
               "unclosed_rawtext_element": unclosed, "invalid_reference_count": p.invalid_refs}
    return features, quality


def label_counts(db, condition="1"):
    result = {s: {l: 0 for l in LABELS} for s in SPLITS}
    for split, label, n in db.execute(f"SELECT split,label,count(*) FROM pages WHERE {condition} GROUP BY split,label"):
        result[split][label] = n
    return result


def checkpoint_init(db, args, balance, config):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS pages(
            id TEXT PRIMARY KEY,label TEXT,split TEXT,site_group TEXT,source_file TEXT,
            html_characters INTEGER,raw_sha TEXT UNIQUE,norm_sha TEXT,
            similarity_candidate INTEGER,sensitivity INTEGER,processed INTEGER DEFAULT 0,
            feature_values TEXT,quality_values TEXT);
        CREATE INDEX IF NOT EXISTS source_lookup ON pages(source_file,processed);
    """)
    saved = db.execute("SELECT value FROM state WHERE key='config'").fetchone()
    if saved:
        if json.loads(saved[0]) != config:
            raise ValueError("Inputs, code, environment or feature caps differ from this checkpoint. Use a new --output-dir for a separate run.")
        return
    if db.execute("SELECT count(*) FROM pages").fetchone()[0]:
        raise ValueError("Incomplete checkpoint metadata. Use a new --output-dir.")
    try:
        db.execute("BEGIN")
        with args.manifest.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = {"sha256", "label", "split", "site_group", "source_file", "html_characters", "raw_html_sha256",
                        "normalized_html_sha256", "in_validation_sensitivity_subset", "sampled_similarity_candidate"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("Use the balanced primary manifest from script 08.")
            batch = []
            for row in reader:
                split, label = row["split"], row["label"]
                candidate, sensitivity = int(row["sampled_similarity_candidate"]), int(row["in_validation_sensitivity_subset"])
                if split not in SPLITS or label not in LABELS or candidate not in (0, 1) or sensitivity not in (0, 1):
                    raise ValueError("Unexpected split, label or flag in balanced manifest.")
                if any(not row[k] for k in required) or int(row["html_characters"]) <= 0:
                    raise ValueError("Missing metadata or empty capture in balanced manifest.")
                if split == "train" and (candidate or sensitivity) or sensitivity and candidate:
                    raise ValueError("Unexpected similarity/sensitivity flag.")
                batch.append((row["sha256"], label, split, row["site_group"], row["source_file"], int(row["html_characters"]),
                              row["raw_html_sha256"], row["normalized_html_sha256"], candidate, sensitivity))
                if len(batch) >= 5000:
                    db.executemany("INSERT INTO pages(id,label,split,site_group,source_file,html_characters,raw_sha,norm_sha,similarity_candidate,sensitivity) VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
                    batch.clear()
            if batch:
                db.executemany("INSERT INTO pages(id,label,split,site_group,source_file,html_characters,raw_sha,norm_sha,similarity_candidate,sensitivity) VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
        if label_counts(db) != balance["balanced_counts"] or label_counts(db, "sensitivity=1")["validation"] != balance["validation_sensitivity_counts"]:
            raise ValueError("Manifest counts differ from balance report.")
        for col in ("site_group", "norm_sha"):
            if db.execute(f"SELECT 1 FROM pages GROUP BY {col} HAVING count(DISTINCT split)>1 LIMIT 1").fetchone():
                raise ValueError(f"Training/validation overlap in {col}.")
        with args.sensitivity_manifest.open(newline="", encoding="utf-8-sig") as f:
            seen = set()
            for row in csv.DictReader(f):
                saved_row = db.execute("SELECT label,split,site_group,source_file,sensitivity FROM pages WHERE id=?", (row["sha256"],)).fetchone()
                if row["sha256"] in seen or saved_row != (row["label"], "validation", row["site_group"], row["source_file"], 1):
                    raise ValueError("Secondary validation manifest differs from primary manifest membership.")
                seen.add(row["sha256"])
            if len(seen) != sum(balance["validation_sensitivity_counts"].values()):
                raise ValueError("Secondary validation manifest membership is incomplete.")
        source_names = {s[0] for s in db.execute("SELECT DISTINCT source_file FROM pages")}
        if not source_names.issubset({f["name"] for f in config["source_files"]}):
            raise ValueError("Manifest references unavailable training Parquet files.")
        db.execute("INSERT INTO state VALUES ('config',?)", (json.dumps(config, sort_keys=True),))
        db.commit()
    except BaseException:
        db.rollback()
        raise


def scan_source(db, path, args, resolver):
    rows = db.execute("SELECT id,label,site_group,html_characters,raw_sha FROM pages WHERE source_file=? AND processed=0", (path.name,)).fetchall()
    if not rows:
        print("  Already checkpointed, or no selected records in this file.", flush=True)
        return
    expected = {row[0]: row[1:] for row in rows}
    placeholders = ",".join("?" for _ in rows)
    con = duckdb.connect()
    processed = 0
    seen = set()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET memory_limit=?", (args.memory_limit,))
        # DuckDB hashes the COMPLETE HTML for content integrity but only sends
        # a bounded prefix to Python. Parquet decoding still needs working RAM.
        query = f"""SELECT sha256,label,url,length(html),sha256(html),substr(html,1,{args.max_html_characters})
            FROM read_parquet(?) WHERE sha256 IN ({placeholders})"""
        # Use the lazy relation's row fetcher, rather than materializing the
        # selected shard's HTML as one connection result.
        cursor = con.sql(query, params=[str(path), *expected])
        db.execute("BEGIN")
        while True:
            record = cursor.fetchone()
            if record is None:
                break
            page_id, label, url, full_length, raw_sha, prefix = record
            if page_id not in expected or page_id in seen:
                raise ValueError(f"Unexpected or repeated selected ID in {path.name}.")
            expected_label, group, expected_length, expected_raw_sha = expected[page_id]
            if label != expected_label or full_length != expected_length or raw_sha != expected_raw_sha:
                raise ValueError(f"HTML or label differs from audited content in {path.name}.")
            actual_group = resolver.resolve(canonical_page_url(url), canonical_page_url(url))[1]
            if actual_group != group:
                raise ValueError(f"URL site group differs from the audited manifest in {path.name}.")
            features, quality = extract_static_features(prefix, url, resolver, args.max_html_characters, args.max_js_characters, full_length)
            db.execute("UPDATE pages SET feature_values=?,quality_values=?,processed=1 WHERE id=?",
                       (json.dumps([features[n] for n in FEATURE_NAMES], separators=(",", ":")),
                        json.dumps([quality[n] for n in QUALITY_COLUMNS], separators=(",", ":")), page_id))
            seen.add(page_id)
            processed += 1
            if processed % 100 == 0:
                db.commit()
                db.execute("BEGIN")
            if processed % 500 == 0:
                print(f"  Extracted {processed:,}/{len(rows):,} pending pages; checkpoints saved every 100.", flush=True)
        if len(seen) != len(expected):
            raise ValueError(f"Missing selected records in {path.name}.")
        db.commit()
        print(f"  Saved features for {processed:,} pages.", flush=True)
    except BaseException:
        db.rollback()
        raise
    finally:
        con.close()


def export_features(db, output_dir):
    definitions = [("train_static_features.csv", "split='train'"),
                   ("validation_static_features.csv", "split='validation'"),
                   ("validation_sensitivity_static_features.csv", "sensitivity=1")]
    outputs = {}
    summaries = {split: {label: Counter() for label in LABELS} for split in SPLITS}
    numeric_quality_keys = dict(zip(QUALITY_COLUMNS, range(len(QUALITY_COLUMNS))))
    for name, condition in definitions:
        path = output_dir / name
        temporary = path.with_suffix(".csv.tmp")
        written = 0
        try:
            with temporary.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([*METADATA_COLUMNS, *QUALITY_COLUMNS, *FEATURE_NAMES])
                query = f"""SELECT id,label,site_group,source_file,similarity_candidate,sensitivity,
                    quality_values,feature_values,split FROM pages WHERE {condition} ORDER BY id"""
                for row in db.execute(query):
                    quality, values = json.loads(row[6]), json.loads(row[7])
                    if len(values) != len(FEATURE_NAMES) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                        raise ValueError("Checkpoint has invalid feature values.")
                    writer.writerow([*row[:6], *quality, *values])
                    written += 1
                    if name != "validation_sensitivity_static_features.csv":
                        summary = summaries[row[8]][row[1]]
                        summary["rows"] += 1
                        for flag, idx in numeric_quality_keys.items():
                            summary[flag + "_rows"] += int(quality[idx] > 0)
                        summary["password_input_present_rows"] += int(values[FEATURE_NAMES.index("password_input_count")] > 0)
                        summary["inline_executable_script_present_rows"] += int(values[FEATURE_NAMES.index("script_inline_executable_count")] > 0)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        outputs[name] = {"rows": written, "sha256": digest_file(path)}
    return outputs, summaries


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def recorded_input_hash(report, filename):
    matches = [h for path, h in report["inputs_sha256"].items() if path.replace("\\", "/").split("/")[-1] == filename]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify {filename} provenance in balance report.")
    return matches[0]


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\new dataset\phreshphish_dataset\data"))
    parser.add_argument("--manifest", type=Path, default=root / "reports/balanced/balanced_training_validation_manifest.csv")
    parser.add_argument("--sensitivity-manifest", type=Path, default=root / "reports/balanced/balanced_validation_without_flagged_candidates.csv")
    parser.add_argument("--balance-report", type=Path, default=root / "reports/balanced/balance_report.json")
    parser.add_argument("--preparation-report", type=Path, default=root / "reports/prepared/preparation_report.json")
    parser.add_argument("--fingerprint-report", type=Path, default=root / "reports/html_fingerprints/html_fingerprint_report.json")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/features")
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--max-html-characters", type=int, default=DEFAULT_MAX_HTML_CHARACTERS)
    parser.add_argument("--max-js-characters", type=int, default=DEFAULT_MAX_JS_CHARACTERS)
    args = parser.parse_args()
    if args.max_html_characters < 1000 or args.max_js_characters < 1000:
        parser.error("Analysis caps must be at least 1000 characters.")
    inputs = [args.manifest, args.sensitivity_manifest, args.balance_report, args.preparation_report, args.fingerprint_report]
    for path in inputs:
        if not path.is_file():
            parser.error(f"Missing input: {path}. Complete scripts 05, 06 and 08 first.")
    balance, preparation, fingerprint = load_json(args.balance_report), load_json(args.preparation_report), load_json(args.fingerprint_report)
    if balance["status"] != "balanced_experimental_manifests_complete":
        raise ValueError("Balanced manifests must be complete.")
    hashes = {str(p.resolve()): digest_file(p) for p in inputs}
    for path in (args.manifest, args.sensitivity_manifest):
        if hashes[str(path.resolve())] != balance["outputs"][path.name]["sha256"]:
            raise ValueError(f"Balanced manifest changed: {path.name}.")
    if hashes[str(args.preparation_report.resolve())] != recorded_input_hash(balance, "preparation_report.json"):
        raise ValueError("Preparation report differs from the balancing inputs.")
    if hashes[str(args.fingerprint_report.resolve())] != preparation["input_fingerprint_report_sha256"]:
        raise ValueError("Fingerprint report differs from the prepared inputs.")
    if preparation["output_retained_manifest_sha256"] != recorded_input_hash(balance, "clean_training_validation_manifest.csv"):
        raise ValueError("Balancing and preparation provenance differ.")
    paths = sorted(args.data_dir.glob("train-*.parquet"))
    if not paths:
        raise ValueError(f"No training Parquet files in {args.data_dir}.")
    source_stats = [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths]
    if source_stats != fingerprint["inputs"]["source_files"]:
        raise ValueError("Training source files differ from the fingerprint audit's file metadata.")
    output_names = {"static_features_checkpoint.sqlite", "train_static_features.csv", "validation_static_features.csv",
                    "validation_sensitivity_static_features.csv", "static_feature_schema.json", "feature_extraction_report.json"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).resolve() in {p.resolve() for p in inputs + paths} for name in output_names):
        raise ValueError("Outputs must not overwrite inputs.")
    config = {"script_version": VERSION, "code_sha256": digest_file(Path(__file__)), "inputs_sha256": hashes,
              "source_files": source_stats, "data_dir": str(args.data_dir.resolve()),
              "python_version": platform.python_version(), "duckdb_version": duckdb.__version__, "tldextract_version": version("tldextract"),
              "parameters": {"max_html_characters": args.max_html_characters, "max_js_characters": args.max_js_characters},
              "feature_names": list(FEATURE_NAMES), "external_js_policy": "references_only_no_download"}
    checkpoint = args.output_dir / "static_features_checkpoint.sqlite"
    if not checkpoint.exists() and shutil.disk_usage(args.output_dir).free < 1024 ** 3:
        raise ValueError("Keep at least 1 GiB free for feature files and checkpoints.")
    print(f"Static feature count: {len(FEATURE_NAMES)}; DuckDB memory limit: {args.memory_limit}; threads: 1", flush=True)
    print(f"Analysis caps: {args.max_html_characters:,} HTML characters; {args.max_js_characters:,} inline/event JS characters per page.", flush=True)
    print("This is static analysis. No websites are opened and no JavaScript is executed.", flush=True)
    db = sqlite3.connect(str(checkpoint))
    db.execute("PRAGMA cache_size=-32768")
    resolver = ReferenceResolver()
    try:
        checkpoint_init(db, args, balance, config)
        complete = db.execute("SELECT count(*) FROM pages WHERE processed=1").fetchone()[0]
        print(f"Checkpointed pages: {complete:,}/{balance['primary_records']:,}.", flush=True)
        for index, path in enumerate(paths, 1):
            print(f"HTML/JS features {index}/{len(paths)}: {path.name}", flush=True)
            scan_source(db, path, args, resolver)
        if label_counts(db, "processed=1") != balance["balanced_counts"]:
            raise ValueError("Feature extraction did not complete all selected records.")
        for path in inputs:
            if digest_file(path) != hashes[str(path.resolve())]:
                raise ValueError("An input report or manifest changed during extraction.")
        if [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths] != source_stats:
            raise ValueError("Training source file metadata changed during extraction.")
        outputs, quality = export_features(db, args.output_dir)
        schema = {"schema_version": VERSION, "feature_count": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
                  "feature_definitions": FEATURE_DEFINITIONS, "metadata_columns": list(METADATA_COLUMNS),
                  "quality_columns_not_classifier_inputs": list(QUALITY_COLUMNS), "parameters": config["parameters"],
                  "implementation_code_sha256": config["code_sha256"], "external_js_policy": config["external_js_policy"],
                  "inference_contract": "Use extract_static_features() from this exact implementation with the same caps and inline/event-only JS policy. Keep label, identifiers and quality metadata outside X.",
                  "method_references": ["https://docs.python.org/3.12/library/html.parser.html", "https://github.com/john-kurkowski/tldextract"]}
        schema_path = args.output_dir / "static_feature_schema.json"
        write_json(schema_path, schema)
        outputs[schema_path.name] = {"sha256": digest_file(schema_path)}
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "script_version": VERSION,
                  "status": "static_feature_extraction_complete", "inputs": config, "feature_count": len(FEATURE_NAMES),
                  "extracted_label_counts": label_counts(db, "processed=1"),
                  "validation_sensitivity_counts": label_counts(db, "processed=1 AND sensitivity=1")["validation"],
                  "quality_summary_by_split_and_label": quality, "outputs": outputs, "model_trained": False,
                  "policy": [
                      "No labels, brand targets, supplied IDs, capture date, language, site names, source shards or similarity flags are classifier features.",
                      "Source labels are checked for integrity only; extract_static_features() has no label argument.",
                      "HTML character caps and a global JS scan cap apply identically to each page and label; truncation is reported as quality metadata.",
                      "No parse-error or truncated page is silently dropped; selected class balance is preserved. Review quality counts before training.",
                      "Secondary validation reuses the exact same feature rows; it is not independently re-extracted.",
                      "Full raw HTML hashes are checked against audited fingerprints for each newly processed record. Checkpoints commit every 100 pages.",
                      "Only training-release Parquet files are opened. No official test contents are read."
                  ],
                  "limitations": [
                      "This is a static baseline, not recorded JavaScript behavior, a web crawler or a classifier.",
                      "External JS code is unavailable under this policy. Do not add downloaded external scripts only at prediction time to an inline-only model.",
                      "Regex indicators are approximate and may match comments/strings or benign library code; no individual feature establishes phishing.",
                      "HTMLParser is tolerant and does not implement browser DOM repair or execution; form ownership and visibility hints are approximations.",
                      "Large HTML pages use a bounded prefix; code after the HTML/JS caps can be missed. Hidden CSS via stylesheets is not resolved.",
                      "Document size and coding style can reflect collection differences. Accuracy, false positives and generalization require later evaluation and ablation.",
                      "Source file metadata is checked for all shards, but completed checkpoint records are not re-hashed on resume. Metadata checks do not prove unchanged file bytes."
                  ]}
        report_path = args.output_dir / "feature_extraction_report.json"
        write_json(report_path, report)
        print("Feature counts by split:", json.dumps(report["extracted_label_counts"], indent=2), flush=True)
        print(f"Saved report: {report_path.resolve()}", flush=True)
        print("Static feature extraction complete. No model trained yet.", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except duckdb.OutOfMemoryException:
        raise SystemExit("Not enough RAM to decode a Parquet row group. Close memory-heavy applications and rerun the same command, increasing --memory-limit only if spare RAM permits. Completed 100-page checkpoints remain. No new complete feature report was written.")
    except (ValueError, KeyError, sqlite3.Error) as error:
        raise SystemExit(f"Feature extraction stopped: {error}. Checkpointed pages remain. A completed report is valid only after the final completion message.")
