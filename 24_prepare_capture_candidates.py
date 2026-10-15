"""Prepare a reproducible URL candidate list; do not browse candidate sites.

Run in the existing project .venv: python 24_prepare_capture_candidates.py
Requires the existing tldextract==5.4.0 and the completed script-23 report.
Default target: 50 proposed benign + 50 reported phishing candidates.

Downloads only OpenPhish's official linked public text feed. Keeps a dated,
hashed local snapshot, filters malformed/private-address/non-page URLs,
deduplicates and selects at most one URL per private-suffix-aware site group.
Fifty curated official endpoints supply the provisional benign candidates.
Their lookup observations and references are provenance, not final labels.

The output has proposed_label but ground_truth_label=null, capture_status=
not_attempted, and training_eligible=false for EVERY candidate. No browser,
credentials, website crawl, dataset scan, model training or test evaluation.
Upload only candidate_preparation_report.json for the next review.

OpenPhish source/terms: https://www.openphish.com/phishing_feeds.html
https://www.openphish.com/terms.html . Keep feed-derived URL lists private;
check the provider's permission requirements before redistribution.
"""

import argparse
import gzip
import hashlib
import io
import ipaddress
import json
import platform
import re
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

VERSION = "1.0"
FEED_URL = "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"
FEED_DOC = "https://www.openphish.com/phishing_feeds.html"
FEED_TERMS = "https://www.openphish.com/terms.html"
PILOT_SCRIPT_SHA = "319b6fdf87659235fe878189ba6fb1eed8113829fac7065fbe23cb420648aa86"
PILOT_FIXTURE_SHA = "ac8caada7e353734e351c6f226ca79b4e6094fa2d116f883e6df49eaf5a22c0b"
REPORT_NAME = "candidate_preparation_report.json"
MAX_RAW_BYTES = 16 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024
NON_PAGE_SUFFIXES = {".js", ".mjs", ".css", ".json", ".xml", ".txt", ".db", ".sql", ".csv",
                     ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".woff", ".woff2",
                     ".ttf", ".mp3", ".mp4", ".pdf", ".zip", ".gz", ".bz2", ".7z", ".exe", ".dll"}
EXPECTED_CASES = {
    "normal_same_origin_login", "ordinary_cross_origin_login", "ordinary_password_visibility_toggle",
    "delayed_contact_form", "mocked_external_script_form", "encoded_delayed_form", "japanese_label_xhr",
    "delayed_dummy_submission", "unknown_request_containment"}

# URL, intended page-type hint and public lookup observation. These are seed
# candidates, not declarations that the page's later capture will be benign.
# Lookup observations refer to public text retrieval on 2026-10-08, which can
# differ from browser rendering. Failed/challenged lookups remain explicit.
BENIGN_SEEDS = [
    ("GitHub", "https://github.com/login", "authentication", "sign_in_text_seen"),
    ("Google", "https://accounts.google.com/", "authentication", "sign_in_text_seen"),
    ("Microsoft", "https://login.live.com/", "authentication", "javascript_required"),
    ("Adobe", "https://account.adobe.com/", "authentication", "javascript_required"),
    ("Dropbox", "https://www.dropbox.com/login", "authentication", "empty_text_response"),
    ("Yahoo", "https://login.yahoo.com/", "authentication", "lookup_rate_limited"),
    ("Apple", "https://account.apple.com/", "authentication", "empty_text_response"),
    ("WordPress", "https://wordpress.com/log-in/", "authentication", "javascript_required"),
    ("Atlassian", "https://id.atlassian.com/login", "authentication", "javascript_required"),
    ("Zoom", "https://www.zoom.com/signin", "authentication", "lookup_unavailable"),
    ("Slack", "https://slack.com/signin", "authentication", "lookup_access_denied"),
    ("Salesforce", "https://login.salesforce.com/", "authentication", "sign_in_text_seen"),
    ("Spotify", "https://accounts.spotify.com/en/login", "authentication", "sign_in_text_seen"),
    ("Netflix", "https://www.netflix.com/login", "authentication", "lookup_unavailable"),
    ("PayPal", "https://www.paypal.com/signin", "authentication", "sign_in_text_seen"),
    ("eBay", "https://signin.ebay.com/signin/", "authentication", "sign_in_text_seen"),
    ("LinkedIn", "https://www.linkedin.com/login", "authentication", "sign_in_text_seen"),
    ("Facebook", "https://www.facebook.com/login/", "authentication", "sign_in_and_block_message_seen"),
    ("Reddit", "https://www.reddit.com/login/", "authentication", "sign_in_text_seen"),
    ("Pinterest", "https://www.pinterest.com/login/", "authentication", "lookup_unavailable"),
    ("GitLab", "https://gitlab.com/users/sign_in", "authentication", "sign_in_text_seen"),
    ("Mozilla Account", "https://accounts.firefox.com/", "authentication", "client_challenge_seen"),
    ("Stack Overflow", "https://stackoverflow.com/users/login", "authentication", "sign_in_text_seen"),
    ("Canva", "https://www.canva.com/login/", "authentication", "sign_in_text_seen"),
    ("Notion", "https://app.notion.com/login", "authentication", "official_endpoint_redirect_and_javascript_required"),
    ("Grammarly", "https://account.grammarly.com/", "authentication", "official_endpoint_redirect_and_sign_in_text_seen"),
    ("Proton", "https://account.proton.me/login", "authentication", "javascript_required"),
    ("Figma", "https://www.figma.com/login", "authentication", "lookup_unavailable"),
    ("Asana", "https://app.asana.com/-/login", "authentication", "sign_in_text_seen"),
    ("Khan Academy", "https://www.khanacademy.org/login", "authentication", "empty_text_response"),
    ("Amazon", "https://www.amazon.com/", "general", "client_challenge_seen"),
    ("Wikipedia", "https://www.wikipedia.org/", "general", "site_text_seen"),
    ("NASA", "https://www.nasa.gov/", "general", "site_text_seen"),
    ("BBC", "https://www.bbc.com/", "general", "lookup_unavailable"),
    ("Mozilla", "https://www.mozilla.org/", "general", "site_text_seen"),
    ("Python", "https://www.python.org/", "general", "site_text_seen"),
    ("DuckDB", "https://duckdb.org/", "general", "site_text_seen"),
    ("PostgreSQL", "https://www.postgresql.org/", "general", "site_text_seen"),
    ("MySQL", "https://www.mysql.com/", "general", "site_text_seen"),
    ("phpMyAdmin", "https://www.phpmyadmin.net/", "general", "site_text_seen"),
    ("SLIIT", "https://www.sliit.lk/", "general", "lookup_access_denied"),
    ("University of Moratuwa", "https://uom.lk/", "general", "site_text_seen"),
    ("MIT", "https://www.mit.edu/", "general", "site_text_seen"),
    ("Stanford", "https://www.stanford.edu/", "general", "site_text_seen"),
    ("United Nations", "https://www.un.org/", "general", "lookup_unavailable"),
    ("WHO", "https://www.who.int/", "general", "site_text_seen"),
    ("W3C", "https://www.w3.org/", "general", "site_text_seen"),
    ("Project Gutenberg", "https://www.gutenberg.org/", "general", "site_text_seen"),
    ("IETF", "https://www.ietf.org/", "general", "site_text_seen"),
    ("LibreOffice", "https://www.libreoffice.org/", "general", "site_text_seen"),
]
OFFICIAL_REFERENCES = {
    "GitHub": "https://docs.github.com/en/enterprise-cloud@latest/authentication/authenticating-with-single-sign-on/authenticating-with-a-managed-user-account",
    "Google": "https://support.google.com/accounts/answer/27441?hl=en",
    "Adobe": "https://helpx.adobe.com/account/individual/sign-in-and-security/sign-in-methods/sign-in-to-adobe-account.html",
    "Dropbox": "https://help.dropbox.com/security/password-reset",
    "Notion": "https://www.notion.so/login",
}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def object_sha(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_limited(path, maximum=MAX_RAW_BYTES):
    with path.open("rb") as file:
        data = file.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"{path.name} exceeds the {maximum // 1048576} MiB input limit.")
    return data


def save_json(path, value):
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_pilot(root, report_path):
    script = root / "23_validate_resource_capture.py"
    if not script.is_file() or sha_bytes(read_limited(script)) != PILOT_SCRIPT_SHA:
        raise ValueError("Keep the original verified 23_validate_resource_capture.py beside script 24.")
    raw = read_limited(report_path, 1_048_576)
    report = json.loads(raw.decode("utf-8-sig"))
    config = report["config"]
    if (report["status"] != "resource_capture_pilot_complete" or report["controlled_cases"] != 9 or
            report["passed_cases"] != 9 or report["failed_cases"] != 0 or
            config["script_sha256"] != PILOT_SCRIPT_SHA or config["fixture_definitions_sha256"] != PILOT_FIXTURE_SHA or
            config["chromium_sandbox_requested"] is not True):
        raise ValueError("Script 23 must finish all nine controlled cases successfully before URL selection.")
    cases = report["cases"]
    if len(cases) != 9 or {c["case_id"] for c in cases} != EXPECTED_CASES:
        raise ValueError("Unexpected controlled case list in the script-23 report.")
    for case in cases:
        if (case["status"] != "passed" or case.get("error") is not None or
                any(case["quality"].values()) or not case["checks"] or
                not all(check["passed"] is True for check in case["checks"])):
            raise ValueError("A controlled capture has incomplete, failed or truncated observations.")
        required = {"initial_html_response_exact", "all_received_response_bodies_exact_and_hashed",
                    "every_referenced_external_script_body_retained", "css_body_retained",
                    "server_received_expected_dummy_identity_and_password_requests", "final_dom_html_retained",
                    "page_requests_only_owned_loopback_or_aborted", "observation_not_truncated_or_failed"}
        if not required.issubset({check["property"] for check in case["checks"]}):
            raise ValueError("Controlled resource-capture checks are missing from the report.")
        for check in case["checks"]:
            if "comparison" in check:
                observed, expected = check["observed"], check["expected"]
                valid = observed == expected if check["comparison"] == "equal" else observed >= expected if check["comparison"] == "at_least" else False
                if not valid or observed != case["behaviour_features"][check["property"]]:
                    raise ValueError("A controlled report check disagrees with its recorded feature.")
    return sha_bytes(raw)


def normalize_url(raw):
    if not isinstance(raw, str) or not raw or len(raw) > 4096:
        raise ValueError("not_a_bounded_url_string")
    raw = raw.strip()
    if any(ord(c) < 33 or ord(c) == 127 for c in raw) or "\\" in raw:
        raise ValueError("url_contains_space_control_or_backslash")
    p = urlsplit(raw)
    if p.scheme.lower() not in ("http", "https") or not p.hostname:
        raise ValueError("not_http_or_https")
    if p.username is not None or p.password is not None:
        raise ValueError("url_has_embedded_credentials")
    host = p.hostname.rstrip(".").encode("idna").decode("ascii").lower()
    if not host or len(host) > 253:
        raise ValueError("invalid_host")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if ("." not in host or host.endswith((".localhost", ".local", ".internal", ".test", ".example", ".invalid")) or
                host.split(".")[-1].isdigit() or
                all(re.fullmatch(r"(?:0x[0-9a-f]+|[0-9]+)", part) for part in host.split(".")) or
                any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", part) for part in host.split("."))):
            raise ValueError("nonpublic_or_invalid_hostname")
    else:
        if not address.is_global:
            raise ValueError("nonpublic_ip_literal")
        host = address.compressed
    port = p.port
    if port is not None and port not in (80, 443):
        raise ValueError("nonstandard_port_outside_pilot_scope")
    netloc = "[" + host + "]" if ":" in host else host
    if port is not None and port != (443 if p.scheme.lower() == "https" else 80):
        netloc += ":" + str(port)
    path = p.path or "/"
    # Queries and fragments can drive phishing content/SPA routing. Preserve
    # their exact ordering/text instead of merging semantically different URLs.
    return urlunsplit((p.scheme.lower(), netloc, path, p.query, p.fragment)), host


def site_group(host, extractor):
    try:
        return "ip:" + ipaddress.ip_address(host).compressed
    except ValueError:
        parsed = extractor(host)
        return "domain:" + parsed.top_domain_under_public_suffix if parsed.domain and parsed.suffix else "host:" + host


class FeedRedirectPolicy(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, newurl):
        if newurl != FEED_URL:
            raise ValueError("The feed redirected outside the fixed official source; download stopped.")
        return super().redirect_request(request, file, code, message, headers, newurl)


def download_feed():
    request = urllib.request.Request(FEED_URL, headers={
        "User-Agent": "HTMLJSPhishingResearch-CandidatePreparation/1.0",
        "Accept": "text/plain", "Accept-Encoding": "identity"})
    opener = urllib.request.build_opener(FeedRedirectPolicy())
    started = time.monotonic()
    with opener.open(request, timeout=15) as response:
        if response.status != 200 or response.geturl() != FEED_URL:
            raise ValueError("Unexpected feed response status or source.")
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" in content_type:
            raise ValueError("Feed source returned HTML instead of its URL text file.")
        chunks, size = [], 0
        while True:
            if time.monotonic() - started > 30:
                raise TimeoutError("Feed download exceeded its bounded time window.")
            part = response.read(min(65536, MAX_RAW_BYTES + 1 - size))
            if not part:
                break
            size += len(part)
            if size > MAX_RAW_BYTES:
                raise ValueError("Feed exceeds the input byte limit.")
            chunks.append(part)
        metadata = {"source_name": "openphish_community", "source_url": FEED_URL,
                    "documentation_url": FEED_DOC, "terms_url": FEED_TERMS,
                    "acquired_utc": utc_now(), "acquisition_mode": "single_fixed_source_https_download",
                    "http_etag": response.headers.get("ETag"), "http_last_modified": response.headers.get("Last-Modified"),
                    "per_url_detection_time_available": False, "per_url_online_status_available": False}
    return b"".join(chunks), metadata


def parse_feed(raw, feed_format, provenance, acquired_utc):
    if raw[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            payload = stream.read(MAX_DECOMPRESSED_BYTES + 1)
        if len(payload) > MAX_DECOMPRESSED_BYTES:
            raise ValueError("Decompressed feed exceeds its bounded input limit.")
    else:
        payload = raw
    text = payload.decode("utf-8-sig")
    if text.lstrip().lower().startswith(("<!doctype html", "<html")):
        raise ValueError("Feed contains an HTML access/error page, not candidate data.")
    rows, rejected = [], Counter()
    if feed_format == "openphish":
        for number, url in enumerate(text.splitlines(), 1):
            if not url.strip():
                continue
            rows.append({"raw_url": url, "proposed_label": "phish", "page_type_hint": "unknown",
                         "source_name": "openphish_community", "source_row": number,
                         "label_evidence_type": "vendor_reported_phishing_url",
                         "label_evidence_url": FEED_URL, "source_claims_verified": None,
                         "source_claims_online": None, "source_detection_time": None})
    else:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("PhishTank JSON must contain an array of feed records.")
        for number, item in enumerate(data, 1):
            if not isinstance(item, dict) or item.get("verified") != "yes" or item.get("online") != "yes":
                rejected["phishtank_not_verified_and_online_in_snapshot"] += 1
                continue
            pid = str(item.get("phish_id", ""))
            try:
                checked = datetime.fromisoformat(str(item["verification_time"]).replace("Z", "+00:00"))
                if checked.tzinfo is None or checked > datetime.fromisoformat(acquired_utc):
                    raise ValueError("Invalid verification time")
            except (KeyError, ValueError):
                rejected["phishtank_missing_or_invalid_verification_time"] += 1
                continue
            if not pid.isdecimal() or int(pid) < 1:
                rejected["phishtank_invalid_id"] += 1
                continue
            rows.append({"raw_url": item.get("url"), "proposed_label": "phish", "page_type_hint": "unknown",
                         "source_name": "phishtank_local_snapshot", "source_row": number,
                         "source_phish_id": pid, "label_evidence_type": "imported_community_verified_snapshot_claim",
                         "label_evidence_url": "https://www.phishtank.net/phish_detail.php?phish_id=" + pid,
                         "source_claims_verified": True, "source_claims_online": True,
                         "source_detection_time": item.get("verification_time")})
    if not rows:
        raise ValueError("Feed has no usable source records. No candidate URLs were visited.")
    for row in rows:
        row["source_snapshot_sha256"] = provenance["snapshot_sha256"]
        row["source_snapshot_acquired_utc"] = acquired_utc
        row["source_origin_authenticated"] = provenance["acquisition_mode"] == "single_fixed_source_https_download"
    return rows, rejected


def prepare_rows(source_rows, extractor):
    accepted, decisions = {}, []
    for row in source_rows:
        decision = {"source_name": row["source_name"], "source_row": row["source_row"],
                    "raw_url_value_sha256": object_sha(row["raw_url"])}
        try:
            canonical, host = normalize_url(row["raw_url"])
            if Path(unquote(urlsplit(canonical).path).rstrip("/")).suffix.lower() in NON_PAGE_SUFFIXES:
                raise ValueError("obvious_non_page_path_suffix")
            group = site_group(host, extractor)
        except (ValueError, UnicodeError) as error:
            decision.update(decision="rejected", reason=str(error)[:100])
            decisions.append(decision)
            continue
        cid = sha_bytes(canonical.encode("utf-8"))
        if cid in accepted:
            decision.update(candidate_id=cid, decision="rejected", reason="duplicate_canonical_url")
        else:
            accepted[cid] = {**row, "candidate_id": cid, "url": canonical, "site_group": group,
                             "ground_truth_label": None, "label_review_status": "pending_capture_and_independent_review",
                             "capture_status": "not_attempted", "observed_page_type": None,
                             "capture_utc": None, "training_eligible": False, "split": "unassigned"}
            decision.update(candidate_id=cid, decision="eligible_for_candidate_selection", reason=None)
        decisions.append(decision)
    return list(accepted.values()), decisions


def select_rows(benign, phish, per_class, seed):
    rank = lambda row: (sha_bytes((str(seed) + "|" + row["candidate_id"]).encode()), row["candidate_id"])
    exact_conflicts = {r["candidate_id"] for r in benign} & {r["candidate_id"] for r in phish}
    # A familiar domain may host compromised paths. Do not infer that a feed
    # entry is wrong from the benign catalogue; expose conflicts and keep both
    # claims out of this initial selection.
    benign_groups = {r["site_group"] for r in benign if r["candidate_id"] not in exact_conflicts}
    chosen, decisions = [], []
    for label, pool in (("benign", benign), ("phish", phish)):
        groups, total = set(), 0
        for row in sorted(pool, key=rank):
            if row["candidate_id"] in exact_conflicts:
                reason = "cross_label_exact_url_conflict"
            elif label == "phish" and row["site_group"] in benign_groups:
                reason = "site_group_overlaps_benign_candidate_catalogue"
            elif row["site_group"] in groups:
                reason = "additional_url_same_site_group"
            elif total >= per_class:
                reason = "reserve_after_requested_candidate_quota"
            else:
                groups.add(row["site_group"])
                total += 1
                chosen.append(row)
                reason = "selected"
            decisions.append({"candidate_id": row["candidate_id"], "proposed_label": label,
                              "site_group": row["site_group"], "decision": reason})
    counts = {label: sum(r["proposed_label"] == label for r in chosen) for label in ("benign", "phish")}
    if counts["benign"] != counts["phish"]:
        # Preserve the shortage and all decisions; don't invent labels or
        # silently shrink the requested target to claim completion/balance.
        balanced = False
    else:
        balanced = True
    return chosen, decisions, counts, balanced


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "reports/capture_candidates_v1")
    parser.add_argument("--pilot-report", type=Path, default=root / "reports/fresh_capture_pilot_v1/resource_capture_report.json")
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feed-file", type=Path, help="Optional existing local text or PhishTank JSON/gzip snapshot; no network download in this mode.")
    parser.add_argument("--feed-format", choices=("openphish", "phishtank"), default="openphish")
    args = parser.parse_args()
    if not 1 <= args.per_class <= 50:
        parser.error("--per-class must be 1..50 for the curated pilot catalogue.")
    if args.feed_format == "phishtank" and args.feed_file is None:
        parser.error("PhishTank mode requires a provided local --feed-file; public access is not assumed.")
    pilot_sha = verify_pilot(root, args.pilot_report)
    try:
        import tldextract
    except ImportError as error:
        raise SystemExit("Install in .venv: python -m pip install tldextract==5.4.0") from error
    if version("tldextract") != "5.4.0":
        raise ValueError("Use the existing validated version: python -m pip install tldextract==5.4.0")
    extractor = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
    output = args.output_dir.resolve()
    reserved = [root / "reports" / name for name in ("features", "features_context_v2", "balanced",
        "baseline_training_v1_1", "form_context_training_v2", "capture_quality_v1", "browser_pilot_v1",
        "archived_browser_pilot_v1", "fresh_capture_pilot_v1")]
    reserved.append(Path(r"D:\new dataset\phreshphish_dataset"))
    if root.is_relative_to(output) or any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in reserved):
        raise ValueError("Use a separate candidate output folder outside the dataset and earlier reports.")
    config = {"script_version": VERSION, "script_sha256": sha_bytes(Path(__file__).read_bytes()),
              "python_version": platform.python_version(), "tldextract_version": version("tldextract"),
              "site_group_policy": "offline_bundled_public_suffix_snapshot_with_private_domains",
              "seed": args.seed, "requested_per_proposed_label": args.per_class,
              "benign_seed_catalogue_sha256": object_sha(BENIGN_SEEDS), "pilot_report_sha256": pilot_sha,
              "feed_format": args.feed_format, "feed_mode": "local_snapshot" if args.feed_file else "official_https_download",
              "local_input_sha256": sha_bytes(read_limited(args.feed_file)) if args.feed_file else None}
    if output.exists() and any(output.iterdir()):
        path = output / REPORT_NAME
        if path.is_file():
            saved = json.loads(read_limited(path).decode("utf-8-sig"))
            if saved.get("config") == config:
                for name, digest in saved["output_sha256"].items():
                    if name not in ("source_feed_snapshot.bin", "candidate_manifest.json", "candidate_decisions.json") or sha_bytes(read_limited(output / name)) != digest:
                        raise ValueError("A saved selection file is missing or changed. Preserve it and use a new --output-dir.")
                print("Existing dated snapshot and selection verified; no new feed request or candidate-site visit.", flush=True)
                print("Selected candidate counts:", json.dumps(saved["selected_candidate_counts"]), flush=True)
                print(f"Saved report: {path}", flush=True)
                return 0 if saved["requested_target_met"] else 1
        raise ValueError("Output belongs to a different/incomplete run. Preserve it and use a new --output-dir, such as .\\reports\\capture_candidates_v1_retry")
    if args.feed_file:
        raw = read_limited(args.feed_file)
        provenance = {"source_name": args.feed_format + "_local_snapshot", "acquired_utc": utc_now(),
                      "source_url": FEED_URL if args.feed_format == "openphish" else "https://www.phishtank.net/developer_info.php",
                      "acquisition_mode": "local_file_origin_and_original_download_time_not_authenticated",
                      "local_file_sha256": sha_bytes(raw), "original_snapshot_download_time": None}
    else:
        print("Downloading the official public URL text feed only; candidate sites will not be opened.", flush=True)
        raw, provenance = download_feed()
    provenance["snapshot_sha256"] = sha_bytes(raw)
    provenance["snapshot_bytes"] = len(raw)
    feed_rows, feed_rejections = parse_feed(raw, args.feed_format, provenance, provenance["acquired_utc"])
    seed_rows = [{"raw_url": url, "proposed_label": "benign", "organization": organization,
                  "page_type_hint": hint, "source_name": "curated_official_endpoint_seed", "source_row": number,
                  "label_evidence_type": "official_endpoint_candidate_requires_capture_review",
                  "label_evidence_url": OFFICIAL_REFERENCES.get(organization, url),
                  "public_reference_lookup_date": "2026-10-08", "public_reference_lookup_observation": observation,
                  "source_claims_verified": None, "source_claims_online": None,
                  "current_content_reviewed": False}
                 for number, (organization, url, hint, observation) in enumerate(BENIGN_SEEDS, 1)]
    benign, benign_decisions = prepare_rows(seed_rows, extractor)
    phish, phish_decisions = prepare_rows(feed_rows, extractor)
    selected, selection_decisions, counts, balanced = select_rows(benign, phish, args.per_class, args.seed)
    target_met = counts == {"benign": args.per_class, "phish": args.per_class}
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_feed_snapshot.bin").write_bytes(raw)
    manifest = {"status": "candidate_list_pending_capture_and_label_review", "created_utc": utc_now(),
                "config": config, "feed_provenance": provenance, "selected_candidate_counts": counts,
                "selection_balanced_by_proposed_label": balanced, "requested_target_met": target_met,
                "ground_truth_confirmed_counts": {"benign": 0, "phish": 0},
                "training_eligible": False, "ready_for_unrestricted_live_execution": False,
                "page_type_hints_are_provenance_not_classifier_features": True, "candidates": selected}
    save_json(output / "candidate_manifest.json", manifest)
    save_json(output / "candidate_decisions.json", {"feed_parse_rejections": dict(feed_rejections),
              "benign_url_validation": benign_decisions, "phish_url_validation": phish_decisions,
              "selection_decisions": selection_decisions})
    by_label = {label: [row for row in selected if row["proposed_label"] == label] for label in ("benign", "phish")}
    report = {"status": "candidate_preparation_complete_pending_review" if target_met else "candidate_preparation_target_shortfall",
              "created_utc": utc_now(), "config": config, "feed_provenance": provenance,
              "raw_source_record_count": len(feed_rows) + sum(feed_rejections.values()),
              "eligible_candidate_pool_counts": {"benign": len(benign), "phish": len(phish)},
              "selected_candidate_counts": counts, "requested_target_met": target_met,
              "selection_balanced_by_proposed_label": balanced,
              "selected_site_group_counts": {label: len({row["site_group"] for row in rows}) for label, rows in by_label.items()},
              "selected_cross_label_shared_site_groups": len({row["site_group"] for row in by_label["benign"]} & {row["site_group"] for row in by_label["phish"]}),
              "selected_page_type_hint_counts": {label: dict(Counter(row["page_type_hint"] for row in rows)) for label, rows in by_label.items()},
              "selected_benign_reference_lookup_counts": dict(Counter(row["public_reference_lookup_observation"] for row in by_label["benign"])),
              "url_validation_rejection_counts": dict(Counter(row["reason"] for row in [*benign_decisions, *phish_decisions] if row["decision"] == "rejected")),
              "feed_parse_rejection_counts": dict(feed_rejections),
              "selection_decision_counts": dict(Counter(row["decision"] for row in selection_decisions)),
              "output_sha256": {name: sha_bytes(read_limited(output / name)) for name in
                  ("source_feed_snapshot.bin", "candidate_manifest.json", "candidate_decisions.json")},
              "ground_truth_confirmed_counts": {"benign": 0, "phish": 0},
              "candidate_sites_requested": False, "candidate_dns_resolved": False,
              "candidate_html_or_js_downloaded": False, "credentials_entered": False,
              "model_trained": False, "official_test_evaluated": False,
              "existing_dataset_or_model_changed": False, "training_eligible": False,
              "review_protocol": [
                  "Capture initial/final URL and timestamps, HTML and resource bodies in an isolated collection environment before executing live reported phishing content.",
                  "Review the actual captured page and independent label evidence. Record benign, phish, uncertain or unavailable outcomes; do not infer a label from model predictions.",
                  "Record page type, scripts missing/blocked, errors, challenge/error/parked pages, timeouts and redirect targets. A feed entry can change or disappear.",
                  "Keep every attempted candidate and capture outcome. Report usable reviewed class counts separately; any supplemental candidates require a new traceable selection.",
                  "Check page-type, source/platform, language, brand and capture-quality differences before claiming a balanced, useful dataset.",
                  "Group domains and deduplicate source/content before future train/validation/test assignment; existing-dataset overlap has not been checked here."],
              "limitations": [
                  "Proposed benign/phish counts balance acquisition candidates, not verified ground-truth pages or successful captures.",
                  "OpenPhish community text has no per-URL verification/online/detection-time fields; null means unavailable, never verified.",
                  "Imported PhishTank fields are source claims at the supplied snapshot time; local file origin and current content are not independently authenticated.",
                  "Official endpoints are provisional negative seeds; reputation, HTTPS or absence from the feed alone cannot establish current benign content.",
                  "The benign list is curated and includes 30 authentication hints and 20 general hints; phishing page types are unknown. This is not a representative accuracy benchmark.",
                  "Blocked, empty or challenge responses during public reference lookup were recorded, not bypassed or converted into confirmed labels.",
                  "Private-suffix-aware site grouping does not fully deduplicate brands, template families or campaigns. One URL per selected group is only a pilot diversity constraint.",
                  "URL screening does not resolve DNS. A later collector must validate resolved addresses and redirects; preparation is not authorization for unrestricted live execution.",
                  "Non-page suffix exclusions are URL heuristics, not checked MIME types; a misleading extension can serve different content. Every exclusion is retained for review.",
                  "OpenPhish feed-derived URL lists remain subject to its source terms; check permission requirements before redistribution. The review report intentionally contains no selected candidate URLs.",
                  "Existing PhreshPhish splits and the 106-feature baseline are unchanged; this step measures no phishing accuracy."]}
    save_json(output / REPORT_NAME, report)
    print("Selected candidate counts:", json.dumps(counts), flush=True)
    print("Confirmed ground-truth counts: benign=0, phish=0; capture and label review pending.", flush=True)
    print(f"Saved candidate list: {output / 'candidate_manifest.json'}", flush=True)
    print(f"Saved report: {output / REPORT_NAME}", flush=True)
    if not target_met:
        print("Candidate target not reached. All shortfalls and exclusions were retained; no URLs visited or model trained.", flush=True)
        return 1
    print("Candidate preparation complete. No candidate sites opened and no model trained.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Candidate preparation interrupted. Preserve any partial output and use a new --output-dir.")
    except Exception as error:
        raise SystemExit("Candidate preparation stopped: " + str(error))
