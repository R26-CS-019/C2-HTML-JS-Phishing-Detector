"""Short-lived browser worker. JSON stdin/stdout; no fitted model loaded here."""
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .browser_network import Resource, fetch_public_resource, fingerprint
from .capture import CaptureError, normalize_url, decode_html
from .controlled_fixtures import enrich_snapshot

USER = "RESEARCH_BEHAVIOR_USER@example.test"
PASSWORD = "RESEARCH_BEHAVIOR_PASSWORD"
MAX_DOM_BYTES = 8 * 1024 * 1024
MAX_ROUTES = 80
MAX_EVENTS = 300
MAX_TOTAL_BYTES = 32 * 1024 * 1024


def origin(url):
    p = urlsplit(url)
    return p.scheme.lower(), (p.hostname or "").lower(), p.port or (443 if p.scheme == "https" else 80)


def clean_url(url):
    # Do not retain values from any query, fragment or userinfo in behavior logs.
    try:
        p = urlsplit(url)
        authority = p.hostname or ""
        if ":" in authority:
            authority = "[" + authority + "]"
        if p.port:
            authority += ":" + str(p.port)
        path = p.path.replace(USER, "[test-user]").replace(PASSWORD, "[test-password]")
        return (p.scheme + "://" + authority + path + ("?[redacted]" if p.query else ""))[:8192]
    except ValueError:
        return "[invalid URL]"


def run_capture(payload):
    from importlib.metadata import version
    from playwright.sync_api import sync_playwright, Error as BrowserError
    url = normalize_url(payload["url"])
    observe_ms = payload.get("observe_ms", 6000)
    action_ms = payload.get("action_ms", 1500)
    if not isinstance(observe_ms, int) or not 500 <= observe_ms <= 15000:
        raise ValueError("Observation time must be 500..15000 ms.")
    if not isinstance(action_ms, int) or not 500 <= action_ms <= 5000:
        raise ValueError("Interaction time must be 500..5000 ms.")
    # An offline manifest enables reproducible built-in demos and archived captures.
    # Missing resources always fail closed; no fallback to the live network.
    manifest = payload.get("resources")
    offline = manifest is not None
    if not offline and tuple(int(n) for n in version("playwright").split('.')[:2]) < (1,63):
        raise ValueError("Install requirements_behavior.txt and its Chromium browser before live browser capture.")
    if offline and (not isinstance(manifest, dict) or len(manifest) > MAX_ROUTES):
        raise ValueError("Invalid offline resource manifest.")
    events, routes, navs, errors, critical = [], [], [], [], []
    state = {"interaction":False, "bytes":0, "route_count":0, "events_dropped":0,
             "deadline":time.monotonic()+55, "main_source":None, "redirects":0}
    started = time.monotonic()

    with sync_playwright() as p:
        # Explicitly enable Chromium's renderer sandbox (Playwright defaults off).
        browser = p.chromium.launch(headless=True, chromium_sandbox=True,
            timeout=15000, args=["--disable-background-networking", "--disable-component-update",
            "--disable-quic", "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
            "--host-resolver-rules=MAP * ~NOTFOUND", "--proxy-server=http://127.0.0.1:9",
            "--proxy-bypass-list=<-loopback>"])
        context = browser.new_context(service_workers="block", accept_downloads=False,
                                      permissions=[], viewport={"width":1280,"height":800})
        context.set_default_timeout(1800)
        page = context.new_page()

        def record_event(source, raw):
            if len(events) >= MAX_EVENTS or not isinstance(raw,str) or len(raw)>60000:
                state["events_dropped"] += 1
                return
            try:
                event = json.loads(raw)
                if not isinstance(event,dict) or not isinstance(event.get("kind"),str):
                    raise ValueError("Invalid event")
                event["main_frame"] = source["frame"] == page.main_frame
                event["phase"] = "interaction" if state["interaction"] else "passive"
                event["elapsed_ms"] = round((time.monotonic()-started)*1000)
                if "snapshot" in event:
                    event["snapshot"] = enrich_snapshot(event["snapshot"])
                events.append(event)
            except (ValueError,KeyError,TypeError):
                state["events_dropped"] += 1

        def handle(route):
            request = route.request
            state["route_count"] += 1
            phase = "interaction" if state["interaction"] else "passive"
            try:
                main = request.is_navigation_request() and request.frame == page.main_frame
            except BrowserError:
                main = False
            body = request.post_data or ""
            probe = unquote(request.url + " " + body)
            row = {"url":clean_url(request.url), "method":request.method,
                   "resource_type":request.resource_type, "main_navigation":main,
                   "phase":phase, "test_password_present":PASSWORD in probe,
                   "test_identity_present":USER in probe,
                   "cross_origin":origin(request.url)!=origin(page.url if page.url.startswith('http') else url),
                   "elapsed_ms":round((time.monotonic()-started)*1000)}
            try:
                if len(routes)>=MAX_ROUTES or time.monotonic()>state["deadline"]:
                    raise CaptureError("Resource/time budget reached.")
                if state["interaction"]:
                    row["action"]="blocked_during_test"
                    route.abort()
                    return
                if request.method != "GET":
                    raise CaptureError("Non-GET requests are blocked.")
                if request.resource_type not in {"document","script","stylesheet","fetch","xhr"}:
                    row["action"]="blocked_optional_resource"
                    route.abort()
                    return
                target = normalize_url(request.url)
                if offline:
                    item = manifest.get(target)
                    if item is None:
                        raise CaptureError("Resource absent from offline capture.")
                    resource = Resource(target,item.get("status",200),item["headers"],item.get("body","").encode())
                else:
                    resource = fetch_public_resource(target, timeout=min(8,max(.1,state["deadline"]-time.monotonic())))
                if len(resource.body)>MAX_DOM_BYTES or state["bytes"]+len(resource.body)>MAX_TOTAL_BYTES:
                    raise CaptureError("Resource byte budget reached.")
                state["bytes"] += len(resource.body)
                if resource.status in {301,302,303,307,308}:
                    state["redirects"] += 1
                    if state["redirects"]>8:
                        raise CaptureError("Redirect budget reached.")
                if main and resource.status==200:
                    content_type=resource.headers.get("content-type","")
                    if content_type.split(';')[0].strip().lower() not in {"text/html","application/xhtml+xml"}:
                        raise CaptureError("Main document is not HTML.")
                    html, decoding=decode_html(resource.body,content_type)
                    if state["main_source"] is None:
                        state["main_source"]={"html":html,"page_url":target,"decoding":decoding,
                                              "response_body_sha256":hashlib.sha256(resource.body).hexdigest()}
                row.update(fingerprint(resource))
                row["url"] = clean_url(target)
                row["action"]="fulfilled_offline" if offline else "fulfilled_public_get"
                route.fulfill(status=resource.status,headers=resource.headers,body=resource.body)
            except (CaptureError, OSError, BrowserError, ValueError, KeyError) as e:
                row["action"]="blocked_or_failed"
                row["error"]=str(e)[:300]
                if phase=="passive" and request.resource_type in {"document","script","stylesheet","fetch","xhr"}:
                    if len(critical)<MAX_ROUTES:
                        critical.append({"url":row["url"],"type":request.resource_type,"error":row["error"]})
                try:
                    route.abort()
                except BrowserError:
                    pass
            finally:
                if len(routes)<MAX_ROUTES:
                    routes.append(row)

        context.expose_binding("__behaviorRecord",record_event)
        context.add_init_script(path=str(Path(__file__).with_name("behavior_probe.js")))
        context.route("**/*",handle)
        context.route_web_socket("**/*",lambda ws:ws.close())
        context.on("page",lambda other:other.close() if other!=page else None)
        page.on("dialog",lambda dialog:dialog.dismiss())
        page.on("pageerror",lambda error:errors.append(str(error)[:300]) if len(errors)<20 else None)
        page.on("framenavigated",lambda frame:navs.append({"url":clean_url(frame.url),
            "phase":"interaction" if state["interaction"] else "passive"}) if frame==page.main_frame and len(navs)<30 else None)
        try:
            page.goto(url,wait_until="domcontentloaded",timeout=20000)
            # Timers, external scripts and JS navigation continue during this window.
            page.wait_for_timeout(observe_ms)
            if not state["main_source"]:
                raise CaptureError("No complete main HTML document arrived.")
            passive_url = normalize_url(page.url)
            passive_html = page.content()
            if len(passive_html.encode())>MAX_DOM_BYTES:
                raise CaptureError("Rendered DOM exceeds the byte limit.")
            snapshot=enrich_snapshot(page.evaluate("window.__behaviorProbe.snapshot()"))
            # All requests, including GET and navigation, are aborted from here on.
            state["interaction"]=True
            interactions=[]
            fields=page.locator("input,textarea")
            for field in snapshot["fields"]:
                if (field["passwordlike"] or field["identitylike"]) and field["rendered_visible"] and not field["disabled"]:
                    try:
                        fields.nth(field["dom_index"]).fill(PASSWORD if field["passwordlike"] else USER)
                        interactions.append({"kind":"fill_test_value","field_index":field["dom_index"]})
                    except BrowserError:
                        interactions.append({"kind":"fill_unavailable","field_index":field["dom_index"]})
                    if len(interactions)>=10:
                        break
            form_indices=sorted(set(f["form_index"] for f in snapshot["fields"] if f["passwordlike"] and f["form_index"] is not None))[:3]
            for index in form_indices:
                try:
                    form=page.locator("form").nth(index)
                    button=form.locator("button[type=submit],button:not([type]),input[type=submit]").first
                    if button.count() and button.is_visible():
                        button.click(timeout=1200,no_wait_after=True)
                    else:
                        form.evaluate("f=>f.requestSubmit()")
                    interactions.append({"kind":"test_submit","form_index":index})
                except BrowserError:
                    interactions.append({"kind":"submit_unavailable","form_index":index})
            page.wait_for_timeout(action_ms)
            interaction_page_unavailable=False
            try:
                page.evaluate("window.__behaviorProbe.flush()")
                after=enrich_snapshot(page.evaluate("window.__behaviorProbe.snapshot()"))
            except BrowserError:
                # Aborted form navigation may replace the document with a browser
                # error page. The passive DOM was already saved and is still valid.
                after=None
                interaction_page_unavailable=True
            counts=Counter(e["kind"] for e in events)
            passive_events=[e for e in events if e.get("phase")=="passive" and e.get("main_frame")]
            snapshots=[snapshot]+[e["snapshot"] for e in passive_events if "snapshot" in e]
            initial=next((e["snapshot"] for e in passive_events if e["kind"]=="dom_ready"),snapshot)
            signals={"initial_passwordlike_controls":initial["passwordlike_count"],
                     "passive_passwordlike_controls":snapshot["passwordlike_count"],
                     "maximum_passwordlike_controls":max(s["passwordlike_count"] for s in snapshots),
                     "forms_added_after_ready":sum(e.get("forms_added",0) for e in passive_events),
                     "controls_added_after_ready":sum(e.get("controls_added",0) for e in passive_events),
                     "main_frame_navigations":sum(n["phase"]=="passive" for n in navs),
                     "base64_decode_calls":counts["base64_decode_call"],
                     "fetch_calls":counts["fetch_call"],"xhr_calls":counts["xhr_call"],
                     "beacon_calls":counts["beacon_call"],"timer_callbacks":counts["timer_fired"],
                     "submit_events":counts["submit_event"]+counts["direct_submit"],
                     "test_password_request_attempts":sum(r["test_password_present"] for r in routes),
                     "cross_origin_test_password_attempts":sum(r["test_password_present"] and r["cross_origin"] for r in routes),
                     "external_scripts_loaded":sum(r["resource_type"]=="script" and r["action"].startswith('fulfilled') for r in routes)}
            quality={"critical_resources_failed":bool(critical),"event_storage_cut":state["events_dropped"]>0 or any(s.get("trace_events_dropped") for s in snapshots),
                     "request_storage_cut":state["route_count"]>MAX_ROUTES,
                     "field_storage_cut":any(s["storage_cut"] for s in snapshots),
                     "page_script_errors":bool(errors),"subframes_present":len(page.frames)>1,
                     "interaction_document_unavailable":interaction_page_unavailable}
            return {"status":"browser_capture_complete", "source":state["main_source"],
                    "rendered_html":passive_html,"page_url":passive_url,"signals":signals,
                    "snapshots":{"initial":initial,"passive":snapshot,"after_test":after},
                    "quality":quality,"events":events,"requests":routes,"navigations":navs,
                    "page_errors":errors,"critical_resource_failures":critical,"interactions":interactions,
                    "capture":{"capture_basis":"browser_rendered","javascript_executed":True,
                        "network_mode":"offline_replay" if offline else "public_get_gateway",
                        "chromium_sandbox_requested":True,"chromium_version":browser.version,
                        "playwright_version":version("playwright"),"observe_ms":observe_ms,"action_ms":action_ms,
                        "requests":state["route_count"],"resource_bytes":state["bytes"],
                        "cookies_forwarded":False,"interaction_requests_forwarded":False,
                        "elapsed_seconds":round(time.monotonic()-started,3)}}
        finally:
            context.close()
            browser.close()


def main():
    try:
        raw=sys.stdin.buffer.read(16*1024*1024+1)
        if len(raw)>16*1024*1024:
            raise ValueError("Capture input exceeds limit.")
        value=run_capture(json.loads(raw))
    except Exception as e:
        value={"status":"browser_capture_failed","error":f"{type(e).__name__}: {e}"[:6000]}
    encoded=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
    if len(encoded)>32*1024*1024:
        encoded=b'{"status":"browser_capture_failed","error":"Browser observation output exceeded 32 MiB."}'
    sys.stdout.buffer.write(encoded)


if __name__=="__main__":
    main()
