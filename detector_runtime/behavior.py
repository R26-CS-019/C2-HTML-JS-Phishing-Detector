"""Connect browser observations to the frozen rendered-HTML classifier."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

VERSION = "browser_pipeline_1.0"
LIMITATIONS = [
    "Finite observation windows and selected test interactions can miss behavior.",
    "Cookies, POST/API writes, images, fonts, media, WebSockets and service workers are blocked; some pages cannot render fully.",
    "JavaScript hooks are observations that a page can evade or tamper with; they are not proof of malicious intent.",
    "The frozen model scores the passive rendered main-frame HTML; browser signals are separate evidence, not extra ML inputs.",
    "Credential forms, encoding and cross-origin login occur on legitimate pages too.",
    "Accuracy of this integrated browser pipeline requires a new independently labelled evaluation set.",
    "A Chromium sandbox and routing policy are not a hardened malware-analysis VM. Use an isolated disposable VM for suspected phishing pages."]


def capture_in_worker(payload, timeout=75):
    raw=json.dumps(payload,ensure_ascii=True,allow_nan=False).encode()
    if len(raw)>16*1024*1024:
        raise ValueError("Browser capture input exceeds 16 MiB.")
    env=os.environ.copy()
    env["PYTHONPATH"]=os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p and Path(p).is_dir())
    env["PYTHONDONTWRITEBYTECODE"]="1"
    # Files bound stderr/output reads without retaining captured HTML on disk.
    with tempfile.TemporaryFile() as stderr:
        process=subprocess.Popen([sys.executable,"-B","-m","detector_runtime.behavior_worker"],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0,
            start_new_session=os.name!="nt")
        job=None
        if os.name=="nt":
            try:
                from .worker_job import WindowsJob
                job=WindowsJob(process)
            except OSError as error:
                process.kill();process.communicate(timeout=10)
                return {"status":"browser_capture_failed","error":"Could not contain the browser worker: "+str(error)}
        try:
            try:
                output,_=process.communicate(raw,timeout=timeout)
            except subprocess.TimeoutExpired:
                if job:
                    job.close()
                else:
                    import signal
                    os.killpg(process.pid,signal.SIGKILL)
                process.kill()
                process.communicate(timeout=10)
                return {"status":"browser_capture_failed","error":"Browser analysis reached its time limit."}
        finally:
            if job:
                job.close()
        if len(output)>32*1024*1024:
            return {"status":"browser_capture_failed","error":"Browser report exceeded the output limit."}
        try:
            value=json.loads(output)
            if not isinstance(value,dict):
                raise ValueError("Invalid worker report")
            return value
        except (ValueError,UnicodeError):
            stderr.seek(0)
            detail=stderr.read(1000).decode(errors="replace")
            return {"status":"browser_capture_failed","error":"Browser worker stopped without a valid report. "+detail}


def rules_for(capture):
    """Evidence rules request review; they never override the classifier label."""
    s=capture["signals"]
    rules=[]
    def add(code,text):
        rules.append({"code":code,"finding":text,"effect":"review_evidence"})
    if s["forms_added_after_ready"] and s["maximum_passwordlike_controls"]:
        add("credential_form_created_after_load","A credential form appeared after the page loaded.")
    if s["main_frame_navigations"]>1:
        add("page_navigation","The page changed its main address during observation.")
    if s["test_password_request_attempts"]:
        add("test_password_in_request","A request attempted to include the test password. The request was blocked.")
    if s["cross_origin_test_password_attempts"]:
        add("cross_origin_test_password","The test password was directed to another origin. Ordinary login services can do this too.")
    if s["base64_decode_calls"]:
        add("encoded_script_executed","A script decoded Base64 content while running.")
    if capture["quality"].get("subframes_present"):
        add("subframe_scope","Embedded frames were present; their HTML is outside the main-frame classifier input.")
    return rules


def combine(detector,capture):
    if capture.get("status")!="browser_capture_complete":
        return {"status":"browser_capture_failed","display_result":"Browser analysis could not complete",
            "analysis_pipeline":VERSION,"model_prediction_made":False,"phishing_score":None,
            "static_model_prediction":None,"needs_review":True,
            "error":capture.get("error","Browser capture failed."),"limitations":LIMITATIONS}
    source=capture["source"]
    source_result=detector.predict_html(source["html"],source["page_url"],"http_source")
    result=detector.predict_html(capture["rendered_html"],capture["page_url"],"browser_rendered")
    result["capture_basis_verified"]=True
    result["javascript_executed_by_detector"]=True
    result["external_javascript_downloaded_by_detector"]=capture["signals"]["external_scripts_loaded"]>0
    result["analysis_pipeline"]=VERSION
    result["decision_basis"]="frozen_model_on_passive_rendered_main_frame"
    result["model_prediction_made"]=True
    result["source_analysis"]=source_result
    result["feature_changes"]={k:{"source":source_result["features"][k],"rendered":v}
        for k,v in result["features"].items() if source_result["features"][k]!=v}
    rules=rules_for(capture)
    result["behavior"]={k:v for k,v in capture.items() if k not in {"source","rendered_html","page_url","status","capture"}}
    result["behavior"]["rules"]=rules
    result["capture"]=capture["capture"]
    result["capture"]["capture_complete"]=not any(capture["quality"].get(k) for k in (
        "critical_resources_failed","event_storage_cut","request_storage_cut","field_storage_cut"))
    result["limits_used"]={"per_resource_bytes":8*1024*1024,"total_resource_bytes":32*1024*1024,
                          "routes":80,"events":300,"worker_timeout_seconds":75}
    result["needs_review"]=result["needs_review"] or bool(rules) or any(capture["quality"].values())
    if not result["capture"]["capture_complete"]:
        # Preserve the diagnostic score/label, but abstain from an evaluated result.
        result["status"]="inconclusive_capture"
        result["display_result"]="Incomplete browser analysis — needs review"
    result["evaluated_prediction"]=result["static_model_prediction"] if result["status"]=="prediction_complete" else None
    result["limitations"]=LIMITATIONS
    return result


def analyze_browser_url(detector,url,observe_ms=6000,action_ms=1500):
    from .capture import normalize_url
    normalized=normalize_url(url)
    return combine(detector,capture_in_worker({"url":normalized,"observe_ms":observe_ms,"action_ms":action_ms}))


def analyze_browser_capture(detector,url,resources,observe_ms=6000,action_ms=1500):
    """Backend integration: replay a caller-provided complete resource manifest."""
    from .capture import normalize_url
    return combine(detector,capture_in_worker({"url":normalize_url(url),"resources":resources,
                                             "observe_ms":observe_ms,"action_ms":action_ms}))
