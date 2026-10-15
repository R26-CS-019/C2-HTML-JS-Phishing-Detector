"""Verify browser + original model + decision on trusted offline pages."""
import argparse
import json
from pathlib import Path

from detector_runtime.model import FrozenDetector
from detector_runtime.behavior_examples import examples, analyze_demo
from detector_runtime.reports import save_report


def checks_for(name,result):
    complete=name not in {"private_resource","missing_script"}
    checks={"expected_completion":result["status"]==("prediction_complete" if complete else "inconclusive_capture")}
    if result.get("behavior"):
        s=result["behavior"]["signals"]
        checks.update({"original_106_features":len(result["features"])==106,
                       "original_model_used":result["model"]["model_bundle_sha256"]=="944e9bcfcb884ab167beaf3b72b992cf4799aee727e31e8afbde1b1b2d4b4ca1",
                       "no_test_requests_forwarded":not result["capture"]["interaction_requests_forwarded"],
                       "no_live_page_requests":result["capture"]["network_mode"]=="offline_replay",
                       "decision_matches_rendered_score":result["static_model_prediction"]==("phish" if result["phishing_score"]>=.5 else "benign")})
        if name in {"delayed_login","encoded_login","external_login"}:
            checks.update({"source_form_absent":result["source_analysis"]["features"]["form_count"]==0,
                           "rendered_form_found":result["features"]["form_count"]==1,
                           "delayed_credential_observed":s["forms_added_after_ready"]>=1 and s["maximum_passwordlike_controls"]>=1})
        if name=="encoded_login":
            checks["encoding_executed"]=s["base64_decode_calls"]>=1
        if name=="external_login":
            checks["external_script_executed"]=s["external_scripts_loaded"]>=1
        if name in {"ordinary_sso","fetch_login"}:
            checks["credential_attempt_blocked"]=s["test_password_request_attempts"]>=1
            checks["cross_origin_not_automatic_label"]=all(r["effect"]=="review_evidence" for r in result["behavior"]["rules"])
        if name=="contact_form":
            checks["contact_has_no_password"]=s["maximum_passwordlike_controls"]==0
        if name=="redirect_login":
            checks["redirect_observed"]=s["main_frame_navigations"]>=2 and result["page_url"].endswith('/destination')
    else:
        checks["browser_observation_present"]=False
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--output-file",type=Path)
    args=parser.parse_args()
    detector=FrozenDetector(args.project_dir)
    cases=[]
    for name in examples():
        result=analyze_demo(detector,name)
        checks=checks_for(name,result)
        passed=all(checks.values())
        cases.append({"case":name,"passed":passed,"checks":checks,"analysis":result})
        print(f"{name}: {'PASS' if passed else 'FAIL'}",flush=True)
        if result.get('error'):
            print(result['error'].encode('ascii','backslashreplace').decode(),flush=True)
            break
    passed=sum(c["passed"] for c in cases)
    report={"status":"integrated_checks_passed" if passed==len(examples()) else "integrated_checks_failed",
            "expected_cases":len(examples()),"executed_cases":len(cases),"passed_cases":passed,
            "accuracy_measured":False,"model_retrained":False,"cases":cases}
    if args.output_file:
        args.output_file.parent.mkdir(parents=True,exist_ok=True)
        args.output_file.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        path=args.output_file
    else:
        path=save_report(args.project_dir,report,"browser_pipeline_v1")
    print(f"Integrated checks: {passed}/{len(examples())}. Saved: {path}")
    print("These checks verify the connected system; they do not measure phishing accuracy.")
    return 0 if passed==len(examples()) else 1


if __name__=="__main__":
    raise SystemExit(main())
