"""Separate per-analysis reports; frozen reports are never overwritten."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def save_report(project_dir, value, category="webpage_analysis_v1"):
    if category not in {"webpage_analysis_v1", "dynamic_runtime_v1", "browser_pipeline_v1"}:
        raise ValueError("Unexpected report category.")
    directory = Path(project_dir).resolve() / "reports" / category
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    path = directory / (now.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8] + ".json")
    value = {"created_utc": now.isoformat(), **value}
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False, ensure_ascii=True)
        f.write("\n")
    return path


def analyze_url(detector, url):
    from .capture import CaptureError, capture_url
    try:
        capture = capture_url(url)
        result = detector.predict_html(capture.html, capture.metadata["final_url"], "http_source")
        result["capture"] = capture.metadata
        result["capture_basis_verified"] = True
        return result
    except CaptureError as e:
        return {"status": "capture_failed", "display_result": "Inconclusive capture",
                "error": str(e), "model_prediction_made": False,
                "phishing_score": None, "static_model_prediction": None}
