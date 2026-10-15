"""Public HTTP API for the frozen C2 static detector (group website / hosting).

Wraps the unchanged FrozenDetector and SSRF-guarded capture. Browser mode is
not exposed: it executes untrusted page JavaScript. Run locally with:
    .\\.venv\\Scripts\\python.exe -m uvicorn api_server:app --port 7860
"""
import hmac
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from detector_runtime import VERSION
from detector_runtime.capture import CaptureError
from detector_runtime.model import FrozenDetector
from detector_runtime.reports import analyze_url

COMPONENT = "C2_html_js"
THRESHOLD = 0.5
MAX_HTML_BYTES = 5 * 1024 * 1024        # public limit; the CLI allows 32 MiB
MAX_REQUEST_BYTES = MAX_HTML_BYTES + 64 * 1024
API_KEY = os.environ.get("C2_API_KEY", "")
ORIGINS = [o.strip() for o in os.environ.get("C2_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
THREADS = int(os.environ.get("C2_THREADS", "2"))

state = {"detector": None, "error": None}
gate = threading.BoundedSemaphore(2)


@asynccontextmanager
async def lifespan(app):
    try:
        state["detector"] = FrozenDetector(Path(__file__).resolve().parent, threads=THREADS)
    except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        state["error"] = str(e)
        print(f"Frozen detector failed to load: {e}")
    yield


app = FastAPI(title="R26-CS-019 C2 HTML/JS Phishing Detector API", version=VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type", "X-API-Key"])


@app.middleware("http")
async def limit_body(request: Request, call_next):
    if request.method == "POST":
        length = request.headers.get("content-length")
        if length is None or not length.isdigit():
            return JSONResponse({"detail": "Send a JSON body with a Content-Length."}, 411)
        if int(length) > MAX_REQUEST_BYTES:
            return JSONResponse({"detail": "Request exceeds 5 MiB."}, 413)
    return await call_next(request)


class AnalyzeRequest(BaseModel):
    url: str | None = None
    html: str | None = None
    page_url: str | None = None


def check_key(supplied):
    if API_KEY and not hmac.compare_digest(supplied or "", API_KEY):
        raise HTTPException(401, "Missing or invalid X-API-Key.")


def contract(status, result=None, message=None):
    """Group result contract: label is null whenever status is not ok."""
    ok = status == "ok"
    score = result.get("phishing_score") if result else None
    return {
        "component": COMPONENT,
        "component_version": VERSION,
        "status": status,
        "label": ("phishing" if score >= THRESHOLD else "legitimate") if ok else None,
        "score": score if ok else None,
        "threshold": THRESHOLD,
        "score_is_calibrated_probability": False,
        "message": message or (result or {}).get("display_result"),
        "evidence": {
            "active_quality_flags": (result or {}).get("active_quality_flags", []),
            "needs_review": (result or {}).get("needs_review"),
            "capture": (result or {}).get("capture"),
            "limitations": (result or {}).get("limitations", []),
        },
        "details": result,
    }


@app.get("/")
def root():
    return {"component": COMPONENT, "health": "/health", "analyze": "POST /analyze", "docs": "/docs"}


@app.get("/health")
def health():
    detector = state["detector"]
    return {"component": COMPONENT, "status": "online", "ready": detector is not None,
            "model": detector.describe() if detector else None,
            "error": None if detector else "Frozen detector is not loaded."}


@app.post("/analyze")
def analyze(body: AnalyzeRequest, x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    detector = state["detector"]
    if detector is None:
        raise HTTPException(503, "Frozen detector is not loaded.")
    if (body.url is None) == (body.html is None):
        return contract("invalid_input", message="Send either 'url' or 'html' (with 'page_url').")
    if body.html is not None and (not body.page_url or len(body.html.encode()) > MAX_HTML_BYTES):
        return contract("invalid_input", message="'html' needs 'page_url' and must be at most 5 MiB.")
    if not gate.acquire(blocking=False):
        raise HTTPException(429, "Two analyses are already running. Try again shortly.")
    try:
        if body.url is not None:
            result = analyze_url(detector, body.url)
        else:
            result = detector.predict_html(body.html, body.page_url, "user_supplied_unknown")
            result["capture"] = {"input": "pasted_html", "capture_complete": "not_independently_verified"}
    except CaptureError as e:
        return contract("invalid_input", message=str(e))
    except (OSError, ValueError, KeyError, TypeError):
        return contract("error", message="Analysis failed.")
    finally:
        gate.release()
    status = {"prediction_complete": "ok", "inconclusive_capture": "inconclusive",
              "capture_failed": "capture_failed"}.get(result.get("status"), "error")
    return contract(status, result, result.get("error"))
