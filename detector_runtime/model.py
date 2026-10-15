"""Reuse the byte-verified training extractor and frozen classifier."""
import hashlib
import io
import importlib.util
import json
import math
import platform
import sys
import threading
from importlib.metadata import version
from pathlib import Path, PureWindowsPath

from . import VERSION

FROZEN_SPEC_SHA = "0443c025950a2fcce451c80c8d657ba2e5a56fbcb25de1e3279748e987c9e1cc"
BASE_SHA = "4c845262a0cb27c085a2fc80c4d37f427addad5405b7319ba20438f7e18ab7ef"
CONTEXT_SHA = "62d468f83d124346cce3866894f85ae4bec4cdcd3163267b23625b61764d16eb"
BASE_FILE = "09_extract_static_features.py"
CONTEXT_FILE = "15_extract_form_context.py"
FREEZE_REL = Path("reports/static_baseline_freeze_v1/static_baseline_freeze_report.json")
BASES = {"http_source", "browser_rendered", "user_supplied_unknown"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def load_verified_helper(path, expected):
    require(Path(path).is_file(), f"Missing original helper: {path}.")
    require(digest(path) == expected, f"Keep the unchanged original helper: {Path(path).name}.")
    name = "detector_" + Path(path).stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


class FeatureAdapter:
    """Exactly script 15's 106 features, with script 09's offline PSL."""

    def __init__(self, project_dir):
        root = Path(project_dir).resolve()
        self.base = load_verified_helper(root / BASE_FILE, BASE_SHA)
        self.context = load_verified_helper(root / CONTEXT_FILE, CONTEXT_SHA)
        self.names = list(self.base.FEATURE_NAMES + self.context.CONTEXT_NAMES)
        self.quality_names = list(self.base.QUALITY_COLUMNS + self.context.CONTEXT_QUALITY)
        self.parameters = {
            "max_html_characters": self.base.DEFAULT_MAX_HTML_CHARACTERS,
            "max_js_characters": self.base.DEFAULT_MAX_JS_CHARACTERS,
        }
        self.resolver = self.base.ReferenceResolver()
        require(len(self.names) == len(set(self.names)) == 106, "Unexpected feature schema.")

    def verify_contract(self, schema):
        require(schema["feature_names"] == self.names
                and schema["feature_count"] == 106
                and schema["quality_columns_not_classifier_inputs"] == self.quality_names
                and schema["parameters"] == self.parameters
                and schema["association_limits"] == self.context.LIMITS
                and schema["feature_definitions"] == {
                    **self.base.FEATURE_DEFINITIONS, **self.context.CONTEXT_DEFINITIONS}
                and schema["external_js_policy"] == "references_only_no_download"
                and schema["base_implementation_code_sha256"] == BASE_SHA
                and schema["implementation_code_sha256"] == CONTEXT_SHA,
                "Frozen feature contract differs from the original extractor.")

    def extract(self, html, page_url):
        require(isinstance(html, str) and html.strip(), "Empty HTML cannot produce a page result.")
        features, quality = self.context.extract_enhanced_static_features(
            html, page_url, self.base, resolver=self.resolver,
            **self.parameters, full_html_characters=len(html))
        require(list(features) == self.names and list(quality) == self.quality_names,
                "Extractor output order changed.")
        require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
                    for v in features.values()), "Invalid extracted features.")
        return features, quality


def _validate_bundle(bundle, spec, run_config_sha, lgb):
    require(isinstance(bundle, dict) and bundle.get("bundle_version") == "form_context_1.0"
            and bundle.get("model_name") == spec["model_name"]
            and bundle.get("feature_names") == spec["feature_names"]
            and bundle.get("label_mapping") == {"benign": 0, "phish": 1}
            and bundle.get("threshold") == spec["threshold"] == 0.5
            and bundle.get("feature_extraction_contract") == spec["feature_extraction_contract"]
            and bundle.get("base_extractor_code_sha256") == BASE_SHA
            and bundle.get("extractor_code_sha256") == CONTEXT_SHA
            and bundle.get("run_config_sha256") == run_config_sha
            and bundle.get("package_versions") == spec["training_package_versions"]
            and bundle.get("training_label_counts") == spec["development_class_counts"]["train"],
            "Model bundle differs from its frozen training provenance.")
    model = bundle["estimator"]
    require(isinstance(model, lgb.LGBMClassifier) and list(model.classes_) == [0, 1]
            and model.objective_ == "binary" and model.n_features_in_ == 106
            and model.booster_.num_feature() == 106
            and model.booster_.feature_name() == spec["feature_names"]
            and model.best_iteration_ == spec["best_iteration"] > 0
            and model.booster_.current_iteration() >= spec["best_iteration"],
            "Estimator feature order, labels or best iteration differs.")
    params = model.get_params(deep=False)
    require(all(params.get(k) == v for k, v in spec["training_hyperparameters"].items()),
            "Estimator hyperparameters differ from the frozen model.")
    return model


class FrozenDetector:
    def __init__(self, project_dir, freeze_report=None, threads=4):
        require(isinstance(threads, int) and 1 <= threads <= 16, "Use 1..16 CPU threads.")
        self.root = Path(project_dir).resolve()
        self.threads = threads
        self.lock = threading.Lock()
        freeze_path = Path(freeze_report or self.root / FREEZE_REL)
        require(freeze_path.is_file(), f"Missing frozen report: {freeze_path}.")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
        self.spec = spec = freeze["specification"]
        require(object_sha(spec) == freeze.get("specification_sha256") == FROZEN_SPEC_SHA,
                "This runtime is for the already frozen 106-feature model. The specification changed.")
        require(spec["model_input_numeric_type"] == "float32", "Unexpected numeric contract.")
        # Relocate original Windows project paths; never accept pickle uploads.
        old_root = PureWindowsPath(spec["input_files"]["trainer"]["path"]).parent
        self.refs = {}
        for key in ("base_extractor", "context_extractor", "context_schema", "run_config", "model_bundle"):
            entry = spec["input_files"][key]
            relative = PureWindowsPath(entry["path"]).relative_to(old_root)
            require(all(p not in {"..", "."} for p in relative.parts), "Invalid frozen artifact path.")
            path = self.root.joinpath(*relative.parts)
            require(path.resolve().is_relative_to(self.root) and path.is_file(), f"Missing original artifact: {path}.")
            require(digest(path) == entry["sha256"], f"Frozen artifact changed: {path.name}.")
            self.refs[key] = path
        schema = json.loads(self.refs["context_schema"].read_text(encoding="utf-8-sig"))
        require(schema == spec["feature_extraction_contract"], "Saved schema differs from the freeze.")
        wanted = {**spec["training_package_versions"], **spec["extraction_package_versions"]}
        installed = {name: version(name) for name in wanted}
        mismatches = [f"{name}: installed {installed[name]}, frozen {value}"
                      for name, value in wanted.items() if installed[name] != value]
        require(not mismatches, "Use your original .venv; dependency versions differ: " + "; ".join(mismatches))
        self.adapter = FeatureAdapter(self.root)
        self.adapter.verify_contract(schema)
        import numpy as np
        import joblib
        import lightgbm as lgb
        from threadpoolctl import threadpool_limits
        self.np, self.threadpool_limits = np, threadpool_limits
        # Deserialize the exact bytes just verified, closing a possible
        # read/replace race between hashing a path and opening it for loading.
        model_bytes = self.refs["model_bundle"].read_bytes()
        require(hashlib.sha256(model_bytes).hexdigest() == spec["model_bundle_sha256"],
                "Frozen model bytes changed before deserialization.")
        bundle = joblib.load(io.BytesIO(model_bytes))
        self.model = _validate_bundle(bundle, spec, digest(self.refs["run_config"]), lgb)
        require(digest(self.refs["model_bundle"]) == spec["model_bundle_sha256"], "Model bytes changed while loading.")
        self.environment = {"python": platform.python_version(), "packages": installed}

    def describe(self):
        return {"runtime_version": VERSION, "model_name": self.spec["model_name"],
                "model_bundle_sha256": self.spec["model_bundle_sha256"],
                "frozen_specification_sha256": FROZEN_SPEC_SHA,
                "feature_count": 106, "numeric_type": "float32", "threshold": 0.5,
                "best_iteration": self.spec["best_iteration"], "threads": self.threads,
                "environment": self.environment, "model_trained": False}

    def predict_html(self, html, page_url, capture_basis="user_supplied_unknown"):
        require(capture_basis in BASES, "Unknown capture basis.")
        from .capture import normalize_url
        page_url = normalize_url(page_url)  # Syntax only; offline HTML does not resolve DNS.
        with self.lock:
            features, quality = self.adapter.extract(html, page_url)
            x = self.np.asarray([[features[n] for n in self.adapter.names]], dtype=self.np.float32)
            require(self.np.isfinite(x).all(), "Features overflowed float32.")
            with self.threadpool_limits(limits=self.threads):
                score = float(self.model.booster_.predict(
                    x, num_iteration=self.spec["best_iteration"], raw_score=False,
                    num_threads=self.threads)[0])
        require(math.isfinite(score) and 0 <= score <= 1, "Invalid model score.")
        flags = [k for k, v in quality.items() if v]
        blocked_capture = any(quality.get(k, 0) for k in (
            "context_human_verification_hint_present", "context_unavailable_page_hint_present"))
        status = "inconclusive_capture" if blocked_capture else "prediction_complete"
        return {"status": status, "page_url": page_url, "capture_basis": capture_basis,
                "capture_basis_verified": False, "html_sha256": hashlib.sha256(html.encode()).hexdigest(),
                "html_characters": len(html), "model": self.describe(),
                "phishing_score": score, "score_is_calibrated_probability": False,
                "static_model_prediction": "phish" if score >= 0.5 else "benign",
                "display_result": "Inconclusive capture" if blocked_capture else (
                    "Potential phishing" if score >= 0.5 else "No phishing predicted"),
                "needs_review": bool(flags) or capture_basis != "browser_rendered",
                "quality_flags": quality, "active_quality_flags": flags,
                "features": features, "javascript_executed_by_detector": False,
                "external_javascript_downloaded_by_detector": False,
                "limitations": [
                    "The frozen model was tested on archived browser-rendered HTML. Live URL accuracy is unmeasured.",
                    "Raw HTTP source may omit forms or content created by JavaScript.",
                    "No phishing predicted is a model result, not a guarantee that a website is safe.",
                    "Dynamic lab signals are separate observations and are not classifier inputs."]}
