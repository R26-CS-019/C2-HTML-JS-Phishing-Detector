"""Verify and reuse the existing runtime. This add-on contains no classifier."""
import importlib
import sys
from pathlib import Path

from .protocol import file_sha, require

RUNTIME_FILES = {
    "detector_runtime/__init__.py": "270f10bcc58e052931a242211fbbaa32a53b70c06dc80ecaa794c48adc730019",
    "detector_runtime/capture.py": "30d8ced35a97b5e8309626d04cb10cd5ea5b3581f167ebbce13aa85095576b2c",
    "detector_runtime/model.py": "f592ca2809136c89d33a5f9b88c872a384d8f49b6dd18e442fdd80fc9030fe8e",
}


def import_runtime(project_dir):
    root = Path(project_dir).resolve()
    for name, expected in RUNTIME_FILES.items():
        path = root / name
        require(path.is_file(), f"Missing {name}. Extract this add-on beside scripts 29–31 and the existing detector_runtime folder.")
        require(file_sha(path) == expected, f"The existing runtime file changed: {name}. Use the original runtime v1 package.")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    capture = importlib.import_module("detector_runtime.capture")
    require(Path(capture.__file__).resolve() == root / "detector_runtime/capture.py", "A different detector_runtime package was imported.")
    return capture


def load_detector(project_dir, threads):
    model = importlib.import_module("detector_runtime.model")
    require(Path(model.__file__).resolve() == Path(project_dir).resolve() / "detector_runtime/model.py",
            "A different detector model module was imported.")
    detector = model.FrozenDetector(project_dir, threads=threads)
    described = detector.describe()
    require(described["model_name"] == "lightgbm_context_all_features"
            and described["feature_count"] == 106 and described["threshold"] == 0.5
            and described["best_iteration"] == 1269 and described["model_trained"] is False,
            "The frozen static model contract differs.")
    return detector
