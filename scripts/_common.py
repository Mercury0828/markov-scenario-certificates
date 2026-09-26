"""Paths and small helpers shared by the scripts."""
import json
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def save_json(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    return path


def load_json(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def write_text(name, text):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def environment():
    import matplotlib
    import numpy
    import scipy
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "platform": platform.platform()}
