import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(root=None):
    root = Path(root) if root else Path(__file__).parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def environment_info():
    result = {"python": sys.version, "platform": platform.platform(), "processor": platform.processor()}
    for name in ("torch", "numpy", "scipy", "pandas", "matplotlib"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result
