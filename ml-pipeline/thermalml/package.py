"""Model packages for the Pi (Part III 4.3.1.2 and 4.3.3.1 C).

Each package folder holds the model file and a metadata.json with: ordered feature
list, scaler parameters, window length, sampling period, horizon and variant
(predictors), version string, training run IDs, split-manifest hash, library
versions, SHA-256 of every shipped file, and validation / test metrics.
models/model_metadata.json indexes them.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
from pathlib import Path


def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "sklearn", "xgboost", "torch", "onnx", "onnxruntime"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            pass
    return out


def write_package(models_dir: Path, name: str, meta: dict) -> Path:
    folder = Path(models_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta.setdefault("version", f"{name}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    meta["library_versions"] = library_versions()
    # pin the shipped files: the Pi-side parity check compares these hashes first
    meta["file_sha256"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in sorted(folder.iterdir()) if f.is_file() and f.name != "metadata.json"}
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    index_path = Path(models_dir) / "model_metadata.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"packages": {}}
    index["packages"][name] = {"folder": name, "version": meta["version"], "kind": meta.get("kind"),
                               "model": meta.get("model"), "variant": meta.get("variant"),
                               "horizon_s": meta.get("horizon_s"), "window_s": meta.get("window_s")}
    index_path.write_text(json.dumps(index, indent=2, default=float))
    return folder


def read_package(folder: Path) -> dict:
    return json.loads((Path(folder) / "metadata.json").read_text())
