"""habitat-radiomics-pca-lnm — shared helpers.

Everything the pipeline needs from the surrounding filesystem lives here, so
that a single ``config/config.yaml`` controls every path.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import yaml

__all__ = [
    "REPO_ROOT",
    "load_config",
    "paths",
    "repo_path",
    "ensure_dir",
    "write_json",
    "read_json",
    "sha256",
    "manifest",
    "log",
]

# __file__ = <repo>/src/hrpca/__init__.py  ->  parents[2] = <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}([^\${]*)")


def _expand_env(value):
    """Expand ``${VAR}`` from the environment (unset -> empty string)."""
    if isinstance(value, str):
        def _repl(m):
            v = os.environ.get(m.group(1), "")
            tail = m.group(2)
            if not v:
                return tail.lstrip("/\\")
            return v.rstrip("/\\") + tail
        return _ENV_RE.sub(_repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def load_config(config: str | os.PathLike | None = None) -> dict:
    """Load ``config/config.yaml`` (or an alternative file)."""
    if config is None:
        config = REPO_ROOT / "config" / "config.yaml"
    with open(config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg = _expand_env(cfg)
    cfg["_config_path"] = str(Path(config).resolve())
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def repo_path(cfg: dict, key: str) -> Path:
    """Resolve a *relative* config key (``data_dir``, ``results_dir``)."""
    return (REPO_ROOT / cfg[key]).resolve()


def paths(cfg: dict) -> dict:
    """Return the standard input/output directories, creating the outputs."""
    out = {
        "data": repo_path(cfg, "data_dir"),
        "results": repo_path(cfg, "results_dir"),
    }
    out["data_scores"] = out["data"] / "scores"
    out["data_features"] = out["data"] / "features"
    out["results_tables"] = out["results"] / "tables"
    out["results_figures"] = out["results"] / "figures"
    out["results_logs"] = out["results"] / "logs"
    for key in ("results_tables", "results_figures", "results_logs", "data_scores"):
        ensure_dir(out[key])
    return out


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# tiny IO helpers
# --------------------------------------------------------------------------
def write_json(obj, path: str | os.PathLike, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False, default=_default)


def read_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _default(obj):
    """Make numpy scalars/arrays JSON-serialisable."""
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:  # pragma: no cover
        pass
    return str(obj)


def sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def manifest(files, base: str | os.PathLike | None = None) -> list:
    """``[{path, bytes, sha256}]`` for every existing file in *files*."""
    base = Path(base) if base is not None else None
    rows = []
    for f in files:
        f = Path(f)
        if not f.exists():
            continue
        rel = str(f.relative_to(base)) if base and str(f).startswith(str(base)) else str(f)
        rows.append({"path": rel, "bytes": f.stat().st_size, "sha256": sha256(f)})
    return sorted(rows, key=lambda r: r["path"])


def log(msg: str) -> None:
    print(msg, flush=True)


def banner(title: str, char: str = "=") -> None:
    line = char * max(len(title) + 4, 72)
    print(f"\n{line}\n  {title}\n{line}", flush=True)
