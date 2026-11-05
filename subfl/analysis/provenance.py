"""Run provenance (code, environment, hardware), written when a run starts, and the run status file."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_GLOBS = ("subfl/**/*.py", "baselines/*.py", "*.py", "*.sh")
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS")
PACKAGES = ("torch", "torchvision", "numpy", "pandas", "scipy")  # scipy: torchvision's SVHN reads the .mat files
STATUS_FILE = "run_status.json"
PROVENANCE_FILE = "provenance.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, record: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)  # atomic


def git_info(project_root: Path = PROJECT_ROOT) -> dict:
    """{"commit", "dirty", "source"} from ``git`` (source "git"), or None values with source "unavailable"."""

    try:
        head = subprocess.run(["git", "-C", str(project_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        status = subprocess.run(["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "source": "unavailable"}
    if head.returncode != 0 or not head.stdout.strip():
        return {"commit": None, "dirty": None, "source": "unavailable"}
    return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "source": "git"}


def code_files(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    """{relative path: sha256} of the code files (subfl/, baselines/, the top-level scripts), sorted, caches excluded."""

    files = sorted({p for pattern in CODE_GLOBS for p in Path(project_root).glob(pattern)
                    if p.is_file() and "__pycache__" not in p.parts})
    return {p.relative_to(project_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def code_sha256(project_root: Path = PROJECT_ROOT) -> tuple[str, int]:
    """(sha256 over the sorted (path, file sha256) pairs of the code files, number of files)."""

    files = code_files(project_root)
    digest = hashlib.sha256()
    for rel, sha in files.items():
        digest.update(rel.encode() + b"\0" + sha.encode() + b"\0")
    return digest.hexdigest(), len(files)


def _version(package: str):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment() -> dict:
    cuda = torch.cuda.is_available()
    try:
        cudnn = torch.backends.cudnn.version() if cuda else None
    except Exception:  # noqa: BLE001 - a missing cuDNN must not stop the run
        cudnn = None
    return {
        "python": platform.python_version(),
        "packages": {name: _version(name) for name in PACKAGES},
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda,
        "cudnn": cudnn,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
        "torch_threads": torch.get_num_threads(),
        "env": {key: os.environ.get(key) for key in ENV_KEYS},
        "hostname": platform.node(),
        "platform": platform.platform(),
    }


def write_provenance(run_dir: Path, *, arm: str, experiment_name: str, project_root: Path = PROJECT_ROOT) -> dict:
    sha, n_files = code_sha256(project_root)
    record = {
        "pid": os.getpid(),
        "started": _now(),
        "arm": arm,
        "experiment_name": experiment_name,
        "argv": list(sys.argv),
        "git": git_info(project_root),
        "code_sha256": sha,
        "code_files": n_files,
        "code_file_sha256": code_files(project_root),
        **environment(),
    }
    _write_json(Path(run_dir) / PROVENANCE_FILE, record)
    return record


def update_provenance(run_dir: Path, **fields) -> dict:
    """Merge ``fields`` into provenance.json (atomic rewrite)."""

    path = Path(run_dir) / PROVENANCE_FILE
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    record.update(fields)
    _write_json(path, record)
    return record


def write_status(run_dir: Path, status: str, **fields) -> None:
    """run_status.json: {"status": started | completed | failed, "updated": UTC time, ...}."""

    if status not in ("started", "completed", "failed"):
        raise ValueError(f"unknown run status {status!r}")
    record = {"status": status, "updated": _now(), "pid": os.getpid(), **fields}
    _write_json(Path(run_dir) / STATUS_FILE, record)
