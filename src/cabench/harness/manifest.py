"""Per-run manifest, stored beside the results.

config_id  hash of the config YAML -> shared by every run of that config
run_id     hash of the whole manifest -> unique per run
"""

import hashlib
import json
import os
import socket
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from cabench.harness.pins import HarnessConfig, ModelPin


def canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, no whitespace variation."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def gpu_info() -> list[dict] | None:
    """Name + driver of each GPU, or None if there's no nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        name, driver = (s.strip() for s in line.split(",", 1))
        gpus.append({"name": name, "driver": driver})
    return gpus


def cuda_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.version.cuda


def runtime_facts() -> dict:
    """Facts about where/when this run happened."""
    return {
        "hostname": socket.gethostname(),
        "utc_time": datetime.now(UTC).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpus": gpu_info(),
        "cuda": cuda_version(),
    }


def build_manifest(
    model: ModelPin,
    config: HarnessConfig,
    benchmark_versions: dict[str, str] | None = None,
    project_git_hash: str | None = None,
    limit: int | None = None,
) -> dict:
    """Assemble the manifest dict, including config_id and run_id.

    project_git_hash is lm-eval's reported `git_hash`, which is the commit of the
    repo the run was launched from (this project), not of lm-eval. limit is the
    per-task item cap of a partial (test) run; None for a full run.
    """
    manifest = {
        "config_file": config.path.name,
        "config_id": config.config_id,
        "model": asdict(model),
        "harness": asdict(config.harness),
        "tasks": [asdict(t) for t in config.tasks],
        "benchmark_versions": benchmark_versions or {},
        "project_git_hash": project_git_hash,
        "limit": limit,
        "runtime": runtime_facts(),
    }
    manifest["run_id"] = sha256_text(canonical_json(manifest))
    return manifest


def write_manifest(manifest: dict, out: Path) -> Path:
    """Write manifest.json into `out` and return its path."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return path
