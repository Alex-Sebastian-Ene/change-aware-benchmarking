"""Verify that the environment matches the pins. Fail loudly on any drift."""

import importlib.util
import subprocess
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cabench.harness.parsers import get_parser
from cabench.harness.pins import HarnessPin, TaskPin


class PinMismatch(RuntimeError):
    """Raised when something on disk / installed does not match its pin."""


def check(name: str, actual, expected) -> None:
    """Raise PinMismatch naming `name` if actual != expected."""
    if actual != expected:
        raise PinMismatch(f"{name}: expected {expected!r}, got {actual!r}")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Hex sha256 of a file, read in chunks (the .sif is several GB)."""
    h = sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    """Run `git -C repo <args>` and return stripped stdout. Raise on failure."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _norm_url(url: str) -> str:
    return url.strip().removesuffix("/").removesuffix(".git").lower()


def verify_harness(pin: HarnessPin, lm_eval_dir: Path, lockfile: Path, sif: Path) -> None:
    """Check code, packages and container all match `pin`."""
    if not pin.lockfile_sha256:
        raise PinMismatch("lockfile_sha256 is empty, fill it in before running")
    if not pin.apptainer_digest:
        raise PinMismatch("apptainer_digest is empty, fill it in before running")

    lm_eval_dir = Path(lm_eval_dir).resolve()
    check("lm_eval_sha", git(lm_eval_dir, "rev-parse", "HEAD"), pin.lm_eval_sha)
    check("lm_eval working tree", git(lm_eval_dir, "status", "--porcelain"), "")
    check(
        "lm_eval_repo",
        _norm_url(git(lm_eval_dir, "remote", "get-url", "origin")),
        _norm_url(pin.lm_eval_repo),
    )

    spec = importlib.util.find_spec("lm_eval")
    imported_from = Path(spec.origin).resolve() if spec and spec.origin else None
    if imported_from is None or not imported_from.is_relative_to(lm_eval_dir):
        raise PinMismatch(
            f"lm_eval is imported from {imported_from}, not the pinned checkout {lm_eval_dir}"
        )

    try:
        installed_vllm = version("vllm")
    except PackageNotFoundError:
        installed_vllm = None
    check("vllm_version", installed_vllm, pin.vllm_version)

    check("lockfile_sha256", sha256_file(lockfile), pin.lockfile_sha256)
    check("apptainer_digest", sha256_file(sif), pin.apptainer_digest)


def verify_task(pin: TaskPin, task_manager) -> None:
    """Check the task lm-eval will actually run matches `pin`."""
    if hasattr(task_manager, "load"):
        task = task_manager.load([pin.name])["tasks"][pin.name]
    else:
        from lm_eval.tasks import get_task_dict

        task = get_task_dict([pin.name], task_manager)[pin.name]
    cfg = task.config

    check(f"{pin.name} yaml_version", str((cfg.metadata or {}).get("version")), pin.yaml_version)
    check(f"{pin.name} dataset_repo", cfg.dataset_path, pin.dataset_repo)
    check(
        f"{pin.name} dataset_sha",
        (cfg.dataset_kwargs or {}).get("revision"),
        pin.dataset_sha,
    )

    if pin.stop:
        until = (cfg.generation_kwargs or {}).get("until", [])
        check(f"{pin.name} stop", tuple(until), pin.stop)

    if pin.parser_version:
        get_parser(pin.name, pin.parser_version)

    if pin.template_path:
        check(
            f"{pin.name} template_sha256",
            sha256_file(Path(pin.template_path)),
            pin.template_sha256,
        )
