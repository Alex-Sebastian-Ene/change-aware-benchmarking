"""Pins: what a run evaluates, loaded from one YAML file per configuration.

The config_id is the hash of that file. Config files are never edited in place: the
filename must end in the first 12 characters of its hash, so an edit breaks the name
and forces a new file.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

ID_LEN = 12
PINNED_SUFFIX = "_pinned"  # configs/tasks overrides are named <upstream task>_pinned


@dataclass(frozen=True)
class TaskPin:
    name: str  # lm-eval task name
    yaml_version: str  # task YAML `metadata.version` -> logged as benchmark_version
    dataset_repo: str
    dataset_sha: str  # 40-char HF dataset commit, never a branch name
    num_fewshot: int
    fewshot_seed: int
    max_gen_toks: int | None  # None for loglikelihood tasks
    stop: tuple[str, ...] = ()
    parser_version: str | None = None  # generative tasks only
    template_path: str | None = None  # None = task default at pinned commit
    template_sha256: str | None = None
    # `benchmark` in the long/items tables and the item_id hash, shared with public runs.
    # None = name without the _pinned suffix.
    benchmark: str | None = None

    def __post_init__(self):
        if self.benchmark is None:
            object.__setattr__(self, "benchmark", self.name.removesuffix(PINNED_SUFFIX))


@dataclass(frozen=True)
class HarnessPin:
    lm_eval_repo: str  # leaderboard fork URL for merged benchmarks
    lm_eval_sha: str
    vllm_version: str
    dtype: str = "bfloat16"
    batch_size: int = 32  # fixed, never "auto"
    apply_chat_template: bool = False  # base models; instruct = open question
    lockfile_sha256: str = ""
    apptainer_digest: str = ""  # sha256 of the .sif


@dataclass(frozen=True)
class ModelPin:
    repo: str
    sha: str  # 40-char model commit


@dataclass(frozen=True)
class HarnessConfig:
    path: Path
    config_id: str
    harness: HarnessPin
    tasks: tuple[TaskPin, ...]


class ConfigError(ValueError):
    """A config file is malformed or breaks a pinning rule."""


def is_full_sha(value: str) -> bool:
    """True if `value` is a 40-char lowercase hex commit hash (not a branch/tag)."""
    return len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def config_hash(path: Path) -> str:
    """sha256 of a config file, with CRLF normalised so Windows and Linux agree."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def expected_filename(path: Path) -> str:
    """The name a config file must have: `<anything>-<hash prefix>.yaml`."""
    stem = path.stem
    if len(stem) > ID_LEN and stem[-ID_LEN - 1] == "-":
        stem = stem[: -ID_LEN - 1]
    return f"{stem}-{config_hash(path)[:ID_LEN]}.yaml"


def _validate(harness: HarnessPin, tasks: tuple[TaskPin, ...]) -> None:
    if not is_full_sha(harness.lm_eval_sha):
        raise ConfigError(f"lm_eval_sha must be a full 40-char sha, got {harness.lm_eval_sha!r}")
    if not isinstance(harness.batch_size, int):
        raise ConfigError(f"batch_size must be a fixed int, got {harness.batch_size!r}")
    if not tasks:
        raise ConfigError("config has no tasks")
    for t in tasks:
        if not is_full_sha(t.dataset_sha):
            raise ConfigError(f"{t.name}: dataset_sha must be a full 40-char sha")
        if t.max_gen_toks is not None:
            if not t.parser_version:
                raise ConfigError(f"{t.name}: generative task needs a parser_version")
            if not t.stop:
                raise ConfigError(f"{t.name}: generative task needs stop sequences")
        if t.template_path and not t.template_sha256:
            raise ConfigError(f"{t.name}: template_path set but template_sha256 is empty")


def load_config(path: Path) -> HarnessConfig:
    """Load and validate a harness config YAML."""
    path = Path(path)
    if path.name != expected_filename(path):
        raise ConfigError(
            f"{path.name} does not match its contents (edited in place?). "
            f"Configs are immutable: save the change as {expected_filename(path)}"
        )

    raw = yaml.safe_load(path.read_text())
    try:
        harness = HarnessPin(**raw["harness"])
        tasks = tuple(
            TaskPin(
                **{**t, "yaml_version": str(t["yaml_version"]), "stop": tuple(t.get("stop", ()))}
            )
            for t in raw["tasks"]
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"{path.name}: {e}") from e

    _validate(harness, tasks)
    return HarnessConfig(path=path, config_id=config_hash(path), harness=harness, tasks=tasks)


def resolve_model_sha(repo: str, revision: str = "main") -> str:
    """Look up the commit SHA for `revision` on the HF Hub.

    Use this only when WRITING pins, never during a run.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        sha = HfApi().model_info(repo, revision=revision).sha
    except GatedRepoError as e:
        raise RuntimeError(f"{repo} is gated: accept its licence on the Hub and log in") from e
    except RevisionNotFoundError as e:
        raise RuntimeError(f"{repo} has no revision {revision!r}") from e
    except RepositoryNotFoundError as e:
        raise RuntimeError(f"{repo} not found (typo, or private without a token?)") from e

    if is_full_sha(revision) and sha != revision:
        raise RuntimeError(f"{repo}: asked for {revision}, Hub returned {sha}")
    return sha
