"""Pins: what a run evaluates, loaded from one YAML file per configuration.

The config_id is the hash of that file. Config files are never edited in place: the
filename must end in the first 12 characters of its hash, so an edit breaks the name
and forces a new file.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ID_LEN = 12
PINNED_SUFFIX = "_pinned"  # configs/tasks overrides are named <upstream task>_pinned
HASH_SUFFIX = re.compile(rf"-[0-9a-f]{{{ID_LEN}}}$")  # "-c5af5974dcec" in mmlu_pro-c5af5974dcec
SHA1_HEX = re.compile(r"[0-9a-f]{40}")  # a git / HF Hub commit
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
EXACT_VERSION = re.compile(r"\d+(\.\d+)+\S*")  # 0.6.3, 0.6.3.post1; not a range or placeholder


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
        # Frozen, so fields are set with object.__setattr__. YAML gives 1.0 as a float
        # and stop as a list.
        object.__setattr__(self, "yaml_version", str(self.yaml_version))
        object.__setattr__(self, "stop", tuple(self.stop))
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


def _matches(pattern: re.Pattern):
    """A check that a value is a string matching `pattern` in full."""
    return lambda value: isinstance(value, str) and pattern.fullmatch(value) is not None


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)  # YAML reads `yes` as True


# field -> (check, what its value must be)
HARNESS_RULES = {
    "lm_eval_sha": (_matches(SHA1_HEX), "a full 40-char commit sha"),
    "vllm_version": (_matches(EXACT_VERSION), 'an exact version as a string, e.g. "0.6.3"'),
    "lockfile_sha256": (_matches(SHA256_HEX), "a 64-char lowercase hex sha256"),
    "apptainer_digest": (_matches(SHA256_HEX), "a 64-char lowercase hex sha256"),
    "batch_size": (_is_int, 'a fixed int, never "auto"'),
}
TASK_RULES = {
    "dataset_sha": (_matches(SHA1_HEX), "a full 40-char commit sha"),
    "num_fewshot": (_is_int, "an int"),
    "fewshot_seed": (_is_int, "an int"),
}


def is_full_sha(value: str) -> bool:
    """True if `value` is a 40-char lowercase hex commit hash (not a branch/tag)."""
    return _matches(SHA1_HEX)(value)


def config_hash(path: Path) -> str:
    """sha256 of a config file, with CRLF normalised so Windows and Linux agree."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def expected_filename(path: Path) -> str:
    """The name a config file must have: `<name>-<hash prefix>.yaml`, any old prefix dropped."""
    return f"{HASH_SUFFIX.sub('', path.stem)}-{config_hash(path)[:ID_LEN]}.yaml"


def _check_fields(pin, rules: dict, where: str = "") -> None:
    for name, (ok, what) in rules.items():
        value = getattr(pin, name)
        if not ok(value):
            raise ConfigError(f"{where}{name} must be {what}, got {value!r}")


def _validate(harness: HarnessPin, tasks: tuple[TaskPin, ...]) -> None:
    _check_fields(harness, HARNESS_RULES)
    if not tasks:
        raise ConfigError("config has no tasks")
    for t in tasks:
        _check_fields(t, TASK_RULES, where=f"{t.name}: ")
        if t.max_gen_toks is not None and not (t.parser_version and t.stop):
            raise ConfigError(f"{t.name}: a generative task needs parser_version and stop")
        if t.template_path and not t.template_sha256:
            raise ConfigError(f"{t.name}: template_path set but template_sha256 is empty")


def load_config(path: Path) -> HarnessConfig:
    """Load and validate a harness config YAML."""
    path = Path(path)
    expected = expected_filename(path)
    if path.name != expected:
        raise ConfigError(
            f"{path.name} does not match its contents (edited in place?). "
            f"Configs are immutable: save the change as {expected}"
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        harness = HarnessPin(**raw["harness"])
        tasks = tuple(TaskPin(**t) for t in raw["tasks"])
    except (UnicodeDecodeError, KeyError, TypeError) as e:  # not UTF-8, missing or unknown key
        raise ConfigError(f"{path.name}: {type(e).__name__}: {e}") from e
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
