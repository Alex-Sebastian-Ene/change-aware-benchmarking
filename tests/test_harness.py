"""Tests for the pinned harness (no GPU or lm-eval needed)."""

import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

from cabench.harness import manifest, parsers, run
from cabench.harness.pins import (
    ConfigError,
    HarnessPin,
    ModelPin,
    TaskPin,
    config_hash,
    expected_filename,
    is_full_sha,
    load_config,
)
from cabench.harness.verify import PinMismatch, check, git, sha256_file, verify_harness

SHA = "a" * 40
LOCK_SHA256 = "b" * 64
SIF_SHA256 = "c" * 64
MODEL = ModelPin(repo="org/model", sha=SHA)
HARNESS = HarnessPin(
    lm_eval_repo="https://example.com/lm-eval",
    lm_eval_sha=SHA,
    vllm_version="0.0.0",
    lockfile_sha256=LOCK_SHA256,
    apptainer_digest=SIF_SHA256,
)
TASK = TaskPin(
    name="arc_easy",
    yaml_version="1.0",
    dataset_repo="allenai/ai2_arc",
    dataset_sha=SHA,
    num_fewshot=25,
    fewshot_seed=1234,
    max_gen_toks=None,
)
RAW = {
    "harness": {
        "lm_eval_repo": "https://example.com/lm-eval",
        "lm_eval_sha": SHA,
        "vllm_version": "0.0.0",
        "lockfile_sha256": LOCK_SHA256,
        "apptainer_digest": SIF_SHA256,
    },
    "tasks": [
        {
            "name": "arc_easy",
            "yaml_version": 1.0,
            "dataset_repo": "allenai/ai2_arc",
            "dataset_sha": SHA,
            "num_fewshot": 25,
            "fewshot_seed": 1234,
            "max_gen_toks": None,
        }
    ],
}


def write_config(tmp_path, raw=RAW, name="test"):
    """Write `raw` as YAML and rename it to its required hashed filename."""
    draft = tmp_path / f"{name}.yaml"
    draft.write_text(yaml.safe_dump(raw))
    final = tmp_path / expected_filename(draft)
    draft.rename(final)
    return final


# ---------- pins / config ----------


def test_is_full_sha():
    assert is_full_sha(SHA)
    assert not is_full_sha("main")
    assert not is_full_sha("a" * 39)
    assert not is_full_sha("g" * 40)


def test_load_config(tmp_path):
    cfg = load_config(write_config(tmp_path))
    assert cfg.harness == HARNESS
    assert cfg.tasks == (TASK,)
    assert cfg.config_id == config_hash(cfg.path)


def test_config_edited_in_place_is_refused(tmp_path):
    path = write_config(tmp_path)
    path.write_text(path.read_text().replace("25", "5"))
    with pytest.raises(ConfigError, match="edited in place"):
        load_config(path)


def test_config_is_read_as_utf8(tmp_path):
    draft = tmp_path / "utf8.yaml"
    draft.write_bytes((yaml.safe_dump(RAW) + "# naïve – café\n").encode("utf-8"))
    path = draft.rename(tmp_path / expected_filename(draft))
    assert load_config(path).tasks == (TASK,)


def test_config_not_utf8_is_refused(tmp_path):
    draft = tmp_path / "latin1.yaml"
    draft.write_bytes((yaml.safe_dump(RAW) + "# café\n").encode("latin-1"))
    path = draft.rename(tmp_path / expected_filename(draft))
    with pytest.raises(ConfigError, match="UTF-8"):
        load_config(path)


def test_config_hash_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.yaml", tmp_path / "crlf.yaml"
    lf.write_bytes(b"a: 1\nb: 2\n")
    crlf.write_bytes(b"a: 1\r\nb: 2\r\n")
    assert config_hash(lf) == config_hash(crlf)


def test_expected_filename_replaces_old_hash(tmp_path):
    path = write_config(tmp_path, name="base")
    assert expected_filename(path) == path.name
    assert path.name.count("-") == 1


@pytest.mark.parametrize(
    "name, benchmark, expected",
    [
        ("leaderboard_mmlu_pro_pinned", None, "leaderboard_mmlu_pro"),
        ("arc_easy", None, "arc_easy"),
        ("arc_easy_pinned", "ai2_arc_easy", "ai2_arc_easy"),
    ],
)
def test_task_benchmark(name, benchmark, expected):
    assert TaskPin(**{**TASK.__dict__, "name": name, "benchmark": benchmark}).benchmark == expected


def test_matrix_config_id_matches_pins(tmp_path):
    """scripts/matrix/utils/config.py must give the IDs the harness and schema check use."""
    spec = importlib.util.spec_from_file_location(
        "matrix_config", Path(__file__).parents[1] / "scripts/matrix/utils/config.py"
    )
    matrix_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix_config)

    path = write_config(tmp_path)
    assert matrix_config.config_id(path) == load_config(path).config_id
    assert matrix_config.load_config(path)["_config_id"] == load_config(path).config_id
    path.write_text(path.read_text().replace("25", "5"))
    with pytest.raises(ConfigError, match="edited in place"):
        matrix_config.load_config(path)


@pytest.mark.parametrize(
    "harness_edit, task_edit, match",
    [
        ({"lm_eval_sha": "main"}, {}, "lm_eval_sha"),
        ({"batch_size": "auto"}, {}, "batch_size"),
        # placeholders left from template.yaml.example
        ({"vllm_version": "<e.g. 0.6.3>"}, {}, "vllm_version"),
        ({"vllm_version": 0.7}, {}, "vllm_version"),  # unquoted in YAML -> float
        ({"lockfile_sha256": "<sha256 of uv.lock>"}, {}, "lockfile_sha256"),
        ({"apptainer_digest": "<sha256 of the .sif>"}, {}, "apptainer_digest"),
        ({"apptainer_digest": ""}, {}, "apptainer_digest"),
        ({}, {"num_fewshot": "5"}, "num_fewshot"),
        ({}, {"fewshot_seed": None}, "fewshot_seed"),
        ({}, {"dataset_sha": "main"}, "dataset_sha"),
        ({}, {"max_gen_toks": 256, "stop": ["\n"]}, "parser_version"),
        ({}, {"max_gen_toks": 256, "parser_version": "v1"}, "stop"),
        ({}, {"template_path": "t.jinja"}, "template_sha256"),
        ({}, {"typo_field": 1}, "typo_field"),
    ],
)
def test_bad_config_is_refused(tmp_path, harness_edit, task_edit, match):
    raw = {
        "harness": {**RAW["harness"], **harness_edit},
        "tasks": [{**RAW["tasks"][0], **task_edit}],
    }
    with pytest.raises(ConfigError, match=match):
        load_config(write_config(tmp_path, raw))


# ---------- verify helpers ----------


def test_check_passes_on_match():
    check("x", 1, 1)


def test_check_raises_and_names_field():
    with pytest.raises(PinMismatch, match="dtype"):
        check("dtype", "float16", "bfloat16")


def test_sha256_file_known_value(tmp_path):
    f = tmp_path / "f.txt"
    f.write_bytes(b"hello")
    assert sha256_file(f) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_sha256_file_small_chunks_same_result(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(bytes(range(256)) * 100)
    assert sha256_file(f, chunk=7) == sha256_file(f)


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repo with one commit."""
    r = tmp_path / "repo"
    r.mkdir()

    def g(*a):
        subprocess.run(["git", "-C", str(r), *a], check=True, capture_output=True)

    g("init")
    g("config", "user.email", "t@t")
    g("config", "user.name", "t")
    (r / "a.txt").write_text("a")
    g("add", ".")
    g("commit", "-m", "init")
    g("remote", "add", "origin", "https://example.com/lm-eval.git")
    return r


def test_git_rev_parse(repo):
    assert is_full_sha(git(repo, "rev-parse", "HEAD"))


def test_verify_harness_rejects_wrong_commit(repo, tmp_path):
    lock, sif = tmp_path / "uv.lock", tmp_path / "img.sif"
    lock.write_text("lock")
    sif.write_text("sif")
    with pytest.raises(PinMismatch, match="lm_eval_sha"):
        verify_harness(HARNESS, repo, lock, sif)


def test_verify_harness_rejects_lm_eval_not_from_checkout(repo, tmp_path):
    pin = HarnessPin(**{**HARNESS.__dict__, "lm_eval_sha": git(repo, "rev-parse", "HEAD")})
    with pytest.raises(PinMismatch, match="lm_eval is imported from"):
        verify_harness(pin, repo, tmp_path / "uv.lock", tmp_path / "img.sif")


def test_verify_harness_refuses_empty_pins(repo, tmp_path):
    empty = HarnessPin(lm_eval_repo="x", lm_eval_sha=SHA, vllm_version="0.0.0")
    with pytest.raises(PinMismatch, match="lockfile_sha256"):
        verify_harness(empty, repo, tmp_path / "uv.lock", tmp_path / "img.sif")


# ---------- manifest ----------


def test_canonical_json_ignores_key_order():
    assert manifest.canonical_json({"a": 1, "b": 2}) == manifest.canonical_json({"b": 2, "a": 1})


def test_manifest_ids(tmp_path):
    cfg = load_config(write_config(tmp_path))
    m = manifest.build_manifest(MODEL, cfg, {"arc_easy": "1.0"}, "abc1234")
    assert m["config_id"] == cfg.config_id
    assert m["benchmark_versions"] == {"arc_easy": "1.0"}
    assert len(m["run_id"]) == 64
    assert m["run_id"] != manifest.build_manifest(MODEL, cfg)["run_id"]


def test_manifest_records_limit(tmp_path):
    cfg = load_config(write_config(tmp_path))
    assert manifest.build_manifest(MODEL, cfg)["limit"] is None
    assert manifest.build_manifest(MODEL, cfg, limit=10)["limit"] == 10


# ---------- parsers ----------


def test_register_and_get_parser():
    @parsers.register("_test_task", "v1")
    def p(text):
        return text.strip()

    assert parsers.get_parser("_test_task", "v1")(" 42 ") == "42"


def test_register_same_version_twice_raises():
    parsers.register("_test_dup", "v1")(lambda s: s)
    with pytest.raises(ValueError):
        parsers.register("_test_dup", "v1")(lambda s: s)


def test_get_parser_missing_raises_keyerror():
    with pytest.raises(KeyError, match="_no_such_task"):
        parsers.get_parser("_no_such_task", "v1")


# ---------- run (string builders only; no GPU) ----------


def test_gen_kwargs_loglikelihood_has_no_max_gen_toks():
    assert "max_gen_toks" not in run.build_gen_kwargs(TASK)


def test_gen_kwargs_generative_includes_max_gen_toks():
    gen = TaskPin(**{**TASK.__dict__, "max_gen_toks": 256})
    assert "max_gen_toks=256" in run.build_gen_kwargs(gen)


def test_model_args_uses_sha_not_branch():
    args = run.build_model_args(MODEL, HARNESS).split(",")
    assert f"revision={SHA}" in args
    assert f"tokenizer_revision={SHA}" in args
