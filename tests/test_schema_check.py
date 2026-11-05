import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cabench.harness.pins import expected_filename
from cabench.schema_check import load_configs, read_tables, run_checks

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_schema.py"
SHA = "a" * 40
REV1, REV2 = "1" * 40, "2" * 40


def item_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


ARC = [item_id(f"arc{i}") for i in range(3)]
HS = [item_id(f"hs{i}") for i in range(2)]


def task(name: str, **extra) -> dict:
    return {
        "name": name,
        "yaml_version": "1.0",
        "dataset_repo": "org/data",
        "dataset_sha": SHA,
        "num_fewshot": 0,
        "fewshot_seed": 0,
        "max_gen_toks": None,
        **extra,
    }


def write_config(directory: Path, name: str, tasks: list[str], **task_extra) -> str:
    raw = {
        "harness": {
            "lm_eval_repo": "https://example.com/lm-eval",
            "lm_eval_sha": SHA,
            "vllm_version": "0.0.0",
            "lockfile_sha256": "b" * 64,
            "apptainer_digest": "c" * 64,
        },
        "tasks": [task(t, **task_extra) for t in tasks],
    }
    draft = directory / f"{name}.yaml"
    draft.write_text(yaml.safe_dump(raw))
    final = directory / expected_filename(draft)
    draft.rename(final)
    return final.stem.rsplit("-", 1)[1]


@pytest.fixture
def configs_dir(tmp_path):
    d = tmp_path / "configs"
    d.mkdir()
    write_config(d, "both", ["arc_pinned", "hs_pinned"])
    write_config(d, "arc", ["arc_pinned"])
    return d


@pytest.fixture
def configs(configs_dir):
    loaded, errors = load_configs(configs_dir)
    assert errors == []
    return loaded


def config_id(configs, n_tasks: int) -> str:
    return next(cid for cid, c in configs.items() if len(c.tasks) == n_tasks)


LONG_COLUMNS = ["run_id", "item_id", "model_id", "model_revision", "config_id", "source"]


def make_tables(configs) -> dict[str, pd.DataFrame]:
    """r1: model m1 on both tasks (5 items). r2: model m2 on arc only (3 items)."""
    both, arc = config_id(configs, 2), config_id(configs, 1)
    rows = [("r1", i, "m1", REV1, both, "own") for i in ARC + HS]
    rows += [("r2", i, "m2", REV2, arc, "own") for i in ARC]
    return {
        "long": pd.DataFrame(rows, columns=LONG_COLUMNS),
        "models": pd.DataFrame({"model_id": ["m1", "m2"], "model_revision": [REV1, REV2]}),
        "items": pd.DataFrame(
            {
                "item_id": ARC + HS,
                # task arc_pinned -> benchmark arc, as in the item_id of public runs
                "benchmark": ["arc"] * 3 + ["hs"] * 2,
                "benchmark_version": ["1.0"] * 5,
            }
        ),
    }


def add_public_run(tables, items) -> None:
    """A leaderboard run of m1: placeholder config_id, any subset of items."""
    rows = [("p1", i, "m1", REV1, "openllm_v2_public", "public") for i in items]
    tables["long"] = pd.concat([tables["long"], pd.DataFrame(rows, columns=LONG_COLUMNS)])


def results(tables, configs, errors=()):
    return run_checks(tables, configs, list(errors))


def names(configs):
    return [r.name for r in results(make_tables(configs), configs)]


def failing(tables, configs, errors=()) -> set[str]:
    return {r.name for r in results(tables, configs, errors) if not r.ok}


def text(tables, configs, errors=()) -> str:
    return "\n".join(line for r in results(tables, configs, errors) for line in r.failures)


def test_clean_batch_passes(configs):
    assert failing(make_tables(configs), configs) == set()


def test_duplicate_key(configs):
    unique, counts, _ = names(configs)
    t = make_tables(configs)
    t["long"] = pd.concat([t["long"], t["long"].iloc[[0]]])
    assert failing(t, configs) == {unique, counts}
    assert f"('r1', '{ARC[0]}')" in text(t, configs)


def test_missing_row(configs):
    _, counts, _ = names(configs)
    t = make_tables(configs)
    t["long"] = t["long"].drop(index=4)
    assert failing(t, configs) == {counts}
    assert "r1: 4 rows, expected 5" in text(t, configs)


def test_missing_whole_benchmark(configs):
    _, counts, _ = names(configs)
    t = make_tables(configs)
    t["long"] = t["long"][~t["long"]["item_id"].isin(HS)]
    assert failing(t, configs) == {counts}
    assert "r1: 3 rows, expected 5" in text(t, configs)


def test_items_of_other_version_not_counted(configs):
    _, counts, _ = names(configs)
    t = make_tables(configs)
    t["items"].loc[0, "benchmark_version"] = "2.0"
    assert counts in failing(t, configs)


@pytest.mark.parametrize(
    "column, value, message",
    [
        ("item_id", "ffffffffffffffff", "item_id: 1 values not in items"),
        ("model_revision", "9" * 40, "model: 1 (model_id, model_revision) pairs not in models"),
    ],
)
def test_unresolved_item_or_model(configs, column, value, message):
    *_, refs = names(configs)
    t = make_tables(configs)
    t["long"].loc[0, column] = value
    assert failing(t, configs) == {refs}
    assert message in text(t, configs)


def test_unresolved_config_skips_count(configs):
    *_, refs = names(configs)
    t = make_tables(configs)
    t["long"].loc[t["long"]["run_id"] == "r1", "config_id"] = "f" * 64
    assert failing(t, configs) == {refs}
    assert "config_id: 1 values match no config file" in text(t, configs)


def test_public_run_skips_config_and_count(configs):
    t = make_tables(configs)
    add_public_run(t, ARC[:2])
    assert failing(t, configs) == set()


def test_public_run_still_checks_keys_and_refs(configs):
    unique, _, refs = names(configs)
    t = make_tables(configs)
    add_public_run(t, [ARC[0], ARC[0], "ffffffffffffffff"])
    assert failing(t, configs) == {unique, refs}
    assert "item_id: 1 values not in items" in text(t, configs)
    assert "config_id" not in text(t, configs)


def test_task_benchmark_override(tmp_path):
    d = tmp_path / "configs"
    d.mkdir()
    write_config(d, "named", ["arc_pinned"], benchmark="arc_challenge")
    configs, errors = load_configs(d)
    assert errors == []
    (cid,) = configs
    t = {
        "long": pd.DataFrame(
            [("r1", i, "m1", REV1, cid, "own") for i in ARC], columns=LONG_COLUMNS
        ),
        "models": pd.DataFrame({"model_id": ["m1"], "model_revision": [REV1]}),
        "items": pd.DataFrame(
            {"item_id": ARC, "benchmark": ["arc_challenge"] * 3, "benchmark_version": ["1.0"] * 3}
        ),
    }
    assert failing(t, configs) == set()


def test_config_edited_in_place(configs_dir, configs):
    *_, refs = names(configs)
    t = make_tables(configs)
    path = next(configs_dir.glob("arc-*.yaml"))
    path.write_text(path.read_text().replace("num_fewshot: 0", "num_fewshot: 5"))
    edited, errors = load_configs(configs_dir)
    assert refs in failing(t, edited, errors)
    assert "edited in place" in text(t, edited, errors)


def write_csvs(tables, directory: Path) -> Path:
    directory.mkdir(exist_ok=True)
    for name, df in tables.items():
        df.to_csv(directory / f"{name}.csv", index=False)
    return directory


def test_read_tables_keeps_ids_as_strings(tmp_path, configs):
    t = make_tables(configs)
    t["models"].loc[0, "model_id"] = "007"
    t["long"].loc[t["long"]["model_id"] == "m1", "model_id"] = "007"
    t["long"]["extra"] = "ignored"
    loaded = read_tables(write_csvs(t, tmp_path / "tables"))
    assert loaded["models"]["model_id"].tolist() == ["007", "m2"]
    assert failing(loaded, configs) == set()


def test_read_tables_missing_column(tmp_path, configs):
    t = make_tables(configs)
    t["long"] = t["long"].drop(columns="config_id")
    with pytest.raises(ValueError, match="long is missing columns"):
        read_tables(write_csvs(t, tmp_path / "tables"))


def test_read_tables_matrix_store_layout(tmp_path, configs):
    """long/<benchmark>/<run_id>.parquet + items/models.parquet, as scripts/matrix writes."""
    pytest.importorskip("pyarrow")  # parquet support comes with `uv sync --group matrix`
    t = make_tables(configs)
    add_public_run(t, ARC)
    store = tmp_path / "data"
    for run_id, run in t["long"].groupby("run_id"):
        out = store / "long" / "arc" / f"{run_id}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        run.assign(correct=1).to_parquet(out, index=False)
    t["items"].to_parquet(store / "items.parquet", index=False)
    t["models"].to_parquet(store / "models.parquet", index=False)
    loaded = read_tables(store)
    assert sorted(loaded["long"]["run_id"].unique()) == ["p1", "r1", "r2"]
    assert failing(loaded, configs) == set()


def run_script(tables_dir: Path, configs_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(tables_dir), "--configs", str(configs_dir)],
        capture_output=True,
        text=True,
    )


def test_script_exit_codes(tmp_path, configs_dir, configs):
    clean = write_csvs(make_tables(configs), tmp_path / "clean")
    assert run_script(clean, configs_dir).returncode == 0

    t = make_tables(configs)
    t["long"] = t["long"].drop(index=4)
    result = run_script(write_csvs(t, tmp_path / "broken"), configs_dir)
    assert result.returncode == 1
    assert "FAIL  row count per run equals the item count" in result.stdout

    assert run_script(tmp_path / "missing", configs_dir).returncode == 2
