"""ingest_public.py and build_tables.py on a fake download_details.py folder.

The details parquets are shaped like the Open LLM Leaderboard's: `doc` a struct,
`arguments` as {"gen_args_0": {"arg_0": ...}}, `resps` strings, MuSR's `target` the
answer text. transformers is replaced by a whitespace tokeniser.
"""
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")  # the matrix group: uv sync --group matrix
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/matrix"))

from utils.ingest import read_samples  # noqa: E402

SAFE = "Org__Model-1B"
STAMP = "2024_09_19T15_51_35.851278"
MUSR = [f"leaderboard_musr_{s}" for s in ("murder_mysteries", "object_placements",
                                          "team_allocation")]
TOKENIZER_STUB = '''
class _Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}
class AutoTokenizer:
    @staticmethod
    def from_pretrained(*a, **k):
        return _Tok()
'''


def _resps(lps):
    return [[[str(lp), "False"]] for lp in lps], [[str(lp), "False"] for lp in lps]


def mmlu_row(doc_id, qid, question, options, answer, lps):
    r, fr = _resps(lps)
    best = max(range(len(lps)), key=lps.__getitem__)
    return dict(
        doc_id=doc_id,
        doc=dict(question_id=qid, question=question, options=options, answer="ABCD"[answer],
                 answer_index=answer, category="math"),
        target="ABCD"[answer],
        arguments={"gen_args_0": {"arg_0": f"prompt {doc_id}", "arg_1": " A"}},
        resps=r, filtered_resps=fr, acc=float(best == answer))


def musr_row(doc_id, narrative, choices, answer, lps):
    r, fr = _resps(lps)
    norm = [lp / len(c) for lp, c in zip(lps, choices, strict=True)]
    best = max(range(len(norm)), key=norm.__getitem__)
    return dict(
        doc_id=doc_id,
        doc=dict(narrative=narrative, question="Who is the most likely murderer?",
                 choices=str(choices), answer_index=answer, answer_choice=choices[answer]),
        target=choices[answer], arguments={"gen_args_0": {"arg_0": f"{narrative} words",
                                                          "arg_1": " x"}},
        resps=r, filtered_resps=fr, acc_norm=float(best == answer))


MMLU_ROWS = [
    mmlu_row(0, 30, "q one", ["a", "b"], 0, [-1.0, -2.0]),
    mmlu_row(1, 20, "dup", ["a", "b"], 1, [-1.0, -2.0]),
    mmlu_row(2, 10, "dup", ["a", "b"], 1, [-3.0, -2.0]),  # lower question_id: kept
]
# "Al" wins on raw log-probability, "Bartholomew" per character: acc_norm picks B
MUSR_ROWS = {t: [musr_row(0, f"{t} story", ["Al", "Bartholomew"], 1, [-2.0, -3.0]),
                 musr_row(1, f"{t} story two", ["X", "Y"], 0, [-1.0, -2.0])] for t in MUSR}


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw" / SAFE
    d.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(MMLU_ROWS), d / f"{SAFE}__leaderboard_mmlu_pro.parquet")
    for t, rows in MUSR_ROWS.items():
        pq.write_table(pa.Table.from_pylist(rows), d / f"{SAFE}__{t}.parquet")
    tasks = ["leaderboard_mmlu_pro", *MUSR]
    (d / "manifest.json").write_text(json.dumps({
        "model": "Org/Model-1B", "repo": f"open-llm-leaderboard/{SAFE}-details",
        "configs": {f"{SAFE}__{t}": {"split_used": "latest", "all_splits": [STAMP, "latest"],
                                     "rows": 1} for t in tasks}}))
    (d / f"results_{STAMP}.json").write_text(json.dumps({
        "config": {"model_sha": "a" * 40}, "git_hash": "abc123",
        "versions": {"leaderboard_mmlu_pro": 0.1, **{t: 1.0 for t in MUSR}}}))
    stub = tmp_path / "stub" / "transformers"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text(TOKENIZER_STUB)
    return d


def ingest_public(raw: Path, *benchmarks: str) -> Path:
    data = raw.parents[1] / "data"
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join([str(raw.parents[1] / "stub"), str(ROOT / "src")])}
    subprocess.run([sys.executable, str(ROOT / "scripts/matrix/ingest_public.py"), str(raw),
                    "--data", str(data), "--benchmark", *benchmarks],
                   check=True, capture_output=True, text=True, env=env)
    return data


def build_tables_module():
    spec = importlib.util.spec_from_file_location("build_tables",
                                                  ROOT / "scripts/matrix/build_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mmlu_pro_public_rows(raw):
    data = ingest_public(raw, "leaderboard_mmlu_pro")
    long = pd.read_parquet(data / "long/leaderboard_mmlu_pro" / f"public_{SAFE}_{STAMP}.parquet")

    assert len(long) == 2  # the duplicate with question_id 20 is dropped
    assert long["correct"].tolist() == [1, 1]
    assert long["logprob"].tolist() == [-1.0, -2.0]
    assert long["pred"].tolist() == ["A", "B"]
    assert long["prompt_tokens"].tolist() == [2, 2]
    assert set(long["benchmark_version"]) == {"0.1"}
    assert set(long["config_id"]) == {"openllm_v2_public"}
    assert long["timestamp"].iloc[0] == pd.Timestamp("2024-09-19 15:51:35.851278", tz="UTC")


def test_musr_public_rows_in_one_file(raw):
    data = ingest_public(raw, "leaderboard_musr")
    files = list((data / "long").rglob("*.parquet"))
    assert [f.relative_to(data / "long").as_posix() for f in files] == [
        f"leaderboard_musr/public_{SAFE}_{STAMP}.parquet"]
    long = pd.read_parquet(files[0])

    assert len(long) == 6 and long["item_id"].is_unique
    assert long["pred"].tolist() == ["B", "A"] * 3  # acc_norm's choice, not raw argmax
    assert long["correct"].tolist() == [1, 1] * 3
    assert set(long["benchmark_version"]) == {"1.0"}


def test_public_and_own_musr_rows_share_item_ids(raw, tmp_path):
    data = ingest_public(raw, "leaderboard_musr")
    public = pd.read_parquet(next((data / "long").rglob("*.parquet")))

    own = []
    for t, rows in MUSR_ROWS.items():
        path = tmp_path / f"{t}.jsonl"
        # own samples log arguments as [[context, continuation], ...]
        lines = [{**r, "arguments": [[r["arguments"]["gen_args_0"]["arg_0"], " x"]]}
                 for r in rows]
        path.write_text("".join(json.dumps(r) + "\n" for r in lines))
        own.append(read_samples(
            path, task=t, benchmark="leaderboard_musr", benchmark_version="1.0", run_id="own",
            source="own", model_id="Org/Model-1B", model_revision="a" * 40, config_id="c" * 64,
            harness_commit="h" * 40, metric="acc_norm",
            timestamp=datetime(2026, 11, 18, tzinfo=UTC))[0])
    own = pd.concat(own, ignore_index=True)

    assert own["item_id"].tolist() == public["item_id"].tolist()
    assert own["pred"].tolist() == public["pred"].tolist()
    assert own["correct"].tolist() == public["correct"].tolist()


def test_items_cover_public_rows_in_source_index_order(raw):
    data = ingest_public(raw, "leaderboard_mmlu_pro", "leaderboard_musr")
    bt = build_tables_module()

    mmlu = bt.public_items(raw.parent, "leaderboard_mmlu_pro", "0.1")
    assert mmlu["source_index"].tolist() == ["10", "30"]  # by question_id, as before
    musr = bt.public_items(raw.parent, "leaderboard_musr", "1.0")
    assert musr["subtask"].tolist() == [s for s in ("murder_mysteries", "object_placements",
                                                    "team_allocation") for _ in (0, 1)]
    assert musr["answer_key"].tolist() == ["1", "0"] * 3

    for bench, items in (("leaderboard_mmlu_pro", mmlu), ("leaderboard_musr", musr)):
        long = pd.read_parquet(data / "long" / bench)
        assert set(long["item_id"]) == set(items["item_id"])
        assert bt.build_wide(long, items, bench, "openllm_v2_public").shape == (1, len(items))


def test_tasks_from_different_runs_are_refused(raw):
    m = json.loads((raw / "manifest.json").read_text())
    m["configs"][f"{SAFE}__{MUSR[1]}"]["all_splits"] = ["2024_10_01T00_00_00.000000"]
    (raw / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(subprocess.CalledProcessError) as e:
        ingest_public(raw, "leaderboard_musr")
    assert "different leaderboard runs" in e.value.stderr
