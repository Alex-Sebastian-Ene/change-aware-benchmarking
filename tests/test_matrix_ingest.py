import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")  # utils.schema needs the matrix group: uv sync --group matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/matrix"))

from utils.ingest import read_samples  # noqa: E402
from utils.matrix import build_wide  # noqa: E402
from utils.schema import LONG, conform  # noqa: E402

TASK = "leaderboard_mmlu_pro"
CFG = "c" * 64
T1 = datetime(2026, 11, 1, 12, tzinfo=UTC)
T2 = datetime(2026, 11, 2, 12, tzinfo=UTC)


def write_samples(path: Path, docs: list[tuple[int, str, int]], target=0) -> Path:
    """One lm-eval sample per (question_id, question, acc), in file order."""
    with open(path, "w") as f:
        for doc_id, (qid, question, acc) in enumerate(docs):
            doc = {
                "question_id": qid,
                "question": question,
                "options": ["x", "y"],
                "answer": "A",
                "category": "math",
            }
            rec = {
                "doc_id": doc_id,
                "doc": doc,
                "target": target,
                "resps": [[["-0.5", True]], [["-1.5", False]]],
                "acc": acc,
            }
            f.write(json.dumps(rec) + "\n")
    return path


def read(path: Path, run_id: str = "run", timestamp: datetime = T1):
    return read_samples(
        path,
        task=TASK,
        benchmark=TASK,
        benchmark_version="0.1",
        run_id=run_id,
        source="own",
        model_id="org/model",
        model_revision="a" * 40,
        config_id=CFG,
        harness_commit="h" * 40,
        metric="acc",
        timestamp=timestamp,
    )


def test_duplicate_question_keeps_lowest_question_id(tmp_path):
    docs = [(5, "q1", 1), (9, "dup", 0), (3, "q2", 1), (7, "dup", 1)]
    long, items = read(write_samples(tmp_path / "s.jsonl", docs))

    assert len(long) == len(items) == 3
    assert long["item_id"].is_unique
    assert list(long["correct"]) == [1, 1, 1]  # q1, q2, then dup from question_id 7
    assert list(items["source_index"]) == ["5", "3", "7"]
    conform(long, LONG, "long")
    assert build_wide(long, items, TASK, CFG).shape == (1, 3)


def test_earliest_run_is_canonical_whatever_the_ingest_order(tmp_path):
    early = read(write_samples(tmp_path / "a.jsonl", [(1, "q1", 0), (2, "q2", 0)]), "a", T1)
    late = read(write_samples(tmp_path / "b.jsonl", [(1, "q1", 1), (2, "q2", 1)]), "b", T2)
    items = early[1]

    for parts in ([early[0], late[0]], [late[0], early[0]]):
        wide = build_wide(pd.concat(parts, ignore_index=True), items, TASK, CFG)
        assert wide.to_numpy().tolist() == [[0.0, 0.0]]


def test_timestamp_needs_a_timezone(tmp_path):
    path = write_samples(tmp_path / "s.jsonl", [(1, "q1", 1)])
    with pytest.raises(ValueError, match="timezone"):
        read(path, timestamp=datetime(2026, 11, 1, 12))


@pytest.mark.parametrize("target", [0, "0", "A", "(A)"])
def test_target_as_index_or_letter(tmp_path, target):
    # MMLU-Pro logs the answer letter as target; other tasks log an index
    long, _ = read(write_samples(tmp_path / "s.jsonl", [(1, "q1", 1)], target=target))
    assert long["logprob"].tolist() == [-0.5]
    assert long["pred"].tolist() == ["A"]  # a letter, as in public rows


def test_target_letter_picks_its_choice(tmp_path):
    long, _ = read(write_samples(tmp_path / "s.jsonl", [(1, "q1", 0)], target="B"))
    assert long["logprob"].tolist() == [-1.5]


# --- MuSR: text targets, acc_norm, doc_id as source index, subtasks sharing a benchmark ---

MUSR = "leaderboard_musr"


def write_musr(path: Path, docs: list[tuple[str, list[str], int, float]], resps=None) -> Path:
    """One lm-eval MuSR sample per (narrative, choices, answer_index, acc_norm), in file order.

    Shaped like the leaderboard's details: choices stored as a string, target = answer text.
    """
    with open(path, "w") as f:
        for doc_id, (narrative, choices, answer, acc_norm) in enumerate(docs):
            doc = {
                "narrative": narrative,
                "question": "Who is the most likely murderer?",
                "choices": str(choices),
                "answer_index": answer,
                "answer_choice": choices[answer],
            }
            lps = resps or [-1.0 - i for i in range(len(choices))]
            rec = {
                "doc_id": doc_id,
                "doc": doc,
                "target": choices[answer],
                "resps": [[[str(lp), "False"]] for lp in lps],
                "acc_norm": acc_norm,
            }
            f.write(json.dumps(rec) + "\n")
    return path


def read_musr(path: Path, subtask: str = "murder_mysteries"):
    return read_samples(
        path,
        task=f"{MUSR}_{subtask}",
        benchmark=MUSR,
        benchmark_version="1.0",
        run_id="run",
        source="own",
        model_id="org/model",
        model_revision="a" * 40,
        config_id=CFG,
        harness_commit="h" * 40,
        metric="acc_norm",
        timestamp=T1,
    )


def test_musr_text_target_and_items(tmp_path):
    docs = [("story one", ["Mackenzie", "Ana"], 1, 0.0), ("story two", ["Bo", "Cy"], 0, 1.0)]
    long, items = read_musr(write_musr(tmp_path / "s.jsonl", docs))

    assert long["logprob"].tolist() == [-2.0, -1.0]  # the gold choice's log-probability
    assert long["correct"].tolist() == [0, 1]
    assert items["answer_key"].tolist() == ["1", "0"]
    assert items["subtask"].tolist() == ["murder_mysteries"] * 2
    assert items["source_index"].tolist() == ["0", "1"]  # doc_id: MuSR has no ID column
    conform(long, LONG, "long")


def test_musr_same_question_different_narrative_is_two_items(tmp_path):
    docs = [("story one", ["A1", "B1"], 0, 1.0), ("story two", ["A1", "B1"], 0, 1.0)]
    long, _ = read_musr(write_musr(tmp_path / "s.jsonl", docs))
    assert long["item_id"].nunique() == 2


def test_acc_norm_pred_divides_by_choice_length(tmp_path):
    # raw argmax picks "Al" (-2.0 > -3.0); per character, "Bartholomew" wins (-3/11 > -2/2)
    docs = [("story", ["Al", "Bartholomew"], 1, 1.0)]
    long, _ = read_musr(write_musr(tmp_path / "s.jsonl", docs, resps=[-2.0, -3.0]))
    assert long["pred"].tolist() == ["B"]


def test_musr_answer_that_looks_like_a_letter(tmp_path):
    # "X" is a capital letter, but for MuSR the target is always the answer text
    long, _ = read_musr(write_musr(tmp_path / "s.jsonl", [("story", ["X", "Y"], 1, 1.0)]))
    assert long["logprob"].tolist() == [-2.0]


def test_musr_target_not_in_choices_is_an_error(tmp_path):
    path = write_musr(tmp_path / "s.jsonl", [("story", ["X", "Y"], 0, 1.0)])
    path.write_text(path.read_text().replace('"target": "X"', '"target": "Z"'))
    with pytest.raises(ValueError, match="not one of the choices"):
        read_musr(path)


def test_mmlu_pro_letters_unaffected_by_choice_text(tmp_path):
    # an option whose text is "B" must not change how MMLU-Pro's letter target is read
    path = tmp_path / "s.jsonl"
    rec = {
        "doc_id": 0,
        "doc": {"question_id": 1, "question": "q", "options": ["B", "y"], "answer": "B",
                "category": "math"},
        "target": "B",
        "resps": [[["-0.5", True]], [["-1.5", False]]],
        "acc": 0,
    }
    path.write_text(json.dumps(rec) + "\n")
    long, _ = read(path)
    assert long["logprob"].tolist() == [-1.5]


def test_ingest_own_writes_one_file_per_benchmark(tmp_path):
    import subprocess

    run = tmp_path / "run"
    tasks = []
    for i, s in enumerate(("murder_mysteries", "team_allocation")):
        name = f"{MUSR}_{s}_pinned"
        (run / name).mkdir(parents=True)
        write_musr(run / name / "samples.jsonl", [(f"{s} story {i}", ["X", "Y"], 0, 1.0)])
        tasks.append({"name": name, "benchmark": MUSR})
    manifest = {
        "limit": None, "run_id": "r" * 64, "config_id": CFG, "tasks": tasks,
        "benchmark_versions": {t["name"]: "1.0" for t in tasks},
        "model": {"repo": "org/model", "sha": "a" * 40},
        "harness": {"lm_eval_sha": "h" * 40},
        "runtime": {"utc_time": T1.isoformat()},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))

    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "scripts/matrix/ingest_own.py"), str(run),
         "--data", str(tmp_path / "data")],
        check=True, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
    )
    out = list((tmp_path / "data/long").rglob("*.parquet"))
    assert [p.relative_to(tmp_path / "data/long").as_posix() for p in out] == [
        f"{MUSR}/own_{'r' * 64}.parquet"
    ]
    assert len(pd.read_parquet(out[0])) == 2


# --- GPQA: shuffled choices, "(X)" targets, labels scored, three subsets as benchmarks ---

GPQA = "leaderboard_gpqa_main"


def gpqa_rec(doc_id: int, question: str, shown: list[str], gold: int, lps: list[float],
             subdomain: str = "Organic Chemistry") -> dict:
    """One lm-eval GPQA sample: choice1..4 in prompt order, target the gold label "(X)"."""
    label = f"({'ABCD'[gold]})"
    best = max(range(4), key=lps.__getitem__)  # "(A)".."(D)" have equal length
    return {
        "doc_id": doc_id,
        "doc": {"Question": question, "Subdomain": subdomain, "answer": label,
                **{f"choice{i + 1}": c for i, c in enumerate(shown)}},
        "target": label,
        "resps": [[[str(lp), "False"]] for lp in lps],
        "acc_norm": float(best == gold),
    }


def read_gpqa(path: Path, recs: list[dict], benchmark: str = GPQA):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return read_samples(
        path, task=benchmark, benchmark=benchmark, benchmark_version="1.0", run_id="run",
        source="own", model_id="org/model", model_revision="a" * 40, config_id=CFG,
        harness_commit="h" * 40, metric="acc_norm", timestamp=T1,
    )


def test_gpqa_item_id_ignores_the_shuffle(tmp_path):
    # the same question shuffled two ways, as different harness commits do; the model
    # picks "wrong 2" in the first run and the gold answer in the second
    a = gpqa_rec(0, "q", ["wrong 2", "gold", "wrong 1", "wrong 3"], 1, [-1.0, -2.0, -3.0, -4.0])
    b = gpqa_rec(0, "q", ["wrong 3", "wrong 1", "wrong 2", "gold"], 3, [-4.0, -3.0, -2.0, -1.0])
    long_a, items_a = read_gpqa(tmp_path / "a.jsonl", [a])
    long_b, items_b = read_gpqa(tmp_path / "b.jsonl", [b])

    assert long_a["item_id"].tolist() == long_b["item_id"].tolist()
    assert items_a["answer_key"].tolist() == items_b["answer_key"].tolist() == ["A"]
    assert items_a["subtask"].tolist() == ["Organic Chemistry"]
    assert long_a["logprob"].tolist() == [-2.0] and long_b["logprob"].tolist() == [-1.0]
    # choices in item order: gold, wrong 1, wrong 2, wrong 3
    assert long_a["pred"].tolist() == ["C"] and long_a["correct"].tolist() == [0]
    assert long_b["pred"].tolist() == ["A"] and long_b["correct"].tolist() == [1]
    conform(long_a, LONG, "long")


def test_gpqa_identical_choice_texts_keep_pred_and_correct_consistent(tmp_path):
    # the leaderboard's preprocessing can leave all four answers reading the same
    recs = [gpqa_rec(0, "q", ["ln(2) = "] * 4, 0, [-2.0, -1.0, -3.0, -4.0]),
            gpqa_rec(1, "q2", ["ln(2) = "] * 4, 0, [-1.0, -2.0, -3.0, -4.0])]
    long, items = read_gpqa(tmp_path / "s.jsonl", recs)
    assert long["correct"].tolist() == [0, 1]
    assert long["pred"].tolist() == ["B", "A"]  # == answer_key "A" exactly when correct


def test_gpqa_subsets_are_separate_benchmarks(tmp_path):
    rec = gpqa_rec(0, "q", ["gold", "w1", "w2", "w3"], 0, [-1.0, -2.0, -3.0, -4.0])
    main, _ = read_gpqa(tmp_path / "m.jsonl", [rec], "leaderboard_gpqa_main")
    diamond, _ = read_gpqa(tmp_path / "d.jsonl", [rec], "leaderboard_gpqa_diamond")
    assert main["item_id"].tolist() != diamond["item_id"].tolist()
    assert set(main["benchmark"]) == {"leaderboard_gpqa_main"}
