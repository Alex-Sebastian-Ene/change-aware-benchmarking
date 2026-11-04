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


def write_samples(path: Path, docs: list[tuple[int, str, int]]) -> Path:
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
                "target": 0,
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
