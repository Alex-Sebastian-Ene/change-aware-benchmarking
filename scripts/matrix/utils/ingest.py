"""lm-eval `--log_samples` output -> long-table rows + items-table rows.

lm-eval writes one samples_<task>_<timestamp>.jsonl per task, one JSON line
per document with keys including doc_id, doc, target, resps, filtered_resps
and one key per metric (acc, acc_norm, exact_match, ...).

Each benchmark stores its question and choices under different `doc` fields,
so the content used for the item ID is pulled by a small per-task extractor.
MMLU-Pro is filled in as the worked example; the rest are TODO and must be
checked against the doc fields at the pinned harness commit before use.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .item_id import make_item_id
from .schema import SCHEMA_VERSION

# task -> fn(doc) -> (question, choices, answer_key, subtask)
Extractor = Callable[[dict], tuple[str, list[str], str, str]]


def _mmlu_pro(doc: dict):
    """
    Args:
        doc (dict): The sample's "doc" entry.

    Returns:
        tuple[str, list[str], str, str]: (question, choices, answer_key, subtask).
    """
    return doc["question"], list(doc["options"]), str(doc["answer"]), doc.get("category", "")


EXTRACTORS: dict[str, Extractor] = {
    "leaderboard_mmlu_pro": _mmlu_pro,
    # TODO: more benchmarks
}

LOGLIK_METRICS = ("acc_norm", "acc")
GEN_METRICS = ("exact_match", "prompt_level_strict_acc")


def _score(rec: dict, metric: str) -> int:
    """
    Args:
        rec (dict): One parsed line of a samples JSONL file.
        metric (str): Metric key to read, e.g. "acc".

    Returns:
        int: 0 or 1.
    """
    v = rec[metric]
    if isinstance(v, bool) or v in (0, 1, 0.0, 1.0):
        return int(v)
    raise ValueError(f"doc {rec.get('doc_id')}: {metric}={v!r} is not 0/1")


def read_samples(
    path: str | Path,
    *,
    task: str,
    benchmark: str,
    benchmark_version: str,
    run_id: str,
    source: str,
    model_id: str,
    model_revision: str,
    config_id: str,
    harness_commit: str,
    metric: str,
    count_tokens: Callable[[str], int] | None = None,
    timestamp: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Args:
        path (str | Path): The samples_<task>_<timestamp>.jsonl file.
        task (str): lm-eval task name; selects the extractor in EXTRACTORS.
        benchmark (str): Benchmark name for the item ID and long table.
        benchmark_version (str): Benchmark version for the item ID and long table.
        run_id (str): ID of this run.
        source (str): "public" or "own".
        model_id (str): Hub model name.
        model_revision (str): Model commit hash.
        config_id (str): ID of the pinned harness config used.
        harness_commit (str): lm-eval commit hash.
        metric (str): Metric key giving `correct`; a GEN_METRICS entry marks the
            task as generative (no logprob, pred is the generated text).
        count_tokens (Callable[[str], int] | None): Tokeniser-based counter for
            prompt_tokens and gen_tokens; if None, prompt_tokens is null and
            gen_tokens is 0.
        timestamp (datetime | None): Timestamp for every row; defaults to now (UTC).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (long rows, one per sample;
        item rows, de-duplicated on item_id).
    """
    if task not in EXTRACTORS:
        raise NotImplementedError(f"no extractor for {task}; add one to EXTRACTORS")
    extract = EXTRACTORS[task]
    ts = timestamp or datetime.now(timezone.utc)
    long_rows, item_rows = [], []

    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        q, choices, answer, subtask = extract(rec["doc"])
        iid = make_item_id(benchmark, benchmark_version, q, choices)
        generative = metric in GEN_METRICS

        logprob = pred = None
        gen_tokens = 0
        if generative:
            pred = rec["filtered_resps"][0] if rec.get("filtered_resps") else None
            gen_tokens = count_tokens(pred) if (count_tokens and pred) else 0
        else:
            lps = [float(r[0][0]) for r in rec["resps"]]
            gold = int(rec["target"])
            logprob = lps[gold]
            pred = str(max(range(len(lps)), key=lps.__getitem__))

        prompt = rec["arguments"][0][0] if rec.get("arguments") else None
        long_rows.append(dict(
            run_id=run_id, source=source, model_id=model_id, model_revision=model_revision,
            benchmark=benchmark, benchmark_version=benchmark_version, item_id=iid,
            config_id=config_id, harness_commit=harness_commit,
            correct=_score(rec, metric), logprob=logprob, pred=pred,
            prompt_tokens=count_tokens(prompt) if (count_tokens and prompt) else None,
            gen_tokens=gen_tokens, timestamp=ts, schema_version=SCHEMA_VERSION))
        item_rows.append(dict(
            item_id=iid, benchmark=benchmark, benchmark_version=benchmark_version,
            subtask=subtask, source_index=str(rec["doc_id"]), answer_key=answer,
            ref_prompt_tokens=None))

    return pd.DataFrame(long_rows), pd.DataFrame(item_rows).drop_duplicates("item_id")