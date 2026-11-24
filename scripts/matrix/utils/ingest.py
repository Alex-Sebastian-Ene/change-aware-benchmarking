"""lm-eval per-document records -> long-table rows + items-table rows.

Used for both sources, so public and own rows get item IDs, scores and preds from
the same code:
- own runs: lm-eval's `--log_samples` output, one JSON line per document
  (read_jsonl; run.py saves it as <task>/samples.jsonl);
- public runs: the Open LLM Leaderboard details parquet for one task, one row per
  document (read_details; download_details.py saves it as <model>__<task>.parquet).
Both have doc_id, doc, target, resps, arguments and one key per metric (acc,
acc_norm, exact_match, ...). They differ only in how `arguments` is nested.

Each benchmark stores its question and choices under different `doc` fields,
so the content used for the item ID is pulled by a small per-task extractor.
MMLU-Pro, MuSR and GPQA are filled in; the rest are TODO and must be checked against
the doc fields at the pinned harness commit before use. EXTRACTORS is keyed by
lm-eval task name (without the _pinned suffix), so a benchmark made of several
subtasks has one entry per subtask.

Identical questions get the same item ID. Only the one with the lowest source
index is kept, so public and own runs keep the same copy.

BENCHMARKS lists, per benchmark, the lm-eval tasks it is made of and the metric
that gives `correct`. ingest_own.py, ingest_public.py and build_tables.py all
read it, so adding a benchmark means adding its extractor(s) and one entry there.
"""
from __future__ import annotations

import ast
import json
import string
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd

from .item_id import make_item_id, normalise
from .schema import SCHEMA_VERSION

# task -> fn(doc) -> (question, choices, answer_key, subtask, source_index)
# source_index None = the dataset has no ID column; the sample's doc_id is used instead.
Extractor = Callable[[dict], tuple[str, list[str], str, str, int | None]]


def _mmlu_pro(doc: dict):
    """
    Args:
        doc (dict): The sample's "doc" entry.

    Returns:
        tuple[str, list[str], str, str, int]: (question, choices, answer_key, subtask,
        source_index). source_index is the dataset's question_id, as in ingest_public.py.
    """
    return (doc["question"], list(doc["options"]), str(doc["answer"]), doc.get("category", ""),
            int(doc["question_id"]))


def _musr(doc: dict, subtask: str):
    """
    Args:
        doc (dict): The sample's "doc" entry.
        subtask (str): The MuSR split, e.g. "murder_mysteries"; the doc doesn't record it.

    Returns:
        tuple[str, list[str], str, str, None]: (question, choices, answer_key, subtask,
        None). The question alone repeats ("Who is the most likely murderer?"), so the
        narrative is part of it. MuSR stores choices as a string of a Python list. It has
        no ID column, so source_index is None and read_samples uses doc_id.
    """
    return (doc["narrative"] + "\n\n" + doc["question"], ast.literal_eval(doc["choices"]),
            str(doc["answer_index"]), subtask, None)


MUSR_SUBTASKS = ("murder_mysteries", "object_placements", "team_allocation")


def _gpqa_shown(doc: dict) -> list[str]:
    """GPQA's four answers in the order the prompt shows them, (A) to (D)."""
    return [str(doc[f"choice{i}"]) for i in range(1, 5)]


def _gpqa_order(doc: dict) -> list[int]:
    """Prompt positions (0-3) in the order _gpqa returns the choices: gold first, then
    the others by normalised text, ties by position.

    Positions, not texts, because some answers are identical as shown: the leaderboard's
    preprocessing deletes "[...]", which leaves e.g. four answers reading "ln(2) = ".
    """
    shown = _gpqa_shown(doc)
    gold = _gold_index(doc["answer"])
    others = sorted((i for i in range(4) if i != gold), key=lambda i: (normalise(shown[i]), i))
    return [gold, *others]


def _gpqa(doc: dict):
    """
    Args:
        doc (dict): The sample's "doc" entry, after the leaderboard task's process_docs,
            which shuffles the four answers into choice1..choice4 and sets `answer` to
            the gold letter in brackets, e.g. "(D)".

    Returns:
        tuple[str, list[str], str, str, None]: (question, choices, answer_key, subtask,
        None). The choices are in a fixed order that doesn't depend on the shuffle: the
        gold answer, then the other three sorted by their normalised text. So the gold
        is always choice "A", answer_key is "A", and read_records maps pred into this
        order (see CHOICE_ORDER). subtask is the dataset's Subdomain, e.g. "Organic
        Chemistry". GPQA's Record ID isn't an integer, so source_index is None and
        doc_id is used.

    The shuffle can't be part of the item ID, as the spec's "choices in the order shown"
    would make it: it depends on the harness run, not the question. lm-eval draws it
    from one random state shared by the GPQA subsets, so it changes with the order the
    tasks are built in. In the leaderboard details, main has the same shuffle in every
    run checked, but diamond and extended each have three, one per group of harness
    commits (d4631ea6; 9694c56; a791d293 and a781a6b). With shown order, one question
    would become up to three items, and our own runs would match none of them.
    """
    shown = _gpqa_shown(doc)
    return (doc["Question"], [shown[i] for i in _gpqa_order(doc)], "A",
            doc.get("Subdomain", ""), None)


# The leaderboard's three GPQA subsets are nested (diamond 198 < main 448 < extended
# 546 questions). Each subset is its own benchmark: combined, the same question would
# count up to three times.
GPQA_SUBSETS = ("diamond", "main", "extended")

EXTRACTORS: dict[str, Extractor] = {
    "leaderboard_mmlu_pro": _mmlu_pro,
    **{f"leaderboard_musr_{s}": partial(_musr, subtask=s) for s in MUSR_SUBTASKS},
    **{f"leaderboard_gpqa_{s}": _gpqa for s in GPQA_SUBSETS},
    # TODO: more benchmarks
}

# For acc_norm, lm-eval divides each log-probability by the length in characters of
# the choice it scored, i.e. doc_to_choice's string. For most tasks that is the
# extractor's choices; tasks listed here score other strings, given by doc. GPQA
# scores the labels "(A)".."(D)", not the answer texts the item ID is built from.
SCORED_CHOICES: dict[str, Callable[[dict], list[str]]] = {
    f"leaderboard_gpqa_{s}": (lambda doc: ["(A)", "(B)", "(C)", "(D)"]) for s in GPQA_SUBSETS
}

# Tasks whose extractor puts the choices in a different order from the prompt. `resps`
# and `target` follow the prompt's order; the function gives, from doc, the prompt
# position of each choice in the extractor's order. pred is mapped through it, so that
# pred == answer_key exactly when the answer is correct.
CHOICE_ORDER: dict[str, Callable[[dict], list[int]]] = {
    f"leaderboard_gpqa_{s}": _gpqa_order for s in GPQA_SUBSETS
}

# Tasks whose doc_to_target is the answer text (MuSR: `answer_choice`), so lm-eval logs
# `target` as that text. Listed explicitly rather than guessed from the value, because
# an answer text can look like a letter or digit ("A", "3").
TEXT_TARGET_TASKS = frozenset(f"leaderboard_musr_{s}" for s in MUSR_SUBTASKS)



@dataclass(frozen=True)
class Benchmark:
    tasks: tuple[str, ...]  # lm-eval task names (no _pinned suffix), each in EXTRACTORS
    metric: str  # the lm-eval metric that gives `correct`, as the leaderboard scored it


BENCHMARKS: dict[str, Benchmark] = {
    "leaderboard_mmlu_pro": Benchmark(("leaderboard_mmlu_pro",), "acc"),
    "leaderboard_musr": Benchmark(tuple(f"leaderboard_musr_{s}" for s in MUSR_SUBTASKS),
                                  "acc_norm"),
    **{f"leaderboard_gpqa_{s}": Benchmark((f"leaderboard_gpqa_{s}",), "acc_norm")
       for s in GPQA_SUBSETS},
}

LOGLIK_METRICS = ("acc_norm", "acc")
LETTERS = string.ascii_uppercase  # pred is stored as a letter, in public and own rows


def _gold_index(target) -> int:
    """Index of the correct choice from a sample's `target`.

    lm-eval logs `target` as doc_to_target returns it: an int for most tasks, but the
    answer letter ("A"-"J") for MMLU-Pro, whose doc_to_target is the `answer` field,
    and the letter in brackets ("(A)"-"(D)") for GPQA.
    Tasks in TEXT_TARGET_TASKS log the answer text instead and don't come here.
    """
    if isinstance(target, int) and not isinstance(target, bool):
        return target
    t = str(target).strip()
    if t.isdigit():
        return int(t)
    if len(t) == 3 and t[0] == "(" and t[2] == ")":
        t = t[1]
    if len(t) == 1 and t in LETTERS:
        return LETTERS.index(t)
    raise ValueError(f"can't read target {target!r} as a choice index or letter")


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


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """
    Args:
        path (str | Path): An lm-eval samples JSONL file (own runs).

    Returns:
        Iterator[dict]: One record per line.
    """
    for line in Path(path).read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def read_details(path: str | Path) -> Iterator[dict]:
    """
    Args:
        path (str | Path): One task's Open LLM Leaderboard details parquet, as
            download_details.py saves it (public runs).

    Returns:
        Iterator[dict]: One record per row, with the same keys as a samples JSONL
        line. List fields come back as numpy arrays, which the readers here accept.
    """
    yield from pd.read_parquet(path).to_dict("records")


def _prompt(rec: dict) -> str | None:
    """The prompt (context) of a record, or None if it has no `arguments`.

    Own samples log `arguments` as [[context, continuation], ...]; the leaderboard
    details store it as {"gen_args_0": {"arg_0": context, "arg_1": ...}, ...}.
    """
    args = rec.get("arguments")
    if args is None:
        return None
    if isinstance(args, dict):
        return args["gen_args_0"]["arg_0"]
    return args[0][0] if len(args) else None


def _item(rec: dict, task: str, benchmark: str, benchmark_version: str):
    """
    Args:
        rec (dict): One record (read_jsonl or read_details).
        task (str): lm-eval task name without the _pinned suffix.
        benchmark (str): Benchmark name for the item ID.
        benchmark_version (str): Benchmark version for the item ID.

    Returns:
        tuple[str, list[str], dict, int]: (item_id, choices, items-table row,
        source_index).
    """
    q, choices, answer, subtask, source_index = EXTRACTORS[task](rec["doc"])
    if source_index is None:
        source_index = int(rec["doc_id"])
    choices = [str(c) for c in choices]
    iid = make_item_id(benchmark, benchmark_version, q, choices)
    row = dict(item_id=iid, benchmark=benchmark, benchmark_version=benchmark_version,
               subtask=subtask, source_index=str(source_index), answer_key=answer,
               ref_prompt_tokens=None)
    return iid, choices, row, int(source_index)


def _keep_lowest(rows: list[dict], source_indices: list[int]) -> pd.DataFrame:
    """Rows deduplicated by item_id, keeping the lowest source index, in input order."""
    keep = pd.Series(source_indices).sort_values(kind="stable").index
    return pd.DataFrame(rows).loc[keep].drop_duplicates("item_id").sort_index() \
        .reset_index(drop=True)


def _check_task(task: str) -> None:
    if task not in EXTRACTORS:
        raise NotImplementedError(f"no extractor for {task}; add one to EXTRACTORS")


def read_items(
    records: Iterable[dict], *, task: str, benchmark: str, benchmark_version: str
) -> pd.DataFrame:
    """
    Args:
        records (Iterable[dict]): One task's records (read_jsonl or read_details).
        task (str): lm-eval task name without the _pinned suffix.
        benchmark (str): Benchmark name for the item ID.
        benchmark_version (str): Benchmark version for the item ID.

    Returns:
        pd.DataFrame: Items-table rows, one per distinct item, as read_records gives
        them. Of identical questions, the one with the lowest source index is kept.
    """
    _check_task(task)
    rows, source_indices = [], []
    for rec in records:
        _, _, row, source_index = _item(rec, task, benchmark, benchmark_version)
        rows.append(row)
        source_indices.append(source_index)
    return _keep_lowest(rows, source_indices)


def read_records(
    records: Iterable[dict],
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
    timestamp: datetime,
    count_tokens: Callable[[str], int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Args:
        records (Iterable[dict]): One task's records (read_jsonl or read_details).
        task (str): lm-eval task name without the _pinned suffix; selects the
            extractor in EXTRACTORS.
        benchmark (str): Benchmark name for the item ID and long table.
        benchmark_version (str): Benchmark version for the item ID and long table.
        run_id (str): ID of this run.
        source (str): "public" or "own".
        model_id (str): Hub model name.
        model_revision (str): Model commit hash.
        config_id (str): ID of the pinned harness config used.
        harness_commit (str): lm-eval commit hash.
        metric (str): Metric key giving `correct`; a GEN_METRICS entry marks the
            task as generative (no logprob, pred is the generated text). For
            "acc_norm", pred is chosen as lm-eval does: each log-probability divided
            by its choice's length in characters.
        timestamp (datetime): When the run happened, with a timezone, e.g.
            datetime.fromisoformat(manifest["runtime"]["utc_time"]). The earliest run of
            a model and config becomes its matrix row, so this must be the run's time,
            not the time of ingest.
        count_tokens (Callable[[str], int] | None): Tokeniser-based counter for
            prompt_tokens and gen_tokens; if None, prompt_tokens is null and
            gen_tokens is 0.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (long rows, one per distinct item;
        item rows, one per distinct item). Of identical questions, the one with the
        lowest source index is kept.
    """
    _check_task(task)
    if timestamp.tzinfo is None:
        raise ValueError("timestamp needs a timezone, e.g. the manifest's runtime.utc_time")
    generative = metric in GEN_METRICS
    long_rows, item_rows, source_indices = [], [], []

    for rec in records:
        iid, choices, item_row, source_index = _item(rec, task, benchmark, benchmark_version)

        logprob = pred = None
        gen_tokens = 0
        if generative:
            fr = rec.get("filtered_resps")
            pred = fr[0] if fr is not None and len(fr) else None
            gen_tokens = count_tokens(pred) if (count_tokens and pred) else 0
        else:
            lps = [float(r[0][0]) for r in rec["resps"]]
            if task in TEXT_TARGET_TASKS:
                if str(rec["target"]) not in choices:
                    raise ValueError(f"doc {rec.get('doc_id')}: target {rec['target']!r} "
                                     f"is not one of the choices {choices}")
                gold = choices.index(str(rec["target"]))
            else:
                gold = _gold_index(rec["target"])
            logprob = lps[gold]
            # acc_norm compares log-probabilities per character of the choice
            # (lm_eval/api/task.py at the pinned commit), so pred must too, or it
            # can disagree with `correct`. The strings lm-eval scored are the
            # extractor's choices (as for _musr) unless the task is in SCORED_CHOICES.
            if metric == "acc_norm":
                scored = SCORED_CHOICES[task](rec["doc"]) if task in SCORED_CHOICES else choices
                scores = [lp / len(c) for lp, c in zip(lps, scored, strict=True)]
            else:
                scores = lps
            best = max(range(len(scores)), key=scores.__getitem__)
            if task in CHOICE_ORDER:
                best = CHOICE_ORDER[task](rec["doc"]).index(best)
            pred = LETTERS[best]

        prompt = _prompt(rec)
        long_rows.append(dict(
            run_id=run_id, source=source, model_id=model_id, model_revision=model_revision,
            benchmark=benchmark, benchmark_version=benchmark_version, item_id=iid,
            config_id=config_id, harness_commit=harness_commit,
            correct=_score(rec, metric), logprob=logprob, pred=pred,
            prompt_tokens=count_tokens(prompt) if (count_tokens and prompt) else None,
            gen_tokens=gen_tokens, timestamp=timestamp, schema_version=SCHEMA_VERSION))
        item_rows.append(item_row)
        source_indices.append(source_index)

    return _keep_lowest(long_rows, source_indices), _keep_lowest(item_rows, source_indices)


def read_samples(path: str | Path, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """read_records on an own run's samples JSONL file; takes read_records' keywords."""
    return read_records(read_jsonl(path), **kwargs)
