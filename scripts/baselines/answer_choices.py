"""Which answer option models pick, question by question, on MMLU.

efficbench's matrix only says right or wrong. The Open LLM Leaderboard's public details also
hold each model's log-likelihood for options A-D, so this fetches them for a sample of
models spread across the score range and joins the answer key from cais/mmlu (both public).
Every file is checked against the matrix before it is used.

Run from the repository root, after scripts/irt/efficbench.py:

    uv run python scripts/baselines/answer_choices.py

Writes data/processed/mmlu_choices.parquet (one row per model and question) and
results/baselines/{choices_confusion,choices_difficulty,choices_checks}.csv and choices.json.
"""

import io
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "data/processed/efficbench_mmlu"
PICKS = ROOT / "data/processed/mmlu_choices.parquet"
PARTS = ROOT / "data/processed/mmlu_choices_parts"  # one file per model, so reruns resume
OUT = ROOT / "results/baselines"

N_MODELS = 24
HUB = "https://huggingface.co"
DETAILS = "open-llm-leaderboard-old/details_{}"
LETTERS = list("ABCD")
BINS = np.arange(0, 1.01, 0.1)


def fetch(url, tries=6):
    """GET with backoff: the Hub rate-limits anonymous requests."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
        time.sleep(5 * 2**attempt)


def quoted(path):
    return urllib.request.quote(path)


def sample_models(Y, models):
    """N_MODELS model indices evenly spaced through the ranking by full score."""
    order = np.argsort(Y.mean(axis=1))
    return order[np.linspace(0, len(order) - 1, N_MODELS).round().astype(int)]


def answer_key(subjects):
    """(subject, question, options) -> index of the correct option, from cais/mmlu."""

    def one(subject):
        url = f"{HUB}/datasets/cais/mmlu/resolve/main/{subject}/test-00000-of-00001.parquet"
        table = pd.read_parquet(io.BytesIO(fetch(url)))
        table["subject"] = subject
        return table

    with ThreadPoolExecutor(8) as pool:
        table = pd.concat(pool.map(one, subjects), ignore_index=True)
    table["key"] = [
        prompt_key(q, list(c)) for q, c in zip(table["question"], table["choices"], strict=True)
    ]
    return table[["subject", "key", "answer"]]


def last_block(text):
    """The text after its last blank line, whitespace collapsed.

    Prompts join few-shot examples with blank lines, and a question can hold one too, so
    both sides of the join are cut the same way.
    """
    return " ".join(text.rsplit("\n\n", 1)[-1].split())


def prompt_key(question, options):
    """cais/mmlu's question and options, written the way a leaderboard prompt writes them."""
    lines = [f"{letter}. {str(o).strip()}" for letter, o in zip(LETTERS, options, strict=True)]
    return last_block("\n".join([question.strip(), *lines]))


def detail_key(prompt):
    """The same key from a leaderboard prompt."""
    return last_block(prompt.removesuffix("Answer:"))


def read_answers(raw):
    """(table, acc) from a details file: older files nest acc inside `metrics`."""
    file = pq.ParquetFile(io.BytesIO(raw))
    flat = "acc" in file.schema_arrow.names
    columns = ["full_prompt", "predictions", "acc" if flat else "metrics"]
    table = file.read(columns=columns).to_pandas()
    acc = table["acc"] if flat else [m["acc"] for m in table["metrics"]]
    return table, np.array(acc, dtype=float)


def model_details(name, row, items):
    """One model's answers, or None if its public files do not match the matrix."""
    safe = name.replace("/", "__")
    cached = PARTS / f"{safe}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    repo = DETAILS.format(safe)
    try:
        info = json.loads(fetch(f"{HUB}/api/datasets/{repo}"))
    except urllib.error.HTTPError as e:
        print(f"  skip {name}: HTTP {e.code}", flush=True)
        return None
    files = sorted((s["rfilename"] for s in info["siblings"]), reverse=True)  # newest first

    def subject_rows(subject, positions):
        want = row[positions].astype(float)
        tag = f"hendrycksTest-{subject}|5_"
        for path in (f for f in files if tag in f and f.endswith(".parquet")):
            url = f"{HUB}/datasets/{repo}/resolve/main/{quoted(path)}"
            table, acc = read_answers(fetch(url))
            if len(acc) == len(want) and (acc == want).all():
                return pd.DataFrame(
                    {
                        "model": name,
                        "subject": subject,
                        "item_idx": positions,
                        "key": table["full_prompt"].map(detail_key),
                        "pick": [int(np.argmax(p)) for p in table["predictions"]],
                        "correct": acc.astype(int),
                    }
                )
        return None

    parts = [subject_rows(s, g.index.to_numpy()) for s, g in items.groupby("subject", sort=False)]
    if any(p is None for p in parts):
        missing = sum(p is None for p in parts)
        print(f"  skip {name}: {missing} subjects do not match the matrix", flush=True)
        return None
    print(f"  ok   {name}", flush=True)
    answers = pd.concat(parts, ignore_index=True)
    answers.to_parquet(cached)
    return answers


def download(Y, models, items):
    """Fetch the sample, keeping only what the analysis needs."""
    PARTS.mkdir(parents=True, exist_ok=True)
    chosen = sample_models(Y, models)
    names = models["model_id"].to_numpy()[chosen]
    with ThreadPoolExecutor(4) as pool:
        tables = list(
            pool.map(lambda c: model_details(names[c[0]], Y[c[1]], items), enumerate(chosen))
        )
    picks = pd.concat([t for t in tables if t is not None], ignore_index=True)
    key = answer_key(items["subject"].unique())
    # identical (question, options) pairs repeat within MMLU; they share an answer unless flagged
    conflicts = key.groupby(["subject", "key"])["answer"].nunique().gt(1).sum()
    print(f"questions with conflicting answers in cais/mmlu: {conflicts}")
    key = key.drop_duplicates(["subject", "key"])
    picks = picks.merge(key, on=["subject", "key"], how="left", validate="many_to_one")
    picks = picks.rename(columns={"answer": "gold"}).drop(columns="key")
    picks["gold"] = picks["gold"].astype("Int8")
    picks["pick"] = picks["pick"].astype("int8")
    PICKS.parent.mkdir(parents=True, exist_ok=True)
    picks.to_parquet(PICKS)
    return picks


def fleiss_kappa(counts):
    """Agreement between models on each question beyond chance (questions x options counts)."""
    n = counts.sum(axis=1)
    agree = (counts * (counts - 1)).sum(axis=1) / (n * (n - 1))
    shares = counts.sum(axis=0) / counts.sum()
    chance = (shares**2).sum()
    return (agree.mean() - chance) / (1 - chance)


def distractors_by_difficulty(picks, difficulty):
    """Share of answers on the correct option and on each wrong option, ranked out of sample.

    Wrong options are ranked on half the models and measured on the other half: ranking and
    measuring on the same answers would make the top option look popular by chance.
    """
    models = picks["model"].unique()
    fit = picks[picks["model"].isin(models[::2])]
    held = picks[picks["model"].isin(models[1::2])]
    votes = fit.groupby(["item_idx", "pick"]).size().unstack(fill_value=0)
    votes = votes.reindex(columns=range(4), fill_value=0)
    gold = picks.groupby("item_idx")["gold"].first().astype(int).reindex(votes.index)

    wrong = votes.to_numpy(dtype=float, copy=True)
    wrong[np.arange(len(wrong)), gold.to_numpy()] = -1
    # small random tie-break so that tied options do not always go to the earliest letter
    wrong += np.random.default_rng(0).random(wrong.shape) * 1e-3
    ranks = np.argsort(-wrong, axis=1)  # columns: wrong options, most picked first, then gold

    answered = held.groupby(["item_idx", "pick"]).size().unstack(fill_value=0)
    answered = answered.reindex(index=votes.index, columns=range(4), fill_value=0).to_numpy()
    rows = np.arange(len(answered))
    table = pd.DataFrame(
        {
            "correct": answered[rows, gold.to_numpy()],
            "top_wrong": answered[rows, ranks[:, 0]],
            "second_wrong": answered[rows, ranks[:, 1]],
            "third_wrong": answered[rows, ranks[:, 2]],
        },
        index=votes.index,
    )
    table["bin"] = pd.cut(difficulty.reindex(table.index), BINS, include_lowest=True)
    summed = table.groupby("bin", observed=True).sum()
    out = summed.div(summed.sum(axis=1), axis=0)
    out["n_questions"] = table.groupby("bin", observed=True).size()
    out["mid"] = [interval.mid for interval in out.index]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Y = np.load(MATRIX / "responses.npy")
    models = pd.read_csv(MATRIX / "models.csv", index_col="model_idx")
    items = pd.read_csv(MATRIX / "items.csv", index_col="item_idx")

    picks = pd.read_parquet(PICKS) if PICKS.exists() else download(Y, models, items)

    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    used = picks["model"].nunique()
    expected = used * Y.shape[1]
    check("every question present", len(picks) == expected, f"{len(picks)} answers, {used} models")
    joined = picks["gold"].notna().mean()
    check("every question has an answer key", joined == 1, f"{joined:.2%} joined to cais/mmlu")
    known = picks.dropna(subset="gold")
    agreement = ((known["pick"] == known["gold"]) == (known["correct"] == 1)).mean()
    check(
        "top option == answer key reproduces the score",
        agreement > 0.999,
        f"{agreement:.3%} of answers; proves the option order and the key",
    )

    known = known.assign(gold=known["gold"].astype(int), pick=known["pick"].astype(int))
    confusion = pd.crosstab(known["gold"], known["pick"], normalize="index")
    confusion.index, confusion.columns = LETTERS, LETTERS
    confusion.to_csv(OUT / "choices_confusion.csv", index_label="correct_option")
    print(confusion.round(3))

    counts = known.groupby(["item_idx", "pick"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(4), fill_value=0).to_numpy(dtype=float)
    kappa = fleiss_kappa(counts)
    picked = known["pick"].value_counts(normalize=True).sort_index()
    gold_share = known.drop_duplicates("item_idx")["gold"].value_counts(normalize=True).sort_index()

    difficulty = pd.Series(Y.mean(axis=0), index=items.index, name="difficulty")
    by_difficulty = distractors_by_difficulty(known, difficulty)
    by_difficulty.to_csv(OUT / "choices_difficulty.csv")
    print(by_difficulty.round(3))

    wrong = known[known["correct"] == 0]
    summary = {
        "n_models": int(used),
        "n_answers": int(len(known)),
        "models": sorted(known["model"].unique().tolist()),
        "accuracy": float(known["correct"].mean()),
        "fleiss_kappa": float(kappa),
        "pick_share": {LETTERS[i]: float(v) for i, v in picked.items()},
        "gold_share": {LETTERS[i]: float(v) for i, v in gold_share.items()},
        "n_wrong": int(len(wrong)),
    }
    (OUT / "choices.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(checks).to_csv(OUT / "choices_checks.csv", index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    main()
