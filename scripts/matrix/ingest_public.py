"""Convert downloaded Open LLM Leaderboard MMLU-Pro details into the long table.

Input:  one model's folder from download_details.py
Output: <data>/long/leaderboard_mmlu_pro/<run_id>.parquet, in the spec's
        "Long response table" format, with duplicate questions removed.

Usage (login node: it fetches the run's results JSON and the model's tokenizer):
    python ingest_public.py /scratch/$USER/leaderboard_raw/Qwen__Qwen2.5-0.5B \
        --data /scratch/$USER/data
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.item_id import make_item_id   # item ID rule

TASK = "leaderboard_mmlu_pro"
CONFIG_ID = "openllm_v2_public"   # placeholder
SCHEMA_VERSION = 1
LETTERS = string.ascii_uppercase

LONG_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("source", pa.string()),
    ("model_id", pa.string()),
    ("model_revision", pa.string()),
    ("benchmark", pa.string()),
    ("benchmark_version", pa.string()),
    ("item_id", pa.string()),
    ("config_id", pa.string()),
    ("harness_commit", pa.string()),
    ("correct", pa.int8()),
    ("logprob", pa.float32()),
    ("pred", pa.string()),
    ("prompt_tokens", pa.int32()),
    ("gen_tokens", pa.int32()),
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("schema_version", pa.int16()),
])


def load_results_json(repo: str, safe: str, stamp: str, model_dir: Path) -> dict:
    """
    Args:
        repo (str): Hub dataset repo holding the details, e.g.
            "open-llm-leaderboard/Qwen__Qwen2.5-0.5B-details".
        safe (str): Model ID with "/" replaced by "__", the folder name inside the repo.
        stamp (str): Run timestamp as it appears in the split name,
            e.g. "2024_06_26T12_00_00.123456".
        model_dir (Path): The model's folder from download_details.py, used as the cache.

    Returns:
        dict: The parsed results JSON (keys include "config", "versions", "git_hash").
    """
    local = model_dir / f"results_{stamp}.json"
    if not local.exists():
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo, f"{safe}/results_{stamp.replace('_', '-')}.json", repo_type="dataset")
        local.write_text(Path(path).read_text())
    return json.loads(local.read_text())


def model_revision(res: dict) -> str | None:
    """
    Args:
        res (dict): Parsed results JSON, as returned by load_results_json.

    Returns:
        str | None: The model revision, or None if the results do not record one.
    """
    cfg = res.get("config", {}) or {}
    for key in ("model_sha", "model_revision", "revision"):
        if cfg.get(key):
            return str(cfg[key])
    m = re.search(r"revision=([0-9a-f]{7,40})", str(cfg.get("model_args", "")))
    return m.group(1) if m else None


def main() -> None:
    """
    Command-line args:
        model_dir (str): One model's folder from download_details.py.
        --data (str): Parquet store root; output goes to <data>/long/<TASK>/<run_id>.parquet.

    Returns:
        None. Writes one Parquet file and prints a summary.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    model_id, repo = manifest["model"], manifest["repo"]
    safe = model_id.replace("/", "__")
    cfg = f"{safe}__{TASK}"
    raw = pd.read_parquet(model_dir / f"{cfg}.parquet")

    info = manifest["configs"][cfg]
    stamp = sorted(s for s in info["all_splits"] if s not in ("latest", "train"))[-1]
    res = load_results_json(repo, safe, stamp, model_dir)
    version = str(res["versions"][TASK])
    revision = model_revision(res)

    docs = list(raw["doc"])
    lps = [np.array([float(p[0]) for p in r]) for r in raw["filtered_resps"]]
    gold = [LETTERS.index(str(t).strip()) for t in raw["target"]]
    pred = [int(np.argmax(lp)) for lp in lps]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    prompts = [a["gen_args_0"]["arg_0"] for a in raw["arguments"]]
    prompt_tokens = [len(x) for x in tok(prompts, add_special_tokens=False)["input_ids"]]

    long = pd.DataFrame({
        "run_id": f"public_{safe}_{stamp}",
        "source": "public",
        "model_id": model_id,
        "model_revision": revision,
        "benchmark": TASK,
        "benchmark_version": version,
        "item_id": [make_item_id(TASK, version, d["question"], [str(o) for o in d["options"]]) for d in docs],
        "config_id": CONFIG_ID,
        "harness_commit": res.get("git_hash"),
        "correct": raw["acc"].astype("int8"),
        "logprob": [np.float32(lp[g]) for lp, g in zip(lps, gold)],
        "pred": [LETTERS[p] for p in pred],
        "prompt_tokens": prompt_tokens,
        "gen_tokens": 0,
        "timestamp": pd.Timestamp(datetime.strptime(stamp, "%Y_%m_%dT%H_%M_%S.%f"), tz="UTC"),
        "schema_version": SCHEMA_VERSION,
    })

    qid = pd.Series([int(d["question_id"]) for d in docs])
    n = len(long)
    long = long.loc[qid.sort_values().index].drop_duplicates("item_id").sort_index()

    out = Path(args.data) / "long" / TASK / f"{long['run_id'].iloc[0]}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(long, schema=LONG_SCHEMA, preserve_index=False), out)
    print(f"{model_id}: {len(long)} questions ({n - len(long)} duplicates removed), mean correct {long['correct'].mean():.6f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
