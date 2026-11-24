"""Convert downloaded Open LLM Leaderboard details into the long table.

Input:  one model's folder from download_details.py
Output: <data>/long/<benchmark>/<run_id>.parquet per benchmark, in the spec's
        "Long response table" format, with duplicate questions removed. A benchmark
        made of several tasks (the three MuSR subtasks) goes into one file.

Rows come from utils/ingest.py's read_records, the same code ingest_own.py uses,
so public and own rows get item IDs, scores and preds the same way. The tasks and
metric of each benchmark are in BENCHMARKS there.

Usage (login node: it fetches the run's results JSON and the model's tokenizer):
    python ingest_public.py /scratch/$USER/leaderboard_raw/Qwen__Qwen2.5-0.5B \
        --data /scratch/$USER/data
    python ingest_public.py /scratch/$USER/leaderboard_raw/Qwen__Qwen2.5-0.5B \
        --data /scratch/$USER/data --benchmark leaderboard_musr
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from utils.ingest import BENCHMARKS, read_details, read_records
from utils.schema import LONG, conform

CONFIG_ID = "openllm_v2_public"   # placeholder


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


def run_stamp(manifest: dict, cfgs: list[str]) -> str:
    """
    Args:
        manifest (dict): The model folder's manifest.json from download_details.py.
        cfgs (list[str]): Details configs of one benchmark, e.g.
            ["Qwen__Qwen2.5-0.5B__leaderboard_mmlu_pro"].

    Returns:
        str: The latest run timestamp, which must be the same for every config.
    """
    stamps = {}
    for cfg in cfgs:
        if cfg not in manifest["configs"]:
            sys.exit(f"{cfg} isn't in manifest.json; rerun download_details.py for it")
        splits = manifest["configs"][cfg]["all_splits"]
        stamps[cfg] = sorted(s for s in splits if s not in ("latest", "train"))[-1]
    if len(set(stamps.values())) != 1:
        sys.exit(f"the tasks come from different leaderboard runs: {stamps}")
    return next(iter(stamps.values()))


def ingest(model_dir: Path, benchmark: str, data: Path) -> Path:
    """
    Args:
        model_dir (Path): One model's folder from download_details.py.
        benchmark (str): A key of BENCHMARKS, e.g. "leaderboard_musr".
        data (Path): Parquet store root.

    Returns:
        Path: The Parquet file written, <data>/long/<benchmark>/<run_id>.parquet.
    """
    manifest = json.loads((model_dir / "manifest.json").read_text())
    model_id, repo = manifest["model"], manifest["repo"]
    safe = model_id.replace("/", "__")
    bench = BENCHMARKS[benchmark]
    cfgs = [f"{safe}__{task}" for task in bench.tasks]

    stamp = run_stamp(manifest, cfgs)
    res = load_results_json(repo, safe, stamp, model_dir)
    versions = {str(res["versions"][task]) for task in bench.tasks}
    if len(versions) != 1:
        sys.exit(f"{benchmark}: tasks report different versions {sorted(versions)}")
    (version,) = versions
    revision = model_revision(res)
    if revision is None:
        sys.exit(f"{model_id}: the results JSON records no model revision")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def count_tokens(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    run_id = f"public_{safe}_{stamp}"
    parts, n = [], 0
    for task, cfg in zip(bench.tasks, cfgs, strict=True):
        long, _ = read_records(
            read_details(model_dir / f"{cfg}.parquet"),
            task=task, benchmark=benchmark, benchmark_version=version,
            run_id=run_id, source="public", model_id=model_id, model_revision=revision,
            config_id=CONFIG_ID, harness_commit=res.get("git_hash"), metric=bench.metric,
            timestamp=datetime.strptime(stamp, "%Y_%m_%dT%H_%M_%S.%f").replace(tzinfo=UTC),
            count_tokens=count_tokens,
        )
        n += len(pd.read_parquet(model_dir / f"{cfg}.parquet", columns=["doc_id"]))
        parts.append(long)
    long = pd.concat(parts, ignore_index=True)
    dup = long["item_id"].duplicated()
    if dup.any():
        sys.exit(f"{benchmark}: {dup.sum()} item_ids appear in more than one task")

    out = data / "long" / benchmark / f"{run_id}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(conform(long, LONG, "long"), out)
    print(f"{model_id} {benchmark}: {len(long)} questions ({n - len(long)} duplicates removed), "
          f"mean correct {long['correct'].mean():.6f}")
    print(f"wrote {out}")
    return out


def main() -> None:
    """
    Command-line args:
        model_dir (str): One model's folder from download_details.py.
        --data (str): Parquet store root; output goes to <data>/long/<benchmark>/<run_id>.parquet.
        --benchmark (list[str]): Keys of BENCHMARKS; default leaderboard_mmlu_pro.

    Returns:
        None. Writes one Parquet file per benchmark and prints a summary.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--data", required=True)
    ap.add_argument("--benchmark", nargs="+", default=["leaderboard_mmlu_pro"],
                    choices=sorted(BENCHMARKS), help="benchmark(s) to ingest")
    args = ap.parse_args()
    for benchmark in args.benchmark:
        ingest(Path(args.model_dir), benchmark, Path(args.data))


if __name__ == "__main__":
    main()
