"""Download raw Open LLM Leaderboard per-question details.

Saves one Parquet file per model per task config.

Usage:
    python download_details.py Qwen/Qwen2.5-0.5B --out /scratch/$USER/leaderboard_raw
    python download_details.py Qwen/Qwen2.5-0.5B meta-llama/Llama-3.2-1B --tasks mmlu_pro bbh --out /scratch/$USER/leaderboard_raw
    python download_details.py Qwen/Qwen2.5-0.5B --tasks all --out ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset


def pick_split(repo: str, cfg: str) -> tuple[str, list[str]]:
    """
    Args:
        repo (str): Hub dataset repo, e.g. "open-llm-leaderboard/Qwen__Qwen2.5-0.5B-details".
        cfg (str): Dataset config (task) name within the repo.

    Returns:
        tuple[str, list[str]]: (split to download, all split names in the config).
    """
    splits = get_dataset_split_names(repo, cfg)
    if "latest" in splits:
        return "latest", splits
    timestamped = sorted(s for s in splits if s not in ("train", "latest"))
    return (timestamped[-1] if timestamped else splits[0]), splits


def main() -> None:
    """
    Command-line args:
        models (list[str]): Hub model names, e.g. Qwen/Qwen2.5-0.5B.
        --tasks (list[str]): Task name fragments to keep, or "all"; default mmlu_pro.
        --out (str): Output folder; each model gets <out>/<org>__<name>/.

    Returns:
        None. Writes one Parquet file per task config plus manifest.json per model.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="Hub model names, e.g. Qwen/Qwen2.5-0.5B")
    ap.add_argument("--tasks", nargs="+", default=["mmlu_pro"], help="task name fragments to keep (e.g. mmlu_pro bbh gpqa), or 'all'")
    ap.add_argument("--out", required=True, help="output folder, on scratch")
    args = ap.parse_args()

    out_root = Path(args.out)
    for model in args.models:
        safe = model.replace("/", "__")
        repo = f"open-llm-leaderboard/{safe}-details"
        print(f"\n== {model}  ({repo})")

        try:
            configs = get_dataset_config_names(repo)
        except Exception as e:
            print(f"  could not list configs: {e}")
            continue

        task_cfgs = [c for c in configs if not c.endswith("__results")]
        if args.tasks != ["all"]:
            task_cfgs = [c for c in task_cfgs if any(t in c for t in args.tasks)]
        print(f"  {len(task_cfgs)} config(s) to download")

        model_dir = out_root / safe
        model_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"model": model, "repo": repo, "configs": {}}

        for cfg in task_cfgs:
            dest = model_dir / f"{cfg}.parquet"
            if dest.exists():
                print(f"  skip (already have) {cfg}")
                continue
            split, all_splits = pick_split(repo, cfg)
            ds = load_dataset(repo, name=cfg, split=split)
            ds.to_parquet(str(dest))
            manifest["configs"][cfg] = {"split_used": split, "all_splits": all_splits, "rows": ds.num_rows}
            print(f"  {cfg}: {ds.num_rows} rows  (split {split})")

        with open(model_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        if task_cfgs and manifest["configs"]:
            first = next(iter(manifest["configs"]))
            ds = load_dataset("parquet", data_files=str(model_dir / f"{first}.parquet"), split="train")
            print(f"\n  columns in {first}: {ds.column_names}")
            row = {k: (str(v)[:150]) for k, v in ds[0].items()}
            print("  first row (truncated):")
            for k, v in row.items():
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
