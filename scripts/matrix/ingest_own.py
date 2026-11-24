"""Own run directory (manifest.json + <task>/samples.jsonl) -> long-table parquet files.

    uv run --group matrix python scripts/matrix/ingest_own.py <RUN_DIR> [--data DIR]

Writes <data>/long/<benchmark>/own_<run_id>.parquet per benchmark and never overwrites one.
Subtasks that share a benchmark (the three MuSR tasks) go into the same file.
Smoke runs (manifest limit set) are refused.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.ingest import BENCHMARKS, read_samples  # noqa: E402
from utils.schema import LONG, conform  # noqa: E402

PINNED_SUFFIX = "_pinned"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--data", type=Path, default=Path("/scratch") / os.environ.get("USER", "") / "data")
    a = p.parse_args()

    m = json.loads((a.run_dir / "manifest.json").read_text())
    if m["limit"] is not None:
        sys.exit(f"limit={m['limit']}: smoke runs don't go in the data store")

    by_benchmark = defaultdict(list)
    for t in m["tasks"]:
        by_benchmark[t["benchmark"]].append(t)

    # read everything before writing anything, so a bad task leaves no partial output
    outputs = []
    for benchmark, tasks in by_benchmark.items():
        if benchmark not in BENCHMARKS:
            sys.exit(f"{benchmark} isn't in BENCHMARKS (utils/ingest.py); add it there")
        versions = {m["benchmark_versions"][t["name"]] for t in tasks}
        if len(versions) != 1:
            sys.exit(f"{benchmark}: subtasks report different versions {sorted(versions)}")
        (version,) = versions
        parts = []
        for t in tasks:
            long_df, _ = read_samples(
                a.run_dir / t["name"] / "samples.jsonl",
                task=t["name"].removesuffix(PINNED_SUFFIX), benchmark=benchmark,
                benchmark_version=version,
                run_id=m["run_id"], source="own",
                model_id=m["model"]["repo"], model_revision=m["model"]["sha"],
                config_id=m["config_id"], harness_commit=m["harness"]["lm_eval_sha"],
                metric=BENCHMARKS[benchmark].metric,
                timestamp=datetime.fromisoformat(m["runtime"]["utc_time"]),
            )
            mean = long_df["correct"].mean()
            print(f"  {t['name']}: {len(long_df)} rows, mean correct {mean:.4f}")
            parts.append(long_df)
        long_df = pd.concat(parts, ignore_index=True)
        dup = long_df["item_id"].duplicated()
        if dup.any():
            sys.exit(f"{benchmark}: {dup.sum()} item_ids appear in more than one subtask")
        out = a.data / "long" / benchmark / f"own_{m['run_id']}.parquet"
        if out.exists():
            sys.exit(f"{out} already exists")
        outputs.append((benchmark, long_df, out))

    for benchmark, long_df, out in outputs:
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(conform(long_df, LONG, "long"), out)
        mean = long_df["correct"].mean()
        print(f"{benchmark}: {len(long_df)} rows, mean correct {mean:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
