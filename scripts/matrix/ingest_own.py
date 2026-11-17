"""Own run directory (manifest.json + <task>/samples.jsonl) -> long-table parquet files.

    uv run --group matrix python scripts/matrix/ingest_own.py <RUN_DIR> [--data DIR]

Writes <data>/long/<benchmark>/own_<run_id>.parquet per task and never overwrites one.
Smoke runs (manifest limit set) are refused.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.ingest import read_samples  # noqa: E402
from utils.schema import LONG, conform  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--data", type=Path, default=Path("/scratch") / os.environ.get("USER", "") / "data")
    a = p.parse_args()

    m = json.loads((a.run_dir / "manifest.json").read_text())
    if m["limit"] is not None:
        sys.exit(f"limit={m['limit']}: smoke runs don't go in the data store")

    for t in m["tasks"]:
        benchmark = t["benchmark"]
        long_df, _ = read_samples(
            a.run_dir / t["name"] / "samples.jsonl",
            task=benchmark, benchmark=benchmark,
            benchmark_version=m["benchmark_versions"][t["name"]],
            run_id=m["run_id"], source="own",
            model_id=m["model"]["repo"], model_revision=m["model"]["sha"],
            config_id=m["config_id"], harness_commit=m["harness"]["lm_eval_sha"],
            metric="acc", timestamp=datetime.fromisoformat(m["runtime"]["utc_time"]),
        )
        out = a.data / "long" / benchmark / f"own_{m['run_id']}.parquet"
        if out.exists():
            sys.exit(f"{out} already exists")
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(conform(long_df, LONG, "long"), out)
        print(f"{benchmark}: {len(long_df)} rows, mean correct {long_df['correct'].mean():.4f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
