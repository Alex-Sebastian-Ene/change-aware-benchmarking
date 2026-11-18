"""Run one pinned evaluation: one model, every task in one harness config.

Run inside the container from the repo root (scripts/slurm/eval.sbatch does this):

    python scripts/run_eval.py --config configs/harness/<config>.yaml \
        --model Qwen/Qwen2.5-0.5B --sha <40-char model commit> [--limit 20]

--check-only runs verify_harness and verify_task and stops, without loading the model or
needing a GPU. Use it on the login node before queueing a job.
"""

import argparse
import os
import sys
from pathlib import Path

from cabench.harness.pins import ModelPin, is_full_sha, load_config
from cabench.harness.run import TASK_DIR, run
from cabench.harness.verify import verify_harness, verify_task

SCR = Path("/scratch") / os.environ.get("USER", "")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", required=True, help="Hub repo, e.g. Qwen/Qwen2.5-0.5B")
    p.add_argument("--sha", required=True, help="full 40-char model commit")
    p.add_argument("--out", type=Path, default=SCR / "eval_runs")
    p.add_argument("--limit", type=int, default=None, help="questions per task (smoke runs)")
    p.add_argument("--lm-eval-dir", type=Path, default=SCR / "lm-evaluation-harness")
    p.add_argument("--sif", type=Path, default=SCR / "vllm.sif")
    p.add_argument("--lockfile", type=Path, default=Path("uv.lock"))
    p.add_argument("--check-only", action="store_true", help="verify pins, then stop")
    a = p.parse_args()

    if not is_full_sha(a.sha):
        sys.exit(f"--sha must be a full 40-char commit, got {a.sha!r}")
    config = load_config(a.config)
    print(f"config {config.path.name}  config_id {config.config_id}  backend {config.harness.backend}")

    if a.check_only:
        from lm_eval.tasks import TaskManager

        verify_harness(config.harness, a.lm_eval_dir, a.lockfile, a.sif)
        tm = TaskManager(verbosity="WARNING", include_path=str(TASK_DIR))
        for t in config.tasks:
            verify_task(t, tm)
        print("PINS_OK")
        return 0

    run_dir = run(
        ModelPin(repo=a.model, sha=a.sha),
        config,
        a.out,
        lm_eval_dir=a.lm_eval_dir,
        lockfile=a.lockfile,
        sif=a.sif,
        limit=a.limit,
    )
    print("RUN_DIR", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
