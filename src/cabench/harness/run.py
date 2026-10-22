"""Run pinned evaluations. The only harness module that touches lm_eval / vllm."""

import json
from datetime import UTC, datetime
from pathlib import Path

from cabench.harness.manifest import build_manifest, write_manifest
from cabench.harness.pins import HarnessConfig, HarnessPin, ModelPin, TaskPin, is_full_sha
from cabench.harness.verify import check, verify_harness, verify_task

TASK_DIR = Path("configs/tasks")


def build_gen_kwargs(task: TaskPin) -> str:
    """Greedy-decoding gen_kwargs string for lm-eval.

    Loglikelihood tasks (max_gen_toks is None) must NOT get max_gen_toks=None.
    """
    parts = ["temperature=0", "do_sample=False"]
    if task.max_gen_toks is not None:
        parts.append(f"max_gen_toks={task.max_gen_toks}")
    return ",".join(parts)


def build_model_args(model: ModelPin, harness: HarnessPin) -> str:
    """model_args string for lm-eval's vllm backend."""
    return f"pretrained={model.repo},revision={model.sha},dtype={harness.dtype}"


def run(
    model: ModelPin,
    config: HarnessConfig,
    out: Path,
    *,
    lm_eval_dir: Path,
    lockfile: Path,
    sif: Path,
    task_dir: Path = TASK_DIR,
    limit: int | None = None,
) -> Path:
    """Verify everything, evaluate each task, write results + manifest. Returns run dir."""
    import lm_eval
    from lm_eval.tasks import TaskManager

    if not is_full_sha(model.sha):
        raise ValueError(f"model sha must be a full 40-char sha, got {model.sha!r}")

    harness = config.harness
    verify_harness(harness, lm_eval_dir, lockfile, sif)
    task_manager = TaskManager(include_path=str(task_dir))
    for t in config.tasks:
        verify_task(t, task_manager)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out) / config.config_id[:12] / model.sha[:12] / stamp

    benchmark_versions = {}
    project_git_hash = None
    for t in config.tasks:
        result = lm_eval.simple_evaluate(
            model="vllm",
            model_args=build_model_args(model, harness),
            tasks=[t.name],
            task_manager=task_manager,
            num_fewshot=t.num_fewshot,
            fewshot_random_seed=t.fewshot_seed,
            batch_size=harness.batch_size,
            apply_chat_template=harness.apply_chat_template,
            gen_kwargs=build_gen_kwargs(t),
            log_samples=True,
            limit=limit,
            random_seed=0,
            numpy_random_seed=0,
            torch_random_seed=0,
        )

        reported = str(result["versions"][t.name])
        check(f"{t.name} benchmark_version", reported, t.yaml_version)
        benchmark_versions[t.name] = reported
        project_git_hash = result.get("git_hash")

        t_dir = run_dir / t.name
        t_dir.mkdir(parents=True, exist_ok=True)
        (t_dir / "results.json").write_text(json.dumps(result["results"], indent=2, default=str))
        with (t_dir / "samples.jsonl").open("w") as f:
            for name, samples in result.get("samples", {}).items():
                for s in samples:
                    f.write(json.dumps({"task": name, **s}, default=str) + "\n")

    manifest = build_manifest(model, config, benchmark_versions, project_git_hash)
    write_manifest(manifest, run_dir)
    return run_dir
