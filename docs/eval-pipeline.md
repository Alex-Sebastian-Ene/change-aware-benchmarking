# Evaluation pipeline: setup and usage

How to run pinned evaluations on Iridis and turn them into rows of the response matrix.
For why things are set up this way, see `docs/decision-log.md`.

## Overview

A run evaluates one model on the tasks in one harness config and writes a samples file per
task plus a `manifest.json`. Ingesting the run turns it into long-table rows with
`source = "own"`, which share the matrix with the Open LLM Leaderboard's public rows.

What runs where:

| Piece | Where it lives | What pins it |
|---|---|---|
| CUDA, torch, vLLM | container image `vllm.sif`, unpacked to `vllm_sandbox/` | `apptainer_digest` |
| lm-eval code | fork checkout on scratch | `lm_eval_sha` |
| lm-eval's other packages | `ctr-venv` on scratch | `configs/harness/ctr-venv-freeze.txt` (recorded, not checked) |
| `cabench` and its packages | this repo, `uv.lock` | `lockfile_sha256` |
| Task prompt and data | `configs/tasks/<task>/` | `yaml_version`, `dataset_sha` |
| Model weights and tokenizer | Hugging Face Hub | model sha, passed per run |

Every run calls `verify_harness` and `verify_task` first and stops if anything on disk
differs from the config.

Paths used below (change them if yours differ):

```bash
SCR=/scratch/$USER                                   # scratch root
REPO=$HOME/gdp/change-aware-benchmarking             # this repo
FORK=$SCR/lm-evaluation-harness                      # pinned lm-eval checkout
SIF=$SCR/vllm.sif                                    # pinned image file
BOX=$SCR/vllm_sandbox                                # unpacked image
VENV=$SCR/ctr-venv                                   # lm-eval's packages
GITENV=$SCR/git-env                                  # git for inside the container
DATA=$SCR/data                                       # parquet store (long/, items, models, matrices)
RAW=$SCR/leaderboard_raw                             # download_details.py output
```

---

## Part 1. One-time setup

Do all of Part 1 on the **login node** unless a step says otherwise. Compute nodes have no
internet, and Iridis has no `tmux` or `screen`, so long downloads use `nohup`.

### 1.1 Shell environment

Add to `~/.bashrc`, then log in again:

```bash
module load apptainer
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer/cache
export APPTAINER_TMPDIR=/scratch/$USER/apptainer/tmp
export HF_HOME=/scratch/$USER/hf_cache

# Run a command inside the container with the fixes it needs.
ctr() { apptainer exec \
  --env PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/scratch/$USER/git-env/bin \
  --bind /scratch /scratch/$USER/vllm_sandbox "$@"; }
```

```bash
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR $HF_HOME
```

**Always `conda deactivate` before using the container**. Apptainer copies your environment
variables into it, and a conda env can make it load the wrong packages.

### 1.2 Repo

```bash
cd $HOME/gdp
git clone https://github.com/Alex-Sebastian-Ene/change-aware-benchmarking.git
cd change-aware-benchmarking
uv sync --group matrix
uv run pytest -q
```

Don't run `uv add`, `uv remove` or `uv lock` on a branch you run evaluations from. Any change
to `uv.lock` makes every existing config fail `verify_harness`.

### 1.3 lm-eval checkout

```bash
cd $SCR
git clone https://github.com/huggingface/lm-evaluation-harness.git
git -C lm-evaluation-harness checkout <lm_eval_sha from the config>
git -C lm-evaluation-harness status --porcelain        # must print nothing
```

Never edit, pull or reset this checkout afterwards. To use a different commit, clone a second
checkout and make a new config.

### 1.4 Container image

Pull by digest, never by tag:

```bash
cd $SCR
nohup apptainer pull vllm.sif \
  docker://vllm/vllm-openai@sha256:327eb19763b5c91a06e750082348170eafb99e476f91eb0cb60328dc18b04961 \
  > pull.log 2>&1 &
echo $! > pull.pid
```

Check progress with `tail -f $SCR/pull.log` and `ps -p $(cat $SCR/pull.pid)`. It takes about
an hour. When the process has ended, check the file is complete:

```bash
apptainer sif list $SIF        # the FS (Squashfs) partition's end position must equal the file size
ls -l $SIF
```

The log may end at `Creating SIF file...` without a `Build complete` line; the `sif list`
check is what counts. Never pull the image again: a new pull can give different bytes and so a
different digest. Keep a backup copy outside scratch if you can.

### 1.5 Sandbox

Iridis has no `fusermount`, so `.sif` files can't be mounted. Unpack the image once into a
folder on a CPU node:

```bash
cd $REPO/scripts
sbatch build_sandbox.sh
```

When the job finishes, check the result:

```bash
cat $BOX/.source_sif_sha256          # sha256 of vllm.sif = apptainer_digest
ctr python3 -c "import vllm, sys; print(vllm.__version__, sys.version)"
```

The sandbox is read-only. To rebuild it (for example after a scratch purge), make it writable,
delete it, and resubmit:

```bash
chmod -R u+w $BOX && rm -rf $BOX && sbatch $REPO/scripts/build_sandbox.sh
```

Check the GPU works from a compute node:

```bash
srun --partition=a100 --gres=gpu:1 --time=0:10:00 --nodes=1 \
  apptainer exec --nv \
  --env PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  $BOX python3 -c "import vllm, torch; print(vllm.__version__, torch.cuda.get_device_name(0))"
```

### 1.6 git for the container

`verify_harness` runs `git` on the fork checkout, and the image has none:

```bash
conda create -y -p $GITENV -c conda-forge git
conda deactivate
ctr git -C $FORK rev-parse HEAD           # must print lm_eval_sha
ctr git -C $FORK status --porcelain       # must print nothing
```

### 1.7 lm-eval's packages

```bash
cd $SCR
ctr python3 -m venv --system-site-packages ctr-venv
ctr $VENV/bin/python -m pip install -e $FORK
git -C $FORK status --porcelain
```

If the last command lists files such as `lm_eval.egg-info/`, add them to
`$FORK/.git/info/exclude` and run it again until it prints nothing.

Check that the image's torch and vLLM are still the ones in use, and that `lm_eval` comes from
the checkout:

```bash
ctr $VENV/bin/python -c "import torch, vllm, lm_eval; print(torch.__file__); print(vllm.__file__); print(lm_eval.__file__)"
```

torch and vLLM must be under `/usr/local/lib/python3.12/dist-packages/`, and `lm_eval` under
`$FORK`. `SyntaxWarning: invalid escape sequence '\s'` messages are harmless; don't fix them in
the checkout.

Record the packages and commit the file:

```bash
ctr $VENV/bin/python -m pip freeze > $REPO/configs/harness/ctr-venv-freeze.txt
```

The `git command not found` error from `pip freeze` is harmless.

### 1.8 Check `cabench` loads inside the container

```bash
ctr env PYTHONPATH=$REPO/src $VENV/bin/python -c "import cabench.harness.run; print('ok')"
```

---

## Part 2. Adding a task

Each task the harness runs is a pinned copy of the fork's task, in its own folder.

1. Copy the task YAML and any `utils.py` **from the pinned checkout**, not from GitHub:

   ```bash
   mkdir -p $REPO/configs/tasks/leaderboard_mmlu_pro
   cp $FORK/lm_eval/tasks/leaderboard/mmlu_pro/utils.py $REPO/configs/tasks/leaderboard_mmlu_pro/
   cp $FORK/lm_eval/tasks/leaderboard/mmlu_pro/mmlu_pro.yaml \
      $REPO/configs/tasks/leaderboard_mmlu_pro/leaderboard_mmlu_pro_pinned.yaml
   ```

2. In the copied YAML, change only:
   - `task:` to `<upstream name>_pinned`
   - add `dataset_kwargs: {revision: <dataset commit>}` (or `revision:` under an existing
     `dataset_kwargs`)

3. Choose the dataset commit. To merge with public data, use the commit the leaderboard used:
   the last dataset commit before the leaderboard's run date (the date in the model's
   `results_*.json` filename), unless later commits changed nothing under `data/`. List commits:

   ```bash
   uv run --group matrix python -c "from huggingface_hub import HfApi; [print(c.commit_id, c.created_at.date(), c.title) for c in HfApi().list_repo_commits('TIGER-Lab/MMLU-Pro', repo_type='dataset')]"
   ```

4. Note `metadata.version` from the YAML. It must match the version the leaderboard recorded
   (`versions[...]` in its `results_*.json`), because `benchmark_version` is part of every item ID.

Generative tasks (MATH, IFEval) also need `stop` sequences and a registered `parser_version`
in `cabench/harness/parsers.py`. Agree those with Seb before adding them.

---

## Part 3. Creating a harness config

A config fixes every setting that can change an answer. Its sha256 is its `config_id`.

1. Copy the template:

   ```bash
   cd $REPO
   cp configs/harness/template.yaml.example configs/harness/draft.yaml
   ```

2. Fill in every field:

   | Field | Where to get it |
   |---|---|
   | `lm_eval_repo` | `https://github.com/huggingface/lm-evaluation-harness` |
   | `lm_eval_sha` | `git -C $FORK rev-parse HEAD` |
   | `vllm_version` | `ctr python3 -c "import vllm; print(vllm.__version__)"` (exact string) |
   | `dtype` | `bfloat16` |
   | `batch_size` | a fixed integer (`32`), never `auto` |
   | `apply_chat_template` | `false` for base models, `true` for instruct models |
   | `lockfile_sha256` | `sha256sum uv.lock \| cut -d' ' -f1` |
   | `apptainer_digest` | `cat $BOX/.source_sif_sha256` |
   | task `name` | the `task:` value in the pinned YAML |
   | `yaml_version` | the pinned YAML's `metadata.version`, in quotes |
   | `dataset_repo`, `dataset_sha` | as in the pinned YAML |
   | `num_fewshot` | the leaderboard's setting (MMLU-Pro: 5) |
   | `fewshot_seed` | any integer (`1234`); no effect with the `first_n` sampler |
   | `max_gen_toks` | `null` for loglikelihood tasks |

   A task's `benchmark` defaults to its name without `_pinned`. Leave it out unless that gives
   the wrong name.

3. Rename the file to its hashed name and check it loads:

   ```bash
   mv configs/harness/draft.yaml configs/harness/<name>.yaml
   uv run python -c "from pathlib import Path; from cabench.harness.pins import expected_filename; print(expected_filename(Path('configs/harness/<name>.yaml')))"
   mv configs/harness/<name>.yaml configs/harness/<printed name>
   uv run python -c "from pathlib import Path; from cabench.harness.pins import load_config; print(load_config(Path('configs/harness/<printed name>')).config_id)"
   ```

   Name configs by what they cover, e.g. `mmlu_pro-<hash12>.yaml` for MMLU-Pro on base models.

4. Commit the config together with `uv.lock` and `ctr-venv-freeze.txt`. Never edit it again:
   any change, even a comment, goes in a new file.

---

## Part 4. Preparing a model

Do this once per model, on the login node. It downloads the weights and the dataset at their
pinned revisions into `HF_HOME`, so offline compute nodes can read them.

1. Choose the model revision. To compare with public data, use the `model_sha` the
   leaderboard recorded:

   ```bash
   python -c "import json,glob; r=json.load(open(glob.glob('$RAW/Qwen__Qwen2.5-0.5B/results_*.json')[0])); print(r['config']['model_sha'], r['versions'])"
   ```

   For any other model, use the full 40-character commit from the model's Hub page.

2. Fill the cache with a two-question CPU run inside the container:

   ```bash
   conda deactivate
   cd $REPO
   ctr env HF_HOME=$HF_HOME $VENV/bin/lm_eval \
     --model hf --model_args pretrained=<repo>,revision=<sha>,dtype=float32 \
     --include_path configs/tasks --tasks leaderboard_mmlu_pro_pinned \
     --limit 2 --device cpu --log_samples --output_path $SCR/smoke_cpu
   ```

   It must run inside the container: the dataset cache has to be written by the same
   `datasets` version that will read it.

3. For a model that's also in the public data, compare one prompt with the leaderboard's.
   They must be identical:

   ```bash
   uv run --group matrix python - $SCR/smoke_cpu $RAW/Qwen__Qwen2.5-0.5B <<'EOF'
   import glob, json, sys
   import pandas as pd

   own_dir, public_dir = sys.argv[1], sys.argv[2]
   own_file = glob.glob(f"{own_dir}/**/samples_*.jsonl", recursive=True)[0]
   own = json.loads(open(own_file).readline())
   public = pd.read_parquet(glob.glob(f"{public_dir}/*leaderboard_mmlu_pro*.parquet")[0])

   def prompt(arguments):
       # in-memory samples hold [[context, continuation], ...]; saved files hold gen_args_0
       if isinstance(arguments, dict):
           return arguments["gen_args_0"]["arg_0"]
       return arguments[0][0]

   qid = own["doc"]["question_id"]
   row = public[[d["question_id"] == qid for d in public["doc"]]].iloc[0]
   print("identical" if prompt(own["arguments"]) == prompt(row["arguments"]) else "DIFFERENT")
   EOF
   ```

   If they differ, stop: the task files, few-shot examples or dataset revision don't match.

---

## Part 5. Running evaluations

### 5.1 Job script

Save as `$REPO/scripts/slurm/eval.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/eval_logs/%x-%j.log
set -euo pipefail

# Inputs, passed with sbatch --export: MODEL_REPO, MODEL_SHA, CONFIG (path relative to the repo),
# optional LIMIT (smoke runs) and OUT.
: "${MODEL_REPO:?}" "${MODEL_SHA:?}" "${CONFIG:?}"
LIMIT=${LIMIT:-None}
SCR=/scratch/$USER
OUT=${OUT:-$SCR/eval_runs}
REPO=$HOME/gdp/change-aware-benchmarking
BOX=$SCR/vllm_sandbox

module load apptainer

# The sandbox must come from the .sif the config pins.
want=$(awk '/apptainer_digest:/ {gsub(/"/, "", $2); print $2}' "$REPO/$CONFIG")
have=$(cat "$BOX/.source_sif_sha256")
[ "$want" = "$have" ] || { echo "sandbox is from $have, config pins $want"; exit 1; }

cd "$REPO"
apptainer exec --nv --cleanenv --bind /scratch \
  --env PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$SCR/git-env/bin \
  --env PYTHONPATH=$REPO/src \
  --env HF_HOME=$SCR/hf_cache --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 \
  --env XDG_CACHE_HOME=$SCR/.cache \
  --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  --env SLURM_JOB_ID=$SLURM_JOB_ID \
  "$BOX" "$SCR/ctr-venv/bin/python" - <<EOF
from pathlib import Path
from cabench.harness.pins import ModelPin, load_config
from cabench.harness.run import run

run_dir = run(
    ModelPin(repo="$MODEL_REPO", sha="$MODEL_SHA"),
    load_config(Path("$CONFIG")),
    Path("$OUT"),
    lm_eval_dir=Path("$SCR/lm-evaluation-harness"),
    lockfile=Path("uv.lock"),
    sif=Path("$SCR/vllm.sif"),
    limit=$LIMIT,
)
print("RUN_DIR", run_dir)
EOF
```

`--cleanenv` stops your shell's variables leaking into the container; everything the run needs
is passed with `--env`. `CUDA_VISIBLE_DEVICES` keeps vLLM on the GPU Slurm gave the job, and
`SLURM_JOB_ID` is recorded in the manifest.

```bash
mkdir -p /scratch/$USER/eval_logs
```

### 5.2 Smoke run

Run 20 questions per task into a separate folder before any full run:

```bash
cd $REPO
sbatch --export=ALL,MODEL_REPO=Qwen/Qwen2.5-0.5B,MODEL_SHA=<sha>,CONFIG=configs/harness/<config>.yaml,LIMIT=20,OUT=/scratch/$USER/eval_smoke \
  scripts/slurm/eval.sbatch
```

It passes if the log ends with `RUN_DIR …` and that folder contains `manifest.json` and
`<task>/samples.jsonl` with 20 lines. Keep smoke output out of the data store. The manifest
records `limit`, so a partial run can always be told apart.

### 5.3 Full runs

Same command without `LIMIT` and `OUT`:

```bash
sbatch --export=ALL,MODEL_REPO=Qwen/Qwen2.5-0.5B,MODEL_SHA=<sha>,CONFIG=configs/harness/<config>.yaml \
  scripts/slurm/eval.sbatch
```

Every submission is a separate run with its own `run_id`, so submitting the same model twice
gives the repeat needed for the noise test. Check on jobs with `squeue -u $USER` and
`tail -f /scratch/$USER/eval_logs/eval-<jobid>.log`.

For many models, write one `repo sha` pair per line in `models.txt` and submit one job each:

```bash
while read repo sha; do
  sbatch --export=ALL,MODEL_REPO=$repo,MODEL_SHA=$sha,CONFIG=configs/harness/<config>.yaml scripts/slurm/eval.sbatch
done < models.txt
```

Run Part 4 for each model on the login node first.

### 5.4 Output

```
$OUT/<config_id[:12]>/<model_sha[:12]>/<UTC time>/
    manifest.json                  config_id, run_id, model, all pins, versions, limit, runtime
    <task>/results.json            aggregate scores
    <task>/samples.jsonl           one line per question
```

---

## Part 6. Ingesting runs

Run on the login node from the repo root. This writes one long-table file per run and task
and never overwrites an existing one:

```bash
cd $REPO
uv run --group matrix python - <RUN_DIR> <<'EOF'
import json, os, sys
from datetime import datetime
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, "scripts/matrix")
from utils.ingest import read_samples
from utils.schema import LONG, conform
from cabench.harness.pins import TaskPin

run_dir = Path(sys.argv[1])
m = json.loads((run_dir / "manifest.json").read_text())
if m["limit"] is not None:
    sys.exit(f"limit={m['limit']}: smoke runs don't go in the data store")
data = Path("/scratch") / os.environ["USER"] / "data"
for t in m["tasks"]:
    task = TaskPin(**t)
    long_df, _ = read_samples(
        run_dir / task.name / "samples.jsonl",
        task=task.benchmark, benchmark=task.benchmark,
        benchmark_version=m["benchmark_versions"][task.name],
        run_id=m["run_id"], source="own",
        model_id=m["model"]["repo"], model_revision=m["model"]["sha"],
        config_id=m["config_id"], harness_commit=m["harness"]["lm_eval_sha"],
        metric="acc", timestamp=datetime.fromisoformat(m["runtime"]["utc_time"]),
    )
    out = data / "long" / task.benchmark / f"own_{m['run_id']}.parquet"
    if out.exists():
        sys.exit(f"{out} already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(conform(long_df, LONG, "long"), out)
    print(f"{task.benchmark}: {len(long_df)} rows, mean correct {long_df['correct'].mean():.4f} -> {out}")
EOF
```

Then rebuild the side tables and matrices and run the schema check:

```bash
uv run --group matrix python scripts/matrix/build_tables.py --data $DATA --raw $RAW
uv run python scripts/check_schema.py $DATA
```

`build_tables.py` refuses to continue if any own `item_id` is missing from the items table;
that means the own questions don't match the public ones.

For Qwen2.5-0.5B on MMLU-Pro, expect 12,032 rows per run after duplicates are removed.

---

## Part 7. Acceptance tests

Run test 2 first, because test 1's threshold depends on it.

**Test 2 (noise):** two full runs of one model with one config (runs A and B).
**Test 1 (reproduction):** a full own run of 2–3 models that are also in the public data,
compared with the public rows.

Per-question agreement between any two runs:

```bash
uv run --group matrix python - <long file 1> <long file 2> <<'EOF'
import sys, pandas as pd
a, b = (pd.read_parquet(p)[["item_id", "correct"]] for p in sys.argv[1:3])
j = a.merge(b, on="item_id", suffixes=("_1", "_2"))
print(f"matched {len(j)} of {len(a)} / {len(b)} items")
print(f"agreement {(j.correct_1 == j.correct_2).mean():.4%}")
print(f"accuracy  {j.correct_1.mean():.4f} vs {j.correct_2.mean():.4f}")
EOF
```

`matched` must equal the number of items in each run; if not, item IDs differ and the
comparison means nothing. Send both results to Seb, who sets the threshold (currently ≥98% on
multiple-choice tasks) and judges the result.

---

## Rules

- Never edit a config file, the lm-eval checkout, `vllm.sif` or the sandbox. Make a new
  config, clone, image or sandbox instead.
- Never re-pull the image.
- Don't change `uv.lock` on a branch you run from.
- Use the leaderboard's model sha and dataset revision for anything compared with public data.
- `conda deactivate` before using the container.
- Smoke runs (`LIMIT` set) never go in the data store.
- Long-table files are written once; a bad run is removed by deleting its file.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `fuse: failed to exec fusermount` | Iridis can't mount `.sif` files | Run from the sandbox (1.5) |
| `"python": executable file not found in $PATH` | The image's PATH omits `/usr/bin` | Pass `--env PATH=…` (the `ctr` helper does) |
| `execve(): apptainer: No such file or directory` | Apptainer module not loaded | `module load apptainer` |
| `libcrypto.so.1.1: cannot open shared object file` | Host `git` used inside the container | Use the conda-forge git (1.6) |
| `PinMismatch: lm_eval working tree` | Files changed or added in the checkout | Restore them, or add build files to `.git/info/exclude` |
| `PinMismatch: lockfile_sha256` | `uv.lock` changed since the config was made | Restore `uv.lock`, or make a new config |
| `PinMismatch: lm_eval is imported from …` | `lm_eval` loaded from somewhere other than the checkout | Reinstall with `pip install -e $FORK` in the venv; deactivate conda |
| `ConfigError: … does not match its contents` | The config was edited | Restore it, or save the change under its new hashed name |
| `sandbox is from …, config pins …` | Sandbox built from a different `.sif` | Rebuild the sandbox (1.5) |
| Login-node job ends without finishing | Login nodes stop heavy CPU work | Run heavy steps as Slurm jobs |

## Known issues

- **`read_samples` fails on MMLU-Pro.** At the pinned lm-eval commit, `target` in the samples
  file is the answer letter (`"A"`–`"J"`), but `read_samples` does `int(rec["target"])`.
  It also writes `pred` as an index (`"3"`) where public rows use a letter (`"D"`). Fix in
  `scripts/matrix/utils/ingest.py` before ingesting any own run.
- **Backend.** Own runs use vLLM; the leaderboard used HF transformers. Test 1 measures the
  combined effect of this and the lm-eval commit difference.
- **Public rows' `config_id`** is the placeholder `openllm_v2_public`, which matches no config
  file, so public rows fail the schema check's config test.
- **Not covered by the pins:** the venv (recorded in `ctr-venv-freeze.txt` only) and the sandbox
  (tied to the `.sif` by its marker file only).