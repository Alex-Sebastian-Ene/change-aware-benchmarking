# Response Matrix

Utilities for turning per-question LLM evaluation results into a long response
table and per-benchmark wide matrices (spec v1). Currently only MMLU-Pro from
the Open LLM Leaderboard is wired up end to end.

## Pipeline

```
download_details.py   Hub details  ->  <raw>/<org>__<name>/*.parquet + manifest.json
ingest_public.py      one model    ->  <data>/long/leaderboard_mmlu_pro/<run_id>.parquet
build_tables.py       long table   ->  <data>/items.parquet, models.parquet, matrices/*.parquet
```

The long table is the source of truth. `items.parquet`, `models.parquet` and
the matrices are derived from it: rerun `build_tables.py` after every ingest
and never edit those outputs by hand.

## Setup

These scripts live in `scripts/matrix/` of the `change-aware-benchmarking`
repo and run on any machine with Python, uv and internet access: a laptop or
the Iridis HPC cluster. Run every command below from the repository root.
Python puts the script's own folder on `sys.path`, so `utils` (in
`scripts/matrix/utils/`) is found automatically when you run a script by
path. To import `utils` from your own code or a notebook, first add
`scripts/matrix` to `sys.path`.

The extra dependencies (`pyarrow`, `datasets`, `huggingface_hub`,
`transformers` and `pyyaml`) are in the opt-in `matrix` dependency group of
`pyproject.toml`. Install them with:

```bash
uv sync --group matrix
```

The Open LLM Leaderboard details datasets are gated, so you need a Hugging
Face token. Accept the dataset terms on the Hub, then run
`uv run hf auth login` once. The token is saved in the Hugging Face cache
folder (`$HF_HOME` if you set it, otherwise `~/.cache/huggingface`).

### Extra steps on Iridis

Skip this section when you run locally.

- Your home directory quota is too small for uv's package cache and the
  Hugging Face cache. Point both at scratch before you run anything, for
  example in your `~/.bashrc`:

  ```bash
  export UV_CACHE_DIR=/scratch/$USER/uv-cache
  export HF_HOME=/scratch/$USER/hf_cache
  ```

- All three scripts need internet access to the Hugging Face Hub, so run
  them on a login node, not a compute node.
- Keep raw downloads and the data store on scratch, for example
  `/scratch/$USER/leaderboard_raw` and `/scratch/$USER/data`.

## Typical run

Locally, use the repo's git-ignored `data/` folders:

```bash
RAW=data/raw/leaderboard
DATA=data/processed
```

On Iridis, use scratch:

```bash
RAW=/scratch/$USER/leaderboard_raw
DATA=/scratch/$USER/data
```

Then:

```bash
uv run --group matrix python scripts/matrix/download_details.py Qwen/Qwen2.5-0.5B meta-llama/Llama-3.2-1B --out $RAW
uv run --group matrix python scripts/matrix/ingest_public.py $RAW/Qwen__Qwen2.5-0.5B     --data $DATA
uv run --group matrix python scripts/matrix/ingest_public.py $RAW/meta-llama__Llama-3.2-1B --data $DATA
uv run --group matrix python scripts/matrix/build_tables.py --data $DATA --raw $RAW
```

---

## Command-line scripts

### `download_details.py`

Downloads the raw per-question details for one or more models from the Open
LLM Leaderboard (`open-llm-leaderboard/<org>__<name>-details` on the Hub).

```bash
uv run --group matrix python scripts/matrix/download_details.py MODEL [MODEL ...] --out OUT [--tasks FRAG [FRAG ...] | --tasks all]
```

| Argument  | Meaning |
|-----------|---------|
| `MODEL`   | Hub model name(s), e.g. `Qwen/Qwen2.5-0.5B`. |
| `--out`   | Output folder (required). Each model gets `<out>/<org>__<name>/`. |
| `--tasks` | Keep only task configs whose name contains one of these fragments (default `mmlu_pro`), or `all`. |

It writes:

- one Parquet file per task config, named `<org>__<name>__<task>.parquet`;
- a `manifest.json` that records, for each config, the split it used and every
  split that exists. `ingest_public.py` needs this file.

The script uses the `latest` split when there is one, and otherwise the newest
timestamped split. It skips files that already exist, so you can rerun it
safely. A model with no details repo is reported and skipped. At the end it
prints the columns and first row of one file so you can check them.

### `ingest_public.py`

Converts one model's MMLU-Pro download into a long-table file.

```bash
uv run --group matrix python scripts/matrix/ingest_public.py MODEL_DIR --data DATA
```

| Argument   | Meaning |
|------------|---------|
| `MODEL_DIR`| One model's folder from `download_details.py`. |
| `--data`   | Parquet store root. Output goes to `<data>/long/leaderboard_mmlu_pro/<run_id>.parquet`. |

What it does:

- Fetches the run's `results_<timestamp>.json` from the Hub and caches it in
  `MODEL_DIR`. That file supplies `benchmark_version`, `model_revision` and
  `harness_commit`.
- Computes `item_id` from the question and options. See `scripts/matrix/utils/item_id.py`.
- Sets `correct` from the leaderboard's `acc`, and `logprob` from the
  log-likelihood of the gold option. Sets `pred` to the letter of the option
  with the highest log-likelihood.
- Loads the model's tokenizer at that revision to count `prompt_tokens`.
  `gen_tokens` is 0, because this is a log-likelihood task.
- Removes duplicate questions, meaning items whose question and options are
  identical. It keeps the one with the lowest `question_id`.

The `run_id` is `public_<org>__<name>_<timestamp>`. `config_id` is currently
the placeholder `openllm_v2_public`.

### `build_tables.py`

Regenerates the side tables and wide matrices from the long table.

```bash
uv run --group matrix python scripts/matrix/build_tables.py --data DATA --raw RAW
```

| Argument | Meaning |
|----------|---------|
| `--data` | Parquet store root containing `long/`. Outputs are written here. |
| `--raw`  | The `download_details.py` output folder, used to rebuild the items table from the published questions. |

| Output | Contents |
|--------|----------|
| `items.parquet` | One row per item: `item_id`, `benchmark`, `benchmark_version`, `subtask`, `source_index`, `answer_key`, `ref_prompt_tokens`. |
| `models.parquet` | One row per `(model_id, model_revision)`: `family`, `params_b` and `parent_model` come from the leaderboard summary dataset `open-llm-leaderboard/contents` for public models. `checkpoint_step` and `split` are left empty. |
| `matrices/<benchmark>__<config_id>.parquet` | Rows are `model_id@model_revision` and columns are `item_id` in items-table order. Values are `1.0`, `0.0` or `NaN`. `NaN` means there was no response and is never 0. |

For each model and config, the earliest run is canonical. Later runs are
repeats that serve as a noise reference, and they do not become matrix rows.

The script exits with an error in any of these cases:

- a `model_revision` is missing;
- a benchmark has no items builder (only `leaderboard_mmlu_pro` has one now,
  in `ITEM_BUILDERS`);
- a long-table `item_id` is not in the items table.

---

## `utils/` library

Importable helpers in `scripts/matrix/utils/`, shared by the scripts, and the
starting point for ingesting our own lm-eval runs. The examples below assume
`scripts/matrix` is on `sys.path`, for example:

```python
import sys
sys.path.insert(0, "scripts/matrix")   # from the repository root
```

### `utils/item_id.py`: stable item IDs

```python
from utils.item_id import make_item_id, item_key, normalise, find_collisions

iid = make_item_id("leaderboard_mmlu_pro", "1.0", question, options)
```

`item_id` is the first 16 hex characters of the SHA-256 of
`benchmark | benchmark_version | question | choice_1 | ... | choice_n`. Before
hashing, the question and each choice are normalised: Unicode NFC, whitespace
collapsed, ends stripped. For free-response items, pass `choices=None`.

- `item_key(...)` returns the string that gets hashed. It is useful for
  debugging.
- `find_collisions(pairs)` takes `(item_key, item_id)` pairs and returns any ID
  that maps to more than one distinct key.

Every producer of item IDs must use this module, so that public and own runs
line up.

### `utils/schema.py`: Arrow schemas

Defines `LONG`, `MODELS` and `ITEMS` (with nullability), `SCHEMA_VERSION = 1`,
and the allowed values in `ALLOWED`.

```python
from utils.schema import LONG, conform
import pyarrow.parquet as pq

table = conform(df, LONG, "long")     # name is "long", "models" or "items"
pq.write_table(table, path)
```

`conform` reorders the columns and casts them to the schema. It raises an
error on missing or extra columns, on values outside `ALLOWED`, or on a null
`correct`. It never guesses.

### `utils/config.py`: pinned harness configs

```python
from utils.config import config_id, load_config, register

cid = config_id("configs/mmlu_pro_v1.yaml")    # first 16 hex chars of SHA-256 of the raw bytes
cfg = load_config("configs/mmlu_pro_v1.yaml")  # parsed dict, plus cfg["_config_id"]
register("configs/mmlu_pro_v1.yaml", "configs/registry.tsv")
```

- `config_id` hashes the raw file bytes, so any edit, even a comment, gives a
  new ID. Never edit a config in place. Copy it to a new file instead.
- `load_config` requires the sections `harness`, `tasks`, `decoding`,
  `backend`, `environment` and `chat_template`. It rejects files that still
  contain `TBD`, `TODO` or `FILL_ME`, unless you pass `allow_unpinned=True`.
- `register` appends `config_id<TAB>filename` to a registry file. It raises an
  error if a registered file has since changed.

### `utils/ingest.py`: our own lm-eval runs

Converts an lm-eval `--log_samples` file (`samples_<task>_<timestamp>.jsonl`)
into long-table rows and items-table rows.

```python
from utils.ingest import read_samples
from utils.schema import LONG, ITEMS, conform

long_df, items_df = read_samples(
    "samples_leaderboard_mmlu_pro_2026-10-17T12-00-00.jsonl",
    task="leaderboard_mmlu_pro", benchmark="leaderboard_mmlu_pro", benchmark_version="1.0",
    run_id="own_...", source="own", model_id="org/name", model_revision="<sha>",
    config_id=cid, harness_commit="<lm-eval sha>", metric="acc",
    count_tokens=lambda s: len(tok(s, add_special_tokens=False)["input_ids"]),
)
table = conform(long_df, LONG, "long")
```

- `metric` selects the column that becomes `correct`. A metric in
  `GEN_METRICS` (`exact_match`, `prompt_level_strict_acc`) marks the task as
  generative: `logprob` is null, `pred` is the generated text, and
  `gen_tokens` is counted. Otherwise the task is treated as log-likelihood.
- Each benchmark needs an extractor in `EXTRACTORS`, mapping `doc` to
  `(question, choices, answer_key, subtask)`. Only MMLU-Pro has one so far.
  Check the `doc` fields at the pinned harness commit before you add others.
- If you leave out `count_tokens`, `prompt_tokens` is null and `gen_tokens`
  is 0.

### `utils/matrix.py`: long table to wide matrix

```python
from utils.matrix import build_wide, build_all, split_canonical

wide = build_wide(long_df, items_df, "leaderboard_mmlu_pro", cid)
all_matrices = build_all(long_df, items_df)    # {(benchmark, config_id): DataFrame}
canon, repeats = split_canonical(long_df)      # canonical runs vs repeat runs
```

This module follows the same rules as `build_tables.py`: one matrix per
`(benchmark, config_id)`, the earliest run canonical, and `NaN` for no
response. It also raises an error on duplicate `(run_id, item_id)` keys and on
item IDs that are not in the items table.
