"""Build the side tables and wide matrices from the long table.

Outputs (spec v1, as in the starter repo's gdp_matrix/schema.py and matrix.py):
    <data>/items.parquet    one row per item:   item_id, benchmark, benchmark_version,
                            subtask, source_index, answer_key, ref_prompt_tokens
    <data>/models.parquet   one row per model:  model_id, model_revision, family, params_b,
                            checkpoint_step, parent_model, source, split
    <data>/matrices/<benchmark>__<config_id>.parquet
                            rows "model_id@model_revision", columns item_id in items-table
                            order, values 1.0 / 0.0 / NaN (NaN = no response, never 0)

The long table is the source of truth; everything here is regenerated from it,
so rerun this script after every ingest and never edit its outputs by hand.

Usage (login node: it reads the leaderboard's summary table for model metadata):
    python build_tables.py --data /scratch/$USER/data --raw /scratch/$USER/leaderboard_raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.ingest import BENCHMARKS, read_details, read_items   # as ingest_public.py

ITEMS_SCHEMA = pa.schema([
    ("item_id", pa.string()),
    ("benchmark", pa.string()),
    ("benchmark_version", pa.string()),
    ("subtask", pa.string()),
    ("source_index", pa.string()),
    ("answer_key", pa.string()),
    ("ref_prompt_tokens", pa.int32()),
])
MODELS_SCHEMA = pa.schema([
    ("model_id", pa.string()),
    ("model_revision", pa.string()),
    ("family", pa.string()),
    ("params_b", pa.float32()),
    ("checkpoint_step", pa.int64()),
    ("parent_model", pa.string()),
    ("source", pa.string()),
    ("split", pa.string()),   # calibration | held_out | extrapolation | checkpoint (Seb)
])


# ---------------------------------------------------------------------------
# Items: built from the published questions with utils/ingest.py's read_items, so
# the ID rule and de-duplication (lowest source index of identical items) are the
# same code that built the long table's rows.
# ---------------------------------------------------------------------------
def public_items(raw_root: Path, benchmark: str, version: str) -> pd.DataFrame:
    """
    Args:
        raw_root (Path): download_details.py output folder (one subfolder per model).
        benchmark (str): A key of BENCHMARKS (utils/ingest.py), e.g. "leaderboard_musr".
        version (str): benchmark_version, as used in the long table's item IDs.

    Returns:
        pd.DataFrame: One row per unique item, columns in ITEMS_SCHEMA order: tasks in
        BENCHMARKS order, ascending source index within each (question_id for MMLU-Pro,
        as before). This order sets the matrix columns. Any model's details file has
        every question, so the first model found is used for each task.
    """
    parts = []
    for task in BENCHMARKS[benchmark].tasks:
        files = sorted(raw_root.glob(f"*/*__{task}.parquet"))
        if not files:
            sys.exit(f"no {task} files under {raw_root}; run download_details.py for it")
        items = read_items(read_details(files[0]), task=task, benchmark=benchmark,
                           benchmark_version=version)
        # matrix columns follow this order: ascending source index within each task
        parts.append(items.iloc[items["source_index"].astype(int).argsort(kind="stable")])
    items = pd.concat(parts, ignore_index=True)
    dup = items["item_id"].duplicated()
    if dup.any():
        sys.exit(f"{benchmark}: {dup.sum()} item_ids appear in more than one task")
    items["ref_prompt_tokens"] = pd.array([pd.NA] * len(items), dtype="Int32")
    return items[ITEMS_SCHEMA.names]


# ---------------------------------------------------------------------------
# Models: one row per (model_id, model_revision) in the long table, with
# metadata from the leaderboard summary table for public models.
# ---------------------------------------------------------------------------
def build_models(long: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        long (pd.DataFrame): The long response table.

    Returns:
        pd.DataFrame: One row per (model_id, model_revision), columns in
        MODELS_SCHEMA order. Metadata columns are NaN where the leaderboard
        summary table has no match; checkpoint_step and split are left empty.
    """
    models = (long[["model_id", "model_revision", "source"]].drop_duplicates(["model_id", "model_revision"]).reset_index(drop=True))
    meta = pd.DataFrame(columns=["model_id", "model_revision", "family", "params_b", "parent_model"])
    if (models["source"] == "public").any():
        from datasets import load_dataset
        s = load_dataset("open-llm-leaderboard/contents", split="train").to_pandas()
        meta = pd.DataFrame({
            "model_id": s["fullname"],
            "model_revision": s["Model sha"],
            "family": s["Architecture"].str.replace(r"For(CausalLM|ConditionalGeneration)$", "", regex=True),
            "params_b": s["#Params (B)"].where(s["#Params (B)"] > 0),
            "parent_model": s["Base Model"].where(s["Base Model"] != s["fullname"]),
        }).drop_duplicates(["model_id", "model_revision"])
    models = models.merge(meta, on=["model_id", "model_revision"], how="left")
    models["checkpoint_step"] = pd.array([pd.NA] * len(models), dtype="Int64")
    models["split"] = None
    return models[MODELS_SCHEMA.names]


# ---------------------------------------------------------------------------
# Wide matrix per (benchmark, config_id). Earliest run per model is canonical;
# later runs of the same model/config are repeats (noise reference), not rows.
# ---------------------------------------------------------------------------
def canonical(long: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        long (pd.DataFrame): Long-table rows.

    Returns:
        pd.DataFrame: Rows of the run with the earliest timestamp (run_id breaks ties).
    """
    keys = ["model_id", "model_revision", "benchmark", "config_id"]
    starts = (long.groupby(keys + ["run_id"], as_index=False)["timestamp"].min().sort_values(keys + ["timestamp", "run_id"]))
    keep = set(starts.drop_duplicates(keys)["run_id"])
    return long[long["run_id"].isin(keep)]


def build_wide(long: pd.DataFrame, items: pd.DataFrame, benchmark: str, config_id: str) -> pd.DataFrame:
    """
    Args:
        long (pd.DataFrame): The long response table.
        items (pd.DataFrame): The items table; sets the column order.
        benchmark (str): Benchmark to select.
        config_id (str): Config to select.

    Returns:
        pd.DataFrame: float32 matrix indexed by "model_id@model_revision" (sorted),
        one column per item_id, values 1.0 / 0.0 / NaN.
    """
    sel = canonical(long[(long["benchmark"] == benchmark) & (long["config_id"] == config_id)])
    order = items.loc[items["benchmark"] == benchmark, "item_id"].tolist()
    wide = (sel.assign(model=sel["model_id"] + "@" + sel["model_revision"]).pivot(index="model", columns="item_id", values="correct").reindex(columns=order).astype("float32").sort_index())
    wide.columns.name = None
    return wide


def write(df: pd.DataFrame, schema: pa.Schema, path: Path) -> None:
    """
    Args:
        df (pd.DataFrame): Table to write; columns must match schema.
        schema (pa.Schema): Arrow schema to cast to.
        path (Path): Output file.

    Returns:
        None.
    """
    pq.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False), path)


def main() -> None:
    """
    Command-line args:
        --data (str): Parquet store root (contains long/); outputs are written here.
        --raw (str): download_details.py output folder, used to build the items table.

    Returns:
        None. Writes items.parquet, models.parquet and matrices/*.parquet; exits
        with an error if any model_revision is missing or an item_id is unknown.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Parquet store root (contains long/)")
    ap.add_argument("--raw", required=True, help="download_details.py output folder")
    args = ap.parse_args()
    data, raw_root = Path(args.data), Path(args.raw)

    long = pd.read_parquet(data / "long")
    no_rev = long.loc[long["model_revision"].isna(), "model_id"].unique().tolist()
    if no_rev:
        sys.exit(f"model_revision is missing for {no_rev}; every matrix row needs one")
    print(f"long table: {len(long)} rows, {long['run_id'].nunique()} runs, {long['model_id'].nunique()} models")

    parts = []
    for (bench, version), _ in long.groupby(["benchmark", "benchmark_version"]):
        if bench not in BENCHMARKS:
            sys.exit(f"{bench} isn't in BENCHMARKS (utils/ingest.py); add it there")
        parts.append(public_items(raw_root, bench, version))
    items = pd.concat(parts, ignore_index=True)
    missing = set(long["item_id"]) - set(items["item_id"])
    if missing:
        sys.exit(f"{len(missing)} long-table item_ids not in items, e.g. {sorted(missing)[:3]}")
    write(items, ITEMS_SCHEMA, data / "items.parquet")
    print(f"items.parquet:  {len(items)} items")

    models = build_models(long)
    no_meta = models.loc[models["family"].isna(), "model_id"].tolist()
    if no_meta:
        print(f"  note: no summary-table metadata for {no_meta[:5]}")
    write(models, MODELS_SCHEMA, data / "models.parquet")
    print(f"models.parquet: {len(models)} models")

    out_dir = data / "matrices"
    out_dir.mkdir(exist_ok=True)
    for (bench, cfg), _ in long.groupby(["benchmark", "config_id"]):
        wide = build_wide(long, items, bench, cfg)
        path = out_dir / f"{bench}__{cfg}.parquet"
        wide.to_parquet(path)
        filled = np.isfinite(wide.to_numpy()).mean()
        print(f"matrix {path.name}: {wide.shape[0]} models x {wide.shape[1]} items, {filled:.1%} filled")


if __name__ == "__main__":
    main()
