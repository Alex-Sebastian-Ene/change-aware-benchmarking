"""Arrow schemas for the long table and the two side tables (spec v1).

Store everything as Parquet with these schemas so types never drift between
batches. `conform` casts a pandas frame to the schema and fails loudly on
missing or extra columns rather than guessing.
"""
from __future__ import annotations

import pandas as pd
import pyarrow as pa

SCHEMA_VERSION = 1

LONG = pa.schema([
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("source", pa.string(), nullable=False),          # public | own
    pa.field("model_id", pa.string(), nullable=False),
    pa.field("model_revision", pa.string(), nullable=False),
    pa.field("benchmark", pa.string(), nullable=False),
    pa.field("benchmark_version", pa.string(), nullable=False),
    pa.field("item_id", pa.string(), nullable=False),
    pa.field("config_id", pa.string(), nullable=False),
    pa.field("harness_commit", pa.string(), nullable=False),
    pa.field("correct", pa.int8(), nullable=False),
    pa.field("logprob", pa.float32()),                        # null for generative
    pa.field("pred", pa.string()),
    pa.field("prompt_tokens", pa.int32()),
    pa.field("gen_tokens", pa.int32(), nullable=False),
    pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("schema_version", pa.int16(), nullable=False),
])

MODELS = pa.schema([
    pa.field("model_id", pa.string(), nullable=False),
    pa.field("model_revision", pa.string(), nullable=False),
    pa.field("family", pa.string()),
    pa.field("params_b", pa.float32()),
    pa.field("checkpoint_step", pa.int64()),
    pa.field("parent_model", pa.string()),
    pa.field("source", pa.string(), nullable=False),
    pa.field("split", pa.string()),   # calibration | held_out | extrapolation | checkpoint
])

ITEMS = pa.schema([
    pa.field("item_id", pa.string(), nullable=False),
    pa.field("benchmark", pa.string(), nullable=False),
    pa.field("benchmark_version", pa.string(), nullable=False),
    pa.field("subtask", pa.string()),
    pa.field("source_index", pa.string()),
    pa.field("answer_key", pa.string()),
    pa.field("ref_prompt_tokens", pa.int32()),
])

ALLOWED = {
    ("long", "source"): {"public", "own"},
    ("models", "source"): {"public", "own"},
    ("models", "split"): {"calibration", "held_out", "extrapolation", "checkpoint", None},
    ("long", "correct"): {0, 1},
}


def conform(df: pd.DataFrame, schema: pa.Schema, name: str) -> pa.Table:
    """
    Args:
        df (pd.DataFrame): Table to convert; must have exactly the schema's columns.
        schema (pa.Schema): Target schema (LONG, MODELS or ITEMS).
        name (str): Table name ("long", "models" or "items"); selects the ALLOWED
            value checks and appears in error messages.

    Returns:
        pa.Table: The table with columns in schema order and schema types.
    """
    cols = set(df.columns)
    want = set(schema.names)
    if cols != want:
        raise ValueError(f"{name}: missing {sorted(want - cols)}, extra {sorted(cols - want)}")
    for (table, col), allowed in ALLOWED.items():
        if table == name:
            bad = set(df[col].where(df[col].notna(), None).unique()) - allowed
            if bad:
                raise ValueError(f"{name}.{col}: invalid values {sorted(map(str, bad))}")
    if name == "long" and df["correct"].isna().any():
        raise ValueError("long.correct must never be null for a scored response")
    return pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False, safe=True)
