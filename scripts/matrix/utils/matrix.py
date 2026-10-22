"""Long table -> wide matrix (spec: "Wide matrix").

One matrix per (benchmark, config_id). Rows model_id@model_revision, columns
item_id in items-table order, values 1/0/NaN. Missing responses stay NaN.
The earliest run per (model, revision, config) is canonical; later runs are
returned separately for the noise reference.
"""
from __future__ import annotations

import pandas as pd


def _row_key(df: pd.DataFrame) -> pd.Series:
    """
    Args:
        df (pd.DataFrame): Rows with model_id and model_revision columns.

    Returns:
        pd.Series: "model_id@model_revision" for each row.
    """
    return df["model_id"] + "@" + df["model_revision"]


def split_canonical(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Args:
        long_df (pd.DataFrame): Long-table rows.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (rows from canonical runs, rows from repeat runs).
    """
    keys = ["model_id", "model_revision", "benchmark", "config_id"]
    starts = (long_df.groupby(keys + ["run_id"], as_index=False)["timestamp"].min().sort_values(keys + ["timestamp", "run_id"]))
    canon_runs = starts.drop_duplicates(keys, keep="first")["run_id"]
    is_canon = long_df["run_id"].isin(set(canon_runs))
    return long_df[is_canon], long_df[~is_canon]


def build_wide(long_df: pd.DataFrame, items_df: pd.DataFrame, benchmark: str, config_id: str) -> pd.DataFrame:
    """
    Args:
        long_df (pd.DataFrame): The long response table.
        items_df (pd.DataFrame): The items table; sets the column order.
        benchmark (str): Benchmark to select.
        config_id (str): Config to select.

    Returns:
        pd.DataFrame: float32 matrix indexed by "model" ("model_id@model_revision"),
        one column per item_id, values 1.0 / 0.0 / NaN.
    """
    sel = long_df[(long_df["benchmark"] == benchmark) & (long_df["config_id"] == config_id)]
    if sel.empty:
        raise ValueError(f"no rows for benchmark={benchmark!r} config_id={config_id!r}")
    canon, _ = split_canonical(sel)

    dup = canon.duplicated(["run_id", "item_id"])
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicate (run_id, item_id) keys; run the schema check first")

    item_order = items_df.loc[items_df["benchmark"] == benchmark, "item_id"].tolist()
    unknown = set(canon["item_id"]) - set(item_order)
    if unknown:
        raise ValueError(f"{len(unknown)} item_ids not in items table, e.g. {sorted(unknown)[:3]}")

    wide = (canon.assign(row=_row_key(canon)).pivot(index="row", columns="item_id", values="correct").reindex(columns=item_order).astype("float32"))
    wide.index.name = "model"
    wide.columns.name = None
    return wide


def build_all(long_df: pd.DataFrame, items_df: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Args:
        long_df (pd.DataFrame): The long response table.
        items_df (pd.DataFrame): The items table.

    Returns:
        dict[tuple[str, str], pd.DataFrame]: (benchmark, config_id) -> matrix from build_wide.
    """
    pairs = long_df[["benchmark", "config_id"]].drop_duplicates().itertuples(index=False)
    return {(b, c): build_wide(long_df, items_df, b, c) for b, c in pairs}
