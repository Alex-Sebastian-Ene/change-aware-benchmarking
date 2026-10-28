"""MMLU response matrix from efficbench's mmlu_fields.pickle.

Builds data/processed/efficbench_mmlu/{responses.npy, models.csv, items.csv}.
Run from the repository root: uv run python scripts/irt/efficbench.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/mmlu_fields.pickle"
PROCESSED = ROOT / "data/processed/efficbench_mmlu"
PREFIX = "harness_hendrycksTest_"


def build():
    # pickle can run code on load; this file was checked to reference only numpy globals
    with open(RAW, "rb") as f:
        src = pickle.load(f)
    subjects = [k for k in src["data"] if k.startswith(PREFIX)]
    blocks = [src["data"][k]["correctness"] for k in subjects]  # each is (items, models)
    Y = np.concatenate(blocks).T
    assert set(np.unique(Y)) <= {0.0, 1.0}

    names = [s[len(PREFIX) : -len("_5")] for s in subjects]  # "_5" = 5-shot
    items = pd.DataFrame({"subject": np.repeat(names, [len(b) for b in blocks])})
    item_ids = items.subject + "_" + items.groupby("subject").cumcount().astype(str)
    items.insert(0, "item_id", item_ids)
    model_ids = [
        m.removeprefix("open-llm-leaderboard/details_").replace("__", "/", 1)
        for m in src["models"]
    ]
    models = pd.DataFrame({"model_id": model_ids})

    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED / "responses.npy", Y.astype(np.int8))
    models.to_csv(PROCESSED / "models.csv", index_label="model_idx")
    items.to_csv(PROCESSED / "items.csv", index_label="item_idx")
    print(f"Y {Y.shape}, {len(names)} subjects")


def load():
    """Y (models x items, int8 0/1), models, items."""
    return (
        np.load(PROCESSED / "responses.npy"),
        pd.read_csv(PROCESSED / "models.csv", index_col="model_idx"),
        pd.read_csv(PROCESSED / "items.csv", index_col="item_idx"),
    )


if __name__ == "__main__":
    build()
