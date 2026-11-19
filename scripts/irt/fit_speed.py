"""Training time and held-out answer fit: gradient descent (fit_mirt) vs Bayesian (fit_mirt_bayes).

Same family splits as the baseline. For each epoch budget, time the fit on the train models
and score the hidden half of each test model's answers (response_diagnostics).

Run: uv run --group irt python scripts/irt/fit_speed.py
"""

import time

import numpy as np
import pandas as pd
from diagnostics import response_diagnostics
from efficbench import ROOT, load
from protocol import fit_context, make_splits

from cabench.irt import fit_mirt, fit_mirt_bayes

FITS = {"gradient descent": fit_mirt, "bayesian": fit_mirt_bayes}
EPOCHS = [100, 250, 500, 1000, 2000]
DIMS = [1, 3]
SPLITS = [0, 1]


def main():
    Y, _, items = load()
    Y = Y.astype(np.float32)
    splits = make_splits(Y)
    rows = []
    for split in SPLITS:
        train, _, test = splits[split]
        for d in DIMS:
            for name, fit in FITS.items():
                for epochs in EPOCHS:
                    start = time.perf_counter()
                    ctx = fit_context(Y[train], items.subject.values, d, fit=fit, epochs=epochs)
                    seconds = time.perf_counter() - start
                    answers = response_diagnostics(Y[test], ctx).query("predictor == 'IRT'")
                    rows.append(
                        {"split": split, "n_dims": d, "fit": name, "epochs": epochs}
                        | {"seconds": seconds}
                        | answers[["log_loss", "brier", "ece"]].iloc[0].to_dict()
                    )
                    print(rows[-1], flush=True)
    out = ROOT / "results/irt/bayesian/fit_speed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
