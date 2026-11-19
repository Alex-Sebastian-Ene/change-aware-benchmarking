"""Bayesian (variational) IRT training versus the gradient-descent baseline.

Same splits, seeds, k and dimensions as the progression runs, so the gradient-descent
side is read from results/irt/progression/{information,voptimal}/ (05 / 07) and only the
Bayesian fit is evaluated here. Both fits are scored on held-out answers as well.

Run: uv run --group irt python scripts/irt/run_bayesian.py
"""

import numpy as np
import pandas as pd
from diagnostics import response_diagnostics
from efficbench import ROOT, load
from protocol import evaluate, fit_context, make_splits
from run_progression import run_selection

from cabench.irt import fit_mirt, fit_mirt_bayes

# D=1 matches the 05 baseline methods; D=3 is the dimension 06 chose on validation.
STAGES = {1: ["random", "IRT anchors"], 3: ["random", "voptimal_adaptive"]}
FITS = {"gradient descent": fit_mirt, "bayesian": fit_mirt_bayes}


def main():
    Y, models, items = load()
    Y = Y.astype(np.float32)
    splits = make_splits(Y)
    subjects, ids = items.subject.values, models.model_id.values
    out = ROOT / "results/irt/bayesian"
    out.mkdir(parents=True, exist_ok=True)

    answers = []
    for d, selectors in STAGES.items():
        contexts = {}
        for name, fit in FITS.items():
            contexts[name] = [
                fit_context(Y[tr], subjects, n_dims=d, fit=fit) for tr, _, _ in splits
            ]
            for s, ((_, _, te), ctx) in enumerate(zip(splits, contexts[name], strict=True)):
                answers.append(response_diagnostics(Y[te], ctx).assign(split=s, fit=name))
            print(f"D={d} {name} fitted", flush=True)
        # check the saved gradient-descent results still reproduce before reusing them
        check = evaluate(
            Y,
            splits[:1],
            contexts["gradient descent"][:1],
            [("random", "p-IRT")],
            ks=[50],
            seeds=range(3),
            solver="newton",
        )
        stage = "information" if d == 1 else "voptimal"
        saved = pd.read_csv(ROOT / f"results/irt/progression/{stage}/test.csv")
        saved = saved.query("split == 0 and k == 50 and method == 'random + p-IRT' and seed < 3")
        assert np.allclose(check.mae.values, saved.mae.values, atol=1e-4), (check.mae, saved.mae)
        run_selection(Y, splits, contexts["bayesian"], selectors, out / f"d{d}", ids)
    pd.concat(answers).to_csv(out / "answers.csv", index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
