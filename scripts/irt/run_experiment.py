"""Full protocol. Run from the repository root (~30-60 min on a laptop CPU):

    uv run --group irt python scripts/irt/run_experiment.py

For each family split:
  validation: fit MIRT for every D, evaluate every method for every k over SEEDS,
  fit the gp-IRT weight per k.
Pick the D with the lowest validation gp-IRT error (mean log-MAE, so every k counts equally).
Test: evaluate every method once with that D and the validation gp-IRT weights.
Writes results/irt/{validation,test,gp_weights}.csv.
"""

import numpy as np
import pandas as pd
from efficbench import ROOT, load
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from cabench.baselines import random_items, stratified_items
from cabench.irt import (
    anchor_estimate,
    fit_mirt,
    gp_estimate,
    gp_weight,
    irt_anchors,
    pirt_estimate,
)
from cabench.splits import behaviour_families, split_models

KS = [10, 20, 50, 100, 200, 500]
DIMS = [1, 2, 3, 5]
SPLIT_SEEDS = range(5)
SEEDS = range(20)
FAMILY_DIST = 0.3


def evaluate_scores(true_scores, pred_scores):
    err = pred_scores - true_scores
    return {
        "mae": np.abs(err).mean(),
        "rmse": np.sqrt((err**2).mean()),
        "bias": err.mean(),
        "p95_abs_err": np.percentile(np.abs(err), 95),
        "spearman": spearmanr(true_scores, pred_scores).statistic,
    }


def predict_all(Y, k, fit, subjects):
    """Predicted full scores per method, shape (len(SEEDS), n_models)."""
    theta, alpha, beta, emb = fit
    pred = {"random": [], "stratified": [], "IRT anchors": []}
    seen = np.zeros((len(SEEDS),) + Y.shape, dtype=bool)
    for i, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        pred["random"].append(Y[:, random_items(Y.shape[1], k, rng)].mean(1))
        pred["stratified"].append(Y[:, stratified_items(subjects, k, rng)].mean(1))
        anchors, weights = irt_anchors(emb, k, seed)
        pred["IRT anchors"].append(anchor_estimate(Y, anchors, weights))
        seen[i][:, anchors] = True
    # one theta fit for all seeds at once: each row is a (seed, model) pair
    Y_tiled = np.tile(Y, (len(SEEDS), 1))
    pirt = pirt_estimate(Y_tiled, seen.reshape(-1, Y.shape[1]), alpha, beta, theta)
    pred = {m: np.array(p) for m, p in pred.items()}
    pred["p-IRT"] = pirt.reshape(len(SEEDS), -1)
    return pred


def score_rows(pred, true, **keys):
    return [
        {**keys, "method": m, "seed": seed, **evaluate_scores(true, p[i])}
        for m, p in pred.items()
        for i, seed in enumerate(SEEDS)
    ]


def main():
    Y, _, items = load()
    Y = Y.astype(np.float32)
    subjects = items.subject.values
    families = behaviour_families(Y, FAMILY_DIST)
    splits = {s: split_models(len(Y), s, groups=families) for s in SPLIT_SEEDS}

    fits, gp_weights, val_rows = {}, {}, []
    for s, (train, val, _) in splits.items():
        true = Y[val].mean(1)
        for D in DIMS:
            theta, alpha, beta, _ = fit_mirt(Y[train], n_dims=D)
            emb = StandardScaler().fit_transform(np.column_stack([alpha, beta]))
            fits[s, D] = theta, alpha, beta, emb
            for k in KS:
                pred = predict_all(Y[val], k, fits[s, D], subjects)
                lam = gp_weight(pred["IRT anchors"] - true, pred["p-IRT"] - true)
                gp_weights[s, D, k] = lam
                pred["gp-IRT"] = gp_estimate(pred["IRT anchors"], pred["p-IRT"], lam)
                val_rows += score_rows(pred, true, split=s, D=D, k=k)
            print(f"split {s}  D={D}  done", flush=True)

    val = pd.DataFrame(val_rows)
    gp = val[val.method == "gp-IRT"]
    best_D = int(np.log(gp.groupby(["D", "k"]).mae.mean()).groupby("D").mean().idxmin())
    print(f"chosen D = {best_D}")

    test_rows = []
    for s, (_, _, test) in splits.items():
        true = Y[test].mean(1)
        for k in KS:
            pred = predict_all(Y[test], k, fits[s, best_D], subjects)
            lam = gp_weights[s, best_D, k]
            pred["gp-IRT"] = gp_estimate(pred["IRT anchors"], pred["p-IRT"], lam)
            test_rows += score_rows(pred, true, split=s, D=best_D, k=k)

    out = ROOT / "results/irt"
    out.mkdir(parents=True, exist_ok=True)
    val.to_csv(out / "validation.csv", index=False)
    pd.DataFrame(test_rows).to_csv(out / "test.csv", index=False)
    weights = pd.Series(gp_weights, name="gp_weight").rename_axis(["split", "D", "k"])
    weights.to_csv(out / "gp_weights.csv")


if __name__ == "__main__":
    main()
