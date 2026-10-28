"""Fixed evaluation protocol for IRT score estimation, and the minimal IRT baseline.

A method is a (selector, estimator) pair:
  selector(ctx, k, seed) -> (items, weights)   which k items a new model answers
  estimator(Y, selections, ctx) -> scores      full-benchmark score from those answers,
                                               one row per selection: (n_seeds, n_models)
`ctx` holds everything learned from the train models (IRT parameters) plus item metadata.

To try a new method, add it to SELECTORS / ESTIMATORS and list it in `methods`:
`evaluate` runs it on the same splits, k values and seeds as the baseline, so its rows
line up with results/irt/baseline.csv.
"""

import numpy as np
import pandas as pd
from run_experiment import FAMILY_DIST, KS, SEEDS, SPLIT_SEEDS, evaluate_scores
from sklearn.preprocessing import StandardScaler

from cabench.baselines import random_items, stratified_items
from cabench.irt import anchor_estimate, fit_mirt, irt_anchors, pirt_estimate
from cabench.splits import behaviour_families, split_models

N_DIMS = 1  # the minimal baseline: unidimensional 2PL
PARTS = {"train": 0, "val": 1, "test": 2}


def make_splits(Y):
    """(train, val, test) model indices for each split seed, behavioural families kept together."""
    families = behaviour_families(Y, FAMILY_DIST)
    return [split_models(len(Y), s, groups=families) for s in SPLIT_SEEDS]


def fit_context(Y_train, subjects, n_dims=N_DIMS):
    """Fit IRT on the train models and bundle what selectors and estimators may use."""
    theta, alpha, beta, history = fit_mirt(Y_train, n_dims=n_dims)
    return {
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "history": history,
        "emb": StandardScaler().fit_transform(np.column_stack([alpha, beta])),
        "subjects": subjects,
    }


def select_random(ctx, k, seed):
    rng = np.random.default_rng(seed)
    return random_items(len(ctx["beta"]), k, rng), np.full(k, 1 / k)


def select_stratified(ctx, k, seed):
    rng = np.random.default_rng(seed)
    return stratified_items(ctx["subjects"], k, rng), np.full(k, 1 / k)


def select_irt_anchors(ctx, k, seed):
    return irt_anchors(ctx["emb"], k, seed)


def estimate_accuracy(Y, selections, ctx):
    """Weighted accuracy on the selected items (plain accuracy when the weights are uniform)."""
    return np.array([anchor_estimate(Y, items, weights) for items, weights in selections])


def estimate_pirt(Y, selections, ctx):
    """p-IRT: infer theta from the selected answers, predict every other item."""
    seen = np.zeros((len(selections),) + Y.shape, dtype=bool)
    for i, (items, _) in enumerate(selections):
        seen[i][:, items] = True
    # one theta fit for all selections at once: each row is a (selection, model) pair
    Y_tiled = np.tile(Y, (len(selections), 1))
    seen = seen.reshape(-1, Y.shape[1])
    scores = pirt_estimate(Y_tiled, seen, ctx["alpha"], ctx["beta"], ctx["theta"])
    return scores.reshape(len(selections), -1)


SELECTORS = {
    "random": select_random,
    "stratified": select_stratified,
    "IRT anchors": select_irt_anchors,
}
ESTIMATORS = {
    "accuracy": estimate_accuracy,
    "p-IRT": estimate_pirt,
}
# non-IRT floors, IRT estimator only, IRT selection only, and both
BASELINE = [
    ("random", "accuracy"),
    ("stratified", "accuracy"),
    ("random", "p-IRT"),
    ("IRT anchors", "accuracy"),
    ("IRT anchors", "p-IRT"),
]


def evaluate(Y, splits, contexts, methods=BASELINE, part="test", ks=KS, seeds=SEEDS):
    """One row per (split, k, method, seed) with the metrics of `evaluate_scores`."""
    rows = []
    for s, (split, ctx) in enumerate(zip(splits, contexts, strict=True)):
        Y_held_out = Y[split[PARTS[part]]]
        true = Y_held_out.mean(1)
        for k in ks:
            # each selector runs once per seed, shared by every estimator that uses it
            selections = {}
            for name in dict.fromkeys(sel for sel, _ in methods):
                selections[name] = [SELECTORS[name](ctx, k, seed) for seed in seeds]
                for items, weights in selections[name]:
                    assert len(set(items)) == k and np.isclose(weights.sum(), 1), name

            for sel, est in methods:
                preds = ESTIMATORS[est](Y_held_out, selections[sel], ctx)
                for seed, pred in zip(seeds, preds, strict=True):
                    rows.append({
                        "split": s, "k": k, "method": f"{sel} + {est}",
                        "selector": sel, "estimator": est, "seed": seed,
                        **evaluate_scores(true, pred),
                    })
        print(f"split {s} done", flush=True)
    return pd.DataFrame(rows)
