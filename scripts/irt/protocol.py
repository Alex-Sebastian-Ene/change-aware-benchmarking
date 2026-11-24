"""Fixed evaluation protocol for IRT score estimation, and the minimal IRT baseline.

A method is a (selector, estimator) pair:
  selector(ctx, k, seed) -> (items, weights)   which k items a new model answers
  estimator(Y, selections, ctx) -> scores      full-benchmark score from those answers,
                                               one row per selection: (n_seeds, n_models)
`ctx` holds everything learned from the train models (IRT parameters) plus item metadata.
Adaptive selectors additionally read selected answers and return per-model item indices.
calibrate_gp fits blend weights on validation before evaluate uses them on test.

To try a new method, add it to SELECTORS / ESTIMATORS and list it in `methods`:
`evaluate` runs it on the same splits, k values and seeds as the baseline, so its rows
line up with results/irt/baseline.csv.
"""

import numpy as np
import pandas as pd
from run_experiment import FAMILY_DIST, KS, SEEDS, SPLIT_SEEDS, evaluate_scores
from sklearn.preprocessing import StandardScaler

from cabench.baselines import random_items, stratified_items
from cabench.design import adaptive_design, as_context, parameter_context, static_design
from cabench.irt import (
    fit_mirt,
    gp_estimate,
    gp_weight,
    irt_anchors,
    pirt_estimate,
)
from cabench.splits import behaviour_families, split_models

N_DIMS = 1  # the minimal baseline: unidimensional 2PL
PARTS = {"train": 0, "val": 1, "test": 2}


def make_splits(Y):
    """(train, val, test) model indices for each split seed, behavioural families kept together."""
    families = behaviour_families(Y, FAMILY_DIST)
    return [split_models(len(Y), s, groups=families) for s in SPLIT_SEEDS]


def fit_context(Y_train, subjects, n_dims=N_DIMS, item_ids=None, **fit_kwargs):
    """Fit IRT on the train models and bundle what selectors and estimators may use."""
    theta, alpha, beta, history = fit_mirt(Y_train, n_dims=n_dims, **fit_kwargs)
    return {
        **parameter_context(theta, alpha, beta, subjects=subjects, item_ids=item_ids),
        "history": history,
        "emb": StandardScaler().fit_transform(np.column_stack([alpha, beta])),
        "subjects": subjects,
        "item_mean": Y_train.mean(0),
    }


def select_random(ctx, k, seed):
    rng = np.random.default_rng(seed)
    return random_items(as_context(ctx)["n_items"], k, rng), np.full(k, 1 / k)


def select_stratified(ctx, k, seed):
    rng = np.random.default_rng(seed)
    return stratified_items(ctx["subjects"], k, rng), np.full(k, 1 / k)


def select_irt_anchors(ctx, k, seed):
    if ctx.get("emb") is None:
        raise ValueError("IRT anchors need item embeddings; supply them in model_context")
    return irt_anchors(ctx["emb"], k, seed)


def select_fisher_static(ctx, k, seed):
    return static_design(ctx, k, "fisher"), np.full(k, 1 / k)


def select_voptimal_static(ctx, k, seed):
    return static_design(ctx, k, "voptimal"), np.full(k, 1 / k)


def selected_answers(Y, items):
    return Y[:, items] if items.ndim == 1 else np.take_along_axis(Y, items, axis=1)


def estimate_accuracy(Y, selections, ctx):
    """Weighted accuracy on the selected items (plain accuracy when the weights are uniform)."""
    return np.array([selected_answers(Y, items) @ weights for items, weights in selections])


def estimate_pirt(Y, selections, ctx, solver="adam", thetas=None):
    """p-IRT: infer theta from the selected answers, predict every other item."""
    if solver not in {"adam", "newton"}:
        raise ValueError(f"Unknown theta solver: {solver}")
    ctx = as_context(ctx)
    if solver == "newton" or thetas is not None or "alpha" not in ctx:
        scores = []
        for s, (items, _) in enumerate(selections):
            observed = selected_answers(Y, items)
            theta = ctx["infer"](observed, items) if thetas is None else thetas[s]
            p = np.asarray(ctx["predict"](theta))
            if p.shape != Y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError("predict must return (models, items) probabilities in [0, 1]")
            p_selected = selected_answers(p, items)
            # Preserve the baseline's p-IRT definition: actual answers on seen items.
            scores.append(p.mean(1) + (observed - p_selected).sum(1) / Y.shape[1])
        return np.array(scores)
    seen = np.zeros((len(selections),) + Y.shape, dtype=bool)
    for i, (items, _) in enumerate(selections):
        if items.ndim == 1:
            seen[i][:, items] = True
        else:
            np.put_along_axis(seen[i], items, True, axis=1)
    # one theta fit for all selections at once: each row is a (selection, model) pair
    Y_tiled = np.tile(Y, (len(selections), 1))
    seen = seen.reshape(-1, Y.shape[1])
    scores = pirt_estimate(Y_tiled, seen, ctx["alpha"], ctx["beta"], ctx["theta"])
    return scores.reshape(len(selections), -1)


def estimate_gpirt(Y, selections, ctx):
    """Validation-fitted blend; evaluate shares the two component predictions."""
    return gp_estimate(
        estimate_accuracy(Y, selections, ctx), estimate_pirt(Y, selections, ctx), ctx["gp_weight"]
    )


SELECTORS = {
    "random": select_random,
    "stratified": select_stratified,
    "IRT anchors": select_irt_anchors,
    "fisher_static": select_fisher_static,
    "voptimal_static": select_voptimal_static,
}
# Adaptive selectors need answers as they go. They return one trajectory for all k.
ADAPTIVE_SELECTORS = {"fisher_adaptive": "fisher", "voptimal_adaptive": "voptimal"}
ESTIMATORS = {
    "accuracy": estimate_accuracy,
    "p-IRT": estimate_pirt,
    "gp-IRT": estimate_gpirt,
}
# non-IRT floors, IRT estimator only, IRT selection only, and both
BASELINE = [
    ("random", "accuracy"),
    ("stratified", "accuracy"),
    ("random", "p-IRT"),
    ("IRT anchors", "accuracy"),
    ("IRT anchors", "p-IRT"),
]


def evaluate(
    Y,
    splits,
    contexts,
    methods=BASELINE,
    part="test",
    ks=KS,
    seeds=SEEDS,
    return_records=False,
    model_ids=None,
    solver="adam",
    item_ids=None,
):
    """Same baseline defaults; optionally return (metrics, per-model records).

    New experiments use solver='newton' for selected-answer-only MAP fits; the baseline
    keeps its original Adam solver. Adaptive designs are deterministic, so their single
    trajectory is shared across seeds. Those repeated rows are not independent trials.
    """
    ks, seeds = list(ks), list(seeds)
    if not ks or not seeds or min(ks) < 1 or max(ks) > Y.shape[1]:
        raise ValueError("Use nonempty seeds and k values between 1 and the number of items")
    if len(set(ks)) != len(ks) or len(set(seeds)) != len(seeds):
        raise ValueError("k values and seeds must be unique")
    rows, records = [], []
    names = list(dict.fromkeys(sel for sel, _ in methods))
    if set(names) - (SELECTORS.keys() | ADAPTIVE_SELECTORS.keys()):
        raise ValueError("Unknown selector")
    if {est for _, est in methods} - ESTIMATORS.keys():
        raise ValueError("Unknown estimator")
    model_ids = np.arange(len(Y)) if model_ids is None else np.asarray(model_ids)
    if len(model_ids) != len(Y) or len(np.unique(model_ids)) != len(Y):
        raise ValueError("Provide one unique model_id per row of Y")
    for s, (split, ctx) in enumerate(zip(splits, contexts, strict=True)):
        ctx = as_context(ctx)
        if ctx["n_items"] != Y.shape[1]:
            raise ValueError("Response columns must match the model's items")
        if item_ids is not None and ctx.get("item_ids") is not None:
            if not np.array_equal(item_ids, ctx["item_ids"]):
                raise ValueError("Model item_ids must match response column order")
        Y_held_out = Y[split[PARTS[part]]]
        true = Y_held_out.mean(1)
        trajectories = {
            name: {
                k: (items, theta)
                for k, items, theta in adaptive_design(
                    Y_held_out,
                    ctx,
                    ks,
                    ADAPTIVE_SELECTORS[name],
                )
            }
            for name in names
            if name in ADAPTIVE_SELECTORS
        }
        static_orders = {
            name: SELECTORS[name](ctx, max(ks), seeds[0])[0]
            for name in names
            if name in {"fisher_static", "voptimal_static"}
        }
        for k in ks:
            # each selector runs once per seed, shared by every estimator that uses it
            selections, predictions = {}, {}
            for name in names:
                deterministic = name in trajectories or name in static_orders
                if name in trajectories:
                    items, theta = trajectories[name][k]
                    selections[name] = [(items, np.full(k, 1 / k))]
                elif name in static_orders:
                    selections[name] = [(static_orders[name][:k], np.full(k, 1 / k))]
                else:
                    selections[name] = [SELECTORS[name](ctx, k, seed) for seed in seeds]
                for items, weights in selections[name]:
                    assert items.shape[-1] == k and np.isclose(weights.sum(), 1), name
                    assert np.all(np.diff(np.sort(items, axis=-1), axis=-1) != 0), name
                estimators = {est for sel, est in methods if sel == name}
                if "gp-IRT" in estimators:
                    estimators |= {"accuracy", "p-IRT"}
                for est in estimators - {"gp-IRT"}:
                    if est == "p-IRT":
                        pred = estimate_pirt(
                            Y_held_out,
                            selections[name],
                            ctx,
                            solver=solver,
                            thetas=[theta] if name in trajectories else None,
                        )
                    else:
                        pred = ESTIMATORS[est](Y_held_out, selections[name], ctx)
                    predictions[name, est] = (
                        np.repeat(pred, len(seeds), axis=0) if deterministic else pred
                    )
                if "gp-IRT" in estimators:
                    # Missing calibration must fail; never fit on test.
                    lam = ctx["gp_weights"][name, k]
                    predictions[name, "gp-IRT"] = gp_estimate(
                        predictions[name, "accuracy"], predictions[name, "p-IRT"], lam
                    )

            for sel, est in methods:
                preds = predictions[sel, est]
                for seed, pred in zip(seeds, preds, strict=True):
                    keys = {
                        "split": s,
                        "k": k,
                        "method": f"{sel} + {est}",
                        "selector": sel,
                        "estimator": est,
                        "seed": seed,
                        "n_dims": ctx["n_dims"],
                    }
                    rows.append({**keys, **evaluate_scores(true, pred)})
                    if return_records:
                        records.extend(
                            {
                                **keys,
                                "model_id": model_id,
                                "true_score": actual,
                                "pred_score": score,
                                "abs_error": abs(score - actual),
                                "signed_error": score - actual,
                            }
                            for model_id, actual, score in zip(
                                model_ids[split[PARTS[part]]], true, pred, strict=True
                            )
                        )
        print(f"split {s} done", flush=True)
    metrics = pd.DataFrame(rows)
    return (metrics, pd.DataFrame(records)) if return_records else metrics


def calibrate_gp(Y, splits, contexts, selectors, ks=KS, seeds=SEEDS, model_ids=None, item_ids=None):
    """Fit the existing gp-IRT blend separately per split, selector and k on validation.

    Returns validation per-model records (including the fitted blend) and the weights.
    Validation blend error is in-sample tuning error, not an unbiased performance estimate.
    """
    methods = [(sel, est) for sel in selectors for est in ("accuracy", "p-IRT")]
    _, records = evaluate(
        Y,
        splits,
        contexts,
        methods,
        part="val",
        ks=ks,
        seeds=seeds,
        model_ids=model_ids,
        solver="newton",
        return_records=True,
        item_ids=item_ids,
    )
    keys = ["split", "selector", "n_dims", "k", "seed", "model_id"]
    accuracy = records[records.estimator == "accuracy"].set_index(keys)
    pirt = records[records.estimator == "p-IRT"].set_index(keys)
    joined = accuracy.join(pirt, lsuffix="_accuracy", rsuffix="_pirt", validate="one_to_one")
    weights, blends = [], []
    for (s, sel, k), group in joined.groupby(["split", "selector", "k"]):
        lam = gp_weight(group.signed_error_accuracy.values, group.signed_error_pirt.values)
        contexts[s].setdefault("gp_weights", {})[sel, k] = lam
        weights.append(
            {
                "split": s,
                "selector": sel,
                "k": k,
                "n_dims": as_context(contexts[s])["n_dims"],
                "gp_weight": lam,
            }
        )
        blend = group.reset_index()[keys].copy()
        blend["method"], blend["estimator"] = f"{sel} + gp-IRT", "gp-IRT"
        blend["true_score"] = group.true_score_pirt.values
        blend["pred_score"] = gp_estimate(
            group.pred_score_accuracy.values, group.pred_score_pirt.values, lam
        )
        blend["signed_error"] = blend.pred_score - blend.true_score
        blend["abs_error"] = blend.signed_error.abs()
        blends.append(blend)
    return pd.concat([records, *blends], ignore_index=True), pd.DataFrame(weights)
