"""Experiments 05-07: gp-IRT / Fisher, dimensionality, then V-optimal MIRT. Stop there.

Run all: uv run --group irt python scripts/irt/run_progression.py
Quick integration check on the real matrix: add --smoke (separate outputs, not results).
"""

import argparse

import numpy as np
import pandas as pd
from diagnostics import (
    choose_dimension,
    dimension_diagnostics,
    paired_comparison,
    response_diagnostics,
)
from efficbench import ROOT, load
from protocol import calibrate_gp, evaluate, fit_context, make_splits
from run_experiment import KS, SEEDS

from cabench.design import as_context

DIMS = [1, 2, 3, 5, 8]
INFORMATION_SELECTORS = ["random", "IRT anchors", "fisher_static", "fisher_adaptive"]
VOPTIMAL_SELECTORS = ["random", "IRT anchors", "voptimal_static", "voptimal_adaptive"]
ESTIMATORS = ["accuracy", "p-IRT", "gp-IRT"]


def save_test(Y, splits, contexts, selectors, out, model_ids, ks=KS, seeds=SEEDS, item_ids=None):
    methods = [(sel, est) for sel in selectors for est in ESTIMATORS]
    metrics, records = evaluate(
        Y,
        splits,
        contexts,
        methods,
        ks=ks,
        seeds=seeds,
        model_ids=model_ids,
        solver="newton",
        return_records=True,
        item_ids=item_ids,
    )
    metrics.to_csv(out / "test.csv", index=False)
    records.to_csv(out / "predictions.csv", index=False)
    paired_comparison(records).to_csv(out / "paired.csv", index=False)
    return metrics, records


def run_selection(
    Y, splits, contexts, selectors, out, model_ids, ks=KS, seeds=SEEDS, item_ids=None
):
    out.mkdir(parents=True, exist_ok=True)
    val, weights = calibrate_gp(Y, splits, contexts, selectors, ks, seeds, model_ids, item_ids)
    # Keep tuning receipts before evaluating test; never infer a blend weight on test.
    val.to_csv(out / "validation_predictions.csv", index=False)
    weights.to_csv(out / "gp_weights.csv", index=False)
    return save_test(Y, splits, contexts, selectors, out, model_ids, ks, seeds, item_ids)


def run_mirt(
    Y,
    splits,
    subjects,
    model_ids,
    out,
    dims=DIMS,
    ks=KS,
    seeds=SEEDS,
    n_null=100,
    context_factory=fit_context,
    contexts_by_dim=None,
    item_ids=None,
    **fit_kwargs,
):
    """Hold selection fixed at random for every D; choose D using validation only."""
    out.mkdir(parents=True, exist_ok=True)
    diagnostic_rows = []
    for s, (train, _, _) in enumerate(splits):
        diagnostic_rows.append(dimension_diagnostics(Y[train], n_null=n_null).assign(split=s))
    pd.concat(diagnostic_rows).to_csv(out / "dimensionality.csv", index=False)

    contexts, validation, weights, responses = {}, [], [], []
    for d in dims:
        contexts[d] = (
            contexts_by_dim[d]
            if contexts_by_dim is not None
            else [
                context_factory(Y[train], subjects, n_dims=d, **fit_kwargs)
                for train, _, _ in splits
            ]
        )
        for (train, _, _), ctx in zip(splits, contexts[d], strict=True):
            if as_context(ctx)["n_dims"] != d:
                raise ValueError(f"The supplied D={d} model has a different ability dimension")
            ctx.setdefault("item_mean", Y[train].mean(0))
        val, w = calibrate_gp(Y, splits, contexts[d], ["random"], ks, seeds, model_ids, item_ids)
        validation.append(val)
        weights.append(w)
        for s, ((_, val_idx, _), ctx) in enumerate(zip(splits, contexts[d], strict=True)):
            responses.append(response_diagnostics(Y[val_idx], ctx).assign(split=s))
        print(f"D={d} validation done", flush=True)

    val = pd.concat(validation, ignore_index=True)
    chosen = choose_dimension(val)
    val.to_csv(out / "validation_predictions.csv", index=False)
    pd.concat(weights).to_csv(out / "gp_weights.csv", index=False)
    pd.concat(responses).to_csv(out / "response_validation.csv", index=False)
    pd.DataFrame(
        {"n_dims": [chosen], "criterion": ["random + gp-IRT validation mean log-MAE"]}
    ).to_csv(out / "chosen_dimension.csv", index=False)
    print(f"D={chosen} chosen on validation; now evaluate test", flush=True)

    test, records = [], []
    for d in dims:
        metrics, pred = evaluate(
            Y,
            splits,
            contexts[d],
            [("random", est) for est in ESTIMATORS],
            ks=ks,
            seeds=seeds,
            solver="newton",
            return_records=True,
            model_ids=model_ids,
            item_ids=item_ids,
        )
        test.append(metrics)
        records.append(pred)
    pd.concat(test).to_csv(out / "test.csv", index=False)
    pred = pd.concat(records, ignore_index=True)
    pred.to_csv(out / "predictions.csv", index=False)
    reference_dims = min(dims)
    paired = paired_comparison(pred, reference_dims=reference_dims)
    paired["reference_dims"] = reference_dims
    paired.to_csv(out / "paired.csv", index=False)
    return chosen, contexts[chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "fisher", "mirt", "voptimal"], default="all")
    parser.add_argument(
        "--smoke", action="store_true", help="one split, two seeds, k=10/20, 100 epochs"
    )
    args = parser.parse_args()
    Y, models, items = load()
    Y = Y.astype(np.float32)
    splits = make_splits(Y)
    ks, seeds, fit_kwargs, n_null = KS, SEEDS, {}, 100
    out = ROOT / "results/irt/progression"
    if args.smoke:
        splits, ks, seeds, fit_kwargs, n_null = splits[:1], [10, 20], range(2), {"epochs": 100}, 5
        out = out / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "smoke": [args.smoke],
            "fit_epochs": [fit_kwargs.get("epochs", 2000)],
            "n_splits": [len(splits)],
            "n_seeds": [len(seeds)],
            "ks": [str(list(ks))],
        }
    ).to_csv(out / "protocol.csv", index=False)
    subjects, ids = items.subject.values, models.model_id.values
    if args.stage in {"all", "fisher"}:
        contexts = [fit_context(Y[train], subjects, **fit_kwargs) for train, _, _ in splits]
        run_selection(
            Y, splits, contexts, INFORMATION_SELECTORS, out / "information", ids, ks, seeds
        )
    chosen_contexts = None
    if args.stage in {"all", "mirt"}:
        _, chosen_contexts = run_mirt(
            Y,
            splits,
            subjects,
            ids,
            out / "mirt",
            ks=ks,
            seeds=seeds,
            n_null=n_null,
            **fit_kwargs,
        )
    if args.stage in {"all", "voptimal"}:
        if chosen_contexts is None:
            choice = out / "mirt/chosen_dimension.csv"
            if not choice.exists():
                parser.error("Run --stage mirt first; V-optimality uses its validation-selected D")
            d = int(pd.read_csv(choice).n_dims.iloc[0])
            chosen_contexts = [
                fit_context(Y[train], subjects, n_dims=d, **fit_kwargs) for train, _, _ in splits
            ]
        run_selection(
            Y, splits, chosen_contexts, VOPTIMAL_SELECTORS, out / "voptimal", ids, ks, seeds
        )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
