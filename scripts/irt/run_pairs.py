"""Does IRT measure the score change between a parent model and its fine-tuned child?

Folds hold out whole lineages, so no model from a tested family helps fit the item
parameters. Each method estimates child score minus parent score from k child answers,
and the change on the full benchmark is the truth.

Run: uv run --group irt --group matrix python scripts/irt/run_pairs.py --data DATA
Quick integration check: add --smoke (separate outputs, not results).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from diagnostics import paired_comparison
from efficbench import ROOT
from protocol import estimate_pirt, fit_context
from run_experiment import KS, SEEDS

from cabench.design import adaptive_design
from cabench.pairs import (
    LEVELS,
    change_interval,
    corrected_change,
    drop_near_duplicates,
    find_pairs,
    heldout_losses,
    learn_prior,
    model_change,
    pair_problem,
    paired_change,
)

N_DIMS = 3  # the dimension 06 chose on validation for MMLU
N_FOLDS = 5
FOLD_SEED = 0

PAIRED, GAP, VOPTIMAL = "random + paired", "random + IRT score gap", "V-optimal + IRT score gap"
MODEL, CORRECTED = "random + parent model", "random + corrected parent model"
FOCUS = "fine-tuned"

# Pass criteria. The thresholds were set before any real pair was looked at.
CHECK_K = 100
MAX_LOSS_GAIN = 0.01
COVERAGE = {80: (0.77, 0.83), 95: (0.93, 0.97)}


def evaluate_pair(problem, pair_id, prior, ks, seeds):
    """One row per method, k and seed. Random items are nested across k within a seed."""
    n_items = len(problem["X"])
    ks = [k for k in ks if k <= n_items]
    y_child = problem["y_child"][None]
    parent_score = problem["y_parent"].mean()
    orders = [np.random.default_rng([seed, pair_id]).permutation(n_items) for seed in seeds]
    voptimal = {
        k: (items, theta)
        for k, items, theta in adaptive_design(y_child, problem["ctx"], ks, "voptimal")
    }
    rows = []

    def add(method, k, seed, estimate, residuals=None):
        interval = {} if residuals is None else change_interval(estimate, residuals, n_items)
        rows.append({"method": method, "k": k, "seed": seed, "pred_score": estimate, **interval})

    for k in ks:
        weights = np.full(k, 1 / k)
        selections = [(order[:k], weights) for order in orders]
        scores = estimate_pirt(y_child, selections, problem["ctx"], solver="newton")[:, 0]
        items, theta = voptimal[k]
        adaptive_score = estimate_pirt(
            y_child, [(items, weights)], problem["ctx"], solver="newton", thetas=[theta]
        )[0, 0]
        for seed, order, score in zip(seeds, orders, scores, strict=True):
            chosen = order[:k]
            add(PAIRED, k, seed, *paired_change(problem, chosen))
            add(GAP, k, seed, score - parent_score)
            add(VOPTIMAL, k, seed, adaptive_score - parent_score)
            add(MODEL, k, seed, model_change(problem, chosen, prior))
            add(CORRECTED, k, seed, *corrected_change(problem, chosen, prior))
    return rows


def conformal_intervals(records, method=VOPTIMAL):
    """Conformal intervals for a method with none of its own.

    The half-width for a fold is a quantile of the method's errors on the other folds,
    which are all out-of-sample. Coverage holds on average over fine-tunes like these,
    not for each one, and every pair gets the same width at a given k.
    """
    rows = records[records.method == method]
    errors = rows.groupby(["k", "split", "model_id"]).abs_error.mean()
    for (k, split), index in rows.groupby(["k", "split"]).groups.items():
        calibration = np.sort(errors.loc[k].drop(split, level="split").to_numpy())
        estimate = records.loc[index, "pred_score"]
        for level in LEVELS:
            rank = int(np.ceil((len(calibration) + 1) * level / 100))
            half_width = calibration[rank - 1] if rank <= len(calibration) else np.inf
            records.loc[index, f"low_{level}"] = estimate - half_width
            records.loc[index, f"high_{level}"] = estimate + half_width
    return records


def make_folds(pairs, n_folds=N_FOLDS, seed=FOLD_SEED):
    roots = np.random.default_rng(seed).permutation(pairs["root"].unique())
    return np.array_split(roots, n_folds)


def run_folds(Y, rows, pairs, roots, subject_codes, n_dims, ks, seeds, folds, **fit_kwargs):
    """Per-prediction records and per-pair held-out losses."""
    index = {row: i for i, row in enumerate(rows)}
    n_subjects = subject_codes.max() + 1

    def problem(pair, ctx):
        y_parent, y_child = Y[index[pair.parent]], Y[index[pair.child]]
        parameters = ctx["alpha"], ctx["beta"], ctx["theta"]
        return pair_problem(y_parent, y_child, *parameters, subject_codes, n_subjects)

    records, losses = [], []
    for fold, test_roots in enumerate(folds):
        train = ~np.isin(roots, test_roots)
        ctx = fit_context(Y[train], subject_codes, n_dims=n_dims, **fit_kwargs)
        is_test = pairs["root"].isin(test_roots)
        prior = learn_prior([problem(p, ctx) for p in pairs[~is_test].itertuples()])
        for pair in pairs[is_test].itertuples():
            current = problem(pair, ctx)
            keys = {
                "split": fold,
                "model_id": pair.child,
                "parent": pair.parent,
                "root": pair.root,
                "kind": pair.kind,
                "n_dims": n_dims,
            }
            for row in evaluate_pair(current, pair.Index, prior, ks, seeds):
                records.append({**keys, "true_score": current["true_change"], **row})
            for model, loss in heldout_losses(current, prior).items():
                losses.append({**keys, "model": model, "log_loss": loss})
        print(f"fold {fold} done", flush=True)
    records = pd.DataFrame(records)
    records["signed_error"] = records.pred_score - records.true_score
    records["abs_error"] = records.signed_error.abs()
    return records, pd.DataFrame(losses)


def summarise(records):
    """Error, direction and interval coverage per method and k, for each kind and overall."""
    r = records.assign(
        squared_error=records.signed_error**2,
        right_direction=(np.sign(records.pred_score) == np.sign(records.true_score)).where(
            records.true_score != 0
        ),
    )
    columns = ["abs_error", "squared_error", "signed_error", "right_direction"]
    for level in LEVELS:
        low, high = r[f"low_{level}"], r[f"high_{level}"]
        r[f"coverage_{level}"] = r.true_score.between(low, high).where(low.notna())
        r[f"width_{level}"] = high - low
        columns += [f"coverage_{level}", f"width_{level}"]
    r[columns] = r[columns].astype(float)
    both = pd.concat([r, r.assign(kind="all")])
    summary = both.groupby(["kind", "method", "k"])[columns].mean().reset_index()
    summary["rmse"] = np.sqrt(summary.pop("squared_error"))
    n_pairs = both.groupby(["kind", "method", "k"]).model_id.nunique()
    return summary.merge(n_pairs.rename("n_pairs").reset_index())


def check_results(summary, paired, losses):
    """Apply the pass criteria to the fine-tuned pairs."""
    at_k = summary[(summary.kind == FOCUS) & (summary.k == CHECK_K)].set_index("method")
    loss = losses[losses.kind == FOCUS].groupby("model").log_loss.mean()
    parent_gain = loss["ability shift"] - loss["+ parent answers"]
    drift_gain = loss["+ parent answers"] - loss["+ subject drift"]
    rows = [
        ("ability alone explains the child", parent_gain, parent_gain < MAX_LOSS_GAIN),
        ("no subject drift beyond parent answers", drift_gain, drift_gain < MAX_LOSS_GAIN),
    ]
    for method in (CORRECTED, VOPTIMAL):
        beat = paired[(paired.method == method) & (paired.k == CHECK_K)].iloc[0]
        rows.append((f"{method} beats paired baseline", beat.delta_mae, beat.ci_high < 0))
        for level, (low, high) in COVERAGE.items():
            value = at_k.loc[method, f"coverage_{level}"]
            rows.append((f"{method}: {level}% interval coverage", value, low <= value <= high))
    return pd.DataFrame(rows, columns=["check", "value", "passed"]).assign(k=CHECK_K)


def run(Y, rows, pairs, roots, subject_codes, out, n_dims, ks, seeds, folds, **fit_kwargs):
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "pairs.csv", index=False)
    pd.DataFrame(
        {
            "n_pairs": [len(pairs)],
            "n_dims": [n_dims],
            "n_folds": [len(folds)],
            "fold_seed": [FOLD_SEED],
            "seeds": [str(list(seeds))],
            "ks": [str(list(ks))],
            "fit_epochs": [fit_kwargs.get("epochs", 2000)],
        }
    ).to_csv(out / "protocol.csv", index=False)
    records, losses = run_folds(
        Y, rows, pairs, roots, subject_codes, n_dims, ks, seeds, folds, **fit_kwargs
    )
    records = conformal_intervals(records)
    summary = summarise(records)
    paired = paired_comparison(records[records.kind == FOCUS], baseline=PAIRED)
    checks = check_results(summary, paired, losses)
    records.to_csv(out / "predictions.csv", index=False)
    losses.to_csv(out / "heldout_losses.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)
    checks.to_csv(out / "checks.csv", index=False)
    return summary, paired, checks


def load_pairs(data, benchmark, config, contents):
    """Response matrix and de-duplicated pairs from the store build_tables.py writes."""
    wide = pd.read_parquet(data / "matrices" / f"{benchmark}__{config}.parquet")
    items = pd.read_parquet(data / "items.parquet").set_index("item_id").loc[wide.columns]
    models = pd.read_parquet(data / "models.parquet")
    Y, rows = wide.to_numpy(dtype=np.float32), wide.index.to_numpy()
    pairs, roots = find_pairs(models, rows)
    if pairs.empty:
        raise ValueError("No model in this matrix has its parent in the matrix too")
    pairs = drop_near_duplicates(pairs, Y, rows)
    types = pd.read_parquet(contents, columns=["fullname", "Type"])
    types = types.drop_duplicates("fullname").set_index("fullname").Type
    fine_tuned = (
        pairs.child.str.rsplit("@", n=1)
        .str[0]
        .map(types)
        .fillna("")
        .str.contains("fine-tuned", regex=False)
    )
    pairs["kind"] = np.where(fine_tuned, FOCUS, "other derived")
    return Y, rows, pairs, roots, pd.factorize(items.subtask)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="store from build_tables.py")
    parser.add_argument("--benchmark", default="leaderboard_mmlu_pro")
    parser.add_argument("--config", default="openllm_v2_public")
    parser.add_argument(
        "--contents", type=Path, default=ROOT / "data/raw/leaderboard_contents/contents.parquet"
    )
    parser.add_argument("--dims", type=int, default=N_DIMS)
    parser.add_argument(
        "--smoke", action="store_true", help="one fold, two seeds, k=20/100, 100 epochs"
    )
    args = parser.parse_args()
    Y, rows, pairs, roots, codes = load_pairs(args.data, args.benchmark, args.config, args.contents)
    print(f"{len(pairs)} pairs, {(pairs.kind == FOCUS).sum()} fine-tuned", flush=True)
    ks, seeds, folds, fit_kwargs = KS, SEEDS, make_folds(pairs), {}
    out = ROOT / "results/irt/pairs" / args.benchmark
    if args.smoke:
        ks, seeds, folds, fit_kwargs = [20, 100], range(2), folds[:1], {"epochs": 100}
        out = out / "smoke"
    summary, _, checks = run(
        Y, rows, pairs, roots, codes, out, args.dims, ks, seeds, folds, **fit_kwargs
    )
    shown = summary[(summary.kind == FOCUS) & (summary.k == CHECK_K)]
    print(shown[["method", "abs_error", "right_direction", "coverage_80", "coverage_95"]])
    print(checks)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
