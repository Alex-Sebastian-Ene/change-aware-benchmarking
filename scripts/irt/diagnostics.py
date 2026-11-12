"""Training dimensionality checks, held-out answer prediction, and paired score comparisons."""

import numpy as np
import pandas as pd

from cabench.design import as_context


def dimension_diagnostics(Y_train, n_components=20, n_null=100, seed=0):
    """PCA of centred binary responses, with a Bernoulli parallel-analysis null.

    Each null matrix has the same shape and per-item expected accuracy as training.
    Work with the model Gram matrix, avoiding a questions x questions covariance matrix.
    Pass training rows only; this is a diagnostic, not a rule that chooses D on test.
    """
    Y_train = np.asarray(Y_train, dtype=float)
    n_components = min(n_components, len(Y_train) - 1, Y_train.shape[1])

    def eigenvalues(Y):
        X = Y - Y.mean(0)
        return np.maximum(np.linalg.eigvalsh(X @ X.T / (len(Y) - 1))[::-1], 0)

    actual = eigenvalues(Y_train)
    rng = np.random.default_rng(seed)
    null = np.array(
        [
            eigenvalues(rng.binomial(1, Y_train.mean(0), size=Y_train.shape))[:n_components]
            for _ in range(n_null)
        ]
    )
    return pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "eigenvalue": actual[:n_components],
            "variance_explained": actual[:n_components] / actual.sum(),
            "null_95": np.quantile(null, 0.95, axis=0),
        }
    )


def response_diagnostics(Y, ctx, seed=0):
    """Reveal a fixed random half of items, score only the hidden half on held-out models.

    Same revealed items for every dimension. Likelihood, Brier score and calibration ECE
    assess responses, separately from full-score MAE. ECE uses ten equal-width bins.
    """
    order = np.random.default_rng(seed).permutation(Y.shape[1])
    seen, hidden = np.split(order, [Y.shape[1] // 2])
    ctx = as_context(ctx)
    theta = ctx["infer"](Y[:, seen], seen)
    predictions = {
        "IRT": ctx["predict"](theta, hidden),
    }
    if "item_mean" in ctx:
        predictions["item mean"] = np.broadcast_to(ctx["item_mean"][hidden], (len(Y), len(hidden)))
    actual = Y[:, hidden].ravel()
    rows = []
    for name, p in predictions.items():
        p = np.clip(p.ravel(), 1e-8, 1 - 1e-8)
        bins = np.minimum((10 * p).astype(int), 9)
        ece = sum(abs((p[bins == b] - actual[bins == b]).sum()) / len(p) for b in range(10))
        rows.append(
            {
                "predictor": name,
                "n_dims": ctx["n_dims"],
                "log_loss": -(actual * np.log(p) + (1 - actual) * np.log1p(-p)).mean(),
                "brier": np.mean((p - actual) ** 2),
                "ece": ece,
            }
        )
    return pd.DataFrame(rows)


def choose_dimension(validation, estimator="gp-IRT"):
    """Choose D using random selection only, mean log validation MAE across k.

    Every budget counts equally; selection is held fixed while testing dimensionality.
    A tie chooses the smaller dimension. No test records may be supplied here.
    """
    rows = validation[(validation.selector == "random") & (validation.estimator == estimator)]
    runs = rows.groupby(["n_dims", "k", "split", "seed"]).abs_error.mean()
    mae = runs.groupby(["n_dims", "k"]).mean()
    return int(np.log(mae.clip(lower=1e-12)).groupby("n_dims").mean().idxmin())


def paired_comparison(
    records, baseline="random + gp-IRT", n_boot=10000, seed=0, reference_dims=None
):
    """Paired delta MAE and a percentile bootstrap over split means (negative wins).

    Match split/seed/k/model first, average seeds within each model, then models within
    each split. Deterministic designs repeated over seeds do not inflate the sample size.
    The five family splits overlap, so this CI describes split variability, not five
    independent populations. reference_dims compares all D against a single baseline D.
    """
    keys = ["split", "seed", "k", "model_id"]
    base = records[records.method == baseline]
    if reference_dims is None:
        keys.append("n_dims")
    else:
        base = base[base.n_dims == reference_dims]
    if records.duplicated(["method", "n_dims", "split", "seed", "k", "model_id"]).any():
        raise ValueError("Duplicate per-model predictions")
    joined = records.merge(
        base[keys + ["abs_error", "true_score"]],
        on=keys,
        how="left",
        suffixes=("", "_baseline"),
        validate="many_to_one",
    )
    if joined.abs_error_baseline.isna().any() or not np.allclose(
        joined.true_score, joined.true_score_baseline
    ):
        raise ValueError("Methods must contain the same split/seed/k/model targets")
    counts = joined.groupby(["method", "n_dims"]).size()
    expected = (
        counts.index.get_level_values("n_dims").map(base.groupby("n_dims").size()).to_numpy()
        if reference_dims is None
        else len(base)
    )
    if np.any(counts.to_numpy() != expected):
        raise ValueError("Methods must contain the same split/seed/k/model targets")
    joined["delta"] = joined.abs_error - joined.abs_error_baseline
    model_means = joined.groupby(["method", "n_dims", "k", "split", "model_id"]).delta.mean()
    split_means = model_means.groupby(["method", "n_dims", "k", "split"]).mean()
    rng, rows = np.random.default_rng(seed), []
    for (method, dims, k), group in split_means.groupby(["method", "n_dims", "k"]):
        values = group.values
        draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "method": method,
                "n_dims": dims,
                "k": k,
                "baseline": baseline,
                "delta_mae": values.mean(),
                "ci_low": lo,
                "ci_high": hi,
                "splits_won": int((values < 0).sum()),
                "n_splits": len(values),
            }
        )
    return pd.DataFrame(rows)
