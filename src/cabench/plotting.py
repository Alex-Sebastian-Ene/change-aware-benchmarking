"""Plots of score-estimation records and frozen-model item designs. No model fitting."""

import matplotlib.pyplot as plt
import numpy as np

from cabench.design import adaptive_design, as_context, static_design


def plot_score_errors(records):
    """Hold estimator fixed in each panel; average seeds, then models, then splits."""
    if records.n_dims.nunique() != 1:
        raise ValueError("Filter records to one dimension before comparing selectors")
    estimators = [e for e in ["accuracy", "p-IRT", "gp-IRT"] if e in set(records.estimator)]
    means = records.groupby(["estimator", "selector", "k", "split", "model_id"]).abs_error.mean()
    split_means = means.groupby(["estimator", "selector", "k", "split"]).mean()
    summary = split_means.groupby(["estimator", "selector", "k"]).agg(["mean", "std"])
    fig, axes = plt.subplots(1, len(estimators), figsize=(5 * len(estimators), 4), squeeze=False)
    for ax, estimator in zip(axes[0], estimators, strict=True):
        for selector in records.selector.unique():
            curve = summary.loc[estimator, selector]
            ax.errorbar(
                curve.index,
                100 * curve["mean"],
                yerr=100 * curve["std"].fillna(0),
                marker="o",
                capsize=3,
                label=selector,
            )
        ax.set(title=estimator, xscale="log", xlabel="answers observed", ylabel="MAE (pp)")
        ax.legend(fontsize=8)
    fig.suptitle(f"D={records.n_dims.iloc[0]}; ±1 SD across splits")
    fig.tight_layout()
    return fig


def plot_estimator_grid(records, k):
    """Selector x estimator ablation at one budget, using the same split-balanced MAE."""
    if records.n_dims.nunique() != 1:
        raise ValueError("Filter records to one dimension before comparing selectors")
    means = (
        records[records.k == k]
        .groupby(["selector", "estimator", "split", "model_id"])
        .abs_error.mean()
    )
    means = means.groupby(["selector", "estimator", "split"]).mean()
    table = 100 * means.groupby(["selector", "estimator"]).mean().unstack("estimator")
    table = table.reindex(columns=[e for e in ["accuracy", "p-IRT", "gp-IRT"] if e in table])
    fig, ax = plt.subplots(figsize=(6, max(3, len(table) * 0.6)))
    image = ax.imshow(table.values, aspect="auto", cmap="viridis_r")
    for row, col in np.ndindex(table.shape):
        ax.text(
            col,
            row,
            f"{table.iloc[row, col]:.2f}",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    ax.set(
        xticks=range(len(table.columns)),
        xticklabels=table.columns,
        yticks=range(len(table)),
        yticklabels=table.index,
        title=f"MAE (pp), k={k}",
    )
    fig.colorbar(image, ax=ax, label="MAE (pp)")
    fig.tight_layout()
    return fig


def plot_predictions(records, k, selector="random"):
    """Parity and signed-error distributions: a shared selector isolates estimation."""
    if records.n_dims.nunique() != 1:
        raise ValueError("Filter records to one dimension before comparing estimators")
    rows = records[(records.k == k) & (records.selector == selector)]
    means = rows.groupby(["estimator", "split", "model_id"])[["true_score", "pred_score"]].mean()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for estimator in rows.estimator.unique():
        data = means.loc[estimator]
        left.scatter(100 * data.true_score, 100 * data.pred_score, s=14, alpha=0.5, label=estimator)
        right.hist(
            100 * (data.pred_score - data.true_score),
            bins=20,
            histtype="step",
            linewidth=1.5,
            label=estimator,
        )
    lo = 100 * means.min().min()
    hi = 100 * means.max().max()
    left.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    left.set(xlabel="true full score (%)", ylabel="predicted score (%)", title=f"{selector}, k={k}")
    right.axvline(0, color="black", linewidth=1)
    right.set(xlabel="signed error (pp)", ylabel="model/split pairs", title="Bias and spread")
    left.legend()
    right.legend()
    fig.tight_layout()
    return fig


def plot_item_choices(ctx, responses, k=10, seed=0):
    """Show the same candidate pool and budget for static / adaptive designs.

    Two response rows show adaptation to different models. MIRT shows the first two
    loading coordinates; 1D shows reference success probability versus discrimination.
    This is model geometry, not an empirical claim about score error.
    """
    ctx = as_context(ctx)
    criterion = "fisher" if ctx["n_dims"] == 1 else "voptimal"
    choices = {"random": np.random.default_rng(seed).choice(ctx["n_items"], k, replace=False)}
    if ctx.get("emb") is not None:
        from cabench.irt import irt_anchors

        choices["IRT anchors"] = irt_anchors(ctx["emb"], k, seed)[0]
    choices[f"{criterion}_static"] = static_design(ctx, k, criterion)
    _, items, _ = list(adaptive_design(responses[:2], ctx, [k], criterion))[-1]
    choices[f"{criterion}_adaptive"] = items
    a = ctx["loadings"]
    if ctx["n_dims"] == 1:
        x, y = ctx["predict"](ctx["theta"].mean(0, keepdims=True))[0], abs(a[:, 0])
        xlabel, ylabel = "P(correct) at training mean", "|discrimination|"
    else:
        x, y = a[:, 0], a[:, 1]
        xlabel, ylabel = "loading 1", "loading 2"
    fig, axes = plt.subplots(1, len(choices), figsize=(4 * len(choices), 3.8), squeeze=False)
    for ax, (name, selected) in zip(axes[0], choices.items(), strict=True):
        ax.scatter(x, y, s=8, color="lightgrey", label="candidate items")
        if selected.ndim == 1:
            ax.scatter(x[selected], y[selected], s=35, color="tab:green", label="shared design")
        else:
            for m, indices in enumerate(selected):
                ax.scatter(x[indices], y[indices], s=35, alpha=0.7, label=f"model {m}")
        ax.set(title=f"{name}, k={k}", xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
