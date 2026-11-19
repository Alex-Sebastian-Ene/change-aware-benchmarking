"""Presentation figures for the parent/child pair results and the IRT fit comparison.

Reads results/irt/pairs/<name>/predictions.csv (efficbench_pairs.py) and
results/irt/bayesian/fit_speed.csv, and saves PNGs to results/figures/ following design.md.

Run: uv run --group irt python scripts/irt/plot_pairs.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from efficbench import ROOT
from run_pairs import CORRECTED, FOCUS, GAP, PAIRED, VOPTIMAL

FIGS = ROOT / "results/figures"
PAIRS = ROOT / "results/irt/pairs/efficbench_mmlu_extended"
FIT_SPEED = ROOT / "results/irt/bayesian/fit_speed.csv"
BIG_CHANGE = 0.02  # direction is only meaningful for changes of 2 points or more

STYLE = {  # method: (label, colour, marker), from design.md
    PAIRED: ("Random questions", "#9CA3AF", "s"),
    GAP: ("Random + IRT", "#93C5FD", "^"),
    CORRECTED: ("Random + parent model", "#2563EB", "D"),
    VOPTIMAL: ("Adaptive IRT (ours)", "#1E3A8A", "o"),
}
FIT_STYLE = {  # fit: (label, colour, marker); gradient descent is the one we use
    "gradient descent": ("Gradient descent (current)", "#1E3A8A", "o"),
    "bayesian": ("Bayesian", "#93C5FD", "^"),
}

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "legend.frameon": False,
    }
)


def by_pair(records, column):
    """Mean over seeds within a pair, then over pairs: every pair counts once."""
    per_pair = records.groupby(["method", "k", "model_id"])[column].mean()
    return per_pair.groupby(["method", "k"]).mean().unstack("method")


def log_questions(ax, ks):
    ax.set(xscale="log", xticks=ks, xticklabels=ks, xlabel="questions asked")
    ax.minorticks_off()


def plot_methods(ax, table, methods, labelled=True):
    for method in methods:
        label, colour, marker = STYLE[method]
        label = label if labelled else None
        ax.plot(table.index, 100 * table[method], marker=marker, color=colour, lw=2.5, label=label)


def error_figure(records):
    table = by_pair(records, "abs_error")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_methods(ax, table, STYLE)
    log_questions(ax, table.index)
    ax.set(
        ylabel="change error (percentage points)",
        title="Adaptive questions measure a fine-tune's change with fewer answers",
    )
    ax.legend()
    return fig


def interval_figure(records):
    methods = [CORRECTED, VOPTIMAL]
    rows = records[records.method.isin(methods)]
    width = by_pair(rows.assign(width=rows.high_95 - rows.low_95), "width")
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    for level in (80, 95):
        inside = rows.true_score.between(rows[f"low_{level}"], rows[f"high_{level}"])
        left.axhline(level, color="#374151", lw=1, ls="--")
        plot_methods(left, by_pair(rows.assign(inside=inside), "inside"), methods, level == 95)
    left.set(
        ylabel="intervals containing the true change (%)",
        ylim=(60, 100),
        title="80% and 95% intervals",
    )
    plot_methods(right, width, methods)
    right.set(ylabel="95% interval width (percentage points)", title="Width")
    for ax in (left, right):
        log_questions(ax, width.index)
    left.legend(loc="lower right")
    fig.suptitle("Intervals hit their stated coverage; adaptive questions make them narrower")
    fig.tight_layout()
    return fig


def direction_figure(records):
    big = records[records.true_score.abs() >= BIG_CHANGE]
    big = big.assign(right=np.sign(big.pred_score) == np.sign(big.true_score))
    table = by_pair(big, "right")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_methods(ax, table, STYLE)
    log_questions(ax, table.index)
    n_pairs = big.model_id.nunique()
    ax.set(
        ylabel="right direction (% of runs)",
        ylim=(50, 101),
        title=f"Up or down? Changes of 2+ points ({n_pairs} fine-tunes)",
    )
    ax.legend(loc="lower right")
    return fig


def parity_figure(records, k=100):
    # one seed: averaging seeds would hide the noise of a single random draw
    at_k = records[(records.k == k) & (records.seed == 0)]
    points = 100 * at_k.set_index(["method", "model_id"])[["true_score", "pred_score"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    lo, hi = points.min().min() - 1, points.max().max() + 1
    for ax, method in zip(axes, [PAIRED, VOPTIMAL], strict=True):
        label, colour, marker = STYLE[method]
        data = points.loc[method]
        ax.plot([lo, hi], [lo, hi], color="#374151", lw=1, ls="--")
        ax.scatter(data.true_score, data.pred_score, color=colour, marker=marker, s=30, alpha=0.8)
        ax.set(title=label, xlabel="true change (points)", xlim=(lo, hi), ylim=(lo, hi))
    axes[0].set_ylabel(f"estimated change from {k} questions")
    fig.suptitle(f"Each dot is one fine-tune: estimates from {k} questions vs the full benchmark")
    fig.tight_layout()
    return fig


def fit_speed_figure(speed):
    table = speed.groupby(["n_dims", "fit", "epochs"])[["seconds", "log_loss"]].mean()
    dims = sorted(speed.n_dims.unique())
    fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 4.8), squeeze=False)
    for ax, d in zip(axes[0], dims, strict=True):
        for fit, (label, colour, marker) in FIT_STYLE.items():
            curve = table.loc[(d, fit)]
            ax.plot(curve.seconds, curve.log_loss, marker=marker, color=colour, lw=2.5, label=label)
        ax.set(
            xscale="log",
            xticks=[0.5, 1, 2, 5, 10],
            xticklabels=["0.5", "1", "2", "5", "10"],
            xlabel="training time (seconds)",
            title=f"{d} ability dimension{'s' if d > 1 else ''}",
        )
        ax.minorticks_off()
    axes[0, 0].set_ylabel("held-out answer loss (lower is better)")
    axes[0, 0].legend()
    fig.suptitle("Bayesian training fits the answers no better and no faster")
    fig.tight_layout()
    return fig


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(PAIRS / "predictions.csv")
    records = records[(records.kind == FOCUS) & records.method.isin(STYLE)]
    figures = {
        "pairs_error_vs_questions": error_figure(records),
        "pairs_interval_coverage": interval_figure(records),
        "pairs_direction": direction_figure(records),
        "pairs_parity_k100": parity_figure(records),
    }
    if FIT_SPEED.exists():
        figures["bayesian_fit_speed"] = fit_speed_figure(pd.read_csv(FIT_SPEED))
    for name, fig in figures.items():
        fig.savefig(FIGS / f"{name}.png", dpi=200)
        print(f"Saved {FIGS / name}.png")


if __name__ == "__main__":
    main()
