"""Graphs of the baseline results. Run from the repository root, after the other two scripts:

    uv run --group irt python scripts/baselines/plot_results.py

Writes results/baselines/{correlation_by_method,answer_choices}.png.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/baselines"

INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
METHODS = {
    "uniform": ("Random", "#2a78d6", "o"),
    "subtask": ("Random within subject", "#eb6834", "s"),
    "difficulty": ("Random within difficulty band", "#1baf7a", "^"),
    "systematic": ("Every n-th by difficulty", "#eda100", "D"),
    "anchors": ("Cluster anchors", "#e87ba4", "P"),
}
OPTIONS = {
    "correct": ("Correct option", "#2a78d6", "o"),
    "top_wrong": ("Most-picked wrong option", "#c04c1d", "s"),
    "second_wrong": ("Second wrong option", "#eb6834", "^"),
    "third_wrong": ("Third wrong option", "#f4a582", "D"),
}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=10, length=0)


def panel_title(ax, title, subtitle):
    ax.text(0, 1.1, title, transform=ax.transAxes, fontsize=12, color=INK, weight="bold")
    ax.text(0, 1.04, subtitle, transform=ax.transAxes, fontsize=10, color=SECONDARY)


def draw_line(ax, x, y, spec):
    label, colour, marker = spec
    ax.plot(
        x,
        y,
        color=colour,
        linewidth=2,
        marker=marker,
        markersize=7,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        solid_capstyle="round",
        label=label,
        clip_on=False,
    )


def correlation_by_method(title):
    table = pd.read_csv(OUT / "methods.csv")
    ks = sorted(table["k"].unique())
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6.2), dpi=200, facecolor=SURFACE)

    for method, spec in METHODS.items():
        rows = table[table["method"] == method].sort_values("k")
        if method == "uniform":
            left.fill_between(
                rows["k"],
                rows["spearman_p10"],
                rows["spearman_p90"],
                color=spec[1],
                alpha=0.14,
                linewidth=0,
                label="Random: 10th to 90th percentile of subsets",
            )
        draw_line(left, rows["k"], rows["spearman"], spec)
        draw_line(right, rows["k"], rows["rmse"] * 100, spec)

    for ax in (left, right):
        style(ax)
        ax.set_xscale("log")
        ax.set_xticks(ks, [f"{k:,}" for k in ks])
        ax.minorticks_off()
        ax.set_xlabel("Questions in the subset (k)", color=SECONDARY, fontsize=10)

    left.set_ylim(top=1.0)
    panel_title(
        left,
        "Do subset scores rank models correctly?",
        "Spearman correlation with the full 14,042-question score, held-out models",
    )
    right.set_yscale("log")
    ticks = [0.5, 1, 2, 5, 10, 20]
    right.set_yticks(ticks, [f"{t:g}" for t in ticks])
    right.minorticks_off()
    panel_title(
        right,
        "How far off is each model's score?",
        "Root-mean-square error, in percentage points of accuracy (log scale)",
    )

    handles, labels = left.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        labelcolor=SECONDARY,
        bbox_to_anchor=(0.5, 0.045),
    )
    left.text(
        60,
        0.74,
        "The four random methods\nare almost on top of each other",
        color=SECONDARY,
        fontsize=10,
    )
    fig.text(
        0.058,
        0.02,
        "125 to 135 held-out models in each of 5 family splits. 200 random subsets per split; "
        "one anchor set per split, so anchors have no band.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.suptitle(title, x=0.058, y=0.975, ha="left", fontsize=15, color=INK, weight="bold")
    fig.subplots_adjust(left=0.058, right=0.975, top=0.8, bottom=0.25, wspace=0.16)
    fig.savefig(OUT / "correlation_by_method.png", facecolor=SURFACE)
    plt.close(fig)


def answer_choices():
    confusion = pd.read_csv(OUT / "choices_confusion.csv", index_col="correct_option")
    table = pd.read_csv(OUT / "choices_difficulty.csv")
    info = json.loads((OUT / "choices.json").read_text())
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13, 6.2), dpi=200, facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.5]}
    )

    values = confusion.to_numpy() * 100
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    left.imshow(values, cmap=cmap, vmin=0, vmax=60, aspect="equal")
    for (row, column), value in np.ndenumerate(values):
        colour = "#ffffff" if value > 35 else INK
        left.text(column, row, f"{value:.0f}%", ha="center", va="center", color=colour, fontsize=12)
    left.set_xticks(range(4), confusion.columns)
    left.set_yticks(range(4), confusion.index)
    left.tick_params(colors=SECONDARY, labelsize=11, length=0)
    for side in left.spines.values():
        side.set_visible(False)
    left.set_xlabel("Option the model picked", color=SECONDARY, fontsize=10)
    left.set_ylabel("Correct option", color=SECONDARY, fontsize=10)
    panel_title(left, "Answer position barely matters", "Each row sums to 100%")

    x = table["mid"] * 100
    for column, spec in OPTIONS.items():
        draw_line(right, x, table[column] * 100, spec)
    style(right)
    right.set_xlim(0, 100)
    right.set_ylim(0, 100)
    right.set_xticks(range(0, 101, 20), [f"{t}%" for t in range(0, 101, 20)])
    right.set_yticks(range(0, 101, 20), [f"{t}%" for t in range(0, 101, 20)])
    right.set_xlabel(
        "How easy the question is: share of all 428 models that get it right",
        color=SECONDARY,
        fontsize=10,
    )
    hardest = table.iloc[0]
    right.annotate(
        f"On the hardest questions, {hardest['top_wrong']:.0%} of\n"
        "answers land on one wrong option",
        xy=(hardest["mid"] * 100, hardest["top_wrong"] * 100),
        xytext=(13, 80),
        color=INK,
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
    )
    panel_title(
        right,
        "Hard questions pull models to the same wrong option",
        "Wrong options ranked on half the models, measured on the other half",
    )
    right.legend(loc="center right", frameon=False, fontsize=10, labelcolor=SECONDARY)

    fig.suptitle(
        "Which answer models pick depends on the question, not on the letter",
        x=0.045,
        y=0.975,
        ha="left",
        fontsize=15,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.045,
        0.02,
        f"{info['n_answers']:,} answers from {info['n_models']} models spread across the score "
        f"range. Agreement between models on each question beyond chance: Fleiss' kappa "
        f"{info['fleiss_kappa']:.2f}.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.045, right=0.975, top=0.8, bottom=0.14, wspace=0.12)
    fig.savefig(OUT / "answer_choices.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    correlation_by_method(
        "Cluster anchors rank models far better on few questions; past about 200, random catches up"
    )
    answer_choices()
