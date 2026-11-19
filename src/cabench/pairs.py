"""Score change between a parent model and its fine-tuned child, from a few child answers.

Owners: A. Vijaykrishnan, A. Chitnis.

The parent has answered every item, the child only k of them. Item parameters stay frozen
and the child is modelled starting from the parent:

    logit P(child correct on i) = theta_parent . alpha_i - beta_i + x_i . b
    x_i = [alpha_i, parent's residual on i, one indicator per subject]

b is the ability shift, how much of the parent's item-level behaviour the child keeps, and
drift per subject. corrected_change uses random items to remove the model's bias, so its
interval holds even when the model is wrong.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from cabench.design import parameter_context

LEVELS = (80, 95)


def find_pairs(models, rows):
    """Match matrix rows ("model_id@revision") to their parents through `parent_model`.

    Returns a (child, parent, root) table and the root of every row. The root is the top
    of the lineage, so a whole family can be held out together.
    """
    rows = pd.Index(rows)
    names = rows.str.rsplit("@", n=1).str[0]
    first_row = pd.Series(rows, index=names).groupby(level=0).first().to_dict()
    known = models.dropna(subset=["parent_model"]).drop_duplicates("model_id")
    parent_of = dict(zip(known.model_id, known.parent_model, strict=True))

    def root(name):
        seen = {name}
        while parent_of.get(name, name) not in seen:
            name = parent_of[name]
            seen.add(name)
        return name

    pairs = [
        {"child": row, "parent": first_row[parent_of[name]], "root": root(name)}
        for row, name in zip(rows, names, strict=True)
        if parent_of.get(name, name) != name and parent_of[name] in first_row
    ]
    roots = np.array([root(name) for name in names])
    return pd.DataFrame(pairs, columns=["child", "parent", "root"]), roots


def agreement(a, b):
    """Share of jointly answered items where both models are right or both wrong."""
    both = np.isfinite(a) & np.isfinite(b)
    return (a[both] == b[both]).mean()


def drop_near_duplicates(pairs, Y, rows, max_agreement=0.99):
    """Keep one child from each group of siblings that answer almost identically."""
    index = {row: i for i, row in enumerate(rows)}
    keep = []
    for _, siblings in pairs.groupby("parent", sort=False):
        kept = []
        for position, child in zip(siblings.index, siblings.child, strict=True):
            y = Y[index[child]]
            if all(agreement(y, Y[index[other]]) < max_agreement for other in kept):
                keep.append(position)
                kept.append(child)
    return pairs.loc[sorted(keep)].reset_index(drop=True)


def pair_problem(y_parent, y_child, alpha, beta, theta_train, subject_codes, n_subjects):
    """Everything the estimators need for one pair, on items both models answered."""
    both = np.flatnonzero(np.isfinite(y_parent) & np.isfinite(y_child))
    alpha, beta = alpha[both].astype(float), beta[both].astype(float)
    ctx = parameter_context(theta_train.astype(float), alpha, beta)
    y_parent, y_child = y_parent[both].astype(float), y_child[both].astype(float)
    theta = ctx["infer"](y_parent[None], np.arange(len(both)))
    offset = (theta @ alpha.T - beta)[0]
    subjects = np.eye(n_subjects)[subject_codes[both]]
    return {
        "y_parent": y_parent,
        "y_child": y_child,
        "offset": offset,
        "X": np.column_stack([alpha, y_parent - expit(offset), subjects]),
        "n_dims": alpha.shape[1],
        "ctx": ctx,
        "true_change": (y_child - y_parent).mean(),
    }


def fit_coefficients(problem, items, prior, n_columns=None):
    """MAP estimate of b by damped Newton, using the first n_columns of X."""
    X = problem["X"][items, :n_columns]
    y, offset = problem["y_child"][items], problem["offset"][items]
    mean, precision = prior[0][: X.shape[1]], prior[1][: X.shape[1]]
    b = mean.copy()
    for _ in range(50):
        p = expit(offset + X @ b)
        gradient = X.T @ (p - y) + precision * (b - mean)
        hessian = (X.T * (p * (1 - p))) @ X + np.diag(precision)
        step = np.linalg.solve(hessian, gradient)
        step /= max(1, np.abs(step).max())  # keeps separable small samples from diverging
        b -= step
        if np.abs(step).max() < 1e-6:
            break
    return b


def predict_child(problem, items, prior, n_columns=None):
    """P(child correct) on every item, fitted from the child's answers on `items`."""
    b = fit_coefficients(problem, items, prior, n_columns)
    return expit(problem["offset"] + problem["X"][:, : len(b)] @ b)


def learn_prior(problems):
    """Gaussian prior for b from fully observed training pairs."""
    n_columns = problems[0]["X"].shape[1]
    flat = np.zeros(n_columns), np.full(n_columns, 0.01)
    b = np.array([fit_coefficients(p, np.arange(len(p["X"])), flat) for p in problems])
    return b.mean(0), 1 / (b.var(0) + 1e-3)


def paired_change(problem, items):
    """Mean answer difference on the items: the no-IRT baseline. Returns (change, residuals)."""
    residuals = problem["y_child"][items] - problem["y_parent"][items]
    return residuals.mean(), residuals


def model_change(problem, items, prior):
    """Change predicted by the parent-based model, using the real answers where seen."""
    p = predict_child(problem, items, prior)
    p[items] = problem["y_child"][items]
    return (p - problem["y_parent"]).mean()


def corrected_change(problem, items, prior):
    """model_change with its bias removed. `items` must be a random sample.

    Each half of the items fits the model that corrects the other half, as in
    prediction-powered inference. Returns (change, residuals).
    """
    halves = items[::2], items[1::2]
    estimate, residuals = 0, []
    for fit, held in (halves, halves[::-1]):
        p = predict_child(problem, fit, prior)
        r = problem["y_child"][held] - p[held]
        estimate += len(held) / len(items) * (p.mean() + r.mean())
        residuals.append(r)
    return estimate - problem["y_parent"].mean(), np.concatenate(residuals)


def change_interval(change, residuals, n_items):
    """Normal 80% / 95% intervals for a random sample of n_items without replacement.

    One pseudo-flip (as in Agresti-Min) keeps the width above zero when none were seen.
    """
    n = len(residuals)
    spread = ((residuals - residuals.mean()) ** 2).sum() + 1
    se = np.sqrt(spread * (1 - n / n_items)) / (n + 2)
    row = {}
    for level in LEVELS:
        half_width = norm.ppf(0.5 + level / 200) * se
        row[f"low_{level}"], row[f"high_{level}"] = change - half_width, change + half_width
    return row


def heldout_losses(problem, prior, seed=0):
    """Log loss on a hidden half of the items as the child model gets more terms.

    If frozen item parameters were enough, the extra terms would barely help.
    """
    order = np.random.default_rng(seed).permutation(len(problem["X"]))
    fit, held = np.array_split(order, 2)
    d, y = problem["n_dims"], problem["y_child"][held]
    levels = {"ability shift": d, "+ parent answers": d + 1, "+ subject drift": None}
    losses = {}
    for name, n_columns in levels.items():
        p = np.clip(predict_child(problem, fit, prior, n_columns)[held], 1e-8, 1 - 1e-8)
        losses[name] = -(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()
    return losses
