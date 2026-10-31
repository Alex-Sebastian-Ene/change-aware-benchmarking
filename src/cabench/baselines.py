"""Noise reference (re-runs of the same model) and random-subset baselines at equal compute.

Core objective 2. Owner: S. Ene.

Matrices are models x items DataFrames, as scripts/matrix/build_tables.py writes them:
index "model_id@model_revision", columns item_id, values 1.0 / 0.0 / NaN (NaN = no
response, never 0). Compute is counted in items: a subset of k items costs k.
"""

import warnings
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

Values = float | np.ndarray | pd.Series


# ---------- core: uniform and subtask-stratified random subsets ----------


def analytic_se(p: Values, k: int, n_items: int) -> Values:
    """Standard error of the accuracy on k items drawn without replacement.

    Args:
        p: the model's full-benchmark accuracy, one float or one value per model.
        k: subset size, 1 <= k <= n_items.
        n_items: number of items in the full benchmark (N).

    Returns:
        sqrt(p(1-p)/k * (N-k)/(N-1)), the same shape as p; 0 when k == N.

    Raises:
        ValueError: if k is not in 1..n_items.
    """
    if not 1 <= k <= n_items:
        raise ValueError(f"subset size k={k} must be in 1..{n_items}")
    p = np.asarray(p)
    return np.sqrt(p * (1 - p) / k * (n_items - k) / (n_items - 1))


def full_scores(M: pd.DataFrame) -> pd.Series:
    """Full-benchmark accuracy of each model.

    Args:
        M: response matrix (models x items, 1/0/NaN).

    Returns:
        One value per model (index = M.index): the mean over answered items, NaN
        skipped. A model with no answers gets NaN.
    """
    return M.sum(axis=1) / M.count(axis=1)


def subset_scores(M: pd.DataFrame, items: pd.Index) -> pd.Series:
    """Each model's accuracy on `items` only, with the same NaN rule as full_scores.

    Args:
        M: response matrix.
        items: item_ids to score on; every one must be a column of M.

    Returns:
        One value per model, index = M.index.
    """
    return full_scores(M[items])


def proportional_allocation(strata: pd.Series, k: int, rng: np.random.Generator) -> pd.Series:
    """Split a budget of k items across strata in proportion to their size.

    Each stratum gets the floor of its exact share k * size / N. The leftover items go to
    strata at random, so that each stratum gets one extra with probability equal to its
    fractional part (systematic rounding: one uniform draw u, then a stratum gets an extra
    item when the running total of fractional parts + u crosses an integer). On average
    every stratum then gets exactly its share, so every item has a k/N chance of being
    picked.

    Args:
        strata: item_id -> stratum label (e.g. the items table's `subtask`).
        k: total items to draw, 0 <= k <= len(strata).
        rng: random generator, created once by the caller.

    Returns:
        stratum label -> number of items, sorted by label; sums to k, never more than a
        stratum's size.

    Raises:
        ValueError: if k is not in 0..len(strata).
    """
    n = strata.count()  # items with a NaN stratum are never drawn
    if not 0 <= k <= n:
        raise ValueError(f"subset size k={k} must be in 0..{n}")
    sizes = strata.value_counts().sort_index()
    # whole numbers throughout: a float running total can miss its last integer
    floors, remainders = np.divmod(k * sizes, n)
    crossed = (remainders.cumsum() + rng.integers(n)) // n
    return floors + crossed.diff().fillna(crossed).astype(int)


def sample_items(
    items: pd.Index,
    k: int,
    rng: np.random.Generator,
    strata: pd.Series | None = None,
) -> pd.Index:
    """Draw k distinct items, uniformly or stratified.

    Args:
        items: item_ids to draw from (usually M.columns).
        k: number of items, 0 <= k <= len(items).
        rng: random generator, created once by the caller; never re-seed in here.
        strata: optional item_id -> stratum label. If given, draw
            proportional_allocation(strata[items], k, rng) items from each stratum.

    Returns:
        The k chosen item_ids, no repeats.
    """
    if strata is None:
        return items[rng.choice(len(items), size=k, replace=False)]
    labels = strata[items]
    counts = proportional_allocation(labels, k, rng)
    groups = labels.groupby(labels.to_numpy()).indices
    picked = [rng.choice(groups[label], size=n, replace=False) for label, n in counts.items()]
    return items[np.concatenate(picked)]


def _error_table(M: pd.DataFrame, rows: list[pd.DataFrame]) -> pd.DataFrame:
    """Join per-subset estimates with each model's full score."""
    errors = pd.concat(rows, ignore_index=True)
    errors["full"] = errors["model"].map(full_scores(M))
    errors["err"] = errors["est"] - errors["full"]
    return errors


def _simulate(M: pd.DataFrame, ks: Sequence[int], n_rep: int, draw: Callable) -> pd.DataFrame:
    """Score every model on draw(k) for each k and repetition; models share each subset."""
    rows = [
        pd.DataFrame(
            {"model": M.index, "k": k, "rep": rep, "est": subset_scores(M, draw(k)).to_numpy()}
        )
        for k in ks
        for rep in range(n_rep)
    ]
    return _error_table(M, rows)


def random_subset_errors(
    M: pd.DataFrame,
    ks: Sequence[int],
    n_rep: int = 1000,
    seed: int = 0,
    strata: pd.Series | None = None,
) -> pd.DataFrame:
    """Simulate random-subset estimates and their errors against the full score.

    For each k and repetition, draw one subset and score every model on it, so all
    models in a repetition share the same items.

    Args:
        M: response matrix.
        ks: subset sizes to try, e.g. (25, 50, 100, 200, 400, 800, 1600).
        n_rep: repetitions per k.
        seed: seed for the single np.random.default_rng made here; record it with results.
        strata: optional item_id -> stratum label, for stratified sampling.

    Returns:
        Long table, one row per (model, k, rep), with columns
        model, k, rep, est (subset score), full (full score), err (est - full).
        est is NaN for a model with no answers in the subset.
    """
    rng = np.random.default_rng(seed)
    return _simulate(M, ks, n_rep, lambda k: sample_items(M.columns, k, rng, strata))


def empirical_vs_analytic(errors: pd.DataFrame, n_items: int) -> pd.DataFrame:
    """Compare the simulated spread of `est` with analytic_se, per model and k.

    Args:
        errors: output of random_subset_errors (uniform sampling, complete matrix).
        n_items: number of items in the matrix.

    Returns:
        One row per (model, k), with columns model, k, empirical (std of est over reps),
        analytic (analytic_se(full, k, n_items)) and ratio (empirical / analytic, about
        1 if everything is right).
    """
    out = errors.groupby(["model", "k"], as_index=False).agg(
        empirical=("est", "std"), full=("full", "first")
    )
    out["analytic"] = [
        float(analytic_se(p, k, n_items)) for p, k in zip(out["full"], out["k"], strict=True)
    ]
    out["ratio"] = out["empirical"] / out["analytic"]
    return out.drop(columns="full")


def summarise_errors(
    errors: pd.DataFrame,
    levels: Sequence[float] = (0.8, 0.95),
    by: str | Sequence[str] = "k",
) -> pd.DataFrame:
    """Error summary per subset size (or per `by` group, e.g. ["method", "k"]).

    Args:
        errors: output of random_subset_errors or subset_errors.
        levels: coverage levels for the |err| quantiles.
        by: column(s) to group on.

    Returns:
        Indexed by `by`, with columns bias (mean err), rmse, mae (mean |err|) and one
        q<level> column per level (e.g. q80, q95): that quantile of |err|. NaN errors
        are ignored.
    """
    by = by if isinstance(by, str) else list(by)
    errors = errors.assign(squared=errors["err"] ** 2, absolute=errors["err"].abs())
    grouped = errors.groupby(by)
    out = pd.DataFrame(
        {
            "bias": grouped["err"].mean(),
            "rmse": np.sqrt(grouped["squared"].mean()),
            "mae": grouped["absolute"].mean(),
        }
    )
    for level in levels:
        out[f"q{level * 100:g}"] = grouped["absolute"].quantile(level)
    return out


def budget_for_error(summary: pd.DataFrame, target: float, metric: str = "rmse") -> int | None:
    """Smallest subset size whose error is at or below `target`.

    Args:
        summary: output of summarise_errors, indexed by k.
        target: error to reach, e.g. the noise reference's sd.
        metric: summary column to compare, e.g. "rmse" or "q95".

    Returns:
        The smallest such k, or None if no k in the summary reaches it.
    """
    reached = summary.index[summary[metric] <= target]
    return int(reached.min()) if len(reached) else None


def noise_reference(long: pd.DataFrame, benchmark: str, config_id: str) -> pd.DataFrame:
    """Run-to-run spread of each model's full-benchmark score under one config.

    Args:
        long: the long table (run_id, model_id, model_revision, benchmark, config_id,
            item_id, correct, ...).
        benchmark: benchmark to select.
        config_id: config to select; only runs of the same pinned config are repeats.

    Returns:
        One row per model, indexed "model_id@model_revision", with columns n_runs,
        mean, sd and range of the per-run accuracy. sd is NaN for a model with one run.
    """
    sel = long[(long["benchmark"] == benchmark) & (long["config_id"] == config_id)]
    model = (sel["model_id"] + "@" + sel["model_revision"]).rename("model")
    per_run = sel.groupby([model, "run_id"])["correct"].mean()
    return per_run.groupby("model").agg(
        n_runs="count", mean="mean", sd="std", range=lambda s: s.max() - s.min()
    )


# ---------- extensions: data-driven selection and single-subset intervals ----------
# Methods that learn from responses (difficulty, systematic, anchors, calibrated) must be
# fitted on calibration models (M_calib) and scored on held-out models (M_eval).


def item_difficulty(M_calib: pd.DataFrame) -> pd.Series:
    """item_id -> fraction of calibration models that answered it correctly (NaN skipped)."""
    return M_calib.mean(axis=0)


def difficulty_strata(difficulty: pd.Series, n_bins: int = 5) -> pd.Series:
    """item_id -> difficulty bin 0..n_bins-1 (equal-count quantile bins, pd.qcut).

    Pass the result as `strata` to sample_items. Items with NaN difficulty get NaN.
    """
    # rank first: difficulties tie a lot, and tied values would otherwise share a bin
    return pd.qcut(difficulty.rank(method="first"), n_bins, labels=False)


def systematic_sample(order: pd.Index, k: int, rng: np.random.Generator) -> pd.Index:
    """Every (len(order) / k)-th item of `order`, starting at a random offset.

    Args:
        order: item_ids in the order to walk, e.g. sorted by difficulty.
        k: number of items, 1 <= k <= len(order).
        rng: random generator for the offset.

    Returns:
        The k chosen item_ids.
    """
    n = len(order)
    if not 1 <= k <= n:
        raise ValueError(f"subset size k={k} must be in 1..{n}")
    return order[(rng.integers(n) + np.arange(k) * n) // k]


def cluster_anchors(M_calib: pd.DataFrame, k: int, seed: int = 0) -> tuple[pd.Index, pd.Series]:
    """Representative items: cluster items by their response pattern, one per cluster.

    Each item is a point (its column of M_calib, NaN filled with the item's mean). Cluster
    the points into k groups (e.g. scipy.cluster.vq.kmeans2, already a dependency), pick
    the member nearest each centre, and weight it by its cluster's share of items.

    Args:
        M_calib: response matrix of the calibration models.
        k: number of clusters (anchors); empty clusters are dropped, so there may be fewer.
        seed: clustering seed.

    Returns:
        The anchor item_ids, and anchor item_id -> weight (weights sum to 1).
    """
    answered = M_calib.dropna(axis=1, how="all")
    points = answered.fillna(answered.mean(axis=0)).T.to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # empty clusters are dropped below
        centres, labels = kmeans2(points, k, minit="points", rng=seed)
    clusters = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    nearest = [
        members[((points[members] - centres[labels[members[0]]]) ** 2).sum(axis=1).argmin()]
        for members in clusters
    ]
    anchors = answered.columns[nearest]
    weights = np.array([len(members) for members in clusters]) / len(points)
    return anchors, pd.Series(weights, index=anchors)


def weighted_scores(M: pd.DataFrame, items: pd.Index, weights: pd.Series) -> pd.Series:
    """Each model's weighted accuracy on `items`.

    Args:
        M: response matrix.
        items: item_ids to score on.
        weights: item_id -> weight. For a model with NaN answers, the weights of its
            answered items are rescaled to sum to 1.

    Returns:
        One value per model, index = M.index.
    """
    answers = M[items]
    w = weights[items]
    return (answers * w).sum(axis=1) / answers.notna().mul(w).sum(axis=1)


def calibrated_scores(M_calib: pd.DataFrame, M_eval: pd.DataFrame, items: pd.Index) -> pd.Series:
    """Predict held-out models' full scores from their subset scores with a linear fit.

    Fit full score = a + b * subset score on the calibration models (np.polyfit), then
    apply it to M_eval's subset scores.

    Returns:
        One predicted full score per model of M_eval.
    """
    subset = subset_scores(M_calib, items)
    answered = subset.notna()
    slope, intercept = np.polyfit(subset[answered], full_scores(M_calib)[answered], deg=1)
    return intercept + slope * subset_scores(M_eval, items)


def bootstrap_se(
    M: pd.DataFrame, items: pd.Index, n_boot: int, rng: np.random.Generator
) -> pd.Series:
    """Standard error of each model's subset score, from one subset alone.

    Resample `items` with replacement n_boot times and take the std of the subset score.
    This is the interval you can give when you only have one subset.

    Returns:
        One value per model, index = M.index.
    """
    draws = [
        subset_scores(M, items[rng.integers(len(items), size=len(items))]) for _ in range(n_boot)
    ]
    return pd.concat(draws, axis=1).std(axis=1)


def _anchor_errors(
    M_eval: pd.DataFrame, M_calib: pd.DataFrame, ks: Sequence[int], seed: int
) -> pd.DataFrame:
    """One rep per k: the anchors are fixed once the calibration models are."""
    rows = []
    for k in ks:
        anchors, weights = cluster_anchors(M_calib, k, seed)
        est = weighted_scores(M_eval, anchors, weights).to_numpy()
        rows.append(pd.DataFrame({"model": M_eval.index, "k": k, "rep": 0, "est": est}))
    return _error_table(M_eval, rows)


def subset_errors(
    M_eval: pd.DataFrame,
    method: str,
    ks: Sequence[int],
    n_rep: int = 1000,
    seed: int = 0,
    M_calib: pd.DataFrame | None = None,
    subtasks: pd.Series | None = None,
) -> pd.DataFrame:
    """random_subset_errors for any selection method, scored on held-out models.

    Args:
        M_eval: response matrix of the held-out models.
        method: "uniform", "subtask" (needs subtasks), "difficulty" or "systematic"
            (need M_calib; systematic walks items sorted by calibration difficulty), or
            "anchors" (needs M_calib; deterministic, so one rep per k).
        ks, n_rep, seed: as in random_subset_errors.
        M_calib: response matrix of the calibration models.
        subtasks: item_id -> subtask label.

    Returns:
        The random_subset_errors columns plus `method`.

    Raises:
        ValueError: for an unknown method or a missing M_calib / subtasks.
    """
    if method == "subtask" and subtasks is None:
        raise ValueError("method 'subtask' needs subtasks")
    if method in ("difficulty", "systematic", "anchors") and M_calib is None:
        raise ValueError(f"method {method!r} needs M_calib")

    match method:
        case "uniform":
            strata = None
        case "subtask":
            strata = subtasks
        case "difficulty":
            strata = difficulty_strata(item_difficulty(M_calib))
        case "systematic":
            rng = np.random.default_rng(seed)
            order = item_difficulty(M_calib).dropna().sort_values(kind="stable").index
            errors = _simulate(M_eval, ks, n_rep, lambda k: systematic_sample(order, k, rng))
            return errors.assign(method=method)
        case "anchors":
            return _anchor_errors(M_eval, M_calib, ks, seed).assign(method=method)
        case _:
            raise ValueError(f"unknown method {method!r}")
    return random_subset_errors(M_eval, ks, n_rep, seed, strata).assign(method=method)


# ---------- numpy-array versions used by scripts/irt (A. Vijaykrishnan) ----------


def random_items(n_items, k, rng):
    return rng.choice(n_items, size=k, replace=False)


def stratified_items(subjects, k, rng):
    """Positions of k random items, allocated to subjects by proportional_allocation."""
    return sample_items(pd.RangeIndex(len(subjects)), k, rng, pd.Series(subjects)).to_numpy()
