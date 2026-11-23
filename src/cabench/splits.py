"""Split definitions (core objective 7). Owners: S. Ene, J. Hall.

Runners should take their splits from all_splits, so the same rules hold everywhere:

  family    split_models(n, seed, groups=families)          whole families held out
  stronger  strength_split(Y, seed, "stronger")             top scorers held out
  weaker    strength_split(Y, seed, "weaker")               bottom scorers held out
  per_size  size_split(params_b, seed)                      part of every size class held out
  larger    extrapolation_split(params_b, seed, "larger")   biggest size classes, after a size gap
  smaller   extrapolation_split(params_b, seed, "smaller")  smallest size classes, after a size gap

Near-duplicate models (duplicate_groups) are never on two sides of any split. Relatives
(fine-tunes of one base model) may be, except in the family split. The size splits need
params_b, which the efficbench MMLU matrix lacks, so all_splits leaves them out without it.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import coo_array, csr_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform

TEST_FRAC, VAL_FRAC = 0.15, 0.15  # the shares split_models also uses (70/15/15)
FAMILY_DIST = 0.3  # behaviour_families cut-off, as the runners use it
DUPLICATE_AGREEMENT = 0.99  # same answer on this share of items: one model (as scripts/irt)
SIZE_GAP = 1.25  # extrapolation: each test model at least 1.25x every train/val model's size
Split = tuple[np.ndarray, np.ndarray, np.ndarray]  # (train, val, test) model indices


def split_models(n_models, seed, groups=None):
    """70/15/15 split of model indices into train / val / test.

    With `groups`, each group stays on one side (held-out family split).
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.permutation(n_models)
        a, b = int(0.70 * n_models), int(0.85 * n_models)
        return idx[:a], idx[a:b], idx[b:]
    groups = np.asarray(groups)
    order = np.concatenate(
        [np.flatnonzero(groups == g) for g in rng.permutation(np.unique(groups))]
    )
    cum = np.arange(1, n_models + 1)
    # cut only at group boundaries
    ends = np.flatnonzero(np.r_[groups[order][1:] != groups[order][:-1], True])
    a = ends[np.abs(cum[ends] - 0.70 * n_models).argmin()] + 1
    b = ends[np.abs(cum[ends] - 0.85 * n_models).argmin()] + 1
    return order[:a], order[a:b], order[b:]


def behaviour_families(Y, max_dist=0.3):
    """Cluster models by answer correlation (average linkage on 1 - corr).

    shortcut: the efficbench data has no base-model lineage, so a family is behavioural;
    use the `family` / `parent_model` columns of models.parquet once that matrix is used.
    """
    D = 1 - np.corrcoef(Y)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), "average"), max_dist, "distance")


# ---------- grouping rules ----------


def duplicate_groups(Y: ArrayLike, min_agreement: float = DUPLICATE_AGREEMENT) -> np.ndarray:
    """One label per model, shared by near-duplicates.

    Two models are near-duplicates when they give the same answer (both right or both
    wrong) on at least `min_agreement` of the items both answered, such as one model
    uploaded under two names. Chains join: if A matches B and B matches C, all three share
    a label, even when A and C are further apart.

    Args:
        Y: models x items, 1 / 0 / NaN (NaN = no response), as a numpy array or DataFrame.
        min_agreement: share of the commonly answered items that must match.

    Returns:
        Integer labels, one per row of Y. A model with no near-duplicate, or with no
        responses, has a label of its own.
    """
    Y = np.asarray(Y, dtype=np.float32)
    seen = ~np.isnan(Y)
    right = (seen & (Y == 1)).astype(np.float32)
    wrong = (seen & (Y == 0)).astype(np.float32)
    shown = seen.astype(np.float32)
    # counts of items, exact in float32 below 2**24 items
    same = right @ right.T + wrong @ wrong.T
    both = shown @ shown.T
    linked = (same >= min_agreement * both) & (both > 0)
    return connected_components(csr_array(linked), directed=False)[1]


def size_classes(params_b: ArrayLike) -> np.ndarray:
    """Size class per model: params_b rounded to whole billions, halves up; NaN stays NaN.

    The leaderboard gives one architecture slightly different sizes (14.766 and 14.77), so
    the size splits treat a whole class as one size.
    """
    return np.floor(np.asarray(params_b, dtype=float) + 0.5)


def _merge(*labels: ArrayLike) -> np.ndarray:
    """Component id per model: models sharing a label in any input are joined, in chains.

    A missing label (None or NaN) joins nothing.
    """
    n = len(labels[0])
    rows, cols, nodes = [], [], n
    for lab in labels:
        codes, uniques = pd.factorize(np.asarray(lab))  # -1 where missing
        has = codes >= 0
        rows.append(np.flatnonzero(has))
        cols.append(codes[has] + nodes)
        nodes += len(uniques)
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_array((np.ones(len(r)), (r, c)), shape=(nodes, nodes))
    return connected_components(graph, directed=False)[1][:n]


def _nested(inner: ArrayLike, outer: ArrayLike) -> bool:
    """True if every `inner` group lies within a single `outer` group."""
    return bool(pd.Series(np.asarray(outer)).groupby(np.asarray(inner)).nunique().max() <= 1)


# ---------- helpers ----------


def _check_fractions(test_frac: float, val_frac: float) -> None:
    if not (0 < test_frac < 1 and 0 < val_frac < 1 and test_frac + val_frac < 1):
        raise ValueError(
            f"test_frac and val_frac must be in (0, 1) and sum to less than 1, "
            f"got {test_frac} and {val_frac}"
        )


def _take(units: list[np.ndarray], target: float, at_least_one: bool = True) -> int:
    """How many leading units to take so that their model count is nearest `target`.

    With at_least_one, a non-empty list always gives at least one unit; otherwise taking
    none is allowed, when 0 is nearest.
    """
    if not units:
        return 0
    ends = np.cumsum([len(u) for u in units])
    if at_least_one:
        return int(np.abs(ends - target).argmin()) + 1
    return int(np.abs(np.r_[0, ends] - target).argmin())


def _join(units: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(units) if units else np.array([], dtype=int)


def _round_random(x: float, rng: np.random.Generator) -> int:
    """x rounded down or up at random, up with probability x - floor(x): right on average."""
    low = np.floor(x)
    return int(low) + int(rng.random() < x - low)


# ---------- distribution-shift splits (spec objective 4) ----------


def shift_split(
    values: ArrayLike,
    seed: int,
    high: bool = True,
    groups: ArrayLike | None = None,
    test_frac: float = TEST_FRAC,
    val_frac: float = VAL_FRAC,
    gap: float = 1.0,
    rest_groups: ArrayLike | None = None,
) -> Split:
    """Test on the models at one end of `values`; train and val split the rest at random.

    The test part is what the predictor never saw: models that score higher, or are
    bigger, than every model it was fitted or tuned on. Val is drawn at random from the
    rest, not from the band next to test, so choosing D or the gp-IRT weights never sees
    the shift, as in real use.

    Args:
        values: one number per model (full-benchmark score, params_b, ...). NaN means
            unknown: that model goes in no part, as it can't be placed on either side.
        seed: seed for the random train / val split. The test part doesn't depend on it.
        high: True tests on the largest values, False on the smallest.
        groups: one label per model, as in split_models: each group stays in one part.
            A group is ranked by the mean value of its known members, and each cut falls
            at the group boundary nearest its share, so part sizes are only near it.
        test_frac, val_frac: shares of the models with a known value; train gets the rest.
        gap: ratio kept clear between test and the rest. With high, a model goes to train
            or val only if gap times its value is at most the smallest test value (with
            low, only if its value is at least gap times the largest); models in between
            go in no part. 1 keeps everyone.
        rest_groups: one label per model, deciding what stays together when train and val
            split the rest; defaults to `groups`. extrapolation_split passes finer labels
            than its size classes, so a class is kept whole only at the test cut.

    Returns:
        (train, val, test) indices into `values`: disjoint, and together every model with
        a known value that the gap doesn't leave out. Without groups, test has
        round(test_frac * n) models and val round(val_frac * n), where n is the number of
        known values; on tiny inputs each gets at least one while models are left.

    Raises:
        ValueError: if test_frac or val_frac is outside (0, 1), or they sum to 1 or more;
            if gap is below 1; or if gap is above 1 and a known value is not positive.
    """
    _check_fractions(test_frac, val_frac)
    if gap < 1:
        raise ValueError(f"gap must be at least 1, got {gap}")
    values = np.asarray(values, dtype=float)
    known = np.flatnonzero(~np.isnan(values))
    if gap > 1 and (values[known] <= 0).any():
        raise ValueError("a gap is a ratio, so every known value must be positive")
    n = len(known)

    # A unit is a group, or a single model without groups; it is never split up.
    labels = known if groups is None else np.asarray(groups)[known]
    unit_ids, unit_of = np.unique(labels, return_inverse=True)
    means = np.bincount(unit_of, weights=values[known]) / np.bincount(unit_of)
    by_rank = np.argsort(-means if high else means, kind="stable")  # test end first
    ranked = [known[unit_of == u] for u in by_rank]

    t = _take(ranked, round(test_frac * n))
    test, rest = _join(ranked[:t]), ranked[t:]
    if gap > 1 and len(test):
        if high:
            smallest = values[test].min()
            rest = [u[values[u] * gap <= smallest] for u in rest]
        else:
            largest = values[test].max()
            rest = [u[values[u] >= largest * gap] for u in rest]
        rest = [u for u in rest if len(u)]
    if rest_groups is not None and rest:
        models = _join(rest)
        _, label_of = np.unique(np.asarray(rest_groups)[models], return_inverse=True)
        order = np.argsort(label_of, kind="stable")
        rest = np.split(models[order], np.flatnonzero(np.diff(label_of[order])) + 1)
    rng = np.random.default_rng(seed)
    rest = [rest[i] for i in rng.permutation(len(rest))]
    v = _take(rest, round(val_frac * n))
    return _join(rest[v:]), _join(rest[:v]), test


def strength_split(
    Y: ArrayLike, seed: int, direction: str = "stronger", groups: ArrayLike | None = None
) -> Split:
    """Test on the strongest (or weakest) models by full-benchmark score.

    The score is the same full-benchmark accuracy the predictors try to recover, so this
    shift is deliberate, not a leak: it asks how far a subset calibrated on weaker models
    carries to stronger ones (and the reverse). Test scores sit in a narrow band, so
    report MAE and bias here, and don't compare its rank correlations with other splits.

    Args:
        Y: models x items, 1 / 0 / NaN (NaN = no response, skipped in the score), as a
            numpy array or a DataFrame like baselines.py uses. A row with no responses has
            no score, so that model goes in no part.
        seed: as in shift_split.
        direction: "stronger" tests on the highest scores, "weaker" on the lowest.
        groups: as in shift_split.

    Returns:
        (train, val, test) row positions in Y, as shift_split gives them for the scores.

    Raises:
        ValueError: if direction is neither "stronger" nor "weaker".
    """
    if direction not in ("stronger", "weaker"):
        raise ValueError(f'direction must be "stronger" or "weaker", got {direction!r}')
    Y = np.asarray(Y, dtype=float)
    answered = (~np.isnan(Y)).sum(axis=1)
    # mean over answered items; NaN (no warning) for a model with none, left out below
    score = np.nansum(Y, axis=1) / np.where(answered > 0, answered, np.nan)
    return shift_split(score, seed, high=direction == "stronger", groups=groups)


def extrapolation_split(
    params_b: ArrayLike,
    seed: int,
    direction: str = "larger",
    groups: ArrayLike | None = None,
    gap: float = SIZE_GAP,
) -> Split:
    """Test on the largest (or smallest) size classes, with a size gap before train and val.

    Whole size classes (size_classes) go to one side, so models of one size are never on
    both. Then every model within `gap` of the test sizes sits out: with "larger" and the
    default 1.25, each test model is at least 1.25 times the size of every train and val
    model. Models in the gap are only left out of this split.

    Args:
        params_b: parameters in billions, one per model, as in models.parquet. NaN (size
            unknown) leaves that model out.
        seed: as in shift_split.
        direction: "larger" tests on the biggest classes, "smaller" on the smallest.
        groups: one label per model that must stay on one side, such as duplicate_groups.
            A group spanning two size classes joins them.
        gap: size ratio kept clear between test and train/val; 1 means no gap.

    Returns:
        (train, val, test) model indices, as shift_split gives them for params_b; models in
        the gap are in no part.

    Raises:
        ValueError: if direction is neither "larger" nor "smaller", gap is below 1, or a
            known size is not positive.
    """
    if direction not in ("larger", "smaller"):
        raise ValueError(f'direction must be "larger" or "smaller", got {direction!r}')
    params_b = np.asarray(params_b, dtype=float)
    classes = size_classes(params_b)
    units = classes if groups is None else _merge(classes, groups)
    # whole classes only at the test cut; train and val split the rest by `groups` alone
    rest_groups = np.arange(len(params_b)) if groups is None else groups
    return shift_split(
        params_b,
        seed,
        high=direction == "larger",
        groups=units,
        gap=gap,
        rest_groups=rest_groups,
    )


# ---------- held-out split across sizes ----------


def size_split(
    params_b: ArrayLike,
    seed: int,
    groups: ArrayLike | None = None,
    test_frac: float = TEST_FRAC,
    val_frac: float = VAL_FRAC,
) -> Split:
    """Hold out part of every size class: test_frac of each as test, val_frac as val.

    Unlike extrapolation_split, test spans every size, so the seen (train) and unseen
    (test) models cover the same range, and their errors and rank correlations compare
    fairly. Relatives of a test model may be in train: that is one of the cases tested.

    A class's counts are test_frac and val_frac of its models, rounded up or down at random
    in proportion (a class of 3 models gets a test model 45% of the time), so the overall
    shares hold on average even with many small classes.

    Args:
        params_b: parameters in billions, one per model. NaN leaves that model out.
        seed: seed for the draw.
        groups: one label per model that must stay on one side, such as duplicate_groups.
            A group spanning two size classes joins them into one stratum.
        test_frac, val_frac: as in shift_split.

    Returns:
        (train, val, test) model indices: disjoint, and together every model with a known
        size.

    Raises:
        ValueError: if test_frac or val_frac is outside (0, 1), or they sum to 1 or more.
    """
    _check_fractions(test_frac, val_frac)
    params_b = np.asarray(params_b, dtype=float)
    known = np.flatnonzero(~np.isnan(params_b))
    classes = size_classes(params_b)[known]
    if groups is None:
        strata, unit_of = classes, known
    else:
        unit_of = np.asarray(groups)[known]
        strata = _merge(classes, unit_of)
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for stratum in np.unique(strata):
        inside = strata == stratum
        members, labels = known[inside], unit_of[inside]
        units = [members[labels == u] for u in np.unique(labels)]
        units = [units[i] for i in rng.permutation(len(units))]
        n = len(members)
        t = _take(units, _round_random(test_frac * n, rng), at_least_one=False)
        v = t + _take(units[t:], _round_random(val_frac * n, rng), at_least_one=False)
        test += units[:t]
        val += units[t:v]
        train += units[v:]
    return _join(train), _join(val), _join(test)


# ---------- every split, one set of rules ----------


def all_splits(
    Y: ArrayLike,
    seed: int,
    params_b: ArrayLike | None = None,
    families: ArrayLike | None = None,
) -> dict[str, Split]:
    """Every split for one seed, by name, with the same rules everywhere.

    Near-duplicates (duplicate_groups) stay on one side in every split. Without params_b,
    as for the efficbench MMLU matrix, only family, stronger and weaker come back.

    Args:
        Y: models x items, 1 / 0 / NaN, as in strength_split. Without `families` it must
            be complete, as behaviour_families needs.
        seed: split seed; the runners use 0 to 4.
        params_b: parameters in billions, one per model, NaN where unknown. Adds per_size,
            larger and smaller.
        families: one family label per model, such as models.parquet's family or lineage
            groups. Defaults to behaviour_families(Y, FAMILY_DIST), as the runners use now,
            so the family split is the one they already draw.

    Returns:
        {name: (train, val, test)} for family, stronger and weaker, and with params_b also
        per_size, larger and smaller.

    Raises:
        ValueError: if params_b or families has a length other than the number of models.
    """
    Y = np.asarray(Y, dtype=float)
    n = len(Y)
    for name, extra in (("params_b", params_b), ("families", families)):
        if extra is not None and len(extra) != n:
            raise ValueError(f"{name} has {len(extra)} entries for {n} models")
    duplicates = duplicate_groups(Y)
    fam = behaviour_families(Y, FAMILY_DIST) if families is None else np.asarray(families)
    if not _nested(duplicates, fam):
        fam = _merge(fam, duplicates)
    splits = {
        "family": split_models(n, seed, groups=fam),
        "stronger": strength_split(Y, seed, "stronger", groups=duplicates),
        "weaker": strength_split(Y, seed, "weaker", groups=duplicates),
    }
    if params_b is not None:
        splits["per_size"] = size_split(params_b, seed, groups=duplicates)
        for direction in ("larger", "smaller"):
            splits[direction] = extrapolation_split(params_b, seed, direction, groups=duplicates)
    return splits
