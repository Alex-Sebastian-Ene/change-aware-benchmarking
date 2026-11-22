import numpy as np
import pandas as pd
import pytest

from cabench.splits import (
    FAMILY_DIST,
    all_splits,
    behaviour_families,
    duplicate_groups,
    extrapolation_split,
    shift_split,
    size_classes,
    size_split,
    split_models,
    strength_split,
)

N = 100
VALUES = np.random.default_rng(1).permutation(N).astype(float)  # 0..99 shuffled, no ties
GROUPS = np.repeat(np.arange(30), np.random.default_rng(0).integers(1, 8, 30))  # sizes 1-7
FOURS = np.arange(N) // 4  # 25 groups of 4, for the N-model tests


def test_family_split_keeps_groups_together():
    groups = np.repeat(np.arange(30), np.random.default_rng(0).integers(1, 8, 30))
    tr, va, te = split_models(len(groups), 0, groups=groups)
    assert len(tr) + len(va) + len(te) == len(groups) == len(set(np.r_[tr, va, te]))
    assert not set(groups[tr]) & set(groups[va])
    assert not set(groups[tr]) & set(groups[te])
    assert not set(groups[va]) & set(groups[te])


def assert_partition(parts, expected) -> None:
    """The parts share no model and together are exactly `expected`."""
    joined = np.concatenate(parts)
    assert len(joined) == len(set(joined))
    assert set(joined) == set(expected)


def assert_same_split(got, expected) -> None:
    for a, b in zip(got, expected, strict=True):
        np.testing.assert_array_equal(a, b)


# ---------- shift_split ----------


def test_shift_split_sizes_and_partition():
    train, val, test = shift_split(VALUES, 0)
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert_partition((train, val, test), range(N))


@pytest.mark.parametrize("high", [True, False])
def test_shift_split_tests_on_one_end(high):
    train, val, test = shift_split(VALUES, 0, high=high)
    others = VALUES[np.r_[train, val]]
    if high:
        assert VALUES[test].min() > others.max()
    else:
        assert VALUES[test].max() < others.min()


def test_shift_split_val_is_random_not_the_next_band():
    train, val, _ = shift_split(VALUES, 0)
    # the band next to test is 70..84; a random val reaches well below it
    assert VALUES[val].min() < np.median(VALUES[np.r_[train, val]])


def test_shift_split_seed_changes_train_and_val_only():
    splits = [shift_split(VALUES, seed) for seed in range(3)]
    assert all(set(s[2]) == set(splits[0][2]) for s in splits)
    assert set(splits[0][1]) != set(splits[1][1])


def test_shift_split_same_seed_same_split():
    assert_same_split(shift_split(VALUES, 7), shift_split(VALUES, 7))


def test_shift_split_leaves_out_unknown_values():
    values = VALUES.copy()
    unknown = np.arange(0, N, 5)  # 20 models
    values[unknown] = np.nan
    train, val, test = shift_split(values, 0)
    assert_partition((train, val, test), np.setdiff1d(np.arange(N), unknown))
    assert (len(train), len(val), len(test)) == (56, 12, 12)  # shares of the 80 known


def test_shift_split_keeps_groups_together():
    values = np.random.default_rng(2).random(len(GROUPS))
    train, val, test = shift_split(values, 0, groups=GROUPS)
    assert_partition((train, val, test), range(len(GROUPS)))
    for a, b in [(train, val), (train, test), (val, test)]:
        assert not set(GROUPS[a]) & set(GROUPS[b])
    means = pd.Series(values).groupby(GROUPS).mean()
    test_groups, other_groups = np.unique(GROUPS[test]), np.unique(GROUPS[np.r_[train, val]])
    assert means[test_groups].min() >= means[other_groups].max()
    assert abs(len(test) - round(0.15 * len(GROUPS))) <= np.bincount(GROUPS).max()


@pytest.mark.parametrize(
    "fractions",
    [{"test_frac": 0}, {"test_frac": 1}, {"val_frac": 0}, {"test_frac": 0.5, "val_frac": 0.5}],
)
def test_shift_split_bad_fractions_raise(fractions):
    with pytest.raises(ValueError):
        shift_split(VALUES, 0, **fractions)


# ---------- strength_split ----------


def response_matrix(seed: int = 3, n_items: int = 40) -> np.ndarray:
    """N models x n_items, 1 / 0 with abilities spread out, about 10% NaN (no response)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.2, 0.9, N)
    Y = (rng.random((N, n_items)) < p[:, None]).astype(float)
    Y[rng.random(Y.shape) < 0.1] = np.nan
    return Y


@pytest.mark.parametrize("groups", [None, FOURS])
@pytest.mark.parametrize("direction, high", [("stronger", True), ("weaker", False)])
def test_strength_split_is_shift_split_on_scores(direction, high, groups):
    Y = response_matrix()
    expected = shift_split(np.nanmean(Y, axis=1), 0, high=high, groups=groups)
    assert_same_split(strength_split(Y, 0, direction, groups=groups), expected)
    assert_same_split(strength_split(pd.DataFrame(Y), 0, direction, groups=groups), expected)


def test_strength_split_tests_on_the_strongest():
    Y = response_matrix()
    score = np.nanmean(Y, axis=1)
    train, val, test = strength_split(Y, 0)
    assert score[test].min() >= score[np.r_[train, val]].max()


def test_strength_split_leaves_out_models_with_no_responses():
    Y = response_matrix()
    Y[[3, 50]] = np.nan
    assert_partition(strength_split(Y, 0, "weaker"), np.setdiff1d(np.arange(N), [3, 50]))


def test_strength_split_bad_direction_raises():
    with pytest.raises(ValueError):
        strength_split(response_matrix(), 0, "up")


def side_of(parts, n: int) -> np.ndarray:
    """0 / 1 / 2 for a model in train / val / test, -1 for a model in no part."""
    side = np.full(n, -1)
    for k, part in enumerate(parts):
        side[part] = k
    return side


def test_shift_split_without_a_gap_keeps_everyone():
    values = VALUES.copy()
    values[[3, 50]] = np.nan
    assert_partition(shift_split(values, 0, gap=1), np.setdiff1d(np.arange(N), [3, 50]))


# ---------- size_classes ----------


def test_size_classes_round_to_whole_billions():
    sizes = [14.66, 14.766, 14.77, 7.242, 0.494, 2.5, np.nan]
    np.testing.assert_array_equal(size_classes(sizes), [15, 15, 15, 7, 0, 3, np.nan])


# ---------- extrapolation_split ----------


def lognormal_sizes() -> np.ndarray:
    """N continuous sizes, about 1-50B, three unknown (NaN)."""
    sizes = np.random.default_rng(4).lognormal(2, 1, N)
    sizes[[0, 1, 2]] = np.nan
    return sizes


def leaderboard_sizes(n: int = 300) -> np.ndarray:
    """Leaderboard-like sizes: many models share an exact size, some differ in the decimals."""
    pool = [0.494, 1.544, 3.821, 7.242, 7.616, 8.03, 12.248, 14.66, 14.766, 14.77, 32.764, 70.554]
    return np.random.default_rng(6).choice(pool, n)


@pytest.mark.parametrize("direction", ["larger", "smaller"])
def test_extrapolation_split_keeps_the_size_gap(direction):
    sizes = lognormal_sizes()
    train, val, test = extrapolation_split(sizes, 0, direction)
    others = sizes[np.r_[train, val]]
    if direction == "larger":
        assert sizes[test].min() >= 1.25 * others.max()
    else:
        assert sizes[test].max() * 1.25 <= others.min()


@pytest.mark.parametrize("direction", ["larger", "smaller"])
def test_extrapolation_split_leaves_out_only_the_gap(direction):
    sizes = lognormal_sizes()
    train, val, test = extrapolation_split(sizes, 0, direction)
    out = np.flatnonzero(side_of((train, val, test), N) == -1)
    known_out = out[~np.isnan(sizes[out])]
    assert set(out) - set(known_out) == {0, 1, 2}  # the unknown sizes
    assert len(known_out) > 0  # the gap is in use with these sizes
    if direction == "larger":
        assert (sizes[known_out] * 1.25 > sizes[test].min()).all()
    else:
        assert (sizes[known_out] < sizes[test].max() * 1.25).all()


def test_extrapolation_split_takes_whole_sizes():
    # the worked example: 3 test models wanted, but there are four 70B models
    sizes = np.r_[[7.0] * 10, [13.0] * 6, [70.0] * 4][np.random.default_rng(0).permutation(20)]
    train, val, test = extrapolation_split(sizes, 0)
    assert sizes[test].tolist() == [70.0] * 4
    assert sizes[np.r_[train, val]].max() == 13.0
    assert_partition((train, val, test), range(20))


@pytest.mark.parametrize("direction", ["larger", "smaller"])
@pytest.mark.parametrize("gap", [1, 1.25])
def test_extrapolation_split_never_splits_a_size_class(direction, gap):
    sizes = leaderboard_sizes()
    classes = size_classes(sizes)
    train, val, test = extrapolation_split(sizes, 0, direction, gap=gap)
    assert not set(classes[test]) & set(classes[np.r_[train, val]])
    if gap == 1:
        assert_partition((train, val, test), range(len(sizes)))


@pytest.mark.parametrize("direction", ["larger", "smaller"])
def test_extrapolation_split_val_is_random_models_not_whole_classes(direction):
    sizes = leaderboard_sizes()
    classes = size_classes(sizes)
    train, val, _ = extrapolation_split(sizes, 0, direction)
    assert len(val) == round(0.15 * len(sizes))
    assert set(classes[val]) & set(classes[train])  # a class can sit on both train and val


@pytest.mark.parametrize("gap", [1, 1.25])
def test_extrapolation_split_keeps_groups_together(gap):
    sizes = lognormal_sizes()
    groups = np.arange(N)
    big, small = np.nanargmax(sizes), np.nanargmin(sizes)
    groups[small] = groups[big]  # a group spanning the smallest and the largest size
    side = side_of(extrapolation_split(sizes, 0, groups=groups, gap=gap), N)
    # never in two different parts; with a gap, a member may sit out (in no part)
    assert len({side[small], side[big]} - {-1}) == 1


def test_extrapolation_split_seed_changes_train_and_val_only():
    sizes = lognormal_sizes()
    splits = [extrapolation_split(sizes, seed) for seed in range(3)]
    assert all(set(s[2]) == set(splits[0][2]) for s in splits)
    assert set(splits[0][1]) != set(splits[1][1])


@pytest.mark.parametrize(
    "kwargs",
    [{"direction": "up"}, {"gap": 0.9}, {"params_b": np.r_[lognormal_sizes()[3:], 0.0]}],
)
def test_extrapolation_split_bad_input_raises(kwargs):
    args = {"params_b": lognormal_sizes(), "seed": 0} | kwargs
    with pytest.raises(ValueError):
        extrapolation_split(**args)


# ---------- duplicate_groups ----------


def near_copies() -> np.ndarray:
    """6 models x 200 items: models 0-3 are near-duplicates (3 only by a chain), 4 and 5 not."""
    rng = np.random.default_rng(5)
    Y = (rng.random((6, 200)) < 0.6).astype(float)
    Y[1] = Y[0]  # an exact copy of 0
    Y[2] = Y[0]
    Y[2, :2] = 1 - Y[2, :2]  # 99% the same as 0
    Y[3] = Y[2]
    Y[3, 2:4] = 1 - Y[3, 2:4]  # 99% as 2 but 98% as 0: joined through 2
    Y[4] = Y[0]
    Y[4, 10:14] = 1 - Y[4, 10:14]  # 98% as 0 and 1, less as 2 and 3
    return Y


def test_duplicate_groups_joins_near_copies_in_chains():
    labels = duplicate_groups(near_copies())
    assert labels[0] == labels[1] == labels[2] == labels[3]
    assert len(set(labels)) == 3


def test_duplicate_groups_threshold():
    labels = duplicate_groups(near_copies(), min_agreement=0.98)
    assert labels[4] == labels[0]  # 98% now counts
    assert labels[5] != labels[0]


def test_duplicate_groups_compares_only_items_both_answered():
    Y = near_copies()
    Y[5] = Y[0]
    Y[5, :20] = 1 - Y[5, :20]
    Y[5, :20] = np.nan  # the differences are hidden: 100% on the items both answered
    Y[4] = np.nan  # no responses at all
    labels = duplicate_groups(pd.DataFrame(Y))
    assert labels[5] == labels[0]
    assert (labels[4] != np.delete(labels, 4)).all()


# ---------- size_split ----------


def test_size_split_holds_out_part_of_every_class():
    sizes = np.repeat([1.0, 3.0, 7.0, 13.0, 70.0], 40)[np.random.default_rng(0).permutation(200)]
    train, val, test = size_split(sizes, 0)
    assert_partition((train, val, test), range(200))
    for size in (1.0, 3.0, 7.0, 13.0, 70.0):
        assert (sizes[test] == size).sum() == 6  # 15% of 40
        assert (sizes[val] == size).sum() == 6


def test_size_split_shares_hold_with_many_small_classes():
    sizes = np.repeat(np.arange(1.0, 301.0), 3)  # 300 classes of 3 models
    train, val, test = size_split(sizes, 0)
    assert_partition((train, val, test), range(900))
    assert abs(len(test) - 135) <= 30 and abs(len(val) - 135) <= 30  # 15% of 900, +-3.5 SD


def test_size_split_leaves_out_unknown_sizes():
    sizes = leaderboard_sizes()
    sizes[[0, 7]] = np.nan
    assert_partition(size_split(sizes, 0), np.setdiff1d(np.arange(len(sizes)), [0, 7]))


def test_size_split_keeps_groups_together():
    sizes = leaderboard_sizes()
    groups = np.arange(len(sizes)) // 3
    groups[0] = groups[-1]  # one group spanning two size classes
    side = side_of(size_split(sizes, 0, groups=groups), len(sizes))
    for g in np.unique(groups):
        assert len(set(side[groups == g])) == 1


def test_size_split_seeds():
    sizes = leaderboard_sizes()
    assert_same_split(size_split(sizes, 3), size_split(sizes, 3))
    assert set(size_split(sizes, 0)[2]) != set(size_split(sizes, 1)[2])


def test_size_split_bad_fractions_raise():
    with pytest.raises(ValueError):
        size_split(leaderboard_sizes(), 0, test_frac=0.5, val_frac=0.5)


# ---------- all_splits ----------


def complete_matrix(n: int = 60, n_items: int = 300) -> np.ndarray:
    """n models x n_items, complete 1 / 0, with near-duplicate pairs (0, 1) and (10, 11)."""
    rng = np.random.default_rng(7)
    p = rng.uniform(0.3, 0.9, n)
    Y = (rng.random((n, n_items)) < p[:, None]).astype(float)
    Y[1] = Y[0]
    Y[11] = Y[10]
    Y[11, :2] = 1 - Y[11, :2]
    return Y


def sizes_for(n: int = 60) -> np.ndarray:
    sizes = leaderboard_sizes(n)
    sizes[1], sizes[11] = sizes[0], sizes[10]  # a duplicate has its twin's size
    sizes[5] = np.nan
    return sizes


def test_all_splits_names():
    Y = complete_matrix()
    always, with_sizes = {"family", "stronger", "weaker"}, {"per_size", "larger", "smaller"}
    assert set(all_splits(Y, 0)) == always
    assert set(all_splits(Y, 0, params_b=sizes_for())) == always | with_sizes


@pytest.mark.parametrize("seed", range(3))
def test_all_splits_never_separate_duplicates(seed):
    Y = complete_matrix()
    for name, parts in all_splits(Y, seed, params_b=sizes_for()).items():
        joined = np.concatenate(parts)
        assert len(joined) == len(set(joined)), name
        side = side_of(parts, len(Y))
        assert side[0] == side[1] and side[10] == side[11], name


@pytest.mark.parametrize("seed", range(5))
def test_all_splits_family_split_is_the_one_runners_draw(seed):
    Y = complete_matrix()
    expected = split_models(len(Y), seed, groups=behaviour_families(Y, FAMILY_DIST))
    assert_same_split(all_splits(Y, seed)["family"], expected)


def test_all_splits_joins_given_families_split_by_a_duplicate():
    Y = complete_matrix()
    families = np.arange(len(Y)) // 2  # puts 10 and 11 in one family, 0 and 1 in another...
    families[1] = 99  # ...until 1 is moved to a family of its own
    side = side_of(all_splits(Y, 0, families=families)["family"], len(Y))
    assert side[0] == side[1]


def test_all_splits_wrong_length_raises():
    with pytest.raises(ValueError):
        all_splits(complete_matrix(), 0, params_b=sizes_for(59))
