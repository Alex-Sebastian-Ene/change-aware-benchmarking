"""Tests for the random-subset baselines and noise reference (cabench.baselines)."""

import numpy as np
import pandas as pd
import pytest

from cabench import baselines as b


def matrix(n_models=3, n_items=400, seed=0):
    """Synthetic models at accuracy 0.2 .. 0.8, no missing answers."""
    rng = np.random.default_rng(seed)
    p = np.linspace(0.2, 0.8, n_models)[:, None]
    data = (rng.random((n_models, n_items)) < p).astype(float)
    return pd.DataFrame(
        data,
        index=[f"m{i}@rev" for i in range(n_models)],
        columns=[f"item{j}" for j in range(n_items)],
    )


# ---------- core ----------


def test_analytic_se():
    assert b.analytic_se(0.3, 100, 12000) == pytest.approx(0.04564, abs=1e-4)
    assert b.analytic_se(0.3, 12000, 12000) == 0
    assert b.analytic_se(np.array([0.3, 0.5]), 100, 12000).shape == (2,)
    with pytest.raises(ValueError):
        b.analytic_se(0.3, 0, 100)


def test_full_scores_skips_nan():
    M = pd.DataFrame([[1, 0, np.nan], [1, 1, 1]], index=["a", "b"])
    assert b.full_scores(M).tolist() == [0.5, 1.0]


def test_subset_scores():
    M = pd.DataFrame([[1, 0, 1], [0, 0, 1]], index=["a", "b"], columns=["x", "y", "z"])
    assert b.subset_scores(M, pd.Index(["x", "z"])).tolist() == [1.0, 0.5]


def test_proportional_allocation():
    strata = pd.Series(["a"] * 50 + ["b"] * 30 + ["c"] * 20)
    rng = np.random.default_rng(0)
    assert b.proportional_allocation(strata, 10, rng).to_dict() == {"a": 5, "b": 3, "c": 2}

    draws = pd.DataFrame([b.proportional_allocation(strata, 7, rng) for _ in range(2000)])
    assert (draws.sum(axis=1) == 7).all()
    assert draws["a"].isin([3, 4]).all() and draws["c"].isin([1, 2]).all()
    # on average each stratum gets exactly its share k * size / N
    assert draws.mean().to_dict() == pytest.approx({"a": 3.5, "b": 2.1, "c": 1.4}, abs=0.05)

    with pytest.raises(ValueError):
        b.proportional_allocation(strata, 101, rng)


def test_proportional_allocation_awkward_shares_and_nan():
    # thirds never add up exactly in floats, so the running total has to stay in integers
    strata = pd.Series(list("abcdefg") * 7 + ["h"] * 3)
    rng = np.random.default_rng(0)
    for k in range(len(strata) + 1):
        counts = b.proportional_allocation(strata, k, rng)
        assert counts.sum() == k and (counts <= strata.value_counts().sort_index()).all()

    assert b.proportional_allocation(pd.Series(["a", "a", None]), 2, rng).to_dict() == {"a": 2}


def test_sample_items_distinct_and_reproducible():
    items = pd.Index([f"i{j}" for j in range(100)])
    first = b.sample_items(items, 10, np.random.default_rng(1))
    assert len(set(first)) == 10 and set(first) <= set(items)
    assert list(first) == list(b.sample_items(items, 10, np.random.default_rng(1)))


def test_sample_items_stratified_counts():
    items = pd.Index([f"i{j}" for j in range(100)])
    strata = pd.Series(["a"] * 50 + ["b"] * 30 + ["c"] * 20, index=items)
    chosen = b.sample_items(items, 10, np.random.default_rng(1), strata=strata)
    assert strata[chosen].value_counts().to_dict() == {"a": 5, "b": 3, "c": 2}


def test_stratified_items_have_equal_inclusion_probability():
    items = pd.Index([f"i{j}" for j in range(30)])
    strata = pd.Series(["a"] * 17 + ["b"] * 11 + ["c"] * 2, index=items)
    rng = np.random.default_rng(0)
    picks = pd.Series(np.concatenate([b.sample_items(items, 7, rng, strata) for _ in range(20000)]))
    assert (picks.value_counts() / 20000).to_numpy() == pytest.approx(7 / 30, abs=0.02)


def test_random_subset_errors_shape_and_bias():
    err = b.random_subset_errors(matrix(), ks=[25, 100], n_rep=200, seed=0)
    assert len(err) == 3 * 2 * 200
    assert set(err.columns) >= {"model", "k", "rep", "est", "full", "err"}
    assert abs(err["err"].mean()) < 0.02


def test_simulation_matches_analytic():
    err = b.random_subset_errors(matrix(n_items=2000), ks=[50, 200], n_rep=2000, seed=0)
    ratio = b.empirical_vs_analytic(err, n_items=2000)["ratio"]
    assert ratio.between(0.9, 1.1).all()


def test_summarise_errors():
    err = pd.DataFrame({"k": [10, 10, 10], "err": [-0.1, 0.1, np.nan]})
    s = b.summarise_errors(err, levels=(0.95,))
    assert s.loc[10, "bias"] == pytest.approx(0)
    assert s.loc[10, "rmse"] == pytest.approx(0.1)
    assert "q95" in s.columns


def test_budget_for_error():
    summary = pd.DataFrame({"rmse": [0.09, 0.06, 0.045]}, index=pd.Index([25, 50, 100], name="k"))
    assert b.budget_for_error(summary, 0.05) == 100
    assert b.budget_for_error(summary, 0.01) is None


def test_noise_reference():
    long = pd.DataFrame(
        {
            "run_id": ["r1"] * 4 + ["r2"] * 4 + ["r3"] * 4,
            "model_id": ["m"] * 8 + ["n"] * 4,
            "model_revision": ["x"] * 12,
            "benchmark": ["bench"] * 12,
            "config_id": ["c"] * 12,
            "item_id": list("abcd") * 3,
            "correct": [1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
        }
    )
    nr = b.noise_reference(long, "bench", "c")
    assert nr.loc["m@x", "n_runs"] == 2
    assert nr.loc["m@x", "range"] == pytest.approx(0.25)
    assert np.isnan(nr.loc["n@x", "sd"])


# ---------- extensions ----------


def test_item_difficulty():
    M = pd.DataFrame([[1, 0, np.nan], [1, 1, 0]], columns=["x", "y", "z"])
    assert b.item_difficulty(M).tolist() == [1.0, 0.5, 0.0]


def test_difficulty_strata_equal_bins():
    diff = pd.Series(np.linspace(0, 1, 100), index=[f"i{j}" for j in range(100)])
    assert b.difficulty_strata(diff, n_bins=5).value_counts().tolist() == [20] * 5


def test_systematic_sample_even_spacing():
    order = pd.Index([f"i{j}" for j in range(100)])
    chosen = b.systematic_sample(order, 10, np.random.default_rng(0))
    assert len(chosen) == 10
    assert np.diff([order.get_loc(i) for i in chosen]).tolist() == [10] * 9


def test_weighted_scores():
    M = pd.DataFrame([[1, 0], [0, 0]], index=["a", "b"], columns=["x", "y"])
    weights = pd.Series({"x": 0.75, "y": 0.25})
    assert b.weighted_scores(M, pd.Index(["x", "y"]), weights).tolist() == [0.75, 0.0]


def test_cluster_anchors_weights():
    anchors, weights = b.cluster_anchors(matrix(n_models=8, n_items=200), k=10, seed=0)
    assert 1 <= len(anchors) <= 10 and len(set(anchors)) == len(anchors)
    assert set(weights.index) == set(anchors)
    assert weights.sum() == pytest.approx(1)


def test_calibrated_scores_identity_on_full_set():
    M = matrix()
    assert b.calibrated_scores(M, M, M.columns).tolist() == pytest.approx(b.full_scores(M).tolist())


def test_bootstrap_se_close_to_binomial():
    M = matrix(n_items=2000)
    items = M.columns[:400]
    se = b.bootstrap_se(M, items, n_boot=2000, rng=np.random.default_rng(0))
    p = b.subset_scores(M, items)
    assert (se / np.sqrt(p * (1 - p) / 400)).between(0.9, 1.1).all()


def test_subset_errors_methods():
    M = matrix(n_models=6, n_items=300)
    calib, held = M.iloc[:4], M.iloc[4:]
    subtasks = pd.Series(np.repeat(list("abc"), 100), index=M.columns)
    for method in ["uniform", "subtask", "difficulty", "systematic"]:
        err = b.subset_errors(
            held, method, ks=[30], n_rep=50, seed=0, M_calib=calib, subtasks=subtasks
        )
        assert set(err["method"]) == {method}
        assert len(err) == 2 * 50
    # 4 calibration models give at most 2**4 distinct item patterns, so keep k well below 16
    anchors = b.subset_errors(held, "anchors", ks=[8], M_calib=calib)
    assert set(anchors["model"]) == set(held.index)
    with pytest.raises(ValueError):
        b.subset_errors(held, "subtask", ks=[30])
