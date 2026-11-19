import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from cabench.pairs import (
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


def simulate_family(rng, n_pairs=40, n_items=1500, n_subjects=5):
    """Parents follow 1-D IRT plus item quirks; children keep most answers and gain in a subject.

    The subject gain and the quirks are outside plain IRT, so the model is misspecified.
    """
    alpha, beta = rng.normal(1, 0.3, (n_items, 1)), rng.normal(size=n_items)
    codes = np.arange(n_items) % n_subjects
    theta = rng.normal(size=(n_pairs, 1))
    P = expit(theta @ alpha.T - beta + rng.normal(0, 1, (n_pairs, n_items)))
    parents = (rng.random(P.shape) < P).astype(float)
    children = parents.copy()
    for m, y in enumerate(children):
        target = codes == m % n_subjects
        gain = (y == 0) & target & (rng.random(n_items) < 0.4)
        forget = (y == 1) & (rng.random(n_items) < 0.03)
        y[gain], y[forget] = 1, 0
    train_theta = rng.normal(size=(100, 1))
    problems = [
        pair_problem(p, c, alpha, beta, train_theta, codes, n_subjects)
        for p, c in zip(parents, children, strict=True)
    ]
    return problems


def test_find_pairs_follows_lineage_and_needs_the_parent_in_the_matrix():
    models = pd.DataFrame(
        {
            "model_id": ["base", "child", "grandchild", "orphan"],
            "parent_model": [None, "base", "child", "missing"],
        }
    )
    rows = ["base@1", "child@2", "grandchild@3", "orphan@4"]
    pairs, roots = find_pairs(models, rows)
    assert pairs.child.tolist() == ["child@2", "grandchild@3"]
    assert pairs.parent.tolist() == ["base@1", "child@2"]
    assert pairs.root.tolist() == ["base", "base"]
    assert roots.tolist() == ["base", "base", "base", "missing"]


def test_near_duplicate_siblings_are_dropped_and_missing_answers_ignored():
    rng = np.random.default_rng(0)
    a, other = rng.integers(0, 2, (2, 500)).astype(float)
    clone = a.copy()
    clone[:50] = np.nan
    Y = np.array([a, clone, other])
    pairs = pd.DataFrame({"child": ["a", "clone", "other"], "parent": "p", "root": "p"})
    kept = drop_near_duplicates(pairs, Y, ["a", "clone", "other"])
    assert kept.child.tolist() == ["a", "other"]


def test_interval_has_width_when_no_answer_flipped():
    row = change_interval(0.0, np.zeros(50), 1000)
    assert row["low_95"] < 0 < row["high_95"]
    assert row["high_95"] - row["low_95"] > row["high_80"] - row["low_80"]


def test_corrected_change_is_unbiased_and_beats_paired_when_the_model_is_wrong():
    rng = np.random.default_rng(0)
    problems = simulate_family(rng)
    train, test = problems[:20], problems[20:]
    prior = learn_prior(train)
    k, paired, corrected, covered = 200, [], [], []
    for problem in test:
        n_items, truth = len(problem["X"]), problem["true_change"]
        for _ in range(10):
            items = rng.permutation(n_items)[:k]
            paired.append(paired_change(problem, items)[0] - truth)
            estimate, residuals = corrected_change(problem, items, prior)
            corrected.append(estimate - truth)
            row = change_interval(estimate, residuals, n_items)
            covered.append(row["low_95"] <= truth <= row["high_95"])
    corrected, paired = np.array(corrected), np.array(paired)
    assert abs(corrected.mean()) < 0.003
    assert np.abs(corrected).mean() < np.abs(paired).mean()
    assert 0.9 <= np.mean(covered) <= 0.99


def test_parent_terms_improve_heldout_prediction_and_model_uses_seen_answers():
    problems = simulate_family(np.random.default_rng(1), n_pairs=12)
    prior = learn_prior(problems[:8])
    losses = pd.DataFrame([heldout_losses(p, prior) for p in problems[8:]]).mean()
    assert losses["+ parent answers"] < losses["ability shift"]
    assert losses["+ subject drift"] < losses["+ parent answers"]

    problem = problems[8]
    everything = np.arange(len(problem["X"]))
    assert model_change(problem, everything, prior) == pytest.approx(problem["true_change"])


def test_run_pairs_end_to_end(tmp_path):
    pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/irt"))
    import run_pairs

    rng = np.random.default_rng(2)
    n_items, n_roots = 300, 12
    alpha, beta = rng.normal(1, 0.3, (n_items, 1)), rng.normal(size=n_items)
    theta = rng.normal(size=(n_roots, 1))
    parents = (rng.random((n_roots, n_items)) < expit(theta @ alpha.T - beta)).astype(np.float32)
    children = parents.copy()
    flips = rng.random(children.shape) < 0.08
    children[flips] = 1 - children[flips]
    Y = np.vstack([parents, children])
    Y[0, :5] = np.nan
    parent_rows = [f"p{i}@a" for i in range(n_roots)]
    child_rows = [f"c{i}@b" for i in range(n_roots)]
    pairs = pd.DataFrame(
        {
            "child": child_rows,
            "parent": parent_rows,
            "root": [f"p{i}" for i in range(n_roots)],
            "kind": run_pairs.FOCUS,
        }
    )
    roots = np.tile(pairs["root"].to_numpy(), 2)
    summary, paired, checks = run_pairs.run(
        Y,
        np.array(parent_rows + child_rows),
        pairs,
        roots,
        np.arange(n_items) % 3,
        tmp_path,
        n_dims=1,
        ks=[20, 100],
        seeds=range(2),
        folds=run_pairs.make_folds(pairs, n_folds=3),
        epochs=50,
    )
    assert set(summary.method) == {
        run_pairs.PAIRED,
        run_pairs.GAP,
        run_pairs.VOPTIMAL,
        run_pairs.MODEL,
        run_pairs.CORRECTED,
    }
    assert summary.abs_error.notna().all() and len(checks) == 8
    assert (tmp_path / "predictions.csv").exists() and len(paired) == 10
    assert summary.loc[summary.method == run_pairs.VOPTIMAL, "coverage_95"].notna().all()


def test_conformal_intervals_cover_at_the_stated_rate():
    pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/irt"))
    from run_pairs import VOPTIMAL, conformal_intervals

    rng = np.random.default_rng(4)
    n_pairs = 400
    truth = rng.normal(0, 0.02, n_pairs)
    records = pd.DataFrame(
        {
            "method": VOPTIMAL,
            "k": 100,
            "split": np.arange(n_pairs) % 5,
            "model_id": np.arange(n_pairs),
            "true_score": truth,
            "pred_score": truth + rng.standard_t(3, n_pairs) * 0.01,
        }
    )
    records["abs_error"] = (records.pred_score - records.true_score).abs()
    records = conformal_intervals(records)
    for level in (80, 95):
        covered = records.true_score.between(records[f"low_{level}"], records[f"high_{level}"])
        assert abs(covered.mean() - level / 100) < 0.04


def test_lineage_groups_join_links_shared_bases_and_families():
    pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/irt"))
    from efficbench_pairs import lineage_groups

    links = pd.DataFrame({"child": ["a", "b"], "parent": ["base", "base"]})
    group = lineage_groups(np.array(["a", "b", "c", "d"]), np.array([1, 2, 2, 3]), links)
    assert group["a"] == group["b"] == group["c"]
    assert group["d"] != group["a"]


def test_pick_run_skips_limited_and_incomplete_runs():
    pytest.importorskip("pyarrow")
    pytest.importorskip("huggingface_hub")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/irt"))
    from efficbench_parents import N_SUBJECTS, pick_run

    def run(name, n_files, size):
        return [(f"{name}/hendrycksTest-{i}.parquet", size) for i in range(n_files)]

    files = run("1-full", N_SUBJECTS, 400_000) + run("2-limited", N_SUBJECTS, 10_000)
    files += run("3-partial", 5, 400_000)
    assert {path.split("/")[0] for path in pick_run(files)} == {"1-full"}
    assert pick_run(run("only-limited", N_SUBJECTS, 10_000)) == []
