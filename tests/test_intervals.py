import itertools

import numpy as np
import pytest
from scipy.special import expit

from cabench.intervals import (
    audit,
    conformal_interval,
    conformal_quantile,
    drift_interval,
    paired_estimates,
    rank_intervals,
    sampling_probabilities,
    score_sd,
)


def test_conformal_rank_rule_and_coverage():
    errors = [0.5, 0.8, 1.1, 1.4, 1.9, 2.2, 2.6, 3.4, 5.0]
    assert conformal_quantile(errors, 80) == 3.4  # the 8th smallest of 9
    assert conformal_quantile(errors, 95) == np.inf  # 9 models are too few for 95%
    rng = np.random.default_rng(1)
    hits = []
    for _ in range(4000):
        e = rng.standard_t(3, 41)  # heavy tails: no normality needed
        low, high = conformal_interval(0.0, np.zeros(40), e[:40], 90)
        hits.append(low <= e[40] <= high)
    assert 89 <= np.mean(hits) * 100 <= 93.5  # at least 90, at most 90 + 100/41, +-noise


def test_score_sd_matches_monte_carlo():
    rng = np.random.default_rng(3)
    alpha, beta = rng.normal(0, 1, (300, 2)), rng.normal(0, 1, 300)
    theta, prior = np.array([[0.3, -0.2]]), np.eye(2) * 0.5
    seen = np.zeros((1, 300), bool)
    seen[0, :20] = True
    w = expit(theta @ alpha.T - beta)[0, :20]
    cov = np.linalg.inv((alpha[:20].T * w * (1 - w)) @ alpha[:20] + np.linalg.inv(prior))
    p = expit(rng.multivariate_normal(theta[0], cov, 20_000) @ alpha.T - beta)[:, 20:]
    scores = (rng.random(p.shape) < p).sum(1) / 300  # seen answers fixed, so they add no spread
    assert score_sd(theta, alpha, beta, seen, prior)[0] == pytest.approx(scores.std(), rel=0.05)


def test_audit_is_exactly_unbiased():
    resid = np.array([0.4, -0.3, 0.1, 0.9, -0.6, 0.2, 0.0])
    every_audit = itertools.combinations(range(7), 3)
    means, variances = zip(*(audit(resid[list(i)], 7, pseudo=0) for i in every_audit), strict=True)
    assert np.mean(means) == pytest.approx(resid.mean())
    assert np.mean(variances) == pytest.approx(np.var(means))


def test_drift_interval_removes_a_shared_bias_and_covers():
    rng = np.random.default_rng(7)

    def batch(n, shared):  # predictions too low by a shared amount plus each model's own
        p = rng.uniform(0.3, 0.9, (n, 2000))
        pred = np.clip(p - shared - rng.normal(0, 0.01, (n, 1)), 0.01, 0.99)
        y = (rng.random(p.shape) < p).astype(float)
        return y - pred, pred.mean(1), y.mean(1)

    left, coverage = [], []
    for _ in range(20):  # a batch shares one bias estimate, so judge over many batches
        resid, pred, truth = batch(240, 0.08)
        _, in_pred, in_truth = batch(100, 0.0)  # in-range models set the tau2 floor
        asked = np.array([rng.choice(2000, 100, replace=False) for _ in range(40)])
        means, variances = audit(np.take_along_axis(resid[:40], asked, 1), 2000)
        floor = np.var(in_truth - in_pred)
        est, low, high = drift_interval(pred[40:], 0.0, 1.0, means, variances, floor)
        left.append((truth[40:] - est).mean())
        coverage.append(np.mean((low <= truth[40:]) & (truth[40:] <= high)))
    assert abs(np.mean(left)) < 0.003  # the 0.08 bias is gone
    assert 90 <= np.mean(coverage) * 100 <= 99


def test_paired_estimates_are_exactly_unbiased():
    y = np.array([[1, 0, 1, 1, 0], [1, 1, 0, 1, 0], [0, 0, 1, 1, 1]], float)
    q = np.array([0.1, 0.3, 0.2, 0.15, 0.25])
    a, b = np.triu_indices(3, 1)
    true_diff = y.mean(1)[a] - y.mean(1)[b]
    scores, diffs, variances, errors = 0, 0, 0, 0
    for i, j in itertools.product(range(5), repeat=2):  # every pair of draws, by probability
        s, pairs = paired_estimates(y[:, [i, j]], np.array([i, j]), q, pseudo=False)
        w = q[i] * q[j]
        scores = scores + w * s.estimate.to_numpy()
        diffs = diffs + w * pairs["diff"].to_numpy()
        variances = variances + w * pairs.se.to_numpy() ** 2
        errors = errors + w * (pairs["diff"].to_numpy() - true_diff) ** 2
    assert scores == pytest.approx(y.mean(1))
    assert diffs == pytest.approx(true_diff)
    assert variances == pytest.approx(errors)  # standard errors are right on average too


def test_paired_differences_cover_and_a_good_guide_narrows_them():
    rng = np.random.default_rng(11)
    base = (rng.random(3000) < 0.7).astype(float)
    hard = np.arange(3000) < 600  # the two models only differ here
    y = np.vstack([base, np.where(hard & (rng.random(3000) < 0.5), 1 - base, base)])
    true_diff = y[0].mean() - y[1].mean()
    mean_se = {}
    for name, q in (
        ("uniform", sampling_probabilities(3000)),
        ("guided", sampling_probabilities(3000, hard)),
    ):
        hits, ses = [], []
        for _ in range(1000):
            items = rng.choice(3000, 300, p=q)
            pairs = paired_estimates(y[:, items], items, q)[1]
            hits.append(abs(pairs["diff"][0] - true_diff) <= 1.96 * pairs.se[0])
            ses.append(pairs.se[0])
        assert 93 <= np.mean(hits) * 100 <= 97
        mean_se[name] = np.mean(ses)
    assert mean_se["guided"] < 0.7 * mean_se["uniform"]


def test_sampling_probabilities_keep_every_question_drawable():
    q = sampling_probabilities(4, [0, 1, 1, 2])
    assert q.sum() == pytest.approx(1)
    assert q[0] == pytest.approx(0.05)  # zero guide weight keeps its uniform share
    with pytest.raises(ValueError):
        sampling_probabilities(4, [0, 1, 1, 2], uniform_share=0)


def test_decisions_and_rank_intervals():
    rng = np.random.default_rng(5)
    y = np.vstack([np.ones(1000), rng.random(1000) < 0.5, np.zeros(1000), np.zeros(1000)])
    q = sampling_probabilities(1000)
    items = rng.choice(1000, 2000, p=q)
    pairs = paired_estimates(y[:, items].astype(float), items, q)[1]
    decision = pairs.set_index(["a", "b"]).decision
    assert decision[(0, 1)] == decision[(1, 2)] == "a"
    assert decision[(2, 3)] == "tie"  # identical models
    ranks = rank_intervals(4, pairs)
    assert ranks.best_rank.tolist() == [1, 2, 3, 3]
    assert ranks.worst_rank.tolist() == [1, 2, 4, 4]
