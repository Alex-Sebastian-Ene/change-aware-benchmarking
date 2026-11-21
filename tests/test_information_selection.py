import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from cabench.irt import (  # noqa: E402
    adaptive_items,
    fisher_information,
    fisher_static,
    fit_theta,
    infer_theta,
    theta_covariance,
    voptimal_gain,
    voptimal_static,
)


def test_fisher_uses_intercept_and_training_average():
    theta = np.array([[-2.0], [0.0], [1.0]])
    alpha, beta = np.array([[2.0], [-3.0], [0.0]]), np.array([2.0, 1.0, 0.0])
    p = expit(theta @ alpha.T - beta)
    expected = alpha[:, 0] ** 2 * p * (1 - p)
    np.testing.assert_allclose(fisher_information(theta, alpha, beta), expected)
    np.testing.assert_array_equal(
        fisher_static(theta, alpha, beta, 2), np.argsort(-expected.mean(0))[:2]
    )
    with pytest.raises(ValueError, match="1D"):
        fisher_information(np.zeros((1, 2)), np.ones((3, 2)), beta)


@pytest.mark.parametrize("dims", [1, 3, 8])
def test_newton_map_matches_independent_optimizer(dims):
    rng = np.random.default_rng(dims)
    train = rng.normal(size=(40, dims))
    alpha, beta = rng.normal(size=(30, dims)), rng.normal(size=30)
    items = np.array([rng.choice(30, 12, replace=False) for _ in range(3)])
    responses = rng.binomial(1, 0.5, size=items.shape).astype(float)
    responses[:2] = np.array([[1], [0]])  # extreme prefixes must stay finite
    theta = infer_theta(responses, items, alpha, beta, train)
    precision, mu = np.linalg.inv(theta_covariance(train)), train.mean(0)
    for m, t in enumerate(theta):
        a, b, y = alpha[items[m]], beta[items[m]], responses[m]

        def objective(x, a=a, b=b, y=y):
            z = a @ x - b
            d = x - mu
            return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * d @ precision @ d

        fit = minimize(objective, mu, method="BFGS", tol=1e-8)
        np.testing.assert_allclose(t, fit.x, atol=2e-5)


def test_newton_agrees_with_baseline_adam():
    rng = np.random.default_rng(12)
    train = rng.normal(size=(100, 2)).astype(np.float32)
    alpha, beta = (
        rng.normal(size=(30, 2)).astype(np.float32),
        rng.normal(size=30).astype(np.float32),
    )
    Y = rng.binomial(1, 0.5, size=(5, 30)).astype(np.float32)
    items = np.arange(10)
    seen = np.zeros_like(Y, dtype=bool)
    seen[:, items] = True
    adam = fit_theta(Y, seen, alpha, beta, train)
    newton = infer_theta(Y[:, items], items, alpha, beta, train)
    np.testing.assert_allclose(newton, adam, atol=2e-4)


def test_voptimal_gain_equals_brute_force_covariance_reduction():
    rng = np.random.default_rng(4)
    alpha = rng.normal(size=(9, 3))
    x = rng.normal(size=(3, 3))
    cov = x @ x.T + np.eye(3)
    target = alpha.T @ alpha / len(alpha)
    weights = rng.uniform(0, 0.25, len(alpha))
    gain = voptimal_gain(alpha, weights, cov, target)
    brute = [
        np.trace(target @ cov)
        - np.trace(target @ np.linalg.inv(np.linalg.inv(cov) + w * np.outer(a, a)))
        for a, w in zip(alpha, weights, strict=True)
    ]
    np.testing.assert_allclose(gain, brute, atol=1e-12)
    batched = voptimal_gain(alpha, np.stack([weights, weights]), np.stack([cov, cov]), target)
    np.testing.assert_allclose(batched, np.stack([gain, gain]))


def test_static_voptimal_accounts_for_accumulated_information():
    # After measuring axis 1, the independent axis 2 is more useful than its duplicate.
    train = np.array([[-1.0, -1], [-1, 1], [1, -1], [1, 1]])
    alpha = np.array([[2.0, 0], [2, 0], [0, 2]])
    items = voptimal_static(train, alpha, np.zeros(3), 3)
    np.testing.assert_array_equal(items, [0, 2, 1])


@pytest.mark.parametrize("criterion,dims", [("fisher", 1), ("voptimal", 3)])
def test_adaptive_path_reads_only_selected_answers(criterion, dims):
    rng = np.random.default_rng(19)
    train = rng.normal(size=(40, dims))
    alpha, beta = rng.normal(size=(35, dims)), rng.normal(size=35)
    Y = rng.binomial(1, 0.5, (4, 35))
    path = list(adaptive_items(Y, alpha, beta, train, [3, 9], criterion))
    _, items, theta = path[-1]
    changed = 1 - Y
    np.put_along_axis(changed, items, np.take_along_axis(Y, items, axis=1), axis=1)
    rerun = list(adaptive_items(changed, alpha, beta, train, [9], criterion))[-1]
    np.testing.assert_array_equal(rerun[1], items)
    np.testing.assert_allclose(rerun[2], theta)
    np.testing.assert_array_equal(path[0][1], items[:, :3])
    short = list(adaptive_items(Y, alpha, beta, train, [3], criterion))[-1]
    np.testing.assert_array_equal(short[1], items[:, :3])
    assert all(len(set(row)) == 9 for row in items)


def test_adaptive_responds_to_answers():
    train = np.linspace(-2, 2, 100)[:, None]
    alpha, beta = np.ones((21, 1)), np.linspace(-3, 3, 21)
    Y = np.stack([np.zeros(21), np.ones(21)])
    _, items, theta = list(adaptive_items(Y, alpha, beta, train, [5]))[-1]
    assert items[0, 0] == items[1, 0]
    assert not np.array_equal(items[0, 1:], items[1, 1:])
    assert theta[0, 0] < 0 < theta[1, 0]
