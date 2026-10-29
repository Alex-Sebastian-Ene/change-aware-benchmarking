"""Tests for src/cabench/adaptive_irt.py (Fluid Benchmarking).

Run from the repository root:
    uv run --group irt pytest tests/test_adaptive_irt.py 
"""

import numpy as np
import pytest
from scipy.special import expit

pytest.importorskip("torch")

from cabench.adaptive_irt import (
    A_MIN,
    ability_estimate,
    fisher_information,
    fit_2pl,
    run_fluid,
    select_mfi,
)


def simulate(theta, n_items=2000, seed=0):
    """Random items (a, b) and one model's 0/1 answers to them, generated from ability theta"""
    rng = np.random.default_rng(seed)
    a, b = rng.lognormal(0, 0.3, n_items), rng.normal(size=n_items)
    y = (rng.random(n_items) < expit(a * (theta - b))).astype(float)
    return y, a, b


# fisher_information
# Tests: where one item (a = 1.5, b = 0.7) is most informative
# Expected: the information peaks at theta = b = 0.7
def test_fisher_information_peaks_at_difficulty():
    grid = np.linspace(-4, 4, 801)
    info = fisher_information(grid, a=1.5, b=0.7)
    assert np.isclose(grid[info.argmax()], 0.7)
    assert np.isclose(info.max(), 1.5**2 / 4)


# ability_estimate
# Tests: answers simulated from a known theta
# Expected: the estimate is within 0.15 of the theta that generated the answers
def test_ability_estimate_recovers_theta():
    for theta in [-2, -0.5, 0, 1, 2.5]:
        y, a, b = simulate(theta)
        assert abs(ability_estimate(y, a, b) - theta) < 0.15, theta


# ability_estimate
# Tests: that it finds the true MAP
# Expected: both give the same theta, within 0.001
def test_map_matches_grid_search():
    y, a, b = simulate(1.0, n_items=30)
    grid = np.linspace(-4, 4, 80_001)
    p = expit(a * (grid[:, None] - b))
    log_post = (y * np.log(p) + (1 - y) * np.log1p(-p)).sum(1) - grid**2 / 2
    assert abs(ability_estimate(y, a, b) - grid[log_post.argmax()]) < 1e-3


# ability_estimate
# Tests: the same answers, with the search started at theta0 = -3, 0 and 3
# Expected: the same estimate every time (within 1e-5), because the log-posterior has one peak
def test_ability_estimate_same_from_any_start():
    y, a, b = simulate(0.8, n_items=100)
    estimates = [ability_estimate(y, a, b, theta0=start) for start in (-3.0, 0.0, 3.0)]
    assert np.ptp(estimates) < 1e-5


# ability_estimate
# Tests: 200 items where the last 100 answers are NaN ("not asked")
# Expected: exactly the same estimate as passing only the first 100 items (NaN items are ignored)
def test_unanswered_items_are_ignored():
    y, a, b = simulate(0.5, n_items=200)
    masked = y.copy()
    masked[100:] = np.nan
    assert ability_estimate(masked, a, b) == ability_estimate(y[:100], a[:100], b[:100])


# ability_estimate
# Tests: extreme answer patterns
# Expected:
#   - no answers at all -> 0.0, the prior mean
#   - all 50 correct / all 50 wrong -> within the (-4, 4) range
#   - 5,000 correct answers on very hard items (b = 4) -> the peak lies beyond the range,
#                                      so the estimate is the upper limit, exactly 4
def test_edge_cases():
    _, a, b = simulate(0.0, n_items=50)
    assert ability_estimate(np.full(50, np.nan), a, b) == 0.0
    assert 0 < ability_estimate(np.ones(50), a, b) < 4
    assert -4 < ability_estimate(np.zeros(50), a, b) < 0
    assert ability_estimate(np.ones(5000), np.full(5000, 3.0), np.full(5000, 4.0)) == 4.0


# _bisection (the fallback when Newton's method fails)
# Tests: the same answers solved by both methods
# Expected: the same estimate, within 1e-4
def test_bisection_fallback_matches_newton():
    from cabench.adaptive_irt import _bisection, _newton
    y, a, b = simulate(0.7, n_items=100)
    assert abs(_bisection(y, a, b) - _newton(0.0, y, a, b)) < 1e-4


# select_mfi
# Tests: 9 items with the same a = 1 and b = -2, -1.5, ..., 2, at theta = 0.6
# Expected:
#   - item 5 (b = 0.5), the closest difficulty to theta
#   - with item 5 marked as used: item 6 (b = 1.0), the next closest
#   - with every item used: RuntimeError
def test_select_mfi_picks_closest_unused_difficulty():
    a, b = np.ones(9), np.linspace(-2, 2, 9)
    used = np.zeros(9, dtype=bool)
    assert select_mfi(0.6, a, b, used) == 5
    used[5] = True
    assert select_mfi(0.6, a, b, used) == 6
    with pytest.raises(RuntimeError):
        select_mfi(0.6, a, b, np.ones(9, dtype=bool))


# select_mfi
# Tests: that discrimination a counts, not only the distance between b and theta
# Expected:
#   - two items with the same b = 0.5 (a = 1 and 2), theta = 0.5 -> item 1
#   - item 0 (a = 1, b = 0) vs item 1 (a = 3, b = 1), theta = 0 -> item 1
def test_select_mfi_prefers_higher_discrimination():
    unused = np.zeros(2, dtype=bool)
    assert select_mfi(0.5, np.array([1.0, 2.0]), np.array([0.5, 0.5]), unused) == 1
    assert select_mfi(0.0, np.array([1.0, 3.0]), np.array([0.0, 1.0]), unused) == 1


# run_fluid
# Tests: one 50-question run for a model with theta = 1, out of 300 items
# Expected:
#   - 50 abilities and 50 items, with no item asked twice
#   - theta goes up after every correct answer and down after every wrong one
#   - every model gets the same first question, no matter its answers
def test_run_fluid():
    y, a, b = simulate(1.0, n_items=300)
    abilities, items = run_fluid(y, a, b, start_ability=0.0, n_max=50)
    assert len(abilities) == len(items) == 50 == len(set(items))
    steps = np.diff(np.r_[0.0, abilities])
    assert (np.sign(steps) == 2 * y[items] - 1).all()
    assert run_fluid(1 - y, a, b, start_ability=0.0, n_max=1)[1] == items[:1]


# run_fluid
# Tests: a 40-question run, replayed step by step outside run_fluid
# Expected, at every step k:
#   - items[k] is what select_mfi picks at the previous theta (start_ability for the first item)
#   - abilities[k] equals ability_estimate on the first k + 1 answers (within 1e-5)
#   The script relies on this: it reads the estimate for budget k from abilities[k - 1].
def test_run_fluid_matches_step_by_step_replay():
    y, a, b = simulate(0.5, n_items=300)
    abilities, items = run_fluid(y, a, b, start_ability=0.0, n_max=40)
    used = np.zeros(len(a), dtype=bool)
    previous = 0.0  # start_ability
    for k, idx in enumerate(items):
        assert idx == select_mfi(previous, a, b, used)
        used[idx] = True
        answered = np.where(used, y, np.nan)
        assert np.isclose(abilities[k], ability_estimate(answered, a, b), atol=1e-5)
        previous = abilities[k]


# run_fluid
# Tests: 40 items, of which items 0-9 are clamped to a = A_MIN
# Expected:
#   - a 30-question run asks exactly the 30 real items (10-39) and none of the clamped ones,
#     because clamped items carry almost no information
#   - a run with n_max = 100 stops after 40 questions, when every item has been asked
def test_run_fluid_never_picks_clamped_items():
    y, a, b = simulate(0.0, n_items=40)
    a[:10] = A_MIN
    _, items = run_fluid(y, a, b, start_ability=0.0, n_max=30)
    assert set(items) == set(range(10, 40))
    _, all_items = run_fluid(y, a, b, start_ability=0.0, n_max=100)
    assert len(all_items) == 40


# fit_2pl
# Tests: a 300 models x 500 items matrix simulated from known theta, a and b
# Expected:
#   - the fitted theta has mean 0 and sd 1 (standardised)
#   - the fitted theta correlates > 0.9 with the true theta
#   - every a >= A_MIN and every b is finite (no NaN)
#   - over 95% of items are not clamped, and their b correlates > 0.9 with the true b
def test_fit_2pl_recovers_simulated_items():
    rng = np.random.default_rng(0)
    theta = rng.normal(size=300)
    a, b = rng.lognormal(0, 0.3, 500), rng.normal(size=500)
    Y = (rng.random((300, 500)) < expit(a * (theta[:, None] - b))).astype(np.float32)
    a_hat, b_hat, theta_hat = fit_2pl(Y)
    assert np.isclose(theta_hat.mean(), 0, atol=1e-5) and np.isclose(theta_hat.std(), 1)
    assert np.corrcoef(theta_hat, theta)[0, 1] > 0.9
    assert (a_hat >= A_MIN).all() and np.isfinite(b_hat).all()
    ok = a_hat > A_MIN  # items that were not clamped
    assert ok.mean() > 0.95
    assert np.corrcoef(b_hat[ok], b[ok])[0, 1] > 0.9


# fit_2pl
# Tests: a simulated matrix where item 0 is reversed: only below-average models get it right. The fit gives such an item a negative a
# Expected: item 0 is clamped to exactly A_MIN, and its b is still a finite number (not NaN)
def test_fit_2pl_clamps_reversed_items():
    rng = np.random.default_rng(1)
    theta = rng.normal(size=200)
    a, b = rng.lognormal(0, 0.3, 100), rng.normal(size=100)
    Y = (rng.random((200, 100)) < expit(a * (theta[:, None] - b))).astype(np.float32)
    Y[:, 0] = theta < 0
    a_hat, b_hat, _ = fit_2pl(Y)
    assert a_hat[0] == A_MIN and np.isfinite(b_hat[0])
