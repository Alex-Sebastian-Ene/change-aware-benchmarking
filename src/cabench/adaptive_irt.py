"""Adaptive IRT benchmarking, following Fluid Benchmarking
(Hofmann et al., 2025, https://github.com/allenai/fluid-benchmarking).

This is the adaptive counterpart of irt.py (static subsets)
Contains the following functions:
    fit_2pl           fit a, b (and training thetas) on known models - gets forzen after computed
    run_fluid         evaluate one new model adaptively, up to n_max items:
      select_mfi        next item = most Fisher information at the current theta
      ability_estimate  MAP theta (N(0, 1) prior) from the answers so far

Everything except fit_2pl (see todo at the end of file) follows the Fluid repo 
    (fluid_benchmarking/engine.py and estimators.py). 

fit_2pl reuses irt.fit_mirt currently, TODO Fluid's version
The response matrix comes from scripts/irt/efficbench.py 
    and the train/test model split from scripts/irt/protocol.py

"""

import numpy as np
from scipy.special import expit  # numerically stable sigmoid, 1 / (1 + exp(-x))

from cabench.irt import fit_mirt

A_MIN = 1e-3  # replace a where a<=0 with this


def fit_2pl(Y, epochs=2000, lr=0.05, l2=1e-3, seed=0):
    """Fit a, b on a complete 0/1 matrix Y (models x items) of known models.
    Returns a (n_items,), b (n_items,) and theta (n_models,) of the training models.
    """
    theta, alpha, beta, _ = fit_mirt(Y, n_dims=1, epochs=epochs, lr=lr, l2=l2, seed=seed)
    theta, alpha = theta[:, 0], alpha[:, 0] # flatten for 1D

    # check whether theta's orientation is as expected
    # negate if model accuracy is negatively correlated with theta
    if np.corrcoef(theta, np.asarray(Y).mean(1))[0, 1] < 0:
        theta, alpha = -theta, -alpha

    # standardise theta and calculate a and b accordingly
    # theta = m + s * z gives theta * alpha - beta = z * (s * alpha) - (beta - m * alpha)
    m, s = theta.mean(), theta.std()
    theta = (theta - m) / s
    alpha, beta = s * alpha, beta - m * alpha

    # convert to Fluid's (a, b) form
    # Fluid's fit forces a > 0. As a stand-in, items with a <=0 are replaced with a small epsilon
    #   these carry no information and won't be picked 
    a = np.maximum(alpha, A_MIN)
    b = beta / a
    return a, b, theta


def fisher_information(theta, a, b, D=1.0):
    """Fisher Information of each item at ability theta: D^2 a^2 p (1 - p).
    """
    p = expit(D * a * (theta - b))  # P(correct) for each item at this theta
    return (D**2) * (a**2) * (p * (1.0 - p))


PRIOR_MEAN = 0.0
PRIOR_SD = 1.0
THETA_MIN, THETA_MAX = -4.0, 4.0  # theta is always kept inside this range
TOL = 1e-6  # theta counts as found once |score| < TOL
MAX_NEWTON_STEPS = 100
MAX_HALVINGS = 15  # per Newton step, if the step overshoots
MAX_BISECTION_STEPS = 80

def ability_estimate(responses, a, b, theta0=PRIOR_MEAN):
    """MAP estimate of one model's ability theta from its 0/1 responses.

    responses  one entry per item: 1 correct, 0 wrong, NaN not asked (ignored)
    a, b       frozen item parameters, same length as responses
    theta0     where the search starts; the adaptive loop passes the previous estimate
    """
    # keep only the items this model has actually answered
    responses = np.asarray(responses, dtype=float)
    seen = ~np.isnan(responses)
    y, a, b = responses[seen], np.asarray(a, dtype=float)[seen], np.asarray(b, dtype=float)[seen]

    theta = _newton(theta0, y, a, b)
    return theta if theta is not None else _bisection(y, a, b)


def _score(theta, y, a, b):
    """First derivative of the log-posterior at theta.
    score > 0: theta is too low (the model got more right than theta predicts), go up
    score < 0: theta is too high, go down
    """
    p = expit(a * (theta - b))  # predicted P(correct) for each answered item
    prior_term = (PRIOR_MEAN - theta) / PRIOR_SD**2  # pulls theta back towards the prior mean
    likelihood_term = np.sum(a * (y - p))  # actual minus expected, weighted by discrimination
    return prior_term + likelihood_term


def _score_prime(theta, y, a, b):
    """Second derivative of the log-posterior at theta.
    """
    p = expit(a * (theta - b))
    prior_term = -1 / PRIOR_SD**2
    likelihood_term = -np.sum(a**2 * p * (1 - p))
    return prior_term + likelihood_term


def _newton(theta0, y, a, b):
    """Find where the score is zero with Newton steps. None if it does not converge.
    """
    theta = float(np.clip(theta0, THETA_MIN, THETA_MAX))
    for _ in range(MAX_NEWTON_STEPS):
        score = _score(theta, y, a, b)
        if abs(score) < TOL:
            return theta  # converged
        
        score_prime = _score_prime(theta, y, a, b)
        if not np.isfinite(score_prime) or score_prime == 0:
            return None  # Newton cannot continue

        # Newton step (jump to where the score would be zero if it changed at a constant rate)
        new = float(np.clip(theta - score / score_prime, THETA_MIN, THETA_MAX))

        # if the step overshoots (|score| does not shrink), halve it
        for _ in range(MAX_HALVINGS):
            new_score = _score(new, y, a, b)
            if abs(new_score) < abs(score) or not np.isfinite(new_score):
                break
            new = 0.5 * (new + theta)
        theta = new
    return None


def _bisection(y, a, b):
    """Fallback: halve [THETA_MIN, THETA_MAX] until the zero-score point is found.
    """
    lo, hi = THETA_MIN, THETA_MAX
    score_lo, score_hi = _score(lo, y, a, b), _score(hi, y, a, b)
    if score_lo * score_hi > 0:
        if score_lo > 0:
            return hi
        else: 
            return lo

    for _ in range(MAX_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        score_mid = _score(mid, y, a, b)
        if abs(score_mid) < TOL:
            return mid
        if score_lo * score_mid > 0:  # same sign as the low end, so the zero is in [mid, hi]
            lo, score_lo = mid, score_mid
        else:  # the zero is in [lo, mid]
            hi = mid
    return 0.5 * (lo + hi)


def select_mfi(theta, a, b, used):
    """Index of the unused item with Maximum Fisher Information (MFI) at theta.

    used : boolean mask of items already asked (these may not be picked again).
    """
    info = fisher_information(theta, a, b)
    info = np.where(used, -np.inf, info)  # used items can never win
    idx = int(np.argmax(info))
    if not np.isfinite(info[idx]):
        raise RuntimeError("no items left to select")
    return idx


def run_fluid(responses, a, b, start_ability, n_max):
    """Fluid Benchmarking for one model: ask up to n_max items adaptively.

    responses      the model's full 0/1 answers on every item
    a, b           frozen item parameters from fit_2pl
    start_ability  theta used to pick the first item (Fluid: the mean theta of the
                   training models), so every model gets the same first question

    Returns (abilities, items): abilities[k - 1] is the theta estimate after k items, and
    items lists the asked item indices in order. One run up to n_max therefore gives the
    result for every smaller budget too.
    """
    responses = np.asarray(responses, dtype=float)
    n_items = len(a)
    used = np.zeros(n_items, dtype=bool)  # items asked so far
    answered = np.full(n_items, np.nan)  # answers revealed so far. NaN = not asked yet
    n_steps = min(n_max, n_items)  # stop early if every item has been asked

    abilities, items = [], []
    theta = start_ability
    for _ in range(n_steps):
        # 1. pick the most informative item at the current ability estimate
        idx = select_mfi(theta, a, b, used)

        # 2. retrieve the model's answer
        r = responses[idx] 
        if r not in (0.0, 1.0):
            raise ValueError(f"response for item {idx} must be 0 or 1, got {r}")
       
        used[idx] = True
        answered[idx] = r

        # 3. re-estimate theta from every answer so far, starting from the previous estimate.
        # A correct answer moves theta up (so the next item is harder), a wrong one down.
        theta = ability_estimate(answered, a, b, theta0=theta)
        items.append(idx)
        abilities.append(theta)
    return abilities, items


# TODO: replace fit_2pl with Fluid's own fit (which is a hierarchical Bayesian 2PL with py-irt)
# Before swapping, check py-irt's training defaults (optimizer, learning rate and decay, how
# best_params is picked) - this is not in the Fluid repo
# Knock-on changes:
#   - a > 0 is guaranteed, so the A_MIN clamp in fit_2pl never triggers: remove it and A_MIN.
#   - The orientation flip in fit_2pl is no longer needed (a > 0 fixes the direction).
#   - The theta standardisation in fit_2pl is ours, not Fluid's: decide whether to keep it.
#     Fluid uses the N(0, 1) prior without rescaling its fitted thetas.
#   - Import Pyro instead of irt.fit_mirt, and add pyro-ppl to the irt dependency group.
#   - Rerun the experiments; results from the two fits are not interchangeable.
