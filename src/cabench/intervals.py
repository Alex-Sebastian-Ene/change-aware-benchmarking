"""Prediction intervals for scores and score differences. Core objective 4. Owner: S. Ene.

conformal_interval  width from a method's errors on held-out models (rank rule), optionally
                    scaled per model by score_sd. Valid while new models resemble those.
audit               a method's error on one model, from questions drawn at random among those
                    it did not ask: unbiased for any model. With predictions of 0 it is plain
                    random sampling.
drift_interval      audits pooled over helper models estimate the error a method shares on a
                    batch (e.g. IRT on frontier models) and correct the rest of the batch.
paired_estimates    scores and pairwise differences from the same questions drawn with known
                    probabilities, with a decision per pair; unbiased with no model at all.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


def conformal_quantile(scores, level):
    """The ceil((n + 1) * level / 100)-th smallest score; inf if n is too small for the level."""
    s = np.sort(np.asarray(scores, dtype=float).ravel())
    rank = int(np.ceil((len(s) + 1) * level / 100))
    return s[rank - 1] if rank <= len(s) else np.inf


def conformal_interval(estimate, calib_estimate, calib_true, level, scale=1.0, calib_scale=1.0):
    """estimate +- q * scale, q the conformal quantile of |calibration error| / calib_scale.

    The calibration models must be held out from whatever fitted the method.
    """
    err = np.abs(np.asarray(calib_true, dtype=float) - calib_estimate) / calib_scale
    half = conformal_quantile(err, level) * np.asarray(scale, dtype=float)
    return estimate - half, estimate + half


def score_sd(theta, alpha, beta, seen, prior_cov):
    """Spread of an IRT score estimate under a Laplace ability posterior.

    The score is the answer where seen, sigmoid(theta . alpha - beta) elsewhere, averaged over
    items. theta (models, D) are MAP abilities under a Gaussian prior with covariance prior_cov;
    alpha (items, D); seen (models, items) bool. Counts question luck only, so it is
    overconfident on its own: use it as scale in conformal_interval.
    """
    w = expit(theta @ alpha.T - beta)
    w = w * (1 - w)
    info = np.einsum("mq,qd,qe->mde", w * seen, alpha, alpha) + np.linalg.inv(prior_cov)
    unseen = w * ~seen
    g = unseen @ alpha / len(beta)  # d score / d theta
    var = np.einsum("md,mde,me->m", g, np.linalg.inv(info), g) + unseen.sum(1) / len(beta) ** 2
    return np.sqrt(var)


def audit(residuals, n_unseen, pseudo=0.25):
    """Mean audit residual per model and the variance of that mean.

    residuals (models, k): answer minus predicted probability on k questions drawn uniformly
    without replacement from the model's n_unseen unasked ones. The mean is unbiased for the
    method's average error over all of them, for any model. The variance has the
    finite-population correction plus `pseudo` of extra spread, so it never reaches 0.
    """
    r = np.atleast_2d(np.asarray(residuals, dtype=float))
    k = r.shape[1]
    ss = ((r - r.mean(1, keepdims=True)) ** 2).sum(1) + pseudo
    return r.mean(1), ss / ((k - 1) * k) * (1 - k / np.asarray(n_unseen, dtype=float))


def drift_interval(score, sd, unseen_share, helper_means, helper_vars, tau2_floor, level=95):
    """Correct a batch's estimates by the error the method shares on it; (estimate, low, high).

    A model's error is a shared bias plus its own deviation of variance tau2. The bias is the
    mean of the helpers' audit means; tau2 is their spread minus audit noise. That difference
    is unreliable (audit noise usually swamps it, intervals come out too narrow), so tau2 is at
    least tau2_floor: set it to the variance of the method's errors on held-out in-range models.
    """
    m = np.asarray(helper_means, dtype=float)
    spread = m.var(ddof=1)
    tau2 = max(spread - np.mean(helper_vars), tau2_floor)
    est = score + unseen_share * m.mean()
    half = norm.ppf(0.5 + level / 200) * np.sqrt(sd**2 + unseen_share**2 * (tau2 + spread / len(m)))
    return est, est - half, est + half


def sampling_probabilities(n_items, guide=None, uniform_share=0.2):
    """uniform_share spread evenly and the rest in proportion to guide (None: uniform).

    Every question keeps a chance, so a bad guide costs precision, never validity.
    """
    if guide is None:
        return np.full(n_items, 1 / n_items)
    if not 0 < uniform_share <= 1:
        raise ValueError("uniform_share must be in (0, 1]: every question needs a chance")
    guide = np.asarray(guide, dtype=float)
    return uniform_share / n_items + (1 - uniform_share) * guide / guide.sum()


def paired_estimates(answers, items, q, level=95, tie_margin=0.01, pseudo=True):
    """Scores and pairwise differences from shared draws, with standard errors and decisions.

    answers (models, k): every model's answers to the same k questions `items`, drawn with
    replacement with probabilities q. Each draw adds answer / (N q_i), whose mean is the full
    score exactly (Hansen-Hurwitz); shared draws cancel the questions models answer alike.
    decision per pair: "tie" if the interval sits within +-tie_margin, else "a" or "b" if that
    model is better at `level` (per pair; for a whole batch use 100 - (100 - level) / n_pairs),
    else "undecided". pseudo adds a quarter draw of spread (as audit does) so models that have
    agreed on every draw so far keep a width; a full draw over-covers under guided sampling.
    Returns (scores, pairs) DataFrames; models are numbered by row.
    """
    w = 1 / (len(q) * q[items])
    t = np.atleast_2d(np.asarray(answers, dtype=float)) * w
    k, cov = t.shape[1], np.atleast_2d(np.cov(t))
    floor = 0.25 * w.mean() ** 2 if pseudo else 0.0

    def se(var):
        return np.sqrt(((k - 1) * var + floor) / ((k - 1) * k))

    a, b = np.triu_indices(len(t), 1)
    est = t.mean(1)
    diff, diff_se = est[a] - est[b], se(cov[a, a] + cov[b, b] - 2 * cov[a, b])
    low = diff - norm.ppf(0.5 + level / 200) * diff_se
    high = 2 * diff - low
    tie, better, worse = (low > -tie_margin) & (high < tie_margin), low > 0, high < 0
    decision = np.select([tie, better, worse], ["tie", "a", "b"], "undecided")
    scores = pd.DataFrame({"estimate": est, "se": se(np.diag(cov))})
    pairs = pd.DataFrame({"a": a, "b": b, "diff": diff, "se": diff_se, "decision": decision})
    return scores, pairs


def rank_intervals(n_models, pairs):
    """Best and worst possible rank (1 = best) per model, from the pairs' decisions."""
    won, lost = pairs.decision == "a", pairs.decision == "b"
    winners = np.r_[pairs.a[won], pairs.b[lost]].astype(int)
    losers = np.r_[pairs.b[won], pairs.a[lost]].astype(int)
    return pd.DataFrame(
        {
            "best_rank": 1 + np.bincount(losers, minlength=n_models),
            "worst_rank": n_models - np.bincount(winners, minlength=n_models),
        }
    )
