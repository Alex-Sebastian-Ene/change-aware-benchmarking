"""IRT model fit (start with 1PL/2PL) and static / adaptive item subset selection.

Core objective 3. Owners: A. Vijaykrishnan, A. Chitnis.

Multidimensional 2PL IRT: P(Y_mi = 1) = sigmoid(theta_m . alpha_i - beta_i), fitted by
L2-regularised maximum likelihood (not the hierarchical Bayesian VI used by tinyBenchmarks).
Static subset: IRT anchors. Score estimators: anchor-weighted accuracy, p-IRT and gp-IRT.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def fit_mirt(Y, n_dims=2, epochs=2000, lr=0.05, l2=1e-3, seed=0):
    """Fit on a complete 0/1 matrix Y (models x items).

    Returns theta (M, D), alpha (Q, D), beta (Q,) and the training loss per epoch.
    """
    Y = torch.as_tensor(Y, dtype=torch.float32)
    assert not torch.isnan(Y).any(), "fit_mirt expects a complete matrix"
    torch.manual_seed(seed)
    n_models, n_items = Y.shape
    theta = (torch.randn(n_models, n_dims) * 0.1).requires_grad_()
    alpha = (torch.randn(n_items, n_dims) * 0.1).requires_grad_()
    beta = torch.zeros(n_items, requires_grad=True)
    opt = torch.optim.Adam([theta, alpha, beta], lr=lr)

    history = []
    for _ in range(epochs):
        opt.zero_grad()
        nll = F.binary_cross_entropy_with_logits(theta @ alpha.T - beta, Y)
        reg = l2 * (theta.pow(2).mean() + alpha.pow(2).mean() + beta.pow(2).mean())
        (nll + reg).backward()
        opt.step()
        history.append(nll.item())
    return theta.detach().numpy(), alpha.detach().numpy(), beta.detach().numpy(), history


def fit_theta(Y_obs, mask, alpha, beta, theta_prior, steps=500, lr=0.05):
    """MAP theta for new models with items frozen, using only entries where mask is set.

    The prior is a Gaussian fitted to the training thetas, so a model seen on few items
    stays near typical ones.
    """
    Y_obs = torch.as_tensor(np.nan_to_num(Y_obs), dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.float32)
    alpha, beta = torch.as_tensor(alpha), torch.as_tensor(beta)
    mu = torch.as_tensor(theta_prior.mean(0))
    prior_cov = np.atleast_2d(np.cov(theta_prior.T))
    precision = torch.as_tensor(np.linalg.inv(prior_cov), dtype=torch.float32)

    theta = mu.repeat(len(Y_obs), 1).requires_grad_()
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = theta @ alpha.T - beta
        nll = (F.binary_cross_entropy_with_logits(logits, Y_obs, reduction="none") * mask).sum()
        d = theta - mu
        (nll + 0.5 * ((d @ precision) * d).sum()).backward()
        opt.step()
    return theta.detach().numpy()


def predict_proba(theta, alpha, beta):
    return 1 / (1 + np.exp(-(theta @ alpha.T - beta)))


def irt_anchors(embeddings, k, seed):
    """One anchor per K-means cluster (the item nearest the centroid).

    Each anchor is weighted by its cluster's share of items. `embeddings` is usually the
    standardised [alpha, beta] of every item.
    """
    km = KMeans(n_clusters=k, random_state=seed).fit(embeddings)
    anchors, _ = pairwise_distances_argmin_min(km.cluster_centers_, embeddings)
    return anchors, np.bincount(km.labels_, minlength=k) / len(embeddings)


def anchor_estimate(Y, anchors, weights):
    return Y[:, anchors] @ weights


def pirt_estimate(Y, seen, alpha, beta, theta_train):
    """Observed answers where `seen` (bool, same shape as Y), IRT-predicted P(correct)
    elsewhere, averaged per model."""
    theta = fit_theta(Y, seen, alpha, beta, theta_prior=theta_train)
    return np.where(seen, Y, predict_proba(theta, alpha, beta)).mean(1)


def gp_weight(err_anchor, err_pirt):
    """Weight on the anchor estimate that minimises the squared error of the blend."""
    p, c = np.mean(err_pirt**2), np.mean(err_anchor * err_pirt)
    d = np.mean((err_anchor - err_pirt) ** 2)
    return float(np.clip((p - c) / d, 0, 1)) if d else 0.5


def gp_estimate(anchor, pirt, lam):
    return lam * anchor + (1 - lam) * pirt
