"""Frozen-model adapters and information-based item design. NumPy / SciPy only.

No fitting backend is required. A model supplies fixed item loadings, prediction and
ability-inference callbacks. Information has form w_i(theta) * a_i * a_i.T.
The default w=p(1-p) is for a linear-logit 1PL/2PL/MIRT model; other links must supply
information_weights explicitly. This does not assume arbitrary IRT models share 2PL Fisher.
"""

import numpy as np
from scipy.special import expit


def model_context(
    theta_train,
    loadings,
    predict,
    infer,
    *,
    information_weights=None,
    prior_cov=None,
    embeddings=None,
    subjects=None,
    item_ids=None,
    target=None,
):
    """Adapt a teammate's frozen model without refitting or depending on its library.

    theta_train: (train models, D); loadings: (Q, D), in the SAME coordinates.
    predict(theta, items=None): probabilities (models, Q) or (models, k).
    infer(responses, items, initial=None): abilities (models, D), using only supplied answers.
    items is (k,) or (models, k); callbacks must support both forms.
    information_weights(theta, items=None): nonnegative scalar weights, same shape as predict.
    Optional target (D, D) defines the V-optimal prediction objective. Default: A.T @ A / Q.
    item_ids, when supplied, must match the response matrix column order exactly.
    """
    theta, a = np.asarray(theta_train, dtype=float), np.asarray(loadings, dtype=float)
    if theta.ndim == 1:
        theta = theta[:, None]
    if a.ndim == 1:
        a = a[:, None]
    if theta.ndim != 2 or a.ndim != 2 or theta.shape[1] != a.shape[1]:
        raise ValueError("theta_train and loadings must use the same D-dimensional coordinates")
    if not np.isfinite(theta).all() or not np.isfinite(a).all():
        raise ValueError("Abilities and loadings must be finite")
    cov = theta_covariance(theta) if prior_cov is None else np.asarray(prior_cov, dtype=float)
    if cov.shape != (a.shape[1], a.shape[1]) or not np.allclose(cov, cov.T):
        raise ValueError("prior_cov must be a symmetric D x D matrix")
    np.linalg.cholesky(cov)
    target = a.T @ a / len(a) if target is None else np.asarray(target, dtype=float)
    if target.shape != cov.shape or not np.allclose(target, target.T):
        raise ValueError("target must be a symmetric D x D matrix")
    if np.linalg.eigvalsh(target).min() < -1e-8:
        raise ValueError("target must be positive semidefinite")
    if item_ids is not None:
        item_ids = np.asarray(item_ids)
        if item_ids.shape != (len(a),) or len(np.unique(item_ids)) != len(a):
            raise ValueError("Provide one unique item_id per loading row")

    def logistic_weights(theta, items=None):
        p = np.asarray(predict(theta, items))
        return p * (1 - p)

    return {
        "theta": theta,
        "loadings": a,
        "n_dims": a.shape[1],
        "n_items": len(a),
        "predict": predict,
        "infer": infer,
        "information_weights": logistic_weights
        if information_weights is None
        else information_weights,
        "prior_cov": cov,
        "target": target,
        "emb": embeddings,
        "subjects": subjects,
        "item_ids": item_ids,
    }


def parameter_context(theta_train, alpha, beta, *, subjects=None, item_ids=None):
    """Adapter for any fitted linear-logit 1PL/2PL/MIRT: sigmoid(theta @ alpha.T - beta).

    For sigmoid(a * (theta-b)), pass beta=a*b. For Rasch, pass alpha=ones((Q,1)).
    PyTorch tensors should be detached and moved to CPU before converting to NumPy.
    Replace ctx['infer'] with a teammate's inference callback to keep their estimator.
    """
    theta_train = np.asarray(theta_train)
    if theta_train.ndim == 1:
        theta_train = theta_train[:, None]
    a, b = np.asarray(alpha), np.asarray(beta).reshape(-1)
    if a.ndim == 1:
        a = a[:, None]
    if b.shape != (len(a),):
        raise ValueError("beta must contain one intercept per item")

    def predict(theta, items=None):
        if items is None:
            return expit(theta @ a.T - b)
        if np.ndim(items) == 1:
            return expit(theta @ a[items].T - b[items])
        return expit(np.einsum("mkd,md->mk", a[items], theta) - b[items])

    def infer(responses, items, initial=None):
        return infer_theta(responses, items, a, b, theta_train, initial)

    emb = np.column_stack([a, b])
    scale = emb.std(0)
    emb = (emb - emb.mean(0)) / np.where(scale > 0, scale, 1)
    ctx = model_context(
        theta_train, a, predict, infer, embeddings=emb, subjects=subjects, item_ids=item_ids
    )
    ctx.update(theta=np.asarray(theta_train), alpha=a, beta=b)
    return ctx


def as_context(ctx):
    """Accept old parameter dictionaries too; preserve their metadata and gp weights."""
    if "loadings" in ctx:
        return ctx
    return {**parameter_context(ctx["theta"], ctx["alpha"], ctx["beta"]), **ctx}


def information_weights(ctx, theta, items=None):
    weights = np.asarray(ctx["information_weights"](theta, items), dtype=float)
    expected = (len(theta), ctx["n_items"] if items is None else np.shape(items)[-1])
    if weights.shape != expected or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(
            "information_weights must return finite nonnegative (models, items) weights"
        )
    return weights


def static_design(ctx, k, criterion="voptimal"):
    """Static Fisher / greedy V-optimal selection, independent of the fitting backend."""
    ctx = as_context(ctx)
    if not 1 <= k <= ctx["n_items"] or criterion not in {"fisher", "voptimal"}:
        raise ValueError("Use a valid budget and criterion: fisher or voptimal")
    a = ctx["loadings"]
    weights = information_weights(ctx, ctx["theta"]).mean(0)
    if criterion == "fisher":
        if ctx["n_dims"] != 1:
            raise ValueError("Fisher ranking requires 1D IRT")
        return np.argsort(-weights * a[:, 0] ** 2, kind="stable")[:k]
    cov = ctx["prior_cov"].copy()
    selected = []
    for _ in range(k):
        gain = voptimal_gain(a, weights, cov, ctx["target"])
        gain[selected] = -np.inf
        item = int(np.argmax(gain))
        selected.append(item)
        c = cov @ a[item]
        cov -= weights[item] * np.outer(c, c) / (1 + weights[item] * a[item] @ c)
    return np.array(selected)


def adaptive_design(Y, ctx, ks, criterion="voptimal"):
    """Yield (k, per-model items, theta). Only selected answers reach model inference."""
    ctx = as_context(ctx)
    ks = set(ks)
    if not ks or min(ks) < 1 or max(ks) > ctx["n_items"]:
        raise ValueError("Checkpoints must lie between 1 and the number of items")
    if criterion not in {"fisher", "voptimal"}:
        raise ValueError("Use criterion fisher or voptimal")
    if criterion == "fisher" and ctx["n_dims"] != 1:
        raise ValueError("Adaptive Fisher requires 1D IRT")
    if Y.shape[1] != ctx["n_items"]:
        raise ValueError("Response columns must match the model's items")
    a, rows = ctx["loadings"], np.arange(len(Y))
    selected = np.empty((len(Y), max(ks)), dtype=int)
    responses = np.empty_like(selected, dtype=float)
    seen = np.zeros(Y.shape, dtype=bool)
    theta = np.broadcast_to(ctx["theta"].mean(0), (len(Y), ctx["n_dims"])).copy()
    precision = np.linalg.inv(ctx["prior_cov"])
    for t in range(max(ks)):
        weights = information_weights(ctx, theta)
        if criterion == "fisher":
            gain = weights * a[:, 0] ** 2
        else:
            chosen_a = a[selected[:, :t]]
            chosen_w = np.take_along_axis(weights, selected[:, :t], axis=1)
            info = precision + np.einsum("mkd,mk,mke->mde", chosen_a, chosen_w, chosen_a)
            gain = voptimal_gain(a, weights, np.linalg.inv(info), ctx["target"])
        gain[seen] = -np.inf
        item = np.argmax(gain, axis=1)
        selected[:, t], responses[:, t] = item, Y[rows, item]
        seen[rows, item] = True
        if not np.isfinite(responses[:, t]).all():
            raise ValueError(
                "A selected response is missing; missing answers cannot be treated as wrong"
            )
        theta = np.asarray(ctx["infer"](responses[:, : t + 1], selected[:, : t + 1], initial=theta))
        if theta.shape != (len(Y), ctx["n_dims"]) or not np.isfinite(theta).all():
            raise ValueError("infer must return finite (models, D) abilities")
        if t + 1 in ks:
            yield t + 1, selected[:, : t + 1].copy(), theta.copy()


def fisher_information(theta, alpha, beta):
    if alpha.shape[1] != 1:
        raise ValueError("Fisher ranking requires 1D IRT")
    p = expit(np.atleast_2d(theta) @ alpha.T - beta)
    return alpha[:, 0] ** 2 * p * (1 - p)


def fisher_static(theta_train, alpha, beta, k):
    return static_design(parameter_context(theta_train, alpha, beta), k, "fisher")


def voptimal_static(theta_train, alpha, beta, k):
    return static_design(parameter_context(theta_train, alpha, beta), k, "voptimal")


def adaptive_items(Y, alpha, beta, theta_train, ks, criterion="fisher"):
    """Compatibility wrapper for the original parameter-array interface."""
    yield from adaptive_design(Y, parameter_context(theta_train, alpha, beta), ks, criterion)


def theta_covariance(theta_train):
    """Training ability covariance, with a small ridge for weak / redundant dimensions."""
    cov = np.atleast_2d(np.cov(np.asarray(theta_train, dtype=float).T))
    return cov + np.eye(cov.shape[0]) * max(np.trace(cov) / len(cov), 1) * 1e-6


def infer_theta(responses, items, alpha, beta, theta_train, initial=None):
    """Newton MAP fit from selected answers only, using the training Gaussian prior.

    responses is (models, k); items is (k,) or (models, k). Warm starts make a fit after
    each adaptive answer cheap. A small covariance ridge handles redundant dimensions;
    otherwise this is the fit_theta objective. Item parameters stay frozen; use float64.
    """
    responses = np.asarray(responses, dtype=float)
    a = np.broadcast_to(np.asarray(alpha, dtype=float)[items], (*responses.shape, alpha.shape[1]))
    b = np.broadcast_to(np.asarray(beta, dtype=float)[items], responses.shape)
    mu = np.asarray(theta_train).mean(0)
    precision = np.linalg.inv(theta_covariance(theta_train))
    theta = np.broadcast_to(mu, (len(responses), len(mu))).copy()
    if initial is not None:
        theta[:] = initial

    def objective(t):
        z = np.einsum("mkd,md->mk", a, t) - b
        d = t - mu
        return (np.logaddexp(0, z) - responses * z).sum(1) + 0.5 * np.sum(d @ precision * d, 1)

    for _ in range(50):
        p = expit(np.einsum("mkd,md->mk", a, theta) - b)
        grad = np.einsum("mkd,mk->md", a, p - responses) + (theta - mu) @ precision
        hessian = np.einsum("mkd,mk,mke->mde", a, p * (1 - p), a) + precision
        step = np.linalg.solve(hessian, grad[..., None])[..., 0]
        if np.max(np.abs(step), initial=0) < 1e-6:
            break
        # Full Newton steps can overshoot for all-correct / all-wrong response prefixes.
        scale = np.ones(len(theta))
        loss = objective(theta)
        for _ in range(25):
            candidate = theta - scale[:, None] * step
            bad = objective(candidate) > loss - 1e-4 * scale * np.sum(grad * step, 1)
            if not bad.any():
                break
            scale[bad] *= 0.5
        theta -= scale[:, None] * step
    return theta


def voptimal_gain(alpha, weights, covariance, target):
    """Reduction in trace(target @ covariance) from each candidate's rank-one information.

    target = alpha.T @ alpha / Q for the full item population: logit prediction variance,
    as in Krumdick et al. (2026), Eq. 6. This is not variance of the mean score.
    covariance / weights may also have a leading model axis.
    """
    c = np.einsum("...de,qe->...qd", covariance, alpha)
    numerator = weights * np.einsum("...qd,de,...qe->...q", c, target, c)
    denominator = 1 + weights * np.einsum("qd,...qd->...q", alpha, c)
    return numerator / denominator
