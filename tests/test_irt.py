import numpy as np
import pytest

pytest.importorskip("torch")  # needs the irt dependency group: uv run --group irt pytest

from cabench.irt import (  # noqa: E402
    fit_mirt,
    fit_theta,
    gp_estimate,
    gp_weight,
    irt_anchors,
    predict_proba,
)


def test_mirt_recovers_simulated_model():
    rng = np.random.default_rng(0)
    th, al, be = rng.normal(size=(300, 2)), rng.normal(size=(2000, 2)), rng.normal(size=2000)
    P = predict_proba(th, al, be)
    Y = (rng.random(P.shape) < P).astype(np.float32)
    t, a, b, _ = fit_mirt(Y, n_dims=2, epochs=1500)
    assert np.corrcoef(predict_proba(t, a, b).ravel(), P.ravel())[0, 1] > 0.9

    # held-out models, half their items observed: does fit_theta recover their scores?
    th_new = rng.normal(size=(50, 2))
    P_new = predict_proba(th_new, al, be)
    Y_new = (rng.random(P_new.shape) < P_new).astype(np.float32)
    mask = rng.random(Y_new.shape) < 0.5
    pred = predict_proba(fit_theta(Y_new, mask, a, b, t), a, b).mean(1)
    assert np.corrcoef(pred, P_new.mean(1))[0, 1] > 0.95


def test_irt_anchors_weights_sum_to_one():
    emb = np.random.default_rng(0).normal(size=(1000, 3))
    anchors, weights = irt_anchors(emb, 20, seed=0)
    assert len(anchors) == 20 and np.isclose(weights.sum(), 1)


def test_gp_blend_beats_both_parts():
    rng = np.random.default_rng(0)
    noisy, biased = rng.normal(0, 0.03, 10_000), rng.normal(0.02, 0.01, 10_000)
    lam = gp_weight(noisy, biased)
    blend = gp_estimate(noisy, biased, lam)
    assert 0 < lam < 1
    assert np.mean(blend**2) < min(np.mean(noisy**2), np.mean(biased**2))
    assert gp_weight(noisy, noisy) == 0.5
