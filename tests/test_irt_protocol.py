import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")  # needs the irt dependency group: uv run --group irt pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/irt"))

import protocol  # noqa: E402
from diagnostics import (
    choose_dimension,
    dimension_diagnostics,
    paired_comparison,
    response_diagnostics,
)  # noqa: E402

from cabench.irt import predict_proba  # noqa: E402


def test_protocol_on_simulated_1d_irt():
    rng = np.random.default_rng(0)
    theta = rng.normal(size=(150, 1))
    alpha, beta = rng.normal(1, 0.3, (400, 1)), rng.normal(size=400)
    P = predict_proba(theta, alpha, beta)
    Y = (rng.random(P.shape) < P).astype(np.float32)
    subjects = np.repeat(["a", "b", "c", "d"], 100)

    splits = protocol.make_splits(Y)[:2]
    contexts = [protocol.fit_context(Y[train], subjects) for train, _, _ in splits]
    results = protocol.evaluate(Y, splits, contexts, ks=[10, 50], seeds=range(3))

    assert len(results) == 2 * 2 * len(protocol.BASELINE) * 3
    assert results.mae.notna().all()
    # when the data really is 1-D IRT, p-IRT should beat plain accuracy at small k
    mae = results[results.k == 10].groupby("method").mae.mean()
    assert mae["IRT anchors + p-IRT"] < mae["random + accuracy"]


@pytest.mark.parametrize("dims", [1, 3])
def test_new_methods_share_validation_tuning_and_model_records(dims):
    rng = np.random.default_rng(10)
    theta = rng.normal(size=(50, dims)).astype(np.float32)
    alpha, beta = (
        rng.normal(size=(45, dims)).astype(np.float32),
        rng.normal(size=45).astype(np.float32),
    )
    Y = rng.binomial(1, predict_proba(theta, alpha, beta)).astype(np.float32)
    split = [(np.arange(30), np.arange(30, 40), np.arange(40, 50))]
    ctx = {
        "theta": theta[:30],
        "alpha": alpha,
        "beta": beta,
        "emb": np.column_stack([alpha, beta]),
        "item_mean": Y[:30].mean(0),
    }
    selectors = ["random", "IRT anchors", "voptimal_static", "voptimal_adaptive"]
    if dims == 1:
        selectors += ["fisher_static", "fisher_adaptive"]
    ks, seeds = [3, 7], [0, 1]
    ids = np.array([f"model-{i}" for i in range(len(Y))])
    val, weights = protocol.calibrate_gp(Y, split, [ctx], selectors, ks, seeds, ids)
    assert weights.gp_weight.between(0, 1).all()
    assert len(weights) == len(selectors) * len(ks)
    assert set(val.model_id) == set(ids[30:40])
    assert set(val.estimator) == {"accuracy", "p-IRT", "gp-IRT"}

    methods = [(sel, est) for sel in selectors for est in protocol.ESTIMATORS]
    metrics, records = protocol.evaluate(
        Y,
        split,
        [ctx],
        methods,
        ks=ks,
        seeds=seeds,
        solver="newton",
        return_records=True,
        model_ids=ids,
    )
    assert len(records) == len(metrics) * 10 == len(methods) * len(ks) * len(seeds) * 10
    assert set(records.model_id) == set(ids[40:])
    assert np.isfinite(records.pred_score).all()
    np.testing.assert_allclose(records.abs_error, abs(records.pred_score - records.true_score))
    keys = ["split", "k", "method", "seed", "n_dims"]
    actual = records.groupby(keys).abs_error.mean().sort_index()
    expected = metrics.set_index(keys).mae.sort_index()
    np.testing.assert_allclose(actual, expected, atol=2e-8)
    for name in set(selectors) - {"random", "IRT anchors"}:
        by_seed = records[records.selector == name].pivot(
            index=["k", "estimator", "model_id"], columns="seed", values="pred_score"
        )
        np.testing.assert_array_equal(by_seed[0], by_seed[1])
    # Test predictions cannot alter the validation-fitted weights.
    fitted = ctx["gp_weights"].copy()
    changed = Y.copy()
    changed[40:] = 1 - changed[40:]
    protocol.evaluate(changed, split, [ctx], methods, ks=ks, seeds=seeds, solver="newton")
    assert fitted == ctx["gp_weights"]
    comparison = paired_comparison(records, n_boot=100)
    assert (comparison[comparison.method == "random + gp-IRT"].delta_mae == 0).all()
    with pytest.raises(ValueError, match="same split"):
        paired_comparison(records.iloc[1:], n_boot=100)


def test_complete_observation_recovers_exact_scores():
    rng = np.random.default_rng(2)
    Y = rng.binomial(1, 0.5, size=(5, 12))
    ctx = {
        "theta": rng.normal(size=(30, 2)),
        "alpha": rng.normal(size=(12, 2)),
        "beta": rng.normal(size=12),
    }
    selections = [(np.tile(np.arange(12), (len(Y), 1)), np.ones(12) / 12)]
    np.testing.assert_allclose(
        protocol.estimate_pirt(Y, selections, ctx, solver="newton")[0], Y.mean(1)
    )


def test_diagnostics_and_dimension_choice_use_random_only():
    rng = np.random.default_rng(2)
    Y = rng.binomial(1, 0.5, size=(25, 30))
    diag = dimension_diagnostics(Y[:20], n_null=5)
    assert (diag.variance_explained >= 0).all()
    np.testing.assert_allclose(diag.variance_explained.sum(), 1)
    ctx = {
        "theta": rng.normal(size=(20, 2)),
        "alpha": rng.normal(size=(30, 2)),
        "beta": rng.normal(size=30),
        "item_mean": Y[:20].mean(0),
    }
    scores = response_diagnostics(Y[20:], ctx)
    assert set(scores.predictor) == {"IRT", "item mean"}
    assert np.isfinite(scores[["log_loss", "brier", "ece"]]).all().all()
    import pandas as pd

    val = pd.DataFrame(
        {
            "selector": ["random", "random", "IRT anchors"],
            "estimator": ["gp-IRT"] * 3,
            "n_dims": [1, 2, 8],
            "k": [10] * 3,
            "abs_error": [0.03, 0.02, 0.001],
            "split": [0] * 3,
            "seed": [0] * 3,
        }
    )
    assert choose_dimension(val) == 2


def test_gp_requires_validation_weights():
    rng = np.random.default_rng(0)
    Y = rng.binomial(1, 0.5, (15, 20))
    ctx = {
        "theta": rng.normal(size=(10, 1)),
        "alpha": rng.normal(size=(20, 1)),
        "beta": rng.normal(size=20),
    }
    split = [(np.arange(10), np.arange(10, 12), np.arange(12, 15))]
    with pytest.raises(KeyError, match="gp_weights"):
        protocol.evaluate(
            Y, split, [ctx], [("random", "gp-IRT")], ks=[3], seeds=[0], solver="newton"
        )


def test_stages_accept_prefitted_external_models(tmp_path):
    from run_progression import run_mirt, run_selection

    from cabench.design import model_context, parameter_context

    rng = np.random.default_rng(6)
    Y = rng.binomial(1, 0.5, (24, 20)).astype(float)
    splits = [(np.arange(14), np.arange(14, 19), np.arange(19, 24))]
    item_ids = np.array([f"item-{i}" for i in range(20)])
    contexts = {}
    for d in [1, 2]:
        native = parameter_context(
            rng.normal(size=(14, d)), rng.normal(size=(20, d)), rng.normal(size=20)
        )
        contexts[d] = [
            model_context(
                native["theta"],
                native["loadings"],
                native["predict"],
                native["infer"],
                embeddings=np.column_stack([native["loadings"], np.arange(20)]),
                item_ids=item_ids,
            )
        ]

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("A prefitted model must not be refitted")

    chosen, ctx = run_mirt(
        Y,
        splits,
        None,
        np.arange(24),
        tmp_path / "mirt",
        dims=[1, 2],
        contexts_by_dim=contexts,
        context_factory=forbidden_fit,
        ks=[3, 5],
        seeds=[0, 1],
        n_null=3,
        item_ids=item_ids,
    )
    assert chosen in {1, 2} and ctx is contexts[chosen]
    metrics, records = run_selection(
        Y,
        splits,
        ctx,
        ["random", "voptimal_static", "voptimal_adaptive"],
        tmp_path / "voptimal",
        np.arange(24),
        ks=[3, 5],
        seeds=[0, 1],
        item_ids=item_ids,
    )
    assert len(records) == len(metrics) * 5
    assert (tmp_path / "voptimal/predictions.csv").exists()
    with pytest.raises(ValueError, match="column order"):
        protocol.evaluate(
            Y, splits, ctx, [("random", "accuracy")], ks=[3], seeds=[0], item_ids=item_ids[::-1]
        )
