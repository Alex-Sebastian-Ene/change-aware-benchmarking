import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")  # needs the irt dependency group: uv run --group irt pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/irt"))

import protocol  # noqa: E402

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
