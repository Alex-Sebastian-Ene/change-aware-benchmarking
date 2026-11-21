import subprocess
import sys

import numpy as np
import pytest

from cabench.design import adaptive_design, model_context, parameter_context, static_design


def test_design_import_does_not_require_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import cabench.design; assert 'torch' not in sys.modules",
        ],
        check=True,
    )


def test_unidimensional_parameter_vectors_are_normalised():
    theta = np.linspace(-2, 2, 20)
    ctx = parameter_context(theta, np.ones(8), np.zeros((8, 1)))
    assert ctx["theta"].shape == (20, 1)
    assert ctx["loadings"].shape == (8, 1)
    assert ctx["predict"](ctx["theta"]).shape == (20, 8)
    assert ctx["infer"](np.ones((2, 3)), np.arange(3)).shape == (2, 1)


@pytest.mark.parametrize("dims", [1, 3])
def test_teammate_callbacks_receive_only_selected_answers(dims):
    rng = np.random.default_rng(4)
    theta, a, b = rng.normal(size=(25, dims)), rng.normal(size=(30, dims)), rng.normal(size=30)
    native = parameter_context(theta, a, b)
    calls = []

    def infer(responses, items, initial=None):
        calls.append((responses.copy(), items.copy()))
        return native["infer"](responses, items, initial)

    external = model_context(theta, a, native["predict"], infer)
    Y = rng.binomial(1, native["predict"](theta[:3]))
    actual = list(adaptive_design(Y, external, [3, 8]))[-1]
    expected = list(adaptive_design(Y, native, [3, 8]))[-1]
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[2], expected[2])
    assert len(calls) == 8
    for responses, items in calls:
        np.testing.assert_array_equal(responses, np.take_along_axis(Y, items, axis=1))
    assert "alpha" not in external and "beta" not in external
    np.testing.assert_array_equal(static_design(external, 8), static_design(native, 8))
    shared = np.arange(5)
    per_model = np.tile(shared, (3, 1))
    np.testing.assert_allclose(
        external["predict"](theta[:3], shared), external["predict"](theta[:3], per_model)
    )


def test_custom_information_and_target_are_used():
    theta = np.linspace(-1, 1, 30)[:, None]
    native = parameter_context(theta, np.ones((4, 1)), np.zeros(4))

    def weights(theta, items=None):
        return np.broadcast_to([0.0, 0.0, 2.0, 1.0], (len(theta), 4))

    custom = model_context(
        theta, np.ones((4, 1)), native["predict"], native["infer"], information_weights=weights
    )
    np.testing.assert_array_equal(static_design(custom, 2), [2, 3])
    np.testing.assert_array_equal(static_design(custom, 2, "fisher"), [2, 3])
    _, selected, _ = list(adaptive_design(np.zeros((2, 4)), custom, [2]))[-1]
    np.testing.assert_array_equal(selected, [[2, 3], [2, 3]])

    theta2 = np.column_stack([theta[:, 0], np.sin(theta[:, 0])])
    a = np.array([[2.0, 0], [0, 2]])
    p = parameter_context(theta2, a, np.zeros(2))
    custom2 = model_context(
        theta2, a, p["predict"], p["infer"], prior_cov=np.eye(2), target=np.diag([0.0, 1.0])
    )
    assert static_design(custom2, 1)[0] == 1


def test_bad_coordinates_and_missing_responses_fail_clearly():
    with pytest.raises(ValueError, match="same D"):
        model_context(np.zeros((10, 2)), np.ones((5, 1)), None, None)
    ctx = parameter_context(np.linspace(-1, 1, 20)[:, None], np.ones((5, 1)), np.zeros(5))
    with pytest.raises(ValueError, match="missing"):
        list(adaptive_design(np.full((2, 5), np.nan), ctx, [1]))
