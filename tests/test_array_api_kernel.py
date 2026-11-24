import numpy as np
import pytest

from structured_linear._solver import _interaction_product


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_interaction_product_preserves_array_api_namespace(dtype):
    xp = pytest.importorskip("array_api_strict", minversion="2.6")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(17, 3)).astype(dtype)
    z = rng.normal(size=(17, 2)).astype(dtype)
    theta = rng.normal(size=(3, 2)).astype(dtype)
    device = xp.Device("CPU_DEVICE")
    arrays = [xp.asarray(a, device=device) for a in (x, z, theta)]
    actual = _interaction_product(*arrays)
    assert actual.__array_namespace__() is xp
    assert actual.device == arrays[0].device
    assert actual.dtype == getattr(xp, dtype)
    expected = np.einsum("ij,ij->i", x, z @ theta.T)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("classifier", [False, True])
@pytest.mark.parametrize("modifiers", [[], [2, 0]])
def test_pliable_fit_predict_stays_in_namespace(dtype, classifier, modifiers):
    from structured_linear import PliableClassifier, PliableRegressor

    xp = pytest.importorskip("array_api_strict", minversion="2.6")
    rng = np.random.default_rng(21)
    X = rng.normal(size=(80, 4)).astype(dtype)
    signal = X[:, 1] + X[:, 0] * X[:, 1] - X[:, 2]
    y = np.where(signal > 0, 3, -1) if classifier else signal
    weights = np.arange(80, dtype=dtype) % 5
    estimator = PliableClassifier if classifier else PliableRegressor
    options = dict(alpha=0.03, modifier_features=modifiers, warm_start=True)
    expected = estimator(**options).fit(X, y, weights)
    device = xp.Device("CPU_DEVICE")
    inputs = [xp.asarray(a, device=device) for a in (X, y, weights)]
    model = estimator(**options).fit(*inputs).fit(*inputs)
    assert model.converged_
    for array in (model.coef_, model.interaction_coef_, model.predict(inputs[0])):
        assert array.__array_namespace__() is xp
        assert array.device == inputs[0].device
    assert model.coef_.dtype == getattr(xp, dtype)
    methods = ("predict", "predict_proba", "decision_function") if classifier else ("predict",)
    for method in methods:
        result = getattr(model, method)(inputs[0])
        assert result.device == inputs[0].device
        np.testing.assert_allclose(np.asarray(result), getattr(expected, method)(X),
                                   atol=3e-4, rtol=3e-4)
