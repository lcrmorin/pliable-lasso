import numpy as np
import pytest

from structured_linear import PliableClassifier, PliableRegressor


@pytest.mark.parametrize("device,dtype", [
    ("cpu", "float32"), ("cpu", "float64"),
    ("mps", "float32"),
    ("cuda", "float32"), ("cuda", "float64"),
])
@pytest.mark.parametrize("classifier", [False, True])
def test_pliable_torch_fit_and_predict(device, dtype, classifier):
    torch = pytest.importorskip("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    rng = np.random.default_rng(29)
    X = rng.normal(size=(120, 4)).astype(dtype)
    signal = X[:, 0] + X[:, 1] * X[:, 3] - X[:, 2]
    y = np.where(signal > 0, 3, -1) if classifier else signal
    weights = np.arange(120, dtype=dtype) % 5
    estimator = PliableClassifier if classifier else PliableRegressor
    options = dict(alpha=0.03, modifier_features=[3], warm_start=True)
    expected = estimator(**options).fit(X, y, weights)
    tensors = [torch.as_tensor(array, device=device) for array in (X, y, weights)]
    model = estimator(**options).fit(*tensors).fit(*tensors)

    assert model.converged_
    for array in (model.coef_, model.interaction_coef_):
        assert array.device == tensors[0].device
        assert array.dtype == tensors[0].dtype
    methods = ("predict", "predict_proba", "decision_function") if classifier else ("predict",)
    for method in methods:
        result = getattr(model, method)(tensors[0])
        assert result.device == tensors[0].device
        if method != "predict" or not classifier:
            assert result.dtype == tensors[0].dtype
        np.testing.assert_allclose(result.cpu().numpy(), getattr(expected, method)(X),
                                   atol=4e-4, rtol=4e-4)
