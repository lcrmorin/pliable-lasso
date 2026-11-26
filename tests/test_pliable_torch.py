import numpy as np
import pytest

from structured_linear import PliableClassifier, PliableRegressor


TORCH_CONFIGURATIONS = [
    ("cpu", "float32"), ("cpu", "float64"),
    ("mps", "float32"),
    ("cuda", "float32"), ("cuda", "float64"),
]


def _torch_for_device(device):
    torch = pytest.importorskip("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch


@pytest.mark.parametrize("device,dtype", TORCH_CONFIGURATIONS)
@pytest.mark.parametrize("classifier", [False, True])
def test_pliable_torch_fit_and_predict(device, dtype, classifier):
    torch = _torch_for_device(device)

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


@pytest.mark.parametrize("device,dtype", TORCH_CONFIGURATIONS)
@pytest.mark.parametrize("classifier", [False, True])
def test_pliable_torch_grid_search(device, dtype, classifier):
    from sklearn import config_context
    from sklearn.model_selection import GridSearchCV

    torch = _torch_for_device(device)
    rng = np.random.default_rng(21)
    X = rng.normal(size=(100, 3)).astype(dtype)
    signal = X[:, 0] + X[:, 0] * X[:, 2] - X[:, 1]
    y = (signal > 0).astype(np.int64) if classifier else signal
    estimator = PliableClassifier if classifier else PliableRegressor
    options = dict(param_grid={"alpha": [0.02, 0.2]}, cv=3, error_score="raise")
    expected = GridSearchCV(estimator(modifier_features=[2]), **options).fit(X, y)
    X_device = torch.as_tensor(X, device=device)
    y_device = torch.as_tensor(y, device=device)

    with config_context(array_api_dispatch=True):
        search = GridSearchCV(estimator(modifier_features=[2]), **options).fit(X_device, y_device)
        assert search.best_params_ == expected.best_params_
        np.testing.assert_allclose(search.cv_results_["mean_test_score"],
                                   expected.cv_results_["mean_test_score"], atol=4e-4)
        model = search.best_estimator_
        for array in (model.coef_, model.interaction_coef_, model.predictor_indices_,
                      model.modifier_indices_):
            assert array.device == X_device.device
        assert model.coef_.dtype == X_device.dtype
        if classifier:
            assert model.classes_.device == X_device.device
        method = "predict_proba" if classifier else "predict"
        result = getattr(search, method)(X_device)
        assert result.device == X_device.device
        assert result.dtype == X_device.dtype
        np.testing.assert_allclose(result.cpu().numpy(), getattr(expected, method)(X),
                                   atol=4e-4, rtol=4e-4)
