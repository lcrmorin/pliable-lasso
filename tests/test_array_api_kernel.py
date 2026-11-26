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
@pytest.mark.parametrize("device_name", ["CPU_DEVICE", "device1"])
def test_pliable_fit_predict_stays_in_namespace(dtype, classifier, modifiers, device_name):
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
    device = xp.Device(device_name)
    inputs = [xp.asarray(a, device=device) for a in (X, y, weights)]
    model = estimator(**options).fit(*inputs).fit(*inputs)
    assert model.converged_
    for array in (model.coef_, model.interaction_coef_, model.predictor_indices_,
                  model.modifier_indices_, model.predict(inputs[0])):
        assert array.__array_namespace__() is xp
        assert array.device == inputs[0].device
    assert model.coef_.dtype == getattr(xp, dtype)
    methods = ("predict", "predict_proba", "decision_function") if classifier else ("predict",)
    for method in methods:
        result = getattr(model, method)(inputs[0])
        assert result.device == inputs[0].device
        np.testing.assert_allclose(np.asarray(result.to_device(xp.Device("CPU_DEVICE"))),
                                   getattr(expected, method)(X),
                                   atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("classifier", [False, True])
def test_pliable_grid_search_with_array_api_dispatch(dtype, classifier):
    from sklearn import config_context
    from sklearn.model_selection import GridSearchCV

    from structured_linear import PliableClassifier, PliableRegressor

    xp = pytest.importorskip("array_api_strict", minversion="2.6")
    rng = np.random.default_rng(21)
    X = rng.normal(size=(100, 3)).astype(dtype)
    signal = X[:, 0] + X[:, 0] * X[:, 2] - X[:, 1]
    y = (signal > 0).astype(np.int64) if classifier else signal
    estimator = PliableClassifier if classifier else PliableRegressor
    options = dict(param_grid={"alpha": [0.02, 0.2]}, cv=3, error_score="raise")
    expected = GridSearchCV(estimator(modifier_features=[2]), **options).fit(X, y)

    X_api, y_api = xp.asarray(X), xp.asarray(y)
    with config_context(array_api_dispatch=True):
        search = GridSearchCV(estimator(modifier_features=[2]), **options).fit(X_api, y_api)
        assert search.best_params_ == expected.best_params_
        np.testing.assert_allclose(search.cv_results_["mean_test_score"],
                                   expected.cv_results_["mean_test_score"], atol=3e-4)
        method = "predict_proba" if classifier else "predict"
        prediction = getattr(search, method)(X_api)
        assert prediction.__array_namespace__() is xp
        assert prediction.device == X_api.device
        assert prediction.dtype == X_api.dtype
        np.testing.assert_allclose(np.asarray(prediction), getattr(expected, method)(X),
                                   atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("family", ["CategoryFusion", "DataSharedLasso", "FusedScorecard"])
@pytest.mark.parametrize("classifier", [False, True])
def test_cpu_estimators_with_array_api_dispatch(family, classifier):
    from sklearn import config_context
    from sklearn.model_selection import GridSearchCV

    import structured_linear

    xp = pytest.importorskip("array_api_strict", minversion="2.6")
    rng = np.random.default_rng(25)
    X = np.tile([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], (20, 1))
    signal = X[:, 0] - X[:, 1] + rng.normal(size=len(X))
    y = (signal > 0).astype(np.int64) if classifier else signal
    weights = rng.uniform(0.5, 2.0, size=len(X))
    suffix = "Classifier" if classifier else "Regressor"
    estimator = getattr(structured_linear, family + suffix)
    options = dict(param_grid={"alpha": [0.02, 0.1]}, cv=3, error_score="raise")
    expected = GridSearchCV(estimator(), **options).fit(X, y, sample_weight=weights)

    X_api, y_api, weights_api = (xp.asarray(value) for value in (X, y, weights))
    with config_context(array_api_dispatch=True):
        search = GridSearchCV(estimator(), **options).fit(X_api, y_api, sample_weight=weights_api)
        assert search.best_params_ == expected.best_params_
        np.testing.assert_allclose(search.cv_results_["mean_test_score"],
                                   expected.cv_results_["mean_test_score"])
        methods = ["predict", "predict_contributions"]
        if classifier:
            methods.extend(["decision_function", "predict_proba"])
        for method in methods:
            output = getattr(search.best_estimator_, method)(X_api)
            assert output.__array_namespace__() is xp
            assert output.device == X_api.device
            np.testing.assert_allclose(np.asarray(output), getattr(expected.best_estimator_, method)(X))


def _pliable_array_api_checks():
    from sklearn.utils.estimator_checks import estimator_checks_generator

    from structured_linear import PliableClassifier, PliableRegressor

    for estimator in (PliableRegressor(), PliableClassifier()):
        for model, check in estimator_checks_generator(estimator):
            if "array_api" in getattr(check, "func", check).__name__:
                options = ", ".join(str(value) for value in check.keywords.values())
                yield pytest.param(model, check, id=f"{type(model).__name__}({options})")


@pytest.mark.parametrize("estimator,check", list(_pliable_array_api_checks()))
def test_pliable_upstream_array_api(estimator, check):
    check(estimator)


@pytest.mark.parametrize("classifier", [False, True])
def test_pliable_numpy_features_with_device_targets_and_weights(classifier):
    from structured_linear import PliableClassifier, PliableRegressor

    xp = pytest.importorskip("array_api_strict", minversion="2.6")
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 0, 1, 0, 1, 1])
    weights = np.arange(1, 7, dtype=float)
    estimator = PliableClassifier if classifier else PliableRegressor
    expected = estimator(alpha=0.02).fit(X, y, sample_weight=weights)
    device = xp.Device("device1")
    result = estimator(alpha=0.02).fit(
        X, xp.asarray(y, device=device), sample_weight=xp.asarray(weights, device=device)
    )
    np.testing.assert_allclose(result.coef_, expected.coef_)
    np.testing.assert_allclose(result.predict(X), expected.predict(X))
