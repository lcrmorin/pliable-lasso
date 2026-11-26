"""Hardware parity for the category, scorecard, and data-shared estimators."""

import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import is_classifier
from sklearn.model_selection import GridSearchCV

from structured_linear import (
    CategoryFusionClassifier, CategoryFusionRegressor,
    DataSharedLassoClassifier, DataSharedLassoRegressor,
    FusedScorecardClassifier, FusedScorecardRegressor,
)


@pytest.mark.parametrize("estimator", [
    CategoryFusionClassifier, CategoryFusionRegressor,
    DataSharedLassoClassifier, DataSharedLassoRegressor,
    FusedScorecardClassifier, FusedScorecardRegressor,
])
@pytest.mark.parametrize("device,dtype", [
    ("cpu", "float32"), ("mps", "float32"),
    ("cuda", "float32"), ("cuda", "float64"),
])
def test_structured_torch_fit_and_grid_search(estimator, device, dtype):
    torch = pytest.importorskip("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    rng = np.random.default_rng(25)
    X = np.tile([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], (20, 1)).astype(dtype)
    signal = X[:, 0] - X[:, 1] + rng.normal(size=len(X))
    classifier = is_classifier(estimator())
    y = (signal > 0).astype(np.int64) if classifier else signal.astype(dtype)
    weights = rng.uniform(0.5, 2.0, len(X)).astype(dtype)
    features, target, weight = [torch.as_tensor(a, device=device) for a in (X, y, weights)]
    options = dict(param_grid={"alpha": [0.02, 0.1]}, cv=3, error_score="raise")
    expected = GridSearchCV(estimator(), **options).fit(X, y, sample_weight=weights)
    with config_context(array_api_dispatch=True):
        actual = GridSearchCV(estimator(), **options).fit(features, target, sample_weight=weight)
        assert actual.best_params_ == expected.best_params_
        np.testing.assert_allclose(actual.cv_results_["mean_test_score"],
                                   expected.cv_results_["mean_test_score"], atol=4e-4)
        methods = ["predict", "predict_contributions"]
        if classifier:
            methods.extend(["decision_function", "predict_proba"])
        for method in methods:
            result = getattr(actual.best_estimator_, method)(features)
            assert result.device == features.device
            if method != "predict" or not classifier:
                assert result.dtype == features.dtype
            reference = getattr(expected.best_estimator_, method)(X)
            np.testing.assert_allclose(result.cpu().numpy(), reference, atol=4e-4)
