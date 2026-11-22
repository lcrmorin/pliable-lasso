import numpy as np
import pytest
from sklearn.base import clone, is_classifier

from structured_linear import (
    DataSharedLassoClassifier, DataSharedLassoRegressor,
    FusedScorecardClassifier, FusedScorecardRegressor,
)

ESTIMATORS = [
    DataSharedLassoClassifier(), DataSharedLassoRegressor(),
    FusedScorecardClassifier(path_alphas=[0.05]),
    FusedScorecardRegressor(path_alphas=[0.05]),
]


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_weights_match_repeated_rows(estimator):
    X = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [9, 9]])
    y = np.array([0, 1, 1, 0, 0, 1, 1])
    weights = np.array([1, 3, 2, 2, 1, 4, 0])
    rows = np.repeat(np.arange(len(X)), weights)
    fit = clone(estimator).fit(X, y, sample_weight=weights)
    repeated = clone(estimator).fit(X[rows], y[rows])
    method = "predict_proba" if is_classifier(estimator) else "predict"
    np.testing.assert_allclose(getattr(fit, method)(X[:-1]),
                               getattr(repeated, method)(X[:-1]), atol=1e-6)
    scaled = clone(estimator).fit(X, y, sample_weight=weights * 1e200)
    np.testing.assert_allclose(getattr(fit, method)(X[:-1]),
                               getattr(scaled, method)(X[:-1]), atol=1e-6)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_invalid_sample_weight(estimator):
    for weights in ([1], [-1, 1], [np.inf, 1], 0):
        with pytest.raises(ValueError, match="sample_weight"):
            clone(estimator).fit([[0, 0], [1, 1]], [0, 1], sample_weight=weights)


def test_weighted_constant_scorecard():
    model = FusedScorecardRegressor().fit([[0], [0]], [0, 4], sample_weight=[1, 3])
    np.testing.assert_allclose(model.predict([[0]]), [3])
