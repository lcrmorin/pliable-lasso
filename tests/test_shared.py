import pickle

import cvxpy as cp
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from structured_linear import DataSharedLassoClassifier, DataSharedLassoRegressor


@pytest.mark.parametrize("classifier", [False, True])
def test_objective_and_unseen_segment(classifier):
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(size=(n, 2))
    segment = np.arange(n) % 3
    design = np.empty((n, 3), dtype=object)
    design[:, :2] = X
    design[:, 2] = np.array(["sme", "retail", "corporate"])[segment]
    score = X[:, 0] + (segment == 1) * X[:, 1]
    alpha, weight = 0.025, 0.8
    cls = DataSharedLassoClassifier if classifier else DataSharedLassoRegressor
    estimator = cls(alpha=alpha, deviation_weight=weight)
    shared, deviation, intercept = cp.Variable(2), cp.Variable((3, 2)), cp.Variable()
    eta = intercept + X @ shared + cp.sum(cp.multiply(X, deviation[segment, :]), axis=1)
    if classifier:
        y = rng.binomial(1, 1 / (1 + np.exp(-score)))
        loss = cp.sum(cp.logistic(cp.multiply(1 - 2 * y, eta))) / n
    else:
        y = score + rng.normal(size=n)
        loss = cp.sum_squares(y - eta) / (2 * n)
    pipeline = make_pipeline(clone(estimator)).fit(design, y)
    model = pipeline[-1]
    penalty = cp.norm1(shared) + weight * cp.norm1(cp.vec(deviation, order="C"))
    reference = cp.Problem(cp.Minimize(loss + alpha * penalty)).solve(solver="CLARABEL")
    predict_score = model.decision_function if classifier else model.predict
    fitted = predict_score(design)
    achieved = (np.logaddexp(0, (1 - 2 * y) * fitted).mean() if classifier
                else np.mean((y - fitted) ** 2) / 2)
    achieved += alpha * (
        np.abs(model.shared_coef_).sum() + weight * np.abs(model.segment_deviations_).sum()
    )
    assert achieved == pytest.approx(reference, abs=1e-7)
    unseen = design[:4].copy()
    unseen[:, -1] = "new"
    np.testing.assert_allclose(predict_score(unseen), model.intercept_ + X[:4] @ model.shared_coef_)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(pipeline)).predict(design),
                                  pipeline.predict(design))
    model.fit(design[:99], y[:99])
    assert model.segment_coef_.shape == (3, 2)
