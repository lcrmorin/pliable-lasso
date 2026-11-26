import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from structured_linear import CategoryFusionClassifier, CategoryFusionRegressor


@pytest.mark.parametrize("logistic", [False, True])
def test_weighted_objective_against_cvxpy(logistic):
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(41)
    x = rng.normal(size=120)
    codes = np.repeat(np.arange(3), 40)
    score = np.array([-0.8, -0.8, 1.5])[codes] * x
    y = rng.binomial(1, 1 / (1 + np.exp(-score))) if logistic else score
    X = np.column_stack((x, codes))
    weights = rng.uniform(0.5, 1.5, len(y))
    alpha = 0.015
    estimator = CategoryFusionClassifier if logistic else CategoryFusionRegressor
    model = clone(estimator(alpha=alpha)).fit(X, y, sample_weight=weights)
    assert model.converged_

    design = np.column_stack((np.ones(len(x)), x))
    beta = cp.Variable((3, 2))
    scores = cp.sum(cp.multiply(design, beta[codes]), axis=1)
    loss = cp.logistic(scores) - cp.multiply(y, scores) if logistic else 0.5 * cp.square(scores - y)
    penalty = sum(cp.norm(beta[i] - beta[j]) for i in range(3) for j in range(i))
    problem = cp.Problem(cp.Minimize(weights @ loss / weights.sum() + alpha * penalty))
    problem.solve(solver="CLARABEL")
    assert problem.status == cp.OPTIMAL
    prediction = model.decision_function(X) if logistic else model.predict(X)
    assert_allclose(prediction, scores.value, atol=1e-4)


def test_fusion_of_nominal_levels_and_unknown_segment():
    X = np.array([[-1, "A"], [1, "A"], [-1, "C"], [1, "C"], [-1, "B"], [1, "B"]], dtype=object)
    y = np.array([-1, 1, -1.005, 1.005, 4, 2])
    model = CategoryFusionRegressor().fit(X, y)
    assert model.segments_ == ("A", "C", "B")
    assert model.n_clusters_ == 2
    assert model.cluster_labels_[0] == model.cluster_labels_[1]
    assert_allclose(model.predict(X)[:2], model.predict(X)[2:4], atol=1e-6)
    with pytest.raises(ValueError, match="Unknown segment"):
        model.predict([[0, "D"]])
    model.fit(X[:2], y[:2])
    assert model.segments_ == ("A",)
    assert model.n_clusters_ == 1
    assert_allclose(model.predict(X[:2]), y[:2], atol=1e-6)


@pytest.mark.parametrize("classifier", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_array_api_fit_stays_on_simulated_device(classifier, dtype):
    xp = pytest.importorskip("array_api_strict")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 3)).astype(dtype)
    X[:, -1] = np.arange(len(X)) % 3
    signal = X[:, 0] - X[:, 1] + rng.normal(size=len(X))
    y = (signal > 0).astype(int) if classifier else signal
    weight = (np.arange(len(X)) % 5).astype(dtype)
    estimator = CategoryFusionClassifier if classifier else CategoryFusionRegressor
    expected = estimator(alpha=0.15).fit(X, y, sample_weight=weight)
    device = xp.Device("device1")
    cpu = xp.Device("CPU_DEVICE")
    features = xp.asarray(X, device=device)
    model = estimator(alpha=0.15).fit(features, y, sample_weight=weight)
    assert model.converged_
    for array in (model.coef_, model.intercept_, model.cluster_labels_):
        assert array.device == device
    method = "predict_proba" if classifier else "predict"
    prediction = getattr(model, method)(features)
    assert prediction.device == device
    assert prediction.dtype == features.dtype
    np.testing.assert_allclose(np.asarray(prediction.to_device(cpu)),
                               getattr(expected, method)(X), atol=1e-4)
    np.testing.assert_array_equal(np.asarray(model.cluster_labels_.to_device(cpu)),
                                  expected.cluster_labels_)
