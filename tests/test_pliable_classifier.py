import pickle

import numpy as np
import pytest
from scipy.special import expit
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from structured_linear import PliableClassifier


def expected_classifier_check_failures(estimator):
    if sklearn_version == "1.6.0":
        return {
            "check_sample_weight_equivalence_on_dense_data":
                "The binary-label fixture encodes the resampled targets independently."
        }
    return {}


@pytest.mark.sklearn_compat
@parametrize_with_checks(
    [PliableClassifier()], expected_failed_checks=expected_classifier_check_failures
)
def test_sklearn_classifier(estimator, check):
    check(estimator)


def data():
    rng = np.random.default_rng(91)
    X = rng.normal(size=(120, 4))
    eta = -0.4 + X[:, 0] - 0.4 * X[:, 3] + 1.2 * X[:, 0] * X[:, 2]
    return X, rng.binomial(1, expit(eta))


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_classifier_convex_reference(fit_intercept):
    cp = pytest.importorskip("cvxpy")
    X, y = data()
    weights = np.arange(len(y)) % 4
    design = np.column_stack((X, (X[:, :2, None] * X[:, None, 2:]).reshape(len(y), -1)))
    coefficients = cp.Variable(8)
    intercept = cp.Variable() if fit_intercept else 0.
    eta = design @ coefficients + intercept
    theta = cp.reshape(coefficients[4:], (2, 2), order="C")
    grouped = sum(cp.norm(cp.hstack([coefficients[j], theta[j]]), 2)
                  + cp.norm(theta[j], 2) for j in range(2))
    alpha, ratio = .03, .4
    objective = (weights / weights.sum()) @ cp.logistic(cp.multiply(1 - 2*y, eta))
    objective += alpha * ((1-ratio) * grouped + ratio * cp.norm1(coefficients[4:]))
    reference = cp.Problem(cp.Minimize(objective))
    reference.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    labels = np.where(y, "yes", "no")
    model = PliableClassifier(alpha=alpha, l1_ratio=ratio, modifier_features=[2, 3],
                              fit_intercept=fit_intercept, tol=1e-10)
    model.fit(X, labels, sample_weight=weights)
    assert model.converged_ and reference.status == cp.OPTIMAL
    np.testing.assert_array_equal(model.classes_, ["no", "yes"])
    np.testing.assert_allclose(model.objective_, reference.value, atol=1e-9, rtol=0)
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], expit(eta.value), atol=2e-5)
    repeated = clone(model).fit(X.repeat(weights, axis=0), labels.repeat(weights))
    np.testing.assert_allclose(model.predict_proba(X), repeated.predict_proba(X), atol=1e-8)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(model.classes_, restored.classes_)
    np.testing.assert_array_equal(model.predict_proba(X), restored.predict_proba(X))


def test_classifier_warm_start_and_grid_search():
    X, y = data()
    warm = PliableClassifier(alpha=.1, modifier_features=[2, 3], warm_start=True, tol=1e-9)
    warm.fit(X, y).set_params(alpha=.025).fit(X, y)
    cold = clone(warm).fit(X, y)
    np.testing.assert_allclose(warm.predict_proba(X), cold.predict_proba(X), atol=1e-7)
    pipe = make_pipeline(StandardScaler(), clone(warm))
    search = GridSearchCV(pipe, {"pliableclassifier__alpha": [.01, .03]}, cv=3,
                          scoring="neg_log_loss")
    search.fit(X, y)
    assert search.predict_proba(X[:2]).shape == (2, 2)
    with pytest.raises(ValueError, match="one class"):
        warm.fit(X, y, sample_weight=y)
    np.testing.assert_allclose(warm.predict_proba(X), cold.predict_proba(X), atol=1e-7)
