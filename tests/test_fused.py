import pickle

import cvxpy as cp
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from structured_linear import FusedScorecardClassifier


def test_objective_and_path():
    counts = np.array([2, 2, 8, 8])
    bins = np.repeat([10, 30, 20, 40], 10)[:, None]
    y = np.concatenate([np.r_[np.ones(n), np.zeros(10 - n)] for n in counts])
    alpha = 0.025
    model = FusedScorecardClassifier(
        alpha=alpha, path_alphas=[1.0, 0.01], bin_order=[[10, 30, 20, 40]]
    )
    pipeline = make_pipeline(clone(model)).fit(bins, y)
    fitted = pipeline[-1]
    np.testing.assert_array_equal(fitted.alphas_, [1.0, alpha, 0.01])
    np.testing.assert_array_equal(fitted.bin_score_path_[0][0], np.zeros(4))
    assert [row["merged_bin"] for row in fitted.score_table()] == [0, 0, 1, 1]

    scores = cp.Variable(4)
    margins = cp.multiply(1 - 2 * y, scores[np.repeat(np.arange(4), 10)])
    reference = cp.Problem(cp.Minimize(
        cp.sum(cp.logistic(margins)) / len(y) + alpha * cp.norm1(cp.diff(scores))
    )).solve(solver="CLARABEL")
    actual = np.logaddexp(0, (1 - 2 * y) * fitted.decision_function(bins)).mean()
    actual += alpha * np.abs(np.diff(fitted.bin_scores_[0])).sum()
    assert actual == pytest.approx(reference, abs=1e-7)

    restored = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_array_equal(restored.predict_proba(bins), pipeline.predict_proba(bins))
    with pytest.raises(ValueError, match="outside the fitted bin order"):
        fitted.predict([[50]])
    original_probabilities = pipeline.predict_proba(bins)
    fitted.fit(bins, 1 - y)
    np.testing.assert_allclose(
        fitted.predict_proba(bins), original_probabilities[:, ::-1], atol=1e-7
    )


def test_fused_squared_loss():
    from structured_linear import FusedScorecardRegressor

    X = np.repeat([0, 1, 2, 3], 5)[:, None]
    y = np.repeat([0.0, 0.0, 3.0, 3.0], 5)
    alpha = 0.1
    fitted = FusedScorecardRegressor(alpha=alpha, path_alphas=[2.0]).fit(X, y)
    score = cp.Variable(4)
    reference = cp.Problem(cp.Minimize(
        cp.sum_squares(y - score[X.ravel()]) / (2 * len(y))
        + alpha * cp.norm1(cp.diff(score))
    )).solve(solver="CLARABEL")
    actual = np.mean((y - fitted.predict(X)) ** 2) / 2
    actual += alpha * np.abs(np.diff(fitted.bin_scores_[0])).sum()
    assert actual == pytest.approx(reference, abs=1e-7)
    assert [row["merged_bin"] for row in fitted.score_table()] == [0, 0, 1, 1]
    np.testing.assert_array_equal(fitted.bin_score_path_[0][0], np.zeros(4))
    fitted.fit(np.zeros_like(X), y)
    np.testing.assert_allclose(fitted.predict(np.zeros_like(X)), y.mean())
