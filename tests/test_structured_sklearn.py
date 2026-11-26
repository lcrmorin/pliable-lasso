"""Run sklearn's generic checks alongside the structured-input tests."""
import pytest
from sklearn import __version__ as sklearn_version
from sklearn.utils.estimator_checks import parametrize_with_checks

from structured_linear import (
    CategoryFusionClassifier,
    CategoryFusionRegressor,
    DataSharedLassoClassifier,
    DataSharedLassoRegressor,
    FusedScorecardClassifier,
    FusedScorecardRegressor,
)


def expected_failures(estimator):
    failures = {}
    if isinstance(estimator, (FusedScorecardClassifier, FusedScorecardRegressor,
                              CategoryFusionClassifier, CategoryFusionRegressor)):
        failures["check_sample_weight_equivalence_on_dense_data"] = (
            "The check scores levels removed by zero weighting; unseen levels are rejected."
        )
        failures["check_fit_idempotent"] = (
            "The check splits continuous columns, generating unseen bin or segment levels."
        )
    if isinstance(estimator, (FusedScorecardClassifier, CategoryFusionClassifier)):
        failures["check_decision_proba_consistency"] = (
            "The held-out continuous fixture has unseen bin or segment levels."
        )
    if isinstance(estimator, (DataSharedLassoClassifier, CategoryFusionClassifier)):
        failures["check_classifiers_train"] = (
            "The generic training-accuracy threshold assumes all columns are numeric predictors; "
            "the final continuous column is consumed as a segment identifier instead."
        )
    if sklearn_version == "1.6.0" and isinstance(estimator, DataSharedLassoClassifier):
        failures["check_sample_weight_equivalence_on_dense_data"] = (
            "scikit-learn 1.6.0 encodes the weighted and resampled binary targets independently."
        )
    return failures


@pytest.mark.sklearn_compat
@parametrize_with_checks(
    [FusedScorecardClassifier(), FusedScorecardRegressor(),
     DataSharedLassoClassifier(), DataSharedLassoRegressor(),
     CategoryFusionClassifier(), CategoryFusionRegressor()],
    expected_failed_checks=expected_failures,
)
def test_sklearn_structured(estimator, check):
    check(estimator)


@pytest.mark.parametrize("estimator", [
    FusedScorecardClassifier(max_iter=1, path_alphas=[0.1]),
    FusedScorecardRegressor(max_iter=1, path_alphas=[0.1]),
    DataSharedLassoClassifier(max_iter=1),
    DataSharedLassoRegressor(max_iter=1),
])
def test_reports_iterations_for_primary_fit(estimator):
    X = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
    estimator.fit(X, [0, 0, 1, 0, 1, 1])
    assert 0 <= estimator.n_iter_ <= 1


def test_fused_regressor_reports_zero_iterations_for_analytic_fit():
    model = FusedScorecardRegressor().fit([[1], [1]], [0, 1])
    assert model.n_iter_ == 0
