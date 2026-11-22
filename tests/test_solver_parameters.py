import pytest

from structured_linear import (
    DataSharedLassoClassifier,
    DataSharedLassoRegressor,
    FusedScorecardClassifier,
    FusedScorecardRegressor,
)


@pytest.mark.parametrize(
    "estimator",
    [DataSharedLassoClassifier, DataSharedLassoRegressor,
     FusedScorecardClassifier, FusedScorecardRegressor],
)
@pytest.mark.parametrize("parameter,value", [("max_iter", 0), ("tol", float("nan"))])
def test_invalid_solver_controls(estimator, parameter, value):
    with pytest.raises(ValueError, match=parameter):
        estimator(**{parameter: value}).fit([[0, 0], [1, 1]], [0, 1])
