# Getting started

Structured Linear requires Python 3.10+ and scikit-learn 1.6+. Install from GitHub:

```bash
python -m pip install "git+https://github.com/lcrmorin/structured-linear.git"
```

From a local checkout, run `python -m pip install .`.

## Fit three regressors

The same toy array represents numeric predictors and modifiers for pliable lasso,
numeric predictors and segments for data-shared lasso, and ordered bin codes for
the fused scorecard. In real data, fit each model's preprocessing within the
training split.

```python
from structured_linear import (
    DataSharedLassoRegressor,
    FusedScorecardRegressor,
    PliableRegressor,
)

X = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
y = [0, 1, 2, 0, 2, 4]

pliable = PliableRegressor(modifier_features=[1], alpha=0.05).fit(X, y)
shared = DataSharedLassoRegressor(segment_feature=1, alpha=0.05).fit(X, y)
fused = FusedScorecardRegressor(alpha=0.05, path_alphas=[0.1, 0.5]).fit(X, y)
print(pliable.predict(X))
print(shared.segment_coef_)
print(fused.score_table())
```

## Classify

Each classifier accepts binary labels and exposes `predict`, `decision_function`,
and `predict_proba`. Probability columns follow `classes_`.
