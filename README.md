# Pliable lasso

Pliable lasso fits regression and binary classification models whose predictor
slopes can vary with selected modifier columns. This scikit-learn-compatible
implementation follows the penalty of [Tibshirani and Friedman (2020)](https://doi.org/10.1080/10618600.2019.1648271).

## Installation

From a source checkout, run `python -m pip install .`. The package requires
Python 3.10 or newer and scikit-learn 1.6 or newer, plus NumPy and SciPy.
Install `".[test]"` for the test suite.

## Usage

Predictors and modifiers share one matrix. Pass the zero-based positions
of the modifier columns **after preprocessing**. The other columns are
predictors. Fit preprocessing on training data and reuse it for predictions.

```python
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pliable_lasso import PliableClassifier, PliableRegressor

X, y = load_diabetes(return_X_y=True)
regressor = make_pipeline(
    StandardScaler(),
    PliableRegressor(alpha=0.75, l1_ratio=0.5, modifier_features=[0, 3]),
)
regressor.fit(X, y)
predictions = regressor.predict(X[:3])

X, y = load_breast_cancer(return_X_y=True)
classifier = make_pipeline(
    StandardScaler(),
    PliableClassifier(alpha=0.01, l1_ratio=0.5, modifier_features=[1, 4]),
)
classifier.fit(X, y)
probabilities = classifier.predict_proba(X[:3])  # Columns follow classifier.classes_.
```

Both estimators accept `sample_weight` and work with `Pipeline` and
`GridSearchCV`. The classifier's `decision_function` returns log odds for
`classes_[1]`. Without modifiers, both estimators use an L1 penalty of
`alpha * (1 - l1_ratio)` on predictor coefficients.

## Model and interpretation

For row \(i\), the model score is
\(b + X_i\beta + Z_i\gamma + \sum_j X_{ij}(Z_i\theta_j)\).
The regressor uses weighted squared loss divided by two. The classifier
uses weighted binary log loss. Weights are normalized to sum to one.

The regularizer is
\(\alpha[(1-\rho)\sum_j(\|(\beta_j,\theta_j)\|_2+\|\theta_j\|_2)
+\rho\sum_{j,k}|\theta_{jk}|]\), where \(\rho\) is `l1_ratio`.
The intercept and modifier main effects are unpenalized. Interactions use
products of the columns supplied to the estimator. Encode categorical
inputs numerically and apply any desired scaling in preprocessing.

- `coef_` contains the main effects in input-column order and `intercept_` is scalar.
- `interaction_coef_` has shape `(n_predictors, n_modifiers)`. Rows follow
  `predictor_indices_` and columns follow `modifier_indices_`.
- `objective_`, `optimality_`, `n_iter_`, and `converged_` report fit
  diagnostics. `optimality_` is a relative proximal-gradient stationarity
  residual. Increase `max_iter` or review feature scaling if the solver
  reaches its iteration limit.

The estimators accept finite dense numerical inputs, with a single
continuous target or exactly two classes. The group penalty encourages
hierarchical sparsity. Individual interaction and main-effect selections
can vary with scaling, collinearity, and regularization.

See the [worked real-data example](docs/real-data-example.md) for conditional
slopes, warm-started regularization paths, and comparison with pooled and
segmented baselines. Those scores illustrate the workflow on two small datasets.

## Development and release

Run `python -m pytest` after installing the test extra. The suite includes
independent convex references for both objectives and scikit-learn API checks.
`python -m build` builds a wheel and source archive. The CI workflow checks
distribution metadata and runs the suite against both installed artifacts.

Before public release, review the hosted CI results, confirm the PyPI
project name, and verify a TestPyPI install. Numerical agreement with the
authors' R implementation remains unverified. The code is licensed under
[BSD-3-Clause](LICENSE).
