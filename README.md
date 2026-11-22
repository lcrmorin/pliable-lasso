# Structured Linear

Scikit-learn estimators for interpretable regression and binary classification with structured coefficients.

| Family | Regression | Binary classification |
| --- | --- | --- |
| Effects that vary with modifiers | `PliableRegressor` | `PliableClassifier` |
| Adjacent bins that share effects | `FusedScorecardRegressor` | `FusedScorecardClassifier` |
| Segments with shared slopes and sparse deviations | `DataSharedLassoRegressor` | `DataSharedLassoClassifier` |

Install with `python -m pip install structured-linear`. Requires Python 3.10+ and scikit-learn 1.6+.

[Read the documentation](https://lcrmorin.github.io/structured-linear/) for the quickstart, estimator guides, worked examples, and benchmark.
