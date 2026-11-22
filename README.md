# Structured Linear

Scikit-learn estimators for interpretable regression and binary classification with structured coefficients.

| Family | Regression | Binary classification |
| --- | --- | --- |
| Effects that vary with modifiers | `PliableRegressor` | `PliableClassifier` |
| Adjacent bins that share effects | `FusedScorecardRegressor` | `FusedScorecardClassifier` |
| Segments with shared slopes and sparse deviations | `DataSharedLassoRegressor` | `DataSharedLassoClassifier` |

Install from a source checkout with `python -m pip install .`. Requires Python 3.10+ and scikit-learn 1.6+.

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

[Read the documentation](https://lcrmorin.github.io/structured-linear/).

See the [estimator comparison](https://lcrmorin.github.io/structured-linear/estimators/) for preprocessing and API conventions, and the [worked example](docs/real-data-example.md) for conditional slopes and regularization paths.

### Benchmark and tests

Run `python benchmarks/compare_classifiers.py` from an installed checkout.
The benchmark uses shared splits, inner-CV tuning by log loss, and a common ridge baseline.
The synthetic settings distinguish stepwise effects, sparse segment deviations,
and continuous effect modification. The real-data segment is derived from a
feature for illustration. See [the benchmark results](benchmarks/classification-results.csv).

Install test dependencies with `python -m pip install '.[test]'` and run `python -m pytest`.

### Attribution

[BSD 3-Clause license](LICENSE). Methods are based on the
[pliable lasso](https://arxiv.org/abs/1712.00484),
the [adjacent-level fusion penalty](https://cran.r-project.org/web/packages/smurf/vignettes/smurf.html),
and the [data-shared lasso](https://pmc.ncbi.nlm.nih.gov/articles/PMC5650251/).
