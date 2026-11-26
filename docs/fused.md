# Fused scorecards

See the [worked scorecard example](fused-example.md) for training-only binning, merged bands, and held-out predictions.

## Input format

`FusedScorecardRegressor` and `FusedScorecardClassifier` take numeric bin codes.
Fit a binner upstream. By default, codes are sorted numerically within each
feature. Use `bin_order` to declare a different order or to allow codes absent
from training. Codes outside the fitted order are rejected.

## Key parameters

For bin effects $s_{jk}$, the penalty is

$$
\alpha\sum_j\sum_{k=2}^{K_j}|s_{jk}-s_{j,k-1}|.
$$

## Fitted outputs

The first bin in each feature has a zero reference effect. `score_table()`
reports contiguous merged bands and their effects in target units for
regression and log odds for classification. Optional `path_alphas` adds
penalties to `alphas_`, `intercept_path_`, and `bin_score_path_`. Merges may
change along the path.

## Prediction behavior

Scores sum the fitted intercept and one bin effect per feature. An explicitly declared bin absent in training has no
direct evidence for its effect.

## Regression and classification example

```python
from structured_linear import FusedScorecardClassifier, FusedScorecardRegressor

X = [[0], [0], [1], [1], [2], [2]]
reg = FusedScorecardRegressor(alpha=0.02, path_alphas=[0.05, 0.2])
reg.fit(X, [0, 0, 1, 1, 3, 3])
clf = FusedScorecardClassifier(alpha=0.02)
clf.fit(X, [0, 0, 0, 1, 1, 1])
print(reg.score_table(), clf.predict_proba(X))
```

Original paper [Tibshirani, Saunders, Rosset, Zhu, and Knight, *Sparsity and Smoothness Via the Fused Lasso*](https://doi.org/10.1111/j.1467-9868.2005.00490.x). The scorecard uses the successive-difference penalty on bin effects.

See also [the adjacent-level fusion penalty in the R smurf vignette](https://cran.r-project.org/web/packages/smurf/vignettes/smurf.html).

See the [prediction breakdown guide](estimators.md#explain-individual-predictions) for per-row bin contributions.

## Array API inputs

Numeric Array API inputs use native bin lookup and implicit design products.
The solver operates on adjacent differences and retains the fitted path and
inference arrays on the input device. The NumPy path continues to use a sparse
design. Strict-array tests cover weighted grid search, custom bin order,
single-level features, and float32 on a simulated non-CPU device. Real GPU
validation is pending.
