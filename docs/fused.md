# Fused scorecards

`FusedScorecardRegressor` and `FusedScorecardClassifier` take numeric bin codes.
Fit a binner upstream. By default, codes are sorted numerically within each
feature. Use `bin_order` to declare a different order or to allow codes absent
from training. Codes outside the fitted order are rejected.

For bin effects $s_{jk}$, the penalty is

$$
\alpha\sum_j\sum_{k=2}^{K_j}|s_{jk}-s_{j,k-1}|.
$$

The first bin in each feature has a zero reference effect. `score_table()`
reports contiguous merged bands and their effects in target units for
regression and log odds for classification. Optional `path_alphas` adds
penalties to `alphas_`, `intercept_path_`, and `bin_score_path_`. Merges may
change along the path. An explicitly declared bin absent in training has no
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

Reference [the adjacent-level fusion penalty in the R smurf vignette](https://cran.r-project.org/web/packages/smurf/vignettes/smurf.html).
