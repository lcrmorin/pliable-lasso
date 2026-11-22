# Data-shared lasso

`DataSharedLassoRegressor` and `DataSharedLassoClassifier` take a segment
column selected by `segment_feature` and finite numeric predictors. Segment
labels can be strings or finite numbers. The slope in segment $g$ is
$\beta + \Delta_g$, with penalty

$$
\alpha\left(\lVert\beta\rVert_1
  + w\sum_g\lVert\Delta_g\rVert_1\right).
$$

`deviation_weight` sets $w$. Larger values penalize departures from shared
slopes more strongly. The fitted values are available in `shared_coef_`,
`segment_deviations_`, and `segment_coef_`, with rows indexed by `segments_`.
An unseen segment uses the shared slopes and global intercept. The
shared/deviation decomposition can be nonunique for some penalty settings.
Add segment indicators as predictor columns if you need segment-specific
baseline shifts.

## Regression and classification example

```python
from structured_linear import DataSharedLassoClassifier, DataSharedLassoRegressor

X = [[0, "A"], [1, "A"], [2, "A"], [0, "B"], [1, "B"], [2, "B"]]
reg = DataSharedLassoRegressor(segment_feature=1, alpha=0.05)
reg.fit(X, [0, 1, 2, 0, 2, 4])
clf = DataSharedLassoClassifier(segment_feature=1, alpha=0.05)
clf.fit(X, [0, 0, 1, 0, 1, 1])
print(reg.segment_coef_, clf.predict_proba([[1, "new segment"]]))
```

Reference [Gross and Tibshirani, *Data Shared Lasso*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5650251/).
