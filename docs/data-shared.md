# Data-shared lasso

See the [German Credit demo](data-shared-example.md) for a comparison with pooled and separate models and a plot of fitted slopes.

## Input format

`DataSharedLassoRegressor` and `DataSharedLassoClassifier` take a segment
column selected by `segment_feature` and finite numeric predictors. Segment
labels can be strings or finite numbers.

## Key parameters

The slope in segment $g$ is
$\beta + \Delta_g$, with penalty

$$
\alpha\left(\lVert\beta\rVert_1
  + w\sum_g\lVert\Delta_g\rVert_1\right).
$$

`deviation_weight` sets $w$. Larger values penalize departures from shared
slopes more strongly.

## Fitted outputs

The fitted values are available in `shared_coef_`,
`segment_deviations_`, and `segment_coef_`, with rows indexed by `segments_`.

## Prediction behavior

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

## Reconstruct a segment slope

Here the training data have slope 2 in three segments and slope 5 in
segment B. The common scale and evenly spaced values make the slope comparison
direct.

```python
import numpy as np
from structured_linear import DataSharedLassoRegressor

values = np.tile(np.linspace(-2, 2, 30), 4)
segments = np.repeat(["A", "B", "C", "D"], 30)
X = np.column_stack((values, segments))
y = values * np.where(segments == "B", 5, 2)

model = DataSharedLassoRegressor(segment_feature=1, alpha=0.02).fit(X, y)
row = model.segments_.index("B")
shared = model.shared_coef_[0]
deviation = model.segment_deviations_[row, 0]
print(f"B slope: {shared:.2f} + {deviation:.2f} = {shared + deviation:.2f}")
print(model.predict([[1, "B"], [1, "unseen"]]))
```

The fitted slope for B is approximately `2.00 + 2.94 = 4.94`.
For a predictor value of 1, the unseen segment gets the shared slope plus
the fitted intercept. B also gets its fitted deviation.

Reference [Gross and Tibshirani, *Data Shared Lasso*](https://doi.org/10.1016/j.csda.2016.02.015).

## Inspect the fitted slopes

On a fitted estimator, call `model.effect_table()` for one row per
segment and numeric predictor. Each row reports the `shared` slope,
the segment's `deviation`, and the resulting `coefficient`. Feature
labels are fitted column names when available, or zero-based input-column
indices. The shared intercept is available as `model.intercept_`.

The table is a list of dictionaries. To review or export it with pandas,
continue the example above.

```python
import pandas as pd

effects = pd.DataFrame(model.effect_table())
print(effects)
effects.to_csv("effects.csv", index=False)
```

See the [prediction breakdown guide](estimators.md#explain-individual-predictions)
to inspect how these slopes contribute to an individual prediction.

## Array API inputs

Numeric Array API inputs use native products for the shared slopes and
segment deviations. The solver applies those products directly to the
features and numeric segment codes. It keeps fitted coefficients and
inference arrays on the input device. Unseen segments use the shared
slopes, as on NumPy. Strict-array tests cover weighted grid search and
the unseen-segment fallback on a simulated non-CPU device. See the [hardware support table](estimators.md#pytorch-hardware-verification)
for verified CUDA and Apple MPS configurations.
