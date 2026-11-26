# Category fusion

These estimators require the [GitHub source installation](getting-started.md).

Use category fusion when a nominal predictor defines segments and some
segments may share an entire regression equation. The model fits a vector
of intercept and slopes for each level, then penalizes the Euclidean
distance between every pair of vectors. Levels have no required ordering.

For segment vectors $b_g$, augmented predictor rows $z_i=(1,x_i)$, and
normalized sample weights $w_i$, the objective is

$$
\sum_i w_i\,\ell(y_i,z_i^\top b_{g_i})
+ \alpha\sum_{g<h}\lVert b_g-b_h\rVert_2.
$$

`CategoryFusionRegressor` uses half squared error.
`CategoryFusionClassifier` uses binary logistic loss.
The fusion penalty includes the intercept. Put numeric predictors on
comparable scales before fitting if their coefficients should contribute
similarly to the penalty.

```python
import numpy as np
from structured_linear import CategoryFusionRegressor

X = np.array([
    [-1, "north"], [1, "north"],
    [-1, "west"], [1, "west"],
    [-1, "south"], [1, "south"],
], dtype=object)
y = [-1, 1, -1, 1, 4, 2]

model = CategoryFusionRegressor(segment_feature=-1, alpha=0.01).fit(X, y)
print(dict(zip(model.segments_, model.cluster_labels_)))
print(model.predict([[0.5, "north"]]))
```

`segments_` gives the order of rows in `coef_` and entries in `intercept_`.
`cluster_labels_` reports connected components of pairs within
`fusion_tol` in coefficient distance. Predictions use the fitted
coefficients at solver precision. Prediction requires a segment seen
during fitting. For a shared fallback for unseen segments, see
[data-shared lasso](data-shared.md). The complete-graph penalty has quadratically many pairs
in the number of levels, so start with modest cardinality.

This uses equal penalty weights for all category pairs for one categorical
predictor, with optional observation weights. The solver uses primal-dual
splitting. Method reference [Song, Zheng, Li and Xue (2025),
*Collapsing Categories for Regression with Mixed Predictors*](https://arxiv.org/abs/2511.06542).
