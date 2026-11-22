# Pliable lasso

## Input format

`PliableRegressor` and `PliableClassifier` use numeric columns designated by
`modifier_features` as modifiers. Other input columns are predictors. The
linear predictor is

$$
\eta(x,z) = b + z^\top\gamma + \sum_j x_j(\beta_j + z^\top\theta_j).
$$

## Key parameters

`alpha` controls penalty strength and `l1_ratio` controls interaction sparsity.
The intercept and modifier main effects are unpenalized. The penalty encourages
hierarchical sparsity. Scale numeric inputs upstream when their units differ.

## Fitted outputs

Use `coef_` for input-column main effects and `interaction_coef_` for modifier
interactions. Its rows follow `predictor_indices_` and columns follow
`modifier_indices_`. Both estimators support sample weights and warm starts.
See the [worked example](real-data-example.md) for conditional slopes and paths.

## Prediction behavior

At prediction time, use the same numeric columns and modifier encoding. The classifier returns probabilities in `classes_` order.

## Regression and classification example

```python
from structured_linear import PliableClassifier, PliableRegressor

X = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
reg = PliableRegressor(modifier_features=[1], alpha=0.05)
reg.fit(X, [0, 1, 2, 0, 2, 4])
clf = PliableClassifier(modifier_features=[1], alpha=0.05)
clf.fit(X, [0, 0, 1, 0, 1, 1])
print(reg.interaction_coef_, clf.predict_proba(X))
```

Reference [Tibshirani and Friedman, *A Pliable Lasso*](https://doi.org/10.1080/10618600.2019.1648271).
