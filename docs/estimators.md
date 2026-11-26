# Estimator guide

Each family has a regressor and a binary classifier. Regressors predict in target
units. Classifiers expose `decision_function` for log odds and `predict_proba`
with columns ordered by `classes_`. All estimators support scikit-learn's
`get_params`, `set_params`, and cloning.

| Capability | Pliable lasso | Fused scorecards | Data-shared lasso | Category fusion |
| --- | --- | --- | --- | --- |
| Model-specific input | Numeric modifier columns | Ordered numeric bin codes | Segment label column | One nominal segment column |
| Penalty strength | `alpha >= 0` | `alpha > 0` | `alpha > 0` | `alpha > 0` |
| Sample weights | Both estimators | Both estimators | Both estimators | Both estimators |
| Intercept | On by default, configurable | Always fitted | Always fitted | Always fitted and included in fusion |
| Reuse a solution | Optional warm start across fits | Warm starts within a requested penalty path | No | No |
| Main fitted effects | `coef_`, `interaction_coef_` | `bin_scores_`, `score_table()` | `shared_coef_`, `segment_deviations_`, `segment_coef_` | `coef_`, `intercept_`, `cluster_labels_` |

## Preprocessing and regularization

- **Pliable lasso.** Scale continuous predictors and modifiers upstream when their units differ. Encode categorical modifiers numerically. Scaling changes the penalty applied to interaction effects.
- **Fused scorecards.** Bin continuous inputs upstream, then pass the ordered codes directly. Column scaling is unnecessary. Declare `bin_order` when numeric sort order differs from the intended order or when the binner can emit codes absent from the fit.
- **Data-shared lasso.** Scale continuous predictors upstream and pass the segment column through unchanged. The segment label selects a coefficient vector. Keep its column at the position specified by `segment_feature`.
- **[Category fusion](category-fusion.md).** Scale numeric predictors upstream and pass the nominal segment column through unchanged. The penalty can fuse entire segment equations, including their intercepts.

Fit preprocessing within each training fold, typically using a pipeline.
Tune `alpha` separately for each estimator on your chosen metric. The same
numeric value has a different effect across penalties. `max_iter` and `tol`
control each family's solver, whose stopping criteria differ.

## Sample weights

All estimators accept a finite, nonnegative scalar or one weight per
training row via `fit(X, y, sample_weight=weights)`. At least one row must
have positive weight. The loss is averaged by total weight, so multiplying
every weight by the same positive constant leaves the objective unchanged.
The penalty strength `alpha` keeps the same meaning.

Zero-weight rows contribute no loss. Fused scorecards also exclude them
when inferring the bin order. Data-shared lasso and category fusion exclude
them when discovering segments. For binary classification, both classes need
positive total weight.

For example, with metadata routing disabled, pass weights to the named
model step as `pipeline.fit(X, y, model__sample_weight=weights)`.
This weights the estimator's loss. Fit any weighted preprocessing
separately or route the weights to the preprocessing step as well.

## Explain individual predictions

Use `model.predict_contributions(X)` to get an array with one row per
observation. The first column is the intercept. Each subsequent column
is an additive contribution in input-column order.

| Family | Contribution columns after the intercept |
| --- | --- |
| Pliable lasso | One per input feature. Interaction terms are assigned to the predictor whose slope the modifier changes. |
| Fused scorecards | One per input feature, containing the fitted effect of its bin. |
| Data-shared lasso | One per numeric predictor. An unseen segment uses the shared slopes. |
| Category fusion | One per numeric predictor. Each row uses the fitted equation for its segment. |

For the two segment models, the segment column selects the equation and
is omitted from the output. Contributions sum to the regression prediction
or the classifier's log odds, up to floating-point precision. They describe
the fitted equation and depend on the preprocessing applied to its inputs.

### Reconstruct a classifier score

This fused scorecard uses one ordered bin column. Pandas is optional and
labels the output for inspection.

```python
import numpy as np
import pandas as pd
from scipy.special import expit
from structured_linear import FusedScorecardClassifier

X = pd.DataFrame({"duration_bin": [0, 0, 0, 1, 1, 1]})
y = [0, 0, 1, 0, 1, 1]
model = FusedScorecardClassifier(alpha=0.01).fit(X, y)

parts = pd.DataFrame(
    model.predict_contributions(X),
    index=X.index,
    columns=["intercept", *model.feature_names_in_],
)
log_odds = parts.sum(axis=1)
np.testing.assert_allclose(log_odds, model.decision_function(X))
np.testing.assert_allclose(expit(log_odds), model.predict_proba(X)[:, 1])
print(parts.drop_duplicates())
```

For a fitted pipeline, transform the inputs through its preprocessing steps before
calling `predict_contributions` on the final estimator. Label the contributions
using the transformed feature order. For classifiers, sum the contributions
before applying the logistic function to obtain the probability of `classes_[1]`.

## Pliable lasso

[PliableRegressor and PliableClassifier](pliable.md)

## Fused scorecards

[FusedScorecardRegressor and FusedScorecardClassifier](fused.md)

## Data-shared lasso

[DataSharedLassoRegressor and DataSharedLassoClassifier](data-shared.md)

## Category fusion

[CategoryFusionRegressor and CategoryFusionClassifier](category-fusion.md)

## Array API and grid search

All eight estimators can run with scikit-learn's
`config_context(array_api_dispatch=True)` and `GridSearchCV` on
`array-api-strict` CPU arrays. The tests use integer `cv` and default scoring.
See the [pliable example](pliable.md#experimental-array-api-inputs) for setup.

All four families use native Array API numerical kernels on numeric
Array API inputs. Fused scorecards and data-shared lasso apply their
design products without materializing an expanded feature matrix.
Category fusion uses the CPU for segment and cluster metadata. Use
numeric or boolean class labels and numeric segment codes with Array
API arrays.

The pliable estimators advertise `array_api_support=True`. The other six
estimators retain `False` pending upstream conformance and real GPU
validation.
