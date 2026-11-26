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
| Reuse a solution | Optional warm start across fits | Warm starts within a requested penalty path | |  |
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

All estimators estimators accept a finite, nonnegative scalar or one weight per
training row via `fit(X, y, sample_weight=weights)`. At least one row must
have positive weight. The loss is averaged by total weight, so multiplying
every weight by the same positive constant leaves the objective unchanged.
The penalty strength `alpha` keeps the same meaning.

Zero-weight rows contribute no loss. Fused scorecards also exclude them
when inferring the bin order, and data-shared lasso and category fusion exclude them when
discovering segments. For binary classification, both classes need
positive total weight.

For example, with metadata routing disabled, pass weights to the named
model step as `pipeline.fit(X, y, model__sample_weight=weights)`.
This weights the estimator's loss. Fit any weighted preprocessing
separately or route the weights to the preprocessing step as well.

## Pliable lasso

[PliableRegressor and PliableClassifier](pliable.md)

## Fused scorecards

[FusedScorecardRegressor and FusedScorecardClassifier](fused.md)

## Data-shared lasso

[DataSharedLassoRegressor and DataSharedLassoClassifier](data-shared.md)
