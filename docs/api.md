# API reference

All estimators are imported from `structured_linear`. See the
[estimator guide](estimators.md) for preprocessing, scoring formulas,
and tuning advice. Classifiers return probabilities in `classes_` order.
[PliableClassifier](#structured_linear.PliableClassifier) has the same constructor
parameters as [PliableRegressor](#structured_linear.PliableRegressor).

::: structured_linear.PliableRegressor
    options:
      members: [fit, predict, predict_contributions]
      inherited_members: true

::: structured_linear.PliableClassifier
    options:
      members: [fit, predict, predict_contributions, decision_function, predict_proba]
      inherited_members: true

::: structured_linear.FusedScorecardRegressor
    options:
      members: [fit, predict, score_table, predict_contributions]
      inherited_members: true

::: structured_linear.FusedScorecardClassifier
    options:
      members: [fit, predict, score_table, predict_contributions, decision_function, predict_proba]
      inherited_members: true

::: structured_linear.DataSharedLassoRegressor
    options:
      members: [fit, predict, effect_table, predict_contributions]
      inherited_members: true

::: structured_linear.DataSharedLassoClassifier
    options:
      members: [fit, predict, effect_table, predict_contributions, decision_function, predict_proba]
      inherited_members: true

::: structured_linear.CategoryFusionRegressor
    options:
      members: [fit, predict, fit_path, effect_table, predict_contributions]
      inherited_members: true

::: structured_linear.CategoryFusionClassifier
    options:
      members: [fit, predict, fit_path, effect_table, predict_contributions, decision_function, predict_proba]
      inherited_members: true
