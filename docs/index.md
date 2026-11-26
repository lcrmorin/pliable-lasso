# Structured Linear

Structured Linear provides four families of scikit-learn estimators for interpretable regression and binary classification.

| Model family | Use it when | Estimators |
| --- | --- | --- |
| Pliable lasso | Predictor slopes vary with numeric or encoded modifiers | `PliableRegressor`, `PliableClassifier` |
| Fused scorecard | Adjacent ordered bins can share an effect | `FusedScorecardRegressor`, `FusedScorecardClassifier` |
| Data-shared lasso | Predefined segments share slopes and need a few deviations | `DataSharedLassoRegressor`, `DataSharedLassoClassifier` |
| [Category fusion](category-fusion.md) | Some nominal categories can share an intercept and all slopes | `CategoryFusionRegressor`, `CategoryFusionClassifier` |

Start with [installation and a runnable example](getting-started.md). The [estimator guide](estimators.md) describes the objectives, inputs, and fitted attributes. The [benchmark](benchmark.md) compares the original three classifier families on common splits.

Structured Linear is distributed under the [BSD 3-Clause license](https://github.com/lcrmorin/structured-linear/blob/main/LICENSE).
