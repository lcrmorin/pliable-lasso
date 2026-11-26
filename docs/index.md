# Structured Linear

[![PyPI version](https://img.shields.io/pypi/v/structured-linear)](https://pypi.org/project/structured-linear/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](getting-started.md)
[![scikit-learn 1.6+](https://img.shields.io/badge/scikit--learn-1.6%2B-F7931E?logo=scikitlearn&logoColor=white)](getting-started.md)
[![Tests on main](https://github.com/lcrmorin/structured-linear/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/lcrmorin/structured-linear/actions/workflows/tests.yml?query=branch%3Amain)
[![NVIDIA CUDA float32 and float64](https://img.shields.io/badge/NVIDIA%20CUDA-float32%20%7C%20float64-237B2D)](estimators.md#pytorch-hardware-verification)
[![Apple MPS float32](https://img.shields.io/badge/Apple%20MPS-float32-555555)](estimators.md#pytorch-hardware-verification)

Structured Linear provides four families of scikit-learn estimators for interpretable regression and binary classification.

All eight estimators in the GitHub source passed PyTorch fit, prediction, and
grid-search checks on CUDA and Apple Silicon MPS. See the linked support table
for verified dtypes.

| Model family | Use it when | Estimators |
| --- | --- | --- |
| Pliable lasso | Predictor slopes vary with numeric or encoded modifiers | `PliableRegressor`, `PliableClassifier` |
| Fused scorecard | Adjacent ordered bins can share an effect | `FusedScorecardRegressor`, `FusedScorecardClassifier` |
| Data-shared lasso | Predefined segments share slopes and need a few deviations | `DataSharedLassoRegressor`, `DataSharedLassoClassifier` |
| [Category fusion](category-fusion.md) | Some nominal categories can share an intercept and all slopes | `CategoryFusionRegressor`, `CategoryFusionClassifier` |

Start with [installation and a runnable example](getting-started.md). The [estimator guide](estimators.md) describes the objectives, inputs, and fitted attributes. The [benchmark](benchmark.md) compares the original three classifier families on common splits.

Structured Linear is distributed under the [BSD 3-Clause license](https://github.com/lcrmorin/structured-linear/blob/main/LICENSE).
