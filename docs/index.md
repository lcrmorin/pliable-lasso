# Structured Linear

[![PyPI version](https://img.shields.io/pypi/v/structured-linear)](https://pypi.org/project/structured-linear/)
[![Tests on main](https://github.com/lcrmorin/structured-linear/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/lcrmorin/structured-linear/actions/workflows/tests.yml?query=branch%3Amain)
[![CUDA for pliable estimators](https://img.shields.io/badge/NVIDIA%20CUDA-Pliable%20estimators-237B2D)](pliable.md#try-pytorch-cuda)
[![Apple MPS for pliable estimators](https://img.shields.io/badge/Apple%20MPS-Pliable%20estimators-555555)](pliable.md#try-pytorch-mps)

Structured Linear provides four families of scikit-learn estimators for interpretable regression and binary classification.

CUDA and Apple MPS badges describe PyTorch support for the pliable estimators
in the GitHub source. See the linked guide for device and dtype requirements.

| Model family | Use it when | Estimators |
| --- | --- | --- |
| Pliable lasso | Predictor slopes vary with numeric or encoded modifiers | `PliableRegressor`, `PliableClassifier` |
| Fused scorecard | Adjacent ordered bins can share an effect | `FusedScorecardRegressor`, `FusedScorecardClassifier` |
| Data-shared lasso | Predefined segments share slopes and need a few deviations | `DataSharedLassoRegressor`, `DataSharedLassoClassifier` |
| [Category fusion](category-fusion.md) | Some nominal categories can share an intercept and all slopes | `CategoryFusionRegressor`, `CategoryFusionClassifier` |

Start with [installation and a runnable example](getting-started.md). The [estimator guide](estimators.md) describes the objectives, inputs, and fitted attributes. The [benchmark](benchmark.md) compares the original three classifier families on common splits.

Structured Linear is distributed under the [BSD 3-Clause license](https://github.com/lcrmorin/structured-linear/blob/main/LICENSE).
