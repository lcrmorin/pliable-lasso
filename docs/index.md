# Structured Linear

[![PyPI version](https://img.shields.io/pypi/v/structured-linear)](https://pypi.org/project/structured-linear/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](getting-started.md)
[![scikit-learn 1.6+](https://img.shields.io/badge/scikit--learn-1.6%2B-F7931E?logo=scikitlearn&logoColor=white)](getting-started.md)
[![Tests on main](https://github.com/lcrmorin/structured-linear/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/lcrmorin/structured-linear/actions/workflows/tests.yml?query=branch%3Amain)

Structured Linear is for problems where one ordinary linear or logistic model
is too rigid, but you still want to explain what each input does. Each model
family has a regressor and a binary classifier.

[Install the package and fit your first model](getting-started.md).

## Which model should I use?

| Model family | Use it when | Estimators |
| --- | --- | --- |
| [Pliable lasso](pliable.md) | An input’s effect changes with another variable | `PliableRegressor`, `PliableClassifier` |
| [Fused scorecard](fused.md) | Neighboring bands, such as income ranges, can share a score | `FusedScorecardRegressor`, `FusedScorecardClassifier` |
| [Data-shared lasso](data-shared.md) | Segments mostly behave alike but may need a few different coefficients | `DataSharedLassoRegressor`, `DataSharedLassoClassifier` |
| [Category fusion](category-fusion.md) | Some unordered categories can share the same equation | `CategoryFusionRegressor`, `CategoryFusionClassifier` |

The [API reference](api.md) lists parameters and fitted attributes.
The [estimator guide](estimators.md) compares input formats and preprocessing.
The [benchmark](benchmark.md) shows where the models helped and where a
conventional baseline did just as well.

## When not to use it

Start with ordinary linear or logistic regression if a single equation is
enough. Use Structured Linear when the structure itself matters: fewer
scorecard bands, shared equations, or an explanation of how effects change
across groups. The German Credit benchmark does not establish a predictive
advantage over simpler models.

## Hardware compatibility

[![NVIDIA CUDA float32 and float64](https://img.shields.io/badge/NVIDIA%20CUDA-float32%20%7C%20float64-237B2D)](estimators.md#pytorch-hardware-verification)
[![Apple MPS float32](https://img.shields.io/badge/Apple%20MPS-float32-555555)](estimators.md#pytorch-hardware-verification)

The badges link to tested PyTorch configurations, not continuous CI.
CUDA and Apple Silicon MPS checks are run on demand. Only the two Pliable
estimators advertise scikit-learn's `array_api_support` tag; the other six
have tested native paths but do not advertise that tag.

Structured Linear is distributed under the [BSD 3-Clause license](https://github.com/lcrmorin/structured-linear/blob/main/LICENSE).
