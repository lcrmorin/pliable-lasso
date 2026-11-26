# Structured Linear

[![PyPI version](https://img.shields.io/pypi/v/structured-linear)](https://pypi.org/project/structured-linear/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://lcrmorin.github.io/structured-linear/getting-started/)
[![scikit-learn 1.6+](https://img.shields.io/badge/scikit--learn-1.6%2B-F7931E?logo=scikitlearn&logoColor=white)](https://lcrmorin.github.io/structured-linear/getting-started/)
[![Tests on main](https://github.com/lcrmorin/structured-linear/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/lcrmorin/structured-linear/actions/workflows/tests.yml?query=branch%3Amain)

Structured Linear provides scikit-learn estimators for interpretable regression and binary classification. In lending, they can model effects that vary with loan characteristics, build scorecards from ordered bins, or share information across loan segments.

| What you want to model | Regressor | Binary classifier |
| --- | --- | --- |
| Predictor slopes that vary with selected modifiers | `PliableRegressor` | `PliableClassifier` |
| Scorecards that can merge adjacent bins | `FusedScorecardRegressor` | `FusedScorecardClassifier` |
| Shared slopes with sparse segment-specific adjustments | `DataSharedLassoRegressor` | `DataSharedLassoClassifier` |
| Nominal categories that can share an entire regression equation | `CategoryFusionRegressor` | `CategoryFusionClassifier` |

## Install

Install the [PyPI distribution](https://pypi.org/project/structured-linear/) with pip. The current source requires Python 3.11+ and scikit-learn 1.6+.

```bash
python -m pip install structured-linear
```

To try unreleased changes, install from GitHub `main`:

```bash
python -m pip install "structured-linear @ git+https://github.com/lcrmorin/structured-linear.git@main"
```

The Python import name is `structured_linear`.

## Fit a lending-style classifier

This toy example uses scaled duration as a predictor and loan purpose as the segment label. The final column selects the segment and is excluded from the numeric predictor matrix.

```python
from structured_linear import DataSharedLassoClassifier

applications = [
    [-1.5, "car"], [-0.5, "car"], [0.5, "car"], [1.5, "car"],
    [-1.5, "household"], [-0.5, "household"],
    [0.5, "household"], [1.5, "household"],
]
bad_credit = [0, 0, 1, 1, 1, 1, 0, 0]

model = DataSharedLassoClassifier(segment_feature=1, alpha=0.02).fit(applications, bad_credit)
new_applications = [[0.5, "car"], [0.5, "household"]]
print(model.predict_proba(new_applications)[:, 1])  # Probability of label 1.
```

For cross-validation, fit scaling or binning inside each training fold. Pliable estimators take numeric modifier columns. Fused scorecards take ordered bin codes. Data-shared estimators take a designated segment column and use the shared fit for an unseen segment.

- [API reference](https://lcrmorin.github.io/structured-linear/api/)
- [Estimator guide and preprocessing](https://lcrmorin.github.io/structured-linear/estimators/)
- [Data-shared lasso on German Credit](https://lcrmorin.github.io/structured-linear/data-shared-example/)
- [Fused scorecard walkthrough](https://lcrmorin.github.io/structured-linear/fused-example/)
- [Category fusion guide](https://lcrmorin.github.io/structured-linear/category-fusion/)
- [Classifier benchmark](https://lcrmorin.github.io/structured-linear/benchmark/)

## When not to use it

Start with ordinary linear or logistic regression if a single equation is
enough. Use Structured Linear when the structure itself matters: fewer
scorecard bands, shared equations, or an explanation of how effects change
across groups. The German Credit benchmark does not establish a predictive
advantage over simpler models.

## Hardware compatibility

[![NVIDIA CUDA float32 and float64](https://img.shields.io/badge/NVIDIA%20CUDA-float32%20%7C%20float64-237B2D)](https://lcrmorin.github.io/structured-linear/estimators/#pytorch-hardware-verification)
[![Apple MPS float32](https://img.shields.io/badge/Apple%20MPS-float32-555555)](https://lcrmorin.github.io/structured-linear/estimators/#pytorch-hardware-verification)

The badges link to tested PyTorch configurations, not continuous CI.
CUDA and Apple Silicon MPS checks are run on demand. Only the two Pliable
estimators advertise scikit-learn's `array_api_support` tag; the other six
have tested native paths but do not advertise that tag.

## Install or build from source

From a checkout of the repository, run `python -m pip install .`. To build a wheel and source distribution in `dist/`, run the following commands.

```bash
python -m pip install build
python -m build
```

## Original papers

- Robert Tibshirani and Jerome Friedman (2020), [*A Pliable Lasso*](https://doi.org/10.1080/10618600.2019.1648271).
- Robert Tibshirani, Michael Saunders, Saharon Rosset, Ji Zhu, and Keith Knight (2005), [*Sparsity and Smoothness Via the Fused Lasso*](https://doi.org/10.1111/j.1467-9868.2005.00490.x). Structured Linear uses the successive-difference penalty on ordered bin effects.
- Samuel M. Gross and Robert Tibshirani (2016), [*Data Shared Lasso: A Novel Tool to Discover Uplift*](https://doi.org/10.1016/j.csda.2016.02.015).
- Chaegeun Song, Zhong Zheng, Bing Li, and Lingzhou Xue (2025), [*Collapsing Categories for Regression with Mixed Predictors*](https://arxiv.org/abs/2511.06542).

The Structured Linear implementation is distributed under the [BSD 3-Clause license](https://github.com/lcrmorin/structured-linear/blob/main/LICENSE).
