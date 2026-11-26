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

## Experimental Array API inputs

Both estimators accept dense arrays exposing `__array_namespace__` with
`float32` or `float64` features. The end-to-end path is tested with
`array-api-strict` on CPU. For classification, labels on this path must be
numeric or boolean. Use the same namespace, dtype, and device at fit and
prediction time.

```python
import array_api_strict as xp
from structured_linear import PliableRegressor

X = xp.asarray([[0., 0.], [1., 0.], [2., 0.], [0., 1.], [1., 1.], [2., 1.]])
y = xp.asarray([0., 1., 2., 0., 2., 4.])
model = PliableRegressor(modifier_features=[1], alpha=0.02).fit(X, y)
prediction = model.predict(X)
```

Predictions and coefficient arrays stay in the input namespace. Scalar
diagnostics such as `objective_` are Python values. This support covers
direct fit and inference calls. Both estimators also work with
`GridSearchCV` using scikit-learn's default scoring and integer `cv`
on `array-api-strict` CPU arrays with dispatch enabled.

Set `SCIPY_ARRAY_API=1` before starting Python with scikit-learn 1.6.
Using the arrays from the example above:

```python
from sklearn import config_context
from sklearn.model_selection import GridSearchCV

with config_context(array_api_dispatch=True):
    search = GridSearchCV(
        PliableRegressor(modifier_features=[1]),
        {"alpha": [0.02, 0.2]},
        cv=3,
    ).fit(X, y)
    prediction = search.predict(X)
```

The estimator tag `array_api_support` remains `False`. The checks here
cover this grid-search configuration. GPU grid search and other
meta-estimator configurations remain unverified.

## Try PyTorch MPS

Use float32 features and regression targets on an MPS device. Classification
labels can be integers. Run the test with CPU fallback disabled so unsupported
operators surface explicitly.

```bash
python -m pip install -e '.[test]' torch
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest -q tests/test_pliable_torch.py -k mps
```

The test checks both estimators against NumPy. MPS cases skip if PyTorch
reports that the backend is unavailable.

For GitHub Actions, use the **Tests** workflow's manual `mps` backend on a
dedicated Apple Silicon runner labeled `structured-linear-mps`. The
workflow checks that PyTorch can allocate a Metal tensor before running
pytest. A standard GitHub-hosted arm64 Mac failed that device check.

## Try PyTorch CUDA

Install a CUDA-enabled PyTorch build for your NVIDIA environment. The parity
checks cover weighted fits, warm starts, and inference in float32 and float64.

```bash
python -m pytest -q tests/test_pliable_torch.py -k cuda
python benchmarks/benchmark_pliable_cuda.py --dtype float32
python benchmarks/benchmark_pliable_cuda.py --dtype float64
```

The benchmark synchronizes the device for fit and inference timings and reports
initial input transfer and peak additional PyTorch-allocated device memory
separately. It compares predictions and objectives against PyTorch CPU. CUDA
checks require a machine reporting `torch.cuda.is_available()`.

In GitHub Actions, manually run the **Tests** workflow with backend
`cuda` and the branch you want to validate. Configure a dedicated Linux
runner with label `structured-linear-cuda` and a Python environment
containing CUDA-enabled PyTorch first. Ordinary pushes and pull
requests run the CPU jobs.

Reference [Tibshirani and Friedman, *A Pliable Lasso*](https://doi.org/10.1080/10618600.2019.1648271).

See the [prediction breakdown guide](estimators.md#explain-individual-predictions) for per-row predictor contributions.
