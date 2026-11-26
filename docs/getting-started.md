# Getting started

The current source requires Python 3.11+ and scikit-learn 1.6+. Install the latest release from PyPI.

```bash
python -m pip install structured-linear
```

To try unreleased changes, install from GitHub `main`:

```bash
python -m pip install "structured-linear @ git+https://github.com/lcrmorin/structured-linear.git@main"
```

From a local checkout, run `python -m pip install .`.

## Fit a binary classifier

This example treats loan purpose as a segment label and scaled loan duration
as a numeric predictor.

```python
from structured_linear import DataSharedLassoClassifier

# Columns are scaled duration and loan purpose, in that order.
applications = [
    [-1.5, "car"], [-0.5, "car"], [0.5, "car"], [1.5, "car"],
    [-1.5, "household"], [-0.5, "household"],
    [0.5, "household"], [1.5, "household"],
]
bad_credit = [0, 0, 1, 1, 0, 1, 1, 0]
model = DataSharedLassoClassifier(segment_feature=1, alpha=0.02).fit(applications, bad_credit)

new_applications = [[0.5, "car"], [0.5, "household"]]
print(model.predict_proba(new_applications)[:, 1])  # Probability of label 1.
```

Probability columns follow `classes_`. In a real workflow, fit any scaling
or binning on the training split before transforming new rows.

- [Pliable lasso](pliable.md) for slopes that vary with modifiers
- [Fused scorecards](fused.md) for adjacent bins that share an effect
- [Data-shared lasso](data-shared.md) for shared and segment-specific slopes
- [Category fusion](category-fusion.md) for grouping nominal segments with similar regression equations

## Save and reload a fitted pipeline

Use joblib to save preprocessing and the estimator together. Load artifacts
from trusted sources in a Python environment with matching package versions.

```python
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from structured_linear import PliableRegressor

pipeline = make_pipeline(StandardScaler(), PliableRegressor(alpha=0.01))
pipeline.fit([[0], [1], [2]], [0, 1, 2])
joblib.dump(pipeline, "model.joblib")
restored = joblib.load("model.joblib")
print(restored.predict([[1.5]]))
```
