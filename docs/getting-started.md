# Getting started

Structured Linear requires Python 3.10+ and scikit-learn 1.6+. Install from PyPI.

```bash
python -m pip install structured-linear
```

From a local checkout, run `python -m pip install .`.

## Fit a binary classifier

This example treats loan purpose as a segment label and scaled loan duration
as a numeric predictor.

```python
from structured_linear import DataSharedLassoClassifier

# Columns are scaled duration and loan purpose, in that order.
applications = [
    [-1.0, "car"], [0.0, "car"], [1.0, "car"],
    [-1.0, "household"], [0.0, "household"], [1.0, "household"],
]
bad_credit = [0, 0, 1, 0, 1, 1]
model = DataSharedLassoClassifier(segment_feature=1).fit(applications, bad_credit)

new_applications = [[0.5, "car"], [0.5, "household"]]
print(model.predict_proba(new_applications)[:, 1])  # Probability of label 1.
```

Probability columns follow `classes_`. In a real workflow, fit any scaling
or binning on the training split before transforming new rows.

- [Pliable lasso](pliable.md) for slopes that vary with modifiers
- [Fused scorecards](fused.md) for adjacent bins that share an effect
- [Data-shared lasso](data-shared.md) for shared and segment-specific slopes
