# Export scoring equations

Use export when you want to score new rows outside Python or inspect the
equation in a format that does not require the original estimator. Use
[joblib](getting-started.md#save-and-reload-a-fitted-pipeline) instead if you
want to reload a fitted Python pipeline, including its preprocessing.

Export any fitted Structured Linear estimator with
`structured_linear.export.to_dict`. The result is a JSON-compatible
description of its scoring equation. It contains the fitted input order,
class labels, a schema version, and an equation made of an intercept,
weighted products of columns, and category or bin lookups. Segmented
models specify the equation for each segment and an optional fallback.

```python
import json
from structured_linear import FusedScorecardRegressor
from structured_linear.export import to_dict, to_sql

model = FusedScorecardRegressor().fit([[0], [0], [1], [1]], [0, 1, 4, 5])
spec = to_dict(model)
payload = json.dumps(spec, allow_nan=False)
expression = to_sql(json.loads(payload), columns=["risk_bin"])
print(expression)
```

The SQL compiler takes the dictionary, so it can run without the original
fitted object. Pass SQL column names in fitted input order. If the fitted
input had string feature names, the compiler can use those by default.
The expression returns regression predictions or binary log odds. Pass
`probability=True` for the probability of `spec["classes"][1]`; the SQL
engine then needs `EXP`. Parity is tested against SQLite.

Apply the same preprocessing as during fitting. For instance, a scorecard
expression expects fitted bin codes. SQL returns `NULL` for missing inputs,
unknown scorecard bins, and unknown category-fusion segments. Data-shared
lasso uses its shared equation for unseen non-null segments. Unlike Python
prediction, the generated SQL does not raise or warn on an unknown bin:
monitor or reject `NULL` scores explicitly in the calling query.

Use [pipeline persistence](getting-started.md#save-and-reload-a-fitted-pipeline)
to retain preprocessing and fitted Python methods.
