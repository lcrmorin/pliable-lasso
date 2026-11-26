# Changelog

## 0.2.0

- Add category-fusion regression and binary classification, with learned
  category groups, effect tables, regularization paths, and a German Credit benchmark.
- Add prediction contributions for all eight estimators and fitted effect
  tables for data-shared lasso.
- Support native Array API numerical kernels across all four estimator
  families. PyTorch fit, prediction, and grid-search checks passed on CUDA
  in float32 and float64 and on Apple Silicon MPS in float32.
- Advertise scikit-learn's `array_api_support` tag for the two pliable
  estimators. The other six retain `False` pending broader conformance.
- Export fitted scoring equations as dictionaries and SQL. Document
  fitted pipeline persistence with joblib.
- Require Python 3.11 or newer. The scikit-learn minimum remains 1.6.
