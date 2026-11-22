# Classifier benchmark

The benchmark uses five outer folds and three inner folds. Each model uses the
same splits and four candidate regularization strengths. Inner folds select the
strength by log loss. Scalers and binners are fitted within each training fold.

## Mean held-out log loss

Lower is better.

| Dataset or scenario | Ridge | Fused scorecard | Data-shared lasso | Pliable lasso |
| --- | ---: | ---: | ---: | ---: |
| Breast cancer | 0.1579 | 0.1704 | 0.1600 | 0.1603 |
| Stepwise main effects | 0.5212 | 0.4737 | 0.5176 | 0.5167 |
| Segment-specific slopes | 0.6141 | 0.6252 | 0.5645 | 0.5663 |
| Continuously varying slopes | 0.5991 | 0.6082 | 0.6043 | 0.4993 |

Ridge led on the breast cancer dataset, where radius thresholds define
illustrative segments. Each structured classifier led in the synthetic setting
designed around its own assumptions. These experiments illustrate model
behavior on four settings.

From an installed checkout, run `python benchmarks/compare_classifiers.py`.
The [fold-level results](https://github.com/lcrmorin/structured-linear/blob/main/benchmarks/classification-results.csv)
also include AUC, Brier score, and fit time.
