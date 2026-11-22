# Classifier benchmark

The benchmark uses five outer folds and three inner folds. Each model uses the
same splits and four candidate regularization strengths. Inner folds select the
strength by log loss. Scalers and binners are fitted within each training fold.

For a focused comparison with separate models and segment-level scores, see
the [data-shared lasso demo](data-shared-example.md).

## Mean held-out log loss

Lower is better.

| Dataset or scenario | Ridge | Fused scorecard | Data-shared lasso | Pliable lasso |
| --- | ---: | ---: | ---: | ---: |
| German Credit | 0.5884 | 0.5905 | 0.5927 | 0.5859 |
| Stepwise main effects | 0.5212 | 0.4737 | 0.5176 | 0.5167 |
| Segment-specific slopes | 0.6141 | 0.6252 | 0.5645 | 0.5663 |
| Continuously varying slopes | 0.5991 | 0.6082 | 0.6043 | 0.4993 |

German Credit has 1,000 rows, including 300 labeled as bad credit. This
experiment uses seven numeric predictors and three groups derived from
loan purpose. Groups distinguish car purchases, household purchases,
and other purposes. Age is the continuous modifier. Mean held-out AUC was
0.632 for ridge, 0.635 for fused scorecards, 0.632 for data-shared lasso,
and 0.639 for pliable lasso. The differences on this dataset are small.

Each structured classifier led on log loss in the synthetic setting
designed around its assumptions. All reported fits use equal row weights.

From an installed checkout, run `python benchmarks/compare_classifiers.py`.
This downloads the archive from UCI. To run offline, pass
`--german-credit-archive /path/to/archive.zip`.

Dataset attribution [Hans Hofmann, Statlog (German Credit Data), UCI Machine Learning Repository](https://doi.org/10.24432/C5NC77).
The dataset is licensed under CC BY 4.0.

The [fold-level results](https://github.com/lcrmorin/structured-linear/blob/main/benchmarks/classification-results.csv)
also include AUC, Brier score, and fit time.
