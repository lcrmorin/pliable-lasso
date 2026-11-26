# Classifier benchmark

The first comparison covers the original three structured classifier families
and a ridge baseline. A separate [category-fusion comparison](#category-fusion-on-german-credit)
appears below.

The benchmark uses five outer folds and three inner folds. Each model uses the
same splits and four candidate regularization strengths. Inner folds select the
strength by log loss. Scalers and binners are fitted within each training fold.

For a focused comparison with separate models and segment-level scores, see
the [data-shared lasso demo](data-shared-example.md). That demo uses
segment-and-label stratification and computes log loss on the combined
out-of-fold predictions. The benchmark below stratifies by label and
averages the individual fold losses.

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

## Category fusion on German Credit

This separate comparison uses five-fold cross-validation on 75% of German
Credit to select one penalty per model, followed by evaluation on the
remaining 25%. Both splits use segment-and-label stratification. Each fit
scales the seven numeric predictors using its training rows. The pooled
logistic model uses only those predictors, and the separate logistic models
fit one equation per segment. The two structured models use the segment as
their designated category column.

| Category-fusion alpha | CV log loss | Groups in each fold |
| ---: | ---: | ---: |
| 0.0003 | 0.5952 | 3 |
| 0.001 | 0.5937 | 3 |
| 0.003 | 0.5903 | 3 |
| 0.01 | 0.5858 | 3 |
| 0.03 | 0.5859 | 1 |
| 0.1 | 0.5859 | 1 |

| Model | Selected alpha | Holdout log loss | Car | Household | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pooled | 0.1 | 0.5999 | 0.6300 | 0.5705 | 0.6266 |
| Separate | 0.1 | 0.6015 | 0.6314 | 0.5703 | 0.6335 |
| Data-shared lasso | 0.01 | 0.6040 | 0.6382 | 0.5758 | 0.6189 |
| Category fusion | 0.01 | 0.6137 | 0.6471 | 0.5862 | 0.6281 |

Minimum CV loss selected three distinct equations for category fusion. At
`alpha=0.03`, all three segments shared one equation at nearly the same
CV loss. These results describe one held-out split.

Run `python benchmarks/compare_category_fusion.py` from an installed checkout.
Use `--german-credit-archive /path/to/archive.zip` to run offline.
