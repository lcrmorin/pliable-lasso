# Classifier benchmark

**What did we learn?** On German Credit, the first comparison found little
difference between the structured classifiers and ridge logistic regression.
On synthetic problems, each structured classifier performed best when the
data matched the pattern it was designed to capture. In a
[separate German Credit split](#category-fusion-on-german-credit), pooled
logistic regression beat category fusion.

## Ridge versus three structured classifiers

The table reports average log loss on held-out rows. Log loss evaluates
predicted probabilities and penalizes confident mistakes. Lower is better.

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

## How the comparison works

All four models use the same five evaluation splits, stratified by outcome
label. Within each training split, three-fold cross-validation picks from
four penalty strengths by log loss. Scaling and binning are fitted only on
the training rows for each fit. All rows have equal weight. The scores are
averages of the five held-out fold losses.

The [data-shared lasso demo](data-shared-example.md) answers a different
question: how does the model compare with fitting one equation per segment?
It stratifies by both segment and outcome and scores the combined held-out
predictions, so the numbers are not directly comparable.

## Reproduce the comparison

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

Cross-validation picked three separate equations for category fusion.
Increasing `alpha` to `0.03` gave one shared equation at nearly the same
cross-validation loss. On the holdout, the pooled baseline had the lowest log
loss. This is one split, not a general ranking of the models.

Run `python benchmarks/compare_category_fusion.py` from an installed checkout.
Use `--german-credit-archive /path/to/archive.zip` to run offline.
