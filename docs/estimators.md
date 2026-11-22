# Estimator guide

All six estimators use additive linear predictors. Regression fits a squared-error loss divided by two. Classification fits binary logistic loss. The structural penalty determines how effects are shared. All models fit an intercept, which is unpenalized.

## Pliable lasso

`PliableRegressor` and `PliableClassifier` use numeric columns designated by
`modifier_features` as modifiers. Other input columns are predictors. The
linear predictor is

$$
\eta(x,z) = b + z^\top\gamma + \sum_j x_j(\beta_j + z^\top\theta_j).
$$

`alpha` controls penalty strength and `l1_ratio` controls interaction sparsity.
The intercept and modifier main effects are unpenalized. The penalty encourages
hierarchical sparsity. Scale numeric inputs upstream when their units differ.

Use `coef_` for input-column main effects and `interaction_coef_` for modifier
interactions. Its rows follow `predictor_indices_` and columns follow
`modifier_indices_`. Both estimators support sample weights and warm starts.
See the [worked example](real-data-example.md) for conditional slopes and paths.

Reference [Tibshirani and Friedman, *A Pliable Lasso*](https://doi.org/10.1080/10618600.2019.1648271).

## Fused scorecards

`FusedScorecardRegressor` and `FusedScorecardClassifier` take numeric bin codes.
Fit a binner upstream. By default, codes are sorted numerically within each
feature. Use `bin_order` to declare a different order or to allow codes absent
from training. Codes outside the fitted order are rejected.

For bin effects $s_{jk}$, the penalty is

$$
\alpha\sum_j\sum_{k=2}^{K_j}|s_{jk}-s_{j,k-1}|.
$$

The first bin in each feature has a zero reference effect. `score_table()`
reports contiguous merged bands and their effects in target units for
regression and log odds for classification. Optional `path_alphas` adds
penalties to `alphas_`, `intercept_path_`, and `bin_score_path_`. Merges may
change along the path. An explicitly declared bin absent in training has no
direct evidence for its effect.

Reference [the adjacent-level fusion penalty in the R smurf vignette](https://cran.r-project.org/web/packages/smurf/vignettes/smurf.html).

## Data-shared lasso

`DataSharedLassoRegressor` and `DataSharedLassoClassifier` take a segment
column selected by `segment_feature` and finite numeric predictors. Segment
labels can be strings or finite numbers. The slope in segment $g$ is
$\beta + \Delta_g$, with penalty

$$
\alpha\left(\lVert\beta\rVert_1
  + w\sum_g\lVert\Delta_g\rVert_1\right).
$$

`deviation_weight` sets $w$. Larger values penalize departures from shared
slopes more strongly. The fitted values are available in `shared_coef_`,
`segment_deviations_`, and `segment_coef_`, with rows indexed by `segments_`.
An unseen segment uses the shared slopes and global intercept. The
shared/deviation decomposition can be nonunique for some penalty settings.
Add segment indicators as predictor columns if you need segment-specific
baseline shifts.

Reference [Gross and Tibshirani, *Data Shared Lasso*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5650251/).
