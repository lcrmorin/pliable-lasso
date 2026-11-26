import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_is_fitted, validate_data

from ._array_lasso import ArrayLassoDesign, _fit_inputs
from ._pliable import _array_api_features, _host_input
from ._solver import _namespace, _sigmoid
from ._sparse_logistic import _logistic_lasso


from ._validation import (
    _BinaryClassifier,
    _binary_target,
    _cpu_input,
    _positive,
    _validate_solver_controls,
    _weighted_rows,
)


class _FusedScorecard(BaseEstimator):
    """Additive scorecard with an L1 penalty on adjacent differences.

    alpha multiplies the sum of absolute adjacent score differences across
    features. The first bin of each feature has a zero reference score.
    Pass bin_order to declare the numeric codes in their intended order.
    Optional path_alphas fits a warm-started path including alpha.
    """

    def __init__(self, *, alpha=0.01, path_alphas=None, bin_order=None, max_iter=1000, tol=1e-8):
        self.alpha = alpha
        self.path_alphas = path_alphas
        self.bin_order = bin_order
        self.max_iter = max_iter
        self.tol = tol

    def _encode(self, X):
        for column, levels in zip(X.T, self.bin_levels_):
            order = np.argsort(levels)
            sorted_levels = levels[order]
            positions = np.searchsorted(sorted_levels, column)
            if np.any(positions == len(levels)) or np.any(sorted_levels[positions] != column):
                raise ValueError("Encountered a bin code outside the fitted bin order.")
            yield order[positions]

    def _design(self, codes):
        # One indicator per boundary. Its coefficient is the adjacent difference.
        row_blocks, counts = [], []
        for column, levels in zip(codes, self.bin_levels_):
            n_boundaries = len(levels) - 1
            boundaries, rows = np.nonzero(np.arange(n_boundaries)[:, None] < column)
            row_blocks.append(rows)
            counts.append(np.bincount(boundaries, minlength=n_boundaries))
        indices = np.concatenate(row_blocks)
        indptr = np.r_[0, np.cumsum(np.concatenate(counts))]
        matrix = sparse.csc_matrix(
            (np.ones(indices.size), indices, indptr),
            shape=(len(column), len(indptr) - 1),
        )
        return matrix.asformat(self._design_format)

    def _fit_path(self, X, y, weight):
        _validate_solver_controls(self.max_iter, self.tol)
        _positive(self.alpha, "alpha")
        alphas = [] if self.path_alphas is None else list(self.path_alphas)
        for alpha in alphas:
            _positive(alpha, "path_alphas entries")
        self.alphas_ = np.array(sorted(set([self.alpha, *alphas]), reverse=True))
        self.bin_levels_ = [np.unique(column) for column in X.T]
        if self.bin_order is not None:
            if len(self.bin_order) != X.shape[1]:
                raise ValueError("bin_order must specify one sequence per feature.")
            levels_by_feature = [np.array(levels, dtype=float) for levels in self.bin_order]
            for levels in levels_by_feature:
                if (levels.ndim != 1 or not levels.size or not np.isfinite(levels).all()
                        or np.unique(levels).size != levels.size):
                    raise ValueError("Each bin order must contain unique finite numeric codes.")
            self.bin_levels_ = levels_by_feature
        design = self._design(self._encode(X))
        offsets = np.cumsum([len(levels) - 1 for levels in self.bin_levels_])
        initial = None
        self.intercept_path_ = []
        self.bin_score_path_ = []
        for alpha in self.alphas_:
            intercept, differences, initial = self._solve(design, y, weight, alpha, initial)
            scores = tuple(
                np.r_[0.0, np.cumsum(block)]
                for block in np.split(differences, offsets[:-1])
            )
            self.intercept_path_.append(intercept)
            self.bin_score_path_.append(scores)
            if alpha == self.alpha:
                self.intercept_ = intercept
                self.bin_scores_ = scores
        self.intercept_path_ = np.asarray(self.intercept_path_)
        return self

    def _encode_array(self, X, xp):
        for i, levels in enumerate(self.bin_levels_):
            column = X[:, i]
            order = xp.argsort(levels)
            sorted_levels = xp.take(levels, order, axis=0)
            positions = xp.searchsorted(sorted_levels, column)
            safe = xp.minimum(positions, xp.full_like(positions, levels.shape[0] - 1))
            if not bool(xp.all(xp.take(sorted_levels, safe, axis=0) == column)):
                raise ValueError("Encountered a bin code outside the fitted bin order.")
            yield xp.take(order, safe, axis=0)

    def _fit_array_api(self, X, y, sample_weight, logistic):
        _validate_solver_controls(self.max_iter, self.tol)
        _positive(self.alpha, "alpha")
        alphas = [] if self.path_alphas is None else list(self.path_alphas)
        for alpha in alphas:
            _positive(alpha, "path_alphas entries")
        X, y, weight = _fit_inputs(self, X, y, sample_weight, logistic)
        xp = _namespace(X)
        if self.bin_order is None:
            self.bin_levels_ = [xp.unique_values(X[:, i]) for i in range(X.shape[1])]
        else:
            if len(self.bin_order) != X.shape[1]:
                raise ValueError("bin_order must specify one sequence per feature.")
            self.bin_levels_ = []
            for levels in self.bin_order:
                levels = xp.asarray(levels, dtype=X.dtype, device=X.device)
                if (levels.ndim != 1 or not levels.shape[0]
                        or not bool(xp.all(xp.isfinite(levels)))
                        or xp.unique_values(levels).shape != levels.shape):
                    raise ValueError("Each bin order must contain unique finite numeric codes.")
                self.bin_levels_.append(levels)
        codes = list(self._encode_array(X, xp))
        counts = [levels.shape[0] for levels in self.bin_levels_]
        rows = [[xp.nonzero(column == level)[0] for level in range(1, count)]
                for column, count in zip(codes, counts)]
        offsets = np.r_[0, np.cumsum([count - 1 for count in counts])]

        def bin_scores(coefficients):
            return tuple(
                xp.cumulative_sum(coefficients[int(start):int(end)], include_initial=True)
                for start, end in zip(offsets[:-1], offsets[1:])
            )

        def forward(coefficients):
            return sum(xp.take(scores, column, axis=0)
                       for scores, column in zip(bin_scores(coefficients), codes))

        def adjoint(residual):
            gradients = []
            for groups in rows:
                if groups:
                    total = xp.stack([xp.sum(xp.take(residual, group, axis=0)) for group in groups])
                    gradients.append(xp.flip(xp.cumulative_sum(xp.flip(total))))
            return xp.concat(gradients) if gradients else xp.zeros_like(residual[:0])

        self.alphas_ = xp.asarray(sorted(set([self.alpha, *alphas]), reverse=True),
                                  dtype=X.dtype, device=X.device)
        design = ArrayLassoDesign(weight, int(offsets[-1]), forward, adjoint, logistic)
        path_intercepts, self.bin_score_path_ = [], []
        params = None
        for alpha in sorted(set([self.alpha, *alphas]), reverse=True):
            params = design.fit(y, alpha, self.max_iter, self.tol, params)
            scores = bin_scores(params[:-1])
            path_intercepts.append(params[-1])
            self.bin_score_path_.append(scores)
            if alpha == self.alpha:
                self.intercept_, self.bin_scores_ = params[-1], scores
        self.intercept_path_ = xp.stack(path_intercepts)
        return self

    def _array_contributions(self, X):
        check_is_fitted(self, "bin_scores_")
        xp = _namespace(X)
        X = _array_api_features(X, xp)
        fitted = self.bin_scores_[0]
        if (xp is not _namespace(fitted) or X.shape[1] != self.n_features_in_
                or X.device != fitted.device or X.dtype != fitted.dtype):
            raise ValueError("Prediction features must match the fitted namespace, shape, dtype, and device.")
        effects = [xp.take(scores, codes, axis=0)
                   for codes, scores in zip(self._encode_array(X, xp), self.bin_scores_)]
        return xp.stack([xp.zeros_like(X[:, 0]) + self.intercept_, *effects], axis=1)

    def _linear_prediction(self, X):
        if _namespace(X) is not np:
            return _namespace(X).sum(self._array_contributions(X), axis=1)
        check_is_fitted(self, "bin_scores_")
        X = validate_data(self, _cpu_input(X), reset=False, dtype=float)
        scores = np.full(len(X), self.intercept_)
        for codes, effects in zip(self._encode(X), self.bin_scores_):
            scores += effects[codes]
        return scores

    def predict_contributions(self, X):
        """Return the intercept and one additive bin effect per feature."""
        if _namespace(X) is not np:
            return self._array_contributions(X)
        check_is_fitted(self, "bin_scores_")
        values = validate_data(self, _cpu_input(X), reset=False, dtype=float)
        effects = [scores[codes] for codes, scores in zip(self._encode(values), self.bin_scores_)]
        return np.column_stack((np.full(len(values), self.intercept_), *effects))

    def score_table(self):
        """Return one row per input bin, with its additive effect and merged-bin ID."""
        check_is_fitted(self, "bin_scores_")
        names = getattr(self, "feature_names_in_", range(self.n_features_in_))
        rows = []
        for name, levels, scores in zip(names, self.bin_levels_, self.bin_scores_):
            levels, scores = _host_input(levels), _host_input(scores)
            groups = np.r_[0, np.cumsum(np.diff(scores) != 0)]
            rows.extend(
                {"feature": name, "bin": level, "merged_bin": int(group),
                 self._effect_name: float(score)}
                for level, group, score in zip(levels, groups, scores)
            )
        return rows

class FusedScorecardClassifier(_BinaryClassifier, _FusedScorecard):
    """Fit an adjacent-bin L1 penalty with mean binary logistic loss."""

    _effect_name = "log_odds"
    _design_format = "csr"

    def fit(self, X, y, sample_weight=None):
        """Fit with optional scalar or per-row nonnegative weights."""
        if _namespace(X) is not np:
            return self._fit_array_api(X, y, sample_weight, logistic=True)
        X, y = validate_data(self, _cpu_input(X), _cpu_input(y), dtype=float)
        X, y, weight = _weighted_rows(X, y, sample_weight)
        self.classes_, y = _binary_target(y)
        return self._fit_path(X, y, weight)

    def _solve(self, design, y, weight, alpha, initial):
        return _logistic_lasso(design, y, weight, alpha, self.max_iter, self.tol, initial)

    def decision_function(self, X):
        return self._linear_prediction(X)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        xp = _namespace(scores)
        return xp.stack((_sigmoid(-scores, xp), _sigmoid(scores, xp)), axis=1)

    def predict(self, X):
        scores = self.decision_function(X)
        return _namespace(scores).where(scores > 0, self.classes_[1], self.classes_[0])


class FusedScorecardRegressor(RegressorMixin, _FusedScorecard):
    """Fit an adjacent-bin L1 penalty with half mean squared error."""

    _effect_name = "effect"
    _design_format = "csc"

    def fit(self, X, y, sample_weight=None):
        """Fit with optional scalar or per-row nonnegative weights."""
        if _namespace(X) is not np:
            return self._fit_array_api(X, y, sample_weight, logistic=False)
        X, y = validate_data(self, _cpu_input(X), _cpu_input(y), dtype=float, y_numeric=True)
        X, y, weight = _weighted_rows(X, y, sample_weight)
        return self._fit_path(X, y, weight)

    def _solve(self, design, y, weight, alpha, model):
        if design.shape[1] == 0:
            return float(weight @ y), np.empty(0), None
        if model is None:
            model = Lasso(warm_start=True, max_iter=self.max_iter, tol=self.tol)
        model.set_params(alpha=alpha).fit(design, y, sample_weight=weight)
        return float(model.intercept_), model.coef_, model

    def predict(self, X):
        return self._linear_prediction(X)
