from numbers import Integral, Real

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

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


class _DataSharedLasso(BaseEstimator):
    """Shared slopes plus segment-specific sparse deviations."""

    def __init__(self, *, segment_feature=-1, alpha=0.01, deviation_weight=1.0,
                 max_iter=1000, tol=1e-8):
        self.segment_feature = segment_feature
        self.alpha = alpha
        self.deviation_weight = deviation_weight
        self.max_iter = max_iter
        self.tol = tol

    def _split(self, X):
        index = self.segment_feature_
        if index == X.shape[1] - 1:
            features = X[:, :-1]
        elif index == 0:
            features = X[:, 1:]
        else:
            features = np.delete(X, index, axis=1)
        features = check_array(features, dtype=float)
        segment_column = X[:, index]
        segments = segment_column.tolist()
        # Numeric input has already passed finite-value validation.
        if segment_column.dtype.kind in "iuf":
            return features, segments
        for segment in segments:
            if not isinstance(segment, (Real, str)) or (
                isinstance(segment, Real) and not np.isfinite(segment)
            ):
                raise ValueError("Segment labels must be strings or finite numbers.")
        return features, segments

    def _fit_design(self, X):
        _validate_solver_controls(self.max_iter, self.tol)
        _positive(self.alpha, "alpha")
        _positive(self.deviation_weight, "deviation_weight")
        if X.shape[1] < 2:
            raise ValueError("Got 1 feature(s); at least one predictor and one segment column are required.")
        if not isinstance(self.segment_feature, Integral) or not (
            -X.shape[1] <= self.segment_feature < X.shape[1]
        ):
            raise ValueError("segment_feature must be a valid column index.")
        self.segment_feature_ = self.segment_feature % X.shape[1]
        self.deviation_weight_ = self.deviation_weight
        features, segments = self._split(X)
        self.segments_ = tuple(dict.fromkeys(segments))
        self._segment_lookup_ = {segment: i for i, segment in enumerate(self.segments_)}
        codes = np.array([self._segment_lookup_[segment] for segment in segments])
        shared = sparse.coo_matrix(features)
        deviations = sparse.csr_matrix(
            (shared.data / self.deviation_weight_,
             (shared.row, codes[shared.row] * features.shape[1] + shared.col)),
            shape=(len(X), len(self.segments_) * features.shape[1]),
        )
        return sparse.hstack((shared, deviations), format="csc")

    def _split_array(self, X, xp):
        index = self.segment_feature_
        return xp.concat((X[:, :index], X[:, index + 1:]), axis=1), X[:, index]

    def _fit_array_api(self, X, y, sample_weight, logistic):
        _validate_solver_controls(self.max_iter, self.tol)
        _positive(self.alpha, "alpha")
        _positive(self.deviation_weight, "deviation_weight")
        X, y, weight = _fit_inputs(self, X, y, sample_weight, logistic)
        if X.shape[1] < 2:
            raise ValueError("Got 1 feature(s); at least one predictor and one segment column are required.")
        if not isinstance(self.segment_feature, Integral) or not (
            -X.shape[1] <= self.segment_feature < X.shape[1]
        ):
            raise ValueError("segment_feature must be a valid column index.")
        self.segment_feature_ = self.segment_feature % X.shape[1]
        self.deviation_weight_ = self.deviation_weight
        xp = _namespace(X)
        features, segments = self._split_array(X, xp)
        self._segment_levels_ = xp.unique_values(segments)
        self.segments_ = tuple(_host_input(self._segment_levels_).tolist())
        codes = xp.searchsorted(self._segment_levels_, segments)
        n_groups = len(self.segments_)
        n_features = features.shape[1]
        rows = [xp.nonzero(codes == i)[0] for i in range(n_groups)]
        blocks = [xp.take(features, indices, axis=0) for indices in rows]

        def forward(coefficients):
            deviations = xp.reshape(coefficients[n_features:], (n_groups, n_features))
            slopes = xp.take(deviations, codes, axis=0) / self.deviation_weight_
            return features @ coefficients[:n_features] + xp.sum(features * slopes, axis=1)

        def adjoint(residual):
            deviations = [
                block.T @ xp.take(residual, indices, axis=0) / self.deviation_weight_
                for block, indices in zip(blocks, rows)
            ]
            return xp.concat((features.T @ residual, *deviations))

        design = ArrayLassoDesign(weight, (n_groups + 1) * n_features, forward, adjoint, logistic)
        result = design.fit(y, self.alpha, self.max_iter, self.tol)
        fitted, self.n_iter_ = result.coefficients, result.n_iter
        self.intercept_ = fitted[-1]
        self.shared_coef_ = fitted[:n_features]
        deviations = xp.reshape(fitted[n_features:-1], (n_groups, n_features))
        self.segment_deviations_ = deviations / self.deviation_weight_
        self.segment_coef_ = self.shared_coef_ + self.segment_deviations_
        return self

    def _array_contributions(self, X):
        check_is_fitted(self, "shared_coef_")
        xp = _namespace(X)
        X = _array_api_features(X, xp)
        coef = self.shared_coef_
        if (xp is not _namespace(coef) or X.shape[1] != self.n_features_in_
                or X.device != coef.device or X.dtype != coef.dtype):
            raise ValueError("Prediction features must match the fitted namespace, shape, dtype, and device.")
        features, segments = self._split_array(X, xp)
        levels = self._segment_levels_
        codes = xp.searchsorted(levels, segments)
        safe = xp.minimum(codes, xp.full_like(codes, levels.shape[0] - 1))
        known = xp.take(levels, safe, axis=0) == segments
        deviations = xp.take(self.segment_deviations_, safe, axis=0)
        slopes = coef + xp.where(known[:, None], deviations, xp.zeros_like(deviations))
        intercept = xp.zeros_like(X[:, :1]) + self.intercept_
        return xp.concat((intercept, features * slopes), axis=1)

    def _set_coefficients(self, coefficients):
        blocks = coefficients.reshape(len(self.segments_) + 1, -1)
        if hasattr(self, "_segment_levels_"):
            del self._segment_levels_
        self.shared_coef_ = blocks[0].copy()
        self.segment_deviations_ = blocks[1:] / self.deviation_weight_
        self.segment_coef_ = self.shared_coef_ + self.segment_deviations_

    def effect_table(self):
        """Return shared, deviation, and total slopes for each segment and feature."""
        check_is_fitted(self, "shared_coef_")
        names = getattr(self, "feature_names_in_", range(self.n_features_in_))
        predictors = [name for i, name in enumerate(names) if i != self.segment_feature_]
        shared_slopes = _host_input(self.shared_coef_)
        return [
            {"segment": segment, "feature": feature, "shared": float(shared),
             "deviation": float(deviation), "coefficient": float(total)}
            for segment, deviations, totals in zip(
                self.segments_, _host_input(self.segment_deviations_), _host_input(self.segment_coef_)
            )
            for feature, shared, deviation, total in zip(
                predictors, shared_slopes, deviations, totals
            )
        ]

    def _prediction_inputs(self, X):
        check_is_fitted(self, "shared_coef_")
        X = validate_data(self, _cpu_input(X), reset=False, dtype=None)
        features, segments = self._split(X)
        codes = np.array([self._segment_lookup_.get(segment, -1) for segment in segments])
        return features, codes

    def predict_contributions(self, X):
        """Return the intercept and each numeric feature's additive contribution."""
        if _namespace(X) is not np:
            return self._array_contributions(X)
        features, codes = self._prediction_inputs(X)
        contributions = features * self.shared_coef_
        known = codes >= 0
        contributions[known] += features[known] * self.segment_deviations_[codes[known]]
        return np.column_stack((np.full(len(features), self.intercept_), contributions))

    def _linear_prediction(self, X):
        if _namespace(X) is not np:
            return _namespace(X).sum(self._array_contributions(X), axis=1)
        features, codes = self._prediction_inputs(X)
        known = codes >= 0
        predictions = self.intercept_ + features @ self.shared_coef_
        predictions[known] += np.einsum(
            "ij,ij->i", features[known], self.segment_deviations_[codes[known]]
        )
        return predictions


class DataSharedLassoRegressor(RegressorMixin, _DataSharedLasso):
    """Minimize half mean squared error plus the data-shared L1 penalty.

    Parameters
    ----------
    segment_feature : int, default=-1
        Column containing segment labels; excluded from the numeric predictors.
    alpha : float, default=0.01
        Positive strength of the L1 penalty.
    deviation_weight : float, default=1.0
        Positive penalty multiplier for segment deviations relative to shared slopes.
    max_iter : int, default=1000
        Solver iteration limit.
    tol : float, default=1e-8
        Solver tolerance.

    Attributes
    ----------
    intercept_ : float
        Shared baseline score.
    segments_ : tuple
        Labels in the order of the fitted segment coefficient rows.
    shared_coef_ : array
        Shared slopes, also used for unseen segments.
    segment_deviations_, segment_coef_ : arrays
        Segment adjustments and the resulting total slopes.
    segment_feature_ : int
        Resolved segment column index.
    n_iter_ : int
        Completed iterations for the fitted solver.
    n_features_in_ : int
        Number of input columns seen during fit.
    """

    def fit(self, X, y, sample_weight=None):
        """Fit with optional scalar or per-row nonnegative weights."""
        if _namespace(X) is not np:
            return self._fit_array_api(X, y, sample_weight, logistic=False)
        X, y = validate_data(self, _cpu_input(X), _cpu_input(y), dtype=None)
        X, y, weight = _weighted_rows(X, y, sample_weight)
        design = self._fit_design(X)
        model = Lasso(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
        model.fit(design, np.asarray(y, dtype=float), sample_weight=weight)
        self.n_iter_ = int(model.n_iter_)
        self.intercept_ = float(model.intercept_)
        self._set_coefficients(model.coef_)
        return self

    def predict(self, X):
        return self._linear_prediction(X)


class DataSharedLassoClassifier(_BinaryClassifier, _DataSharedLasso):
    """Minimize mean binary logistic loss plus the data-shared L1 penalty.

    Parameters
    ----------
    segment_feature : int, default=-1
        Column containing segment labels; excluded from the numeric predictors.
    alpha : float, default=0.01
        Positive strength of the L1 penalty.
    deviation_weight : float, default=1.0
        Positive penalty multiplier for segment deviations relative to shared slopes.
    max_iter : int, default=1000
        Solver iteration limit.
    tol : float, default=1e-8
        Solver tolerance.

    Attributes
    ----------
    intercept_ : float
        Shared baseline score.
    segments_ : tuple
        Labels in the order of the fitted segment coefficient rows.
    shared_coef_ : array
        Shared slopes, also used for unseen segments.
    segment_deviations_, segment_coef_ : arrays
        Segment adjustments and the resulting total slopes.
    segment_feature_ : int
        Resolved segment column index.
    n_iter_ : int
        Completed iterations for the fitted solver.
    n_features_in_ : int
        Number of input columns seen during fit.
    classes_ : array
        Class labels in probability-column order.
    """

    def fit(self, X, y, sample_weight=None):
        """Fit with optional scalar or per-row nonnegative weights."""
        if _namespace(X) is not np:
            return self._fit_array_api(X, y, sample_weight, logistic=True)
        X, y = validate_data(self, _cpu_input(X), _cpu_input(y), dtype=None)
        X, y, weight = _weighted_rows(X, y, sample_weight)
        self.classes_, y = _binary_target(y)
        design = self._fit_design(X)
        self.intercept_, coefficients, _, self.n_iter_ = _logistic_lasso(
            design, y, weight, self.alpha, self.max_iter, self.tol
        )
        self._set_coefficients(coefficients)
        return self

    def decision_function(self, X):
        return self._linear_prediction(X)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        xp = _namespace(scores)
        return xp.stack((_sigmoid(-scores, xp), _sigmoid(scores, xp)), axis=1)

    def predict(self, X):
        scores = self.decision_function(X)
        return _namespace(scores).where(scores > 0, self.classes_[1], self.classes_[0])
