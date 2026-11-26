from numbers import Integral, Real

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from ._sparse_logistic import _logistic_lasso


from ._validation import (
    _BinaryClassifier,
    _binary_target,
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
            raise ValueError("At least one predictor and one segment column are required.")
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

    def _set_coefficients(self, coefficients):
        blocks = coefficients.reshape(len(self.segments_) + 1, -1)
        self.shared_coef_ = blocks[0].copy()
        self.segment_deviations_ = blocks[1:] / self.deviation_weight_
        self.segment_coef_ = self.shared_coef_ + self.segment_deviations_

    def effect_table(self):
        """Return shared, deviation, and total slopes for each segment and feature."""
        check_is_fitted(self, "shared_coef_")
        names = getattr(self, "feature_names_in_", range(self.n_features_in_))
        predictors = [name for i, name in enumerate(names) if i != self.segment_feature_]
        return [
            {"segment": segment, "feature": feature, "shared": float(shared),
             "deviation": float(deviation), "coefficient": float(total)}
            for segment, deviations, totals in zip(
                self.segments_, self.segment_deviations_, self.segment_coef_
            )
            for feature, shared, deviation, total in zip(
                predictors, self.shared_coef_, deviations, totals
            )
        ]

    def _prediction_inputs(self, X):
        check_is_fitted(self, "shared_coef_")
        X = validate_data(self, X, reset=False, dtype=None)
        features, segments = self._split(X)
        codes = np.array([self._segment_lookup_.get(segment, -1) for segment in segments])
        return features, codes

    def predict_contributions(self, X):
        """Return the intercept and each numeric feature's additive contribution."""
        features, codes = self._prediction_inputs(X)
        contributions = features * self.shared_coef_
        known = codes >= 0
        contributions[known] += features[known] * self.segment_deviations_[codes[known]]
        return np.column_stack((np.full(len(features), self.intercept_), contributions))

    def _linear_prediction(self, X):
        features, codes = self._prediction_inputs(X)
        known = codes >= 0
        predictions = self.intercept_ + features @ self.shared_coef_
        predictions[known] += np.einsum(
            "ij,ij->i", features[known], self.segment_deviations_[codes[known]]
        )
        return predictions


class DataSharedLassoRegressor(RegressorMixin, _DataSharedLasso):
    """Minimize half mean squared error plus the data-shared L1 penalty."""

    def fit(self, X, y, sample_weight=None):
        """Fit with optional scalar or per-row nonnegative weights."""
        X, y = validate_data(self, X, y, dtype=None)
        X, y, weight = _weighted_rows(X, y, sample_weight)
        design = self._fit_design(X)
        model = Lasso(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
        model.fit(design, np.asarray(y, dtype=float), sample_weight=weight)
        self.intercept_ = float(model.intercept_)
        self._set_coefficients(model.coef_)
        return self

    def predict(self, X):
        return self._linear_prediction(X)


class DataSharedLassoClassifier(_BinaryClassifier, _DataSharedLasso):
    """Minimize mean binary logistic loss plus the data-shared L1 penalty."""

    def fit(self, X, y, sample_weight=None):
        """Fit with optional scalar or per-row nonnegative weights."""
        X, y = validate_data(self, X, y, dtype=None)
        X, y, weight = _weighted_rows(X, y, sample_weight)
        self.classes_, y = _binary_target(y)
        design = self._fit_design(X)
        self.intercept_, coefficients, _ = _logistic_lasso(
            design, y, weight, self.alpha, self.max_iter, self.tol
        )
        self._set_coefficients(coefficients)
        return self

    def decision_function(self, X):
        return self._linear_prediction(X)
