from numbers import Integral, Real

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from ._sparse_logistic import _logistic_lasso


from ._validation import _BinaryClassifier, _binary_target, _positive


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
        features = check_array(np.delete(X, self.segment_feature_, axis=1), dtype=float)
        segments = X[:, self.segment_feature_].tolist()
        for segment in segments:
            if not isinstance(segment, (Real, str)) or (
                isinstance(segment, Real) and not np.isfinite(segment)
            ):
                raise ValueError("Segment labels must be strings or finite numbers.")
        return features, segments

    def _fit_design(self, X):
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

    def _linear_prediction(self, X):
        check_is_fitted(self, "shared_coef_")
        X = validate_data(self, X, reset=False, dtype=None)
        features, segments = self._split(X)
        codes = np.array([self._segment_lookup_.get(segment, -1) for segment in segments])
        known = codes >= 0
        predictions = self.intercept_ + features @ self.shared_coef_
        predictions[known] += np.einsum(
            "ij,ij->i", features[known], self.segment_deviations_[codes[known]]
        )
        return predictions


class DataSharedLassoRegressor(RegressorMixin, _DataSharedLasso):
    """Minimize half mean squared error plus the data-shared L1 penalty."""

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype=None)
        design = self._fit_design(X)
        model = Lasso(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
        model.fit(design, np.asarray(y, dtype=float))
        self.intercept_ = float(model.intercept_)
        self._set_coefficients(model.coef_)
        return self

    def predict(self, X):
        return self._linear_prediction(X)


class DataSharedLassoClassifier(_BinaryClassifier, _DataSharedLasso):
    """Minimize mean binary logistic loss plus the data-shared L1 penalty."""

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype=None)
        self.classes_, y = _binary_target(y)
        design = self._fit_design(X)
        self.intercept_, coefficients, _ = _logistic_lasso(
            design, y, self.alpha, self.max_iter, self.tol
        )
        self._set_coefficients(coefficients)
        return self

    def decision_function(self, X):
        return self._linear_prediction(X)
