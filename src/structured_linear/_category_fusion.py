"""Estimators that fuse regressions across levels of one nominal predictor."""

import warnings
from numbers import Integral, Real

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted, validate_data

from ._category_fusion_solver import _solve_category_fusion
from ._validation import (
    _BinaryClassifier,
    _binary_target,
    _positive,
    _validate_solver_controls,
    _weighted_rows,
)


class _CategoryFusion(BaseEstimator):
    def __init__(
        self, *, segment_feature=-1, alpha=0.01, max_iter=5000, tol=1e-8,
        fusion_tol=1e-5,
    ):
        self.segment_feature = segment_feature
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fusion_tol = fusion_tol

    def _split(self, X):
        index = self.segment_feature_
        features = np.delete(X, index, axis=1).astype(float)
        if not np.isfinite(features).all():
            raise ValueError("Numeric predictors must be finite.")
        segments = X[:, index].tolist()
        if any(
            not isinstance(segment, (Real, str))
            or (isinstance(segment, Real) and not np.isfinite(segment))
            for segment in segments
        ):
            raise ValueError("Segment labels must be strings or finite numbers.")
        return features, segments

    def _fit(self, X, y, sample_weight, *, logistic):
        _validate_solver_controls(self.max_iter, self.tol)
        _positive(self.alpha, "alpha")
        _positive(self.fusion_tol, "fusion_tol")
        X, y = validate_data(self, X, y, dtype=None, y_numeric=not logistic)
        X, y, weight = _weighted_rows(X, y, sample_weight)
        if X.shape[1] < 2 or not isinstance(self.segment_feature, Integral) or not (
            -X.shape[1] <= self.segment_feature < X.shape[1]
        ):
            raise ValueError("Select one segment column and at least one numeric predictor.")
        self.segment_feature_ = self.segment_feature % X.shape[1]
        features, segments = self._split(X)
        if logistic:
            self.classes_, y = _binary_target(y)
        self.segments_ = tuple(dict.fromkeys(segments))
        self._segment_lookup_ = {segment: i for i, segment in enumerate(self.segments_)}
        codes = np.array([self._segment_lookup_[segment] for segment in segments])
        design = np.column_stack((np.ones(len(X)), features))
        order = np.argsort(codes, kind="stable")
        groups = np.split(order, np.cumsum(np.bincount(codes))[:-1])
        beta, self.n_iter_, self.converged_ = _solve_category_fusion(
            [design[group] for group in groups],
            [y[group] for group in groups],
            [weight[group] for group in groups],
            self.alpha, logistic, self.max_iter, self.tol,
        )
        if not self.converged_:
            warnings.warn("Category fusion reached max_iter.", ConvergenceWarning, stacklevel=3)

        left, right = np.triu_indices(len(beta), k=1)
        close = np.linalg.norm(beta[left] - beta[right], axis=1) <= self.fusion_tol

        edges = csr_matrix(
            (np.ones(close.sum()), (left[close], right[close])),
            shape=(len(beta), len(beta)),
        )
        self.n_clusters_, self.cluster_labels_ = connected_components(edges, directed=False)
        self.intercept_ = beta[:, 0].copy()
        self.coef_ = beta[:, 1:].copy()
        return self

    def _linear_prediction(self, X):
        check_is_fitted(self, "coef_")
        X = validate_data(self, X, reset=False, dtype=None)
        features, segments = self._split(X)
        try:
            codes = np.array([self._segment_lookup_[segment] for segment in segments])
        except KeyError as error:
            raise ValueError(f"Unknown segment: {error.args[0]!r}.") from error
        return self.intercept_[codes] + np.einsum("ij,ij->i", features, self.coef_[codes])


class CategoryFusionRegressor(RegressorMixin, _CategoryFusion):
    """Linear regression with pairwise fusion of segment intercepts and slopes."""

    def fit(self, X, y, sample_weight=None):
        return self._fit(X, y, sample_weight, logistic=False)

    def predict(self, X):
        return self._linear_prediction(X)


class CategoryFusionClassifier(_BinaryClassifier, _CategoryFusion):
    """Binary logistic regression with pairwise fusion of segment equations."""

    def fit(self, X, y, sample_weight=None):
        return self._fit(X, y, sample_weight, logistic=True)

    def decision_function(self, X):
        return self._linear_prediction(X)
