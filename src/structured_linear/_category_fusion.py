"""Estimators that fuse regressions across levels of one nominal predictor."""

import warnings
from numbers import Integral, Real

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted, validate_data

from ._category_fusion_solver import _solve_category_fusion
from ._pliable import _array_api_features, _array_api_input, _array_api_weights, _host_input
from ._solver import _namespace, _sigmoid
from ._validation import (
    _BinaryClassifier,
    _binary_target,
    _cpu_input,
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
        if _namespace(X) is not np:
            return self._fit_array_api(X, y, sample_weight, logistic)
        X, y = validate_data(self, _cpu_input(X), _cpu_input(y), dtype=None, y_numeric=not logistic)
        X, y, weight = _weighted_rows(X, y, sample_weight)
        if X.shape[1] == 1:
            raise ValueError("Got 1 feature(s); a predictor and a segment column are required.")
        if not isinstance(self.segment_feature, Integral) or not (
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
        return self._finish_fit(beta)

    def _fit_array_api(self, X, y, sample_weight, logistic):
        xp = _namespace(X)
        X = _array_api_features(X, xp)
        _, n_features = X.shape
        if n_features == 1:
            raise ValueError("Got 1 feature(s); a predictor and a segment column are required.")
        if not isinstance(self.segment_feature, Integral) or not (
            -n_features <= self.segment_feature < n_features
        ):
            raise ValueError("Select one segment column and at least one numeric predictor.")
        self.segment_feature_ = self.segment_feature % n_features
        self.n_features_in_ = n_features
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        y = _array_api_input(y, X, xp)
        if y.shape != (X.shape[0],) or not bool(xp.all(xp.isfinite(y))):
            raise ValueError("y must have shape (n_samples,) and contain finite values.")
        weight = _array_api_weights(sample_weight, X, xp)
        features, segments = self._split_array_api(X, xp)
        # Only the categorical column and row-selection mask are needed on CPU.
        keep = _host_input(weight > 0)
        self.segments_ = tuple(dict.fromkeys(segments[keep].tolist()))
        self._segment_lookup_ = {segment: i for i, segment in enumerate(self.segments_)}
        design = xp.concat((xp.ones_like(features[:, :1]), features), axis=1)
        groups = [
            xp.asarray(np.flatnonzero(keep & (segments == segment)), device=X.device)
            for segment in self.segments_
        ]
        if logistic:
            classes = xp.unique_values(y[weight > 0])
            if classes.shape[0] == 1:
                raise ValueError("Only one class has positive sample weight.")
            if classes.shape != (2,):
                raise ValueError("Only binary classification is supported.")
            if xp.isdtype(y.dtype, "real floating") and not bool(xp.all(y == xp.floor(y))):
                raise ValueError("Classification targets must be discrete.")
            self.classes_ = classes
            y = y == classes[1]
        beta, self.n_iter_, self.converged_ = _solve_category_fusion(
            [xp.take(design, group, axis=0) for group in groups],
            [xp.astype(xp.take(y, group, axis=0), X.dtype) for group in groups],
            [xp.take(weight, group, axis=0) for group in groups],
            self.alpha, logistic, self.max_iter, self.tol,
        )
        return self._finish_fit(beta)

    def _split_array_api(self, X, xp):
        index = self.segment_feature_
        features = xp.concat((X[:, :index], X[:, index + 1:]), axis=1)
        return features, _host_input(X[:, index])

    def _finish_fit(self, beta):
        if not self.converged_:
            warnings.warn("Category fusion reached max_iter.", ConvergenceWarning, stacklevel=3)
        xp = _namespace(beta)
        beta_host = _host_input(beta)
        left, right = np.triu_indices(beta.shape[0], k=1)
        close = np.linalg.norm(beta_host[left] - beta_host[right], axis=1) <= self.fusion_tol
        edges = csr_matrix(
            (np.ones(close.sum()), (left[close], right[close])),
            shape=(beta.shape[0], beta.shape[0]),
        )
        self.n_clusters_, labels = connected_components(edges, directed=False)
        self.cluster_labels_ = labels if xp is np else xp.asarray(labels, device=beta.device)
        self.intercept_ = beta[:, 0]
        self.coef_ = beta[:, 1:]
        return self

    def fit_path(self, X, y, alphas, sample_weight=None):
        """Return independently fitted models in the order of the supplied alphas."""
        alphas = tuple(alphas)
        if not alphas:
            raise ValueError("alphas must contain at least one value.")
        for alpha in alphas:
            _positive(alpha, "alpha")
        return [
            clone(self).set_params(alpha=alpha).fit(X, y, sample_weight=sample_weight)
            for alpha in alphas
        ]

    def effect_table(self):
        """Return one row per segment and term, including the intercept."""
        check_is_fitted(self, "coef_")
        names = getattr(self, "feature_names_in_", range(self.n_features_in_))
        predictors = [name for i, name in enumerate(names) if i != self.segment_feature_]
        return [
            {"segment": segment, "cluster": int(cluster), "feature": feature,
             "coefficient": float(coefficient)}
            for segment, cluster, intercept, slopes in zip(
                self.segments_, _host_input(self.cluster_labels_),
                _host_input(self.intercept_), _host_input(self.coef_)
            )
            for feature, coefficient in zip([None, *predictors], [intercept, *slopes])
        ]

    def _prediction_inputs(self, X):
        check_is_fitted(self, "coef_")
        xp = _namespace(X)
        if xp is not _namespace(self.coef_):
            raise ValueError("Prediction features must use the fitted array namespace.")
        if xp is np:
            X = validate_data(self, X, reset=False, dtype=None)
            features, segments = self._split(X)
        else:
            X = _array_api_features(X, xp)
            if X.shape[1] != self.n_features_in_ or X.dtype != self.coef_.dtype or X.device != self.coef_.device:
                raise ValueError("Prediction features must match the fitted shape, dtype, and device.")
            features, segments = self._split_array_api(X, xp)
        try:
            codes = np.array([self._segment_lookup_[segment] for segment in segments])
        except KeyError as error:
            raise ValueError(f"Unknown segment: {error.args[0]!r}.") from error
        return features, codes if xp is np else xp.asarray(codes, device=X.device)

    def predict_contributions(self, X):
        """Return the intercept and each numeric feature's additive contribution."""
        features, codes = self._prediction_inputs(X)
        xp = _namespace(features)
        intercept = xp.take(self.intercept_, codes, axis=0)
        effects = features * xp.take(self.coef_, codes, axis=0)
        if xp is np:
            return np.column_stack((intercept, effects))
        return xp.concat((intercept[:, None], effects), axis=1)

    def _linear_prediction(self, X):
        contributions = self.predict_contributions(X)
        return _namespace(contributions).sum(contributions, axis=1)


class CategoryFusionRegressor(RegressorMixin, _CategoryFusion):
    """Linear regression with pairwise fusion of segment intercepts and slopes.

    Parameters
    ----------
    segment_feature : int, default=-1
        Column containing the unordered segment labels.
    alpha : float, default=0.01
        Positive strength of the pairwise fusion penalty on slopes and intercepts.
    max_iter : int, default=5000
        Solver iteration limit.
    tol : float, default=1e-8
        Solver tolerance.
    fusion_tol : float, default=1e-5
        Distance threshold for assigning fitted equations to connected clusters.

    Attributes
    ----------
    segments_ : tuple
        Labels in fitted coefficient-row order.
    coef_, intercept_ : arrays
        Slopes and intercept for each segment.
    cluster_labels_ : array
        Cluster ID for each segment.
    n_clusters_, n_iter_ : int
        Number of learned clusters and completed solver iterations.
    converged_ : bool
        Whether the solver met its stopping criterion.
    segment_feature_ : int
        Resolved segment column index.
    n_features_in_ : int
        Number of input columns seen during fit.
    """

    def fit(self, X, y, sample_weight=None):
        return self._fit(X, y, sample_weight, logistic=False)

    def predict(self, X):
        return self._linear_prediction(X)


class CategoryFusionClassifier(_BinaryClassifier, _CategoryFusion):
    """Binary logistic regression with pairwise fusion of segment equations.

    Parameters
    ----------
    segment_feature : int, default=-1
        Column containing the unordered segment labels.
    alpha : float, default=0.01
        Positive strength of the pairwise fusion penalty on slopes and intercepts.
    max_iter : int, default=5000
        Solver iteration limit.
    tol : float, default=1e-8
        Solver tolerance.
    fusion_tol : float, default=1e-5
        Distance threshold for assigning fitted equations to connected clusters.

    Attributes
    ----------
    segments_ : tuple
        Labels in fitted coefficient-row order.
    coef_, intercept_ : arrays
        Slopes and intercept for each segment.
    cluster_labels_ : array
        Cluster ID for each segment.
    n_clusters_, n_iter_ : int
        Number of learned clusters and completed solver iterations.
    converged_ : bool
        Whether the solver met its stopping criterion.
    segment_feature_ : int
        Resolved segment column index.
    n_features_in_ : int
        Number of input columns seen during fit.
    classes_ : array
        Class labels in probability-column order.
    """

    def fit(self, X, y, sample_weight=None):
        return self._fit(X, y, sample_weight, logistic=True)

    def decision_function(self, X):
        return self._linear_prediction(X)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        xp = _namespace(scores)
        return xp.stack((_sigmoid(-scores, xp), _sigmoid(scores, xp)), axis=1)

    def predict(self, X):
        scores = self.decision_function(X)
        xp = _namespace(scores)
        return xp.where(scores > 0, self.classes_[1], self.classes_[0])
