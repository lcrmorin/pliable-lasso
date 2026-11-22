"""A scikit-learn regressor for the published pliable-lasso objective."""
from copy import copy
from numbers import Integral, Real
import warnings

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_scalar, validate_data

from ._solver import InteractionDesign, LogisticInteractionDesign, solve
from ._validation import _normalized_weights


class _PliableBase(BaseEstimator):
    def __init__(self, *, alpha=1.0, l1_ratio=0.5, modifier_features=None,
                 fit_intercept=True, max_iter=10000, tol=1e-6, warm_start=False):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.modifier_features = modifier_features
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start

    def _validate_parameters(self):
        for name in ['alpha', 'l1_ratio', 'tol']:
            value = getattr(self, name)
            check_scalar(value, name, Real, min_val=0,
                         max_val=1 if name == 'l1_ratio' else None,
                         include_boundaries='neither' if name == 'tol' else 'both')
            if not np.isfinite(value):
                raise ValueError(f'{name} must be finite.')
        check_scalar(self.max_iter, 'max_iter', Integral, min_val=1)
        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise ValueError('fit_intercept must be a boolean.')
        if not isinstance(self.warm_start, (bool, np.bool_)):
            raise ValueError('warm_start must be a boolean.')

    def _fit(self, X, y, sample_weight, *, logistic):
        self._validate_parameters()
        # Stage the input metadata until validation and optimization succeed.
        metadata = copy(self)
        X, y = validate_data(metadata, X, y, y_numeric=not logistic, multi_output=False,
                             dtype=np.float64, accept_sparse=False)
        n, d = X.shape
        modifiers = np.asarray([] if self.modifier_features is None else self.modifier_features)
        if modifiers.ndim != 1 or (len(modifiers) and modifiers.dtype.kind not in 'iu'):
            raise ValueError('modifier_features must be a one-dimensional sequence of integer column indices.')
        modifiers = modifiers.astype(np.intp)
        if len(np.unique(modifiers)) != len(modifiers) or np.any((modifiers < 0) | (modifiers >= d)):
            raise ValueError('modifier_features must contain distinct indices within the input columns.')
        predictors = np.setdiff1d(np.arange(d), modifiers)
        if not len(predictors):
            raise ValueError('At least one non-modifier predictor column is required.')
        weight = _normalized_weights(sample_weight, n)
        classes = None
        if logistic:
            check_classification_targets(y)
            classes = np.unique(y[weight > 0])
            if len(classes) == 1:
                raise ValueError('Only one class has positive sample weight.')
            if len(classes) > 2:
                raise ValueError('Only binary classification is supported.')
            y = (y == classes[1]).astype(float)
        offset = float(weight @ y) if self.fit_intercept and not logistic else 0.
        design = InteractionDesign(X[:, predictors], X[:, modifiers], weight, self.fit_intercept)
        names = getattr(metadata, 'feature_names_in_', None)
        initial = None
        if (self.warm_start and getattr(self, '_fit_intercept_', None) == self.fit_intercept
                and self.n_features_in_ == d
                and np.array_equal(self.modifier_indices_, modifiers)
                and np.array_equal(getattr(self, 'feature_names_in_', None), names)
                and np.array_equal(getattr(self, 'classes_', None), classes)):
            initial = np.concatenate((self.coef_[predictors], self.coef_[modifiers],
                                      self.interaction_coef_.ravel()))
        solver_design = design
        if logistic:
            solver_design = LogisticInteractionDesign(design, self.fit_intercept)
            if self.fit_intercept:
                if initial is None:
                    offset = float(np.log(weight[y == 1].sum()) - np.log(weight[y == 0].sum()))
                    initial = np.zeros(design.size)
                else:
                    offset = self.intercept_ + design.mean @ initial
                initial = np.append(initial, offset)
        result = solve(solver_design, y - (0. if logistic else offset), self.alpha,
                       self.l1_ratio, self.max_iter, self.tol, initial)
        coefficients = result.coefficients[:design.size]
        beta, gamma, theta = design.unpack(coefficients)
        if logistic and self.fit_intercept:
            offset = result.coefficients[-1]
        self.n_features_in_ = d
        self._fit_intercept_ = self.fit_intercept
        if names is not None:
            self.feature_names_in_ = names
        elif hasattr(self, 'feature_names_in_'):
            del self.feature_names_in_
        self.coef_ = np.zeros(d)
        self.coef_[predictors] = beta
        self.coef_[modifiers] = gamma
        self.interaction_coef_ = theta.copy()
        self.intercept_ = float(offset - design.mean @ coefficients)
        if logistic:
            self.classes_ = classes
        self.predictor_indices_ = predictors
        self.modifier_indices_ = modifiers
        self.n_iter_ = result.n_iter
        self.objective_ = result.objective
        self.optimality_ = result.optimality
        self.converged_ = result.converged
        if not result.converged:
            warnings.warn(f'{type(self).__name__} did not converge. Increase max_iter, scale '
                          'the inputs, or relax tol. Inspect optimality_.',
                          ConvergenceWarning, stacklevel=3)
        return self

    def _linear_score(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64, accept_sparse=False)
        result = self.intercept_ + X @ self.coef_
        if len(self.modifier_indices_):
            result += np.einsum('ij,ij->i', X[:, self.predictor_indices_],
                                X[:, self.modifier_indices_] @ self.interaction_coef_.T)
        return result

class PliableRegressor(RegressorMixin, _PliableBase):
    """Linear regression with modifier-dependent coefficients.

    Parameters
    ----------
    alpha : float, default=1.0
        Overall nonnegative penalty strength. Unlike the original paper, alpha
        names regularization strength following sklearn's Lasso convention;
        the paper's mixing parameter is ``l1_ratio`` here.
    l1_ratio : float, default=0.5
        Mixing weight in [0, 1] for the elementwise interaction penalty.
    modifier_features : array-like of int or None, default=None
        Distinct zero-based column indices in the single input matrix that
        supply modifiers Z. Remaining columns are predictors X. None means
        no modifiers, reducing to Lasso with penalty alpha * (1 - l1_ratio).
        Supply numerical encodings; categorical encoding belongs in a pipeline.
    fit_intercept : bool, default=True
        Whether to fit an unpenalized intercept.
    max_iter : int, default=10000
        Maximum number of proximal-gradient iterations.
    tol : float, default=1e-6
        Relative proximal-gradient stationarity tolerance, strictly positive.
    warm_start : bool, default=False
        Initialize from the previous fit when column names, modifier indices
        and intercept settings match. Cloning starts with no fitted state.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Main-effect coefficients in original input column order, including the
        unpenalized modifier main effects.
    interaction_coef_ : ndarray of shape (n_predictors, n_modifiers)
        Interaction coefficients; rows follow predictor_indices_, columns
        follow modifier_indices_. These multiply raw input products X_j Z_k.
    intercept_ : float
        Intercept in the original input coordinates.
    predictor_indices_ : ndarray of int
        Original input columns used as predictors.
    modifier_indices_ : ndarray of int
        Original input columns used as modifiers, in supplied order.
    n_iter_ : int
        Iterations performed.
    objective_ : float
        Weighted mean squared error divided by two plus the pliable penalty.
    optimality_ : float
        Relative infinity norm of the proximal-gradient mapping. This is not
        a duality gap; values at or below tol indicate convergence.
    converged_ : bool
        Whether the stationarity tolerance was reached.
    n_features_in_ : int
        Number of columns seen during fit.
    feature_names_in_ : ndarray of str
        Input feature names, present when all names are strings.

    Notes
    -----
    Fits y = intercept + X beta + Z gamma + sum_j X_j (Z theta_j), minimizing
    weighted squared error / (2 * sum(weights)) plus
    alpha * ((1-l1_ratio) * sum_j (||(beta_j, theta_j)||_2 + ||theta_j||_2)
             + l1_ratio * sum_jk |theta_jk|).

    The intercept and gamma are unpenalized. The penalty encourages hierarchical
    sparsity but does not enforce a hard rule that every nonzero interaction
    requires a nonzero main coefficient. No automatic feature scaling is used.
    The solver centers design columns to eliminate the intercept, including
    interaction columns; it does not redefine interactions as centered X/Z
    products. Scaling and modifier encoding affect the penalized model.

    Only finite dense inputs and a single continuous target are supported.
    See Tibshirani and Friedman (2020), A Pliable Lasso,
    https://doi.org/10.1080/10618600.2019.1648271.
    """

    def fit(self, X, y, sample_weight=None):
        """Fit a continuous target from predictor and modifier columns."""
        return self._fit(X, y, sample_weight, logistic=False)

    def predict(self, X):
        """Predict the conditional mean."""
        return self._linear_score(X)


class PliableClassifier(ClassifierMixin, _PliableBase):
    """Binary logistic regression with modifier-dependent coefficients.

    Uses the same parameters and structured penalty as PliableRegressor.
    Coefficients and interactions are on the log-odds scale.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def fit(self, X, y, sample_weight=None):
        """Fit a binary target from predictor and modifier columns."""
        return self._fit(X, y, sample_weight, logistic=True)

    def decision_function(self, X):
        """Return log odds for classes_[1]."""
        return self._linear_score(X)

    def predict_proba(self, X):
        """Return class probabilities in classes_ order."""
        scores = self.decision_function(X)
        return np.column_stack((expit(-scores), expit(scores)))

    def predict(self, X):
        """Predict the most probable class."""
        positive = self.decision_function(X) > 0
        return self.classes_[positive.astype(np.intp)]
