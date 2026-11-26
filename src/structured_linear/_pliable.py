"""A scikit-learn regressor for the published pliable-lasso objective."""
from copy import copy
from numbers import Integral, Real
import warnings

import numpy as np
from array_api_compat import is_cupy_array, is_torch_array, to_device
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_scalar, validate_data

from ._solver import (
    InteractionDesign,
    LogisticInteractionDesign,
    _interaction_product,
    _namespace,
    _sigmoid,
    solve,
)
from ._validation import _normalized_weights


def _array_api_features(X, xp):
    if X.ndim != 2 or min(X.shape) == 0:
        raise ValueError("X must be a nonempty two-dimensional array.")
    if getattr(X.device, "type", None) == "mps" and X.dtype != xp.float32:
        raise ValueError("MPS X must have dtype float32.")
    if X.dtype not in (xp.float32, xp.float64):
        raise ValueError("Array API X must have dtype float32 or float64.")
    if not bool(xp.all(xp.isfinite(X))):
        raise ValueError("X must contain only finite values.")
    return X


def _host_input(value):
    source = _namespace(value)
    if source is np:
        return value
    if is_cupy_array(value):
        return value.get()
    return np.asarray(to_device(value, source.asarray(0).device))


def _array_api_input(value, X, xp, *, dtype=None):
    if _namespace(value) is not xp:
        value = _host_input(value)
        if is_torch_array(X):
            # Construct from host memory before transferring to the fitted device.
            return to_device(xp.asarray(value, dtype=dtype, device="cpu"), X.device)
    return xp.asarray(value, dtype=dtype, device=X.device)


def _array_api_weights(sample_weight, X, xp):
    weight = (xp.ones(X.shape[0], dtype=X.dtype, device=X.device)
              if sample_weight is None
              else _array_api_input(sample_weight, X, xp, dtype=X.dtype))
    if weight.ndim == 0:
        weight = xp.broadcast_to(weight, (X.shape[0],))
    if weight.shape != (X.shape[0],):
        raise ValueError("sample_weight must be a scalar or have shape (n_samples,).")
    if not bool(xp.all(xp.isfinite(weight) & (weight >= 0))):
        raise ValueError("sample_weight must be finite and nonnegative.")
    maximum = xp.max(weight)
    if not bool(maximum > 0):
        raise ValueError("sample_weight must contain at least one non-zero value.")
    weight = weight / maximum
    return weight / xp.sum(weight)


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
        xp = _namespace(X)
        output_dtype = getattr(X, "dtype", None)
        if xp is np:
            sample_weight = _host_input(sample_weight)
            X, y = validate_data(metadata, X, _host_input(y), y_numeric=not logistic, multi_output=False,
                                 dtype=np.float64, accept_sparse=False)
        else:
            X = _array_api_features(X, xp)
            y = _array_api_input(y, X, xp)
            if y.shape != (X.shape[0],):
                raise ValueError("y must have shape (n_samples,).")
            if not xp.isdtype(y.dtype, ("real floating", "integral", "bool")):
                raise ValueError("Array API y must contain real numerical or boolean values.")
            if xp.isdtype(y.dtype, "real floating") and not bool(xp.all(xp.isfinite(y))):
                raise ValueError("y must contain only finite values.")
            metadata = None
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
        weight = (_normalized_weights(sample_weight, n) if xp is np
                  else _array_api_weights(sample_weight, X, xp))
        classes = None
        if logistic:
            if xp is np:
                check_classification_targets(y)
                classes = np.unique(y[weight > 0])
            else:
                if xp.isdtype(y.dtype, "real floating") and not bool(xp.all(y == xp.floor(y))):
                    raise ValueError("Continuous targets are not supported for classification.")
                classes = xp.unique_values(y[weight > 0])
            if classes.shape[0] == 1:
                raise ValueError('Only one class has positive sample weight.')
            if classes.shape[0] != 2:
                raise ValueError('Only binary classification is supported.')
            y = (y == classes[1]).astype(float) if xp is np else xp.astype(y == classes[1], X.dtype)
        elif xp is not np:
            y = xp.astype(y, X.dtype)
        offset = float(weight @ y) if self.fit_intercept and not logistic else 0.
        if xp is np:
            x, z = X[:, predictors], X[:, modifiers]
        else:
            x = xp.take(X, xp.asarray(predictors, device=X.device), axis=1)
            z = xp.take(X, xp.asarray(modifiers, device=X.device), axis=1)
        design = InteractionDesign(x, z, weight, self.fit_intercept)
        names = getattr(metadata, 'feature_names_in_', None)
        initial = None
        if (self.warm_start and getattr(self, '_fit_intercept_', None) == self.fit_intercept
                and self.n_features_in_ == d
                and getattr(self, "_modifier_layout", None) == tuple(modifiers)
                and np.array_equal(getattr(self, 'feature_names_in_', None), names)):
            old_coef = self.coef_
            old_classes = getattr(self, 'classes_', None)
            if xp is np and isinstance(old_coef, np.ndarray) and np.array_equal(old_classes, classes):
                initial = np.concatenate((old_coef[predictors], old_coef[modifiers],
                                          self.interaction_coef_.ravel()))
            elif (xp is not np and _namespace(old_coef) is xp
                  and old_coef.device == X.device and old_coef.dtype == X.dtype
                  and (classes is None or (old_classes.shape == classes.shape
                                           and old_classes.dtype == classes.dtype
                                           and bool(xp.all(old_classes == classes))))):
                initial = xp.concat((
                    xp.take(old_coef, xp.asarray(predictors, device=X.device)),
                    xp.take(old_coef, xp.asarray(modifiers, device=X.device)),
                    xp.reshape(self.interaction_coef_, (-1,)),
                ))
        solver_design = design
        if logistic:
            solver_design = LogisticInteractionDesign(design, self.fit_intercept)
            if self.fit_intercept:
                if initial is None:
                    offset = float(xp.log(xp.sum(weight[y == 1])) - xp.log(xp.sum(weight[y == 0])))
                    initial = (np.zeros(design.size) if xp is np else
                               xp.zeros(design.size, dtype=X.dtype, device=X.device))
                else:
                    offset = self.intercept_ + design.mean @ initial
                initial = (np.append(initial, offset) if xp is np else xp.concat((
                    initial, xp.reshape(xp.asarray(offset, dtype=X.dtype, device=X.device), (1,))
                )))
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
        if xp is np:
            self.coef_ = np.zeros(d)
            self.coef_[predictors] = beta
            self.coef_[modifiers] = gamma
            self.interaction_coef_ = theta.copy()
            if output_dtype == np.float32:
                self.coef_ = self.coef_.astype(np.float32)
                self.interaction_coef_ = self.interaction_coef_.astype(np.float32)
        else:
            order = np.argsort(np.concatenate((predictors, modifiers)))
            self.coef_ = xp.take(xp.concat((beta, gamma)), xp.asarray(order, device=X.device))
            self.interaction_coef_ = theta
        self.intercept_ = float(offset - design.mean @ coefficients)
        if logistic:
            self.classes_ = classes
        self.predictor_indices_ = predictors if xp is np else xp.asarray(predictors, device=X.device)
        self.modifier_indices_ = modifiers if xp is np else xp.asarray(modifiers, device=X.device)
        self._modifier_layout = tuple(modifiers)
        self.n_iter_ = result.n_iter
        self.objective_ = result.objective
        self.optimality_ = result.optimality
        self.converged_ = result.converged
        if not result.converged:
            warnings.warn(f'{type(self).__name__} did not converge. Increase max_iter, scale '
                          'the inputs, or relax tol. Inspect optimality_.',
                          ConvergenceWarning, stacklevel=3)
        return self

    def _prediction_input(self, X, method):
        check_is_fitted(self)
        xp = _namespace(X)
        if _namespace(self.coef_) is not xp:
            raise ValueError(f"{type(self).__name__}.{method}() must use the same namespace as fit().")
        if xp is not np:
            X = _array_api_features(X, xp)
            if X.shape[1] != self.n_features_in_:
                raise ValueError(f"X must have {self.n_features_in_} features.")
            if X.device != self.coef_.device or X.dtype != self.coef_.dtype:
                raise ValueError("Prediction X must match the fitted device and dtype.")
            return X, xp
        return validate_data(self, X, reset=False, dtype=self.coef_.dtype, accept_sparse=False), xp

    def predict_contributions(self, X):
        """Return the intercept and additive effects, assigning interactions to predictors."""
        X, xp = self._prediction_input(X, "predict_contributions")
        effects = X * self.coef_
        concat = np.concatenate if xp is np else xp.concat
        zeros = xp.zeros_like(X[:, :1])
        if self.modifier_indices_.shape[0]:
            predictors = xp.take(X, self.predictor_indices_, axis=1)
            modifiers = xp.take(X, self.modifier_indices_, axis=1)
            interaction = predictors * (modifiers @ self.interaction_coef_.T)
            order = xp.argsort(concat((self.predictor_indices_, self.modifier_indices_)))
            effects = effects + xp.take(concat((interaction, xp.zeros_like(modifiers)), axis=1), order, axis=1)
        return concat((self.intercept_ + zeros, effects), axis=1)

    def _linear_score(self, X, method):
        X, xp = self._prediction_input(X, method)
        if xp is not np:
            result = self.intercept_ + X @ self.coef_
            if self.modifier_indices_.shape[0]:
                x = xp.take(X, self.predictor_indices_, axis=1)
                z = xp.take(X, self.modifier_indices_, axis=1)
                result = result + _interaction_product(x, z, self.interaction_coef_)
            return result
        result = self.intercept_ + X @ self.coef_
        if self.modifier_indices_.shape[0]:
            result += _interaction_product(X[:, self.predictor_indices_],
                                           X[:, self.modifier_indices_],
                                           self.interaction_coef_)
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
    predictor_indices_ : array of int
        Original input columns used as predictors.
    modifier_indices_ : array of int
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
        return self._linear_score(X, "predict")


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
        return self._linear_score(X, "decision_function")

    def predict_proba(self, X):
        """Return class probabilities in classes_ order."""
        scores = self._linear_score(X, "predict_proba")
        xp = _namespace(scores)
        if xp is not np:
            return xp.stack((_sigmoid(-scores, xp), _sigmoid(scores, xp)), axis=1)
        return np.column_stack((expit(-scores), expit(scores)))

    def predict(self, X):
        """Predict the most probable class."""
        positive = self._linear_score(X, "predict") > 0
        xp = _namespace(positive)
        if xp is not np:
            return xp.where(positive, self.classes_[1], self.classes_[0])
        return self.classes_[positive.astype(np.intp)]
