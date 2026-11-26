"""Array API L1 solver adapter for implicit structured designs."""

import warnings

from sklearn.exceptions import ConvergenceWarning

from ._pliable import _array_api_features, _array_api_input, _array_api_weights
from ._solver import _namespace, _sigmoid, solve


def _fit_inputs(estimator, X, y, sample_weight, logistic):
    xp = _namespace(X)
    X = _array_api_features(X, xp)
    y = _array_api_input(y, X, xp)
    if y.shape != (X.shape[0],) or not bool(xp.all(xp.isfinite(y))):
        raise ValueError("y must have shape (n_samples,) and contain finite values.")
    weight = _array_api_weights(sample_weight, X, xp)
    keep = weight > 0
    X, y, weight = xp.take(X, xp.nonzero(keep)[0], axis=0), y[keep], weight[keep]
    if logistic:
        classes = xp.unique_values(y)
        if classes.shape != (2,):
            raise ValueError("Exactly two classes are required.")
        if xp.isdtype(y.dtype, "real floating") and not bool(xp.all(y == xp.floor(y))):
            raise ValueError("Classification targets must be discrete.")
        estimator.classes_ = classes
        y = y == classes[1]
    estimator.n_features_in_ = X.shape[1]
    if hasattr(estimator, "feature_names_in_"):
        del estimator.feature_names_in_
    return X, xp.astype(y, X.dtype), weight


class ArrayLassoDesign:
    """Use a design's products without materializing the expanded matrix."""

    def __init__(self, weight, size, forward, adjoint, logistic):
        self.weight = weight
        self.xp = _namespace(weight)
        self.device = weight.device
        self.p, self.k, self.size = size, 0, size + 1
        self.forward = forward
        self.adjoint = adjoint
        self.logistic = logistic

    def loss(self, params, target):
        xp = self.xp
        score = self.forward(params[:-1]) + params[-1]
        if self.logistic:
            margin = (1 - 2 * target) * score
            value = self.weight @ xp.logaddexp(xp.zeros_like(margin), margin)
            return float(value), self.weight * (_sigmoid(score, xp) - target)
        residual = score - target
        weighted = self.weight * residual
        return float(residual @ weighted / 2), weighted

    def rmatvec(self, residual):
        xp = self.xp
        return xp.concat((self.adjoint(residual), xp.reshape(xp.sum(residual), (1,))))

    def smooth(self, params, target):
        value, residual = self.loss(params, target)
        return value, self.rmatvec(residual)

    def fit(self, target, alpha, max_iter, tol, initial=None):
        if initial is None:
            xp = self.xp
            center = self.weight @ target
            if self.logistic:
                center = xp.log(self.weight @ target) - xp.log(self.weight @ (1 - target))
            initial = xp.concat((
                xp.zeros(self.p, dtype=target.dtype, device=self.device),
                xp.reshape(center, (1,)),
            ))
        result = solve(self, target, alpha, 0, max_iter,
                       max(tol, 4 * self.xp.finfo(target.dtype).eps), initial)
        if not result.converged:
            warnings.warn("Array API L1 solver reached max_iter.", ConvergenceWarning, stacklevel=3)
        return result.coefficients
