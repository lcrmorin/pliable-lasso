import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning


def _logistic_lasso(design, y, alpha, max_iter, tol, initial=None):
    """Solve mean logistic loss + alpha * L1 with an unpenalized intercept."""
    n, p = design.shape
    if initial is None:
        initial = np.r_[np.log(y.mean() / (1 - y.mean())), np.zeros(2 * p)]

    def objective(params):
        positive = params[1:p + 1]
        negative = params[p + 1:]
        score = params[0] + design @ (positive - negative)
        loss = np.logaddexp(0, (1 - 2 * y) * score).mean()
        residual = (expit(score) - y) / n
        gradient = design.T @ residual
        return loss + alpha * (positive.sum() + negative.sum()), np.r_[
            residual.sum(), gradient + alpha, alpha - gradient
        ]

    result = minimize(
        objective, initial, jac=True, method="L-BFGS-B",
        bounds=[(None, None)] + [(0, None)] * (2 * p),
        options={"maxiter": max_iter, "gtol": tol, "ftol": np.finfo(float).eps},
    )
    if not result.success:
        warnings.warn(str(result.message), ConvergenceWarning, stacklevel=3)
    params = result.x
    return float(params[0]), params[1:p + 1] - params[p + 1:], params
