import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning


def _logistic_lasso(design, y, weight, alpha, max_iter, tol, initial=None):
    """Solve weighted mean logistic loss + alpha * L1 with an unpenalized intercept."""
    p = design.shape[1]
    if initial is None:
        log_odds = np.log(weight[y == 1].sum()) - np.log(weight[y == 0].sum())
        initial = np.r_[log_odds, np.zeros(2 * p)]

    def objective(params):
        positive = params[1:p + 1]
        negative = params[p + 1:]
        score = params[0] + design @ (positive - negative)
        loss = weight @ np.logaddexp(0, (1 - 2 * y) * score)
        residual = weight * (expit(score) - y)
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
