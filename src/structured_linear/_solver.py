"""Matrix-free proximal gradient for the convex pliable-lasso objective.

Fresh implementation from the objective, without code from earlier packages.
The penalty groups are nested: scalar theta entries, each theta row, then
(beta_j, theta_j). Their proximal maps compose in that order.
"""
from dataclasses import dataclass
from math import sqrt

import numpy as np
from array_api_compat import array_namespace, is_torch_array
from scipy.special import expit


@dataclass
class SolverResult:
    coefficients: object
    n_iter: int
    objective: float
    optimality: float
    converged: bool


def _namespace(X):
    if is_torch_array(X):
        return array_namespace(X)
    if isinstance(X, np.ndarray):
        return np
    get_namespace = getattr(X, "__array_namespace__", None)
    return np if get_namespace is None else get_namespace()


def _interaction_product(x, z, theta):
    """Compute one interaction contribution per row on the input namespace."""
    xp = _namespace(x)
    if xp is np:
        return np.einsum("ij,ij->i", x, (theta @ z.T).T)
    return xp.sum(x * xp.matmul(z, xp.matrix_transpose(theta)), axis=1)


class InteractionDesign:
    """Weighted-centered design with a bounded cache for small interactions."""

    def __init__(self, x, z, weight, fit_intercept):
        self.xp = _namespace(x)
        self.device = None if self.xp is np else x.device
        self.x = x
        self.z = z
        self.weight = weight
        self.p = x.shape[1]
        self.k = z.shape[1]
        self.size = self.p + self.k + self.p * self.k
        self.matrix = None
        # Leave larger designs matrix-free. The cache is local to a fit.
        if self.xp is np and self.k and x.shape[0] * self.size * x.itemsize <= 16 * 1024**2:
            self.matrix = np.empty((x.shape[0], self.size), dtype=x.dtype, order="F")
            self.matrix[:, :self.p] = x
            self.matrix[:, self.p:self.p + self.k] = z
            for j in range(self.p):
                start = self.p + self.k + j * self.k
                np.multiply(x[:, j, None], z, out=self.matrix[:, start:start + self.k])
        self.mean = (np.zeros(self.size) if self.xp is np else
                     self.xp.zeros(self.size, dtype=x.dtype, device=self.device))
        if fit_intercept:
            self.mean = self.rmatvec(weight)

    def unpack(self, v):
        return (v[:self.p], v[self.p:self.p + self.k],
                self.xp.reshape(v[self.p + self.k:], (self.p, self.k)))

    def matvec(self, v):
        if self.matrix is not None:
            return (self.matrix @ v) - self.mean @ v
        beta, gamma, theta = self.unpack(v)
        result = self.x @ beta
        if self.k:
            result = result + self.z @ gamma
            result = result + _interaction_product(self.x, self.z, theta)
        return result - self.mean @ v

    def rmatvec(self, residual):
        if self.matrix is not None:
            return (self.matrix.T @ residual) - self.mean * residual.sum()
        xp = self.xp
        parts = [self.x.T @ residual, self.z.T @ residual]
        if self.k:
            parts.append(xp.reshape((self.x * residual[:, None]).T @ self.z, (-1,)))
        gradient = np.concatenate(parts) if xp is np else xp.concat(tuple(parts))
        return gradient - self.mean * xp.sum(residual)

    def smooth(self, v, target):
        value, weighted = self.loss(v, target)
        return value, self.rmatvec(weighted)

    def loss(self, v, target):
        residual = self.matvec(v) - target
        weighted = self.weight * residual
        return float(residual @ weighted / 2), weighted


class LogisticInteractionDesign:
    """Add a free intercept to a centered interaction design when requested."""

    def __init__(self, design, fit_intercept):
        self.design = design
        self.xp = design.xp
        self.device = design.device
        self.fit_intercept = fit_intercept
        self.p, self.k = design.p, design.k
        self.size = design.size + int(fit_intercept)

    def rmatvec(self, residual):
        gradient = self.design.rmatvec(residual)
        if self.fit_intercept:
            if self.xp is np:
                gradient = np.append(gradient, residual.sum())
            else:
                intercept_gradient = self.xp.reshape(self.xp.sum(residual), (1,))
                gradient = self.xp.concat((gradient, intercept_gradient))
        return gradient

    def loss(self, v, target):
        eta = self.design.matvec(v[:self.design.size])
        if self.fit_intercept:
            eta = eta + v[-1]
        margin = (2 * target - 1) * eta
        if self.xp is not np:
            xp = self.xp
            derivative = self.design.weight * (1 - 2 * target) * _sigmoid(-margin, xp)
            zero = xp.zeros_like(margin)
            return float(self.design.weight @ xp.logaddexp(zero, -margin)), derivative
        derivative = self.design.weight * (1 - 2 * target) * expit(-margin)
        return float(self.design.weight @ np.logaddexp(0, -margin)), derivative

    def smooth(self, v, target):
        value, derivative = self.loss(v, target)
        return value, self.rmatvec(derivative)


def _sigmoid(v, xp):
    e = xp.exp(-xp.abs(v))
    return xp.where(v >= 0, 1 / (1 + e), e / (1 + e))


def _portable_proximal(v, p, k, step, alpha, l1_ratio, xp):
    beta = v[:p]
    end = p + k + p * k
    theta = xp.reshape(v[p + k:end], (p, k))
    magnitude = xp.abs(theta) - step * alpha * l1_ratio
    theta = xp.sign(theta) * xp.maximum(magnitude, xp.zeros_like(magnitude))
    group = step * alpha * (1 - l1_ratio)

    def shrink(norm):
        safe = xp.where(norm > 0, norm, xp.ones_like(norm))
        return xp.maximum(1 - group / safe, xp.zeros_like(norm))

    theta = theta * shrink(xp.sqrt(xp.sum(theta**2, axis=1)))[:, None]
    factor = shrink(xp.sqrt(beta**2 + xp.sum(theta**2, axis=1)))
    theta = theta * factor[:, None]
    return xp.concat((beta * factor, v[p:p + k], xp.reshape(theta, (-1,)), v[end:]))


def _portable_penalty(v, p, k, alpha, l1_ratio, xp):
    beta = v[:p]
    theta = xp.reshape(v[p + k:p + k + p * k], (p, k))
    squared = xp.sum(theta**2, axis=1)
    groups = xp.sum(xp.sqrt(beta**2 + squared) + xp.sqrt(squared))
    entries = xp.sum(xp.abs(theta))
    return float(alpha * ((1 - l1_ratio) * groups + l1_ratio * entries))


def penalty(v, p, k, alpha, l1_ratio):
    beta = v[:p]
    theta = v[p + k:p + k + p * k].reshape(p, k)
    theta_norm = np.linalg.norm(theta, axis=1)
    return float(alpha * ((1 - l1_ratio) *
                          (np.hypot(beta, theta_norm).sum() + theta_norm.sum())
                          + l1_ratio * np.abs(theta).sum()))


def proximal(v, p, k, step, alpha, l1_ratio):
    """Exact proximal operator for the nested pliable penalty; gamma is free."""
    result = v.copy()
    beta = result[:p]
    theta = result[p + k:p + k + p * k].reshape(p, k)
    lam = step * alpha
    theta[:] = np.sign(theta) * np.maximum(np.abs(theta) - lam * l1_ratio, 0)
    group = lam * (1 - l1_ratio)
    norms = np.linalg.norm(theta, axis=1)
    factors = np.maximum(1 - np.divide(group, norms, out=np.zeros_like(norms),
                                      where=norms > 0), 0)
    theta *= factors[:, None]
    norms = np.hypot(beta, np.linalg.norm(theta, axis=1))
    factors = np.maximum(1 - np.divide(group, norms, out=np.zeros_like(norms),
                                      where=norms > 0), 0)
    beta *= factors
    theta *= factors[:, None]
    return result


def solve(design, target, alpha, l1_ratio, max_iter, tol, initial=None):
    """Accelerated proximal gradient with backtracking and adaptive restart.

    Stop on the infinity norm of the proximal-gradient mapping, relative to
    max(1, ||gradient_at_zero||_inf). This is a stationarity residual, not a
    duality-gap certificate. Every successful line search bounds the smooth
    loss from above; no spectral-norm approximation is required.
    """
    xp = design.xp
    if xp is np:
        zero = np.zeros(design.size)
        prox_operator = proximal
    else:
        zero = xp.zeros(design.size, device=design.device, dtype=target.dtype)

        def prox_operator(v, p, k, step, alpha, l1_ratio):
            return _portable_proximal(v, p, k, step, alpha, l1_ratio, xp)

    current = zero if initial is None else initial
    extrapolated = current
    momentum = 1.
    lipschitz = 1.
    _, initial_gradient = design.smooth(zero, target)
    scale = max(1., float(xp.max(xp.abs(initial_gradient))))
    optimality = np.inf
    converged = False
    roundoff = 1e-14 if xp is np else max(1e-14, 4 * xp.finfo(target.dtype).eps)
    for iteration in range(1, max_iter + 1):
        value, gradient = design.smooth(extrapolated, target)
        for _ in range(100):
            candidate = prox_operator(extrapolated - gradient / lipschitz,
                                 design.p, design.k, 1 / lipschitz, alpha, l1_ratio)
            difference = candidate - extrapolated
            trial, weighted = design.loss(candidate, target)
            bound = value + gradient @ difference + lipschitz / 2 * (difference @ difference)
            slack = roundoff * max(1., abs(value), abs(trial))
            if np.isfinite(trial) and trial <= bound + slack:
                break
            lipschitz *= 2
        else:
            raise FloatingPointError('Could not find a finite descent step; rescale the input.')

        candidate_gradient = design.rmatvec(weighted)
        prox = prox_operator(candidate - candidate_gradient / lipschitz,
                        design.p, design.k, 1 / lipschitz, alpha, l1_ratio)
        optimality = float(lipschitz * xp.max(xp.abs(candidate - prox)) / scale)
        if optimality <= tol:
            current = candidate
            converged = True
            break
        next_momentum = (1 + sqrt(1 + 4 * momentum**2)) / 2
        if (extrapolated - candidate) @ (candidate - current) > 0:
            extrapolated = candidate
            next_momentum = 1.
        else:
            extrapolated = candidate + (momentum - 1) / next_momentum * (candidate - current)
        current = candidate
        momentum = next_momentum
    regularization = (penalty(current, design.p, design.k, alpha, l1_ratio) if xp is np
                      else _portable_penalty(current, design.p, design.k, alpha, l1_ratio, xp))
    objective = trial + regularization
    return SolverResult(current, iteration, objective, optimality, converged)
