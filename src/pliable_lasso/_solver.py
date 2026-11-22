"""Matrix-free proximal gradient for the convex pliable-lasso objective.

Fresh implementation from the objective, without code from earlier packages.
The penalty groups are nested: scalar theta entries, each theta row, then
(beta_j, theta_j). Their proximal maps compose in that order.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass
class SolverResult:
    coefficients: np.ndarray
    n_iter: int
    objective: float
    optimality: float
    converged: bool


class InteractionDesign:
    """Weighted-centered [X, Z, X_j * Z_k], without materializing interactions."""

    def __init__(self, x, z, weight, fit_intercept):
        self.x = x
        self.z = z
        self.weight = weight
        self.p = x.shape[1]
        self.k = z.shape[1]
        self.size = self.p + self.k + self.p * self.k
        self.mean = np.zeros(self.size)
        if fit_intercept:
            self.mean = self.rmatvec(weight)

    def unpack(self, v):
        return (v[:self.p], v[self.p:self.p + self.k],
                v[self.p + self.k:].reshape(self.p, self.k))

    def matvec(self, v):
        beta, gamma, theta = self.unpack(v)
        result = self.x @ beta
        if self.k:
            result = result + self.z @ gamma
            # Match the column-major layout of the estimator's predictor slice.
            interaction = (theta @ self.z.T).T
            result += np.einsum('ij,ij->i', self.x, interaction)
        return result - self.mean @ v

    def rmatvec(self, residual):
        parts = [self.x.T @ residual, self.z.T @ residual]
        if self.k:
            parts.append(((self.x * residual[:, None]).T @ self.z).ravel())
        return np.concatenate(parts) - self.mean * residual.sum()

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
        self.fit_intercept = fit_intercept
        self.p, self.k = design.p, design.k
        self.size = design.size + int(fit_intercept)

    def rmatvec(self, residual):
        gradient = self.design.rmatvec(residual)
        if self.fit_intercept:
            gradient = np.append(gradient, residual.sum())
        return gradient

    def loss(self, v, target):
        eta = self.design.matvec(v[:self.design.size])
        if self.fit_intercept:
            eta += v[-1]
        margin = (2 * target - 1) * eta
        derivative = self.design.weight * (1 - 2 * target) * expit(-margin)
        return float(self.design.weight @ np.logaddexp(0, -margin)), derivative

    def smooth(self, v, target):
        value, derivative = self.loss(v, target)
        return value, self.rmatvec(derivative)


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
    zero = np.zeros(design.size)
    current = zero if initial is None else initial
    extrapolated = current.copy()
    momentum = 1.
    lipschitz = 1.
    _, initial_gradient = design.smooth(zero, target)
    scale = max(1., float(np.max(np.abs(initial_gradient))))
    optimality = np.inf
    converged = False
    for iteration in range(1, max_iter + 1):
        value, gradient = design.smooth(extrapolated, target)
        for _ in range(100):
            candidate = proximal(extrapolated - gradient / lipschitz,
                                 design.p, design.k, 1 / lipschitz, alpha, l1_ratio)
            difference = candidate - extrapolated
            trial, weighted = design.loss(candidate, target)
            bound = value + gradient @ difference + lipschitz / 2 * (difference @ difference)
            slack = 1e-14 * max(1., abs(value), abs(trial))
            if np.isfinite(trial) and trial <= bound + slack:
                break
            lipschitz *= 2
        else:
            raise FloatingPointError('Could not find a finite descent step; rescale the input.')

        candidate_gradient = design.rmatvec(weighted)
        prox = proximal(candidate - candidate_gradient / lipschitz,
                        design.p, design.k, 1 / lipschitz, alpha, l1_ratio)
        optimality = float(lipschitz * np.max(np.abs(candidate - prox)) / scale)
        if optimality <= tol:
            current = candidate
            converged = True
            break
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        if (extrapolated - candidate) @ (candidate - current) > 0:
            extrapolated = candidate.copy()
            next_momentum = 1.
        else:
            extrapolated = candidate + (momentum - 1) / next_momentum * (candidate - current)
        current = candidate
        momentum = next_momentum
    objective = trial + penalty(current, design.p, design.k, alpha, l1_ratio)
    return SolverResult(current, iteration, objective, optimality, converged)
