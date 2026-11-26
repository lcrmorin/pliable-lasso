"""Primal-dual solver for complete-graph vector fusion."""

import numpy as np
from scipy.special import expit


def _solve_category_fusion(designs, targets, weights, alpha, logistic, max_iter, tol):
    n_groups = len(designs)
    n_parameters = designs[0].shape[1]
    left, right = np.triu_indices(n_groups, k=1)
    beta = np.zeros((n_groups, n_parameters))
    dual = np.zeros((len(left), n_parameters))

    def differences(value):
        return value[left] - value[right]

    def gradient(value):
        result = []
        for design, target, weight, row in zip(designs, targets, weights, value):
            score = design @ row
            residual = expit(score) - target if logistic else score - target
            result.append(design.T @ (weight * residual))
        return np.asarray(result)

    lipschitz = max(
        np.linalg.norm(design * np.sqrt(weight[:, None]), ord=2) ** 2
        for design, weight in zip(designs, weights)
    )
    if logistic:
        lipschitz *= 0.25

    # The complete-graph incidence operator has squared norm n_groups.
    # These steps satisfy 1/primal_step - dual_step * ||D||² > L/2.
    primal_step = 1 / lipschitz
    dual_step = lipschitz / (4 * n_groups)
    for iteration in range(1, max_iter + 1):
        grad = gradient(beta)
        np.add.at(grad, left, dual)
        np.add.at(grad, right, -dual)
        next_beta = beta - primal_step * grad
        candidate = dual + dual_step * differences(2 * next_beta - beta)
        norms = np.linalg.norm(candidate, axis=1, keepdims=True)
        next_dual = candidate * (alpha / np.maximum(alpha, norms))

        if (
            np.max(np.abs(next_beta - beta)) <= tol * (1 + np.max(np.abs(beta)))
            and np.max(np.abs(next_dual - dual), initial=0)
            <= dual_step * tol * (1 + np.max(np.abs(beta)))
        ):
            return next_beta, iteration, True
        beta, dual = next_beta, next_dual

    return beta, max_iter, False
