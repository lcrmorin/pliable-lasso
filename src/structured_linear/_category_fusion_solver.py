"""Primal-dual solver for complete-graph vector fusion."""

import numpy as np
from ._solver import _namespace, _sigmoid


def _solve_category_fusion(designs, targets, weights, alpha, logistic, max_iter, tol):
    n_groups = len(designs)
    n_parameters = designs[0].shape[1]
    left, right = np.triu_indices(n_groups, k=1)
    xp = _namespace(designs[0])
    options = {} if xp is np else {"device": designs[0].device}
    beta = xp.zeros((n_groups, n_parameters), dtype=designs[0].dtype, **options)
    dual = xp.zeros((len(left), n_parameters), dtype=beta.dtype, **options)
    if xp is not np:
        outgoing = [xp.asarray(np.flatnonzero(left == group), **options) for group in range(n_groups)]
        incoming = [xp.asarray(np.flatnonzero(right == group), **options) for group in range(n_groups)]
        left, right = xp.asarray(left, **options), xp.asarray(right, **options)

    def differences(value):
        return xp.take(value, left, axis=0) - xp.take(value, right, axis=0)

    def gradient(value):
        result = []
        for i, (design, target, weight) in enumerate(zip(designs, targets, weights)):
            score = design @ value[i, :]
            residual = _sigmoid(score, xp) - target if logistic else score - target
            result.append(design.T @ (weight * residual))
        return xp.stack(result)

    def curvature(design, weight):
        if xp is np:
            return np.linalg.norm(design * np.sqrt(weight[:, None]), ord=2) ** 2
        gram = design.T @ (weight[:, None] * design)
        return float(xp.max(xp.sum(xp.abs(gram), axis=1)))

    lipschitz = max(curvature(design, weight) for design, weight in zip(designs, weights))
    if logistic:
        lipschitz *= 0.25

    # The complete-graph incidence operator has squared norm n_groups.
    # These steps satisfy 1/primal_step - dual_step * ||D||² > L/2.
    if xp is not np:
        tol = max(tol, 4 * xp.finfo(beta.dtype).eps)
    primal_step = 1 / lipschitz
    dual_step = lipschitz / (4 * n_groups)
    for iteration in range(1, max_iter + 1):
        grad = gradient(beta)
        if xp is np:
            np.add.at(grad, left, dual)
            np.add.at(grad, right, -dual)
        elif n_groups > 1:
            grad = grad + xp.stack([
                xp.sum(xp.take(dual, out, axis=0), axis=0)
                - xp.sum(xp.take(dual, inc, axis=0), axis=0)
                for out, inc in zip(outgoing, incoming)
            ])
        next_beta = beta - primal_step * grad
        candidate = dual + dual_step * differences(2 * next_beta - beta)
        norms = xp.sqrt(xp.sum(candidate * candidate, axis=1, keepdims=True))
        next_dual = candidate * (alpha / xp.maximum(xp.full_like(norms, alpha), norms))
        dual_change = xp.max(xp.abs(next_dual - dual)) if n_groups > 1 else 0
        scale = 1 + xp.max(xp.abs(beta))

        if (
            xp.max(xp.abs(next_beta - beta)) <= tol * scale
            and dual_change <= dual_step * tol * scale
        ):
            return next_beta, iteration, True
        beta, dual = next_beta, next_dual

    return beta, max_iter, False
