"""Compare solvers on generated data at a common stationarity tolerance.

Pass the original handoff's _solver.py to include the baseline. Run with
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/compare_solvers.py.
"""
import argparse
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import tracemalloc

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from pliable_lasso._solver import InteractionDesign, penalty, proximal, solve

TOL = 1e-6
RATIO = .5
MAX_ITER = 10000


def block_solve(design, target, alpha, initial):
    """Cyclic block proximal updates with a residual and a full convergence check."""
    p, k = design.p, design.k
    coefficients = np.zeros(design.size) if initial is None else initial.copy()
    residual = design.matvec(coefficients) - target
    groups = [
        np.r_[j, np.arange(p + k + j * k, p + k + (j + 1) * k)]
        for j in range(p)
    ]
    if k:
        groups.append(np.arange(p, p + k))
    blocks = []
    modifiers = np.column_stack((np.ones(len(target)), design.z))
    for j, indices in enumerate(groups):
        if j < p:
            block = design.x[:, j, None] * modifiers
        else:
            block = design.z
        block = block - design.mean[indices]
        weighted_transpose = block.T * design.weight
        lipschitz = max(1e-12, np.linalg.eigvalsh(weighted_transpose @ block)[-1])
        blocks.append((indices, block, weighted_transpose, lipschitz))
    scale = max(1., np.max(np.abs(design.rmatvec(-design.weight * target))))
    for iteration in range(1, MAX_ITER + 1):
        for j, (indices, block, weighted_transpose, lipschitz) in enumerate(blocks):
            old = coefficients[indices]
            update = old - (weighted_transpose @ residual) / lipschitz
            if j < p:
                update = proximal_block(update, lipschitz, alpha)
            residual += block @ (update - old)
            coefficients[indices] = update
        # Refresh residuals and verify all coordinates, including the gamma block.
        residual = design.matvec(coefficients) - target
        gradient = design.rmatvec(design.weight * residual)
        mapped = proximal(coefficients - gradient, p, k, 1., alpha, RATIO)
        if np.max(np.abs(coefficients - mapped)) / scale <= TOL / 2:
            break
    return coefficients, iteration


def proximal_block(update, lipschitz, alpha):
    # The block omits gamma. Insert its unused slots for the common proximal map.
    k = len(update) - 1
    packed = np.concatenate((update[:1], np.zeros(k), update[1:]))
    mapped = proximal(packed, 1, k, 1 / lipschitz, alpha, RATIO)
    return np.concatenate((mapped[:1], mapped[1 + k:]))


def fixture(n, p, k, correlated, dense, weighted):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n, p))
    if correlated:
        x = .6 * x + .8 * rng.normal(size=(n, 1))
    z = rng.normal(size=(n, k))
    weight = rng.uniform(.2, 2., size=n) if weighted else np.ones(n)
    weight /= weight.sum()
    active = p if dense else min(p, 4)
    y = (1. + x[:, :active] @ rng.normal(size=active) / np.sqrt(active)
         + np.einsum('ij,ij->i', x[:, :active],
                     z @ rng.normal(size=(k, active)) / np.sqrt(active * k))
         + z[:, 0] + rng.normal(scale=.2, size=n))
    return x, z, weight, y


def diagnostics(data, coefficients, alpha, predictions=None):
    x, z, weight, y = data
    n, p = x.shape
    k = z.shape[1]
    full = np.column_stack((x, z, (x[:, :, None] * z[:, None, :]).reshape(n, -1)))
    full -= weight @ full
    target = y - weight @ y
    residual = full @ coefficients - target if predictions is None else predictions - y
    gradient = full.T @ (weight * residual)
    scale = max(1., np.max(np.abs(full.T @ (weight * target))))
    mapped = proximal(coefficients - gradient, p, k, 1., alpha, RATIO)
    return {
        'objective': float(residual @ (weight * residual) / 2
                           + penalty(coefficients, p, k, alpha, RATIO)),
        'stationarity': float(np.max(np.abs(coefficients - mapped)) / scale),
        'intercept_residual': float(abs(weight @ residual)),
    }


def run(data, solver, penalties, warm):
    x, z, weight, y = data
    design = InteractionDesign(x, z, weight, True)
    target = y - weight @ y
    coefficients = None
    fits = []
    for alpha in penalties:
        coefficients, iterations = solver(design, target, alpha, coefficients if warm else None)
        fits.append((coefficients, iterations))
    return fits


def measure(data, solver, penalties, warm):
    times = []
    for _ in range(3):
        start = perf_counter()
        fits = run(data, solver, penalties, warm)
        times.append(perf_counter() - start)
    tracemalloc.start()
    try:
        run(data, solver, penalties, warm)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        'first_seconds': times[0],
        'repeat_seconds': float(np.median(times[1:])),
        'peak_traced_bytes': peak,
        'fits': [
            dict(alpha=alpha, iterations=iterations, **diagnostics(data, coefficients, alpha))
            for (coefficients, iterations), alpha in zip(fits, penalties)
        ],
    }


def measure_r(rscript, data, penalties):
    x, z, weight, y = data
    size = x.shape[1] + z.shape[1] + x.shape[1] * z.shape[1]
    script = Path(__file__).with_name('compare_pliable.R')
    with TemporaryDirectory() as tmp:
        inputs, output = Path(tmp) / 'data.csv', Path(tmp) / 'fit.csv'
        np.savetxt(inputs, np.column_stack((x, z, y, len(y) * weight)), delimiter=',')
        durations = []
        for _ in range(3):
            run_r = subprocess.run(
                [rscript, str(script), str(inputs), str(output), str(x.shape[1]),
                 *map(str, penalties)], capture_output=True, text=True,
            )
            if run_r.returncode:
                raise RuntimeError(f'R pliable failed:\n{run_r.stdout}{run_r.stderr}')
            elapsed, pliable_version, r_version = run_r.stdout.splitlines()[-1].split()
            durations.append(float(elapsed))
        rows = np.loadtxt(output, delimiter=',', ndmin=2)
    design = InteractionDesign(x, z, weight, True)
    offset = weight @ y
    fits = []
    for alpha, row in zip(penalties, rows, strict=True):
        reference = solve(design, y - offset, alpha, RATIO, MAX_ITER, TOL / 2)
        predicted = row[size:]
        error = predicted - (offset + design.matvec(reference.coefficients))
        fits.append(dict(alpha=alpha, prediction_rmse=float(np.sqrt(weight @ error**2)),
                         **diagnostics(data, row[:size], alpha, predicted)))
    return {
        'r_version': r_version, 'pliable_version': pliable_version,
        'first_seconds': durations[0], 'repeat_seconds': float(np.median(durations[1:])),
        'peak_traced_bytes': None,
        'fits': fits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', help='Path to the original handoff _solver.py')
    parser.add_argument('--rscript', help='Rscript executable with CRAN pliable installed')
    args = parser.parse_args()
    solvers = {}
    if args.reference:
        spec = importlib.util.spec_from_file_location('handoff_solver', args.reference)
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)

        def baseline(design, target, alpha, initial):
            result = reference.solve(design, target, alpha, RATIO, MAX_ITER, TOL / 2)
            return result.coefficients, result.n_iter

        solvers['handoff'] = baseline, False

    def optimized(design, target, alpha, initial):
        result = solve(design, target, alpha, RATIO, MAX_ITER, TOL / 2, initial)
        return result.coefficients, result.n_iter

    solvers.update(optimized=(optimized, False), optimized_warm=(optimized, True),
                   block=(block_solve, True))
    with threadpool_limits(1):
        print(json.dumps({'versions': {'python': sys.version.split()[0],
                                       **{p: version(p) for p in ('numpy', 'scikit-learn')}},
                          'platform': platform.platform(),
                          'thread_environment': {key: os.environ.get(key) for key in (
                              'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                              'VECLIB_MAXIMUM_THREADS',
                          )},
                          'threadpools': threadpool_info()}))
        for label, dimensions in [('small_sparse', (120, 12, 3, False, False, False)),
                                  ('larger_correlated_dense', (1500, 96, 5, True, True, True))]:
            data = fixture(*dimensions)
            if args.rscript and label == 'small_sparse':
                x, z, weight, y = data
                # R centers each input before building interactions.
                data = x - x.mean(axis=0), z - z.mean(axis=0), weight, y
            for task, penalties in [('fit', [.035]), ('path', [.2, .1, .06, .035])]:
                for name, (solver, warm) in solvers.items():
                    if task == 'fit' and name == 'optimized_warm':
                        continue
                    metrics = measure(data, solver, penalties, warm)
                    print(json.dumps(dict(case=label, task=task, solver=name, **metrics)))
                if args.rscript and label == 'small_sparse':
                    metrics = measure_r(args.rscript, data, penalties)
                    print(json.dumps(dict(case=label, task=task, solver='R pliable', **metrics)))


if __name__ == '__main__':
    main()
