"""Benchmark low-event-rate PliableClassifier fits with one BLAS thread.

Install the checkout first, or put its src directory on PYTHONPATH.
Pass --compare with another checkout's src directory for alternating
timings and a check of objective and probability agreement.
"""
import argparse
import cProfile
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import pstats
from statistics import median
import sys
from time import perf_counter
import tracemalloc

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from pliable_lasso import PliableClassifier

CASES = [(4_000, 40, 2), (24_000, 72, 4), (24_000, 72, 12)]
PENALTIES = {"fit": [0.002], "path": [0.012, 0.006, 0.003, 0.002]}


def fixture(n, p, k):
    rng = np.random.default_rng(42)
    predictors = rng.standard_normal((n, p))
    predictors = 0.8 * predictors + 0.6 * rng.standard_normal((n, 1))
    modifiers = rng.standard_normal((n, k))
    score = -3.7 + 0.55 * predictors[:, 0] + 0.25 * predictors[:, 2]
    score += 0.65 * predictors[:, 1] * modifiers[:, 0]
    score -= 0.45 * predictors[:, 3] * modifiers[:, -1]
    return np.column_stack((predictors, modifiers)), rng.binomial(1, expit(score))


def fit(estimator, X, y, modifiers, penalties):
    model = estimator(modifier_features=modifiers, tol=1e-6, warm_start=len(penalties) > 1)
    metrics = []
    for alpha in penalties:
        model.set_params(alpha=alpha).fit(X, y)
        if not model.converged_:
            raise RuntimeError(f"Fit at alpha={alpha} did not converge.")
        metrics.append({
            "alpha": alpha, "objective": float(model.objective_),
            "stationarity": float(model.optimality_), "iterations": int(model.n_iter_),
        })
    return metrics, model.predict_proba(X)[:, 1]


def load_estimator(source_dir):
    init = source_dir / "pliable_lasso" / "__init__.py"
    spec = spec_from_file_location("comparison_pliable_lasso", init,
                                   submodule_search_locations=[str(init.parent)])
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PliableClassifier


def measure(estimators, fit_args, repeats):
    samples = {label: [] for label in estimators}
    outcomes = {}
    order = list(estimators)
    for _ in range(repeats):
        for label in order:
            start = perf_counter()
            outcomes[label] = fit(estimators[label], *fit_args)
            samples[label].append(perf_counter() - start)
        order.reverse()
    if "comparison" in outcomes:
        reference, reference_prob = outcomes["baseline"]
        comparison, comparison_prob = outcomes["comparison"]
        np.testing.assert_allclose(
            [row["objective"] for row in comparison],
            [row["objective"] for row in reference], rtol=1e-7, atol=1e-10,
        )
        np.testing.assert_allclose(comparison_prob, reference_prob, rtol=1e-6, atol=1e-7)
    for label, estimator in estimators.items():
        tracemalloc.start()
        try:
            fit(estimator, *fit_args)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        yield {
            "implementation": label, "seconds": median(samples[label]),
            "samples": samples[label], "peak_traced_mib": peak / 2**20,
            "fits": outcomes[label][0],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    estimators = {"baseline": PliableClassifier}
    if args.compare:
        estimators["comparison"] = load_estimator(args.compare)

    with threadpool_limits(limits=1):
        for n, p, k in CASES:
            X, y = fixture(n, p, k)
            for task, penalties in PENALTIES.items():
                fit_args = (X, y, list(range(p, p+k)), penalties)
                for result in measure(estimators, fit_args, args.repeats):
                    print(json.dumps({"n": n, "p": p, "k": k, "task": task,
                                      "event_rate": float(y.mean()), **result}), flush=True)
        if args.profile:
            profile = cProfile.Profile()
            profile.runcall(fit, PliableClassifier, X, y, list(range(p, p+k)), PENALTIES["fit"])
            pstats.Stats(profile).strip_dirs().sort_stats("tottime").print_stats(16)


if __name__ == "__main__":
    main()
