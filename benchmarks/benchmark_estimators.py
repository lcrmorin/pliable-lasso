"""Single-process fit and prediction timings with reproducible quality checks.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. Compare revisions by
running this same script with each checkout's src directory on PYTHONPATH.
"""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss, mean_squared_error

from structured_linear import (
    DataSharedLassoClassifier,
    DataSharedLassoRegressor,
    FusedScorecardClassifier,
    FusedScorecardRegressor,
    PliableClassifier,
    PliableRegressor,
)


def cases(n, p, modifiers):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, p))
    segments = rng.integers(0, 8, size=n)
    score = X[:, 0] + X[:, 1] * X[:, -1] + 0.25 * (segments == 1)
    continuous = score + rng.normal(size=n)
    labels = rng.binomial(1, expit(score))
    bins = np.clip(np.floor(2 * X + 4), 0, 7)
    orders = [range(8)] * p
    shared = np.column_stack((X, segments))
    options = dict(alpha=0.03)
    for estimator, features in (
        (PliableRegressor(modifier_features=modifiers, **options), X),
        (PliableClassifier(modifier_features=modifiers, **options), X),
        (FusedScorecardRegressor(bin_order=orders, **options), bins),
        (FusedScorecardClassifier(bin_order=orders, **options), bins),
        (DataSharedLassoRegressor(**options), shared),
        (DataSharedLassoClassifier(**options), shared),
    ):
        is_classifier = hasattr(estimator, "predict_proba")
        yield estimator, features, labels if is_classifier else continuous, is_classifier


def run(n, p, k, repeats):
    rows = {}
    split = n * 4 // 5
    for model, X, y, classifier in cases(n, p, list(range(p - k, p))):
        fit_times, predict_times = [], []
        for _ in range(repeats + 1):
            start = perf_counter()
            model.fit(X[:split], y[:split])
            fit_times.append(perf_counter() - start)
            predict = model.predict_proba if classifier else model.predict
            start = perf_counter()
            prediction = predict(X[split:])
            predict_times.append(perf_counter() - start)
        metric = log_loss if classifier else mean_squared_error
        rows[type(model).__name__] = dict(
            fit_seconds=median(fit_times[1:]),
            predict_seconds=median(predict_times[1:]),
            test_loss=metric(y[split:], prediction),
            predictions=prediction.tolist(),
            objective=getattr(model, "objective_", None),
            optimality=getattr(model, "optimality_", None),
            n_iter=getattr(model, "n_iter_", None),
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--modifiers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(
        args.samples, args.features, args.modifiers, args.repeats
    ), indent=2) + "\n")
