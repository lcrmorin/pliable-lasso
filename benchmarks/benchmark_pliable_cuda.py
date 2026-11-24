"""Compare PyTorch CPU and CUDA execution at matched solution quality."""
import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np
import torch
from scipy.special import expit

from structured_linear import PliableClassifier, PliableRegressor


def benchmark(estimator, inputs, device, repeats):
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    started = perf_counter()
    X, y, weight = [array.to(device) for array in inputs]
    synchronize()
    transfer = perf_counter() - started

    fit_times, predict_times, peaks = [], [], []
    for _ in range(repeats + 1):
        prediction = None
        model = estimator(alpha=0.03, modifier_features=[X.shape[1] - 1])
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            allocated = torch.cuda.memory_allocated(device)
        else:
            allocated = None
        started = perf_counter()
        model.fit(X, y, weight)
        synchronize()
        fit_times.append(perf_counter() - started)

        predict = model.predict_proba if isinstance(model, PliableClassifier) else model.predict
        started = perf_counter()
        prediction = predict(X)
        synchronize()
        predict_times.append(perf_counter() - started)
        if allocated is not None:
            peaks.append(torch.cuda.max_memory_allocated(device) - allocated)
        if prediction.device != X.device or model.coef_.device != X.device:
            raise RuntimeError("Fit or inference left the requested device.")
    summary = {
        "fit_seconds": median(fit_times[1:]),
        "predict_seconds": median(predict_times[1:]),
        "input_transfer_seconds": transfer,
        "peak_additional_device_bytes": max(peaks[1:], default=None),
        "objective": model.objective_,
        "optimality": model.optimality_,
        "iterations": model.n_iter_,
        "converged": model.converged_,
    }
    return summary, prediction.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.samples, args.features) < 2 or args.repeats < 1:
        parser.error("Require at least two samples and features and one repeat.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable to PyTorch.")

    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(args.samples, args.features)).astype(args.dtype)
    signal = X[:, 0] + X[:, 0] * X[:, -1]
    weights = torch.ones(args.samples, dtype=getattr(torch, args.dtype))
    outputs = {}
    for estimator, y in (
        (PliableRegressor, signal),
        (PliableClassifier, rng.binomial(1, expit(signal))),
    ):
        inputs = (torch.from_numpy(X), torch.from_numpy(y), weights)
        cpu, expected = benchmark(estimator, inputs, torch.device("cpu"), args.repeats)
        measured, actual = benchmark(estimator, inputs, torch.device(args.device), args.repeats)
        np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-4)
        np.testing.assert_allclose(measured["objective"], cpu["objective"], atol=1e-6)
        if not (measured["converged"] and cpu["converged"]):
            raise RuntimeError("Both runs must converge before comparing timings.")
        outputs[estimator.__name__] = {
            "cpu": cpu,
            args.device + "_measured": measured,
            "max_prediction_error": float(np.max(np.abs(actual - expected))),
        }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
