"""Compare three structured classifiers with paired nested CV."""
import argparse
import csv
from io import BytesIO
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from structured_linear import (
    DataSharedLassoClassifier,
    FusedScorecardClassifier,
    PliableClassifier,
)


GERMAN_CREDIT_URL = (
    "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
)


def german_credit(archive):
    if archive is None:
        with urlopen(GERMAN_CREDIT_URL, timeout=30) as response:
            archive = BytesIO(response.read())
    with ZipFile(archive) as source, source.open("german.data") as data:
        rows = np.loadtxt(data, dtype=str)
    # Duration, amount, installment rate, residence duration, existing credits,
    # dependents, and age (also used as the continuous modifier).
    X = rows[:, [1, 4, 7, 10, 15, 17, 12]].astype(float)
    purpose = rows[:, 3]
    car = np.isin(purpose, ["A40", "A41"])
    household = np.isin(purpose, ["A42", "A43", "A44", "A45"])
    segment = np.select([car, household], [0, 1], default=2)
    return "german_credit", X, segment, (rows[:, -1] == "2").astype(int)


def datasets(archive=None):
    yield german_credit(archive)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(800, 7))
    g = rng.choice(3, len(X), p=[0.6, 0.3, 0.1])
    group_effect = np.array([-0.5, 0.2, 0.6])[g]
    for name, signal in (
        ("stepwise", 1.6 * np.sign(X[:, 0]) - 1.4 * (abs(X[:, 1]) < 0.6)),
        ("segment_slopes", X[:, 0] + 2 * (g == 1) * X[:, 1] - 2 * (g == 2) * X[:, 2]),
        ("varying_slopes", X[:, 0] + 1.5 * X[:, 1] * X[:, -1]),
    ):
        yield name, X, g, rng.binomial(1, expit(signal + group_effect))


def candidates(X, y, g, Z, h):
    scaler = StandardScaler().fit(X)
    G, H = np.eye(3)[g], np.eye(3)[h]
    pooled_x, pooled_z = np.c_[scaler.transform(X), G], np.c_[scaler.transform(Z), H]
    shared_x, shared_z = np.c_[pooled_x, g], np.c_[pooled_z, h]
    binner = KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="uniform")
    fused_x = np.c_[binner.fit_transform(X), G]
    fused_z = np.c_[binner.transform(Z), H]
    bin_order = [range(n_bins) for n_bins in binner.n_bins_] + [range(2)] * 3
    modifiers = list(range(X.shape[1] - 1, pooled_x.shape[1]))
    for alpha in (0.003, 0.01, 0.03, 0.1):
        for name, model, train, test in (
            ("ridge", LogisticRegression(C=1 / (len(y) * alpha)), pooled_x, pooled_z),
            ("fused", FusedScorecardClassifier(alpha=alpha, bin_order=bin_order), fused_x, fused_z),
            ("data_shared", DataSharedLassoClassifier(alpha=alpha), shared_x, shared_z),
            ("pliable", PliableClassifier(alpha=alpha, modifier_features=modifiers), pooled_x, pooled_z),
        ):
            start = perf_counter()
            model.fit(train, y)
            elapsed = perf_counter() - start
            yield (name, alpha), model.predict_proba(test)[:, 1], elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--german-credit-archive", type=Path,
                        help="Local UCI dataset ZIP. Downloads from UCI if omitted.")
    args = parser.parse_args()
    rows = []
    for dataset, X, g, y in datasets(args.german_credit_archive):
        outer = StratifiedKFold(5, shuffle=True, random_state=42)
        for fold, (train, test) in enumerate(outer.split(X, y)):
            losses = {}
            inner = StratifiedKFold(3, shuffle=True, random_state=fold)
            for a, b in inner.split(X[train], y[train]):
                t, v = train[a], train[b]
                for key, p, _ in candidates(X[t], y[t], g[t], X[v], g[v]):
                    losses.setdefault(key, []).append(log_loss(y[v], p))
            winners = {
                min((key for key in losses if key[0] == name), key=lambda k: np.mean(losses[k]))
                for name in {key[0] for key in losses}
            }
            for key, p, elapsed in candidates(X[train], y[train], g[train], X[test], g[test]):
                if key not in winners:
                    continue
                rows.append(dict(dataset=dataset, fold=fold, model=key[0], alpha=key[1],
                                 log_loss=log_loss(y[test], p),
                                 auc=roc_auc_score(y[test], p),
                                 brier=brier_score_loss(y[test], p), fit_seconds=elapsed))
    output = Path(__file__).with_name("classification-results.csv")
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
