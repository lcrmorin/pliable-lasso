"""Compare four ways of sharing information across German Credit segments."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from compare_classifiers import german_credit
from structured_linear import CategoryFusionClassifier, DataSharedLassoClassifier

ALPHAS = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
NAMES = ("Pooled", "Separate", "Data-shared", "Category fusion")


def candidates(X, y, groups, Z, target_groups):
    scaler = StandardScaler().fit(X)
    X, Z = scaler.transform(X), scaler.transform(Z)
    for alpha in ALPHAS:
        pooled = LogisticRegression(C=1 / (len(y) * alpha)).fit(X, y)
        yield "Pooled", alpha, pooled.predict_proba(Z)[:, 1], None

        separate = np.empty(len(Z))
        for group in np.unique(groups):
            train, test = groups == group, target_groups == group
            model = LogisticRegression(C=1 / (train.sum() * alpha)).fit(X[train], y[train])
            if test.any():
                separate[test] = model.predict_proba(Z[test])[:, 1]
        yield "Separate", alpha, separate, None

        train_X, test_X = np.c_[X, groups], np.c_[Z, target_groups]
        shared = DataSharedLassoClassifier(alpha=alpha).fit(train_X, y)
        yield "Data-shared", alpha, shared.predict_proba(test_X)[:, 1], None

    fusion = CategoryFusionClassifier()
    for model in fusion.fit_path(np.c_[X, groups], y, ALPHAS):
        if not model.converged_:
            raise RuntimeError(f"Category fusion failed to converge at alpha={model.alpha}.")
        p = model.predict_proba(np.c_[Z, target_groups])[:, 1]
        yield "Category fusion", model.alpha, p, dict(zip(model.segments_, model.cluster_labels_))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--german-credit-archive", type=Path)
    args = parser.parse_args()
    _, X, groups, y = german_credit(args.german_credit_archive)
    strata = 2 * groups + y
    train, test = train_test_split(np.arange(len(y)), stratify=strata, random_state=42, test_size=0.25)
    scores = {}
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for fit, validation in cv.split(X[train], strata[train]):
        fit, validation = train[fit], train[validation]
        for name, alpha, p, clusters in candidates(
            X[fit], y[fit], groups[fit], X[validation], groups[validation]
        ):
            losses, counts = scores.setdefault((name, alpha), ([], []))
            losses.append(log_loss(y[validation], p))
            if clusters is not None:
                counts.append(len(set(clusters.values())))
    best = {
        name: min(ALPHAS, key=lambda alpha: np.mean(scores[name, alpha][0]))
        for name in NAMES
    }
    print("Category fusion path (alpha, CV log loss, cluster count by fold):")
    for alpha in ALPHAS:
        losses, counts = scores["Category fusion", alpha]
        print(f"{alpha:g} {np.mean(losses):.4f} {counts}")
    print("Holdout (model, alpha, overall, car, household, other):")
    for name, alpha, p, clusters in candidates(X[train], y[train], groups[train], X[test], groups[test]):
        if alpha != best[name]:
            continue
        losses = [log_loss(y[test], p)]
        for group in np.unique(groups):
            mask = groups[test] == group
            losses.append(log_loss(y[test][mask], p[mask]))
        print(name, alpha, *(f"{loss:.4f}" for loss in losses))
        if clusters is not None:
            print("Selected fit clusters:", {int(k): int(v) for k, v in sorted(clusters.items())})


if __name__ == "__main__":
    main()
