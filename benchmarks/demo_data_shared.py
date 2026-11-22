"""Compare pooled, separate and data-shared classifiers on German Credit."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from compare_classifiers import german_credit
from structured_linear import DataSharedLassoClassifier


FEATURES = (
    "Duration", "Credit amount", "Installment rate", "Residence duration",
    "Existing credits", "Dependents", "Age",
)
SEGMENTS = ("Car", "Household", "Other")
ALPHAS = (0.003, 0.01, 0.03, 0.1)
MODELS = ("Pooled + segment indicators", "Separate per segment", "Data-shared lasso")


def fits(X, y, group, Z, target_group):
    scaler = StandardScaler().fit(X)
    X, Z = scaler.transform(X), scaler.transform(Z)
    pooled_X, pooled_Z = np.c_[X, np.eye(3)[group]], np.c_[Z, np.eye(3)[target_group]]
    for alpha in ALPHAS:
        pooled = LogisticRegression(C=1 / (len(y) * alpha), max_iter=1000)
        pooled.fit(pooled_X, y)
        yield (MODELS[0], alpha), pooled.predict_proba(pooled_Z)[:, 1], None

        separate = np.empty(len(Z))
        for segment in range(len(SEGMENTS)):
            train, test = group == segment, target_group == segment
            model = LogisticRegression(C=1 / (train.sum() * alpha), max_iter=1000)
            model.fit(X[train], y[train])
            separate[test] = model.predict_proba(Z[test])[:, 1]
        yield (MODELS[1], alpha), separate, None

        shared = DataSharedLassoClassifier(alpha=alpha).fit(np.c_[pooled_X, group], y)
        p = shared.predict_proba(np.c_[pooled_Z, target_group])[:, 1]
        yield (MODELS[2], alpha), p, shared


def choose(X, y, group):
    scores = {}
    cv = StratifiedKFold(3, shuffle=True, random_state=17)
    for train, test in cv.split(X, 2 * group + y):
        for key, p, _ in fits(X[train], y[train], group[train], X[test], group[test]):
            scores.setdefault(key, []).append(log_loss(y[test], p))
    return {min((key for key in scores if key[0] == name),
                key=lambda key: np.mean(scores[key])) for name in MODELS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--german-credit-archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data-shared-effects.png"))
    args = parser.parse_args()
    _, X, group, y = german_credit(args.german_credit_archive)
    predictions = {name: np.empty(len(y)) for name in MODELS}
    aucs = {name: [] for name in MODELS}
    outer = StratifiedKFold(5, shuffle=True, random_state=42)
    for train, test in outer.split(X, 2 * group + y):
        winners = choose(X[train], y[train], group[train])
        for key, p, _ in fits(X[train], y[train], group[train], X[test], group[test]):
            if key in winners:
                predictions[key[0]][test] = p
                aucs[key[0]].append(roc_auc_score(y[test], p))

    for name, p in predictions.items():
        losses = [log_loss(y[group == g], p[group == g]) for g in range(len(SEGMENTS))]
        print(name, *(f"{value:.4f}" for value in [log_loss(y, p), np.mean(aucs[name]), *losses]))
    print("Segment counts:", np.bincount(group))

    # Refit for interpretation after producing all held-out predictions.
    winners = choose(X, y, group)
    key, _, shared = next(
        (key, p, model) for key, p, model in fits(X, y, group, X, group)
        if key in winners and key[0] == MODELS[2]
    )
    print(f"Full-data fit alpha: {key[1]}")
    shared_effect = shared.shared_coef_[:len(FEATURES)]
    deviations = shared.segment_deviations_[:, :len(FEATURES)]
    positions = np.arange(len(FEATURES))
    fig, (ax, grid) = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    ax.barh(positions, shared_effect, color="#00796b")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set(yticks=positions, yticklabels=FEATURES, title="Shared slopes",
           xlabel="Log-odds change per training SD")
    ax.invert_yaxis()
    limit = max(0.01, np.abs(deviations).max())
    image = grid.imshow(deviations.T, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    grid.set(xticks=np.arange(len(SEGMENTS)),
             xticklabels=[SEGMENTS[int(g)] for g in shared.segments_],
             title="Deviation from shared slope")
    fig.colorbar(image, ax=grid, label="Log-odds change per training SD")
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
