from numbers import Integral, Real

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_scalar


def _binary_target(y):
    check_classification_targets(y)
    classes, codes = np.unique(y, return_inverse=True)
    if len(classes) != 2:
        raise ValueError("Exactly two classes are required.")
    return classes, codes


def _positive(value, name):
    check_scalar(value, name, Real, min_val=0, include_boundaries="neither")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be positive and finite.")


def _validate_solver_controls(max_iter, tol):
    check_scalar(max_iter, "max_iter", Integral, min_val=1)
    _positive(tol, "tol")


class _BinaryClassifier(ClassifierMixin, BaseEstimator):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def predict_proba(self, X):
        probability = expit(self.decision_function(X))
        return np.column_stack((1 - probability, probability))

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
