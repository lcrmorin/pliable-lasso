import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets


def _binary_target(y):
    check_classification_targets(y)
    classes, codes = np.unique(y, return_inverse=True)
    if len(classes) != 2:
        raise ValueError("Exactly two classes are required.")
    return classes, codes


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")


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
