"""Structured linear regression and classification estimators."""

from ._category_fusion import CategoryFusionClassifier, CategoryFusionRegressor
from ._fused import FusedScorecardClassifier, FusedScorecardRegressor
from ._pliable import PliableClassifier, PliableRegressor
from ._shared import DataSharedLassoClassifier, DataSharedLassoRegressor

__version__ = "0.1.0"
__all__ = [
    "CategoryFusionClassifier",
    "CategoryFusionRegressor",
    "DataSharedLassoClassifier",
    "DataSharedLassoRegressor",
    "FusedScorecardClassifier",
    "FusedScorecardRegressor",
    "PliableClassifier",
    "PliableRegressor",
]
