import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from structured_linear import CategoryFusionClassifier, CategoryFusionRegressor


@pytest.mark.parametrize("model", [CategoryFusionRegressor(), CategoryFusionClassifier()])
def test_category_pipeline_persistence(model, tmp_path):
    X = [[0, "a"], [1, "a"], [2, "a"], [3, "a"], [0, "b"], [1, "b"], [2, "b"], [3, "b"]]
    y = [0, 0, 1, 0, 1, 1, 0, 1]
    pipeline = Pipeline([("transform", FunctionTransformer()), ("model", model)]).fit(X, y)
    filename = tmp_path / "model.joblib"
    joblib.dump(pipeline, filename)
    restored = joblib.load(filename)
    method = "predict_proba" if hasattr(model, "predict_proba") else "predict"
    assert_allclose(getattr(restored, method)(X), getattr(pipeline, method)(X))
    np.testing.assert_array_equal(restored[-1].cluster_labels_, pipeline[-1].cluster_labels_)
