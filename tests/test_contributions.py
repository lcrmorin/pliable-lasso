import numpy as np
import pytest
from numpy.testing import assert_allclose

from structured_linear import (
    CategoryFusionClassifier,
    CategoryFusionRegressor,
    DataSharedLassoClassifier,
    DataSharedLassoRegressor,
    FusedScorecardClassifier,
    FusedScorecardRegressor,
    PliableClassifier,
    PliableRegressor,
)


@pytest.mark.parametrize("estimator", [
    CategoryFusionClassifier(), CategoryFusionRegressor(),
    DataSharedLassoClassifier(), DataSharedLassoRegressor(),
    FusedScorecardClassifier(), FusedScorecardRegressor(),
    PliableClassifier(modifier_features=[1]), PliableRegressor(modifier_features=[1]),
])
def test_contributions_reconstruct_scores(estimator):
    X = np.tile([[0, 0], [1, 0], [0, 1], [1, 1]], (10, 1))
    y = np.tile([0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                 0, 1, 0, 1], 2)
    estimator.fit(X, y)
    expected = getattr(estimator, "decision_function", estimator.predict)(X)
    effects = estimator.predict_contributions(X)
    assert effects.shape == (len(X), 2 if hasattr(estimator, "segments_") else 3)
    assert_allclose(effects.sum(axis=1), expected, atol=1e-12)


def test_segment_effect_tables_and_shared_fallback():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"segment": ["A", "A", "B", "B"], "duration": [-1, 1, -1, 1]})
    y = [-1, 1, 3, 5]
    fused = CategoryFusionRegressor(segment_feature=0).fit(X, y)
    rows = fused.effect_table()
    assert rows[0] == {"segment": "A", "cluster": int(fused.cluster_labels_[0]),
                       "feature": None, "coefficient": fused.intercept_[0]}
    assert rows[1]["feature"] == "duration"
    assert rows[1]["coefficient"] == fused.coef_[0, 0]
    shared = DataSharedLassoRegressor(segment_feature=0).fit(X, y)
    for i, row in enumerate(shared.effect_table()):
        assert row["segment"] == shared.segments_[i]
        assert row["feature"] == "duration"
        assert row["coefficient"] == row["shared"] + row["deviation"]
    unseen = pd.DataFrame({"segment": ["C"], "duration": [2]})
    assert_allclose(shared.predict_contributions(unseen), [[shared.intercept_, 2 * shared.shared_coef_[0]]])
    with pytest.raises(ValueError, match="Unknown segment"):
        fused.predict_contributions(unseen)


def test_pliable_interaction_assignment_on_array_api():
    xp = pytest.importorskip("array_api_strict")
    X = xp.asarray([[0., -1.], [0., 1.], [1., -1.], [1., 1.]])
    y = xp.asarray([0., 0., -2., 2.])
    model = PliableRegressor(alpha=0.01, modifier_features=[0]).fit(X, y)
    effects = model.predict_contributions(X)
    assert float(xp.max(xp.abs(model.interaction_coef_))) > 0.1
    assert_allclose(np.asarray(effects[:, 1]), np.asarray(X[:, 0] * model.coef_[0]))
    assert_allclose(np.asarray(xp.sum(effects, axis=1)), np.asarray(model.predict(X)), atol=1e-12)
