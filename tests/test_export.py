import json
import math
import sqlite3

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
from structured_linear.export import to_dict, to_sql


@pytest.mark.parametrize("model", [
    CategoryFusionClassifier(), CategoryFusionRegressor(),
    DataSharedLassoClassifier(), DataSharedLassoRegressor(),
    FusedScorecardClassifier(), FusedScorecardRegressor(),
    PliableClassifier(alpha=0.01, modifier_features=[1]),
    PliableRegressor(alpha=0.01, modifier_features=[1]),
])
def test_sql_parity(model):
    X = np.tile([[0, 0], [1, 0], [0, 1], [1, 1]], (10, 1))
    y = np.random.default_rng(25).binomial(1, np.tile([.3, .6, .7, .3], 10))
    model.fit(X, y)
    spec = json.loads(json.dumps(to_dict(model), allow_nan=False))
    expression = to_sql(spec, ["x", 'group"id'])
    with sqlite3.connect(":memory:") as conn:
        conn.execute('CREATE TABLE inputs (x, "group""id")')
        conn.executemany("INSERT INTO inputs VALUES (?, ?)", X.tolist())
        actual = [row[0] for row in conn.execute(f"SELECT {expression} FROM inputs")]
        expected = getattr(model, "decision_function", model.predict)(X)
        assert_allclose(actual, expected, atol=1e-12)
        if spec["classes"]:
            conn.create_function("EXP", 1, lambda x: None if x is None else math.exp(x))
            probability = to_sql(spec, ["x", 'group"id'], probability=True)
            actual = [row[0] for row in conn.execute(f"SELECT {probability} FROM inputs")]
            assert_allclose(actual, model.predict_proba(X)[:, 1], atol=1e-12)


def test_sql_segment_fallback_and_quoting():
    X = [[0, "a's"], [1, "a's"], [0, "b"], [1, "b"]]
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE inputs (x, segment)")
        conn.executemany("INSERT INTO inputs VALUES (?, ?)", [*X, [2, "unseen"], [2, None]])
        for model in (DataSharedLassoRegressor(), CategoryFusionRegressor()):
            model.fit(X, [0, 1, 2, 3])
            expression = to_sql(to_dict(model), ["x", "segment"])
            actual = [row[0] for row in conn.execute(f"SELECT {expression} FROM inputs")]
            assert_allclose(actual[:4], model.predict(X))
            assert actual[-1] is None
            if isinstance(model, DataSharedLassoRegressor):
                assert_allclose(actual[-2], model.predict([[2, "unseen"]])[0])
            else:
                assert actual[-2] is None
