"""Export fitted Structured Linear equations for scoring."""

import math

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._category_fusion import _CategoryFusion
from ._fused import _FusedScorecard
from ._pliable import _PliableBase
from ._shared import _DataSharedLasso


def _scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("Export requires finite numeric or string values.")


def _equation(intercept, coefficients, columns):
    return {
        "intercept": _scalar(float(intercept)),
        "terms": [
            {"columns": [column], "coefficient": _scalar(float(coefficient))}
            for column, coefficient in zip(columns, coefficients)
        ],
        "lookups": [],
    }


def to_dict(model):
    """Return a JSON-compatible description of a fitted scoring equation."""
    if not isinstance(model, (_CategoryFusion, _DataSharedLasso, _FusedScorecard, _PliableBase)):
        raise TypeError("Expected a Structured Linear estimator.")
    check_is_fitted(model)
    n_features = model.n_features_in_
    spec = {
        "version": 1,
        "features": list(getattr(model, "feature_names_in_", range(n_features))),
        "classes": [_scalar(label) for label in getattr(model, "classes_", ())],
    }
    if isinstance(model, (_CategoryFusion, _DataSharedLasso)):
        segment = int(model.segment_feature_)
        columns = [i for i in range(n_features) if i != segment]
        shared = isinstance(model, _DataSharedLasso)
        intercepts = [model.intercept_] * len(model.segments_) if shared else model.intercept_
        coefficients = model.segment_coef_ if shared else model.coef_
        spec.update(
            segment_column=segment,
            segments=[
                {"value": _scalar(label), "equation": _equation(intercept, slopes, columns)}
                for label, intercept, slopes in zip(model.segments_, intercepts, coefficients)
            ],
            equation=_equation(model.intercept_, model.shared_coef_, columns) if shared else None,
        )
    elif isinstance(model, _FusedScorecard):
        equation = _equation(model.intercept_, (), ())
        equation["lookups"] = [
            {"column": i, "values": [[_scalar(level), _scalar(score)]
                                     for level, score in zip(levels, scores)]}
            for i, (levels, scores) in enumerate(zip(model.bin_levels_, model.bin_scores_))
        ]
        spec["equation"] = equation
    else:
        equation = _equation(model.intercept_, model.coef_, range(n_features))
        equation["terms"].extend(
            {"columns": [int(i), int(j)], "coefficient": _scalar(float(coefficient))}
            for i, row in zip(model.predictor_indices_, model.interaction_coef_)
            for j, coefficient in zip(model.modifier_indices_, row)
        )
        spec["equation"] = equation
    return spec


def _literal(value):
    value = _scalar(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return repr(value)


def to_sql(spec, columns=None, *, probability=False):
    """Compile a version-1 scoring dictionary to a quoted SQL expression."""
    if spec["version"] != 1:
        raise ValueError("Unsupported scoring dictionary version.")
    columns = spec["features"] if columns is None else list(columns)
    if len(columns) != len(spec["features"]) or any(not isinstance(col, str) for col in columns):
        raise ValueError("Supply one SQL column name per model feature.")
    names = ['"' + col.replace('"', '""') + '"' for col in columns]

    def equation_sql(equation):
        if equation is None:
            return "NULL"
        parts = [_literal(equation["intercept"])]
        for term in equation["terms"]:
            factors = [f"({names[index]})" for index in term["columns"]]
            parts.append("(" + " * ".join([_literal(term["coefficient"]), *factors]) + ")")
        for lookup in equation["lookups"]:
            cases = " ".join(
                f"WHEN {_literal(value)} THEN {_literal(effect)}"
                for value, effect in lookup["values"]
            )
            parts.append(f"(CASE {names[lookup['column']]} {cases} ELSE NULL END)")
        return "(" + " + ".join(parts) + ")"

    result = equation_sql(spec["equation"])
    if "segment_column" in spec:
        column = names[spec["segment_column"]]
        cases = " ".join(
            f"WHEN {_literal(segment['value'])} THEN {equation_sql(segment['equation'])}"
            for segment in spec["segments"]
        )
        result = f"(CASE {column} {cases} ELSE {result} END)"
    result = f"(CASE WHEN {' OR '.join(name + ' IS NULL' for name in names)} THEN NULL ELSE {result} END)"
    if probability:
        if len(spec["classes"]) != 2:
            raise ValueError("Probability export requires a binary classifier.")
        result = (
            f"(CASE WHEN {result} >= 0 THEN 1.0 / (1.0 + EXP(-({result}))) "
            f"ELSE EXP({result}) / (1.0 + EXP({result})) END)"
        )
    return result
