"""
Reusable value-coercion lambdas for BaseTranslator.VALUE_CONSTRAINTS.

These defend against a target framework's hard type/range requirements
being violated by whatever value lands in that parameter slot. They stop a crash.
"""

from __future__ import annotations

NUMERIC = (int, float)



def is_number(v) -> bool:
    return isinstance(v, NUMERIC) and not isinstance(v, bool)


def clamp_int_min(minimum: int):
    # Simple clamping of ints using minimum of permitted range
    return lambda v: max(minimum, int(v)) if is_number(v) else v


def clamp_float_range(lo: float, hi: float):
    # Clamping of floats using given range
    return lambda v: max(lo, min(hi, float(v))) if is_number(v) else v


def drop_value():
    return lambda v: None



def max_features_constraint(v):
    # sklearn's max_features must be either: int >= 1, or a float in (0, 1], a string like {'sqrt', 'log2'}, or None.
    if v is None:
        return None
    if isinstance(v, str):
        return v if v in ("sqrt", "log2") else None
    if not is_number(v):
        return None
    fv = float(v)
    if fv <= 0:
        return None
    if fv <= 1:
        return fv
    return max(1, int(round(fv)))


def max_depth_constraint(v):
    # sklearn uses max_depth which has to be a positive int or none.
    if v is None:
        return None
    if not is_number(v):
        return None
    fv = float(v)
    if fv <= 0:
        return 1
    return max(1, int(round(fv)))


def max_iter_constraint(v):
    # sklearn's max_iter must be a non-negative int. Negative values (e.g. -1)
    # are an "unlimited" convention some frameworks use but sklearn rejects.
    if not is_number(v):
        return v
    iv = int(v)
    return 1000 if iv < 0 else iv


# Constraints shared by every sklearn module
SKLEARN_HARD_CONSTRAINTS = {
    "min_samples_leaf": clamp_int_min(1),
    "min_samples_split": clamp_int_min(2),
    "min_weight_fraction_leaf": clamp_float_range(0.0, 0.5),
    "max_leaf_nodes": clamp_int_min(2),
    "max_features": max_features_constraint,
    "max_depth": max_depth_constraint,
    "max_iter": max_iter_constraint,
    "oob_score": drop_value(),
}
