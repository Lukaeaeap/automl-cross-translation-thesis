from .base import BaseTranslator
from .value_constraints import SKLEARN_HARD_CONSTRAINTS


class FLAMLTranslator(BaseTranslator):
    FRAMEWORK = "flaml"

    # framework specific name : canonical param name

    RECEIVE_TRANSFORMS = {
        "log_max_bin": "max_bin",
        "max_leaves": "max_leaf_nodes",
    }
    # canonical name : framework specific param name

    EMIT_TRANSFORMS = {
        "max_bin": "log_max_bin",
        "max_leaf_nodes": "max_leaves",
    }

    VALUE_CONSTRAINTS = dict(SKLEARN_HARD_CONSTRAINTS)
