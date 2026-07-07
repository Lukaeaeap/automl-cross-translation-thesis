from .base import BaseTranslator
from .value_constraints import SKLEARN_HARD_CONSTRAINTS, clamp_int_min


class AutoGluonTranslator(BaseTranslator):
    FRAMEWORK = "autogluon"

    # framework specific name : canonical param name

    RECEIVE_TRANSFORMS = {
        "seed": "random_state",
        "random_seed": "random_state",
        "seed_value": "random_state",
        "num_boost_round": "n_estimators",
    }
    # canonical name : framework specific param name

    EMIT_TRANSFORMS = {
        "reg_alpha": "lambda_l1",
        "reg_lambda": "lambda_l2",
        "min_child_samples": "min_data_in_leaf",
    }

    # applies constraints, num_leaves and min_data_in_leaf have to be clamped to be compatible
    VALUE_CONSTRAINTS = {
        **SKLEARN_HARD_CONSTRAINTS,
        "num_leaves": clamp_int_min(2),
        "min_data_in_leaf": clamp_int_min(1),
    }
