from .base import BaseTranslator

ACTIVATION_TO_CANONICAL = {
    "Rectifier": "relu",
    "RectifierWithDropout": "relu",
    "Tanh": "tanh",
    "TanhWithDropout": "tanh",
    "Maxout": "relu",
    "MaxoutWithDropout": "relu",
    "ExpRectifier": "relu",
    "ExpRectifierWithDropout": "relu",
}
CANONICAL_TO_ACTIVATION = {
    "relu": "Rectifier",
    "tanh": "Tanh",
    "logistic": "Tanh",
    "identity": "Rectifier",
}


def hidden_to_list(v):
    # h2o requires a list of layer widths
    if isinstance(v, (int, float)):
        return [int(v)]
    return v


def mtries_sentinel_to_none(v):
    #sklearn's max_features has no equivalent, it uses none instead.
    if isinstance(v, (int, float)) and v < 0:
        return None
    return v


class H2OTranslator(BaseTranslator):
    FRAMEWORK = "h2o"

    # framework specific name : canonical param name

    RECEIVE_TRANSFORMS = {
        "ntrees": "n_estimators",
        "learn_rate": "learning_rate",
        "min_rows": "min_samples_leaf",
        "min_child_weight": "min_samples_leaf",
        "sample_rate": "subsample",
        "col_sample_rate": "colsample_bytree",
        "col_sample_rate_per_tree": "colsample_bytree",
        "seed": "random_state",
        "activation": ("activation", lambda v: ACTIVATION_TO_CANONICAL.get(v, v)),
    }

    # canonical name : framework specific param name

    EMIT_TRANSFORMS = {
        "n_estimators": "ntrees",
        "learning_rate": "learn_rate",
        "min_samples_leaf": "min_rows",
        "subsample": "sample_rate",
        "colsample_bytree": "col_sample_rate",
        "random_state": "seed",
        "activation": ("activation", lambda v: CANONICAL_TO_ACTIVATION.get(v, v)),
    }

    VALUE_CONSTRAINTS = {
        "hidden": hidden_to_list,
        "mtries": mtries_sentinel_to_none,
    }
