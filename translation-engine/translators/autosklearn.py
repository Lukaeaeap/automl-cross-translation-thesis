from .base import BaseTranslator
from .value_constraints import SKLEARN_HARD_CONSTRAINTS

# auto-sklearn's SGD uses loss name "log", but modern sklearn uses "log_loss".
OLD_TO_NEW_LOSS = {"log": "log_loss"}
NEW_TO_OLD_LOSS = {v: k for k, v in OLD_TO_NEW_LOSS.items()}


class AutoSklearnTranslator(BaseTranslator):
    FRAMEWORK = "autosklearn"

    # framework specific name : canonical param name

    RECEIVE_TRANSFORMS = {
        "early_stopping": "early_stopping",
        "min_samples_split": "min_samples_split",
        "loss": ("loss", lambda v: OLD_TO_NEW_LOSS.get(v, v)),
    }

    # canonical name : framework specific param name

    EMIT_TRANSFORMS = {}

    # applies constraints and new to old loss function
    VALUE_CONSTRAINTS = {
        **SKLEARN_HARD_CONSTRAINTS,
        "loss": lambda v: NEW_TO_OLD_LOSS.get(v, v),
    }
