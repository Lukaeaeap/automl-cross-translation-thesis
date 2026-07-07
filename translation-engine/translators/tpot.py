from .base import BaseTranslator
from .value_constraints import SKLEARN_HARD_CONSTRAINTS


class TPOTTranslator(BaseTranslator):
    FRAMEWORK = "tpot"

    # framework specific name : canonical param name

    RECEIVE_TRANSFORMS = {
        "nthread": "n_jobs",
    }

    # canonical name : framework specific param name

    EMIT_TRANSFORMS = {
        "n_jobs": "nthread",
    }

    VALUE_CONSTRAINTS = dict(SKLEARN_HARD_CONSTRAINTS)
