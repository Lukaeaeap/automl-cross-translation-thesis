from .flaml import FLAMLTranslator
from .h2o import H2OTranslator
from .tpot import TPOTTranslator
from .autosklearn import AutoSklearnTranslator
from .autogluon import AutoGluonTranslator

REGISTRY = {
    "flaml": FLAMLTranslator,
    "h2o": H2OTranslator,
    "tpot": TPOTTranslator,
    "autosklearn": AutoSklearnTranslator,
    "autogluon": AutoGluonTranslator,
}
