# Examples of using translation system.
import sys

sys.path.insert(0, "translation-engine")

from engine import TranslationEngine

engine = TranslationEngine()

# autosklearn -> tpot with gradient_boosting
result = engine.translate(
    config={
        "model": "gradient_boosting",
        "params": {"learning_rate": 0.1, "n_estimators": 200, "max_depth": 4},
    },
    source="autosklearn",
    target="tpot",
)
print("autosklearn -> TPOT (gradient_boosting)")
print(f"model : {result['model']}")
print(f"params: {result['params']}")
print(f"unmapped: {result['unmapped_params']}\n")


# use broadcast to translate from one source framework to all others
# flaml -> Autogluon, Autosklearn, 
result2 = engine.broadcast(
    config={
        "model": "XGBoostSklearnEstimator",
        "params": {"n_estimators": 100, "learning_rate": 0.05},
    },
    source="flaml",
)
print("FLAML -> all frameworks (XGBoostSklearnEstimator)")
for fw, r in result2.items():
    print(f"{fw} -> {r['model']} params: {r['params']}")
