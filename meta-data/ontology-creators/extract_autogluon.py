"""
Extract AutoGluon hyperparameter ontology.

The extraction works in two steps:
    1. Framework introspection: _get_default_searchspace(), "_get_search_space()", and "search_space()" find the searchspace per model.
    2. Sklearn introspection: Find models of which the search space is empty, but that are wrapped around sklearn estimators.

Outputs: /meta-data/ontologies/autogluon_ontology_mapping.csv

Run from root:
    python meta-data/ontology-creators/extract_autogluon.py
"""

import importlib
from pathlib import Path
import extraction_utils
import autogluon.tabular

OUT = extraction_utils.out_path("autogluon_ontology_mapping.csv")
FIELDS = extraction_utils.FIELDS

# models that are wrapped around sklearn estimators
UNDERLYING = {
    "RFModel": ("sklearn.ensemble", "RandomForestClassifier"),
    "XTModel": ("sklearn.ensemble", "ExtraTreesClassifier"),
    "KNNModel": ("sklearn.neighbors", "KNeighborsClassifier"),
    "LinearModel": ("sklearn.linear_model", "LogisticRegression"),
}

# Autogluon models
AG_MODELS = [
    "LGBModel",
    "XGBoostModel",
    "RFModel",
    "XTModel",
    "CatBoostModel",
    "KNNModel",
    "LinearModel",
    "TabularNeuralNetTorchModel",
    "NNFastAiTabularModel",
    "BoostedRulesModel",
    "FTTransformerModel",
    "FastTextModel",
    "VowpalWabbitModel",
]


def domain_to_str(domain):
    # Return domain type as string
    if isinstance(domain, (str, int, float, bool, type(None))):
        return type(domain).__name__
    cls_name = type(domain).__name__
    if hasattr(domain, "lower") and hasattr(domain, "upper"):
        low, high = domain.lower, domain.upper
        return f"{cls_name}({low}, {high})"
    if hasattr(domain, "categories"):
        return f"Categorical({list(domain.categories)})"
    return cls_name


def get_family(cls_name):
    # Get family of class
    name = cls_name.lower()

    if ("gbm" in name and "xgb" not in name) or "lgb" in name:
        return "lgb"
    if "xgb" in name:
        return "xgboost"

    if name.startswith("rf"):
        return "rf"
    if name.startswith("xt"):
        return "xt"

    if "catboost" in name:
        return "catboost"
    if "knn" in name:
        return "knn"
    if "linear" in name:
        return "linear"
    if "ebm" in name:
        return "ebm"

    if "boostedrules" in name or "imodels" in name:
        return "imodels"
    if "ft" in name or "transformer" in name:
        return "automm"
    if "nn" in name or "neural" in name or "torch" in name or "fastai" in name:
        return "nn"

    return "other"


def extract_model(cls_name):
    # Extract models hyperparameters by class name
    rows = []
    try:
        mod = importlib.import_module("autogluon.tabular.models")
        cls = getattr(mod, cls_name)
        model = cls(path="temp_dir")
    except Exception as e:
        print(f"Exception for {cls_name}: {e}")
        return rows

    family = get_family(cls_name)
    extracted = set()

    # Step 1: frameworks search space
    space = None
    for method in ("_get_default_searchspace", "_get_search_space", "search_space"):
        try:
            fn = getattr(model, method)
            space = fn() if method != "search_space" else fn(data_size=(1000, 10))
            if isinstance(space, dict) and space:
                break
        except Exception:
            continue

    if space:
        for k, v in space.items():
            extracted.add(k)

            if isinstance(v, (int, float, str, bool, type(None))):
                default_val = v
            else:
                default_val = getattr(v, "default", "")

            rows.append(
                {
                    "framework": "autogluon",
                    "family": family,
                    "class": cls_name,
                    "parameter_name": k,
                    "value_default": str(default_val),
                    "value_type": domain_to_str(v),
                    "description": "",
                }
            )

    # Step 2: extract underlying sklearn classes from RF, XT, KNN, LinearModel
    entry = UNDERLYING.get(cls_name)
    if entry is None:
        print("Entry is empty!")
    else:
        mod_name, cls_attr = entry
        try:
            underlying = getattr(importlib.import_module(mod_name), cls_attr)
            rows.extend(
                extraction_utils.full_api_rows(
                    "autogluon", family, cls_name, underlying, extracted
                )
            )
        except Exception:
            pass

    return rows


def main(out=None):
    # Run meta data extraction for every model
    rows = []
    for model in AG_MODELS:
        model_rows = extract_model(model)
        print(f"{model}: {len(model_rows)} params")
        rows.extend(model_rows)

    out = Path(out) if out else OUT
    extraction_utils.write_csv(out, rows)

    print(f"Wrote all {len(rows)} rows to: {out}")


if __name__ == "__main__":
    main(out=extraction_utils.cli_out_arg())
