"""
Extract FLAML hyperparameter ontology.

The extraction works in two steps:
    1. Framework introspection: cls.search_space() exposes the tunable HPs with ranges.
    2. Wrapped Library Class introspection: inspect.signature() is used on the underlying library classes

Outputs: /meta-data/ontologies/flaml_ontology_mapping.csv

Run from root:
    python meta-data/ontology-creators/extract_flaml.py
"""

import inspect
import importlib
from pathlib import Path
import extraction_utils
import flaml

OUT = extraction_utils.out_path("flaml_ontology_mapping.csv")
FIELDS = extraction_utils.FIELDS


def domain_to_str(domain):
    # Convert the domain type to a string
    name = type(domain).__name__
    if hasattr(domain, "lower") and hasattr(domain, "upper"):
        return f"{name}({domain.lower}, {domain.upper})"
    if hasattr(domain, "categories"):
        return f"Categorical({domain.categories})"
    return str(domain)


def full_api_rows_for(cls_name, underlying_cls, seen_params):
    # Retrieve rows for all class parameters not already covered by step 1
    rows = []
    try:
        sig = inspect.signature(underlying_cls.__init__)
    except (ValueError, TypeError):
        return rows

    param_docs = extraction_utils.parse_numpy_params(underlying_cls)

    for pname, param in sig.parameters.items():
        if pname in ("self", "args", "kwargs") or pname in seen_params:
            continue
        default = "" if param.default is inspect.Parameter.empty else param.default
        vtype = type(default).__name__ if default != "" else ""
        rows.append(
            {
                "framework": "flaml",
                "family": "model",
                "class": cls_name,
                "parameter_name": pname,
                "value_default": str(default),
                "value_type": vtype,
                "description": param_docs.get(pname, ""),
            }
        )
    return rows


def extract_estimator(cls):
    rows = []
    cls_name = cls.__name__
    seen_params = set()

    # Step 1: search space, get search space by inspecting dummy initiations
    space = None
    for task in ("classification", "regression", "binary"):
        try:
            space = cls.search_space(data_size=(1000, 10), task=task)
            if space:
                break
        except Exception:
            pass
    if space is None:
        try:
            space = cls.space
        except AttributeError:
            space = {}

    if isinstance(space, dict):
        try:
            inst = cls()
            defaults = inst.get_params() if hasattr(inst, "get_params") else {}
        except Exception:
            defaults = {}

        for param, domain in space.items():
            seen_params.add(param)
            rows.append(
                {
                    "framework": "flaml",
                    "family": "model",
                    "class": cls_name,
                    "parameter_name": param,
                    "value_default": str(defaults.get(param, "")),
                    "value_type": domain_to_str(domain),
                    "description": "",
                }
            )

    # Step 2: fully exposed class documentation from the underlying library classes.
    underlying = UNDERLYING_for(cls_name)
    if underlying is None:
        for attr in ("estimator_class", "base_class", "_model_type"):
            underlying = getattr(cls, attr, None)
            if underlying is not None:
                break
    if underlying is not None and underlying is not cls:
        rows.extend(full_api_rows_for(cls_name, underlying, seen_params))

    return rows


def UNDERLYING_for(cls_name: str):
    # Underlying libraries sklearn, ligtbm, xgboost, catboost which are used to expose classes
    mapping = {
        "LGBMEstimator": ("lightgbm", "LGBMClassifier"),
        "XGBoostEstimator": ("xgboost", "XGBClassifier"),
        "XGBoostSklearnEstimator": ("xgboost", "XGBClassifier"),
        "XGBoostLimitDepthEstimator": ("xgboost", "XGBClassifier"),
        "RandomForestEstimator": ("sklearn.ensemble", "RandomForestClassifier"),
        "ExtraTreesEstimator": ("sklearn.ensemble", "ExtraTreesClassifier"),
        "LRL1Classifier": ("sklearn.linear_model", "LogisticRegression"),
        "LRL2Classifier": ("sklearn.linear_model", "LogisticRegression"),
        "SGDEstimator": ("sklearn.linear_model", "SGDClassifier"),
        "CatBoostEstimator": ("catboost", "CatBoostClassifier"),
        "ElasticNetEstimator": ("sklearn.linear_model", "ElasticNet"),
        "KNeighborsEstimator": ("sklearn.neighbors", "KNeighborsClassifier"),
        "LassoLarsEstimator": ("sklearn.linear_model", "LassoLars"),
        "SVCEstimator": ("sklearn.svm", "LinearSVC"),
        "TransformersEstimator": None,
        "TransformersEstimatorModelSelection": None,
    }
    entry = mapping.get(cls_name)
    if entry is None:
        return None
    mod_name, cls_attr = entry
    return extraction_utils.attempt_import(mod_name, cls_attr)


def collect_estimators():
    # Install estimators from flaml
    from flaml.automl.model import (
        LGBMEstimator,
        XGBoostEstimator,
        XGBoostSklearnEstimator,
        RandomForestEstimator,
        ExtraTreesEstimator,
        LRL1Classifier,
        LRL2Classifier,
        SGDEstimator,
    )

    estimators = [
        LGBMEstimator,
        XGBoostEstimator,
        XGBoostSklearnEstimator,
        RandomForestEstimator,
        ExtraTreesEstimator,
        LRL1Classifier,
        LRL2Classifier,
        SGDEstimator,
    ]

    optional = [
        ("flaml.automl.model", "CatBoostEstimator"),
        ("flaml.automl.model", "ElasticNetEstimator"),
        ("flaml.automl.model", "KNeighborsEstimator"),
        ("flaml.automl.model", "LassoLarsEstimator"),
        ("flaml.automl.model", "SVCEstimator"),
        ("flaml.automl.model", "XGBoostLimitDepthEstimator"),
        ("flaml.automl.model", "TransformersEstimator"),
        ("flaml.automl.model", "TransformersEstimatorModelSelection"),
    ]


    for module_path, cls_name in optional:
        try:
            mod = importlib.import_module(module_path)
            estimators.append(getattr(mod, cls_name))
        except Exception:
            pass

    return estimators


def main(out=None):
    # 
    estimators = collect_estimators()
    rows = []
    collected_classes = set()

    for cls in estimators:
        cls_name = cls.__name__
        if cls_name in collected_classes:
            continue
        collected_classes.add(cls_name)
        cls_rows = extract_estimator(cls)
        print(f"{cls_name}: {len(cls_rows)} params")
        rows.extend(cls_rows)

    out = Path(out) if out else OUT
    extraction_utils.write_csv(out, rows)
    print(f"Wrote all {len(rows)} rows to: {out}")


if __name__ == "__main__":
    main(out=extraction_utils.cli_out_arg())
