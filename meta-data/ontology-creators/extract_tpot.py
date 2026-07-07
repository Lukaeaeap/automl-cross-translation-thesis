"""
Extract TPOT hyperparameter ontology.

The extraction works in two steps:
    1. Framework introspection: TPOT config dicts for tunable search-space params.
    2. Sklearn introspection: inspect.signature() on the underlying class.

Outputs: /meta-data/ontologies/tpot_ontology_mapping.csv

Run from root:
    python meta-data/ontology-creators/extract_tpot.py
"""

import importlib
import warnings
from pathlib import Path

import extraction_utils
from extraction_utils import attempt_import
from tpot.config import classifier_config_dict, regressor_config_dict

warnings.filterwarnings("ignore")

OUT = extraction_utils.out_path("tpot_ontology_mapping.csv")
FIELDS = extraction_utils.FIELDS


# Maps TPOT class names to underlying library's classes for API and docstring extraction
UNDERLYING = {
    "AdaBoostClassifier": attempt_import("sklearn.ensemble", "AdaBoostClassifier"),
    "BaggingClassifier": attempt_import("sklearn.ensemble", "BaggingClassifier"),
    "BernoulliNB": attempt_import("sklearn.naive_bayes", "BernoulliNB"),
    "DecisionTreeClassifier": attempt_import("sklearn.tree", "DecisionTreeClassifier"),
    "ExtraTreesClassifier": attempt_import("sklearn.ensemble", "ExtraTreesClassifier"),
    "GaussianNB": attempt_import("sklearn.naive_bayes", "GaussianNB"),
    "GaussianProcessClassifier": attempt_import("sklearn.gaussian_process", "GaussianProcessClassifier"),
    "GradientBoostingClassifier": attempt_import("sklearn.ensemble", "GradientBoostingClassifier"),
    "HistGradientBoostingClassifier": attempt_import("sklearn.ensemble", "HistGradientBoostingClassifier"),
    "KNeighborsClassifier": attempt_import("sklearn.neighbors", "KNeighborsClassifier"),
    "LGBMClassifier": attempt_import("lightgbm", "LGBMClassifier"),
    "LinearDiscriminantAnalysis": attempt_import("sklearn.discriminant_analysis", "LinearDiscriminantAnalysis"),
    "LinearSVC": attempt_import("sklearn.svm", "LinearSVC"),
    "LogisticRegression": attempt_import("sklearn.linear_model", "LogisticRegression"),
    "MLPClassifier": attempt_import("sklearn.neural_network", "MLPClassifier"),
    "MultinomialNB": attempt_import("sklearn.naive_bayes", "MultinomialNB"),
    "PassiveAggressiveClassifier": attempt_import("sklearn.linear_model", "PassiveAggressiveClassifier"),
    "QuadraticDiscriminantAnalysis": attempt_import("sklearn.discriminant_analysis", "QuadraticDiscriminantAnalysis"),
    "RandomForestClassifier": attempt_import("sklearn.ensemble", "RandomForestClassifier"),
    "SGDClassifier": attempt_import("sklearn.linear_model", "SGDClassifier"),
    "SVC": attempt_import("sklearn.svm", "SVC"),
    "XGBClassifier": attempt_import("xgboost", "XGBClassifier"),
    "ARDRegression": attempt_import("sklearn.linear_model", "ARDRegression"),
    "AdaBoostRegressor": attempt_import("sklearn.ensemble", "AdaBoostRegressor"),
    "BaggingRegressor": attempt_import("sklearn.ensemble", "BaggingRegressor"),
    "BayesianRidge": attempt_import("sklearn.linear_model", "BayesianRidge"),
    "DecisionTreeRegressor": attempt_import("sklearn.tree", "DecisionTreeRegressor"),
    "ElasticNet": attempt_import("sklearn.linear_model", "ElasticNet"),
    "ExtraTreesRegressor": attempt_import("sklearn.ensemble", "ExtraTreesRegressor"),
    "GaussianProcessRegressor": attempt_import("sklearn.gaussian_process", "GaussianProcessRegressor"),
    "GradientBoostingRegressor": attempt_import("sklearn.ensemble", "GradientBoostingRegressor"),
    "HistGradientBoostingRegressor": attempt_import("sklearn.ensemble", "HistGradientBoostingRegressor"),
    "KNeighborsRegressor": attempt_import("sklearn.neighbors", "KNeighborsRegressor"),
    "LGBMRegressor": attempt_import("lightgbm", "LGBMRegressor"),
    "Lars": attempt_import("sklearn.linear_model", "Lars"),
    "Lasso": attempt_import("sklearn.linear_model", "Lasso"),
    "LassoLars": attempt_import("sklearn.linear_model", "LassoLars"),
    "LinearSVR": attempt_import("sklearn.svm", "LinearSVR"),
    "MLPRegressor": attempt_import("sklearn.neural_network", "MLPRegressor"),
    "OrthogonalMatchingPursuit": attempt_import("sklearn.linear_model", "OrthogonalMatchingPursuit"),
    "Perceptron": attempt_import("sklearn.linear_model", "Perceptron"),
    "RandomForestRegressor": attempt_import("sklearn.ensemble", "RandomForestRegressor"),
    "Ridge": attempt_import("sklearn.linear_model", "Ridge"),
    "SGDRegressor": attempt_import("sklearn.linear_model", "SGDRegressor"),
    "SVR": attempt_import("sklearn.svm", "SVR"),
    "TheilSenRegressor": attempt_import("sklearn.linear_model", "TheilSenRegressor"),
    "XGBRegressor": attempt_import("xgboost", "XGBRegressor"),
}

parse_numpy_params = extraction_utils.parse_numpy_params


def infer_type(values):
    # Infer the type from the passed value
    if values is None or len(values) == 0:
        return "unknown"
    if isinstance(values, dict):
        return "nested"
    sample = [v for v in values if v is not None and v is not True and v is not False]
    bools = [v for v in values if isinstance(v, bool)]
    ints = [v for v in sample if isinstance(v, int)]
    floats = [v for v in sample if isinstance(v, float)]
    strings = [v for v in sample if isinstance(v, str)]
    if bools and len(bools) == len(values):
        return "bool"
    if strings:
        return f"Categorical({values})"
    if floats or (ints and any(v != int(v) for v in floats)):
        lo, hi = min(floats or ints), max(floats or ints)
        return f"Float({lo}, {hi})"
    if ints:
        lo, hi = min(ints), max(ints)
        return f"Integer({lo}, {hi})"
    return f"Categorical({values})"


def load_class(full_name):
    # Loads class from module path
    module_path, cls_name = full_name.rsplit(".", 1)
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)
    except Exception:
        return None


def extract_step1(config_dict, family):
    # Step 1: config dict (search space)
    rows = []
    for full_name, param_space in config_dict.items():
        cls_name = full_name.rsplit(".", 1)[-1]
        cls = load_class(full_name)
        if cls is None:
            continue
        underlying = UNDERLYING.get(cls_name)
        lib_descs = parse_numpy_params(underlying or cls)
        try:
            defaults = cls().get_params()
        except Exception:
            defaults = {}

        if not param_space:
            rows.append(
                {
                    "framework": "tpot",
                    "family": family,
                    "class": cls_name,
                    "parameter_name": "(none)",
                    "value_default": "",
                    "value_type": "fixed",
                    "description": "No tunable hyperparameters",
                }
            )
            continue

        for param, values in param_space.items():
            rows.append(
                {
                    "framework": "tpot",
                    "family": family,
                    "class": cls_name,
                    "parameter_name": param,
                    "value_default": str(defaults.get(param, "")),
                    "value_type": infer_type(values)
                    if not isinstance(values, dict)
                    else "nested",
                    "description": lib_descs.get(param, ""),
                }
            )
    return rows


def full_api_rows_for(family, cls_name, underlying_cls, already_captured):
    return extraction_utils.full_api_rows(
        "tpot", family, cls_name, underlying_cls, already_captured
    )


def add_full_api_step(step1_rows):
    # Step 2: extend models with sklearn classes parameters
    from collections import defaultdict

    captured = defaultdict(set)
    family_map = {}
    for row in step1_rows:
        key = row["class"]
        captured[key].add(row["parameter_name"])
        family_map.setdefault(key, row["family"])

    extra = []
    for cls_name, underlying in UNDERLYING.items():
        if underlying is None:
            continue
        already = captured.get(cls_name, set())
        family = family_map.get(cls_name, "classification")
        extra.extend(extraction_utils.full_api_rows(
        "tpot", family, cls_name, underlying, already))
    return extra


def main(out=None):
    # Run meta data extraction for every model

    # Step 1
    rows = extract_step1(classifier_config_dict, "classification")
    rows += extract_step1(regressor_config_dict, "regression")

    # Step 2
    rows += add_full_api_step(rows)

    rows = extraction_utils.dedup_rows(rows)

    out = Path(out) if out else OUT
    extraction_utils.write_csv(out, rows)

    print(f"Wrote all {len(rows)} rows to: {out}")


if __name__ == "__main__":
    main(out=extraction_utils.cli_out_arg())
