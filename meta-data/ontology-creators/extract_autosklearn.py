"""
Extract auto-sklearn hyperparameter ontology.

The extraction works in two steps:
    1. Framework introspection: ConfigSpace search space hyperparameters with ranges and choices.
    2. Sklearn introspection: using inspect.signature() on the models underlying class.

Outputs: /meta-data/ontologies/autosklearn_ontology_mapping.csv

Run from root:
    python meta-data/ontology-creators/extract_autosklearn.py
"""

import warnings
from pathlib import Path

import extraction_utils
from extraction_utils import attempt_import
from autosklearn.pipeline.components.classification import ClassifierChoice
from autosklearn.pipeline.components.regression import RegressorChoice
import ConfigSpace.hyperparameters as CSH
from collections import defaultdict

warnings.filterwarnings("ignore")

OUT = extraction_utils.out_path("autosklearn_ontology_mapping.csv")
FIELDS = extraction_utils.FIELDS


# models that are wrapped around sklearn estimators
UNDERLYING = {
    "adaboost": attempt_import("sklearn.ensemble", "AdaBoostClassifier"),
    "bernoulli_nb": attempt_import("sklearn.naive_bayes", "BernoulliNB"),
    "decision_tree": attempt_import("sklearn.tree", "DecisionTreeClassifier"),
    "extra_trees": attempt_import("sklearn.ensemble", "ExtraTreesClassifier"),
    "gaussian_nb": attempt_import("sklearn.naive_bayes", "GaussianNB"),
    "gradient_boosting": attempt_import("sklearn.ensemble", "GradientBoostingClassifier"),
    "hist_gradient_boosting": attempt_import("sklearn.ensemble", "HistGradientBoostingClassifier"),
    "k_nearest_neighbors": attempt_import("sklearn.neighbors", "KNeighborsClassifier"),
    "lda": attempt_import("sklearn.discriminant_analysis", "LinearDiscriminantAnalysis"),
    "liblinear_svc": attempt_import("sklearn.svm", "LinearSVC"),
    "libsvm_svc": attempt_import("sklearn.svm", "SVC"),
    "mlp": attempt_import("sklearn.neural_network", "MLPClassifier"),
    "multinomial_nb": attempt_import("sklearn.naive_bayes", "MultinomialNB"),
    "passive_aggressive": attempt_import("sklearn.linear_model", "PassiveAggressiveClassifier"),
    "qda": attempt_import("sklearn.discriminant_analysis", "QuadraticDiscriminantAnalysis"),
    "random_forest": attempt_import("sklearn.ensemble", "RandomForestClassifier"),
    "sgd": attempt_import("sklearn.linear_model", "SGDClassifier"),
    "ard_regression": attempt_import("sklearn.linear_model", "ARDRegression"),
    "gaussian_process": attempt_import("sklearn.gaussian_process", "GaussianProcessRegressor"),
    "liblinear_svr": attempt_import("sklearn.svm", "LinearSVR"),
    "libsvm_svr": attempt_import("sklearn.svm", "SVR"),
}


parse_numpy_params = extraction_utils.parse_numpy_params


def hp_type_to_str(hp):

    if isinstance(hp, CSH.CategoricalHyperparameter):
        return f"Categorical({list(hp.choices)})"
    if isinstance(
        hp, (CSH.UniformIntegerHyperparameter, CSH.NormalIntegerHyperparameter)
    ):
        log = getattr(hp, "log", False)
        return f"Integer({hp.lower}, {hp.upper}{'_log' if log else ''})"
    if isinstance(hp, (CSH.UniformFloatHyperparameter, CSH.NormalFloatHyperparameter)):
        log = getattr(hp, "log", False)
        return f"Float({hp.lower}, {hp.upper}{'_log' if log else ''})"
    if isinstance(hp, CSH.OrdinalHyperparameter):
        return f"Ordinal({list(hp.sequence)})"
    return type(hp).__name__


def extract_components(registry, family):
    #Step 1: ConfigSpace search space extraction.
    rows = []

    for comp_name, comp_cls in registry.items():
        try:
            space = comp_cls.get_hyperparameter_search_space()
        except Exception as e:
            print(f"Exception for {comp_name}: {e}")
            continue
        hps = space.get_hyperparameters()
        lib_descs = parse_numpy_params(UNDERLYING.get(comp_name))

        if not hps:
            rows.append(
                {
                    "framework": "autosklearn",
                    "family": family,
                    "class": comp_name,
                    "parameter_name": "(none)",
                    "value_default": "",
                    "value_type": "fixed",
                    "description": "No tunable HPs",
                }
            )
        for hp in hps:
            rows.append(
                {
                    "framework": "autosklearn",
                    "family": family,
                    "class": comp_name,
                    "parameter_name": hp.name,
                    "value_default": str(hp.default_value),
                    "value_type": hp_type_to_str(hp),
                    "description": lib_descs.get(hp.name, ""),
                }
            )
    return rows


def add_full_api_step(step1_rows):
    # Step 2: extend models with sklearn classes parameters

    captured = defaultdict(set)
    for row in step1_rows:
        captured[(row["family"], row["class"])].add(row["parameter_name"])

    extra_rows = []
    
    for (family, comp_name), already in captured.items():
        underlying = UNDERLYING.get(comp_name)
        if underlying is None:
            continue

        descriptions = parse_numpy_params(underlying)

        extra_rows.extend(extraction_utils.full_api_rows(
            "autosklearn", family, comp_name, underlying, already, descriptions
    ) )
    return extra_rows


def main(out=None):
    # Run meta data extraction for every model

    # Step 1
    # Feature preprocessing components are excluded in the search
    rows = extract_components(ClassifierChoice.get_components(), "classification")
    rows += extract_components(RegressorChoice.get_components(), "regression")

    # Step 2
    rows += add_full_api_step(rows)

    rows = extraction_utils.dedup_rows(rows)

    covered = sum(1 for r in rows if r.get("description"))
    print(f"Rows with description: {covered}/{len(rows)}"
    )

    out = Path(out) if out else OUT
    extraction_utils.write_csv(out, rows)
    
    print(f"Wrote all {len(rows)} rows to: {out}")

if __name__ == "__main__":
    main(out=extraction_utils.cli_out_arg())

