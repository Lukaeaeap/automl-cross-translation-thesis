"""
Lookup constants for both the translation engine and the experiments.
"""

import json
from pathlib import Path

# Load all classifer models from ontology
ROLES = json.loads(
    (Path(__file__).parent / "ontology" / "algorithm_roles.json").read_text()
)
CLASSIFIER_ROLES = {"classifier", "both"}

CLASSIFIER_CANONICALS = set(
    k for k, v in ROLES.items() if isinstance(v, str) and v in CLASSIFIER_ROLES
)

# Keywords to recognize meta estimator ensembles in order to drop them
ENSEMBLE_KEYWORDS = ("ensemble", "stacked", "weighted")

# H2O infrastructure parameters that should be dropped in translations and ontology building
H2O_INFRA_SUFFIXES = ("_frame", "_column", "_id")

H2O_INFRA_EXACT = set(
    {
        "checkpoint",
        "keep_cross_validation_models",
        "keep_cross_validation_predictions",
        "keep_cross_validation_fold_assignment",
        "export_checkpoints_dir",
        "score_each_iteration",
        "build_tree_one_node",
        "custom_metric_func",
        "check_constant_response",
        "ignore_const_cols",
        "fold_assignment",
        "nfolds",
        "parallelize_cross_validation",
        "gainslift_bins",
        "auc_type",
        "score_eval_metric_only",
        "quiet_mode",
        "max_runtime_secs",
        "calibrate_model",
        "calibration_method",
        "ignored_columns",
        "score_tree_interval",
        "in_training_checkpoints_tree_interval",
        "auto_rebalance",
        "max_confusion_matrix_size",
        "nthread",
        "backend",
        "dmatrix_type",
        "save_matrix_directory",
        "sample_type",
        "normalize_type",
        "eta",
        "quantile_alpha",
        "huber_alpha",
        "tweedie_power",
        "r2_stopping",
        "pred_noise_bandwidth",
    }
)

# AutoGluon preprocessing/global settings that are not per-model hyperparameters
AUTOGLUON_INFRA_PREFIXES = ("proc.",)



# The Excel automl results files with configurations use short algorithm names. Below are the mappings of the short names to the actually used names by the frameworks internally.
# Per framework: Algorithm name -> engine class name

H2O_ALGO_TO_CLASS = { 
    "gbm": "GBM",
    "xgboost": "XGBoost", 
    "drf": "DRF",
    "deeplearning": "H2ODeepLearningEstimator",
    "glm": "H2OGeneralizedLinearEstimator",
}

AG_ALGO_TO_CLASS = {
    "lgbm": "LGBModel",
    "extra_trees": "XTModel",
    "random_forest": "RFModel",
    "xgboost": "XGBoostModel",
    "catboost": "CatBoostModel",
    "nn_torch": "TabularNeuralNetTorchModel",
    "fastai": "NNFastAiTabularModel",
    "knn": "KNNModel", 
    "linear": "LinearModel",
}

FLAML_ALGO_TO_CLASS = {
    "lgbm": "LGBMEstimator",
    "xgboost": "XGBoostEstimator",
    "xgb_limitdepth": "XGBoostSklearnEstimator",
    "random_forest": "RandomForestEstimator",
    "extra_trees": "ExtraTreesEstimator", 
    "linear": "LRL1Classifier",
    "sgd": "SGDEstimator",
}

AUTOSKLEARN_ALGO_TO_CLASS = {
    "adaboost": "AdaBoostClassifier",
    "ard_regression": "ARDRegression",
    "bernoulli_nb": "BernoulliNB",
    "decision_tree": "DecisionTreeClassifier",
    "extra_trees": "ExtraTreesClassifier",
    "gaussian_nb": "GaussianNB", 
    "gaussian_process": "GaussianProcessRegressor",
    "gradient_boosting": "HistGradientBoostingClassifier",
    "k_nearest_neighbors": "KNeighborsClassifier",
    "lda": "LinearDiscriminantAnalysis", 
    "liblinear_svc": "LinearSVC",
    "liblinear_svr": "LinearSVR",
    "libsvm_svc": "SVC",
    "libsvm_svr": "SVR",
    "mlp": "MLPClassifier", 
    "multinomial_nb": "MultinomialNB",
    "passive_aggressive": "PassiveAggressiveClassifier",
    "qda": "QuadraticDiscriminantAnalysis",
    "random_forest": "RandomForestClassifier", 
    "sgd": "SGDClassifier",
}

# Exclusions for TPOT and sklearn model specific hyperparameters that were not accepted in previous experiment runs 

TPOT_CLASS_EXCLUDED = {
    "RandomForestClassifier": {"nthread"},
    "RandomForestRegressor": {"nthread"},
    "ExtraTreesClassifier": {"nthread"},
    "ExtraTreesRegressor": {"nthread"},
    "LinearSVC": {"kernel", "gamma", "degree", "coef0", "shrinking"},
    "LinearSVR": {"kernel", "gamma", "degree", "coef0", "shrinking"},
}

# Sklearn HistGradientBoosting
HIST_GBM_UNSUPPORTED = set(
    {
        "subsample",
        "max_features",
        "min_samples_split",
        "min_weight_fraction_leaf",
        "min_impurity_decrease",
        "criterion",
        "presort",
        "init",
    }
)

# AutoGluon internal models classes with full import path
AG_CLASS_MAP = {
    "LGBModel": "autogluon.tabular.models.LGBModel",
    "RFModel": "autogluon.tabular.models.RFModel",
    "XGBoostModel": "autogluon.tabular.models.XGBoostModel",
    "XTModel": "autogluon.tabular.models.XTModel",
    "KNNModel": "autogluon.tabular.models.KNNModel",
    "CatBoostModel": "autogluon.tabular.models.CatBoostModel",
    "LinearModel": "autogluon.tabular.models.LinearModel",
    "TabularNeuralNetTorchModel": "autogluon.tabular.models.TabularNeuralNetTorchModel",
    "NNFastAiTabularModel": "autogluon.tabular.models.NNFastAiTabularModel",
}


# Checkpoint saved after
# 30 min is 1800s
CHECKPOINT_INTERVAL_SECS = 1800

# Experiment outcome
# Ensure naming consistency between experiment, data and analysis.
FIT_SUCCESS = "FIT_SUCCESS"
FIT_FAIL = "FIT_FAIL"
FIT_FAIL_RANGE = "FIT_FAIL_RANGE"
FIT_SKIP = "SKIP"
