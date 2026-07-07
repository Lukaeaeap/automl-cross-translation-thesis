"""
Helper functions for experiment_fit_success.py.

Centralises dataset loading and framework instantiation helpers used in the
E3 fit success testing experiment.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
for lg in ["flaml", "autogluon", "h2o"]:
    logging.getLogger(lg).setLevel(logging.ERROR)

HERE = Path(__file__).resolve().parent
ENGINE_DIR = HERE.parent
DATASETS = ENGINE_DIR.parent / "automl-data" / "datasets"

sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(HERE))

# Generic helper functions

def clean_params(params: dict) -> dict:
    # Drop None values
    return {k: v for k, v in params.items() if v is not None}


def filter_sig(cls, params: dict) -> dict:
    # Drop parameers that are not accepted by cls.__init__
    # Pass everything if **kwargs present.
    try:
        sig = inspect.signature(cls.__init__)
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return params
        accepted = set(sig.parameters) - {"self"}
        return {k: v for k, v in params.items() if k in accepted}
    except (ValueError, TypeError):
        return params


def sklearn_class(name: str):
    # Return the sklearn/xgboost/lightgbm class for a given class name.
    if not name or name.startswith("["):
        return None
    for mod in [
        "sklearn.ensemble",
        "sklearn.linear_model",
        "sklearn.tree",
        "sklearn.svm",
        "sklearn.naive_bayes",
        "sklearn.neighbors",
        "sklearn.discriminant_analysis",
        "sklearn.neural_network",
        "sklearn.gaussian_process",
        "xgboost",
        "lightgbm"]:
        try:
            cls = getattr(importlib.import_module(mod), name, None)
            if cls:
                return cls
        except ImportError:
            pass
    return None


# FLAML helper function

def flaml_filter(cls, params: dict, task: str = "binary") -> dict:
    # Filter params to those in cls.search_space. Fall back to sig filter.
    try:
        space = cls.search_space(data_size=(1000, 10), task=task)
        valid = set(space.keys())
        return {k: v for k, v in params.items() if k in valid}
    except Exception:
        return filter_sig(cls, params)


# H2O helper function 

H2O_ALIASES: dict | None = None


def resolve_h2o_class(model_name: str) -> str:
    # Resolve abbreviated H2O names (such as DRF, GBM) to full Python class names
    global H2O_ALIASES
    if H2O_ALIASES is None:
        try:
            aliases_path = ENGINE_DIR / "ontology" / "aliases.json"
            H2O_ALIASES = json.loads(aliases_path.read_text()).get("h2o", {})
        except Exception:
            H2O_ALIASES = {}
    return H2O_ALIASES.get(model_name, model_name)


# Dataset loading
def resolve_dataset_id(name_or_id: str) -> str:
    # Accept display name ('Australian') or numeric ID ('dataset_146818') to identify given dataset.
    if (DATASETS / f"{name_or_id}.csv").exists():
        return name_or_id
    for meta in DATASETS.glob("*_metadata.json"):
        try:
            m = json.loads(meta.read_text())
            if m.get("dataset_name", "").lower() == name_or_id.lower():
                return meta.stem.replace("_metadata", "")
        except Exception:
            pass
    return name_or_id


def load_dataset(dataset_id: str, n_sample: int = 500, seed: int = 1):
    # Load and preprocess given dataset CSV. Returns either (X, y, df_original, dataset_name) or (None, None, None, None).
    dataset_id = resolve_dataset_id(dataset_id)
    csv = DATASETS / f"{dataset_id}.csv"
    meta_f = DATASETS / f"{dataset_id}_metadata.json"
    if not csv.exists():
        return None, None, None, None
    
    m = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    target_col = m.get("target_name", "class")
    cat_feats = set(m.get("categorical_features", []))
    name = m.get("dataset_name", dataset_id)

    df = pd.read_csv(csv)
    if target_col not in df.columns:
        candidates = [c for c in df.columns if c.lower() == target_col.lower()]
        if not candidates:
            return None, None, None, None
        target_col = candidates[0]

    if len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=seed)

    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col])
    y = (y_raw == sorted(y_raw.unique())[-1]).astype(int).values

    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import SimpleImputer

    cat_cols = [c for c in X_df.columns if c in cat_feats or X_df[c].dtype == object]
    num_cols = [c for c in X_df.columns if c not in cat_cols]

    parts = []
    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        parts.append(num_imputer.fit_transform(X_df[num_cols]))
    if cat_cols:
        cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        cat_data = X_df[cat_cols].fillna("__missing__")
        parts.append(cat_encoder.fit_transform(cat_data))
    X = np.hstack(parts) if parts else np.zeros((len(df), 1))
    return X, y, df.assign(**{target_col: y_raw}), name
