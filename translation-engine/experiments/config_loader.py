"""
Load and normalize AutoML configs from the configuration Excel files. 
Here the ensembles of algorithms get filtered out of the config.

Usage:
from config_loader import load_configs_from_excel, load_configs_from_june_excel

Each returned config dict:
    {"framework": str, "model": str, "params": dict,
     "dataset_id": str, "cv_accuracy": float, "candidate_id": str}
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constants import (
    ENSEMBLE_KEYWORDS,
    H2O_INFRA_SUFFIXES,
    H2O_INFRA_EXACT,
    H2O_ALGO_TO_CLASS,
    AG_ALGO_TO_CLASS,
    FLAML_ALGO_TO_CLASS,
)


def is_ensemble(algorithm: str) -> bool:
    lower = str(algorithm).lower()
    return any(kw in lower for kw in ENSEMBLE_KEYWORDS)


def coerce(v):
    if v is None or isinstance(v, (int, float, bool)):
        return v
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s.lower() == "none":
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return v


def parse_raw(row, params_col):
    raw = row.get(params_col)
    if pd.isna(raw) or raw is None:
        return None
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None


def parse_autosklearn(algorithm: str, raw: dict) -> dict:
    prefix = f"classifier:{algorithm}:"
    result = {}
    for k, v in raw.items():
        if not k.startswith(prefix):
            continue
        key = k[len(prefix) :]
        result[key] = coerce(v)
    return result


def parse_h2o(algorithm: str, raw: dict) -> tuple:
    class_name = H2O_ALGO_TO_CLASS.get(algorithm.lower(), algorithm)
    params = {
        k: coerce(v)
        for k, v in raw.items()
        if v is not None
        and not k.endswith(H2O_INFRA_SUFFIXES)
        and k not in H2O_INFRA_EXACT
        and not isinstance(v, (dict, list))
    }
    return class_name, params


def parse_tpot(algorithm: str, raw: dict) -> dict:
    pipeline_str = raw.get("pipeline", "")
    if not pipeline_str or not algorithm:
        return {}
    return {
        m.group(1): coerce(m.group(2).strip())
        for m in re.finditer(rf"{re.escape(algorithm)}__(\w+)=([^,)]+)", pipeline_str)
    }


def parse_autogluon(algorithm: str, raw: dict) -> tuple:
    class_name = AG_ALGO_TO_CLASS.get(algorithm.lower(), algorithm)
    params = {k: coerce(v) for k, v in raw.items() if v is not None}
    return class_name, params


def parse_flaml(algorithm: str, raw: dict) -> tuple:
    class_name = FLAML_ALGO_TO_CLASS.get(algorithm.lower(), algorithm)
    params = {k: coerce(v) for k, v in raw.items() if v is not None}
    return class_name, params


def build_config(
    framework, model_name, params, row, acc_col=None, candidate_col="candidate_id"
):
    dataset_id = row.get("dataset_id")
    candidate = row.get(candidate_col)
    cv_val = row.get(acc_col) if acc_col else None
    try:
        cv_val = float(cv_val) if cv_val is not None and not pd.isna(cv_val) else None
    except (TypeError, ValueError):
        cv_val = None
    return {
        "framework": framework,
        "model": model_name,
        "params": params,
        "dataset_id": str(dataset_id)
        if dataset_id is not None and not pd.isna(dataset_id)
        else None,
        "cv_accuracy": cv_val,
        "candidate_id": str(candidate)
        if candidate is not None and not pd.isna(candidate)
        else None,
    }


def load_configs_from_excel(
    path: str, frameworks: list = None, best_only: bool = False
) -> list:
    df = pd.read_excel(path)
    supported = {"autosklearn", "tpot", "h2o"}
    df = df[df["framework"].isin(set(frameworks or supported) & supported)]
    if best_only:
        df = df[df["is_best_model"].astype(str).str.lower().isin(["true", "1", "1.0"])]

    configs, skipped = [], 0
    for _, row in df.iterrows():
        framework = str(row.get("framework", "") or "")
        algorithm = str(row.get("algorithm", "") or "").strip()
        if is_ensemble(algorithm):
            continue
        raw = parse_raw(row, "params_json")
        if not isinstance(raw, dict):
            skipped += 1
            continue

        if framework == "autosklearn":
            model_name, params = algorithm, parse_autosklearn(algorithm, raw)
        elif framework == "tpot":
            model_name, params = algorithm, parse_tpot(algorithm, raw)
        elif framework == "h2o":
            model_name, params = parse_h2o(algorithm, raw)
        else:
            continue
        configs.append(build_config(framework, model_name, params, row, "cv_accuracy"))

    print(
        f"Loaded {len(configs)} configs from {Path(path).name} ({skipped} parse errors)"
    )
    return configs


def load_configs_from_june_excel(
    path: str,
    frameworks: list = None,
    best_only: bool = False,
    sheet_name: str = "TopK_Summary",
) -> list:
    df = pd.read_excel(path, sheet_name=sheet_name)
    supported = {"h2o", "autogluon", "flaml"}
    df = df[df["framework"].isin(set(frameworks or supported) & supported)]

    algo_col = "algo" if "algo" in df.columns else "algorithm"

    if "is_ensemble" in df.columns:
        df = df[df["is_ensemble"] == 0]
    else:
        df = df[
            ~df[algo_col]
            .str.lower()
            .str.contains("|".join(ENSEMBLE_KEYWORDS), na=False)
        ]

    if best_only:
        metric_col = "cv_metric_value" if "cv_metric_value" in df.columns else "cv_auc"
        df = (
            df.sort_values(metric_col, ascending=False)
            .groupby(["dataset_id", "framework"], as_index=False)
            .first()
        )

    params_col = "params_json" if "params_json" in df.columns else "refit_params_json"
    acc_col = next(
        (c for c in ("cv_auc", "cv_accuracy", "cv_metric_value") if c in df.columns),
        None,
    )

    configs, skipped = [], 0
    for _, row in df.iterrows():
        framework = str(row.get("framework", "") or "")
        algorithm = str(row.get(algo_col, "") or "").strip()
        if is_ensemble(algorithm):
            continue
        raw = parse_raw(row, params_col)
        if not isinstance(raw, dict):
            skipped += 1
            continue

        if framework == "h2o":
            model_name, params = parse_h2o(algorithm, raw)
        elif framework == "autogluon":
            model_name, params = parse_autogluon(algorithm, raw)
        elif framework == "flaml":
            model_name, params = parse_flaml(algorithm, raw)
        else:
            continue
        configs.append(build_config(framework, model_name, params, row, acc_col))

    print(
        f"Loaded {len(configs)} configs from {Path(path).name} [{sheet_name}] ({skipped} parse errors)"
    )
    return configs
