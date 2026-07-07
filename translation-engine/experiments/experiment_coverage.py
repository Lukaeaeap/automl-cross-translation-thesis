"""
Experiment 2 (E2): Ontology Coverage
1a. Hyperparameter coverage gives two views:
    - RAW: every non-null key seen in config dicts, no infra filtering.
    - SEARCH SPACE: keys remaining after framework-specific infra filtering (H2O blocklist, autosklearn prefix-strip, tpot pipeline-regex).
    Only the search spaces hyperparameters are included in the Fit Success experiments

1b. Model coverage:
    How many unique algorithm classes in the excel configuration datasets also exist in the translation systems ontology 
    in models.json + aliases.json. 

Usage from root:
    python translation-engine/experiments/experiment_coverage.py
    python translation-engine/experiments/experiment_coverage.py --out results/ontology_coverage.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ENGINE_DIR = HERE.parent
NEW_DATA = ENGINE_DIR.parent / "automl-data" / "New-Data"
MAY_XLSX = NEW_DATA / "automl_results_may.xlsx"
JUNE_XLSX = NEW_DATA / "results_binary_June_15_new.xlsx"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(HERE))

from engine import TranslationEngine
from config_loader import is_ensemble, parse_raw
from translators import REGISTRY
from constants import (
    H2O_INFRA_SUFFIXES,
    H2O_INFRA_EXACT,
    AUTOGLUON_INFRA_PREFIXES,
    H2O_ALGO_TO_CLASS,
    AG_ALGO_TO_CLASS,
    FLAML_ALGO_TO_CLASS,
    AUTOSKLEARN_ALGO_TO_CLASS,
    CLASSIFIER_CANONICALS,
)

ALGO_TO_CLASS = {}
for fw, mapping in [
    ("h2o", H2O_ALGO_TO_CLASS),
    ("autogluon", AG_ALGO_TO_CLASS),
    ("flaml", FLAML_ALGO_TO_CLASS),
    ("autosklearn", AUTOSKLEARN_ALGO_TO_CLASS),
]:
    ALGO_TO_CLASS[fw] = {k: v.lower() for k, v in mapping.items()}

def extract_param_keys_raw(framework: str, algorithm: str, raw: dict) -> set:
    # For autosklearn we still strip the classifier:{algo}: prefix 
    # so param names are comparable to the ontology.
    
    if framework == "tpot":
        return extract_param_keys_search(framework, algorithm, raw)
    if framework == "autosklearn":
        prefix = f"classifier:{algorithm}:"
        keys = set()
        for k, v in raw.items():
            if v is None or isinstance(v, (dict, list)):
                continue
            keys.add(k[len(prefix) :] if k.startswith(prefix) else k)
        return keys
    return {
        k for k, v in raw.items() if v is not None and not isinstance(v, (dict, list))
    }


def extract_param_keys_search(framework: str, algorithm: str, raw: dict) -> set:
    # find only the search space hyperparameters
    if framework == "h2o":
        return {
            k
            for k, v in raw.items()
            if v is not None
            and not k.endswith(H2O_INFRA_SUFFIXES)
            and k not in H2O_INFRA_EXACT
            and not isinstance(v, (dict, list))
        }
    if framework == "autosklearn":
        prefix = f"classifier:{algorithm}:"
        return {k[len(prefix) :] for k in raw if k.startswith(prefix)}
    if framework == "tpot":
        pipeline_str = raw.get("pipeline", "")
        if not pipeline_str or not algorithm:
            return set()
        return set(re.findall(rf"{re.escape(algorithm)}__(\w+)=", pipeline_str))
    if framework == "autogluon":
        return {
            k
            for k, v in raw.items()
            if v is not None
            and not isinstance(v, (dict, list))
            and not k.startswith(AUTOGLUON_INFRA_PREFIXES)
        }
    if framework == "flaml":
        return {
            k
            for k, v in raw.items()
            if v is not None and not isinstance(v, (dict, list))
        }
    return set()


def iter_rows(df: pd.DataFrame, params_col: str, engine=None):
    algo_col = "algo" if "algo" in df.columns else "algorithm"
    for _, row in df.iterrows():
        fw = str(row.get("framework", "") or "").strip()
        alg = str(row.get(algo_col, "") or "").strip()
        if not fw or is_ensemble(alg):
            continue
        if engine is not None:
            resolved = ALGO_TO_CLASS.get(fw, {}).get(alg.lower(), alg)
            canonical = engine.resolve_model(fw, resolved) or engine.resolve_model(
                fw, alg
            )
            if canonical not in CLASSIFIER_CANONICALS:
                continue
        raw = parse_raw(row, params_col)
        if isinstance(raw, dict):
            yield fw, alg, raw


def load_frames():
    frames = []
    for path in (MAY_XLSX, JUNE_XLSX):
        if not path.exists():
            continue
        print(f"Loading {path.name}")
        try:
            df = pd.read_excel(path, sheet_name="TopK_Summary")
        except Exception:
            df = pd.read_excel(path)
        col = "params_json" if "params_json" in df.columns else "refit_params_json"
        frames.append((df, col))
    return frames


def coverage_rows(
    expected_keys: dict, mapped_keys: dict, item_key: str, count_key: str
) -> list:
    def row(fw, exp, cov):
        miss = exp - cov
        return {
            "framework": fw,
            item_key: len(exp),
            count_key: len(cov),
            "coverage_pct": round(100.0 * len(cov) / len(exp), 1) if exp else 0.0,
            "missing_count": len(miss),
            "missing": sorted(miss),
        }

    covs = {fw: exp & mapped_keys.get(fw, set()) for fw, exp in expected_keys.items()}
    rows = [row(fw, expected_keys[fw], covs[fw]) for fw in sorted(expected_keys)]
    overall = row("OVERALL", set().union(*expected_keys.values()), set().union(*covs.values()))
    overall["missing"] = []
    rows.append(overall)
    return rows


def compute_param_coverage(engine: TranslationEngine, frames) -> dict:
    """
    Returns {"raw": [rows], "search_space": [rows], "search_space_cross_framework": [rows]}
    raw: portion of all existing scalar keys in configs present in the ontology
    search_space: portion of tunable hyperparameter keys present in the ontology
    search_space_cross_framework: portion of tunable hyperparameter keys mapped to a
    canonical param that exists in 2 or more frameworks, i.e. actually translatable
    """

    ontology_keys: dict = defaultdict(set)
    ontology_keys_cross_fw: dict = defaultdict(set)
    for canonical, model_params in engine.params.items():
        if canonical not in CLASSIFIER_CANONICALS:
            continue
        for param_info in model_params.values():
            mappings = param_info.get("mappings", {})
            for fw, fw_name in mappings.items():
                if not fw_name:
                    continue
                ontology_keys[fw].add(fw_name)
                if len(mappings) > 1:
                    ontology_keys_cross_fw[fw].add(fw_name)

    for fw, translator_cls in REGISTRY.items():
        ontology_keys[fw] |= set(translator_cls.RECEIVE_TRANSFORMS.keys())

    expected_raw: dict = defaultdict(set)
    expected_search: dict = defaultdict(set)
    for df, params_col in frames:
        for fw, alg, raw in iter_rows(df, params_col, engine):
            expected_raw[fw] |= extract_param_keys_raw(fw, alg, raw)
            expected_search[fw] |= extract_param_keys_search(fw, alg, raw)

    return {
        "raw": coverage_rows(
            expected_raw, ontology_keys, "expected_params", "covered_params"
        ),
        "search_space": coverage_rows(
            expected_search, ontology_keys, "expected_params", "covered_params"
        ),
        "search_space_cross_framework": coverage_rows(
            expected_search, ontology_keys_cross_fw, "expected_params", "covered_params"
        ),
    }


def compute_model_coverage(engine: TranslationEngine, frames) -> list:
    ontology_names: dict = defaultdict(set)
    for canonical, fw_map in engine.models.items():
        if canonical not in CLASSIFIER_CANONICALS:
            continue
        for fw, names in fw_map.items():
            for n in names if isinstance(names, list) else [names]:
                ontology_names[fw].add(str(n).lower())
    for fw, alias_map in engine.class_to_canonical.items():
        for alias, canonical in alias_map.items():
            if canonical not in CLASSIFIER_CANONICALS:
                continue
            ontology_names[fw].add(str(alias).lower())

    expected_models: dict = defaultdict(set)
    for df, params_col in frames:
        algo_col = "algo" if "algo" in df.columns else "algorithm"
        for _, row in df.iterrows():
            fw = str(row.get("framework", "") or "").strip()
            alg = str(row.get(algo_col, "") or "").strip()
            if not fw or not alg or is_ensemble(alg):
                continue
            resolved = ALGO_TO_CLASS.get(fw, {}).get(alg.lower(), alg.lower())
            canonical = engine.resolve_model(fw, resolved) or engine.resolve_model(
                fw, alg
            )
            if canonical not in CLASSIFIER_CANONICALS:
                continue
            expected_models[fw].add(resolved)

    return coverage_rows(
        expected_models, ontology_names, "expected_models", "covered_models"
    )


def print_table(title: str, rows: list, columns: dict):
    print(f"\n{title}")
    print(pd.DataFrame(rows)[list(columns)].rename(columns=columns).to_string(index=False))


def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=str(RESULTS_DIR / f"{timestamp}_ontology_coverage.csv")
    )
    args = parser.parse_args()

    engine = TranslationEngine()
    frames = load_frames()

    param_result = compute_param_coverage(engine, frames)
    model_rows = compute_model_coverage(engine, frames)

    suffix = " (classifiers only)"
    param_columns = {
        "framework": "Framework",
        "expected_params": "Total keys",
        "covered_params": "In ontology",
        "coverage_pct": "Coverage %",
        "missing_count": "Missing",
    }

    print_table(
        f"E2a RAW: all HP keys in config vs ontology{suffix}",
        param_result["raw"],
        param_columns,
    )

    print_table(
        f"E2a SEARCH SPACE: infra-filtered HP keys vs ontology{suffix}",
        param_result["search_space"],
        param_columns,
    )

    print_table(
        f"E2a SEARCH SPACE, CROSS-FRAMEWORK: keys mapped to a canonical param shared by 2+ frameworks{suffix}",
        param_result["search_space_cross_framework"],
        {
            "framework": "Framework",
            "expected_params": "Total keys",
            "covered_params": "Cross-framework mapping",
            "coverage_pct": "Coverage %",
            "missing_count": "Missing",
        },
    )

    print_table(
        f"E2b Model Coverage{suffix}",
        model_rows,
        {
            "framework": "Framework",
            "expected_models": "Expected",
            "covered_models": "Covered",
            "coverage_pct": "Coverage %",
            "missing_count": "Missing",
        },
    )

    for row in model_rows:
        if row.get("missing"):
            print(f"\nMissing models for {row['framework']}: {row['missing']}")

    combined = []
    for view, rows in (
        ("raw", param_result["raw"]),
        ("search_space", param_result["search_space"]),
        ("search_space_cross_framework", param_result["search_space_cross_framework"]),
        ("model", model_rows),
    ):
        for r in rows:
            combined.append(
                {
                    "view": view,
                    "framework": r["framework"],
                    "expected": r.get("expected_params", r.get("expected_models")),
                    "covered": r.get("covered_params", r.get("covered_models")),
                    "coverage_pct": r["coverage_pct"],
                    "missing_count": r["missing_count"],
                    "missing": "|".join(r["missing"]),
                }
            )

    out_path = Path(args.out)
    pd.DataFrame(combined).to_csv(out_path, index=False)
    print(f"Results saved in {out_path}")


if __name__ == "__main__":
    main()
