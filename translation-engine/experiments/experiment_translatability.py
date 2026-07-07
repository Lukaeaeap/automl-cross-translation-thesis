"""
Experiment 1 (E1): Translatability
It measures real translation possibility (no fitting yet), from configuration datasets to target frameworks

Attempts to translate every classifier configiguration from source to every other framework.
Finds if engine.translate() actually produced a usable target model.

Usage from root:
    python translation-engine/experiments/experiment_translatability.py
"""

from __future__ import annotations
import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ENGINE_DIR = HERE.parent
NEW_DATA = ENGINE_DIR.parent / "automl-data" / "New-Data"
MAY_XLSX = NEW_DATA / "automl_results_may.xlsx"
JUNE_XLSX = NEW_DATA / "results_binary_June_15_new.xlsx"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(HERE))

from engine import TranslationEngine, FRAMEWORKS
from config_loader import load_configs_from_excel, load_configs_from_june_excel
from constants import CLASSIFIER_CANONICALS

def run() -> list[dict]:
    engine = TranslationEngine()

    configs = []
    if MAY_XLSX.exists():
        print(f"Loading May: {MAY_XLSX.name}")
        configs.extend(load_configs_from_excel(str(MAY_XLSX)))
    else:
        print("May excel file not found.")
    if JUNE_XLSX.exists():
        print(f"Loading June: {JUNE_XLSX.name}")
        configs.extend(load_configs_from_june_excel(str(JUNE_XLSX)))
    else:
        print("June excel file not found.")

    print(f"Loaded {len(configs)} configs total")

    results = []

    for i, cfg in enumerate(configs):
        src_fw = cfg["framework"]
        model_cls = cfg["model"]
        params = cfg["params"]

        canonical = engine.resolve_model(src_fw, model_cls)
        if canonical is None or canonical not in CLASSIFIER_CANONICALS:
            continue

        for tgt_fw in FRAMEWORKS:
            if tgt_fw == src_fw:
                continue

            translation = engine.translate(
                {"model": model_cls, "params": params}, source=src_fw, target=tgt_fw
            )
            tgt_model = translation["model"]
            translatable = not tgt_model.startswith("[")

            results.append(
                {
                    "source_framework": src_fw,
                    "source_model": model_cls,
                    "target_framework": tgt_fw,
                    "target_model": tgt_model,
                    "canonical_model": canonical,
                    "translatable": translatable,
                }
            )

        if (i + 1) % 1000 == 0:
            print(f"processed {i + 1}/{len(configs)} configs, {len(results)} translations so far")

    return results


def print_summary(results: list[dict]):
    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results)
    print(f"\nAttempted: {len(df)}, translatable: {df['translatable'].sum()}")

    grp = df.groupby(["source_framework", "target_framework"])
    pair = pd.DataFrame({"n": grp.size(), "translatable": grp["translatable"].sum()})
    pair["rate %"] = (100 * pair["translatable"] / pair["n"]).round(1)
    print(f"\n{pair}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=str(RESULTS_DIR / f"{timestamp}_translatability_full.csv")
    )
    args = parser.parse_args()

    results = run()
    print_summary(results)
    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"\n{len(results)} results -> {args.out}")


if __name__ == "__main__":
    main()
