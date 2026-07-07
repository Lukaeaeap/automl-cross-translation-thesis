"""
Build canonical model + parameter ontology from framework CSVs.

Usage from root:
    python translation-engine/ontology/build.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ontology.align import cluster_models, cluster_params, normalize_model_name
from constants import H2O_INFRA_SUFFIXES, H2O_INFRA_EXACT

HERE = Path(__file__).resolve().parent
DEFAULT_CSV_DIR = HERE.parent.parent / "meta-data" / "ontologies"

CSV_NAMES = {
    "h2o": "h2o_ontology_mapping.csv",
    "flaml": "flaml_ontology_mapping.csv",
    "autogluon": "autogluon_ontology_mapping.csv",
    "tpot": "tpot_ontology_mapping.csv",
    "autosklearn": "autosklearn_ontology_mapping.csv",
}

FRAMEWORKS = list(CSV_NAMES.keys())

# Extra pip packages found from class name substrings and their package
CLASS_DEPENDENCY_RULES = [
    ("CatBoost", "catboost"),
    ("XGBoost", "xgboost"),
    ("LGBM", "lightgbm"),
]


def load_all(csv_dir: Path) -> pd.DataFrame:
    # Load all the csv's with ontologies and remove infrastructure keys from H2O entries
    frames = []
    for fw, fname in CSV_NAMES.items():
        path = csv_dir / fname
        if not path.exists():
            print(f"Warning: {fname} not found, skipping {fw}")
            continue
        df = pd.read_csv(path)
        df["framework"] = fw
        if fw == "h2o":
            before = len(df)
            mask = df["parameter_name"].str.endswith(H2O_INFRA_SUFFIXES) | df[
                "parameter_name"
            ].isin(H2O_INFRA_EXACT)
            df = df[~mask]
            print(
                f"loaded {fw}: {len(df)} rows ({before - len(df)} infra params removed)"
            )
        else:
            print(f"loaded {fw}: {len(df)} rows")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def apply_aliases(model_ontology: dict) -> dict:
    aliases_path = HERE / "aliases.json"
    if not aliases_path.exists():
        return model_ontology

    aliases = json.loads(aliases_path.read_text())
    added = 0

    reverse: dict[tuple, str] = {}
    for canonical, fw_map in model_ontology.items():
        for fw, names in fw_map.items():
            for name in names:
                reverse[(fw, name)] = canonical

    for fw, fw_aliases in aliases.items():
        if fw == "_comment":
            continue
        for alias, canonical_class in fw_aliases.items():
            canonical_model = reverse.get((fw, canonical_class))
            if canonical_model is None:
                guessed = normalize_model_name(canonical_class)
                if guessed in model_ontology:
                    canonical_model = guessed
                else:
                    model_ontology[alias] = {fw: [alias]}
                    reverse[(fw, alias)] = alias
                    added += 1
                    continue

            fw_list = model_ontology[canonical_model].setdefault(fw, [])
            if alias not in fw_list:
                fw_list.append(alias)
                reverse[(fw, alias)] = canonical_model
                added += 1

    for canonical in model_ontology:
        for fw in model_ontology[canonical]:
            model_ontology[canonical][fw] = sorted(set(model_ontology[canonical][fw]))

    print(f"-> {added} aliases injected from aliases.json")
    return model_ontology


def build_dependencies(model_ontology: dict) -> dict:
    deps: dict = {}
    for canonical, fw_map in model_ontology.items():
        fw_deps: dict = {}
        for fw, classes in fw_map.items():
            primary = classes[0] if classes else ""
            needed = {
                pkg
                for sub, pkg in CLASS_DEPENDENCY_RULES
                if sub.lower() in primary.lower()
            }
            if needed:
                fw_deps[fw] = sorted(needed)
        if fw_deps:
            deps[canonical] = fw_deps
    return deps


def build_matrix(model_ontology: dict) -> dict:
    return {
        canonical: {fw: bool(model_ontology[canonical].get(fw)) for fw in FRAMEWORKS}
        for canonical in model_ontology
    }


def build_models(df: pd.DataFrame) -> dict:
    print("\nStep 1: clustering models...")
    clusters = cluster_models(df)
    print(f"-> {len(clusters)} canonical models")
    result = {
        k: {fw: sorted(v) for fw, v in sorted(fws.items())}
        for k, fws in sorted(clusters.items())
    }
    return apply_aliases(result)


def fill_singletons(df: pd.DataFrame, fw_models: dict, params: dict) -> int:
    represented: set = {
        (fw, pname)
        for info in params.values()
        for fw, pname in info["mappings"].items()
    }
    added = 0
    for fw, model_names in fw_models.items():
        sub = df[(df["framework"] == fw) & (df["class"].isin(model_names))]
        for _, row in sub.drop_duplicates("parameter_name").iterrows():
            pname = row["parameter_name"]
            if (fw, pname) in represented:
                continue
            if pname in params:
                params[pname]["mappings"][fw] = pname
                if pd.notna(row.get("value_type")):
                    params[pname].setdefault("value_types", {})[fw] = str(
                        row["value_type"]
                    )
                if pd.notna(row.get("value_default")):
                    params[pname].setdefault("value_defaults", {})[fw] = str(
                        row["value_default"]
                    )
            else:
                entry = {
                    "description": str(row.get("description", "") or "").strip(),
                    "mappings": {fw: pname},
                    "value_types": {},
                    "value_defaults": {},
                }
                if pd.notna(row.get("value_type")):
                    entry["value_types"][fw] = str(row["value_type"])
                if pd.notna(row.get("value_default")):
                    entry["value_defaults"][fw] = str(row["value_default"])
                params[pname] = entry
            represented.add((fw, pname))
            added += 1
    return added


def build_params(df: pd.DataFrame, model_ontology: dict) -> dict:
    print("\nStep 2: clustering parameters per model...")
    result, total_singletons = {}, 0
    for canonical, fw_models in model_ontology.items():
        params = cluster_params(df, fw_models) or {}
        added = fill_singletons(df, fw_models, params)
        total_singletons += added
        if params:
            result[canonical] = params
            n = len(params)
            covered = sum(1 for p in params.values() if len(p["mappings"]) > 1)
            print(
                f"{canonical}: {n} params, {covered} cross-framework, {added} singletons"
            )
    print(f"-> {total_singletons} singleton params total")
    return result


def main():
    csv_dir = DEFAULT_CSV_DIR
    out_dir = HERE
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CSV dir: {csv_dir}")
    df = load_all(csv_dir)
    print(f"Total: {len(df)} rows across {df['framework'].nunique()} frameworks")

    model_ont = build_models(df)
    (out_dir / "models.json").write_text(json.dumps(model_ont, indent=2))
    print(f"\nSaved {len(model_ont)} canonical models -> models.json")

    (out_dir / "models_matrix.json").write_text(
        json.dumps(build_matrix(model_ont), indent=2)
    )
    print("Saved model availability matrix -> models_matrix.json")

    deps = build_dependencies(model_ont)
    (out_dir / "model_dependencies.json").write_text(json.dumps(deps, indent=2))
    print("Saved model dependencies -> model_dependencies.json")

    param_ont = build_params(df, model_ont)
    (out_dir / "params.json").write_text(json.dumps(param_ont, indent=2))
    total = sum(len(v) for v in param_ont.values())
    print(
        f"Saved {total} canonical params across {len(param_ont)} models -> params.json"
    )


if __name__ == "__main__":
    main()
