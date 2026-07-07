"""
Experiment 3 (E3): Runtime Success Rate
Translate every config source -> target and call .fit() on a real dataset.

Outcomes: FIT_SUCCESS, FIT_FAIL, FIT_FAIL_RANGE (bad value, not an engine bug), SKIP (missing dep)

Usage from root:
    python translation-engine/experiments/experiment_fit_success.py
    python translation-engine/experiments/experiment_fit_success.py --sample 300 --max-per-triple 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
for lg in ["flaml", "autogluon", "h2o"]:
    logging.getLogger(lg).setLevel(logging.ERROR)

HERE = Path(__file__).resolve().parent
ENGINE_DIR = HERE.parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(HERE))

from engine import TranslationEngine
from config_loader import load_configs_from_excel, load_configs_from_june_excel
from constants import (
    TPOT_CLASS_EXCLUDED,
    HIST_GBM_UNSUPPORTED,
    AG_CLASS_MAP,
    CLASSIFIER_CANONICALS,
    H2O_INFRA_SUFFIXES,
    H2O_INFRA_EXACT,
    FIT_SUCCESS,
    FIT_FAIL,
    FIT_FAIL_RANGE,
    FIT_SKIP,
    CHECKPOINT_INTERVAL_SECS,
)
from pipeline_utils import (
    load_dataset,
    resolve_dataset_id,
    clean_params,
    filter_sig,
    sklearn_class,
    resolve_h2o_class,
    flaml_filter,
)

MAY_XLSX = ENGINE_DIR.parent / "automl-data" / "New-Data" / "automl_results_may.xlsx"
JUNE_XLSX = (
    ENGINE_DIR.parent / "automl-data" / "New-Data" / "results_binary_June_15_new.xlsx"
)

 # Default experiment dataset is: Australian
DEFAULT_DATASETS = ["dataset_146818"] 
# Extra datasets that can be added are: wilt, phoneme, credit-g
EXTRA_DATASETS = ["dataset_146820", "dataset_168350", "dataset_168757"]  


RANGE_KEYWORDS = (
    "must be",
    "got ",
    "invalid value",
    "parameter of ",
    "combination of",
)


def classify_error(msg: str) -> str:
    if not msg:
        return FIT_FAIL
    return (
        FIT_FAIL_RANGE if any(kw in msg.lower() for kw in RANGE_KEYWORDS) else FIT_FAIL
    )


def fit_autosklearn(model_name, params, X, y, df_orig, target_col, meta):
   # auto-sklearn components map 1:1 to sklearn classes. The models can be fit via sklearn directly.
    cls = sklearn_class(model_name)
    if not cls:
        return FIT_SKIP, f"[autosklearn] no sklearn class for component '{model_name}'"
    cleaned = filter_sig(cls, clean_params(params))
    try:
        cls(**cleaned).fit(X, y)
        return FIT_SUCCESS, None
    except Exception as e:
        return classify_error(str(e)), str(e)[:250]


def fit_tpot(model_name, params, X, y, df_orig, target_col, meta):
    cls = sklearn_class(model_name)
    if not cls:
        return FIT_SKIP, f"[tpot] sklearn class not found for '{model_name}'"
    cleaned = {
        k: v
        for k, v in clean_params(params).items()
        if k not in TPOT_CLASS_EXCLUDED.get(model_name, set())
    }
    if "HistGradient" in model_name:
        cleaned = {k: v for k, v in cleaned.items() if k not in HIST_GBM_UNSUPPORTED}
    try:
        cls(**cleaned).fit(X, y)
        return FIT_SUCCESS, None
    except Exception as e:
        return classify_error(str(e)), str(e)[:250]


def fit_flaml(model_name, params, X, y, df_orig, target_col, meta):
    try:
        import flaml.automl.model as fm
    except ImportError:
        return FIT_SKIP, "[flaml] not installed - run pip install flaml"
    cls = getattr(fm, model_name, None)
    if not cls:
        return FIT_SKIP, f"[flaml] class not found: {model_name}"
    cleaned = clean_params(params)
    if "Spark" in model_name:
        return (
            FIT_SKIP,
            "[flaml] Spark model - requires external cluster, excluded from experiments",
        )
    try:
        cls(task="binary", **flaml_filter(cls, cleaned)).fit(X, y)
        return FIT_SUCCESS, None
    except Exception as e:
        return classify_error(str(e)), str(e)[:250]


def fit_autogluon(model_name, params, X, y, df_orig, target_col, meta):
    import tempfile

    module_path = AG_CLASS_MAP.get(model_name)
    if not module_path:
        return FIT_SKIP, f"[autogluon] no class map entry for {model_name}"
    module_name, class_name = module_path.rsplit(".", 1)
    try:
        import importlib

        cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as e:
        return FIT_SKIP, str(e)[:150]
    if df_orig is None:
        return FIT_SKIP, "[autogluon] requires DataFrame input"
    feature_cols = [c for c in df_orig.columns if c != target_col]
    X_df = df_orig[feature_cols].reset_index(drop=True)
    y_ser = df_orig[target_col].reset_index(drop=True)
    y_ser = (y_ser == sorted(y_ser.unique())[-1]).astype(int)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            cls(
                path=tmp,
                name="fit_test",
                problem_type="binary",
                eval_metric="roc_auc",
                hyperparameters=clean_params(params),
            ).fit(X=X_df, y=y_ser)
            return FIT_SUCCESS, None
        except Exception as e:
            msg = str(e)
            if any(dep in msg for dep in ("torch", "fastai", "interpret", "xgboost")):
                return (
                    FIT_SKIP,
                    f"[autogluon] optional dependency not installed: {msg[:120]}",
                )
            return classify_error(msg), msg[:250]


def fit_h2o(model_name, params, X, y, df_orig, target_col, meta):
    try:
        import h2o.estimators as h2o_est
    except ImportError:
        return FIT_SKIP, "[h2o] not installed - run pip install h2o"
    resolved = resolve_h2o_class(model_name)
    cls = getattr(h2o_est, resolved, None) or getattr(h2o_est, model_name, None)
    if not cls:
        return FIT_SKIP, f"[h2o] class not found: {model_name}"
    cleaned = {
        k: v
        for k, v in clean_params(params).items()
        if k not in H2O_INFRA_EXACT and not k.endswith(H2O_INFRA_SUFFIXES)
    }
    try:
        cls(**filter_sig(cls, cleaned))
        return FIT_SUCCESS, None
    except Exception as e:
        return classify_error(str(e)), str(e)[:250]


FIT_FUNCTIONS = {
    "autosklearn": fit_autosklearn,
    "tpot": fit_tpot,
    "flaml": fit_flaml,
    "autogluon": fit_autogluon,
    "h2o": fit_h2o,
}

SEP = "|||"


def load_checkpoint(path: Path) -> tuple[list, dict, set]:
    if not path.exists():
        return [], {}, set()
    try:
        data = json.loads(path.read_text())
        seen = {
            tuple(k.split(SEP)): v for k, v in data.get("seen_triples", {}).items()
        }
        results = data.get("results", [])
        done_idx = set(data.get("done_idx", []))
        print(
            f"Resumed from checkpoint: {len(results)} results, {len(done_idx)} configs done"
        )
        return results, seen, done_idx
    except Exception as e:
        print(f"Warning: could not load checkpoint, {e}")
        return [], {}, set()


def save_checkpoint(results: list, seen: dict, done_idx: set, path: Path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "results": results,
                "seen_triples": {SEP.join(k): v for k, v in seen.items()},
                "done_idx": sorted(done_idx),
            }
        )
    )
    tmp.replace(path)


def maybe_checkpoint(
    results, seen_triples, done_idx, checkpoint_path, checkpoint_interval, last_save
):
    if (time.time() - last_save) < checkpoint_interval:
        return last_save
    if checkpoint_path:
        save_checkpoint(results, dict(seen_triples), done_idx, checkpoint_path)
        print(f"Checkpoint saved, {len(results)} results")
    return time.time()


def run(
    may_path=MAY_XLSX,
    june_path=JUNE_XLSX,
    max_per_triple=3,
    n_sample=50,
    seed=1,
    dataset_ids=None,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = CHECKPOINT_INTERVAL_SECS,
):
    engine = TranslationEngine()
    matrix = engine.matrix()

    configs = []
    if may_path.exists():
        configs.extend(load_configs_from_excel(str(may_path)))
    if june_path.exists():
        configs.extend(load_configs_from_june_excel(str(june_path)))

    loaded_datasets = {}
    for did in dataset_ids or DEFAULT_DATASETS:
        resolved = resolve_dataset_id(did)
        X, y, df_orig, name = load_dataset(resolved, n_sample, seed)
        if X is not None:
            meta_path = (
                ENGINE_DIR.parent
                / "automl-data"
                / "datasets"
                / f"{resolved}_metadata.json"
            )
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            target_col = meta.get("target_name", "class")
            loaded_datasets[resolved] = {
                "X": X,
                "y": y,
                "df": df_orig,
                "name": name,
                "target": target_col,
                "meta": meta,
            }
            print(f"Loaded {name}: {X.shape[0]} rows x {X.shape[1]} features")
        else:
            print(f"Skipping {did}, not found")

    if not loaded_datasets:
        print("No datasets found.")
        return []

    prior_results, prior_seen, done_idx = (
        load_checkpoint(checkpoint_path) if checkpoint_path else ([], {}, set())
    )
    results = list(prior_results)
    seen_triples = defaultdict(int, prior_seen)
    tested = len(results)
    last_save = time.time()

    print(f"\nRunning on {len(loaded_datasets)} datasets, max {max_per_triple} per triple\n")

    canonicals = [engine.resolve_model(cfg["framework"], cfg["model"]) for cfg in configs]
    groups = defaultdict(list)
    for i, (cfg, canonical) in enumerate(zip(configs, canonicals)):
        if canonical is not None and canonical in CLASSIFIER_CANONICALS:
            groups[(canonical, cfg["framework"])].append(i)

    selected = set()
    for idxs in groups.values():
        k = min(max_per_triple, len(idxs))
        step = len(idxs) / k
        selected.update(idxs[int(i * step)] for i in range(k))

    for i, cfg in enumerate(configs):
        if i not in selected or i in done_idx:
            continue
        src_fw, model_cls, params = cfg["framework"], cfg["model"], cfg["params"]
        canonical = canonicals[i]
        fw_row = matrix.get(canonical, {})

        for tgt_fw in sorted(FIT_FUNCTIONS):
            if tgt_fw == src_fw or not fw_row.get(tgt_fw):
                continue
            triple = (canonical, src_fw, tgt_fw)
            seen_triples[triple] += 1

            translation = engine.translate(
                {"model": model_cls, "params": params}, source=src_fw, target=tgt_fw
            )
            tgt_model = translation["model"]
            if tgt_model.startswith("["):
                continue

            n_params = len(params)
            n_unmapped = len(translation["unmapped_params"])
            param_cov = (
                round((n_params - n_unmapped) / n_params, 4) if n_params else 1.0
            )

            for did, ds in loaded_datasets.items():
                tested += 1
                outcome, error = FIT_FUNCTIONS[tgt_fw](
                    tgt_model,
                    translation["params"],
                    ds["X"],
                    ds["y"],
                    ds["df"],
                    ds["target"],
                    ds["meta"],
                )
                line = f"{tested}: {src_fw}/{model_cls} -> {tgt_fw}/{tgt_model} ({ds['name']}) cov={param_cov:.0%} outcome={outcome}"
                if error:
                    line += f"  {error[:100]}"
                print(line)

                results.append(
                    {
                        "dataset_id": did,
                        "dataset_name": ds["name"],
                        "source_framework": src_fw,
                        "source_model": model_cls,
                        "target_framework": tgt_fw,
                        "target_model": tgt_model,
                        "canonical_model": canonical,
                        "param_coverage": param_cov,
                        "n_source_params": n_params,
                        "n_unmapped_params": n_unmapped,
                        "fit_outcome": outcome,
                        "fit_error": error,
                    }
                )

                last_save = maybe_checkpoint(
                    results,
                    seen_triples,
                    done_idx,
                    checkpoint_path,
                    checkpoint_interval,
                    last_save,
                )

        done_idx.add(i)

    if checkpoint_path:
        save_checkpoint(results, dict(seen_triples), done_idx, checkpoint_path)
    return results


def print_summary(results: list):
    if not results:
        print("No results.")
        return
    df = pd.DataFrame(results)
    print(f"\nTotal: {len(df)}")
    print(df["fit_outcome"].value_counts())

    pair = df.groupby(["source_framework", "target_framework"])["fit_outcome"].value_counts().unstack(fill_value=0)
    print(f"\n{pair}")

    errs = df[df["fit_error"].notna()].drop_duplicates(
        ["source_framework", "target_framework", "fit_error"]
    )
    if len(errs):
        print(f"\nDistinct errors:\n{errs[['source_framework', 'target_framework', 'fit_outcome', 'fit_error']]}")


# CLI defaults for the full exhaustive run
# run()'s own defaults are smaller for quick usage
DEFAULT_N_SAMPLE = 50
DEFAULT_MAX_PER_TRIPLE = 99999


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--max-per-triple", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--datasets", nargs="+", default=None, help="add EXTRA_DATASETS entries to test more than the default")
    parser.add_argument("--resume", default=None, help="path to an existing checkpoint json to continue from")
    args = parser.parse_args()

    sample = args.sample or DEFAULT_N_SAMPLE
    max_per_triple = args.max_per_triple or DEFAULT_MAX_PER_TRIPLE
    datasets = args.datasets or DEFAULT_DATASETS

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    out = args.out or str(RESULTS_DIR / f"{timestamp}_fit_success_run.csv")
    if args.resume:
        checkpoint_path = Path(args.resume)
    else:
        checkpoint_dir = RESULTS_DIR / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{Path(out).stem}_{timestamp}.json"

    results = run(
        max_per_triple=max_per_triple,
        n_sample=sample,
        dataset_ids=datasets,
        checkpoint_path=checkpoint_path,
    )
    print_summary(results)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n{len(results)} results -> {out}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
