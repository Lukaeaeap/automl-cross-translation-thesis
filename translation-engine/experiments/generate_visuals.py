"""
Creates all key E1, E2, and E3 result tables.

Run from root:
    python translation-engine/experiments/generate_visuals.py

Outputs to experiments/results/tables/
"""

import json
import sys
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from constants import ENSEMBLE_KEYWORDS, FIT_SUCCESS, FIT_FAIL, FIT_FAIL_RANGE
from config_loader import is_ensemble, parse_raw
from translators import REGISTRY

RESULTS = HERE / "results"
CSV_DIR = RESULTS / "tables"
CSV_DIR.mkdir(exist_ok=True)
ONTOLOGY_DIR = HERE.parent / "ontology"

FW_ORDER = ["autogluon", "autosklearn", "flaml", "h2o", "tpot"]
FW_LABELS = ["AutoGluon", "auto-sklearn", "FLAML", "H2O", "TPOT"]
FW_MAP = dict(zip(FW_ORDER, FW_LABELS))


# newest results file containing stem in its name, csv or json
def latest(stem):
    matches = sorted(
        [*RESULTS.glob(f"*{stem}*.csv"), *RESULTS.glob(f"*{stem}*.json")],
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"No results file matching '*{stem}*' in {RESULTS}")
    return matches[-1]


# load results as a list of dicts, works for either csv or json
def load_records(stem):
    path = latest(stem)
    print(f"Using {stem} results: {path.name}")
    if path.suffix == ".csv":
        return pd.read_csv(path).to_dict("records")
    return json.loads(path.read_text())


# Load results data
m3_data = load_records("fit_success")
translatability_data = load_records("translatability")

def save_csv(filename, headers, rows):
    # Save a table as a CSV.
    df = pd.DataFrame(rows, columns=headers)
    path = CSV_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8")
    return path

def get_outcome(r):
    return r.get("fit_outcome") or r.get("outcome") or ""


PAIR = ["source_framework", "target_framework"]


def pair_counts(df, mask=None):
    # (src, tgt) -> row count
    # if a mask is provided, the data is filtered using a boolean mask
    return (df[mask] if mask is not None else df).groupby(PAIR).size().to_dict()


m3 = pd.DataFrame(m3_data)
m3["outcome"] = [get_outcome(r) for r in m3_data]

total = len(m3)
pair_total = pair_counts(m3)  
pair_ok = pair_counts(m3, m3["outcome"] == FIT_SUCCESS) 

# translatable = engine.translate() produced a real model, not just a models.json matrix hit
translatability = pd.DataFrame(translatability_data)
pair_attempt_n = pair_counts(translatability)
pair_attempt_tr = pair_counts(translatability, translatability["translatable"])


pair_cov = (
    m3.dropna(subset=["param_coverage"]).groupby(PAIR)["param_coverage"].mean().to_dict()
)

def mean_of(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0


src_avg = {src: mean_of(pair_cov.get((src, tgt)) for tgt in FW_ORDER) for src in FW_ORDER}
tgt_avg = {tgt: mean_of(pair_cov.get((src, tgt)) for src in FW_ORDER) for tgt in FW_ORDER}

overall_src = sum(src_avg[fw] for fw in FW_ORDER) / len(FW_ORDER)
overall_tgt = sum(tgt_avg[fw] for fw in FW_ORDER) / len(FW_ORDER)


overall_cov = (overall_src + overall_tgt) / 2

h_cov = ["Source \\ Target"] + FW_LABELS + ["Total (as source)"]
r_cov = []
for src in FW_ORDER:
    row = [FW_MAP[src]]
    for tgt in FW_ORDER:
        cov = pair_cov.get((src, tgt))
        row.append(f"{cov:.1%}" if cov is not None else "-")
    row.append(f"{src_avg[src]:.1%}")
    r_cov.append(row)

total_row_cov = (
    ["Total (as target)"]
    + [f"{tgt_avg[tgt]:.1%}" for tgt in FW_ORDER]
    + [f"{overall_cov:.1%}"]
)
r_cov.append(total_row_cov)

save_csv("results_table5_pairwise_param_coverage.csv", h_cov, r_cov)


fw_outcome_headers = [
    "Framework",
    "Translations",
    "Success",
    "Error from fit",
    "Error from incorrect value range",
]


def outcome_breakdown(role):
    # role: 'source_framework' or 'target_framework'
    counts = m3.groupby([role, "outcome"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=FW_ORDER, columns=counts.columns, fill_value=0)

    def row(label, c):
        return [
            label,
            f"{int(c.sum()):,}",
            f"{int(c.get(FIT_SUCCESS, 0)):,}",
            f"{int(c.get(FIT_FAIL, 0)):,}",
            f"{int(c.get(FIT_FAIL_RANGE, 0)):,}"
        ]

    rows_out = [row(label, counts.loc[fw]) for fw, label in zip(FW_ORDER, FW_LABELS)]
    rows_out.append(row("Total", counts.sum()))
    return rows_out


rows_by_source = outcome_breakdown("source_framework")
save_csv("results_table3a_fit_outcomes_by_source.csv", fw_outcome_headers, rows_by_source)

rows_by_target = outcome_breakdown("target_framework")
save_csv("results_table3b_fit_outcomes_by_target.csv", fw_outcome_headers, rows_by_target)

def pair_total_table(n_map, hit_map, fmt):
    # FW_ORDER x FW_ORDER table

    def totals(fixed_is_src, fixed_fw):
        others = FW_ORDER
        if fixed_is_src:
            pairs = [(fixed_fw, o) for o in others if o != fixed_fw]
        else:
            pairs = [(o, fixed_fw) for o in others if o != fixed_fw]
        n = sum(n_map.get(p, 0) for p in pairs)
        hit = sum(hit_map.get(p, 0) for p in pairs)
        return n, hit

    headers = ["Source \\ Target"] + FW_LABELS + ["Total"]
    rows_out = []
    for src, src_label in zip(FW_ORDER, FW_LABELS):
        row = [src_label]
        for tgt in FW_ORDER:
            n = n_map.get((src, tgt), 0)
            row.append(fmt(n, hit_map.get((src, tgt), 0)) if n else "-")
        row.append(fmt(*totals(True, src)))
        rows_out.append(row)

    total_row = ["Total"]
    for tgt in FW_ORDER:
        total_row.append(fmt(*totals(False, tgt)))
    total_row.append(fmt(sum(n_map.values()), sum(hit_map.values())))
    rows_out.append(total_row)
    return headers, rows_out


def fmt_count_rate(n, ok):
    pct = 100 * ok / n if n else 0
    if pct >= 100.0:
        return f"{n:,} / {ok:,}"
    return f"{n:,} / {ok:,} ({pct:.1f}%)"


def fmt_count(n, hit):
    return f"{hit:,} / {n:,}"


save_csv(
    "results_table4_fit_success_pair_count_and_rate.csv",
    *pair_total_table(pair_total, pair_ok, fmt_count_rate),
)

save_csv(
    "results_table2_translatability_pair_counts.csv",
    *pair_total_table(pair_attempt_n, pair_attempt_tr, fmt_count),
)

pair_fail = pair_counts(m3, m3["outcome"] == FIT_FAIL)
pair_fail_range = pair_counts(m3, m3["outcome"] == FIT_FAIL_RANGE)

h_st = ["source", "target", "count", "success", "fail", "fail_range", "avg param cov %"]
r_st = []
for src in FW_ORDER:
    if src not in {k[0] for k in pair_total}:
        continue
    for tgt in FW_ORDER:
        n = pair_total.get((src, tgt), 0)
        if n == 0:
            continue
        ok = pair_ok.get((src, tgt), 0)
        fail = pair_fail.get((src, tgt), 0)
        fail_range = pair_fail_range.get((src, tgt), 0)
        avg_cov = (pair_cov.get((src, tgt)) or 0) * 100
        r_st.append([src, tgt, f"{n:,}", f"{ok:,}", fail, fail_range,round(avg_cov, 1)])

save_csv("total_results_table_source_target_success_breakdown.csv", h_st, r_st)

CLASSIFIER_ROLES = {"classifier", "both"}
roles = json.loads((ONTOLOGY_DIR / "algorithm_roles.json").read_text())
tested_set = set(m3["canonical_model"])

# Source excel config data usage: raw configs per framework, and how many are dropped
# (ensemble models, invalid params from json) before use by the experiments.
NEW_DATA = HERE.parent.parent / "automl-data" / "New-Data"
MAY_XLSX = NEW_DATA / "automl_results_may.xlsx"
JUNE_XLSX = NEW_DATA / "results_binary_June_15_new.xlsx"


def analyze_source(df_raw, frameworks, algo_col, params_col, use_keyword_fallback=False):
    stats = {}
    for fw in frameworks:
        raw_rows = df_raw[df_raw["framework"] == fw]
        total = len(raw_rows)
        non_ensemble = raw_rows
        if "is_ensemble" in raw_rows.columns:
            non_ensemble = non_ensemble[non_ensemble["is_ensemble"] == 0]
        elif use_keyword_fallback:
            non_ensemble = non_ensemble[
                ~non_ensemble[algo_col].str.lower().str.contains("|".join(ENSEMBLE_KEYWORDS), na=False)
            ]
        non_ensemble = non_ensemble[
            ~non_ensemble[algo_col].astype(str).str.strip().apply(is_ensemble)
        ]
        parse_ok = sum(
            1
            for _, row in non_ensemble.iterrows()
            if isinstance(parse_raw(row, params_col), dict)
        )
        stats[fw] = {
            "raw_total": total,
            "ensemble_dropped": total - len(non_ensemble),
            "final": parse_ok,
        }
    return stats


may_stats = analyze_source(pd.read_excel(MAY_XLSX), ("autosklearn", "tpot", "h2o"), "algorithm", "params_json")

df_june = pd.read_excel(JUNE_XLSX, sheet_name="TopK_Summary")
june_algo_col = "algo" if "algo" in df_june.columns else "algorithm"
june_params_col = "params_json" if "params_json" in df_june.columns else "refit_params_json"
june_stats = analyze_source(
    df_june, ("h2o", "autogluon", "flaml"), june_algo_col, june_params_col, use_keyword_fallback=True
)

# merge per framework across both source files (h2o appears in both)
combined = pd.DataFrame(may_stats).T.add(pd.DataFrame(june_stats).T, fill_value=0)
combined = combined[["raw_total", "ensemble_dropped", "final"]].astype(int).sort_index()
combined.loc["TOTAL"] = combined.sum()

h11 = ["framework", "raw_rows", "dropped_ensemble", "final_configs_used"]
r11 = [list(row) for row in combined.itertuples(name=None)]

save_csv("results_table1_source_data_usage.csv", h11, r11)

# Per model cross-framework hyperparameter overlap. 
# A structural property of the ontology.
models_ont = json.loads((ONTOLOGY_DIR / "models.json").read_text())
params_ont = json.loads((ONTOLOGY_DIR / "params.json").read_text())
params_patch = json.loads((ONTOLOGY_DIR / "params_patch.json").read_text())
for canonical, overrides in params_patch.items():
    if canonical.startswith("_"):
        continue
    params_ont.setdefault(canonical, {}).update(overrides)

from translators.value_constraints import SKLEARN_HARD_CONSTRAINTS

shared_constraint_keys = set(SKLEARN_HARD_CONSTRAINTS.keys())
patch_mappings_per_fw = {fw: 0 for fw in FW_ORDER}
for canonical, params in params_patch.items():
    if canonical.startswith("_"):
        continue
    for pname, info in params.items():
        for fw in info.get("mappings", {}):
            patch_mappings_per_fw[fw] += 1

h_manual = [
    "Framework",
    "Receive-side renames or value transformations",
    "Emit-side renames",
    "Framework specific value constraints",
    "Ontology patch mappings (corrected/added)",
    "Total manual interventions",
]
r_manual = []
grand_total = 0
for fw, label in zip(FW_ORDER, FW_LABELS):
    cls = REGISTRY[fw]
    n_receive = len(cls.RECEIVE_TRANSFORMS)
    n_emit = len(cls.EMIT_TRANSFORMS)
    n_own_constraints = len(set(cls.VALUE_CONSTRAINTS.keys()) - shared_constraint_keys)
    n_patch = patch_mappings_per_fw[fw]
    row_total = n_receive + n_emit + n_own_constraints + n_patch
    grand_total += row_total
    r_manual.append([label, n_receive, n_emit, n_own_constraints, n_patch, row_total])

n_shared = len(shared_constraint_keys)
grand_total += n_shared
r_manual.append(["Shared sklearn-family value constraints", "", "", n_shared, "", n_shared])
r_manual.append(["Total", "", "", "", "", grand_total])
save_csv("results_table10_manual_interventions.csv", h_manual, r_manual)

n_translations_by_model = m3.groupby("canonical_model").size().to_dict()

def transform_canonicals(translator_cls):
    canonicals = set(translator_cls.EMIT_TRANSFORMS.keys())
    for entry in translator_cls.RECEIVE_TRANSFORMS.values():
        canonicals.add(entry[0] if isinstance(entry, tuple) else entry)
    return canonicals

TRANSFORM_CANONICALS = {fw: transform_canonicals(cls) for fw, cls in REGISTRY.items()}

model_rows = []
for canonical, pinfo in params_ont.items():
    applicable = [fw for fw in FW_ORDER if models_ont.get(canonical, {}).get(fw)]
    if len(applicable) <= 1 or not pinfo:
        continue
    if roles.get(canonical) not in CLASSIFIER_ROLES:
        continue
    shared_gt1 = sum(
        1
        for pname, info in pinfo.items()
        if sum(
            1
            for fw in applicable
            if fw in info.get("mappings", {}) or pname in TRANSFORM_CANONICALS.get(fw, set())
        ) > 1
    )
    model_rows.append(
        {
            "canonical_model": canonical,
            "frameworks_with_model": len(applicable),
            "shared_hyperparams": shared_gt1,
            "role": roles.get(canonical, "unknown"),
            "tested_in_fit_experiment": "yes" if canonical in tested_set else "no",
            "n_translations_tested": n_translations_by_model.get(canonical, 0),
        }
    )
model_overlap = pd.DataFrame(model_rows).sort_values(
    ["frameworks_with_model", "shared_hyperparams"], ascending=[False, False]
)

t8_cols = [
    "canonical_model",
    "frameworks_with_model",
    "shared_hyperparams",
    "role",
    "tested_in_fit_experiment",
    "n_translations_tested",
]
save_csv(
    "results_table6_hyperparam_overlap_by_model.csv",
    t8_cols,
    model_overlap[t8_cols].values.tolist(),
)

# Distinct fit-failure error messages, listed verbatim
err_rows = (
    m3[m3["fit_error"].notna()]
    .groupby(["outcome", "fit_error"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
save_csv(
    "results_table7_fit_error_breakdown.csv",
    ["outcome", "fit_error", "count"],
    err_rows.values.tolist(),
)

cov_by_model_pair = m3.groupby(["canonical_model", "source_framework", "target_framework"])["param_coverage"].mean()
succ_by_model_pair = m3[m3["outcome"] == FIT_SUCCESS].groupby(["canonical_model", "source_framework", "target_framework"]).size()

h_mf = ["canonical_model"] + FW_LABELS
for src, src_label in zip(FW_ORDER, FW_LABELS):
    models = sorted(m3[m3["source_framework"] == src]["canonical_model"].unique())
    r_mf = []
    for model in models:
        row = [model]
        for tgt in FW_ORDER:
            key = (model, src, tgt)
            if key in cov_by_model_pair.index:
                row.append(f"{cov_by_model_pair[key]:.1%} ({succ_by_model_pair.get(key, 0)})")
            else:
                row.append("-")
        r_mf.append(row)
    save_csv(f"results_table8_model_by_target_source_{src}.csv", h_mf, r_mf)

cov_by_model_fw = m3.groupby(["canonical_model", "source_framework"])["param_coverage"].mean()
succ_by_model_fw = m3[m3["outcome"] == FIT_SUCCESS].groupby(["canonical_model", "source_framework"]).size()

r_mf_agg = []
for model in sorted(m3["canonical_model"].unique()):
    row = [model]
    for fw in FW_ORDER:
        key = (model, fw)
        if key in cov_by_model_fw.index:
            row.append(f"{cov_by_model_fw[key]:.1%} ({succ_by_model_fw.get(key, 0)})")
        else:
            row.append("-")
    r_mf_agg.append(row)
save_csv("results_table8_model_by_framework_transferability.csv", h_mf, r_mf_agg)

ontology_coverage_path = latest("ontology_coverage")
print(f"Using ontology coverage results: {ontology_coverage_path.name}")
if ontology_coverage_path.suffix == ".csv":
    cov_df = pd.read_csv(ontology_coverage_path)
    cov_row = cov_df[(cov_df["view"] == "raw") & (cov_df["framework"] == "OVERALL")].iloc[0]
    raw_overall = {"expected_params": cov_row["expected"], "covered_params": cov_row["covered"]}

    search_df = cov_df[cov_df["view"] == "search_space"]
    cross_fw_df = cov_df[cov_df["view"] == "search_space_cross_framework"]
    r_tsc = []
    for fw, label in zip(FW_ORDER, FW_LABELS):
        row = search_df[search_df["framework"] == fw].iloc[0]
        cross_row = cross_fw_df[cross_fw_df["framework"] == fw].iloc[0]
        r_tsc.append([
            label,
            f"{row['covered']}/{row['expected']}",
            f"{cross_row['covered']}/{cross_row['expected']}",
        ])
    total_row = search_df[search_df["framework"] == "OVERALL"].iloc[0]
    total_cross_row = cross_fw_df[cross_fw_df["framework"] == "OVERALL"].iloc[0]
    r_tsc.append([
        "Total",
        f"{total_row['covered']}/{total_row['expected']}",
        f"{total_cross_row['covered']}/{total_cross_row['expected']}",
    ])
    save_csv(
        "results_table9_translation_system_coverage.csv",
        ["Framework", "Present in ontology", "Cross-framework mapping"],
        r_tsc,
    )
else:
    ontology_coverage = json.loads(ontology_coverage_path.read_text())
    raw_overall = next(
        r
        for r in ontology_coverage["hyperparameter_coverage"]["raw"]
        if r["framework"] == "OVERALL"
    )

ontology_model_roles = {k: v for k, v in roles.items() if not k.startswith("_")}
classifier_or_both = {k for k, v in ontology_model_roles.items() if v in CLASSIFIER_ROLES}

multi_fw_models = {
    canonical
    for canonical, fw_map in models_ont.items()
    if sum(1 for fw in FW_ORDER if fw_map.get(fw)) > 1
}
multi_fw_classifier_or_both = multi_fw_models & classifier_or_both

summary_rows = [
    ["Total models in ontology", len(ontology_model_roles)],
    ["Models with classifier/both role", len(classifier_or_both)],
    ["Models tested in fit experiment", len(tested_set & multi_fw_classifier_or_both)],
    ["Models present in more than 1 framework", len(multi_fw_models)],
    ["Models present in more than 1 framework with classifier/both role", len(multi_fw_classifier_or_both)],
    ["Unique config parameters found in configurations datasets", raw_overall["expected_params"]],
    ["Unique config hyperparameters covered by translation system", raw_overall["covered_params"]],
]

save_csv("results_table0_overall_summary.csv", ["metric", "value"], summary_rows)

print(f"Saved all tables to {CSV_DIR}")

