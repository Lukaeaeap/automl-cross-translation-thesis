"""
Alignment of model and hyperparameter names across AutoML frameworks using Natural Language Processing (NLP).

Summary of model alignment:
- Normalizing of names by converting CamelCase -> snake_case + noise-token removal
- Abbreviation expansion (RF->random_forest, XGB->xgboost, ...)
- rapidfuzz fuzzy merge for residual ambiguities

Summary of parameter alignment:
  - rapidfuzz name similarity on normalised param names
  - Sentence-transformer semantic similarity on name + description texts
  - Union-Find clustering cross-framework pairs only
"""

import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

# Shared: Union-Find with path compression
class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a, b, prefer=None):
        # Merge groups of a and b. prefer(ra, rb) -> root to keep.
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        keep, drop = prefer(ra, rb) if prefer else (ra, rb)
        self.parent[drop] = keep
        return True


# Attempt to initialize sentence transformer model with "all-MiniLM-L6-v2"
ST_MODEL = None

def get_st_model():
    global ST_MODEL
    if ST_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer

            ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            raise RuntimeError(f"sentence-transformers is required but failed to load: {e}") from e
    return ST_MODEL


def desc_similarity(texts: list) -> np.ndarray:
    # Calculate cosine similarity of encoded model
    model = get_st_model()
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return cosine_similarity(emb)

# Abbreviations of models used internally
ABBREV: Dict[str, str] = {
    "rf": "random_forest",
    "xt": "extra_trees",
    "knn": "k_nearest_neighbors",
    "lgb": "lightgbm",
    "lgbm": "lightgbm",
    "lightgbm": "lightgbm",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "xg_boost": "xgboost",
    "mlp": "neural_network",
    "deeplearning": "neural_network",
    "deep_learning": "neural_network",
    "svc": "svm",
    "svr": "svm",
    "lda": "linear_discriminant_analysis",
    "qda": "quadratic_discriminant_analysis",
    "lrl1": "logistic_regression",
    "lrl2": "logistic_regression",
    "lrl": "logistic_regression",
    "gbm": "gradient_boosting",
    "gbt": "gradient_boosting",
    "dt": "decision_tree",
    "nb": "naive_bayes",
    "ada": "adaboost",
    "ica": "independent_component_analysis",
    "cat_boost": "catboost",
    "support_vector_machine": "svm",
    "liblinear_svc": "svm",
    "libsvm_svc": "svm",
    "liblinear_svr": "svm",
    "libsvm_svr": "svm",
    "k_neighbors": "k_nearest_neighbors",
    "kneighbors": "k_nearest_neighbors",
    "bernoullinb": "naive_bayes",
    "multinomialnb": "naive_bayes",
    "gaussiannb": "naive_bayes",
    "bernoulli_nb": "naive_bayes",
    "multinomial_nb": "naive_bayes",
    "gaussian_nb": "naive_bayes",
    "linear_discriminant": "linear_discriminant_analysis",
    "quadratic_discriminant": "quadratic_discriminant_analysis",
    "discriminant_analysis": "linear_discriminant_analysis",
}

# Noise to ignore
NOISE = {
    "h2o",
    "estimator",
    "classifier",
    "regressor",
    "model",
    "spark",
    "sklearn",
    "tabular",
    "neural",
    "net",
    "torch",
    "fast",
    "ai",
    "transformers",
    "selection",
    "preproc",
    "preprocessor",
    "large",
    "limit",
    "depth",
    "autoencoder",
    "prep",
    "stack",
    "linear",
    "auto",
    "encoder",
}


def camel_to_snake(name: str) -> str:
    # Split acronyms from words ("XGBoost" to "XG_Boost")
    # Than split remaining lower/upper letter neighbours.

    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def normalize_model_name(name: str) -> str:
    # normalize model names by convertion to snake case, removing noise and using abbreviaions

    stripped = name
    for prefix in ("H2O", "Spark"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].lstrip("_") or stripped
            break

    raw_tokens = re.split(r"[^a-z0-9]+", camel_to_snake(stripped))
    tokens = [t for t in raw_tokens if t and t not in NOISE]
    if not tokens:
        return name.lower()

    key = "_".join(tokens)
    joined = "".join(tokens)

    for candidate in (
        key,
        joined,
        tokens[0],
        "".join(tokens[:2]) if len(tokens) >= 2 else None,
    ):
        if candidate and candidate in ABBREV:
            return ABBREV[candidate]

    return key


def cluster_models(
    df: pd.DataFrame, fuzzy_threshold: float = 96.0
) -> Dict[str, Dict[str, List[str]]]:

    # Attempt to group (framework, class) pairs into canonical model buckets by using exact grouping and a fuzzy clustering.

    records = df[["framework", "class"]].drop_duplicates().copy()
    records["normalized"] = records["class"].map(normalize_model_name)

    # Attempt exact grouping by normalized name
    exact = records.groupby('normalized').apply(
        lambda g: g.groupby('framework')['class'].apply(list).to_dict()).to_dict()

    exact = defaultdict(lambda: defaultdict(list), exact)

    # Do a fuzzy merge where a longer (more descriptive) canonical name is preffered
    names = list(exact.keys())
    uf = UnionFind(names)

    def find_longer(ra, rb):
        return (ra, rb) if len(ra) >= len(rb) else (rb, ra)

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if not uf.find(a) == uf.find(b):
                if (fuzz.token_sort_ratio(a.replace("_", " "), b.replace("_", " ")) >= fuzzy_threshold):
                    uf.union(a, b, prefer=find_longer)
    
    clusters: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    for norm, fw_dict in exact.items():
        canonical = uf.find(norm)
        for fw, names_list in fw_dict.items():
            clusters[canonical][fw].extend(names_list)

    return {
        k: {fw: sorted(set(v)) for fw, v in fwd.items()}
        for k, fwd in sorted(clusters.items())
    }

# Parameter clustering 
def normalize_param_name(name: str) -> str:
    # Normalize parameter names by filtering out dots, underscores, and common unneccesary pre- and postfixes to prep for fuzzy merge
    s = name.lower().replace(".", "_")
    for pfx in ("n_", "num_", "number_of_", "nb_"):
        if s.startswith(pfx) and len(s) > len(pfx):
            s = s[len(pfx) :]
            break
    for sfx in ("_count", "_number", "_size"):
        if s.endswith(sfx):
            s = s[: -len(sfx)]
            break
    return s.replace("_", " ").strip()


def collect_param_rows(
    df: pd.DataFrame, model_mapping: Dict[str, List[str]]
) -> Optional[pd.DataFrame]:
    # Build table of hyperparameters to cluster
    rows = []
    for fw, model_names in model_mapping.items():
        sub = (
            df[(df["framework"] == fw) & (df["class"].isin(model_names))][
                ["parameter_name", "description", "value_type", "value_default"]
            ]
            .drop_duplicates("parameter_name")
            .copy()
        )
        sub["framework"] = fw
        rows.append(sub)
    if not rows:
        return None
    df_out = pd.concat(rows, ignore_index=True).rename(
        columns={"parameter_name": "param"}
    )
    df_out["description"] = df_out["description"].fillna("").astype(str)
    return df_out if len(df_out) > 0 else None


def similarity_matrices(
    param_df: pd.DataFrame, name_w: float, desc_w: float
) -> np.ndarray:

    # Calculate similarity between parameters based on name and description

    norms = param_df["param"].map(normalize_param_name).tolist()
    n = len(norms)
    texts = [
        f"{r['param']}: {r['description']}" if r["description"].strip() else r["param"]
        for _, r in param_df.iterrows()
    ]

    name_sim = np.array(
        [
            [fuzz.token_sort_ratio(norms[i], norms[j]) / 100.0 for j in range(n)]
            for i in range(n)
        ]
    )
    desc_sim = desc_similarity(texts) if n >= 2 else np.eye(n)

    combined = name_w * name_sim + desc_w * desc_sim

    # Zero for pairs from the same framework, so they do not get merged
    fws = param_df["framework"].tolist()
    for i in range(n):
        for j in range(n):
            if i != j and fws[i] == fws[j]:
                combined[i, j] = 0.0

    return combined, name_sim, desc_sim


def cluster_indices(
    n: int,
    frameworks: list,
    combined: np.ndarray,
    name_sim: np.ndarray,
    desc_sim: np.ndarray,
    threshold: float,
    min_name: float,
    min_desc: float,
) -> List[List[int]]:
    uf = UnionFind(range(n))
    component_fws: list = [{fw} for fw in frameworks]

    for i in range(n):
        for j in range(i + 1, n):
            if (frameworks[i] == frameworks[j]
                or combined[i, j] < threshold
                or name_sim[i, j] < min_name
                or desc_sim[i, j] < min_desc):
                continue
            ri, rj = uf.find(i), uf.find(j)
            if ri == rj or (component_fws[ri] & component_fws[rj]):
                continue  # would create same-framework conflict
            uf.union(i, j)
            component_fws[ri] |= component_fws[rj]

    buckets = defaultdict(list)
    for i in range(n):
        buckets[uf.find(i)].append(i)
    return list(buckets.values())


def cluster_params(
    df: pd.DataFrame,
    model_mapping: Dict[str, List[str]],
    name_weight: float = 0.55,
    desc_weight: float = 0.45,
    threshold: float = 0.50,
    min_name_sim: float = 0.25,
    min_desc_sim: float = 0.42,
) -> Dict[str, Dict]:
    # Cluster hyperparameter names per canonical model from all frameworks by grouping based on similarity score per canonical models set of hyperparameters 

    param_df = collect_param_rows(df, model_mapping)
    if param_df is None:
        return {}

    combined, name_sim, desc_sim = similarity_matrices(
        param_df, name_weight, desc_weight
    )
    groups = cluster_indices(
        len(param_df),
        param_df["framework"].tolist(),
        combined,
        name_sim,
        desc_sim,
        threshold,
        min_name_sim,
        min_desc_sim,
    )

    result: Dict[str, Dict] = {}
    for members in groups:
        rows = [param_df.iloc[m] for m in members]
        params_list = [r["param"] for r in rows]
        param_counts = Counter(params_list)
        canonical = max(param_counts, key=lambda p: (param_counts[p], -len(p)))
        best_desc = max(
            (r["description"] for r in rows),
            key=lambda d: len(d) if d.strip() else 0,
            default="",
        )

        mappings, value_types, value_defaults = {}, {}, {}
        for row in rows:
            fw = row["framework"]
            if fw not in mappings:
                mappings[fw] = row["param"]
                if pd.notna(row.get("value_type")):
                    value_types[fw] = str(row["value_type"])
                if pd.notna(row.get("value_default")):
                    value_defaults[fw] = str(row["value_default"])

        result[canonical] = {
            "description": best_desc.strip(),
            "mappings": mappings,
            "value_types": value_types,
            "value_defaults": value_defaults,
        }

    return result
