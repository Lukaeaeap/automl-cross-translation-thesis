"""
Extracts H2O hyperparameter ontology from the installed h2o-3 framework using its REST API.

H2O requires a running Java process.

Output: meta-data/ontologies/h2o_ontology_mapping.csv

Run from the repo root:
    python meta-data/ontology-creators/extract_h2o.py
"""

import importlib
import shutil
import socket
import subprocess
import time
from pathlib import Path

import extraction_utils
import h2o

OUT = extraction_utils.out_path("h2o_ontology_mapping.csv")

H2O_ESTIMATORS = [
    ("H2OANOVAGLMEstimator", "h2o.estimators.anovaglm"),
    ("H2OAdaBoostEstimator", "h2o.estimators.adaboost"),
    ("H2OAggregatorEstimator", "h2o.estimators.aggregator"),
    ("H2OAutoEncoderEstimator", "h2o.estimators.deeplearning"),
    ("H2OCoxProportionalHazardsEstimator", "h2o.estimators.coxph"),
    ("H2ODecisionTreeEstimator", "h2o.estimators.decision_tree"),
    ("H2ODeepLearningEstimator", "h2o.estimators.deeplearning"),
    ("H2OExtendedIsolationForestEstimator", "h2o.estimators.extended_isolation_forest"),
    ("H2OGeneralizedAdditiveEstimator", "h2o.estimators.gam"),
    ("H2OGeneralizedLinearEstimator", "h2o.estimators.glm"),
    ("H2OGeneralizedLowRankEstimator", "h2o.estimators.glrm"),
    ("H2OGenericEstimator", "h2o.estimators.generic"),
    ("H2OGradientBoostingEstimator", "h2o.estimators.gbm"),
    ("H2OHGLMEstimator", "h2o.estimators.hglm"),
    ("H2OInfogram", "h2o.estimators.infogram"),
    ("H2OIsolationForestEstimator", "h2o.estimators.isolation_forest"),
    ("H2OIsotonicRegressionEstimator", "h2o.estimators.isotonicregression"),
    ("H2OKMeansEstimator", "h2o.estimators.kmeans"),
    ("H2OModelSelectionEstimator", "h2o.estimators.model_selection"),
    ("H2ONaiveBayesEstimator", "h2o.estimators.naive_bayes"),
    ("H2OPrincipalComponentAnalysisEstimator", "h2o.estimators.pca"),
    ("H2ORandomForestEstimator", "h2o.estimators.random_forest"),
    ("H2ORuleFitEstimator", "h2o.estimators.rulefit"),
    ("H2OSingularValueDecompositionEstimator", "h2o.estimators.svd"),
    ("H2OSupportVectorMachineEstimator", "h2o.estimators.psvm"),
    ("H2OTargetEncoderEstimator", "h2o.estimators.targetencoder"),
    ("H2OUpliftRandomForestEstimator", "h2o.estimators.uplift_random_forest"),
    ("H2OWord2vecEstimator", "h2o.estimators.word2vec"),
    ("H2OXGBoostEstimator", "h2o.estimators.xgboost"),
]


def type_str(meta):
    # Convert H2O parameter metadata to a type string
    h2o_type = str(meta.get("type", "")).lower()
    values = meta.get("values", [])
    if values:
        return f"Categorical({values})"
    if "int" in h2o_type:
        return "int"
    if "float" in h2o_type or "double" in h2o_type:
        return "float"
    if "bool" in h2o_type:
        return "bool"
    if h2o_type:
        return h2o_type
    
    return "unknown"


def start_h2o_cluster(port=54321, timeout=60):
    # Start a local, single-node H2O cluster..

    # Ensure the right java version is installed
    java = shutil.which("java")
    if java is None:
        raise RuntimeError("java not found on your system PATH, H2O requires Java version of 17+")

    temp_folder = Path(f"h2o_temp_port_{port}")
    temp_folder.mkdir(exist_ok=True)

    flatfile = temp_folder / "flatfile.txt"
    flatfile.write_text(f"127.0.0.1:{port}\n")

    # Locate h2o.jar file
    h2o_folder = Path(h2o.__file__).resolve().parent
    jar_path = h2o_folder / "backend" / "bin" / "h2o.jar"

    # Launch java process
    proc = subprocess.Popen(
        [
            java,
            "-ea",
            "-jar",
            str(jar_path),
            "-ip",
            "127.0.0.1",
            "-web_ip",
            "127.0.0.1",
            "-baseport",
            str(port),
            "-ice_root",
            str(temp_folder),
            "-name",
            "h2o_ontology_extract",
            "-log_level",
            "WARN",
            "-allow_unsupported_java",
            "-flatfile",
            str(flatfile),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server to start listening to port
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"H2O process fail to start, gave code {proc.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(1)
    else:
        proc.terminate()
        raise RuntimeError(f"H2O cluster took longer then {timeout}s to start")

    # Connect python to running cluster
    h2o.init(ip="127.0.0.1", port=port, start_h2o=False, verbose=False)
    return proc


def extract_estimator(cls_name, module_path):
    # H2O exposes each algorithm's full parameter schema trough API (type, default, and a help text)
    rows = []
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        algo = cls.algo

        resp = h2o.api(f"GET /3/ModelBuilders/{algo}")
        params = resp["model_builders"][algo]["parameters"]

        # Process hyperparameters
        for metadata in params:
            param = metadata.get("name")
            if not param:
                continue
            rows.append(
                {
                    "framework": "h2o",
                    "family": cls_name.replace("H2O", "").replace("Estimator", "").lower(),
                    "class": cls_name,
                    "parameter_name": param,
                    "value_default": str(metadata.get("default_value", "")),
                    "value_type": type_str(metadata),
                    "description": str(metadata.get("help", "")).replace("\n", " ").strip(),
                }
            )
    except Exception as e:
        print(f"Exception for {cls_name}: {e}")

    return rows


def main(out=None):
    print("Starting H2O cluster...")
    process = start_h2o_cluster()

    try:
        rows = []
        for cls_name, module_path in H2O_ESTIMATORS:
            model_rows = extract_estimator(cls_name, module_path)
            print(f"{cls_name}: {len(model_rows)} params")
            rows.extend(model_rows)
    finally:
        print("Shutting down H2O cluster...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    out = Path(out) if out else OUT
    extraction_utils.write_csv(out, rows)
    print(f"Wrote all {len(rows)} rows to: {out}")


if __name__ == "__main__":
    main(out=extraction_utils.cli_out_arg())
