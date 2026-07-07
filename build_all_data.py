"""
Regenerates all framework ontology CSVs and rebuild the canonical JSON.

Outputs: 
/meta-data/ontologies/*.csv
/ontology/models_matrix.json
/ontology/models.json
/ontology/params.json
/ontology/model_dependencies.json

Run from root:
    python build_all_data.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

def run(script: str):
    subprocess.run([sys.executable, script], cwd=ROOT, check=True)


run("meta-data/ontology-creators/extract_tpot.py")
run("meta-data/ontology-creators/extract_autosklearn.py")
run("meta-data/ontology-creators/extract_flaml.py")
run("meta-data/ontology-creators/extract_autogluon.py")
run("meta-data/ontology-creators/extract_h2o.py")
run("translation-engine/ontology/build.py")

print("Ontology Build Complete")
