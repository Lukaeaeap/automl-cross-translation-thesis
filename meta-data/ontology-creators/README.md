# Ontology Creators

One Python script per AutoML framework. Each script introspects the installed
framework and writes a `*_ontology_mapping.csv` to `meta-data/ontologies/`.

## How to regenerate all CSVs from scratch

```bash
cd <repo-root>
# Activate whichever venv has all frameworks installed
source .venv/bin/activate   # or create a fresh one

pip install h2o flaml auto-sklearn autogluon tpot lightgbm xgboost scikit-learn
# auto-sklearn is Linux/WSL-only and needs an older scikit-learn pin - see
# requirements-autosklearn.txt and setup_env.sh for the
# --no-deps workaround used to install it alongside the newer scikit-learn.

python meta-data/ontology-creators/extract_tpot.py
python meta-data/ontology-creators/extract_autosklearn.py
python meta-data/ontology-creators/extract_flaml.py
python meta-data/ontology-creators/extract_autogluon.py
python meta-data/ontology-creators/extract_h2o.py
# extract_h2o.py starts a local H2O Java cluster and shuts it down when done -
# requires Java 17+ on PATH.

# Every extract_*.py accepts --out <path> to write the CSV somewhere other
# than the default meta-data/ontologies/ location.

# Rebuild the canonical ontology from the new CSVs
python translation-engine/ontology/build.py
```

## Design principles

- No manual edits to CSVs - all content is extracted programmatically.
- Manual overrides live in `translation-engine/ontology/`:
  - `aliases.json`      - model name aliases (class renames across versions)
  - `params_patch.json` - param additions missed by NLP clustering
