# AutoML Configuration Translation Engine

Translates model configurations (class name + hyperparameters) between five AutoML frameworks: H2O, auto-sklearn, TPOT, FLAML, AutoGluon. A canonical ontology maps model names and hyperparameter names across frameworks, so a config produced by one framework can be converted to run in another.

Three experiments measure how well this works:
- **E1 Translatability** - can a config be translated to a real target model at all?
- **E2 Ontology Coverage** - how many hyperparameters/models does the ontology actually cover?
- **E3 Runtime Success** - does the translated config actually `.fit()` without error?

## Install

```bash
bash setup_env.sh
source .venv/bin/activate
```

Requires Python 3.8+, Java 17+ (for H2O), and Linux/WSL (auto-sklearn is Linux-only).

## Build the ontology

```bash
cd translation-engine
python ontology/build.py
```

Reads the CSVs in `meta-data/ontologies/` and writes `models.json`/`params.json` etc. into `translation-engine/ontology/`.

## Run the translator

```bash
python example_translate.py
```

Or in code:

```python
from engine import TranslationEngine
engine = TranslationEngine()
result = engine.translate(config={"model": "...", "params": {...}}, source="flaml", target="h2o")
```

## Run the experiments

From `translation-engine/`:

```bash
python experiments/experiment_translatability.py   # E1
python experiments/experiment_coverage.py           # E2
python experiments/experiment_fit_success.py        # E3, slow - calls .fit()
python experiments/generate_visuals.py              # builds CSV tables from the results above
```

Each run writes a timestamp-prefixed CSV to `experiments/results/` so older runs are never overwritten.
