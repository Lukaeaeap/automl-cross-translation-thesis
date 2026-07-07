import json
from pathlib import Path

HERE = Path(__file__).parent


def load_models() -> dict:
    with open(HERE / "models.json") as f:
        return json.load(f)


def load_params() -> dict:
    with open(HERE / "params.json") as f:
        return json.load(f)


def load_matrix() -> dict:
    # Load matrix with model existence per framework
    with open(HERE / "models_matrix.json") as f:
        return json.load(f)


def load_dependencies() -> dict:
    # Load dependencies for models that require extra packages.
    path = HERE / "model_dependencies.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

