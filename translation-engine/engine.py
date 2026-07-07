"""
TranslationEngine maps the AutoML configs between frameworks using the canonical ontology and override rules.

Usage:
engine = TranslationEngine()
result = engine.translate(
            config={"model": "XGBoostEstimator", "params": {"n_estimators": 100}},
            source="flaml", target="h2o",
            )
See example_translate.py for more examples.
"""

import ast
import importlib
import inspect
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).parent

from translators import REGISTRY

FRAMEWORKS = list(REGISTRY.keys())


class TranslationEngine:
    def __init__(self, ontology_dir: Optional[Path] = None):
        ontology_path = Path(ontology_dir) if ontology_dir else HERE / "ontology"
        if not (ontology_path / "models.json").exists():
            raise FileNotFoundError(
                f"Ontology not built yet. Run: python ontology/build.py\n"
                f"Expected: {ontology_path / 'models.json'}")
        self.models: Dict = json.loads((ontology_path / "models.json").read_text())
        self.params: Dict = json.loads((ontology_path / "params.json").read_text())
        self.apply_params_patch(ontology_path)
        self.translators = {fw: cls() for fw, cls in REGISTRY.items()}

        self.class_to_canonical: Dict[str, Dict[str, str]] = {}
        for canonical, fw_map in self.models.items():
            for fw, names in fw_map.items():
                self.class_to_canonical.setdefault(fw, {})
                for name in names:
                    self.class_to_canonical[fw][name] = canonical

    # Public API for translation

    def translate(
        self, config: Dict[str, Any], source: str, target: str
    ) -> Dict[str, Any]:
        self.check_framework(source)
        self.check_framework(target)

        model_name = config.get("model", "")
        params = config.get("params", {})

        canonical_model = self.resolve_model(source, model_name)
        if canonical_model is None:
            warnings.warn(
                f"[engine] '{model_name}' (source={source}) not found in model ontology.",
                stacklevel=2,
            )
            canonical_model = model_name.lower()

        param_ont = self.params.get(canonical_model, {})
        src_translator = self.translators[source]
        canonical_params = src_translator.to_canonical_params(params, param_ont)

        tgt_translator = self.translators[target]
        target_params = tgt_translator.from_canonical_params(
            canonical_params, param_ont
        )
        target_params = self.validate_params(target_params, param_ont, target)
        target_params = tgt_translator.constrain_params(target_params)
        target_model = self.resolve_target_model(canonical_model, target)
        target_params = self.filter_to_class_signature(
            target, target_model, target_params
        )

        # A canonical param is "unmapped" when it produced no target value, has no explicit ontology mapping for this target, 
        # and is also not handled with EMIT_TRANSFORMS
        unmapped = [
            cp
            for cp in canonical_params
            if not cp.startswith("__")
            and cp not in target_params.values()
            and param_ont.get(cp, {}).get("mappings", {}).get(target) is None
            and cp not in tgt_translator.EMIT_TRANSFORMS
        ]

        return {
            "canonical_model": canonical_model,
            "model": target_model,
            "params": target_params,
            "unmapped_params": unmapped,
        }

    def broadcast(
        self, config: Dict[str, Any], source: str, targets: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        targets = targets or [f for f in FRAMEWORKS if f != source]
        return {fw: self.translate(config, source=source, target=fw) for fw in targets}

    # Ontology inspection functions

    def canonical_models(self) -> List[str]:
        return sorted(self.models.keys())

    def frameworks_for(self, canonical_model: str) -> Dict[str, List[str]]:
        return self.models.get(canonical_model, {})

    def matrix(self) -> Dict[str, Dict[str, bool]]:
        return {
            canonical: {fw: bool(fw_map.get(fw)) for fw in FRAMEWORKS}
            for canonical, fw_map in self.models.items()
        }

    def coverage(self) -> Dict[str, Dict]:
        totals = {fw: 0 for fw in FRAMEWORKS}
        covered = {fw: 0 for fw in FRAMEWORKS}
        for model_params in self.params.values():
            for param_info in model_params.values():
                mappings = param_info.get("mappings", {})
                for fw in FRAMEWORKS:
                    totals[fw] += 1
                    if fw in mappings:
                        covered[fw] += 1
        return {
            fw: {
                "covered": covered[fw],
                "total": totals[fw],
                "pct": round(100 * covered[fw] / totals[fw], 1) if totals[fw] else 0,
            }
            for fw in FRAMEWORKS
        }

    # Internal helper functions

    def apply_params_patch(self, ontology_dir: Path):
        patch_path = ontology_dir / "params_patch.json"
        if not patch_path.exists():
            return
        patch = json.loads(patch_path.read_text())
        for canonical, param_overrides in patch.items():
            if canonical.startswith("_"):
                continue
            if canonical not in self.params:
                self.params[canonical] = {}
            for param_name, param_info in param_overrides.items():
                self.params[canonical][param_name] = param_info

    def check_framework(self, fw: str):
        if fw not in FRAMEWORKS:
            raise ValueError(f"Unknown framework '{fw}'. Known: {FRAMEWORKS}")

    def resolve_model(self, framework: str, class_name: str) -> Optional[str]:
        fw_map = self.class_to_canonical.get(framework, {})
        if class_name in fw_map:
            return fw_map[class_name]
        lower = class_name.lower()
        for name, canonical in fw_map.items():
            if name.lower() == lower:
                return canonical
        return None

    def resolve_target_model(self, canonical_model: str, target: str) -> str:
        names = self.models.get(canonical_model, {}).get(target, [])
        return names[0] if names else f"[{canonical_model}]"

    # Module search paths per framework, used to resolve a class name to its import path
    FW_MODULES = {
        "tpot": [
            "sklearn.ensemble",
            "sklearn.linear_model",
            "sklearn.tree",
            "sklearn.svm",
            "sklearn.naive_bayes",
            "sklearn.neighbors",
            "sklearn.neural_network",
            "sklearn.gaussian_process",
            "sklearn.discriminant_analysis",
            "xgboost",
            "lightgbm",
        ],
        "flaml": ["flaml.automl.model"],
        "autogluon": ["autogluon.tabular.models"],
        "h2o": ["h2o.estimators"],
        "autosklearn": [
            "autosklearn.pipeline.components.classification",
            "autosklearn.pipeline.components.regression",
            "sklearn.gaussian_process",
            "sklearn.discriminant_analysis",
            "sklearn.ensemble",
            "sklearn.linear_model",
            "sklearn.tree",
            "sklearn.svm",
            "sklearn.naive_bayes",
            "sklearn.neighbors",
            "sklearn.neural_network",
        ],
    }

    def filter_to_class_signature(
        self, target_fw: str, model_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Drop params that the target class __init__ doesn't accept.
        if model_name.startswith("["):
            return params
        for mod_path in self.FW_MODULES.get(target_fw, []):
            try:
                mod = importlib.import_module(mod_path)
                cls = getattr(mod, model_name, None)
                if cls is None:
                    continue
                sig = inspect.signature(cls.__init__)
                # If **kwargs is present, class accepts anything, don't filter for anything
                for p in sig.parameters.values():
                    if p.kind == inspect.Parameter.VAR_KEYWORD:
                        return params
                accepted = set(sig.parameters) - {"self"}
                return {k: v for k, v in params.items() if k in accepted}
            except Exception:
                continue
        return params

    def validate_params(
        self, target_params: Dict[str, Any], param_ont: Dict[str, Dict], target: str
    ) -> Dict[str, Any]:
        # Validate all hyperparameters and attempt to coerce to target type
        reverse_map = {}
        for c_param, info in param_ont.items():
            fw_name = info.get("mappings", {}).get(target)
            if fw_name:
                reverse_map[fw_name] = c_param

        result = {}
        for fw_param, value in target_params.items():
            if fw_param in reverse_map:
                c_param = reverse_map[fw_param]
                value_type = (
                    param_ont.get(c_param, {}).get("value_types", {}).get(target)
                )
                result[fw_param] = self.coerce(value, value_type)
            else:
                result[fw_param] = value
        return result

    def coerce(self, value: Any, vt: Optional[str]) -> Any:
        # Coerce value to the correct Python type for the target framework.

        if not vt:
            return value
        vt = str(vt).strip()
        try:
            if "UniformInteger" in vt or vt == "int":
                return int(float(value))
            if "UniformFloat" in vt or vt == "float":
                return float(value)
            if vt == "bool":
                return str(value).lower() not in ("false", "0", "none")
            if vt.startswith(("Integer(", "Int(")):
                return int(float(value))
            if vt.startswith(("Real(", "Float(")):
                return float(value)
            if vt.startswith("Categorical("):
                m = re.match(r"Categorical\s*\(\s*(\[.*\])\s*\)", vt)
                if m and value in ast.literal_eval(m.group(1)):
                    return value
                # Ignore invalid categories
                return None  
        except Exception:
            pass
        return value
 