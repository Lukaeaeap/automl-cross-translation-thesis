"""
BaseTranslator handles ontology lookups.
It refers to each frameworks translator subclass for speciifc overriding rules.
"""

from typing import Callable, Dict, Any, Optional, Tuple, Union

# A mapping is either a rename ("fw_name") or a (fw_name, value_transform)
TransformEntry = Union[str, Tuple[str, Optional[Callable[[Any], Any]]]]

class BaseTranslator:
    """
    Allows translations in two directions:

    to_canonical_params(model, params) = framework dialect -> canonical
    from_canonical_params(model, params) = canonical -> framework dialect

    
    Subclasses override:
        RECEIVE_TRANSFORMS {fw_param: canonical_param}
        EMIT_TRANSFORMS {canonical_param: fw_param}
        VALUE_CONSTRAINTS {fw_param: value_transform} 
    """

    FRAMEWORK: str = ""

    # fw_param -> canonical_param
    RECEIVE_TRANSFORMS: Dict[str, TransformEntry] = {}

    # canonical_param -> fw_param
    EMIT_TRANSFORMS: Dict[str, TransformEntry] = {}

    # fw_param -> value_transform, applied last, after name resolution
    VALUE_CONSTRAINTS: Dict[str, Callable[[Any], Any]] = {}

    def resolve(self, entry: TransformEntry, value: Any) -> Tuple[str, Any]:
        if isinstance(entry, tuple):
            name, transform = entry
            return name, (transform(value) if transform else value)
        return entry, value

    def constrain(self, fw_param: str, value: Any) -> Any:
        constraint = self.VALUE_CONSTRAINTS.get(fw_param)
        return constraint(value) if constraint else value

    def constrain_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Force constrains on parameters
        return {k: self.constrain(k, v) for k, v in params.items()}

    def to_canonical_params(
        self,
        params: Dict[str, Any],
        param_ontology: Dict[str, Dict],
    ) -> Dict[str, Any]:
        # Translate framework specific parameter names to canonical parameter names.
        result: Dict[str, Any] = {}

        # Build reverse map from param ontology: fw_param_name -> canonical_param
        reverse: Dict[str, str] = {}
        for canonical, info in param_ontology.items():
            fw_name = info.get("mappings", {}).get(self.FRAMEWORK)
            if fw_name:
                reverse[fw_name] = canonical

        # Apply the receive remapping, do ontology lookup and store unknown parameters for debugging
        for fw_param, value in params.items():
            value = self.constrain(fw_param, value)
            if fw_param in self.RECEIVE_TRANSFORMS:
                name, val = self.resolve(self.RECEIVE_TRANSFORMS[fw_param], value)
                result[name] = val

            elif fw_param in reverse:
                result[reverse[fw_param]] = value

            else:
                result[f"__{self.FRAMEWORK}__{fw_param}"] = value

        return result


    def from_canonical_params(
        self,
        params: Dict[str, Any],
        param_ontology: Dict[str, Dict],
    ) -> Dict[str, Any]:
        # Translate canonical parameter names to framework specific parameter names .

        result: Dict[str, Any] = {}
        for canonical_param, value in params.items():
            # Skip parameters that are unknown
            if canonical_param.startswith("__"):
                continue

            if canonical_param in self.EMIT_TRANSFORMS:
                fw_param, val = self.resolve(self.EMIT_TRANSFORMS[canonical_param], value)

            else:
                # Ontology lookup
                ontology_fw_param = (
                    param_ontology.get(canonical_param, {})
                    .get("mappings", {})
                    .get(self.FRAMEWORK)
                )
                if ontology_fw_param:
                    fw_param, val = ontology_fw_param, value
                else:
                    continue

            result[fw_param] = self.constrain(fw_param, val)

        return result
