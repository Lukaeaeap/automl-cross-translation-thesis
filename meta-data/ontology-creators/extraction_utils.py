"""
Helper functions for the ontology creation scripts which are the extract_*.py scripts.
"""

import argparse
import csv
import importlib
import inspect
import re
import textwrap
from pathlib import Path

from docstring_parser import parse as parse_docstring

FIELDS = [
    "framework",
    "family",
    "class",
    "parameter_name",
    "value_default",
    "value_type",
    "description",
]

def out_path(filename):
    # Output file path
    return Path(__file__).resolve().parent.parent / "ontologies" / filename


def attempt_import(module, attr):
    # Try to import a module from a library
    try:
        return getattr(importlib.import_module(module), attr)
    except Exception:
        return None


def normalize_numpydoc(doc):
    # Normalize different numpydocs to work in docstring_parser
    m = re.search(r"(?m)^(Parameters)\s*\n(-+)\s*\n", doc)
    if not m:
        return doc
    start = m.end()
    rest = doc[start:]
    end_m = re.search(r"(?m)^\S.*\n-+\s*\n", rest)
    section = rest[: end_m.start()] if end_m else rest
    tail = rest[end_m.start():] if end_m else ""
    section = section.lstrip("\n")
    section = textwrap.dedent(section)
    section = re.sub(r"\n{2,}(?=\s{0,4}\S+\s*:)", "\n", section)
    return doc[:start] + section + tail


def parse_numpy_params(cls):
    # Return {param: description} from either a NumPy or Google-style docstring, dependent on module.
    if cls is None:
        return {}
    for target in (cls, getattr(cls, "__init__", None)):
        try:
            doc = inspect.getdoc(target) or ""
        except Exception:
            continue
        if "Parameters" not in doc and "Args" not in doc:
            continue
        parsed = parse_docstring(normalize_numpydoc(doc))
        descriptions = {p.arg_name: p.description or "" for p in parsed.params}
        if descriptions:
            return descriptions
    return {}


def full_api_rows(
    framework, family, cls_name, underlying_cls, already_captured, lib_descs=None
):
    # Step 2: retrieve the constructor parameters that are not already captured, using docstring descriptions.
    if underlying_cls is None:
        return []
    if lib_descs is None:
        lib_descs = parse_numpy_params(underlying_cls)
    rows = []
    try:
        sig = inspect.signature(underlying_cls.__init__)
    except (ValueError, TypeError):
        return []
    for pn, param in sig.parameters.items():
        if pn in ("self", "args", "kwargs") or pn.startswith("__"):
            continue
        if pn in already_captured:
            continue
        desc = lib_descs.get(pn, "")
        if not desc:
            continue
        default = param.default
        if default is inspect.Parameter.empty:
            default = ""
        rows.append(
            {
                "framework": framework,
                "family": family,
                "class": cls_name,
                "parameter_name": pn,
                "value_default": str(default),
                "value_type": type(default).__name__ if default != "" else "",
                "description": desc,
            }
        )
    return rows


def dedup_rows(rows):
    # Deduplicate exact row duplicates, only keep truly different variations.
    seen: dict = {}
    deduped = []
    for r in rows:
        key = (r["class"], r["parameter_name"])
        sig = (r.get("value_type", ""), r.get("value_default", ""))
        if key not in seen:
            seen[key] = sig
            deduped.append(r)
        elif seen[key] != sig:
            deduped.append(r)
    return deduped


def write_csv(path, rows):
    # Write ontology to csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def cli_out_arg():
    # Get CLI argument for name of csv export
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    return ap.parse_args().out

