"""Plugin manifest serialisation helpers.

Status: scaffolding — not yet wired.  ``PluginRegistry`` and ``PluginSpec``
exist but in-tree operations are not discovered from the registry at runtime.
This module will be called once the registry is wired into the CLI/GUI loader.

Manifests describe operation metadata for parser, transformation, measurement,
and writer plugins so the CLI, GUI, and provenance layer can share one source
of truth.  Keep manifest serialisation here; do not add operation
implementations, graph node dataclasses, or UI behaviour.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from probeflow.plugins.api import PluginOperation, PluginSpec


def manifest_from_spec(spec: PluginSpec) -> dict[str, Any]:
    """Return a JSON-compatible manifest dictionary for a plugin spec."""

    return {
        "name": spec.name,
        "version": spec.version,
        "operations": [_operation_to_manifest(op) for op in spec.operations],
    }


def _operation_to_manifest(op: PluginOperation) -> dict[str, Any]:
    return {
        "name": op.name,
        "kind": op.kind,
        "version": op.version,
        "function": _callable_ref(op.function),
        "input_types": list(op.input_types),
        "output_types": list(op.output_types),
        "parameters": _jsonable(op.parameters),
    }


def _callable_ref(fn) -> str:
    module = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
    return f"{module}:{qualname}" if module else str(qualname)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return _callable_ref(value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)
