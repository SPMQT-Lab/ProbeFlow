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

from dataclasses import asdict
from typing import Any

from probeflow.plugins.api import PluginSpec


def manifest_from_spec(spec: PluginSpec) -> dict[str, Any]:
    """Return a JSON-compatible manifest dictionary for a plugin spec."""

    return asdict(spec)
