"""Minimal plugin-facing types — **experimental, not wired to any dispatch**.

Architectural role
------------------
Plugin operations map directly to the intended operation classes: parser,
transformation, measurement, and writer.  See
:mod:`probeflow.plugins` for the broader status note — these dataclasses
are kept as a stable shape for a future extension layer; no in-tree
caller dispatches through them today.

Boundary rules
--------------
Keep the API small.  Do not migrate existing processing functions into
plugins as part of unrelated cleanup, and do not define provenance
dataclasses or GUI/CLI behaviour here; this module describes operations
and callables, not their presentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

OperationKind = Literal["parser", "transformation", "measurement", "writer"]


@dataclass(frozen=True)
class PluginOperation:
    """Description of one operation supplied by a plugin."""

    name: str
    kind: OperationKind
    version: str
    function: Callable[..., Any]
    input_types: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginSpec:
    """Description of a ProbeFlow plugin package."""

    name: str
    version: str
    operations: tuple[PluginOperation, ...] = ()
