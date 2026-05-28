"""Plugin registry foundation — **experimental, not wired to any caller**.

Status (2026-05-28, review arch-backend #7)
-------------------------------------------
``PluginRegistry`` / :class:`PluginSpec` / :class:`PluginOperation` are
implemented but no CLI command, GUI panel, or processing path currently
discovers operations from this registry — the in-tree processing kernels
are dispatched directly from
:func:`probeflow.processing.state.apply_processing_state`.  The module is
kept and tested so a future extension layer can build on a working
baseline; treat anything here as not-yet-shipped public surface.

Conceptual goal
---------------
Plugins should declare parser, transformation, measurement, and writer
operations with enough metadata for CLI, GUI, and provenance wrappers to
use the same registry — letting new operations be discovered once instead
of hand-wired into several command and panel files.

Boundary rules
--------------
Keep plugin metadata, registration, manifests, and adapters here.  Do not
move current processing kernels into plugins during cleanup, and do not
define provenance dataclasses here (those live in ``probeflow.provenance``).
"""

from probeflow.plugins.api import PluginOperation, PluginSpec
from probeflow.plugins.registry import PluginRegistry

__all__ = ["PluginOperation", "PluginSpec", "PluginRegistry"]
