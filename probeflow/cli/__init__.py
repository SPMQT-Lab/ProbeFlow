"""Command-line orchestration package for ProbeFlow.

Architectural role
------------------
The CLI is an orchestration layer over the intended Session -> Probe ->
Scan/Spectrum -> ScanGraph architecture. Commands should call parser,
transformation, measurement, writer, provenance, and future plugin-registry
services without owning those concepts themselves.

Future cleanup
--------------
The current command implementation is parked in :mod:`probeflow.cli._legacy`
for compatibility with existing private imports. Continue moving command
runners into ``commands/`` and shared processing wrappers into
``processing_ops.py``. New commands should call canonical package APIs or the
future plugin registry rather than reaching into GUI modules.

Do not add ``Scan``/``Spectrum`` model definitions, graph node dataclasses,
numerical kernels, vendor parser logic, or GUI widgets here.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any
import sys

from probeflow.cli import _legacy as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})


def main(argv=None) -> int:
    """Run the ProbeFlow CLI."""
    return _impl.main(argv)


class _CliCompatModule(ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        legacy = sys.modules.get("probeflow.cli._legacy")
        if legacy is not None and hasattr(legacy, name):
            setattr(legacy, name, value)


sys.modules[__name__].__class__ = _CliCompatModule

__all__ = [name for name in globals() if not name.startswith("__")]
