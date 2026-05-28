"""Command-line orchestration package for ProbeFlow.

Architectural role
------------------
The CLI is an orchestration layer.  Commands compose the production
domain stack — parsers in :mod:`probeflow.io`, the :class:`Scan` /
:class:`ROI` model in :mod:`probeflow.core`, the
:func:`apply_processing_state` pipeline in :mod:`probeflow.processing`,
measurement helpers in :mod:`probeflow.measurements` /
:mod:`probeflow.analysis`, and the
:class:`probeflow.provenance.records.ProcessingHistory` /
:class:`ExportRecord` sidecar machinery in :mod:`probeflow.provenance` —
without owning those concepts themselves.

Future cleanup
--------------
The current command implementation is parked in :mod:`probeflow.cli._legacy`
for compatibility with existing private imports.  Continue moving command
runners into ``commands/`` and shared processing wrappers into
``processing_ops.py``.

Do not add :class:`Scan` / spectrum model definitions, provenance records,
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
