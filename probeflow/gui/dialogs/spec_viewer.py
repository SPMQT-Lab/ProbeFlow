"""Compatibility exports for spectroscopy viewer dialogs.

The implementation lives in :mod:`probeflow.gui.spec_viewer`. Keep this module
for existing callers that import ``probeflow.gui.dialogs.spec_viewer``.
"""

from __future__ import annotations

from probeflow.gui.spec_viewer import SpecOverlayDialog, SpecViewerDialog

SpecViewerDialog.__module__ = __name__
SpecOverlayDialog.__module__ = __name__

__all__ = ["SpecViewerDialog", "SpecOverlayDialog"]
