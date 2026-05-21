"""Spectroscopy viewer dialogs."""

from __future__ import annotations

from .overlay import SpecOverlayDialog
from .single import SpecViewerDialog

_PUBLIC_MODULE = "probeflow.gui.dialogs.spec_viewer"
SpecViewerDialog.__module__ = _PUBLIC_MODULE
SpecOverlayDialog.__module__ = _PUBLIC_MODULE

__all__ = ["SpecViewerDialog", "SpecOverlayDialog"]
