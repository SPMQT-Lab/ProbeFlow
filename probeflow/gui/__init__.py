"""PySide6 interface package for ProbeFlow.

Architectural role
------------------
The GUI is an interface layer over the production domain stack — parser
results from :mod:`probeflow.io`, the :class:`Scan` / :class:`ROI` model
in :mod:`probeflow.core`, the :func:`apply_processing_state` pipeline in
:mod:`probeflow.processing`, measurement helpers in
:mod:`probeflow.measurements`, and the
:class:`probeflow.provenance.records.ProcessingHistory` /
:class:`ExportRecord` machinery in :mod:`probeflow.provenance`.  GUI code
should present those results, not own them.

Future cleanup
--------------
A small backward-compatibility shim lives in ``probeflow.gui.compat``
(formerly ``_legacy.py``) so public ``from probeflow.gui import X`` imports
remain stable while widgets and dialogs are transplanted into ``browse/``,
``viewer/``, ``convert/``, ``features/``, ``terminal/``, and ``dialogs/``.
Keep GUI code Qt-facing only: do not add provenance dataclasses, numerical
kernels, measurement algorithms, readers, or writers here.

Dialog / dock keep-alive convention
-----------------------------------
- Non-modal top-level viewers spawned from ``ProbeFlowWindow`` are tracked in
  ``self._open_viewers`` (a strong-ref set reaped by Qt's ``destroyed`` signal,
  see :class:`probeflow.gui.app.ProbeFlowWindow`).
- Per-viewer modeless children spawned from an ``ImageViewerDialog`` are
  tracked in ``self._modeless_children`` (registry maintained by
  ``_track_modeless_child`` / ``_close_modeless_children`` and reaped via
  ``destroyed``).
- Docks docked into a main window are stored as ``self._<name>_dock``
  (e.g. ``self._roi_dock``).

New dialog / dock code should follow one of these patterns rather than
introducing a fresh ad-hoc keep-alive attribute.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any
import sys

from probeflow.gui import models, rendering
from probeflow.gui.config import (
    CONFIG_PATH,
    DEFAULT_CUSHION,
    LOGO_PATH,
    LOGO_GIF_PATH,
    LOGO_NAV_PATH,
    GITHUB_URL,
    GUI_FONT_SIZES,
    GUI_FONT_DEFAULT,
    normalise_gui_font_size,
    load_config,
    save_config,
)
from probeflow.gui.styling import (
    THEMES,
    _sep,
    _build_qss,
)
from probeflow.gui.utils import _open_url, _format_scan_conditions
from probeflow.gui.models import (
    PLANE_NAMES,
    SxmFile,
    VertFile,
    _card_meta_str,
    _scan_items_to_sxm,
    _spec_items_to_vert,
    scan_image_folder,
    scan_vert_folder,
)
from probeflow.gui.rendering import (
    CMAP_KEY,
    CMAP_NAMES,
    DEFAULT_CMAP_KEY,
    DEFAULT_CMAP_LABEL,
    STM_COLORMAPS,
    THUMBNAIL_CHANNEL_DEFAULT,
    THUMBNAIL_CHANNEL_OPTIONS,
    _apply_processing,
    clip_range_from_arr,
    render_scan_image,
    render_scan_thumbnail,
    render_spec_thumbnail,
    render_with_processing,
    resolve_thumbnail_plane_index,
)

_LEGACY_EXPORTS = {
    "AboutDialog",
    "BrowseInfoPanel",
    "BrowseToolPanel",
    "ConvertPanel",
    "ConvertSidebar",
    "DeveloperTerminalWidget",
    "EdgeDetectionDialog",
    "FFTViewerDialog",
    "ImageViewerDialog",
    "PeriodicFilterDialog",
    "ProbeFlowWindow",
    "ProcessingControlPanel",
    "SpecMappingDialog",
    "SpecOverlayDialog",
    "SpecViewerDialog",
    "STMBackgroundDialog",
    "ThumbnailGrid",
    "_DefinitionsDialog",
    "_DevSidebar",
    "_TerminalPane",
}

# Names that have been fully extracted to probeflow.gui.dialogs.
# Exposed here for backward compatibility without loading compat.
_DIALOGS_EXPORTS = {
    "ViewerSpecMappingDialog",
    "_DEFINITIONS_HTML",
    "_DefinitionsPanel",
}


def _load_compat():
    existing = {
        name: value
        for name, value in globals().items()
        if not (name.startswith("__") and name.endswith("__"))
        and name not in {"Any", "ModuleType", "annotations", "import_module", "main", "models", "rendering", "sys"}
    }
    compat = import_module("probeflow.gui.compat")
    for name, value in existing.items():
        if hasattr(compat, name) and getattr(compat, name) is not value:
            setattr(compat, name, value)
    globals().update({
        name: value
        for name, value in vars(compat).items()
        if not (name.startswith("__") and name.endswith("__"))
        and name not in {"main"}
    })
    return compat


def __getattr__(name: str) -> Any:
    if name in _DIALOGS_EXPORTS:
        from probeflow.gui import dialogs as _dialogs
        return getattr(_dialogs, name)
    if name in _LEGACY_EXPORTS:
        return getattr(_load_compat(), name)
    raise AttributeError(f"module 'probeflow.gui' has no attribute {name!r}")


def main(*, open_survey: Any = None, browse_folder: Any = None) -> None:
    """Start the Qt GUI, importing PySide6 only when the GUI is launched.

    When ``open_survey`` is set to a path, ProbeFlow boots straight into
    Survey mode with that ScanFlow manifest loaded.
    When ``browse_folder`` is set, the Browse tab opens that folder on startup.
    """

    _load_compat().main(open_survey=open_survey, browse_folder=browse_folder)


class _GuiCompatModule(ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        compat = sys.modules.get("probeflow.gui.compat")
        if compat is not None and hasattr(compat, name):
            setattr(compat, name, value)


sys.modules[__name__].__class__ = _GuiCompatModule


__all__ = [
    "AboutDialog",
    "BrowseInfoPanel",
    "BrowseToolPanel",
    "CMAP_KEY",
    "CMAP_NAMES",
    "CONFIG_PATH",
    "ConvertPanel",
    "ConvertSidebar",
    "DEFAULT_CMAP_KEY",
    "DEFAULT_CMAP_LABEL",
    "DEFAULT_CUSHION",
    "DeveloperTerminalWidget",
    "EdgeDetectionDialog",
    "FFTViewerDialog",
    "GUI_FONT_DEFAULT",
    "GUI_FONT_SIZES",
    "ImageViewerDialog",
    "PLANE_NAMES",
    "PeriodicFilterDialog",
    "ProbeFlowWindow",
    "ProcessingControlPanel",
    "STM_COLORMAPS",
    "SpecMappingDialog",
    "SpecOverlayDialog",
    "SpecViewerDialog",
    "SxmFile",
    "STMBackgroundDialog",
    "THEMES",
    "THUMBNAIL_CHANNEL_DEFAULT",
    "THUMBNAIL_CHANNEL_OPTIONS",
    "ThumbnailGrid",
    "VertFile",
    "ViewerSpecMappingDialog",
    "clip_range_from_arr",
    "load_config",
    "main",
    "models",
    "normalise_gui_font_size",
    "render_scan_image",
    "render_scan_thumbnail",
    "render_spec_thumbnail",
    "render_with_processing",
    "rendering",
    "resolve_thumbnail_plane_index",
    "save_config",
    "scan_image_folder",
    "scan_vert_folder",
]
