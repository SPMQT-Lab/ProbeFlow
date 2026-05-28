"""PySide6 interface package for ProbeFlow.

Architectural role
------------------
The GUI is an interface layer over the intended Session -> Probe ->
Scan/Spectrum -> ScanGraph model. It should present parser results,
transformations, measurements, writer artifacts, and provenance graphs without
owning those domain concepts itself.

Future cleanup
--------------
The legacy main-window implementation still lives in
``probeflow.gui._legacy`` so public imports remain stable while widgets and
dialogs are transplanted into ``browse/``, ``viewer/``, ``convert/``,
``features/``, ``terminal/``, and ``dialogs/``. Keep GUI code Qt-facing only:
do not add graph node dataclasses, numerical kernels, measurement algorithms,
readers, or writers here.
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
    NAVBAR_DARK_BG,
    NAVBAR_LIGHT_BG,
    NAVBAR_H,
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
    "FFTViewerDialog",
    "ImageViewerDialog",
    "Navbar",
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
# Exposed here for backward compatibility without loading _legacy.
_DIALOGS_EXPORTS = {
    "ViewerSpecMappingDialog",
    "_DEFINITIONS_HTML",
    "_DefinitionsPanel",
}


def _load_legacy():
    existing = {
        name: value
        for name, value in globals().items()
        if not (name.startswith("__") and name.endswith("__"))
        and name not in {"Any", "ModuleType", "annotations", "import_module", "main", "models", "rendering", "sys"}
    }
    legacy = import_module("probeflow.gui._legacy")
    for name, value in existing.items():
        if hasattr(legacy, name) and getattr(legacy, name) is not value:
            setattr(legacy, name, value)
    globals().update({
        name: value
        for name, value in vars(legacy).items()
        if not (name.startswith("__") and name.endswith("__"))
        and name not in {"main"}
    })
    return legacy


def __getattr__(name: str) -> Any:
    if name in _DIALOGS_EXPORTS:
        from probeflow.gui import dialogs as _dialogs
        return getattr(_dialogs, name)
    if name in _LEGACY_EXPORTS:
        return getattr(_load_legacy(), name)
    raise AttributeError(f"module 'probeflow.gui' has no attribute {name!r}")


def main(*, open_survey: Any = None, browse_folder: Any = None) -> None:
    """Start the Qt GUI, importing PySide6 only when the GUI is launched.

    When ``open_survey`` is set to a path, ProbeFlow boots straight into
    Survey mode with that ScanFlow manifest loaded.
    When ``browse_folder`` is set, the Browse tab opens that folder on startup.
    """

    _load_legacy().main(open_survey=open_survey, browse_folder=browse_folder)


class _GuiCompatModule(ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        legacy = sys.modules.get("probeflow.gui._legacy")
        if legacy is not None and hasattr(legacy, name):
            setattr(legacy, name, value)


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
    "FFTViewerDialog",
    "GUI_FONT_DEFAULT",
    "GUI_FONT_SIZES",
    "ImageViewerDialog",
    "Navbar",
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
