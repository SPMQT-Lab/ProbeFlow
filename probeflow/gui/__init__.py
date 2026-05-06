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
    "CONFIG_PATH",
    "ConvertPanel",
    "ConvertSidebar",
    "DEFAULT_CUSHION",
    "DeveloperTerminalWidget",
    "FFTViewerDialog",
    "GUI_FONT_DEFAULT",
    "GUI_FONT_SIZES",
    "ImageViewerDialog",
    "Navbar",
    "PeriodicFilterDialog",
    "ProbeFlowWindow",
    "ProcessingControlPanel",
    "SpecMappingDialog",
    "SpecViewerDialog",
    "STMBackgroundDialog",
    "THEMES",
    "ThumbnailGrid",
    "ViewerSpecMappingDialog",
    "_DefinitionsPanel",
    "_DevSidebar",
    "_TerminalPane",
    "_build_qss",
    "_open_url",
    "_sep",
    "load_config",
    "normalise_gui_font_size",
    "save_config",
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
    if name in _LEGACY_EXPORTS:
        return getattr(_load_legacy(), name)
    raise AttributeError(f"module 'probeflow.gui' has no attribute {name!r}")


def main() -> None:
    """Start the Qt GUI, importing PySide6 only when the GUI is launched."""

    _load_legacy().main()


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
