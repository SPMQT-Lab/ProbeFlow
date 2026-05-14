"""Viewer widgets and helper modules.

Public names are resolved lazily so importing one viewer submodule does not
eagerly import every Qt-backed helper in this package.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, str] = {
    "BadLinePreviewController": "bad_line_preview",
    "DeferredPlaneAction": "deferred_action",
    "DisplaySliderController": "display_sliders",
    "ImageMeasurementController": "image_measurements",
    "ProcessingUndoController": "processing_undo",
    "SetZeroPlaneController": "set_zero_plane",
    "SpecOverlayController": "spec_overlay",
    "activate_roi": "roi_ops",
    "active_roi": "roi_ops",
    "active_roi_id": "roi_ops",
    "delete_active_roi": "roi_ops",
    "delete_roi": "roi_ops",
    "export_histogram": "histogram_export",
    "export_line_profile": "line_profile_export",
    "has_roi_aware_local_filter": "roi_ops",
    "invert_active_roi": "roi_ops",
    "invert_roi": "roi_ops",
    "load_roi_set": "roi_sidecar",
    "plot_roi_line_profile": "roi_analysis",
    "rename_roi": "roi_ops",
    "resolve_channel_unit": "channel_util",
    "roi_canvas_created": "roi_ops",
    "roi_canvas_moved": "roi_ops",
    "roi_line_endpoint_changed": "roi_ops",
    "roi_line_set_width": "roi_ops",
    "save_roi_set": "roi_sidecar",
    "save_viewer_png": "png_export",
    "select_nth_roi": "roi_ops",
    "selected_or_active_roi_id": "roi_ops",
    "show_roi_fft": "roi_analysis",
    "show_roi_histogram": "roi_analysis",
    "transform_roi_set_for_display_op": "geometric_ops",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
