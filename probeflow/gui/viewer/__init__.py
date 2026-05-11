"""Viewer widgets and helper modules."""

from probeflow.gui.viewer.bad_line_preview import BadLinePreviewController
from probeflow.gui.viewer.channel_util import resolve_channel_unit
from probeflow.gui.viewer.deferred_action import DeferredPlaneAction
from probeflow.gui.viewer.display_sliders import DisplaySliderController
from probeflow.gui.viewer.geometric_ops import transform_roi_set_for_display_op
from probeflow.gui.viewer.histogram_export import export_histogram
from probeflow.gui.viewer.line_profile_export import export_line_profile
from probeflow.gui.viewer.png_export import save_viewer_png
from probeflow.gui.viewer.processing_undo import ProcessingUndoController
from probeflow.gui.viewer.roi_analysis import (
    plot_roi_line_profile,
    show_roi_fft,
    show_roi_histogram,
)
from probeflow.gui.viewer.roi_ops import (
    activate_roi,
    active_roi,
    active_roi_id,
    delete_active_roi,
    delete_roi,
    has_roi_aware_local_filter,
    invert_active_roi,
    invert_roi,
    rename_roi,
    roi_canvas_created,
    roi_canvas_moved,
    roi_line_endpoint_changed,
    roi_line_set_width,
    select_nth_roi,
    selected_or_active_roi_id,
)
from probeflow.gui.viewer.roi_sidecar import load_roi_set, save_roi_set
from probeflow.gui.viewer.set_zero_plane import SetZeroPlaneController
from probeflow.gui.viewer.spec_overlay import SpecOverlayController

__all__ = [
    "BadLinePreviewController",
    "DeferredPlaneAction",
    "DisplaySliderController",
    "ProcessingUndoController",
    "SetZeroPlaneController",
    "SpecOverlayController",
    "activate_roi",
    "active_roi",
    "active_roi_id",
    "delete_active_roi",
    "delete_roi",
    "export_histogram",
    "export_line_profile",
    "has_roi_aware_local_filter",
    "invert_active_roi",
    "invert_roi",
    "load_roi_set",
    "plot_roi_line_profile",
    "rename_roi",
    "resolve_channel_unit",
    "roi_canvas_created",
    "roi_canvas_moved",
    "roi_line_endpoint_changed",
    "roi_line_set_width",
    "save_roi_set",
    "save_viewer_png",
    "select_nth_roi",
    "selected_or_active_roi_id",
    "show_roi_fft",
    "show_roi_histogram",
    "transform_roi_set_for_display_op",
]
