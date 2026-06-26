"""ImageViewerDialog — double-click viewer with scroll/zoom, histogram, processing, export.

Extracted from the historical probeflow.gui._legacy as part of the ongoing
GUI refactor (now re-exported via ``probeflow.gui.compat``).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

_log = logging.getLogger(__name__)

import os as _os
_os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import (
    Qt, QThreadPool,
    Signal,
)
from PySide6.QtGui import (
    QPixmap,
)
from PySide6.QtWidgets import (
    QDialog,
)

from probeflow.gui.utils import _format_scan_conditions
from probeflow.gui.models import PLANE_NAMES, SxmFile
from probeflow.gui.workers import ViewerLoader
from probeflow.gui.viewer.display_range import DisplayRangeController
from probeflow.gui.viewer import (
    BadLinePreviewController,
    DeferredPlaneAction,
    DisplaySliderController,
    ProcessingUndoController,
    SetZeroPlaneController,
    SpecOverlayController,
    resolve_channel_unit,
)
from probeflow.processing.gui_adapter import processing_state_from_gui
from probeflow.processing.state import (
    apply_processing_state_with_calibration,
    missing_roi_references,
)
from probeflow.provenance import (
    ProcessingHistory,
    append_processing_state,
    display_lines,
)
from probeflow.gui.roi_context import (
    area_roi_mask,
)
from probeflow.gui.viewer.scan_load import load_scan_for_viewer, ViewerScanData
from probeflow.gui.dialogs.image_viewer_build_mixin import ImageViewerBuildMixin
from probeflow.gui.dialogs.image_viewer_chrome_mixin import ImageViewerChromeMixin
from probeflow.gui.viewer.image_viewer_display_mixin import ImageViewerDisplayMixin
from probeflow.gui.viewer.image_viewer_processing_export_mixin import (
    ImageViewerProcessingExportMixin,
)
from probeflow.gui.viewer.image_viewer_mask_mixin import ImageViewerMaskMixin
from probeflow.gui.viewer.image_viewer_roi_mixin import ImageViewerRoiMixin
from probeflow.gui.viewer.image_viewer_selection_mixin import ImageViewerSelectionMixin
from probeflow.gui.viewer.image_viewer_toolbar_mixin import ImageViewerToolbarMixin
from probeflow.gui.viewer.image_viewer_tools_mixin import ImageViewerToolsMixin

# Dialogs imported from their specific submodule files to avoid circular imports
# (this module lives inside probeflow.gui.dialogs).
from probeflow.gui.dialogs.stm_background import STMBackgroundDialog


class ImageViewerDialog(
    ImageViewerBuildMixin,
    ImageViewerChromeMixin,
    ImageViewerMaskMixin,
    ImageViewerRoiMixin,
    ImageViewerSelectionMixin,
    ImageViewerToolbarMixin,
    ImageViewerDisplayMixin,
    ImageViewerToolsMixin,
    ImageViewerProcessingExportMixin,
    QDialog,
):
    """Double-click viewer with scroll/zoom, histogram display, processing, export."""

    # Emitted by "→ Feature Counting" / "→ TV Denoising" buttons so the
    # parent can act immediately without the viewer closing.
    immediate_action_requested = Signal(str)   # "features" | "tv"

    # Width of the collapsed-sidebar rail (px).
    _SIDEBAR_RAIL_W = 46

    def __init__(self, entry: SxmFile, entries: list[SxmFile],
                 colormap: str, t: dict, parent=None,
                 clip_low: float = 1.0, clip_high: float = 99.0,
                 processing: dict = None,
                 spec_image_map: Optional[dict] = None,
                 initial_plane_idx: int = 0):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.setWindowTitle(entry.stem)
        self.setMinimumSize(1100, 720)
        self.resize(1400, 860)
        self._show_maximized_on_start = False

        self._entries    = entries
        self._colormap   = colormap
        # Per-image colormap — independent of the global browser colormap.
        # Inherits the browser colormap at open time, but changes here don't
        # propagate back to thumbnails.
        self._viewer_colormap = colormap
        self._t          = t
        self._idx        = next((i for i, e in enumerate(entries) if e.stem == entry.stem), 0)
        self._pool       = QThreadPool.globalInstance()
        self._token      = object()
        self._clip_low   = clip_low
        self._clip_high  = clip_high
        self._drs        = DisplayRangeController(clip_low=clip_low, clip_high=clip_high, parent=self)
        # Per-region (per-area-ROI) display ranges for composite rendering.
        # ``_display_scope`` selects whether the contrast sliders edit the
        # global range or the active area ROI's own range.  ``_rois_hidden``
        # hides every ROI overlay so the composited image can be inspected.
        self._region_drs: dict[str, DisplayRangeController] = {}
        self._display_scope: str = "global"  # "global" | "roi"
        self._rois_hidden: bool = False
        self._processing = dict(processing) if processing else {}
        self._processing_roi_error: str = ""
        self._processing_error: str = ""
        # Undo / redo stacks for processing state. Each entry is a deep copy
        # of the full processing dict at a prior point. Apply / Reset push
        # the previous state onto _undo_stack and clear _redo_stack; the
        # Undo / Redo buttons swap between the two.
        self._proc_undo_btn = None
        self._proc_redo_btn = None
        # Mutable mapping shared with the parent window: spec_stem → image_stem.
        # Empty dict by default — markers only appear after explicit mapping.
        self._spec_image_map = spec_image_map if spec_image_map is not None else {}
        self._raw_arr: Optional[np.ndarray] = None
        self._display_arr: Optional[np.ndarray] = None  # raw or processed, for histogram/export
        self._source_processing_history: Optional[ProcessingHistory] = None
        self._processing_history: Optional[ProcessingHistory] = None
        self._last_export_record = None
        # Data range for sliders lives in self._hist_panel.data_min_si / data_max_si
        self._scan_header: dict = {}
        self._scan_range_m: Optional[tuple] = None
        # Post-processing scan_range_m; tracks shape-changing steps so the
        # scale bar stays calibrated against the displayed array shape
        # (review image-proc #4).  Updated by _refresh_display_array.
        self._display_scan_range_m: Optional[tuple] = None
        self._scan_shape: Optional[tuple] = None
        self._scan_format: str = ""
        self._scan_plane_names: list[str] = list(PLANE_NAMES)
        self._scan_plane_units: list[str] = ["m", "m", "A", "A"]
        # Modeless child dialogs (FeatureFinder, ImageInfo, PairCorrelation,
        # FeatureLattice, FFTViewer, PointFFT, LinePeriodicityPlot,
        # Threshold, STMBackground, …) — tracked here so closeEvent can
        # iterate-close them before the viewer tears down (review
        # gui-arch #22).  Without this their signal handlers can fire on
        # a partially-destroyed viewer (the f6aac4a class of bug).
        self._modeless_children: list[Any] = []
        # Controllers initialised inside _build() after their dependent widgets are created.
        self._spec_overlay: "SpecOverlayController | None" = None
        self._zero_ctrl: "SetZeroPlaneController | None" = None
        self._angle_overlay: "object | None" = None  # AngleOverlayItem, imported lazily
        self._angle_measurement_id: "str | None" = None  # measurement tied to overlay
        self._proc_undo_ctrl: "ProcessingUndoController | None" = None
        self._display_slider_ctrl: "DisplaySliderController | None" = None
        self._bad_line_preview_ctrl: "BadLinePreviewController | None" = None
        self._pending_initial_plane_idx: Optional[int] = max(0, int(initial_plane_idx))
        self._reset_zoom_on_next_pixmap = True
        self._deferred = DeferredPlaneAction()

        self._build()
        self._restore_viewer_desktop_layout()
        self._drs.rangeChanged.connect(self._refresh_display_range)
        self._processing_panel.set_state(self._processing)
        self._set_advanced_processing_state(self._processing)
        self._load_current()

    # ── Navigation ─────────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        k = event.key()

        # ── drawing tool shortcuts ────────────────────────────────────────────
        # Mnemonic keys: the letter matches the tool name.
        _tool_keys = {
            Qt.Key_R: "rectangle",
            Qt.Key_E: "ellipse",
            Qt.Key_L: "line",
            Qt.Key_P: "point",
            Qt.Key_G: "polygon",   # polyGon (P is taken by Point)
            Qt.Key_F: "freehand",
        }
        if k in _tool_keys and not event.modifiers():
            self._set_drawing_tool(_tool_keys[k])
            event.accept()
            return

        # ── Escape: cancel drawing, or close dialog if idle ───────────────────
        if k == Qt.Key_Escape:
            canvas_tool = getattr(self._zoom_lbl, "tool", lambda: "pan")()
            canvas_drawing = (canvas_tool != "pan" or
                              self._zoom_lbl._draw_pts or
                              self._zoom_lbl._draw_start is not None)
            if canvas_drawing:
                self._zoom_lbl.cancel_drawing()
                self._set_drawing_tool("pan")
                event.accept()
                return
            self.accept()
            return

        # ── ROI keyboard actions ──────────────────────────────────────────────
        if k == Qt.Key_Delete and not event.modifiers():
            self._delete_active_image_roi()
            event.accept()
            return

        if k == Qt.Key_I and not event.modifiers():
            self._invert_active_image_roi()
            event.accept()
            return

        if Qt.Key_1 <= k <= Qt.Key_9 and not event.modifiers():
            self._select_nth_image_roi(k - Qt.Key_0)
            event.accept()
            return

        # ── arrow keys: navigate between images ──────────────────────────────
        if k == Qt.Key_Left:
            self._go_prev()
        elif k == Qt.Key_Right:
            self._go_next()
        else:
            super().keyPressEvent(event)

    def _go_prev(self):
        if self._idx > 0:
            self._idx -= 1
            self._load_current(reset_zoom=True)

    def _go_next(self):
        if self._idx < len(self._entries) - 1:
            self._idx += 1
            self._load_current(reset_zoom=True)

    # ── Load / render ──────────────────────────────────────────────────────────
    def _load_current(self, reset_zoom: bool = True):
        entry = self._entries[self._idx]
        if hasattr(self, "_processing_panel"):
            self._clear_bad_line_preview()
        self._load_current_source(entry, reset_zoom=reset_zoom)
        self._refresh_display_array(reset_zoom_if_shape_changed=not reset_zoom)
        self._refresh_histogram_and_markers(entry)
        self._refresh_viewer_pixmap(reset_zoom=reset_zoom)
        self._sync_line_profile_visibility()

    def _load_current_source(self, entry: SxmFile, reset_zoom: bool = True):
        if hasattr(self, "_image_measurements"):
            self._image_measurements.clear_feature_points(silent=True)
        self._title_lbl.setText(entry.stem)
        self.setWindowTitle(entry.stem)
        self._conditions_lbl.setText(_format_scan_conditions(entry))
        self._pos_lbl.setText(f"{self._idx + 1} / {len(self._entries)}")
        self._prev_btn.setEnabled(self._idx > 0)
        self._next_btn.setEnabled(self._idx < len(self._entries) - 1)
        if reset_zoom:
            self._zoom_lbl.setText("Loading…")
            self._zoom_lbl.setPixmap(QPixmap())
        self._zoom_lbl.set_markers([])
        # Quick selections are per-image and ephemeral — drop on navigation.
        if hasattr(self, "_clear_quick_selection"):
            self._clear_quick_selection()
        self._roi_filter_scope_id = None
        self._load_image_roi_set(entry)
        self._load_image_mask_set(entry)
        if self._pending_initial_plane_idx is not None:
            target_ch = self._pending_initial_plane_idx
            self._pending_initial_plane_idx = None
        else:
            target_ch = self._ch_cb.currentIndex()
        data: ViewerScanData = load_scan_for_viewer(entry.path, target_ch)
        self._set_scan_channel_choices_from_names(data.plane_names, data.plane_units)
        clamped = max(0, min(target_ch, max(data.n_planes - 1, 0)))
        self._ch_cb.blockSignals(True)
        try:
            self._ch_cb.setCurrentIndex(clamped)
        finally:
            # An exception here must not leave the combo permanently muted —
            # that turns every later channel change into a silent no-op.
            self._ch_cb.blockSignals(False)
        self._raw_arr          = data.raw_arr
        self._scan_header      = data.scan_header
        self._scan_range_m     = data.scan_range_m
        # Updated by _refresh_display_array to reflect any shape-changing
        # processing step (review image-proc #4); used by _scan_extent_nm
        # so the scale bar and rulers stay calibrated against the display
        # array's actual physical extent.
        self._display_scan_range_m = data.scan_range_m
        self._scan_shape       = data.scan_shape
        self._scan_format      = data.source_format
        self._scan_plane_names = data.plane_names
        self._scan_plane_units = data.plane_units
        self._source_processing_history = data.processing_history
        self._rebuild_processing_history()
        # Readers degrade gracefully on partial/odd files (e.g. a scan still
        # being written yields only its complete planes) and record why on
        # the Scan; without this the user just sees missing channels with no
        # explanation (2026-06-12 parser review).
        if data.scan_warnings and hasattr(self, "_status_lbl"):
            extra = (
                f" (+{len(data.scan_warnings) - 1} more)"
                if len(data.scan_warnings) > 1 else ""
            )
            self._status_lbl.setText(
                f"Loaded with warnings: {data.scan_warnings[0]}{extra}"
            )

    def _rebuild_processing_history(self) -> None:
        if self._source_processing_history is None:
            self._processing_history = None
            self._sync_history_panel()
            return
        history = ProcessingHistory.from_dict(self._source_processing_history.to_dict())
        try:
            append_processing_state(
                history,
                processing_state_from_gui(self._processing or {}),
            )
        except Exception:
            _log.warning("Could not append current processing state to history "
                         "panel; panel may lag the actual pipeline", exc_info=True)
        self._processing_history = history
        self._sync_history_panel()

    def _sync_history_panel(self) -> None:
        if not hasattr(self, "_history_text"):
            return
        if self._processing_history is None:
            self._history_text.setText("Source: unknown\nChannel: unknown")
            return
        self._history_text.setText("\n".join(display_lines(self._processing_history)))

    def _processing_pixel_sizes_m(self) -> tuple[float | None, float | None]:
        """Return (pixel_size_x_m, pixel_size_y_m) derived from the loaded scan.

        Returns ``(None, None)`` when calibration data is missing.  Forwarded
        to :func:`probeflow.gui.rendering._apply_processing` so step-tolerance
        and facet-level operations interpret ``step_threshold_deg`` as a real
        surface slope (review image-proc #1).
        """
        scan_range = getattr(self, "_scan_range_m", None)
        shape = getattr(self, "_scan_shape", None)
        if scan_range is None or shape is None:
            return None, None
        try:
            w_m, h_m = float(scan_range[0]), float(scan_range[1])
            Ny, Nx = int(shape[0]), int(shape[1])
        except (TypeError, ValueError, IndexError):
            return None, None
        if Nx <= 0 or Ny <= 0 or w_m <= 0 or h_m <= 0:
            return None, None
        return w_m / Nx, h_m / Ny

    def _refresh_display_array(self, reset_zoom_if_shape_changed: bool = False):
        # Keep the Gaussian-blur σ readout calibrated to the loaded scan.
        panel = getattr(self, "_processing_panel", None)
        if panel is not None and hasattr(panel, "set_pixel_size_nm"):
            psx, _psy = self._processing_pixel_sizes_m()
            panel.set_pixel_size_nm(psx * 1e9 if psx else None)
        old_shape = self._display_arr.shape if self._display_arr is not None else None
        # display array: raw with processing applied (no grain overlay — that's visual only)
        if self._raw_arr is not None and self._processing:
            try:
                self._processing_roi_error = ""
                self._processing_error = ""
                state = processing_state_from_gui(self._processing or {})
                missing = missing_roi_references(
                    state, self._image_roi_set,
                    getattr(self, "_image_mask_set", None),
                )
                if missing:
                    refs = ", ".join(
                        f"{m['param']}={m['value']}" for m in missing[:3]
                    )
                    if len(missing) > 3:
                        refs += f", +{len(missing) - 3} more"
                    self._processing_roi_error = (
                        "Processing paused: missing ROI reference(s): " + refs
                    )
                    if hasattr(self, "_status_lbl"):
                        self._status_lbl.setText(self._processing_roi_error)
                    self._display_arr = self._raw_arr
                    self._display_scan_range_m = getattr(self, "_scan_range_m", None)
                    self._update_export_summary()
                    return
                # Forward calibration through the pipeline so:
                #   (a) step-tolerance / facet ops interpret step_threshold_deg
                #       as a real surface slope (review image-proc #1), and
                #   (b) scan_range_m grows with the canvas for canvas-expanding
                #       ops (rotate_arbitrary / shear / affine_lattice_correction),
                #       keeping the scale bar / FFT k-axes consistent with the
                #       displayed array shape (review image-proc #4).
                raw_range = getattr(self, "_scan_range_m", None)
                self._display_arr, self._display_scan_range_m = (
                    apply_processing_state_with_calibration(
                        self._raw_arr, state, self._image_roi_set,
                        mask_set=getattr(self, "_image_mask_set", None),
                        scan_range_m=raw_range,
                    )
                )
            except Exception as exc:
                # The viewer falls back to the raw image; without a loud
                # message the failed step reads as "nothing happened".
                _log.warning("Processing pipeline failed; displaying raw image",
                             exc_info=True)
                self._processing_error = f"⚠ Processing failed — showing raw image: {exc}"
                if hasattr(self, "_status_lbl"):
                    self._status_lbl.setText(self._processing_error)
                self._display_arr = self._raw_arr
                self._display_scan_range_m = getattr(self, "_scan_range_m", None)
        else:
            self._processing_roi_error = ""
            self._processing_error = ""
            self._display_arr = self._raw_arr
            self._display_scan_range_m = getattr(self, "_scan_range_m", None)
        # A re-levelling op (e.g. Set Zero Plane) can shift the data range out
        # from under a manual contrast window; drop limits that no longer
        # overlap the data so the image doesn't render as a flat clamped field.
        self._revalidate_manual_display_range()
        new_shape = self._display_arr.shape if self._display_arr is not None else None
        if reset_zoom_if_shape_changed and old_shape is not None and new_shape != old_shape:
            self._reset_zoom_on_next_pixmap = True
        self._update_export_summary()

    def _refresh_histogram_and_markers(self, entry: SxmFile):
        self._update_histogram()
        self._load_spec_markers(entry)

    def _refresh_display_range(self):
        self._update_histogram()
        self._refresh_viewer_pixmap(reset_zoom=False)

    def _refresh_processing_display(self):
        entry = self._entries[self._idx]
        self._refresh_display_array(reset_zoom_if_shape_changed=True)
        self._rebuild_processing_history()
        self._refresh_histogram_and_markers(entry)
        self._refresh_viewer_pixmap(reset_zoom=False)
        self._sync_line_profile_visibility()

    def _on_bad_line_preview_settings_changed(self) -> None:
        if self._bad_line_preview_ctrl is None:
            return
        msg = self._bad_line_preview_ctrl.on_settings_changed()
        if msg and hasattr(self, "_status_lbl"):
            self._status_lbl.setText(msg)

    def _on_preview_bad_lines(self) -> None:
        if self._bad_line_preview_ctrl is None:
            return
        # Always navigate to the Processing tab so the user can see and
        # configure the bad-line settings (method, polarity, threshold…).
        self._show_sidebar_tab("processing")
        if hasattr(self, "_sidebar_tabs") and hasattr(self, "_processing_panel"):
            idx = self._sidebar_tab_indices.get("processing")
            if idx is not None:
                scroll = self._sidebar_tabs.widget(idx)
                if hasattr(scroll, "ensureWidgetVisible"):
                    scroll.ensureWidgetVisible(
                        self._processing_panel._bad_lines_combo
                    )
        msg = self._bad_line_preview_ctrl.run()
        if hasattr(self, "_status_lbl"):
            if msg:
                self._status_lbl.setText(msg)
            elif self._processing_panel.bad_line_method() is None:
                self._status_lbl.setText(
                    "Bad line correction: select a method and polarity in the "
                    "Processing panel, then click 'Preview detection'."
                )

    def _clear_bad_line_preview(self, summary: str = "Preview: not run") -> None:
        if self._bad_line_preview_ctrl is not None:
            self._bad_line_preview_ctrl.clear(summary)

    def _on_open_stm_background(self) -> None:
        self._open_stm_background_for_roi(None)

    def _open_stm_background_for_roi(self, roi_id: str | None = None) -> None:
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("STM Background: no image loaded.")
            return
        active_roi = (
            self._image_roi_set.get(roi_id)
            if self._image_roi_set is not None and roi_id is not None
            else self._active_image_roi()
        )
        roi_mask = area_roi_mask(active_roi, arr.shape[:2])
        roi_id = active_roi.id if roi_mask is not None else None
        roi_name = active_roi.name if roi_mask is not None else None
        dlg = STMBackgroundDialog(
            arr,
            theme=self._t,
            active_roi_mask=roi_mask,
            active_roi_id=roi_id,
            active_roi_name=roi_name,
            prior_row_alignment=self._processing.get("align_rows") or None,
            parent=self,
        )
        dlg.applied.connect(self._on_stm_background_applied)
        self._stm_background_dialog = dlg
        self._present_modal_tool(dlg)

    def _on_stm_background_applied(self, params: dict) -> None:
        self._push_proc_undo_snapshot()
        self._processing["stm_background"] = dict(params)
        self._clear_bad_line_preview()
        self._refresh_processing_display()
        model = str(params.get("model", "linear")).replace("_", " ")
        fit_region = str(params.get("fit_region", "whole_image")).replace("_", " ")
        self._status_lbl.setText(
            f"Applied STM Background ({model}; fit region: {fit_region})."
        )

    def _refresh_viewer_pixmap(self, reset_zoom: bool = False):
        if self._display_arr is None:
            self._zoom_lbl.setText("No image data")
            self._zoom_lbl.setPixmap(QPixmap())
            return
        # Resolve display limits (percentile or manual) from current array
        vmin, vmax = self._drs.resolve(self._display_arr) if self._display_arr is not None else (None, None)
        entry = self._entries[self._idx]
        self._token = object()
        loader = ViewerLoader(entry, self._viewer_colormap, self._token, None,
                              self._ch_cb.currentIndex(),
                              self._clip_low, self._clip_high,
                              None,
                              vmin=vmin, vmax=vmax,
                              arr=self._display_arr,
                              region_levels=self._region_levels_for_render())
        self._reset_zoom_on_next_pixmap = bool(reset_zoom or self._reset_zoom_on_next_pixmap)
        loader.signals.loaded.connect(self._on_loaded)
        loader.signals.failed.connect(self._on_viewer_pixmap_failed)
        self._current_viewer_loader = loader
        self._pool.start(loader)

    def _channel_unit(self) -> tuple[float, str, str]:
        """Return (scale, unit_label, axis_label) for the current channel."""
        idx = self._ch_cb.currentIndex()
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        return resolve_channel_unit(
            self._scan_plane_units,
            self._scan_plane_names,
            idx,
            self._ch_cb.currentText(),
            arr,
        )

    def _set_scan_channel_choices(self, scan) -> None:
        names = list(scan.plane_names) if scan.plane_names else [
            f"Channel {i}" for i in range(scan.n_planes)
        ]
        self._set_scan_channel_choices_from_names(names, list(getattr(scan, "plane_units", [])))

    def _set_scan_channel_choices_from_names(self, names: list[str], units: list[str]) -> None:
        if not names:
            return
        current = self._ch_cb.currentIndex()
        if [self._ch_cb.itemText(i) for i in range(self._ch_cb.count())] == names:
            return
        self._ch_cb.blockSignals(True)
        try:
            self._ch_cb.clear()
            self._ch_cb.addItems(names)
            self._ch_cb.setCurrentIndex(max(0, min(current, len(names) - 1)))
        finally:
            self._ch_cb.blockSignals(False)

    def _update_histogram(self):
        arr = self._display_arr
        if arr is None:
            self._hist_panel.clear(self._t)
            return

        flat = arr[np.isfinite(arr)].ravel()
        if flat.size < 2:
            self._hist_panel.clear(self._t)
            return

        scale, unit, axis_label = self._channel_unit()
        flat_phys = flat.astype(np.float64) * scale

        vmin_si, vmax_si = self._drs.resolve(arr)
        if vmin_si is not None:
            lo_phys = float(vmin_si) * scale
            hi_phys = float(vmax_si) * scale
        else:
            lo_phys, hi_phys = float(flat_phys.min()), float(flat_phys.max())

        self._hist_panel.render(
            flat_phys, lo_phys, hi_phys, unit, axis_label, self._t, scale=scale)
        self._update_display_sliders()
