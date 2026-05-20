"""ImageViewerDialog — double-click viewer with scroll/zoom, histogram, processing, export.

Extracted from probeflow.gui._legacy as part of the ongoing GUI refactor.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import os as _os
_os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import (
    Qt, QThreadPool,
    Signal, Slot,
)
from PySide6.QtGui import (
    QAction, QActionGroup, QCursor, QFont, QKeySequence,
    QPixmap, QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox,
    QDialog, QDockWidget, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMenu, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QStackedWidget,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QVBoxLayout, QWidget,
)

from probeflow.gui.config import (
    CONFIG_PATH,
    GITHUB_URL,
    GUI_FONT_SIZES,
    GUI_FONT_DEFAULT,
    LOGO_PATH,
    normalise_gui_font_size,
    load_config,
    save_config,
)
from probeflow.gui.styling import THEMES, _build_qss, _sep
from probeflow.gui.utils import _open_url, _format_scan_conditions
from probeflow.gui.models import PLANE_NAMES, SxmFile
from probeflow.gui.rendering import (
    CMAP_KEY,
    CMAP_NAMES,
    DEFAULT_CMAP_KEY,
    DEFAULT_CMAP_LABEL,
    STM_COLORMAPS,
    _apply_processing,
)
from probeflow.gui.workers import ViewerLoader
from probeflow.gui.viewer.display_range import DisplayRangeController
from probeflow.gui.viewer.histogram import HistogramPanel
from probeflow.gui.viewer.widgets import LineProfilePanel, RulerWidget, ScaleBarWidget
from probeflow.gui.viewer import (
    BadLinePreviewController,
    DeferredPlaneAction,
    DisplaySliderController,
    ImageMeasurementController,
    ProcessingUndoController,
    SetZeroPlaneController,
    SpecOverlayController,
    activate_roi,
    active_roi,
    active_roi_id,
    delete_active_roi,
    delete_all_rois,
    delete_roi,
    export_histogram,
    export_line_profile,
    has_roi_aware_local_filter,
    invert_active_roi,
    invert_roi,
    load_roi_set,
    plot_roi_line_profile,
    rename_roi,
    resolve_channel_unit,
    roi_canvas_created,
    roi_canvas_moved,
    roi_line_endpoint_changed,
    roi_line_set_width,
    save_roi_set,
    save_viewer_png,
    select_nth_roi,
    selected_or_active_roi_id,
    show_roi_fft,
    show_roi_histogram,
    transform_roi_set_for_display_op,
)
from probeflow.gui.widgets import ImageMeasurementsPanel
from probeflow.gui.image_canvas import ImageCanvas
from probeflow.gui.roi_manager_dock import ROIManagerDock
from probeflow.processing.gui_adapter import processing_state_from_gui
from probeflow.processing.state import (
    assert_roi_references_resolved,
    missing_roi_references,
)
from probeflow.provenance import (
    ProcessingHistory,
    append_processing_state,
    build_export_record,
    display_lines,
    processing_history_from_scan,
)
from probeflow.gui.processing import ProcessingControlPanel
from probeflow.core import AREA_ROI_KINDS
from probeflow.gui.roi_context import (
    active_area_roi_context,
    active_area_roi_area_m2,
    active_line_roi_context,
    area_roi_mask,
    collect_point_source_records,
    point_source_arrays_m,
    point_source_arrays_px,
    point_source_metadata,
    selected_or_active_area_roi_context,
)
from probeflow.core.scan_loader import load_scan
from probeflow.gui.viewer.scan_load import load_scan_for_viewer, ViewerScanData
from probeflow.gui.viewer.processed_export import (
    build_processed_scan_for_export,
    build_processed_export_provenance,
    save_processed_image,
    save_provenance_json,
    write_processed_export_sidecar,
)

# Dialogs imported from their specific submodule files to avoid circular imports
# (this module lives inside probeflow.gui.dialogs).
from probeflow.gui.dialogs.about import AboutDialog
from probeflow.gui.dialogs.definitions import _DefinitionsDialog
from probeflow.gui.dialogs.fft_viewer import FFTViewerDialog
from probeflow.gui.dialogs.periodic_filter import PeriodicFilterDialog
from probeflow.gui.dialogs.stm_background import STMBackgroundDialog

class ImageViewerDialog(QDialog):
    """Double-click viewer with scroll/zoom, histogram display, processing, export."""

    def __init__(self, entry: SxmFile, entries: list[SxmFile],
                 colormap: str, t: dict, parent=None,
                 clip_low: float = 1.0, clip_high: float = 99.0,
                 processing: dict = None,
                 spec_image_map: Optional[dict] = None,
                 initial_plane_idx: int = 0):
        super().__init__(parent)
        self.setWindowTitle(entry.stem)
        self.setMinimumSize(960, 680)
        self.resize(1260, 800)

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
        self._scan_shape: Optional[tuple] = None
        self._scan_format: str = ""
        self._scan_plane_names: list[str] = list(PLANE_NAMES)
        self._scan_plane_units: list[str] = ["m", "m", "A", "A"]
        # Controllers initialised inside _build() after their dependent widgets are created.
        self._spec_overlay: "SpecOverlayController | None" = None
        self._zero_ctrl: "SetZeroPlaneController | None" = None
        self._angle_overlay: "object | None" = None  # AngleOverlayItem, imported lazily
        self._proc_undo_ctrl: "ProcessingUndoController | None" = None
        self._display_slider_ctrl: "DisplaySliderController | None" = None
        self._bad_line_preview_ctrl: "BadLinePreviewController | None" = None
        self._pending_initial_plane_idx: Optional[int] = max(0, int(initial_plane_idx))
        self._reset_zoom_on_next_pixmap = True
        self._deferred = DeferredPlaneAction()

        self._build()
        self._drs.rangeChanged.connect(self._refresh_display_range)
        self._processing_panel.set_state(self._processing)
        self._set_advanced_processing_state(self._processing)
        self._load_current()

    # ── Build ──────────────────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # title
        self._title_lbl = QLabel()
        self._title_lbl.setFont(QFont("Helvetica", 12, QFont.Bold))
        self._title_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self._title_lbl)

        self._conditions_lbl = QLabel()
        self._conditions_lbl.setFont(QFont("Helvetica", 9))
        self._conditions_lbl.setAlignment(Qt.AlignCenter)
        self._conditions_lbl.setStyleSheet("color: palette(mid);")
        root.addWidget(self._conditions_lbl)

        # main splitter: image | right panel
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ── Left: scrollable zoom image ────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(500)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(False)
        self._scroll_area.setAlignment(Qt.AlignCenter)
        self._zoom_lbl = ImageCanvas()
        self._zoom_lbl.setText("Loading…")

        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)

        self._zoom_out_btn = QPushButton("−")
        self._zoom_out_btn.setFixedSize(30, 26)
        self._zoom_out_btn.setFont(QFont("Helvetica", 11))
        self._zoom_out_btn.setToolTip("Zoom out")
        self._zoom_out_btn.clicked.connect(lambda: self._zoom_lbl.zoom_by(1 / 1.25))
        toolbar.addWidget(self._zoom_out_btn)

        self._zoom_reset_btn = QPushButton("1:1")
        self._zoom_reset_btn.setMinimumWidth(42)
        self._zoom_reset_btn.setFixedHeight(26)
        self._zoom_reset_btn.setFont(QFont("Helvetica", 9))
        self._zoom_reset_btn.setToolTip("Reset to native raster size")
        self._zoom_reset_btn.clicked.connect(self._zoom_lbl.reset_zoom)
        toolbar.addWidget(self._zoom_reset_btn)

        self._zoom_fit_btn = QPushButton("Fit")
        self._zoom_fit_btn.setMinimumWidth(40)
        self._zoom_fit_btn.setFixedHeight(26)
        self._zoom_fit_btn.setFont(QFont("Helvetica", 9))
        self._zoom_fit_btn.setToolTip("Fit image to available space")
        self._zoom_fit_btn.clicked.connect(self._zoom_lbl.fit_to_view)
        toolbar.addWidget(self._zoom_fit_btn)

        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setFixedSize(30, 26)
        self._zoom_in_btn.setFont(QFont("Helvetica", 11))
        self._zoom_in_btn.setToolTip("Zoom in")
        self._zoom_in_btn.clicked.connect(lambda: self._zoom_lbl.zoom_by(1.25))
        toolbar.addWidget(self._zoom_in_btn)

        channel_lbl = QLabel("Channel")
        channel_lbl.setFont(QFont("Helvetica", 8, QFont.Bold))
        toolbar.addSpacing(8)
        toolbar.addWidget(channel_lbl)

        self._ch_cb = QComboBox()
        self._ch_cb.addItems(PLANE_NAMES)
        self._ch_cb.setFont(QFont("Helvetica", 8))
        self._ch_cb.setMinimumWidth(170)
        self._ch_cb.currentIndexChanged.connect(self._on_channel_changed)
        toolbar.addWidget(self._ch_cb)

        # Per-image colormap — does not affect browser thumbnails
        cmap_lbl = QLabel("Colormap")
        cmap_lbl.setFont(QFont("Helvetica", 8, QFont.Bold))
        toolbar.addSpacing(8)
        toolbar.addWidget(cmap_lbl)
        self._viewer_cmap_cb = QComboBox()
        self._viewer_cmap_cb.addItems(CMAP_NAMES)
        self._viewer_cmap_cb.setFont(QFont("Helvetica", 8))
        _initial_cmap_label = next(
            (lbl for lbl, k in STM_COLORMAPS
             if k == self._viewer_colormap or lbl == self._viewer_colormap),
            DEFAULT_CMAP_LABEL,
        )
        self._viewer_cmap_cb.setCurrentText(_initial_cmap_label)
        self._viewer_cmap_cb.currentTextChanged.connect(self._on_viewer_colormap_changed)
        toolbar.addWidget(self._viewer_cmap_cb)

        self._coord_lbl = QLabel("—")
        self._coord_lbl.setFont(QFont("Helvetica", 8))
        self._coord_lbl.setMinimumWidth(140)
        toolbar.addWidget(self._coord_lbl)

        zoom_hint = QLabel("Ctrl+scroll to zoom")
        zoom_hint.setFont(QFont("Helvetica", 8))
        toolbar.addWidget(zoom_hint)
        toolbar.addStretch()
        left_lay.addLayout(toolbar)

        from probeflow.gui.image_quick_toolbar import ImageQuickToolbar
        self._quick_toolbar = ImageQuickToolbar(self)
        self._quick_toolbar.mode_requested.connect(self._on_quick_toolbar_mode)
        self._quick_toolbar.action_requested.connect(self._on_quick_toolbar_action)
        left_lay.addWidget(self._quick_toolbar)

        # Rulers scroll together with the image (placed in the same scroll
        # viewport via a small grid container).
        self._ruler_top  = RulerWidget("horizontal")
        self._ruler_left = RulerWidget("vertical")
        ruler_corner = QWidget()
        ruler_corner.setFixedSize(RulerWidget.THICKNESS_PX, RulerWidget.THICKNESS_PX)
        self._ruler_container = QWidget()
        ruler_grid = QGridLayout(self._ruler_container)
        ruler_grid.setContentsMargins(0, 0, 0, 0)
        ruler_grid.setSpacing(0)
        ruler_grid.addWidget(ruler_corner,    0, 0)
        ruler_grid.addWidget(self._ruler_top, 0, 1)
        ruler_grid.addWidget(self._ruler_left, 1, 0)
        ruler_grid.addWidget(self._zoom_lbl,  1, 1)
        self._scroll_area.setWidget(self._ruler_container)
        left_lay.addWidget(self._scroll_area, 1)

        self._scale_bar = ScaleBarWidget()
        left_lay.addWidget(self._scale_bar)

        self._line_profile_panel = LineProfilePanel()
        self._line_profile_panel.setVisible(False)
        left_lay.addWidget(self._line_profile_panel)

        splitter.addWidget(left)

        # ── Right: task-focused sidebar ───────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(380)
        right.setMaximumWidth(420)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 4, 8, 4)
        right_lay.setSpacing(6)

        self._sidebar_tabs = QTabWidget()
        self._sidebar_tabs.setDocumentMode(True)
        self._sidebar_tabs.setMinimumWidth(360)
        self._sidebar_tabs.setElideMode(Qt.ElideNone)
        self._sidebar_tabs.tabBar().setUsesScrollButtons(False)
        right_lay.addWidget(self._sidebar_tabs, 1)
        self._sidebar_tab_indices: dict[str, int] = {}

        def _sidebar_tab(key: str, label: str) -> tuple[QWidget, QVBoxLayout]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            body = QWidget()
            lay = QVBoxLayout(body)
            lay.setContentsMargins(6, 6, 6, 6)
            lay.setSpacing(6)
            scroll.setWidget(body)
            idx = self._sidebar_tabs.addTab(scroll, label)
            self._sidebar_tab_indices[key] = idx
            return body, lay

        _display_tab, display_lay = _sidebar_tab("display", "View")
        _processing_tab, processing_lay = _sidebar_tab("processing", "Process")
        _roi_tab, roi_lay = _sidebar_tab("roi", "ROI")
        _measurements_tab, measurements_lay = _sidebar_tab("measurements", "Measure")
        _export_tab, export_lay = _sidebar_tab("export", "Export")

        def _collapsible_section(
            target_lay: QVBoxLayout,
            title: str,
            expanded: bool = False,
        ):
            btn = QPushButton(("[−] " if expanded else "[+] ") + title)
            btn.setCheckable(True)
            btn.setChecked(expanded)
            btn.setFlat(True)
            btn.setFont(QFont("Helvetica", 9, QFont.Bold))
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            target_lay.addWidget(btn)

            body = QWidget()
            body_lay = QVBoxLayout(body)
            body_lay.setContentsMargins(2, 2, 0, 2)
            body_lay.setSpacing(4)
            body.setVisible(expanded)
            target_lay.addWidget(body)

            def _sync(checked: bool):
                body.setVisible(bool(checked))
                btn.setText(("[−] " if checked else "[+] ") + title)

            btn.toggled.connect(_sync)
            return btn, body, body_lay

        def _spin_row(label: str, mn: float, mx: float, init: float,
                      step: float, decimals: int) -> tuple[QWidget, QDoubleSpinBox]:
            w = QWidget()
            row = QHBoxLayout(w)
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label)
            lbl.setFont(QFont("Helvetica", 8))
            spin = QDoubleSpinBox()
            spin.setRange(float(mn), float(mx))
            spin.setDecimals(decimals)
            spin.setSingleStep(float(step))
            spin.setValue(float(init))
            spin.setFont(QFont("Helvetica", 8))
            row.addWidget(lbl)
            row.addWidget(spin, 1)
            return w, spin

        # ── Histogram / contrast panel (placed in its own dock after _viewer_main) ──
        self._hist_panel = HistogramPanel(parent=self)
        self._hist_panel.rangeReleased.connect(self._on_hist_range_released)
        self._hist_panel.autoClipRequested.connect(self._on_auto_clip)
        self._hist_panel.resetRequested.connect(self._on_reset_display)
        self._hist_panel.contextMenuRequested.connect(self._on_hist_context_menu)
        self._hist_panel.minReleased.connect(self._on_min_slider_changed)
        self._hist_panel.maxReleased.connect(self._on_max_slider_changed)
        self._hist_panel.brightnessReleased.connect(self._on_brightness_slider_changed)
        self._hist_panel.contrastReleased.connect(self._on_contrast_slider_changed)
        display_lay.addWidget(self._hist_panel)
        self._processing_panel = ProcessingControlPanel("viewer_full")
        self._processing_panel.bad_line_preview_requested.connect(
            self._on_preview_bad_lines)
        self._processing_panel.bad_line_preview_settings_changed.connect(
            self._on_bad_line_preview_settings_changed)
        self._processing_panel.stm_background_requested.connect(
            self._on_open_stm_background)
        self._processing_panel.simple_background_requested.connect(
            self._on_simple_background)
        self._processing_panel._align_combo.currentIndexChanged.connect(
            self._on_align_rows_changed)
        processing_lay.addWidget(self._processing_panel)

        processing_lay.addWidget(_sep())

        _, self._history_widget, history_lay = _collapsible_section(
            processing_lay, "Processing history", expanded=False
        )
        self._history_text = QLabel("")
        self._history_text.setFont(QFont("Helvetica", 8))
        self._history_text.setWordWrap(True)
        self._history_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        history_lay.addWidget(self._history_text)

        processing_lay.addWidget(_sep())

        # ── Zero reference | ROI filter scope (compact 2-column row) ──────────
        zs_row = QHBoxLayout()
        zs_row.setSpacing(6)
        zs_row.setContentsMargins(0, 0, 0, 0)

        zero_col = QVBoxLayout()
        zero_col.setSpacing(3)
        zero_col.setContentsMargins(0, 0, 0, 0)
        _zero_hdr = QLabel("Zero ref.")
        _zero_hdr.setFont(QFont("Helvetica", 7, QFont.Bold))
        _zero_hdr.setAlignment(Qt.AlignCenter)
        zero_col.addWidget(_zero_hdr)
        self._set_zero_plane_btn = QPushButton("Set zero plane")
        self._set_zero_plane_btn.setCheckable(True)
        self._set_zero_plane_btn.setFont(QFont("Helvetica", 8))
        self._set_zero_plane_btn.setFixedHeight(24)
        self._set_zero_plane_btn.setToolTip("Click 3 points on the image to define a zero-height plane.")
        self._set_zero_plane_btn.toggled.connect(self._on_set_zero_plane_mode_toggled)
        zero_col.addWidget(self._set_zero_plane_btn)
        self._set_zero_clear_btn = QPushButton("Clear")
        self._set_zero_clear_btn.setFont(QFont("Helvetica", 8))
        self._set_zero_clear_btn.setFixedHeight(22)
        self._set_zero_clear_btn.setToolTip("Clear all zero-plane reference points.")
        self._set_zero_clear_btn.clicked.connect(self._on_clear_set_zero)
        zero_col.addWidget(self._set_zero_clear_btn)
        zero_col.addStretch()

        sel_col = QVBoxLayout()
        sel_col.setSpacing(3)
        sel_col.setContentsMargins(0, 0, 0, 0)
        _sel_hdr = QLabel("ROI filters")
        _sel_hdr.setFont(QFont("Helvetica", 7, QFont.Bold))
        _sel_hdr.setAlignment(Qt.AlignCenter)
        sel_col.addWidget(_sel_hdr)
        self._scope_cb = QComboBox()
        self._scope_cb.addItems(["Whole image", "ROI filters only"])
        self._scope_cb.setFont(QFont("Helvetica", 8))
        self._scope_cb.setToolTip(
            "ROI filters only: smooth/high-pass/edge/FFT apply inside the "
            "active area ROI; background and scan-line corrections remain whole-image.")
        sel_col.addWidget(self._scope_cb)
        sel_col.addStretch()

        zs_row.addLayout(zero_col, 1)
        zs_row.addLayout(sel_col, 1)
        processing_lay.addLayout(zs_row)

        self._roi_status_lbl = QLabel("ROI filter scope: whole image")
        self._roi_status_lbl.setFont(QFont("Helvetica", 8))
        self._roi_status_lbl.setWordWrap(True)
        processing_lay.addWidget(self._roi_status_lbl)

        processing_lay.addWidget(_sep())

        # ── Apply / Reset — always visible ────────────────────────────────────
        ar_row = QHBoxLayout()
        ar_row.setSpacing(4)
        proc_apply_btn = QPushButton("Apply processing")
        proc_apply_btn.setFont(QFont("Helvetica", 8, QFont.Bold))
        proc_apply_btn.setFixedHeight(28)
        proc_apply_btn.setObjectName("accentBtn")
        proc_apply_btn.setToolTip(
            "Apply queued in-panel filters and bad-line correction settings. "
            "Align rows updates immediately; STM Background has its own Apply."
        )
        proc_apply_btn.clicked.connect(self._on_apply_processing)
        proc_reset_btn = QPushButton("Reset")
        proc_reset_btn.setFont(QFont("Helvetica", 8))
        proc_reset_btn.setFixedHeight(28)
        proc_reset_btn.setToolTip(
            "Discard all processing (background, FFT, smoothing, set-zero, …) "
            "and reload the raw on-disk data for the current image.")
        proc_reset_btn.clicked.connect(self._on_reset_processing)
        ar_row.addWidget(proc_apply_btn, 2)
        ar_row.addWidget(proc_reset_btn, 1)
        processing_lay.addLayout(ar_row)

        # ── Undo / Redo — restore previous processing snapshots ───────────────
        ur_row = QHBoxLayout()
        ur_row.setSpacing(4)
        self._proc_undo_btn = QPushButton("↶ Undo")
        self._proc_undo_btn.setFont(QFont("Helvetica", 8))
        self._proc_undo_btn.setFixedHeight(24)
        self._proc_undo_btn.setToolTip(
            "Restore the processing state from before the last Apply / Reset "
            "(Ctrl+Z).")
        self._proc_undo_btn.clicked.connect(self._on_undo_processing)
        self._proc_redo_btn = QPushButton("Redo ↷")
        self._proc_redo_btn.setFont(QFont("Helvetica", 8))
        self._proc_redo_btn.setFixedHeight(24)
        self._proc_redo_btn.setToolTip(
            "Reapply a state that was just undone (Ctrl+Y or Ctrl+Shift+Z).")
        self._proc_redo_btn.clicked.connect(self._on_redo_processing)
        ur_row.addWidget(self._proc_undo_btn, 1)
        ur_row.addWidget(self._proc_redo_btn, 1)
        processing_lay.addLayout(ur_row)
        self._proc_undo_ctrl = ProcessingUndoController(
            self._proc_undo_btn, self._proc_redo_btn, self._sync_viewer_menu_actions,
        )
        self._update_undo_redo_buttons()

        QShortcut(QKeySequence("Ctrl+Z"), self,
                  activated=self._on_undo_processing)
        QShortcut(QKeySequence("Ctrl+Y"), self,
                  activated=self._on_redo_processing)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self,
                  activated=self._on_redo_processing)

        processing_lay.addWidget(_sep())

        # ── Save PNG — always visible ─────────────────────────────────────────
        save_btn = QPushButton("⬇  Save PNG copy…")
        save_btn.setFont(QFont("Helvetica", 8, QFont.Bold))
        save_btn.setFixedHeight(26)
        save_btn.setObjectName("accentBtn")
        save_btn.clicked.connect(self._on_save_png)
        export_lay.addWidget(save_btn)

        # ── Send to tool (collapsible) ────────────────────────────────────────
        _, self._export_widget, send_lay = _collapsible_section(
            export_lay, "→ Send to tool", expanded=False
        )

        send_feat_btn = QPushButton("→ Feature Counting")
        send_feat_btn.setFont(QFont("Helvetica", 8))
        send_feat_btn.setFixedHeight(24)
        send_feat_btn.setToolTip(
            "Send the current processed image to the Feature Counting tab and close viewer")
        send_feat_btn.clicked.connect(self._on_send_to_features)
        send_lay.addWidget(send_feat_btn)

        send_tv_btn = QPushButton("→ TV Denoising")
        send_tv_btn.setFont(QFont("Helvetica", 8))
        send_tv_btn.setFixedHeight(24)
        send_tv_btn.setToolTip(
            "Send the current processed image to the TV Denoising tab and close viewer")
        send_tv_btn.clicked.connect(self._on_send_to_tv)
        send_lay.addWidget(send_tv_btn)

        # ── Advanced tools (collapsible) ──────────────────────────────────────
        _, self._advanced_widget, advanced_lay = _collapsible_section(
            processing_lay, "Advanced tools", expanded=False
        )

        periodic_btn = QPushButton("Periodic FFT filter…")
        periodic_btn.setFont(QFont("Helvetica", 8))
        periodic_btn.setFixedHeight(24)
        periodic_btn.clicked.connect(self._on_periodic_filter)
        advanced_lay.addWidget(periodic_btn)

        fft_viewer_btn = QPushButton("FFT viewer…")
        fft_viewer_btn.setFont(QFont("Helvetica", 8))
        fft_viewer_btn.setFixedHeight(24)
        fft_viewer_btn.setToolTip(
            "Open a side-by-side real-space / FFT window with zoom, pan, "
            "and cursor readout in nm⁻¹.")
        fft_viewer_btn.clicked.connect(self._on_open_fft_viewer)
        advanced_lay.addWidget(fft_viewer_btn)

        radial_fft_lbl = QLabel("Radial FFT")
        radial_fft_lbl.setFont(QFont("Helvetica", 7, QFont.Bold))
        radial_fft_lbl.setAlignment(Qt.AlignCenter)
        advanced_lay.addWidget(radial_fft_lbl)
        fft_mode_row = QHBoxLayout()
        fft_mode_row.setContentsMargins(0, 0, 0, 0)
        fft_mode_lbl = QLabel("Mode:")
        fft_mode_lbl.setFont(QFont("Helvetica", 8))
        self._advanced_fft_combo = QComboBox()
        self._advanced_fft_combo.addItems(["None", "Low-pass", "High-pass"])
        self._advanced_fft_combo.setFont(QFont("Helvetica", 8))
        self._advanced_fft_combo.setToolTip(
            "Global radial low/high-pass FFT filter. Use Apply processing to commit it."
        )
        fft_mode_row.addWidget(fft_mode_lbl)
        fft_mode_row.addWidget(self._advanced_fft_combo, 1)
        advanced_lay.addLayout(fft_mode_row)
        self._advanced_fft_cutoff_w, self._advanced_fft_cutoff_spin = _spin_row(
            "Cutoff:", 0.01, 0.50, 0.10, 0.01, 2)
        self._advanced_fft_cutoff_spin.setToolTip(
            "Fraction of the Nyquist radius used by the radial FFT filter."
        )
        advanced_lay.addWidget(self._advanced_fft_cutoff_w)
        self._advanced_fft_soft_cb = QCheckBox("Soft border")
        self._advanced_fft_soft_cb.setFont(QFont("Helvetica", 8))
        self._advanced_fft_soft_cb.setToolTip(
            "Cosine-taper the image edges before FFT to suppress ringing artefacts."
        )
        advanced_lay.addWidget(self._advanced_fft_soft_cb)

        undistort_lbl = QLabel("Linear undistort (drift)")
        undistort_lbl.setFont(QFont("Helvetica", 7, QFont.Bold))
        undistort_lbl.setAlignment(Qt.AlignCenter)
        advanced_lay.addWidget(undistort_lbl)

        self._undistort_shear_w, self._undistort_shear_spin = _spin_row(
            "Shear x (px):", -20.0, 20.0, 0.0, 0.25, 2)
        advanced_lay.addWidget(self._undistort_shear_w)
        self._undistort_scale_w, self._undistort_scale_spin = _spin_row(
            "Scale y:", 0.80, 1.20, 1.0, 0.005, 3)
        advanced_lay.addWidget(self._undistort_scale_w)

        # ── Spectroscopy overlay (collapsible) ────────────────────────────────
        _, self._spec_overlay_widget, spec_lay = _collapsible_section(
            display_lay, "Spectroscopy overlay", expanded=False
        )

        self._spec_show_cb = QCheckBox("Show spec positions")
        self._spec_show_cb.setFont(QFont("Helvetica", 8))
        self._spec_show_cb.setChecked(False)
        self._spec_show_cb.toggled.connect(self._on_spec_show_toggled)
        spec_lay.addWidget(self._spec_show_cb)

        self._map_spectra_here_btn = QPushButton("Map spectra to this image…")
        self._map_spectra_here_btn.setFont(QFont("Helvetica", 8))
        self._map_spectra_here_btn.setFixedHeight(24)
        self._map_spectra_here_btn.setToolTip(
            "Pick which spectroscopy files in this folder belong to the "
            "currently displayed image. Markers are drawn at each spectrum's "
            "recorded (x,y) position.")
        self._map_spectra_here_btn.clicked.connect(self._on_map_spectra_here)
        spec_lay.addWidget(self._map_spectra_here_btn)

        self._zoom_lbl.marker_clicked.connect(self._on_marker_clicked)
        self._zoom_lbl.pixel_clicked.connect(self._on_set_zero_pick)
        self._zoom_lbl.pixmap_resized.connect(self._on_pixmap_resized)
        self._zoom_lbl.context_menu_requested.connect(self._on_image_context_menu)
        self._zoom_lbl.pixel_hovered.connect(self._on_pixel_hovered)
        self._zoom_lbl.roi_created.connect(self._on_canvas_roi_created)
        self._zoom_lbl.roi_move_requested.connect(self._on_canvas_roi_move)
        self._zoom_lbl.roi_line_preview.connect(self._on_line_roi_preview)
        self._zoom_lbl.roi_line_geometry_changed.connect(self._on_line_roi_geometry_changed)
        self._zoom_lbl.roi_delete_requested.connect(self._on_canvas_roi_delete)
        self._zoom_lbl.roi_copy_requested.connect(self._on_canvas_roi_copy)
        self._zoom_lbl.roi_paste_requested.connect(self._on_canvas_roi_paste)
        self._zoom_lbl.roi_activate_requested.connect(self._on_canvas_roi_activate)
        self._zoom_lbl.tool_changed.connect(self._on_canvas_tool_changed)
        self._zoom_lbl.roi_context_menu_requested.connect(self._on_roi_canvas_context_menu)
        self._zoom_lbl.angle_points_ready.connect(self._on_angle_points_ready)
        self._line_profile_panel.export_csv_clicked.connect(self._on_export_line_profile_csv)
        self._line_profile_panel.width_changed.connect(self._on_line_profile_width_changed)

        roi_empty_lbl = QLabel(
            "ROI tools live in the ROI Manager dock. Choose a drawing tool above "
            "to create an ROI, or reopen the manager here."
        )
        roi_empty_lbl.setFont(QFont("Helvetica", 8))
        roi_empty_lbl.setWordWrap(True)
        roi_lay.addWidget(roi_empty_lbl)
        show_roi_btn = QPushButton("Show ROI Manager")
        show_roi_btn.setDefault(False)
        show_roi_btn.setAutoDefault(False)
        show_roi_btn.clicked.connect(self._show_roi_manager)
        roi_lay.addWidget(show_roi_btn)
        roi_lay.addStretch(1)

        def _sec_lbl(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setFont(QFont("Helvetica", 8))
            lbl.setStyleSheet("font-weight: 600; color: palette(mid);")
            return lbl

        measurements_lay.addWidget(_sec_lbl("Quick measurements"))
        distance_btn = QPushButton("Distance (active line ROI)")
        distance_btn.setFont(QFont("Helvetica", 8))
        distance_btn.setFixedHeight(26)
        distance_btn.setToolTip(
            "Measure the length and angle of the active line ROI."
        )
        distance_btn.clicked.connect(self._on_measure_distance)
        measurements_lay.addWidget(distance_btn)

        angle_btn = QPushButton("Angle (two selected line ROIs)")
        angle_btn.setFont(QFont("Helvetica", 8))
        angle_btn.setFixedHeight(26)
        angle_btn.setToolTip(
            "Measure the acute angle between two selected line ROIs."
        )
        angle_btn.clicked.connect(self._on_measure_angle)
        measurements_lay.addWidget(angle_btn)

        measurements_lay.addWidget(_sep())
        measurements_lay.addWidget(_sec_lbl("ROI measurements"))
        roi_stats_btn = QPushButton("ROI statistics (active area ROI)")
        roi_stats_btn.setFont(QFont("Helvetica", 8))
        roi_stats_btn.setFixedHeight(26)
        roi_stats_btn.setToolTip(
            "Compute area, mean, RMS roughness, and range for the active area ROI."
        )
        roi_stats_btn.clicked.connect(self._on_measure_roi_stats)
        measurements_lay.addWidget(roi_stats_btn)

        measurements_lay.addWidget(_sep())
        measurements_lay.addWidget(_sec_lbl("Feature & Lattice"))
        lattice_btn = QPushButton("Add lattice grid…")
        lattice_btn.setFont(QFont("Helvetica", 8))
        lattice_btn.setFixedHeight(26)
        lattice_btn.setToolTip(
            "Create an interactive lattice/grid overlay on the current image "
            "for atomic-lattice measurement."
        )
        lattice_btn.clicked.connect(self._on_open_lattice_grid)
        measurements_lay.addWidget(lattice_btn)

        feature_finder_btn = QPushButton("Feature finder…")
        feature_finder_btn.setFont(QFont("Helvetica", 8))
        feature_finder_btn.setFixedHeight(26)
        feature_finder_btn.setToolTip(
            "Find local maxima or minima, set thresholds, export coordinates, "
            "and generate a feature image for selective FFT analysis."
        )
        feature_finder_btn.clicked.connect(self._on_open_feature_finder)
        measurements_lay.addWidget(feature_finder_btn)

        measurements_lay.addWidget(_sep())
        measurements_lay.addWidget(_sec_lbl("Feature measurements"))
        pair_corr_btn = QPushButton("Pair correlation…")
        pair_corr_btn.setFont(QFont("Helvetica", 8))
        pair_corr_btn.setFixedHeight(26)
        pair_corr_btn.setToolTip(
            "Compute g(r) radial pair-correlation from feature points or point ROIs."
        )
        pair_corr_btn.clicked.connect(self._on_open_pair_correlation)
        measurements_lay.addWidget(pair_corr_btn)

        feat_lat_btn = QPushButton("Feature-to-lattice…")
        feat_lat_btn.setFont(QFont("Helvetica", 8))
        feat_lat_btn.setFixedHeight(26)
        feat_lat_btn.setToolTip(
            "Compare detected features to the active lattice grid: "
            "matching, off-lattice count, RMS displacement and occupancy."
        )
        feat_lat_btn.clicked.connect(self._on_open_feature_lattice)
        measurements_lay.addWidget(feat_lat_btn)

        measurements_lay.addWidget(_sep())

        results_hint = QLabel("Results appear in the Measurements panel →")
        results_hint.setFont(QFont("Helvetica", 8))
        results_hint.setWordWrap(True)
        results_hint.setStyleSheet("color: palette(mid);")
        measurements_lay.addWidget(results_hint)

        self._status_lbl = QLabel("")
        self._status_lbl.setFont(QFont("Helvetica", 8))
        self._status_lbl.setWordWrap(True)
        right_lay.addWidget(self._status_lbl)

        display_lay.addStretch(1)
        processing_lay.addStretch(1)
        export_lay.addStretch(1)

        splitter.addWidget(right)
        splitter.setSizes([720, 380])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setCollapsible(1, False)

        # Embed splitter in a QMainWindow so we can host the ROI dock widget
        self._viewer_main = QMainWindow()
        self._viewer_main.setWindowFlags(Qt.Widget)
        self._viewer_main.setCentralWidget(splitter)
        self._viewer_main.setDockNestingEnabled(False)

        self._image_roi_set = None
        self._copy_roi_buffer = None  # ROI object held for Ctrl+V paste

        self._measurement_panel = ImageMeasurementsPanel(parent=self._viewer_main)
        self._measurement_table = self._measurement_panel.table
        self._feature_detection_panel = self._measurement_panel.feature_panel
        self._measurement_dock = QDockWidget("Measurements", self._viewer_main)
        self._measurement_dock.setWidget(self._measurement_panel)
        self._measurement_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        self._measurement_dock.setMinimumWidth(240)
        self._image_measurements = ImageMeasurementController(
            self,
            self._measurement_table,
            self._feature_detection_panel,
            self._measurement_panel.point_mask_panel,
            self._measurement_panel.line_periodicity_panel,
        )
        self._measurement_panel.roiStatsRequested.connect(
            self._image_measurements.add_active_roi_stats_measurement
        )
        self._measurement_panel.stepHeightRequested.connect(
            self._image_measurements.add_selected_step_height_measurement
        )
        self._measurement_panel.lineProfileRequested.connect(
            self._image_measurements.add_current_line_profile_measurement
        )
        self._line_profile_panel.add_delta_measurement_clicked.connect(
            self._image_measurements.add_current_line_profile_delta_measurement
        )
        self._line_profile_panel.add_profile_summary_clicked.connect(
            self._image_measurements.add_current_line_profile_measurement
        )

        self._roi_dock = ROIManagerDock(
            roi_set_getter=lambda: self._image_roi_set,
            callbacks={
                "on_roi_set_changed":    self._on_image_roi_set_changed,
                "on_fft_roi":            self._on_roi_fft,
                "on_histogram_roi":      self._on_roi_histogram,
                "on_roi_stats_measurement": self._image_measurements.add_roi_stats_measurement,
                "on_step_height_measurement": self._image_measurements.add_step_height_measurement_for_rois,
                "on_line_profile_roi":   self._on_roi_line_profile,
                "on_line_profile_measurement": self._image_measurements.add_line_profile_measurement_for_roi,
                "on_feature_maxima_roi": self._image_measurements.detect_feature_maxima_for_roi,
                "on_stm_background_roi": self._open_stm_background_for_roi,
                "on_roi_selection_changed": self._sync_viewer_menu_actions,
                "get_image_shape":       self._current_array_shape,
            },
            parent=self._viewer_main,
        )

        # ROI and measurements are powerful but task-specific, so they start
        # hidden and remain reachable from the sidebar and top menus.
        self._viewer_main.addDockWidget(Qt.RightDockWidgetArea, self._roi_dock)
        self._viewer_main.addDockWidget(Qt.RightDockWidgetArea, self._measurement_dock)
        self._viewer_main.splitDockWidget(self._roi_dock, self._measurement_dock, Qt.Vertical)
        self._viewer_main.resizeDocks(
            [self._roi_dock, self._measurement_dock], [220, 220], Qt.Horizontal
        )
        self._roi_dock.hide()
        self._measurement_dock.hide()
        self._build_viewer_menu_bar()
        root.addWidget(self._viewer_main, 1)

        # navigation row
        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("← Prev")
        self._prev_btn.setFont(QFont("Helvetica", 10))
        self._prev_btn.setFixedWidth(90)
        self._prev_btn.clicked.connect(self._go_prev)

        self._pos_lbl = QLabel()
        self._pos_lbl.setAlignment(Qt.AlignCenter)
        self._pos_lbl.setFont(QFont("Helvetica", 10))

        self._next_btn = QPushButton("Next →")
        self._next_btn.setFont(QFont("Helvetica", 10))
        self._next_btn.setFixedWidth(90)
        self._next_btn.clicked.connect(self._go_next)

        close_btn = QPushButton("Close")
        close_btn.setFont(QFont("Helvetica", 10))
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.accept)

        nav_row.addWidget(self._prev_btn)
        nav_row.addStretch()
        nav_row.addWidget(self._pos_lbl)
        nav_row.addStretch()
        nav_row.addWidget(self._next_btn)
        nav_row.addSpacing(16)
        nav_row.addWidget(close_btn)
        root.addLayout(nav_row)

        # Controllers that need widgets created above.
        self._spec_overlay = SpecOverlayController(self._zoom_lbl, self._spec_image_map)
        self._zero_ctrl = SetZeroPlaneController(self._zoom_lbl)
        self._display_slider_ctrl = DisplaySliderController(
            self._drs, self._hist_panel,
            lambda: self._display_arr,
            self._channel_unit,
        )
        self._bad_line_preview_ctrl = BadLinePreviewController(
            self._zoom_lbl,
            self._processing_panel,
            lambda: self._display_arr if self._display_arr is not None else self._raw_arr,
        )

    def _build_viewer_menu_bar(self) -> None:
        menu_bar = self._viewer_main.menuBar()
        self._viewer_processing_actions: dict[str, QAction | dict[str, QAction]] = {}
        self._viewer_roi_tool_actions: dict[str, QAction] = {}
        self._viewer_roi_actions: dict[str, QAction] = {}
        self._viewer_measurement_actions: dict[str, QAction] = {}

        file_menu = menu_bar.addMenu("File")
        close_action = QAction("Close", self)
        close_action.setShortcut(QKeySequence.Close)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

        view_menu = menu_bar.addMenu("View")
        auto_contrast_action = QAction("Auto contrast", self)
        auto_contrast_action.triggered.connect(self._on_auto_clip)
        view_menu.addAction(auto_contrast_action)
        reset_contrast_action = QAction("Reset contrast", self)
        reset_contrast_action.triggered.connect(self._on_reset_display)
        view_menu.addAction(reset_contrast_action)
        view_menu.addSeparator()
        fit_action = QAction("Fit image to window", self)
        fit_action.triggered.connect(self._zoom_lbl.fit_to_view)
        view_menu.addAction(fit_action)
        native_action = QAction("View at 1:1", self)
        native_action.triggered.connect(self._zoom_lbl.reset_zoom)
        view_menu.addAction(native_action)
        view_menu.addSeparator()
        display_panel_action = QAction("Histogram / Contrast", self)
        display_panel_action.triggered.connect(lambda: self._show_sidebar_tab("display"))
        view_menu.addAction(display_panel_action)
        processing_panel_action = QAction("Processing panel", self)
        processing_panel_action.triggered.connect(lambda: self._show_sidebar_tab("processing"))
        view_menu.addAction(processing_panel_action)
        roi_panel_action = QAction("ROI panel", self)
        roi_panel_action.triggered.connect(lambda: self._show_sidebar_tab("roi"))
        view_menu.addAction(roi_panel_action)
        measurements_panel_action = QAction("Measurements panel", self)
        measurements_panel_action.triggered.connect(
            lambda: self._show_sidebar_tab("measurements")
        )
        view_menu.addAction(measurements_panel_action)
        export_panel_action = QAction("Export panel", self)
        export_panel_action.triggered.connect(lambda: self._show_sidebar_tab("export"))
        view_menu.addAction(export_panel_action)
        view_menu.addSeparator()
        for label, dock in (
            ("ROI Manager", self._roi_dock),
            ("Measurements", self._measurement_dock),
        ):
            action = dock.toggleViewAction()
            action.setText(label)
            view_menu.addAction(action)

        processing_menu = menu_bar.addMenu("Processing")
        plane_action = QAction("Plane/background subtraction…", self)
        plane_action.triggered.connect(self._on_simple_background)
        processing_menu.addAction(plane_action)
        stm_bg_top_action = QAction("STM scan-line background…", self)
        stm_bg_top_action.triggered.connect(self._on_open_stm_background)
        processing_menu.addAction(stm_bg_top_action)
        bad_lines_top_action = QAction("Bad scan-line correction…", self)
        bad_lines_top_action.triggered.connect(self._on_preview_bad_lines)
        processing_menu.addAction(bad_lines_top_action)
        processing_menu.addSeparator()
        self._add_combo_menu(
            processing_menu, "Align rows", self._processing_panel._align_combo,
            ["None", "Median", "Mean"],
        )
        self._add_combo_menu(
            processing_menu, "Bad line correction", self._processing_panel._bad_lines_combo,
            ["None", "Step segments", "MAD/outlier segments"],
        )
        self._add_combo_menu(
            processing_menu, "Smooth", self._processing_panel._smooth_combo,
            ["None", "Gaussian"],
        )
        self._add_combo_menu(
            processing_menu, "Hi-pass", self._processing_panel._highpass_combo,
            ["None", "Gaussian"],
        )
        self._add_combo_menu(
            processing_menu, "Edge filter", self._processing_panel._edge_combo,
            ["None", "Laplacian", "LoG", "DoG"],
        )
        processing_menu.addSeparator()

        zero_action = QAction("Zero plane", self)
        zero_action.setCheckable(True)
        zero_action.triggered.connect(self._set_zero_plane_btn.setChecked)
        self._set_zero_plane_btn.toggled.connect(self._sync_viewer_menu_actions)
        self._viewer_processing_actions["zero_plane"] = zero_action
        processing_menu.addAction(zero_action)
        clear_zero_action = QAction("Clear zero plane", self)
        clear_zero_action.triggered.connect(self._on_clear_set_zero)
        processing_menu.addAction(clear_zero_action)
        processing_menu.addSeparator()

        apply_action = QAction("Apply processing", self)
        apply_action.triggered.connect(self._on_apply_processing)
        processing_menu.addAction(apply_action)
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self._on_undo_processing)
        self._viewer_processing_actions["undo"] = undo_action
        processing_menu.addAction(undo_action)
        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(self._on_redo_processing)
        self._viewer_processing_actions["redo"] = redo_action
        processing_menu.addAction(redo_action)
        reset_action = QAction("Reset processing", self)
        reset_action.triggered.connect(self._on_reset_processing)
        processing_menu.addAction(reset_action)

        roi_menu = menu_bar.addMenu("ROI")
        show_roi_manager_action = QAction("Show ROI Manager", self)
        show_roi_manager_action.triggered.connect(self._show_roi_manager)
        roi_menu.addAction(show_roi_manager_action)
        roi_menu.addSeparator()
        tool_group = QActionGroup(self)
        tool_group.setExclusive(True)
        for key, label in (
            ("pan", "Pan"),
            ("rectangle", "Rectangle"),
            ("ellipse", "Ellipse"),
            ("polygon", "Polygon"),
            ("freehand", "Freehand"),
            ("line", "Line"),
            ("point", "Point"),
        ):
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=key: self._set_drawing_tool(value)
            )
            tool_group.addAction(action)
            self._viewer_roi_tool_actions[key] = action
            roi_menu.addAction(action)
        roi_menu.addSeparator()

        rename_action = QAction("Rename ROI", self)
        rename_action.triggered.connect(self._rename_active_image_roi)
        self._viewer_roi_actions["rename"] = rename_action
        roi_menu.addAction(rename_action)
        delete_action = QAction("Delete ROI", self)
        delete_action.triggered.connect(self._delete_active_image_roi)
        self._viewer_roi_actions["delete"] = delete_action
        roi_menu.addAction(delete_action)
        set_active_action = QAction("Set active ROI", self)
        set_active_action.triggered.connect(self._set_selected_or_active_image_roi)
        self._viewer_roi_actions["set_active"] = set_active_action
        roi_menu.addAction(set_active_action)
        invert_action = QAction("Invert ROI", self)
        invert_action.triggered.connect(self._invert_active_image_roi)
        self._viewer_roi_actions["invert"] = invert_action
        roi_menu.addAction(invert_action)
        mask_action = QAction("Mask from selection", self)
        mask_action.triggered.connect(self._on_mask_selection)
        self._viewer_roi_actions["mask"] = mask_action
        roi_menu.addAction(mask_action)
        # Planned ROI menu additions: Grow ROI, Shrink ROI, Specify ROI.
        # These should remain hidden or disabled until implemented in the ROI backend.

        measurements_menu = menu_bar.addMenu("Measurements")
        ruler_action = QAction("Ruler / distance…", self)
        ruler_action.triggered.connect(self._on_measure_distance)
        measurements_menu.addAction(ruler_action)
        angle_action = QAction("Angle measurement…", self)
        angle_action.triggered.connect(self._on_measure_angle)
        measurements_menu.addAction(angle_action)
        roi_stats_new_action = QAction("ROI statistics…", self)
        roi_stats_new_action.triggered.connect(self._on_measure_roi_stats)
        measurements_menu.addAction(roi_stats_new_action)
        measurements_menu.addSeparator()
        add_roi_stats_action = QAction("Add active ROI statistics", self)
        add_roi_stats_action.triggered.connect(
            self._image_measurements.add_active_roi_stats_measurement
        )
        self._viewer_measurement_actions["roi_stats"] = add_roi_stats_action
        measurements_menu.addAction(add_roi_stats_action)
        add_step_height_action = QAction("Add step height from selected ROIs", self)
        add_step_height_action.triggered.connect(
            self._image_measurements.add_selected_step_height_measurement
        )
        self._viewer_measurement_actions["step_height"] = add_step_height_action
        measurements_menu.addAction(add_step_height_action)
        add_line_profile_action = QAction("Add current line profile", self)
        add_line_profile_action.triggered.connect(
            self._image_measurements.add_current_line_profile_measurement
        )
        self._viewer_measurement_actions["line_profile"] = add_line_profile_action
        measurements_menu.addAction(add_line_profile_action)
        find_periodicity_action = QAction("Find periodicity from line profile…", self)
        find_periodicity_action.triggered.connect(
            self._image_measurements.find_periodicity_for_active_line_roi
        )
        self._viewer_measurement_actions["line_periodicity"] = find_periodicity_action
        measurements_menu.addAction(find_periodicity_action)
        detect_maxima_action = QAction("Detect maxima in active ROI", self)
        detect_maxima_action.triggered.connect(
            self._image_measurements.detect_feature_maxima_for_active_roi
        )
        self._viewer_measurement_actions["feature_maxima"] = detect_maxima_action
        measurements_menu.addAction(detect_maxima_action)
        feature_finder_action = QAction("Feature finder…", self)
        feature_finder_action.triggered.connect(self._on_open_feature_finder)
        measurements_menu.addAction(feature_finder_action)
        pair_corr_action = QAction("Pair correlation…", self)
        pair_corr_action.triggered.connect(self._on_open_pair_correlation)
        measurements_menu.addAction(pair_corr_action)
        feat_lat_action = QAction("Feature-to-lattice comparison…", self)
        feat_lat_action.triggered.connect(self._on_open_feature_lattice)
        measurements_menu.addAction(feat_lat_action)
        measurements_menu.addSeparator()
        self._image_measurements.add_detected_point_menu_actions(
            measurements_menu,
            self,
            self._viewer_measurement_actions,
        )
        measurements_menu.addSeparator()
        lattice_grid_action = QAction("Lattice/Grid tool…", self)
        lattice_grid_action.triggered.connect(self._on_open_lattice_grid)
        measurements_menu.addAction(lattice_grid_action)
        measurements_menu.addSeparator()
        show_measurements_action = QAction("Show measurements", self)
        show_measurements_action.triggered.connect(self._show_measurements)
        measurements_menu.addAction(show_measurements_action)
        show_measure_tab_action = QAction("Measurement table (Measure tab)", self)
        show_measure_tab_action.triggered.connect(
            lambda: self._show_sidebar_tab("measurements")
        )
        measurements_menu.addAction(show_measure_tab_action)

        fft_menu = menu_bar.addMenu("FFT")
        open_fft_action = QAction("Open FFT viewer…", self)
        open_fft_action.triggered.connect(self._on_open_fft_viewer)
        fft_menu.addAction(open_fft_action)
        periodic_filter_action = QAction("Periodic filter…", self)
        periodic_filter_action.triggered.connect(self._on_periodic_filter)
        fft_menu.addAction(periodic_filter_action)

        export_menu = menu_bar.addMenu("Export")
        save_png_action = QAction("Save PNG copy", self)
        save_png_action.triggered.connect(self._on_save_png)
        export_menu.addAction(save_png_action)
        save_processed_action = QAction("Save processed image", self)
        save_processed_action.triggered.connect(self._on_save_processed_image)
        export_menu.addAction(save_processed_action)
        save_provenance_action = QAction("Save provenance", self)
        save_provenance_action.triggered.connect(self._on_save_provenance)
        export_menu.addAction(save_provenance_action)

        help_menu = menu_bar.addMenu("Help")
        github_action = QAction("GitHub", self)
        github_action.triggered.connect(lambda: _open_url(GITHUB_URL))
        help_menu.addAction(github_action)
        about_action = QAction("About ProbeFlow", self)
        about_action.triggered.connect(self._show_viewer_about)
        help_menu.addAction(about_action)
        definitions_action = QAction("Definitions", self)
        definitions_action.triggered.connect(self._show_viewer_definitions)
        help_menu.insertAction(github_action, definitions_action)
        help_menu.insertSeparator(github_action)

        self._sync_viewer_menu_actions()

    def _add_combo_menu(
        self,
        parent_menu: QMenu,
        title: str,
        combo: QComboBox,
        labels: list[str],
    ) -> None:
        menu = parent_menu.addMenu(title)
        group = QActionGroup(self)
        group.setExclusive(True)
        action_map: dict[str, QAction] = {}
        for label in labels:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=label, cb=combo: cb.setCurrentText(value)
            )
            group.addAction(action)
            menu.addAction(action)
            action_map[label] = action
        self._viewer_processing_actions[f"combo:{title}"] = action_map
        combo.currentTextChanged.connect(self._sync_viewer_menu_actions)

    def _show_sidebar_tab(self, key: str) -> None:
        if not hasattr(self, "_sidebar_tabs"):
            return
        idx = self._sidebar_tab_indices.get(key)
        if idx is not None:
            self._sidebar_tabs.setCurrentIndex(idx)

    def _show_roi_manager(self) -> None:
        self._show_sidebar_tab("roi")
        if not hasattr(self, "_roi_dock"):
            return
        self._roi_dock.show()
        self._roi_dock.raise_()

    def _show_measurements(self) -> None:
        self._show_sidebar_tab("measurements")
        if not hasattr(self, "_measurement_dock"):
            return
        self._measurement_dock.show()
        self._measurement_dock.raise_()

    # ── Navigation ─────────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        k = event.key()

        # ── drawing tool shortcuts ────────────────────────────────────────────
        _tool_keys = {
            Qt.Key_R: "rectangle",
            Qt.Key_E: "ellipse",
            Qt.Key_P: "polygon",
            Qt.Key_F: "freehand",
            Qt.Key_L: "line",
            Qt.Key_T: "point",
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
        self._load_image_roi_set(entry)
        if self._pending_initial_plane_idx is not None:
            target_ch = self._pending_initial_plane_idx
            self._pending_initial_plane_idx = None
        else:
            target_ch = self._ch_cb.currentIndex()
        data: ViewerScanData = load_scan_for_viewer(entry.path, target_ch)
        self._set_scan_channel_choices_from_names(data.plane_names, data.plane_units)
        clamped = max(0, min(target_ch, max(data.n_planes - 1, 0)))
        self._ch_cb.blockSignals(True)
        self._ch_cb.setCurrentIndex(clamped)
        self._ch_cb.blockSignals(False)
        self._raw_arr          = data.raw_arr
        self._scan_header      = data.scan_header
        self._scan_range_m     = data.scan_range_m
        self._scan_shape       = data.scan_shape
        self._scan_format      = data.source_format
        self._scan_plane_names = data.plane_names
        self._scan_plane_units = data.plane_units
        self._source_processing_history = data.processing_history
        self._rebuild_processing_history()

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
            pass
        self._processing_history = history
        self._sync_history_panel()

    def _sync_history_panel(self) -> None:
        if not hasattr(self, "_history_text"):
            return
        if self._processing_history is None:
            self._history_text.setText("Source: unknown\nChannel: unknown")
            return
        self._history_text.setText("\n".join(display_lines(self._processing_history)))

    def _refresh_display_array(self, reset_zoom_if_shape_changed: bool = False):
        old_shape = self._display_arr.shape if self._display_arr is not None else None
        # display array: raw with processing applied (no grain overlay — that's visual only)
        if self._raw_arr is not None and self._processing:
            try:
                self._processing_roi_error = ""
                self._processing_error = ""
                state = processing_state_from_gui(self._processing or {})
                missing = missing_roi_references(state, self._image_roi_set)
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
                    return
                try:
                    self._display_arr = _apply_processing(
                        self._raw_arr,
                        self._processing,
                        roi_set=self._image_roi_set,
                    )
                except TypeError as exc:
                    if "roi_set" not in str(exc):
                        raise
                    self._display_arr = _apply_processing(self._raw_arr, self._processing)
            except Exception as exc:
                self._processing_error = f"Processing failed: {exc}"
                if hasattr(self, "_status_lbl"):
                    self._status_lbl.setText(self._processing_error)
                self._display_arr = self._raw_arr
        else:
            self._processing_roi_error = ""
            self._processing_error = ""
            self._display_arr = self._raw_arr
        new_shape = self._display_arr.shape if self._display_arr is not None else None
        if reset_zoom_if_shape_changed and old_shape is not None and new_shape != old_shape:
            self._reset_zoom_on_next_pixmap = True

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
        msg = self._bad_line_preview_ctrl.run()
        if msg and hasattr(self, "_status_lbl"):
            self._status_lbl.setText(msg)

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
        roi_mask = None
        roi_id = None
        roi_name = None
        if (
            active_roi is not None
            and active_roi.kind in AREA_ROI_KINDS
        ):
            try:
                roi_mask = active_roi.to_mask(arr.shape[:2])
                if not roi_mask.any():
                    roi_mask = None
                else:
                    roi_id = active_roi.id
                    roi_name = active_roi.name
            except Exception:
                roi_mask = None
        dlg = STMBackgroundDialog(
            arr,
            theme=self._t,
            active_roi_mask=roi_mask,
            active_roi_id=roi_id,
            active_roi_name=roi_name,
            parent=self,
        )
        dlg.applied.connect(self._on_stm_background_applied)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        self._stm_background_dialog = dlg

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
                              arr=self._display_arr)
        self._reset_zoom_on_next_pixmap = bool(reset_zoom or self._reset_zoom_on_next_pixmap)
        loader.signals.loaded.connect(self._on_loaded)
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
        self._ch_cb.clear()
        self._ch_cb.addItems(names)
        self._ch_cb.setCurrentIndex(max(0, min(current, len(names) - 1)))
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

    # ── Spec position overlay ─────────────────────────────────────────────────
    def _load_spec_markers(self, entry):
        self._spec_overlay.load(
            entry,
            self._scan_range_m,
            self._scan_shape,
            self._scan_format,
            self._scan_header,
            show=self._spec_show_cb.isChecked(),
        )

    def _on_spec_show_toggled(self, checked: bool):
        self._spec_overlay.apply_visibility(checked)

    # ── Image-level ROI set ───────────────────────────────────────────────────

    def _load_image_roi_set(self, entry: "SxmFile") -> None:
        """Load ROIs from <stem>.rois.json sidecar if it exists, else create empty set."""
        self._image_roi_set, _err = load_roi_set(entry.path)
        self._zoom_lbl.set_roi_set(self._image_roi_set)
        if hasattr(self, "_roi_dock"):
            self._roi_dock.refresh(self._image_roi_set)
        self._sync_viewer_menu_actions()

    def _save_image_roi_set(self) -> None:
        """Persist the current ROISet to its sidecar file."""
        if self._image_roi_set is None:
            return
        entry = self._entries[self._idx]
        err = save_roi_set(self._image_roi_set, entry.path)
        if err and hasattr(self, "_status_lbl"):
            self._status_lbl.setText(err)

    def _on_image_roi_set_changed(self) -> None:
        self._zoom_lbl.set_roi_set(self._image_roi_set)
        self._save_image_roi_set()
        if hasattr(self, "_roi_dock"):
            self._roi_dock.refresh(self._image_roi_set)
        self._sync_line_profile_visibility()
        self._sync_viewer_menu_actions()

    def _on_pixel_hovered(self, col: int, row: int, val) -> None:
        if not hasattr(self, "_coord_lbl"):
            return
        if val is None:
            self._coord_lbl.setText(f"({col}, {row})")
        else:
            scale, unit, _ = self._channel_unit()
            val_disp = float(val) * scale
            unit_str = f" {unit}" if unit else ""
            self._coord_lbl.setText(f"({col}, {row}): {val_disp:.4g}{unit_str}")

    # ── Canvas drawing-tool callbacks ─────────────────────────────────────────

    def _on_canvas_roi_created(self, roi) -> None:
        roi_canvas_created(
            self._image_roi_set, roi,
            self._on_image_roi_set_changed, self._set_drawing_tool,
        )

    def _on_canvas_roi_move(self, roi_id: str, dx: int, dy: int) -> None:
        roi_canvas_moved(
            self._image_roi_set, roi_id, dx, dy, self._on_image_roi_set_changed,
        )

    def _on_canvas_tool_changed(self, kind: str) -> None:
        """Canvas emitted tool_changed (e.g. after Escape or drawing completion)."""
        if hasattr(self, "_quick_toolbar"):
            self._quick_toolbar.set_active_mode(kind)
        self._sync_line_profile_visibility(kind)
        from probeflow.gui.tool_manager import _TOOL_HINTS
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(_TOOL_HINTS.get(kind, ""))
        self._sync_viewer_menu_actions()

    def _on_roi_canvas_context_menu(self, roi_id: str, global_pos) -> None:
        """Right-click on a ROI in the canvas — show a small ROI action menu."""
        from PySide6.QtWidgets import QMenu
        roi_set = self._image_roi_set
        roi = roi_set.get(roi_id) if roi_set else None
        if roi is None:
            return
        is_area = roi.kind in AREA_ROI_KINDS
        is_line = roi.kind == "line"
        menu = QMenu(self)
        act_active = menu.addAction("Set Active")
        act_active.triggered.connect(lambda: self._set_active_image_roi(roi_id))
        act_rename = menu.addAction("Rename…")
        act_rename.triggered.connect(lambda: self._rename_image_roi(roi_id))
        act_delete = menu.addAction("Delete")
        act_delete.triggered.connect(lambda: self._delete_image_roi(roi_id))
        act_invert = menu.addAction("Invert")
        act_invert.setEnabled(is_area)
        act_invert.triggered.connect(lambda: self._invert_image_roi(roi_id))
        menu.addSeparator()
        act_stm_bg = menu.addAction("STM Background fit from ROI...")
        act_stm_bg.setEnabled(is_area)
        act_stm_bg.triggered.connect(lambda: self._open_stm_background_for_roi(roi_id))
        act_fft = menu.addAction("FFT this region")
        act_fft.setEnabled(is_area)
        act_fft.triggered.connect(lambda: self._on_roi_fft(roi_id))
        act_hist = menu.addAction("Histogram of this region")
        act_hist.setEnabled(is_area)
        act_hist.triggered.connect(lambda: self._on_roi_histogram(roi_id))
        act_stats = menu.addAction("Add ROI statistics to measurements")
        act_stats.setEnabled(is_area)
        act_stats.triggered.connect(
            lambda: self._image_measurements.add_roi_stats_measurement(roi_id)
        )
        act_maxima = menu.addAction("Detect maxima in this region")
        act_maxima.setEnabled(is_area)
        act_maxima.triggered.connect(
            lambda: self._image_measurements.detect_feature_maxima_for_roi(roi_id)
        )
        act_profile = menu.addAction("Line profile")
        act_profile.setEnabled(is_line)
        act_profile.triggered.connect(lambda: self._on_roi_line_profile(roi_id))
        act_profile_measure = menu.addAction("Add line profile measurement")
        act_profile_measure.setEnabled(is_line)
        act_profile_measure.triggered.connect(
            lambda: self._image_measurements.add_line_profile_measurement_for_roi(roi_id)
        )
        menu.exec(global_pos)

    # ── ROI helper actions ────────────────────────────────────────────────────

    def _set_active_image_roi(self, roi_id: str) -> None:
        activate_roi(self._image_roi_set, roi_id, self._on_image_roi_set_changed)

    def _rename_image_roi(self, roi_id: str) -> None:
        rename_roi(self._image_roi_set, roi_id, self._on_image_roi_set_changed, parent=self)

    def _delete_image_roi(self, roi_id: str) -> None:
        delete_roi(self._image_roi_set, roi_id, self._on_image_roi_set_changed)

    def _delete_active_image_roi(self) -> None:
        delete_active_roi(self._image_roi_set, self._on_image_roi_set_changed)

    def _clear_all_image_marks(self) -> None:
        delete_all_rois(self._image_roi_set, self._on_image_roi_set_changed)
        if self._angle_overlay is not None:
            scene = self._zoom_lbl.scene()
            self._angle_overlay.remove_from_scene(scene)
            self._angle_overlay = None

    def _to_dock_result(self, legacy_r, measurement_id: str):
        """Convert a legacy MeasurementResult to the newer dock format."""
        from probeflow.measurements.adapters import legacy_measurement_to_result

        return legacy_measurement_to_result(legacy_r, measurement_id)

    def _add_dialog_measurement_result(self, result) -> None:
        from dataclasses import replace

        from probeflow.measurements.models import MeasurementResult

        mid = self._measurement_table.next_measurement_id()
        if isinstance(result, MeasurementResult):
            dock_result = replace(result, measurement_id=mid)
        else:
            dock_result = self._to_dock_result(result, mid)
        self._measurement_table.add_result(dock_result)
        self._measurement_dock.show()
        self._measurement_dock.raise_()

    def _invert_image_roi(self, roi_id: str) -> None:
        invert_roi(
            self._image_roi_set, roi_id,
            self._current_array_shape(), self._on_image_roi_set_changed,
        )

    def _invert_active_image_roi(self) -> None:
        had_area = (
            self._active_image_roi() is not None
            and self._active_image_roi().kind in AREA_ROI_KINDS
        )
        invert_active_roi(
            self._image_roi_set, self._current_array_shape(), self._on_image_roi_set_changed,
        )
        if had_area and hasattr(self, "_scope_cb"):
            self._scope_cb.setCurrentText("ROI filters only")
            if hasattr(self, "_status_lbl"):
                self._status_lbl.setText(
                    "ROI inverted. Filters will apply inside the inverted area."
                )

    def _select_nth_image_roi(self, n: int) -> None:
        select_nth_roi(self._image_roi_set, n, self._on_image_roi_set_changed)

    # ── ROI operation callbacks ───────────────────────────────────────────────

    def _on_roi_fft(self, roi_id: str) -> None:
        roi = self._image_roi_set.get(roi_id) if self._image_roi_set else None
        if roi is None or self._display_arr is None:
            return
        show_roi_fft(roi, self._display_arr, parent=self)

    def _on_roi_histogram(self, roi_id: str) -> None:
        roi = self._image_roi_set.get(roi_id) if self._image_roi_set else None
        if roi is None or self._display_arr is None:
            return
        show_roi_histogram(roi, self._display_arr, self._channel_unit, parent=self)

    def _on_roi_line_profile(self, roi_id: str) -> None:
        roi = self._image_roi_set.get(roi_id) if self._image_roi_set else None
        if roi is None or roi.kind != "line" or self._display_arr is None:
            return
        self._line_profile_panel.set_width(int(roi.geometry.get("width", 1)))
        plot_roi_line_profile(
            roi, self._display_arr,
            self._pixel_size_xy_m(),
            self._channel_unit,
            self._line_profile_panel,
            self._t,
        )

    def _on_line_roi_preview(
        self, roi_id: str, x1: float, y1: float, x2: float, y2: float,
    ) -> None:
        """Live endpoint drag: update profile without touching the data model."""
        if self._display_arr is None:
            return
        from probeflow.core.roi import ROI as _ROI
        tmp_roi = _ROI(
            id=roi_id, name="", kind="line",
            geometry={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        )
        plot_roi_line_profile(
            tmp_roi, self._display_arr,
            self._pixel_size_xy_m(),
            self._channel_unit,
            self._line_profile_panel,
            self._t,
        )

    def _on_line_roi_geometry_changed(
        self, roi_id: str, x1: float, y1: float, x2: float, y2: float,
    ) -> None:
        """Endpoint drag released: commit new geometry and persist."""
        roi_line_endpoint_changed(
            self._image_roi_set, roi_id, x1, y1, x2, y2,
            self._on_image_roi_set_changed,
        )

    def _on_line_profile_width_changed(self, width: int) -> None:
        """Width spinbox changed: update active line ROI geometry and re-plot."""
        roi_id = self._active_line_roi_id()
        if roi_id is None:
            return
        roi_line_set_width(
            self._image_roi_set, roi_id, width,
            self._on_image_roi_set_changed,
        )

    def _on_canvas_roi_activate(self, roi_id: str) -> None:
        if self._image_roi_set is None:
            return
        self._image_roi_set.set_active(roi_id)
        self._on_image_roi_set_changed()

    def _on_canvas_roi_delete(self, roi_id: str) -> None:
        if self._image_roi_set is None:
            return
        self._image_roi_set.remove(roi_id)
        self._on_image_roi_set_changed()

    def _on_canvas_roi_copy(self, roi_id: str) -> None:
        roi = self._image_roi_set.get(roi_id) if self._image_roi_set else None
        if roi is not None:
            self._copy_roi_buffer = roi

    def _on_canvas_roi_paste(self) -> None:
        roi = self._copy_roi_buffer
        if roi is None or self._image_roi_set is None:
            return
        from probeflow.core.roi import ROI as _ROI, translate as _translate
        # Create a new ROI (new id) offset by 10 pixels so it doesn't overlap
        offset_roi = _translate(roi, 10.0, 10.0)
        pasted = _ROI.new(
            offset_roi.kind, offset_roi.geometry,
            name=f"{roi.name}_copy",
        )
        self._image_roi_set.add(pasted)
        self._image_roi_set.set_active(pasted.id)
        self._on_image_roi_set_changed()

    def _on_map_spectra_here(self):
        """Open the per-image spec→this-image mapping dialog."""
        entry = self._entries[self._idx]
        accepted, n = self._spec_overlay.open_map_dialog(entry, self)
        if not accepted and n == 0 and not self._spec_image_map:
            self._status_lbl.setText(
                "No spectroscopy files found alongside this image.")
            return
        if accepted:
            self._status_lbl.setText(
                f"{n} spec(s) mapped to this image. Reloading markers…")
            self._load_spec_markers(entry)

    def _on_marker_clicked(self, entry):
        self._spec_overlay.open_spec_viewer(entry, self._t, self)

    def _current_array_shape(self) -> tuple[int, int] | None:
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        return None if arr is None else arr.shape

    def _active_image_roi(self):
        return active_roi(self._image_roi_set)

    def _processing_has_roi_aware_local_filter(self, state: dict) -> bool:
        return has_roi_aware_local_filter(state)

    def _selected_or_active_image_roi_id(self) -> "str | None":
        return selected_or_active_roi_id(
            getattr(self, "_image_roi_set", None), getattr(self, "_roi_dock", None),
        )

    def _rename_active_image_roi(self) -> None:
        roi_id = self._selected_or_active_image_roi_id()
        if roi_id:
            self._rename_image_roi(roi_id)

    def _set_selected_or_active_image_roi(self) -> None:
        roi_id = self._selected_or_active_image_roi_id()
        if roi_id:
            self._set_active_image_roi(roi_id)

    def _show_viewer_about(self) -> None:
        dlg = AboutDialog(self._t, self)
        dlg.exec()

    def _show_viewer_definitions(self) -> None:
        dlg = getattr(self, "_definitions_dialog", None)
        if dlg is None:
            dlg = _DefinitionsDialog(self._t, self)
            self._definitions_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _sync_viewer_menu_actions(self) -> None:
        if hasattr(self, "_viewer_processing_actions"):
            for key, value in self._viewer_processing_actions.items():
                if isinstance(value, dict):
                    title = key.removeprefix("combo:")
                    combo = {
                        "Align rows": self._processing_panel._align_combo,
                        "Bad line correction": self._processing_panel._bad_lines_combo,
                        "Smooth": self._processing_panel._smooth_combo,
                        "Hi-pass": self._processing_panel._highpass_combo,
                        "Edge filter": self._processing_panel._edge_combo,
                    }.get(title)
                    current = combo.currentText() if combo is not None else ""
                    for label, action in value.items():
                        action.blockSignals(True)
                        action.setChecked(label == current)
                        action.blockSignals(False)
                    continue
                if key == "zero_plane":
                    value.blockSignals(True)
                    value.setChecked(self._set_zero_plane_btn.isChecked())
                    value.blockSignals(False)
                elif key == "undo":
                    value.setEnabled(self._proc_undo_ctrl.can_undo)
                elif key == "redo":
                    value.setEnabled(self._proc_undo_ctrl.can_redo)

        if hasattr(self, "_viewer_roi_tool_actions"):
            tool = self._zoom_lbl.tool()
            for key, action in self._viewer_roi_tool_actions.items():
                action.blockSignals(True)
                action.setChecked(key == tool)
                action.blockSignals(False)
            if tool in {"rectangle", "ellipse", "polygon", "freehand", "point"}:
                self._show_sidebar_tab("roi")
            elif tool == "line":
                self._show_sidebar_tab("measurements")
                if hasattr(self, "_measurement_panel"):
                    self._measurement_panel.set_measurement_type("line_profile")

        roi = None
        is_area = False
        if hasattr(self, "_image_roi_set"):
            roi_id = self._selected_or_active_image_roi_id()
            roi = self._image_roi_set.get(roi_id) if (self._image_roi_set and roi_id) else None
            is_area = roi is not None and roi.kind in AREA_ROI_KINDS

        if hasattr(self, "_viewer_roi_actions"):
            for key, action in self._viewer_roi_actions.items():
                action.setEnabled(roi is not None)
                if key in ("invert", "mask"):
                    action.setEnabled(is_area)

        if hasattr(self, "_viewer_measurement_actions"):
            states = self._image_measurements.action_enabled_state()
            for key, action in self._viewer_measurement_actions.items():
                action.setEnabled(states.get(key, True))

        if hasattr(self, "_quick_toolbar"):
            is_line = bool(self._active_line_roi_id())
            self._quick_toolbar.set_action_enabled("line_periodicity", is_line)
            self._quick_toolbar.set_action_enabled("line_profile", is_line)
            self._quick_toolbar.set_action_enabled("mask_selection", is_area)
            self._quick_toolbar.set_action_enabled("invert_selection", is_area)

    def _set_selection_tool(self, kind: str) -> None:
        """Compat shim: delegates to _set_drawing_tool, mapping 'none' → 'pan'."""
        self._set_drawing_tool(kind if kind and kind != "none" else "pan")

    def _set_drawing_tool(self, kind: str) -> None:
        """Activate a drawing tool both on the canvas and in the toolbar."""
        kind = str(kind or "pan")
        from probeflow.gui.tool_manager import TOOLS
        if kind not in TOOLS:
            kind = "pan"
        if hasattr(self, "_quick_toolbar"):
            self._quick_toolbar.set_active_mode(kind)
        self._zoom_lbl.set_tool(kind)
        self._sync_line_profile_visibility(kind)
        from probeflow.gui.tool_manager import _TOOL_HINTS
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(_TOOL_HINTS.get(kind, ""))
        self._sync_viewer_menu_actions()

    def _on_quick_toolbar_mode(self, key: str) -> None:
        """Handle a drawing-mode request from the quick toolbar."""
        if self._set_zero_plane_btn.isChecked():
            self._set_zero_plane_btn.setChecked(False)
        self._set_drawing_tool(key)

    def _on_quick_toolbar_action(self, key: str) -> None:
        """Dispatch an action request from the quick toolbar to existing handlers."""
        dispatch = {
            "clear_selection":   self._clear_all_image_marks,
            "auto_contrast":     self._on_auto_clip,
            "plane_background":  self._on_simple_background,
            "stm_background":    self._on_open_stm_background,
            "bad_lines":         self._on_preview_bad_lines,
            "open_fft":          self._on_open_fft_viewer,
            "open_lattice_grid": self._on_open_lattice_grid,
            "line_periodicity":  self._image_measurements.find_periodicity_for_active_line_roi,
            "line_profile":      self._image_measurements.add_current_line_profile_measurement,
            "mask_selection":    self._on_mask_selection,
            "invert_selection":  self._invert_active_image_roi,
        }
        handler = dispatch.get(key)
        if handler is not None:
            handler()

    def _on_mask_selection(self) -> None:
        """Apply ROI-scoped filter mask from the active area ROI."""
        roi_ctx = selected_or_active_area_roi_context(
            self._image_roi_set,
            getattr(self, "_roi_dock", None),
        )
        if roi_ctx.roi is None:
            if hasattr(self, "_status_lbl"):
                self._status_lbl.setText(
                    "Select an area ROI first to use mask-based processing."
                )
            return
        self._scope_cb.setCurrentText("ROI filters only")
        self._show_sidebar_tab("processing")
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(
                f"ROI filter scope set to '{roi_ctx.roi.name}'. "
                "Filters now apply inside the ROI only."
            )

    def _active_line_roi_id(self) -> "str | None":
        """Return the active ROI id if it is a line ROI, else None."""
        return active_line_roi_context(getattr(self, "_image_roi_set", None)).roi_id

    def _sync_line_profile_visibility(self, kind: str | None = None) -> None:
        if not hasattr(self, "_line_profile_panel"):
            return
        tool_is_line = (kind or self._zoom_lbl.selection_tool()) == "line"
        active_line_id = self._active_line_roi_id()
        is_line = tool_is_line or (active_line_id is not None)
        self._line_profile_panel.setVisible(is_line)
        if is_line:
            if active_line_id is not None:
                self._on_roi_line_profile(active_line_id)
            else:
                self._line_profile_panel.show_empty(theme=self._t)
        else:
            self._line_profile_panel.show_empty(theme=self._t)

    def _pixel_size_xy_m(self) -> tuple[float, float]:
        shape = self._current_array_shape()
        if shape is None or self._scan_range_m is None:
            return 1e-10, 1e-10
        Ny, Nx = shape
        try:
            w_m = float(self._scan_range_m[0])
            h_m = float(self._scan_range_m[1])
        except (TypeError, ValueError, IndexError):
            return 1e-10, 1e-10
        px_x = w_m / Nx if Nx > 0 and w_m > 0 else 1e-10
        px_y = h_m / Ny if Ny > 0 and h_m > 0 else 1e-10
        return px_x, px_y

    def _on_set_zero_plane_mode_toggled(self, checked: bool):
        msg = self._zero_ctrl.toggle(checked, self._set_selection_tool)
        self._zoom_lbl.set_set_zero_mode(checked)
        if msg:
            self._status_lbl.setText(msg)
        if not checked:
            self._refresh_zero_markers()

    def _on_set_zero_pick(self, frac_x: float, frac_y: float):
        """Handle image clicks while manual zero-plane mode is active."""
        rerender, msg = self._zero_ctrl.on_canvas_pick(
            frac_x, frac_y,
            self._raw_arr,
            self._processing,
            self._set_zero_plane_btn.isChecked(),
        )
        if msg:
            self._status_lbl.setText(msg)
        if rerender:
            if self._set_zero_plane_btn.isChecked():
                self._set_zero_plane_btn.setChecked(False)
            self._refresh_processing_display()

    def _refresh_zero_markers(self):
        self._zero_ctrl.refresh_markers(self._raw_arr, self._processing)

    def _on_clear_set_zero(self):
        if self._set_zero_plane_btn.isChecked():
            self._set_zero_plane_btn.setChecked(False)
        msg = self._zero_ctrl.clear()
        self._status_lbl.setText(msg)

    # ── Histogram range and clip handlers ─────────────────────────────────────
    def _on_hist_range_released(self, lo_phys: float, hi_phys: float) -> None:
        """Receive drag-release from HistogramPanel and update display range."""
        scale, _, _ = self._channel_unit()
        if not scale:
            return
        self._drs.set_manual(lo_phys / scale, hi_phys / scale)

    def _on_auto_clip(self):
        """Reset to 1%–99% percentile autoscale."""
        self._clip_low  = 1.0
        self._clip_high = 99.0
        self._drs.reset()

    def _on_reset_display(self):
        """Reset display range to default percentile state."""
        self._drs.reset()

    # ── Per-image colormap ────────────────────────────────────────────────────
    def _on_viewer_colormap_changed(self, label: str) -> None:
        """Update the viewer colormap without touching browser thumbnails."""
        from probeflow.gui.rendering import CMAP_KEY as _CMAP_KEY
        self._viewer_colormap = _CMAP_KEY.get(label, label)
        self._refresh_viewer_pixmap(reset_zoom=False)

    # ── Display range sliders ─────────────────────────────────────────────────

    def _update_display_sliders(self) -> None:
        self._display_slider_ctrl.update()

    def _on_min_slider_changed(self, v: int) -> None:
        self._display_slider_ctrl.on_min_changed(v)

    def _on_max_slider_changed(self, v: int) -> None:
        self._display_slider_ctrl.on_max_changed(v)

    def _on_brightness_slider_changed(self, v: int) -> None:
        self._display_slider_ctrl.on_brightness_changed(v)

    def _on_contrast_slider_changed(self, v: int) -> None:
        self._display_slider_ctrl.on_contrast_changed(v)

    # ── Simple background subtraction ─────────────────────────────────────────
    def _on_simple_background(self) -> None:
        """Apply automated plane subtraction (order-1 polynomial fit, whole image)."""
        if self._display_arr is None:
            return
        self._push_proc_undo_snapshot()
        self._processing["plane_bg"] = {"order": 1}
        self._clear_bad_line_preview()
        self._refresh_processing_display()
        self._status_lbl.setText("Simple background: plane subtracted.")

    def _on_hist_context_menu(self, pos):
        menu = QMenu(self)
        auto_action = menu.addAction("Auto display range")
        export_action = menu.addAction("Export histogram...")
        chosen = menu.exec(self._hist_panel._canvas.mapToGlobal(pos))
        if chosen is auto_action:
            self._on_auto_clip()
        elif chosen is export_action:
            self._on_export_histogram()

    def _on_export_histogram(self):
        """Save the current histogram (bin centres + counts) as a TSV file."""
        ok, msg = export_histogram(
            self._hist_panel.flat_phys,
            self._entries[self._idx].stem,
            self._hist_panel.unit or "",
            self._ch_cb.currentText(),
            parent=self,
        )
        if msg:
            self._status_lbl.setText(msg)

    def _on_channel_changed(self, _: int):
        # Different channels have different physical units — reset manual limits.
        # Use reset_silent to avoid a premature refresh with stale channel data.
        self._drs.reset_silent(self._clip_low, self._clip_high)
        self._hist_panel.clear(self._t)
        self._load_current(reset_zoom=True)

    @Slot(QPixmap, object)
    def _on_loaded(self, pixmap: QPixmap, token):
        if token is not self._token:
            return
        self._zoom_lbl.setText("")
        reset_zoom = self._reset_zoom_on_next_pixmap
        self._reset_zoom_on_next_pixmap = False
        self._zoom_lbl.set_source(pixmap, reset_zoom=reset_zoom)
        self._zoom_lbl.set_raw_array(self._display_arr)
        self._refresh_zero_markers()
        self._refresh_scale_bar()

    def _scan_extent_nm(self) -> tuple[float, float]:
        """Return (width_nm, height_nm) for the current scan, or (0,0)."""
        if self._scan_range_m is None:
            return 0.0, 0.0
        try:
            w_nm = float(self._scan_range_m[0]) * 1e9
            h_nm = float(self._scan_range_m[1]) * 1e9
        except (TypeError, ValueError, IndexError):
            return 0.0, 0.0
        return max(0.0, w_nm), max(0.0, h_nm)

    def _refresh_scale_bar(self):
        """Re-bind the scale bar + axes rulers to current scan/pixmap dimensions."""
        w_nm, h_nm = self._scan_extent_nm()
        pw = self._zoom_lbl.width()
        ph = self._zoom_lbl.height()
        self._scale_bar.set_scan_size(w_nm, pw)
        self._ruler_top.set_extent(w_nm, pw)
        self._ruler_left.set_extent(h_nm, ph)
        # The scroll area hosts a container (rulers + image), not the image
        # label directly. When the pixmap/ruler fixed sizes change, Qt does not
        # automatically resize that non-resizable scroll widget; without this,
        # the container can stay at its tiny construction-time size and show
        # only a postage-stamp slice of the large image.
        self._ruler_container.adjustSize()

    def _on_pixmap_resized(self, new_width_px: int):
        new_h = self._zoom_lbl.height()
        w_nm, h_nm = self._scan_extent_nm()
        self._scale_bar.set_scan_size(w_nm, new_width_px)
        self._ruler_top.set_extent(w_nm, new_width_px)
        self._ruler_left.set_extent(h_nm, new_h)
        self._ruler_container.adjustSize()

    # ── Controls ───────────────────────────────────────────────────────────────
    def _on_align_rows_changed(self, _index: int) -> None:
        """Apply row-alignment changes immediately without committing queued filters."""
        if not hasattr(self, "_processing_panel"):
            return
        align_value = self._processing_panel.state().get("align_rows")
        current_value = self._processing.get("align_rows")
        if current_value == align_value:
            self._sync_viewer_menu_actions()
            return
        if align_value is None and "align_rows" not in self._processing:
            self._sync_viewer_menu_actions()
            return
        base_state = copy.deepcopy(self._processing)
        base_state.pop("align_rows", None)
        coalesced_align_undo = self._proc_undo_ctrl.try_coalesce(base_state)
        if not coalesced_align_undo:
            self._proc_undo_ctrl.push(self._processing)
        if align_value is None:
            self._processing.pop("align_rows", None)
            label = "None"
        else:
            self._processing["align_rows"] = align_value
            label = str(align_value).replace("_", " ").title()
        if coalesced_align_undo:
            self._proc_undo_ctrl.discard_last_undo_if_eq(self._processing)
        self._clear_bad_line_preview()
        self._refresh_processing_display()
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(f"Align rows: {label}.")

    def _advanced_processing_state(self) -> dict:
        if not hasattr(self, "_undistort_shear_spin"):
            return {}
        shear_x = float(self._undistort_shear_spin.value())
        scale_y = float(self._undistort_scale_spin.value())
        state = {
            "linear_undistort": (shear_x != 0.0 or scale_y != 1.0),
            "undistort_shear_x": shear_x,
            "undistort_scale_y": scale_y,
        }
        if hasattr(self, "_advanced_fft_combo"):
            fft_map = {0: None, 1: "low_pass", 2: "high_pass"}
            fft_mode = fft_map.get(self._advanced_fft_combo.currentIndex())
            fft_cutoff = float(self._advanced_fft_cutoff_spin.value())
            if fft_mode is not None:
                state.update({
                    "fft_mode": fft_mode,
                    "fft_cutoff": fft_cutoff,
                    "fft_window": "hanning",
                })
            if self._advanced_fft_soft_cb.isChecked():
                state.update({
                    "fft_soft_border": True,
                    "fft_soft_mode": fft_mode or "low_pass",
                    "fft_soft_cutoff": fft_cutoff,
                    "fft_soft_border_frac": 0.12,
                })
        return state

    def _set_advanced_processing_state(self, state: dict | None) -> None:
        if not hasattr(self, "_undistort_shear_spin"):
            return
        state = state or {}
        if hasattr(self, "_advanced_fft_combo"):
            fft_mode = state.get("fft_mode") or state.get("fft_soft_mode")
            self._advanced_fft_combo.setCurrentIndex(
                {None: 0, "low_pass": 1, "high_pass": 2}.get(fft_mode, 0)
            )
            cutoff = state.get("fft_cutoff", state.get("fft_soft_cutoff", 0.10))
            self._advanced_fft_cutoff_spin.setValue(float(cutoff))
            self._advanced_fft_soft_cb.setChecked(bool(state.get("fft_soft_border", False)))
        self._undistort_shear_spin.setValue(float(state.get("undistort_shear_x", 0.0)))
        self._undistort_scale_spin.setValue(float(state.get("undistort_scale_y", 1.0)))

    def _on_open_lattice_grid(self):
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return
        from probeflow.gui.lattice_grid_tool import open_real_space_tool
        scan_range = self._scan_range_m or (float(arr.shape[1]) * 1e-9,
                                            float(arr.shape[0]) * 1e-9)

        def _get_image():
            return self._display_arr if self._display_arr is not None else self._raw_arr

        def _preview_lattice_correction(corrected_arr) -> None:
            self._display_arr = corrected_arr
            self._refresh_viewer_pixmap(reset_zoom=False)

        def _clear_lattice_correction_preview() -> None:
            self._refresh_display_array(reset_zoom_if_shape_changed=False)
            self._refresh_viewer_pixmap(reset_zoom=False)

        def _apply_lattice_correction(op_name: str, op_params: dict) -> None:
            ops = list(self._processing.get("geometric_ops") or [])
            ops.append({"op": op_name, "params": op_params})
            self._processing["geometric_ops"] = ops
            self._refresh_processing_display()

        item, panel = open_real_space_tool(
            self._zoom_lbl, scan_range, arr.shape, parent=self,
            get_image_fn=_get_image,
            apply_correction_fn=_apply_lattice_correction,
            preview_image_fn=_preview_lattice_correction,
            clear_preview_fn=_clear_lattice_correction_preview,
        )
        self._lattice_grid_item = item
        dock = QDockWidget("Lattice Grid", self._viewer_main)
        dock.setWidget(panel)
        dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        dock.setMinimumWidth(220)
        self._viewer_main.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.show()
        dock.raise_()

        def _on_dock_closed():
            if self._zoom_lbl.scene() and item.scene():
                self._zoom_lbl.scene().removeItem(item)

        dock.visibilityChanged.connect(lambda v: _on_dock_closed() if not v else None)

    def _on_open_feature_finder(self):
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return
        px_x_m, px_y_m = self._pixel_size_xy_m()
        roi_mask = None
        roi_ctx = selected_or_active_area_roi_context(
            self._image_roi_set,
            getattr(self, "_roi_dock", None),
        )
        if roi_ctx.roi is not None:
            roi_mask = area_roi_mask(roi_ctx.roi, arr.shape[:2])
        from probeflow.gui.dialogs.feature_finder import FeatureFinderDialog
        dlg = FeatureFinderDialog(
            arr,
            pixel_size_x_m=px_x_m,
            pixel_size_y_m=px_y_m,
            roi_mask=roi_mask,
            theme=self._t,
            parent=self,
        )
        self._feature_finder_dlg = dlg
        dlg.show()

    def _on_measure_distance(self) -> None:
        """Measure length/angle of the active line ROI → new panel."""
        roi_id = self._active_line_roi_id()
        if roi_id is None:
            self._status_lbl.setText("Select a line ROI first.")
            return
        roi = self._image_roi_set.get(roi_id) if self._image_roi_set else None
        if roi is None:
            return
        from probeflow.analysis.simple_measurements import measure_line_distance
        px_x_m, px_y_m = self._pixel_size_xy_m()
        mid = self._measurement_table.next_measurement_id()
        _, ch_unit, _ = self._channel_unit()
        result = measure_line_distance(
            roi, px_x_m, px_y_m,
            measurement_id=mid,
            source=self._source_label(),
            channel=ch_unit,
        )
        self._measurement_table.add_result(result)
        self._measurement_dock.show()
        self._measurement_dock.raise_()
        self._status_lbl.setText(str(result.context.get("summary") or ""))

    def _on_measure_angle(self) -> None:
        """Switch to the 3-point angle tool; handles emitted from angle_points_ready."""
        self._zoom_lbl.set_tool("angle")
        self._status_lbl.setText("Click P1, P2 (vertex), P3 to measure angle")

    def _on_angle_points_ready(self, p1, p2, p3) -> None:
        """Create angle overlay and record result from the 3-point angle tool."""
        from probeflow.gui.angle_overlay import AngleOverlayItem
        scene = self._zoom_lbl.scene()
        if self._angle_overlay is not None:
            self._angle_overlay.remove_from_scene(scene)
        self._angle_overlay = AngleOverlayItem(p1, p2, p3, scene)
        deg = self._angle_overlay.angle_deg
        from probeflow.measurements.models import MeasurementResult as R
        mid = self._measurement_table.next_measurement_id()
        result = R(
            measurement_id=mid,
            kind="angle",
            source_label=self._source_label(),
            source_path=self._source_label(),
            channel=None,
            x_unit="°",
            y_unit=None,
            z_unit=None,
            values={"angle_deg": deg},
            context={},
            notes="",
        )
        self._measurement_table.add_result(result)
        self._measurement_dock.show()
        self._measurement_dock.raise_()
        self._status_lbl.setText(f"Angle: {deg:.2f}°  (drag handles to adjust)")

    def _on_measure_roi_stats(self) -> None:
        """Compute statistics for the active area ROI → new panel."""
        roi_set = self._image_roi_set
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return
        if roi_set is None:
            self._status_lbl.setText("No ROIs loaded.")
            return
        roi_ctx = selected_or_active_area_roi_context(
            roi_set,
            getattr(self, "_roi_dock", None),
        )
        roi = roi_ctx.roi
        if roi is None:
            self._status_lbl.setText("Select an area ROI first.")
            return
        mask = area_roi_mask(roi, arr.shape[:2])
        if mask is None:
            self._status_lbl.setText("Could not create a non-empty ROI mask.")
            return
        scale, ch_unit, _ = self._channel_unit()
        phys_arr = arr.astype(float) * float(scale)
        from probeflow.analysis.roi_statistics import compute_roi_statistics
        px_x_m, px_y_m = self._pixel_size_xy_m()
        mid = self._measurement_table.next_measurement_id()
        result = compute_roi_statistics(
            phys_arr, mask,
            pixel_size_x_m=px_x_m,
            pixel_size_y_m=px_y_m,
            z_unit=ch_unit,
            measurement_id=mid,
            source=self._source_label(),
            channel=ch_unit,
            roi_id=roi.id,
            roi_name=roi.name,
        )
        self._measurement_table.add_result(result)
        self._measurement_dock.show()
        self._measurement_dock.raise_()
        self._status_lbl.setText(str(result.context.get("summary") or ""))

    def _source_label(self) -> str:
        """Short label for the currently loaded file, for measurement provenance."""
        try:
            return self._entries[self._idx].stem
        except (AttributeError, IndexError, TypeError):
            return ""

    def _point_source_records(self):
        px_x, px_y = self._pixel_size_xy_m()
        ff_dlg = getattr(self, "_feature_finder_dlg", None)
        measure_ctrl = getattr(self, "_image_measurements", None)
        dock = getattr(self, "_roi_dock", None)
        sel_ids = list(dock.selected_roi_ids()) if dock and hasattr(dock, "selected_roi_ids") else []
        return collect_point_source_records(
            pixel_size_x_m=px_x,
            pixel_size_y_m=px_y,
            feature_finder_result=getattr(ff_dlg, "result", None),
            measurement_points=getattr(measure_ctrl, "feature_points", []) or [],
            measurement_metadata=getattr(measure_ctrl, "feature_metadata", {}) or {},
            roi_set=self._image_roi_set,
            selected_roi_ids=sel_ids,
        )

    def _collect_point_sources_m(self) -> dict[str, "np.ndarray"]:
        """Collect available point sources as (N,2) arrays in metres."""
        return point_source_arrays_m(self._point_source_records())

    def _collect_point_sources_px(self) -> dict[str, "np.ndarray"]:
        """Collect available point sources as (N,2) arrays in pixel coordinates."""
        return point_source_arrays_px(self._point_source_records())

    def _collect_point_source_metadata(self) -> dict[str, dict[str, object]]:
        """Collect metadata for available point sources."""
        return point_source_metadata(self._point_source_records())

    def _on_open_pair_correlation(self) -> None:
        sources = self._collect_point_sources_m()
        if not sources:
            self._status_lbl.setText(
                "Run Feature finder or select point ROIs first."
            )
            return
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        px_x, px_y = self._pixel_size_xy_m()
        roi_area_m2 = None
        if arr is not None and self._image_roi_set is not None:
            roi_area_m2 = active_area_roi_area_m2(
                self._active_image_roi(),
                arr.shape[:2],
                pixel_size_x_m=px_x,
                pixel_size_y_m=px_y,
            )
        _, ch_unit, _ = self._channel_unit()
        entries = getattr(self, "_entries", [])
        entry = entries[self._idx] if entries else None
        from probeflow.gui.dialogs.pair_correlation import PairCorrelationDialog

        def _add(result):
            self._add_dialog_measurement_result(result)

        dlg = PairCorrelationDialog(
            sources,
            roi_area_m2=roi_area_m2,
            pixel_size_x_m=px_x,
            pixel_size_y_m=px_y,
            source_label=self._source_label(),
            source_path=str(entry.path) if entry is not None else None,
            channel=ch_unit,
            source_metadata=self._collect_point_source_metadata(),
            on_add_result=_add,
            theme=self._t,
            parent=self,
        )
        dlg.show()

    def _on_open_feature_lattice(self) -> None:
        sources = self._collect_point_sources_px()
        if not sources:
            self._status_lbl.setText(
                "Run Feature finder or select point ROIs first."
            )
            return
        item = getattr(self, "_lattice_grid_item", None)
        if item is None:
            self._status_lbl.setText("Open the Lattice/Grid tool first.")
            return
        grid = item.grid()
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        px_x, px_y = self._pixel_size_xy_m()
        _, ch_unit, _ = self._channel_unit()
        entries = getattr(self, "_entries", [])
        entry = entries[self._idx] if entries else None
        from probeflow.gui.dialogs.feature_lattice_dialog import FeatureLatticeDialog

        def _add(result):
            self._add_dialog_measurement_result(result)

        dlg = FeatureLatticeDialog(
            sources,
            lattice_origin_px=grid.origin_px,
            a_px=grid.a_px,
            b_px=grid.b_px,
            pixel_size_x_m=px_x,
            pixel_size_y_m=px_y,
            image_shape=arr.shape[:2] if arr is not None else None,
            source_label=self._source_label(),
            source_path=str(entry.path) if entry is not None else None,
            channel=ch_unit,
            source_metadata=self._collect_point_source_metadata(),
            on_add_result=_add,
            theme=self._t,
            parent=self,
        )
        dlg.show()

    def _on_open_fft_viewer(self):
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return
        dlg = FFTViewerDialog(
            arr,
            self._scan_range_m or (1e-9, 1e-9),
            colormap=self._colormap,
            theme=self._t,
            channel_unit=self._channel_unit(),
            parent=self,
        )
        dlg.show()

    def _on_periodic_filter(self):
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image data available for FFT filtering.")
            return
        dlg = PeriodicFilterDialog(
            arr,
            peaks=self._processing.get("periodic_notches", []),
            radius_px=float(self._processing.get("periodic_notch_radius", 3.0)),
            scan_range_m=self._scan_range_m,
            colormap=self._colormap,
            theme=self._t,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        peaks = dlg.selected_peaks()
        if peaks:
            self._processing["periodic_notches"] = peaks
            self._processing["periodic_notch_radius"] = dlg.radius_px()
            self._status_lbl.setText(f"Periodic FFT filter: {len(peaks)} peak(s) selected.")
        else:
            self._processing.pop("periodic_notches", None)
            self._processing.pop("periodic_notch_radius", None)
            self._status_lbl.setText("Periodic FFT filter cleared.")
        self._refresh_processing_display()

    def _on_apply_processing(self):
        panel_state = self._processing_panel.state()
        panel_state.update(self._advanced_processing_state())
        has_roi_aware_local_filter = self._processing_has_roi_aware_local_filter(panel_state)
        active_roi = self._active_image_roi()
        active_area_roi_id = active_area_roi_context(self._image_roi_set).roi_id
        wants_filter_roi = self._scope_cb.currentIndex() == 1
        if (
            wants_filter_roi
            and active_roi is not None
            and active_area_roi_id is None
            and has_roi_aware_local_filter
        ):
            self._status_lbl.setText(
                f"Active {active_roi.kind} ROI is not valid for area processing; "
                "select an area ROI or delete/deselect it before applying local filters."
            )
            return
        if wants_filter_roi:
            if active_area_roi_id is None:
                self._status_lbl.setText("Select an active area ROI before using ROI filters.")
                return
        # Snapshot for undo before any mutation. Validation has passed; this
        # apply is going to change the state.
        self._push_proc_undo_snapshot()
        preserve = {
            key: self._processing[key]
            for key in (
                "set_zero_xy",
                "set_zero_plane_points",
                "set_zero_patch",
                "periodic_notches",
                "periodic_notch_radius",
                "geometric_ops",
                "stm_background",
                "plane_bg",
            )
            if key in self._processing
        }
        self._processing = panel_state
        self._processing.update(preserve)
        if wants_filter_roi and active_area_roi_id is not None:
            self._processing["processing_scope"] = "roi"
            self._processing["processing_roi_id"] = active_area_roi_id
        else:
            self._processing.pop("processing_scope", None)
            self._processing.pop("processing_roi_id", None)
        self._clear_bad_line_preview()
        self._refresh_processing_display()

    def _on_reset_processing(self):
        """Clear all processing for the current image and reload raw data."""
        has_zero = bool(self._zero_ctrl.points)
        if not self._processing and not has_zero:
            self._status_lbl.setText("Already showing the original — nothing to reset.")
            return
        # Snapshot for undo before clearing.
        self._push_proc_undo_snapshot()
        self._processing = {}
        self._processing_panel.set_state({})
        self._set_advanced_processing_state({})
        self._clear_bad_line_preview()
        # Untoggle any active set-zero pick modes so we don't re-pick on reload.
        if self._set_zero_plane_btn.isChecked():
            self._set_zero_plane_btn.setChecked(False)
        self._zero_ctrl.clear()
        self._set_selection_tool("none")
        self._scope_cb.setCurrentIndex(0)
        self._roi_status_lbl.setText("ROI filter scope: whole image")
        self._refresh_zero_markers()
        self._status_lbl.setText("Reset: showing original on-disk data.")
        self._refresh_processing_display()

    # ── Processing undo / redo ────────────────────────────────────────────────

    def _push_proc_undo_snapshot(self) -> None:
        self._proc_undo_ctrl.push(self._processing)

    def _restore_processing_state(self, state: dict) -> None:
        """Apply a snapshot to ``self._processing`` and resync the GUI."""
        self._processing = copy.deepcopy(state)
        self._processing_panel.set_state(self._processing)
        self._set_advanced_processing_state(self._processing)
        self._refresh_processing_display()

    def _on_undo_processing(self) -> None:
        state = self._proc_undo_ctrl.undo(self._processing)
        if state is None:
            return
        self._restore_processing_state(state)
        self._status_lbl.setText("Undo: restored previous processing.")

    def _on_redo_processing(self) -> None:
        state = self._proc_undo_ctrl.redo(self._processing)
        if state is None:
            return
        self._restore_processing_state(state)
        self._status_lbl.setText("Redo: reapplied processing.")

    def _update_undo_redo_buttons(self) -> None:
        if self._proc_undo_ctrl is not None:
            self._proc_undo_ctrl.update_buttons()

    def _on_save_png(self):
        entry = self._entries[self._idx]
        if getattr(self, "_processing_roi_error", ""):
            self._status_lbl.setText(
                f"Cannot export while processing has stale ROI references. {self._processing_roi_error}"
            )
            return
        if getattr(self, "_processing_error", ""):
            self._status_lbl.setText(f"Export blocked: {self._processing_error}")
            return
        try:
            ps = processing_state_from_gui(self._processing or {})
            assert_roi_references_resolved(ps, self._image_roi_set)
        except ValueError as _roi_err:
            self._status_lbl.setText(f"Export blocked: {_roi_err}")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save PNG", str(Path.home() / f"{entry.stem}_viewer.png"),
            "PNG images (*.png)")
        if not out_path:
            return
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No data to save.")
            return
        msg = save_viewer_png(
            arr, out_path, entry.path,
            self._colormap, self._clip_low, self._clip_high,
            self._drs, self._processing, self._image_roi_set,
            self._ch_cb.currentIndex(), self._ch_cb.currentText() or None,
            processing_history=(
                self._processing_history.to_dict()
                if self._processing_history is not None else None
            ),
        )
        if msg.startswith("Saved") and self._processing_history is not None:
            self._mark_history_export(out_path, export_parameters={"export_kind": "viewer_png"})
        self._status_lbl.setText(msg)

    def _assert_exportable_processing(self) -> bool:
        if getattr(self, "_processing_roi_error", ""):
            self._status_lbl.setText(
                f"Cannot export while processing has stale ROI references. {self._processing_roi_error}"
            )
            return False
        if getattr(self, "_processing_error", ""):
            self._status_lbl.setText(f"Export blocked: {self._processing_error}")
            return False
        try:
            ps = processing_state_from_gui(self._processing or {})
            assert_roi_references_resolved(ps, self._image_roi_set)
        except ValueError as _roi_err:
            self._status_lbl.setText(f"Export blocked: {_roi_err}")
            return False
        return True

    def _current_display_settings(self) -> dict:
        from probeflow.provenance.export import png_display_state

        return png_display_state(
            self._drs,
            clip_low=self._clip_low,
            clip_high=self._clip_high,
            colormap=self._viewer_colormap,
            add_scalebar=True,
            scalebar_unit="nm",
            scalebar_pos="bottom-right",
        )

    def _processed_scan_for_export(self):
        entry = self._entries[self._idx]
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        return build_processed_scan_for_export(
            entry.path, self._ch_cb.currentIndex(), arr, self._processing or {},
        )

    def _on_save_processed_image(self):
        if not self._assert_exportable_processing():
            return
        entry = self._entries[self._idx]
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save processed image",
            str(Path.home() / f"{entry.stem}_processed.sxm"),
            (
                "Supported images (*.sxm *.png *.csv *.pdf *.gwy);;"
                "Nanonis SXM (*.sxm);;PNG images (*.png);;"
                "CSV grids (*.csv);;PDF figures (*.pdf);;Gwyddion (*.gwy)"
            ),
        )
        if not out_path:
            return
        out = Path(out_path)
        if not out.suffix:
            out = out.with_suffix(".sxm")
        try:
            scan, plane_idx = self._processed_scan_for_export()
        except ValueError as exc:
            self._status_lbl.setText(str(exc))
            return
        msg = save_processed_image(
            scan, plane_idx, out,
            colormap=self._viewer_colormap,
            clip_low=self._clip_low,
            clip_high=self._clip_high,
            display_settings=self._current_display_settings(),
            roi_set=self._image_roi_set,
            processing_history=(
                self._processing_history.to_dict()
                if self._processing_history is not None else None
            ),
        )
        self._status_lbl.setText(msg)

    def _on_save_provenance(self):
        if not self._assert_exportable_processing():
            return
        if self._processing_history is None:
            self._status_lbl.setText("No provenance available to save.")
            return
        entry = self._entries[self._idx]
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save provenance",
            str(Path.home() / f"{entry.stem}.probeflow.json"),
            "ProbeFlow provenance (*.probeflow.json *.json)",
        )
        if not out_path:
            return
        out = Path(out_path)
        if not out.suffix:
            out = out.with_suffix(".probeflow.json")
        try:
            msg, record = save_provenance_json(
                self._processing_history,
                out,
                display_settings=self._current_display_settings(),
            )
            self._last_export_record = record
            self._history_text.setText(
                "\n".join(display_lines(record.processing_history))
            )
            self._status_lbl.setText(msg)
        except Exception as exc:
            self._status_lbl.setText(f"Save provenance error: {exc}")

    def _mark_history_export(self, out_path: str, export_parameters: dict | None = None) -> None:
        try:
            record = build_export_record(
                self._processing_history,
                export_path=out_path,
                export_format="png",
                display_settings=self._current_display_settings(),
                export_parameters=export_parameters,
            )
            self._last_export_record = record
            self._history_text.setText(
                "\n".join(display_lines(record.processing_history))
            )
        except Exception:
            pass

    def _on_send_to_features(self):
        self._deferred.action = "features"
        self._deferred.plane_idx = self._ch_cb.currentIndex()
        self.accept()

    def _on_send_to_tv(self):
        self._deferred.action = "tv"
        self._deferred.plane_idx = self._ch_cb.currentIndex()
        self.accept()

    def _on_image_context_menu(self, pos):
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QAction
        menu = QMenu(self)
        a_feat = QAction("→ Feature Counting", self)
        a_feat.setToolTip("Send processed image to Feature Counting tab")
        a_feat.triggered.connect(self._on_send_to_features)
        menu.addAction(a_feat)
        a_tv = QAction("→ TV Denoising", self)
        a_tv.setToolTip("Send processed image to TV Denoising tab")
        a_tv.triggered.connect(self._on_send_to_tv)
        menu.addAction(a_tv)
        menu.addSeparator()
        from PySide6.QtWidgets import QMenu as _QMenu
        transform_menu = _QMenu("Transform", self)
        for label, op in [
            ("Flip Horizontal", "flip_horizontal"),
            ("Flip Vertical",   "flip_vertical"),
            ("Rotate 90° CW",   "rotate_90_cw"),
            ("Rotate 180°",     "rotate_180"),
            ("Rotate 270° CW",  "rotate_270_cw"),
        ]:
            act = transform_menu.addAction(label)
            act.triggered.connect(
                (lambda _op=op: lambda: self._on_geometric_op(_op))()
            )
        arb_act = transform_menu.addAction("Rotate Arbitrary…")
        arb_act.triggered.connect(self._on_rotate_arbitrary)
        menu.addMenu(transform_menu)
        menu.addSeparator()
        a_png = QAction("⬇ Save PNG copy…", self)
        a_png.triggered.connect(self._on_save_png)
        menu.addAction(a_png)
        prof = self._line_profile_panel.profile_data()
        if prof is not None:
            a_csv = QAction("Export line profile as CSV…", self)
            a_csv.triggered.connect(self._on_export_line_profile_csv)
            menu.addAction(a_csv)
        menu.exec(pos)

    def _on_geometric_op(self, op_name: str) -> None:
        self._transform_image_roi_set_for_display_op(op_name)
        ops = list(self._processing.get("geometric_ops") or [])
        ops.append({"op": op_name, "params": {}})
        self._processing["geometric_ops"] = ops
        self._refresh_processing_display()

    def _on_rotate_arbitrary(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        angle, ok = QInputDialog.getDouble(
            self, "Rotate Arbitrary",
            "Angle (degrees, positive = counter-clockwise):",
            0.0, -360.0, 360.0, 1,
        )
        if not ok:
            return
        self._transform_image_roi_set_for_display_op(
            "rotate_arbitrary",
            {"angle_degrees": angle},
        )
        ops = list(self._processing.get("geometric_ops") or [])
        ops.append({"op": "rotate_arbitrary", "params": {"angle_degrees": angle}})
        self._processing["geometric_ops"] = ops
        self._refresh_processing_display()

    def _transform_image_roi_set_for_display_op(
        self,
        op_name: str,
        params: dict | None = None,
    ) -> None:
        transform_roi_set_for_display_op(
            self._image_roi_set,
            op_name,
            params,
            self._current_array_shape(),
            status_fn=(
                self._status_lbl.setText if hasattr(self, "_status_lbl") else None
            ),
            roi_changed_fn=self._on_image_roi_set_changed,
        )

    def _on_export_line_profile_csv(self):
        prof = self._line_profile_panel.profile_data()
        if prof is None:
            self._status_lbl.setText("No line profile to export (draw a line first).")
            return
        x_vals, y_vals, x_label, y_label = prof
        entry = self._entries[self._idx]
        ok, msg = export_line_profile(
            x_vals, y_vals, x_label, y_label,
            entry.stem,
            self._scan_header or {},
            parent=self,
        )
        if msg:
            self._status_lbl.setText(msg)

    def closeEvent(self, event):
        # Invalidate the in-flight worker token so any pending loaded() signal
        # is dropped rather than delivered to widgets that are being torn down.
        self._token = object()
        # Close any modeless child dialogs that hold a reference to self or to
        # the currently displayed Scan.  Without this they outlive the viewer.
        stm_dlg = getattr(self, "_stm_background_dialog", None)
        if stm_dlg is not None and stm_dlg.isVisible():
            stm_dlg.close()
        super().closeEvent(event)
