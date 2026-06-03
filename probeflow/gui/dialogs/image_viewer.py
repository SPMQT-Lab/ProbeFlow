"""ImageViewerDialog — double-click viewer with scroll/zoom, histogram, processing, export.

Extracted from the historical probeflow.gui._legacy as part of the ongoing
GUI refactor (now re-exported via ``probeflow.gui.compat``).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

import os as _os
_os.environ.setdefault("QT_API", "pyside6")

from probeflow.gui.typography import ui_font
from PySide6.QtCore import (
    Qt, QThreadPool,
    Signal, Slot,
)
from PySide6.QtGui import (
    QAction, QActionGroup, QCursor, QFont, QKeySequence,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox,
    QDialog, QDockWidget, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMenu, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QStackedWidget,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QToolButton, QVBoxLayout, QWidget,
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
from probeflow.gui.desktop_layout import (
    apply_screen_fraction_geometry,
    b64_to_qbytearray,
    qbytearray_to_b64,
    restore_geometry_or_default,
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
    resolve_channel_unit,
)
from probeflow.gui.widgets import ImageMeasurementsPanel
from probeflow.gui.image_canvas import ImageCanvas
from probeflow.gui.roi_manager_dock import ROIManagerPanel
from probeflow.gui.viewer.floating_panel import FloatingPanelManager, ModalOverlay
from probeflow.processing.gui_adapter import processing_state_from_gui
from probeflow.processing.state import (
    apply_processing_state_with_calibration,
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
    active_line_roi_context,
    area_roi_mask,
    collect_point_source_records,
    point_source_arrays_m,
    point_source_arrays_px,
    point_source_metadata,
    selected_or_active_area_roi_context,
)
from probeflow.gui.viewer.tool_launch import (
    feature_lattice_launch_context,
    lattice_grid_launch_context,
    pair_correlation_launch_context,
)
from probeflow.gui.viewer.shortcuts import viewer_command
from probeflow.core.scan_loader import load_scan
from probeflow.gui.viewer.scan_load import load_scan_for_viewer, ViewerScanData
from probeflow.gui.viewer.image_viewer_display_mixin import ImageViewerDisplayMixin
from probeflow.gui.viewer.image_viewer_processing_export_mixin import (
    ImageViewerProcessingExportMixin,
)
from probeflow.gui.viewer.image_viewer_roi_mixin import ImageViewerRoiMixin
from probeflow.gui.viewer.image_viewer_toolbar_mixin import ImageViewerToolbarMixin
from probeflow.gui.viewer.image_viewer_tools_mixin import ImageViewerToolsMixin
from probeflow.gui.viewer.window_menu import populate_window_menu

# Dialogs imported from their specific submodule files to avoid circular imports
# (this module lives inside probeflow.gui.dialogs).
from probeflow.gui.dialogs.about import AboutDialog
from probeflow.gui.dialogs.definitions import _DefinitionsDialog
from probeflow.gui.dialogs.fft_viewer import FFTViewerDialog
from probeflow.gui.dialogs.periodic_filter import PeriodicFilterDialog
from probeflow.gui.dialogs.stm_background import STMBackgroundDialog


class _ElidedLabel(QLabel):
    """Single-line label that keeps the full text available as a tooltip."""

    def __init__(self, text: str = "", parent=None):
        super().__init__("", parent)
        self._full_text = ""
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self._refresh_elide()

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._refresh_elide()

    def _refresh_elide(self) -> None:
        width = max(24, self.width() - 2)
        super().setText(
            self.fontMetrics().elidedText(self._full_text, Qt.ElideMiddle, width)
        )


class ImageViewerDialog(
    ImageViewerRoiMixin,
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

    # ── Build ──────────────────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # title
        # Compact header: file name and measurement conditions share one row.
        header_row = QHBoxLayout()
        header_row.setContentsMargins(2, 0, 2, 0)
        header_row.setSpacing(10)
        self._title_lbl = QLabel()
        self._title_lbl.setFont(ui_font(12, weight=QFont.Bold))
        header_row.addWidget(self._title_lbl)
        self._conditions_lbl = QLabel()
        self._conditions_lbl.setFont(ui_font(9))
        self._conditions_lbl.setStyleSheet("color: palette(mid);")
        self._conditions_lbl.setAlignment(Qt.AlignVCenter)
        header_row.addWidget(self._conditions_lbl)
        header_row.addStretch(1)
        root.addLayout(header_row)

        # main splitter: image | right panel
        self._viewer_splitter = QSplitter(Qt.Horizontal)
        splitter = self._viewer_splitter
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(5)

        # ── Left: scrollable zoom image ────────────────────────────────────────
        left = QWidget()
        # Floating tool panels hover over this canvas column.
        self._canvas_host = left
        left.setMinimumWidth(600)
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
        self._zoom_out_btn.setFont(ui_font(11))
        self._zoom_out_btn.setToolTip("Zoom out")
        self._zoom_out_btn.clicked.connect(lambda: self._zoom_lbl.zoom_by(1 / 1.25))
        toolbar.addWidget(self._zoom_out_btn)

        self._zoom_reset_btn = QPushButton("1:1")
        self._zoom_reset_btn.setMinimumWidth(42)
        self._zoom_reset_btn.setFixedHeight(26)
        self._zoom_reset_btn.setFont(ui_font(9))
        self._zoom_reset_btn.setToolTip("Reset to native raster size")
        self._zoom_reset_btn.clicked.connect(self._zoom_lbl.reset_zoom)
        toolbar.addWidget(self._zoom_reset_btn)

        self._zoom_fit_btn = QPushButton("Fit")
        self._zoom_fit_btn.setMinimumWidth(40)
        self._zoom_fit_btn.setFixedHeight(26)
        self._zoom_fit_btn.setFont(ui_font(9))
        self._zoom_fit_btn.setToolTip("Fit image to available space")
        self._zoom_fit_btn.clicked.connect(self._zoom_lbl.fit_to_view)
        toolbar.addWidget(self._zoom_fit_btn)

        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setFixedSize(30, 26)
        self._zoom_in_btn.setFont(ui_font(11))
        self._zoom_in_btn.setToolTip("Zoom in")
        self._zoom_in_btn.clicked.connect(lambda: self._zoom_lbl.zoom_by(1.25))
        toolbar.addWidget(self._zoom_in_btn)

        channel_lbl = QLabel("Channel")
        channel_lbl.setFont(ui_font(8, weight=QFont.Bold))
        toolbar.addSpacing(8)
        toolbar.addWidget(channel_lbl)

        self._ch_cb = QComboBox()
        self._ch_cb.addItems(PLANE_NAMES)
        self._ch_cb.setFont(ui_font(8))
        self._ch_cb.setMinimumWidth(170)
        self._ch_cb.currentIndexChanged.connect(self._on_channel_changed)
        toolbar.addWidget(self._ch_cb)

        # Per-image colormap — does not affect browser thumbnails
        cmap_lbl = QLabel("Colormap")
        cmap_lbl.setFont(ui_font(8, weight=QFont.Bold))
        toolbar.addSpacing(8)
        toolbar.addWidget(cmap_lbl)
        self._viewer_cmap_cb = QComboBox()
        self._viewer_cmap_cb.addItems(CMAP_NAMES)
        self._viewer_cmap_cb.setFont(ui_font(8))
        _initial_cmap_label = next(
            (lbl for lbl, k in STM_COLORMAPS
             if k == self._viewer_colormap or lbl == self._viewer_colormap),
            DEFAULT_CMAP_LABEL,
        )
        self._viewer_cmap_cb.setCurrentText(_initial_cmap_label)
        self._viewer_cmap_cb.currentTextChanged.connect(self._on_viewer_colormap_changed)
        toolbar.addWidget(self._viewer_cmap_cb)

        self._coord_lbl = QLabel("—")
        self._coord_lbl.setFont(ui_font(8))
        self._coord_lbl.setMinimumWidth(140)
        toolbar.addWidget(self._coord_lbl)

        zoom_hint = QLabel("Ctrl+scroll to zoom")
        zoom_hint.setFont(ui_font(8))
        toolbar.addWidget(zoom_hint)
        toolbar.addStretch()
        # Visible entry point for the command finder (also Ctrl+K / Help menu) so
        # new users discover it.
        search_btn = QPushButton("⌕ Search")
        search_btn.setToolTip("Find and run any command  (Ctrl+K)")
        search_btn.setDefault(False)
        search_btn.setAutoDefault(False)
        search_btn.clicked.connect(self._show_command_finder)
        toolbar.addWidget(search_btn)
        help_btn = QPushButton("?")
        help_btn.setFixedSize(24, 24)
        help_btn.setToolTip("Show image viewer shortcuts")
        help_btn.clicked.connect(self._show_image_viewer_shortcuts)
        toolbar.addWidget(help_btn)
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
        right.setMinimumWidth(340)
        self._sidebar_panel = right
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 4, 8, 4)
        right_lay.setSpacing(6)

        self._sidebar_tabs = QTabWidget()
        self._sidebar_tabs.setDocumentMode(True)
        self._sidebar_tabs.setMinimumWidth(320)
        # Elide (not clip) when the row is tight so every tab stays visible and
        # clickable — never silently drop a tab off the right edge.
        self._sidebar_tabs.setElideMode(Qt.ElideRight)
        self._sidebar_tabs.tabBar().setUsesScrollButtons(False)
        self._sidebar_tabs.tabBar().setExpanding(False)
        right_lay.addWidget(self._sidebar_tabs, 1)
        self._sidebar_tab_indices: dict[str, int] = {}
        # (key, label, tooltip) in tab order — also drives the collapsed rail.
        self._sidebar_tab_meta: list[tuple[str, str, str]] = []

        # Collapse chevron lives in the tab bar's corner.
        self._sidebar_collapse_btn = QToolButton()
        self._sidebar_collapse_btn.setObjectName("sidebarCollapseBtn")
        self._sidebar_collapse_btn.setText("›")
        self._sidebar_collapse_btn.setFixedSize(30, 30)
        self._sidebar_collapse_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._sidebar_collapse_btn.setToolTip("Collapse the panel to widen the image")
        self._sidebar_collapse_btn.setAutoRaise(True)
        self._sidebar_collapse_btn.clicked.connect(lambda: self._set_sidebar_collapsed(True))
        self._sidebar_tabs.setCornerWidget(self._sidebar_collapse_btn, Qt.TopRightCorner)

        def _sidebar_tab(key: str, label: str, tip: str = "") -> tuple[QWidget, QVBoxLayout]:
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
            if tip:
                self._sidebar_tabs.setTabToolTip(idx, tip)
            self._sidebar_tab_indices[key] = idx
            self._sidebar_tab_meta.append((key, label, tip))
            return body, lay

        _display_tab, display_lay = _sidebar_tab(
            "display", "View",
            "Colormap, contrast, histogram and ROI overlay visibility.",
        )
        _processing_tab, processing_lay = _sidebar_tab(
            "processing", "Process",
            "Line corrections, background subtraction, filters and FFT tools.",
        )
        _roi_tab, roi_lay = _sidebar_tab(
            "roi", "ROI",
            "Create, edit and combine regions of interest.",
        )
        _measurements_tab, measurements_lay = _sidebar_tab(
            "measurements", "Measure",
            "Distances, angles, ROI statistics, features and results.",
        )
        _export_tab, export_lay = _sidebar_tab(
            "export", "Export",
            "Save images (PNG/PDF/SXM/GWY), provenance and hand-off to tools.",
        )

        def _collapsible_section(
            target_lay: QVBoxLayout,
            title: str,
            expanded: bool = False,
        ):
            btn = QPushButton(("[−] " if expanded else "[+] ") + title)
            btn.setCheckable(True)
            btn.setChecked(expanded)
            btn.setFlat(True)
            btn.setFont(ui_font(9, weight=QFont.Bold))
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
            lbl.setFont(ui_font(8))
            spin = QDoubleSpinBox()
            spin.setRange(float(mn), float(mx))
            spin.setDecimals(decimals)
            spin.setSingleStep(float(step))
            spin.setValue(float(init))
            spin.setFont(ui_font(8))
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

        # ── Per-region contrast scope + overlay visibility ────────────────────
        disp_scope_row = QHBoxLayout()
        disp_scope_row.setSpacing(6)
        disp_scope_row.setContentsMargins(0, 0, 0, 0)
        _disp_scope_lbl = QLabel("Contrast scope")
        _disp_scope_lbl.setFont(ui_font(8))
        self._display_scope_cb = QComboBox()
        self._display_scope_cb.addItems(["Whole image", "Active ROI"])
        self._display_scope_cb.setFont(ui_font(8))
        self._display_scope_cb.setToolTip(
            "Active ROI: the brightness/contrast sliders edit the active area\n"
            "ROI's own range. Each region is composited with its own scaling,\n"
            "so a split scan can show both areas well in one image."
        )
        self._display_scope_cb.currentIndexChanged.connect(self._on_display_scope_changed)
        disp_scope_row.addWidget(_disp_scope_lbl)
        disp_scope_row.addWidget(self._display_scope_cb, 1)
        display_lay.addLayout(disp_scope_row)

        self._hide_rois_cb = QCheckBox("Hide ROI overlays")
        self._hide_rois_cb.setFont(ui_font(8))
        self._hide_rois_cb.setToolTip(
            "Hide every ROI overlay so the composited image can be inspected.\n"
            "ROIs are unchanged; untick to show them again."
        )
        self._hide_rois_cb.toggled.connect(self._on_toggle_rois_hidden)
        display_lay.addWidget(self._hide_rois_cb)

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

        # Processing history is shown via Image → Info; keep the label alive (hidden,
        # owned by the dialog) so its existing updaters stay safe without cluttering
        # the Process tab.
        self._history_text = QLabel("", self)
        self._history_text.setFont(ui_font(8))
        self._history_text.setWordWrap(True)
        self._history_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._history_text.hide()

        processing_lay.addWidget(_sep())

        # ── Zero reference | ROI filter scope (compact 2-column row) ──────────
        zs_row = QHBoxLayout()
        zs_row.setSpacing(6)
        zs_row.setContentsMargins(0, 0, 0, 0)

        zero_col = QVBoxLayout()
        zero_col.setSpacing(3)
        zero_col.setContentsMargins(0, 0, 0, 0)
        _zero_hdr = QLabel("Zero ref.")
        _zero_hdr.setFont(ui_font(7, weight=QFont.Bold))
        _zero_hdr.setAlignment(Qt.AlignCenter)
        zero_col.addWidget(_zero_hdr)
        self._set_zero_plane_btn = QPushButton("Set zero plane")
        self._set_zero_plane_btn.setCheckable(True)
        self._set_zero_plane_btn.setFont(ui_font(8))
        self._set_zero_plane_btn.setFixedHeight(24)
        self._set_zero_plane_btn.setToolTip("Click 3 points on the image to define a zero-height plane.")
        self._set_zero_plane_btn.toggled.connect(self._on_set_zero_plane_mode_toggled)
        zero_col.addWidget(self._set_zero_plane_btn)
        self._set_zero_clear_btn = QPushButton("Hide Points")
        self._set_zero_clear_btn.setFont(ui_font(8))
        self._set_zero_clear_btn.setFixedHeight(22)
        self._set_zero_clear_btn.setToolTip(
            "Hide zero-plane reference point markers; processing is unchanged."
        )
        self._set_zero_clear_btn.clicked.connect(self._on_clear_set_zero)
        zero_col.addWidget(self._set_zero_clear_btn)
        zero_col.addStretch()

        sel_col = QVBoxLayout()
        sel_col.setSpacing(3)
        sel_col.setContentsMargins(0, 0, 0, 0)
        _sel_hdr = QLabel("ROI filters")
        _sel_hdr.setFont(ui_font(7, weight=QFont.Bold))
        _sel_hdr.setAlignment(Qt.AlignCenter)
        sel_col.addWidget(_sel_hdr)
        self._scope_cb = QComboBox()
        self._scope_cb.addItems(["Whole image", "ROI filters only"])
        self._scope_cb.setFont(ui_font(8))
        self._scope_cb.setToolTip(
            "ROI filters only: smooth/high-pass/edge/FFT apply inside the "
            "active area ROI; background and scan-line corrections remain whole-image.")
        sel_col.addWidget(self._scope_cb)
        sel_col.addStretch()

        zs_row.addLayout(zero_col, 1)
        zs_row.addLayout(sel_col, 1)
        processing_lay.addLayout(zs_row)

        self._roi_status_lbl = QLabel("ROI filter scope: whole image")
        self._roi_status_lbl.setFont(ui_font(8))
        self._roi_status_lbl.setWordWrap(True)
        processing_lay.addWidget(self._roi_status_lbl)

        processing_lay.addWidget(_sep())

        # ── Apply / Reset — always visible ────────────────────────────────────
        ar_row = QHBoxLayout()
        ar_row.setSpacing(4)
        proc_apply_btn = QPushButton("Apply processing")
        proc_apply_btn.setFont(ui_font(8, weight=QFont.Bold))
        proc_apply_btn.setFixedHeight(28)
        proc_apply_btn.setObjectName("accentBtn")
        proc_apply_btn.setToolTip(
            "Apply queued in-panel filters and bad-line correction settings. "
            "Align rows updates immediately; STM Background has its own Apply."
        )
        proc_apply_btn.clicked.connect(self._on_apply_processing)
        proc_reset_btn = QPushButton("Reset")
        proc_reset_btn.setFont(ui_font(8))
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
        self._proc_undo_btn.setFont(ui_font(8))
        self._proc_undo_btn.setFixedHeight(24)
        self._proc_undo_btn.setToolTip(
            "Restore the processing state from before the last Apply / Reset "
            "(Ctrl+Z).")
        self._proc_undo_btn.clicked.connect(self._on_undo_processing)
        self._proc_redo_btn = QPushButton("Redo ↷")
        self._proc_redo_btn.setFont(ui_font(8))
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

        processing_lay.addWidget(_sep())

        # ── Save PNG — always visible ─────────────────────────────────────────
        self._save_png_btn = QPushButton("⬇  Save PNG copy…")
        self._save_png_btn.setFont(ui_font(8, weight=QFont.Bold))
        self._save_png_btn.setFixedHeight(26)
        self._save_png_btn.setObjectName("accentBtn")
        self._save_png_btn.clicked.connect(self._on_save_png)
        export_lay.addWidget(self._save_png_btn)

        summary = QWidget()
        summary_lay = QGridLayout(summary)
        summary_lay.setContentsMargins(2, 4, 2, 4)
        summary_lay.setHorizontalSpacing(8)
        summary_lay.setVerticalSpacing(2)

        def _summary_row(row: int, name: str, attr: str, *, elide: bool = False) -> QLabel:
            key_lbl = QLabel(name)
            key_lbl.setFont(ui_font(8, weight=QFont.Bold))
            key_lbl.setStyleSheet("color: palette(mid);")
            val_lbl = _ElidedLabel("--") if elide else QLabel("--")
            val_lbl.setFont(ui_font(8))
            val_lbl.setWordWrap(False)
            val_lbl.setSizePolicy(
                QSizePolicy.Ignored if elide else QSizePolicy.Preferred,
                QSizePolicy.Preferred,
            )
            val_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            val_lbl.setObjectName(attr)
            summary_lay.addWidget(key_lbl, row, 0)
            summary_lay.addWidget(val_lbl, row, 1)
            setattr(self, attr, val_lbl)
            return val_lbl

        _summary_row(0, "PNG", "_export_png_size_lbl")
        _summary_row(1, "File", "_export_png_file_lbl", elide=True)
        _summary_row(2, "Bias", "_export_bias_lbl")
        _summary_row(3, "Current", "_export_current_lbl")
        _summary_row(4, "Data", "_export_precision_lbl")
        summary_lay.setColumnStretch(1, 1)
        export_lay.addWidget(summary)

        self._export_provenance_chk = QCheckBox("Write provenance")
        self._export_provenance_chk.setFont(ui_font(8))
        self._export_provenance_chk.setChecked(True)
        export_lay.addWidget(self._export_provenance_chk)

        self._export_scalebar_chk = QCheckBox("Include scale bar")
        self._export_scalebar_chk.setFont(ui_font(8))
        self._export_scalebar_chk.setChecked(True)
        export_lay.addWidget(self._export_scalebar_chk)

        self._save_pdf_btn = QPushButton("Save PDF copy…")
        self._save_pdf_btn.setFont(ui_font(8, weight=QFont.Bold))
        self._save_pdf_btn.setFixedHeight(26)
        self._save_pdf_btn.clicked.connect(self._on_save_pdf)
        export_lay.addWidget(self._save_pdf_btn)

        self._save_sxm_btn = QPushButton("Save SXM copy…")
        self._save_sxm_btn.setFont(ui_font(8, weight=QFont.Bold))
        self._save_sxm_btn.setFixedHeight(26)
        self._save_sxm_btn.clicked.connect(self._on_save_sxm)
        export_lay.addWidget(self._save_sxm_btn)

        self._save_gwy_btn = QPushButton("Save GWY copy…")
        self._save_gwy_btn.setFont(ui_font(8, weight=QFont.Bold))
        self._save_gwy_btn.setFixedHeight(26)
        self._save_gwy_btn.setToolTip(
            "Export a Gwyddion .gwy file. Requires the optional gwyfile package."
        )
        self._save_gwy_btn.clicked.connect(self._on_save_gwy)
        export_lay.addWidget(self._save_gwy_btn)
        self._update_export_summary()

        # ── Send to tool (collapsible) ────────────────────────────────────────
        _, self._export_widget, send_lay = _collapsible_section(
            export_lay, "→ Send to tool", expanded=False
        )

        send_feat_btn = QPushButton("→ Feature Counting")
        send_feat_btn.setFont(ui_font(8))
        send_feat_btn.setFixedHeight(24)
        send_feat_btn.setToolTip(
            "Send the current processed image to Feature Counting (viewer stays open)")
        send_feat_btn.clicked.connect(self._on_send_to_features)
        send_lay.addWidget(send_feat_btn)

        send_tv_btn = QPushButton("→ TV Denoising")
        send_tv_btn.setFont(ui_font(8))
        send_tv_btn.setFixedHeight(24)
        send_tv_btn.setToolTip(
            "Send the current processed image to TV Denoising (viewer stays open)")
        send_tv_btn.clicked.connect(self._on_send_to_tv)
        send_lay.addWidget(send_tv_btn)

        # ── Advanced tools (opened as a dismissible overlay) ──────────────────
        # Built as a standalone, viewer-owned widget; an "Advanced…" button in the
        # Process tab presents it over the canvas so the tab stays focused on the
        # core flatten/background/filter workflow.
        advanced_btn = QPushButton("Advanced…")
        advanced_btn.setToolTip(
            "Periodic / radial FFT filters and linear drift undistortion."
        )
        advanced_btn.clicked.connect(self._open_advanced_tools)
        processing_lay.addWidget(advanced_btn)

        self._advanced_widget = QWidget(self)
        self._advanced_widget.setObjectName("advancedToolsPanel")
        self._advanced_widget.setMinimumWidth(320)
        advanced_lay = QVBoxLayout(self._advanced_widget)
        advanced_lay.setContentsMargins(8, 8, 8, 8)
        advanced_lay.setSpacing(6)
        self._advanced_widget.hide()
        _adv_title = QLabel("Advanced tools")
        _adv_title.setFont(ui_font(10, weight=QFont.Bold))
        advanced_lay.addWidget(_adv_title)

        periodic_btn = QPushButton("Periodic FFT filter…")
        periodic_btn.setFont(ui_font(8))
        periodic_btn.setFixedHeight(24)
        periodic_btn.clicked.connect(self._on_periodic_filter)
        advanced_lay.addWidget(periodic_btn)

        fft_viewer_btn = QPushButton("FFT viewer…")
        fft_viewer_btn.setFont(ui_font(8))
        fft_viewer_btn.setFixedHeight(24)
        fft_viewer_btn.setToolTip(
            "Open the FFT viewer to fit reciprocal Bragg peaks to a known "
            "structure and preview affine undistortion.")
        fft_viewer_btn.clicked.connect(self._on_open_fft_viewer)
        advanced_lay.addWidget(fft_viewer_btn)

        radial_fft_lbl = QLabel("Radial FFT")
        radial_fft_lbl.setFont(ui_font(7, weight=QFont.Bold))
        radial_fft_lbl.setAlignment(Qt.AlignCenter)
        advanced_lay.addWidget(radial_fft_lbl)
        fft_mode_row = QHBoxLayout()
        fft_mode_row.setContentsMargins(0, 0, 0, 0)
        fft_mode_lbl = QLabel("Mode:")
        fft_mode_lbl.setFont(ui_font(8))
        self._advanced_fft_combo = QComboBox()
        self._advanced_fft_combo.addItems(["None", "Low-pass", "High-pass"])
        self._advanced_fft_combo.setFont(ui_font(8))
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
        self._advanced_fft_soft_cb.setFont(ui_font(8))
        self._advanced_fft_soft_cb.setToolTip(
            "Cosine-taper the image edges before FFT to suppress ringing artefacts."
        )
        advanced_lay.addWidget(self._advanced_fft_soft_cb)

        undistort_lbl = QLabel("Linear undistort (drift)")
        undistort_lbl.setFont(ui_font(7, weight=QFont.Bold))
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
        self._spec_show_cb.setFont(ui_font(8))
        self._spec_show_cb.setChecked(False)
        self._spec_show_cb.toggled.connect(self._on_spec_show_toggled)
        spec_lay.addWidget(self._spec_show_cb)

        self._map_spectra_here_btn = QPushButton("Map spectra to this image…")
        self._map_spectra_here_btn.setFont(ui_font(8))
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
        self._zoom_lbl.object_hovered.connect(self._on_canvas_object_hovered)
        self._zoom_lbl.roi_created.connect(self._on_canvas_roi_created)
        self._zoom_lbl.roi_move_requested.connect(self._on_canvas_roi_move)
        self._zoom_lbl.roi_geometry_preview.connect(self._on_roi_geometry_preview)
        self._zoom_lbl.roi_geometry_changed.connect(self._on_roi_geometry_changed)
        self._zoom_lbl.roi_delete_requested.connect(self._on_canvas_roi_delete)
        self._zoom_lbl.roi_copy_requested.connect(self._on_canvas_roi_copy)
        self._zoom_lbl.roi_paste_requested.connect(self._on_canvas_roi_paste)
        self._zoom_lbl.roi_activate_requested.connect(self._on_canvas_roi_activate)
        self._zoom_lbl.tool_changed.connect(self._on_canvas_tool_changed)
        self._zoom_lbl.roi_context_menu_requested.connect(self._on_roi_canvas_context_menu)
        self._zoom_lbl.angle_points_ready.connect(self._on_angle_points_ready)
        self._line_profile_panel.export_csv_clicked.connect(self._on_export_line_profile_csv)
        self._line_profile_panel.width_changed.connect(self._on_line_profile_width_changed)

        roi_hint_lbl = QLabel(
            "Choose a drawing tool above to create an ROI. Click an ROI to select "
            "it, then drag the active ROI (or its handles) to edit. For a split "
            "scan, set the View tab's “Contrast scope” to “Active ROI” to give "
            "each region its own brightness/contrast, and use “Hide ROI "
            "overlays” there to inspect the result."
        )
        roi_hint_lbl.setFont(ui_font(8))
        roi_hint_lbl.setWordWrap(True)
        roi_hint_lbl.setStyleSheet("color: palette(mid);")
        roi_lay.addWidget(roi_hint_lbl)
        # The ROI manager panel itself is added to ``roi_lay`` after it is
        # constructed below (it needs the ROI-set getter and callbacks).

        # The Measure tab is driven entirely by the ImageMeasurementsPanel's own
        # tool menu (added to ``measurements_lay`` after it is constructed below);
        # its tool buttons emit signals the viewer connects to its handlers.

        self._status_lbl = QLabel("")
        self._status_lbl.setFont(ui_font(8))
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setText(
            "Tip: click ROIs to select them. Right-click the image or an ROI for actions."
        )
        right_lay.addWidget(self._status_lbl)

        display_lay.addStretch(1)
        processing_lay.addStretch(1)
        export_lay.addStretch(1)

        # Collapsed sidebar rail: a thin strip of tab buttons shown when the
        # full panel is hidden, so the image can take the full width.
        self._sidebar_rail = QWidget()
        self._sidebar_rail.setObjectName("sidebarRail")
        self._sidebar_rail.setFixedWidth(self._SIDEBAR_RAIL_W)
        rail_lay = QVBoxLayout(self._sidebar_rail)
        rail_lay.setContentsMargins(3, 4, 3, 4)
        rail_lay.setSpacing(4)

        expand_btn = QToolButton()
        expand_btn.setObjectName("sidebarExpandBtn")
        expand_btn.setText("‹")
        expand_btn.setCursor(QCursor(Qt.PointingHandCursor))
        expand_btn.setToolTip("Expand the panel")
        expand_btn.setAutoRaise(True)
        expand_btn.clicked.connect(lambda: self._set_sidebar_collapsed(False))
        rail_lay.addWidget(expand_btn)

        rail_sep = QFrame()
        rail_sep.setFrameShape(QFrame.HLine)
        rail_sep.setFrameShadow(QFrame.Sunken)
        rail_lay.addWidget(rail_sep)

        _rail_abbrev = {
            "View": "View", "Process": "Proc", "ROI": "ROI",
            "Measure": "Meas", "Export": "Exp",
        }
        for _key, _label, _tip in self._sidebar_tab_meta:
            rail_btn = QToolButton()
            rail_btn.setObjectName("sidebarRailBtn")
            rail_btn.setText(_rail_abbrev.get(_label, _label[:4]))
            rail_btn.setFont(ui_font(8))
            rail_btn.setToolTip(_tip or _label)
            rail_btn.setCursor(QCursor(Qt.PointingHandCursor))
            rail_btn.setAutoRaise(True)
            rail_btn.setFixedWidth(self._SIDEBAR_RAIL_W - 6)
            rail_btn.clicked.connect(
                lambda _c=False, k=_key: self._open_sidebar_from_rail(k)
            )
            rail_lay.addWidget(rail_btn)
        rail_lay.addStretch(1)
        self._sidebar_rail.setVisible(False)

        sidebar_wrap = QWidget()
        wrap_lay = QHBoxLayout(sidebar_wrap)
        wrap_lay.setContentsMargins(0, 0, 0, 0)
        wrap_lay.setSpacing(0)
        wrap_lay.addWidget(self._sidebar_rail)
        wrap_lay.addWidget(right, 1)
        self._sidebar_wrap = sidebar_wrap
        self._sidebar_collapsed = False

        splitter.addWidget(sidebar_wrap)
        splitter.setSizes([900, 400])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setCollapsible(1, False)

        # Embed the splitter in a QMainWindow so transient tool docks (e.g. the
        # lattice grid) can still be hosted.  The ROI and Measurements panels are
        # no longer docks — they live directly in the ROI / Measure sidebar tabs.
        self._viewer_main = QMainWindow()
        self._viewer_main.setWindowFlags(Qt.Widget)
        self._viewer_main.setCentralWidget(splitter)
        self._viewer_main.setDockNestingEnabled(False)

        self._image_roi_set = None
        self._copy_roi_buffer = None  # ROI object held for Ctrl+V paste

        self._measurement_panel = ImageMeasurementsPanel(parent=self)
        self._measurement_table = self._measurement_panel.table
        self._feature_detection_panel = self._measurement_panel.feature_panel
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
        # One-shot + dialog tools the panel's menu now hosts: connect each to the
        # viewer's existing handler (formerly the Measure-tab action buttons).
        self._measurement_panel.distanceRequested.connect(self._on_measure_distance)
        self._measurement_panel.angleRequested.connect(self._on_measure_angle)
        self._measurement_panel.updateAngleRequested.connect(
            self._on_update_angle_measurement
        )
        self._measurement_panel.featureFinderRequested.connect(self._on_open_feature_finder)
        self._measurement_panel.pairCorrelationRequested.connect(
            self._on_open_pair_correlation
        )
        self._measurement_panel.featureToLatticeRequested.connect(
            self._on_open_feature_lattice
        )
        self._measurement_panel.latticeGridRequested.connect(self._on_open_lattice_grid)
        self._line_profile_panel.add_delta_measurement_clicked.connect(
            self._image_measurements.add_current_line_profile_delta_measurement
        )
        self._line_profile_panel.add_profile_summary_clicked.connect(
            self._image_measurements.add_current_line_profile_measurement
        )

        self._roi_panel = ROIManagerPanel(
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
            parent=self,
        )
        self._roi_panel.setObjectName("imageViewerRoiManagerPanel")

        # ROI manager and measurements now live in their sidebar tabs (built
        # above) rather than in separate floating docks.
        roi_lay.addWidget(self._roi_panel, 1)
        measurements_lay.addWidget(self._measurement_panel, 1)

        # Manager for floating, dismissible tool panels over the canvas.
        self._floating_panels = FloatingPanelManager(self._canvas_host)

        self._build_viewer_menu_bar()
        root.addWidget(self._viewer_main, 1)

        # navigation row
        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("← Prev")
        self._prev_btn.setFont(ui_font(10))
        self._prev_btn.setFixedWidth(90)
        self._prev_btn.clicked.connect(self._go_prev)

        self._pos_lbl = QLabel()
        self._pos_lbl.setAlignment(Qt.AlignCenter)
        self._pos_lbl.setFont(ui_font(10))

        self._next_btn = QPushButton("Next →")
        self._next_btn.setFont(ui_font(10))
        self._next_btn.setFixedWidth(90)
        self._next_btn.clicked.connect(self._go_next)

        close_btn = QPushButton("Close")
        close_btn.setFont(ui_font(10))
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
            self._target_drs, self._hist_panel,
            lambda: self._display_arr,
            self._channel_unit,
        )
        self._bad_line_preview_ctrl = BadLinePreviewController(
            self._zoom_lbl,
            self._processing_panel,
            lambda: self._display_arr if self._display_arr is not None else self._raw_arr,
        )

    def _configure_viewer_action(self, action: QAction, command_id: str) -> QAction:
        command = viewer_command(command_id)
        action.setText(command.label)
        if command.shortcuts:
            action.setShortcuts([QKeySequence(s) for s in command.shortcuts])
        if command.status_tip:
            action.setStatusTip(command.status_tip)
            action.setToolTip(command.status_tip)
        self._viewer_command_actions[command_id] = action
        return action

    def _viewer_action(
        self,
        command_id: str,
        handler=None,
        *,
        register: dict[str, QAction] | None = None,
    ) -> QAction:
        action = self._configure_viewer_action(QAction(self), command_id)
        if handler is not None:
            action.triggered.connect(handler)
        command = viewer_command(command_id)
        if register is not None:
            key = command.enabled_state_key or command.command_id
            register[key] = action
        return action

    def _build_viewer_menu_bar(self) -> None:
        menu_bar = self._viewer_main.menuBar()
        self._viewer_processing_actions: dict[str, QAction | dict[str, QAction]] = {}
        self._viewer_roi_tool_actions: dict[str, QAction] = {}
        self._viewer_roi_actions: dict[str, QAction] = {}
        self._viewer_measurement_actions: dict[str, QAction] = {}
        self._viewer_command_actions: dict[str, QAction] = {}

        file_menu = menu_bar.addMenu("File")
        close_action = QAction("Close", self)
        close_action.setShortcut(QKeySequence.Close)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

        view_menu = menu_bar.addMenu("View")
        auto_contrast_action = self._viewer_action(
            "view.auto_contrast",
            self._on_auto_clip,
        )
        view_menu.addAction(auto_contrast_action)
        reset_contrast_action = self._viewer_action(
            "view.reset_contrast",
            self._on_reset_display,
        )
        view_menu.addAction(reset_contrast_action)
        view_menu.addSeparator()
        fit_action = self._viewer_action("view.fit", self._zoom_lbl.fit_to_view)
        view_menu.addAction(fit_action)
        native_action = self._viewer_action("view.one_to_one", self._zoom_lbl.reset_zoom)
        view_menu.addAction(native_action)
        view_menu.addSeparator()
        reset_layout_action = self._viewer_action(
            "view.reset_layout",
            self._reset_viewer_window_layout,
        )
        view_menu.addAction(reset_layout_action)
        view_menu.addSeparator()
        display_panel_action = self._viewer_action(
            "panel.view",
            lambda: self._show_sidebar_tab("display"),
        )
        view_menu.addAction(display_panel_action)
        processing_panel_action = self._viewer_action(
            "panel.process",
            lambda: self._show_sidebar_tab("processing"),
        )
        view_menu.addAction(processing_panel_action)
        roi_panel_action = self._viewer_action(
            "panel.roi",
            lambda: self._show_sidebar_tab("roi"),
        )
        view_menu.addAction(roi_panel_action)
        measurements_panel_action = self._viewer_action(
            "panel.measure",
            lambda: self._show_sidebar_tab("measurements"),
        )
        view_menu.addAction(measurements_panel_action)
        export_panel_action = self._viewer_action(
            "panel.export",
            lambda: self._show_sidebar_tab("export"),
        )
        view_menu.addAction(export_panel_action)
        view_menu.addSeparator()
        dock_panels_action = self._viewer_action(
            "view.dock_panels",
            self._dock_panels_into_window,
        )
        view_menu.addAction(dock_panels_action)

        # ── Image menu ────────────────────────────────────────────────────────
        image_menu = menu_bar.addMenu("Image")

        info_action = self._viewer_action("image.info", self._on_show_image_info)
        image_menu.addAction(info_action)
        image_menu.addSeparator()

        # Color submenu
        color_menu = image_menu.addMenu("Color")
        _QUICK_CMAPS = [
            ("Gray",         "gray"),
            ("Gray (inv.)",  "gray_r"),
            ("Hot",          "hot"),
            ("AFM Hot",      "afmhot"),
            ("Viridis",      "viridis"),
            ("Plasma",       "plasma"),
            ("Inferno",      "inferno"),
            ("Cividis",      "cividis"),
        ]
        for _cmap_label, _mpl_key in _QUICK_CMAPS:
            _act = QAction(_cmap_label, self)
            _act.triggered.connect(
                lambda _c=False, k=_mpl_key, lbl=_cmap_label:
                    self._set_viewer_colormap_by_key(k, lbl)
            )
            color_menu.addAction(_act)
        color_menu.addSeparator()
        colormap_action = self._viewer_action("image.colormap", self._on_colormap_picker)
        color_menu.addAction(colormap_action)
        image_menu.addSeparator()

        # Adjust submenu
        adjust_menu = image_menu.addMenu("Adjust")
        bc_action = self._viewer_action(
            "panel.view",
            lambda: self._show_sidebar_tab("display"),
        )
        adjust_menu.addAction(bc_action)
        threshold_action = self._viewer_action("image.threshold", self._on_threshold)
        adjust_menu.addAction(threshold_action)

        scale_action = self._viewer_action("image.scale", self._on_scale_image)
        image_menu.addAction(scale_action)

        # Type submenu (bit-depth quantization)
        type_menu = image_menu.addMenu("Type")
        type_menu.addAction(
            self._viewer_action("image.type_8bit",
                                lambda: self._on_convert_bit_depth(8))
        )
        type_menu.addAction(
            self._viewer_action("image.type_16bit",
                                lambda: self._on_convert_bit_depth(16))
        )
        image_menu.addSeparator()

        # Transform submenu
        transform_menu = image_menu.addMenu("Transform")
        transform_menu.addAction(
            self._viewer_action("image.flip_h",
                                lambda: self._on_geometric_op("flip_horizontal"))
        )
        transform_menu.addAction(
            self._viewer_action("image.flip_v",
                                lambda: self._on_geometric_op("flip_vertical"))
        )
        transform_menu.addSeparator()
        transform_menu.addAction(
            self._viewer_action("image.rotate_90_cw",
                                lambda: self._on_geometric_op("rotate_90_cw"))
        )
        transform_menu.addAction(
            self._viewer_action("image.rotate_180",
                                lambda: self._on_geometric_op("rotate_180"))
        )
        transform_menu.addAction(
            self._viewer_action("image.rotate_270_cw",
                                lambda: self._on_geometric_op("rotate_270_cw"))
        )
        transform_menu.addAction(
            self._viewer_action("image.rotate_arbitrary", self._on_rotate_arbitrary)
        )
        transform_menu.addSeparator()
        transform_menu.addAction(
            self._viewer_action("image.shear", self._on_shear)
        )

        processing_menu = menu_bar.addMenu("Processing")
        plane_action = self._viewer_action(
            "processing.plane_background",
            self._on_simple_background,
        )
        processing_menu.addAction(plane_action)
        stm_bg_top_action = self._viewer_action(
            "processing.stm_background",
            self._on_open_stm_background,
        )
        processing_menu.addAction(stm_bg_top_action)
        bad_lines_top_action = self._viewer_action(
            "processing.bad_lines",
            self._on_preview_bad_lines,
        )
        processing_menu.addAction(bad_lines_top_action)
        image_ops_action = self._viewer_action(
            "processing.image_operations",
            self._on_open_image_operations,
        )
        processing_menu.addAction(image_ops_action)
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

        zero_action = self._viewer_action(
            "processing.zero_plane",
            self._set_zero_plane_btn.setChecked,
            register=self._viewer_processing_actions,
        )
        zero_action.setCheckable(True)
        self._set_zero_plane_btn.toggled.connect(self._sync_viewer_menu_actions)
        processing_menu.addAction(zero_action)
        clear_zero_action = self._viewer_action(
            "processing.clear_zero",
            self._on_clear_set_zero,
        )
        processing_menu.addAction(clear_zero_action)
        processing_menu.addSeparator()

        apply_action = self._viewer_action(
            "processing.apply",
            self._on_apply_processing,
        )
        processing_menu.addAction(apply_action)
        undo_action = self._viewer_action(
            "processing.undo",
            self._on_undo_processing,
            register=self._viewer_processing_actions,
        )
        processing_menu.addAction(undo_action)
        redo_action = self._viewer_action(
            "processing.redo",
            self._on_redo_processing,
            register=self._viewer_processing_actions,
        )
        processing_menu.addAction(redo_action)
        reset_action = self._viewer_action(
            "processing.reset",
            self._on_reset_processing,
        )
        processing_menu.addAction(reset_action)

        roi_menu = menu_bar.addMenu("ROI")
        show_roi_manager_action = QAction("Show ROI Manager", self)
        show_roi_manager_action.triggered.connect(self._show_roi_manager)
        roi_menu.addAction(show_roi_manager_action)
        roi_reference_action = self._viewer_action(
            "help.roi_reference",
            self._show_viewer_roi_reference,
        )
        roi_menu.addAction(roi_reference_action)
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
        ruler_action = self._viewer_action("measure.distance", self._on_measure_distance)
        measurements_menu.addAction(ruler_action)
        angle_action = self._viewer_action("measure.angle", self._on_measure_angle)
        measurements_menu.addAction(angle_action)
        update_angle_action = self._viewer_action(
            "measure.update_angle", self._on_update_angle_measurement,
        )
        measurements_menu.addAction(update_angle_action)
        roi_stats_new_action = self._viewer_action(
            "measure.roi_stats",
            self._on_measure_roi_stats,
        )
        measurements_menu.addAction(roi_stats_new_action)
        measurements_menu.addSeparator()
        add_roi_stats_action = self._viewer_action(
            "measure.add_roi_stats",
            self._image_measurements.add_active_roi_stats_measurement,
            register=self._viewer_measurement_actions,
        )
        measurements_menu.addAction(add_roi_stats_action)
        add_step_height_action = self._viewer_action(
            "measure.step_height",
            self._image_measurements.add_selected_step_height_measurement,
            register=self._viewer_measurement_actions,
        )
        measurements_menu.addAction(add_step_height_action)
        add_line_profile_action = self._viewer_action(
            "measure.line_profile",
            self._image_measurements.add_current_line_profile_measurement,
            register=self._viewer_measurement_actions,
        )
        measurements_menu.addAction(add_line_profile_action)
        find_periodicity_action = self._viewer_action(
            "measure.line_periodicity",
            self._image_measurements.find_periodicity_for_active_line_roi,
            register=self._viewer_measurement_actions,
        )
        measurements_menu.addAction(find_periodicity_action)
        detect_maxima_action = self._viewer_action(
            "measure.feature_maxima",
            self._image_measurements.detect_feature_maxima_for_active_roi,
            register=self._viewer_measurement_actions,
        )
        measurements_menu.addAction(detect_maxima_action)
        feature_finder_action = self._viewer_action(
            "measure.feature_finder",
            self._on_open_feature_finder,
        )
        measurements_menu.addAction(feature_finder_action)
        pair_corr_action = self._viewer_action(
            "measure.pair_correlation",
            self._on_open_pair_correlation,
        )
        measurements_menu.addAction(pair_corr_action)
        feat_lat_action = self._viewer_action(
            "measure.feature_lattice",
            self._on_open_feature_lattice,
        )
        measurements_menu.addAction(feat_lat_action)
        measurements_menu.addSeparator()
        self._image_measurements.add_detected_point_menu_actions(
            measurements_menu,
            self,
            self._viewer_measurement_actions,
        )
        measurements_menu.addSeparator()
        lattice_grid_action = self._viewer_action(
            "measure.lattice_grid",
            self._on_open_lattice_grid,
        )
        measurements_menu.addAction(lattice_grid_action)
        clear_lattice_grid_action = self._viewer_action(
            "measure.clear_lattice_grid",
            self._on_clear_lattice_grid,
        )
        measurements_menu.addAction(clear_lattice_grid_action)
        measurements_menu.addSeparator()
        open_fft_action = self._viewer_action("fft.open", self._on_open_fft_viewer)
        measurements_menu.addAction(open_fft_action)
        periodic_filter_action = self._viewer_action(
            "fft.periodic_filter",
            self._on_periodic_filter,
        )
        measurements_menu.addAction(periodic_filter_action)
        measurements_menu.addSeparator()
        show_measure_tab_action = self._viewer_action(
            "measure.show_panel",
            lambda: self._show_sidebar_tab("measurements"),
        )
        measurements_menu.addAction(show_measure_tab_action)

        export_menu = menu_bar.addMenu("Export")
        save_png_action = self._viewer_action("export.save_png", self._on_save_png)
        export_menu.addAction(save_png_action)
        save_processed_action = self._viewer_action(
            "export.save_processed",
            self._on_save_processed_image,
        )
        export_menu.addAction(save_processed_action)
        save_provenance_action = self._viewer_action(
            "export.save_provenance",
            self._on_save_provenance,
        )
        export_menu.addAction(save_provenance_action)

        window_menu = menu_bar.addMenu("Window")
        self._viewer_window_menu = window_menu
        window_menu.aboutToShow.connect(lambda: populate_window_menu(window_menu, self))
        populate_window_menu(window_menu, self)

        # Persistent shortcut for cycling windows (Cmd+` on macOS, Ctrl+` elsewhere).
        # Registered as a QShortcut on self so it fires even when a floating tool
        # window (e.g. FFT viewer) has keyboard focus, not just the main window.
        from probeflow.gui.viewer.window_menu import cycle_viewer_windows
        from PySide6.QtGui import QShortcut
        _cycle_shortcut = QShortcut(QKeySequence("Ctrl+`"), self)
        _cycle_shortcut.setContext(Qt.WindowShortcut)
        _cycle_shortcut.activated.connect(lambda: cycle_viewer_windows(self))

        help_menu = menu_bar.addMenu("Help")
        command_finder_action = self._viewer_action(
            "viewer.command_finder",
            self._show_command_finder,
        )
        help_menu.addAction(command_finder_action)
        shortcuts_action = self._viewer_action(
            "help.shortcuts",
            self._show_image_viewer_shortcuts,
        )
        help_menu.addAction(shortcuts_action)
        help_menu.addSeparator()
        github_action = QAction("GitHub", self)
        github_action.triggered.connect(lambda: _open_url(GITHUB_URL))
        help_menu.addAction(github_action)
        about_action = self._viewer_action("help.about", self._show_viewer_about)
        help_menu.addAction(about_action)
        howto_action = self._viewer_action(
            "help.howto",
            self._show_viewer_howto,
        )
        help_menu.insertAction(github_action, howto_action)
        definitions_action = self._viewer_action(
            "help.definitions",
            self._show_viewer_definitions,
        )
        help_menu.insertAction(github_action, definitions_action)
        roi_reference_help_action = self._viewer_action(
            "help.roi_reference",
            self._show_viewer_roi_reference,
        )
        help_menu.insertAction(github_action, roi_reference_help_action)
        help_menu.insertSeparator(github_action)

        self._sync_viewer_menu_actions()

    def _restore_viewer_desktop_layout(self) -> None:
        if not hasattr(self, "_viewer_splitter"):
            return
        cfg = load_config()
        layout = cfg.get("layout", {}).get("image_viewer", {})
        restore_geometry_or_default(self, layout.get("geometry"), 0.90)
        state = layout.get("state")
        if state and hasattr(self, "_viewer_main"):
            try:
                self._viewer_main.restoreState(b64_to_qbytearray(state))
            except Exception:
                pass

        sizes = layout.get("splitter_sizes")
        if sizes and len(sizes) == self._viewer_splitter.count():
            self._viewer_splitter.setSizes([max(1, int(x)) for x in sizes])
        else:
            self._viewer_splitter.setSizes([900, 400])

        tab_key = layout.get("sidebar_tab")
        if tab_key and hasattr(self, "_sidebar_tabs"):
            idx = self._sidebar_tab_indices.get(tab_key)
            if idx is not None:
                self._sidebar_tabs.setCurrentIndex(idx)

        zoom_mode = layout.get("zoom_mode", "fit")
        if zoom_mode in {"fit", "one_to_one", "manual"} and hasattr(self, "_zoom_lbl"):
            self._zoom_lbl._view_scale_mode = zoom_mode

        if layout.get("sidebar_collapsed") and hasattr(self, "_sidebar_panel"):
            self._set_sidebar_collapsed(True)

        self._show_maximized_on_start = bool(layout.get("was_maximized"))
        if self._show_maximized_on_start:
            self.setWindowState(self.windowState() | Qt.WindowMaximized)

    def _save_viewer_desktop_layout_into(self, cfg: dict) -> None:
        if not hasattr(self, "_viewer_splitter"):
            return
        layout_root = cfg.setdefault("layout", {})
        layout = layout_root.setdefault("image_viewer", {})
        layout["geometry"] = qbytearray_to_b64(self.saveGeometry())
        if hasattr(self, "_viewer_main"):
            layout["state"] = qbytearray_to_b64(self._viewer_main.saveState())
        layout["was_maximized"] = self.isMaximized()
        # When collapsed the live splitter sizes are not the expanded layout, so
        # persist the remembered expanded sizes instead.
        if getattr(self, "_sidebar_collapsed", False):
            layout["splitter_sizes"] = getattr(
                self, "_expanded_sidebar_sizes", self._viewer_splitter.sizes()
            )
        else:
            layout["splitter_sizes"] = self._viewer_splitter.sizes()
        layout["sidebar_collapsed"] = bool(getattr(self, "_sidebar_collapsed", False))
        layout["zoom_mode"] = getattr(getattr(self, "_zoom_lbl", None), "_view_scale_mode", "fit")

        if hasattr(self, "_sidebar_tabs") and hasattr(self, "_sidebar_tab_indices"):
            current = self._sidebar_tabs.currentIndex()
            for key, idx in self._sidebar_tab_indices.items():
                if idx == current:
                    layout["sidebar_tab"] = key
                    break

    def _dock_panels_into_window(self) -> None:
        """Re-dock any floating tool docks (e.g. the lattice grid) back in place.

        A QDockWidget dragged out to float has no obvious way back for most
        users; this pulls every floating tool dock back into the right dock area.
        The ROI and Measurements panels are no longer docks (they live in the
        sidebar tabs), so they are unaffected.
        """
        if not hasattr(self, "_viewer_main"):
            return
        restored = 0
        for dock in self._viewer_main.findChildren(QDockWidget):
            if dock.isFloating():
                self._viewer_main.addDockWidget(Qt.RightDockWidgetArea, dock)
                dock.setFloating(False)
                restored += 1
        if hasattr(self, "_status_lbl"):
            if restored:
                self._status_lbl.setText("Tool panels docked back into the window.")
            else:
                self._status_lbl.setText("No floating tool panels to dock.")

    def _reset_viewer_window_layout(self) -> None:
        cfg = load_config()
        if isinstance(cfg.get("layout"), dict):
            cfg["layout"].pop("image_viewer", None)
        save_config(cfg)
        apply_screen_fraction_geometry(self, 0.90)
        self._dock_panels_into_window()
        self._set_sidebar_collapsed(False)
        self._viewer_splitter.setSizes([900, 400])
        self._sidebar_tabs.setCurrentIndex(self._sidebar_tab_indices.get("display", 0))
        self._zoom_lbl._view_scale_mode = "fit"
        self._zoom_lbl.fit_to_view()
        self._show_maximized_on_start = False
        self._status_lbl.setText("Image viewer layout reset.")

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
        # A request to show a tab implies the panel should be visible.
        if getattr(self, "_sidebar_collapsed", False):
            self._set_sidebar_collapsed(False)
        idx = self._sidebar_tab_indices.get(key)
        if idx is not None:
            self._sidebar_tabs.setCurrentIndex(idx)

    def _set_sidebar_collapsed(self, collapsed: bool) -> None:
        """Collapse the task sidebar to a thin rail (image takes full width)."""
        if not hasattr(self, "_sidebar_panel"):
            return
        collapsed = bool(collapsed)
        if collapsed:
            sizes = self._viewer_splitter.sizes()
            total = sum(sizes)
            if len(sizes) == 2 and sizes[1] > 0:
                self._expanded_sidebar_sizes = sizes
            self._sidebar_panel.setVisible(False)
            self._sidebar_rail.setVisible(True)
            # Shrink the sidebar pane to the rail width so the image reclaims the
            # freed space (otherwise the splitter leaves an empty band).
            rail_w = self._SIDEBAR_RAIL_W
            if total > rail_w:
                self._viewer_splitter.setSizes([total - rail_w, rail_w])
        else:
            self._sidebar_rail.setVisible(False)
            self._sidebar_panel.setVisible(True)
            sizes = getattr(self, "_expanded_sidebar_sizes", None)
            if sizes:
                self._viewer_splitter.setSizes(sizes)
        self._sidebar_collapsed = collapsed

    def _toggle_sidebar_collapsed(self) -> None:
        self._set_sidebar_collapsed(not getattr(self, "_sidebar_collapsed", False))

    def _open_sidebar_from_rail(self, key: str) -> None:
        self._set_sidebar_collapsed(False)
        self._show_sidebar_tab(key)

    def _present_modal_tool(self, dialog, *, persistent: bool = False) -> "ModalOverlay":
        """Show *dialog* as a dimmed, click-outside-to-dismiss overlay.

        The dialog is reused whole (reparented as a child of a scrim overlay) instead
        of opening as a separate window that can slip behind the viewer.  Transient
        dialogs are tracked as modeless children so viewer close tears them down;
        ``persistent`` widgets are reused across opens (state preserved) and are owned
        by the viewer, so they are not tracked/destroyed here.
        """
        if not persistent:
            self._track_modeless_child(dialog)
        return ModalOverlay(self, dialog, persistent=persistent)

    def _open_advanced_tools(self) -> None:
        """Open the advanced-processing controls as a dismissible overlay."""
        self._present_modal_tool(self._advanced_widget, persistent=True)

    def _show_roi_manager(self) -> None:
        self._show_sidebar_tab("roi")

    def _show_measurements(self) -> None:
        self._show_sidebar_tab("measurements")

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
                try:
                    self._display_arr, self._display_scan_range_m = (
                        apply_processing_state_with_calibration(
                            self._raw_arr, state, self._image_roi_set,
                            scan_range_m=raw_range,
                        )
                    )
                except Exception:
                    raise
            except Exception as exc:
                self._processing_error = f"Processing failed: {exc}"
                if hasattr(self, "_status_lbl"):
                    self._status_lbl.setText(self._processing_error)
                self._display_arr = self._raw_arr
                self._display_scan_range_m = getattr(self, "_scan_range_m", None)
        else:
            self._processing_roi_error = ""
            self._processing_error = ""
            self._display_arr = self._raw_arr
            self._display_scan_range_m = getattr(self, "_scan_range_m", None)
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
