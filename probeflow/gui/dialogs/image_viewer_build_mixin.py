"""Image-viewer UI construction (the ``_build`` method).

Split out of ``image_viewer.py`` to keep the dialog module focused on behaviour.
``ImageViewerBuildMixin._build`` assembles the whole viewer layout — toolbar,
canvas, rulers, sidebar tabs, docks, and the controller objects — onto ``self``;
it runs once from ``ImageViewerDialog.__init__`` and relies on the other mixins
(resolved via the class MRO) for the handlers it wires up.
"""

from __future__ import annotations


from probeflow.gui.image_canvas import ImageCanvas
from probeflow.gui.models import PLANE_NAMES
from probeflow.gui.processing import ProcessingControlPanel
from probeflow.gui.rendering import CMAP_NAMES, DEFAULT_CMAP_LABEL, STM_COLORMAPS
from probeflow.gui.mask_manager import MaskManagerPanel
from probeflow.gui.roi_manager_dock import ROIManagerPanel
from probeflow.gui.styling import _sep
from probeflow.gui.typography import ui_font
from probeflow.gui.viewer import (
    BadLinePreviewController,
    DisplaySliderController,
    ImageMeasurementController,
    ProcessingUndoController,
    SetZeroPlaneController,
    SpecOverlayController,
)
from probeflow.gui.viewer.floating_panel import FloatingPanelManager
from probeflow.gui.viewer.histogram import HistogramPanel
from probeflow.gui.viewer.widgets import LineProfilePanel, RulerWidget, ScaleBarWidget
from probeflow.gui.widgets import ImageMeasurementsPanel
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QScrollArea, QSizePolicy, QSplitter,
    QStackedWidget, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)


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


class ImageViewerBuildMixin:
    """Builds the image-viewer layout in ``_build``."""

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
        ruler_fg = (self._t or {}).get("fg", "#cdd6f4")
        self._ruler_top  = RulerWidget("horizontal", fg=ruler_fg)
        self._ruler_left = RulerWidget("vertical", fg=ruler_fg)
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
        right.setMinimumWidth(360)
        self._sidebar_panel = right
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 4, 8, 4)
        right_lay.setSpacing(6)

        self._sidebar_tabs = QTabWidget()
        self._sidebar_tabs.setDocumentMode(True)
        self._sidebar_tabs.setMinimumWidth(340)
        # Elide (not clip) when the row is tight so every tab stays visible and
        # clickable — never silently drop a tab off the right edge.
        self._sidebar_tabs.setElideMode(Qt.ElideRight)
        self._sidebar_tabs.tabBar().setUsesScrollButtons(False)
        self._sidebar_tabs.tabBar().setExpanding(False)

        # The sidebar can show either the tab strip (page 0) or a single tool's
        # controls (page 1) — e.g. the lattice grid — so an interactive tool gets
        # the whole right column instead of opening a second dock.
        self._sidebar_stack = QStackedWidget()
        self._sidebar_stack.addWidget(self._sidebar_tabs)            # page 0
        self._sidebar_tool_host = QWidget()
        _tool_lay = QVBoxLayout(self._sidebar_tool_host)
        _tool_lay.setContentsMargins(0, 0, 0, 0)
        _tool_lay.setSpacing(6)
        _tool_header = QHBoxLayout()
        self._sidebar_tool_back = QPushButton("←  Back")
        self._sidebar_tool_back.setObjectName("ghostBtn")
        self._sidebar_tool_back.setCursor(QCursor(Qt.PointingHandCursor))
        self._sidebar_tool_back.setDefault(False)
        self._sidebar_tool_back.setAutoDefault(False)
        self._sidebar_tool_back.clicked.connect(self._on_sidebar_tool_back)
        _tool_header.addWidget(self._sidebar_tool_back)
        self._sidebar_tool_title = QLabel("")
        self._sidebar_tool_title.setStyleSheet("font-weight: 700;")
        _tool_header.addWidget(self._sidebar_tool_title, 1)
        _tool_lay.addLayout(_tool_header)
        self._sidebar_tool_content = QWidget()
        QVBoxLayout(self._sidebar_tool_content).setContentsMargins(0, 0, 0, 0)
        _tool_lay.addWidget(self._sidebar_tool_content, 1)
        self._sidebar_stack.addWidget(self._sidebar_tool_host)       # page 1
        self._sidebar_tool_widget = None
        self._sidebar_tool_on_close = None

        right_lay.addWidget(self._sidebar_stack, 1)
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
        _measurements_tab, measurements_lay = _sidebar_tab(
            "measurements", "Measure",
            "Distances, angles, ROI statistics, features and results.",
        )
        # ROI and Masks share one tab; each is a collapsible section (built when
        # the panels are created, below).
        _roimask_tab, roimask_lay = _sidebar_tab(
            "roi", "ROI/Mask",
            "Regions of interest and the active mask layer (edge-detection "
            "output, cleanup, conversion to ROIs).",
        )
        # Legacy alias so any "masks" navigation lands on the merged tab; no extra
        # rail entry (the rail is driven by _sidebar_tab_meta only).
        self._sidebar_tab_indices["masks"] = self._sidebar_tab_indices["roi"]
        _export_tab, export_lay = _sidebar_tab(
            "export", "Export",
            "Save images (PNG/PDF/SXM/GWY), provenance and hand-off to tools.",
        )

        # Size the sidebar so every tab label fits un-elided at the current
        # GUI font.  With a hardcoded minimum, Medium/Large fonts elided every
        # label ("Vi…/Proc…/Meas…"), hiding which tab is which — the elide
        # mode is a safety net for extreme cases, not the normal state.
        _tab_bar = self._sidebar_tabs.tabBar()
        _tabs_w = sum(
            _tab_bar.tabSizeHint(i).width() for i in range(_tab_bar.count())
        ) + self._sidebar_collapse_btn.width() + 12
        self._sidebar_tabs.setMinimumWidth(max(340, _tabs_w))
        right.setMinimumWidth(max(360, _tabs_w + 16))

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
        self._processing_panel.advanced_edge_requested.connect(
            self._on_open_advanced_edge)
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
        _sel_hdr = QLabel("Filter scope")
        _sel_hdr.setFont(ui_font(7, weight=QFont.Bold))
        _sel_hdr.setAlignment(Qt.AlignCenter)
        sel_col.addWidget(_sel_hdr)
        # The old Whole-image / ROI dropdown is retired: scope is auto-resolved
        # (quick selection → region; "Set as ROI filter scope" → that ROI; else
        # whole image).  This hint replaces it.
        _sel_hint = QLabel("Draw a region to\nprocess inside it")
        _sel_hint.setFont(ui_font(8))
        _sel_hint.setAlignment(Qt.AlignCenter)
        _sel_hint.setToolTip(
            "Smooth/high-pass/edge/FFT apply inside the active quick selection "
            "(or the ROI set as filter scope); background and scan-line "
            "corrections remain whole-image.")
        sel_col.addWidget(_sel_hint)
        sel_col.addStretch()

        zs_row.addLayout(zero_col, 1)
        zs_row.addLayout(sel_col, 1)
        processing_lay.addLayout(zs_row)

        self._roi_status_lbl = QLabel("Processing scope: whole image")
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
        # ±20 % was too tight for real thermal-drift corrections; allow up to
        # a factor of 2 either way (extreme values are still the user's call).
        self._undistort_scale_w, self._undistort_scale_spin = _spin_row(
            "Scale y:", 0.50, 2.00, 1.0, 0.005, 3)
        advanced_lay.addWidget(self._undistort_scale_w)

        # The overlay is modal (scrim blocks the panel behind it): without an
        # Apply button here, the user must dismiss the card and then find
        # "Apply processing" underneath — which reads as an abandoned choice,
        # not a commit. Apply directly and close the card.
        adv_apply_btn = QPushButton("Apply processing")
        adv_apply_btn.setFont(ui_font(9, weight=QFont.Bold))
        adv_apply_btn.setToolTip(
            "Apply the processing pipeline including these advanced settings, "
            "then close this panel."
        )
        adv_apply_btn.clicked.connect(self._on_apply_advanced_tools)
        advanced_lay.addWidget(adv_apply_btn)

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
        self._zoom_lbl.selection_drawn.connect(self._on_canvas_selection_drawn)
        self._zoom_lbl.selection_cleared.connect(self._on_canvas_selection_cleared)
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

        roi_hint_lbl = QLabel(
            "Drawing tools make a quick selection (gold dashes) — drag its "
            "handles to resize, drag its outline to move, Esc to clear, or "
            "promote it to a named ROI via right-click. Click an ROI to select "
            "it, then drag the active ROI (or its handles) to edit. For a split "
            "scan, set the View tab's “Contrast scope” to “Active ROI” to give "
            "each region its own brightness/contrast, and use “Hide ROI "
            "overlays” there to inspect the result."
        )
        roi_hint_lbl.setFont(ui_font(8))
        roi_hint_lbl.setWordWrap(True)
        roi_hint_lbl.setStyleSheet("color: palette(mid);")
        # The hint and the ROI manager panel are added to the ROI collapsible
        # section of the "ROI/Mask" tab after the panel is constructed below.

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
            "View": "View", "Process": "Proc", "Measure": "Meas",
            "ROI/Mask": "R/M", "Export": "Exp",
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
        splitter.setSizes([840, 460])
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
        self._image_mask_set = None   # active-mask layer (MaskSet); loaded per image
        self._roi_filter_scope_id = None  # ROI explicitly set as filter scope
        self._edge_detection_dialog = None

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
            self._roi_stats_active_and_show
        )
        self._measurement_panel.stepHeightRequested.connect(
            self._image_measurements.add_selected_step_height_measurement
        )
        self._measurement_panel.lineProfileRequested.connect(
            self._image_measurements.add_current_line_profile_measurement
        )
        self._measurement_panel.lineProfileWidthChanged.connect(
            self._on_line_profile_width_changed
        )
        # One-shot + dialog tools the panel's menu now hosts: connect each to the
        # viewer's existing handler (formerly the Measure-tab action buttons).
        self._measurement_panel.distanceRequested.connect(self._on_measure_distance)
        self._measurement_panel.angleRequested.connect(self._on_measure_angle)
        self._measurement_panel.updateAngleRequested.connect(
            self._on_update_angle_measurement
        )
        self._measurement_panel.clearAngleRequested.connect(self._clear_angle_overlay)
        self._measurement_panel.featureFinderRequested.connect(self._on_open_feature_finder)
        self._measurement_panel.pairCorrelationRequested.connect(
            self._on_open_pair_correlation
        )
        self._measurement_panel.featureToLatticeRequested.connect(
            self._on_open_feature_lattice
        )
        self._measurement_panel.latticeGridRequested.connect(self._on_open_lattice_grid)
        self._measurement_panel.fftViewerRequested.connect(self._on_open_fft_viewer)
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
                "on_roi_stats_measurement": self._roi_stats_roi_and_show,
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

        self._mask_panel = MaskManagerPanel(
            mask_set_getter=lambda: self._image_mask_set,
            callbacks={
                "on_mask_set_changed": self._on_image_mask_set_changed,
                "convert_to_roi":      self._convert_mask_to_rois,
                "add_mask_stats":      self._add_active_mask_stats,
                "export_mask":         self._export_mask_to_file,
            },
            parent=self,
        )
        self._mask_panel.setObjectName("imageViewerMaskManagerPanel")

        # ROI and Masks share the "ROI/Mask" tab as two collapsible sections
        # (same pattern as the View tab's "Spectroscopy overlay"). ROI is open by
        # default; Masks starts collapsed.
        _roi_btn, _roi_body, roi_section_lay = _collapsible_section(
            roimask_lay, "Regions of interest", expanded=True
        )
        roi_section_lay.addWidget(roi_hint_lbl)
        roi_section_lay.addWidget(self._roi_panel, 1)

        self._mask_section_btn, _mask_body, mask_section_lay = _collapsible_section(
            roimask_lay, "Masks", expanded=False
        )
        mask_hint = QLabel(
            "Masks come from Advanced Edge Detection (Process tab). The active "
            "mask (●) restricts statistics and can become ROI(s)."
        )
        mask_hint.setWordWrap(True)
        mask_hint.setFont(ui_font(8))
        mask_section_lay.addWidget(mask_hint)
        mask_section_lay.addWidget(self._mask_panel, 1)
        roimask_lay.addStretch(1)

        # Measurements now lives in its sidebar tab (built above) rather than in a
        # separate floating dock.
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
