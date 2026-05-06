"""ProbeFlow — PySide6 GUI for STM scan browsing, processing, and Createc→Nanonis conversion.

NOTE — this is the working GUI implementation, not deprecated code.
The `_legacy` suffix reflects an in-progress refactor: classes will be
moved out of this file into dedicated submodules (`gui/dialogs/`,
`gui/viewer/`, `gui/browse/`, `gui/convert/`, `gui/features/`,
`gui/terminal/`) opportunistically as features touch them. Until that
work completes, the bulk of the GUI lives here. New widgets / dialogs
should still go in their proper subpackage; only edits to existing
classes belong in this file.

Boundary rules: do not add parsers, writers, numerical kernels,
analysis algorithms, model definitions, or graph-node dataclasses
here — those belong in `io/`, `processing/`, `analysis/`, `core/`, or
`provenance/` respectively.
"""

from __future__ import annotations

import copy
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image

import os as _os
_os.environ.setdefault("QT_API", "pyside6")
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PySide6.QtCore import (
    Qt, QEvent, QObject, QRect, QRunnable, QThreadPool, QTimer,
    QSize, Signal, Slot,
)
from PySide6.QtGui import (
    QAction, QActionGroup, QBrush, QColor, QCursor, QFont, QImage, QKeySequence, QMovie,
    QPainter, QPen, QPixmap, QShortcut, QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
    QDialog, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QStackedWidget,
    QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QTextEdit, QToolTip, QVBoxLayout, QWidget,
)
import shutil
import subprocess
import webbrowser

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices


def _open_url(url: str) -> None:
    """Open URL in default browser. Tries Qt first, then Windows (WSL), then webbrowser."""
    try:
        if QDesktopServices.openUrl(QUrl(url)):
            return
    except Exception:
        pass
    if shutil.which("cmd.exe"):
        try:
            subprocess.Popen(["cmd.exe", "/c", "start", "", url],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return
        except Exception:
            pass
    try:
        webbrowser.open(url)
    except Exception:
        pass

from probeflow import processing as _proc
from probeflow.processing.display import (
    array_to_uint8 as _array_to_uint8,
    clip_range_from_array as _clip_range_from_array,
    histogram_from_array as _histogram_from_array,
)
from probeflow.processing.display_state import DisplayRangeState
from probeflow.provenance.export import build_scan_export_provenance, png_display_state
from probeflow.processing.gui_adapter import (
    processing_state_from_gui,
)
from probeflow.processing.state import missing_roi_references
from probeflow.gui.features import (
    FeaturesPanel,
    FeaturesSidebar,
    _FeaturesWorker,
    _FeaturesWorkerSignals,
)
from probeflow.gui.features.tv import (
    TVPanel,
    TVSidebar,
    _TVWorker,
    _TVWorkerSignals,
)
from probeflow.gui.processing import ProcessingControlPanel
from probeflow.gui.terminal import DeveloperTerminalWidget, _TerminalPane
from probeflow.gui.dialogs import (
    AboutDialog,
    FFTViewerDialog,
    PeriodicFilterDialog,
    SpecMappingDialog,
    SpecViewerDialog,
    STMBackgroundDialog,
    ViewerSpecMappingDialog,
)
from probeflow.core.scan_loader import SUPPORTED_SUFFIXES, load_scan

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_PATH     = Path.home() / ".probeflow_config.json"
REPO_ROOT       = Path(__file__).resolve().parents[2]
DEFAULT_CUSHION = REPO_ROOT / "src" / "file_cushions"
LOGO_PATH       = REPO_ROOT / "assets" / "logo.png"
LOGO_GIF_PATH   = REPO_ROOT / "assets" / "logo.gif"
LOGO_NAV_PATH   = REPO_ROOT / "assets" / "logo_nav.png"
GITHUB_URL      = "https://github.com/SPMQT-Lab/ProbeFlow"

NAVBAR_DARK_BG  = "#3273dc"
NAVBAR_LIGHT_BG = "#ffffff"
NAVBAR_H        = 58

# ── Themes ────────────────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg":         "#1e1e2e",
        "fg":         "#cdd6f4",
        "entry_bg":   "#313244",
        "btn_bg":     "#45475a",
        "btn_fg":     "#cdd6f4",
        "log_bg":     "#181825",
        "log_fg":     "#cdd6f4",
        "ok_fg":      "#a6e3a1",
        "err_fg":     "#f38ba8",
        "warn_fg":    "#fab387",
        "info_fg":    "#cdd6f4",
        "accent_bg":  "#89b4fa",
        "accent_fg":  "#1e1e2e",
        "sep":        "#45475a",
        "sub_fg":     "#6c7086",
        "sidebar_bg": "#181825",
        "main_bg":    "#1e1e2e",
        "status_bg":  "#313244",
        "status_fg":  "#6c7086",
        "card_bg":    "#313244",
        "card_sel":   "#4a4f6a",
        "card_fg":    "#cdd6f4",
        "tab_act":    "#313244",
        "tab_inact":  "#1e1e2e",
        "tree_bg":    "#181825",
        "tree_fg":    "#cdd6f4",
        "tree_sel":   "#45475a",
        "tree_head":  "#313244",
        "splitter":   "#45475a",
    },
    "light": {
        "bg":         "#f8f9fa",
        "fg":         "#1e1e2e",
        "entry_bg":   "#ffffff",
        "btn_bg":     "#d0d4da",
        "btn_fg":     "#1e1e2e",
        "log_bg":     "#ffffff",
        "log_fg":     "#1e1e2e",
        "ok_fg":      "#1a7a1a",
        "err_fg":     "#c0392b",
        "warn_fg":    "#b07800",
        "info_fg":    "#1e1e2e",
        "accent_bg":  "#3273dc",
        "accent_fg":  "#ffffff",
        "sep":        "#b0bec5",
        "sub_fg":     "#4a5568",
        "sidebar_bg": "#f0f2f5",
        "main_bg":    "#ffffff",
        "status_bg":  "#f0f2f5",
        "status_fg":  "#4a5568",
        "card_bg":    "#dce8f5",
        "card_sel":   "#b8d4ee",
        "card_fg":    "#1e1e2e",
        "tab_act":    "#ffffff",
        "tab_inact":  "#e4edf8",
        "tree_bg":    "#ffffff",
        "tree_fg":    "#1e1e2e",
        "tree_sel":   "#cce0f5",
        "tree_head":  "#e8f0f8",
        "splitter":   "#dee2e6",
    },
}

# ── Extracted GUI helpers (re-exported for compatibility) ─────────────────────
from probeflow.gui.models import (
    PLANE_NAMES,
    FolderEntry,
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
    _fit_image_to_box,
    _get_lut,
    _make_lut,
    clip_range_from_arr,
    pil_to_pixmap,
    render_scan_image,
    render_scan_thumbnail,
    render_spec_thumbnail,
    render_with_processing,
    resolve_thumbnail_plane_index,
)
from probeflow.gui.workers import (
    ChannelLoader,
    ChannelSignals,
    ConversionSignals,
    ConversionWorker,
    ThumbnailLoader,
    ThumbnailSignals,
    ViewerLoader,
    ViewerSignals,
)
from probeflow.gui.browse import ScanCard, SpecCard, ThumbnailGrid, _BrowseCard
from probeflow.gui.viewer.widgets import (
    LineProfilePanel,
    RulerWidget,
    ScaleBarWidget,
    _ZoomLabel,
)
from probeflow.gui.image_canvas import ImageCanvas
from probeflow.gui.roi_manager_dock import ROIManagerDock


# ── Config ────────────────────────────────────────────────────────────────────
GUI_FONT_SIZES = {"Small": 9, "Medium": 12, "Large": 14}
GUI_FONT_DEFAULT = "Medium"


def normalise_gui_font_size(label: str | None) -> str:
    return label if label in GUI_FONT_SIZES else GUI_FONT_DEFAULT


def load_config() -> dict:
    defaults = {
        "dark_mode":       True,
        "input_dir":       "",
        "output_dir":      "",
        "custom_output":   False,
        "do_png":          False,
        "do_sxm":          True,
        "clip_low":        1.0,
        "clip_high":       99.0,
        "colormap":        DEFAULT_CMAP_LABEL,
        "browse_filter":   "all",
        "gui_font_size":   GUI_FONT_DEFAULT,
    }
    try:
        if CONFIG_PATH.exists():
            defaults.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass
    defaults["gui_font_size"] = normalise_gui_font_size(defaults.get("gui_font_size"))
    return defaults


def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


# ── Viewer and browse support lives in extracted GUI modules. ───────────────

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
        self._t          = t
        self._idx        = next((i for i, e in enumerate(entries) if e.stem == entry.stem), 0)
        self._pool       = QThreadPool.globalInstance()
        self._token      = object()
        self._clip_low   = clip_low
        self._clip_high  = clip_high
        self._drs        = DisplayRangeState(low_pct=clip_low, high_pct=clip_high)
        self._processing = dict(processing) if processing else {}
        self._processing_roi_error: str = ""
        # Undo / redo stacks for processing state. Each entry is a deep copy
        # of the full processing dict at a prior point. Apply / Reset push
        # the previous state onto _undo_stack and clear _redo_stack; the
        # Undo / Redo buttons swap between the two.
        self._proc_undo_stack: list[dict] = []
        self._proc_redo_stack: list[dict] = []
        self._proc_undo_btn = None
        self._proc_redo_btn = None
        # Mutable mapping shared with the parent window: spec_stem → image_stem.
        # Empty dict by default — markers only appear after explicit mapping.
        self._spec_image_map = spec_image_map if spec_image_map is not None else {}
        self._raw_arr: Optional[np.ndarray] = None
        self._display_arr: Optional[np.ndarray] = None  # raw or processed, for histogram/export
        self._spec_markers: list[dict] = []
        self._spec_roi_set: "object | None" = None  # ROISet when spec positions are loaded
        self._scan_header: dict = {}
        self._scan_range_m: Optional[tuple] = None
        self._scan_shape: Optional[tuple] = None
        self._scan_format: str = ""
        self._scan_plane_names: list[str] = list(PLANE_NAMES)
        self._scan_plane_units: list[str] = ["m", "m", "A", "A"]
        self._roi_rect_px: Optional[tuple[int, int, int, int]] = None
        self._selection_geometry: Optional[dict] = None
        self._line_profile_geometry: Optional[dict] = None
        self._zero_pick_mode: str = "plane"
        self._zero_plane_points_px: list[tuple[int, int]] = []
        self._zero_markers_hidden = False
        self._pending_initial_plane_idx: Optional[int] = max(0, int(initial_plane_idx))
        self._reset_zoom_on_next_pixmap = True
        self._deferred_action: str = ""
        self._deferred_plane_idx: int = 0

        self._build()
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
        self._zoom_out_btn.setFixedSize(28, 24)
        self._zoom_out_btn.setFont(QFont("Helvetica", 11))
        self._zoom_out_btn.setToolTip("Zoom out")
        self._zoom_out_btn.clicked.connect(lambda: self._zoom_lbl.zoom_by(1 / 1.25))
        toolbar.addWidget(self._zoom_out_btn)

        self._zoom_reset_btn = QPushButton("1:1")
        self._zoom_reset_btn.setFixedSize(36, 24)
        self._zoom_reset_btn.setFont(QFont("Helvetica", 9))
        self._zoom_reset_btn.setToolTip("Reset to native raster size")
        self._zoom_reset_btn.clicked.connect(self._zoom_lbl.reset_zoom)
        toolbar.addWidget(self._zoom_reset_btn)

        self._zoom_fit_btn = QPushButton("Fit")
        self._zoom_fit_btn.setFixedSize(36, 24)
        self._zoom_fit_btn.setFont(QFont("Helvetica", 9))
        self._zoom_fit_btn.setToolTip("Fit image to available space")
        self._zoom_fit_btn.clicked.connect(self._zoom_lbl.fit_to_view)
        toolbar.addWidget(self._zoom_fit_btn)

        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setFixedSize(28, 24)
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

        self._coord_lbl = QLabel("—")
        self._coord_lbl.setFont(QFont("Helvetica", 8))
        self._coord_lbl.setMinimumWidth(140)
        toolbar.addWidget(self._coord_lbl)

        zoom_hint = QLabel("Ctrl+scroll to zoom")
        zoom_hint.setFont(QFont("Helvetica", 8))
        toolbar.addWidget(zoom_hint)
        toolbar.addStretch()
        left_lay.addLayout(toolbar)

        drawing_bar = QHBoxLayout()
        drawing_bar.setSpacing(4)
        drawing_lbl = QLabel("Draw")
        drawing_lbl.setFont(QFont("Helvetica", 8, QFont.Bold))
        drawing_bar.addWidget(drawing_lbl)
        self._drawing_group = QButtonGroup(self)
        self._drawing_group.setExclusive(True)
        # Keep old name for backward-compat references
        self._selection_group = self._drawing_group
        for key, label, tip in (
            ("pan",       "✋ Pan",     "Pan (drag to scroll)"),
            ("rectangle", "▭ Rect",    "Rectangle ROI  [R]"),
            ("ellipse",   "◯ Ellipse", "Ellipse ROI  [E]"),
            ("polygon",   "⬠ Poly",   "Polygon ROI — click vertices, double-click to close  [P]"),
            ("freehand",  "〰 Free",   "Freehand ROI — drag to draw  [F]"),
            ("line",      "— Line",    "Line ROI  [L]"),
            ("point",     "• Point",   "Point ROI  [T]"),
        ):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(24)
            btn.setMinimumWidth(44)
            btn.setFont(QFont("Helvetica", 8))
            btn.setToolTip(tip)
            self._drawing_group.addButton(btn)
            btn.setProperty("drawing_tool", key)
            if key == "pan":
                btn.setChecked(True)
            drawing_bar.addWidget(btn)
        self._drawing_group.buttonClicked.connect(self._on_drawing_tool_clicked)
        drawing_bar.addStretch()
        left_lay.addLayout(drawing_bar)

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

        # ── Right: control panel ───────────────────────────────────────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setMinimumWidth(300)
        right_scroll.setMaximumWidth(380)
        right_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        right = QWidget()
        right.setMinimumWidth(300)
        right.setMaximumWidth(380)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 4, 8, 4)
        right_lay.setSpacing(6)

        def _collapsible_section(title: str, expanded: bool = False):
            btn = QPushButton(("[−] " if expanded else "[+] ") + title)
            btn.setCheckable(True)
            btn.setChecked(expanded)
            btn.setFlat(True)
            btn.setFont(QFont("Helvetica", 9, QFont.Bold))
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            right_lay.addWidget(btn)

            body = QWidget()
            body_lay = QVBoxLayout(body)
            body_lay.setContentsMargins(2, 2, 0, 2)
            body_lay.setSpacing(4)
            body.setVisible(expanded)
            right_lay.addWidget(body)

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

        # histogram
        hist_lbl = QLabel("Histogram / contrast")
        hist_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        right_lay.addWidget(hist_lbl)

        self._fig  = Figure(figsize=(3.0, 1.6), dpi=80)
        self._fig.patch.set_alpha(0)
        self._ax   = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setFixedHeight(140)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self._canvas.customContextMenuRequested.connect(self._on_hist_context_menu)
        right_lay.addWidget(self._canvas)

        # histogram drag state
        self._low_line      = None
        self._high_line     = None
        self._hist_flat_phys: Optional[np.ndarray] = None
        self._hist_unit     = ""
        self._dragging      = None  # 'low' | 'high' | None
        self._canvas.mpl_connect("button_press_event",   self._on_hist_press)
        self._canvas.mpl_connect("motion_notify_event",  self._on_hist_motion)
        self._canvas.mpl_connect("button_release_event", self._on_hist_release)

        hist_actions = QHBoxLayout()
        self._auto_clip_btn = QPushButton("Auto")
        self._auto_clip_btn.setFont(QFont("Helvetica", 8))
        self._auto_clip_btn.setFixedHeight(22)
        self._auto_clip_btn.setToolTip(
            "Autoscale display bounds to the current image's 1%–99% percentiles.")
        self._auto_clip_btn.clicked.connect(self._on_auto_clip)
        hist_actions.addStretch()
        hist_actions.addWidget(self._auto_clip_btn)
        right_lay.addLayout(hist_actions)

        # Å / pA value readout for current display bounds
        self._clip_val_lbl = QLabel("")
        self._clip_val_lbl.setFont(QFont("Helvetica", 8))
        self._clip_val_lbl.setAlignment(Qt.AlignCenter)
        right_lay.addWidget(self._clip_val_lbl)

        right_lay.addWidget(_sep())

        self._processing_panel = ProcessingControlPanel("viewer_full")
        self._processing_panel.bad_line_preview_requested.connect(
            self._on_preview_bad_lines)
        self._processing_panel.bad_line_preview_settings_changed.connect(
            self._on_bad_line_preview_settings_changed)
        right_lay.addWidget(self._processing_panel)

        right_lay.addWidget(_sep())

        # ── Zero reference | Selection use (compact 2-column row) ─────────────
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
        _sel_hdr = QLabel("Selection use")
        _sel_hdr.setFont(QFont("Helvetica", 7, QFont.Bold))
        _sel_hdr.setAlignment(Qt.AlignCenter)
        sel_col.addWidget(_sel_hdr)
        self._scope_cb = QComboBox()
        self._scope_cb.addItems(["Whole image", "ROI filters only"])
        self._scope_cb.setFont(QFont("Helvetica", 8))
        self._scope_cb.setToolTip(
            "ROI filters only: smooth/high-pass/edge/FFT apply inside the "
            "drawn selection; background and scan-line corrections remain whole-image.")
        sel_col.addWidget(self._scope_cb)
        self._bg_fit_roi_cb = QCheckBox("Fit bg from sel.")
        self._bg_fit_roi_cb.setFont(QFont("Helvetica", 8))
        self._bg_fit_roi_cb.setToolTip(
            "Fits Plane/Quadratic/Cubic/Quartic background using selected area pixels, "
            "then subtracts that fitted surface from the whole image."
        )
        sel_col.addWidget(self._bg_fit_roi_cb)
        self._patch_roi_cb = QCheckBox("Patch selection")
        self._patch_roi_cb.setFont(QFont("Helvetica", 8))
        self._patch_roi_cb.setToolTip(
            "Fills the selected area by patch interpolation. "
            "Line selections cannot be patch-interpolated."
        )
        sel_col.addWidget(self._patch_roi_cb)
        _pm_row = QHBoxLayout()
        _pm_lbl = QLabel("Method:")
        _pm_lbl.setFont(QFont("Helvetica", 8))
        _pm_lbl.setFixedWidth(46)
        self._patch_method_combo = QComboBox()
        self._patch_method_combo.addItems(["Line-fit", "Laplace"])
        self._patch_method_combo.setFont(QFont("Helvetica", 8))
        self._patch_method_combo.setToolTip(
            "Line-fit: extrapolates scan-line slope from rim pixels — "
            "recommended for STM terraces.\n"
            "Laplace: isotropic harmonic fill — smooth but does not preserve surface tilt."
        )
        _pm_row.addWidget(_pm_lbl)
        _pm_row.addWidget(self._patch_method_combo, 1)
        sel_col.addLayout(_pm_row)
        sel_col.addStretch()

        zs_row.addLayout(zero_col, 1)
        zs_row.addLayout(sel_col, 1)
        right_lay.addLayout(zs_row)

        self._roi_status_lbl = QLabel("Selection: none")
        self._roi_status_lbl.setFont(QFont("Helvetica", 8))
        self._roi_status_lbl.setWordWrap(True)
        right_lay.addWidget(self._roi_status_lbl)

        right_lay.addWidget(_sep())

        # ── Apply / Reset — always visible ────────────────────────────────────
        ar_row = QHBoxLayout()
        ar_row.setSpacing(4)
        proc_apply_btn = QPushButton("Apply processing")
        proc_apply_btn.setFont(QFont("Helvetica", 8, QFont.Bold))
        proc_apply_btn.setFixedHeight(28)
        proc_apply_btn.setObjectName("accentBtn")
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
        right_lay.addLayout(ar_row)

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
        right_lay.addLayout(ur_row)
        self._update_undo_redo_buttons()

        QShortcut(QKeySequence("Ctrl+Z"), self,
                  activated=self._on_undo_processing)
        QShortcut(QKeySequence("Ctrl+Y"), self,
                  activated=self._on_redo_processing)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self,
                  activated=self._on_redo_processing)

        right_lay.addWidget(_sep())

        # ── Save PNG — always visible ─────────────────────────────────────────
        save_btn = QPushButton("⬇  Save PNG copy…")
        save_btn.setFont(QFont("Helvetica", 8, QFont.Bold))
        save_btn.setFixedHeight(26)
        save_btn.setObjectName("accentBtn")
        save_btn.clicked.connect(self._on_save_png)
        right_lay.addWidget(save_btn)

        # ── Send to tool (collapsible) ────────────────────────────────────────
        _, self._export_widget, send_lay = _collapsible_section("→ Send to tool", expanded=False)

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
        _, self._advanced_widget, advanced_lay = _collapsible_section("Advanced tools", expanded=False)

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
        _, self._spec_overlay_widget, spec_lay = _collapsible_section("Spectroscopy overlay", expanded=False)

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
        self._zoom_lbl.selection_preview_changed.connect(self._on_selection_preview_changed)
        self._zoom_lbl.selection_changed.connect(self._on_selection_changed)
        self._zoom_lbl.pixmap_resized.connect(self._on_pixmap_resized)
        self._zoom_lbl.context_menu_requested.connect(self._on_image_context_menu)
        self._zoom_lbl.pixel_hovered.connect(self._on_pixel_hovered)
        self._zoom_lbl.roi_created.connect(self._on_canvas_roi_created)
        self._zoom_lbl.roi_move_requested.connect(self._on_canvas_roi_move)
        self._zoom_lbl.tool_changed.connect(self._on_canvas_tool_changed)
        self._zoom_lbl.roi_context_menu_requested.connect(self._on_roi_canvas_context_menu)
        self._line_profile_panel.export_csv_clicked.connect(self._on_export_line_profile_csv)

        self._status_lbl = QLabel("")
        self._status_lbl.setFont(QFont("Helvetica", 8))
        self._status_lbl.setWordWrap(True)
        right_lay.addWidget(self._status_lbl)

        right_lay.addStretch()

        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setSizes([740, 320])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setCollapsible(1, False)

        # Embed splitter in a QMainWindow so we can host the ROI dock widget
        self._viewer_main = QMainWindow()
        self._viewer_main.setWindowFlags(Qt.Widget)
        self._viewer_main.setCentralWidget(splitter)
        self._viewer_main.setDockNestingEnabled(False)

        self._image_roi_set = None
        self._roi_dock = ROIManagerDock(
            roi_set_getter=lambda: self._image_roi_set,
            callbacks={
                "on_roi_set_changed":    self._on_image_roi_set_changed,
                "on_bg_subtract_fit":    self._on_roi_bg_subtract_fit,
                "on_bg_subtract_exclude": self._on_roi_bg_subtract_exclude,
                "on_fft_roi":            self._on_roi_fft,
                "on_histogram_roi":      self._on_roi_histogram,
                "on_line_profile_roi":   self._on_roi_line_profile,
                "on_roi_selection_changed": self._sync_viewer_menu_actions,
                "get_image_shape":       self._current_array_shape,
            },
            parent=self._viewer_main,
        )
        self._viewer_main.addDockWidget(Qt.RightDockWidgetArea, self._roi_dock)
        self._viewer_main.resizeDocks([self._roi_dock], [200], Qt.Horizontal)
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

    def _build_viewer_menu_bar(self) -> None:
        menu_bar = self._viewer_main.menuBar()
        self._viewer_processing_actions: dict[str, QAction | dict[str, QAction]] = {}
        self._viewer_roi_tool_actions: dict[str, QAction] = {}
        self._viewer_roi_actions: dict[str, QAction] = {}

        file_menu = menu_bar.addMenu("File")
        close_action = QAction("Close", self)
        close_action.setShortcut(QKeySequence.Close)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

        view_menu = menu_bar.addMenu("View")
        fit_action = QAction("Fit image", self)
        fit_action.triggered.connect(self._zoom_lbl.fit_to_view)
        view_menu.addAction(fit_action)
        native_action = QAction("Native size", self)
        native_action.triggered.connect(self._zoom_lbl.reset_zoom)
        view_menu.addAction(native_action)

        processing_menu = menu_bar.addMenu("Processing")
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
        self._add_combo_menu(
            processing_menu, "Radial FFT", self._processing_panel._fft_combo,
            ["None", "Low-pass", "High-pass"],
        )
        fft_soft_action = QAction("FFT soft border", self)
        fft_soft_action.setCheckable(True)
        fft_soft_action.triggered.connect(self._processing_panel._fft_soft_cb.setChecked)
        self._processing_panel._fft_soft_cb.toggled.connect(self._sync_viewer_menu_actions)
        self._viewer_processing_actions["fft_soft_border"] = fft_soft_action
        processing_menu.addAction(fft_soft_action)
        processing_menu.addSeparator()

        stm_background_action = QAction("STM Background...", self)
        stm_background_action.triggered.connect(self._on_open_stm_background)
        processing_menu.addAction(stm_background_action)
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

        export_menu = menu_bar.addMenu("Export")
        save_png_action = QAction("Save PNG copy", self)
        save_png_action.triggered.connect(self._on_save_png)
        export_menu.addAction(save_png_action)
        save_processed_action = QAction("Save processed image", self)
        save_processed_action.setEnabled(False)
        export_menu.addAction(save_processed_action)
        save_provenance_action = QAction("Save provenance", self)
        save_provenance_action.setEnabled(False)
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

    def _show_roi_manager(self) -> None:
        if not hasattr(self, "_roi_dock"):
            return
        self._roi_dock.show()
        self._roi_dock.raise_()

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

        # Return closes the dialog (when not drawing)
        if k == Qt.Key_Return:
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

        # ── arrow keys: nudge line profile or navigate ────────────────────────
        if k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            if self._nudge_line_profile(k):
                event.accept()
                return
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
        self._refresh_line_profile_from_selection()

    def _load_current_source(self, entry: SxmFile, reset_zoom: bool = True):
        self._title_lbl.setText(entry.stem)
        self.setWindowTitle(entry.stem)
        self._pos_lbl.setText(f"{self._idx + 1} / {len(self._entries)}")
        self._prev_btn.setEnabled(self._idx > 0)
        self._next_btn.setEnabled(self._idx < len(self._entries) - 1)
        if reset_zoom:
            self._zoom_lbl.setText("Loading…")
            self._zoom_lbl.setPixmap(QPixmap())
        self._zoom_lbl.set_markers([])
        self._load_image_roi_set(entry)
        try:
            _scan = load_scan(entry.path)
            self._set_scan_channel_choices(_scan)
            if self._pending_initial_plane_idx is not None:
                target = max(0, min(self._pending_initial_plane_idx, _scan.n_planes - 1))
                self._ch_cb.blockSignals(True)
                self._ch_cb.setCurrentIndex(target)
                self._ch_cb.blockSignals(False)
                self._pending_initial_plane_idx = None
            idx = self._ch_cb.currentIndex()
            self._raw_arr = _scan.planes[idx] if idx < _scan.n_planes else None
            self._scan_header  = _scan.header or {}
            self._scan_range_m = _scan.scan_range_m
            self._scan_shape   = _scan.planes[0].shape if _scan.planes else None
            self._scan_format  = entry.source_format
            self._scan_plane_names = list(_scan.plane_names)
            self._scan_plane_units = list(_scan.plane_units)
        except Exception:
            self._raw_arr      = None
            self._scan_header  = {}
            self._scan_range_m = None
            self._scan_shape   = None
            self._scan_format  = ""
            self._scan_plane_names = list(PLANE_NAMES)
            self._scan_plane_units = ["m", "m", "A", "A"]

    def _refresh_display_array(self, reset_zoom_if_shape_changed: bool = False):
        old_shape = self._display_arr.shape if self._display_arr is not None else None
        # display array: raw with processing applied (no grain overlay — that's visual only)
        if self._raw_arr is not None and self._processing:
            try:
                self._processing_roi_error = ""
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
            except Exception:
                self._display_arr = self._raw_arr
        else:
            self._processing_roi_error = ""
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
        self._refresh_histogram_and_markers(entry)
        self._refresh_viewer_pixmap(reset_zoom=False)
        self._refresh_line_profile_from_selection()

    def _on_bad_line_preview_settings_changed(self) -> None:
        if getattr(self, "_bad_line_preview_active", False):
            self._on_preview_bad_lines()
            return
        if hasattr(self, "_processing_panel"):
            method = self._processing_panel.bad_line_method()
            if method is None:
                self._processing_panel.set_bad_line_preview_summary(
                    "Preview: select a method")
            else:
                self._processing_panel.set_bad_line_preview_summary(
                    "Preview: run detection")
        if hasattr(self._zoom_lbl, "clear_bad_segment_overlay"):
            self._zoom_lbl.clear_bad_segment_overlay()

    def _on_preview_bad_lines(self) -> None:
        if self._display_arr is None and self._raw_arr is None:
            self._processing_panel.set_bad_line_preview_summary("Preview: no image")
            return
        method = self._processing_panel.bad_line_method()
        if method is None:
            self._clear_bad_line_preview("Preview: select a method")
            return
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        try:
            from probeflow.processing import (
                detect_bad_scanline_segments,
                repair_bad_scanline_segments,
            )
            segments = detect_bad_scanline_segments(
                arr,
                threshold=self._processing_panel.bad_line_threshold(),
                method=method,
                polarity=self._processing_panel.bad_line_polarity(),
                min_segment_length_px=(
                    self._processing_panel.bad_line_min_segment_length_px()
                ),
                max_adjacent_bad_lines=(
                    self._processing_panel.bad_line_max_adjacent_bad_lines()
                ),
            )
            _preview_arr, preview_info = repair_bad_scanline_segments(
                arr,
                segments,
                max_adjacent_bad_lines=(
                    self._processing_panel.bad_line_max_adjacent_bad_lines()
                ),
                threshold=self._processing_panel.bad_line_threshold(),
                polarity=self._processing_panel.bad_line_polarity(),
                min_segment_length_px=(
                    self._processing_panel.bad_line_min_segment_length_px()
                ),
            )
        except Exception as exc:
            self._clear_bad_line_preview(f"Preview error: {exc}")
            return
        self._bad_line_preview_segments = list(segments)
        self._bad_line_preview_active = True
        if hasattr(self._zoom_lbl, "set_bad_segment_overlay"):
            self._zoom_lbl.set_bad_segment_overlay(segments)
        n = len(segments)
        n_lines = len({seg.line_index for seg in segments})
        skipped = len(preview_info.skipped_segments)
        skipped_lines = len({seg.line_index for seg in preview_info.skipped_segments})
        if n == 0:
            summary = "Detected 0 segments"
        elif n == 1:
            summary = f"Detected 1 segment on 1 scan line"
        else:
            summary = f"Detected {n} segments across {n_lines} scan lines"
        if skipped:
            summary += (
                f"; skipped {skipped} segment{'s' if skipped != 1 else ''} "
                f"across {skipped_lines} line{'s' if skipped_lines != 1 else ''} "
                "because the adjacent-line limit was exceeded"
            )
        self._processing_panel.set_bad_line_preview_summary(summary)
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(summary)

    def _clear_bad_line_preview(self, summary: str = "Preview: not run") -> None:
        self._bad_line_preview_segments = []
        self._bad_line_preview_active = False
        if hasattr(self, "_processing_panel"):
            self._processing_panel.set_bad_line_preview_summary(summary)
        if hasattr(self._zoom_lbl, "clear_bad_segment_overlay"):
            self._zoom_lbl.clear_bad_segment_overlay()

    def _on_open_stm_background(self) -> None:
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("STM Background: no image loaded.")
            return
        active_roi = self._active_image_roi()
        roi_mask = None
        roi_id = None
        roi_name = None
        if (
            active_roi is not None
            and active_roi.kind in {"rectangle", "ellipse", "polygon", "freehand", "multipolygon"}
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
        loader = ViewerLoader(entry, self._colormap, self._token, None,
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
        unit = self._scan_plane_units[idx] if idx < len(self._scan_plane_units) else ""
        name = self._scan_plane_names[idx] if idx < len(self._scan_plane_names) else self._ch_cb.currentText()
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        from probeflow.analysis.spec_plot import choose_display_unit
        scale, unit_label = choose_display_unit(unit, arr)
        axis_label = name.rsplit(" ", 1)[0] if name.endswith((" forward", " backward")) else name
        return scale, unit_label, axis_label

    def _set_scan_channel_choices(self, scan) -> None:
        names = list(scan.plane_names) if scan.plane_names else [
            f"Channel {i}" for i in range(scan.n_planes)
        ]
        current = self._ch_cb.currentIndex()
        if [self._ch_cb.itemText(i) for i in range(self._ch_cb.count())] == names:
            return
        self._ch_cb.blockSignals(True)
        self._ch_cb.clear()
        self._ch_cb.addItems(names)
        self._ch_cb.setCurrentIndex(max(0, min(current, len(names) - 1)))
        self._ch_cb.blockSignals(False)

    def _update_histogram(self):
        # Use processed display array so histogram tracks what the user sees
        arr = self._display_arr
        self._ax.cla()
        self._low_line  = None
        self._high_line = None
        self._hist_flat_phys = None

        if arr is None:
            self._canvas.draw_idle()
            return

        flat = arr[np.isfinite(arr)].ravel()
        if flat.size < 2:
            self._canvas.draw_idle()
            return

        scale, unit, axis_label = self._channel_unit()
        flat_phys = flat.astype(np.float64) * scale
        self._hist_flat_phys = flat_phys
        self._hist_unit = unit

        # Clip lines: position from resolved display range (manual or percentile).
        # arr is in SI; convert to physical display units for the histogram.
        vmin_si, vmax_si = self._drs.resolve(arr)
        if vmin_si is not None:
            lo_phys = float(vmin_si) * scale
            hi_phys = float(vmax_si) * scale
        else:
            lo_phys, hi_phys = float(flat_phys.min()), float(flat_phys.max())

        bg = self._t.get("bg", "#1e1e2e")
        fg = self._t.get("fg", "#cdd6f4")
        self._fig.patch.set_facecolor(bg)
        self._ax.set_facecolor(bg)

        # Bin over a wide robust range (0.1–99.9 %) so bars represent useful
        # signal and are not stretched by outliers.  Uses the shared display
        # pipeline for consistent finite-pixel handling.
        try:
            counts, edges = _histogram_from_array(
                flat_phys, bins=128, clip_percentiles=(0.1, 99.9))
            x_min, x_max = float(edges[0]), float(edges[-1])
        except ValueError:
            counts, edges = np.histogram(flat_phys, bins=128)
            x_min, x_max = None, None

        counts = np.maximum(counts, 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)
        self._ax.bar(centers, counts, width=widths,
                     color=self._t.get("accent_bg", "#89b4fa"),
                     alpha=0.85, linewidth=0)
        self._ax.set_yscale("log")
        if x_min is not None:
            x0 = min(float(x_min), float(lo_phys), float(hi_phys))
            x1 = max(float(x_max), float(lo_phys), float(hi_phys))
            span = x1 - x0
            pad = 0.02 * span if span > 0 else max(abs(x0) * 0.02, 1.0)
            self._ax.set_xlim(x0 - pad, x1 + pad)
        self._low_line  = self._ax.axvline(lo_phys, color="#f38ba8",
                                            linewidth=1.6, picker=6)
        self._high_line = self._ax.axvline(hi_phys, color="#a6e3a1",
                                            linewidth=1.6, picker=6)

        self._ax.tick_params(axis="x", colors=fg, labelsize=7)
        self._ax.tick_params(axis="y", left=False, labelleft=False)
        self._ax.yaxis.set_visible(False)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(self._t.get("sep", "#45475a"))
        self._ax.set_xlabel(f"{axis_label} [{unit}]", fontsize=7, color=fg)
        self._fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.22)
        self._canvas.draw_idle()

        self._clip_val_lbl.setText(
            f"{lo_phys:.3g} {unit}  →  {hi_phys:.3g} {unit}")

    # ── Spec position overlay ─────────────────────────────────────────────────
    def _load_spec_markers(self, entry):
        """Show markers ONLY for spec files explicitly mapped to this image.

        Coordinate-based auto-matching used to be done here, but it
        attached spectra to the wrong scans for users with overlapping
        scan windows. Use the "Map spectra…" dialogs (folder-level on the
        toolbar, or per-image inside this viewer) to establish the
        spec→image mapping explicitly. Without a mapping, no markers
        appear — that's intentional, not a bug.
        """
        self._spec_markers = []
        self._zoom_lbl.set_markers([])

        if self._scan_range_m is None or self._scan_shape is None:
            return

        # Walk the spec→image mapping; only specs assigned to this stem are
        # candidates. We still need their coordinates to position the marker.
        from probeflow.io.file_type import FileType, sniff_file_type
        from probeflow.io.spectroscopy import read_spec_file
        from probeflow.analysis.spec_plot import spec_position_to_pixel, _parse_sxm_offset

        try:
            folder = entry.path.parent
            assigned_specs = {
                spec_stem for spec_stem, img_stem in self._spec_image_map.items()
                if img_stem == entry.stem
            }
            if not assigned_specs:
                return
            spec_types = (FileType.CREATEC_SPEC, FileType.NANONIS_SPEC)
            candidates = [
                f for f in sorted(folder.iterdir())
                if f.is_file()
                   and f.stem in assigned_specs
                   and sniff_file_type(f) in spec_types
            ]

            if self._scan_format == "sxm" and self._scan_header:
                scan_offset_m = _parse_sxm_offset(self._scan_header)
                raw_angle = self._scan_header.get("SCAN_ANGLE", "0").strip()
                try:
                    scan_angle_deg = float(raw_angle) if raw_angle else 0.0
                except ValueError:
                    scan_angle_deg = 0.0
            else:
                scan_offset_m = (0.0, 0.0)
                scan_angle_deg = 0.0

            markers = []
            for spec_path in candidates:
                try:
                    spec = read_spec_file(spec_path)
                    x_m, y_m = spec.position
                    result = spec_position_to_pixel(
                        x_m, y_m,
                        scan_shape=self._scan_shape,
                        scan_range_m=self._scan_range_m,
                        scan_offset_m=scan_offset_m,
                        scan_angle_deg=scan_angle_deg,
                    )
                    if result is None:
                        # User explicitly mapped this spec to this image, but
                        # the coordinates don't actually fall in-frame. Show
                        # the marker anyway, clamped to the centre, so the
                        # user can see the assignment exists.
                        frac_x, frac_y = 0.5, 0.5
                    else:
                        frac_x, frac_y = result
                    markers.append({
                        "frac_x": frac_x,
                        "frac_y": frac_y,
                        "entry": VertFile(
                            path=spec_path,
                            stem=spec_path.stem,
                            sweep_type=spec.metadata.get("sweep_type", "unknown"),
                            bias_mv=spec.metadata.get("bias_mv"),
                        ),
                    })
                except Exception:
                    continue

            # Build a parallel ROISet with the same positions as point ROIs.
            try:
                from probeflow.core.roi import ROI, ROISet
                _roi_set = ROISet(image_id=str(entry.path))
                for m in markers:
                    frac_x = float(m.get("frac_x", 0.5))
                    frac_y = float(m.get("frac_y", 0.5))
                    _entry = m.get("entry")
                    stem = getattr(_entry, "stem", None) or "spectrum"
                    name = f"spectrum_{stem}"
                    linked = str(getattr(_entry, "path", "") or "")
                    _shape = self._scan_shape or (1, 1)
                    px_x = frac_x * (_shape[1] - 1)
                    px_y = frac_y * (_shape[0] - 1)
                    _roi_set.add(ROI.new("point", {"x": px_x, "y": px_y},
                                         name=name,
                                         linked_file=linked or None))
                self._spec_roi_set = _roi_set
            except Exception:
                self._spec_roi_set = None
            self._spec_markers = markers
            if self._spec_show_cb.isChecked():
                self._zoom_lbl.set_markers(markers)
        except Exception:
            pass

    def _on_spec_show_toggled(self, checked: bool):
        if checked:
            self._zoom_lbl.set_markers(self._spec_markers)
        else:
            self._zoom_lbl.set_markers([])

    # ── Image-level ROI set ───────────────────────────────────────────────────

    def _load_image_roi_set(self, entry: "SxmFile") -> None:
        """Load ROIs from <stem>.rois.json sidecar if it exists, else create empty set."""
        from probeflow.core.roi import ROISet
        from probeflow.io.roi_sidecar import load_roi_set_sidecar
        try:
            loaded, _sidecar = load_roi_set_sidecar(entry.path, missing_ok=True)
        except Exception as exc:
            self._image_roi_set = ROISet(image_id=str(entry.path))
            if hasattr(self, "_status_lbl"):
                self._status_lbl.setText(f"Could not load ROI sidecar: {exc}")
        else:
            self._image_roi_set = loaded or ROISet(image_id=str(entry.path))
        self._zoom_lbl.set_roi_set(self._image_roi_set)
        if hasattr(self, "_roi_dock"):
            self._roi_dock.refresh(self._image_roi_set)
        self._sync_viewer_menu_actions()

    def _save_image_roi_set(self) -> None:
        """Persist the current ROISet to its sidecar file."""
        if self._image_roi_set is None:
            return
        entry = self._entries[self._idx]
        from probeflow.io.roi_sidecar import save_roi_set_sidecar
        try:
            save_roi_set_sidecar(self._image_roi_set, entry.path)
        except Exception as exc:
            if hasattr(self, "_status_lbl"):
                self._status_lbl.setText(f"Could not save ROI sidecar: {exc}")

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
        """A drawing tool completed; add the new ROI and make it active."""
        if self._image_roi_set is None:
            return
        self._image_roi_set.add(roi)
        self._image_roi_set.set_active(roi.id)
        self._on_image_roi_set_changed()
        # Canvas already switched to pan internally; sync toolbar
        self._set_drawing_tool("pan")

    def _on_canvas_roi_move(self, roi_id: str, dx: int, dy: int) -> None:
        """Active ROI was drag-moved on the canvas; translate its geometry."""
        if self._image_roi_set is None or (dx == 0 and dy == 0):
            return
        roi = self._image_roi_set.get(roi_id)
        if roi is None:
            return
        from probeflow.core.roi import translate as _translate_roi
        new_roi = _translate_roi(roi, float(dx), float(dy))
        self._image_roi_set.remove(roi_id)
        self._image_roi_set.add(new_roi)
        self._image_roi_set.set_active(new_roi.id)
        self._on_image_roi_set_changed()

    def _on_canvas_tool_changed(self, kind: str) -> None:
        """Canvas emitted tool_changed (e.g. after Escape or drawing completion)."""
        for btn in self._drawing_group.buttons():
            btn.setChecked(btn.property("drawing_tool") == kind)
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
        is_area = roi.kind in {"rectangle", "ellipse", "polygon", "freehand", "multipolygon"}
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
        act_bg_fit = menu.addAction("Background subtract (fit region)")
        act_bg_fit.setEnabled(is_area)
        act_bg_fit.triggered.connect(lambda: self._on_roi_bg_subtract_fit(roi_id))
        act_bg_exclude = menu.addAction("Background subtract (exclude region)")
        act_bg_exclude.setEnabled(is_area)
        act_bg_exclude.triggered.connect(lambda: self._on_roi_bg_subtract_exclude(roi_id))
        act_fft = menu.addAction("FFT this region")
        act_fft.setEnabled(is_area)
        act_fft.triggered.connect(lambda: self._on_roi_fft(roi_id))
        act_hist = menu.addAction("Histogram of this region")
        act_hist.setEnabled(is_area)
        act_hist.triggered.connect(lambda: self._on_roi_histogram(roi_id))
        act_profile = menu.addAction("Line profile")
        act_profile.setEnabled(is_line)
        act_profile.triggered.connect(lambda: self._on_roi_line_profile(roi_id))
        menu.exec(global_pos)

    # ── ROI helper actions ────────────────────────────────────────────────────

    def _set_active_image_roi(self, roi_id: str) -> None:
        if self._image_roi_set is None:
            return
        self._image_roi_set.set_active(roi_id)
        self._on_image_roi_set_changed()

    def _rename_image_roi(self, roi_id: str) -> None:
        from PySide6.QtWidgets import QInputDialog
        roi_set = self._image_roi_set
        roi = roi_set.get(roi_id) if roi_set else None
        if roi is None:
            return
        new_name, ok = QInputDialog.getText(self, "Rename ROI", "New name:", text=roi.name)
        if ok and new_name.strip():
            roi.name = new_name.strip()
            self._on_image_roi_set_changed()

    def _delete_image_roi(self, roi_id: str) -> None:
        if self._image_roi_set is None:
            return
        self._image_roi_set.remove(roi_id)
        self._on_image_roi_set_changed()

    def _delete_active_image_roi(self) -> None:
        if self._image_roi_set is None:
            return
        active_id = self._image_roi_set.active_roi_id
        if active_id is not None:
            self._delete_image_roi(active_id)

    def _invert_image_roi(self, roi_id: str) -> None:
        roi_set = self._image_roi_set
        roi = roi_set.get(roi_id) if roi_set else None
        if roi is None:
            return
        shape = self._current_array_shape()
        if shape is None:
            return
        from probeflow.core import roi as _roi_module
        inverted = _roi_module.invert(roi, shape)
        roi_set.add(inverted)
        self._on_image_roi_set_changed()

    def _invert_active_image_roi(self) -> None:
        if self._image_roi_set is None:
            return
        active_id = self._image_roi_set.active_roi_id
        if active_id is not None:
            self._invert_image_roi(active_id)

    def _select_nth_image_roi(self, n: int) -> None:
        if self._image_roi_set is None:
            return
        rois = list(self._image_roi_set.rois)
        if 1 <= n <= len(rois):
            roi_id = rois[n - 1].id
            self._image_roi_set.set_active(roi_id)
            self._on_image_roi_set_changed()

    # ── ROI operation callbacks ───────────────────────────────────────────────

    def _on_roi_bg_subtract_fit(self, roi_id: str) -> None:
        self._push_proc_undo_snapshot()
        self._processing.setdefault("bg_order", 1)
        self._processing["background_fit_roi_id"] = roi_id
        self._processing.pop("background_fit_rect", None)
        self._processing.pop("background_fit_geometry", None)
        self._refresh_processing_display()

    def _on_roi_bg_subtract_exclude(self, roi_id: str) -> None:
        self._push_proc_undo_snapshot()
        self._processing.setdefault("bg_order", 1)
        self._processing["background_exclude_roi_id"] = roi_id
        self._refresh_processing_display()

    def _on_roi_fft(self, roi_id: str) -> None:
        roi = self._image_roi_set.get(roi_id) if self._image_roi_set else None
        if roi is None or self._display_arr is None:
            return
        try:
            from probeflow.processing.image import fft_magnitude
            mag, _qx, _qy = fft_magnitude(self._display_arr, roi=roi)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "FFT",
                f"FFT computed for ROI '{roi.name}'.\n"
                f"Output shape: {mag.shape[0]} × {mag.shape[1]}",
            )
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "FFT error", str(exc))

    def _on_roi_histogram(self, roi_id: str) -> None:
        roi = self._image_roi_set.get(roi_id) if self._image_roi_set else None
        if roi is None or self._display_arr is None:
            return
        mask = roi.to_mask(self._display_arr.shape[:2])
        vals = self._display_arr[mask]
        from PySide6.QtWidgets import QMessageBox
        if len(vals) == 0:
            QMessageBox.information(self, "Histogram", "No pixels in ROI.")
            return
        scale, unit, _ = self._channel_unit()
        unit_str = f" {unit}" if unit else ""
        QMessageBox.information(
            self, f"Histogram: {roi.name}",
            f"Pixels: {len(vals)}\n"
            f"Min:  {float(vals.min()) * scale:.4g}{unit_str}\n"
            f"Max:  {float(vals.max()) * scale:.4g}{unit_str}\n"
            f"Mean: {float(vals.mean()) * scale:.4g}{unit_str}",
        )

    def _on_roi_line_profile(self, roi_id: str) -> None:
        roi = self._image_roi_set.get(roi_id) if self._image_roi_set else None
        if roi is None or roi.kind != "line" or self._display_arr is None:
            return
        try:
            px_x, px_y = self._pixel_size_xy_m()
            from probeflow.processing.image import line_profile
            s_m, values = line_profile(
                self._display_arr, roi=roi,
                pixel_size_x_m=px_x, pixel_size_y_m=px_y,
            )
            scale, unit, name = self._channel_unit()
            from probeflow.analysis.spec_plot import choose_display_unit
            x_scale, x_unit = choose_display_unit("m", s_m)
            self._line_profile_panel.setVisible(True)
            self._line_profile_panel.plot_profile(
                s_m * x_scale,
                values * scale,
                x_label=f"Distance [{x_unit}]",
                y_label=f"{name} [{unit}]" if unit else name,
                theme=self._t,
            )
            if hasattr(self._line_profile_panel, "set_source_label"):
                self._line_profile_panel.set_source_label(
                    f"Line ROI: {roi.name} ({roi.id[:8]})",
                    theme=self._t,
                )
        except Exception as exc:
            self._line_profile_panel.show_empty(str(exc), theme=self._t)

    def _on_map_spectra_here(self):
        """Open the per-image spec→this-image mapping dialog."""
        entry = self._entries[self._idx]
        # Find sibling .VERT files in the same folder.
        from probeflow.io.file_type import FileType, sniff_file_type
        try:
            spec_paths = sorted(
                f for f in entry.path.parent.iterdir()
                if f.is_file() and sniff_file_type(f) in (
                    FileType.CREATEC_SPEC, FileType.NANONIS_SPEC)
            )
        except Exception:
            spec_paths = []
        if not spec_paths:
            self._status_lbl.setText(
                "No spectroscopy files found alongside this image.")
            return
        # Build minimal VertFile placeholders (read_spec_file is slow; the
        # dialog only needs the stem).
        vert_entries = [VertFile(path=p, stem=p.stem) for p in spec_paths]
        dlg = ViewerSpecMappingDialog(
            entry.stem, vert_entries, self._spec_image_map, self)
        if dlg.exec() == QDialog.Accepted:
            new_map = dlg.updated_map()
            self._spec_image_map.clear()
            self._spec_image_map.update(new_map)
            n_for_this = sum(1 for v in new_map.values() if v == entry.stem)
            self._status_lbl.setText(
                f"{n_for_this} spec(s) mapped to this image. Reloading markers…")
            self._load_spec_markers(entry)

    def _on_marker_clicked(self, entry):
        dlg = SpecViewerDialog(entry, self._t, self)
        dlg.exec()

    def _current_array_shape(self) -> tuple[int, int] | None:
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        return None if arr is None else arr.shape

    def _active_image_roi_id(self) -> "str | None":
        if self._image_roi_set is None:
            return None
        return self._image_roi_set.active_roi_id

    def _active_image_roi(self):
        roi_id = self._active_image_roi_id()
        if self._image_roi_set is None or roi_id is None:
            return None
        return self._image_roi_set.get(roi_id)

    def _processing_has_roi_aware_local_filter(self, state: dict) -> bool:
        return bool(
            state.get("smooth_sigma")
            or state.get("highpass_sigma")
            or state.get("edge_method")
            or state.get("fft_mode") is not None
            or state.get("fft_soft_border")
        )

    def _selected_or_active_image_roi_id(self) -> "str | None":
        if hasattr(self, "_roi_dock"):
            try:
                selected = self._roi_dock._selected_roi_id()
            except Exception:
                selected = None
            if selected:
                return selected
        return self._active_image_roi_id()

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
                        "Radial FFT": self._processing_panel._fft_combo,
                    }.get(title)
                    current = combo.currentText() if combo is not None else ""
                    for label, action in value.items():
                        action.blockSignals(True)
                        action.setChecked(label == current)
                        action.blockSignals(False)
                    continue
                if key == "fft_soft_border":
                    value.blockSignals(True)
                    value.setChecked(self._processing_panel._fft_soft_cb.isChecked())
                    value.blockSignals(False)
                elif key == "zero_plane":
                    value.blockSignals(True)
                    value.setChecked(self._set_zero_plane_btn.isChecked())
                    value.blockSignals(False)
                elif key == "undo":
                    value.setEnabled(bool(self._proc_undo_stack))
                elif key == "redo":
                    value.setEnabled(bool(self._proc_redo_stack))

        if hasattr(self, "_viewer_roi_tool_actions"):
            tool = self._zoom_lbl.tool()
            for key, action in self._viewer_roi_tool_actions.items():
                action.blockSignals(True)
                action.setChecked(key == tool)
                action.blockSignals(False)

        if hasattr(self, "_viewer_roi_actions"):
            roi_id = self._selected_or_active_image_roi_id()
            roi = self._image_roi_set.get(roi_id) if (self._image_roi_set and roi_id) else None
            is_area = roi is not None and roi.kind in {
                "rectangle", "ellipse", "polygon", "freehand", "multipolygon"
            }
            for key, action in self._viewer_roi_actions.items():
                action.setEnabled(roi is not None)
                if key == "invert":
                    action.setEnabled(is_area)

    def _set_selection_tool(self, kind: str) -> None:
        """Compat shim: delegates to _set_drawing_tool, mapping 'none' → 'pan'."""
        self._set_drawing_tool(kind if kind and kind != "none" else "pan")

    def _set_drawing_tool(self, kind: str) -> None:
        """Activate a drawing tool both on the canvas and in the toolbar."""
        kind = str(kind or "pan")
        from probeflow.gui.tool_manager import TOOLS
        if kind not in TOOLS:
            kind = "pan"
        for btn in self._drawing_group.buttons():
            btn.setChecked(btn.property("drawing_tool") == kind)
        self._zoom_lbl.set_tool(kind)
        self._sync_line_profile_visibility(kind)
        from probeflow.gui.tool_manager import _TOOL_HINTS
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(_TOOL_HINTS.get(kind, ""))
        self._sync_viewer_menu_actions()

    def _on_selection_tool_clicked(self, button) -> None:
        """Compat shim kept for any lingering external references."""
        self._on_drawing_tool_clicked(button)

    def _on_drawing_tool_clicked(self, button) -> None:
        if self._set_zero_plane_btn.isChecked():
            self._set_zero_plane_btn.setChecked(False)
        kind = button.property("drawing_tool") or "pan"
        self._zoom_lbl.set_tool(kind)
        self._sync_line_profile_visibility(kind)
        from probeflow.gui.tool_manager import _TOOL_HINTS
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(_TOOL_HINTS.get(kind, ""))
        self._sync_viewer_menu_actions()

    def _active_line_roi_id(self) -> "str | None":
        """Return the active ROI id if it is a line ROI, else None."""
        if not self._image_roi_set:
            return None
        active_id = self._image_roi_set.active_roi_id
        if not active_id:
            return None
        roi = self._image_roi_set.get(active_id)
        return active_id if (roi and roi.kind == "line") else None

    def _sync_line_profile_visibility(self, kind: str | None = None) -> None:
        if not hasattr(self, "_line_profile_panel"):
            return
        tool_is_line = (kind or self._zoom_lbl.selection_tool()) == "line"
        active_line_id = self._active_line_roi_id()
        is_line = tool_is_line or (active_line_id is not None)
        self._line_profile_panel.setVisible(is_line)
        if is_line:
            if active_line_id is not None:
                # Prefer the ROI-based line profile over the old selection geometry
                self._on_roi_line_profile(active_line_id)
            else:
                self._refresh_line_profile_from_selection()
        else:
            self._line_profile_geometry = None
            self._line_profile_panel.show_empty(theme=self._t)

    def _selection_geometry_to_pixels(self, geometry: dict | None) -> dict | None:
        shape = self._current_array_shape()
        if not geometry or shape is None:
            return None
        Ny, Nx = shape
        kind = str(geometry.get("kind", ""))
        if kind not in {"rectangle", "ellipse", "polygon", "line"}:
            return None
        out = {"kind": kind}
        if geometry.get("bounds_frac") is not None:
            try:
                x0f, y0f, x1f, y1f = [float(v) for v in geometry["bounds_frac"]]
            except (TypeError, ValueError):
                return None
            x0 = max(0, min(Nx - 1, int(round(min(x0f, x1f) * (Nx - 1)))))
            x1 = max(0, min(Nx - 1, int(round(max(x0f, x1f) * (Nx - 1)))))
            y0 = max(0, min(Ny - 1, int(round(min(y0f, y1f) * (Ny - 1)))))
            y1 = max(0, min(Ny - 1, int(round(max(y0f, y1f) * (Ny - 1)))))
            if x1 <= x0 or y1 <= y0:
                return None
            out["bounds_frac"] = tuple(float(v) for v in geometry["bounds_frac"])
            out["rect_px"] = (x0, y0, x1, y1)
        if geometry.get("points_frac") is not None:
            points_px = []
            points_frac = []
            for item in geometry.get("points_frac", ()):
                try:
                    xf, yf = float(item[0]), float(item[1])
                except (TypeError, ValueError, IndexError):
                    continue
                xf = max(0.0, min(1.0, xf))
                yf = max(0.0, min(1.0, yf))
                points_frac.append((xf, yf))
                points_px.append((
                    max(0, min(Nx - 1, int(round(xf * (Nx - 1))))),
                    max(0, min(Ny - 1, int(round(yf * (Ny - 1))))),
                ))
            if out["kind"] == "polygon" and len(points_px) < 3:
                return None
            if out["kind"] == "line" and len(points_px) < 2:
                return None
            out["points_frac"] = points_frac
            out["points_px"] = points_px
        return out

    def _area_selection_geometry_px(self) -> dict | None:
        geometry = self._selection_geometry
        if not geometry:
            return None
        if geometry.get("kind") == "line":
            return None
        return geometry if geometry.get("kind") in {"rectangle", "ellipse", "polygon"} else None

    def _selection_status_text(self, geometry: dict | None) -> str:
        if not geometry:
            return "Selection: none"
        kind = geometry.get("kind", "selection")
        if kind == "line" and geometry.get("points_px"):
            (x0, y0), (x1, y1) = geometry["points_px"][:2]
            return f"Selection: line ({x0}, {y0}) → ({x1}, {y1}); display only"
        if kind == "polygon" and geometry.get("points_px"):
            return f"Selection: polygon, {len(geometry['points_px'])} vertices"
        if geometry.get("rect_px"):
            x0, y0, x1, y1 = geometry["rect_px"]
            return (
                f"Selection: {kind}, x {x0}-{x1}, y {y0}-{y1} "
                f"({x1 - x0 + 1} x {y1 - y0 + 1} px)"
            )
        return f"Selection: {kind}"

    def _on_selection_preview_changed(self, geometry) -> None:
        converted = self._selection_geometry_to_pixels(dict(geometry or {}))
        if converted is None or converted.get("kind") != "line":
            self._line_profile_geometry = None
            if self._zoom_lbl.selection_tool() == "line":
                self._line_profile_panel.show_empty(theme=self._t)
            return
        self._line_profile_geometry = converted
        self._refresh_line_profile(converted)

    def _on_selection_changed(self, geometry) -> None:
        converted = self._selection_geometry_to_pixels(dict(geometry or {}))
        if converted is None:
            self._selection_geometry = None
            self._roi_rect_px = None
            self._roi_status_lbl.setText("Selection: none")
            self._line_profile_geometry = None
            if self._zoom_lbl.selection_tool() == "line":
                self._line_profile_panel.show_empty(theme=self._t)
            return
        self._selection_geometry = converted
        self._roi_rect_px = (
            converted.get("rect_px") if converted.get("kind") == "rectangle" else None
        )
        self._roi_status_lbl.setText(self._selection_status_text(converted))
        if converted.get("kind") == "line":
            self._line_profile_geometry = converted
            self._refresh_line_profile(converted)
        elif self._zoom_lbl.selection_tool() == "line":
            self._line_profile_geometry = None
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

    def _refresh_line_profile_from_selection(self) -> None:
        if not hasattr(self, "_line_profile_panel"):
            return
        if self._zoom_lbl.selection_tool() != "line":
            return
        geometry = self._line_profile_geometry
        if (
            geometry is None
            and self._selection_geometry
            and self._selection_geometry.get("kind") == "line"
        ):
            geometry = self._selection_geometry
        if geometry is None:
            current = self._zoom_lbl.current_selection()
            geometry = self._selection_geometry_to_pixels(current) if current else None
        if geometry is None or geometry.get("kind") != "line":
            self._line_profile_panel.show_empty(theme=self._t)
            return
        self._line_profile_geometry = geometry
        self._refresh_line_profile(geometry)

    def _refresh_line_profile(self, geometry: dict | None = None) -> None:
        if not hasattr(self, "_line_profile_panel"):
            return
        if self._zoom_lbl.selection_tool() != "line":
            return
        arr = self._display_arr
        geometry = geometry or self._line_profile_geometry
        if arr is None or not geometry or geometry.get("kind") != "line":
            self._line_profile_panel.show_empty(theme=self._t)
            return
        points = geometry.get("points_px") or []
        if len(points) < 2:
            self._line_profile_panel.show_empty(theme=self._t)
            return
        try:
            px_x, px_y = self._pixel_size_xy_m()
            s_m, values = _proc.line_profile(
                arr,
                tuple(points[0]),
                tuple(points[1]),
                pixel_size_x_m=px_x,
                pixel_size_y_m=px_y,
                width_px=1.0,
                interp="linear",
            )
            scale, unit, axis_label = self._channel_unit()
            y_label = f"{axis_label} [{unit}]" if unit else axis_label
            from probeflow.analysis.spec_plot import choose_display_unit
            x_scale, x_unit = choose_display_unit("m", s_m)
            x_label = f"Distance [{x_unit}]"
            self._line_profile_panel.plot_profile(
                s_m * x_scale,
                values.astype(np.float64) * scale,
                x_label=x_label,
                y_label=y_label,
                theme=self._t,
            )
            if hasattr(self._line_profile_panel, "set_source_label"):
                self._line_profile_panel.set_source_label(None, theme=self._t)
        except Exception as exc:
            self._line_profile_panel.show_empty(
                f"Profile unavailable: {exc}",
                theme=self._t,
            )

    def _nudge_line_profile(self, key: int) -> bool:
        if not hasattr(self, "_zoom_lbl"):
            return False
        if self._zoom_lbl.selection_tool() != "line":
            return False
        if not (self._selection_geometry and self._selection_geometry.get("kind") == "line"):
            return False
        dx = dy = 0
        if key == Qt.Key_Left:
            dx = -1
        elif key == Qt.Key_Right:
            dx = 1
        elif key == Qt.Key_Up:
            dy = -1
        elif key == Qt.Key_Down:
            dy = 1
        else:
            return False
        return self._zoom_lbl.nudge_line(dx, dy, self._current_array_shape())

    def _on_set_zero_plane_mode_toggled(self, checked: bool):
        cleared_partial_points = False
        if checked:
            self._set_selection_tool("none")
            self._zero_pick_mode = "plane"
            self._zero_plane_points_px = []
            self._zero_markers_hidden = False
            self._status_lbl.setText("Click 3 reference points to define the zero plane.")
        elif self._zero_pick_mode == "plane" and len(self._zero_plane_points_px) < 3:
            self._zero_plane_points_px = []
            cleared_partial_points = True
        self._zoom_lbl.set_set_zero_mode(checked)
        if cleared_partial_points:
            self._refresh_zero_markers()

    def _on_clear_roi(self):
        had_processing_selection = any(
            key in self._processing
            for key in (
                "processing_scope",
                "processing_roi_id",
                "roi_rect",
                "roi_geometry",
                "background_fit_rect",
                "background_fit_geometry",
                "patch_interpolate_rect",
                "patch_interpolate_geometry",
                "patch_interpolate_iterations",
            )
        )
        self._roi_rect_px = None
        self._selection_geometry = None
        self._processing.pop("processing_scope", None)
        self._processing.pop("processing_roi_id", None)
        self._processing.pop("roi_rect", None)
        self._processing.pop("roi_geometry", None)
        self._processing.pop("background_fit_rect", None)
        self._processing.pop("background_fit_geometry", None)
        self._processing.pop("patch_interpolate_rect", None)
        self._processing.pop("patch_interpolate_geometry", None)
        self._processing.pop("patch_interpolate_iterations", None)
        self._zoom_lbl.clear_roi()
        self._set_selection_tool("none")
        self._scope_cb.setCurrentIndex(0)
        self._bg_fit_roi_cb.setChecked(False)
        self._patch_roi_cb.setChecked(False)
        self._roi_status_lbl.setText("Selection: none")
        if had_processing_selection:
            self._refresh_processing_display()

    def _on_set_zero_pick(self, frac_x: float, frac_y: float):
        """Handle image clicks while manual zero-plane mode is active."""
        if self._raw_arr is None:
            return
        Ny, Nx = self._raw_arr.shape
        x_px = int(round(frac_x * (Nx - 1)))
        y_px = int(round(frac_y * (Ny - 1)))
        x_px = max(0, min(x_px, Nx - 1))
        y_px = max(0, min(y_px, Ny - 1))

        if self._zero_pick_mode == "plane" and self._set_zero_plane_btn.isChecked():
            self._zero_markers_hidden = False
            self._zero_plane_points_px.append((x_px, y_px))
            n = len(self._zero_plane_points_px)
            self._refresh_zero_markers()  # show partial pick immediately
            if n < 3:
                self._status_lbl.setText(
                    f"Zero plane point {n}/3 set at ({x_px}, {y_px}); click {3 - n} more."
                )
                return
            self._processing['set_zero_plane_points'] = self._zero_plane_points_px[:3]
            self._processing['set_zero_patch'] = 1
            self._processing.pop('set_zero_xy', None)
            if self._set_zero_plane_btn.isChecked():
                self._set_zero_plane_btn.setChecked(False)
            self._status_lbl.setText("Zero plane set from 3 reference points.")
            self._refresh_processing_display()
            return

        return

    def _refresh_zero_markers(self):
        """Push the current set-zero pick state into _ZoomLabel for drawing.

        Sources, in order of priority:
          1. In-progress plane picks (``self._zero_plane_points_px``).
          2. Committed plane points (``processing['set_zero_plane_points']``).
          3. Legacy committed single-point zero (``processing['set_zero_xy']``).
        """
        if self._raw_arr is None:
            self._zoom_lbl.set_zero_markers([])
            return
        if self._zero_markers_hidden:
            self._zoom_lbl.set_zero_markers([])
            return
        Ny, Nx = self._raw_arr.shape
        denom_x = max(1, Nx - 1)
        denom_y = max(1, Ny - 1)

        def _to_marker(pt, label):
            x_px, y_px = pt
            return {
                "frac_x": float(x_px) / denom_x,
                "frac_y": float(y_px) / denom_y,
                "label": label,
            }

        markers: list[dict] = []
        if self._zero_plane_points_px:
            for i, pt in enumerate(self._zero_plane_points_px[:3]):
                markers.append(_to_marker(pt, str(i + 1)))
        elif self._processing.get("set_zero_plane_points"):
            for i, pt in enumerate(self._processing["set_zero_plane_points"][:3]):
                markers.append(_to_marker(pt, str(i + 1)))
        elif self._processing.get("set_zero_xy") is not None:
            markers.append(_to_marker(self._processing["set_zero_xy"], "0"))
        self._zoom_lbl.set_zero_markers(markers)

    def _on_clear_set_zero(self):
        self._zero_plane_points_px = []
        self._zero_markers_hidden = True
        if self._set_zero_plane_btn.isChecked():
            self._set_zero_plane_btn.setChecked(False)
        self._zoom_lbl.set_zero_markers([])
        self._status_lbl.setText(
            "Zero reference markers hidden. Processing is unchanged; use Reset to original to undo leveling."
        )

    # ── Histogram drag handlers ────────────────────────────────────────────────
    def _on_hist_press(self, event):
        if (event.inaxes is not self._ax or event.xdata is None
                or event.button != 1
                or self._low_line is None or self._high_line is None):
            return
        lo = self._low_line.get_xdata()[0]
        hi = self._high_line.get_xdata()[0]
        x0, x1 = self._ax.get_xlim()
        tol = 0.04 * (x1 - x0) if x1 > x0 else 0.0
        d_lo = abs(event.xdata - lo)
        d_hi = abs(event.xdata - hi)
        # pick closest line; if far from both, move the closer one
        if d_lo <= d_hi:
            self._dragging = 'low'
        else:
            self._dragging = 'high'
        # only engage drag if within tolerance OR click outside both lines
        if min(d_lo, d_hi) > tol and (lo <= event.xdata <= hi):
            self._dragging = None

    def _on_hist_motion(self, event):
        if (self._dragging is None or event.inaxes is not self._ax
                or event.xdata is None
                or self._low_line is None or self._high_line is None):
            return
        x = float(event.xdata)
        lo = self._low_line.get_xdata()[0]
        hi = self._high_line.get_xdata()[0]
        if self._dragging == 'low':
            x = min(x, hi - 1e-12)
            self._low_line.set_xdata([x, x])
        else:
            x = max(x, lo + 1e-12)
            self._high_line.set_xdata([x, x])
        if self._hist_flat_phys is not None and self._clip_val_lbl is not None:
            new_lo = self._low_line.get_xdata()[0]
            new_hi = self._high_line.get_xdata()[0]
            self._clip_val_lbl.setText(
                f"{new_lo:.3g} {self._hist_unit}  →  {new_hi:.3g} {self._hist_unit}")
        self._canvas.draw_idle()

    def _on_hist_release(self, event):
        if self._dragging is None or self._hist_flat_phys is None:
            self._dragging = None
            return
        lo_x = float(self._low_line.get_xdata()[0])
        hi_x = float(self._high_line.get_xdata()[0])
        # Convert physical display units (Å or pA) back to SI array units.
        scale, _, _ = self._channel_unit()
        vmin_si = lo_x / scale
        vmax_si = hi_x / scale
        self._drs.set_manual(vmin_si, vmax_si)
        self._dragging = None
        self._refresh_display_range()

    def _on_auto_clip(self):
        """Reset to 1%–99% percentile autoscale."""
        self._drs.reset()
        self._clip_low  = 1.0
        self._clip_high = 99.0
        self._refresh_display_range()

    def _on_hist_context_menu(self, pos):
        menu = QMenu(self)
        auto_action = menu.addAction("Auto display range")
        export_action = menu.addAction("Export histogram...")
        chosen = menu.exec(self._canvas.mapToGlobal(pos))
        if chosen is auto_action:
            self._on_auto_clip()
        elif chosen is export_action:
            self._on_export_histogram()

    def _on_export_histogram(self):
        """Save the current histogram (bin centres + counts) as a TSV file."""
        flat = self._hist_flat_phys
        if flat is None or flat.size < 2:
            self._status_lbl.setText("No histogram data to export.")
            return
        entry = self._entries[self._idx]
        unit = self._hist_unit or ""
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export histogram",
            str(Path.home() / f"{entry.stem}_histogram.txt"),
            "Text files (*.txt *.tsv *.csv)",
        )
        if not out_path:
            return
        try:
            n_bins = 256
            counts, edges = np.histogram(flat, bins=n_bins)
            centres = 0.5 * (edges[:-1] + edges[1:])
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write("# ProbeFlow histogram export\n")
                fh.write(f"# source: {entry.stem}\n")
                fh.write(f"# channel: {self._ch_cb.currentText()}\n")
                fh.write(f"# n_samples: {flat.size}\n")
                fh.write(f"# n_bins: {n_bins}\n")
                fh.write(f"# unit: {unit}\n")
                fh.write(f"bin_center_{unit}\tcount\n")
                for c, n in zip(centres, counts):
                    fh.write(f"{c:.8g}\t{int(n)}\n")
            self._status_lbl.setText(f"Histogram → {out_path}")
        except Exception as exc:
            self._status_lbl.setText(f"Export error: {exc}")

    def _on_channel_changed(self, _: int):
        # Different channels have different physical units — reset manual limits.
        self._drs.reset(self._clip_low, self._clip_high)
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
    def _advanced_processing_state(self) -> dict:
        if not hasattr(self, "_undistort_shear_spin"):
            return {}
        shear_x = float(self._undistort_shear_spin.value())
        scale_y = float(self._undistort_scale_spin.value())
        return {
            "linear_undistort": (shear_x != 0.0 or scale_y != 1.0),
            "undistort_shear_x": shear_x,
            "undistort_scale_y": scale_y,
        }

    def _set_advanced_processing_state(self, state: dict | None) -> None:
        if not hasattr(self, "_undistort_shear_spin"):
            return
        state = state or {}
        self._undistort_shear_spin.setValue(float(state.get("undistort_shear_x", 0.0)))
        self._undistort_scale_spin.setValue(float(state.get("undistort_scale_y", 1.0)))

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
        active_area_roi_id = (
            active_roi.id
            if active_roi is not None
            and active_roi.kind in {"rectangle", "ellipse", "polygon", "freehand", "multipolygon"}
            else None
        )
        wants_filter_roi = (
            self._scope_cb.currentIndex() == 1
            or (active_area_roi_id is not None and has_roi_aware_local_filter)
        )
        wants_bg_fit_roi = self._bg_fit_roi_cb.isChecked()
        wants_patch_roi = self._patch_roi_cb.isChecked()
        selection_geometry = self._area_selection_geometry_px()
        if (
            active_roi is not None
            and active_area_roi_id is None
            and has_roi_aware_local_filter
        ):
            self._status_lbl.setText(
                f"Active {active_roi.kind} ROI is not valid for area processing; "
                "select an area ROI or delete/deselect it before applying local filters."
            )
            return
        if wants_filter_roi or wants_bg_fit_roi or wants_patch_roi:
            if (
                active_area_roi_id is None
                and self._selection_geometry
                and self._selection_geometry.get("kind") == "line"
            ):
                self._status_lbl.setText(
                    "Line selections are display-only; choose an area selection for processing."
                )
                return
            if active_area_roi_id is None and selection_geometry is None:
                self._status_lbl.setText("Select an area before using selection-based processing.")
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
                "background_fit_roi_id",
                "background_exclude_roi_id",
                "stm_background",
            )
            if key in self._processing
        }
        self._processing = panel_state
        self._processing.update(preserve)
        if wants_filter_roi and active_area_roi_id is not None:
            self._processing["processing_scope"] = "roi"
            self._processing["processing_roi_id"] = active_area_roi_id
            self._processing.pop("roi_rect", None)
            self._processing.pop("roi_geometry", None)
        elif wants_filter_roi:
            self._processing["processing_scope"] = "roi"
            self._processing.pop("processing_roi_id", None)
            self._processing["roi_geometry"] = dict(selection_geometry)
            if selection_geometry.get("kind") == "rectangle":
                self._processing["roi_rect"] = selection_geometry.get("rect_px")
            else:
                self._processing.pop("roi_rect", None)
        else:
            self._processing.pop("processing_scope", None)
            self._processing.pop("processing_roi_id", None)
            self._processing.pop("roi_rect", None)
            self._processing.pop("roi_geometry", None)
        if wants_bg_fit_roi and self._processing.get("bg_order") is not None:
            self._processing["background_fit_geometry"] = dict(selection_geometry)
            if selection_geometry.get("kind") == "rectangle":
                self._processing["background_fit_rect"] = selection_geometry.get("rect_px")
            else:
                self._processing.pop("background_fit_rect", None)
        else:
            self._processing.pop("background_fit_rect", None)
            self._processing.pop("background_fit_geometry", None)
        if wants_patch_roi:
            self._processing["patch_interpolate_geometry"] = dict(selection_geometry)
            if selection_geometry.get("kind") == "rectangle":
                self._processing["patch_interpolate_rect"] = selection_geometry.get("rect_px")
            else:
                self._processing.pop("patch_interpolate_rect", None)
            self._processing["patch_interpolate_iterations"] = 200
            self._processing["patch_interpolate_method"] = (
                "line_fit" if self._patch_method_combo.currentIndex() == 0 else "laplace"
            )
        else:
            self._processing.pop("patch_interpolate_rect", None)
            self._processing.pop("patch_interpolate_geometry", None)
            self._processing.pop("patch_interpolate_iterations", None)
        self._clear_bad_line_preview()
        self._refresh_processing_display()

    def _on_reset_processing(self):
        """Clear all processing for the current image and reload raw data."""
        has_selection = self._selection_geometry is not None or self._roi_rect_px is not None
        has_zero = bool(self._zero_plane_points_px)
        if not self._processing and not has_selection and not has_zero:
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
        self._zero_plane_points_px = []
        self._zero_markers_hidden = False
        self._roi_rect_px = None
        self._selection_geometry = None
        self._zoom_lbl.clear_roi()
        self._set_selection_tool("none")
        self._scope_cb.setCurrentIndex(0)
        self._bg_fit_roi_cb.setChecked(False)
        self._patch_roi_cb.setChecked(False)
        self._roi_status_lbl.setText("Selection: none")
        self._refresh_zero_markers()
        self._status_lbl.setText("Reset: showing original on-disk data.")
        self._refresh_processing_display()

    # ── Processing undo / redo ────────────────────────────────────────────────

    _PROC_UNDO_DEPTH = 50

    def _push_proc_undo_snapshot(self) -> None:
        """Record the current processing state for Undo, then clear redo.

        Called by Apply / Reset (after their validation passes) before any
        mutation of ``self._processing``. The snapshot is a deep copy so
        nested ROI / geometry dicts don't alias the live state.
        """
        self._proc_undo_stack.append(copy.deepcopy(self._processing))
        if len(self._proc_undo_stack) > self._PROC_UNDO_DEPTH:
            self._proc_undo_stack.pop(0)
        self._proc_redo_stack.clear()
        self._update_undo_redo_buttons()

    def _restore_processing_state(self, state: dict) -> None:
        """Apply a snapshot to ``self._processing`` and resync the GUI."""
        self._processing = copy.deepcopy(state)
        self._processing_panel.set_state(self._processing)
        self._set_advanced_processing_state(self._processing)
        self._refresh_processing_display()

    def _on_undo_processing(self) -> None:
        if not self._proc_undo_stack:
            return
        self._proc_redo_stack.append(copy.deepcopy(self._processing))
        previous = self._proc_undo_stack.pop()
        self._restore_processing_state(previous)
        self._status_lbl.setText("Undo: restored previous processing.")
        self._update_undo_redo_buttons()

    def _on_redo_processing(self) -> None:
        if not self._proc_redo_stack:
            return
        self._proc_undo_stack.append(copy.deepcopy(self._processing))
        target = self._proc_redo_stack.pop()
        self._restore_processing_state(target)
        self._status_lbl.setText("Redo: reapplied processing.")
        self._update_undo_redo_buttons()

    def _update_undo_redo_buttons(self) -> None:
        if self._proc_undo_btn is not None:
            self._proc_undo_btn.setEnabled(bool(self._proc_undo_stack))
        if self._proc_redo_btn is not None:
            self._proc_redo_btn.setEnabled(bool(self._proc_redo_stack))
        self._sync_viewer_menu_actions()

    def _on_save_png(self):
        entry = self._entries[self._idx]
        if getattr(self, "_processing_roi_error", ""):
            self._status_lbl.setText(
                f"Cannot export while processing has stale ROI references. "
                f"{self._processing_roi_error}"
            )
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save PNG", str(Path.home() / f"{entry.stem}_viewer.png"),
            "PNG images (*.png)")
        if not out_path:
            return
        # Save the same array the viewer is displaying (processed if active)
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No data to save.")
            return
        try:
            try:
                _scan = load_scan(entry.path)
                w_m, h_m = _scan.scan_range_m
            except Exception:
                _scan = None
                w_m = h_m = 0.0
            vmin, vmax = self._drs.resolve(arr)
            provenance = None
            if _scan is not None:
                try:
                    ch_idx = self._ch_cb.currentIndex()
                    ps = processing_state_from_gui(self._processing or {})
                    provenance = build_scan_export_provenance(
                        _scan,
                        channel_index=ch_idx,
                        channel_name=self._ch_cb.currentText() or None,
                        processing_state=ps,
                        display_state=png_display_state(
                            self._drs,
                            colormap=self._colormap,
                            add_scalebar=True,
                            scalebar_unit="nm",
                            scalebar_pos="bottom-right",
                        ),
                        export_kind="viewer_png",
                        output_path=out_path,
                        roi_set=self._image_roi_set,
                    )
                except Exception:
                    pass
            _proc.export_png(
                arr, out_path, self._colormap,
                self._clip_low, self._clip_high,
                lut_fn=lambda key: _get_lut(key),
                scan_range_m=(w_m, h_m),
                vmin=vmin, vmax=vmax,
                provenance=provenance,
            )
            self._status_lbl.setText(f"Saved → {Path(out_path).name}")
        except Exception as exc:
            self._status_lbl.setText(f"Export error: {exc}")

    def _on_send_to_features(self):
        self._deferred_action = "features"
        self._deferred_plane_idx = self._ch_cb.currentIndex()
        self.accept()

    def _on_send_to_tv(self):
        self._deferred_action = "tv"
        self._deferred_plane_idx = self._ch_cb.currentIndex()
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
        """Keep GUI ROI coordinates in the same frame as display transforms."""
        if self._image_roi_set is None or not self._image_roi_set.rois:
            return
        shape = self._current_array_shape()
        if shape is None:
            return
        params = params or {}
        invalidated = self._image_roi_set.transform_all(op_name, params, shape)
        if invalidated:
            invalid = set(invalidated)
            self._image_roi_set.rois = [
                roi for roi in self._image_roi_set.rois if roi.id not in invalid
            ]
            if self._image_roi_set.active_roi_id in invalid:
                self._image_roi_set.active_roi_id = None
            if hasattr(self, "_status_lbl"):
                self._status_lbl.setText(
                    f"{op_name} invalidated {len(invalidated)} ROI(s); removed them."
                )
        self._on_image_roi_set_changed()

    def _on_export_line_profile_csv(self):
        prof = self._line_profile_panel.profile_data()
        if prof is None:
            self._status_lbl.setText("No line profile to export (draw a line first).")
            return
        x_vals, y_vals, x_label, y_label = prof
        entry = self._entries[self._idx]
        hdr = self._scan_header or {}
        bias_mv = None
        current_a = None
        for k, v in hdr.items():
            kl = k.lower()
            if "biasvolt" in kl or "vgap" in kl:
                try:
                    bias_mv = float(str(v).replace(",", "."))
                except (ValueError, TypeError):
                    pass
            elif k == "Current[A]":
                try:
                    current_a = float(str(v).replace(",", "."))
                except (ValueError, TypeError):
                    pass
        parts = [entry.stem]
        if bias_mv is not None:
            parts.append(f"{bias_mv:.0f}mV")
        if current_a is not None:
            parts.append(f"{current_a * 1e12:.0f}pA")
        suggested_name = "_".join(parts) + "_lineprofile.csv"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export line profile as CSV",
            str(Path.home() / suggested_name),
            "CSV files (*.csv)")
        if not out_path:
            return
        try:
            import csv
            with open(out_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["# File", entry.stem])
                if bias_mv is not None:
                    w.writerow(["# Bias (mV)", f"{bias_mv:.3f}"])
                if current_a is not None:
                    w.writerow(["# Setpoint current (A)", f"{current_a:.3e}"])
                w.writerow([x_label, y_label])
                for x, y in zip(x_vals, y_vals):
                    w.writerow([f"{float(x):.6g}", f"{float(y):.6g}"])
            self._status_lbl.setText(f"Profile → {Path(out_path).name}")
        except Exception as exc:
            self._status_lbl.setText(f"CSV export error: {exc}")


# ── Spec → image mapping dialogs ─────────────────────────────────────────────
class BrowseToolPanel(QWidget):
    """Left-side control panel for browsing and live thumbnail appearance."""
    open_folder_requested      = Signal()
    colormap_changed           = Signal(str)
    thumbnail_align_changed    = Signal(str)
    map_spectra_requested      = Signal()
    filter_changed             = Signal(str)   # "all" | "images" | "spectra"
    thumbnail_channel_changed  = Signal(str)

    def __init__(self, t: dict, cfg: dict, parent=None):
        super().__init__(parent)
        self._t            = t
        self._filter_mode  = cfg.get("browse_filter", "all")
        self._build(cfg)

    def _build(self, cfg: dict):
        # Wrap everything in a scroll area so nothing gets clipped on small screens
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 6)
        lay.setSpacing(6)

        # ── Open folder button ─────────────────────────────────────────────────
        open_btn = QPushButton("Open folder…")
        open_btn.setFont(QFont("Helvetica", 9))
        open_btn.setFixedHeight(30)
        open_btn.setCursor(QCursor(Qt.PointingHandCursor))
        open_btn.setObjectName("accentBtn")
        open_btn.clicked.connect(self.open_folder_requested.emit)
        lay.addWidget(open_btn)

        # ── Filter toggle (All / Images / Spectra) ─────────────────────────────
        filter_row = QWidget()
        filter_lay = QHBoxLayout(filter_row)
        filter_lay.setContentsMargins(0, 0, 0, 0)
        filter_lay.setSpacing(0)

        self._filter_group = QButtonGroup(self)
        self._filter_group.setExclusive(True)
        self._filter_btns: dict[str, QPushButton] = {}
        _modes = [("All", "all"), ("Images", "images"), ("Spectra", "spectra")]
        for i, (label, mode) in enumerate(_modes):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFont(QFont("Helvetica", 9))
            btn.setFixedHeight(26)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            # Segmented-control shape: rounded corners only on outer edges
            if i == 0:
                btn.setObjectName("segBtnLeft")
            elif i == len(_modes) - 1:
                btn.setObjectName("segBtnRight")
            else:
                btn.setObjectName("segBtnMid")
            btn.clicked.connect(lambda _c=False, m=mode: self._on_filter_click(m))
            self._filter_group.addButton(btn)
            filter_lay.addWidget(btn, 1)
            self._filter_btns[mode] = btn

        # Set the initially checked button from config.
        initial = self._filter_mode if self._filter_mode in self._filter_btns else "all"
        self._filter_btns[initial].setChecked(True)
        self._filter_mode = initial

        lay.addWidget(filter_row)
        lay.addWidget(_sep())

        # ── Thumbnail appearance ──────────────────────────────────────────────
        appearance_lbl = QLabel("Thumbnail appearance")
        appearance_lbl.setFont(QFont("Helvetica", 11, QFont.Bold))
        lay.addWidget(appearance_lbl)

        cm_lbl = QLabel("Colormap")
        cm_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        lay.addWidget(cm_lbl)

        self.cmap_cb = QComboBox()
        self.cmap_cb.addItems(CMAP_NAMES)
        self.cmap_cb.setCurrentText(cfg.get("colormap", DEFAULT_CMAP_LABEL))
        self.cmap_cb.setFont(QFont("Helvetica", 10))
        self.cmap_cb.currentTextChanged.connect(self._on_colormap_changed)
        lay.addWidget(self.cmap_cb)

        thumb_lbl = QLabel("Thumbnail channel")
        thumb_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        lay.addWidget(thumb_lbl)

        self.thumbnail_channel_cb = QComboBox()
        self.thumbnail_channel_cb.addItems(THUMBNAIL_CHANNEL_OPTIONS)
        self.thumbnail_channel_cb.setCurrentText(THUMBNAIL_CHANNEL_DEFAULT)
        self.thumbnail_channel_cb.setFont(QFont("Helvetica", 10))
        self.thumbnail_channel_cb.setToolTip(
            "Choose which forward scan channel is used for browse thumbnails. "
            "Files without that channel fall back to Z."
        )
        self.thumbnail_channel_cb.currentTextChanged.connect(
            self.thumbnail_channel_changed.emit)
        lay.addWidget(self.thumbnail_channel_cb)

        align_lbl = QLabel("Align rows")
        align_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        lay.addWidget(align_lbl)
        self.align_rows_cb = QComboBox()
        self.align_rows_cb.addItems(["None", "Median", "Mean"])
        self.align_rows_cb.setCurrentText("None")
        self.align_rows_cb.setFont(QFont("Helvetica", 10))
        self.align_rows_cb.setToolTip(
            "Preview-only thumbnail row alignment. Full-size viewer data opens raw."
        )
        self.align_rows_cb.currentTextChanged.connect(self._on_align_changed)
        lay.addWidget(self.align_rows_cb)
        lay.addWidget(_sep())

        self._map_spectra_btn = QPushButton("Map spectra to images\u2026")
        self._map_spectra_btn.setFont(QFont("Helvetica", 9))
        self._map_spectra_btn.setFixedHeight(28)
        self._map_spectra_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._map_spectra_btn.setToolTip(
            "Pick the parent image for each .VERT spectrum in the current "
            "folder. Spectra without a mapping show no marker on any image. "
            "You can also map per-image inside the viewer.")
        self._map_spectra_btn.clicked.connect(self.map_spectra_requested.emit)
        lay.addWidget(self._map_spectra_btn)

        lay.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

    # ── Slots ──────────────────────────────────────────────────────────────────
    def _on_colormap_changed(self):
        cmap_key = CMAP_KEY.get(self.cmap_cb.currentText(), DEFAULT_CMAP_KEY)
        self.colormap_changed.emit(cmap_key)

    def _on_align_changed(self, text: str):
        self.thumbnail_align_changed.emit(text)

    def _on_filter_click(self, mode: str):
        self._filter_mode = mode
        self.filter_changed.emit(mode)

    # ── Public API ─────────────────────────────────────────────────────────────
    def get_filter_mode(self) -> str:
        return self._filter_mode

    def set_filter_mode(self, mode: str) -> None:
        if mode not in self._filter_btns:
            mode = "all"
        self._filter_mode = mode
        btn = self._filter_btns[mode]
        if not btn.isChecked():
            btn.setChecked(True)

    def update_selection_hint(self, n: int):
        if n == 0:
            return
        elif n == 1:
            return
        else:
            return

    def apply_theme(self, t: dict):
        self._t = t


# ── Browse info panel (RIGHT) ─────────────────────────────────────────────────
class BrowseInfoPanel(QWidget):
    """Right-side info panel: selected file name, channel thumbnails, metadata."""

    def __init__(self, t: dict, cfg: dict, parent=None):
        super().__init__(parent)
        self._t         = t
        self._pool      = QThreadPool.globalInstance()
        self._ch_token  = object()
        self._meta_rows: list[tuple[str, str]] = []
        self._clip_low  = 1.0
        self._clip_high = 99.0
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        self._main_lay = lay
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(3)

        summary = QWidget()
        summary.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        summary_lay = QVBoxLayout(summary)
        summary_lay.setContentsMargins(0, 0, 0, 0)
        summary_lay.setSpacing(3)

        self.name_lbl = QLabel("No scan selected")
        self.name_lbl.setFont(QFont("Helvetica", 10, QFont.Bold))
        self.name_lbl.setWordWrap(True)
        self.name_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        summary_lay.addWidget(self.name_lbl)

        # Compact key scan summary. Keep this tight so channels sit high.
        qi_widget = QWidget()
        qi_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        qi_grid = QGridLayout()
        qi_grid.setHorizontalSpacing(8)
        qi_grid.setVerticalSpacing(2)
        qi_grid.setContentsMargins(0, 0, 0, 0)
        qi_widget.setLayout(qi_grid)
        self._qi: dict[str, QLabel] = {}
        _QI_ROWS = [("Pixels", "pixels"), ("Size", "size"),
                    ("Bias",   "bias"),   ("Setp.", "setp")]
        for i, (title, key) in enumerate(_QI_ROWS):
            r, c = divmod(i, 2)
            t_lbl = QLabel(title + ":")
            t_lbl.setFont(QFont("Helvetica", 8))
            v_lbl = QLabel("—")
            v_lbl.setFont(QFont("Helvetica", 10, QFont.Bold))
            qi_grid.addWidget(t_lbl, r, c * 2)
            qi_grid.addWidget(v_lbl, r, c * 2 + 1)
            self._qi[key] = v_lbl
        summary_lay.addWidget(qi_widget)
        summary_lay.addWidget(_sep())

        ch_hdr = QLabel("Channels")
        ch_hdr.setFont(QFont("Helvetica", 11, QFont.Bold))
        summary_lay.addWidget(ch_hdr)

        self._ch_widget = QWidget()
        self._ch_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self._ch_grid = QGridLayout()
        self._ch_grid.setSpacing(8)
        self._ch_grid.setContentsMargins(0, 0, 0, 0)
        self._ch_widget.setLayout(self._ch_grid)
        self._ch_cells: list[QWidget] = []
        self._ch_img_lbls:  list[QLabel] = []
        self._ch_name_lbls: list[QLabel] = []
        self._set_channel_preview_slots(PLANE_NAMES)
        summary_lay.addWidget(self._ch_widget)
        summary_lay.addWidget(_sep())

        # Full metadata is hidden behind a toggle. The quick-info grid above
        # (Pixels / Size / Bias / Setpoint) is what users want at a glance;
        # the full header table is dense and only useful occasionally.
        self._meta_toggle = QPushButton("[+] Show all metadata")
        self._meta_toggle.setFont(QFont("Helvetica", 9, QFont.Bold))
        self._meta_toggle.setFixedHeight(24)
        self._meta_toggle.setCursor(QCursor(Qt.PointingHandCursor))
        self._meta_toggle.setToolTip(
            "Expand to show the full scan header (also accessible via "
            "right-click → Show full metadata).")
        self._meta_toggle.clicked.connect(self._toggle_meta)
        summary_lay.addWidget(self._meta_toggle)
        lay.addWidget(summary)

        self._meta_widget = QWidget()
        meta_lay = QVBoxLayout(self._meta_widget)
        meta_lay.setContentsMargins(0, 4, 0, 0)
        meta_lay.setSpacing(4)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search…")
        self.search_box.setFont(QFont("Helvetica", 10))
        self.search_box.setFixedHeight(28)
        self.search_box.textChanged.connect(self._filter_meta)
        meta_lay.addWidget(self.search_box)

        self.meta_table = QTableWidget(0, 2)
        self.meta_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.meta_table.setWordWrap(True)
        self.meta_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.meta_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.meta_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.meta_table.setColumnWidth(0, 92)
        self.meta_table.verticalHeader().setVisible(False)
        self.meta_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.meta_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.meta_table.setAlternatingRowColors(True)
        self.meta_table.setFont(QFont("Helvetica", 10))
        self.meta_table.verticalHeader().setDefaultSectionSize(22)
        self.meta_table.setShowGrid(False)
        meta_lay.addWidget(self.meta_table, 1)
        self._meta_widget.setVisible(False)
        lay.addWidget(self._meta_widget, 0)
        self._bottom_spacer = QWidget()
        self._bottom_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._bottom_spacer, 1)

    def _toggle_meta(self):
        vis = not self._meta_widget.isVisible()
        self._meta_widget.setVisible(vis)
        self._meta_toggle.setText(
            "[-] Hide all metadata" if vis else "[+] Show all metadata")
        self._main_lay.setStretchFactor(self._meta_widget, 1 if vis else 0)
        self._main_lay.setStretchFactor(self._bottom_spacer, 0 if vis else 1)
        self._bottom_spacer.setVisible(not vis)
        if vis:
            self.meta_table.resizeRowsToContents()

    # ── Public API ─────────────────────────────────────────────────────────────
    def show_entry(self, entry: SxmFile, colormap_key: str,
                    processing: dict = None):
        self.name_lbl.setText(entry.stem)
        self._qi["pixels"].setText(f"{entry.Nx} × {entry.Ny}")
        self._qi["size"].setText(f"{entry.scan_nm:.1f} nm" if entry.scan_nm is not None else "—")
        self._qi["bias"].setText(f"{entry.bias_mv:.0f} mV" if entry.bias_mv is not None else "—")
        self._qi["setp"].setText(f"{entry.current_pa:.1f} pA" if entry.current_pa is not None else "—")
        self.load_channels(entry, colormap_key, processing=None)
        self._load_metadata(entry)

    def show_vert_entry(self, entry: VertFile):
        self.name_lbl.setText(entry.stem)
        sweep = entry.sweep_type.replace("_", " ") if entry.sweep_type != "unknown" else "—"
        self._qi["pixels"].setText(sweep)
        self._qi["size"].setText(f"{entry.n_points} pts" if entry.n_points else "—")
        self._qi["bias"].setText(f"{entry.bias_mv:.0f} mV" if entry.bias_mv is not None else "—")
        freq = entry.spec_freq_hz
        self._qi["setp"].setText(f"{freq:.0f} Hz" if freq is not None else "—")
        for lbl in self._ch_img_lbls:
            lbl.clear()
            lbl.setText("—")
        self._load_vert_metadata(entry)

    def _load_vert_metadata(self, entry: VertFile):
        from probeflow.io.spectroscopy import parse_spec_header
        try:
            hdr = parse_spec_header(entry.path)
        except Exception:
            hdr = {}
        rows: list[tuple[str, str]] = [
            ("Sweep type", entry.sweep_type.replace("_", " ")),
            ("Points", str(entry.n_points)),
        ]
        if entry.bias_mv is not None:
            rows.append(("Bias", f"{entry.bias_mv:.1f} mV"))
        if entry.spec_freq_hz is not None:
            rows.append(("Freq", f"{entry.spec_freq_hz:.0f} Hz"))
        seen = {"sweep_type", "n_points", "bias_mv", "spec_freq_hz"}
        for k, v in hdr.items():
            if k not in seen and v.strip():
                rows.append((k, v.strip()))
        self._meta_rows = rows
        self._filter_meta()

    def clear(self):
        self.name_lbl.setText("No scan selected")
        for v in self._qi.values():
            v.setText("—")
        for lbl in self._ch_img_lbls:
            lbl.clear()
        self._meta_rows = []
        self.meta_table.setRowCount(0)

    def apply_theme(self, t: dict):
        self._t = t
        self._filter_meta()

    # ── Public ─────────────────────────────────────────────────────────────────
    def load_channels(self, entry: SxmFile, colormap_key: str,
                       processing: dict = None):
        self._ch_token = object()
        sigs = ChannelSignals()
        sigs.loaded.connect(self._on_ch_loaded)
        self._ch_sigs = sigs
        planes = []
        try:
            scan = load_scan(entry.path)
            plane_names = list(scan.plane_names)
            n_planes = scan.n_planes
            planes = list(getattr(scan, "planes", []) or [])
        except Exception:
            plane_names = list(PLANE_NAMES)
            n_planes = len(plane_names)
        self._set_channel_preview_slots(plane_names)
        for i in range(n_planes):
            arr = planes[i] if i < len(planes) else None
            loader = ChannelLoader(entry, i, colormap_key,
                                   self._ch_token, 124, 98, sigs,
                                   self._clip_low, self._clip_high,
                                   processing=processing,
                                   arr=arr)
            self._pool.start(loader)

    # Back-compat alias used internally
    _load_channels = load_channels

    @Slot(int, QPixmap, object)
    def _on_ch_loaded(self, idx: int, pixmap: QPixmap, token):
        if token is not self._ch_token:
            return
        if idx >= len(self._ch_img_lbls):
            return
        lbl = self._ch_img_lbls[idx]
        lbl.setPixmap(pixmap.scaled(lbl.width(), lbl.height(),
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _set_channel_preview_slots(self, names: list[str]) -> None:
        names = names or PLANE_NAMES
        if [lbl.text() for lbl in self._ch_name_lbls] == names:
            for lbl in self._ch_img_lbls:
                lbl.clear()
                lbl.setText("—")
            return
        for cell in self._ch_cells:
            self._ch_grid.removeWidget(cell)
            cell.deleteLater()
        self._ch_cells.clear()
        self._ch_img_lbls.clear()
        self._ch_name_lbls.clear()
        for i, name in enumerate(names):
            r, c = divmod(i, 2)
            cell = QWidget()
            cell_lay = QVBoxLayout(cell)
            cell_lay.setContentsMargins(0, 0, 0, 0)
            cell_lay.setSpacing(2)
            img_lbl = QLabel()
            img_lbl.setFixedSize(128, 102)
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setFrameShape(QFrame.StyledPanel)
            img_lbl.setText("—")
            nm_lbl = QLabel(name)
            nm_lbl.setFont(QFont("Helvetica", 9))
            nm_lbl.setAlignment(Qt.AlignCenter)
            nm_lbl.setWordWrap(True)
            cell_lay.addWidget(img_lbl)
            cell_lay.addWidget(nm_lbl)
            self._ch_grid.addWidget(cell, r, c)
            self._ch_cells.append(cell)
            self._ch_img_lbls.append(img_lbl)
            self._ch_name_lbls.append(nm_lbl)

    def _load_metadata(self, entry: SxmFile):
        try:
            hdr = load_scan(entry.path).header
        except Exception:
            hdr = {}
        priority = [
            "REC_DATE", "REC_TIME", "SCAN_PIXELS", "SCAN_RANGE",
            "SCAN_OFFSET", "SCAN_ANGLE", "SCAN_DIR", "BIAS",
            "REC_TEMP", "ACQ_TIME", "SCAN_TIME", "COMMENT",
        ]
        rows: list[tuple[str, str]] = []
        seen: set[str]              = set()
        for k in priority:
            v = hdr.get(k)
            if isinstance(v, str) and v.strip():
                rows.append((k, v.strip()))
                seen.add(k)
        for k, v in hdr.items():
            if k in seen:
                continue
            if isinstance(v, str) and v.strip():
                rows.append((k, v.strip()))
            elif v is not None and not isinstance(v, (bytes, bytearray)):
                s = str(v).strip()
                if s:
                    rows.append((k, s))
        self._meta_rows = rows
        self._filter_meta()

    def _filter_meta(self):
        query = self.search_box.text().lower()
        self.meta_table.setRowCount(0)
        t = self._t
        for param, value in self._meta_rows:
            if not query or query in param.lower() or query in value.lower():
                row    = self.meta_table.rowCount()
                self.meta_table.insertRow(row)
                p_item = QTableWidgetItem(param)
                p_item.setForeground(QColor(t["accent_bg"]))
                v_item = QTableWidgetItem(value)
                v_item.setForeground(QColor(t["fg"]))
                self.meta_table.setItem(row, 0, p_item)
                self.meta_table.setItem(row, 1, v_item)
        self.meta_table.resizeRowsToContents()


# ── Features tab integration ────────────────────────────────────────────────
# Specialized add-on workflows live in probeflow.gui.features. Keep this main
# GUI file focused on Browse/Viewer/Convert orchestration; Features owns tools
# like particle counting, template counting, lattice extraction, and future
# TV-denoise/background-removal panels so optional analysis dependencies do not
# leak into routine browsing or image manipulation.


# ── Spec viewer dialog ───────────────────────────────────────────────────────
# ── Convert panel ─────────────────────────────────────────────────────────────
class ConvertPanel(QWidget):
    def __init__(self, t: dict, cfg: dict, parent=None):
        super().__init__(parent)
        self._t = t
        lay     = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 12)
        lay.setSpacing(10)

        # Input folder (always visible)
        in_row = QHBoxLayout()
        in_lbl = QLabel("Input folder:")
        in_lbl.setFixedWidth(110)
        in_lbl.setFont(QFont("Helvetica", 11))
        self.input_entry = QLineEdit()
        self.input_entry.setFont(QFont("Helvetica", 11))
        self.input_entry.setPlaceholderText("Select folder with .dat files…")
        in_btn = QPushButton("Browse")
        in_btn.setFont(QFont("Helvetica", 10))
        in_btn.setFixedWidth(80)
        in_btn.clicked.connect(self._browse_input)
        in_row.addWidget(in_lbl)
        in_row.addWidget(self.input_entry)
        in_row.addWidget(in_btn)
        lay.addLayout(in_row)

        # Custom output checkbox + row (hidden by default)
        self._custom_out_cb = QCheckBox("Custom output folder")
        self._custom_out_cb.setFont(QFont("Helvetica", 11))
        self._custom_out_cb.setChecked(cfg.get("custom_output", False))
        self._custom_out_cb.toggled.connect(self._toggle_output_row)
        lay.addWidget(self._custom_out_cb)

        self._out_row_widget = QWidget()
        out_row = QHBoxLayout(self._out_row_widget)
        out_row.setContentsMargins(0, 0, 0, 0)
        out_lbl = QLabel("Output folder:")
        out_lbl.setFixedWidth(110)
        out_lbl.setFont(QFont("Helvetica", 11))
        self.output_entry = QLineEdit()
        self.output_entry.setFont(QFont("Helvetica", 11))
        self.output_entry.setPlaceholderText("Defaults to input folder…")
        out_btn = QPushButton("Browse")
        out_btn.setFont(QFont("Helvetica", 10))
        out_btn.setFixedWidth(80)
        out_btn.clicked.connect(self._browse_output)
        out_row.addWidget(out_lbl)
        out_row.addWidget(self.output_entry)
        out_row.addWidget(out_btn)
        lay.addWidget(self._out_row_widget)
        self._out_row_widget.setVisible(cfg.get("custom_output", False))

        lay.addWidget(_sep())

        log_hdr = QHBoxLayout()
        log_lbl = QLabel("Conversion log")
        log_lbl.setFont(QFont("Helvetica", 11, QFont.Bold))
        clear_btn = QPushButton("Clear")
        clear_btn.setFont(QFont("Helvetica", 10))
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(lambda: self.log_text.clear())
        log_hdr.addWidget(log_lbl)
        log_hdr.addStretch()
        log_hdr.addWidget(clear_btn)
        lay.addLayout(log_hdr)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 11))
        lay.addWidget(self.log_text, 1)

        if cfg.get("input_dir"):
            self.input_entry.setText(cfg["input_dir"])
        if cfg.get("output_dir"):
            self.output_entry.setText(cfg["output_dir"])

    def _toggle_output_row(self, checked: bool):
        self._out_row_widget.setVisible(checked)

    def get_output_dir(self) -> str:
        """Returns custom output if checked, otherwise empty string (→ use input dir)."""
        if self._custom_out_cb.isChecked():
            return self.output_entry.text().strip()
        return ""

    def apply_theme(self, t: dict):
        self._t = t

    def log(self, msg: str, tag: str = "info"):
        color = self._t.get(f"{tag}_fg", self._t.get("info_fg", self._t["fg"]))
        self.log_text.append(f'<span style="color:{color}">{msg}</span>')
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum())

    def _browse_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select input folder with .dat files")
        if d:
            self.input_entry.setText(d)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.output_entry.setText(d)


# ── Convert sidebar ───────────────────────────────────────────────────────────
class ConvertSidebar(QWidget):
    def __init__(self, t: dict, cfg: dict, parent=None):
        super().__init__(parent)
        self._t = t
        lay     = QVBoxLayout(self)
        lay.setContentsMargins(12, 14, 12, 10)
        lay.setSpacing(8)

        hdr = QLabel("Output format")
        hdr.setFont(QFont("Helvetica", 12, QFont.Bold))
        lay.addWidget(hdr)

        self.png_cb = QCheckBox("PNG preview")
        self.sxm_cb = QCheckBox("SXM (Nanonis)")
        self.png_cb.setChecked(cfg.get("do_png", False))
        self.sxm_cb.setChecked(cfg.get("do_sxm", True))
        for cb in (self.png_cb, self.sxm_cb):
            cb.setFont(QFont("Helvetica", 11))
            lay.addWidget(cb)

        lay.addWidget(_sep())

        self._adv_btn = QPushButton("[+] Advanced")
        self._adv_btn.setFlat(True)
        self._adv_btn.setFont(QFont("Helvetica", 10))
        self._adv_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._adv_btn.clicked.connect(self._toggle_adv)
        lay.addWidget(self._adv_btn)

        self._adv_widget = QWidget()
        adv_lay = QVBoxLayout(self._adv_widget)
        adv_lay.setContentsMargins(0, 0, 0, 0)
        adv_lay.setSpacing(6)

        def _spin_row(label: str, val: float, mn: float, mx: float):
            row  = QHBoxLayout()
            lbl  = QLabel(label)
            lbl.setFont(QFont("Helvetica", 10))
            lbl.setFixedWidth(100)
            spin = QDoubleSpinBox()
            spin.setRange(mn, mx)
            spin.setValue(val)
            spin.setSingleStep(0.5)
            spin.setFont(QFont("Helvetica", 10))
            row.addWidget(lbl)
            row.addWidget(spin)
            adv_lay.addLayout(row)
            return spin

        self.clip_low_spin  = _spin_row("Clip low (%):",  cfg.get("clip_low",  1.0),  0.0, 10.0)
        self.clip_high_spin = _spin_row("Clip high (%):", cfg.get("clip_high", 99.0), 90.0, 100.0)
        self._adv_widget.setVisible(False)
        lay.addWidget(self._adv_widget)

        lay.addWidget(_sep())

        self.run_btn = QPushButton("  RUN  ")
        self.run_btn.setFont(QFont("Helvetica", 14, QFont.Bold))
        self.run_btn.setFixedHeight(48)
        self.run_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.run_btn.setObjectName("accentBtn")
        lay.addWidget(self.run_btn)

        lay.addWidget(_sep())

        self.fcount_lbl = QLabel("")
        self.fcount_lbl.setFont(QFont("Helvetica", 10))
        self.fcount_lbl.setWordWrap(True)
        lay.addWidget(self.fcount_lbl)

        lay.addStretch()

        credit = QLabel(
            "SPMQT-Lab  |  Dr. Peter Jacobson\n"
            "The University of Queensland\n"
            "Original code by Rohan Platts"
        )
        credit.setFont(QFont("Helvetica", 9))
        credit.setAlignment(Qt.AlignCenter)
        lay.addWidget(credit)

    def _toggle_adv(self):
        vis = not self._adv_widget.isVisible()
        self._adv_widget.setVisible(vis)
        self._adv_btn.setText("[-] Advanced" if vis else "[+] Advanced")

    def update_file_count(self, n: int):
        self.fcount_lbl.setText(f"{n} .dat file(s) in input folder" if n >= 0 else "")


# ── About dialog ──────────────────────────────────────────────────────────────
# ── Navbar ────────────────────────────────────────────────────────────────────
class Navbar(QWidget):
    theme_toggle_clicked = Signal()
    font_size_changed    = Signal(str)
    about_clicked        = Signal()

    def __init__(self, dark: bool, font_size_label: str = GUI_FONT_DEFAULT, parent=None):
        super().__init__(parent)
        self._dark            = dark
        self._font_size_label = normalise_gui_font_size(font_size_label)
        self._btns:           list[QPushButton] = []
        self.setFixedHeight(50)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(6)

        if LOGO_NAV_PATH.exists():
            self._logo_lbl = QLabel()
            self._logo_lbl.setStyleSheet("background: transparent;")
            pix = QPixmap(str(LOGO_NAV_PATH))
            self._logo_lbl.setPixmap(
                pix.scaled(9999, 46, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self._logo_lbl.setCursor(QCursor(Qt.PointingHandCursor))
            self._logo_lbl.mousePressEvent = lambda e: _open_url(GITHUB_URL)
            lay.addWidget(self._logo_lbl)

        title_lbl = QLabel("ProbeFlow")
        title_lbl.setFont(QFont("Helvetica", 12, QFont.Bold))
        title_lbl.setStyleSheet("background: transparent;")
        lay.addWidget(title_lbl)
        lay.addStretch()
        self._font_size_actions: dict[str, QAction] = {}

        self._apply_nav_theme()

    def set_dark(self, dark: bool):
        self._dark = dark
        self._apply_nav_theme()

    def set_font_size(self, label: str):
        label = normalise_gui_font_size(label)
        if label == self._font_size_label:
            self._sync_font_size_button()
            return
        self._font_size_label = label
        self._sync_font_size_button()
        self.font_size_changed.emit(label)

    def _sync_font_size_button(self):
        for label, action in self._font_size_actions.items():
            action.setChecked(label == self._font_size_label)

    def _apply_nav_theme(self):
        if self._dark:
            self.setStyleSheet(
                f"background-color: {NAVBAR_DARK_BG};"
            )
            btn_qss = """
                QPushButton {
                    color: #ffffff;
                    background-color: transparent;
                    border: 2px solid rgba(255,255,255,0.6);
                    border-radius: 4px;
                    padding: 4px 14px;
                }
                QPushButton:hover {
                    background-color: rgba(255,255,255,0.18);
                }
            """
        else:
            self.setStyleSheet(
                f"background-color: {NAVBAR_LIGHT_BG};"
                "border-bottom: 2px solid #b0bec5;"
            )
            btn_qss = """
                QPushButton {
                    color: #1e1e2e;
                    background-color: #f0f2f5;
                    border: 2px solid #b0bec5;
                    border-radius: 4px;
                    padding: 4px 14px;
                }
                QPushButton:hover {
                    background-color: #e4edf8;
                    border-color: #3273dc;
                }
            """
        if hasattr(self, "_logo_lbl"):
            self._logo_lbl.setStyleSheet("background: transparent;")
        for btn in self._btns:
            btn.setStyleSheet(btn_qss)


# ── Processing definitions panel ──────────────────────────────────────────────
_DEFINITIONS_HTML = """
<style>
  body  { font-family: Helvetica, Arial, sans-serif; font-size: 13px;
          color: #cdd6f4; background: transparent; margin: 0; padding: 0; }
  h1    { font-size: 15px; color: #cba6f7; margin: 0 0 12px 0; }
  h2    { font-size: 13px; color: #89b4fa; margin: 18px 0 2px 0; }
  .sub  { font-size: 11px; color: #a6adc8; font-style: italic; margin: 0 0 4px 0; }
  p     { margin: 3px 0 6px 0; line-height: 1.45; }
  .kw   { color: #f38ba8; font-family: monospace; }
  .param{ color: #a6e3a1; font-family: monospace; }
  .note { color: #fab387; }
  hr    { border: none; border-top: 1px solid #45475a; margin: 14px 0; }
</style>
<body>
<h1>Processing Algorithm Reference</h1>
<p>Each step transforms the raw height data. Steps are applied in the order
listed in the viewer's processing panel. All functions operate on
float64 arrays in physical metres — no display-unit clipping involved.</p>
<hr/>

<h2>remove_bad_lines</h2>
<p class="sub">Params: <span class="param">method</span> = step | mad,
<span class="param">polarity</span> = bright | dark,
<span class="param">threshold_mad</span>,
<span class="param">min_segment_length_px</span>,
<span class="param">max_adjacent_bad_lines</span></p>
<p><b>step:</b> Compares each fast-scan row with neighbouring rows, then finds
paired positive/negative jumps along the row.  Only the segment between the
jumps is corrected.</p>
<p><b>mad:</b> Compares each fast-scan row with neighbouring rows and flags
contiguous outlier segments in the row residual.  This is more direct for
plateau-like partial defects.</p>
<p>Both methods repair a detected segment from local neighbouring scan lines;
pixels outside detected segments remain unchanged.  Preview detection in the
viewer is non-destructive.</p>

<h2>Bad scan-line segment</h2>
<p>A short damaged part of a fast-scan line. ProbeFlow corrects the segment
only, not the whole row or column.</p>
<h2>Threshold</h2>
<p>Detection sensitivity. A higher value detects fewer, more obvious artifacts.
A lower value detects more candidate artifacts. It is not a pixel length.</p>
<h2>Minimum segment length (px)</h2>
<p>The shortest run of neighbouring pixels along the fast-scan direction that
can be treated as a bad segment.</p>
<h2>Maximum adjacent bad lines</h2>
<p>The largest number of neighbouring scan lines that ProbeFlow will attempt to
repair as a local bad-line artifact. Broader damaged regions are skipped
because local interpolation becomes unreliable.</p>
<h2>Bright bad segment</h2>
<p>A segment that is higher or brighter than nearby scan lines.</p>
<h2>Dark bad segment</h2>
<p>A segment that is lower or darker than nearby scan lines.</p>
<h2>Preview detection</h2>
<p>Shows candidate bad segments without modifying the image.</p>
<h2>Apply correction</h2>
<p>Repairs the currently detected and accepted bad segments; skipped unsafe
groups remain unchanged.</p>

<hr/>

<h2>align_rows</h2>
<p class="sub">Params: <span class="param">method</span> = median | mean | linear</p>
<p>Removes per-row DC offsets — the most common first step for raw STM data,
where each scan line has an independent height datum due to thermal drift or
tip jumps between lines.</p>
<p><b>median:</b> Subtracts each row's median.  Robust to tip crashes and
outlier pixels within a row.  <b>mean:</b> Subtracts each row's mean — faster
but sensitive to outliers.  <b>linear:</b> Fits and subtracts a first-order
polynomial (slope + offset) per row, correcting both offset and tilt within
each scan line.</p>

<hr/>

<h2>plane_bg (subtract_background)</h2>
<p class="sub">Params: <span class="param">order</span> 1–4,
<span class="param">step_tolerance</span>, optional <span class="param">fit_geometry</span></p>
<p>Fits a 2-D polynomial background to the image and subtracts it.  The
polynomial basis contains all monomials x<sup>i</sup>&nbsp;y<sup>j</sup> with
i+j &le; order (6 terms for order=2, 10 for order=3).  Coordinates are
normalised to [&minus;1,&nbsp;1] for numerical stability.  The fit uses only
finite pixels and, optionally, only pixels inside a user-drawn ROI
(<span class="param">fit_geometry</span>).</p>
<p><b>order=1</b> (plane): equivalent to ImageJ's "Subtract Plane" /
<em>Fit_Polynomial("linear")</em>.  <b>step_tolerance:</b> excludes steep
pixels (gradient &gt; tan(3&deg;)) from the fit so that atomic steps do not
bias the background — analogous to the "step tolerant" area mode in the ImageJ
STM_Background plugin.</p>

<hr/>

<h2>stm_line_bg (stm_line_background)</h2>
<p class="sub">Params: <span class="param">mode</span> = step_tolerant</p>
<p>Corrects inter-line height offsets in images with atomic steps.  For each
pair of adjacent scan lines, estimates the dominant row-to-row shift from the
<em>modal peak</em> of the distribution of pixelwise row differences.  The
modal peak tracks the most common shift (the flat terrace baseline) rather than
the mean, so step edges — which produce large, infrequent differences — do not
bias the correction.  A cumulative shift profile is built and subtracted.</p>
<p class="note">Addresses the same artefact as the ImageJ STM_Background
"line by line" mode, but uses the modal estimator instead of the median for
better step tolerance.  Does not model slow background curvature; combine with
plane_bg for that.</p>

<hr/>

<h2>STM Background</h2>
<p class="sub">Params: <span class="param">fit_region</span>,
<span class="param">line_statistic</span>, <span class="param">model</span>,
<span class="param">linear_x_first</span>, <span class="param">blur_length</span>,
<span class="param">jump_threshold</span></p>
<p>Estimates one background value per fast-scan line, fits or smooths that
scan-line profile, then subtracts the fitted background from the whole image.
This is distinct from ROI-scoped processing: the fit region determines where
the background is estimated, but subtraction is applied to the full image.</p>
<h2>Scan-line profile</h2>
<p>The one-dimensional background estimate, one value per image row.</p>
<h2>Line statistic</h2>
<p>The row value used for the scan-line profile. Median is robust against
adsorbates, pits, and spikes; mean follows all selected pixels.</p>
<h2>Linear fit in x first</h2>
<p>Optionally fits and removes a straight x-direction slope from each scan line
before estimating the y-direction background profile.</p>
<h2>Piezo creep</h2>
<p>A future nonlinear background model for slow scanner relaxation, based on a
logarithmic creep-like curve. It is not exposed in the first ProbeFlow STM
Background dialog until robust constrained fitting is available.</p>
<h2>Piezo creep + x^2 / x^3</h2>
<p>Piezo creep models with additional polynomial terms. These are useful for
more complex slow-scan drift, but should be previewed carefully once enabled.</p>
<h2>Sqrt creep</h2>
<p>A future nonlinear background model based on a square-root creep-like curve.</p>
<h2>Low-pass background</h2>
<p>Smooths the scan-line profile using the blur length. Larger blur lengths
produce a slower, smoother background.</p>
<h2>Line-by-line background</h2>
<p>Uses the raw scan-line profile directly as the background. This is
aggressive and should be previewed carefully.</p>
<h2>Fit region</h2>
<p>Whole image uses all finite pixels. Active ROI uses only the selected area
ROI to estimate the profile, then applies the subtraction to the full image.</p>
<h2>Preview background</h2>
<p>Shows the fitted background image without modifying the data.</p>
<h2>Preview corrected image</h2>
<p>Shows the proposed corrected image before applying the processing step.</p>

<hr/>

<h2>facet_level</h2>
<p class="sub">Params: <span class="param">threshold_deg</span> (default 3.0)</p>
<p>Levels the image using only the nearly-flat pixels as the reference plane.
Local surface slope is estimated via central finite differences; pixels whose
slope angle exceeds <span class="param">threshold_deg</span> are excluded from
the plane fit.  The fitted plane is subtracted from the whole image.  Analogous
to Gwyddion's "Facet Level" — essential for stepped surfaces (Au(111), Si(111))
where step edges would bias a naive plane fit.</p>

<hr/>

<h2>smooth (gaussian_smooth)</h2>
<p class="sub">Params: <span class="param">sigma_px</span> (default 1.0)</p>
<p>Isotropic 2-D Gaussian blur.  NaN pixels are handled by weighted
normalisation (a NaN never propagates into its neighbours).  Typical STM
values: 0.5–2&nbsp;px.  Equivalent to ImageJ's Gaussian blur on float data.</p>

<hr/>

<h2>gaussian_high_pass</h2>
<p class="sub">Params: <span class="param">sigma_px</span> (default 8.0)</p>
<p>Subtracts a Gaussian-blurred version of the image from itself, retaining
only high-spatial-frequency detail.  Output = original &minus; blur(original).
Equivalent to the ImageJ "Highpass" plugin (M.&nbsp;Schmid): the ImageJ version
adds an integer offset for byte/short images; for float data the offset is zero,
so the algorithms are identical.</p>

<hr/>

<h2>fft_soft_border</h2>
<p class="sub">Params: <span class="param">mode</span> low_pass|high_pass,
<span class="param">cutoff</span> [0–1], <span class="param">border_frac</span></p>
<p>FFT-based frequency filter with a Tukey-tapered border.  Before
transforming, pixels within <span class="param">border_frac</span> of any edge
are smoothly ramped to the image mean, eliminating the wrap-around
discontinuity that causes ringing artefacts in DFT-based filters.  After the
inverse FFT, the taper is compensated so the image interior is preserved.
The <span class="kw">low_pass</span> mode keeps frequencies inside a radial
cutoff (fraction of Nyquist); <span class="kw">high_pass</span> keeps
frequencies outside.</p>
<p class="note">The name is inherited from the ImageJ FFT_Soft_Border plugin
(M.&nbsp;Schmid), but the operations differ: the ImageJ plugin computed only
the forward FFT spectrum for display.  This implementation is a complete
forward+filter+inverse pipeline returning a filtered image.</p>

<hr/>

<h2>fourier_filter</h2>
<p class="sub">Params: <span class="param">mode</span> low_pass|high_pass,
<span class="param">cutoff</span>, <span class="param">window</span> hanning|hamming|none</p>
<p>Global radial FFT filter without border compensation.  A 2-D Hanning (or
Hamming) window is applied before transforming to reduce edge discontinuities.
The frequency-domain filter is a hard circular cutoff.  The mean is preserved
for low-pass; removed for high-pass.</p>

<hr/>

<h2>periodic_notch_filter</h2>
<p class="sub">Params: <span class="param">peaks</span> list of (dx,dy),
<span class="param">radius_px</span></p>
<p>Suppresses selected periodic FFT peaks and their Hermitian conjugates using
Gaussian notches.  Peaks are specified as integer pixel offsets from the
centred FFT origin.  Used to remove lattice periodicity from topography so that
defects and adsorbates stand out.</p>
<p class="note">Complementary to (not a port of) the ImageJ Periodic_Filter
(M.&nbsp;Schmid), which <em>extracts</em> the periodic component by convolution
with a lattice-frequency kernel.  Notch removal is the standard workflow for
defect imaging; the ImageJ bandpass extraction is useful for lattice
characterisation.</p>

<hr/>

<h2>patch_interpolate</h2>
<p class="sub">Params: <span class="param">method</span> line_fit|laplace,
<span class="param">rim_px</span> (default 20),
<span class="param">iterations</span> (laplace only, default 200)</p>
<p><b>line_fit (default):</b> For each masked row, fits a linear function to
the non-masked pixels within <span class="param">rim_px</span> columns of the
masked boundary (the "rim"), then extrapolates that line through the masked
columns.  This preserves the local surface slope — physically correct for STM
scan-line repair where nearby scan lines share the same tilt.  Based on the
ImageJ Patch_Interpolation plugin (M.&nbsp;Schmid), mode "lines with individual
slopes".  Rows with no rim data fall back to row-blended interpolation from
vertical neighbours.</p>
<p><b>laplace:</b> Iterative Jacobi relaxation of the discrete Laplace
equation: each masked pixel converges to the average of its four neighbours.
Isotropic and smooth, but does not preserve scan-line slope — creates
artificial height bumps on sloped terraces.  Useful for non-directional patches
(e.g., circular defect sites away from step edges).</p>

<hr/>

<h2>linear_undistort</h2>
<p class="sub">Params: <span class="param">shear_x</span> (px),
<span class="param">scale_y</span></p>
<p>Affine drift/creep correction.  <span class="param">shear_x</span> is the
total horizontal pixel drift accumulated over the slow-scan height (positive =
right drift).  <span class="param">scale_y</span> corrects a y/x pixel-size
mismatch.  Each output pixel is bilinearly interpolated from the input.
Equivalent to the ImageJ Linear_Undistort plugin (M.&nbsp;Schmid);
parameterisation differs: ImageJ uses shear angle in degrees and a y/x ratio,
with the conversion: shear_x&nbsp;=&nbsp;tan(angle)&nbsp;&times;&nbsp;scale_y&nbsp;&times;&nbsp;(Ny&minus;1).</p>

<hr/>

<h2>blend_forward_backward</h2>
<p class="sub">Params: <span class="param">weight</span> (default 0.5)</p>
<p>Blends a forward scan plane with a left-right-mirrored backward scan plane.
The backward scan is automatically flipped before blending because it is
recorded right-to-left in the fast-scan direction.  This differs from the
ImageJ Blend_Images plugin (M.&nbsp;Schmid), which does a generic weighted
sum without flipping — the flip is required for correct physical alignment of
STM forward/backward pairs.</p>

<hr/>

<h2>edge_detect</h2>
<p class="sub">Params: <span class="param">method</span> laplacian|log|dog,
<span class="param">sigma</span>, <span class="param">sigma2</span></p>
<p><b>laplacian:</b> Discrete second-derivative operator — enhances sharp
edges and atomic corrugation peaks.  <b>log (Laplacian of Gaussian):</b>
Pre-smooths with a Gaussian of width <span class="param">sigma</span> before
applying the Laplacian; reduces noise sensitivity.  <b>dog (Difference of
Gaussians):</b> Difference between two Gaussians of width
<span class="param">sigma</span> and <span class="param">sigma2</span> — a
band-pass approximation to the LoG, useful for isolating features of a
specific spatial scale.</p>
</body>
"""


class _DefinitionsPanel(QWidget):
    """Scrollable reference panel listing every processing algorithm."""

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        inner = QTextEdit()
        inner.setReadOnly(True)
        inner.setHtml(_DEFINITIONS_HTML)
        inner.setStyleSheet(f"""
            QTextEdit {{
                background-color: {t.get('bg', '#1e1e2e')};
                border: none;
                padding: 16px;
            }}
        """)
        inner.document().setDefaultStyleSheet(
            f"body {{ color: {t.get('fg', '#cdd6f4')}; }}"
        )

        scroll.setWidget(inner)
        lay.addWidget(scroll)


class _DefinitionsDialog(QDialog):
    """Closeable utility window for processing definitions/help."""

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ProbeFlow Definitions")
        self.resize(760, 640)
        self.setModal(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._panel = _DefinitionsPanel(t, self)
        lay.addWidget(self._panel)


# ── Developer terminal sidebar ────────────────────────────────────────────────
class _DevSidebar(QWidget):
    """Sidebar for the Dev tab: shows cwd, quick links, and info."""

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        title = QLabel("Developer Mode")
        title.setFont(QFont("Helvetica", 11, QFont.Bold))
        lay.addWidget(title)

        info = QLabel(
            "Run shell commands and Python scripts in the ProbeFlow environment.\n\n"
            "↑/↓ — command history   Ctrl+C — interrupt\n\n"
            "All ProbeFlow packages available: import probeflow, numpy, scipy…\n\n"
            "⚠  Interactive tools (claude, ipython, vim) need a real PTY.\n"
            "Use 'Open External Terminal' to launch WSL / Windows Terminal."
        )
        info.setFont(QFont("Helvetica", 9))
        info.setWordWrap(True)
        lay.addWidget(info)

        example_lbl = QLabel("Quick examples:")
        example_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        lay.addWidget(example_lbl)

        examples = QLabel(
            "python3 -c \"import probeflow; print(probeflow.__file__)\"\n\n"
            "python3 scripts/my_analysis.py\n\n"
            "ls -lh *.dat\n\n"
            "python3 -c \"from probeflow.io.readers.createc_scan import read_dat; "
            "import numpy as np; s = read_dat('scan.dat'); "
            "print(np.nanmin(s.planes[0])*1e10, 'A')\""
        )
        examples.setFont(QFont("Courier New" if sys.platform == "win32" else "Monospace", 8))
        examples.setWordWrap(True)
        examples.setStyleSheet("color: #888;")
        lay.addWidget(examples)

        lay.addStretch()


# ── Developer terminal ────────────────────────────────────────────────────────
# ── Main window ───────────────────────────────────────────────────────────────
class ProbeFlowWindow(QMainWindow):
    LEFT_SIDEBAR_DEFAULT_W = 280
    LEFT_SIDEBAR_MIN_W = 240
    RIGHT_INSPECTOR_DEFAULT_W = 340
    RIGHT_INSPECTOR_MIN_W = 300
    CENTRAL_BROWSER_MIN_W = 500

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProbeFlow")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 800)

        self._cfg      = load_config()
        self._dark     = self._cfg.get("dark_mode", True)
        self._gui_font_size = normalise_gui_font_size(self._cfg.get("gui_font_size"))
        self._mode     = "browse"
        self._running  = False
        self._n_loaded = 0
        # Spec → image mapping (populated by user via "Map spectra…" dialogs;
        # kept empty by default so we never auto-attach spectra to the wrong
        # image based on coordinate guesses alone). Keys are spec stems,
        # values are image stems within the currently loaded folder.
        self._spec_image_map: dict[str, str] = {}

        self._build_ui()
        self._apply_theme()

    # ── Build ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        v_lay = QVBoxLayout(central)
        v_lay.setContentsMargins(0, 0, 0, 0)
        v_lay.setSpacing(0)

        self._navbar = Navbar(self._dark, self._gui_font_size)
        self._navbar.theme_toggle_clicked.connect(self._toggle_theme)
        self._navbar.font_size_changed.connect(self._on_gui_font_size_changed)
        self._navbar.about_clicked.connect(self._show_about)
        v_lay.addWidget(self._navbar)

        # Body splitter
        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setHandleWidth(5)
        v_lay.addWidget(self._splitter, 1)

        t = THEMES["dark" if self._dark else "light"]

        # ── Content stack (center+left area) ──────────────────────────────────
        self._content_stack = QStackedWidget()

        # Browse mode: inner splitter [BrowseToolPanel | ThumbnailGrid]
        self._browse_tools = BrowseToolPanel(t, self._cfg)
        self._browse_tools.setMinimumWidth(self.LEFT_SIDEBAR_MIN_W)
        self._browse_tools.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._grid         = ThumbnailGrid(t)
        self._grid.setMinimumWidth(self.CENTRAL_BROWSER_MIN_W)
        self._browse_splitter = QSplitter(Qt.Horizontal)
        self._browse_splitter.setHandleWidth(3)
        self._browse_splitter.setChildrenCollapsible(False)
        self._browse_splitter.addWidget(self._browse_tools)
        self._browse_splitter.addWidget(self._grid)
        self._browse_splitter.setStretchFactor(0, 0)
        self._browse_splitter.setStretchFactor(1, 1)

        self._conv_panel    = ConvertPanel(t, self._cfg)
        self._features_panel = FeaturesPanel(t)
        self._tv_panel       = TVPanel(t)
        self._dev_terminal   = DeveloperTerminalWidget(t)
        self._content_stack.addWidget(self._browse_splitter)
        self._content_stack.addWidget(self._conv_panel)
        self._content_stack.addWidget(self._features_panel)
        self._content_stack.addWidget(self._tv_panel)
        self._content_stack.addWidget(self._dev_terminal)
        self._splitter.addWidget(self._content_stack)

        # ── Right: sidebar stack ───────────────────────────────────────────────
        self._sidebar_stack    = QStackedWidget()
        self._sidebar_stack.setMinimumWidth(self.RIGHT_INSPECTOR_MIN_W)
        self._sidebar_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._browse_info      = BrowseInfoPanel(t, self._cfg)
        self._convert_sidebar  = ConvertSidebar(t, self._cfg)
        self._features_sidebar = FeaturesSidebar(t)
        self._tv_sidebar       = TVSidebar(t)
        self._dev_sidebar      = _DevSidebar(t)
        self._sidebar_stack.addWidget(self._browse_info)
        self._sidebar_stack.addWidget(self._convert_sidebar)
        self._sidebar_stack.addWidget(self._features_sidebar)
        self._sidebar_stack.addWidget(self._tv_sidebar)
        self._sidebar_stack.addWidget(self._dev_sidebar)
        self._splitter.addWidget(self._sidebar_stack)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        self._apply_default_splitter_sizes()

        # Features tab plumbing
        self._features_pool    = QThreadPool.globalInstance()
        self._features_signals = _FeaturesWorkerSignals()
        self._features_signals.finished.connect(self._on_features_finished)

        # TV-denoise tab plumbing
        self._tv_pool    = QThreadPool.globalInstance()
        self._tv_signals = _TVWorkerSignals()
        self._tv_signals.finished.connect(self._on_tv_finished)
        self._tv_sidebar.load_from_browse_requested.connect(
            self._on_tv_load_from_browse)
        self._tv_sidebar.run_requested.connect(self._on_tv_run)
        self._tv_sidebar.revert_requested.connect(self._on_tv_revert)
        self._tv_sidebar.save_png_requested.connect(self._on_tv_save_png)

        # Wire signals
        self._browse_tools.open_folder_requested.connect(self._open_browse_folder)
        self._grid.entry_selected.connect(self._on_entry_select)
        self._grid.selection_changed.connect(self._on_selection_changed)
        self._grid.view_requested.connect(self._open_viewer)
        self._grid.card_context_action.connect(self._on_card_context_action)
        self._grid.folder_changed.connect(self._on_grid_folder_changed)
        self._browse_tools.colormap_changed.connect(self._on_thumbnail_colormap_changed)
        self._browse_tools.thumbnail_align_changed.connect(self._on_thumbnail_align_changed)
        self._browse_tools.map_spectra_requested.connect(self._on_map_spectra)
        self._browse_tools.filter_changed.connect(self._on_filter_changed)
        self._browse_tools.thumbnail_channel_changed.connect(self._on_thumbnail_channel_changed)
        # Sync initial filter state from the toolbar into the grid so the
        # two agree even before the first folder is opened.
        self._grid.apply_filter(self._browse_tools.get_filter_mode())
        self._convert_sidebar.run_btn.clicked.connect(self._run)
        self._conv_panel.input_entry.textChanged.connect(self._update_count)

        self._features_sidebar.load_from_browse_requested.connect(
            self._on_features_load_from_browse)
        self._features_sidebar.run_requested.connect(self._on_features_run)
        self._features_sidebar.export_requested.connect(self._on_features_export)
        self._features_sidebar.crop_template_requested.connect(
            self._features_panel.begin_template_crop)

        # Status bar
        self._status_bar = QStatusBar()
        self._status_bar.setFont(QFont("Helvetica", 10))
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Open a folder to browse scans")

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        self._mode_actions: dict[str, QAction] = {}
        self._font_size_actions: dict[str, QAction] = {}
        self._theme_actions: dict[str, QAction] = {}
        self._thumbnail_cmap_actions: dict[str, QAction] = {}
        self._thumbnail_channel_actions: dict[str, QAction] = {}
        self._thumbnail_align_actions: dict[str, QAction] = {}

        self._mode_action_group = QActionGroup(self)
        self._mode_action_group.setExclusive(True)
        self._font_size_action_group = QActionGroup(self)
        self._font_size_action_group.setExclusive(True)
        self._theme_action_group = QActionGroup(self)
        self._theme_action_group.setExclusive(True)
        self._thumbnail_cmap_action_group = QActionGroup(self)
        self._thumbnail_cmap_action_group.setExclusive(True)
        self._thumbnail_channel_action_group = QActionGroup(self)
        self._thumbnail_channel_action_group.setExclusive(True)
        self._thumbnail_align_action_group = QActionGroup(self)
        self._thumbnail_align_action_group.setExclusive(True)

        def _mode_action(menu, text: str, mode: str, shortcut: str | None = None):
            action = QAction(text, self)
            action.setCheckable(True)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(
                lambda _checked=False, value=mode: self._switch_mode(value)
            )
            self._mode_action_group.addAction(action)
            self._mode_actions[mode] = action
            menu.addAction(action)
            return action

        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open folder...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._menu_open_folder)
        file_menu.addAction(open_action)
        recent_action = QAction("Open recent", self)
        recent_action.setEnabled(False)
        file_menu.addAction(recent_action)
        file_menu.addSeparator()
        export_image_action = QAction("Export image...", self)
        export_image_action.setEnabled(False)
        file_menu.addAction(export_image_action)
        export_processed_action = QAction("Export processed image...", self)
        export_processed_action.setEnabled(False)
        file_menu.addAction(export_processed_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = menu_bar.addMenu("View")
        _mode_action(view_menu, "Browse", "browse", "Ctrl+1")
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("Theme")
        for label, dark in (("Dark mode", True), ("Light mode", False)):
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=dark: self._set_dark_mode(value)
            )
            self._theme_action_group.addAction(action)
            self._theme_actions["dark" if dark else "light"] = action
            theme_menu.addAction(action)
        self._theme_actions["dark"].setShortcut(QKeySequence("Ctrl+Shift+T"))
        text_menu = view_menu.addMenu("Text size")
        for label in GUI_FONT_SIZES:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=label: self._on_gui_font_size_changed(value)
            )
            self._font_size_action_group.addAction(action)
            self._font_size_actions[label] = action
            text_menu.addAction(action)
        cmap_menu = view_menu.addMenu("Thumbnail colourmap")
        for label in ("Gray", "Viridis", "Inferno", "Magma", "Plasma", "Cividis"):
            if label not in CMAP_KEY:
                continue
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=label: self._set_thumbnail_colormap(value)
            )
            self._thumbnail_cmap_action_group.addAction(action)
            self._thumbnail_cmap_actions[label] = action
            cmap_menu.addAction(action)
        channel_menu = view_menu.addMenu("Thumbnail channel")
        for label in THUMBNAIL_CHANNEL_OPTIONS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=label: self._set_thumbnail_channel(value)
            )
            self._thumbnail_channel_action_group.addAction(action)
            self._thumbnail_channel_actions[label] = action
            channel_menu.addAction(action)

        processing_menu = menu_bar.addMenu("Processing")
        align_menu = processing_menu.addMenu("Align rows")
        for label in ("None", "Mean", "Median"):
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=label: self._set_thumbnail_align(value)
            )
            self._thumbnail_align_action_group.addAction(action)
            self._thumbnail_align_actions[label] = action
            align_menu.addAction(action)

        convert_menu = menu_bar.addMenu("Convert")
        _mode_action(convert_menu, "Convert Createc .dat to .sxm...", "convert", "Ctrl+2")
        batch_convert_action = QAction("Batch convert folder...", self)
        batch_convert_action.triggered.connect(lambda: self._switch_mode("convert"))
        convert_menu.addAction(batch_convert_action)

        tools_menu = menu_bar.addMenu("Tools")
        map_action = QAction("Map Spectra to Images...", self)
        map_action.triggered.connect(self._on_map_spectra)
        tools_menu.addAction(map_action)
        tools_menu.addSeparator()
        _mode_action(tools_menu, "Feature counting", "features", "Ctrl+3")
        _mode_action(tools_menu, "TV denoise", "tv", "Ctrl+4")
        tools_menu.addSeparator()
        _mode_action(tools_menu, "Developer tools", "dev", "Ctrl+5")
        prefs_action = QAction("Preferences...", self)
        prefs_action.setEnabled(False)
        tools_menu.addAction(prefs_action)

        help_menu = menu_bar.addMenu("Help")
        definitions_action = QAction("Definitions", self)
        definitions_action.setShortcut(QKeySequence("Ctrl+6"))
        definitions_action.triggered.connect(self._show_definitions)
        help_menu.addAction(definitions_action)
        help_menu.addSeparator()
        github_action = QAction("GitHub", self)
        github_action.triggered.connect(lambda: _open_url(GITHUB_URL))
        help_menu.addAction(github_action)
        report_action = QAction("Report issue", self)
        report_action.triggered.connect(lambda: _open_url(f"{GITHUB_URL}/issues"))
        help_menu.addAction(report_action)
        help_menu.addSeparator()
        about_action = QAction("About ProbeFlow", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        self._sync_menu_actions()

    def _apply_default_splitter_sizes(self) -> None:
        center_default = max(
            self.CENTRAL_BROWSER_MIN_W,
            self.width()
            - self.LEFT_SIDEBAR_DEFAULT_W
            - self.RIGHT_INSPECTOR_DEFAULT_W,
        )
        self._browse_splitter.setSizes([
            self.LEFT_SIDEBAR_DEFAULT_W,
            center_default,
        ])
        self._splitter.setSizes([
            self.LEFT_SIDEBAR_DEFAULT_W + center_default,
            self.RIGHT_INSPECTOR_DEFAULT_W,
        ])

    def _menu_open_folder(self) -> None:
        self._switch_mode("browse")
        self._open_browse_folder()

    def _set_dark_mode(self, dark: bool) -> None:
        dark = bool(dark)
        if self._dark == dark:
            self._sync_menu_actions()
            return
        self._dark = dark
        self._navbar.set_dark(self._dark)
        self._apply_theme()

    def _set_thumbnail_colormap(self, label: str) -> None:
        if hasattr(self, "_browse_tools") and self._browse_tools.cmap_cb.currentText() != label:
            self._browse_tools.cmap_cb.setCurrentText(label)
        else:
            self._on_thumbnail_colormap_changed(CMAP_KEY.get(label, DEFAULT_CMAP_KEY))
        self._sync_menu_actions()

    def _set_thumbnail_channel(self, channel: str) -> None:
        if hasattr(self, "_browse_tools") and self._browse_tools.thumbnail_channel_cb.currentText() != channel:
            self._browse_tools.thumbnail_channel_cb.setCurrentText(channel)
        else:
            self._on_thumbnail_channel_changed(channel)
        self._sync_menu_actions()

    def _set_thumbnail_align(self, mode: str) -> None:
        if hasattr(self, "_browse_tools") and self._browse_tools.align_rows_cb.currentText() != mode:
            self._browse_tools.align_rows_cb.setCurrentText(mode)
        else:
            self._on_thumbnail_align_changed(mode)
        self._sync_menu_actions()

    def _show_definitions(self) -> None:
        theme = THEMES["dark" if self._dark else "light"]
        dlg = getattr(self, "_definitions_dialog", None)
        if dlg is None:
            dlg = _DefinitionsDialog(theme, self)
            self._definitions_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    # ── Mode switching ─────────────────────────────────────────────────────────
    def _switch_mode(self, mode: str):
        if mode == "defs":
            self._show_definitions()
            self._status_bar.showMessage(
                "Processing algorithm reference opened in a separate window")
            self._sync_menu_actions()
            return
        self._mode = mode
        if mode == "browse":
            self._content_stack.setCurrentIndex(0)
            self._sidebar_stack.setCurrentIndex(0)
            n = len(self._grid.get_entries())
            self._status_bar.showMessage(
                f"{n} scan(s) loaded" if n else "Open a folder to browse scans")
        elif mode == "features":
            self._content_stack.setCurrentIndex(2)
            self._sidebar_stack.setCurrentIndex(2)
            if self._features_panel.current_array() is None:
                self._status_bar.showMessage(
                    "Pick a scan in Browse, then 'Load primary scan from Browse'")
            else:
                self._status_bar.showMessage("FeatureCounting — pick a mode and Run")
        elif mode == "tv":
            self._content_stack.setCurrentIndex(3)
            self._sidebar_stack.setCurrentIndex(3)
            if self._tv_panel.current_array() is None:
                self._status_bar.showMessage(
                    "Pick a scan in Browse, then 'Load primary scan from Browse'")
            else:
                self._status_bar.showMessage("TV-denoise — adjust parameters and Run")
        elif mode == "dev":
            self._content_stack.setCurrentIndex(4)
            self._sidebar_stack.setCurrentIndex(4)
            self._status_bar.showMessage(
                "Developer terminal — run shell commands and Python scripts")
        else:
            self._content_stack.setCurrentIndex(1)
            self._sidebar_stack.setCurrentIndex(1)
            self._update_count(self._conv_panel.input_entry.text())
        self._update_tab_styles()

    def _update_tab_styles(self):
        self._sync_menu_actions()

    def _sync_menu_actions(self) -> None:
        if hasattr(self, "_mode_actions"):
            for mode, action in self._mode_actions.items():
                action.blockSignals(True)
                action.setChecked(self._mode == mode)
                action.blockSignals(False)
        if hasattr(self, "_theme_actions"):
            dark_key = "dark" if self._dark else "light"
            for key, action in self._theme_actions.items():
                action.blockSignals(True)
                action.setChecked(key == dark_key)
                action.blockSignals(False)
        if hasattr(self, "_font_size_actions"):
            for label, action in self._font_size_actions.items():
                action.blockSignals(True)
                action.setChecked(label == self._gui_font_size)
                action.blockSignals(False)
        if hasattr(self, "_thumbnail_cmap_actions"):
            cmap_label = (
                self._browse_tools.cmap_cb.currentText()
                if hasattr(self, "_browse_tools")
                else self._cfg.get("colormap", DEFAULT_CMAP_LABEL)
            )
            for label, action in self._thumbnail_cmap_actions.items():
                action.blockSignals(True)
                action.setChecked(label == cmap_label)
                action.blockSignals(False)
        if hasattr(self, "_thumbnail_channel_actions"):
            channel = (
                self._browse_tools.thumbnail_channel_cb.currentText()
                if hasattr(self, "_browse_tools")
                else THUMBNAIL_CHANNEL_DEFAULT
            )
            for label, action in self._thumbnail_channel_actions.items():
                action.blockSignals(True)
                action.setChecked(label == channel)
                action.blockSignals(False)
        if hasattr(self, "_thumbnail_align_actions"):
            align = (
                self._browse_tools.align_rows_cb.currentText()
                if hasattr(self, "_browse_tools")
                else "None"
            )
            for label, action in self._thumbnail_align_actions.items():
                action.blockSignals(True)
                action.setChecked(label == align)
                action.blockSignals(False)

    # ── Browse ─────────────────────────────────────────────────────────────────
    def _open_browse_folder(self):
        dialog = QFileDialog(self, "Open folder containing scan / .VERT files")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, False)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        if not dialog.exec():
            return
        selected = dialog.selectedFiles()
        d = selected[0] if selected else ""
        if not d:
            return
        self._switch_mode("browse")
        # Shallow navigation: the grid drives indexing. Set the new root and
        # let the grid render the immediate folder + subfolder cards.
        self._grid.set_root(Path(d))
        # New root → discard previous spec mapping; user can rebuild it.
        self._spec_image_map = {}
        self._browse_info.clear()
        self._update_browse_status()

    def _on_grid_folder_changed(self, path: Path):
        """Status-bar update + clear any selection-driven UI when navigating."""
        self._browse_info.clear()
        self._update_browse_status()

    def _update_browse_status(self):
        entries = self._grid.get_entries()
        n_folders = sum(1 for e in entries if isinstance(e, FolderEntry))
        n_sxm     = sum(1 for e in entries if isinstance(e, SxmFile))
        n_vert    = sum(1 for e in entries if isinstance(e, VertFile))
        self._n_loaded = n_sxm + n_vert
        cur = self._grid.current_dir()
        parts: list[str] = []
        if n_folders:
            parts.append(f"{n_folders} folder{'s' if n_folders != 1 else ''}")
        if n_sxm:
            parts.append(f"{n_sxm} scan{'s' if n_sxm != 1 else ''}")
        if n_vert:
            parts.append(f"{n_vert} spec{'s' if n_vert != 1 else ''}")
        desc = ", ".join(parts) if parts else "0 items"
        loc = cur.name if cur else "?"
        self._status_bar.showMessage(
            f"{loc}: {desc} — Double-click a folder to navigate, a scan to view")

    def _on_entry_select(self, entry):
        if isinstance(entry, VertFile):
            self._browse_info.show_vert_entry(entry)
            n_sel = len(self._grid.get_selected())
            sweep = entry.sweep_type.replace("_", " ")
            self._status_bar.showMessage(
                f"{entry.stem}  |  {sweep}  |  {entry.n_points} pts  |  "
                f"{n_sel} selected / {self._n_loaded} total  |  Double-click to view")
        else:
            cmap_key, _, proc = self._grid.get_card_state(entry.stem)
            self._browse_info.show_entry(entry, cmap_key, proc)
            n_sel = len(self._grid.get_selected())
            self._status_bar.showMessage(
                f"{entry.stem}  |  {entry.Nx}×{entry.Ny} px  |  "
                f"{n_sel} selected / {self._n_loaded} total  |  Double-click to view full size")

    def _on_selection_changed(self, n_selected: int):
        self._browse_tools.update_selection_hint(n_selected)

    def _on_filter_changed(self, mode: str):
        self._grid.apply_filter(mode)
        entries = self._grid.get_entries()
        n_sxm  = sum(1 for e in entries if isinstance(e, SxmFile))
        n_vert = sum(1 for e in entries if isinstance(e, VertFile))
        img_word  = "image"    if n_sxm  == 1 else "images"
        spec_word = "spectrum" if n_vert == 1 else "spectra"
        if mode == "images":
            msg = f"{n_sxm} {img_word}  ({n_vert} {spec_word} hidden)"
        elif mode == "spectra":
            msg = f"{n_vert} {spec_word}  ({n_sxm} {img_word} hidden)"
        else:
            msg = f"{n_sxm} {img_word}, {n_vert} {spec_word}"
        self._status_bar.showMessage(msg)

    def _on_thumbnail_channel_changed(self, channel: str):
        n = self._grid.set_thumbnail_channel(channel)
        if n == 0:
            self._status_bar.showMessage(f"Thumbnail channel: {channel}")
        else:
            self._status_bar.showMessage(
                f"Thumbnail channel: {channel} — queued {n} image thumbnail"
                f"{'s' if n != 1 else ''}")
        self._sync_menu_actions()

    def _on_thumbnail_colormap_changed(self, cmap_key: str):
        n = self._grid.set_thumbnail_colormap(cmap_key)
        label = next((l for l, k in CMAP_KEY.items() if k == cmap_key), cmap_key)
        if n == 0:
            self._status_bar.showMessage(f"Thumbnail colormap: {label}")
        else:
            self._status_bar.showMessage(
                f"Thumbnail colormap: {label} — queued {n} image thumbnail"
                f"{'s' if n != 1 else ''}")
        self._refresh_primary_channel_previews()
        self._sync_menu_actions()

    def _on_thumbnail_align_changed(self, mode: str):
        n = self._grid.set_thumbnail_align_rows(mode)
        label = mode if mode in ("Median", "Mean") else "None"
        if n == 0:
            self._status_bar.showMessage(f"Thumbnail align rows: {label}")
        else:
            self._status_bar.showMessage(
                f"Thumbnail align rows: {label} — queued {n} image thumbnail"
                f"{'s' if n != 1 else ''}")
        self._sync_menu_actions()

    def _refresh_primary_channel_previews(self):
        primary = self._grid.get_primary()
        if primary:
            entry = next((e for e in self._grid.get_entries()
                          if e.stem == primary), None)
            if entry and isinstance(entry, SxmFile):
                self._browse_info.load_channels(
                    entry, self._grid.thumbnail_colormap(), processing=None)

    def _on_map_spectra(self):
        """Open the folder-level spec→image mapping dialog."""
        entries = self._grid.get_entries()
        sxm_entries  = [e for e in entries if isinstance(e, SxmFile)]
        vert_entries = [e for e in entries if isinstance(e, VertFile)]
        if not vert_entries:
            self._status_bar.showMessage("No spectroscopy files in the current folder.")
            return
        if not sxm_entries:
            self._status_bar.showMessage("No images loaded — open a folder with .sxm files first.")
            return
        dlg = SpecMappingDialog(sxm_entries, vert_entries, self._spec_image_map, self)
        if dlg.exec() == QDialog.Accepted:
            new_map = dlg.get_mapping()
            self._spec_image_map.clear()
            self._spec_image_map.update(new_map)
            self._status_bar.showMessage(
                f"Spec mapping updated: {len(new_map)} of "
                f"{len(vert_entries)} spectra assigned.")

    def _on_card_context_action(self, entry, action: str):
        """Dispatch ScanCard right-click actions (Send to Features, export, show metadata)."""
        if action == "features":
            self._switch_mode("features")
            try:
                _scan = load_scan(entry.path)
            except Exception as exc:
                self._status_bar.showMessage(f"Could not read scan: {exc}")
                return
            plane_idx = self._features_sidebar.plane_index()
            if plane_idx >= _scan.n_planes:
                plane_idx = 0
            arr = _scan.planes[plane_idx]
            if arr is None:
                self._status_bar.showMessage("Could not read scan plane.")
                return
            w_m, h_m = _scan.scan_range_m
            Ny, Nx = arr.shape
            if Nx <= 0 or Ny <= 0 or w_m <= 0 or h_m <= 0:
                px_m = 1e-10
            else:
                px_m = float(np.sqrt((w_m / Nx) * (h_m / Ny)))
            self._features_panel.load_entry(entry, plane_idx, arr, px_m)
            self._features_sidebar.set_status(
                f"Loaded {entry.stem} (plane {plane_idx})")
            self._status_bar.showMessage(f"{entry.stem} sent to FeatureCounting")

        elif action == "export_metadata_csv":
            try:
                _scan = load_scan(entry.path)
                header = dict(getattr(_scan, "header", {}) or {})
            except Exception as exc:
                self._status_bar.showMessage(f"Could not read scan: {exc}")
                return
            if not header:
                self._status_bar.showMessage("No metadata to export")
                return
            out_path, _ = QFileDialog.getSaveFileName(
                self, "Export metadata as CSV",
                str(Path.home() / f"{entry.stem}_metadata.csv"),
                "CSV files (*.csv)")
            if not out_path:
                return
            try:
                import csv
                with open(out_path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["key", "value"])
                    for k in sorted(header):
                        w.writerow([k, header[k]])
                self._status_bar.showMessage(f"Metadata → {out_path}")
            except Exception as exc:
                self._status_bar.showMessage(f"Export error: {exc}")

        elif action == "show_metadata":
            try:
                _scan = load_scan(entry.path)
                header = dict(getattr(_scan, "header", {}) or {})
            except Exception as exc:
                self._status_bar.showMessage(f"Could not read scan: {exc}")
                return
            dlg = QDialog(self)
            dlg.setWindowTitle(f"Metadata — {entry.stem}")
            dlg.resize(560, 600)
            v = QVBoxLayout(dlg)
            tbl = QTableWidget(len(header), 2, dlg)
            tbl.setHorizontalHeaderLabels(["Key", "Value"])
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tbl.verticalHeader().setVisible(False)
            tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
            tbl.setFont(QFont("Helvetica", 9))
            for row, k in enumerate(sorted(header)):
                tbl.setItem(row, 0, QTableWidgetItem(str(k)))
                tbl.setItem(row, 1, QTableWidgetItem(str(header[k])))
            v.addWidget(tbl)
            close_btn = QPushButton("Close", dlg)
            close_btn.clicked.connect(dlg.accept)
            v.addWidget(close_btn)
            dlg.exec()

    # ── Features tab handlers ──────────────────────────────────────────────────
    def _on_features_load_from_browse(self):
        primary = self._grid.get_primary()
        if not primary:
            self._features_sidebar.set_status("Select a scan in the Browse tab first.")
            self._status_bar.showMessage("Pick a scan in Browse to load it into FeatureCounting")
            return
        entry = next((e for e in self._grid.get_entries() if e.stem == primary), None)
        if not entry or isinstance(entry, VertFile):
            self._features_sidebar.set_status("Selected entry is not a topography scan.")
            return
        plane_idx = self._features_sidebar.plane_index()
        try:
            _scan = load_scan(entry.path)
            if plane_idx >= _scan.n_planes:
                plane_idx = 0
            arr = _scan.planes[plane_idx]
            w_m, h_m = _scan.scan_range_m
        except Exception as exc:
            self._features_sidebar.set_status(f"Could not read scan: {exc}")
            return
        if arr is None:
            self._features_sidebar.set_status("Could not read scan plane.")
            return
        Ny, Nx = arr.shape
        if Nx <= 0 or Ny <= 0 or w_m <= 0 or h_m <= 0:
            px_m = 1e-10
        else:
            px_m = float(np.sqrt((w_m / Nx) * (h_m / Ny)))
        self._features_panel.load_entry(entry, plane_idx, arr, px_m)
        self._features_sidebar.set_status(
            f"Loaded {entry.stem} (plane {plane_idx}, px = {px_m * 1e12:.1f} pm)")

    def _on_features_run(self, mode: str):
        arr = self._features_panel.current_array()
        if arr is None:
            self._features_sidebar.set_status("Load a scan first.")
            return
        px_m = self._features_panel.current_pixel_size()
        if px_m <= 0:
            self._features_sidebar.set_status("Scan has no physical pixel size.")
            return

        if mode == "particles":
            params = self._features_sidebar.particles_params()
        elif mode == "template":
            tmpl = self._features_panel.get_template()
            if tmpl is None:
                self._features_sidebar.set_status(
                    "Crop a template first (Template mode → 'Crop template…').")
                return
            params = self._features_sidebar.template_params()
            params["template"] = tmpl
        elif mode == "lattice":
            params = {}
        else:
            self._features_sidebar.set_status(f"Unknown mode {mode!r}")
            return

        self._features_sidebar.set_status(f"Running {mode}…")
        worker = _FeaturesWorker(mode, arr, px_m, params, self._features_signals)
        self._features_pool.start(worker)

    def _on_features_finished(self, mode: str, result, error: str):
        if error:
            self._features_sidebar.set_status(f"{mode} failed: {error}")
            self._status_bar.showMessage(f"{mode} failed: {error}")
            return
        if mode == "particles":
            self._features_panel.set_particles(result)
            self._features_sidebar.set_status(
                f"Found {len(result)} particle(s).")
        elif mode == "template":
            self._features_panel.set_detections(result)
            self._features_sidebar.set_status(
                f"Found {len(result)} match(es).")
        elif mode == "lattice":
            self._features_panel.set_lattice(result)
            self._features_sidebar.set_status(
                f"|a|={result.a_length_m * 1e9:.3f} nm  "
                f"|b|={result.b_length_m * 1e9:.3f} nm  "
                f"γ={result.gamma_deg:.1f}°")

    def _on_features_export(self, mode: str):
        if mode == "particles":
            items = self._features_panel.get_particles()
            kind  = "particles"
        elif mode == "template":
            items = self._features_panel.get_detections()
            kind  = "detections"
        elif mode == "lattice":
            lat = self._features_panel.get_lattice()
            items = [lat] if lat is not None else []
            kind  = "lattice"
        else:
            return
        if not items:
            self._features_sidebar.set_status("Nothing to export — run an analysis first.")
            return
        entry = self._features_panel.current_entry()
        suggested = (Path.home() / f"{entry.stem if entry else 'features'}_{kind}.json")
        out_path, _ = QFileDialog.getSaveFileName(
            self, f"Export {kind} JSON", str(suggested), "JSON (*.json)")
        if not out_path:
            return
        try:
            from probeflow.io.writers.json import write_json
            write_json(out_path, items, kind=kind,
                       extra_meta={"source": str(entry.path) if entry else None})
            self._features_sidebar.set_status(f"Exported → {out_path}")
            self._status_bar.showMessage(f"Exported {kind} → {out_path}")
        except Exception as exc:
            self._features_sidebar.set_status(f"Export failed: {exc}")

    # ── TV-denoise tab handlers ────────────────────────────────────────────────
    def _on_tv_load_from_browse(self):
        primary = self._grid.get_primary()
        if not primary:
            self._tv_sidebar.set_status("Select a scan in the Browse tab first.")
            self._status_bar.showMessage("Pick a scan in Browse to load it into TV-denoise")
            return
        entry = next((e for e in self._grid.get_entries() if e.stem == primary), None)
        if not entry or isinstance(entry, VertFile):
            self._tv_sidebar.set_status("Selected entry is not a topography scan.")
            return
        plane_idx = self._tv_sidebar.plane_index()
        try:
            _scan = load_scan(entry.path)
            if plane_idx >= _scan.n_planes:
                plane_idx = 0
            arr = _scan.planes[plane_idx]
            w_m, h_m = _scan.scan_range_m
        except Exception as exc:
            self._tv_sidebar.set_status(f"Could not read scan: {exc}")
            return
        if arr is None:
            self._tv_sidebar.set_status("Could not read scan plane.")
            return
        Ny, Nx = arr.shape
        if Nx <= 0 or Ny <= 0 or w_m <= 0 or h_m <= 0:
            px_m = 1e-10
        else:
            px_m = float(np.sqrt((w_m / Nx) * (h_m / Ny)))
        self._tv_panel.load_entry(entry, plane_idx, arr, px_m)
        self._tv_sidebar.set_status(
            f"Loaded {entry.stem} (plane {plane_idx}). Adjust parameters and Run.")

    def _on_tv_run(self):
        arr = self._tv_panel.current_array()
        if arr is None:
            self._tv_sidebar.set_status("Load a scan first.")
            return
        params = self._tv_sidebar.params()
        self._tv_sidebar.set_running(True)
        self._tv_sidebar.set_status(f"Running TV-denoise ({params['method']})…")
        worker = _TVWorker(arr, params, self._tv_signals)
        self._tv_pool.start(worker)

    def _on_tv_finished(self, result, error: str):
        self._tv_sidebar.set_running(False)
        if error:
            self._tv_sidebar.set_status(f"TV-denoise failed: {error}")
            self._status_bar.showMessage(f"TV-denoise failed: {error}")
            return
        self._tv_panel.set_denoised(result)
        self._tv_sidebar.set_status("Done. Save the denoised PNG, or Run again.")

    def _on_tv_revert(self):
        self._tv_panel.set_denoised(None)
        self._tv_sidebar.set_status("Reverted to original.")

    def _on_tv_save_png(self):
        out = self._tv_panel.current_denoised()
        if out is None:
            self._tv_sidebar.set_status("Run TV-denoise first.")
            return
        entry = self._tv_panel.current_entry()
        plane_idx = self._tv_panel.current_plane_idx()
        suggested = (Path.home() /
                     f"{entry.stem if entry else 'tv'}_p{plane_idx}_tvdenoise.png")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save denoised PNG", str(suggested), "PNG (*.png)")
        if not out_path:
            return
        try:
            from probeflow.processing import export_png
            from probeflow.io.writers.png import lut_from_matplotlib
            px_m = self._tv_panel.current_pixel_size()
            Ny, Nx = out.shape
            scan_range_m = (px_m * Nx, px_m * Ny)
            export_png(
                out, out_path, "gray", 1.0, 99.0,
                lut_fn=lut_from_matplotlib,
                scan_range_m=scan_range_m,
                add_scalebar=True,
                scalebar_unit="nm",
                scalebar_pos="bottom-right",
            )
            self._tv_sidebar.set_status(f"Saved → {out_path}")
            self._status_bar.showMessage(f"Saved {out_path}")
        except Exception as exc:
            self._tv_sidebar.set_status(f"Save failed: {exc}")

    def _open_viewer(self, entry):
        t = THEMES["dark" if self._dark else "light"]
        if isinstance(entry, VertFile):
            dlg = SpecViewerDialog(entry, t, self)
            dlg.exec()
        else:
            cmap_key, clip, proc = self._grid.get_card_state(entry.stem)
            sxm_entries = [e for e in self._grid.get_entries() if isinstance(e, SxmFile)]
            initial_plane_idx = self._grid.thumbnail_plane_index_for_entry(entry)
            dlg = ImageViewerDialog(entry, sxm_entries, cmap_key, t, self,
                                    clip_low=clip[0], clip_high=clip[1],
                                    processing=proc,
                                    spec_image_map=self._spec_image_map,
                                    initial_plane_idx=initial_plane_idx)
            dlg.exec()
            # Handle "Send to …" actions requested from inside the viewer.
            action = getattr(dlg, "_deferred_action", "")
            if action in ("features", "tv"):
                self._load_from_viewer(dlg, action)

    def _load_from_viewer(self, dlg, action: str):
        """Load the processed array from a closed ImageViewerDialog into Features or TV."""
        entry = dlg._entries[dlg._idx]
        plane_idx = getattr(dlg, "_deferred_plane_idx", dlg._ch_cb.currentIndex())
        arr = dlg._display_arr if dlg._display_arr is not None else dlg._raw_arr
        if arr is None:
            self._status_bar.showMessage("Viewer had no image data to send.")
            return
        scan_range = dlg._scan_range_m
        shape = arr.shape
        if scan_range and shape and shape[0] > 0 and shape[1] > 0:
            w_m, h_m = float(scan_range[0]), float(scan_range[1])
            px_m = float(np.sqrt((w_m / shape[1]) * (h_m / shape[0])))
        else:
            px_m = 1e-10
        if action == "features":
            self._switch_mode("features")
            self._features_panel.load_entry(entry, plane_idx, arr, px_m)
            self._features_sidebar.set_status(
                f"Loaded {entry.stem} (processed, plane {plane_idx}, px = {px_m * 1e12:.1f} pm)")
            self._status_bar.showMessage(f"{entry.stem} → Feature Counting")
        elif action == "tv":
            self._switch_mode("tv")
            self._tv_panel.load_entry(entry, plane_idx, arr, px_m)
            self._tv_sidebar.set_status(
                f"Loaded {entry.stem} (processed, plane {plane_idx}). Adjust parameters and Run.")
            self._status_bar.showMessage(f"{entry.stem} → TV Denoising")

    # ── Convert ────────────────────────────────────────────────────────────────
    def _update_count(self, text: str = ""):
        d = (text or self._conv_panel.input_entry.text()).strip()
        if d and Path(d).is_dir():
            n = len(list(Path(d).glob("*.dat")))
            self._convert_sidebar.update_file_count(n)
        else:
            self._convert_sidebar.update_file_count(-1)

    def _run(self):
        if self._running:
            return
        in_dir  = self._conv_panel.input_entry.text().strip()
        out_dir = self._conv_panel.get_output_dir()
        do_png  = self._convert_sidebar.png_cb.isChecked()
        do_sxm  = self._convert_sidebar.sxm_cb.isChecked()
        clip_lo = self._convert_sidebar.clip_low_spin.value()
        clip_hi = self._convert_sidebar.clip_high_spin.value()

        if not in_dir:
            self._conv_panel.log("ERROR: Please select an input folder.", "err"); return
        if out_dir and not Path(out_dir).is_dir():
            self._conv_panel.log(f"ERROR: Output folder not found: {out_dir}", "err"); return
        if not do_png and not do_sxm:
            self._conv_panel.log("ERROR: Select at least one output format.", "err"); return
        if not Path(in_dir).is_dir():
            self._conv_panel.log(f"ERROR: Input folder not found: {in_dir}", "err"); return

        self._running = True
        self._convert_sidebar.run_btn.setText("  Running…  ")
        self._convert_sidebar.run_btn.setEnabled(False)
        self._status_bar.showMessage("Converting…")

        worker = ConversionWorker(in_dir, out_dir, do_png, do_sxm, clip_lo, clip_hi)
        worker.signals.log_msg.connect(self._conv_panel.log)
        worker.signals.finished.connect(self._on_done)
        QThreadPool.globalInstance().start(worker)

    @Slot(str)
    def _on_done(self, out_dir: str):
        self._running = False
        self._convert_sidebar.run_btn.setText("  RUN  ")
        self._convert_sidebar.run_btn.setEnabled(True)
        sxm_dir = Path(out_dir) / "sxm"
        entries = scan_image_folder(sxm_dir) if sxm_dir.exists() else []
        if entries:
            self._grid.load(entries, folder_path=str(sxm_dir))
            self._n_loaded = len(entries)
            self._switch_mode("browse")
            self._status_bar.showMessage(
                f"Done — {self._n_loaded} scan(s) ready to browse")
        else:
            self._status_bar.showMessage("Done")

    # ── Theme ──────────────────────────────────────────────────────────────────
    def _toggle_theme(self):
        self._set_dark_mode(not self._dark)

    def _on_gui_font_size_changed(self, label: str):
        self._gui_font_size = normalise_gui_font_size(label)
        self._navbar.blockSignals(True)
        self._navbar.set_font_size(self._gui_font_size)
        self._navbar.blockSignals(False)
        self._apply_theme()
        self._status_bar.showMessage(f"Text size: {self._gui_font_size}")

    def _apply_theme(self):
        t = THEMES["dark" if self._dark else "light"]
        app = QApplication.instance()
        app.setFont(QFont("Helvetica", GUI_FONT_SIZES[self._gui_font_size]))
        app.setStyleSheet(_build_qss(t, GUI_FONT_SIZES[self._gui_font_size]))
        self._grid.apply_theme(t)
        self._browse_tools.apply_theme(t)
        self._browse_info.apply_theme(t)
        self._conv_panel.apply_theme(t)
        self._update_tab_styles()

    # ── About ──────────────────────────────────────────────────────────────────
    def _show_about(self):
        t   = THEMES["dark" if self._dark else "light"]
        dlg = AboutDialog(t, self)
        dlg.exec()

    # ── Close ──────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        save_config({
            "dark_mode":     self._dark,
            "input_dir":     self._conv_panel.input_entry.text(),
            "output_dir":    self._conv_panel.output_entry.text(),
            "custom_output": self._conv_panel._custom_out_cb.isChecked(),
            "do_png":        self._convert_sidebar.png_cb.isChecked(),
            "do_sxm":        self._convert_sidebar.sxm_cb.isChecked(),
            "clip_low":      self._convert_sidebar.clip_low_spin.value(),
            "clip_high":     self._convert_sidebar.clip_high_spin.value(),
            "colormap":      self._browse_tools.cmap_cb.currentText(),
            "browse_filter": self._browse_tools.get_filter_mode(),
            "gui_font_size": self._gui_font_size,
        })
        super().closeEvent(event)


# ── Helper widgets ─────────────────────────────────────────────────────────────
def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedHeight(1)
    return line


# ── QSS stylesheet ─────────────────────────────────────────────────────────────
def _build_qss(t: dict, font_pt: int = GUI_FONT_SIZES[GUI_FONT_DEFAULT]) -> str:
    return f"""
QMainWindow, QWidget {{
    background-color: {t['main_bg']};
    color: {t['fg']};
    font-family: Helvetica, Arial, sans-serif;
    font-size: {font_pt}pt;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background-color: {t['main_bg']};
    border: none;
}}
BrowseToolPanel, BrowseToolPanel QWidget,
BrowseInfoPanel, BrowseInfoPanel QWidget,
ConvertSidebar, ConvertSidebar QWidget,
ConvertPanel, ConvertPanel QWidget {{
    background-color: {t['sidebar_bg']};
}}
BrowseToolPanel QLabel, BrowseInfoPanel QLabel,
ConvertSidebar QLabel, ConvertPanel QLabel {{
    color: {t['fg']};
    background: transparent;
}}
QPushButton {{
    background-color: {t['btn_bg']};
    color: {t['btn_fg']};
    border: none;
    border-radius: 4px;
    padding: 5px 12px;
}}
QPushButton:hover {{ background-color: {t['sep']}; }}
QPushButton:disabled {{
    background-color: {t['entry_bg']};
    color: {t['sub_fg']};
}}
QPushButton#accentBtn {{
    background-color: {t['accent_bg']};
    color: {t['accent_fg']};
    font-weight: bold;
}}
QPushButton#accentBtn:disabled {{
    background-color: {t['entry_bg']};
    color: {t['sub_fg']};
}}
QPushButton#segBtnLeft, QPushButton#segBtnMid, QPushButton#segBtnRight {{
    background-color: {t['btn_bg']};
    color: {t['btn_fg']};
    border: none;
    padding: 0px 8px;
    margin: 0px;
    border-radius: 0px;
}}
QPushButton#segBtnLeft {{
    border-top-left-radius: 4px;
    border-bottom-left-radius: 4px;
}}
QPushButton#segBtnRight {{
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}}
QPushButton#segBtnLeft:hover, QPushButton#segBtnMid:hover,
QPushButton#segBtnRight:hover {{
    background-color: {t['sep']};
}}
QPushButton#segBtnLeft:checked, QPushButton#segBtnMid:checked,
QPushButton#segBtnRight:checked {{
    background-color: {t['accent_bg']};
    color: {t['accent_fg']};
    font-weight: bold;
}}
QPushButton#navBtn {{
    color: #ffffff;
    background-color: transparent;
    border: 1px solid rgba(255,255,255,0.40);
    border-radius: 4px;
    padding: 4px 12px;
}}
QPushButton#navBtn:hover {{
    background-color: rgba(255,255,255,0.18);
}}
QComboBox {{
    background-color: {t['entry_bg']};
    color: {t['fg']};
    border: 1px solid {t['sep']};
    border-radius: 3px;
    padding: 4px 8px;
    selection-background-color: {t['accent_bg']};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background-color: {t['entry_bg']};
    color: {t['fg']};
    selection-background-color: {t['accent_bg']};
    selection-color: {t['accent_fg']};
    border: 1px solid {t['sep']};
    outline: none;
    font-size: 11pt;
}}
QComboBox QAbstractItemView::item {{
    min-height: 24px;
    padding: 2px 8px;
}}
QLineEdit {{
    background-color: {t['entry_bg']};
    color: {t['fg']};
    border: 1px solid {t['sep']};
    border-radius: 3px;
    padding: 4px 8px;
}}
QLineEdit:focus {{ border: 1px solid {t['accent_bg']}; }}
QTextEdit {{
    background-color: {t['log_bg']};
    color: {t['log_fg']};
    border: 1px solid {t['sep']};
    border-radius: 3px;
    font-family: monospace;
}}
QTableWidget {{
    background-color: {t['tree_bg']};
    color: {t['tree_fg']};
    border: none;
    gridline-color: transparent;
    alternate-background-color: {t['main_bg']};
}}
QTableWidget::item {{ padding: 3px 6px; }}
QTableWidget::item:selected {{
    background-color: {t['tree_sel']};
    color: {t['fg']};
}}
QHeaderView::section {{
    background-color: {t['tree_head']};
    color: {t['fg']};
    border: none;
    padding: 5px 6px;
    font-weight: bold;
}}
QScrollBar:vertical {{
    background-color: {t['main_bg']};
    width: 10px;
    border-radius: 5px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background-color: {t['sep']};
    border-radius: 5px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background-color: {t['main_bg']};
    height: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal {{
    background-color: {t['sep']};
    border-radius: 5px;
    min-width: 20px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QCheckBox {{ color: {t['fg']}; spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {t['sep']};
    border-radius: 3px;
    background-color: {t['entry_bg']};
}}
QCheckBox::indicator:checked {{
    background-color: {t['accent_bg']};
    border-color: {t['accent_bg']};
}}
QDoubleSpinBox {{
    background-color: {t['entry_bg']};
    color: {t['fg']};
    border: 1px solid {t['sep']};
    border-radius: 3px;
    padding: 3px 5px;
}}
QSplitter::handle {{ background-color: {t['splitter']}; }}
QStatusBar {{
    background-color: {t['status_bg']};
    color: {t['status_fg']};
    border-top: 1px solid {t['sep']};
    font-size: 10pt;
}}
QDialog {{ background-color: {t['bg']}; color: {t['fg']}; }}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {t['sep']};
    background-color: {t['sep']};
}}
"""


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    import sys
    app    = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ProbeFlow")
    window = ProbeFlowWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
