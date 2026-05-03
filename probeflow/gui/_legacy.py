"""ProbeFlow — PySide6 GUI for STM scan browsing, processing, and Createc→Nanonis conversion."""

from __future__ import annotations

# Layout cleanup note: this module is a compatibility parking area while the
# remaining Qt classes are moved into dedicated gui/* modules. New parser,
# writer, processing, analysis, plugin, and graph-node code belongs elsewhere.

import io
import json
import re as _re
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
    Qt, QEvent, QObject, QProcess, QRect, QRunnable, QThreadPool, QTimer,
    QSize, Signal, Slot,
)
from PySide6.QtGui import (
    QAction, QBrush, QColor, QCursor, QFont, QImage, QMovie, QPainter, QPen,
    QPixmap, QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
    QDialog, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSplitter, QStackedWidget,
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

class ProcessingControlPanel(QWidget):
    """Internal processing controls shared by Browse and Viewer."""

    QUICK_KEYS = ("align_rows", "remove_bad_lines")

    def __init__(self, mode: str, parent=None):
        super().__init__(parent)
        if mode not in ("browse_quick", "viewer_full"):
            raise ValueError(f"Unknown processing panel mode: {mode}")
        self._mode = mode
        self._build()

    _TWO_COL_THRESHOLD = 260  # px — switch to 1-col below this panel width

    def resizeEvent(self, event: "QResizeEvent"):
        super().resizeEvent(event)
        if not hasattr(self, "_two_col"):
            return
        use_two_col = event.size().width() >= self._TWO_COL_THRESHOLD
        if use_two_col == self._using_two_col:
            return
        self._using_two_col = use_two_col
        hbox = self._two_col.layout()
        if use_two_col:
            # Move bg_section back into the right half of two_col
            hbox.addWidget(self._bg_section, 1)
        else:
            # Pull bg_section out of two_col, place it below in the main column
            hbox.removeWidget(self._bg_section)
            idx = self._lay.indexOf(self._two_col)
            self._lay.insertWidget(idx + 1, self._bg_section)
        self._bg_section.show()

    def _build(self):
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(4, 2, 0, 2)
        self._lay.setSpacing(4)
        lay = self._lay  # local alias for closures

        def _combo_row(label: str, items: list[str],
                       target=None, lbl_width: int = 90) -> QComboBox:
            if target is None:
                target = lay
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFont(QFont("Helvetica", 8))
            if lbl_width >= 90:
                lbl.setFixedWidth(lbl_width)
            else:
                lbl.setMaximumWidth(lbl_width)
                lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            cb = QComboBox()
            cb.addItems(items)
            cb.setFont(QFont("Helvetica", 8))
            cb.setMinimumWidth(0)
            cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.addWidget(lbl)
            row.addWidget(cb, 1)
            target.addLayout(row)
            return cb

        def _sub_slider(label: str, mn: int, mx: int, init: int,
                        fmt="{v}") -> tuple[QWidget, QSlider, QLabel]:
            w = QWidget()
            rl = QHBoxLayout(w)
            rl.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label)
            lbl.setFont(QFont("Helvetica", 8))
            lbl.setMaximumWidth(44)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(mn, mx)
            sl.setValue(init)
            val_lbl = QLabel(fmt.format(v=init))
            val_lbl.setFont(QFont("Helvetica", 8))
            val_lbl.setFixedWidth(28)
            sl.valueChanged.connect(
                lambda v, vl=val_lbl, f=fmt: vl.setText(f.format(v=v)))
            rl.addWidget(lbl)
            rl.addWidget(sl, 1)
            rl.addWidget(val_lbl)
            return w, sl, val_lbl

        def _col_lbl(text: str, target):
            lbl = QLabel(text)
            lbl.setFont(QFont("Helvetica", 7, QFont.Bold))
            lbl.setAlignment(Qt.AlignCenter)
            target.addWidget(lbl)

        # ── Line corrections (full-width in both modes) ───────────────────────
        line_lbl = QLabel("Line corrections")
        line_lbl.setFont(QFont("Helvetica", 7, QFont.Bold))
        line_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(line_lbl)

        self._align_combo = _combo_row("Align rows:", ["None", "Median", "Mean"])

        self._bad_lines_combo = _combo_row(
            "Bad lines:", ["None", "MAD (rows)", "Step (cols)"]
        )
        self._bad_lines_combo.setToolTip(
            "MAD: flags rows whose median deviates by >5× MAD from the image median.\n"
            "Step: per-column scan for large vertical jumps; detects partial bad lines."
        )

        if self._mode == "browse_quick":
            lay.addStretch()
            return

        # ── Filter section (left column / full-width in 1-col) ────────────────
        self._filter_section = QWidget()
        L = QVBoxLayout(self._filter_section)
        L.setContentsMargins(0, 0, 0, 0)
        L.setSpacing(3)

        _col_lbl("Filters", L)

        self._smooth_combo = _combo_row("Smooth:", ["None", "Gaussian"], L, 54)
        self._smooth_sigma_w, self._smooth_sigma_sl, _ = _sub_slider(
            "sigma:", 1, 20, 1, "{v}px")
        L.addWidget(self._smooth_sigma_w)
        self._smooth_sigma_w.setVisible(False)
        self._smooth_combo.currentIndexChanged.connect(
            lambda i: self._smooth_sigma_w.setVisible(i != 0))

        self._highpass_combo = _combo_row("Hi-pass:", ["None", "Gaussian"], L, 54)
        self._highpass_sigma_w, self._highpass_sigma_sl, _ = _sub_slider(
            "sigma:", 1, 80, 8, "{v}px")
        L.addWidget(self._highpass_sigma_w)
        self._highpass_sigma_w.setVisible(False)
        self._highpass_combo.setToolTip(
            "ImageJ-like high-pass: subtracts a broad Gaussian-blurred background."
        )
        self._highpass_combo.currentIndexChanged.connect(
            lambda i: self._highpass_sigma_w.setVisible(i != 0))

        self._edge_combo = _combo_row("Edge:", ["None", "Laplacian", "LoG", "DoG"], L, 54)
        self._edge_sigma_w, self._edge_sigma_sl, _ = _sub_slider(
            "sigma:", 1, 20, 1, "{v}px")
        L.addWidget(self._edge_sigma_w)
        self._edge_sigma_w.setVisible(False)
        self._edge_combo.currentIndexChanged.connect(
            lambda i: self._edge_sigma_w.setVisible(i != 0))

        _col_lbl("Radial FFT", L)

        self._fft_combo = _combo_row("Mode:", ["None", "Low-pass", "High-pass"], L, 54)
        self._fft_combo.setToolTip(
            "Simple global radial low/high-pass filter. "
            "This is not the ImageJ Periodic Filter workflow."
        )
        self._fft_cutoff_widget, self._fft_sl, _ = _sub_slider(
            "cutoff:", 1, 50, 10, "{v}%")
        L.addWidget(self._fft_cutoff_widget)
        self._fft_cutoff_widget.setVisible(False)
        self._fft_combo.currentIndexChanged.connect(
            lambda i: self._fft_cutoff_widget.setVisible(i != 0))

        self._fft_soft_cb = QCheckBox("Soft border")
        self._fft_soft_cb.setFont(QFont("Helvetica", 8))
        self._fft_soft_cb.setToolTip(
            "Cosine-taper the image edges before FFT to suppress ringing artefacts "
            "(ImageJ FFT_Soft_Border approach)."
        )
        L.addWidget(self._fft_soft_cb)
        L.addStretch()

        # ── Background section (right column / below filters in 1-col) ────────
        self._bg_section = QWidget()
        R = QVBoxLayout(self._bg_section)
        R.setContentsMargins(0, 0, 0, 0)
        R.setSpacing(3)

        _col_lbl("Background", R)

        self._bg_combo = _combo_row(
            "Order:",
            ["None", "Plane", "Quad.", "Cubic", "Quart."],
            R, 46,
        )
        self._bg_step_cb = QCheckBox("Step-tolerant")
        self._bg_step_cb.setFont(QFont("Helvetica", 8))
        self._bg_step_cb.setToolTip(
            "Ignores steep pixels during polynomial surface fitting. "
            "This is not the STM line-background algorithm."
        )
        R.addWidget(self._bg_step_cb)

        self._stm_line_bg_combo = _combo_row(
            "STM line:",
            ["None", "Step-tol."],
            R, 54,
        )

        self._facet_cb = QCheckBox("Facet level")
        self._facet_cb.setFont(QFont("Helvetica", 8))
        self._facet_cb.setToolTip(
            "Level each atomically flat terrace to a common height reference."
        )
        R.addWidget(self._facet_cb)
        R.addStretch()

        # ── Two-column container (2-col mode: filter | bg side by side) ───────
        self._two_col = QWidget()
        hbox = QHBoxLayout(self._two_col)
        hbox.setContentsMargins(0, 2, 0, 0)
        hbox.setSpacing(4)
        hbox.addWidget(self._filter_section, 1)
        hbox.addWidget(self._bg_section, 1)
        lay.addWidget(self._two_col)
        self._using_two_col = True

    def state(self) -> dict:
        align_map = {0: None, 1: "median", 2: "mean"}
        bad_map = {0: None, 1: "mad", 2: "step"}
        cfg = {
            "align_rows": align_map[self._align_combo.currentIndex()],
            "remove_bad_lines": bad_map[self._bad_lines_combo.currentIndex()],
        }
        if self._mode == "browse_quick":
            return {k: cfg[k] for k in self.QUICK_KEYS}

        bg_map = {0: None, 1: 1, 2: 2, 3: 3, 4: 4}
        fft_map = {0: None, 1: "low_pass", 2: "high_pass"}
        edge_map = {0: None, 1: "laplacian", 2: "log", 3: "dog"}
        smooth_i = self._smooth_combo.currentIndex()
        highpass_i = self._highpass_combo.currentIndex()
        edge_i = self._edge_combo.currentIndex()
        fft_idx = self._fft_combo.currentIndex()
        cfg.update({
            "bg_order": bg_map[self._bg_combo.currentIndex()],
            "bg_step_tolerance": self._bg_step_cb.isChecked(),
            "stm_line_bg": (
                "step_tolerant"
                if self._stm_line_bg_combo.currentIndex() == 1
                else None
            ),
            "facet_level": self._facet_cb.isChecked(),
            "smooth_sigma": self._smooth_sigma_sl.value() if smooth_i != 0 else None,
            "highpass_sigma": self._highpass_sigma_sl.value() if highpass_i != 0 else None,
            "edge_method": edge_map[edge_i],
            "edge_sigma": self._edge_sigma_sl.value(),
            "edge_sigma2": self._edge_sigma_sl.value() * 2,
            "fft_mode": fft_map[fft_idx],
            "fft_cutoff": self._fft_sl.value() / 100.0,
            "fft_window": "hanning",
            "fft_soft_border": self._fft_soft_cb.isChecked(),
            "fft_soft_mode": fft_map.get(fft_idx) or "low_pass",
            "fft_soft_cutoff": self._fft_sl.value() / 100.0,
            "fft_soft_border_frac": 0.12,
        })
        return cfg

    def set_state(self, state: dict | None) -> None:
        state = state or {}
        self._align_combo.setCurrentIndex(
            {None: 0, "median": 1, "mean": 2}.get(state.get("align_rows"), 0))
        self._bad_lines_combo.setCurrentIndex(
            {None: 0, "mad": 1, "step": 2}.get(state.get("remove_bad_lines"), 0))
        if self._mode == "browse_quick":
            return

        self._bg_combo.setCurrentIndex(
            {None: 0, 1: 1, 2: 2, 3: 3, 4: 4}.get(state.get("bg_order"), 0))
        self._bg_step_cb.setChecked(bool(state.get("bg_step_tolerance", False)))
        self._stm_line_bg_combo.setCurrentIndex(
            {"step_tolerant": 1}.get(state.get("stm_line_bg"), 0))
        self._facet_cb.setChecked(bool(state.get("facet_level", False)))

        sigma = state.get("smooth_sigma")
        if sigma:
            self._smooth_combo.setCurrentIndex(1)
            self._smooth_sigma_sl.setValue(int(sigma))
        else:
            self._smooth_combo.setCurrentIndex(0)

        highpass = state.get("highpass_sigma")
        if highpass:
            self._highpass_combo.setCurrentIndex(1)
            self._highpass_sigma_sl.setValue(int(highpass))
        else:
            self._highpass_combo.setCurrentIndex(0)

        edge = state.get("edge_method")
        self._edge_combo.setCurrentIndex(
            {None: 0, "laplacian": 1, "log": 2, "dog": 3}.get(edge, 0))
        self._edge_sigma_sl.setValue(int(state.get("edge_sigma", 1)))

        fft_mode = state.get("fft_mode")
        self._fft_combo.setCurrentIndex(
            {None: 0, "low_pass": 1, "high_pass": 2}.get(fft_mode, 0))
        self._fft_sl.setValue(int(round(float(state.get("fft_cutoff", 0.10)) * 100)))
        self._fft_soft_cb.setChecked(bool(state.get("fft_soft_border", False)))


class FFTViewerDialog(QDialog):
    """Side-by-side real-space / FFT inspection window."""

    def __init__(
        self,
        arr: np.ndarray,
        scan_range_m: tuple,
        colormap: str = "gray",
        theme: dict | None = None,
        channel_unit: tuple | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("FFT Viewer")
        self.resize(1000, 700)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._arr = arr.astype(np.float64, copy=True)
        self._scan_range_m = scan_range_m
        self._colormap = colormap
        self._theme = theme or {}
        self._ch_unit = channel_unit  # (scale, unit, axis_label) or None

        self._fft_mag: np.ndarray | None = None
        self._qx: np.ndarray | None = None
        self._qy: np.ndarray | None = None
        self._fft_xlim: tuple = (0.0, 1.0)
        self._fft_ylim: tuple = (1.0, 0.0)
        self._scale_mode = "log"
        self._window_mode = "hann"
        self._dc_mode = "keep"
        self._pan_anchor: tuple | None = None
        self._disp_range: tuple = (0.0, 1.0)
        self._vmin_frac: float = 0.0
        self._vmax_frac: float = 1.0

        self._build()
        self._recompute_fft()
        self._update_info_panel()
        self._redraw()

    # ── layout ─────────────────────────────────────────────────────────────────

    def _build(self):
        bg = self._theme.get("bg", "#1e1e1e")
        fg = self._theme.get("fg", "#dddddd")

        lay = QVBoxLayout(self)
        lay.setSpacing(4)
        lay.setContentsMargins(6, 6, 6, 4)

        # ── toolbar ───────────────────────────────────────────────────────────
        tb = QHBoxLayout()
        tb.setSpacing(6)

        def _lbl(text):
            lb = QLabel(text)
            lb.setFont(QFont("Helvetica", 8))
            return lb

        def _combo(items, min_width=80):
            c = QComboBox()
            c.addItems(items)
            c.setFont(QFont("Helvetica", 8))
            c.setFixedHeight(22)
            c.setMinimumWidth(min_width)
            return c

        tb.addWidget(_lbl("Scale:"))
        self._scale_combo = _combo(["Log", "Linear"], 72)
        self._scale_combo.currentIndexChanged.connect(self._on_scale_changed)
        tb.addWidget(self._scale_combo)

        tb.addWidget(_lbl("Window:"))
        self._window_combo = _combo(["Hann", "None", "Tukey"], 76)
        self._window_combo.currentIndexChanged.connect(self._on_window_changed)
        tb.addWidget(self._window_combo)

        tb.addWidget(_lbl("DC:"))
        self._dc_combo = _combo(["Zero DC", "Keep DC", "Mask DC"], 88)
        self._dc_combo.setCurrentIndex(1)
        self._dc_combo.currentIndexChanged.connect(self._on_dc_changed)
        tb.addWidget(self._dc_combo)

        tb.addStretch(1)

        for label, tip, slot in [
            ("Fit", "Zoom to fit full FFT extent",    self._zoom_fit),
            ("Ctr", "Zoom to centre (quarter range)", self._zoom_centre),
            ("+",   "Zoom in",                        lambda: self._zoom_by(0.5)),
            ("-",   "Zoom out",                       lambda: self._zoom_by(2.0)),
        ]:
            btn = QPushButton(label)
            btn.setFont(QFont("Helvetica", 8))
            btn.setFixedSize(28, 22)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            tb.addWidget(btn)

        tb.addSpacing(8)
        exp_btn = QPushButton("Export PNG…")
        exp_btn.setFont(QFont("Helvetica", 8))
        exp_btn.setFixedHeight(22)
        exp_btn.clicked.connect(self._on_export)
        tb.addWidget(exp_btn)

        lay.addLayout(tb)

        # ── body row: left column | right column ──────────────────────────────
        body_row = QHBoxLayout()
        body_row.setSpacing(4)

        # ── left column: real-space image + info panel ────────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(2)

        self._fig_real = Figure(figsize=(4.8, 4.5), dpi=90)
        self._fig_real.patch.set_facecolor(bg)
        self._canvas_real = FigureCanvasQTAgg(self._fig_real)
        self._ax_real = self._fig_real.add_subplot(111)
        self._ax_real.set_facecolor(bg)
        for sp in self._ax_real.spines.values():
            sp.set_color(fg)
        self._ax_real.tick_params(colors=fg, labelsize=8)
        self._fig_real.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.13)
        left_col.addWidget(self._canvas_real, 1)

        info_frame = QFrame()
        info_frame.setFixedHeight(190)
        info_lay = QVBoxLayout(info_frame)
        info_lay.setContentsMargins(8, 6, 8, 4)
        info_lay.setSpacing(2)
        self._info_lbl = QLabel("")
        self._info_lbl.setFont(QFont("Courier", 8))
        self._info_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_lay.addWidget(self._info_lbl)
        info_lay.addStretch(1)
        left_col.addWidget(info_frame)

        body_row.addLayout(left_col, 1)

        # ── right column: FFT image + tab panel ───────────────────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(2)

        self._fig_fft = Figure(figsize=(4.8, 4.5), dpi=90)
        self._fig_fft.patch.set_facecolor(bg)
        self._canvas_fft = FigureCanvasQTAgg(self._fig_fft)
        self._ax_fft = self._fig_fft.add_subplot(111)
        self._ax_fft.set_facecolor(bg)
        for sp in self._ax_fft.spines.values():
            sp.set_color(fg)
        self._ax_fft.tick_params(colors=fg, labelsize=8)
        self._fig_fft.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.13)
        right_col.addWidget(self._canvas_fft, 1)

        # tab panel (fixed height, aligned with info_frame)
        self._tab_widget = QTabWidget()
        self._tab_widget.setFixedHeight(190)
        self._tab_widget.setFont(QFont("Helvetica", 8))

        # ── Intensity tab ─────────────────────────────────────────────────────
        intensity_tab = QWidget()
        int_lay = QVBoxLayout(intensity_tab)
        int_lay.setSpacing(2)
        int_lay.setContentsMargins(4, 2, 4, 2)

        self._hist_fig = Figure(figsize=(1, 1), dpi=90)
        self._hist_fig.patch.set_facecolor(bg)
        self._hist_canvas = FigureCanvasQTAgg(self._hist_fig)
        self._hist_canvas.setFixedHeight(95)
        self._hist_ax = self._hist_fig.add_axes([0.05, 0.18, 0.92, 0.76])
        self._hist_ax.set_facecolor(bg)
        int_lay.addWidget(self._hist_canvas)

        def _ctrl_row(label_text, slider_val):
            row = QHBoxLayout()
            row.setSpacing(4)
            lbl = QLabel(label_text)
            lbl.setFont(QFont("Helvetica", 8))
            lbl.setFixedWidth(24)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 1000)
            sl.setValue(slider_val)
            sl.setFixedHeight(14)
            val_edit = QLineEdit("0")
            val_edit.setFont(QFont("Helvetica", 8))
            val_edit.setFixedWidth(78)
            val_edit.setFixedHeight(18)
            val_edit.setAlignment(Qt.AlignRight)
            row.addWidget(lbl)
            row.addWidget(sl, 1)
            row.addWidget(val_edit)
            return row, sl, val_edit

        min_row, self._vmin_slider, self._vmin_edit = _ctrl_row("Min:", 0)
        self._vmin_slider.valueChanged.connect(self._on_vmin_slider_changed)
        self._vmin_edit.editingFinished.connect(self._on_vmin_edit_changed)
        int_lay.addLayout(min_row)

        max_row, self._vmax_slider, self._vmax_edit = _ctrl_row("Max:", 1000)
        self._vmax_slider.valueChanged.connect(self._on_vmax_slider_changed)
        self._vmax_edit.editingFinished.connect(self._on_vmax_edit_changed)
        int_lay.addLayout(max_row)

        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        reset_intensity_btn = QPushButton("Reset")
        reset_intensity_btn.setFont(QFont("Helvetica", 8))
        reset_intensity_btn.setFixedSize(55, 18)
        reset_intensity_btn.setToolTip("Reset intensity to full range")
        reset_intensity_btn.clicked.connect(self._reset_intensity)
        reset_row.addWidget(reset_intensity_btn)
        int_lay.addLayout(reset_row)

        self._tab_widget.addTab(intensity_tab, "Intensity")

        # ── Radial profile tab ────────────────────────────────────────────────
        radial_tab = QWidget()
        rad_lay = QVBoxLayout(radial_tab)
        rad_lay.setSpacing(0)
        rad_lay.setContentsMargins(2, 2, 2, 2)

        self._radial_fig = Figure(figsize=(1, 1), dpi=90)
        self._radial_fig.patch.set_facecolor(bg)
        self._radial_canvas = FigureCanvasQTAgg(self._radial_fig)
        self._radial_ax = self._radial_fig.add_axes([0.11, 0.24, 0.85, 0.66])
        self._radial_ax.set_facecolor(bg)
        rad_lay.addWidget(self._radial_canvas)

        self._tab_widget.addTab(radial_tab, "Radial profile")

        right_col.addWidget(self._tab_widget)
        body_row.addLayout(right_col, 1)

        lay.addLayout(body_row, 1)

        # ── status bar ────────────────────────────────────────────────────────
        self._status_lbl = QLabel("")
        self._status_lbl.setFont(QFont("Helvetica", 7))
        self._status_lbl.setAlignment(Qt.AlignLeft)
        lay.addWidget(self._status_lbl)

        # ── canvas event connections ──────────────────────────────────────────
        self._canvas_fft.mpl_connect("scroll_event",         self._on_scroll)
        self._canvas_fft.mpl_connect("button_press_event",   self._on_press)
        self._canvas_fft.mpl_connect("button_release_event", self._on_release)
        self._canvas_fft.mpl_connect("motion_notify_event",  self._on_motion)
        self._canvas_real.mpl_connect("motion_notify_event", self._on_motion)

    # ── FFT computation ────────────────────────────────────────────────────────

    @staticmethod
    def _tukey_1d(N: int, alpha: float = 0.5) -> np.ndarray:
        w = np.ones(N, dtype=np.float64)
        ramp = int(alpha * N / 2)
        if ramp > 0:
            t = np.linspace(0, 1, ramp)
            w[:ramp] = 0.5 * (1 - np.cos(np.pi * t))
            w[N - ramp:] = w[:ramp][::-1]
        return w

    def _make_window(self) -> np.ndarray:
        Ny, Nx = self._arr.shape
        if self._window_mode == "hann":
            return np.outer(np.hanning(Ny), np.hanning(Nx))
        if self._window_mode == "tukey":
            return np.outer(self._tukey_1d(Ny), self._tukey_1d(Nx))
        return np.ones((Ny, Nx), dtype=np.float64)

    def _recompute_fft(self):
        arr = self._arr.copy()
        finite = arr[np.isfinite(arr)]
        arr[~np.isfinite(arr)] = float(np.nanmedian(finite)) if finite.size > 0 else 0.0
        arr -= arr.mean()
        arr *= self._make_window()

        F = np.fft.fftshift(np.fft.fft2(arr))
        self._fft_mag = np.abs(F)

        Ny, Nx = arr.shape
        try:
            w_nm = float(self._scan_range_m[0]) * 1e9
            h_nm = float(self._scan_range_m[1]) * 1e9
            dx_nm = w_nm / Nx if Nx > 0 else 1.0
            dy_nm = h_nm / Ny if Ny > 0 else 1.0
        except (TypeError, ValueError, IndexError):
            dx_nm, dy_nm = 1.0, 1.0

        self._qx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx_nm))
        self._qy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy_nm))
        self._fft_xlim = (float(self._qx[0]),  float(self._qx[-1]))
        self._fft_ylim = (float(self._qy[-1]), float(self._qy[0]))

    def _compute_display_fft(self) -> np.ndarray:
        mag = self._fft_mag.copy()
        Ny, Nx = mag.shape
        cy, cx = Ny // 2, Nx // 2
        r = max(1, min(Ny, Nx) // 60)
        y0, y1 = max(0, cy - r), min(Ny, cy + r + 1)
        x0, x1 = max(0, cx - r), min(Nx, cx + r + 1)
        if self._dc_mode == "zero":
            mag[y0:y1, x0:x1] = 0.0
        elif self._dc_mode == "mask":
            mag = mag.astype(np.float64)
            mag[y0:y1, x0:x1] = np.nan
        if self._scale_mode == "log":
            mag = np.log1p(mag)
        finite = mag[np.isfinite(mag)]
        self._disp_range = (
            (float(finite.min()), float(finite.max())) if finite.size > 0 else (0.0, 1.0)
        )
        return mag

    # ── drawing ────────────────────────────────────────────────────────────────

    def _redraw(self):
        bg = self._theme.get("bg", "#1e1e1e")
        fg = self._theme.get("fg", "#dddddd")

        ax = self._ax_real
        ax.cla()
        ax.set_facecolor(bg)
        try:
            w_nm = float(self._scan_range_m[0]) * 1e9
            h_nm = float(self._scan_range_m[1]) * 1e9
        except Exception:
            w_nm, h_nm = float(self._arr.shape[1]), float(self._arr.shape[0])
        ax.imshow(
            self._arr, cmap=self._colormap, origin="upper",
            extent=[0, w_nm, h_nm, 0], aspect="equal",
        )
        ax.set_title("Real space", fontsize=8, color=fg)
        ax.set_xlabel("nm", fontsize=7, color=fg)
        ax.set_ylabel("nm", fontsize=7, color=fg)
        ax.tick_params(colors=fg, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(fg)

        ax = self._ax_fft
        ax.cla()
        ax.set_facecolor(bg)
        disp = self._compute_display_fft()
        lo, hi = self._disp_range
        vmin_val = lo + self._vmin_frac * (hi - lo)
        vmax_val = lo + self._vmax_frac * (hi - lo)
        extent_q = [
            float(self._qx[0]), float(self._qx[-1]),
            float(self._qy[-1]), float(self._qy[0]),
        ]
        ax.imshow(
            disp, cmap="inferno", origin="upper",
            extent=extent_q, aspect="equal",
            vmin=vmin_val, vmax=vmax_val,
        )
        ax.set_xlim(*self._fft_xlim)
        ax.set_ylim(*self._fft_ylim)
        scale_lbl = "log|FFT|" if self._scale_mode == "log" else "|FFT|"
        ax.set_title(f"FFT  ({scale_lbl})", fontsize=8, color=fg)
        ax.set_xlabel("q_x  (nm⁻¹)", fontsize=7, color=fg)
        ax.set_ylabel("q_y  (nm⁻¹)", fontsize=7, color=fg)
        ax.tick_params(colors=fg, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(fg)
        ax.axhline(0, color=fg, lw=0.4, alpha=0.35)
        ax.axvline(0, color=fg, lw=0.4, alpha=0.35)

        self._update_histogram(disp)
        self._update_radial_profile(disp)
        self._sync_intensity_labels()
        self._canvas_real.draw_idle()
        self._canvas_fft.draw_idle()

    # ── zoom / pan ─────────────────────────────────────────────────────────────

    def _zoom_fit(self):
        self._fft_xlim = (float(self._qx[0]),  float(self._qx[-1]))
        self._fft_ylim = (float(self._qy[-1]), float(self._qy[0]))
        self._ax_fft.set_xlim(*self._fft_xlim)
        self._ax_fft.set_ylim(*self._fft_ylim)
        self._canvas_fft.draw_idle()

    def _zoom_centre(self):
        qx_half = (float(self._qx[-1]) - float(self._qx[0])) * 0.25
        qy_half = (float(self._qy[-1]) - float(self._qy[0])) * 0.25
        self._fft_xlim = (-qx_half,  qx_half)
        self._fft_ylim = (-qy_half, qy_half)
        self._ax_fft.set_xlim(*self._fft_xlim)
        self._ax_fft.set_ylim(*self._fft_ylim)
        self._canvas_fft.draw_idle()

    def _zoom_by(self, factor: float, cx: float | None = None, cy: float | None = None):
        xl, xr = self._fft_xlim
        yb, yt = self._fft_ylim
        xc = cx if cx is not None else (xl + xr) / 2
        yc = cy if cy is not None else (yb + yt) / 2
        self._fft_xlim = (xc + (xl - xc) * factor, xc + (xr - xc) * factor)
        self._fft_ylim = (yc + (yb - yc) * factor, yc + (yt - yc) * factor)
        self._ax_fft.set_xlim(*self._fft_xlim)
        self._ax_fft.set_ylim(*self._fft_ylim)
        self._canvas_fft.draw_idle()

    # ── canvas events ──────────────────────────────────────────────────────────

    def _on_scroll(self, event):
        if event.inaxes is not self._ax_fft:
            return
        factor = 0.65 if event.step > 0 else 1.0 / 0.65
        self._zoom_by(factor, event.xdata, event.ydata)

    def _on_press(self, event):
        if event.inaxes is self._ax_fft and event.button == 1:
            self._pan_anchor = (
                event.xdata, event.ydata,
                self._fft_xlim, self._fft_ylim,
            )

    def _on_release(self, event):
        self._pan_anchor = None

    def _on_motion(self, event):
        if self._pan_anchor is not None and event.inaxes is self._ax_fft:
            x0, y0, xlim0, ylim0 = self._pan_anchor
            dx = x0 - event.xdata
            dy = y0 - event.ydata
            self._fft_xlim = (xlim0[0] + dx, xlim0[1] + dx)
            self._fft_ylim = (ylim0[0] + dy, ylim0[1] + dy)
            self._ax_fft.set_xlim(*self._fft_xlim)
            self._ax_fft.set_ylim(*self._fft_ylim)
            self._canvas_fft.draw_idle()

        if event.inaxes is self._ax_fft and event.xdata is not None:
            qx, qy = event.xdata, event.ydata
            q = np.hypot(qx, qy)
            if q > 0:
                d_nm = 1.0 / q
                d_str = f"{d_nm:.2f} nm" if d_nm >= 1.0 else f"{d_nm * 10:.2f} Å"
            else:
                d_str = "∞"
            theta = np.degrees(np.arctan2(qy, qx))
            self._status_lbl.setText(
                f"q_x={qx:+.3f}  q_y={qy:+.3f}  |q|={q:.3f} nm⁻¹  "
                f"d={d_str}  θ={theta:.1f}°"
            )
        elif event.inaxes is self._ax_real and event.xdata is not None:
            self._status_lbl.setText(
                f"x={event.xdata:.2f} nm  y={event.ydata:.2f} nm"
            )
        else:
            self._status_lbl.setText("")

    # ── toolbar callbacks ──────────────────────────────────────────────────────

    def _on_scale_changed(self, idx: int):
        self._scale_mode = "log" if idx == 0 else "linear"
        self._redraw()

    def _on_window_changed(self, idx: int):
        self._window_mode = ["hann", "none", "tukey"][idx]
        self._recompute_fft()
        self._redraw()

    def _on_dc_changed(self, idx: int):
        self._dc_mode = ["zero", "keep", "mask"][idx]
        self._redraw()

    def _update_histogram(self, disp: np.ndarray):
        fg = self._theme.get("fg", "#dddddd")
        bg = self._theme.get("bg", "#1e1e1e")
        ax = self._hist_ax
        ax.cla()
        ax.set_facecolor(bg)
        finite = disp[np.isfinite(disp)]
        if finite.size > 0:
            lo, hi = self._disp_range
            ax.hist(finite.ravel(), bins=256, range=(lo, hi),
                    color="#6699bb", alpha=0.9, linewidth=0, log=True)
            vmin_val = lo + self._vmin_frac * (hi - lo)
            vmax_val = lo + self._vmax_frac * (hi - lo)
            ylo, yhi = ax.get_ylim()
            ax.axvline(vmin_val, color="#ff6060", lw=1.5, zorder=5)
            ax.axvline(vmax_val, color="#50ee70", lw=1.5, zorder=5)
            ax.set_xlim(lo, hi)
            ax.set_ylim(ylo, yhi)
        ax.set_yticks([])
        ax.tick_params(colors=fg, labelsize=6, length=2)
        for sp in ax.spines.values():
            sp.set_color(fg)
        self._hist_fig.tight_layout(pad=0.15)
        self._hist_canvas.draw_idle()

    def _sync_intensity_labels(self):
        lo, hi = self._disp_range
        vmin_val = lo + self._vmin_frac * (hi - lo)
        vmax_val = lo + self._vmax_frac * (hi - lo)
        self._vmin_edit.setText(f"{vmin_val:.4g}")
        self._vmax_edit.setText(f"{vmax_val:.4g}")

    def _on_vmin_slider_changed(self, val: int):
        new_frac = val / 1000.0
        if new_frac > self._vmax_frac:
            new_frac = self._vmax_frac
            self._vmin_slider.blockSignals(True)
            self._vmin_slider.setValue(int(self._vmax_frac * 1000))
            self._vmin_slider.blockSignals(False)
        self._vmin_frac = new_frac
        self._redraw()

    def _on_vmax_slider_changed(self, val: int):
        new_frac = val / 1000.0
        if new_frac < self._vmin_frac:
            new_frac = self._vmin_frac
            self._vmax_slider.blockSignals(True)
            self._vmax_slider.setValue(int(self._vmin_frac * 1000))
            self._vmax_slider.blockSignals(False)
        self._vmax_frac = new_frac
        self._redraw()

    def _on_vmin_edit_changed(self):
        lo, hi = self._disp_range
        try:
            val = float(self._vmin_edit.text())
            frac = (val - lo) / (hi - lo) if hi > lo else 0.0
            self._vmin_frac = max(0.0, min(frac, self._vmax_frac))
        except ValueError:
            pass
        self._vmin_slider.blockSignals(True)
        self._vmin_slider.setValue(int(self._vmin_frac * 1000))
        self._vmin_slider.blockSignals(False)
        self._redraw()

    def _on_vmax_edit_changed(self):
        lo, hi = self._disp_range
        try:
            val = float(self._vmax_edit.text())
            frac = (val - lo) / (hi - lo) if hi > lo else 1.0
            self._vmax_frac = max(self._vmin_frac, min(frac, 1.0))
        except ValueError:
            pass
        self._vmax_slider.blockSignals(True)
        self._vmax_slider.setValue(int(self._vmax_frac * 1000))
        self._vmax_slider.blockSignals(False)
        self._redraw()

    def _reset_intensity(self):
        self._vmin_frac = 0.0
        self._vmax_frac = 1.0
        self._vmin_slider.blockSignals(True)
        self._vmax_slider.blockSignals(True)
        self._vmin_slider.setValue(0)
        self._vmax_slider.setValue(1000)
        self._vmin_slider.blockSignals(False)
        self._vmax_slider.blockSignals(False)
        self._redraw()

    def _update_info_panel(self):
        Ny, Nx = self._arr.shape
        try:
            w_nm = float(self._scan_range_m[0]) * 1e9
            h_nm = float(self._scan_range_m[1]) * 1e9
            dx_nm = w_nm / Nx if Nx > 0 else 1.0
            dy_nm = h_nm / Ny if Ny > 0 else 1.0
        except Exception:
            w_nm = h_nm = float(Nx)
            dx_nm = dy_nm = 1.0
        dqx = 1.0 / w_nm if w_nm > 0 else 0.0
        q_ny = 1.0 / (2.0 * dx_nm) if dx_nm > 0 else 0.0
        self._info_lbl.setText(
            f"Image:    {Nx} × {Ny} px\n"
            f"Size:     {w_nm:.3g} × {h_nm:.3g} nm\n"
            f"px size:  {dx_nm:.4g} nm\n"
            f"q-res:    {dqx:.4g} nm⁻¹\n"
            f"Nyquist:  {q_ny:.3g} nm⁻¹"
        )

    def _update_radial_profile(self, disp: np.ndarray):
        fg = self._theme.get("fg", "#dddddd")
        bg = self._theme.get("bg", "#1e1e1e")
        ax = self._radial_ax
        ax.cla()
        ax.set_facecolor(bg)
        if self._qx is None or self._qy is None:
            self._radial_canvas.draw_idle()
            return
        qx_2d = self._qx[np.newaxis, :]
        qy_2d = self._qy[:, np.newaxis]
        q_map = np.sqrt(qx_2d ** 2 + qy_2d ** 2)
        Ny, Nx = disp.shape
        n_bins = min(Nx, Ny) // 2
        q_max = float(q_map.max())
        q_bins = np.linspace(0.0, q_max, n_bins + 1)
        q_centers = 0.5 * (q_bins[:-1] + q_bins[1:])
        flat_q = q_map.ravel()
        flat_d = disp.ravel()
        valid = np.isfinite(flat_d)
        idx = np.clip(np.digitize(flat_q, q_bins) - 1, 0, n_bins - 1)
        profile = np.zeros(n_bins)
        counts  = np.zeros(n_bins, dtype=np.int64)
        np.add.at(profile, idx[valid], flat_d[valid])
        np.add.at(counts,  idx[valid], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            profile = np.where(counts > 0, profile / counts, np.nan)
        good = np.isfinite(profile)
        if good.any():
            ax.plot(q_centers[good], profile[good], color="#88bbee", lw=1.0)
        scale_lbl = "log|FFT|" if self._scale_mode == "log" else "|FFT|"
        ax.set_xlabel("q  (nm⁻¹)", fontsize=7, color=fg)
        ax.set_ylabel(f"⟨{scale_lbl}⟩", fontsize=7, color=fg)
        ax.tick_params(colors=fg, labelsize=6, length=2)
        for sp in ax.spines.values():
            sp.set_color(fg)
        self._radial_fig.tight_layout(pad=0.3)
        self._radial_canvas.draw_idle()

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export FFT view", "fft_view.png",
            "PNG image (*.png);;All files (*)"
        )
        if path:
            self._fig_fft.savefig(path, dpi=150, bbox_inches="tight")


class PeriodicFilterDialog(QDialog):
    """Interactive centred-FFT peak picker for periodic notch filtering."""

    def __init__(self, arr: np.ndarray, peaks=None, radius_px: float = 3.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Periodic FFT filter")
        self.resize(680, 620)
        self._arr = arr.astype(np.float64, copy=True)
        self._peaks: list[tuple[int, int]] = [
            (int(p[0]), int(p[1])) for p in (peaks or [])
        ]

        lay = QVBoxLayout(self)
        help_lbl = QLabel(
            "Click bright periodic peaks in the FFT power spectrum. "
            "Each click suppresses that peak and its conjugate in the processed image."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setFont(QFont("Helvetica", 9))
        lay.addWidget(help_lbl)

        self._fig = Figure(figsize=(6.0, 5.0), dpi=90)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(111)
        lay.addWidget(self._canvas, 1)

        radius_row = QHBoxLayout()
        radius_lbl = QLabel("Notch radius:")
        radius_lbl.setFont(QFont("Helvetica", 8))
        self._radius_sl = QSlider(Qt.Horizontal)
        self._radius_sl.setRange(1, 20)
        self._radius_sl.setValue(max(1, min(20, int(round(radius_px)))))
        self._radius_val = QLabel(f"{self._radius_sl.value()} px")
        self._radius_val.setFont(QFont("Helvetica", 8))
        self._radius_sl.valueChanged.connect(
            lambda v: self._radius_val.setText(f"{v} px"))
        radius_row.addWidget(radius_lbl)
        radius_row.addWidget(self._radius_sl, 1)
        radius_row.addWidget(self._radius_val)
        lay.addLayout(radius_row)

        self._selected_lbl = QLabel("")
        self._selected_lbl.setWordWrap(True)
        self._selected_lbl.setFont(QFont("Helvetica", 8))
        lay.addWidget(self._selected_lbl)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear peaks")
        clear_btn.clicked.connect(self._clear)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("Use selected peaks")
        apply_btn.setObjectName("accentBtn")
        apply_btn.clicked.connect(self.accept)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(apply_btn)
        lay.addLayout(btn_row)

        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._draw()

    def selected_peaks(self) -> list[tuple[int, int]]:
        return list(self._peaks)

    def radius_px(self) -> float:
        return float(self._radius_sl.value())

    def _spectrum(self) -> np.ndarray:
        a = self._arr
        nan_mask = ~np.isfinite(a)
        fill = float(np.nanmean(a)) if (~nan_mask).any() else 0.0
        centered = np.where(nan_mask, fill, a) - fill
        win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
        F = np.fft.fftshift(np.fft.fft2(centered * win))
        return np.log1p(np.abs(F) ** 2)

    def _draw(self):
        power = self._spectrum()
        Ny, Nx = power.shape
        cx, cy = Nx // 2, Ny // 2
        self._ax.clear()
        self._ax.imshow(power, cmap="magma", origin="upper")
        self._ax.set_title("FFT power spectrum")
        self._ax.set_xlabel("kx")
        self._ax.set_ylabel("ky")
        self._ax.axvline(cx, color="white", alpha=0.25, linewidth=0.8)
        self._ax.axhline(cy, color="white", alpha=0.25, linewidth=0.8)
        for dx, dy in self._peaks:
            for sx, sy in ((dx, dy), (-dx, -dy)):
                self._ax.plot(cx + sx, cy + sy, "o", color="#89b4fa",
                              markerfacecolor="none", markersize=9, markeredgewidth=1.8)
        self._fig.tight_layout()
        self._canvas.draw_idle()
        if self._peaks:
            text = ", ".join(f"({dx:+d}, {dy:+d})" for dx, dy in self._peaks)
            self._selected_lbl.setText(f"Selected peaks: {text}")
        else:
            self._selected_lbl.setText("Selected peaks: none")

    def _on_click(self, event):
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        Ny, Nx = self._arr.shape
        cx, cy = Nx // 2, Ny // 2
        dx = int(round(event.xdata)) - cx
        dy = int(round(event.ydata)) - cy
        if dx == 0 and dy == 0:
            return
        canonical = (dx, dy)
        conjugate = (-dx, -dy)
        if conjugate in self._peaks:
            canonical = conjugate
        if canonical in self._peaks:
            self._peaks.remove(canonical)
        else:
            self._peaks.append(canonical)
        self._draw()

    def _clear(self):
        self._peaks.clear()
        self._draw()


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
        # Mutable mapping shared with the parent window: spec_stem → image_stem.
        # Empty dict by default — markers only appear after explicit mapping.
        self._spec_image_map = spec_image_map if spec_image_map is not None else {}
        self._raw_arr: Optional[np.ndarray] = None
        self._display_arr: Optional[np.ndarray] = None  # raw or processed, for histogram/export
        self._spec_markers: list[dict] = []
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
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(False)
        self._scroll_area.setAlignment(Qt.AlignCenter)
        self._zoom_lbl = _ZoomLabel()
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

        zoom_hint = QLabel("Ctrl+scroll to zoom")
        zoom_hint.setFont(QFont("Helvetica", 8))
        toolbar.addWidget(zoom_hint)
        toolbar.addStretch()
        left_lay.addLayout(toolbar)

        selection_bar = QHBoxLayout()
        selection_bar.setSpacing(4)
        selection_lbl = QLabel("Selection")
        selection_lbl.setFont(QFont("Helvetica", 8, QFont.Bold))
        selection_bar.addWidget(selection_lbl)
        self._selection_group = QButtonGroup(self)
        self._selection_group.setExclusive(True)
        for key, label, tip in (
            ("none", "Pointer", "Pointer / no selection tool"),
            ("rectangle", "Rect.", "Rectangular area selection"),
            ("ellipse", "Ellipse", "Elliptical area selection"),
            ("polygon", "Polygon", "Polygon area selection; double-click to finish"),
            ("line", "Line", "Line selection for display/status only"),
        ):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(24)
            btn.setMinimumWidth(44)
            btn.setFont(QFont("Helvetica", 8))
            btn.setToolTip(tip)
            self._selection_group.addButton(btn)
            btn.setProperty("selection_tool", key)
            if key == "none":
                btn.setChecked(True)
            selection_bar.addWidget(btn)
        self._selection_group.buttonClicked.connect(self._on_selection_tool_clicked)
        clear_selection_btn = QPushButton("Clear")
        clear_selection_btn.setFont(QFont("Helvetica", 8))
        clear_selection_btn.setFixedHeight(24)
        clear_selection_btn.clicked.connect(self._on_clear_roi)
        selection_bar.addWidget(clear_selection_btn)
        selection_bar.addStretch()
        left_lay.addLayout(selection_bar)

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
        right_scroll.setMinimumWidth(240)
        right_scroll.setMaximumWidth(480)

        right = QWidget()
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
        self._canvas.setFixedHeight(160)
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
        self._line_profile_panel.export_csv_clicked.connect(self._on_export_line_profile_csv)

        self._status_lbl = QLabel("")
        self._status_lbl.setFont(QFont("Helvetica", 8))
        self._status_lbl.setWordWrap(True)
        right_lay.addWidget(self._status_lbl)

        right_lay.addStretch()

        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setSizes([880, 300])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setCollapsible(1, False)
        root.addWidget(splitter, 1)

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

    # ── Navigation ─────────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        k = event.key()
        if k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            if self._nudge_line_profile(k):
                event.accept()
                return
        if k in (Qt.Key_Escape, Qt.Key_Return):
            self.accept()
        elif k == Qt.Key_Left:
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
                self._display_arr = _apply_processing(self._raw_arr, self._processing)
            except Exception:
                self._display_arr = self._raw_arr
        else:
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
        arr = self._raw_arr if self._raw_arr is not None else self._display_arr
        return None if arr is None else arr.shape

    def _set_selection_tool(self, kind: str) -> None:
        kind = str(kind or "none")
        for btn in self._selection_group.buttons():
            if btn.property("selection_tool") == kind:
                btn.setChecked(True)
                break
        self._zoom_lbl.set_selection_tool(kind)
        self._sync_line_profile_visibility(kind)

    def _on_selection_tool_clicked(self, button) -> None:
        if self._set_zero_plane_btn.isChecked():
            self._set_zero_plane_btn.setChecked(False)
        kind = button.property("selection_tool") or "none"
        self._zoom_lbl.set_selection_tool(kind)
        self._sync_line_profile_visibility(kind)

    def _sync_line_profile_visibility(self, kind: str | None = None) -> None:
        if not hasattr(self, "_line_profile_panel"):
            return
        is_line = (kind or self._zoom_lbl.selection_tool()) == "line"
        self._line_profile_panel.setVisible(is_line)
        if is_line:
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
                fh.write(f"# ProbeFlow histogram export\n")
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
        pix = self._zoom_lbl.pixmap()
        if pix is not None and not pix.isNull():
            pw, ph = pix.width(), pix.height()
        else:
            pw = ph = 0
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
        # The signal carries width only — read height off the pixmap directly.
        pix = self._zoom_lbl.pixmap()
        new_h = pix.height() if pix is not None and not pix.isNull() else 0
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
        wants_filter_roi = self._scope_cb.currentIndex() == 1
        wants_bg_fit_roi = self._bg_fit_roi_cb.isChecked()
        wants_patch_roi = self._patch_roi_cb.isChecked()
        selection_geometry = self._area_selection_geometry_px()
        if wants_filter_roi or wants_bg_fit_roi or wants_patch_roi:
            if self._selection_geometry and self._selection_geometry.get("kind") == "line":
                self._status_lbl.setText(
                    "Line selections are display-only; choose an area selection for processing."
                )
                return
            if selection_geometry is None:
                self._status_lbl.setText("Select an area before using selection-based processing.")
                return
        preserve = {
            key: self._processing[key]
            for key in (
                "set_zero_xy",
                "set_zero_plane_points",
                "set_zero_patch",
                "periodic_notches",
                "periodic_notch_radius",
            )
            if key in self._processing
        }
        self._processing = self._processing_panel.state()
        self._processing.update(self._advanced_processing_state())
        self._processing.update(preserve)
        if wants_filter_roi:
            self._processing["processing_scope"] = "roi"
            self._processing["roi_geometry"] = dict(selection_geometry)
            if selection_geometry.get("kind") == "rectangle":
                self._processing["roi_rect"] = selection_geometry.get("rect_px")
            else:
                self._processing.pop("roi_rect", None)
        else:
            self._processing.pop("processing_scope", None)
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
        self._refresh_processing_display()

    def _on_reset_processing(self):
        """Clear all processing for the current image and reload raw data."""
        has_selection = self._selection_geometry is not None or self._roi_rect_px is not None
        has_zero = bool(self._zero_plane_points_px)
        if not self._processing and not has_selection and not has_zero:
            self._status_lbl.setText("Already showing the original — nothing to reset.")
            return
        self._processing = {}
        self._processing_panel.set_state({})
        self._set_advanced_processing_state({})
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

    def _on_save_png(self):
        entry = self._entries[self._idx]
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
        a_png = QAction("⬇ Save PNG copy…", self)
        a_png.triggered.connect(self._on_save_png)
        menu.addAction(a_png)
        prof = self._line_profile_panel.profile_data()
        if prof is not None:
            a_csv = QAction("Export line profile as CSV…", self)
            a_csv.triggered.connect(self._on_export_line_profile_csv)
            menu.addAction(a_csv)
        menu.exec(pos)

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
class SpecMappingDialog(QDialog):
    """Folder-level spec→image mapping editor.

    Lists every loaded .VERT spec file with a dropdown of every loaded
    .sxm image in the same folder (plus a leading "(none)" entry). The
    user picks the parent image for each spectrum; the result is returned
    as a ``dict[spec_stem, image_stem]`` containing only the assigned
    rows. Unassigned spectra are simply omitted from the result.

    A "Suggest all" button populates dropdowns by reading each scan's
    physical extent (offset, range, angle) and picking the smallest
    image whose scan-frame contains the spec's recorded coordinates.
    The suggestion is just a starting point — the user can change any
    row before accepting.
    """

    NONE_LABEL = "(none)"

    def __init__(self, sxm_entries: list, vert_entries: list,
                 current_map: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Map spectra to images")
        self.resize(720, 520)
        self._sxm_entries = list(sxm_entries)
        self._vert_entries = list(vert_entries)
        self._current = dict(current_map or {})
        self._combos: dict[str, QComboBox] = {}
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        hdr = QLabel(
            "Pick the parent image for each spectrum. Unassigned spectra "
            "show no marker on any image.")
        hdr.setFont(QFont("Helvetica", 10))
        hdr.setWordWrap(True)
        v.addWidget(hdr)

        if not self._vert_entries:
            v.addWidget(QLabel("No spectroscopy files in the current folder."))
        else:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            inner = QWidget()
            grid = QGridLayout(inner)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(4)
            grid.addWidget(QLabel("Spectrum"), 0, 0)
            grid.addWidget(QLabel("Parent image"), 0, 1)

            image_options = [self.NONE_LABEL] + [e.stem for e in self._sxm_entries]
            for i, vert in enumerate(self._vert_entries, start=1):
                grid.addWidget(QLabel(vert.stem), i, 0)
                cb = QComboBox()
                cb.addItems(image_options)
                cur = self._current.get(vert.stem)
                if cur and cur in image_options:
                    cb.setCurrentText(cur)
                else:
                    cb.setCurrentText(self.NONE_LABEL)
                grid.addWidget(cb, i, 1)
                self._combos[vert.stem] = cb
            grid.setColumnStretch(1, 1)
            scroll.setWidget(inner)
            v.addWidget(scroll, 1)

        # Action row
        btn_row = QHBoxLayout()
        suggest_btn = QPushButton("Suggest all (by coordinates)")
        suggest_btn.setToolTip(
            "For each spectrum, look at its recorded (x,y) position and pick "
            "the smallest loaded scan whose frame contains it. Existing "
            "selections are overwritten.")
        suggest_btn.clicked.connect(self._on_suggest)
        btn_row.addWidget(suggest_btn)
        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self._on_clear_all)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        ok_btn = QPushButton("Apply")
        ok_btn.setObjectName("accentBtn")
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        v.addLayout(btn_row)

    def _on_clear_all(self):
        for cb in self._combos.values():
            cb.setCurrentText(self.NONE_LABEL)

    def _on_suggest(self):
        """Pick the smallest containing image per spec, by coordinate."""
        # We avoid importing this at module top because it pulls in scan I/O
        # — keeping the dialog responsive on large folders matters more
        # than saving the import here.
        from probeflow.io.spectroscopy import read_spec_file
        from probeflow.analysis.spec_plot import spec_position_to_pixel, _parse_sxm_offset

        # Pre-load image headers once (slow if many files).
        scan_info = []
        for img in self._sxm_entries:
            try:
                _scan = load_scan(img.path)
                shape = _scan.planes[0].shape if _scan.planes else None
                if shape is None or _scan.scan_range_m is None:
                    continue
                hdr = _scan.header or {}
                offset_m = (0.0, 0.0)
                angle_deg = 0.0
                if img.source_format == "sxm" and hdr:
                    offset_m = _parse_sxm_offset(hdr)
                    raw = hdr.get("SCAN_ANGLE", "0").strip()
                    try:
                        angle_deg = float(raw) if raw else 0.0
                    except ValueError:
                        angle_deg = 0.0
                # "Size" used to break ties when several images contain a spec:
                # smaller scan range = better localisation = preferred.
                rng_m = _scan.scan_range_m
                area_m2 = float(rng_m[0]) * float(rng_m[1])
                scan_info.append((img.stem, shape, rng_m, offset_m, angle_deg, area_m2))
            except Exception:
                continue

        for vert in self._vert_entries:
            try:
                spec = read_spec_file(vert.path)
                x_m, y_m = spec.position
            except Exception:
                continue
            best: Optional[tuple[float, str]] = None  # (area, stem)
            for stem, shape, rng_m, offset_m, angle_deg, area in scan_info:
                hit = spec_position_to_pixel(
                    x_m, y_m,
                    scan_shape=shape,
                    scan_range_m=rng_m,
                    scan_offset_m=offset_m,
                    scan_angle_deg=angle_deg,
                )
                if hit is None:
                    continue
                if best is None or area < best[0]:
                    best = (area, stem)
            if best is not None and vert.stem in self._combos:
                self._combos[vert.stem].setCurrentText(best[1])

    def get_mapping(self) -> dict[str, str]:
        """Return the user's selection as ``{spec_stem: image_stem}``."""
        out: dict[str, str] = {}
        for spec_stem, cb in self._combos.items():
            sel = cb.currentText()
            if sel and sel != self.NONE_LABEL:
                out[spec_stem] = sel
        return out


class ViewerSpecMappingDialog(QDialog):
    """In-viewer mapping editor for ONE image.

    Lists all .VERT spec files in the same folder; the user ticks which
    spectra belong to the currently displayed image. Multiple images can
    share a parent in principle (e.g. different planes of the same scan)
    but our mapping is one-to-one, so ticking a spec here moves it from
    any prior parent to the current image.
    """

    def __init__(self, image_stem: str, vert_entries: list,
                 current_map: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Map spectra to {image_stem}")
        self.resize(420, 460)
        self._image_stem  = image_stem
        self._vert_entries = list(vert_entries)
        self._current     = dict(current_map or {})
        self._checks: dict[str, QCheckBox] = {}
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)
        hdr = QLabel(
            f"Tick the spectra to associate with <b>{self._image_stem}</b>. "
            "Ticking one that is already mapped to a different image will move it.")
        hdr.setFont(QFont("Helvetica", 9))
        hdr.setWordWrap(True)
        v.addWidget(hdr)

        if not self._vert_entries:
            v.addWidget(QLabel("No .VERT files in the current folder."))
        else:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            inner = QWidget()
            inner_lay = QVBoxLayout(inner)
            inner_lay.setContentsMargins(0, 0, 0, 0)
            inner_lay.setSpacing(2)
            for vert in self._vert_entries:
                cb = QCheckBox(vert.stem)
                cb.setChecked(self._current.get(vert.stem) == self._image_stem)
                # Annotate other-image assignments so the user knows what
                # ticking this row will displace.
                other = self._current.get(vert.stem)
                if other and other != self._image_stem:
                    cb.setText(f"{vert.stem}   (currently → {other})")
                inner_lay.addWidget(cb)
                self._checks[vert.stem] = cb
            inner_lay.addStretch()
            scroll.setWidget(inner)
            v.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("Apply")
        ok_btn.setObjectName("accentBtn")
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        v.addLayout(btn_row)

    def updated_map(self) -> dict[str, str]:
        """Return a NEW mapping dict reflecting the user's choices."""
        out = dict(self._current)
        for spec_stem, cb in self._checks.items():
            if cb.isChecked():
                out[spec_stem] = self._image_stem
            else:
                # Only clear if this row WAS pointing at the current image.
                if out.get(spec_stem) == self._image_stem:
                    out.pop(spec_stem, None)
        return out


# ── Browse tool panel (LEFT) ──────────────────────────────────────────────────
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
class SpecViewerDialog(QDialog):
    """Full-size viewer for a spectroscopy file (Createc .VERT or Nanonis .dat).

    The viewer is channel-agnostic: it builds a toggleable list from
    ``spec.channel_order`` and stacks one subplot per selected channel.
    """

    # Dark-theme colours for plot elements.
    _BG = "#1e1e2e"
    _FG = "#cdd6f4"
    # Plot curve colours, cycled across selected channels.
    _COLORS = ("#89b4fa", "#a6e3a1", "#fab387", "#f5c2e7",
               "#94e2d5", "#f9e2af", "#cba6f7", "#f38ba8")

    def __init__(self, entry: VertFile, t: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(entry.stem)
        self.setMinimumSize(900, 560)
        self.resize(1100, 640)
        self._entry = entry
        self._t = t
        self._spec = None
        self._checkboxes: dict[str, QCheckBox] = {}
        self._channel_check_widgets: list[QCheckBox] = []
        self._canvas = None
        self._fig = None
        # Unit-override choice per base SI unit. "Auto" means use
        # choose_display_unit; otherwise lookup_unit_scale picks a fixed scale.
        self._unit_choice: dict[str, str] = {"m": "Auto", "A": "Auto", "V": "Auto"}
        self._build()
        self._load()

    # ── UI construction ─────────────────────────────────────────────────

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        self._title = QLabel(self._entry.stem)
        self._title.setFont(QFont("Helvetica", 12, QFont.Bold))
        self._title.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._title)

        splitter = QSplitter(Qt.Horizontal)

        # Left panel: scrollable channel list.
        self._channels_panel = QWidget()
        self._channels_lay = QVBoxLayout(self._channels_panel)
        self._channels_lay.setContentsMargins(6, 6, 6, 6)
        self._channels_lay.setSpacing(4)
        ch_header = QLabel("Channels")
        ch_header.setFont(QFont("Helvetica", 10, QFont.Bold))
        self._channels_lay.addWidget(ch_header)

        # Unit-override selectors for height (Z) and current channels.
        unit_box = QGroupBox("Display units")
        unit_box.setFont(QFont("Helvetica", 9, QFont.Bold))
        unit_lay = QGridLayout(unit_box)
        unit_lay.setContentsMargins(6, 4, 6, 4)
        unit_lay.setSpacing(2)

        z_lbl = QLabel("Z:")
        z_lbl.setFont(QFont("Helvetica", 9))
        self._z_unit_cb = QComboBox()
        self._z_unit_cb.addItems(["Auto", "pm", "Å", "nm", "µm", "m"])
        self._z_unit_cb.setFont(QFont("Helvetica", 9))
        self._z_unit_cb.currentTextChanged.connect(
            lambda v: self._on_unit_changed("m", v))

        i_lbl = QLabel("I:")
        i_lbl.setFont(QFont("Helvetica", 9))
        self._i_unit_cb = QComboBox()
        self._i_unit_cb.addItems(["Auto", "fA", "pA", "nA", "µA", "A"])
        self._i_unit_cb.setFont(QFont("Helvetica", 9))
        self._i_unit_cb.currentTextChanged.connect(
            lambda v: self._on_unit_changed("A", v))

        v_lbl = QLabel("V:")
        v_lbl.setFont(QFont("Helvetica", 9))
        self._v_unit_cb = QComboBox()
        self._v_unit_cb.addItems(["Auto", "µV", "mV", "V"])
        self._v_unit_cb.setFont(QFont("Helvetica", 9))
        self._v_unit_cb.currentTextChanged.connect(
            lambda v: self._on_unit_changed("V", v))

        unit_lay.addWidget(z_lbl, 0, 0)
        unit_lay.addWidget(self._z_unit_cb, 0, 1)
        unit_lay.addWidget(i_lbl, 1, 0)
        unit_lay.addWidget(self._i_unit_cb, 1, 1)
        unit_lay.addWidget(v_lbl, 2, 0)
        unit_lay.addWidget(self._v_unit_cb, 2, 1)

        self._channels_lay.addWidget(unit_box)
        self._channels_lay.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(self._channels_panel)
        left_scroll.setMinimumWidth(200)
        left_scroll.setMaximumWidth(300)
        splitter.addWidget(left_scroll)

        # Right panel: plot canvas.
        self._canvas_widget = QWidget()
        self._canvas_lay = QVBoxLayout(self._canvas_widget)
        self._canvas_lay.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(self._canvas_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        lay.addWidget(splitter, 1)

        self._status = QLabel("Loading…")
        self._status.setFont(QFont("Helvetica", 9))
        lay.addWidget(self._status)

        btn_row = QHBoxLayout()
        self._raw_btn = QPushButton("Show raw data")
        self._raw_btn.setFixedWidth(140)
        self._raw_btn.clicked.connect(self._show_raw_data)
        btn_row.addWidget(self._raw_btn)

        self._export_csv_btn = QPushButton("Export CSV…")
        self._export_csv_btn.setFixedWidth(120)
        self._export_csv_btn.setToolTip(
            "Save the spectrum as a CSV file with one column per selected channel.")
        self._export_csv_btn.clicked.connect(self._export_csv)
        btn_row.addWidget(self._export_csv_btn)

        self._export_grace_btn = QPushButton("Export xmgrace…")
        self._export_grace_btn.setFixedWidth(160)
        self._export_grace_btn.setToolTip(
            "Render via xmgrace (Helvetica default). "
            "Produces three files in the chosen folder: "
            ".agr (re-editable Grace project), .png, and .pdf.")
        self._export_grace_btn.clicked.connect(self._export_xmgrace)
        btn_row.addWidget(self._export_grace_btn)

        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

    # ── Data load + channel list population ─────────────────────────────

    def _load(self) -> None:
        from probeflow.io.spectroscopy import read_spec_file

        try:
            spec = read_spec_file(self._entry.path)
        except Exception as exc:
            self._status.setText(f"Error: {exc}")
            return
        self._spec = spec

        # Pull channel_order off the spec; fall back to whatever keys are
        # present for old SpecData objects that don't carry it.
        order = list(spec.channel_order) if spec.channel_order else list(spec.channels.keys())
        defaults = set(spec.default_channels)

        # Remove only channel checkboxes from prior loads. Static controls such
        # as the unit QGroupBox own combo boxes and must survive dialog lifetime.
        for w in self._channel_check_widgets:
            self._channels_lay.removeWidget(w)
            w.deleteLater()
        self._channel_check_widgets.clear()
        self._checkboxes.clear()
        for ch in order:
            if ch not in spec.channels:
                continue
            cb = QCheckBox(self._channel_display_label(ch))
            cb.setChecked(ch in defaults)
            cb.toggled.connect(self._redraw)
            self._channels_lay.insertWidget(self._channels_lay.count() - 1, cb)
            self._checkboxes[ch] = cb
            self._channel_check_widgets.append(cb)

        sweep = spec.metadata.get("sweep_type", "").replace("_", " ")
        n_pts = spec.metadata.get("n_points", 0)
        self._status.setText(
            f"{sweep}  |  {n_pts} points  |  "
            f"pos ({spec.position[0]*1e9:.2f}, {spec.position[1]*1e9:.2f}) nm"
        )

        self._redraw()

    # ── Plotting ────────────────────────────────────────────────────────

    def _on_unit_changed(self, base: str, label: str) -> None:
        self._unit_choice[base] = label
        self._refresh_channel_labels()
        self._redraw()

    def _display_values_for_channel(self, ch: str) -> tuple[np.ndarray, str]:
        if self._spec is None or ch not in self._spec.channels:
            return np.array([], dtype=float), ""
        from probeflow.analysis.spec_plot import choose_display_unit, lookup_unit_scale

        y = np.asarray(self._spec.channels[ch], dtype=float)
        unit = self._spec.y_units.get(ch, "")
        choice = self._unit_choice.get(unit, "Auto")
        override = lookup_unit_scale(unit, choice) if choice != "Auto" else None
        if override is not None:
            scale, disp_unit = override
        else:
            scale, disp_unit = choose_display_unit(unit, y)
        return y * scale, disp_unit

    def _channel_display_label(self, ch: str) -> str:
        _values, disp_unit = self._display_values_for_channel(ch)
        label = self._channel_label(ch)
        return f"{label}  ({disp_unit})" if disp_unit else label

    def _channel_label(self, ch: str) -> str:
        if self._spec is None:
            return ch
        info = getattr(self._spec, "channel_info", {}).get(ch)
        return getattr(info, "display_label", ch) if info is not None else ch

    def _refresh_channel_labels(self) -> None:
        for ch, cb in self._checkboxes.items():
            cb.setText(self._channel_display_label(ch))

    def _redraw(self) -> None:
        if self._spec is None:
            return
        # Remove existing canvas if present.
        if self._canvas is not None:
            self._canvas_lay.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas = None
            self._fig = None

        selected = [ch for ch, cb in self._checkboxes.items() if cb.isChecked()]
        if not selected:
            # Empty figure keeps the area from collapsing.
            fig = Figure(figsize=(8.5, 4.5), tight_layout=True)
            fig.patch.set_facecolor(self._BG)
            canvas = FigureCanvasQTAgg(fig)
            self._canvas_lay.addWidget(canvas)
            self._canvas = canvas
            self._fig = fig
            return

        fig = Figure(figsize=(8.5, 4.5), tight_layout=True)
        fig.patch.set_facecolor(self._BG)
        axes = fig.subplots(nrows=len(selected), ncols=1, sharex=True)
        if len(selected) == 1:
            axes = [axes]

        spec = self._spec
        for i, (ch, ax) in enumerate(zip(selected, axes)):
            y_disp, disp_unit = self._display_values_for_channel(ch)

            ax.set_facecolor(self._BG)
            ax.plot(spec.x_array, y_disp, linewidth=1.0,
                    color=self._COLORS[i % len(self._COLORS)])
            label = self._channel_label(ch)
            ax.set_ylabel(f"{label} ({disp_unit})" if disp_unit else label,
                          color=self._FG, fontsize=8)
            ax.tick_params(colors=self._FG, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(self._FG)
                spine.set_linewidth(0.6)
            if i < len(selected) - 1:
                ax.tick_params(axis="x", labelbottom=False)

        axes[-1].set_xlabel(spec.x_label, color=self._FG, fontsize=8)

        canvas = FigureCanvasQTAgg(fig)
        self._canvas_lay.addWidget(canvas)
        self._canvas = canvas
        self._fig = fig

    # ── Export ──────────────────────────────────────────────────────────

    def _selected_channels_in_display_units(self):
        """Yield (ch, y_disp, disp_unit) for each ticked channel.

        Applies the user's unit override (or auto-pick) so the exported
        values match what's currently shown in the plot.
        """
        spec = self._spec
        if spec is None:
            return
        for ch, cb in self._checkboxes.items():
            if not cb.isChecked() or ch not in spec.channels:
                continue
            y_disp, disp_unit = self._display_values_for_channel(ch)
            yield ch, self._channel_label(ch), y_disp, disp_unit

    def _export_csv(self) -> None:
        if self._spec is None:
            self._status.setText("Nothing to export — spectrum failed to load.")
            return
        rows = list(self._selected_channels_in_display_units())
        if not rows:
            self._status.setText("Tick at least one channel before exporting.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export spectrum CSV",
            str(Path.home() / f"{self._entry.stem}.csv"),
            "CSV files (*.csv)")
        if not out_path:
            return
        try:
            import csv
            x = self._spec.x_array
            with open(out_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow([self._spec.x_label]
                           + [f"{label} ({u})" if u else label
                              for _ch, label, _y, u in rows])
                for i in range(len(x)):
                    w.writerow([f"{x[i]:.10g}"]
                               + [f"{y[i]:.10g}" for _ch, _label, y, _u in rows])
            self._status.setText(f"CSV → {out_path}")
        except Exception as exc:
            self._status.setText(f"CSV export error: {exc}")

    def _export_xmgrace(self) -> None:
        if self._spec is None:
            self._status.setText("Nothing to export — spectrum failed to load.")
            return
        rows = list(self._selected_channels_in_display_units())
        if not rows:
            self._status.setText("Tick at least one channel before exporting.")
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "Choose output folder for xmgrace export",
            str(Path.home()))
        if not out_dir:
            return
        try:
            from probeflow.analysis.xmgrace_export import Curve, export_bundle
        except ImportError as exc:
            self._status.setText(f"xmgrace export unavailable: {exc}")
            return
        # Group all channels under one Y axis label if they share a unit;
        # otherwise we keep the legend per-channel and a generic "value" label.
        units = {u for _, _label, _y, u in rows}
        if len(units) == 1:
            y_label = f"value ({rows[0][3]})" if rows[0][3] else "value"
        else:
            y_label = "value"
        curves = [
            Curve(name=ch, y=y, legend=f"{label} ({u})" if u else label)
            for ch, label, y, u in rows
        ]
        try:
            paths = export_bundle(
                Path(out_dir),
                self._entry.stem,
                self._spec.x_array,
                curves,
                x_label=self._spec.x_label,
                y_label=y_label,
                title=self._entry.stem,
                subtitle="ProbeFlow xmgrace export",
                font="Helvetica",
            )
        except FileNotFoundError as exc:
            self._status.setText(f"Export error: {exc}")
            return
        except Exception as exc:
            self._status.setText(f"xmgrace failed: {exc}")
            return
        names = ", ".join(p.name for p in paths.values())
        self._status.setText(f"Exported to {out_dir}: {names}")

    # ── Raw-data table ──────────────────────────────────────────────────

    def _show_raw_data(self) -> None:
        if self._spec is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Raw data — {self._entry.stem}")
        dlg.resize(640, 400)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(8, 8, 8, 8)

        table = self._raw_data_table()
        v.addWidget(table)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        v.addLayout(btn_row)

        dlg.exec()

    def _raw_data_table(self) -> QTableWidget:
        spec = self._spec
        if spec is None:
            return QTableWidget(0, 0)
        order = [ch for ch in spec.channel_order if ch in spec.channels]
        if not order:
            order = list(spec.channels.keys())
        n_rows = len(spec.x_array)
        n_cols = 1 + len(order)

        table = QTableWidget(n_rows, n_cols)
        display_rows = [
            (ch, self._channel_label(ch), *self._display_values_for_channel(ch))
            for ch in order
        ]
        headers = [spec.x_label] + [
            f"{label} ({unit})" if unit else label
            for _ch, label, _values, unit in display_rows
        ]
        table.setHorizontalHeaderLabels(headers)

        for r in range(n_rows):
            table.setItem(r, 0, QTableWidgetItem(f"{spec.x_array[r]:.6g}"))
            for c, (_ch, _label, values, _unit) in enumerate(display_rows, start=1):
                table.setItem(r, c, QTableWidgetItem(f"{values[r]:.6g}"))
        table.horizontalHeader().setStretchLastSection(True)
        return table


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
class AboutDialog(QDialog):
    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About ProbeFlow")
        self.setFixedSize(420, 640)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 18, 24, 18)
        lay.setSpacing(4)

        logo_path = LOGO_GIF_PATH if LOGO_GIF_PATH.exists() else LOGO_PATH
        if logo_path.exists():
            logo_lbl = QLabel()
            logo_lbl.setAlignment(Qt.AlignCenter)
            if str(logo_path).endswith(".gif"):
                movie = QMovie(str(logo_path))
                movie.setScaledSize(QSize(372, 372))  # square — matches logo aspect ratio
                logo_lbl.setMovie(movie)
                movie.start()
                self._about_movie = movie
            else:
                pix = QPixmap(str(logo_path))
                logo_lbl.setPixmap(pix.scaledToWidth(372, Qt.SmoothTransformation))
            lay.addWidget(logo_lbl)

        def _row(text, size=11, bold=False, sub=False):
            lbl = QLabel(text)
            f   = QFont("Helvetica", size)
            f.setBold(bold)
            lbl.setFont(f)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setWordWrap(True)
            if sub:
                lbl.setStyleSheet(f"color: {t['sub_fg']};")
            lay.addWidget(lbl)

        _row("ProbeFlow", 16, bold=True)
        _row("Createc → Nanonis File Conversion", 11, sub=True)
        lay.addWidget(_sep())
        _row("Developed at SPMQT-Lab", 11, bold=True)
        _row("Under the supervision of Dr. Peter Jacobson\nThe University of Queensland", 10, sub=True)
        lay.addWidget(_sep())
        _row("Original code by Rohan Platts", 11, bold=True)
        _row("The core conversion algorithms were built by Rohan Platts.\n"
             "This GUI is a refactored and extended version.", 10, sub=True)
        lay.addWidget(_sep())

        gh_btn = QPushButton("View on GitHub")
        gh_btn.setFont(QFont("Helvetica", 11))
        gh_btn.setCursor(QCursor(Qt.PointingHandCursor))
        gh_btn.setObjectName("accentBtn")
        gh_btn.setFixedHeight(36)
        gh_btn.clicked.connect(lambda: _open_url(GITHUB_URL))
        lay.addWidget(gh_btn)


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
        self.setFixedHeight(NAVBAR_H)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
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

        lay.addStretch()

        def _nbtn(text: str, slot) -> QPushButton:
            btn = QPushButton(text)
            btn.setFont(QFont("Helvetica", 11))
            btn.setObjectName("navBtn")
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.clicked.connect(slot)
            lay.addWidget(btn)
            self._btns.append(btn)
            return btn

        self._theme_btn = _nbtn(
            "Light mode" if dark else "Dark mode",
            self.theme_toggle_clicked.emit,
        )
        self._font_size_btn = _nbtn(
            f"Text: {self._font_size_label}",
            lambda: None,
        )
        font_menu = QMenu(self._font_size_btn)
        self._font_size_actions: dict[str, QAction] = {}
        for label in GUI_FONT_SIZES:
            action = QAction(label, font_menu)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked=False, value=label: self.set_font_size(value))
            font_menu.addAction(action)
            self._font_size_actions[label] = action
        self._font_size_btn.setMenu(font_menu)
        self._sync_font_size_button()
        _nbtn("GitHub", lambda: _open_url(GITHUB_URL))
        _nbtn("About",  self.about_clicked.emit)

        self._apply_nav_theme()

    def set_dark(self, dark: bool):
        self._dark = dark
        self._theme_btn.setText("Light mode" if dark else "Dark mode")
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
        self._font_size_btn.setText(f"Text: {self._font_size_label}")
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
<p class="sub">Params: <span class="param">threshold_mad</span> (default 5.0),
<span class="param">method</span> = mad | step</p>
<p><b>mad (default):</b> Computes the median of each row and flags rows whose
median deviates from the image-wide median by more than
<span class="param">threshold_mad</span> &times; MAD (median absolute deviation of row
medians).  Flagged rows are replaced by a distance-weighted column-wise blend
of the nearest good rows above and below.  Self-calibrating — works without
knowing physical step heights.  Best for tip crashes and vibration bursts that
shift an entire row.</p>
<p><b>step:</b> Scans each column top-to-bottom and detects transitions where
the pixel-to-pixel step exceeds an auto-computed threshold.  The elevated
region between a positive and subsequent negative crossing is flagged and
interpolated per-column.  Handles <em>partial</em> bad lines (only some
columns affected) and isolated pixel spikes that the row-median detector
misses.  Based on the ImageJ Remove_Bad_Lines plugin by M.&nbsp;Schmid.</p>

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
class _TerminalPane(QTextEdit):
    """Single-pane terminal edit: prompt + input + output all in one area.

    Prompt text is protected from editing.  Up/Down navigate history.
    Ctrl+C sends SIGTERM to the running process.
    """

    command_entered = Signal(str)
    interrupt_requested = Signal()

    _MONO = "Cascadia Code, Consolas, Courier New" if sys.platform == "win32" else "Monospace"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setUndoRedoEnabled(False)
        self.setAcceptRichText(False)
        self._history: list[str] = []
        self._history_idx: int = -1
        self._input_start: int = 0   # document position right after prompt
        self._prompt_str: str = "$ "

    # ── Public API ─────────────────────────────────────────────────────────────
    def set_prompt(self, prompt: str):
        self._prompt_str = prompt

    def show_prompt(self):
        """Append a new prompt line and position cursor after it."""
        from PySide6.QtGui import QTextCharFormat, QTextCursor
        c = self.textCursor()
        c.movePosition(QTextCursor.End)

        # Coloured prompt: cyan cwd portion, white "$"
        fmt_prompt = QTextCharFormat()
        fmt_prompt.setForeground(QColor("#89dceb"))
        fmt_prompt.setFont(QFont(self._MONO, 10))
        c.insertText(self._prompt_str, fmt_prompt)

        # Reset to plain style for typed input
        fmt_plain = QTextCharFormat()
        fmt_plain.setForeground(QColor("#cdd6f4"))
        fmt_plain.setFont(QFont(self._MONO, 10))
        c.setCharFormat(fmt_plain)

        self.setTextCursor(c)
        self._input_start = c.position()
        self._history_idx = -1
        self.ensureCursorVisible()

    def append_output(self, text: str):
        """Insert text at end (before the current input line)."""
        from PySide6.QtGui import QTextCharFormat, QTextCursor
        c = self.textCursor()
        c.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#cdd6f4"))
        fmt.setFont(QFont(self._MONO, 10))
        c.insertText(text, fmt)
        self.setTextCursor(c)
        self.ensureCursorVisible()

    def append_error(self, text: str):
        from PySide6.QtGui import QTextCharFormat, QTextCursor
        c = self.textCursor()
        c.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#f38ba8"))
        fmt.setFont(QFont(self._MONO, 10))
        c.insertText(text, fmt)
        self.setTextCursor(c)
        self.ensureCursorVisible()

    # ── Input text helpers ─────────────────────────────────────────────────────
    def _current_input(self) -> str:
        from PySide6.QtGui import QTextCursor
        c = self.textCursor()
        c.setPosition(self._input_start)
        c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        return c.selectedText()

    def _replace_input(self, text: str):
        from PySide6.QtGui import QTextCursor
        c = self.textCursor()
        c.setPosition(self._input_start)
        c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        c.insertText(text)
        self.setTextCursor(c)

    def _clamp_cursor(self):
        from PySide6.QtGui import QTextCursor
        c = self.textCursor()
        if c.position() < self._input_start and not c.hasSelection():
            c.movePosition(QTextCursor.End)
            self.setTextCursor(c)

    # ── Key handling ───────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        from PySide6.QtGui import QTextCursor
        key = event.key()
        mods = event.modifiers()

        # Ctrl+C → interrupt
        if mods & Qt.ControlModifier and key == Qt.Key_C:
            self.interrupt_requested.emit()
            return

        # Ctrl+A → select all input (not entire document)
        if mods & Qt.ControlModifier and key == Qt.Key_A:
            c = self.textCursor()
            c.setPosition(self._input_start)
            c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            self.setTextCursor(c)
            return

        # Enter → submit
        if key in (Qt.Key_Return, Qt.Key_Enter):
            cmd = self._current_input().strip()
            c = self.textCursor()
            c.movePosition(QTextCursor.End)
            c.insertText("\n")
            self.setTextCursor(c)
            self._input_start = c.position()
            if cmd:
                self._history = [x for x in self._history if x != cmd]
                self._history.insert(0, cmd)
            self.command_entered.emit(cmd)
            return

        # Up → history back
        if key == Qt.Key_Up:
            if self._history:
                self._history_idx = min(self._history_idx + 1, len(self._history) - 1)
                self._replace_input(self._history[self._history_idx])
            return

        # Down → history forward
        if key == Qt.Key_Down:
            if self._history_idx > 0:
                self._history_idx -= 1
                self._replace_input(self._history[self._history_idx])
            elif self._history_idx == 0:
                self._history_idx = -1
                self._replace_input("")
            return

        # Home → jump to start of input
        if key == Qt.Key_Home and not (mods & Qt.ShiftModifier):
            c = self.textCursor()
            c.setPosition(self._input_start)
            self.setTextCursor(c)
            return

        # Don't allow editing before input start
        c = self.textCursor()
        if c.position() < self._input_start and not c.hasSelection():
            c.movePosition(QTextCursor.End)
            self.setTextCursor(c)

        # Backspace: never go before the prompt
        if key == Qt.Key_Backspace:
            c = self.textCursor()
            if c.position() <= self._input_start and not c.hasSelection():
                return

        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        # Allow click-to-position for selection/copy; but editing keys will
        # clamp back to the input zone automatically.

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        menu.exec(event.globalPos())


class DeveloperTerminalWidget(QWidget):
    """VS Code-style embedded terminal: prompt + input + output in one pane."""

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self._cwd = Path.cwd()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Top bar
        bar = QWidget()
        bar.setFixedHeight(32)
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(8, 0, 8, 0)
        bar_lay.setSpacing(6)
        title_lbl = QLabel("TERMINAL")
        title_lbl.setFont(QFont("Helvetica", 8, QFont.Bold))
        bar_lay.addWidget(title_lbl)
        bar_lay.addStretch()

        ext_btn = QPushButton("Open External Terminal")
        ext_btn.setFont(QFont("Helvetica", 8))
        ext_btn.setFixedHeight(22)
        ext_btn.setToolTip(
            "Open a real system terminal in this directory.\n"
            "Use this for interactive tools like claude, ipython, vim, etc.")
        ext_btn.clicked.connect(self._open_external)
        bar_lay.addWidget(ext_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setFont(QFont("Helvetica", 8))
        clear_btn.setFixedHeight(22)
        clear_btn.clicked.connect(self._clear)
        bar_lay.addWidget(clear_btn)
        lay.addWidget(bar)

        # Terminal pane
        self._pane = _TerminalPane(self)
        self._pane.command_entered.connect(self._run_command)
        self._pane.interrupt_requested.connect(self._interrupt)
        lay.addWidget(self._pane, 1)

        # Process
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)
        self._process.setWorkingDirectory(str(self._cwd))

        self._apply_theme(t)
        self._set_cwd(self._cwd)

    def _set_cwd(self, cwd: Path):
        self._cwd = cwd
        self._process.setWorkingDirectory(str(cwd))
        self._pane.set_prompt(f"{cwd.name} $ ")
        self._pane.show_prompt()

    def _apply_theme(self, t: dict):
        bg = t.get("bg", "#1e1e2e")
        fg = t.get("fg", "#cdd6f4")
        self._pane.setStyleSheet(
            f"QTextEdit {{ background-color: {bg}; color: {fg}; "
            f"border: none; padding: 6px; }}")
        self.setStyleSheet(f"background-color: {bg};")
        # tint the top bar slightly darker
        bar_bg = t.get("sidebar_bg", "#181825")
        self.findChild(QWidget).setStyleSheet(
            f"background-color: {bar_bg}; color: {fg};")

    def _clear(self):
        self._pane.clear()
        self._set_cwd(self._cwd)

    # Tools that require a real PTY — intercept and open external terminal.
    _PTY_TOOLS = frozenset({
        "claude", "ipython", "ipython3",
        "vim", "vi", "nvim", "nano", "emacs",
        "htop", "top", "btop", "less", "more", "man",
        "ssh", "sftp", "ftp", "telnet",
    })

    def _run_command(self, cmd: str):
        if not cmd:
            self._pane.show_prompt()
            return
        if self._process.state() != QProcess.NotRunning:
            self._pane.append_error("[busy — press Ctrl+C to interrupt]\n")
            self._pane.show_prompt()
            return

        # Detect interactive tools that need a real PTY
        first = cmd.split()[0].lstrip("./") if cmd.split() else ""
        # plain `python` / `python3` with no args also needs a PTY
        is_bare_python = first in ("python", "python3", "python3.x") and len(cmd.split()) == 1
        if first in self._PTY_TOOLS or is_bare_python:
            self._pane.append_output(
                f"['{first}' needs an interactive terminal — opening external terminal…]\n")
            self._pane.show_prompt()
            self._open_external()
            return

        import shutil
        if sys.platform.startswith("win"):
            shell, args = "cmd.exe", ["/c", cmd]
        else:
            shell = shutil.which("bash") or "/bin/sh"
            args = ["-c", cmd]
        self._process.start(shell, args)

    def _interrupt(self):
        if self._process.state() != QProcess.NotRunning:
            self._process.kill()

    def _open_external(self):
        """Launch the real system terminal in the current working directory.

        Interactive tools (claude, ipython, vim, etc.) need a proper PTY —
        use this button instead of running them in the embedded pane.
        """
        cwd = str(self._cwd)
        try:
            if sys.platform.startswith("win"):
                # Try Windows Terminal first, fall back to wt, then cmd
                import shutil
                if shutil.which("wt"):
                    subprocess.Popen(["wt", "--startingDirectory", cwd])
                else:
                    subprocess.Popen(["cmd.exe"], cwd=cwd,
                                     creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                # On WSL: open wsl.exe so we stay in the Linux env
                # Check if we're inside WSL
                is_wsl = Path("/proc/version").exists() and \
                    "microsoft" in Path("/proc/version").read_text().lower()
                if is_wsl:
                    subprocess.Popen(
                        ["wsl.exe", "--cd", cwd],
                        start_new_session=True,
                    )
                else:
                    # Plain Linux: try common terminal emulators
                    import shutil as _sh
                    for term in ("gnome-terminal", "xterm", "konsole", "xfce4-terminal"):
                        if _sh.which(term):
                            subprocess.Popen([term], cwd=cwd, start_new_session=True)
                            break
            self._pane.append_output("[External terminal opened]\n")
        except Exception as exc:
            self._pane.append_error(f"[Could not open external terminal: {exc}]\n")
        self._pane.show_prompt()

    def _on_output(self):
        raw = bytes(self._process.readAllStandardOutput())
        text = raw.decode("utf-8", errors="replace")
        text = _re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
        text = _re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", text)
        if text:
            self._pane.append_output(text)

    def _on_finished(self, exit_code: int, exit_status):
        if exit_code != 0:
            self._pane.append_error(f"[exited {exit_code}]\n")
        self._pane.show_prompt()

    def _on_error(self, error):
        self._pane.append_error(f"[error: {error}]\n")
        self._pane.show_prompt()


# ── Main window ───────────────────────────────────────────────────────────────
class ProbeFlowWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProbeFlow")
        self.setMinimumSize(1100, 720)

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

        # Tab bar
        self._tab_bar = QWidget()
        self._tab_bar.setFixedHeight(44)
        tab_lay = QHBoxLayout(self._tab_bar)
        tab_lay.setContentsMargins(0, 0, 0, 0)
        tab_lay.setSpacing(0)
        self._tab_browse   = QPushButton("Browse")
        self._tab_convert  = QPushButton("Convert")
        self._tab_features = QPushButton("FeatureCounting")
        self._tab_tv       = QPushButton("TV-denoise")
        self._tab_dev      = QPushButton("Dev")
        self._tab_defs     = QPushButton("Defs")
        for btn in (self._tab_browse, self._tab_convert, self._tab_features,
                    self._tab_tv, self._tab_dev, self._tab_defs):
            btn.setFont(QFont("Helvetica", 11, QFont.Bold))
            btn.setFixedHeight(44)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.setFlat(True)
            tab_lay.addWidget(btn)
        tab_lay.addStretch()
        self._tab_browse.clicked.connect(lambda: self._switch_mode("browse"))
        self._tab_convert.clicked.connect(lambda: self._switch_mode("convert"))
        self._tab_features.clicked.connect(lambda: self._switch_mode("features"))
        self._tab_tv.clicked.connect(lambda: self._switch_mode("tv"))
        self._tab_dev.clicked.connect(lambda: self._switch_mode("dev"))
        self._tab_defs.clicked.connect(lambda: self._switch_mode("defs"))
        v_lay.addWidget(self._tab_bar)

        # Body splitter
        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setHandleWidth(5)
        v_lay.addWidget(self._splitter, 1)

        t = THEMES["dark" if self._dark else "light"]

        # ── Content stack (center+left area) ──────────────────────────────────
        self._content_stack = QStackedWidget()

        # Browse mode: inner splitter [BrowseToolPanel | ThumbnailGrid]
        self._browse_tools = BrowseToolPanel(t, self._cfg)
        self._browse_tools.setFixedWidth(265)
        self._grid         = ThumbnailGrid(t)
        browse_split = QSplitter(Qt.Horizontal)
        browse_split.setHandleWidth(3)
        browse_split.addWidget(self._browse_tools)
        browse_split.addWidget(self._grid)
        browse_split.setStretchFactor(0, 0)
        browse_split.setStretchFactor(1, 1)

        self._conv_panel    = ConvertPanel(t, self._cfg)
        self._features_panel = FeaturesPanel(t)
        self._tv_panel       = TVPanel(t)
        self._dev_terminal   = DeveloperTerminalWidget(t)
        self._content_stack.addWidget(browse_split)
        self._content_stack.addWidget(self._conv_panel)
        self._content_stack.addWidget(self._features_panel)
        self._content_stack.addWidget(self._tv_panel)
        self._content_stack.addWidget(self._dev_terminal)
        self._defs_panel     = _DefinitionsPanel(t)
        self._content_stack.addWidget(self._defs_panel)
        self._splitter.addWidget(self._content_stack)

        # ── Right: sidebar stack ───────────────────────────────────────────────
        self._sidebar_stack    = QStackedWidget()
        self._sidebar_stack.setFixedWidth(300)
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
        self._defs_sidebar   = QWidget()
        self._sidebar_stack.addWidget(self._defs_sidebar)
        self._splitter.addWidget(self._sidebar_stack)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

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

    # ── Mode switching ─────────────────────────────────────────────────────────
    def _switch_mode(self, mode: str):
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
        elif mode == "defs":
            self._content_stack.setCurrentIndex(5)
            self._sidebar_stack.setCurrentIndex(5)
            self._status_bar.showMessage(
                "Processing algorithm reference — definitions and scientific details")
        else:
            self._content_stack.setCurrentIndex(1)
            self._sidebar_stack.setCurrentIndex(1)
            self._update_count(self._conv_panel.input_entry.text())
        self._update_tab_styles()

    def _update_tab_styles(self):
        t = THEMES["dark" if self._dark else "light"]
        for btn, name in ((self._tab_browse, "browse"),
                          (self._tab_convert, "convert"),
                          (self._tab_features, "features"),
                          (self._tab_tv, "tv"),
                          (self._tab_dev, "dev"),
                          (self._tab_defs, "defs")):
            active = (self._mode == name)
            if active:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {t['tab_act']};
                        color: {t['accent_bg']};
                        border-bottom: 2px solid {t['accent_bg']};
                        border-top: none; border-left: none; border-right: none;
                        padding: 0 18px;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {t['tab_inact']};
                        color: {t['fg']};
                        border: none;
                        padding: 0 18px;
                    }}
                    QPushButton:hover {{ color: {t['accent_bg']}; }}
                """)

    # ── Browse ─────────────────────────────────────────────────────────────────
    def _open_browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Open folder containing scan / .VERT files")
        if not d:
            return
        self._switch_mode("browse")
        from probeflow.core.indexing import index_folder
        all_items    = index_folder(Path(d), recursive=True, include_errors=True)
        sxm_entries  = _scan_items_to_sxm(all_items)
        vert_entries = _spec_items_to_vert(all_items)
        n_errors     = sum(1 for it in all_items if it.load_error)
        entries = sorted(sxm_entries + vert_entries, key=lambda e: e.stem)
        self._grid.load(entries, folder_path=d)
        self._n_loaded = len(entries)
        # New folder → discard previous spec mapping; user can rebuild it.
        self._spec_image_map = {}
        n_sxm  = len(sxm_entries)
        n_vert = len(vert_entries)
        parts  = []
        if n_sxm:
            parts.append(f"{n_sxm} scan{'s' if n_sxm != 1 else ''}")
        if n_vert:
            parts.append(f"{n_vert} spec{'s' if n_vert != 1 else ''}")
        if n_errors:
            parts.append(f"{n_errors} error{'s' if n_errors != 1 else ''}")
        desc = ", ".join(parts) if parts else "0 files"
        self._status_bar.showMessage(
            f"Loaded {desc} — Double-click to view  |  "
            "Thumbnail controls update the whole browse grid")
        self._browse_info.clear()

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

    def _on_thumbnail_align_changed(self, mode: str):
        n = self._grid.set_thumbnail_align_rows(mode)
        label = mode if mode in ("Median", "Mean") else "None"
        if n == 0:
            self._status_bar.showMessage(f"Thumbnail align rows: {label}")
        else:
            self._status_bar.showMessage(
                f"Thumbnail align rows: {label} — queued {n} image thumbnail"
                f"{'s' if n != 1 else ''}")

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
        self._dark = not self._dark
        self._navbar.set_dark(self._dark)
        self._apply_theme()

    def _on_gui_font_size_changed(self, label: str):
        self._gui_font_size = normalise_gui_font_size(label)
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
        self._tab_bar.setStyleSheet(f"background-color: {t['main_bg']};")
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
