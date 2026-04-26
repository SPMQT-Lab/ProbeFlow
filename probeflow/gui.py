"""ProbeFlow — PySide6 GUI for STM scan browsing, processing, and Createc→Nanonis conversion."""

from __future__ import annotations

import io
import json
import re as _re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PySide6.QtCore import (
    Qt, QObject, QRect, QRunnable, QThreadPool, QTimer, QSize, Signal, Slot,
)
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QFont, QImage, QMovie, QPainter, QPen,
    QPixmap, QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
    QDialog, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QRadioButton, QScrollArea, QSlider, QSplitter, QStackedWidget,
    QStatusBar, QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
    QToolTip, QVBoxLayout, QWidget,
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
from probeflow.common import mark_processed_stem
from probeflow.display import (
    array_to_uint8 as _array_to_uint8,
    clip_range_from_array as _clip_range_from_array,
    histogram_from_array as _histogram_from_array,
)
from probeflow.display_state import DisplayRangeState
from probeflow.gui_processing import (
    NUMERIC_PROC_KEYS,
    apply_processing_state_to_scan,
    processing_state_from_gui,
)
from probeflow.scan import SUPPORTED_SUFFIXES, load_scan

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_PATH     = Path.home() / ".probeflow_config.json"
REPO_ROOT       = Path(__file__).resolve().parent.parent
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

# ── Colormaps (25 — most cited in STM/SPM publications) ──────────────────────
STM_COLORMAPS: list[tuple[str, str]] = [
    # Topography
    ("AFM Hot",      "afmhot"),
    ("Hot",          "hot"),
    ("Gray",         "gray"),
    ("Copper",       "copper"),
    ("Bone",         "bone"),
    # Perceptually uniform
    ("Viridis",      "viridis"),
    ("Plasma",       "plasma"),
    ("Inferno",      "inferno"),
    ("Magma",        "magma"),
    ("Cividis",      "cividis"),
    # Diverging — dI/dV and STS maps
    ("Cool-Warm",    "coolwarm"),
    ("RdBu",         "RdBu_r"),
    ("Seismic",      "seismic"),
    ("BWR",          "bwr"),
    ("Spectral",     "Spectral_r"),
    ("PiYG",         "PiYG"),
    # Sequential
    ("YlOrRd",       "YlOrRd"),
    ("Blues",        "Blues"),
    ("Oranges",      "Oranges"),
    ("Greens",       "Greens"),
    # Legacy / full spectrum
    ("Jet",          "jet"),
    ("Turbo",        "turbo"),
    ("Rainbow",      "gist_rainbow"),
    # Cyclic — phase maps
    ("Twilight",     "twilight"),
    ("HSV",          "hsv"),
]

CMAP_NAMES = [label for label, _ in STM_COLORMAPS]
CMAP_KEY   = {label: key for label, key in STM_COLORMAPS}

_LUTS: dict[str, np.ndarray] = {}

DEFAULT_CMAP_LABEL = "Gray"
DEFAULT_CMAP_KEY   = "gray"


def _make_lut(mpl_name: str) -> np.ndarray:
    try:
        import matplotlib.cm as _mcm
        cmap = _mcm.get_cmap(mpl_name)
        x    = np.linspace(0, 1, 256)
        rgba = cmap(x)
        return (rgba[:, :3] * 255).astype(np.uint8)
    except Exception:
        pass
    # fallback: grayscale
    x = np.linspace(0, 1, 256)
    lut = (np.stack([x, x, x], axis=1) * 255).astype(np.uint8)
    return lut


def _get_lut(label_or_key: str) -> np.ndarray:
    key = CMAP_KEY.get(label_or_key, label_or_key)
    if key not in _LUTS:
        _LUTS[key] = _make_lut(key)
    return _LUTS[key]


# ── Data model ────────────────────────────────────────────────────────────────
PLANE_NAMES = ["Z fwd", "Z bwd", "I fwd", "I bwd"]


@dataclass
class SxmFile:
    path:          Path
    stem:          str
    Nx:            int            = 512
    Ny:            int            = 512
    bias_mv:       Optional[float] = None
    current_pa:    Optional[float] = None
    scan_nm:       Optional[float] = None
    source_format: str            = "sxm"


@dataclass
class VertFile:
    path:         Path
    stem:         str
    sweep_type:   str            = "unknown"
    n_points:     int            = 0
    bias_mv:      Optional[float] = None
    spec_freq_hz: Optional[float] = None


def _card_meta_str(entry: SxmFile) -> str:
    """Format key physical parameters for the thumbnail card label."""
    line1 = "  |  ".join(filter(None, [
        f"{entry.Nx}×{entry.Ny}" if entry.Nx > 0 else "",
        f"{entry.scan_nm:.1f} nm" if entry.scan_nm is not None else "",
    ]))
    line2 = "  |  ".join(filter(None, [
        f"{entry.bias_mv:.0f} mV"    if entry.bias_mv    is not None else "",
        f"{entry.current_pa:.0f} pA" if entry.current_pa is not None else "",
    ]))
    return "\n".join(filter(None, [line1, line2]))


# ── SXM parsing / plane I/O live in probeflow.sxm_io (imported above). ──────


def clip_range_from_arr(
    arr: Optional[np.ndarray],
    pct_lo: float = 1.0,
    pct_hi: float = 99.0,
) -> tuple[Optional[float], Optional[float]]:
    """Return (vmin, vmax) for display clipping, or (None, None) on failure.

    Thin adapter over :func:`probeflow.display.clip_range_from_array` that
    preserves the GUI contract of returning (None, None) rather than raising.
    """
    if arr is None:
        return None, None
    try:
        return _clip_range_from_array(arr, pct_lo, pct_hi)
    except ValueError:
        return None, None


def _apply_processing(
    arr: np.ndarray,
    processing: dict,
) -> np.ndarray:
    """Apply the array-transform portion of the processing pipeline.

    Converts the GUI processing dict to a canonical ProcessingState and
    delegates to apply_processing_state().  Grain detection / colormap / clip
    settings are display-only and are silently ignored.  Returns a new float64
    array; the input is never modified.
    """
    from probeflow.processing_state import apply_processing_state
    return apply_processing_state(arr, processing_state_from_gui(processing or {}))


def render_scan_thumbnail(
    scan_path: Path,
    plane_idx: int            = 0,
    colormap:  str            = "gray",
    clip_low:  float          = 1.0,
    clip_high: float          = 99.0,
    size:      tuple          = (148, 116),
    vmin:      Optional[float] = None,
    vmax:      Optional[float] = None,
) -> Optional[Image.Image]:
    """Render a thumbnail of any supported scan format via the unified reader.

    If *vmin* and *vmax* are provided, they are used directly (manual mode).
    Otherwise *clip_low*/*clip_high* percentiles are applied (percentile mode).
    """
    try:
        scan = load_scan(scan_path)
        if plane_idx >= scan.n_planes:
            return None
        arr = scan.planes[plane_idx]

        if vmin is None or vmax is None:
            vmin, vmax = clip_range_from_arr(arr, clip_low, clip_high)
        if vmin is None:
            return None

        u8      = _array_to_uint8(arr, vmin=vmin, vmax=vmax)
        colored = _get_lut(colormap)[u8]
        img     = Image.fromarray(colored, mode="RGB")
        img.thumbnail(size, Image.LANCZOS)
        return img
    except Exception:
        return None


def render_spec_thumbnail(
    vert_path: Path,
    size: tuple = (148, 116),
    dark: bool = True,
) -> Optional[Image.Image]:
    """Render a small matplotlib plot of a .VERT file as a PIL Image."""
    try:
        from .spec_io import read_spec_file
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        spec = read_spec_file(vert_path)
        # Prefer the first default channel; fall back to 'I' then first available
        ch = None
        if spec.default_channels:
            ch = spec.default_channels[0]
        elif "I" in spec.channels:
            ch = "I"
        elif spec.channels:
            ch = next(iter(spec.channels))
        y = spec.channels.get(ch) if ch else None
        if y is None or len(y) == 0:
            return None
        x = spec.x_array

        bg   = "#1e1e2e" if dark else "#ffffff"
        line = "#89b4fa" if dark else "#1e66f5"

        fig = Figure(figsize=(size[0] / 80, size[1] / 80), dpi=80)
        FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.plot(x, y, linewidth=0.9, color=line)
        ax.set_facecolor(bg)
        fig.patch.set_facecolor(bg)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = Image.open(buf).convert("RGB").copy()
        img = img.resize(size, Image.LANCZOS)
        return img
    except Exception:
        return None


def render_with_processing(
    arr:        np.ndarray,
    colormap:   str,
    clip_low:   float,
    clip_high:  float,
    processing: dict,
    size:       Optional[tuple] = None,
    vmin:       Optional[float] = None,
    vmax:       Optional[float] = None,
) -> Optional[Image.Image]:
    """Apply the full processing pipeline to *arr* then render to a PIL Image.

    processing keys (all optional):
        remove_bad_lines : bool
        align_rows       : str | None  — 'median' | 'mean' | 'linear'
        bg_order         : None | 1 | 2
        facet_level      : bool
        smooth_sigma     : float | None  — sigma in pixels (Gaussian)
        edge_method      : str | None   — 'laplacian' | 'log' | 'dog'
        edge_sigma       : float
        edge_sigma2      : float
        fft_mode         : None | 'low_pass' | 'high_pass'
        fft_cutoff       : float  (0.01–0.50)
        fft_window       : str    ('hanning')
        grain_threshold  : float | None  — percentile for grain detection
        grain_above      : bool
    """
    if processing is None:
        processing = {}
    try:
        a = _apply_processing(arr, processing)

        if vmin is None or vmax is None:
            vmin, vmax = clip_range_from_arr(a, clip_low, clip_high)
        if vmin is None:
            return None

        # Grain overlay: colour-code labelled regions on top of the image
        grain_thresh = processing.get('grain_threshold')
        u8 = _array_to_uint8(a, vmin=vmin, vmax=vmax)
        if grain_thresh is not None:
            label_map, n_grains, _ = _proc.detect_grains(
                a,
                threshold_pct=float(grain_thresh),
                above=bool(processing.get('grain_above', True)),
            )
            colored = _get_lut(colormap)[u8].copy()
            # Tint grain pixels: blend toward red
            grain_px = label_map > 0
            colored[grain_px, 0] = np.clip(colored[grain_px, 0].astype(int) // 2 + 128, 0, 255)
            colored[grain_px, 1] = colored[grain_px, 1] // 3
            colored[grain_px, 2] = colored[grain_px, 2] // 3
            img = Image.fromarray(colored, mode="RGB")
        else:
            colored = _get_lut(colormap)[u8]
            img     = Image.fromarray(colored, mode="RGB")

        if size:
            img.thumbnail(size, Image.LANCZOS)
        return img
    except Exception:
        return None


def _scan_items_to_sxm(items) -> list[SxmFile]:
    """Convert ProbeFlowItem scan entries to SxmFile for the existing GUI.

    Preserves the stem-deduplication behaviour of the old scan_image_folder:
    if two files share a stem, only the first (by index_folder sort order)
    is kept.
    """
    _fmt = {"createc_dat": "dat", "nanonis_sxm": "sxm"}
    seen: set[str] = set()
    result: list[SxmFile] = []
    for item in items:
        if item.item_type != "scan" or item.path.stem in seen:
            continue
        seen.add(item.path.stem)
        src_fmt = _fmt.get(item.source_format, item.source_format)
        if item.load_error or item.shape is None:
            result.append(SxmFile(path=item.path, stem=item.path.stem, source_format=src_fmt))
        else:
            Ny, Nx = item.shape
            result.append(SxmFile(
                path=item.path, stem=item.path.stem,
                Nx=Nx, Ny=Ny,
                bias_mv=item.bias * 1000 if item.bias is not None else None,
                current_pa=item.setpoint * 1e12 if item.setpoint is not None else None,
                scan_nm=item.scan_range[0] * 1e9 if item.scan_range else None,
                source_format=src_fmt,
            ))
    return result


def _spec_items_to_vert(items) -> list[VertFile]:
    """Convert ProbeFlowItem spectrum entries to VertFile for the existing GUI."""
    result: list[VertFile] = []
    for item in items:
        if item.item_type != "spectrum":
            continue
        if item.load_error:
            result.append(VertFile(path=item.path, stem=item.path.stem))
        else:
            result.append(VertFile(
                path=item.path,
                stem=item.path.stem,
                sweep_type=str(item.metadata.get("sweep_type") or "unknown"),
                n_points=int(item.metadata.get("n_points") or 0),
                bias_mv=item.bias * 1000 if item.bias is not None else None,
                spec_freq_hz=item.metadata.get("spec_freq_hz"),
            ))
    return result


def scan_image_folder(root: Path) -> list[SxmFile]:
    """Find all supported scan files under root and return lightweight SxmFile entries.

    Delegates to :func:`~probeflow.indexing.index_folder` for discovery and
    sniffing, then converts the resulting :class:`~probeflow.indexing.ProbeFlowItem`
    objects to the GUI-internal :class:`SxmFile` type.
    """
    from probeflow.indexing import index_folder
    items = index_folder(Path(root), recursive=True, include_errors=True)
    return _scan_items_to_sxm(items)


def scan_vert_folder(root: Path) -> list[VertFile]:
    """Find all spectroscopy files under root and return lightweight VertFile entries.

    Delegates to :func:`~probeflow.indexing.index_folder` for discovery and
    sniffing, then converts the resulting :class:`~probeflow.indexing.ProbeFlowItem`
    objects to the GUI-internal :class:`VertFile` type.
    """
    from probeflow.indexing import index_folder
    items = index_folder(Path(root), recursive=True, include_errors=True)
    return _spec_items_to_vert(items)


# ── Config ────────────────────────────────────────────────────────────────────
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
    }
    try:
        if CONFIG_PATH.exists():
            defaults.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass
    return defaults


def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


# ── PIL → QPixmap ─────────────────────────────────────────────────────────────
def pil_to_pixmap(img: Image.Image) -> QPixmap:
    img  = img.convert("RGB")
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ── Worker: thumbnail ─────────────────────────────────────────────────────────
class ThumbnailSignals(QObject):
    loaded = Signal(str, QPixmap, object)  # stem, pixmap, token


class ThumbnailLoader(QRunnable):
    def __init__(self, entry: SxmFile, colormap: str, token, w: int, h: int,
                 clip_low: float = 1.0, clip_high: float = 99.0,
                 processing: dict = None):
        super().__init__()
        self.setAutoDelete(True)
        self.signals    = ThumbnailSignals()
        self.entry      = entry
        self.colormap   = colormap
        self.token      = token
        self.w          = w
        self.h          = h
        self.clip_low   = clip_low
        self.clip_high  = clip_high
        self.processing = processing or {}

    def run(self):
        if self.processing:
            # Use raw array path so processing functions receive unscaled data
            try:
                scan = load_scan(self.entry.path)
                arr = scan.planes[0] if scan.n_planes > 0 else None
            except Exception:
                arr = None
            if arr is not None:
                img = render_with_processing(
                    arr, self.colormap, self.clip_low, self.clip_high,
                    self.processing, size=(self.w, self.h))
            else:
                img = None
        else:
            img = render_scan_thumbnail(self.entry.path, 0, self.colormap,
                                        self.clip_low, self.clip_high,
                                        size=(self.w, self.h))
        if img is not None:
            self.signals.loaded.emit(self.entry.stem, pil_to_pixmap(img), self.token)


# ── Worker: spec thumbnail ────────────────────────────────────────────────────
class SpecThumbnailLoader(QRunnable):
    def __init__(self, entry: VertFile, token, w: int, h: int, dark: bool = True):
        super().__init__()
        self.setAutoDelete(True)
        self.signals = ThumbnailSignals()
        self.entry   = entry
        self.token   = token
        self.w       = w
        self.h       = h
        self.dark    = dark

    def run(self):
        img = render_spec_thumbnail(self.entry.path, size=(self.w, self.h),
                                    dark=self.dark)
        if img is not None:
            self.signals.loaded.emit(self.entry.stem, pil_to_pixmap(img), self.token)


# ── Worker: channel thumbnails ────────────────────────────────────────────────
class ChannelSignals(QObject):
    loaded = Signal(int, QPixmap, object)


class ChannelLoader(QRunnable):
    def __init__(self, entry: SxmFile, idx: int, colormap: str,
                 token, w: int, h: int, signals: ChannelSignals,
                 clip_low: float = 1.0, clip_high: float = 99.0,
                 processing: dict = None):
        super().__init__()
        self.setAutoDelete(True)
        self.signals    = signals
        self.entry      = entry
        self.idx        = idx
        self.colormap   = colormap
        self.token      = token
        self.w          = w
        self.h          = h
        self.clip_low   = clip_low
        self.clip_high  = clip_high
        self.processing = processing or {}

    def run(self):
        if self.processing:
            try:
                scan = load_scan(self.entry.path)
                arr = scan.planes[self.idx] if self.idx < scan.n_planes else None
            except Exception:
                arr = None
            if arr is not None:
                img = render_with_processing(
                    arr, self.colormap, self.clip_low, self.clip_high,
                    self.processing, size=(self.w, self.h))
            else:
                img = None
        else:
            img = render_scan_thumbnail(self.entry.path, self.idx, self.colormap,
                                        self.clip_low, self.clip_high,
                                        size=(self.w, self.h))
        if img is not None:
            self.signals.loaded.emit(self.idx, pil_to_pixmap(img), self.token)


# ── Worker: full-size viewer image ────────────────────────────────────────────
class ViewerSignals(QObject):
    loaded = Signal(QPixmap, object)


class ViewerLoader(QRunnable):
    def __init__(self, entry: SxmFile, colormap: str, token, w: int, h: int,
                 plane_idx: int = 0, clip_low: float = 1.0,
                 clip_high: float = 99.0, processing: dict = None,
                 vmin: Optional[float] = None, vmax: Optional[float] = None):
        super().__init__()
        self.setAutoDelete(True)
        self.signals    = ViewerSignals()
        self.entry      = entry
        self.colormap   = colormap
        self.token      = token
        self.w          = w
        self.h          = h
        self.plane_idx  = plane_idx
        self.clip_low   = clip_low
        self.clip_high  = clip_high
        self.processing = processing or {}
        self.vmin       = vmin
        self.vmax       = vmax

    def run(self):
        if self.processing:
            try:
                scan = load_scan(self.entry.path)
                arr = scan.planes[self.plane_idx] if self.plane_idx < scan.n_planes else None
            except Exception:
                arr = None
            if arr is not None:
                img = render_with_processing(arr, self.colormap,
                                             self.clip_low, self.clip_high,
                                             self.processing,
                                             size=(self.w, self.h),
                                             vmin=self.vmin, vmax=self.vmax)
            else:
                img = None
        else:
            img = render_scan_thumbnail(self.entry.path, self.plane_idx,
                                        self.colormap, self.clip_low, self.clip_high,
                                        size=(self.w, self.h),
                                        vmin=self.vmin, vmax=self.vmax)
        if img is not None:
            self.signals.loaded.emit(pil_to_pixmap(img), self.token)


# ── Worker: conversion ────────────────────────────────────────────────────────
class ConversionSignals(QObject):
    log_msg  = Signal(str, str)
    finished = Signal(str)


class ConversionWorker(QRunnable):
    def __init__(self, in_dir: str, out_dir: str,
                 do_png: bool, do_sxm: bool,
                 clip_low: float, clip_high: float):
        super().__init__()
        self.setAutoDelete(True)
        self.signals   = ConversionSignals()
        self.in_dir    = in_dir
        # if no custom output, use the input dir as base
        self.out_dir   = out_dir if out_dir else in_dir
        self.do_png    = do_png
        self.do_sxm    = do_sxm
        self.clip_low  = clip_low
        self.clip_high = clip_high

    def run(self):
        def _log(msg, tag="info"):
            self.signals.log_msg.emit(msg, tag)

        in_path  = Path(self.in_dir)
        out_path = Path(self.out_dir)
        try:
            if self.do_png:
                from probeflow.dat_png import main as png_main
                _log("── PNG conversion ──", "info")
                png_main(src=in_path, out_root=out_path / "png",
                         clip_low=self.clip_low, clip_high=self.clip_high,
                         verbose=False)
                _log("PNG done.", "ok")

            if self.do_sxm:
                from probeflow.dat_sxm import convert_dat_to_sxm
                _log("── SXM conversion ──", "info")
                files = sorted(in_path.glob("*.dat"))
                if not files:
                    _log(f"No .dat files found in {in_path}", "warn")
                else:
                    sxm_out = out_path / "sxm"
                    sxm_out.mkdir(parents=True, exist_ok=True)
                    errors: dict = {}
                    _log(f"Found {len(files)} .dat file(s)", "info")
                    for i, dat in enumerate(files, 1):
                        _log(f"[{i}/{len(files)}] {dat.name} …", "info")
                        try:
                            convert_dat_to_sxm(dat, sxm_out, DEFAULT_CUSHION,
                                               self.clip_low, self.clip_high)
                            _log(f"  [OK] {dat.name}", "ok")
                        except Exception as exc:
                            _log(f"  FAILED {dat.name}: {exc}", "err")
                            errors[dat.name] = str(exc)
                    if errors:
                        import json as _j
                        (sxm_out / "errors.json").write_text(_j.dumps(errors, indent=2))
                        _log(f"{len(errors)} file(s) failed — see errors.json", "warn")
                    else:
                        _log("All SXM files processed successfully.", "ok")
                    _log(f"Output: {sxm_out}", "info")
        except Exception as exc:
            _log(f"Unexpected error: {exc}", "err")
        finally:
            self.signals.finished.emit(self.out_dir)


# ── ScanCard ──────────────────────────────────────────────────────────────────
class ScanCard(QFrame):
    """Single thumbnail card. Supports single-click, Ctrl+click, and double-click."""
    clicked        = Signal(object, bool)  # SxmFile, ctrl_held
    double_clicked = Signal(object)        # SxmFile

    CARD_W = 200
    CARD_H = 220
    IMG_W  = 180
    IMG_H  = 150

    def __init__(self, entry: SxmFile, t: dict, parent=None):
        super().__init__(parent)
        self.entry     = entry
        self._t        = t
        self._sel      = False
        self._colormap = DEFAULT_CMAP_KEY

        self.setFixedSize(self.CARD_W, self.CARD_H)
        self.setCursor(QCursor(Qt.PointingHandCursor))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 6)
        lay.setSpacing(3)

        self.img_lbl = QLabel()
        self.img_lbl.setFixedSize(self.IMG_W, self.IMG_H)
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setText("…")

        lbl_text = entry.stem if len(entry.stem) <= 22 else entry.stem[:20] + ".."
        self.name_lbl = QLabel(lbl_text)
        self.name_lbl.setAlignment(Qt.AlignCenter)
        self.name_lbl.setFont(QFont("Helvetica", 10))

        self.meta_lbl = QLabel(_card_meta_str(entry))
        self.meta_lbl.setAlignment(Qt.AlignCenter)
        self.meta_lbl.setFont(QFont("Helvetica", 9))

        lay.addWidget(self.img_lbl)
        lay.addWidget(self.name_lbl)
        lay.addWidget(self.meta_lbl)
        self._refresh_style()

    def set_pixmap(self, pixmap: QPixmap):
        self.img_lbl.setPixmap(
            pixmap.scaled(self.IMG_W, self.IMG_H,
                          Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.img_lbl.setText("")

    def set_selected(self, val: bool):
        self._sel = val
        self._refresh_style()

    def apply_theme(self, t: dict):
        self._t = t
        self._refresh_style()

    def _refresh_style(self):
        t = self._t
        if self._sel:
            bg, border, bw, fg = t["card_sel"], t["accent_bg"], 3, t["accent_bg"]
        else:
            bg, border, bw, fg = t["card_bg"], t["sep"], 1, t["card_fg"]
        self.setStyleSheet(f"""
            ScanCard {{
                background-color: {bg};
                border: {bw}px solid {border};
                border-radius: 6px;
            }}
            ScanCard:hover {{
                border: {bw}px solid {t["accent_bg"]};
            }}
        """)
        self.name_lbl.setStyleSheet(f"color: {fg}; background: transparent;")
        self.meta_lbl.setStyleSheet(f"color: {t['sub_fg']}; background: transparent;")
        self.img_lbl.setStyleSheet(f"color: {t['sub_fg']}; background: transparent;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            ctrl = bool(event.modifiers() & Qt.ControlModifier)
            self.clicked.emit(self.entry, ctrl)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(self.entry)
        super().mouseDoubleClickEvent(event)


# ── SpecCard ──────────────────────────────────────────────────────────────────
class SpecCard(QFrame):
    """Thumbnail card for a .VERT spectroscopy file."""
    clicked        = Signal(object, bool)
    double_clicked = Signal(object)

    CARD_W = 200
    CARD_H = 220
    IMG_W  = 180
    IMG_H  = 150

    def __init__(self, entry: VertFile, t: dict, parent=None):
        super().__init__(parent)
        self.entry = entry
        self._t    = t
        self._sel  = False

        self.setFixedSize(self.CARD_W, self.CARD_H)
        self.setCursor(QCursor(Qt.PointingHandCursor))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 6)
        lay.setSpacing(3)

        self.img_lbl = QLabel()
        self.img_lbl.setFixedSize(self.IMG_W, self.IMG_H)
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setText("…")

        lbl_text = entry.stem if len(entry.stem) <= 22 else entry.stem[:20] + ".."
        self.name_lbl = QLabel(lbl_text)
        self.name_lbl.setAlignment(Qt.AlignCenter)
        self.name_lbl.setFont(QFont("Helvetica", 10))

        sweep = entry.sweep_type.replace("_", " ") if entry.sweep_type != "unknown" else "VERT"
        pts   = f"{entry.n_points} pts" if entry.n_points else ""
        meta  = "  |  ".join(filter(None, [sweep, pts]))
        self.meta_lbl = QLabel(meta)
        self.meta_lbl.setAlignment(Qt.AlignCenter)
        self.meta_lbl.setFont(QFont("Helvetica", 9))

        lay.addWidget(self.img_lbl)
        lay.addWidget(self.name_lbl)
        lay.addWidget(self.meta_lbl)
        self._refresh_style()

    def set_pixmap(self, pixmap: QPixmap):
        self.img_lbl.setPixmap(
            pixmap.scaled(self.IMG_W, self.IMG_H,
                          Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.img_lbl.setText("")

    def set_selected(self, val: bool):
        self._sel = val
        self._refresh_style()

    def apply_theme(self, t: dict):
        self._t = t
        self._refresh_style()

    def _refresh_style(self):
        t = self._t
        if self._sel:
            bg, border, bw, fg = t["card_sel"], t["accent_bg"], 3, t["accent_bg"]
        else:
            bg, border, bw, fg = t["card_bg"], t["sep"], 1, t["card_fg"]
        self.setStyleSheet(f"""
            SpecCard {{
                background-color: {bg};
                border: {bw}px solid {border};
                border-radius: 6px;
            }}
            SpecCard:hover {{
                border: {bw}px solid {t["accent_bg"]};
            }}
        """)
        self.name_lbl.setStyleSheet(f"color: {fg}; background: transparent;")
        self.meta_lbl.setStyleSheet(f"color: {t['sub_fg']}; background: transparent;")
        self.img_lbl.setStyleSheet(f"color: {t['sub_fg']}; background: transparent;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            ctrl = bool(event.modifiers() & Qt.ControlModifier)
            self.clicked.emit(self.entry, ctrl)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(self.entry)
        super().mouseDoubleClickEvent(event)


# ── ThumbnailGrid ─────────────────────────────────────────────────────────────
class ThumbnailGrid(QWidget):
    """
    Browse panel: folder toolbar + thumbnail grid.

    - All images default to grayscale on load.
    - Click = single-select; Ctrl+click = multi-select toggle.
    - Double-click = open full-size image viewer.
    - set_colormap_for_selection() reloads ONLY selected cards with the new colormap.
    - Unselected cards keep their current colormap (gray by default).
    """
    entry_selected    = Signal(object)   # primary SxmFile for sidebar
    selection_changed = Signal(int)      # count of selected items
    view_requested    = Signal(object)   # SxmFile to open in full-size viewer

    GAP = 10

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self._t    = t
        self._pool = QThreadPool.globalInstance()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Path strip (folder name + count) ────────────────────────────────
        self._toolbar = QWidget()
        self._toolbar.setFixedHeight(28)
        tb_lay = QHBoxLayout(self._toolbar)
        tb_lay.setContentsMargins(10, 4, 8, 4)
        tb_lay.setSpacing(0)

        self._path_lbl = QLabel("No folder open")
        self._path_lbl.setFont(QFont("Helvetica", 10))
        self._path_lbl.setStyleSheet("background: transparent;")

        tb_lay.addWidget(self._path_lbl, 1)
        outer.addWidget(self._toolbar)

        # ── Scroll area with grid ────────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.NoFrame)

        self._content = QWidget()
        self._grid    = QGridLayout(self._content)
        self._grid.setSpacing(self.GAP)
        self._grid.setContentsMargins(self.GAP, self.GAP, self.GAP, self.GAP)
        self._grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._scroll.setWidget(self._content)
        outer.addWidget(self._scroll, 1)

        # state
        self._cards:          dict[str, Union[ScanCard, SpecCard]] = {}
        self._entries:        list[Union[SxmFile, VertFile]]       = []
        self._selected:       set[str]                         = set()
        self._primary:        Optional[str]                    = None
        self._card_colormaps: dict[str, str]                   = {}
        self._card_processing: dict[str, dict]                 = {}   # current processing per stem
        self._card_clip:      dict[str, tuple[float, float]]   = {}   # current clip per stem
        # per-stem undo stack: list of (colormap, clip_low, clip_high, processing)
        self._history:        dict[str, list[tuple]]           = {}
        self._load_token                                       = object()
        self._current_cols: int                                = 1
        self._filter_mode: str                                 = "all"

        # empty-state placeholder
        self._empty_lbl = QLabel("Open a folder to browse SXM scans")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setFont(QFont("Helvetica", 12))
        self._grid.addWidget(self._empty_lbl, 0, 0)

    # ── Public API ────────────────────────────────────────────────────────────
    def load(self, entries: list[SxmFile], folder_path: str = ""):
        self._entries         = entries
        self._selected        = set()
        self._primary         = None
        self._card_colormaps  = {}
        self._card_processing = {}
        self._card_clip       = {}
        self._history         = {}
        self._load_token      = object()

        if folder_path:
            p = Path(folder_path)
            n_sxm  = sum(1 for e in entries if isinstance(e, SxmFile))
            n_vert = sum(1 for e in entries if isinstance(e, VertFile))
            parts = []
            if n_sxm:
                parts.append(f"{n_sxm} scan{'s' if n_sxm != 1 else ''}")
            if n_vert:
                parts.append(f"{n_vert} spec{'s' if n_vert != 1 else ''}")
            self._path_lbl.setText(f"{p.name}  ({', '.join(parts) if parts else '0 files'})")

        # clear grid
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards = {}

        if not entries:
            self._empty_lbl = QLabel("No .sxm or .VERT files found in this folder")
            self._empty_lbl.setAlignment(Qt.AlignCenter)
            self._empty_lbl.setFont(QFont("Helvetica", 12))
            self._grid.addWidget(self._empty_lbl, 0, 0)
            return

        cols = self._calc_cols()
        self._current_cols = cols
        dark = True  # will be updated via apply_theme if needed
        for entry in entries:
            if isinstance(entry, VertFile):
                card = SpecCard(entry, self._t)
            else:
                card = ScanCard(entry, self._t)
                self._card_colormaps[entry.stem] = DEFAULT_CMAP_KEY
            card.clicked.connect(self._on_card_click)
            card.double_clicked.connect(self._on_card_dbl)
            self._cards[entry.stem] = card

        # Populate the grid honouring the current filter (fresh loads default
        # to the filter previously set via apply_filter()).
        self._relayout_filtered()

        # load all thumbnails
        token = self._load_token
        for entry in entries:
            if isinstance(entry, VertFile):
                loader = SpecThumbnailLoader(entry, token,
                                             SpecCard.IMG_W, SpecCard.IMG_H)
            else:
                loader = ThumbnailLoader(entry, DEFAULT_CMAP_KEY, token,
                                         ScanCard.IMG_W, ScanCard.IMG_H)
            loader.signals.loaded.connect(self._on_thumb)
            self._pool.start(loader)

    HISTORY_MAX = 30

    def set_colormap_for_selection(self, colormap_key: str,
                                    clip_low: float = 1.0,
                                    clip_high: float = 99.0,
                                    processing: dict = None,
                                    push_history: bool = True) -> int:
        """Apply colormap, scale and optional processing to selected cards. Returns count updated."""
        if not self._selected:
            return 0
        token = self._load_token
        for stem in self._selected:
            entry = next((e for e in self._entries if e.stem == stem), None)
            card  = self._cards.get(stem)
            if entry and card and isinstance(entry, SxmFile):
                if push_history:
                    prev_cmap = self._card_colormaps.get(stem, DEFAULT_CMAP_KEY)
                    prev_clip = self._card_clip.get(stem, (1.0, 99.0))
                    prev_proc = self._card_processing.get(stem, {})
                    stack = self._history.setdefault(stem, [])
                    stack.append((prev_cmap, prev_clip[0], prev_clip[1],
                                   dict(prev_proc)))
                    if len(stack) > self.HISTORY_MAX:
                        del stack[0:len(stack) - self.HISTORY_MAX]
                # apply new state
                self._card_colormaps[stem]  = colormap_key
                self._card_clip[stem]       = (clip_low, clip_high)
                self._card_processing[stem] = dict(processing) if processing else {}
                loader = ThumbnailLoader(entry, colormap_key, token,
                                         ScanCard.IMG_W, ScanCard.IMG_H,
                                         clip_low, clip_high,
                                         processing=processing)
                loader.signals.loaded.connect(self._on_thumb)
                self._pool.start(loader)
        return len(self._selected)

    def set_processing_for_all_images(self, processing: dict,
                                      push_history: bool = True) -> int:
        """Apply thumbnail processing to every scan, preserving colour/clip."""
        token = self._load_token
        count = 0
        for entry in self._entries:
            if not isinstance(entry, SxmFile):
                continue
            card = self._cards.get(entry.stem)
            if not card:
                continue
            stem = entry.stem
            prev_cmap = self._card_colormaps.get(stem, DEFAULT_CMAP_KEY)
            prev_clip = self._card_clip.get(stem, (1.0, 99.0))
            prev_proc = self._card_processing.get(stem, {})
            if push_history:
                stack = self._history.setdefault(stem, [])
                stack.append((prev_cmap, prev_clip[0], prev_clip[1],
                              dict(prev_proc)))
                if len(stack) > self.HISTORY_MAX:
                    del stack[0:len(stack) - self.HISTORY_MAX]

            self._card_processing[stem] = dict(processing) if processing else {}
            loader = ThumbnailLoader(entry, prev_cmap, token,
                                     ScanCard.IMG_W, ScanCard.IMG_H,
                                     prev_clip[0], prev_clip[1],
                                     processing=processing or None)
            loader.signals.loaded.connect(self._on_thumb)
            self._pool.start(loader)
            count += 1
        return count

    def update_clip_for_selection(self, clip_low: float, clip_high: float) -> int:
        """Re-render selected cards with new clip, preserving their colormap and
        processing. Does NOT push to undo history (used by live scale slider)."""
        if not self._selected:
            return 0
        token = self._load_token
        for stem in self._selected:
            entry = next((e for e in self._entries if e.stem == stem), None)
            card  = self._cards.get(stem)
            if entry and card and isinstance(entry, SxmFile):
                cmap = self._card_colormaps.get(stem, DEFAULT_CMAP_KEY)
                proc = self._card_processing.get(stem, {})
                self._card_clip[stem] = (clip_low, clip_high)
                loader = ThumbnailLoader(entry, cmap, token,
                                         ScanCard.IMG_W, ScanCard.IMG_H,
                                         clip_low, clip_high,
                                         processing=proc or None)
                loader.signals.loaded.connect(self._on_thumb)
                self._pool.start(loader)
        return len(self._selected)

    def get_card_state(self, stem: str) -> tuple[str, tuple[float, float], dict]:
        """Return (colormap_key, (clip_low, clip_high), processing_dict) for a stem."""
        cmap = self._card_colormaps.get(stem, DEFAULT_CMAP_KEY)
        clip = self._card_clip.get(stem, (1.0, 99.0))
        proc = self._card_processing.get(stem, {})
        return cmap, clip, dict(proc)

    def undo_last(self, stems: set[str]) -> int:
        """Revert the last applied colormap/clip/processing for the given stems."""
        count = 0
        token = self._load_token
        for stem in stems:
            stack = self._history.get(stem)
            if not stack:
                continue
            prev_cmap, prev_low, prev_high, prev_proc = stack.pop()
            entry = next((e for e in self._entries if e.stem == stem), None)
            card  = self._cards.get(stem)
            if entry and card and isinstance(entry, SxmFile):
                self._card_colormaps[stem]  = prev_cmap
                self._card_clip[stem]       = (prev_low, prev_high)
                self._card_processing[stem] = prev_proc
                loader = ThumbnailLoader(entry, prev_cmap, token,
                                         ScanCard.IMG_W, ScanCard.IMG_H,
                                         prev_low, prev_high,
                                         processing=prev_proc or None)
                loader.signals.loaded.connect(self._on_thumb)
                self._pool.start(loader)
                count += 1
        return count

    def get_entries(self) -> list[Union[SxmFile, VertFile]]:
        return self._entries

    def get_selected(self) -> set[str]:
        return self._selected.copy()

    def get_primary(self) -> Optional[str]:
        return self._primary

    def apply_theme(self, t: dict):
        self._t = t
        self._content.setStyleSheet(f"background-color: {t['main_bg']};")
        self._toolbar.setStyleSheet(f"background-color: {t['main_bg']};")
        self._path_lbl.setStyleSheet(f"color: {t['sub_fg']}; background: transparent;")
        for card in self._cards.values():
            card.apply_theme(t)

    # ── Slots ──────────────────────────────────────────────────────────────────
    @Slot(str, QPixmap, object)
    def _on_thumb(self, stem: str, pixmap: QPixmap, token):
        if token is not self._load_token:
            return
        card = self._cards.get(stem)
        if card:
            card.set_pixmap(pixmap)

    def _on_card_click(self, entry: SxmFile, ctrl: bool):
        if ctrl:
            # toggle this card in/out of selection
            if entry.stem in self._selected:
                self._selected.discard(entry.stem)
                self._cards[entry.stem].set_selected(False)
                self._primary = next(iter(self._selected), None) if self._selected else None
            else:
                self._selected.add(entry.stem)
                self._cards[entry.stem].set_selected(True)
                self._primary = entry.stem
        else:
            # single select: deselect all others
            for stem in list(self._selected):
                c = self._cards.get(stem)
                if c:
                    c.set_selected(False)
            self._selected = {entry.stem}
            self._primary  = entry.stem
            self._cards[entry.stem].set_selected(True)

        self.selection_changed.emit(len(self._selected))
        if self._primary:
            primary_entry = next(
                (e for e in self._entries if e.stem == self._primary), None)
            if primary_entry:
                self.entry_selected.emit(primary_entry)

    def _on_card_dbl(self, entry: SxmFile):
        self.view_requested.emit(entry)

    # ── Layout helpers ─────────────────────────────────────────────────────────
    def _calc_cols(self) -> int:
        vw = self._scroll.viewport().width()
        if vw < 10:
            vw = 880
        return max(1, (vw - self.GAP) // (ScanCard.CARD_W + self.GAP))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._entries:
            QTimer.singleShot(60, self._relayout)

    def _relayout(self):
        if not self._entries:
            return
        new_cols = self._calc_cols()
        if new_cols == self._current_cols:
            return
        self._current_cols = new_cols
        self._relayout_filtered()

    def apply_filter(self, mode: str):
        """Switch between showing all entries, only images, or only spectra.

        Does not clear the selection or re-scan the folder; it only re-lays
        out the grid so hidden cards don't leave empty slots.
        """
        if mode not in ("all", "images", "spectra"):
            mode = "all"
        self._filter_mode = mode
        self._relayout_filtered()

    def _is_entry_visible(self, entry) -> bool:
        mode = self._filter_mode
        if mode == "images":
            return isinstance(entry, SxmFile)
        if mode == "spectra":
            return isinstance(entry, VertFile)
        return True  # "all"

    def _relayout_filtered(self):
        """Re-populate the grid with only cards matching the current filter.

        Selections are preserved on the cards themselves; we merely remove
        all widgets from the QGridLayout and re-add visible ones in row/col
        order, which avoids gaps caused by ``setVisible(False)``.
        """
        if not self._entries:
            return
        # Remove every card from the layout (do not delete the widgets).
        for card in self._cards.values():
            self._grid.removeWidget(card)
            card.setVisible(False)

        cols = self._calc_cols()
        self._current_cols = cols

        i = 0
        for entry in self._entries:
            card = self._cards.get(entry.stem)
            if not card or not self._is_entry_visible(entry):
                continue
            row, col = divmod(i, cols)
            self._grid.addWidget(card, row, col, Qt.AlignTop | Qt.AlignLeft)
            card.setVisible(True)
            i += 1


# ── Full-size image viewer dialog ─────────────────────────────────────────────
class _ZoomLabel(QLabel):
    """QLabel inside a scroll area that supports Ctrl+Wheel zoom and spec-position markers."""

    marker_clicked = Signal(object)  # emits VertFile when user clicks a marker
    pixel_clicked  = Signal(float, float)  # (frac_x, frac_y) — only when set_zero_mode is on

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._pixmap_orig: Optional[QPixmap] = None
        self._zoom = 1.0
        self._markers: list[dict] = []   # each: {frac_x, frac_y, entry}
        self._show_markers: bool = True
        self._set_zero_mode: bool = False
        self.setMouseTracking(True)

    def set_set_zero_mode(self, enabled: bool):
        self._set_zero_mode = bool(enabled)
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def set_source(self, pixmap: QPixmap):
        self._pixmap_orig = pixmap
        self._apply_zoom()

    def zoom_by(self, factor: float):
        self._zoom = max(0.25, min(8.0, self._zoom * factor))
        self._apply_zoom()

    def reset_zoom(self):
        self._zoom = 1.0
        self._apply_zoom()

    def set_markers(self, markers: list[dict]):
        self._markers = markers
        self.update()

    def set_show_markers(self, visible: bool):
        self._show_markers = visible
        self.update()

    def _apply_zoom(self):
        if self._pixmap_orig is None:
            return
        w = int(self._pixmap_orig.width()  * self._zoom)
        h = int(self._pixmap_orig.height() * self._zoom)
        scaled = self._pixmap_orig.scaled(w, h, Qt.KeepAspectRatio,
                                           Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self.resize(scaled.size())
        self.update()

    def _marker_px(self, frac_x: float, frac_y: float) -> tuple[int, int]:
        """Fractional image coords → label pixel coords."""
        return int(frac_x * self.width()), int(frac_y * self.height())

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._show_markers or not self._markers or self._pixmap_orig is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        r = 7
        for i, m in enumerate(self._markers):
            sx, sy = self._marker_px(m["frac_x"], m["frac_y"])
            painter.setBrush(QBrush(QColor("#FFD700")))
            painter.setPen(QPen(QColor("black"), 1.5))
            painter.drawEllipse(sx - r, sy - r, 2 * r, 2 * r)
            painter.setFont(QFont("Helvetica", 6, QFont.Bold))
            painter.setPen(QPen(QColor("black")))
            painter.drawText(QRect(sx - r, sy - r, 2 * r, 2 * r),
                             Qt.AlignCenter, str(i + 1))
        painter.end()

    def mouseMoveEvent(self, event):
        if self._show_markers and self._markers and self._pixmap_orig is not None:
            pos = event.pos()
            for m in self._markers:
                sx, sy = self._marker_px(m["frac_x"], m["frac_y"])
                if abs(pos.x() - sx) <= 10 and abs(pos.y() - sy) <= 10:
                    entry = m["entry"]
                    lines = [entry.stem]
                    if entry.sweep_type and entry.sweep_type != "unknown":
                        lines.append(entry.sweep_type)
                    if entry.bias_mv is not None:
                        lines.append(f"Bias: {entry.bias_mv:.0f} mV")
                    QToolTip.showText(event.globalPosition().toPoint(),
                                      "\n".join(lines), self)
                    return
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton and self._show_markers
                and self._markers and self._pixmap_orig is not None):
            pos = event.pos()
            for m in self._markers:
                sx, sy = self._marker_px(m["frac_x"], m["frac_y"])
                if abs(pos.x() - sx) <= 12 and abs(pos.y() - sy) <= 12:
                    self.marker_clicked.emit(m["entry"])
                    return
        if (event.button() == Qt.LeftButton and self._set_zero_mode
                and self._pixmap_orig is not None
                and self.width() > 0 and self.height() > 0):
            pos = event.pos()
            fx = max(0.0, min(1.0, pos.x() / float(self.width())))
            fy = max(0.0, min(1.0, pos.y() / float(self.height())))
            self.pixel_clicked.emit(fx, fy)
            return
        super().mousePressEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            self.zoom_by(1.12 if delta > 0 else 1 / 1.12)
            event.accept()
        else:
            super().wheelEvent(event)


class ProcessingControlPanel(QWidget):
    """Internal processing controls shared by Browse and Viewer."""

    QUICK_KEYS = ("align_rows",)

    def __init__(self, mode: str, parent=None):
        super().__init__(parent)
        if mode not in ("browse_quick", "viewer_full"):
            raise ValueError(f"Unknown processing panel mode: {mode}")
        self._mode = mode
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 0, 2)
        lay.setSpacing(4)

        def _combo_row(label: str, items: list[str]) -> QComboBox:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFont(QFont("Helvetica", 8))
            lbl.setFixedWidth(90)
            cb = QComboBox()
            cb.addItems(items)
            cb.setFont(QFont("Helvetica", 8))
            row.addWidget(lbl)
            row.addWidget(cb, 1)
            lay.addLayout(row)
            return cb

        def _sub_slider(label: str, mn: int, mx: int, init: int,
                        fmt="{v}") -> tuple[QWidget, QSlider, QLabel]:
            w = QWidget()
            rl = QHBoxLayout(w)
            rl.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label)
            lbl.setFont(QFont("Helvetica", 8))
            lbl.setFixedWidth(56)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(mn, mx)
            sl.setValue(init)
            val_lbl = QLabel(fmt.format(v=init))
            val_lbl.setFont(QFont("Helvetica", 8))
            val_lbl.setFixedWidth(30)
            sl.valueChanged.connect(
                lambda v, vl=val_lbl, f=fmt: vl.setText(f.format(v=v)))
            rl.addWidget(lbl)
            rl.addWidget(sl, 1)
            rl.addWidget(val_lbl)
            return w, sl, val_lbl

        line_lbl = QLabel("Line corrections")
        line_lbl.setFont(QFont("Helvetica", 7, QFont.Bold))
        line_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(line_lbl)

        self._align_combo = _combo_row("Align rows:", ["None", "Median", "Mean"])

        if self._mode == "browse_quick":
            lay.addStretch()
            return

        bg_lbl = QLabel("Background subtraction")
        bg_lbl.setFont(QFont("Helvetica", 7, QFont.Bold))
        bg_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(bg_lbl)

        self._bg_combo = _combo_row(
            "Background:",
            ["None", "Plane", "Quadratic", "Cubic", "Quartic"],
        )
        self._bg_step_cb = QCheckBox("Step-edge tolerant (mask high-grad)")
        self._bg_step_cb.setFont(QFont("Helvetica", 8))
        lay.addWidget(self._bg_step_cb)

        self._facet_cb = QCheckBox("Facet level (flat-terrace ref)")
        self._facet_cb.setFont(QFont("Helvetica", 8))
        lay.addWidget(self._facet_cb)

        smooth_lbl = QLabel("Smoothing / Edge detection")
        smooth_lbl.setFont(QFont("Helvetica", 7, QFont.Bold))
        smooth_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(smooth_lbl)

        self._smooth_combo = _combo_row("Smooth:", ["None", "Gaussian"])
        self._smooth_sigma_w, self._smooth_sigma_sl, _ = _sub_slider(
            "sigma (px):", 1, 20, 1, "{v}")
        lay.addWidget(self._smooth_sigma_w)
        self._smooth_sigma_w.setVisible(False)
        self._smooth_combo.currentIndexChanged.connect(
            lambda i: self._smooth_sigma_w.setVisible(i != 0))

        self._edge_combo = _combo_row("Edge detect:", ["None", "Laplacian", "LoG", "DoG"])
        self._edge_sigma_w, self._edge_sigma_sl, _ = _sub_slider(
            "sigma (px):", 1, 20, 1, "{v}")
        lay.addWidget(self._edge_sigma_w)
        self._edge_sigma_w.setVisible(False)
        self._edge_combo.currentIndexChanged.connect(
            lambda i: self._edge_sigma_w.setVisible(i != 0))

        fft_lbl = QLabel("FFT filter")
        fft_lbl.setFont(QFont("Helvetica", 7, QFont.Bold))
        fft_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(fft_lbl)

        self._fft_combo = _combo_row("FFT filter:", ["None", "Low-pass", "High-pass"])
        self._fft_cutoff_widget, self._fft_sl, _ = _sub_slider(
            "Cutoff:", 1, 50, 10, "{v}%")
        lay.addWidget(self._fft_cutoff_widget)
        self._fft_cutoff_widget.setVisible(False)
        self._fft_combo.currentIndexChanged.connect(
            lambda i: self._fft_cutoff_widget.setVisible(i != 0))

        self._fft_soft_cb = QCheckBox("Soft border (Tukey taper)")
        self._fft_soft_cb.setFont(QFont("Helvetica", 8))
        lay.addWidget(self._fft_soft_cb)

        undistort_lbl = QLabel("Linear undistort (drift)")
        undistort_lbl.setFont(QFont("Helvetica", 7, QFont.Bold))
        undistort_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(undistort_lbl)

        self._undistort_shear_w, self._undistort_shear_sl, _ = _sub_slider(
            "Shear x:", -20, 20, 0, "{v}px")
        lay.addWidget(self._undistort_shear_w)
        self._undistort_scale_w, self._undistort_scale_sl, _ = _sub_slider(
            "Scale y:", 80, 120, 100, "{v}%")
        lay.addWidget(self._undistort_scale_w)

    def state(self) -> dict:
        align_map = {0: None, 1: "median", 2: "mean"}
        cfg = {
            "align_rows": align_map[self._align_combo.currentIndex()],
        }
        if self._mode == "browse_quick":
            return {k: cfg[k] for k in self.QUICK_KEYS}

        bg_map = {0: None, 1: 1, 2: 2, 3: 3, 4: 4}
        fft_map = {0: None, 1: "low_pass", 2: "high_pass"}
        edge_map = {0: None, 1: "laplacian", 2: "log", 3: "dog"}
        smooth_i = self._smooth_combo.currentIndex()
        edge_i = self._edge_combo.currentIndex()
        fft_idx = self._fft_combo.currentIndex()
        shear_x = float(self._undistort_shear_sl.value())
        scale_y = self._undistort_scale_sl.value() / 100.0
        cfg.update({
            "bg_order": bg_map[self._bg_combo.currentIndex()],
            "bg_step_tolerance": self._bg_step_cb.isChecked(),
            "facet_level": self._facet_cb.isChecked(),
            "smooth_sigma": self._smooth_sigma_sl.value() if smooth_i != 0 else None,
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
            "linear_undistort": (shear_x != 0.0 or scale_y != 1.0),
            "undistort_shear_x": shear_x,
            "undistort_scale_y": scale_y,
        })
        return cfg

    def set_state(self, state: dict | None) -> None:
        state = state or {}
        self._align_combo.setCurrentIndex(
            {None: 0, "median": 1, "mean": 2}.get(state.get("align_rows"), 0))
        if self._mode == "browse_quick":
            return

        self._bg_combo.setCurrentIndex(
            {None: 0, 1: 1, 2: 2, 3: 3, 4: 4}.get(state.get("bg_order"), 0))
        self._bg_step_cb.setChecked(bool(state.get("bg_step_tolerance", False)))
        self._facet_cb.setChecked(bool(state.get("facet_level", False)))

        sigma = state.get("smooth_sigma")
        if sigma:
            self._smooth_combo.setCurrentIndex(1)
            self._smooth_sigma_sl.setValue(int(sigma))
        else:
            self._smooth_combo.setCurrentIndex(0)

        edge = state.get("edge_method")
        self._edge_combo.setCurrentIndex(
            {None: 0, "laplacian": 1, "log": 2, "dog": 3}.get(edge, 0))
        self._edge_sigma_sl.setValue(int(state.get("edge_sigma", 1)))

        fft_mode = state.get("fft_mode")
        self._fft_combo.setCurrentIndex(
            {None: 0, "low_pass": 1, "high_pass": 2}.get(fft_mode, 0))
        self._fft_sl.setValue(int(round(float(state.get("fft_cutoff", 0.10)) * 100)))
        self._fft_soft_cb.setChecked(bool(state.get("fft_soft_border", False)))

        self._undistort_shear_sl.setValue(int(state.get("undistort_shear_x", 0)))
        self._undistort_scale_sl.setValue(
            int(round(float(state.get("undistort_scale_y", 1.0)) * 100)))


class ImageViewerDialog(QDialog):
    """Double-click viewer with scroll/zoom, histogram, clip sliders, processing, export."""

    def __init__(self, entry: SxmFile, entries: list[SxmFile],
                 colormap: str, t: dict, parent=None,
                 clip_low: float = 1.0, clip_high: float = 99.0,
                 processing: dict = None):
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
        self._raw_arr: Optional[np.ndarray] = None
        self._display_arr: Optional[np.ndarray] = None  # raw or processed, for histogram/export
        self._spec_markers: list[dict] = []
        self._scan_header: dict = {}
        self._scan_range_m: Optional[tuple] = None
        self._scan_shape: Optional[tuple] = None
        self._scan_format: str = ""

        self._build()
        self._processing_panel.set_state(self._processing)
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
        self._scroll_area.setWidget(self._zoom_lbl)
        left_lay.addWidget(self._scroll_area, 1)

        zoom_row = QHBoxLayout()
        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFixedSize(28, 24)
        zoom_out_btn.setFont(QFont("Helvetica", 11))
        zoom_out_btn.clicked.connect(lambda: self._zoom_lbl.zoom_by(1 / 1.25))
        zoom_reset_btn = QPushButton("1:1")
        zoom_reset_btn.setFixedSize(36, 24)
        zoom_reset_btn.setFont(QFont("Helvetica", 9))
        zoom_reset_btn.clicked.connect(self._zoom_lbl.reset_zoom)
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedSize(28, 24)
        zoom_in_btn.setFont(QFont("Helvetica", 11))
        zoom_in_btn.clicked.connect(lambda: self._zoom_lbl.zoom_by(1.25))
        zoom_hint = QLabel("Ctrl+scroll to zoom")
        zoom_hint.setFont(QFont("Helvetica", 8))
        zoom_row.addWidget(zoom_out_btn)
        zoom_row.addWidget(zoom_reset_btn)
        zoom_row.addWidget(zoom_in_btn)
        zoom_row.addWidget(zoom_hint)
        zoom_row.addStretch()
        left_lay.addLayout(zoom_row)

        splitter.addWidget(left)

        # ── Right: control panel ───────────────────────────────────────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setFixedWidth(280)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 4, 8, 4)
        right_lay.setSpacing(6)

        # channel selector
        ch_lbl = QLabel("Channel")
        ch_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        right_lay.addWidget(ch_lbl)
        self._ch_cb = QComboBox()
        self._ch_cb.addItems(PLANE_NAMES)
        self._ch_cb.setFont(QFont("Helvetica", 9))
        self._ch_cb.currentIndexChanged.connect(self._on_channel_changed)
        right_lay.addWidget(self._ch_cb)
        right_lay.addWidget(_sep())

        # histogram
        hist_lbl = QLabel("Histogram — drag the red/green lines to clip")
        hist_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        hist_lbl.setWordWrap(True)
        right_lay.addWidget(hist_lbl)

        self._fig  = Figure(figsize=(2.8, 2.4), dpi=80)
        self._fig.patch.set_alpha(0)
        self._ax   = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setFixedHeight(220)
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

        right_lay.addWidget(_sep())

        # clip sliders (percentile — live)
        clip_lbl = QLabel("Clip (percentile)")
        clip_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        right_lay.addWidget(clip_lbl)

        def _make_clip_row(label, init_val, mn, mx):
            row = QHBoxLayout()
            l = QLabel(label)
            l.setFont(QFont("Helvetica", 8))
            l.setFixedWidth(32)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(mn, mx)
            sl.setValue(int(init_val))
            vl = QLabel(f"{int(init_val)}%")
            vl.setFont(QFont("Helvetica", 8))
            vl.setFixedWidth(32)
            sl.valueChanged.connect(lambda v, vl=vl: vl.setText(f"{v}%"))
            row.addWidget(l)
            row.addWidget(sl, 1)
            row.addWidget(vl)
            right_lay.addLayout(row)
            return sl

        self._low_sl  = _make_clip_row("Low:", self._clip_low,   0,  20)
        self._high_sl = _make_clip_row("High:", self._clip_high, 80, 100)
        # live update on release (avoid re-rendering on every intermediate tick)
        self._low_sl.sliderReleased.connect(self._on_slider_clip)
        self._high_sl.sliderReleased.connect(self._on_slider_clip)

        # Auto-reset button
        auto_row = QHBoxLayout()
        auto_btn = QPushButton("Auto")
        auto_btn.setFont(QFont("Helvetica", 8))
        auto_btn.setFixedHeight(22)
        auto_btn.setToolTip("Reset to 1%–99% percentile autoscale")
        auto_btn.clicked.connect(self._on_auto_clip)
        auto_row.addStretch()
        auto_row.addWidget(auto_btn)
        right_lay.addLayout(auto_row)

        # Å / pA value readout for current clip
        self._clip_val_lbl = QLabel("")
        self._clip_val_lbl.setFont(QFont("Helvetica", 8))
        self._clip_val_lbl.setAlignment(Qt.AlignCenter)
        right_lay.addWidget(self._clip_val_lbl)

        right_lay.addWidget(_sep())

        # processing
        proc_toggle = QPushButton("[+] Processing")
        proc_toggle.setFlat(True)
        proc_toggle.setFont(QFont("Helvetica", 9, QFont.Bold))
        proc_toggle.setCursor(QCursor(Qt.PointingHandCursor))
        right_lay.addWidget(proc_toggle)

        self._processing_widget = QWidget()
        processing_lay = QVBoxLayout(self._processing_widget)
        processing_lay.setContentsMargins(2, 2, 0, 2)
        processing_lay.setSpacing(4)

        self._processing_panel = ProcessingControlPanel("viewer_full")
        processing_lay.addWidget(self._processing_panel)

        proc_apply_btn = QPushButton("Apply processing")
        proc_apply_btn.setFont(QFont("Helvetica", 8))
        proc_apply_btn.setFixedHeight(24)
        proc_apply_btn.setObjectName("accentBtn")
        proc_apply_btn.clicked.connect(self._on_apply_processing)
        processing_lay.addWidget(proc_apply_btn)

        self._processing_widget.setVisible(False)
        right_lay.addWidget(self._processing_widget)

        proc_toggle.clicked.connect(lambda: (
            self._processing_widget.setVisible(not self._processing_widget.isVisible()),
            proc_toggle.setText(
                "[-] Processing" if self._processing_widget.isVisible()
                else "[+] Processing")
        ))

        right_lay.addWidget(_sep())

        # spec position overlay toggle
        self._spec_show_cb = QCheckBox("Show spec positions")
        self._spec_show_cb.setFont(QFont("Helvetica", 8))
        self._spec_show_cb.setChecked(True)
        self._spec_show_cb.toggled.connect(self._on_spec_show_toggled)
        right_lay.addWidget(self._spec_show_cb)

        # Gwyddion-style "Set Z=0 here": toggle, then click on the image.
        self._set_zero_btn = QPushButton("Set Z=0 (click on image)")
        self._set_zero_btn.setCheckable(True)
        self._set_zero_btn.setFont(QFont("Helvetica", 8))
        self._set_zero_btn.setFixedHeight(24)
        self._set_zero_btn.toggled.connect(self._zoom_lbl.set_set_zero_mode)
        right_lay.addWidget(self._set_zero_btn)

        self._set_zero_clear_btn = QPushButton("Clear Z=0 reference")
        self._set_zero_clear_btn.setFont(QFont("Helvetica", 8))
        self._set_zero_clear_btn.setFixedHeight(22)
        self._set_zero_clear_btn.clicked.connect(self._on_clear_set_zero)
        right_lay.addWidget(self._set_zero_clear_btn)

        self._zoom_lbl.marker_clicked.connect(self._on_marker_clicked)
        self._zoom_lbl.pixel_clicked.connect(self._on_set_zero_pick)

        right_lay.addWidget(_sep())

        # save PNG copy
        save_btn = QPushButton("⬇ Save PNG copy…")
        save_btn.setFont(QFont("Helvetica", 8, QFont.Bold))
        save_btn.setFixedHeight(26)
        save_btn.setObjectName("accentBtn")
        save_btn.clicked.connect(self._on_save_png)
        right_lay.addWidget(save_btn)

        self._status_lbl = QLabel("")
        self._status_lbl.setFont(QFont("Helvetica", 8))
        self._status_lbl.setWordWrap(True)
        right_lay.addWidget(self._status_lbl)

        right_lay.addStretch()

        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setSizes([900, 280])
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
            self._load_current()

    def _go_next(self):
        if self._idx < len(self._entries) - 1:
            self._idx += 1
            self._load_current()

    # ── Load / render ──────────────────────────────────────────────────────────
    def _load_current(self):
        entry = self._entries[self._idx]
        self._title_lbl.setText(entry.stem)
        self.setWindowTitle(entry.stem)
        self._pos_lbl.setText(f"{self._idx + 1} / {len(self._entries)}")
        self._prev_btn.setEnabled(self._idx > 0)
        self._next_btn.setEnabled(self._idx < len(self._entries) - 1)
        self._zoom_lbl.setText("Loading…")
        self._zoom_lbl.setPixmap(QPixmap())
        self._zoom_lbl.set_markers([])
        # load raw array; compute display array (with processing if active)
        try:
            _scan = load_scan(entry.path)
            idx = self._ch_cb.currentIndex()
            self._raw_arr = _scan.planes[idx] if idx < _scan.n_planes else None
            self._scan_header  = _scan.header or {}
            self._scan_range_m = _scan.scan_range_m
            self._scan_shape   = _scan.planes[0].shape if _scan.planes else None
            self._scan_format  = entry.source_format
        except Exception:
            self._raw_arr      = None
            self._scan_header  = {}
            self._scan_range_m = None
            self._scan_shape   = None
            self._scan_format  = ""
        # display array: raw with processing applied (no grain overlay — that's visual only)
        if self._raw_arr is not None and self._processing:
            try:
                self._display_arr = _apply_processing(self._raw_arr, self._processing)
            except Exception:
                self._display_arr = self._raw_arr
        else:
            self._display_arr = self._raw_arr
        self._update_histogram()
        self._load_spec_markers(entry)
        # Resolve display limits (percentile or manual) from current array
        vmin, vmax = self._drs.resolve(self._display_arr) if self._display_arr is not None else (None, None)
        # load rendered image
        self._token = object()
        loader = ViewerLoader(entry, self._colormap, self._token, 900, 800,
                              self._ch_cb.currentIndex(),
                              self._clip_low, self._clip_high,
                              self._processing or None,
                              vmin=vmin, vmax=vmax)
        loader.signals.loaded.connect(self._on_loaded)
        self._pool.start(loader)

    def _channel_unit(self) -> tuple[float, str, str]:
        """Return (scale, unit_label, axis_label) for the current channel.
        Z channels (0,1) → Å; current channels (2,3) → pA."""
        idx = self._ch_cb.currentIndex()
        if idx < 2:
            return 1e10, "Å", "Height"
        return 1e12, "pA", "Current"

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
            self._ax.set_xlim(x_min, x_max)
        self._low_line  = self._ax.axvline(lo_phys, color="#f38ba8",
                                            linewidth=1.6, picker=6)
        self._high_line = self._ax.axvline(hi_phys, color="#a6e3a1",
                                            linewidth=1.6, picker=6)

        self._ax.tick_params(colors=fg, labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(self._t.get("sep", "#45475a"))
        self._ax.set_xlabel(f"{axis_label} [{unit}]", fontsize=8, color=fg)
        self._ax.set_ylabel("Count", fontsize=8, color=fg)

        self._fig.tight_layout(pad=0.4)
        self._canvas.draw_idle()

        self._clip_val_lbl.setText(
            f"{lo_phys:.3g} {unit}  →  {hi_phys:.3g} {unit}")

    # ── Spec position overlay ─────────────────────────────────────────────────
    def _load_spec_markers(self, entry):
        """Scan the image's parent folder for spec files and set markers."""
        from probeflow.file_type import FileType, sniff_file_type
        from probeflow.spec_io import read_spec_file
        from probeflow.spec_plot import spec_position_to_pixel, _parse_sxm_offset

        self._spec_markers = []
        self._zoom_lbl.set_markers([])

        if self._scan_range_m is None or self._scan_shape is None:
            return

        try:
            folder = entry.path.parent
            spec_types = (FileType.CREATEC_SPEC, FileType.NANONIS_SPEC)
            candidates = [
                f for f in sorted(folder.iterdir())
                if f.is_file() and sniff_file_type(f) in spec_types
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
                        continue
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

    def _on_marker_clicked(self, entry):
        dlg = SpecViewerDialog(entry, self._t, self)
        dlg.exec()

    def _on_set_zero_pick(self, frac_x: float, frac_y: float):
        """Handle a click on the image while Set Z=0 mode is active."""
        if self._raw_arr is None:
            return
        Ny, Nx = self._raw_arr.shape
        x_px = int(round(frac_x * (Nx - 1)))
        y_px = int(round(frac_y * (Ny - 1)))
        x_px = max(0, min(x_px, Nx - 1))
        y_px = max(0, min(y_px, Ny - 1))
        self._processing['set_zero_xy'] = (x_px, y_px)
        self._processing['set_zero_patch'] = 1
        # Toggle off so the next click navigates normally.
        if self._set_zero_btn.isChecked():
            self._set_zero_btn.setChecked(False)
        self._status_lbl.setText(f"Z=0 set at pixel ({x_px}, {y_px})")
        self._load_current()

    def _on_clear_set_zero(self):
        if 'set_zero_xy' in self._processing:
            self._processing.pop('set_zero_xy', None)
            self._processing.pop('set_zero_patch', None)
            self._status_lbl.setText("Z=0 reference cleared")
            self._load_current()

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
        self._load_current()

    def _on_slider_clip(self):
        self._clip_low  = float(self._low_sl.value())
        self._clip_high = float(self._high_sl.value())
        self._drs.set_percentile(self._clip_low, self._clip_high)
        self._load_current()

    def _on_auto_clip(self):
        """Reset to 1%–99% percentile autoscale."""
        self._drs.reset()
        self._clip_low  = 1.0
        self._clip_high = 99.0
        self._low_sl.blockSignals(True)
        self._high_sl.blockSignals(True)
        self._low_sl.setValue(1)
        self._high_sl.setValue(99)
        self._low_sl.blockSignals(False)
        self._high_sl.blockSignals(False)
        self._load_current()

    def _on_channel_changed(self, _: int):
        # Different channels have different physical units — reset manual limits.
        self._drs.reset(self._clip_low, self._clip_high)
        self._load_current()

    @Slot(QPixmap, object)
    def _on_loaded(self, pixmap: QPixmap, token):
        if token is not self._token:
            return
        self._zoom_lbl.setText("")
        self._zoom_lbl.set_source(pixmap)

    # ── Controls ───────────────────────────────────────────────────────────────
    def _on_apply_processing(self):
        set_zero = {
            key: self._processing[key]
            for key in ("set_zero_xy", "set_zero_patch")
            if key in self._processing
        }
        self._processing = self._processing_panel.state()
        self._processing.update(set_zero)
        self._load_current()

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
                    from probeflow.export_provenance import ExportProvenance
                    ch_idx = self._ch_cb.currentIndex()
                    ps = processing_state_from_gui(self._processing or {})
                    provenance = ExportProvenance.from_scan_export(
                        _scan,
                        channel_index=ch_idx,
                        channel_name=PLANE_NAMES[ch_idx] if 0 <= ch_idx < len(PLANE_NAMES) else None,
                        processing_state=ps,
                        display_state=self._drs,
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


# ── Browse tool panel (LEFT) ──────────────────────────────────────────────────
class BrowseToolPanel(QWidget):
    """Left-side control panel: folder, colormap, scale, processing, export."""
    open_folder_requested      = Signal()
    colormap_apply_requested   = Signal(str)
    scale_changed              = Signal(float, float)
    processing_apply_requested = Signal(dict)
    processing_apply_all_requested = Signal(dict)
    autoclip_requested         = Signal()
    export_requested           = Signal()
    undo_requested             = Signal()
    filter_changed             = Signal(str)   # "all" | "images" | "spectra"

    def __init__(self, t: dict, cfg: dict, parent=None):
        super().__init__(parent)
        self._t            = t
        self._clip_low     = cfg.get("clip_low",  1.0)
        self._clip_high    = cfg.get("clip_high", 99.0)
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

        # ── Colormap ───────────────────────────────────────────────────────────
        cm_lbl = QLabel("Colormap")
        cm_lbl.setFont(QFont("Helvetica", 11, QFont.Bold))
        lay.addWidget(cm_lbl)

        self.cmap_cb = QComboBox()
        self.cmap_cb.addItems(CMAP_NAMES)
        self.cmap_cb.setCurrentText(cfg.get("colormap", DEFAULT_CMAP_LABEL))
        self.cmap_cb.setFont(QFont("Helvetica", 10))
        lay.addWidget(self.cmap_cb)

        self._apply_btn = QPushButton("Apply to selection")
        self._apply_btn.setFont(QFont("Helvetica", 10))
        self._apply_btn.setFixedHeight(30)
        self._apply_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._apply_btn.setObjectName("accentBtn")
        self._apply_btn.clicked.connect(self._on_apply)
        lay.addWidget(self._apply_btn)

        self._sel_hint = QLabel("Select images first (Ctrl+click for multi-select)")
        self._sel_hint.setFont(QFont("Helvetica", 9))
        self._sel_hint.setWordWrap(True)
        lay.addWidget(self._sel_hint)
        lay.addWidget(_sep())

        # ── Display Scale ──────────────────────────────────────────────────────
        scale_hdr = QLabel("Display Scale")
        scale_hdr.setFont(QFont("Helvetica", 9, QFont.Bold))
        lay.addWidget(scale_hdr)

        def _slider_row(label: str, init_val: float, mn: int, mx: int, callback):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFont(QFont("Helvetica", 8))
            lbl.setFixedWidth(36)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(mn, mx)
            sl.setValue(int(init_val))
            val_lbl = QLabel(f"{init_val:.0f}%")
            val_lbl.setFont(QFont("Helvetica", 8))
            val_lbl.setFixedWidth(36)
            def _upd(v, vl=val_lbl, cb=callback):
                vl.setText(f"{v}%")
                cb(v)
            sl.valueChanged.connect(_upd)
            row.addWidget(lbl)
            row.addWidget(sl, 1)
            row.addWidget(val_lbl)
            lay.addLayout(row)
            return sl

        self._low_slider  = _slider_row("Low:", cfg.get("clip_low",  1.0),  0,  20, self._on_low_changed)
        self._high_slider = _slider_row("High:", cfg.get("clip_high", 99.0), 80, 100, self._on_high_changed)
        lay.addWidget(_sep())

        # ── Quick corrections (collapsible) ───────────────────────────────────
        self._proc_toggle = QPushButton("[+] Quick corrections")
        self._proc_toggle.setFlat(True)
        self._proc_toggle.setFont(QFont("Helvetica", 9, QFont.Bold))
        self._proc_toggle.setCursor(QCursor(Qt.PointingHandCursor))
        self._proc_toggle.clicked.connect(self._toggle_proc)
        lay.addWidget(self._proc_toggle)

        self._proc_widget = QWidget()
        proc_lay = QVBoxLayout(self._proc_widget)
        proc_lay.setContentsMargins(4, 2, 0, 2)
        proc_lay.setSpacing(4)

        self._processing_panel = ProcessingControlPanel("browse_quick")
        proc_lay.addWidget(self._processing_panel)

        # ── Apply + Auto-clip ──────────────────────────────────────────────────
        proc_lay.addWidget(_sep())

        self._autoclip_btn = QPushButton("Auto clip (GMM)")
        self._autoclip_btn.setFont(QFont("Helvetica", 8))
        self._autoclip_btn.setFixedHeight(26)
        self._autoclip_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._autoclip_btn.clicked.connect(self.autoclip_requested.emit)
        proc_lay.addWidget(self._autoclip_btn)

        self._proc_apply_btn = QPushButton("Apply to selected thumbnails")
        self._proc_apply_btn.setFont(QFont("Helvetica", 8))
        self._proc_apply_btn.setFixedHeight(26)
        self._proc_apply_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._proc_apply_btn.setObjectName("accentBtn")
        self._proc_apply_btn.clicked.connect(self._on_proc_apply)
        proc_lay.addWidget(self._proc_apply_btn)

        self._proc_apply_all_btn = QPushButton("Apply to all thumbnails")
        self._proc_apply_all_btn.setFont(QFont("Helvetica", 8))
        self._proc_apply_all_btn.setFixedHeight(26)
        self._proc_apply_all_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._proc_apply_all_btn.clicked.connect(self._on_proc_apply_all)
        proc_lay.addWidget(self._proc_apply_all_btn)

        self._undo_btn = QPushButton("↩ Undo last thumbnail change")
        self._undo_btn.setFont(QFont("Helvetica", 8))
        self._undo_btn.setFixedHeight(26)
        self._undo_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._undo_btn.clicked.connect(self.undo_requested.emit)
        proc_lay.addWidget(self._undo_btn)

        lay.addWidget(self._proc_widget)
        self._proc_widget.setVisible(False)
        lay.addWidget(_sep())

        # ── Export ─────────────────────────────────────────────────────────────
        self._export_btn = QPushButton("\u2b07 Export PNG\u2026")
        self._export_btn.setFont(QFont("Helvetica", 9, QFont.Bold))
        self._export_btn.setFixedHeight(30)
        self._export_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._export_btn.setObjectName("accentBtn")
        self._export_btn.clicked.connect(self.export_requested.emit)
        lay.addWidget(self._export_btn)

        lay.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

    # ── Slots ──────────────────────────────────────────────────────────────────
    def _on_low_changed(self, v: int):
        self._clip_low = float(v)
        self.scale_changed.emit(self._clip_low, self._clip_high)

    def _on_high_changed(self, v: int):
        self._clip_high = float(v)
        self.scale_changed.emit(self._clip_low, self._clip_high)

    def _on_apply(self):
        cmap_key = CMAP_KEY.get(self.cmap_cb.currentText(), DEFAULT_CMAP_KEY)
        self.colormap_apply_requested.emit(cmap_key)

    def _toggle_proc(self):
        vis = not self._proc_widget.isVisible()
        self._proc_widget.setVisible(vis)
        self._proc_toggle.setText(
            "[-] Quick corrections" if vis else "[+] Quick corrections")

    def _on_proc_apply(self):
        self.processing_apply_requested.emit(self._processing_panel.state())

    def _on_proc_apply_all(self):
        self.processing_apply_all_requested.emit(self._processing_panel.state())

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

    def get_clip_values(self) -> tuple[float, float]:
        return self._clip_low, self._clip_high

    def update_selection_hint(self, n: int):
        if n == 0:
            self._sel_hint.setText("Select images first (Ctrl+click for multi-select)")
        elif n == 1:
            self._sel_hint.setText("1 image selected")
        else:
            self._sel_hint.setText(f"{n} images selected")

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
        self._clip_low  = cfg.get("clip_low",  1.0)
        self._clip_high = cfg.get("clip_high", 99.0)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 6)
        lay.setSpacing(4)

        self.name_lbl = QLabel("No scan selected")
        self.name_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        self.name_lbl.setWordWrap(True)
        lay.addWidget(self.name_lbl)

        # Quick-info grid (pixels, size, bias, setpoint). Filled by show_entry().
        qi_grid = QGridLayout()
        qi_grid.setSpacing(4)
        qi_grid.setContentsMargins(0, 2, 0, 2)
        self._qi: dict[str, QLabel] = {}
        _QI_ROWS = [("Pixels", "pixels"), ("Size", "size"),
                    ("Bias",   "bias"),   ("Setp.", "setp")]
        for i, (title, key) in enumerate(_QI_ROWS):
            r, c = divmod(i, 2)
            t_lbl = QLabel(title + ":")
            t_lbl.setFont(QFont("Helvetica", 8))
            v_lbl = QLabel("—")
            v_lbl.setFont(QFont("Helvetica", 8, QFont.Bold))
            qi_grid.addWidget(t_lbl, r, c * 2)
            qi_grid.addWidget(v_lbl, r, c * 2 + 1)
            self._qi[key] = v_lbl
        lay.addLayout(qi_grid)
        lay.addWidget(_sep())

        ch_hdr = QLabel("Channels")
        ch_hdr.setFont(QFont("Helvetica", 11, QFont.Bold))
        lay.addWidget(ch_hdr)

        ch_grid = QGridLayout()
        ch_grid.setSpacing(8)
        ch_grid.setContentsMargins(0, 0, 0, 0)
        self._ch_img_lbls:  list[QLabel] = []
        self._ch_name_lbls: list[QLabel] = []
        for i, name in enumerate(PLANE_NAMES):
            r, c = divmod(i, 2)
            cell = QWidget()
            cell_lay = QVBoxLayout(cell)
            cell_lay.setContentsMargins(0, 0, 0, 0)
            cell_lay.setSpacing(2)
            img_lbl = QLabel()
            img_lbl.setFixedSize(128, 102)
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setFrameShape(QFrame.StyledPanel)
            nm_lbl = QLabel(name)
            nm_lbl.setFont(QFont("Helvetica", 9))
            nm_lbl.setAlignment(Qt.AlignCenter)
            cell_lay.addWidget(img_lbl)
            cell_lay.addWidget(nm_lbl)
            ch_grid.addWidget(cell, r, c)
            self._ch_img_lbls.append(img_lbl)
            self._ch_name_lbls.append(nm_lbl)
        lay.addLayout(ch_grid)
        lay.addWidget(_sep())

        meta_hdr_row = QHBoxLayout()
        meta_hdr = QLabel("Metadata")
        meta_hdr.setFont(QFont("Helvetica", 11, QFont.Bold))
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search…")
        self.search_box.setFont(QFont("Helvetica", 10))
        self.search_box.setFixedHeight(28)
        self.search_box.textChanged.connect(self._filter_meta)
        meta_hdr_row.addWidget(meta_hdr)
        meta_hdr_row.addStretch()
        meta_hdr_row.addWidget(self.search_box)
        lay.addLayout(meta_hdr_row)

        self.meta_table = QTableWidget(0, 2)
        self.meta_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.meta_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.meta_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.meta_table.verticalHeader().setVisible(False)
        self.meta_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.meta_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.meta_table.setAlternatingRowColors(True)
        self.meta_table.setFont(QFont("Helvetica", 10))
        self.meta_table.verticalHeader().setDefaultSectionSize(22)
        self.meta_table.setShowGrid(False)
        lay.addWidget(self.meta_table, 1)

    # ── Public API ─────────────────────────────────────────────────────────────
    def show_entry(self, entry: SxmFile, colormap_key: str,
                    processing: dict = None):
        self.name_lbl.setText(entry.stem)
        self._qi["pixels"].setText(f"{entry.Nx} × {entry.Ny}")
        self._qi["size"].setText(f"{entry.scan_nm:.1f} nm" if entry.scan_nm is not None else "—")
        self._qi["bias"].setText(f"{entry.bias_mv:.0f} mV" if entry.bias_mv is not None else "—")
        self._qi["setp"].setText(f"{entry.current_pa:.1f} pA" if entry.current_pa is not None else "—")
        self.load_channels(entry, colormap_key, processing)
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
        from .spec_io import parse_spec_header
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

    def update_clip(self, clip_low: float, clip_high: float):
        self._clip_low  = clip_low
        self._clip_high = clip_high

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
        for i in range(4):
            loader = ChannelLoader(entry, i, colormap_key,
                                   self._ch_token, 124, 98, sigs,
                                   self._clip_low, self._clip_high,
                                   processing=processing)
            self._pool.start(loader)

    # Back-compat alias used internally
    _load_channels = load_channels

    @Slot(int, QPixmap, object)
    def _on_ch_loaded(self, idx: int, pixmap: QPixmap, token):
        if token is not self._ch_token:
            return
        lbl = self._ch_img_lbls[idx]
        lbl.setPixmap(pixmap.scaled(lbl.width(), lbl.height(),
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation))

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


# ── Features tab (Phase 3: particles / template / lattice) ──────────────────

class _FeaturesWorkerSignals(QObject):
    finished = Signal(str, object, str)   # mode, result-or-None, error-or-""


class _FeaturesWorker(QRunnable):
    """Runs a Phase 3 analysis in the thread pool, so the GUI stays responsive."""

    def __init__(self, mode: str, arr: np.ndarray, pixel_size_m: float,
                 params: dict, signals: _FeaturesWorkerSignals):
        super().__init__()
        self._mode    = mode
        self._arr     = arr
        self._px      = float(pixel_size_m)
        self._params  = params
        self._signals = signals

    @Slot()
    def run(self):
        try:
            if self._mode == "particles":
                from probeflow.features import segment_particles
                res = segment_particles(
                    self._arr, self._px,
                    threshold=self._params["threshold"],
                    manual_value=self._params.get("manual_value"),
                    invert=self._params.get("invert", False),
                    min_area_nm2=self._params.get("min_area_nm2", 0.5),
                    max_area_nm2=self._params.get("max_area_nm2"),
                    size_sigma_clip=self._params.get("size_sigma_clip", 2.0),
                )
            elif self._mode == "template":
                from probeflow.features import count_features
                res = count_features(
                    self._arr, self._params["template"], self._px,
                    min_correlation=self._params.get("min_correlation", 0.5),
                    min_distance_m=self._params.get("min_distance_m"),
                )
            elif self._mode == "lattice":
                from probeflow.lattice import extract_lattice, LatticeParams
                res = extract_lattice(self._arr, self._px,
                                      params=LatticeParams())
            else:
                raise ValueError(f"Unknown mode {self._mode!r}")
            self._signals.finished.emit(self._mode, res, "")
        except Exception as exc:
            self._signals.finished.emit(self._mode, None, str(exc))


class FeaturesPanel(QWidget):
    """Center widget for the Features tab: image canvas + overlay + results table."""

    analysis_requested = Signal(str)        # mode name
    template_crop_requested = Signal()

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self._t          = t
        self._entry      = None            # current SxmFile
        self._plane_idx  = 0
        self._arr        = None            # np.ndarray
        self._pixel_size_m = 1e-10
        self._overlay_mode = "none"        # "particles" | "template" | "lattice"
        self._particles  = []
        self._detections = []
        self._lattice    = None
        self._template_arr = None
        self._cropping   = False
        self._crop_start = None
        self._crop_rect  = None
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 4)
        lay.setSpacing(6)

        self._title = QLabel("Features — load a scan from the Browse tab, then run an analysis.")
        self._title.setFont(QFont("Helvetica", 11, QFont.Bold))
        self._title.setWordWrap(True)
        lay.addWidget(self._title)

        self._fig    = Figure(figsize=(6, 6), dpi=90)
        self._fig.patch.set_alpha(0)
        self._ax     = self._fig.add_subplot(111)
        self._ax.set_axis_off()
        self._canvas = FigureCanvasQTAgg(self._fig)
        lay.addWidget(self._canvas, 1)

        self._canvas.mpl_connect("button_press_event",   self._on_press)
        self._canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)

        self._results_table = QTableWidget(0, 4)
        self._results_table.setHorizontalHeaderLabels(["#", "x (nm)", "y (nm)", "value"])
        self._results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._results_table.setFixedHeight(160)
        self._results_table.setFont(QFont("Helvetica", 9))
        lay.addWidget(self._results_table)

    # ── Public API ─────────────────────────────────────────────────────────
    def load_entry(self, entry, plane_idx: int, arr: np.ndarray,
                    pixel_size_m: float):
        self._entry        = entry
        self._plane_idx    = plane_idx
        self._arr          = arr
        self._pixel_size_m = pixel_size_m
        self._particles    = []
        self._detections   = []
        self._lattice      = None
        self._template_arr = None
        self._overlay_mode = "none"
        self._redraw()
        self._results_table.setRowCount(0)
        plane_lbl = PLANE_NAMES[plane_idx] if 0 <= plane_idx < len(PLANE_NAMES) else f"plane {plane_idx}"
        self._title.setText(
            f"{entry.stem}  —  {plane_lbl}  —  "
            f"{arr.shape[1]}×{arr.shape[0]} px  "
            f"(px = {pixel_size_m * 1e12:.1f} pm)")

    def current_entry(self):
        return self._entry

    def current_array(self):
        return self._arr

    def current_pixel_size(self):
        return self._pixel_size_m

    def set_particles(self, particles):
        self._particles    = particles
        self._overlay_mode = "particles"
        self._redraw()
        self._results_table.setColumnCount(4)
        self._results_table.setHorizontalHeaderLabels(
            ["#", "x (nm)", "y (nm)", "area (nm²)"])
        self._results_table.setRowCount(len(particles))
        for i, p in enumerate(particles):
            self._results_table.setItem(i, 0, QTableWidgetItem(str(p.index)))
            self._results_table.setItem(i, 1, QTableWidgetItem(f"{p.centroid_x_m * 1e9:.2f}"))
            self._results_table.setItem(i, 2, QTableWidgetItem(f"{p.centroid_y_m * 1e9:.2f}"))
            self._results_table.setItem(i, 3, QTableWidgetItem(f"{p.area_nm2:.2f}"))

    def set_detections(self, detections):
        self._detections   = detections
        self._overlay_mode = "template"
        self._redraw()
        self._results_table.setColumnCount(4)
        self._results_table.setHorizontalHeaderLabels(
            ["#", "x (nm)", "y (nm)", "corr"])
        self._results_table.setRowCount(len(detections))
        for i, d in enumerate(detections):
            self._results_table.setItem(i, 0, QTableWidgetItem(str(d.index)))
            self._results_table.setItem(i, 1, QTableWidgetItem(f"{d.x_m * 1e9:.2f}"))
            self._results_table.setItem(i, 2, QTableWidgetItem(f"{d.y_m * 1e9:.2f}"))
            self._results_table.setItem(i, 3, QTableWidgetItem(f"{d.correlation:.3f}"))

    def set_lattice(self, lat):
        self._lattice      = lat
        self._overlay_mode = "lattice"
        self._redraw()
        self._results_table.setColumnCount(2)
        self._results_table.setHorizontalHeaderLabels(["parameter", "value"])
        rows = [
            ("|a|",  f"{lat.a_length_m * 1e9:.3f} nm"),
            ("|b|",  f"{lat.b_length_m * 1e9:.3f} nm"),
            ("γ",    f"{lat.gamma_deg:.2f}°"),
            ("a vec (nm)", f"({lat.a_vector_m[0]*1e9:.3f}, {lat.a_vector_m[1]*1e9:.3f})"),
            ("b vec (nm)", f"({lat.b_vector_m[0]*1e9:.3f}, {lat.b_vector_m[1]*1e9:.3f})"),
            ("keypoints (used/total)", f"{lat.n_keypoints_used} / {lat.n_keypoints}"),
        ]
        self._results_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self._results_table.setItem(i, 0, QTableWidgetItem(k))
            self._results_table.setItem(i, 1, QTableWidgetItem(v))

    def get_particles(self):
        return list(self._particles)

    def get_detections(self):
        return list(self._detections)

    def get_lattice(self):
        return self._lattice

    def get_template(self):
        return self._template_arr

    # ── Template crop mode ────────────────────────────────────────────────
    def begin_template_crop(self):
        if self._arr is None:
            return
        self._cropping = True
        self._crop_start = None
        self._crop_rect  = None
        self._title.setText("Template crop — drag a rectangle over one motif, release to set.")
        self._canvas.setCursor(QCursor(Qt.CrossCursor))

    def cancel_template_crop(self):
        self._cropping = False
        self._crop_start = None
        self._crop_rect  = None
        self._canvas.setCursor(QCursor(Qt.ArrowCursor))
        self._redraw()

    def _on_press(self, event):
        if not self._cropping or event.inaxes is not self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._crop_start = (int(event.xdata), int(event.ydata))
        self._crop_rect  = (self._crop_start[0], self._crop_start[1],
                            self._crop_start[0], self._crop_start[1])

    def _on_motion(self, event):
        if not self._cropping or self._crop_start is None:
            return
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        x0, y0 = self._crop_start
        x1, y1 = int(event.xdata), int(event.ydata)
        self._crop_rect = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self._redraw()

    def _on_release(self, event):
        if not self._cropping or self._crop_start is None:
            return
        if self._crop_rect is None:
            return
        x0, y0, x1, y1 = self._crop_rect
        if x1 - x0 < 4 or y1 - y0 < 4 or self._arr is None:
            self.cancel_template_crop()
            self._title.setText("Template crop cancelled — rectangle too small.")
            return
        Ny, Nx = self._arr.shape
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(Nx, x1), min(Ny, y1)
        self._template_arr = self._arr[y0c:y1c, x0c:x1c].copy()
        self._cropping   = False
        self._crop_start = None
        self._canvas.setCursor(QCursor(Qt.ArrowCursor))
        th, tw = self._template_arr.shape
        self._title.setText(
            f"Template captured — {tw}×{th} px.  Press 'Run' to count matches.")
        self._redraw()

    # ── Drawing ───────────────────────────────────────────────────────────
    def _redraw(self):
        self._ax.clear()
        self._ax.set_axis_off()
        if self._arr is None:
            self._canvas.draw_idle()
            return

        try:
            vmin, vmax = _clip_range_from_array(self._arr, 1.0, 99.0)
        except ValueError:
            vmin, vmax = 0.0, 1.0
        self._ax.imshow(self._arr, cmap="gray", vmin=vmin, vmax=vmax,
                         interpolation="nearest", origin="upper")

        if self._overlay_mode == "particles":
            for p in self._particles:
                xs = [c[0] / self._pixel_size_m for c in p.contour_xy_m]
                ys = [c[1] / self._pixel_size_m for c in p.contour_xy_m]
                if xs and ys:
                    xs.append(xs[0]); ys.append(ys[0])
                    self._ax.plot(xs, ys, color="#f38ba8", lw=0.8)
                cx = p.centroid_x_m / self._pixel_size_m
                cy = p.centroid_y_m / self._pixel_size_m
                self._ax.plot(cx, cy, marker="+", color="#a6e3a1", ms=5)
        elif self._overlay_mode == "template":
            for d in self._detections:
                self._ax.plot(d.x_px, d.y_px, marker="o", mfc="none",
                              mec="#89b4fa", ms=8, mew=1.2)
        elif self._overlay_mode == "lattice" and self._lattice is not None:
            lat = self._lattice
            Ny, Nx = self._arr.shape
            cx, cy = Nx / 2, Ny / 2
            ax_, ay_ = lat.a_vector_px
            bx_, by_ = lat.b_vector_px
            self._ax.plot([cx, cx + ax_], [cy, cy + ay_], color="#f38ba8", lw=1.8)
            self._ax.plot([cx, cx + bx_], [cy, cy + by_], color="#89b4fa", lw=1.8)
            # Unit cell parallelogram
            pts_x = [cx, cx + ax_, cx + ax_ + bx_, cx + bx_, cx]
            pts_y = [cy, cy + ay_, cy + ay_ + by_, cy + by_, cy]
            self._ax.plot(pts_x, pts_y, color="#fab387", lw=1.0, ls="--")

        if self._crop_rect is not None:
            x0, y0, x1, y1 = self._crop_rect
            self._ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                          color="#f9e2af", lw=1.2)

        if self._template_arr is not None and self._overlay_mode == "template":
            th, tw = self._template_arr.shape
            self._ax.text(5, 15,
                          f"template: {tw}×{th} px",
                          color="#f9e2af", fontsize=8,
                          bbox=dict(boxstyle="round", fc="#1e1e2e88", ec="none"))

        self._canvas.draw_idle()


class FeaturesSidebar(QWidget):
    """Right sidebar for the Features tab: mode selector + per-mode parameters."""

    mode_changed          = Signal(str)      # "particles" / "template" / "lattice"
    load_from_browse_requested = Signal()
    run_requested         = Signal(str)      # mode
    export_requested      = Signal(str)      # mode
    crop_template_requested = Signal()

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self._t = t
        self._build()

    def _build(self):
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

        # ── Load scan ──
        load_btn = QPushButton("Load primary scan from Browse")
        load_btn.setFont(QFont("Helvetica", 10))
        load_btn.setFixedHeight(30)
        load_btn.setCursor(QCursor(Qt.PointingHandCursor))
        load_btn.setObjectName("accentBtn")
        load_btn.clicked.connect(self.load_from_browse_requested.emit)
        lay.addWidget(load_btn)

        plane_row = QHBoxLayout()
        plane_row.addWidget(QLabel("Plane:"))
        self._plane_cb = QComboBox()
        self._plane_cb.addItems(PLANE_NAMES)
        self._plane_cb.setCurrentIndex(0)
        plane_row.addWidget(self._plane_cb, 1)
        lay.addLayout(plane_row)
        lay.addWidget(_sep())

        # ── Mode selector ──
        mode_lbl = QLabel("Analysis mode")
        mode_lbl.setFont(QFont("Helvetica", 11, QFont.Bold))
        lay.addWidget(mode_lbl)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_btns = {}
        for key, label in [("particles", "Particles"),
                           ("template",  "Template"),
                           ("lattice",   "Lattice")]:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setFont(QFont("Helvetica", 9))
            b.setFixedHeight(26)
            b.setCursor(QCursor(Qt.PointingHandCursor))
            b.clicked.connect(lambda _=False, k=key: self._select_mode(k))
            self._mode_group.addButton(b)
            mode_row.addWidget(b)
            self._mode_btns[key] = b
        lay.addLayout(mode_row)

        # ── Per-mode stack ──
        self._mode_stack = QStackedWidget()
        self._mode_stack.addWidget(self._build_particles_tab())
        self._mode_stack.addWidget(self._build_template_tab())
        self._mode_stack.addWidget(self._build_lattice_tab())
        lay.addWidget(self._mode_stack)

        lay.addWidget(_sep())

        # ── Action buttons ──
        self._run_btn = QPushButton("Run")
        self._run_btn.setFont(QFont("Helvetica", 10, QFont.Bold))
        self._run_btn.setFixedHeight(32)
        self._run_btn.setObjectName("accentBtn")
        self._run_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._run_btn.clicked.connect(lambda: self.run_requested.emit(self._current_mode()))
        lay.addWidget(self._run_btn)

        self._export_btn = QPushButton("Export JSON…")
        self._export_btn.setFont(QFont("Helvetica", 9))
        self._export_btn.setFixedHeight(28)
        self._export_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._export_btn.clicked.connect(lambda: self.export_requested.emit(self._current_mode()))
        lay.addWidget(self._export_btn)

        self._status_lbl = QLabel("Load a scan to begin.")
        self._status_lbl.setFont(QFont("Helvetica", 9))
        self._status_lbl.setWordWrap(True)
        lay.addWidget(self._status_lbl)

        lay.addStretch(1)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

        self._select_mode("particles")

    def _build_particles_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 4, 0, 4)
        l.setSpacing(4)

        l.addWidget(QLabel("Threshold"))
        self._thr_cb = QComboBox()
        self._thr_cb.addItems(["otsu", "manual", "adaptive"])
        l.addWidget(self._thr_cb)

        row = QHBoxLayout()
        row.addWidget(QLabel("Manual (0-255):"))
        self._manual_spin = QDoubleSpinBox()
        self._manual_spin.setRange(0.0, 255.0)
        self._manual_spin.setValue(128.0)
        self._manual_spin.setDecimals(0)
        row.addWidget(self._manual_spin)
        l.addLayout(row)

        self._invert_cb = QCheckBox("Invert (segment dark features)")
        l.addWidget(self._invert_cb)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("min area (nm²):"))
        self._min_area_spin = QDoubleSpinBox()
        self._min_area_spin.setRange(0.0, 1e6)
        self._min_area_spin.setDecimals(2)
        self._min_area_spin.setValue(0.5)
        row2.addWidget(self._min_area_spin)
        l.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("max area (nm²):"))
        self._max_area_spin = QDoubleSpinBox()
        self._max_area_spin.setRange(0.0, 1e9)
        self._max_area_spin.setDecimals(2)
        self._max_area_spin.setValue(0.0)  # 0 → None
        row3.addWidget(self._max_area_spin)
        l.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("σ-clip:"))
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.0, 10.0)
        self._sigma_spin.setDecimals(1)
        self._sigma_spin.setValue(2.0)
        row4.addWidget(self._sigma_spin)
        l.addLayout(row4)

        return w

    def _build_template_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 4, 0, 4)
        l.setSpacing(4)

        crop_btn = QPushButton("Crop template from image…")
        crop_btn.setFont(QFont("Helvetica", 9))
        crop_btn.setFixedHeight(28)
        crop_btn.setCursor(QCursor(Qt.PointingHandCursor))
        crop_btn.clicked.connect(self.crop_template_requested.emit)
        l.addWidget(crop_btn)

        row = QHBoxLayout()
        row.addWidget(QLabel("min correlation:"))
        self._corr_spin = QDoubleSpinBox()
        self._corr_spin.setRange(0.0, 1.0)
        self._corr_spin.setDecimals(2)
        self._corr_spin.setSingleStep(0.05)
        self._corr_spin.setValue(0.5)
        row.addWidget(self._corr_spin)
        l.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("min distance (nm):"))
        self._dist_spin = QDoubleSpinBox()
        self._dist_spin.setRange(0.0, 1e4)
        self._dist_spin.setDecimals(3)
        self._dist_spin.setValue(0.0)   # 0 → auto (½ template side)
        row2.addWidget(self._dist_spin)
        l.addLayout(row2)

        hint = QLabel("Tip: draw a tight rectangle over one motif. Distance of 0 → auto.")
        hint.setFont(QFont("Helvetica", 8))
        hint.setWordWrap(True)
        l.addWidget(hint)
        return w

    def _build_lattice_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 4, 0, 4)
        l.setSpacing(4)

        info = QLabel(
            "Extracts primitive lattice vectors via SIFT keypoint clustering.\n"
            "Best on atomically-resolved images with a clear repeating motif.")
        info.setFont(QFont("Helvetica", 9))
        info.setWordWrap(True)
        l.addWidget(info)
        return w

    def _select_mode(self, key: str):
        for k, b in self._mode_btns.items():
            b.setChecked(k == key)
        idx = {"particles": 0, "template": 1, "lattice": 2}[key]
        self._mode_stack.setCurrentIndex(idx)
        self.mode_changed.emit(key)

    def _current_mode(self) -> str:
        for k, b in self._mode_btns.items():
            if b.isChecked():
                return k
        return "particles"

    # ── Public getters ─────────────────────────────────────────────────────
    def current_mode(self) -> str:
        return self._current_mode()

    def plane_index(self) -> int:
        return int(self._plane_cb.currentIndex())

    def particles_params(self) -> dict:
        return {
            "threshold":       self._thr_cb.currentText(),
            "manual_value":    self._manual_spin.value(),
            "invert":          self._invert_cb.isChecked(),
            "min_area_nm2":    self._min_area_spin.value(),
            "max_area_nm2":    None if self._max_area_spin.value() <= 0
                               else self._max_area_spin.value(),
            "size_sigma_clip": None if self._sigma_spin.value() <= 0
                               else self._sigma_spin.value(),
        }

    def template_params(self) -> dict:
        return {
            "min_correlation": self._corr_spin.value(),
            "min_distance_m":  None if self._dist_spin.value() <= 0
                               else self._dist_spin.value() * 1e-9,
        }

    def set_status(self, text: str):
        self._status_lbl.setText(text)


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

        z_lbl = QLabel("Z (m):")
        z_lbl.setFont(QFont("Helvetica", 9))
        self._z_unit_cb = QComboBox()
        self._z_unit_cb.addItems(["Auto", "pm", "Å", "nm", "µm", "m"])
        self._z_unit_cb.setFont(QFont("Helvetica", 9))
        self._z_unit_cb.currentTextChanged.connect(
            lambda v: self._on_unit_changed("m", v))

        i_lbl = QLabel("I (A):")
        i_lbl.setFont(QFont("Helvetica", 9))
        self._i_unit_cb = QComboBox()
        self._i_unit_cb.addItems(["Auto", "fA", "pA", "nA", "µA", "A"])
        self._i_unit_cb.setFont(QFont("Helvetica", 9))
        self._i_unit_cb.currentTextChanged.connect(
            lambda v: self._on_unit_changed("A", v))

        v_lbl = QLabel("V (V):")
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
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

    # ── Data load + channel list population ─────────────────────────────

    def _load(self) -> None:
        from .spec_io import read_spec_file

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
            unit = spec.y_units.get(ch, "")
            label = f"{ch}  ({unit})" if unit else ch
            cb = QCheckBox(label)
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
        self._redraw()

    def _redraw(self) -> None:
        if self._spec is None:
            return
        from .spec_plot import choose_display_unit, lookup_unit_scale

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
            y = np.asarray(spec.channels[ch], dtype=float)
            unit = spec.y_units.get(ch, "")
            choice = self._unit_choice.get(unit, "Auto")
            override = lookup_unit_scale(unit, choice) if choice != "Auto" else None
            if override is not None:
                scale, disp_unit = override
            else:
                scale, disp_unit = choose_display_unit(unit, y)
            y_disp = y * scale

            ax.set_facecolor(self._BG)
            ax.plot(spec.x_array, y_disp, linewidth=1.0,
                    color=self._COLORS[i % len(self._COLORS)])
            ax.set_ylabel(f"{ch} ({disp_unit})" if disp_unit else ch,
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

    # ── Raw-data table ──────────────────────────────────────────────────

    def _show_raw_data(self) -> None:
        if self._spec is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Raw data — {self._entry.stem}")
        dlg.resize(640, 400)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(8, 8, 8, 8)

        spec = self._spec
        order = [ch for ch in spec.channel_order if ch in spec.channels]
        if not order:
            order = list(spec.channels.keys())
        n_rows = min(20, len(spec.x_array))
        n_cols = 1 + len(order)  # + 1 for x axis column

        table = QTableWidget(n_rows, n_cols)
        headers = [spec.x_label] + [
            f"{ch} ({spec.y_units.get(ch, '')})" if spec.y_units.get(ch) else ch
            for ch in order
        ]
        table.setHorizontalHeaderLabels(headers)

        for r in range(n_rows):
            table.setItem(r, 0, QTableWidgetItem(f"{spec.x_array[r]:.6g}"))
            for c, ch in enumerate(order, start=1):
                val = spec.channels[ch][r]
                table.setItem(r, c, QTableWidgetItem(f"{val:.6g}"))
        table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(table)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        v.addLayout(btn_row)

        dlg.exec()


# ── Export dialog ────────────────────────────────────────────────────────────
_EXPORT_FORMATS: list[tuple[str, str, str]] = [
    # (label, suffix without dot, QFileDialog filter string)
    ("PNG image",    "png", "PNG images (*.png)"),
    ("PDF figure",   "pdf", "PDF figures (*.pdf)"),
    ("CSV grid",     "csv", "CSV grids (*.csv)"),
    ("Nanonis .sxm", "sxm", "Nanonis files (*.sxm)"),
]


class ExportDialog(QDialog):
    """Pick an output format and per-format options before saving."""

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export scan")
        self.setFixedSize(380, 360)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(8)

        # ── Format picker ───────────────────────────────────────────────────
        fmt_row = QHBoxLayout()
        fmt_lbl = QLabel("Format:")
        fmt_lbl.setFixedWidth(70)
        self._fmt_cb = QComboBox()
        for label, _suffix, _filt in _EXPORT_FORMATS:
            self._fmt_cb.addItem(label)
        self._fmt_cb.currentIndexChanged.connect(self._update_visible_options)
        fmt_row.addWidget(fmt_lbl)
        fmt_row.addWidget(self._fmt_cb, 1)
        lay.addLayout(fmt_row)
        lay.addWidget(_sep())

        # ── PNG / PDF options: scale bar toggle + unit + position ───────────
        self._sb_group = QWidget()
        sb_outer = QVBoxLayout(self._sb_group)
        sb_outer.setContentsMargins(0, 0, 0, 0)
        sb_outer.setSpacing(4)

        self._scalebar_cb = QCheckBox("Add scale bar")
        self._scalebar_cb.setChecked(True)
        self._scalebar_cb.toggled.connect(self._on_scalebar_toggled)
        sb_outer.addWidget(self._scalebar_cb)

        self._sb_opts = QWidget()
        sb_lay = QVBoxLayout(self._sb_opts)
        sb_lay.setContentsMargins(12, 0, 0, 0)
        sb_lay.setSpacing(4)

        unit_row = QHBoxLayout()
        unit_lbl = QLabel("Unit:")
        unit_lbl.setFixedWidth(60)
        self._unit_group = QButtonGroup(self)
        self._nm_rb  = QRadioButton("nm")
        self._ang_rb = QRadioButton("Å")
        self._pm_rb  = QRadioButton("pm")
        self._nm_rb.setChecked(True)
        for rb in (self._nm_rb, self._ang_rb, self._pm_rb):
            self._unit_group.addButton(rb)
            unit_row.addWidget(rb)
        unit_row.insertWidget(0, unit_lbl)
        unit_row.addStretch()
        sb_lay.addLayout(unit_row)

        pos_row = QHBoxLayout()
        pos_lbl = QLabel("Position:")
        pos_lbl.setFixedWidth(60)
        self._pos_cb = QComboBox()
        self._pos_cb.addItems(["bottom-right", "bottom-left", "top-right", "top-left"])
        pos_row.addWidget(pos_lbl)
        pos_row.addWidget(self._pos_cb, 1)
        sb_lay.addLayout(pos_row)

        sb_outer.addWidget(self._sb_opts)
        lay.addWidget(self._sb_group)

        # ── "No extra options" note for formats that don't need one ─────────
        self._nooptions_lbl = QLabel(
            "No extra options — the file will be written as-is."
        )
        self._nooptions_lbl.setWordWrap(True)
        self._nooptions_lbl.setStyleSheet("color: gray; font-style: italic;")
        lay.addWidget(self._nooptions_lbl)

        lay.addStretch()

        # ── Buttons ─────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("Export…")
        ok_btn.setObjectName("accentBtn")
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(ok_btn)
        lay.addLayout(btn_row)

        self._update_visible_options()

    def _on_scalebar_toggled(self, checked: bool):
        self._sb_opts.setEnabled(checked)

    def _update_visible_options(self):
        label = self._fmt_cb.currentText()
        scalebar_fmts = ("PNG image", "PDF figure")
        show_sb = label in scalebar_fmts
        self._sb_group.setVisible(show_sb)
        self._nooptions_lbl.setVisible(not show_sb)

    def get_settings(self) -> dict:
        """Return everything the caller needs to route to writers.save_scan.

        Keys:
            format_label : str    — human-readable dropdown entry
            suffix       : str    — lower-case extension (no leading dot)
            file_filter  : str    — QFileDialog filter pattern
            add_scalebar / scalebar_unit / scalebar_pos  — only meaningful for PNG/PDF
        """
        idx = self._fmt_cb.currentIndex()
        label, suffix, filt = _EXPORT_FORMATS[idx]

        unit = "nm"
        if self._ang_rb.isChecked():
            unit = "Å"
        elif self._pm_rb.isChecked():
            unit = "pm"

        return {
            "format_label":  label,
            "suffix":        suffix,
            "file_filter":   filt,
            "add_scalebar":  self._scalebar_cb.isChecked(),
            "scalebar_unit": unit,
            "scalebar_pos":  self._pos_cb.currentText(),
        }


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
    about_clicked        = Signal()

    def __init__(self, dark: bool, parent=None):
        super().__init__(parent)
        self._dark    = dark
        self._btns:   list[QPushButton] = []
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
        _nbtn("GitHub", lambda: _open_url(GITHUB_URL))
        _nbtn("About",  self.about_clicked.emit)

        self._apply_nav_theme()

    def set_dark(self, dark: bool):
        self._dark = dark
        self._theme_btn.setText("Light mode" if dark else "Dark mode")
        self._apply_nav_theme()

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


# ── Main window ───────────────────────────────────────────────────────────────
class ProbeFlowWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProbeFlow")
        self.setMinimumSize(1100, 720)

        self._cfg      = load_config()
        self._dark     = self._cfg.get("dark_mode", True)
        self._mode     = "browse"
        self._running  = False
        self._n_loaded = 0

        self._build_ui()
        self._apply_theme()

    # ── Build ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        v_lay = QVBoxLayout(central)
        v_lay.setContentsMargins(0, 0, 0, 0)
        v_lay.setSpacing(0)

        self._navbar = Navbar(self._dark)
        self._navbar.theme_toggle_clicked.connect(self._toggle_theme)
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
        self._tab_features = QPushButton("Features")
        for btn in (self._tab_browse, self._tab_convert, self._tab_features):
            btn.setFont(QFont("Helvetica", 11, QFont.Bold))
            btn.setFixedHeight(44)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.setFlat(True)
            tab_lay.addWidget(btn)
        tab_lay.addStretch()
        self._tab_browse.clicked.connect(lambda: self._switch_mode("browse"))
        self._tab_convert.clicked.connect(lambda: self._switch_mode("convert"))
        self._tab_features.clicked.connect(lambda: self._switch_mode("features"))
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
        self._content_stack.addWidget(browse_split)
        self._content_stack.addWidget(self._conv_panel)
        self._content_stack.addWidget(self._features_panel)
        self._splitter.addWidget(self._content_stack)

        # ── Right: sidebar stack ───────────────────────────────────────────────
        self._sidebar_stack    = QStackedWidget()
        self._sidebar_stack.setFixedWidth(300)
        self._browse_info      = BrowseInfoPanel(t, self._cfg)
        self._convert_sidebar  = ConvertSidebar(t, self._cfg)
        self._features_sidebar = FeaturesSidebar(t)
        self._sidebar_stack.addWidget(self._browse_info)
        self._sidebar_stack.addWidget(self._convert_sidebar)
        self._sidebar_stack.addWidget(self._features_sidebar)
        self._splitter.addWidget(self._sidebar_stack)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

        # Features tab plumbing
        self._features_pool    = QThreadPool.globalInstance()
        self._features_signals = _FeaturesWorkerSignals()
        self._features_signals.finished.connect(self._on_features_finished)

        # Wire signals
        self._browse_tools.open_folder_requested.connect(self._open_browse_folder)
        self._grid.entry_selected.connect(self._on_entry_select)
        self._grid.selection_changed.connect(self._on_selection_changed)
        self._grid.view_requested.connect(self._open_viewer)
        self._browse_tools.colormap_apply_requested.connect(self._on_apply_colormap)
        self._browse_tools.scale_changed.connect(self._on_scale_changed)
        self._browse_tools.processing_apply_requested.connect(self._on_processing_apply)
        self._browse_tools.processing_apply_all_requested.connect(self._on_processing_apply_all)
        self._browse_tools.autoclip_requested.connect(self._on_autoclip)
        self._browse_tools.export_requested.connect(self._on_export)
        self._browse_tools.undo_requested.connect(self._on_undo)
        self._browse_tools.filter_changed.connect(self._on_filter_changed)
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
                self._status_bar.showMessage("Features — pick a mode and Run")
        else:
            self._content_stack.setCurrentIndex(1)
            self._sidebar_stack.setCurrentIndex(1)
            self._update_count(self._conv_panel.input_entry.text())
        self._update_tab_styles()

    def _update_tab_styles(self):
        t = THEMES["dark" if self._dark else "light"]
        for btn, name in ((self._tab_browse, "browse"),
                          (self._tab_convert, "convert"),
                          (self._tab_features, "features")):
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
        from probeflow.indexing import index_folder
        all_items    = index_folder(Path(d), recursive=True, include_errors=True)
        sxm_entries  = _scan_items_to_sxm(all_items)
        vert_entries = _spec_items_to_vert(all_items)
        n_errors     = sum(1 for it in all_items if it.load_error)
        entries = sorted(sxm_entries + vert_entries, key=lambda e: e.stem)
        self._grid.load(entries, folder_path=d)
        self._n_loaded = len(entries)
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
            "Select scans + Apply to colorize")
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

    def _on_apply_colormap(self, cmap_key: str):
        clip_low, clip_high = self._browse_tools.get_clip_values()
        n = self._grid.set_colormap_for_selection(cmap_key,
                                                   clip_low=clip_low,
                                                   clip_high=clip_high)
        if n == 0:
            self._status_bar.showMessage(
                "No images selected — click images first (Ctrl+click for multi-select)")
        else:
            label = next((l for l, k in CMAP_KEY.items() if k == cmap_key), cmap_key)
            self._status_bar.showMessage(
                f"Applied {label} colormap to {n} image{'s' if n > 1 else ''}")
            primary = self._grid.get_primary()
            if primary:
                entry = next((e for e in self._grid.get_entries()
                              if e.stem == primary), None)
                if entry:
                    _, _, proc = self._grid.get_card_state(primary)
                    self._browse_info.load_channels(entry, cmap_key, proc)

    def _on_scale_changed(self, clip_low: float, clip_high: float):
        self._browse_info.update_clip(clip_low, clip_high)
        n = self._grid.update_clip_for_selection(clip_low, clip_high)
        if n > 0:
            self._status_bar.showMessage(
                f"Scale: {clip_low:.0f}%–{clip_high:.0f}% on {n} image{'s' if n > 1 else ''}")
            primary = self._grid.get_primary()
            if primary:
                entry = next((e for e in self._grid.get_entries()
                              if e.stem == primary), None)
                if entry:
                    cmap, _, proc = self._grid.get_card_state(primary)
                    self._browse_info.load_channels(entry, cmap, proc)

    def _on_processing_apply(self, cfg: dict):
        clip_low, clip_high = self._browse_tools.get_clip_values()
        cmap_key = CMAP_KEY.get(
            self._browse_tools.cmap_cb.currentText(), DEFAULT_CMAP_KEY)
        n = self._grid.set_colormap_for_selection(
            cmap_key, clip_low=clip_low, clip_high=clip_high, processing=cfg)
        if n == 0:
            self._status_bar.showMessage(
                "No images selected — click images first")
        else:
            steps = []
            if cfg.get('align_rows'):
                steps.append(f"align({cfg['align_rows']})")
            desc = ", ".join(steps) if steps else "none"
            self._status_bar.showMessage(
                f"Quick corrections [{desc}] applied to {n} thumbnail{'s' if n > 1 else ''}")
            primary = self._grid.get_primary()
            if primary:
                entry = next((e for e in self._grid.get_entries()
                              if e.stem == primary), None)
                if entry:
                    self._browse_info.load_channels(entry, cmap_key, cfg)

    def _on_processing_apply_all(self, cfg: dict):
        n = self._grid.set_processing_for_all_images(cfg)
        if n == 0:
            self._status_bar.showMessage("No scan thumbnails to update")
            return

        steps = []
        if cfg.get('align_rows'):
            steps.append(f"align({cfg['align_rows']})")
        desc = ", ".join(steps) if steps else "none"
        self._status_bar.showMessage(
            f"Queued quick corrections [{desc}] for {n} thumbnail{'s' if n > 1 else ''}")

        primary = self._grid.get_primary()
        if primary:
            entry = next((e for e in self._grid.get_entries()
                          if e.stem == primary), None)
            if entry:
                cmap_key, _, proc = self._grid.get_card_state(primary)
                self._browse_info.load_channels(entry, cmap_key, proc)

    def _on_autoclip(self):
        primary = self._grid.get_primary()
        if not primary:
            self._status_bar.showMessage("Select an image first for auto clip")
            return
        entry = next((e for e in self._grid.get_entries() if e.stem == primary), None)
        if not entry:
            return
        try:
            _scan = load_scan(entry.path)
            arr = _scan.planes[0] if _scan.n_planes > 0 else None
        except Exception:
            arr = None
        if arr is None:
            self._status_bar.showMessage("Could not read scan data for auto clip")
            return
        try:
            clip_low, clip_high = _proc.gmm_autoclip(arr)
            self._browse_tools._low_slider.setValue(int(round(clip_low)))
            self._browse_tools._high_slider.setValue(int(round(clip_high)))
            self._status_bar.showMessage(
                f"Auto clip: {clip_low:.1f}% – {clip_high:.1f}%")
        except Exception as exc:
            self._status_bar.showMessage(f"Auto clip error: {exc}")

    def _on_undo(self):
        selected = self._grid.get_selected()
        if not selected:
            self._status_bar.showMessage("Select an image first to undo")
            return
        n = self._grid.undo_last(selected)
        if n == 0:
            self._status_bar.showMessage("Nothing to undo — no history for current selection")
        else:
            self._status_bar.showMessage(f"Undo applied to {n} image{'s' if n > 1 else ''}")

    def _on_export(self):
        primary = self._grid.get_primary()
        if not primary:
            self._status_bar.showMessage("Select an image first")
            return
        entry = next((e for e in self._grid.get_entries() if e.stem == primary), None)
        if not entry:
            return

        # Only topography entries are exportable through this dialog; spec
        # entries are handled by the spec viewer.
        if not hasattr(entry, "path") or entry.path.suffix.lower() == ".vert":
            self._status_bar.showMessage(
                "Selected entry isn't a topography scan — exporting .VERT "
                "spectra uses the spec viewer."
            )
            return

        t   = THEMES["dark" if self._dark else "light"]
        dlg = ExportDialog(t, self)
        dlg.setStyleSheet(QApplication.instance().styleSheet())
        if dlg.exec() != QDialog.Accepted:
            return
        settings = dlg.get_settings()

        suffix     = settings["suffix"]
        filt       = settings["file_filter"]
        label      = settings["format_label"]

        _, _, proc_state = self._grid.get_card_state(entry.stem)
        has_processing = any(proc_state.get(k) for k in NUMERIC_PROC_KEYS)
        out_stem = mark_processed_stem(entry.stem) if has_processing else entry.stem
        suggested = str(Path.home() / f"{out_stem}.{suffix}")

        if suffix == "sxm":
            msg = QMessageBox(self)
            msg.setWindowTitle("Save as .sxm")
            msg.setIcon(QMessageBox.Icon.Information)
            if has_processing:
                msg.setText(
                    "The exported <b>.sxm</b> will include source provenance "
                    "and processing operations in the <tt>COMMENT</tt> header field."
                )
            else:
                msg.setText(
                    "The exported <b>.sxm</b> will include source provenance "
                    "(<tt>Source: &lt;filename&gt;</tt>) in the <tt>COMMENT</tt> header field."
                )
            msg.setInformativeText(f"Suggested filename: <b>{out_stem}.sxm</b>")
            msg.setStandardButtons(
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
            )
            msg.setDefaultButton(QMessageBox.StandardButton.Ok)
            if msg.exec() != QMessageBox.StandardButton.Ok:
                return

        out_path, _ = QFileDialog.getSaveFileName(
            self, f"Save as {label}", suggested, filt)
        if not out_path:
            return

        try:
            scan = load_scan(entry.path)
        except Exception as exc:
            self._status_bar.showMessage(f"Could not read scan: {exc}")
            return
        if scan.n_planes == 0:
            self._status_bar.showMessage("Could not read scan data")
            return

        apply_processing_state_to_scan(scan, proc_state, plane_idx=0)

        clip_low, clip_high = self._browse_tools.get_clip_values()
        cmap_key = self._grid._card_colormaps.get(entry.stem, DEFAULT_CMAP_KEY)

        # Per-format kwargs forwarded to writers.save_scan.
        kwargs: dict = {}
        if suffix in ("png", "pdf"):
            kwargs.update(
                colormap=cmap_key,
                clip_low=clip_low,
                clip_high=clip_high,
            )
        if suffix == "png":
            kwargs.update(
                add_scalebar=settings["add_scalebar"],
                scalebar_unit=settings["scalebar_unit"],
                scalebar_pos=settings["scalebar_pos"],
            )

        try:
            scan.save(out_path, plane_idx=0, **kwargs)
            self._status_bar.showMessage(f"Exported → {out_path}")
        except Exception as exc:
            self._status_bar.showMessage(f"Export error: {exc}")

    # ── Features tab handlers ──────────────────────────────────────────────────
    def _on_features_load_from_browse(self):
        primary = self._grid.get_primary()
        if not primary:
            self._features_sidebar.set_status("Select a scan in the Browse tab first.")
            self._status_bar.showMessage("Pick a scan in Browse to load it into Features")
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
            from probeflow.writers.json import write_json
            write_json(out_path, items, kind=kind,
                       extra_meta={"source": str(entry.path) if entry else None})
            self._features_sidebar.set_status(f"Exported → {out_path}")
            self._status_bar.showMessage(f"Exported {kind} → {out_path}")
        except Exception as exc:
            self._features_sidebar.set_status(f"Export failed: {exc}")

    def _open_viewer(self, entry):
        t = THEMES["dark" if self._dark else "light"]
        if isinstance(entry, VertFile):
            dlg = SpecViewerDialog(entry, t, self)
            dlg.exec()
        else:
            cmap_key, clip, proc = self._grid.get_card_state(entry.stem)
            if entry.stem not in self._grid._card_clip:
                # Always open new images with robust 1%–99% clip, independent
                # of whatever the browse-tool scale sliders are set to.
                clip = (1.0, 99.0)
            sxm_entries = [e for e in self._grid.get_entries() if isinstance(e, SxmFile)]
            dlg = ImageViewerDialog(entry, sxm_entries, cmap_key, t, self,
                                    clip_low=clip[0], clip_high=clip[1],
                                    processing=proc)
            dlg.exec()

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

    def _apply_theme(self):
        t = THEMES["dark" if self._dark else "light"]
        QApplication.instance().setStyleSheet(_build_qss(t))
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
        cl, ch = self._browse_tools.get_clip_values()
        save_config({
            "dark_mode":     self._dark,
            "input_dir":     self._conv_panel.input_entry.text(),
            "output_dir":    self._conv_panel.output_entry.text(),
            "custom_output": self._conv_panel._custom_out_cb.isChecked(),
            "do_png":        self._convert_sidebar.png_cb.isChecked(),
            "do_sxm":        self._convert_sidebar.sxm_cb.isChecked(),
            "clip_low":      cl,
            "clip_high":     ch,
            "colormap":      self._browse_tools.cmap_cb.currentText(),
            "browse_filter": self._browse_tools.get_filter_mode(),
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
def _build_qss(t: dict) -> str:
    return f"""
QMainWindow, QWidget {{
    background-color: {t['main_bg']};
    color: {t['fg']};
    font-family: Helvetica, Arial, sans-serif;
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
