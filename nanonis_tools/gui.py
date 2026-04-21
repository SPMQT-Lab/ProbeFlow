"""ProbeFlow — PySide6 GUI for Createc-to-Nanonis file conversion."""

from __future__ import annotations

import json
import re as _re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from PySide6.QtCore import (
    Qt, QObject, QRunnable, QThreadPool, QTimer, QSize, Signal, Slot,
)
from PySide6.QtGui import (
    QColor, QCursor, QFont, QImage, QMovie, QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
    QDialog, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton,
    QRadioButton, QScrollArea, QSlider, QSplitter, QStackedWidget,
    QStatusBar, QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
    QVBoxLayout, QWidget,
)
import webbrowser

from nanonis_tools import processing as _proc

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_PATH     = Path.home() / ".probeflow_config.json"
REPO_ROOT       = Path(__file__).resolve().parent.parent
DEFAULT_CUSHION = REPO_ROOT / "src" / "file_cushions"
LOGO_PATH       = REPO_ROOT / "assets" / "logo.png"
LOGO_GIF_PATH   = REPO_ROOT / "assets" / "logo.gif"
LOGO_NAV_PATH   = REPO_ROOT / "assets" / "logo_nav.png"
GITHUB_URL      = "https://github.com/SPMQT-Lab/ProbeFlow"

NAVBAR_BG = "#3273dc"
NAVBAR_H  = 58

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
        "btn_bg":     "#e0e0e0",
        "btn_fg":     "#1e1e2e",
        "log_bg":     "#ffffff",
        "log_fg":     "#1e1e2e",
        "ok_fg":      "#1a7a1a",
        "err_fg":     "#c0392b",
        "warn_fg":    "#b07800",
        "accent_bg":  "#3273dc",
        "accent_fg":  "#ffffff",
        "sep":        "#dee2e6",
        "sub_fg":     "#6c757d",
        "sidebar_bg": "#eff5fb",
        "main_bg":    "#eef6fc",
        "status_bg":  "#f5f5f5",
        "status_fg":  "#6c757d",
        "card_bg":    "#d0e4f4",
        "card_sel":   "#b8d4ee",
        "card_fg":    "#1e1e2e",
        "tab_act":    "#ffffff",
        "tab_inact":  "#d8eaf8",
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
    path: Path
    stem: str
    Nx:   int = 512
    Ny:   int = 512


# ── SXM parsing ───────────────────────────────────────────────────────────────
def parse_sxm_header(sxm_path: Path) -> dict:
    params: dict              = {}
    current_key: Optional[str] = None
    buf: list[str]            = []

    def _flush():
        if current_key is not None:
            params[current_key] = " ".join(buf).strip()

    try:
        with open(sxm_path, "rb") as fh:
            for raw in fh:
                if raw.strip() == b":SCANIT_END:":
                    break
                line = raw.decode("latin-1", errors="replace").rstrip("\r\n")
                if line.startswith(":") and line.endswith(":") and len(line) > 2:
                    _flush()
                    current_key = line[1:-1]
                    buf = []
                elif current_key is not None:
                    s = line.strip()
                    if s:
                        buf.append(s)
        _flush()
    except Exception:
        pass
    return params


def _sxm_dims(hdr: dict) -> tuple[int, int]:
    nums = [int(x) for x in _re.findall(r"\d+", hdr.get("SCAN_PIXELS", ""))]
    return (nums[0], nums[1]) if len(nums) >= 2 else (512, 512)


def render_sxm_plane(
    sxm_path:  Path,
    plane_idx: int   = 0,
    colormap:  str   = "gray",
    clip_low:  float = 1.0,
    clip_high: float = 99.0,
    size:      tuple = (148, 116),
) -> Optional[Image.Image]:
    try:
        hdr    = parse_sxm_header(sxm_path)
        Nx, Ny = _sxm_dims(hdr)
        if Nx <= 0 or Ny <= 0:
            return None

        data_offset = int((DEFAULT_CUSHION / "data_offset.txt").read_text().strip())
        raw = sxm_path.read_bytes()

        plane_bytes = Ny * Nx * 4
        start = data_offset + plane_idx * plane_bytes
        if start + plane_bytes > len(raw):
            return None

        arr = np.frombuffer(raw[start: start + plane_bytes], dtype=">f4").copy()
        arr = arr.reshape((Ny, Nx))

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        vmin = float(np.percentile(finite, clip_low))
        vmax = float(np.percentile(finite, clip_high))
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max())
        if vmax <= vmin:
            return None

        safe    = np.where(np.isfinite(arr), arr, vmin).astype(np.float64)
        u8      = np.clip((safe - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        colored = _get_lut(colormap)[u8]
        img     = Image.fromarray(colored, mode="RGB")
        img.thumbnail(size, Image.LANCZOS)
        return img
    except Exception:
        return None


def read_sxm_plane_raw(sxm_path: Path, plane_idx: int = 0) -> Optional[np.ndarray]:
    """Read raw float32 array for a plane from an SXM file, returns None on error."""
    try:
        hdr    = parse_sxm_header(sxm_path)
        Nx, Ny = _sxm_dims(hdr)
        if Nx <= 0 or Ny <= 0:
            return None

        data_offset = int((DEFAULT_CUSHION / "data_offset.txt").read_text().strip())
        raw = sxm_path.read_bytes()

        plane_bytes = Ny * Nx * 4
        start = data_offset + plane_idx * plane_bytes
        if start + plane_bytes > len(raw):
            return None

        arr = np.frombuffer(raw[start: start + plane_bytes], dtype=">f4").copy()
        return arr.reshape((Ny, Nx))
    except Exception:
        return None


def _sxm_scan_range(hdr: dict) -> tuple[float, float]:
    """Return (width_m, height_m) from the SCAN_RANGE header entry."""
    nums = _re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                       hdr.get("SCAN_RANGE", ""))
    if len(nums) >= 2:
        try:
            return (float(nums[0]), float(nums[1]))
        except ValueError:
            pass
    return (0.0, 0.0)


def render_with_processing(
    arr:        np.ndarray,
    colormap:   str,
    clip_low:   float,
    clip_high:  float,
    processing: dict,
    size:       Optional[tuple] = None,
) -> Optional[Image.Image]:
    """
    Apply the processing pipeline to *arr* then render to a PIL Image.

    processing keys (all optional / defaulted):
        remove_bad_lines : bool
        bg_order         : None | 1 | 2
        fft_mode         : None | 'low_pass' | 'high_pass'
        fft_cutoff       : float  (0.01 – 0.50)
        fft_window       : str    ('hanning')
    """
    if processing is None:
        processing = {}
    try:
        a = arr.astype(np.float64, copy=True)

        if processing.get('remove_bad_lines'):
            a = _proc.remove_bad_lines(a)

        bg_order = processing.get('bg_order')
        if bg_order is not None:
            a = _proc.subtract_background(a, order=int(bg_order))

        fft_mode = processing.get('fft_mode')
        if fft_mode is not None:
            a = _proc.fourier_filter(
                a,
                mode=fft_mode,
                cutoff=float(processing.get('fft_cutoff', 0.10)),
                window=str(processing.get('fft_window', 'hanning')),
            )

        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return None
        vmin = float(np.percentile(finite, clip_low))
        vmax = float(np.percentile(finite, clip_high))
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max())
        if vmax <= vmin:
            return None

        safe    = np.where(np.isfinite(a), a, vmin).astype(np.float64)
        u8      = np.clip((safe - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        colored = _get_lut(colormap)[u8]
        img     = Image.fromarray(colored, mode="RGB")
        if size:
            img.thumbnail(size, Image.LANCZOS)
        return img
    except Exception:
        return None


def scan_sxm_folder(root: Path) -> list[SxmFile]:
    entries = []
    for sxm in sorted(Path(root).rglob("*.sxm")):
        try:
            hdr    = parse_sxm_header(sxm)
            Nx, Ny = _sxm_dims(hdr)
            entries.append(SxmFile(path=sxm, stem=sxm.stem, Nx=Nx, Ny=Ny))
        except Exception:
            entries.append(SxmFile(path=sxm, stem=sxm.stem))
    return entries


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
            arr = read_sxm_plane_raw(self.entry.path, 0)
            if arr is not None:
                img = render_with_processing(
                    arr, self.colormap, self.clip_low, self.clip_high,
                    self.processing, size=(self.w, self.h))
            else:
                img = None
        else:
            img = render_sxm_plane(self.entry.path, 0, self.colormap,
                                   self.clip_low, self.clip_high,
                                   size=(self.w, self.h))
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
            arr = read_sxm_plane_raw(self.entry.path, self.idx)
            if arr is not None:
                img = render_with_processing(
                    arr, self.colormap, self.clip_low, self.clip_high,
                    self.processing, size=(self.w, self.h))
            else:
                img = None
        else:
            img = render_sxm_plane(self.entry.path, self.idx, self.colormap,
                                   self.clip_low, self.clip_high,
                                   size=(self.w, self.h))
        if img is not None:
            self.signals.loaded.emit(self.idx, pil_to_pixmap(img), self.token)


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
                from nanonis_tools.dats_to_pngs import main as png_main
                _log("── PNG conversion ──", "info")
                png_main(src=in_path, out_root=out_path / "png",
                         clip_low=self.clip_low, clip_high=self.clip_high,
                         verbose=True)
                _log("PNG done.", "ok")

            if self.do_sxm:
                from nanonis_tools.dat_sxm_cli import convert_dat_to_sxm
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
    """Single thumbnail card. Supports single-click and Ctrl+click selection."""
    clicked = Signal(object, bool)  # SxmFile, ctrl_held

    CARD_W = 172
    CARD_H = 158
    IMG_W  = 156
    IMG_H  = 124

    def __init__(self, entry: SxmFile, t: dict, parent=None):
        super().__init__(parent)
        self.entry     = entry
        self._t        = t
        self._sel      = False
        self._colormap = DEFAULT_CMAP_KEY

        self.setFixedSize(self.CARD_W, self.CARD_H)
        self.setCursor(QCursor(Qt.PointingHandCursor))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 2)
        lay.setSpacing(2)

        self.img_lbl = QLabel()
        self.img_lbl.setFixedSize(self.IMG_W, self.IMG_H)
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setText("…")

        lbl_text = entry.stem if len(entry.stem) <= 22 else entry.stem[:20] + ".."
        self.name_lbl = QLabel(lbl_text)
        self.name_lbl.setAlignment(Qt.AlignCenter)
        self.name_lbl.setFont(QFont("Helvetica", 7))

        lay.addWidget(self.img_lbl)
        lay.addWidget(self.name_lbl)
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
                border-radius: 5px;
            }}
        """)
        self.name_lbl.setStyleSheet(f"color: {fg}; background: transparent;")
        self.img_lbl.setStyleSheet(f"color: {t['sub_fg']}; background: transparent;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            ctrl = bool(event.modifiers() & Qt.ControlModifier)
            self.clicked.emit(self.entry, ctrl)
        super().mousePressEvent(event)


# ── ThumbnailGrid ─────────────────────────────────────────────────────────────
class ThumbnailGrid(QWidget):
    """
    Browse panel: folder toolbar + thumbnail grid.

    - All images default to grayscale on load.
    - Click = single-select; Ctrl+click = multi-select toggle.
    - set_colormap_for_selection() reloads ONLY selected cards with the new colormap.
    - Unselected cards keep their current colormap (gray by default).
    """
    entry_selected    = Signal(object)   # primary SxmFile for sidebar
    selection_changed = Signal(int)      # count of selected items

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
        self._path_lbl.setFont(QFont("Helvetica", 8))
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
        self._cards:         dict[str, ScanCard] = {}
        self._entries:       list[SxmFile]       = []
        self._selected:      set[str]            = set()
        self._primary:       Optional[str]       = None  # last clicked
        self._card_colormaps: dict[str, str]     = {}    # per-card colormap
        self._load_token                         = object()
        self._current_cols: int                  = 1

        # empty-state placeholder
        self._empty_lbl = QLabel("Open a folder to browse SXM scans")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setFont(QFont("Helvetica", 10))
        self._grid.addWidget(self._empty_lbl, 0, 0)

    # ── Public API ────────────────────────────────────────────────────────────
    def load(self, entries: list[SxmFile], folder_path: str = ""):
        self._entries        = entries
        self._selected       = set()
        self._primary        = None
        self._card_colormaps = {}
        self._load_token     = object()

        if folder_path:
            p = Path(folder_path)
            self._path_lbl.setText(f"{p.name}  ({len(entries)} scan{'s' if len(entries)!=1 else ''})")

        # clear grid
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards = {}

        if not entries:
            self._empty_lbl = QLabel("No .sxm files found in this folder")
            self._empty_lbl.setAlignment(Qt.AlignCenter)
            self._grid.addWidget(self._empty_lbl, 0, 0)
            return

        cols = self._calc_cols()
        self._current_cols = cols
        for i, entry in enumerate(entries):
            card = ScanCard(entry, self._t)
            card.clicked.connect(self._on_card_click)
            self._card_colormaps[entry.stem] = DEFAULT_CMAP_KEY
            row, col = divmod(i, cols)
            self._grid.addWidget(card, row, col, Qt.AlignTop | Qt.AlignLeft)
            self._cards[entry.stem] = card

        # load all thumbnails as grayscale
        token = self._load_token
        for entry in entries:
            loader = ThumbnailLoader(entry, DEFAULT_CMAP_KEY, token,
                                     ScanCard.IMG_W, ScanCard.IMG_H)
            loader.signals.loaded.connect(self._on_thumb)
            self._pool.start(loader)

    def set_colormap_for_selection(self, colormap_key: str,
                                    clip_low: float = 1.0,
                                    clip_high: float = 99.0,
                                    processing: dict = None) -> int:
        """Apply colormap, scale and optional processing to selected cards. Returns count updated."""
        if not self._selected:
            return 0
        token = self._load_token
        for stem in self._selected:
            entry = next((e for e in self._entries if e.stem == stem), None)
            card  = self._cards.get(stem)
            if entry and card:
                self._card_colormaps[stem] = colormap_key
                loader = ThumbnailLoader(entry, colormap_key, token,
                                         ScanCard.IMG_W, ScanCard.IMG_H,
                                         clip_low, clip_high,
                                         processing=processing)
                loader.signals.loaded.connect(self._on_thumb)
                self._pool.start(loader)
        return len(self._selected)

    def get_entries(self) -> list[SxmFile]:
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
        for card in self._cards.values():
            self._grid.removeWidget(card)
        for i, entry in enumerate(self._entries):
            card = self._cards.get(entry.stem)
            if card:
                row, col = divmod(i, new_cols)
                self._grid.addWidget(card, row, col, Qt.AlignTop | Qt.AlignLeft)


# ── Browse tool panel (LEFT) ──────────────────────────────────────────────────
class BrowseToolPanel(QWidget):
    """Left-side control panel: folder, colormap, scale, processing, export."""
    open_folder_requested      = Signal()
    colormap_apply_requested   = Signal(str)
    scale_changed              = Signal(float, float)
    processing_apply_requested = Signal(dict)
    measure_requested          = Signal()
    export_requested           = Signal()

    def __init__(self, t: dict, cfg: dict, parent=None):
        super().__init__(parent)
        self._t         = t
        self._clip_low  = cfg.get("clip_low",  1.0)
        self._clip_high = cfg.get("clip_high", 99.0)
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
        lay.setSpacing(4)

        # ── Open folder button ─────────────────────────────────────────────────
        open_btn = QPushButton("Open SXM folder…")
        open_btn.setFont(QFont("Helvetica", 9))
        open_btn.setFixedHeight(30)
        open_btn.setCursor(QCursor(Qt.PointingHandCursor))
        open_btn.setObjectName("accentBtn")
        open_btn.clicked.connect(self.open_folder_requested.emit)
        lay.addWidget(open_btn)
        lay.addWidget(_sep())

        # ── Colormap ───────────────────────────────────────────────────────────
        cm_lbl = QLabel("Colormap")
        cm_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        lay.addWidget(cm_lbl)

        cm_row = QHBoxLayout()
        self.cmap_cb = QComboBox()
        self.cmap_cb.addItems(CMAP_NAMES)
        self.cmap_cb.setCurrentText(cfg.get("colormap", DEFAULT_CMAP_LABEL))
        self.cmap_cb.setFont(QFont("Helvetica", 9))
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setFont(QFont("Helvetica", 8))
        self._apply_btn.setFixedHeight(26)
        self._apply_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._apply_btn.setObjectName("accentBtn")
        self._apply_btn.clicked.connect(self._on_apply)
        cm_row.addWidget(self.cmap_cb, 1)
        cm_row.addWidget(self._apply_btn)
        lay.addLayout(cm_row)

        self._sel_hint = QLabel("Select images first (Ctrl+click for multi-select)")
        self._sel_hint.setFont(QFont("Helvetica", 7))
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
            lbl.setFixedWidth(44)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(mn, mx)
            sl.setValue(int(init_val))
            val_lbl = QLabel(f"{init_val:.0f}%")
            val_lbl.setFont(QFont("Helvetica", 8))
            val_lbl.setFixedWidth(32)
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

        # ── Processing (collapsible) ───────────────────────────────────────────
        self._proc_toggle = QPushButton("[+] Processing")
        self._proc_toggle.setFlat(True)
        self._proc_toggle.setFont(QFont("Helvetica", 9, QFont.Bold))
        self._proc_toggle.setCursor(QCursor(Qt.PointingHandCursor))
        self._proc_toggle.clicked.connect(self._toggle_proc)
        lay.addWidget(self._proc_toggle)

        self._proc_widget = QWidget()
        proc_lay = QVBoxLayout(self._proc_widget)
        proc_lay.setContentsMargins(4, 2, 0, 2)
        proc_lay.setSpacing(4)

        self._rbl_cb = QCheckBox("Remove bad lines")
        self._rbl_cb.setFont(QFont("Helvetica", 8))
        proc_lay.addWidget(self._rbl_cb)

        bg_row = QHBoxLayout()
        bg_lbl = QLabel("Background:")
        bg_lbl.setFont(QFont("Helvetica", 8))
        bg_lbl.setFixedWidth(76)
        self._bg_combo = QComboBox()
        self._bg_combo.addItems(["None", "Plane", "Quadratic"])
        self._bg_combo.setFont(QFont("Helvetica", 8))
        bg_row.addWidget(bg_lbl)
        bg_row.addWidget(self._bg_combo, 1)
        proc_lay.addLayout(bg_row)

        fft_row = QHBoxLayout()
        fft_lbl = QLabel("FFT filter:")
        fft_lbl.setFont(QFont("Helvetica", 8))
        fft_lbl.setFixedWidth(76)
        self._fft_combo = QComboBox()
        self._fft_combo.addItems(["None", "Low-pass", "High-pass"])
        self._fft_combo.setFont(QFont("Helvetica", 8))
        self._fft_combo.currentIndexChanged.connect(self._on_fft_mode_changed)
        fft_row.addWidget(fft_lbl)
        fft_row.addWidget(self._fft_combo, 1)
        proc_lay.addLayout(fft_row)

        self._fft_cutoff_widget = QWidget()
        cutoff_lay = QHBoxLayout(self._fft_cutoff_widget)
        cutoff_lay.setContentsMargins(0, 0, 0, 0)
        cutoff_lbl = QLabel("Cutoff:")
        cutoff_lbl.setFont(QFont("Helvetica", 8))
        cutoff_lbl.setFixedWidth(46)
        self._fft_sl = QSlider(Qt.Horizontal)
        self._fft_sl.setRange(1, 50)
        self._fft_sl.setValue(10)
        self._fft_cutoff_lbl = QLabel("10%")
        self._fft_cutoff_lbl.setFont(QFont("Helvetica", 8))
        self._fft_cutoff_lbl.setFixedWidth(32)
        self._fft_sl.valueChanged.connect(lambda v: self._fft_cutoff_lbl.setText(f"{v}%"))
        cutoff_lay.addWidget(cutoff_lbl)
        cutoff_lay.addWidget(self._fft_sl, 1)
        cutoff_lay.addWidget(self._fft_cutoff_lbl)
        proc_lay.addWidget(self._fft_cutoff_widget)
        self._fft_cutoff_widget.setVisible(False)

        self._proc_apply_btn = QPushButton("Apply processing")
        self._proc_apply_btn.setFont(QFont("Helvetica", 8))
        self._proc_apply_btn.setFixedHeight(26)
        self._proc_apply_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._proc_apply_btn.setObjectName("accentBtn")
        self._proc_apply_btn.clicked.connect(self._on_proc_apply)
        proc_lay.addWidget(self._proc_apply_btn)

        proc_lay.addWidget(_sep())

        self._meas_btn = QPushButton("Measure Periodicity")
        self._meas_btn.setFont(QFont("Helvetica", 8))
        self._meas_btn.setFixedHeight(26)
        self._meas_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._meas_btn.clicked.connect(self.measure_requested.emit)
        proc_lay.addWidget(self._meas_btn)

        self._meas_result = QLabel("")
        self._meas_result.setFont(QFont("Courier", 8))
        self._meas_result.setWordWrap(True)
        self._meas_result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        proc_lay.addWidget(self._meas_result)

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
        self._proc_toggle.setText("[-] Processing" if vis else "[+] Processing")

    def _on_fft_mode_changed(self, idx: int):
        self._fft_cutoff_widget.setVisible(idx != 0)

    def _on_proc_apply(self):
        bg_map  = {0: None, 1: 1, 2: 2}
        fft_map = {0: None, 1: 'low_pass', 2: 'high_pass'}
        cfg = {
            'remove_bad_lines': self._rbl_cb.isChecked(),
            'bg_order':         bg_map[self._bg_combo.currentIndex()],
            'fft_mode':         fft_map[self._fft_combo.currentIndex()],
            'fft_cutoff':       self._fft_sl.value() / 100.0,
            'fft_window':       'hanning',
        }
        self.processing_apply_requested.emit(cfg)

    # ── Public API ─────────────────────────────────────────────────────────────
    def get_clip_values(self) -> tuple[float, float]:
        return self._clip_low, self._clip_high

    def update_selection_hint(self, n: int):
        if n == 0:
            self._sel_hint.setText("Select images first (Ctrl+click for multi-select)")
        elif n == 1:
            self._sel_hint.setText("1 image selected")
        else:
            self._sel_hint.setText(f"{n} images selected")

    def show_periodicity_result(self, results: list):
        if not results:
            self._meas_result.setText("No peaks found.")
            return
        lines = [f"Peak {i}: {r['period_m']*1e9:.3f} nm  {r['angle_deg']:.1f}°"
                 for i, r in enumerate(results, 1)]
        self._meas_result.setText("\n".join(lines))

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
        self.dim_lbl = QLabel("")
        self.dim_lbl.setFont(QFont("Helvetica", 8))
        lay.addWidget(self.name_lbl)
        lay.addWidget(self.dim_lbl)
        lay.addWidget(_sep())

        ch_hdr = QLabel("Channels")
        ch_hdr.setFont(QFont("Helvetica", 9, QFont.Bold))
        lay.addWidget(ch_hdr)

        ch_grid = QGridLayout()
        ch_grid.setSpacing(4)
        ch_grid.setContentsMargins(0, 0, 0, 0)
        self._ch_img_lbls:  list[QLabel] = []
        self._ch_name_lbls: list[QLabel] = []
        for i, name in enumerate(PLANE_NAMES):
            r, c = divmod(i, 2)
            cell = QWidget()
            cell_lay = QVBoxLayout(cell)
            cell_lay.setContentsMargins(0, 0, 0, 0)
            cell_lay.setSpacing(1)
            img_lbl = QLabel()
            img_lbl.setFixedSize(118, 94)
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setFrameShape(QFrame.StyledPanel)
            nm_lbl = QLabel(name)
            nm_lbl.setFont(QFont("Helvetica", 7))
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
        meta_hdr.setFont(QFont("Helvetica", 9, QFont.Bold))
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search…")
        self.search_box.setFont(QFont("Helvetica", 8))
        self.search_box.setFixedHeight(22)
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
        self.meta_table.setFont(QFont("Helvetica", 8))
        self.meta_table.verticalHeader().setDefaultSectionSize(17)
        self.meta_table.setShowGrid(False)
        lay.addWidget(self.meta_table, 1)

    # ── Public API ─────────────────────────────────────────────────────────────
    def show_entry(self, entry: SxmFile, colormap_key: str):
        self.name_lbl.setText(entry.stem)
        self.dim_lbl.setText(f"{entry.Nx} × {entry.Ny} px")
        self._load_channels(entry, colormap_key)
        self._load_metadata(entry)

    def clear(self):
        self.name_lbl.setText("No scan selected")
        self.dim_lbl.setText("")
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

    # ── Internal ───────────────────────────────────────────────────────────────
    def _load_channels(self, entry: SxmFile, colormap_key: str):
        self._ch_token = object()
        sigs = ChannelSignals()
        sigs.loaded.connect(self._on_ch_loaded)
        self._ch_sigs = sigs
        for i in range(4):
            loader = ChannelLoader(entry, i, colormap_key,
                                   self._ch_token, 114, 90, sigs,
                                   self._clip_low, self._clip_high)
            self._pool.start(loader)

    @Slot(int, QPixmap, object)
    def _on_ch_loaded(self, idx: int, pixmap: QPixmap, token):
        if token is not self._ch_token:
            return
        lbl = self._ch_img_lbls[idx]
        lbl.setPixmap(pixmap.scaled(lbl.width(), lbl.height(),
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _load_metadata(self, entry: SxmFile):
        hdr      = parse_sxm_header(entry.path)
        priority = [
            "REC_DATE", "REC_TIME", "SCAN_PIXELS", "SCAN_RANGE",
            "SCAN_OFFSET", "SCAN_ANGLE", "SCAN_DIR", "BIAS",
            "REC_TEMP", "ACQ_TIME", "SCAN_TIME", "COMMENT",
        ]
        rows: list[tuple[str, str]] = []
        seen: set[str]              = set()
        for k in priority:
            if k in hdr and hdr[k].strip():
                rows.append((k, hdr[k].strip()))
                seen.add(k)
        for k, v in hdr.items():
            if k not in seen and v.strip():
                rows.append((k, v.strip()))
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


# ── Export dialog ────────────────────────────────────────────────────────────
class ExportDialog(QDialog):
    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export PNG")
        self.setFixedSize(340, 240)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(8)

        self._scalebar_cb = QCheckBox("Add scale bar")
        self._scalebar_cb.setChecked(True)
        self._scalebar_cb.toggled.connect(self._on_scalebar_toggled)
        lay.addWidget(self._scalebar_cb)

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

        lay.addWidget(self._sb_opts)
        lay.addWidget(_sep())

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
        lay.addStretch()

    def _on_scalebar_toggled(self, checked: bool):
        self._sb_opts.setEnabled(checked)

    def get_settings(self) -> dict:
        unit = "nm"
        if self._ang_rb.isChecked():
            unit = "Å"
        elif self._pm_rb.isChecked():
            unit = "pm"
        return {
            'add_scalebar':  self._scalebar_cb.isChecked(),
            'scalebar_unit': unit,
            'scalebar_pos':  self._pos_cb.currentText(),
        }


# ── Convert panel ─────────────────────────────────────────────────────────────
class ConvertPanel(QWidget):
    def __init__(self, t: dict, cfg: dict, parent=None):
        super().__init__(parent)
        self._t = t
        lay     = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 8)
        lay.setSpacing(8)

        # Input folder (always visible)
        in_row = QHBoxLayout()
        in_lbl = QLabel("Input folder:")
        in_lbl.setFixedWidth(100)
        in_lbl.setFont(QFont("Helvetica", 9))
        self.input_entry = QLineEdit()
        self.input_entry.setFont(QFont("Helvetica", 9))
        self.input_entry.setPlaceholderText("Select folder with .dat files…")
        in_btn = QPushButton("Browse")
        in_btn.setFont(QFont("Helvetica", 8))
        in_btn.setFixedWidth(70)
        in_btn.clicked.connect(self._browse_input)
        in_row.addWidget(in_lbl)
        in_row.addWidget(self.input_entry)
        in_row.addWidget(in_btn)
        lay.addLayout(in_row)

        # Custom output checkbox + row (hidden by default)
        self._custom_out_cb = QCheckBox("Custom output folder")
        self._custom_out_cb.setFont(QFont("Helvetica", 9))
        self._custom_out_cb.setChecked(cfg.get("custom_output", False))
        self._custom_out_cb.toggled.connect(self._toggle_output_row)
        lay.addWidget(self._custom_out_cb)

        self._out_row_widget = QWidget()
        out_row = QHBoxLayout(self._out_row_widget)
        out_row.setContentsMargins(0, 0, 0, 0)
        out_lbl = QLabel("Output folder:")
        out_lbl.setFixedWidth(100)
        out_lbl.setFont(QFont("Helvetica", 9))
        self.output_entry = QLineEdit()
        self.output_entry.setFont(QFont("Helvetica", 9))
        self.output_entry.setPlaceholderText("Defaults to input folder…")
        out_btn = QPushButton("Browse")
        out_btn.setFont(QFont("Helvetica", 8))
        out_btn.setFixedWidth(70)
        out_btn.clicked.connect(self._browse_output)
        out_row.addWidget(out_lbl)
        out_row.addWidget(self.output_entry)
        out_row.addWidget(out_btn)
        lay.addWidget(self._out_row_widget)
        self._out_row_widget.setVisible(cfg.get("custom_output", False))

        lay.addWidget(_sep())

        log_hdr = QHBoxLayout()
        log_lbl = QLabel("Conversion log")
        log_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        clear_btn = QPushButton("Clear")
        clear_btn.setFont(QFont("Helvetica", 8))
        clear_btn.setFixedWidth(50)
        clear_btn.clicked.connect(lambda: self.log_text.clear())
        log_hdr.addWidget(log_lbl)
        log_hdr.addStretch()
        log_hdr.addWidget(clear_btn)
        lay.addLayout(log_hdr)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
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

    def log(self, msg: str, tag: str = "info"):
        colors = {
            "ok":   "#a6e3a1",
            "err":  "#f38ba8",
            "warn": "#fab387",
            "info": "#cdd6f4",
        }
        color = colors.get(tag, "#cdd6f4")
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
        lay.setContentsMargins(10, 12, 10, 8)
        lay.setSpacing(6)

        hdr = QLabel("Output format")
        hdr.setFont(QFont("Helvetica", 9, QFont.Bold))
        lay.addWidget(hdr)

        self.png_cb = QCheckBox("PNG preview")
        self.sxm_cb = QCheckBox("SXM (Nanonis)")
        self.png_cb.setChecked(cfg.get("do_png", False))
        self.sxm_cb.setChecked(cfg.get("do_sxm", True))
        for cb in (self.png_cb, self.sxm_cb):
            cb.setFont(QFont("Helvetica", 9))
            lay.addWidget(cb)

        lay.addWidget(_sep())

        self._adv_btn = QPushButton("[+] Advanced")
        self._adv_btn.setFlat(True)
        self._adv_btn.setFont(QFont("Helvetica", 9))
        self._adv_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._adv_btn.clicked.connect(self._toggle_adv)
        lay.addWidget(self._adv_btn)

        self._adv_widget = QWidget()
        adv_lay = QVBoxLayout(self._adv_widget)
        adv_lay.setContentsMargins(0, 0, 0, 0)

        def _spin_row(label: str, val: float, mn: float, mx: float):
            row  = QHBoxLayout()
            lbl  = QLabel(label)
            lbl.setFont(QFont("Helvetica", 8))
            lbl.setFixedWidth(92)
            spin = QDoubleSpinBox()
            spin.setRange(mn, mx)
            spin.setValue(val)
            spin.setSingleStep(0.5)
            spin.setFont(QFont("Helvetica", 8))
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
        self.run_btn.setFont(QFont("Helvetica", 12, QFont.Bold))
        self.run_btn.setFixedHeight(42)
        self.run_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.run_btn.setObjectName("accentBtn")
        lay.addWidget(self.run_btn)

        lay.addWidget(_sep())

        self.fcount_lbl = QLabel("")
        self.fcount_lbl.setFont(QFont("Helvetica", 8))
        self.fcount_lbl.setWordWrap(True)
        lay.addWidget(self.fcount_lbl)

        lay.addStretch()

        credit = QLabel(
            "SPMQT-Lab  |  Dr. Peter Jacobson\n"
            "The University of Queensland\n"
            "Original code by Rohan Platts"
        )
        credit.setFont(QFont("Helvetica", 7))
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

        def _row(text, size=10, bold=False, sub=False):
            lbl = QLabel(text)
            f   = QFont("Helvetica", size)
            f.setBold(bold)
            lbl.setFont(f)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setWordWrap(True)
            if sub:
                lbl.setStyleSheet(f"color: {t['sub_fg']};")
            lay.addWidget(lbl)

        _row("ProbeFlow", 15, bold=True)
        _row("Createc → Nanonis File Conversion", 10, sub=True)
        lay.addWidget(_sep())
        _row("Developed at SPMQT-Lab", 10, bold=True)
        _row("Under the supervision of Dr. Peter Jacobson\nThe University of Queensland", 9, sub=True)
        lay.addWidget(_sep())
        _row("Original code by Rohan Platts", 10, bold=True)
        _row("The core conversion algorithms were built by Rohan Platts.\n"
             "This GUI is a refactored and extended version.", 9, sub=True)
        lay.addWidget(_sep())

        gh_btn = QPushButton("View on GitHub")
        gh_btn.setFont(QFont("Helvetica", 9))
        gh_btn.setCursor(QCursor(Qt.PointingHandCursor))
        gh_btn.setObjectName("accentBtn")
        gh_btn.setFixedHeight(32)
        gh_btn.clicked.connect(lambda: webbrowser.open(GITHUB_URL))
        lay.addWidget(gh_btn)


# ── Navbar ────────────────────────────────────────────────────────────────────
class Navbar(QWidget):
    theme_toggle_clicked = Signal()
    about_clicked        = Signal()

    def __init__(self, dark: bool, parent=None):
        super().__init__(parent)
        self._dark = dark
        self.setFixedHeight(NAVBAR_H)
        self.setStyleSheet(f"background-color: {NAVBAR_BG};")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(6)

        nav_logo_path = LOGO_NAV_PATH if LOGO_NAV_PATH.exists() else None
        if nav_logo_path:
            logo_lbl = QLabel()
            logo_lbl.setStyleSheet("background: transparent;")
            pix = QPixmap(str(nav_logo_path))
            logo_lbl.setPixmap(
                pix.scaled(9999, 46, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logo_lbl.setCursor(QCursor(Qt.PointingHandCursor))
            logo_lbl.mousePressEvent = lambda e: webbrowser.open(GITHUB_URL)
            lay.addWidget(logo_lbl)

        lay.addStretch()

        def _nbtn(text: str, slot) -> QPushButton:
            btn = QPushButton(text)
            btn.setFont(QFont("Helvetica", 9))
            btn.setObjectName("navBtn")
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.clicked.connect(slot)
            lay.addWidget(btn)
            return btn

        self._theme_btn = _nbtn(
            "Light mode" if dark else "Dark mode",
            self.theme_toggle_clicked.emit,
        )
        _nbtn("GitHub", lambda: webbrowser.open(GITHUB_URL))
        _nbtn("About",  self.about_clicked.emit)

    def set_dark(self, dark: bool):
        self._dark = dark
        self._theme_btn.setText("Light mode" if dark else "Dark mode")


# ── Main window ───────────────────────────────────────────────────────────────
class ProbeFlowWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProbeFlow")
        self.setMinimumSize(980, 660)

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
        self._tab_bar.setFixedHeight(38)
        tab_lay = QHBoxLayout(self._tab_bar)
        tab_lay.setContentsMargins(0, 0, 0, 0)
        tab_lay.setSpacing(0)
        self._tab_browse  = QPushButton("Browse")
        self._tab_convert = QPushButton("Convert")
        for btn in (self._tab_browse, self._tab_convert):
            btn.setFont(QFont("Helvetica", 9, QFont.Bold))
            btn.setFixedHeight(38)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.setFlat(True)
            tab_lay.addWidget(btn)
        tab_lay.addStretch()
        self._tab_browse.clicked.connect(lambda: self._switch_mode("browse"))
        self._tab_convert.clicked.connect(lambda: self._switch_mode("convert"))
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
        self._browse_tools.setFixedWidth(230)
        self._grid         = ThumbnailGrid(t)
        browse_split = QSplitter(Qt.Horizontal)
        browse_split.setHandleWidth(3)
        browse_split.addWidget(self._browse_tools)
        browse_split.addWidget(self._grid)
        browse_split.setStretchFactor(0, 0)
        browse_split.setStretchFactor(1, 1)

        self._conv_panel = ConvertPanel(t, self._cfg)
        self._content_stack.addWidget(browse_split)
        self._content_stack.addWidget(self._conv_panel)
        self._splitter.addWidget(self._content_stack)

        # ── Right: sidebar stack ───────────────────────────────────────────────
        self._sidebar_stack   = QStackedWidget()
        self._sidebar_stack.setFixedWidth(280)
        self._browse_info     = BrowseInfoPanel(t, self._cfg)
        self._convert_sidebar = ConvertSidebar(t, self._cfg)
        self._sidebar_stack.addWidget(self._browse_info)
        self._sidebar_stack.addWidget(self._convert_sidebar)
        self._splitter.addWidget(self._sidebar_stack)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

        # Wire signals
        self._browse_tools.open_folder_requested.connect(self._open_browse_folder)
        self._grid.entry_selected.connect(self._on_entry_select)
        self._grid.selection_changed.connect(self._on_selection_changed)
        self._browse_tools.colormap_apply_requested.connect(self._on_apply_colormap)
        self._browse_tools.scale_changed.connect(self._on_scale_changed)
        self._browse_tools.processing_apply_requested.connect(self._on_processing_apply)
        self._browse_tools.measure_requested.connect(self._on_measure_periodicity)
        self._browse_tools.export_requested.connect(self._on_export)
        self._convert_sidebar.run_btn.clicked.connect(self._run)
        self._conv_panel.input_entry.textChanged.connect(self._update_count)

        # Status bar
        self._status_bar = QStatusBar()
        self._status_bar.setFont(QFont("Helvetica", 8))
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
        else:
            self._content_stack.setCurrentIndex(1)
            self._sidebar_stack.setCurrentIndex(1)
            self._update_count(self._conv_panel.input_entry.text())
        self._update_tab_styles()

    def _update_tab_styles(self):
        t = THEMES["dark" if self._dark else "light"]
        for btn, name in ((self._tab_browse, "browse"),
                          (self._tab_convert, "convert")):
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
                        color: {t['sub_fg']};
                        border: none;
                        padding: 0 18px;
                    }}
                    QPushButton:hover {{ color: {t['fg']}; }}
                """)

    # ── Browse ─────────────────────────────────────────────────────────────────
    def _open_browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Open folder containing .sxm files")
        if not d:
            return
        self._switch_mode("browse")
        entries = scan_sxm_folder(Path(d))
        self._grid.load(entries, folder_path=d)
        self._n_loaded = len(entries)
        self._status_bar.showMessage(
            f"Loaded {self._n_loaded} scan(s) — grayscale by default | "
            "Select + Apply to colorize")
        self._browse_info.clear()

    def _on_entry_select(self, entry: SxmFile):
        cmap_key = self._grid._card_colormaps.get(entry.stem, DEFAULT_CMAP_KEY)
        self._browse_info.show_entry(entry, cmap_key)
        n_sel = len(self._grid.get_selected())
        self._status_bar.showMessage(
            f"{entry.stem}  |  {entry.Nx}×{entry.Ny} px  |  "
            f"{n_sel} selected / {self._n_loaded} total")

    def _on_selection_changed(self, n_selected: int):
        self._browse_tools.update_selection_hint(n_selected)

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
                    self._browse_info._load_channels(entry, cmap_key)

    def _on_scale_changed(self, clip_low: float, clip_high: float):
        cmap_key = CMAP_KEY.get(
            self._browse_tools.cmap_cb.currentText(), DEFAULT_CMAP_KEY)
        self._browse_info.update_clip(clip_low, clip_high)
        n = self._grid.set_colormap_for_selection(cmap_key,
                                                   clip_low=clip_low,
                                                   clip_high=clip_high)
        if n > 0:
            self._status_bar.showMessage(
                f"Scale: {clip_low:.0f}%–{clip_high:.0f}% on {n} image{'s' if n > 1 else ''}")
            primary = self._grid.get_primary()
            if primary:
                entry = next((e for e in self._grid.get_entries()
                              if e.stem == primary), None)
                if entry:
                    self._browse_info._load_channels(entry, cmap_key)

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
            if cfg.get('remove_bad_lines'):
                steps.append("bad lines removed")
            if cfg.get('bg_order') is not None:
                steps.append(f"bg order {cfg['bg_order']}")
            if cfg.get('fft_mode'):
                steps.append(
                    f"FFT {cfg['fft_mode']} {cfg.get('fft_cutoff', 0.1)*100:.0f}%")
            desc = ", ".join(steps) if steps else "no processing"
            self._status_bar.showMessage(
                f"Processing ({desc}) applied to {n} image{'s' if n > 1 else ''}")

    def _on_measure_periodicity(self):
        primary = self._grid.get_primary()
        if not primary:
            self._status_bar.showMessage("Select an image first")
            return
        entry = next((e for e in self._grid.get_entries() if e.stem == primary), None)
        if not entry:
            return
        arr = read_sxm_plane_raw(entry.path, 0)
        if arr is None:
            self._status_bar.showMessage("Could not read scan data")
            return
        hdr = parse_sxm_header(entry.path)
        w_m, h_m = _sxm_scan_range(hdr)
        Ny, Nx = arr.shape
        px_x = w_m / Nx if Nx > 0 and w_m > 0 else 1e-10
        px_y = h_m / Ny if Ny > 0 and h_m > 0 else 1e-10
        try:
            results = _proc.measure_periodicity(arr, px_x, px_y)
            self._browse_tools.show_periodicity_result(results)
            self._status_bar.showMessage(
                f"Periodicity: {len(results)} peak(s) found")
        except Exception as exc:
            self._status_bar.showMessage(f"Periodicity error: {exc}")

    def _on_export(self):
        primary = self._grid.get_primary()
        if not primary:
            self._status_bar.showMessage("Select an image first")
            return
        entry = next((e for e in self._grid.get_entries() if e.stem == primary), None)
        if not entry:
            return

        t   = THEMES["dark" if self._dark else "light"]
        dlg = ExportDialog(t, self)
        dlg.setStyleSheet(QApplication.instance().styleSheet())
        if dlg.exec() != QDialog.Accepted:
            return
        settings = dlg.get_settings()

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save PNG", str(Path.home() / f"{entry.stem}.png"),
            "PNG images (*.png)")
        if not out_path:
            return

        arr = read_sxm_plane_raw(entry.path, 0)
        if arr is None:
            self._status_bar.showMessage("Could not read scan data")
            return

        hdr = parse_sxm_header(entry.path)
        w_m, h_m = _sxm_scan_range(hdr)
        clip_low, clip_high = self._browse_tools.get_clip_values()
        cmap_key = self._grid._card_colormaps.get(entry.stem, DEFAULT_CMAP_KEY)

        try:
            _proc.export_png(
                arr, out_path, cmap_key, clip_low, clip_high,
                lut_fn=lambda key: _get_lut(key),
                scan_range_m=(w_m, h_m),
                add_scalebar=settings['add_scalebar'],
                scalebar_unit=settings['scalebar_unit'],
                scalebar_pos=settings['scalebar_pos'],
            )
            self._status_bar.showMessage(f"Exported → {out_path}")
        except Exception as exc:
            self._status_bar.showMessage(f"Export error: {exc}")

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
        entries = scan_sxm_folder(sxm_dir) if sxm_dir.exists() else []
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
    padding: 4px 10px;
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
    padding: 3px 6px;
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
}}
QLineEdit {{
    background-color: {t['entry_bg']};
    color: {t['fg']};
    border: 1px solid {t['sep']};
    border-radius: 3px;
    padding: 3px 6px;
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
QTableWidget::item {{ padding: 1px 4px; }}
QTableWidget::item:selected {{
    background-color: {t['tree_sel']};
    color: {t['fg']};
}}
QHeaderView::section {{
    background-color: {t['tree_head']};
    color: {t['fg']};
    border: none;
    padding: 4px;
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
QCheckBox {{ color: {t['fg']}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {t['sep']};
    border-radius: 2px;
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
    padding: 2px 4px;
}}
QSplitter::handle {{ background-color: {t['splitter']}; }}
QStatusBar {{
    background-color: {t['status_bg']};
    color: {t['status_fg']};
    border-top: 1px solid {t['sep']};
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
