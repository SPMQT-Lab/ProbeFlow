"""TV-denoise tab — exposes probeflow.processing.tv_denoise in the GUI.

Sibling to :mod:`probeflow.gui.features`. Kept separate so the optional
TV-denoise dependencies and UI live alongside the kernel they wrap, without
bloating the main Browse/Viewer file.
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("QT_API", "pyside6")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import QObject, QRunnable, Qt, Signal, Slot
from PySide6.QtGui import QCursor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from probeflow.processing.display import clip_range_from_array as _clip_range_from_array


PLANE_NAMES = ["Z fwd", "Z bwd", "I fwd", "I bwd"]


def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


class _TVWorkerSignals(QObject):
    finished = Signal(object, str)   # denoised-or-None, error-or-""


class _TVWorker(QRunnable):
    """Run tv_denoise off the GUI thread."""

    def __init__(self, arr: np.ndarray, params: dict,
                 signals: _TVWorkerSignals):
        super().__init__()
        self._arr     = arr
        self._params  = params
        self._signals = signals

    @Slot()
    def run(self):
        try:
            from probeflow.processing import tv_denoise
            out = tv_denoise(self._arr, **self._params)
            self._signals.finished.emit(out, "")
        except Exception as exc:
            self._signals.finished.emit(None, str(exc))


class TVPanel(QWidget):
    """Center widget: side-by-side Original / Denoised preview."""

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self._t              = t
        self._entry          = None
        self._plane_idx      = 0
        self._arr            = None       # original
        self._denoised       = None       # last result
        self._pixel_size_m   = 1e-10
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 4)
        lay.setSpacing(6)

        self._title = QLabel("TV-denoise - load a scan from the Browse tab.")
        self._title.setFont(QFont("Helvetica", 11, QFont.Bold))
        self._title.setWordWrap(True)
        lay.addWidget(self._title)

        self._fig = Figure(figsize=(10, 5), dpi=90)
        self._fig.patch.set_alpha(0)
        self._ax_in  = self._fig.add_subplot(1, 2, 1)
        self._ax_out = self._fig.add_subplot(1, 2, 2)
        for ax in (self._ax_in, self._ax_out):
            ax.set_axis_off()
        self._canvas = FigureCanvasQTAgg(self._fig)
        lay.addWidget(self._canvas, 1)

        powered = QLabel("Powered by AiSurf")
        powered.setFont(QFont("Helvetica", 8))
        powered.setAlignment(Qt.AlignCenter)
        powered.setStyleSheet("color: #888;")
        lay.addWidget(powered)

    # ── Public API ─────────────────────────────────────────────────────────────
    def load_entry(self, entry, plane_idx: int, arr: np.ndarray,
                   pixel_size_m: float):
        self._entry        = entry
        self._plane_idx    = plane_idx
        self._arr          = arr
        self._denoised     = None
        self._pixel_size_m = pixel_size_m
        plane_lbl = (PLANE_NAMES[plane_idx]
                     if 0 <= plane_idx < len(PLANE_NAMES)
                     else f"plane {plane_idx}")
        self._title.setText(
            f"{entry.stem}  -  {plane_lbl}  -  "
            f"{arr.shape[1]}x{arr.shape[0]} px  "
            f"(px = {pixel_size_m * 1e12:.1f} pm)")
        self._redraw()

    def set_denoised(self, arr: np.ndarray | None):
        self._denoised = arr
        self._redraw()

    def current_entry(self):
        return self._entry

    def current_array(self):
        return self._arr

    def current_denoised(self):
        return self._denoised

    def current_pixel_size(self):
        return self._pixel_size_m

    def current_plane_idx(self) -> int:
        return self._plane_idx

    def _redraw(self):
        for ax in (self._ax_in, self._ax_out):
            ax.clear()
            ax.set_axis_off()

        if self._arr is not None:
            try:
                vmin, vmax = _clip_range_from_array(self._arr, 1.0, 99.0)
            except ValueError:
                vmin, vmax = 0.0, 1.0
            self._ax_in.imshow(self._arr, cmap="gray", vmin=vmin, vmax=vmax,
                               interpolation="nearest", origin="upper")
            self._ax_in.set_title("Original", fontsize=9, color="#888")

        if self._denoised is not None:
            try:
                vmin, vmax = _clip_range_from_array(self._denoised, 1.0, 99.0)
            except ValueError:
                vmin, vmax = 0.0, 1.0
            self._ax_out.imshow(self._denoised, cmap="gray",
                                vmin=vmin, vmax=vmax,
                                interpolation="nearest", origin="upper")
            self._ax_out.set_title("Denoised", fontsize=9, color="#888")
        elif self._arr is not None:
            self._ax_out.text(0.5, 0.5, "Run to preview",
                              transform=self._ax_out.transAxes,
                              ha="center", va="center",
                              color="#888", fontsize=10)

        self._canvas.draw_idle()


class TVSidebar(QWidget):
    """Right sidebar: TV-denoise parameters and actions."""

    load_from_browse_requested = Signal()
    run_requested              = Signal()
    save_png_requested         = Signal()
    revert_requested           = Signal()

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

        params_lbl = QLabel("Denoise parameters")
        params_lbl.setFont(QFont("Helvetica", 11, QFont.Bold))
        lay.addWidget(params_lbl)

        # Method ───────────────────────────────────────────────────────────────
        m_row = QHBoxLayout()
        m_row.addWidget(QLabel("Method:"))
        self._method_cb = QComboBox()
        self._method_cb.addItems(["huber_rof", "tv_l1"])
        self._method_cb.setToolTip(
            "huber_rof = smooth TV (good default).\n"
            "tv_l1     = aggressive on impulsive noise; staircases on curved terraces.")
        self._method_cb.currentTextChanged.connect(self._on_method_changed)
        m_row.addWidget(self._method_cb, 1)
        lay.addLayout(m_row)

        # Lambda ───────────────────────────────────────────────────────────────
        lam_row = QHBoxLayout()
        lam_row.addWidget(QLabel("λ (data fidelity):"))
        self._lam_spin = QDoubleSpinBox()
        self._lam_spin.setRange(0.001, 100.0)
        self._lam_spin.setDecimals(3)
        self._lam_spin.setSingleStep(0.01)
        self._lam_spin.setValue(0.05)
        self._lam_spin.setToolTip("Larger λ stays closer to the input.")
        lam_row.addWidget(self._lam_spin)
        lay.addLayout(lam_row)

        # Alpha (Huber only) ───────────────────────────────────────────────────
        alpha_row = QHBoxLayout()
        self._alpha_lbl = QLabel("α (Huber):")
        alpha_row.addWidget(self._alpha_lbl)
        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.0, 1.0)
        self._alpha_spin.setDecimals(3)
        self._alpha_spin.setSingleStep(0.01)
        self._alpha_spin.setValue(0.05)
        self._alpha_spin.setToolTip("Huber smoothing parameter (huber_rof only).")
        alpha_row.addWidget(self._alpha_spin)
        lay.addLayout(alpha_row)

        # Max iter ─────────────────────────────────────────────────────────────
        it_row = QHBoxLayout()
        it_row.addWidget(QLabel("Max iter:"))
        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1, 5000)
        self._iter_spin.setSingleStep(50)
        self._iter_spin.setValue(500)
        it_row.addWidget(self._iter_spin)
        lay.addLayout(it_row)

        # Nabla axis ───────────────────────────────────────────────────────────
        nb_row = QHBoxLayout()
        nb_row.addWidget(QLabel("Gradient axis:"))
        self._nabla_cb = QComboBox()
        self._nabla_cb.addItems(["both", "x", "y"])
        self._nabla_cb.setToolTip(
            "both = isotropic.\n"
            "x = removes vertical scratches.\n"
            "y = removes horizontal scratches (typical raster noise).")
        nb_row.addWidget(self._nabla_cb, 1)
        lay.addLayout(nb_row)

        lay.addWidget(_sep())

        # Run / Revert / Save ──────────────────────────────────────────────────
        self._run_btn = QPushButton("Run")
        self._run_btn.setFont(QFont("Helvetica", 10, QFont.Bold))
        self._run_btn.setFixedHeight(32)
        self._run_btn.setObjectName("accentBtn")
        self._run_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._run_btn.clicked.connect(self.run_requested.emit)
        lay.addWidget(self._run_btn)

        self._revert_btn = QPushButton("Revert")
        self._revert_btn.setFont(QFont("Helvetica", 9))
        self._revert_btn.setFixedHeight(26)
        self._revert_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._revert_btn.clicked.connect(self.revert_requested.emit)
        lay.addWidget(self._revert_btn)

        self._save_btn = QPushButton("Save denoised PNG…")
        self._save_btn.setFont(QFont("Helvetica", 9))
        self._save_btn.setFixedHeight(28)
        self._save_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._save_btn.clicked.connect(self.save_png_requested.emit)
        lay.addWidget(self._save_btn)

        self._status_lbl = QLabel("Load a scan to begin.")
        self._status_lbl.setFont(QFont("Helvetica", 9))
        self._status_lbl.setWordWrap(True)
        lay.addWidget(self._status_lbl)

        lay.addStretch(1)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

        self._on_method_changed(self._method_cb.currentText())

    # ── Public API ─────────────────────────────────────────────────────────────
    def plane_index(self) -> int:
        return self._plane_cb.currentIndex()

    def params(self) -> dict:
        method = self._method_cb.currentText()
        out = dict(
            method=method,
            lam=float(self._lam_spin.value()),
            max_iter=int(self._iter_spin.value()),
            nabla_comp=self._nabla_cb.currentText(),
        )
        if method == "huber_rof":
            out["alpha"] = float(self._alpha_spin.value())
        return out

    def set_status(self, msg: str):
        self._status_lbl.setText(msg)

    def set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._run_btn.setText("Running…" if running else "Run")

    def _on_method_changed(self, name: str):
        is_huber = (name == "huber_rof")
        self._alpha_spin.setEnabled(is_huber)
        self._alpha_lbl.setEnabled(is_huber)
