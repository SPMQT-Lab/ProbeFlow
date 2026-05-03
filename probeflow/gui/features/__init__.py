"""Specialized GUI tools for the ProbeFlow Features tab.

This module is intentionally separate from :mod:`probeflow.gui`.

The Browse tab should stay focused on file selection, thumbnails, display
scale, and lightweight thumbnail corrections. The Viewer should stay focused
on canonical image-processing/export operations. Tools in this file are
different: they are feature analyses or specialized one-off transforms that
act on a selected scan after the user explicitly loads it into the Features
workspace.

Future Codex/Claude/readthrough note:
    Keep particle counting, template counting, lattice extraction, and future
    TV-denoise/background-removal panels here (or in sibling Features modules),
    not in Browse/Viewer, unless the tool becomes a normal canonical processing
    operation. This boundary prevents optional feature dependencies and more
    experimental workflows from creating odd dependencies in basic browsing,
    conversion, thumbnail rendering, or standard image manipulation.
"""

from __future__ import annotations

import numpy as np
import os as _os
_os.environ.setdefault("QT_API", "pyside6")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import QObject, QRunnable, Qt, Signal, Slot
from PySide6.QtGui import QCursor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from probeflow.processing.display import clip_range_from_array as _clip_range_from_array


PLANE_NAMES = ["Z fwd", "Z bwd", "I fwd", "I bwd"]


def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedHeight(1)
    return line


class _FeaturesWorkerSignals(QObject):
    finished = Signal(str, object, str)   # mode, result-or-None, error-or-""


class _FeaturesWorker(QRunnable):
    """Run Features-tab analyses off the GUI thread.

    The imports stay lazy on purpose. OpenCV/scikit-learn/lattice dependencies
    belong to Features workflows and should not be imported merely to browse a
    folder or open a normal image-processing Viewer.
    """

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
                from probeflow.analysis.features import segment_particles
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
                from probeflow.analysis.features import count_features
                res = count_features(
                    self._arr, self._params["template"], self._px,
                    min_correlation=self._params.get("min_correlation", 0.5),
                    min_distance_m=self._params.get("min_distance_m"),
                )
            elif self._mode == "lattice":
                from probeflow.analysis.lattice import extract_lattice, LatticeParams
                res = extract_lattice(self._arr, self._px,
                                      params=LatticeParams())
            else:
                raise ValueError(f"Unknown mode {self._mode!r}")
            self._signals.finished.emit(self._mode, res, "")
        except Exception as exc:
            self._signals.finished.emit(self._mode, None, str(exc))


class FeaturesPanel(QWidget):
    """Center widget for the Features tab.

    This is a dedicated analysis workspace: load one selected Browse scan,
    inspect overlays/results, and export analysis JSON. It intentionally does
    not mutate Browse thumbnails or Viewer processing state.
    """

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

        self._title = QLabel("FeatureCounting - load a scan from the Browse tab, then run an analysis.")
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

        powered = QLabel("Powered by UniMR")
        powered.setFont(QFont("Helvetica", 8))
        powered.setAlignment(Qt.AlignCenter)
        powered.setStyleSheet("color: #888;")
        lay.addWidget(powered)

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
            f"{entry.stem}  -  {plane_lbl}  -  "
            f"{arr.shape[1]}x{arr.shape[0]} px  "
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
            ["#", "x (nm)", "y (nm)", "area (nm^2)"])
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
            ("gamma", f"{lat.gamma_deg:.2f} deg"),
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

    def begin_template_crop(self):
        if self._arr is None:
            return
        self._cropping = True
        self._crop_start = None
        self._crop_rect  = None
        self._title.setText("Template crop - drag a rectangle over one motif, release to set.")
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
            self._title.setText("Template crop cancelled - rectangle too small.")
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
            f"Template captured - {tw}x{th} px.  Press 'Run' to count matches.")
        self._redraw()

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
                          f"template: {tw}x{th} px",
                          color="#f9e2af", fontsize=8,
                          bbox=dict(boxstyle="round", fc="#1e1e2e88", ec="none"))

        self._canvas.draw_idle()


class FeaturesSidebar(QWidget):
    """Right sidebar for Features-tab analysis parameters."""

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

        self._mode_stack = QStackedWidget()
        self._mode_stack.addWidget(self._build_particles_tab())
        self._mode_stack.addWidget(self._build_template_tab())
        self._mode_stack.addWidget(self._build_lattice_tab())
        lay.addWidget(self._mode_stack)

        lay.addWidget(_sep())

        self._run_btn = QPushButton("Run")
        self._run_btn.setFont(QFont("Helvetica", 10, QFont.Bold))
        self._run_btn.setFixedHeight(32)
        self._run_btn.setObjectName("accentBtn")
        self._run_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._run_btn.clicked.connect(lambda: self.run_requested.emit(self._current_mode()))
        lay.addWidget(self._run_btn)

        self._export_btn = QPushButton("Export JSON...")
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
        row2.addWidget(QLabel("min area (nm^2):"))
        self._min_area_spin = QDoubleSpinBox()
        self._min_area_spin.setRange(0.0, 1e6)
        self._min_area_spin.setDecimals(2)
        self._min_area_spin.setValue(0.5)
        row2.addWidget(self._min_area_spin)
        l.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("max area (nm^2):"))
        self._max_area_spin = QDoubleSpinBox()
        self._max_area_spin.setRange(0.0, 1e9)
        self._max_area_spin.setDecimals(2)
        self._max_area_spin.setValue(0.0)  # 0 -> None
        row3.addWidget(self._max_area_spin)
        l.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("sigma-clip:"))
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

        crop_btn = QPushButton("Crop template from image...")
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
        self._dist_spin.setValue(0.0)   # 0 -> auto (half template side)
        row2.addWidget(self._dist_spin)
        l.addLayout(row2)

        hint = QLabel("Tip: draw a tight rectangle over one motif. Distance of 0 -> auto.")
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
