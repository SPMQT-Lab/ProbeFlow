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
                 pixel_size_x_m: float, pixel_size_y_m: float,
                 params: dict, signals: _FeaturesWorkerSignals):
        super().__init__()
        self._mode    = mode
        self._arr     = arr
        self._px      = float(pixel_size_m)
        self._px_x    = float(pixel_size_x_m)
        self._px_y    = float(pixel_size_y_m)
        self._params  = params
        self._signals = signals

    @Slot()
    def run(self):
        try:
            if self._mode == "particles":
                from probeflow.analysis.features import segment_particles
                res = segment_particles(
                    self._arr, self._px,
                    pixel_size_x_m=self._px_x,
                    pixel_size_y_m=self._px_y,
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
                    pixel_size_x_m=self._px_x,
                    pixel_size_y_m=self._px_y,
                    min_correlation=self._params.get("min_correlation", 0.5),
                    min_distance_m=self._params.get("min_distance_m"),
                )
            elif self._mode == "lattice":
                from probeflow.analysis.lattice import extract_lattice, LatticeParams
                res = extract_lattice(
                    self._arr,
                    self._px,
                    pixel_size_x_m=self._px_x,
                    pixel_size_y_m=self._px_y,
                    params=LatticeParams(),
                )
            elif self._mode == "classify":
                from probeflow.analysis.features import classify_particles
                particles = self._params["particles"]
                samples   = self._params["samples"]   # list of (class_name, Particle)
                res = classify_particles(self._arr, particles, samples)
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
        self._pixel_size_x_m = 1e-10
        self._pixel_size_y_m = 1e-10
        self._overlay_mode = "none"        # "particles" | "template" | "lattice"
        self._particles  = []
        self._detections = []
        self._lattice    = None
        self._template_arr = None
        self._cropping   = False
        self._crop_start = None
        self._crop_rect  = None
        self._current_mode: str = "particles"
        self._params_signature = None
        self._params_meta: dict | None = None
        self._sample_armed: bool = False
        self._sample_labels: dict = {}     # particle_index → {"name": str, "color": tuple}
        self._classifications: list = []
        self._classification_meta: dict | None = None
        self._label_history: list = []     # undo stack for sample labels
        self._full_xlim = None             # zoom: full-image x limits after first draw
        self._full_ylim = None             # zoom: full-image y limits after first draw
        self._reset_zoom_pending: bool = False
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
        self._canvas.mpl_connect("scroll_event",         self._on_scroll)

        # ── Zoom / view controls ────────────────────────────────────────────
        _zoom_row = QHBoxLayout()
        _zoom_row.setSpacing(4)
        _zi = QPushButton("🔍+")
        _zi.setFixedSize(36, 24)
        _zi.setToolTip("Zoom in  (or scroll ↑)")
        _zi.setFont(QFont("Helvetica", 9))
        _zi.clicked.connect(lambda: self._zoom_view(1.5))
        _zo = QPushButton("🔍−")
        _zo.setFixedSize(36, 24)
        _zo.setToolTip("Zoom out  (or scroll ↓)")
        _zo.setFont(QFont("Helvetica", 9))
        _zo.clicked.connect(lambda: self._zoom_view(1 / 1.5))
        _fit = QPushButton("⟲ Fit")
        _fit.setFixedHeight(24)
        _fit.setToolTip("Reset zoom to show the full image")
        _fit.setFont(QFont("Helvetica", 9))
        _fit.clicked.connect(self.reset_view)
        _zoom_row.addWidget(_zi)
        _zoom_row.addWidget(_zo)
        _zoom_row.addWidget(_fit)
        _zoom_row.addStretch(1)
        lay.addLayout(_zoom_row)

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
                    pixel_size_m: float, pixel_size_x_m: float | None = None,
                    pixel_size_y_m: float | None = None):
        self._entry        = entry
        self._plane_idx    = plane_idx
        self._arr          = arr
        self._pixel_size_m = pixel_size_m
        self._pixel_size_x_m = (
            float(pixel_size_x_m) if pixel_size_x_m is not None else float(pixel_size_m)
        )
        self._pixel_size_y_m = (
            float(pixel_size_y_m) if pixel_size_y_m is not None else float(pixel_size_m)
        )
        self._particles    = []
        self._detections   = []
        self._lattice      = None
        self._template_arr = None
        self._overlay_mode = "none"
        self._sample_labels = {}
        self._label_history = []
        self._full_xlim    = None
        self._full_ylim    = None
        self._reset_zoom_pending = False
        self._redraw()
        self._results_table.setRowCount(0)
        plane_lbl = PLANE_NAMES[plane_idx] if 0 <= plane_idx < len(PLANE_NAMES) else f"plane {plane_idx}"
        self._title.setText(
            f"{entry.stem}  -  {plane_lbl}  -  "
            f"{arr.shape[1]}x{arr.shape[0]} px  "
            f"(px = {self._pixel_size_x_m * 1e12:.1f} x "
            f"{self._pixel_size_y_m * 1e12:.1f} pm)")

    def current_entry(self):
        return self._entry

    def current_array(self):
        return self._arr

    def current_pixel_size(self):
        return self._pixel_size_m

    def current_pixel_sizes(self):
        return self._pixel_size_x_m, self._pixel_size_y_m

    def set_mode(self, mode: str) -> None:
        self._current_mode = mode

    def set_sample_selection_armed(self, armed: bool) -> None:
        self._sample_armed = armed

    def has_sample_labels(self) -> bool:
        return bool(self._sample_labels)

    def sample_label_rows(self) -> list:
        return [
            {
                "class_name": v["name"],
                "color_rgb": list(v["color"]),
                "particle_index": k,
            }
            for k, v in self._sample_labels.items()
        ]

    def clear_sample_labels(self) -> None:
        self._sample_labels = {}

    def _prompt_sample_label(
        self, current_name: str = "", current_color: tuple = (255, 255, 255)
    ) -> dict | None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Label", "Class name:", text=current_name)
        if not ok or not name.strip():
            return None
        return {"name": name.strip(), "color": current_color}

    def _edit_sample_label(self, particle) -> None:
        existing = self._sample_labels.get(particle.index, {})
        result = self._prompt_sample_label(
            current_name=existing.get("name", ""),
            current_color=existing.get("color", (255, 255, 255)),
        )
        if result is not None:
            self._sample_labels[particle.index] = {
                "name": result["name"],
                "color": tuple(result["color"]),
            }

    def set_classifications(self, classifications: list, meta: dict | None = None) -> None:
        self._classifications = classifications
        self._classification_meta = meta
        self._overlay_mode = "classify"
        # Populate results table with class counts
        counts: dict[str, int] = {}
        for c in classifications:
            counts[c.class_name] = counts.get(c.class_name, 0) + 1
        self._results_table.setColumnCount(2)
        self._results_table.setHorizontalHeaderLabels(["class", "count"])
        rows = sorted(counts.items())
        self._results_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self._results_table.setItem(i, 0, QTableWidgetItem(k))
            self._results_table.setItem(i, 1, QTableWidgetItem(str(v)))
        self._redraw()

    def get_classifications(self) -> list:
        return list(self._classifications)

    def set_particles(self, particles, *, params_signature=None, params_meta=None):
        self._particles    = particles
        self._params_signature = params_signature
        self._params_meta = params_meta
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

    # ── Zoom helpers ─────────────────────────────────────────────────────────

    def _on_scroll(self, event):
        """Zoom the matplotlib view with the mouse wheel."""
        if event.inaxes is not self._ax or self._arr is None:
            return
        factor = 1.25 if event.step > 0 else 0.8
        xl, xr = self._ax.get_xlim()
        yb, yt = self._ax.get_ylim()
        xc = float(event.xdata) if event.xdata is not None else (xl + xr) / 2
        yc = float(event.ydata) if event.ydata is not None else (yb + yt) / 2
        xhalf = (xr - xl) / 2 / factor
        yhalf = (yt - yb) / 2 / factor
        self._ax.set_xlim(xc - xhalf, xc + xhalf)
        self._ax.set_ylim(yc - yhalf, yc + yhalf)
        self._canvas.draw_idle()

    def _zoom_view(self, factor: float):
        """Zoom by *factor* around the image centre (>1 = in, <1 = out)."""
        if self._arr is None:
            return
        xl, xr = self._ax.get_xlim()
        yb, yt = self._ax.get_ylim()
        xc = (xl + xr) / 2
        yc = (yb + yt) / 2
        xhalf = (xr - xl) / 2 / factor
        yhalf = (yt - yb) / 2 / factor
        self._ax.set_xlim(xc - xhalf, xc + xhalf)
        self._ax.set_ylim(yc - yhalf, yc + yhalf)
        self._canvas.draw_idle()

    def reset_view(self):
        """Reset zoom to show the full image (back to thumbnail view)."""
        if self._full_xlim is not None:
            self._ax.set_xlim(self._full_xlim)
            self._ax.set_ylim(self._full_ylim)
            self._canvas.draw_idle()

    # ── Classify-label undo ──────────────────────────────────────────────────

    def undo_last_label(self):
        """Undo the most recent sample-label assignment."""
        if self._label_history:
            self._sample_labels = self._label_history.pop()
            self._redraw()

    def _on_press(self, event):
        if event.inaxes is not self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # ── Template-crop mode ───────────────────────────────────────────────
        if self._cropping:
            self._crop_start = (int(event.xdata), int(event.ydata))
            self._crop_rect  = (self._crop_start[0], self._crop_start[1],
                                self._crop_start[0], self._crop_start[1])
            return

        # ── Classify label-by-clicking mode ─────────────────────────────────
        if (self._sample_armed and self._current_mode == "classify"
                and self._particles and self._arr is not None):
            click_x, click_y = event.xdata, event.ydata
            best_p, best_dist_sq = None, float("inf")
            for p in self._particles:
                cx = p.centroid_x_m / self._pixel_size_x_m
                cy = p.centroid_y_m / self._pixel_size_y_m
                dist_sq = (cx - click_x) ** 2 + (cy - click_y) ** 2
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_p = p
            # Accept click only if within 15 % of the image diagonal
            max_dist_sq = (max(self._arr.shape) * 0.15) ** 2
            if best_p is not None and best_dist_sq < max_dist_sq:
                import copy
                self._label_history.append(copy.deepcopy(self._sample_labels))
                self._edit_sample_label(best_p)
                self._redraw()

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
        # ── Preserve current zoom across redraws ─────────────────────────────
        save_zoom = (
            self._arr is not None
            and self._full_xlim is not None
            and not self._reset_zoom_pending
        )
        if save_zoom:
            cur_xlim = list(self._ax.get_xlim())
            cur_ylim = list(self._ax.get_ylim())

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
            if self._current_mode == "classify":
                # Show particles colored by their label assignment
                for p in self._particles:
                    label_info = self._sample_labels.get(p.index)
                    color = (
                        "#{:02x}{:02x}{:02x}".format(*label_info["color"])
                        if label_info else "#f38ba8"
                    )
                    xs = [c[0] / self._pixel_size_x_m for c in p.contour_xy_m]
                    ys = [c[1] / self._pixel_size_y_m for c in p.contour_xy_m]
                    if xs and ys:
                        xs.append(xs[0]); ys.append(ys[0])
                        self._ax.plot(xs, ys, color=color, lw=0.8)
                    cx = p.centroid_x_m / self._pixel_size_x_m
                    cy = p.centroid_y_m / self._pixel_size_y_m
                    center_color = "#a6e3a1" if label_info else "#585b70"
                    self._ax.plot(cx, cy, marker="+", color=center_color, ms=5)
                    if label_info:
                        self._ax.text(cx + 2, cy - 2, label_info["name"],
                                     color=color, fontsize=7, clip_on=True)
            else:
                for p in self._particles:
                    xs = [c[0] / self._pixel_size_x_m for c in p.contour_xy_m]
                    ys = [c[1] / self._pixel_size_y_m for c in p.contour_xy_m]
                    if xs and ys:
                        xs.append(xs[0]); ys.append(ys[0])
                        self._ax.plot(xs, ys, color="#f38ba8", lw=0.8)
                    cx = p.centroid_x_m / self._pixel_size_x_m
                    cy = p.centroid_y_m / self._pixel_size_y_m
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
        elif self._overlay_mode == "classify" and self._classifications:
            # Show classification results: particles colored by assigned class
            classify_map = {c.particle_index: c for c in self._classifications}
            label_colors = {
                v["name"]: "#{:02x}{:02x}{:02x}".format(*v["color"])
                for v in self._sample_labels.values()
            }
            for p in self._particles:
                cx = p.centroid_x_m / self._pixel_size_x_m
                cy = p.centroid_y_m / self._pixel_size_y_m
                c = classify_map.get(p.index)
                if c and c.class_name != "other":
                    color = label_colors.get(c.class_name, "#89b4fa")
                    self._ax.plot(cx, cy, marker="o", color=color, ms=7,
                                  mfc=color, mew=0, alpha=0.85)
                    self._ax.text(cx + 2, cy - 2, c.class_name,
                                 color=color, fontsize=7, clip_on=True)
                else:
                    self._ax.plot(cx, cy, marker=".", color="#585b70",
                                  ms=4, alpha=0.5)

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

        # ── Zoom state management ─────────────────────────────────────────────
        if self._full_xlim is None:
            # First draw of this image: capture full-view limits
            self._full_xlim = list(self._ax.get_xlim())
            self._full_ylim = list(self._ax.get_ylim())
        elif self._reset_zoom_pending:
            self._ax.set_xlim(self._full_xlim)
            self._ax.set_ylim(self._full_ylim)
            self._reset_zoom_pending = False
        elif save_zoom:
            self._ax.set_xlim(cur_xlim)
            self._ax.set_ylim(cur_ylim)

        self._canvas.draw_idle()


class FeaturesSidebar(QWidget):
    """Right sidebar for Features-tab analysis parameters."""

    mode_changed                    = Signal(str)   # "particles" / "template" / "lattice" / "classify"
    classify_params_changed         = Signal()
    segment_for_classify_requested  = Signal()
    undo_label_requested            = Signal()
    load_from_browse_requested      = Signal()
    run_requested                   = Signal(str)   # mode
    export_requested                = Signal(str)   # mode
    crop_template_requested         = Signal()

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
                           ("lattice",   "Lattice"),
                           ("classify",  "Classify")]:
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
        self._mode_stack.addWidget(self._build_classify_tab())
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

    def _build_classify_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 4, 0, 4)
        l.setSpacing(4)

        info = QLabel(
            "① Set threshold below and click\n"
            "    'Segment particles'.\n"
            "② Click any particle on the image\n"
            "    to assign it a class label.\n"
            "③ Press Run to classify all.")
        info.setFont(QFont("Helvetica", 9))
        info.setWordWrap(True)
        l.addWidget(info)

        l.addWidget(_sep())

        l.addWidget(QLabel("Threshold"))
        self._cls_thr_cb = QComboBox()
        self._cls_thr_cb.addItems(["otsu", "manual", "adaptive"])
        l.addWidget(self._cls_thr_cb)

        self._cls_invert_cb = QCheckBox("Invert (segment dark features)")
        l.addWidget(self._cls_invert_cb)

        row = QHBoxLayout()
        row.addWidget(QLabel("min area (nm²):"))
        self._cls_min_area_spin = QDoubleSpinBox()
        self._cls_min_area_spin.setRange(0.0, 1e6)
        self._cls_min_area_spin.setDecimals(2)
        self._cls_min_area_spin.setValue(0.5)
        self._cls_min_area_spin.valueChanged.connect(lambda _: self.classify_params_changed.emit())
        row.addWidget(self._cls_min_area_spin)
        l.addLayout(row)

        seg_btn = QPushButton("① Segment particles")
        seg_btn.setFont(QFont("Helvetica", 9, QFont.Bold))
        seg_btn.setFixedHeight(28)
        seg_btn.setCursor(QCursor(Qt.PointingHandCursor))
        seg_btn.clicked.connect(self.segment_for_classify_requested.emit)
        l.addWidget(seg_btn)

        undo_btn = QPushButton("↩ Undo last label")
        undo_btn.setFont(QFont("Helvetica", 9))
        undo_btn.setFixedHeight(26)
        undo_btn.setCursor(QCursor(Qt.PointingHandCursor))
        undo_btn.clicked.connect(self.undo_label_requested.emit)
        l.addWidget(undo_btn)

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
        idx = {"particles": 0, "template": 1, "lattice": 2, "classify": 3}[key]
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

    def classify_segmentation_params(self) -> dict:
        return {
            "threshold":    self._cls_thr_cb.currentText(),
            "invert":       self._cls_invert_cb.isChecked(),
            "min_area_nm2": self._cls_min_area_spin.value(),
        }

    def set_status(self, text: str):
        self._status_lbl.setText(text)
