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

from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, Signal, Slot
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QFont, QImage, QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
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


# ── Auto-color palette for classify results ───────────────────────────────────
# Catppuccin Mocha-inspired: perceptually distinct, SPM-dark-background-friendly.
_CLASSIFY_PALETTE: list[str] = [
    "#f38ba8",  # pink  (class 0)
    "#89b4fa",  # blue  (class 1)
    "#a6e3a1",  # green (class 2)
    "#fab387",  # orange(class 3)
    "#cba6f7",  # purple(class 4)
    "#f9e2af",  # yellow(class 5)
    "#89dceb",  # cyan  (class 6)
    "#94e2d5",  # teal  (class 7)
    "#eba0ac",  # mauve (class 8)
    "#b4befe",  # lavender (class 9)
]
_CLASSIFY_OTHER_COLOR = "#6c7086"   # muted gray for "other" particles


def _auto_class_colors(class_names: list) -> dict:
    """Return {class_name: hex_color} with stable, distinct colors.

    "other" always gets the muted gray.  Remaining names are assigned in
    sorted order so the mapping is deterministic across reruns.
    """
    colors: dict = {}
    palette_idx = 0
    for name in sorted(class_names):
        if name == "other":
            colors[name] = _CLASSIFY_OTHER_COLOR
        else:
            colors[name] = _CLASSIFY_PALETTE[palette_idx % len(_CLASSIFY_PALETTE)]
            palette_idx += 1
    return colors


def _arr_to_pixmap(arr: np.ndarray) -> QPixmap:
    """Convert a 2-D float array to a grayscale QPixmap (no PIL dependency)."""
    try:
        vmin, vmax = _clip_range_from_array(arr, 1.0, 99.0)
    except ValueError:
        vmin, vmax = float(arr.min()), float(arr.max())
    rng = vmax - vmin
    if rng == 0:
        u8 = np.zeros(arr.shape, dtype=np.uint8)
    else:
        u8 = np.clip((arr - vmin) / rng * 255.0, 0, 255).astype(np.uint8)
    u8 = np.ascontiguousarray(u8)
    h, w = u8.shape
    data = u8.tobytes()
    qimg = QImage(data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg)


PLANE_NAMES = ["Z fwd", "Z bwd", "I fwd", "I bwd"]


def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedHeight(1)
    return line


# ─────────────────────────────────────────────────────────────────────────────
# QGraphicsView-based image canvas for the Features panel
# Replaces the previous matplotlib canvas — gives the same native Qt
# scroll-wheel zoom / click-drag pan as the thumbnail double-click viewer.
# ─────────────────────────────────────────────────────────────────────────────

class _FeatureView(QGraphicsView):
    """Lightweight QGraphicsView for Features-panel image display.

    Modes
    -----
    normal
        Scroll-wheel zooms, left-drag pans (ScrollHandDrag).
    classify
        Left-click emits ``particle_clicked(scene_x, scene_y)``; drag pans.
    crop
        Left-drag draws a rectangle; release emits
        ``crop_completed(x0, y0, x1, y1)`` in image-pixel coordinates.
    """

    particle_clicked = Signal(float, float)        # scene (image-pixel) coords
    crop_completed   = Signal(int, int, int, int)  # x0, y0, x1, y1 in image px

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtGui import QPainter
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._overlay_items: list = []

        self._classify_armed = False
        self._cropping       = False
        self._crop_start     = None
        self._crop_rect_item = None

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMinimumSize(300, 300)

    # ── Image display ─────────────────────────────────────────────────────────

    def set_pixmap(self, pixmap: QPixmap, reset_view: bool = False) -> None:
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        if reset_view:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def fit_view(self) -> None:
        """Reset zoom to show the full image."""
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    # ── Overlay items ─────────────────────────────────────────────────────────

    def clear_overlay(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    def _add(self, item) -> None:
        item.setZValue(10)
        self._scene.addItem(item)
        self._overlay_items.append(item)

    def add_path(self, xs: list, ys: list, color: str, lw: float = 0.8) -> None:
        if len(xs) < 2:
            return
        path = QPainterPath()
        path.moveTo(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(x, y)
        item = QGraphicsPathItem(path)
        item.setPen(QPen(QColor(color), lw))
        item.setBrush(QBrush(Qt.NoBrush))
        self._add(item)

    def add_cross(self, cx: float, cy: float, color: str, size: float = 5.0) -> None:
        h = size / 2
        for x1, y1, x2, y2 in [(cx - h, cy, cx + h, cy), (cx, cy - h, cx, cy + h)]:
            item = QGraphicsLineItem(x1, y1, x2, y2)
            item.setPen(QPen(QColor(color), 0.8))
            self._add(item)

    def add_dot(self, cx: float, cy: float, color: str, r: float = 4.0) -> None:
        item = QGraphicsEllipseItem(cx - r, cy - r, r * 2, r * 2)
        item.setPen(QPen(Qt.NoPen))
        item.setBrush(QBrush(QColor(color)))
        self._add(item)

    def add_circle(self, cx: float, cy: float, r: float, color: str,
                   lw: float = 1.2) -> None:
        item = QGraphicsEllipseItem(cx - r, cy - r, r * 2, r * 2)
        item.setPen(QPen(QColor(color), lw))
        item.setBrush(QBrush(Qt.NoBrush))
        self._add(item)

    def add_text(self, x: float, y: float, text: str, color: str,
                 font_size: int = 7) -> None:
        item = QGraphicsTextItem(text)
        item.setPos(x, y)
        item.setDefaultTextColor(QColor(color))
        item.setFont(QFont("Helvetica", font_size))
        self._add(item)

    def add_line(self, x1: float, y1: float, x2: float, y2: float,
                 color: str, lw: float = 1.8) -> None:
        item = QGraphicsLineItem(x1, y1, x2, y2)
        item.setPen(QPen(QColor(color), lw))
        self._add(item)

    # ── Mode switches ─────────────────────────────────────────────────────────

    def set_classify_armed(self, armed: bool) -> None:
        self._classify_armed = armed
        self.setCursor(Qt.CrossCursor if armed else Qt.ArrowCursor)

    def set_cropping(self, cropping: bool) -> None:
        self._cropping   = cropping
        self._crop_start = None
        if self._crop_rect_item:
            self._scene.removeItem(self._crop_rect_item)
            self._crop_rect_item = None
        if cropping:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.ArrowCursor)

    # ── Events ────────────────────────────────────────────────────────────────

    def wheelEvent(self, event) -> None:
        """Scroll-wheel zoom — no Ctrl needed (same feel as the thumbnail viewer)."""
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else 1 / 1.12
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            if self._cropping:
                self._crop_start = (pos.x(), pos.y())
                if self._crop_rect_item:
                    self._scene.removeItem(self._crop_rect_item)
                self._crop_rect_item = QGraphicsRectItem(QRectF(pos, QPointF(pos.x() + 1, pos.y() + 1)))
                self._crop_rect_item.setPen(QPen(QColor("#f9e2af"), 1.2, Qt.DashLine))
                self._crop_rect_item.setBrush(QBrush(Qt.NoBrush))
                self._crop_rect_item.setZValue(20)
                self._scene.addItem(self._crop_rect_item)
                return
            if self._classify_armed:
                self.particle_clicked.emit(pos.x(), pos.y())
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._cropping and self._crop_start and (event.buttons() & Qt.LeftButton):
            pos = self.mapToScene(event.pos())
            x0, y0 = self._crop_start
            rect = QRectF(
                QPointF(min(x0, pos.x()), min(y0, pos.y())),
                QPointF(max(x0, pos.x()), max(y0, pos.y())),
            )
            if self._crop_rect_item:
                self._crop_rect_item.setRect(rect)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._cropping and self._crop_start and event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            x0, y0 = self._crop_start
            x1, y1 = pos.x(), pos.y()
            ix0, iy0 = int(min(x0, x1)), int(min(y0, y1))
            ix1, iy1 = int(max(x0, x1)), int(max(y0, y1))
            self._crop_start = None
            self.crop_completed.emit(ix0, iy0, ix1, iy1)
            return
        super().mouseReleaseEvent(event)


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
                res = classify_particles(
                    self._arr, particles, samples,
                    use_sharpness=self._params.get("use_sharpness", False),
                )
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

    analysis_requested    = Signal(str)   # mode name
    template_crop_requested = Signal()
    go_to_browse_requested  = Signal()    # ← Browse button

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
        self._show_overlay: bool = True    # toggle: show/hide classify overlay
        self._class_colors: dict = {}      # class_name → hex color string
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 4)
        lay.setSpacing(6)

        # ── Top bar: title + Browse back-button ─────────────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(6)
        self._title = QLabel("FeatureCounting - load a scan from the Browse tab, then run an analysis.")
        self._title.setFont(QFont("Helvetica", 11, QFont.Bold))
        self._title.setWordWrap(True)
        top_row.addWidget(self._title, 1)

        _back_btn = QPushButton("← Browse")
        _back_btn.setFont(QFont("Helvetica", 9))
        _back_btn.setFixedHeight(28)
        _back_btn.setToolTip("Go back to the thumbnail browser")
        _back_btn.setCursor(QCursor(Qt.PointingHandCursor))
        _back_btn.clicked.connect(self.go_to_browse_requested.emit)
        top_row.addWidget(_back_btn)
        lay.addLayout(top_row)

        # ── Image view (QGraphicsView — same zoom engine as thumbnail viewer) ─
        self._view = _FeatureView(self)
        self._view.particle_clicked.connect(self._on_particle_clicked)
        self._view.crop_completed.connect(self._on_crop_completed)
        lay.addWidget(self._view, 1)

        # ── View toolbar ────────────────────────────────────────────────────
        _vt = QHBoxLayout()
        _vt.setSpacing(4)
        _fit_btn = QPushButton("⟲ Fit")
        _fit_btn.setFixedHeight(24)
        _fit_btn.setToolTip("Reset zoom to show the full image")
        _fit_btn.setFont(QFont("Helvetica", 9))
        _fit_btn.clicked.connect(self.reset_view)
        _vt.addWidget(_fit_btn)

        self._overlay_toggle_btn = QPushButton("👁 Original")
        self._overlay_toggle_btn.setFixedHeight(24)
        self._overlay_toggle_btn.setFont(QFont("Helvetica", 9))
        self._overlay_toggle_btn.setToolTip(
            "Toggle between the original image and the classified overlay.\n"
            "Use this to judge how accurate the segmentation is.")
        self._overlay_toggle_btn.setVisible(False)   # hidden until classify runs
        self._overlay_toggle_btn.clicked.connect(self._toggle_overlay)
        _vt.addWidget(self._overlay_toggle_btn)

        _vt.addStretch(1)
        _hint = QLabel("Scroll to zoom · Drag to pan")
        _hint.setFont(QFont("Helvetica", 8))
        _hint.setStyleSheet("color: #888;")
        _vt.addWidget(_hint)
        lay.addLayout(_vt)

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
        self._class_colors  = {}
        self._show_overlay  = True
        self._overlay_toggle_btn.setVisible(False)
        self._overlay_toggle_btn.setText("👁 Original")
        self._redraw(reset_view=True)
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
        self._view.set_classify_armed(armed)

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
        self._show_overlay = True   # always start with overlay visible after a new run

        # ── Auto-assign a distinct color to every class ───────────────────────
        all_classes = list({c.class_name for c in classifications})
        self._class_colors = _auto_class_colors(all_classes)

        # Reveal the compare toggle button now that classify results exist
        self._overlay_toggle_btn.setVisible(True)
        self._overlay_toggle_btn.setText("👁 Original")

        # ── Per-class orientation statistics ──────────────────────────────────
        import math
        class_angles: dict[str, list] = {}
        for c in classifications:
            class_angles.setdefault(c.class_name, []).append(
                getattr(c, "particle_orientation_deg", 0.0)
            )

        total = len(classifications)
        rows = sorted(class_angles.items())

        self._results_table.setColumnCount(5)
        self._results_table.setHorizontalHeaderLabels(
            ["class", "N", "%", "angle (°)", "± std (°)"])
        self._results_table.setRowCount(len(rows))

        for i, (cls_name, angles) in enumerate(rows):
            n = len(angles)
            pct = 100.0 * n / total if total > 0 else 0.0

            # Circular stats on headless orientations
            valid = [a for a in angles if not math.isnan(a)]
            if valid:
                a2 = np.radians(np.array(valid) * 2.0)
                sin_m = float(np.sin(a2).mean())
                cos_m = float(np.cos(a2).mean())
                mean_ang = float(np.degrees(np.arctan2(sin_m, cos_m))) / 2.0
                if mean_ang < 0.0:
                    mean_ang += 180.0
                R = math.sqrt(sin_m ** 2 + cos_m ** 2)
                std_ang = float(np.degrees(math.sqrt(max(0.0, -2.0 * math.log(R + 1e-12))))) / 2.0
                ang_str = f"{mean_ang:.1f}"
                std_str = f"{std_ang:.1f}"
            else:
                ang_str = "—"
                std_str = "—"

            # Class name cell — colored text matching the overlay color
            hex_color = self._class_colors.get(cls_name, _CLASSIFY_OTHER_COLOR)
            name_item = QTableWidgetItem(f"● {cls_name}")
            name_item.setForeground(QBrush(QColor(hex_color)))
            self._results_table.setItem(i, 0, name_item)
            self._results_table.setItem(i, 1, QTableWidgetItem(str(n)))
            self._results_table.setItem(i, 2, QTableWidgetItem(f"{pct:.1f}"))
            self._results_table.setItem(i, 3, QTableWidgetItem(ang_str))
            self._results_table.setItem(i, 4, QTableWidgetItem(std_str))

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
        self._view.set_cropping(True)
        self._title.setText("Template crop — drag a rectangle over one motif, release to set.")

    def cancel_template_crop(self):
        self._view.set_cropping(False)
        self._redraw()

    def reset_view(self) -> None:
        """Reset zoom to fit the full image (⟲ Fit button)."""
        self._view.fit_view()

    def undo_last_label(self):
        """Undo the most recent sample-label assignment."""
        if self._label_history:
            self._sample_labels = self._label_history.pop()
            self._redraw()

    def _toggle_overlay(self) -> None:
        """Switch between original image and classified overlay (compare button)."""
        self._show_overlay = not self._show_overlay
        if self._show_overlay:
            self._overlay_toggle_btn.setText("👁 Original")
            self._overlay_toggle_btn.setToolTip(
                "Click to hide the overlay and see the original scan.\n"
                "Compare to judge segmentation accuracy.")
        else:
            self._overlay_toggle_btn.setText("✦ Overlay")
            self._overlay_toggle_btn.setToolTip(
                "Click to show the classified overlay again.")
        self._redraw()

    # ── Events from _FeatureView ──────────────────────────────────────────────

    def _on_particle_clicked(self, scene_x: float, scene_y: float) -> None:
        """Handle a classify-mode click — find the nearest particle and label it."""
        if not (self._sample_armed and self._current_mode == "classify"
                and self._particles and self._arr is not None):
            return
        best_p, best_dist_sq = None, float("inf")
        for p in self._particles:
            cx = p.centroid_x_m / self._pixel_size_x_m
            cy = p.centroid_y_m / self._pixel_size_y_m
            dist_sq = (cx - scene_x) ** 2 + (cy - scene_y) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_p = p
        max_dist_sq = (max(self._arr.shape) * 0.15) ** 2
        if best_p is not None and best_dist_sq < max_dist_sq:
            import copy
            self._label_history.append(copy.deepcopy(self._sample_labels))
            self._edit_sample_label(best_p)
            self._redraw()

    def _on_crop_completed(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Handle template crop rectangle from _FeatureView."""
        self._view.set_cropping(False)
        if self._arr is None or x1 - x0 < 4 or y1 - y0 < 4:
            self._title.setText("Template crop cancelled — rectangle too small.")
            return
        Ny, Nx = self._arr.shape
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(Nx, x1), min(Ny, y1)
        self._template_arr = self._arr[y0c:y1c, x0c:x1c].copy()
        th, tw = self._template_arr.shape
        self._title.setText(
            f"Template captured — {tw}×{th} px.  Press 'Run' to count matches.")
        self._redraw()

    def _redraw(self, *, reset_view: bool = False) -> None:
        """Rebuild the QGraphicsView display from scratch."""
        self._view.clear_overlay()

        if self._arr is None:
            return

        # ── Background pixmap ────────────────────────────────────────────────
        pixmap = _arr_to_pixmap(self._arr)
        self._view.set_pixmap(pixmap, reset_view=reset_view)

        # ── Particle overlays ────────────────────────────────────────────────
        if self._overlay_mode == "particles":
            if self._current_mode == "classify":
                # Labeling step: show contour colored by the label already assigned.
                # Colors come from the auto-palette keyed by class name.
                label_name_colors: dict[str, str] = {}   # name → hex color cache
                _pidx = 0
                for p in self._particles:
                    label_info = self._sample_labels.get(p.index)
                    if label_info:
                        lname = label_info["name"]
                        if lname not in label_name_colors:
                            label_name_colors[lname] = _CLASSIFY_PALETTE[
                                len(label_name_colors) % len(_CLASSIFY_PALETTE)]
                        color = label_name_colors[lname]
                    else:
                        color = "#585b70"   # unlabeled: muted gray
                    xs = [c[0] / self._pixel_size_x_m for c in p.contour_xy_m]
                    ys = [c[1] / self._pixel_size_y_m for c in p.contour_xy_m]
                    if xs:
                        xs.append(xs[0]); ys.append(ys[0])
                        self._view.add_path(xs, ys, color, lw=1.5 if label_info else 0.7)
                    cx = p.centroid_x_m / self._pixel_size_x_m
                    cy = p.centroid_y_m / self._pixel_size_y_m
                    if label_info:
                        self._view.add_text(cx + 2, cy - 8, label_info["name"], color)
            else:
                for p in self._particles:
                    xs = [c[0] / self._pixel_size_x_m for c in p.contour_xy_m]
                    ys = [c[1] / self._pixel_size_y_m for c in p.contour_xy_m]
                    if xs:
                        xs.append(xs[0]); ys.append(ys[0])
                        self._view.add_path(xs, ys, "#f38ba8")
                    cx = p.centroid_x_m / self._pixel_size_x_m
                    cy = p.centroid_y_m / self._pixel_size_y_m
                    self._view.add_cross(cx, cy, "#a6e3a1")

        elif self._overlay_mode == "template":
            for d in self._detections:
                self._view.add_circle(d.x_px, d.y_px, 5.0, "#89b4fa")

        elif self._overlay_mode == "lattice" and self._lattice is not None:
            lat = self._lattice
            Ny, Nx = self._arr.shape
            cx, cy = Nx / 2.0, Ny / 2.0
            ax_, ay_ = lat.a_vector_px
            bx_, by_ = lat.b_vector_px
            self._view.add_line(cx, cy, cx + ax_, cy + ay_, "#f38ba8")
            self._view.add_line(cx, cy, cx + bx_, cy + by_, "#89b4fa")
            # Unit cell outline
            pts_x = [cx, cx + ax_, cx + ax_ + bx_, cx + bx_, cx]
            pts_y = [cy, cy + ay_, cy + ay_ + by_, cy + by_, cy]
            self._view.add_path(pts_x, pts_y, "#fab387", lw=1.0)

        elif self._overlay_mode == "classify" and self._classifications and self._show_overlay:
            # ── Classify overlay: colored contour borders, no text/dots ────────
            # Colors come from the auto-assigned palette stored in _class_colors.
            # "other" particles get a thin muted border so they're visible but
            # don't compete visually with the classified ones.
            classify_map = {c.particle_index: c for c in self._classifications}
            for p in self._particles:
                c = classify_map.get(p.index)
                if c is None:
                    continue
                color = self._class_colors.get(c.class_name, _CLASSIFY_OTHER_COLOR)
                is_other = c.class_name == "other"
                lw = 0.6 if is_other else 2.0

                # Contour border in class color
                xs = [pt[0] / self._pixel_size_x_m for pt in p.contour_xy_m]
                ys = [pt[1] / self._pixel_size_y_m for pt in p.contour_xy_m]
                if xs:
                    xs.append(xs[0])
                    ys.append(ys[0])
                    self._view.add_path(xs, ys, color, lw=lw)

                # For classified (non-other) particles: draw an orientation tick
                # as a short line through the centroid — same color, same thickness.
                if not is_other:
                    cx = p.centroid_x_m / self._pixel_size_x_m
                    cy = p.centroid_y_m / self._pixel_size_y_m
                    bw = p.bbox_px[2] - p.bbox_px[0]
                    bh = p.bbox_px[3] - p.bbox_px[1]
                    half = max(5.0, 0.4 * float(np.sqrt(bw ** 2 + bh ** 2)))
                    orient_rad = np.radians(getattr(c, "particle_orientation_deg", 0.0))
                    dx_ = half * np.cos(orient_rad)
                    dy_ = half * np.sin(orient_rad)
                    self._view.add_line(cx - dx_, cy - dy_, cx + dx_, cy + dy_,
                                        color, lw=1.2)


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

        l.addWidget(_sep())

        self._sharpness_cb = QCheckBox("Sharpness-sensitive")
        self._sharpness_cb.setFont(QFont("Helvetica", 9))
        self._sharpness_cb.setToolTip(
            "Enable when two molecule types look the same in shape but one is\n"
            "fuzzy/blurred and the other is sharp. Adds the Laplacian variance\n"
            "of each particle as an extra classification feature.")
        l.addWidget(self._sharpness_cb)

        sharp_hint = QLabel("Use when one class is fuzzy,\nthe other is sharp.")
        sharp_hint.setFont(QFont("Helvetica", 8))
        sharp_hint.setStyleSheet("color: #888;")
        sharp_hint.setWordWrap(True)
        l.addWidget(sharp_hint)

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

    def classify_run_params(self) -> dict:
        """Extra parameters forwarded to classify_particles() at run time."""
        return {
            "use_sharpness": self._sharpness_cb.isChecked(),
        }

    def set_status(self, text: str):
        self._status_lbl.setText(text)
