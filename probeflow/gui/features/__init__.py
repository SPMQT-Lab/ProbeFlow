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

from probeflow.gui.typography import ui_font
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QTimer, Signal, Slot
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QFont, QImage, QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
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
    QMenu,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


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

# ── Histogram theme (Catppuccin Mocha — matches image viewer) ────────────────
_FC_THEME: dict = {
    "bg":        "#1e1e2e",
    "fg":        "#cdd6f4",
    "accent_bg": "#89b4fa",
    "sep":       "#45475a",
}


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


def _arr_to_pixmap(arr: np.ndarray, *, vmin=None, vmax=None) -> QPixmap:
    """Convert a 2-D float array to a grayscale QPixmap (no PIL dependency)."""
    from probeflow.processing.display import array_to_uint8
    u8 = array_to_uint8(arr, vmin=vmin, vmax=vmax, clip_percentiles=(1.0, 99.0))
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


# Shared tooltip wrapper (promoted to probeflow.gui._tooltips so the FFT viewer
# and other modules can reuse the same multi-line-near-the-cursor behaviour).
from probeflow.gui._tooltips import tip as _tip


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
    mask_paint
        Left-click / drag paints exclusion-mask brush strokes, emitting
        ``mask_painted(scene_x, scene_y)`` for each sampled point.
    """

    particle_clicked       = Signal(float, float)        # scene (image-pixel) coords
    particle_right_clicked = Signal(float, float)        # right-click — context menu
    crop_completed         = Signal(int, int, int, int)  # x0, y0, x1, y1 in image px
    mask_painted           = Signal(float, float)        # scene x, scene y (brush centre)
    zero_plane_pick        = Signal(float, float)        # scene x, scene y for zero-plane pts

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtGui import QPainter
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._overlay_items: list = []

        self._classify_armed    = False
        self._cropping          = False
        self._crop_start        = None
        self._crop_rect_item    = None
        self._mask_painting     = False
        self._brush_radius_px   = 10
        self._zero_plane_armed  = False

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
        item.setFont(ui_font(font_size))
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

    def set_mask_painting(self, painting: bool, brush_radius_px: int = 10) -> None:
        """Enter/exit exclusion-mask paint mode."""
        self._mask_painting   = painting
        self._brush_radius_px = brush_radius_px
        if painting:
            self.setDragMode(QGraphicsView.NoDrag)
            # Must set cursor on the viewport — that is the widget the user
            # actually interacts with; setting it on the view has no effect.
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            if not self._cropping:
                self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.ArrowCursor)

    def set_zero_plane_armed(self, armed: bool) -> None:
        """Enter/exit interactive zero-plane point-picking mode."""
        self._zero_plane_armed = armed
        if armed:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            if not self._cropping and not self._mask_painting:
                self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.ArrowCursor)

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
        if event.button() == Qt.RightButton:
            pos = self.mapToScene(event.pos())
            self.particle_right_clicked.emit(pos.x(), pos.y())
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            if self._zero_plane_armed:
                self.zero_plane_pick.emit(pos.x(), pos.y())
                event.accept()
                return
            if self._mask_painting:
                self.mask_painted.emit(pos.x(), pos.y())
                event.accept()   # explicit accept so Qt delivers mouseMoveEvent
                return
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
        if self._mask_painting and (event.buttons() & Qt.LeftButton):
            pos = self.mapToScene(event.pos())
            self.mask_painted.emit(pos.x(), pos.y())
            return
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

    Review gui-arch #2 (fixed 2026-05-28): ``signals`` is now optional —
    when omitted, each worker owns its own ``_FeaturesWorkerSignals`` and
    exposes it as ``worker.signals``.  Callers should prefer per-worker
    signals so that two concurrent (or rapidly back-to-back) workers can
    never deliver their ``finished`` to the same connected slot — the
    "second click of Run while the first hasn't finished" cross-talk
    that the agent flagged.  Passing a shared ``signals`` instance
    remains supported for backward compatibility but is not recommended.
    """

    def __init__(self, mode: str, arr: np.ndarray, pixel_size_m: float,
                 pixel_size_x_m: float, pixel_size_y_m: float,
                 params: dict,
                 signals: "_FeaturesWorkerSignals | None" = None):
        super().__init__()
        self._mode    = mode
        self._arr     = arr
        self._px      = float(pixel_size_m)
        self._px_x    = float(pixel_size_x_m)
        self._px_y    = float(pixel_size_y_m)
        self._params  = params
        # Lifetime safety: this QRunnable is auto-deleted by QThreadPool on the
        # *worker* thread when run() returns. If it were the sole owner of a
        # parentless QObject (the signals), that QObject — which carries
        # cross-thread signal/slot connections — would be destroyed off the main
        # thread, corrupting Qt's internals (observed as a SIGSEGV in an
        # unrelated app-level event filter). Parent the auto-created signals to a
        # main-thread owner (the QApplication, created on the main thread) so
        # Shiboken never C++-destroys it from the worker thread; run() then
        # deleteLater()s it so there is no per-run accumulation.
        if signals is not None:
            self.signals = signals
            self._owns_signals = False
        else:
            self.signals = _FeaturesWorkerSignals(QApplication.instance())
            self._owns_signals = True
        # Keep _signals for the existing internal .emit() calls below.
        self._signals = self.signals

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
                    exclude_mask=self._params.get("exclude_mask"),
                    max_exclude_overlap=self._params.get("max_exclude_overlap", 0.25),
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
                    threshold_method=self._params.get("threshold_method", "gmm"),
                    manual_threshold=self._params.get("manual_threshold", 0.5),
                    encoder=self._params.get("encoder", "raw"),
                    rotate_augment=self._params.get("rotate_augment", False),
                )
            else:
                raise ValueError(f"Unknown mode {self._mode!r}")
            self._signals.finished.emit(self._mode, res, "")
        except Exception as exc:
            self._signals.finished.emit(self._mode, None, str(exc))
        finally:
            # Reclaim a worker-owned signals object on the main thread (it is
            # parented to the QApplication). deleteLater() is thread-safe and
            # posts the deletion to the object's (main) thread, so it never runs
            # during this worker's off-thread teardown.
            if self._owns_signals:
                self._signals.deleteLater()


class FeaturesPanel(QWidget):
    """Center widget for the Features tab.

    This is a dedicated analysis workspace: load one selected Browse scan,
    inspect overlays/results, and export analysis JSON. It intentionally does
    not mutate Browse thumbnails or Viewer processing state.
    """

    analysis_requested      = Signal(str)    # mode name
    template_crop_requested = Signal()
    go_to_browse_requested  = Signal()       # ← Browse button
    scan_loaded             = Signal(object) # emitted after load_entry with the arr
    zero_plane_applied      = Signal()       # emitted after zero-plane correction

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self._t          = t
        self._entry      = None            # current SxmFile
        self._plane_idx  = 0
        self._arr        = None            # np.ndarray
        self._pixel_size_m = 1e-10
        self._pixel_size_x_m = 1e-10
        self._pixel_size_y_m = 1e-10
        self._scan = None                  # source Scan, when available (for export provenance)
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
        # FUTURE OPPORTUNITY: this is a raw bool ndarray.  Representing it as a
        # ProbeFlow ROI (a freehand/multipolygon with .to_mask()) would let the
        # exclusion region be saved, restored, and shared with the rest of the
        # ROI tooling instead of being a one-off per-session array.
        self._exclusion_mask: np.ndarray | None = None   # bool mask, same shape as _arr
        self._mask_brush_radius: int = 10
        self._mask_color: tuple[int, int, int] = (220, 50, 50)   # R, G, B
        self._mask_overlay_item = None     # QGraphicsPixmapItem — lives outside _overlay_items
        # Algorithmic step-edge exclusion (separate from the painted mask above).
        self._step_mask: np.ndarray | None = None        # computed step band, bool
        self._step_mask_sig = None                       # params signature for caching
        self._step_overlay_item = None                   # amber band overlay item
        # Zero-plane interactive correction ───────────────────────────────────
        self._arr_original: np.ndarray | None = None    # raw array before any zero-plane
        self._zero_plane_pts: list = []                 # picked (x_px, y_px) points (≤3)
        # Display contrast clip (SI — metres).  None → default 1–99 % auto-clip.
        self._display_vmin: float | None = None
        self._display_vmax: float | None = None
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 4)
        lay.setSpacing(6)

        # ── Top bar: title + Browse back-button ─────────────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(6)
        self._title = QLabel("FeatureCounting - load a scan from the Browse tab, then run an analysis.")
        self._title.setFont(ui_font(11, weight=QFont.Bold))
        self._title.setWordWrap(True)
        top_row.addWidget(self._title, 1)

        _back_btn = QPushButton("← Browse")
        _back_btn.setFont(ui_font(9))
        _back_btn.setFixedHeight(28)
        _back_btn.setToolTip("Go back to the thumbnail browser")
        _back_btn.setCursor(QCursor(Qt.PointingHandCursor))
        _back_btn.clicked.connect(self.go_to_browse_requested.emit)
        top_row.addWidget(_back_btn)
        lay.addLayout(top_row)

        # ── Image view (QGraphicsView — same zoom engine as thumbnail viewer) ─
        self._view = _FeatureView(self)
        self._view.particle_clicked.connect(self._on_particle_clicked)
        self._view.particle_right_clicked.connect(self._on_particle_right_clicked)
        self._view.crop_completed.connect(self._on_crop_completed)
        self._view.mask_painted.connect(self._on_mask_painted)
        self._view.zero_plane_pick.connect(self._on_zero_plane_pick)
        lay.addWidget(self._view, 1)

        # ── View toolbar ────────────────────────────────────────────────────
        _vt = QHBoxLayout()
        _vt.setSpacing(4)
        _fit_btn = QPushButton("⟲ Fit")
        _fit_btn.setFixedHeight(24)
        _fit_btn.setToolTip("Reset zoom to show the full image")
        _fit_btn.setFont(ui_font(9))
        _fit_btn.clicked.connect(self.reset_view)
        _vt.addWidget(_fit_btn)

        self._overlay_toggle_btn = QPushButton("👁 Original")
        self._overlay_toggle_btn.setFixedHeight(24)
        self._overlay_toggle_btn.setFont(ui_font(9))
        self._overlay_toggle_btn.setToolTip(
            "Toggle between the original image and the classified overlay.\n"
            "Use this to judge how accurate the segmentation is.")
        self._overlay_toggle_btn.setVisible(False)   # hidden until classify runs
        self._overlay_toggle_btn.clicked.connect(self._toggle_overlay)
        _vt.addWidget(self._overlay_toggle_btn)

        _vt.addStretch(1)
        _hint = QLabel("Scroll to zoom · Drag to pan")
        _hint.setFont(ui_font(8))
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
        self._results_table.setFont(ui_font(9))
        lay.addWidget(self._results_table)


    def load_entry(self, entry, plane_idx: int, arr: np.ndarray,
                    pixel_size_m: float, pixel_size_x_m: float | None = None,
                    pixel_size_y_m: float | None = None, scan=None):
        # ``scan`` is the source :class:`probeflow.core.scan_model.Scan`, when the
        # caller has one.  Held only so exports can record full provenance
        # (scan_range, pixel sizes, plane names/units, processing state) the same
        # way the CLI does; analysis still runs on ``arr``.
        self._entry        = entry
        self._plane_idx    = plane_idx
        self._arr          = arr
        self._arr_original = np.asarray(arr, dtype=np.float64).copy()  # snapshot before any zero-plane
        self._zero_plane_pts = []                                        # reset on every new load
        self._display_vmin = None   # reset contrast clip; histogram will set auto-clip
        self._display_vmax = None
        self._scan         = scan
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
        self.clear_exclusion_mask()   # discard any mask from the previous image
        self.clear_step_mask()        # discard any computed step band too
        self._redraw(reset_view=True)
        self._results_table.setRowCount(0)
        plane_lbl = PLANE_NAMES[plane_idx] if 0 <= plane_idx < len(PLANE_NAMES) else f"plane {plane_idx}"
        self._title.setText(
            f"{entry.stem}  -  {plane_lbl}  -  "
            f"{arr.shape[1]}x{arr.shape[0]} px  "
            f"(px = {self._pixel_size_x_m * 1e12:.1f} x "
            f"{self._pixel_size_y_m * 1e12:.1f} pm)")
        self.scan_loaded.emit(self._arr)   # update sidebar histogram

    def current_entry(self):
        return self._entry

    def current_scan(self):
        """Source Scan for export provenance, or None if loaded without one."""
        return self._scan

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

    # ── Zero-plane interactive correction ─────────────────────────────────────

    def set_zero_plane_armed(self, armed: bool) -> None:
        """Toggle interactive 3-point zero-plane picking on/off."""
        self._view.set_zero_plane_armed(armed)
        if not armed:
            # If the user cancels before picking 3 points, clear the partial list
            # only if no plane has been applied yet (i.e. pts < 3).
            if len(self._zero_plane_pts) < 3:
                self._zero_plane_pts = []
                self._redraw()    # remove partial markers

    def _on_zero_plane_pick(self, scene_x: float, scene_y: float) -> None:
        """Handle a canvas click while zero-plane mode is armed.

        Scene coords map 1:1 to image pixels in the ``_FeatureView``.
        """
        if self._arr is None:
            return
        Ny, Nx = self._arr.shape
        x_px = max(0, min(int(round(scene_x)), Nx - 1))
        y_px = max(0, min(int(round(scene_y)), Ny - 1))
        self._zero_plane_pts.append((x_px, y_px))
        self._redraw()   # show numbered markers as they accumulate

        n = len(self._zero_plane_pts)
        if n < 3:
            return  # still collecting — controller will update the status

        # All 3 points collected: apply the plane correction.
        from probeflow.processing.geometry import set_zero_plane as _set_zero_plane
        try:
            corrected = _set_zero_plane(
                self._arr_original if self._arr_original is not None else self._arr,
                self._zero_plane_pts,
            )
        except ValueError:
            # Degenerate triangle or collinear points — clear and let user retry.
            self._zero_plane_pts = []
            self._redraw()
            self.zero_plane_applied.emit()   # signal controller to show error
            return
        self._arr = corrected
        # Discard segmentation results — they were computed on the old array.
        self._particles   = []
        self._detections  = []
        self._lattice     = None
        self._classifications = []
        self._overlay_mode = "none"
        self._show_overlay = True
        self._overlay_toggle_btn.setVisible(False)
        self._redraw(reset_view=False)
        self._results_table.setRowCount(0)
        self.zero_plane_applied.emit()

    def reset_to_original(self) -> None:
        """Revert to the array as loaded (undo zero-plane correction)."""
        if self._arr_original is None:
            return
        self._arr = np.asarray(self._arr_original, dtype=np.float64).copy()
        self._zero_plane_pts = []
        self._display_vmin = None   # revert contrast clip to auto
        self._display_vmax = None
        self._particles   = []
        self._detections  = []
        self._lattice     = None
        self._classifications = []
        self._overlay_mode = "none"
        self._show_overlay = True
        self._overlay_toggle_btn.setVisible(False)
        self._redraw(reset_view=False)
        self._results_table.setRowCount(0)
        self.scan_loaded.emit(self._arr)   # update histogram to original

    def set_display_range(self, lo_si: float, hi_si: float) -> None:
        """Update the display contrast range and redraw.

        Called by the controller when the histogram drag lines or sliders are
        released.  Values are in SI (metres).  Pass ``None`` for either
        argument to fall back to the default 1–99 % auto-clip.
        """
        self._display_vmin = float(lo_si) if lo_si is not None else None
        self._display_vmax = float(hi_si) if hi_si is not None else None
        if self._arr is not None:
            self._redraw()

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

        # ── Per-class orientation sub-rows ────────────────────────────────────
        # For each class, bin orientations into 15° windows (0–180°).
        # Each non-empty bin becomes its own row:  "T (30°)", "T (45°)", …
        # "other" is never sub-divided — it just gets one summary row.
        import math
        _BIN_DEG = 15           # window width in degrees
        _N_BINS  = 180 // _BIN_DEG   # 12 bins for [0, 180)

        class_angles: dict[str, list] = {}
        for c in classifications:
            class_angles.setdefault(c.class_name, []).append(
                getattr(c, "particle_orientation_deg", 0.0)
            )

        total = len(classifications)

        # Build flat list of (label, n, pct, hex_color, is_summary) rows.
        # Each non-"other" class gets a bold summary row (total) followed by
        # indented sub-rows for each non-empty 15° orientation bin.
        table_rows: list[tuple[str, int, float, str, bool]] = []

        for cls_name in sorted(class_angles.keys()):
            angles = class_angles[cls_name]
            hex_color = self._class_colors.get(cls_name, _CLASSIFY_OTHER_COLOR)
            n_total = len(angles)
            pct_total = 100.0 * n_total / total if total > 0 else 0.0

            # "other" — single summary row only (no orientation breakdown).
            if cls_name == "other":
                table_rows.append(("other", n_total, pct_total, hex_color, True))
                continue

            # Summary row for this class (bold, all orientations combined).
            table_rows.append((cls_name, n_total, pct_total, hex_color, True))

            valid = np.array([a for a in angles if not math.isnan(a)],
                             dtype=np.float64)
            if valid.size == 0:
                continue

            # Assign each angle to a 15° bin
            bin_idx = np.floor(valid / _BIN_DEG).astype(int)
            bin_idx = np.clip(bin_idx, 0, _N_BINS - 1)

            for b in range(_N_BINS):
                mask = bin_idx == b
                if not mask.any():
                    continue
                bin_angles = valid[mask]
                n_bin = int(mask.sum())
                pct = 100.0 * n_bin / total if total > 0 else 0.0
                # Mean angle of the particles in this bin (arithmetic ok for
                # a 15° window — no wrap issue within such a narrow range).
                mean_ang = float(bin_angles.mean())
                label = f"  {cls_name} ({mean_ang:.0f}°)"   # indented sub-row
                table_rows.append((label, n_bin, pct, hex_color, False))

        # Populate the table — 3 columns: class (angle) | N | %
        _bold_font   = ui_font(9, weight=QFont.Bold)
        _normal_font = ui_font(9)
        self._results_table.setColumnCount(3)
        self._results_table.setHorizontalHeaderLabels(["class (angle)", "N", "%"])
        self._results_table.setRowCount(len(table_rows))

        for i, (label, n, pct, hex_color, is_summary) in enumerate(table_rows):
            font = _bold_font if is_summary else _normal_font
            prefix = "●" if is_summary else " "
            name_item = QTableWidgetItem(f"{prefix} {label}")
            name_item.setForeground(QBrush(QColor(hex_color)))
            name_item.setFont(font)
            n_item = QTableWidgetItem(str(n))
            n_item.setFont(font)
            pct_item = QTableWidgetItem(f"{pct:.1f}")
            pct_item.setFont(font)
            self._results_table.setItem(i, 0, name_item)
            self._results_table.setItem(i, 1, n_item)
            self._results_table.setItem(i, 2, pct_item)

        self._redraw()

    def get_classifications(self) -> list:
        return list(self._classifications)

    def set_particles(self, particles, *, params_signature=None, params_meta=None):
        self._particles    = particles
        self._params_signature = params_signature
        self._params_meta = params_meta
        self._overlay_mode = "particles"
        self._show_overlay = True
        self._overlay_toggle_btn.setVisible(bool(particles))
        self._overlay_toggle_btn.setText("👁 Original")
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
        self._show_overlay = True
        self._overlay_toggle_btn.setVisible(bool(detections))
        self._overlay_toggle_btn.setText("👁 Original")
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
        self._show_overlay = True
        self._overlay_toggle_btn.setVisible(lat is not None)
        self._overlay_toggle_btn.setText("👁 Original")
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

    def clear_particles(self) -> None:
        """Remove segmentation overlay — returns the panel to the raw image."""
        self._particles    = []
        self._overlay_mode = "none"
        self._overlay_toggle_btn.setVisible(False)
        self._overlay_toggle_btn.setText("👁 Original")
        self._show_overlay = True
        self._results_table.setRowCount(0)
        self._redraw()

    def clear_classifications(self) -> None:
        """Remove classification overlay — keeps particle contours visible."""
        self._classifications = []
        self._classification_meta = None
        self._overlay_mode = "particles"
        # Keep the toggle button visible — particle contours are still shown.
        self._overlay_toggle_btn.setVisible(bool(self._particles))
        self._overlay_toggle_btn.setText("👁 Original")
        self._show_overlay = True
        self._results_table.setRowCount(0)
        self._redraw()

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
        """Switch between the raw image and the analysis overlay (compare button)."""
        self._show_overlay = not self._show_overlay
        if self._show_overlay:
            self._overlay_toggle_btn.setText("👁 Original")
            self._overlay_toggle_btn.setToolTip(
                "Click to hide the overlay and see the original scan.")
        else:
            self._overlay_toggle_btn.setText("✦ Overlay")
            self._overlay_toggle_btn.setToolTip(
                "Click to show the analysis overlay again.")
        self._redraw()

    # ── Exclusion mask ────────────────────────────────────────────────────────

    def set_mask_color(self, r: int, g: int, b: int) -> None:
        """Change the highlight colour of the exclusion mask overlay."""
        self._mask_color = (int(r), int(g), int(b))
        if self.has_exclusion_mask():
            self._update_mask_overlay()   # redraw with new colour immediately

    def set_mask_painting(self, painting: bool, brush_radius: int = 10) -> None:
        """Enter or exit mask-paint mode on the view."""
        self._mask_brush_radius = brush_radius
        self._view.set_mask_painting(painting, brush_radius)

    def clear_exclusion_mask(self) -> None:
        """Remove the exclusion mask and its overlay from the scene."""
        self._exclusion_mask = None
        if self._mask_overlay_item is not None:
            self._view._scene.removeItem(self._mask_overlay_item)
            self._mask_overlay_item = None

    def has_exclusion_mask(self) -> bool:
        return self._exclusion_mask is not None and bool(self._exclusion_mask.any())

    def get_analysis_array(self) -> np.ndarray | None:
        """Return the image array with excluded pixels set to the local minimum.

        Excluded pixels become dark background in the uint8 view used by
        ``segment_particles``, so the algorithm naturally ignores them.
        """
        if self._arr is None:
            return None
        if not self.has_exclusion_mask():
            return self._arr
        masked = self._arr.copy()
        finite_vals = masked[np.isfinite(masked)]
        fill = float(finite_vals.min()) if finite_vals.size > 0 else 0.0
        masked[self._exclusion_mask] = fill
        return masked

    def _on_mask_painted(self, scene_x: float, scene_y: float) -> None:
        """Paint a circular brush stroke into the exclusion mask."""
        if self._arr is None:
            return
        if self._exclusion_mask is None:
            self._exclusion_mask = np.zeros(self._arr.shape, dtype=bool)

        ix, iy = int(round(scene_x)), int(round(scene_y))
        r = self._mask_brush_radius
        Ny, Nx = self._arr.shape
        y0, y1 = max(0, iy - r), min(Ny, iy + r + 1)
        x0, x1 = max(0, ix - r), min(Nx, ix + r + 1)

        ys, xs = np.ogrid[y0:y1, x0:x1]
        circle = (xs - ix) ** 2 + (ys - iy) ** 2 <= r ** 2
        self._exclusion_mask[y0:y1, x0:x1] |= circle
        self._update_mask_overlay()

    def _update_mask_overlay(self) -> None:
        """Re-render the semi-transparent red exclusion-mask overlay in the scene."""
        if self._exclusion_mask is None or not self._exclusion_mask.any():
            if self._mask_overlay_item is not None:
                self._view._scene.removeItem(self._mask_overlay_item)
                self._mask_overlay_item = None
            return

        h, w = self._exclusion_mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        r, g, b = self._mask_color
        rgba[self._exclusion_mask] = [r, g, b, 160]   # R, G, B, A — user-selected colour

        # Use tobytes() exactly like _arr_to_pixmap does: QImage references the
        # bytes buffer so it must stay alive until QPixmap.fromImage() copies it.
        data = np.ascontiguousarray(rgba).tobytes()
        qimg = QImage(data, w, h, 4 * w, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)   # deep-copies image data into the pixmap

        if self._mask_overlay_item is None:
            self._mask_overlay_item = QGraphicsPixmapItem(pixmap)
            self._mask_overlay_item.setPos(0, 0)
            self._mask_overlay_item.setZValue(5)   # above image, below particle overlays
            self._view._scene.addItem(self._mask_overlay_item)
        else:
            self._mask_overlay_item.setPixmap(pixmap)

    # ── Algorithmic step-edge exclusion ───────────────────────────────────────

    def compute_step_mask(
        self, *, threshold_deg: float, molecule_diameter_m: float,
        dilate_m: float, min_step_height_m: float | None,
        suppress_dark: bool = False,
    ) -> np.ndarray | None:
        """Compute (and cache) the step-edge band from the RAW scan plane.

        Computed on ``self._arr`` — the raw, unmasked array — so the painted
        mask's artificial cliffs never read as fake steps.  Cached against the
        parameter signature so dragging a slider doesn't recompute every tick.
        Returns the boolean band, or None when no scan is loaded.
        """
        if self._arr is None:
            return None
        sig = (threshold_deg, molecule_diameter_m, dilate_m,
               min_step_height_m, suppress_dark, id(self._arr))
        if sig != self._step_mask_sig:
            from probeflow.analysis.step_edges import step_edge_mask
            px_x, px_y = self.current_pixel_sizes()
            self._step_mask = step_edge_mask(
                self._arr,
                pixel_size_x_m=px_x, pixel_size_y_m=px_y,
                molecule_diameter_m=molecule_diameter_m,
                threshold_deg=threshold_deg, dilate_m=dilate_m,
                min_step_height_m=min_step_height_m, suppress_dark=suppress_dark,
            )
            self._step_mask_sig = sig
        self._update_step_overlay()
        return self._step_mask

    def step_mask(self) -> np.ndarray | None:
        """The most recently computed step band (without recomputing)."""
        return self._step_mask

    def clear_step_mask(self) -> None:
        """Drop the computed step band and its overlay."""
        self._step_mask = None
        self._step_mask_sig = None
        if self._step_overlay_item is not None:
            self._view._scene.removeItem(self._step_overlay_item)
            self._step_overlay_item = None

    def _update_step_overlay(self) -> None:
        """Render the step band as a translucent amber overlay (distinct from the
        red painted mask), so the user sees exactly what will be excluded."""
        mask = self._step_mask
        if mask is None or not mask.any():
            if self._step_overlay_item is not None:
                self._view._scene.removeItem(self._step_overlay_item)
                self._step_overlay_item = None
            return
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask] = [250, 179, 135, 150]   # amber, semi-transparent
        data = np.ascontiguousarray(rgba).tobytes()
        qimg = QImage(data, w, h, 4 * w, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        if self._step_overlay_item is None:
            self._step_overlay_item = QGraphicsPixmapItem(pixmap)
            self._step_overlay_item.setPos(0, 0)
            self._step_overlay_item.setZValue(6)   # above painted mask, below particles
            self._view._scene.addItem(self._step_overlay_item)
        else:
            self._step_overlay_item.setPixmap(pixmap)

        self._view.viewport().update()   # force immediate repaint

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

    def _on_particle_right_clicked(self, scene_x: float, scene_y: float) -> None:
        """Right-click on the canvas — context menu for both labeling phases.

        * **Labeling phase** (``_overlay_mode == "particles"`` and
          ``_current_mode == "classify"``): offer to remove the manual sample
          label assigned to the nearest particle (supports Ctrl+Z undo too).
        * **Post-classification phase** (``_overlay_mode == "classify"``):
          offer to unclassify a non-'other' particle (set its class to 'other').
        """
        if not self._particles or self._arr is None:
            return

        # ── Find the nearest particle within 15 % of the image diagonal ─────────
        best_p, best_dist_sq = None, float("inf")
        for p in self._particles:
            cx = p.centroid_x_m / self._pixel_size_x_m
            cy = p.centroid_y_m / self._pixel_size_y_m
            dist_sq = (cx - scene_x) ** 2 + (cy - scene_y) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_p = p

        max_dist_sq = (max(self._arr.shape) * 0.15) ** 2
        if best_p is None or best_dist_sq > max_dist_sq:
            return

        # ── CASE 1: manual labeling phase (before running classification) ────────
        if self._overlay_mode == "particles" and self._current_mode == "classify":
            label_info = self._sample_labels.get(best_p.index)
            if label_info is None:
                return   # particle has no label — nothing to remove
            menu = QMenu(self)
            menu.addAction(
                f"Remove label '{label_info['name']}'  (particle #{best_p.index})")
            chosen = menu.exec(QCursor.pos())
            if chosen is None:
                return
            import copy
            self._label_history.append(copy.deepcopy(self._sample_labels))
            del self._sample_labels[best_p.index]
            self._redraw()
            return

        # ── CASE 2: post-classification phase ────────────────────────────────────
        if self._overlay_mode == "classify" and self._classifications:
            classify_map = {c.particle_index: c for c in self._classifications}
            cls = classify_map.get(best_p.index)
            if cls is None or cls.class_name == "other":
                return   # already 'other' or not classified — nothing to undo
            menu = QMenu(self)
            menu.addAction(
                f"Unclassify #{best_p.index}  (class '{cls.class_name}' → 'other')")
            chosen = menu.exec(QCursor.pos())
            if chosen is None:
                return
            from dataclasses import replace as _dc_replace
            new_cls = [
                _dc_replace(c, class_name="other") if c.particle_index == best_p.index else c
                for c in self._classifications
            ]
            # Re-apply via set_classifications so the table and overlay both update.
            self.set_classifications(new_cls, meta=self._classification_meta)

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
        pixmap = _arr_to_pixmap(self._arr,
                                vmin=self._display_vmin,
                                vmax=self._display_vmax)
        self._view.set_pixmap(pixmap, reset_view=reset_view)

        # ── Zero-plane reference point markers ────────────────────────────────
        _ZP_COLOR = "#f9e2af"   # warm yellow — visible on both bright and dark scans
        for i, (x_px, y_px) in enumerate(self._zero_plane_pts[:3]):
            r = 7.0
            self._view.add_circle(x_px, y_px, r, _ZP_COLOR, lw=1.8)
            self._view.add_text(x_px + r + 2, y_px - r, str(i + 1), _ZP_COLOR, font_size=8)

        # ── Particle overlays ────────────────────────────────────────────────
        if self._overlay_mode == "particles" and self._show_overlay:
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

        elif self._overlay_mode == "template" and self._show_overlay:
            for d in self._detections:
                self._view.add_circle(d.x_px, d.y_px, 5.0, "#89b4fa")

        elif self._overlay_mode == "lattice" and self._lattice is not None and self._show_overlay:
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
    """Right sidebar — two-phase UniMR-style interface.

    Phase 1 (Segmentation)
        Threshold slider (0–255), min/max area sliders (as % of image area),
        invert checkbox, exclusion mask tools, "Apply Settings" button.
        Clicking "Apply Settings" runs segmentation and auto-advances to Phase 2.

    Phase 2 (Analysis)
        Shows particle count, mode tabs (Particles / Template / Lattice /
        Classify), Run, and Export buttons.  "← Settings" returns to Phase 1.
        The Classify tab exposes threshold method (Manual/Otsu/GMM/Distribution),
        encoding (Raw/PCA+KMeans/Auto), and rotation-augmentation options that
        mirror the UniMR classification interface.
    """

    mode_changed               = Signal(str)   # "particles"/"template"/"lattice"/"classify"
    classify_params_changed    = Signal()      # Phase-1 segmentation params changed
    segment_requested              = Signal()  # "Apply Segmentation" — run + stay in Phase 1
    clear_segmentation_requested   = Signal()  # "Remove Segmentation" — clear overlay
    advance_phase2_requested       = Signal()  # "Move to Phase 2" — advance using found particles
    clear_classification_requested = Signal()  # "Remove Classification" — drop labels (Phase 2)
    add_to_bank_requested          = Signal()  # "Add samples to bank" — persist labelled CLIP embeddings
    preview_requested              = Signal()  # debounced live preview (slider drag)
    undo_label_requested       = Signal()
    load_from_browse_requested = Signal()
    run_requested              = Signal(str)   # mode
    export_requested           = Signal(str)   # mode
    send_to_particle_statistics_requested = Signal(str)  # mode
    crop_template_requested    = Signal()
    mask_paint_toggled         = Signal(bool)  # True = start painting
    mask_clear_requested       = Signal()
    mask_color_changed         = Signal(int, int, int)  # R, G, B
    step_exclude_changed       = Signal()      # algorithmic step-edge controls changed
    zero_plane_armed           = Signal(bool)  # True = enter 3-point picking mode
    reset_to_original_requested = Signal()     # undo zero-plane correction
    display_clip_changed       = Signal(float, float)  # lo_si, hi_si — contrast range

    MASK_COLORS: dict[str, tuple[int, int, int]] = {
        "Red":     (220,  50,  50),
        "Blue":    ( 50, 100, 220),
        "Green":   ( 50, 200,  50),
        "Yellow":  (220, 220,  50),
        "Cyan":    ( 50, 200, 220),
        "Magenta": (200,  50, 200),
    }

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self._t = t
        # ── Histogram interaction state (populated by update_histogram) ────────
        self._hist_scale: float = 1.0
        self._hist_auto_lo_si: float | None = None  # 1-pct clip in SI
        self._hist_auto_hi_si: float | None = None  # 99-pct clip in SI
        self._hist_cur_lo_si:  float | None = None  # current drag-line lo in SI
        self._hist_cur_hi_si:  float | None = None  # current drag-line hi in SI
        self._build()

    # ── Top-level layout ──────────────────────────────────────────────────────

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
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Two-phase stacked widget: 0 = Segmentation, 1 = Analysis
        self._phase_stack = QStackedWidget()
        self._phase_stack.addWidget(self._build_phase1())
        self._phase_stack.addWidget(self._build_phase2())
        lay.addWidget(self._phase_stack, 1)

        # Shared status label — always visible below the phase stack
        sw = QWidget()
        sl = QVBoxLayout(sw)
        sl.setContentsMargins(10, 4, 10, 8)
        sl.setSpacing(0)
        self._status_lbl = QLabel("Load a scan to begin.")
        self._status_lbl.setFont(ui_font(9))
        self._status_lbl.setWordWrap(True)
        sl.addWidget(self._status_lbl)
        lay.addWidget(sw)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # Default mode (also initialises the Run button label)
        self._select_mode("particles")

    # ── Phase 1 — Segmentation ────────────────────────────────────────────────

    def _build_phase1(self) -> QWidget:
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setContentsMargins(10, 10, 10, 6)
        lay.setSpacing(6)

        # ── Load & Plane ───────────────────────────────────────────────────────
        load_btn = QPushButton("Load primary scan from Browse")
        load_btn.setFont(ui_font(10))
        load_btn.setFixedHeight(30)
        load_btn.setCursor(QCursor(Qt.PointingHandCursor))
        load_btn.setObjectName("accentBtn")
        load_btn.setToolTip(_tip(
            "Pull the scan currently selected in the Browse tab into this "
            "workspace, with any Viewer processing already applied. Select a "
            "different thumbnail in Browse, then click here to swap it in."))
        load_btn.clicked.connect(self.load_from_browse_requested.emit)
        lay.addWidget(load_btn)

        plane_row = QHBoxLayout()
        plane_row.addWidget(QLabel("Plane:"))
        self._plane_cb = QComboBox()
        self._plane_cb.addItems(PLANE_NAMES)
        self._plane_cb.setCurrentIndex(0)
        self._plane_cb.setToolTip(_tip(
            "Which acquired data channel to analyse: Z = topography (height), "
            "I = tunnelling current; fwd / bwd are the forward and backward "
            "scan directions. Most particle work uses Z fwd."))
        plane_row.addWidget(self._plane_cb, 1)
        lay.addLayout(plane_row)

        # ── Scan Processing ────────────────────────────────────────────────────
        lay.addWidget(_sep())
        proc_title = QLabel("Scan Processing")
        proc_title.setFont(ui_font(10, weight=QFont.Bold))
        lay.addWidget(proc_title)

        proc_hint = QLabel(
            "Apply before segmenting. Zero-plane subtracts sample tilt "
            "by clicking 3 substrate reference points.")
        proc_hint.setFont(ui_font(8))
        proc_hint.setWordWrap(True)
        proc_hint.setStyleSheet("color: #888;")
        lay.addWidget(proc_hint)

        # Interactive height histogram — same HistogramPanel as the image viewer.
        # Brightness/Contrast sliders are hidden (not meaningful for a scan view);
        # Min, Max, Auto, and Reset remain active.
        from probeflow.gui.viewer.histogram import HistogramPanel as _HistogramPanel
        self._histogram = _HistogramPanel(self)
        self._histogram._brightness_w.setVisible(False)
        self._histogram._contrast_w.setVisible(False)
        self._histogram.rangeReleased.connect(self._on_hist_range_released)
        self._histogram.minReleased.connect(self._on_hist_min_released)
        self._histogram.maxReleased.connect(self._on_hist_max_released)
        self._histogram.autoClipRequested.connect(self._on_hist_auto)
        self._histogram.resetRequested.connect(self._on_hist_reset)
        lay.addWidget(self._histogram)

        self._zero_plane_btn = QPushButton("📐 Set Zero Plane")
        self._zero_plane_btn.setCheckable(True)
        self._zero_plane_btn.setFont(ui_font(9))
        self._zero_plane_btn.setFixedHeight(28)
        self._zero_plane_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._zero_plane_btn.setToolTip(_tip(
            "Click 3 reference points on the bare substrate — a tilted plane "
            "through those 3 heights is subtracted from the whole image. "
            "Removes sample tilt and slow scanner drift. The corrected array "
            "is used for all subsequent segmentation and analysis."))
        self._zero_plane_btn.toggled.connect(self.zero_plane_armed.emit)
        lay.addWidget(self._zero_plane_btn)

        self._reset_proc_btn = QPushButton("↩ Reset to Original")
        self._reset_proc_btn.setFont(ui_font(9))
        self._reset_proc_btn.setFixedHeight(26)
        self._reset_proc_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._reset_proc_btn.setToolTip(_tip(
            "Undo the zero-plane correction and restore the scan as loaded "
            "from Browse (Viewer processing still applied). Also clears any "
            "segmentation results."))
        self._reset_proc_btn.clicked.connect(self.reset_to_original_requested.emit)
        lay.addWidget(self._reset_proc_btn)

        lay.addWidget(_sep())

        # ── Segmentation Settings ──────────────────────────────────────────────
        seg_title = QLabel("Segmentation Settings")
        seg_title.setFont(ui_font(10, weight=QFont.Bold))
        lay.addWidget(seg_title)

        # Threshold slider (0–255)
        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Threshold:"))
        thr_row.addStretch(1)
        self._thr_val_lbl = QLabel("128")
        self._thr_val_lbl.setMinimumWidth(28)
        self._thr_val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        thr_row.addWidget(self._thr_val_lbl)
        lay.addLayout(thr_row)

        self._thr_slider = QSlider(Qt.Horizontal)
        self._thr_slider.setRange(0, 255)
        self._thr_slider.setValue(128)
        self._thr_slider.setToolTip(_tip(
            "Normalised image intensity threshold (0–255). Pixels brighter "
            "than this become particle foreground; everything darker is "
            "background. Lower it to capture faint features, raise it to keep "
            "only the strongest. With 'Invert' on, the comparison flips."))
        self._thr_slider.valueChanged.connect(self._on_thr_slider_changed)
        self._thr_slider.valueChanged.connect(
            lambda _: self.classify_params_changed.emit())
        lay.addWidget(self._thr_slider)

        # Min Area slider (0–100 → 0–0.100% of image area)
        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Min Area:"))
        min_row.addStretch(1)
        self._min_area_val_lbl = QLabel("0.001%")
        self._min_area_val_lbl.setMinimumWidth(50)
        self._min_area_val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        min_row.addWidget(self._min_area_val_lbl)
        lay.addLayout(min_row)

        self._min_area_slider = QSlider(Qt.Horizontal)
        self._min_area_slider.setRange(0, 100)   # each unit = 0.001%  →  0–0.100%
        self._min_area_slider.setValue(1)
        self._min_area_slider.setToolTip(_tip(
            "Minimum particle area, as a percentage of the whole image "
            "(0–0.100%). Anything smaller is treated as noise and discarded. "
            "Raise this to clear away salt-and-pepper speckle; lower it to "
            "keep small particles."))
        self._min_area_slider.valueChanged.connect(self._on_min_area_slider_changed)
        self._min_area_slider.valueChanged.connect(
            lambda _: self.classify_params_changed.emit())
        lay.addWidget(self._min_area_slider)

        # Max Area slider (0–1000 → 0–1.000% of image area)
        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max Area:"))
        max_row.addStretch(1)
        self._max_area_val_lbl = QLabel("off")
        self._max_area_val_lbl.setMinimumWidth(50)
        self._max_area_val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        max_row.addWidget(self._max_area_val_lbl)
        lay.addLayout(max_row)

        self._max_area_slider = QSlider(Qt.Horizontal)
        self._max_area_slider.setRange(0, 1000)  # each unit = 0.001%  →  0–1.000%
        self._max_area_slider.setValue(0)
        self._max_area_slider.setToolTip(_tip(
            "Maximum particle area, as a percentage of the whole image "
            "(0–1.000%). Anything larger is discarded — useful for rejecting "
            "merged blobs or whole-terrace artefacts. Set to 0 for no upper "
            "limit."))
        self._max_area_slider.valueChanged.connect(self._on_max_area_slider_changed)
        self._max_area_slider.valueChanged.connect(
            lambda _: self.classify_params_changed.emit())
        lay.addWidget(self._max_area_slider)

        self._invert_cb = QCheckBox("Invert (segment dark features)")
        self._invert_cb.setFont(ui_font(9))
        self._invert_cb.setToolTip(_tip(
            "Segment dark features instead of bright ones. Turn this on to "
            "find pits, vacancies or depressions; leave it off for adatoms, "
            "molecules and islands that sit above the surface."))
        self._invert_cb.stateChanged.connect(
            lambda _: self.classify_params_changed.emit())
        lay.addWidget(self._invert_cb)

        # ── Exclusion Mask ─────────────────────────────────────────────────────
        lay.addWidget(_sep())
        mask_lbl = QLabel("Exclusion Mask")
        mask_lbl.setFont(ui_font(10, weight=QFont.Bold))
        lay.addWidget(mask_lbl)

        mask_hint = QLabel(
            "Paint step edges or regions to exclude from segmentation.")
        mask_hint.setFont(ui_font(8))
        mask_hint.setWordWrap(True)
        mask_hint.setStyleSheet("color: #888;")
        lay.addWidget(mask_hint)

        # ── Algorithmic step-edge exclusion (reproducible alternative to painting) ─
        self._step_exclude_cb = QCheckBox("Exclude step edges (auto)")
        self._step_exclude_cb.setFont(ui_font(9))
        self._step_exclude_cb.setToolTip(_tip(
            "Detect substrate step edges from the topography and drop molecules "
            "sitting on them — a reproducible alternative to painting over the "
            "step by hand. The computed band is shown in amber."))
        self._step_exclude_cb.toggled.connect(lambda _: self.step_exclude_changed.emit())
        lay.addWidget(self._step_exclude_cb)

        def _step_spin(label, lo, hi, val, step, decimals, suffix, tip):
            row = QHBoxLayout()
            lab = QLabel(label)
            lab.setFont(ui_font(9))
            row.addWidget(lab)
            row.addStretch(1)
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(decimals)
            sp.setValue(val)
            if suffix:
                sp.setSuffix(suffix)
            sp.setToolTip(_tip(tip))
            sp.valueChanged.connect(lambda _: self.step_exclude_changed.emit())
            row.addWidget(sp)
            lay.addLayout(row)
            return sp

        self._step_angle_spin = _step_spin(
            "Step sensitivity:", 5.0, 45.0, 20.0, 1.0, 0, "°",
            "Surface-slope angle above which a pixel counts as a step. A "
            "monatomic step is far steeper than terrace tilt, so the 20° "
            "default separates them with margin. Lower = more sensitive.")
        self._step_molsize_spin = _step_spin(
            "Molecule size:", 0.2, 20.0, 1.0, 0.1, 2, " nm",
            "Approximate molecule diameter. Molecules are suppressed before the "
            "step is detected so their own steep edges don't self-trigger; set "
            "this to roughly your molecule width.")
        self._step_margin_spin = _step_spin(
            "Edge margin:", 0.0, 5.0, 0.3, 0.1, 2, " nm",
            "Extra margin grown around the step band, so molecules sitting next "
            "to (not only squarely on) the step are also excluded.")
        self._step_minheight_spin = _step_spin(
            "Min step height:", 0.0, 5.0, 0.0, 0.05, 2, " nm",
            "Only exclude at steps at least this tall (0 = any steep edge). "
            "Distinguishes a real atomic step from a shallow undulation.")

        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Brush (px):"))
        self._brush_spin = QSpinBox()
        self._brush_spin.setRange(1, 300)
        self._brush_spin.setValue(10)
        self._brush_spin.setToolTip(_tip(
            "Radius of the exclusion-mask brush, in image pixels. Larger "
            "values paint over wide regions quickly; smaller values let you "
            "trace tightly around a step edge."))
        brush_row.addWidget(self._brush_spin, 1)
        lay.addLayout(brush_row)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Colour:"))
        self._mask_color_cb = QComboBox()
        for name in self.MASK_COLORS:
            self._mask_color_cb.addItem(name)
        self._mask_color_cb.setCurrentText("Red")
        self._mask_color_cb.setToolTip(_tip(
            "Colour used to draw the exclusion-mask overlay on the image. "
            "Pick whatever contrasts best with the current scan so the masked "
            "regions stand out."))
        self._mask_color_cb.currentTextChanged.connect(self._on_mask_color_changed)
        color_row.addWidget(self._mask_color_cb, 1)
        lay.addLayout(color_row)

        self._mask_btn = QPushButton("✏  Draw exclusion mask")
        self._mask_btn.setCheckable(True)
        self._mask_btn.setFont(ui_font(9))
        self._mask_btn.setFixedHeight(28)
        self._mask_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._mask_btn.setToolTip(_tip(
            "Toggle brush mode, then click or drag on the image to paint "
            "regions to ignore during segmentation — e.g. step edges, "
            "scan glitches, or a busy area you don't want to count. Toggle "
            "off again to resume panning and clicking particles."))
        self._mask_btn.toggled.connect(self._on_mask_btn_toggled)
        lay.addWidget(self._mask_btn)

        self._clear_mask_btn = QPushButton("🗑  Clear mask")
        self._clear_mask_btn.setFont(ui_font(9))
        self._clear_mask_btn.setFixedHeight(26)
        self._clear_mask_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._clear_mask_btn.setToolTip(_tip(
            "Erase the entire exclusion mask so the whole image is eligible "
            "for segmentation again."))
        self._clear_mask_btn.clicked.connect(self.mask_clear_requested.emit)
        lay.addWidget(self._clear_mask_btn)

        # ── Live-preview debounce timer ────────────────────────────────────────
        # Fires preview_requested 300 ms after the last slider/checkbox change.
        # Using a single-shot QTimer means rapid dragging only triggers one
        # background segmentation, not one per slider tick.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self.preview_requested.emit)

        # Hook sliders and invert checkbox to the debounce timer.
        # (classify_params_changed connections are kept separately so they still
        # clear sample labels when segmentation params change.)
        self._thr_slider.valueChanged.connect(lambda _: self._schedule_preview())
        self._min_area_slider.valueChanged.connect(lambda _: self._schedule_preview())
        self._max_area_slider.valueChanged.connect(lambda _: self._schedule_preview())
        self._invert_cb.stateChanged.connect(lambda _: self._schedule_preview())

        # ── Phase 1 action buttons ─────────────────────────────────────────────
        lay.addWidget(_sep())
        self._segment_btn = QPushButton("Apply Segmentation")
        self._segment_btn.setFont(ui_font(10, weight=QFont.Bold))
        self._segment_btn.setFixedHeight(34)
        self._segment_btn.setObjectName("accentBtn")
        self._segment_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._segment_btn.setToolTip(_tip(
            "Run segmentation at full resolution with the current settings and "
            "draw the particle contours on the image. Stays in Phase 1 so you "
            "can keep tuning the threshold and area filters before moving on."))
        self._segment_btn.clicked.connect(self.segment_requested.emit)
        lay.addWidget(self._segment_btn)

        self._clear_seg_btn = QPushButton("Remove Segmentation")
        self._clear_seg_btn.setFont(ui_font(9))
        self._clear_seg_btn.setFixedHeight(28)
        self._clear_seg_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._clear_seg_btn.setToolTip(_tip(
            "Remove the segmentation contour overlay and show the raw image "
            "again. Your threshold and area settings are kept."))
        self._clear_seg_btn.clicked.connect(self.clear_segmentation_requested.emit)
        lay.addWidget(self._clear_seg_btn)

        self._advance_btn = QPushButton("Move to Phase 2 →")
        self._advance_btn.setFont(ui_font(10))
        self._advance_btn.setFixedHeight(30)
        self._advance_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._advance_btn.setToolTip(_tip(
            "Move on to the Analysis phase, carrying over the particles found "
            "by 'Apply Segmentation'. Run 'Apply Segmentation' at least once "
            "first, otherwise there is nothing to analyse."))
        self._advance_btn.clicked.connect(self.advance_phase2_requested.emit)
        lay.addWidget(self._advance_btn)

        lay.addStretch(1)
        return page

    # ── Phase 2 — Analysis ────────────────────────────────────────────────────

    def _build_phase2(self) -> QWidget:
        page = QWidget()
        lay  = QVBoxLayout(page)
        lay.setContentsMargins(10, 10, 10, 6)
        lay.setSpacing(6)

        # ── Back button + segmentation count ──────────────────────────────────
        back_row = QHBoxLayout()
        back_btn = QPushButton("← Settings")
        back_btn.setFont(ui_font(9))
        back_btn.setFixedHeight(26)
        back_btn.setCursor(QCursor(Qt.PointingHandCursor))
        back_btn.setToolTip(_tip(
            "Go back to Phase 1 to change the threshold, area filters or "
            "exclusion mask. Your current particles and labels are kept."))
        back_btn.clicked.connect(self._on_back_to_phase1)
        back_row.addWidget(back_btn)
        back_row.addStretch(1)
        lay.addLayout(back_row)

        self._segment_count_lbl = QLabel("Segmentation not run yet.")
        self._segment_count_lbl.setFont(ui_font(9, weight=QFont.Bold))
        self._segment_count_lbl.setWordWrap(True)
        lay.addWidget(self._segment_count_lbl)

        lay.addWidget(_sep())

        # ── Mode tabs ──────────────────────────────────────────────────────────
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_btns: dict = {}
        _mode_tips = {
            "particles": "Count and characterise every segmented particle — "
                         "area, position, height, orientation and sharpness.",
            "template":  "Crop one example motif and find every place it "
                         "repeats by normalised cross-correlation. Good for "
                         "counting identical atoms or molecules.",
            "lattice":   "Extract the primitive lattice vectors (a, b and the "
                         "angle γ) from an atomically-resolved image using "
                         "SIFT keypoint clustering.",
            "classify":  "Sort particles into classes you define: label a few "
                         "examples by clicking them, then match the rest by "
                         "image similarity.",
        }
        for key, label in [("particles", "Particles"),
                            ("template",  "Template"),
                            ("lattice",   "Lattice"),
                            ("classify",  "Classify")]:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setFont(ui_font(9))
            b.setFixedHeight(26)
            b.setCursor(QCursor(Qt.PointingHandCursor))
            b.setToolTip(_tip(_mode_tips[key]))
            b.clicked.connect(lambda _=False, k=key: self._select_mode(k))
            self._mode_group.addButton(b)
            mode_row.addWidget(b)
            self._mode_btns[key] = b
        lay.addLayout(mode_row)

        self._mode_stack = QStackedWidget()
        self._mode_stack.addWidget(self._build_particles_tab())   # 0
        self._mode_stack.addWidget(self._build_template_tab())    # 1
        self._mode_stack.addWidget(self._build_lattice_tab())     # 2
        self._mode_stack.addWidget(self._build_classify_tab())    # 3
        lay.addWidget(self._mode_stack)

        lay.addWidget(_sep())

        self._run_btn = QPushButton("▶ Run")
        self._run_btn.setFont(ui_font(10, weight=QFont.Bold))
        self._run_btn.setFixedHeight(32)
        self._run_btn.setObjectName("accentBtn")
        self._run_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._run_btn.setToolTip(_tip(
            "Run the analysis for the mode selected above (Particles, "
            "Template, Lattice or Classify). Runs in the background, so the "
            "interface stays responsive while it works."))
        self._run_btn.clicked.connect(
            lambda: self.run_requested.emit(self._current_mode()))
        lay.addWidget(self._run_btn)

        self._clear_cls_btn = QPushButton("Remove Classification")
        self._clear_cls_btn.setFont(ui_font(9))
        self._clear_cls_btn.setFixedHeight(28)
        self._clear_cls_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._clear_cls_btn.setToolTip(_tip(
            "Discard the classification result and colours, returning to the "
            "plain particle contours. Your sample labels are kept so you can "
            "re-run without re-labelling."))
        self._clear_cls_btn.clicked.connect(self.clear_classification_requested.emit)
        lay.addWidget(self._clear_cls_btn)

        self._export_btn = QPushButton("Export JSON…")
        self._export_btn.setFont(ui_font(9))
        self._export_btn.setFixedHeight(28)
        self._export_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._export_btn.setToolTip(_tip(
            "Save the current mode's results to a JSON file — all values in SI "
            "units, with the source scan recorded for provenance. Run an "
            "analysis first so there is something to export."))
        self._export_btn.clicked.connect(
            lambda: self.export_requested.emit(self._current_mode()))
        lay.addWidget(self._export_btn)

        # Feature bank (Phase 1): persist the human-labelled CLIP embeddings so
        # future classifications can reuse examples across scans. CLIP-only —
        # the raw/PCA pixel vectors aren't comparable between images.
        self._add_bank_btn = QPushButton("➕ Add samples to bank…")
        self._add_bank_btn.setFont(ui_font(9))
        self._add_bank_btn.setFixedHeight(28)
        self._add_bank_btn.setCursor(QCursor(Qt.PointingHandCursor))
        if getattr(self, "_clip_available", False):
            self._add_bank_btn.setToolTip(_tip(
                "Save the CLIP embeddings of your labelled sample molecules to a "
                "reusable bank, after a confirmation step. Builds a library of "
                "examples across scans for better future classification."))
        else:
            self._add_bank_btn.setEnabled(False)
            self._add_bank_btn.setToolTip(_tip(
                "Needs the CLIP encoder (install 'probeflow[clip]'). The bank "
                "stores CLIP embeddings, which the raw/PCA encoders don't produce."))
        self._add_bank_btn.clicked.connect(self.add_to_bank_requested.emit)
        lay.addWidget(self._add_bank_btn)

        self._send_stats_btn = QPushButton("Send to Particle Statistics…")
        self._send_stats_btn.setFont(ui_font(9))
        self._send_stats_btn.setFixedHeight(28)
        self._send_stats_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._send_stats_btn.setToolTip(_tip(
            "Send the current mode's particle/detection positions to Particle "
            "Statistics as a feature set for spatial-statistics analysis. Run an "
            "analysis first so there are positions to send."))
        self._send_stats_btn.clicked.connect(
            lambda: self.send_to_particle_statistics_requested.emit(self._current_mode()))
        lay.addWidget(self._send_stats_btn)

        lay.addStretch(1)
        return page

    # ── Mode tab pages ─────────────────────────────────────────────────────────

    def _build_particles_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 4, 0, 4)
        l.setSpacing(4)
        info = QLabel(
            "Counts and characterises all segmented particles.\n\n"
            "Segmentation settings (threshold, area filters, exclusion mask) "
            "are in Phase 1 — press '← Settings' to change them.\n\n"
            "Press ▶ Run to re-run segmentation with the current settings.")
        info.setFont(ui_font(9))
        info.setWordWrap(True)
        info.setStyleSheet("color: #888;")
        l.addWidget(info)
        return w

    def _build_template_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 4, 0, 4)
        l.setSpacing(4)

        crop_btn = QPushButton("Crop template from image…")
        crop_btn.setFont(ui_font(9))
        crop_btn.setFixedHeight(28)
        crop_btn.setCursor(QCursor(Qt.PointingHandCursor))
        crop_btn.setToolTip(_tip(
            "Draw a tight rectangle over one example of the motif you want to "
            "count. That crop becomes the template matched across the whole "
            "image when you press Run."))
        crop_btn.clicked.connect(self.crop_template_requested.emit)
        l.addWidget(crop_btn)

        row = QHBoxLayout()
        row.addWidget(QLabel("min correlation:"))
        self._corr_spin = QDoubleSpinBox()
        self._corr_spin.setRange(0.0, 1.0)
        self._corr_spin.setDecimals(2)
        self._corr_spin.setSingleStep(0.05)
        self._corr_spin.setValue(0.5)
        self._corr_spin.setToolTip(_tip(
            "How closely a spot must match the template to count as a hit "
            "(0–1). Higher is stricter and finds fewer, cleaner matches; "
            "lower finds more but risks false positives. 0.4–0.6 is a good "
            "starting range."))
        row.addWidget(self._corr_spin)
        l.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("min distance (nm):"))
        self._dist_spin = QDoubleSpinBox()
        self._dist_spin.setRange(0.0, 1e4)
        self._dist_spin.setDecimals(3)
        self._dist_spin.setValue(0.0)   # 0 → auto
        self._dist_spin.setToolTip(_tip(
            "Smallest allowed spacing between two matches, in nanometres. "
            "Stops a single feature being counted several times. Set to 0 to "
            "let it choose automatically from the template size."))
        row2.addWidget(self._dist_spin)
        l.addLayout(row2)

        hint = QLabel("Tip: draw a tight rectangle over one motif. Distance 0 → auto.")
        hint.setFont(ui_font(8))
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
        info.setFont(ui_font(9))
        info.setWordWrap(True)
        l.addWidget(info)
        return w

    def _build_classify_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 4, 0, 4)
        l.setSpacing(6)

        info = QLabel(
            "① Press 'Apply Settings' to find particles.\n"
            "② Click particles on the image to assign class labels.\n"
            "③ Press ▶ Run to classify all remaining particles.")
        info.setFont(ui_font(9))
        info.setWordWrap(True)
        l.addWidget(info)

        # ── Threshold Settings group ───────────────────────────────────────────
        thr_box = QGroupBox("Threshold Settings")
        thr_box.setFont(ui_font(9))
        thr_box_lay = QVBoxLayout(thr_box)
        thr_box_lay.setSpacing(4)

        man_row = QHBoxLayout()
        man_row.addWidget(QLabel("Manual value (0–1):"))
        self._cls_manual_spin = QDoubleSpinBox()
        self._cls_manual_spin.setRange(0.0, 1.0)
        self._cls_manual_spin.setDecimals(3)
        self._cls_manual_spin.setSingleStep(0.05)
        self._cls_manual_spin.setValue(0.5)
        self._cls_manual_spin.setEnabled(False)   # only active in "Manual" mode
        self._cls_manual_spin.setToolTip(_tip(
            "Similarity cutoff used only in Manual mode (0–1). A particle is "
            "given its nearest sample's class when its best similarity is at "
            "or above this value, otherwise it is labelled 'other'. Lower it "
            "to label more particles, raise it to be stricter."))
        man_row.addWidget(self._cls_manual_spin)
        thr_box_lay.addLayout(man_row)

        self._cls_thr_group = QButtonGroup(thr_box)
        self._cls_thr_group.setExclusive(True)
        self._cls_thr_btns: dict = {}
        _thr_tips = {
            "manual": "Use the fixed cutoff set above. Best when you know the "
                      "similarity scale and want full, repeatable control.",
            "otsu":   "Pick the cutoff automatically by splitting the "
                      "similarity histogram (Otsu's method). Needs a genuine "
                      "spread between matches and outliers.",
            "gmm":    "Pick the cutoff automatically by fitting two Gaussians "
                      "to the similarities and cutting between them. If the "
                      "two groups are not clearly separated — e.g. every "
                      "particle is the same kind of thing — it falls back to "
                      "labelling them all by nearest sample rather than "
                      "wrongly dumping most into 'other'.",
            "distribution": "Set the cutoff one standard deviation above the "
                            "mean similarity. A simple, permissive choice.",
        }
        for key, label in [("manual", "Manual"), ("otsu", "Otsu"),
                            ("gmm", "GMM"), ("distribution", "Distribution")]:
            rb = QRadioButton(label)
            rb.setFont(ui_font(9))
            rb.setToolTip(_tip(_thr_tips[key]))
            rb.toggled.connect(
                lambda checked, k=key:
                    self._on_cls_thr_mode_changed(k) if checked else None)
            self._cls_thr_group.addButton(rb)
            thr_box_lay.addWidget(rb)
            self._cls_thr_btns[key] = rb
        self._cls_thr_btns["gmm"].setChecked(True)
        l.addWidget(thr_box)

        # ── Encoding Settings group ────────────────────────────────────────────
        enc_box = QGroupBox("Encoding Settings")
        enc_box.setFont(ui_font(9))
        enc_box_lay = QVBoxLayout(enc_box)
        enc_box_lay.setSpacing(4)

        from probeflow.analysis.features import clip_available
        self._clip_available = clip_available()

        self._enc_group = QButtonGroup(enc_box)
        self._enc_group.setExclusive(True)
        self._enc_btns: dict = {}
        _clip_tip = (
            "Embed each molecule crop with OpenAI CLIP ViT-B/32 and match by "
            "cosine similarity — the encoder the upstream UniMR tool uses. Far "
            "more discriminative than the pixel encoders."
        )
        if not self._clip_available:
            _clip_tip += (
                "\n\nUnavailable: install with  pip install 'probeflow[clip]'  "
                "(torch + openai-clip), then restart."
            )
        _enc_tips = {
            "raw":        "Compare particles by their raw pixel patterns "
                          "(brightness-normalised). Simplest and usually a "
                          "good default for small particles.",
            "pca_kmeans": "Reduce each particle crop to its main components "
                          "with PCA before comparing. Can be more robust to "
                          "noise when you have many particles.",
            "clip":       _clip_tip,
            "auto":       "Let ProbeFlow choose the encoding: CLIP when it is "
                          "installed, otherwise Raw Features.",
        }
        for key, label in [("raw",       "Raw Features"),
                            ("pca_kmeans", "PCA + KMeans"),
                            ("clip",      "CLIP (ViT-B/32)"),
                            ("auto",      "Auto Select")]:
            rb = QRadioButton(label)
            rb.setFont(ui_font(9))
            rb.setToolTip(_tip(_enc_tips[key]))
            if key == "clip" and not self._clip_available:
                rb.setEnabled(False)
            self._enc_group.addButton(rb)
            enc_box_lay.addWidget(rb)
            self._enc_btns[key] = rb
        # Default to CLIP when available (matches UniMR), else Raw.
        self._enc_btns["clip" if self._clip_available else "raw"].setChecked(True)
        l.addWidget(enc_box)

        # ── Augmentation & options ─────────────────────────────────────────────
        self._rotate_aug_cb = QCheckBox("Rotation Augmentation")
        self._rotate_aug_cb.setFont(ui_font(9))
        self._rotate_aug_cb.setToolTip(_tip(
            "Generate 36 rotated copies (0–350°, every 10°) of each labelled "
            "sample before matching, so a molecule is recognised whatever its "
            "orientation. Slightly slower to run."))
        l.addWidget(self._rotate_aug_cb)

        rot_hint = QLabel("36 rotations × 10° — orientation-invariant matching.")
        rot_hint.setFont(ui_font(8))
        rot_hint.setStyleSheet("color: #888;")
        rot_hint.setWordWrap(True)
        l.addWidget(rot_hint)

        self._sharpness_cb = QCheckBox("Sharpness-sensitive")
        self._sharpness_cb.setFont(ui_font(9))
        self._sharpness_cb.setToolTip(_tip(
            "Add edge sharpness (variance of the Laplacian) as an extra "
            "matching feature. Use it when two classes have the same shape "
            "but one is crisp and the other blurred."))
        l.addWidget(self._sharpness_cb)

        l.addWidget(_sep())

        undo_btn = QPushButton("↩ Undo last label")
        undo_btn.setFont(ui_font(9))
        undo_btn.setFixedHeight(26)
        undo_btn.setCursor(QCursor(Qt.PointingHandCursor))
        undo_btn.setToolTip(_tip(
            "Remove the most recently assigned sample label. Use it to step "
            "back through labels one at a time if you click the wrong "
            "particle."))
        undo_btn.clicked.connect(self.undo_label_requested.emit)
        l.addWidget(undo_btn)

        return w

    # ── Phase navigation ──────────────────────────────────────────────────────

    def _on_back_to_phase1(self) -> None:
        """Return to Phase 1 (Segmentation)."""
        self._phase_stack.setCurrentIndex(0)

    def set_segment_count(self, n: int) -> None:
        """Called after segmentation completes — shows count and switches to Phase 2."""
        self._segment_count_lbl.setText(
            f"{n} particle{'s' if n != 1 else ''} found.")
        self._phase_stack.setCurrentIndex(1)
        self._update_run_btn_label()

    def _update_run_btn_label(self) -> None:
        mode = self._current_mode()
        labels = {
            "particles": "▶ Run (Particles)",
            "template":  "▶ Run (Template)",
            "lattice":   "▶ Run (Lattice)",
            "classify":  "▶ Start Classification",
        }
        self._run_btn.setText(labels.get(mode, "▶ Run"))

    # ── Mode selection ─────────────────────────────────────────────────────────

    def _select_mode(self, key: str) -> None:
        for k, b in self._mode_btns.items():
            b.setChecked(k == key)
        idx = {"particles": 0, "template": 1, "lattice": 2, "classify": 3}[key]
        self._mode_stack.setCurrentIndex(idx)
        self.mode_changed.emit(key)
        self._update_run_btn_label()

    def _current_mode(self) -> str:
        for k, b in self._mode_btns.items():
            if b.isChecked():
                return k
        return "particles"

    def current_mode(self) -> str:
        return self._current_mode()

    # ── Live-preview scheduling ───────────────────────────────────────────────

    def _schedule_preview(self) -> None:
        """Restart the debounce timer — emits preview_requested after 300 ms idle."""
        self._preview_timer.start()   # calling start() on a running timer resets it

    # ── Slider value display handlers ─────────────────────────────────────────

    def _on_thr_slider_changed(self, value: int) -> None:
        self._thr_val_lbl.setText(str(value))

    def _on_min_area_slider_changed(self, value: int) -> None:
        pct = value * 0.001
        self._min_area_val_lbl.setText(f"{pct:.3f}%")

    def _on_max_area_slider_changed(self, value: int) -> None:
        if value == 0:
            self._max_area_val_lbl.setText("off")
        else:
            pct = value * 0.001
            self._max_area_val_lbl.setText(f"{pct:.3f}%")

    # ── Classify threshold method ─────────────────────────────────────────────

    def _on_cls_thr_mode_changed(self, key: str) -> None:
        """Enable the manual-value spinbox only when 'Manual' is selected."""
        self._cls_manual_spin.setEnabled(key == "manual")

    # ── Public parameter getters ──────────────────────────────────────────────

    def plane_index(self) -> int:
        return int(self._plane_cb.currentIndex())

    def particles_params(self) -> dict:
        """Return segmentation parameters.

        Area values are returned as percentage of image area (0–100 scale, e.g.
        0.001 means 0.001%).  Callers must convert to nm² using image shape and
        pixel sizes::

            pixel_area_nm2 = Nx * Ny * px_x_m * px_y_m * 1e18
            min_nm2 = (params["min_area_pct"] / 100.0) * pixel_area_nm2
        """
        return {
            "threshold":       "manual",
            "manual_value":    float(self._thr_slider.value()),
            "invert":          self._invert_cb.isChecked(),
            "min_area_pct":    self._min_area_slider.value() * 0.001,  # 0–0.100 %
            "max_area_pct":    self._max_area_slider.value() * 0.001,  # 0–1.000 %
            "size_sigma_clip": None,
        }

    def step_exclude_params(self) -> dict:
        """Algorithmic step-edge exclusion settings (physical units in metres).

        ``min_step_height_m`` is ``None`` when the spin is at 0 (slope-only).
        """
        return {
            "enabled":             self._step_exclude_cb.isChecked(),
            "threshold_deg":       float(self._step_angle_spin.value()),
            "molecule_diameter_m": float(self._step_molsize_spin.value()) * 1e-9,
            "dilate_m":            float(self._step_margin_spin.value()) * 1e-9,
            "min_step_height_m":   (float(self._step_minheight_spin.value()) * 1e-9
                                    if self._step_minheight_spin.value() > 0 else None),
        }

    def template_params(self) -> dict:
        return {
            "min_correlation": self._corr_spin.value(),
            "min_distance_m":  None if self._dist_spin.value() <= 0
                               else self._dist_spin.value() * 1e-9,
        }

    def classify_segmentation_params(self) -> dict:
        """Segmentation params for the classify workflow — proxies particles_params."""
        return self.particles_params()

    def classify_run_params(self) -> dict:
        """Classification parameters forwarded to classify_particles()."""
        thr_method = next(
            (k for k, rb in self._cls_thr_btns.items() if rb.isChecked()), "gmm"
        )
        encoder = next(
            (k for k, rb in self._enc_btns.items() if rb.isChecked()), "raw"
        )
        if encoder == "auto":
            encoder = "clip" if getattr(self, "_clip_available", False) else "raw"
        return {
            "use_sharpness":    self._sharpness_cb.isChecked(),
            "threshold_method": thr_method,
            "manual_threshold": self._cls_manual_spin.value(),
            "encoder":          encoder,
            "rotate_augment":   self._rotate_aug_cb.isChecked(),
        }

    # ── Mask helpers ──────────────────────────────────────────────────────────

    def mask_color_rgb(self) -> tuple[int, int, int]:
        name = self._mask_color_cb.currentText()
        return self.MASK_COLORS.get(name, (220, 50, 50))

    def brush_size(self) -> int:
        return int(self._brush_spin.value())

    def stop_mask_painting(self) -> None:
        """Uncheck the mask-draw button, stopping paint mode if it is active.

        Called automatically when advancing to Phase 2 so that mouse clicks
        on the image land on particles (for Classify labelling) rather than
        painting the exclusion mask.
        """
        if self._mask_btn.isChecked():
            self._mask_btn.setChecked(False)   # triggers _on_mask_btn_toggled(False)

    def set_status(self, text: str) -> None:
        self._status_lbl.setText(text)

    def update_histogram(self, arr) -> None:
        """Replot the height histogram and apply 1–99 % auto-clip to the display."""
        if arr is None:
            return
        flat = arr[np.isfinite(arr)].ravel()
        if flat.size == 0:
            self._histogram.clear(_FC_THEME)
            return
        # Pick a human-readable unit based on the median magnitude.
        med = abs(float(np.nanmedian(flat)))
        if med > 1e-9:
            scale, unit = 1e9,  "nm"
        elif med > 1e-12:
            scale, unit = 1e10, "Å"
        else:
            scale, unit = 1e12, "pm"
        self._hist_scale          = scale
        flat_phys                 = flat * scale
        lo_phys                   = float(np.percentile(flat_phys,  1.0))
        hi_phys                   = float(np.percentile(flat_phys, 99.0))
        self._hist_auto_lo_si     = lo_phys / scale
        self._hist_auto_hi_si     = hi_phys / scale
        self._hist_cur_lo_si      = self._hist_auto_lo_si
        self._hist_cur_hi_si      = self._hist_auto_hi_si
        data_min_phys             = float(np.percentile(flat_phys,  0.1))
        data_max_phys             = float(np.percentile(flat_phys, 99.9))
        self._histogram.render(
            flat_phys=flat_phys,
            lo_phys=lo_phys,
            hi_phys=hi_phys,
            unit=unit,
            axis_label="Height",
            theme=_FC_THEME,
            scale=scale,
            data_min_phys=data_min_phys,
            data_max_phys=data_max_phys,
        )
        # Notify the panel to apply auto-clip whenever data changes.
        self.display_clip_changed.emit(self._hist_auto_lo_si, self._hist_auto_hi_si)

    def disarm_zero_plane(self) -> None:
        """Un-toggle the Set Zero Plane button without re-emitting the signal."""
        self._zero_plane_btn.blockSignals(True)
        self._zero_plane_btn.setChecked(False)
        self._zero_plane_btn.blockSignals(False)

    def _on_mask_btn_toggled(self, checked: bool) -> None:
        self._mask_btn.setText(
            "✋  Stop drawing" if checked else "✏  Draw exclusion mask")
        self.mask_paint_toggled.emit(checked)

    def _on_mask_color_changed(self, name: str) -> None:
        r, g, b = self.MASK_COLORS.get(name, (220, 50, 50))
        self.mask_color_changed.emit(r, g, b)

    # ── Histogram contrast interaction ────────────────────────────────────────

    def _on_hist_range_released(self, lo_phys: float, hi_phys: float) -> None:
        """Drag line released — convert physical → SI and notify the panel."""
        scale = self._hist_scale
        if scale:
            lo_si = lo_phys / scale
            hi_si = hi_phys / scale
            self._hist_cur_lo_si = lo_si
            self._hist_cur_hi_si = hi_si
            self.display_clip_changed.emit(lo_si, hi_si)

    def _on_hist_min_released(self, v: int) -> None:
        """Min slider released — update the lo clip and move the drag line."""
        lo_si = self._histogram.sl_to_si(v)
        hi_si = self._hist_cur_hi_si
        if hi_si is not None and lo_si < hi_si:
            self._hist_cur_lo_si = lo_si
            self.display_clip_changed.emit(lo_si, hi_si)
            self._histogram.update_drag_lines(lo_si * self._hist_scale,
                                              hi_si * self._hist_scale)

    def _on_hist_max_released(self, v: int) -> None:
        """Max slider released — update the hi clip and move the drag line."""
        hi_si = self._histogram.sl_to_si(v)
        lo_si = self._hist_cur_lo_si
        if lo_si is not None and hi_si > lo_si:
            self._hist_cur_hi_si = hi_si
            self.display_clip_changed.emit(lo_si, hi_si)
            self._histogram.update_drag_lines(lo_si * self._hist_scale,
                                              hi_si * self._hist_scale)

    def _on_hist_auto(self) -> None:
        """Auto button — reset clip to 1–99 % percentile."""
        if self._hist_auto_lo_si is not None and self._hist_auto_hi_si is not None:
            self._hist_cur_lo_si = self._hist_auto_lo_si
            self._hist_cur_hi_si = self._hist_auto_hi_si
            self.display_clip_changed.emit(self._hist_auto_lo_si,
                                           self._hist_auto_hi_si)
            self._histogram.update_drag_lines(
                self._hist_auto_lo_si * self._hist_scale,
                self._hist_auto_hi_si * self._hist_scale)

    def _on_hist_reset(self) -> None:
        """Reset button — same as Auto for the FC sidebar."""
        self._on_hist_auto()
