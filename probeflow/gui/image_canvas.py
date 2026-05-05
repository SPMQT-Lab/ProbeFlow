"""
QGraphicsView-based image canvas for ProbeFlow.

Scene coordinates map 1:1 to image pixel coordinates:
  pixel (col, row) → scene QPointF(col, row).
  Scene rect (0, 0, Nx, Ny) spans the full image.

This widget is a drop-in replacement for the _ZoomLabel widget in
probeflow.gui.viewer.widgets.  It exposes the same signals and the same
public API so ImageViewerDialog requires only targeted edits.

Drawing tools (Phase 4b) are not implemented here.  The selection-tool
buttons remain in the UI but are disabled until Phase 4b lands.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen, QPixmap, QTransform,
)
from PySide6.QtWidgets import (
    QFrame, QGraphicsItem, QGraphicsItemGroup, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsTextItem, QGraphicsView, QScrollArea, QToolTip,
)

from probeflow.gui.roi_items import make_roi_item, update_roi_item_style


class ImageCanvas(QGraphicsView):
    """QGraphicsView-based image canvas — drop-in replacement for _ZoomLabel."""

    marker_clicked            = Signal(object)
    pixel_clicked             = Signal(float, float)
    selection_preview_changed = Signal(object)
    selection_changed         = Signal(object)
    pixmap_resized            = Signal(int)
    context_menu_requested    = Signal(object)
    pixel_hovered             = Signal(int, int, object)

    # ── inner items ──────────────────────────────────────────────────────────

    class _SpecMarkerItem(QGraphicsItem):
        """Fixed-screen-size yellow labelled circle for spec position markers."""

        def __init__(self, label: str, entry):
            super().__init__()
            self._label = label
            self._entry = entry
            self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

        @property
        def entry(self):
            return self._entry

        @property
        def label(self) -> str:
            return self._label

        def boundingRect(self) -> QRectF:
            return QRectF(-9, -9, 18, 18)

        def paint(self, painter: QPainter, option, widget=None):
            painter.setBrush(QBrush(QColor("#FFD700")))
            painter.setPen(QPen(QColor("black"), 1.5))
            painter.drawEllipse(QPointF(0, 0), 7, 7)
            painter.setFont(QFont("Helvetica", 6, QFont.Bold))
            painter.setPen(QPen(QColor("black")))
            from PySide6.QtCore import QRectF
            painter.drawText(QRectF(-7, -7, 14, 14), Qt.AlignCenter, self._label)

    class _ZeroMarkerItem(QGraphicsItem):
        """Fixed-screen-size cyan crosshair + circle for set-zero picks."""

        def __init__(self, label: str):
            super().__init__()
            self._label = label
            self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

        def boundingRect(self) -> QRectF:
            return QRectF(-14, -14, 28, 28)

        def paint(self, painter: QPainter, option, widget=None):
            r = 8
            painter.setPen(QPen(QColor(0, 0, 0, 200), 1))
            painter.drawLine(-r - 4, 0, -r, 0)
            painter.drawLine(r, 0, r + 4, 0)
            painter.drawLine(0, -r - 4, 0, -r)
            painter.drawLine(0, r, 0, r + 4)
            painter.setBrush(QBrush(QColor("#22D3EE")))
            painter.setPen(QPen(QColor("black"), 1.5))
            painter.drawEllipse(QPointF(0, 0), r, r)
            if self._label:
                painter.setFont(QFont("Helvetica", 6, QFont.Bold))
                painter.setPen(QPen(QColor("black")))
                from PySide6.QtCore import QRectF
                painter.drawText(QRectF(-r, -r, 2 * r, 2 * r), Qt.AlignCenter, self._label)

    # ── init ─────────────────────────────────────────────────────────────────

    def __init__(self, parent=None):
        super().__init__(parent)

        scene = QGraphicsScene(self)
        self.setScene(scene)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.Antialiasing)

        self._zoom: float = 1.0
        self._image_pixmap: Optional[QPixmap] = None
        self._image_size: Optional[tuple[int, int]] = None
        self._raw_arr: Optional[np.ndarray] = None

        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setZValue(0)
        self._pixmap_item.setTransformationMode(Qt.FastTransformation)
        scene.addItem(self._pixmap_item)

        self._roi_group = scene.createItemGroup([])
        self._roi_group.setZValue(10)

        self._marker_items: list[ImageCanvas._SpecMarkerItem] = []
        self._zero_marker_items: list[ImageCanvas._ZeroMarkerItem] = []

        self._text_overlay_item = QGraphicsTextItem()
        self._text_overlay_item.setDefaultTextColor(QColor("#cdd6f4"))
        self._text_overlay_item.setFont(QFont("Helvetica", 14))
        self._text_overlay_item.setZValue(30)
        self._text_overlay_item.setVisible(False)
        scene.addItem(self._text_overlay_item)

        self._markers: list[dict] = []
        self._show_markers: bool = True
        self._zero_markers: list[dict] = []
        self._set_zero_mode: bool = False

        self._selection_tool: str = "none"
        self._selection_geometry = None

        self._image_roi_set = None
        self._roi_items: dict[str, QGraphicsItemGroup] = {}

    # ── public image API ─────────────────────────────────────────────────────

    def set_source(self, pixmap: QPixmap, reset_zoom: bool = True) -> None:
        self._image_pixmap = pixmap
        if pixmap is None or pixmap.isNull():
            self._pixmap_item.setPixmap(QPixmap())
            self._image_size = None
            self._text_overlay_item.setPlainText("No image")
            self._text_overlay_item.setVisible(True)
            return

        self._pixmap_item.setPixmap(pixmap)
        self._image_size = (pixmap.width(), pixmap.height())
        self._text_overlay_item.setVisible(False)

        if reset_zoom:
            self._zoom = 1.0

        self._rebuild_marker_items()
        self._rebuild_zero_marker_items()
        self._rebuild_roi_items()
        self._apply_zoom()

    def _apply_zoom(self) -> None:
        if self._image_pixmap is None or self._image_pixmap.isNull():
            return
        Nx, Ny = self._image_size
        w = max(1, int(Nx * self._zoom))
        h = max(1, int(Ny * self._zoom))
        self.scene().setSceneRect(0, 0, Nx, Ny)
        self.setTransform(QTransform().scale(self._zoom, self._zoom))
        self.setFixedSize(w, h)
        self.pixmap_resized.emit(w)

    def zoom_by(self, factor: float) -> None:
        self._zoom = max(0.25, min(8.0, self._zoom * factor))
        self._apply_zoom()

    def reset_zoom(self) -> None:
        self._zoom = 1.0
        self._apply_zoom()

    def zoom(self) -> float:
        return self._zoom

    def fit_to_view(self) -> None:
        if self._image_size is None:
            return
        Nx, Ny = self._image_size
        # Walk up to find a scroll area viewport
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                vp = parent.viewport()
                avail_w = vp.width()
                avail_h = vp.height()
                break
            parent = parent.parent() if hasattr(parent, "parent") else None
        else:
            avail_w = self.parentWidget().width() if self.parentWidget() else Nx
            avail_h = self.parentWidget().height() if self.parentWidget() else Ny

        if Nx <= 0 or Ny <= 0 or avail_w <= 0 or avail_h <= 0:
            return
        zoom = min(avail_w / Nx, avail_h / Ny)
        self._zoom = max(0.25, min(8.0, zoom))
        self._apply_zoom()

    # ── compat shims (QLabel interface) ──────────────────────────────────────

    def setText(self, text: str) -> None:
        """Show loading/error text overlay. Empty string hides it."""
        self._text_overlay_item.setPlainText(text)
        self._text_overlay_item.setVisible(bool(text))

    def setPixmap(self, pm: QPixmap) -> None:
        """Compat shim: null pixmap clears image, non-null delegates to set_source."""
        if pm is None or pm.isNull():
            self._image_pixmap = None
            self._pixmap_item.setPixmap(QPixmap())
            self._image_size = None
        else:
            self.set_source(pm, reset_zoom=False)

    def pixmap(self) -> QPixmap:
        """Compat shim: returns a QPixmap sized to current zoomed widget dimensions."""
        if self._image_pixmap is None or self._image_pixmap.isNull():
            return QPixmap()
        return self._image_pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.FastTransformation,
        )

    # ── raw array ────────────────────────────────────────────────────────────

    def set_raw_array(self, arr) -> None:
        self._raw_arr = arr

    # ── spec markers ─────────────────────────────────────────────────────────

    def set_markers(self, markers: list[dict]) -> None:
        self._markers = list(markers or [])
        self._rebuild_marker_items()

    def set_show_markers(self, visible: bool) -> None:
        self._show_markers = bool(visible)
        for item in self._marker_items:
            item.setVisible(self._show_markers)

    def _rebuild_marker_items(self) -> None:
        for item in self._marker_items:
            self.scene().removeItem(item)
        self._marker_items.clear()
        if not self._markers or self._image_size is None:
            return
        Nx, Ny = self._image_size
        for i, m in enumerate(self._markers):
            item = ImageCanvas._SpecMarkerItem(str(i + 1), m["entry"])
            item.setPos(m["frac_x"] * Nx, m["frac_y"] * Ny)
            item.setZValue(20)
            item.setVisible(self._show_markers)
            self.scene().addItem(item)
            self._marker_items.append(item)

    # ── zero markers ─────────────────────────────────────────────────────────

    def set_zero_markers(self, markers: list[dict]) -> None:
        self._zero_markers = list(markers or [])
        self._rebuild_zero_marker_items()

    def _rebuild_zero_marker_items(self) -> None:
        for item in self._zero_marker_items:
            self.scene().removeItem(item)
        self._zero_marker_items.clear()
        if not self._zero_markers or self._image_size is None:
            return
        Nx, Ny = self._image_size
        for m in self._zero_markers:
            item = ImageCanvas._ZeroMarkerItem(str(m.get("label", "")))
            item.setPos(m["frac_x"] * Nx, m["frac_y"] * Ny)
            item.setZValue(20)
            self.scene().addItem(item)
            self._zero_marker_items.append(item)

    # ── ROI set display ───────────────────────────────────────────────────────

    def set_roi_set(self, roi_set) -> None:
        self._image_roi_set = roi_set
        self._rebuild_roi_items()

    def set_active_roi_id(self, roi_id: "str | None") -> None:
        if self._image_roi_set is not None:
            self._image_roi_set.active_roi_id = roi_id
        self._update_roi_styles()

    def _rebuild_roi_items(self) -> None:
        for item in list(self._roi_items.values()):
            # Also remove any PointROIItem stored in data slot 1
            point = item.data(1)
            if point is not None:
                self.scene().removeItem(point)
            self.scene().removeItem(item)
        self._roi_items.clear()
        if self._image_roi_set is None:
            return
        active_id = self._image_roi_set.active_roi_id
        for roi in self._image_roi_set.rois:
            self._add_roi_item_internal(roi, active=(roi.id == active_id))

    def _add_roi_item_internal(self, roi, active: bool) -> None:
        item = make_roi_item(roi, active=active)
        self.scene().addItem(item)
        point = item.data(1)
        if point is not None:
            self.scene().addItem(point)
        self._roi_items[roi.id] = item

    def add_roi_item(self, roi) -> None:
        if roi.id in self._roi_items:
            self.update_roi_item(roi)
            return
        active = bool(
            self._image_roi_set
            and self._image_roi_set.active_roi_id == roi.id
        )
        self._add_roi_item_internal(roi, active=active)

    def remove_roi_item(self, roi_id: str) -> None:
        item = self._roi_items.pop(roi_id, None)
        if item is not None:
            point = item.data(1)
            if point is not None:
                self.scene().removeItem(point)
            self.scene().removeItem(item)

    def update_roi_item(self, roi) -> None:
        self.remove_roi_item(roi.id)
        active = bool(
            self._image_roi_set
            and self._image_roi_set.active_roi_id == roi.id
        )
        self._add_roi_item_internal(roi, active=active)

    def _update_roi_styles(self) -> None:
        if not self._image_roi_set:
            return
        active_id = self._image_roi_set.active_roi_id
        for roi_id, item in self._roi_items.items():
            update_roi_item_style(item, active=(roi_id == active_id))

    # ── selection tool stubs (Phase 4b) ───────────────────────────────────────

    def set_selection_tool(self, kind: str) -> None:
        self._selection_tool = "none"

    def selection_tool(self) -> str:
        return self._selection_tool

    def current_selection(self):
        return None

    def clear_roi(self) -> None:
        self._selection_geometry = None

    def nudge_line(self, dx_px, dy_px, image_shape) -> bool:
        return False

    def set_set_zero_mode(self, enabled: bool) -> None:
        self._set_zero_mode = bool(enabled)
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    # ── mouse events ─────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event) -> None:
        pos = self.mapToScene(event.pos())
        col = int(pos.x())
        row = int(pos.y())
        if self._image_size is not None and self._raw_arr is not None:
            Nx, Ny = self._image_size
            if 0 <= col < Nx and 0 <= row < Ny:
                try:
                    val = float(self._raw_arr[row, col])
                except Exception:
                    val = None
                self.pixel_hovered.emit(col, row, val)

        if self._show_markers and self._markers and self._image_size:
            for item in self._marker_items:
                sp = self.mapFromScene(item.pos())
                if (abs(sp.x() - event.pos().x()) <= 10
                        and abs(sp.y() - event.pos().y()) <= 10):
                    entry = item.entry
                    lines = [entry.stem]
                    if getattr(entry, "measurement_label", None):
                        lines.append(entry.measurement_label)
                    if getattr(entry, "sweep_type", None) and entry.sweep_type != "unknown":
                        lines.append(entry.sweep_type)
                    if getattr(entry, "bias_mv", None) is not None:
                        lines.append(f"Bias: {entry.bias_mv:.0f} mV")
                    QToolTip.showText(event.globalPosition().toPoint(),
                                      "\n".join(lines), self)
                    super().mouseMoveEvent(event)
                    return
            QToolTip.hideText()

        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            if self._set_zero_mode and self._image_size:
                pos = self.mapToScene(event.pos())
                Nx, Ny = self._image_size
                fx = max(0.0, min(1.0, pos.x() / Nx))
                fy = max(0.0, min(1.0, pos.y() / Ny))
                self.pixel_clicked.emit(fx, fy)
                return

            if self._show_markers and self._marker_items:
                for item in self._marker_items:
                    sp = self.mapFromScene(item.pos())
                    if (abs(sp.x() - event.pos().x()) <= 12
                            and abs(sp.y() - event.pos().y()) <= 12):
                        self.marker_clicked.emit(item.entry)
                        return

        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            self.zoom_by(1.12 if delta > 0 else 1 / 1.12)
            event.accept()
        else:
            super().wheelEvent(event)

    def contextMenuEvent(self, event) -> None:
        self.context_menu_requested.emit(event.globalPos())
        event.accept()
