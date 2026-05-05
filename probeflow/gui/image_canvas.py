"""
QGraphicsView-based image canvas for ProbeFlow.

Scene coordinates map 1:1 to image pixel coordinates:
  pixel (col, row) → scene QPointF(col, row).
  Scene rect (0, 0, Nx, Ny) spans the full image.

This widget is a drop-in replacement for the _ZoomLabel widget in
probeflow.gui.viewer.widgets.  It exposes the same signals and the same
public API so ImageViewerDialog requires only targeted edits.

Phase 4b: Full drawing-tool support — rectangle, ellipse, polygon,
freehand, line, point — plus pan-mode ROI drag-move and hover highlight.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPainterPath, QPen,
    QPixmap, QTransform,
)
from PySide6.QtWidgets import (
    QFrame, QGraphicsEllipseItem, QGraphicsItem, QGraphicsItemGroup,
    QGraphicsLineItem, QGraphicsPathItem, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsTextItem, QGraphicsView,
    QScrollArea, QToolTip,
)

from probeflow.gui.roi_items import make_roi_item, update_roi_item_style

# ── Preview-item styling ──────────────────────────────────────────────────────

_PEN_PREVIEW   = QPen(QColor("#fab387"), 2.0, Qt.DashLine)
_BRUSH_PREVIEW = QBrush(QColor(250, 179, 135, 40))
_PEN_VERTEX    = QPen(QColor("#fab387"), 1.0)
_BRUSH_VERTEX  = QBrush(QColor("#fab387"))

_DRAWING_TOOLS = frozenset({"rectangle", "ellipse", "polygon", "freehand", "line", "point"})


class ImageCanvas(QGraphicsView):
    """QGraphicsView-based image canvas — drop-in replacement for _ZoomLabel."""

    marker_clicked            = Signal(object)
    pixel_clicked             = Signal(float, float)
    selection_preview_changed = Signal(object)
    selection_changed         = Signal(object)
    pixmap_resized            = Signal(int)
    context_menu_requested    = Signal(object)
    pixel_hovered             = Signal(int, int, object)

    # Phase 4b signals
    roi_created               = Signal(object)        # new ROI object
    roi_move_requested        = Signal(str, int, int) # (roi_id, dx, dy)
    tool_changed              = Signal(str)
    roi_context_menu_requested = Signal(str, object)  # (roi_id, global_pos)

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
        self.setFocusPolicy(Qt.StrongFocus)

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

        self._selection_tool: str = "pan"
        self._selection_geometry = None

        self._image_roi_set = None
        self._roi_items: dict[str, QGraphicsItemGroup] = {}

        # ── Phase 4b drawing state ────────────────────────────────────────────
        self._tool: str = "pan"
        self._draw_start: Optional[QPointF] = None
        self._draw_pts: list[QPointF] = []
        self._freehand_active: bool = False
        self._preview_item: Optional[QGraphicsItem] = None
        self._preview_vertices: list[QGraphicsItem] = []

        # Pan drag state (left-button pan or middle-button pan)
        self._left_pan_start: Optional[QPoint] = None
        self._mid_pan_start: Optional[QPoint] = None

        # Active-ROI drag-move state
        self._move_roi_id: Optional[str] = None
        self._move_scene_start: Optional[QPointF] = None
        self._move_item_start_pos: QPointF = QPointF(0, 0)
        self._move_point_start_pos: Optional[QPointF] = None

        # Hover state
        self._hover_roi_id: Optional[str] = None

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
        self._text_overlay_item.setPlainText(text)
        self._text_overlay_item.setVisible(bool(text))

    def setPixmap(self, pm: QPixmap) -> None:
        if pm is None or pm.isNull():
            self._image_pixmap = None
            self._pixmap_item.setPixmap(QPixmap())
            self._image_size = None
        else:
            self.set_source(pm, reset_zoom=False)

    def pixmap(self) -> QPixmap:
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
            point = item.data(1)
            if point is not None:
                self.scene().removeItem(point)
            self.scene().removeItem(item)
        self._roi_items.clear()
        self._hover_roi_id = None
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
            update_roi_item_style(
                item,
                active=(roi_id == active_id),
                hover=(roi_id == self._hover_roi_id and roi_id != active_id),
            )

    # ── drawing tool API ─────────────────────────────────────────────────────

    def set_tool(self, kind: str) -> None:
        """Switch to the named drawing tool, cancelling any in-progress drawing."""
        if kind not in ("pan", "rectangle", "ellipse", "polygon",
                        "freehand", "line", "point"):
            kind = "pan"
        if kind != self._tool:
            self._cancel_drawing()
            self._tool = kind
            self._selection_tool = kind
            self.setCursor(self._cursor_for_tool(kind))
            self.tool_changed.emit(kind)

    def tool(self) -> str:
        return self._tool

    def cancel_drawing(self) -> None:
        """Public entry-point: cancel any in-progress drawing and return to pan."""
        if self._tool != "pan" or self._draw_start is not None or self._draw_pts:
            self._cancel_drawing()
            self._tool = "pan"
            self._selection_tool = "pan"
            self.setCursor(Qt.ArrowCursor)
            self.tool_changed.emit("pan")

    # ── compat shims for old _ZoomLabel API ──────────────────────────────────

    def set_selection_tool(self, kind: str) -> None:
        self.set_tool(kind if kind != "none" else "pan")

    def selection_tool(self) -> str:
        return self._tool

    def current_selection(self):
        return None

    def clear_roi(self) -> None:
        self._selection_geometry = None

    def nudge_line(self, dx_px, dy_px, image_shape) -> bool:
        return False

    def set_set_zero_mode(self, enabled: bool) -> None:
        self._set_zero_mode = bool(enabled)
        self.setCursor(Qt.CrossCursor if enabled else self._cursor_for_tool(self._tool))

    # ── drawing helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _cursor_for_tool(tool: str) -> Qt.CursorShape:
        return Qt.ArrowCursor if tool == "pan" else Qt.CrossCursor

    @staticmethod
    def _snap(pos: QPointF) -> QPointF:
        return QPointF(int(pos.x()), int(pos.y()))

    def _roi_at_pos(self, view_pos: QPoint) -> "str | None":
        """Return the ROI id under the given view-space position, or None."""
        items_under = self.items(view_pos)
        for item in items_under:
            roi_id = item.data(0) if callable(getattr(item, "data", None)) else None
            if roi_id and roi_id in self._roi_items:
                return roi_id
            parent = item.parentItem()
            while parent is not None:
                roi_id = parent.data(0) if callable(getattr(parent, "data", None)) else None
                if roi_id and roi_id in self._roi_items:
                    return roi_id
                parent = parent.parentItem()
        # Check PointROI items stored outside their group
        for rid, grp in self._roi_items.items():
            point = grp.data(1)
            if point is not None and point in items_under:
                return rid
        return None

    def _find_scroll_area(self) -> "QScrollArea | None":
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                return parent
            parent = parent.parent() if hasattr(parent, "parent") else None
        return None

    def _scroll_by(self, dx: int, dy: int) -> None:
        sa = self._find_scroll_area()
        if sa is None:
            return
        hbar = sa.horizontalScrollBar()
        vbar = sa.verticalScrollBar()
        hbar.setValue(hbar.value() + dx)
        vbar.setValue(vbar.value() + dy)

    # ── preview item management ───────────────────────────────────────────────

    def _clear_preview(self) -> None:
        if self._preview_item is not None:
            self.scene().removeItem(self._preview_item)
            self._preview_item = None
        for v in self._preview_vertices:
            self.scene().removeItem(v)
        self._preview_vertices.clear()

    def _set_preview(self, item: QGraphicsItem) -> None:
        self._clear_preview()
        item.setPen(_PEN_PREVIEW)
        if hasattr(item, "setBrush"):
            item.setBrush(_BRUSH_PREVIEW)
        item.setZValue(50)
        self.scene().addItem(item)
        self._preview_item = item

    def _update_rect_preview(self, p1: QPointF, p2: QPointF) -> None:
        x = min(p1.x(), p2.x())
        y = min(p1.y(), p2.y())
        w = abs(p2.x() - p1.x())
        h = abs(p2.y() - p1.y())
        self._set_preview(QGraphicsRectItem(QRectF(x, y, w, h)))

    def _update_ellipse_preview(self, p1: QPointF, p2: QPointF) -> None:
        x = min(p1.x(), p2.x())
        y = min(p1.y(), p2.y())
        w = abs(p2.x() - p1.x())
        h = abs(p2.y() - p1.y())
        self._set_preview(QGraphicsEllipseItem(QRectF(x, y, w, h)))

    def _update_line_preview(self, p1: QPointF, p2: QPointF) -> None:
        item = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
        item.setPen(_PEN_PREVIEW)
        item.setZValue(50)
        self._clear_preview()
        self.scene().addItem(item)
        self._preview_item = item

    def _update_polygon_preview(self, cursor_pos: "QPointF | None" = None) -> None:
        pts = self._draw_pts
        if not pts:
            return
        # Build path through collected vertices + optional cursor position
        path = QPainterPath()
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        if cursor_pos is not None:
            path.lineTo(cursor_pos)
        path_item = QGraphicsPathItem(path)
        path_item.setPen(_PEN_PREVIEW)
        path_item.setBrush(QBrush(Qt.NoBrush))
        path_item.setZValue(50)
        # Small vertex circles
        self._clear_preview()
        self.scene().addItem(path_item)
        self._preview_item = path_item
        for p in pts:
            dot = QGraphicsEllipseItem(QRectF(p.x() - 2, p.y() - 2, 4, 4))
            dot.setPen(_PEN_VERTEX)
            dot.setBrush(_BRUSH_VERTEX)
            dot.setZValue(51)
            self.scene().addItem(dot)
            self._preview_vertices.append(dot)

    def _update_freehand_preview(self) -> None:
        pts = self._draw_pts
        if len(pts) < 2:
            return
        path = QPainterPath()
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        path_item = QGraphicsPathItem(path)
        path_item.setPen(_PEN_PREVIEW)
        path_item.setBrush(QBrush(Qt.NoBrush))
        path_item.setZValue(50)
        self._clear_preview()
        self.scene().addItem(path_item)
        self._preview_item = path_item

    # ── ROI creation ─────────────────────────────────────────────────────────

    def _auto_name(self, kind: str) -> str:
        if self._image_roi_set is None:
            return f"{kind}_1"
        existing = {r.name for r in self._image_roi_set.rois}
        i = 1
        while f"{kind}_{i}" in existing:
            i += 1
        return f"{kind}_{i}"

    def _finish_roi(self, kind: str, geometry: dict) -> None:
        """Create a ROI, emit roi_created, and return to pan."""
        from probeflow.core.roi import ROI
        name = self._auto_name(kind)
        roi = ROI.new(kind, geometry, name=name)
        self._cancel_drawing()
        self._tool = "pan"
        self._selection_tool = "pan"
        self.setCursor(Qt.ArrowCursor)
        self.tool_changed.emit("pan")
        self.roi_created.emit(roi)

    def _finish_polygon(self) -> None:
        pts = self._draw_pts[:]
        self._clear_preview()
        self._draw_pts.clear()
        if len(pts) >= 3:
            vertices = [[p.x(), p.y()] for p in pts]
            self._finish_roi("polygon", {"vertices": vertices})
        else:
            # Too few points — cancel silently
            self._tool = "pan"
            self._selection_tool = "pan"
            self.setCursor(Qt.ArrowCursor)
            self.tool_changed.emit("pan")

    def _cancel_drawing(self) -> None:
        self._clear_preview()
        self._draw_start = None
        self._draw_pts.clear()
        self._freehand_active = False

    # ── hover highlight ───────────────────────────────────────────────────────

    def _update_hover(self, view_pos: QPoint) -> None:
        roi_id = self._roi_at_pos(view_pos)
        if roi_id == self._hover_roi_id:
            return
        active_id = self._image_roi_set.active_roi_id if self._image_roi_set else None
        # Un-highlight previous
        if self._hover_roi_id and self._hover_roi_id in self._roi_items:
            update_roi_item_style(
                self._roi_items[self._hover_roi_id],
                active=(self._hover_roi_id == active_id),
                hover=False,
            )
        self._hover_roi_id = roi_id
        # Apply hover highlight (if not the active ROI)
        if roi_id and roi_id in self._roi_items and roi_id != active_id:
            update_roi_item_style(self._roi_items[roi_id], active=False, hover=True)

    # ── mouse events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        # ── middle-mouse pan ──────────────────────────────────────────────────
        if event.button() == Qt.MiddleButton:
            self._mid_pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        # ── set-zero mode ─────────────────────────────────────────────────────
        if self._set_zero_mode and self._image_size:
            pos = self.mapToScene(event.pos())
            Nx, Ny = self._image_size
            fx = max(0.0, min(1.0, pos.x() / Nx))
            fy = max(0.0, min(1.0, pos.y() / Ny))
            self.pixel_clicked.emit(fx, fy)
            return

        # ── spec marker click ─────────────────────────────────────────────────
        if self._show_markers and self._marker_items:
            for item in self._marker_items:
                sp = self.mapFromScene(item.pos())
                if (abs(sp.x() - event.pos().x()) <= 12
                        and abs(sp.y() - event.pos().y()) <= 12):
                    self.marker_clicked.emit(item.entry)
                    return

        scene_pos = self._snap(self.mapToScene(event.pos()))
        tool = self._tool

        # ── pan tool ──────────────────────────────────────────────────────────
        if tool == "pan":
            roi_id = self._roi_at_pos(event.pos())
            active_id = self._image_roi_set.active_roi_id if self._image_roi_set else None
            if roi_id and roi_id == active_id:
                # Start drag-move for active ROI
                item = self._roi_items.get(roi_id)
                self._move_roi_id = roi_id
                self._move_scene_start = scene_pos
                self._move_item_start_pos = item.pos() if item else QPointF(0, 0)
                point = item.data(1) if item else None
                self._move_point_start_pos = point.pos() if point else None
                self.setCursor(Qt.SizeAllCursor)
            else:
                self._left_pan_start = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        # ── drawing tools ─────────────────────────────────────────────────────
        if tool == "point":
            self._finish_roi("point", {"x": float(scene_pos.x()), "y": float(scene_pos.y())})
            event.accept()
            return

        if tool == "polygon":
            self._draw_pts.append(scene_pos)
            self._update_polygon_preview(cursor_pos=scene_pos)
            event.accept()
            return

        if tool == "freehand":
            self._draw_pts = [scene_pos]
            self._freehand_active = True
            event.accept()
            return

        if tool in ("rectangle", "ellipse", "line"):
            self._draw_start = scene_pos
            if tool == "rectangle":
                self._update_rect_preview(scene_pos, scene_pos)
            elif tool == "ellipse":
                self._update_ellipse_preview(scene_pos, scene_pos)
            else:
                self._update_line_preview(scene_pos, scene_pos)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        # ── middle-mouse pan ──────────────────────────────────────────────────
        if self._mid_pan_start is not None and event.buttons() & Qt.MiddleButton:
            delta = event.pos() - self._mid_pan_start
            self._mid_pan_start = event.pos()
            self._scroll_by(-delta.x(), -delta.y())
            event.accept()
            # fall through to update pixel readout below (don't return early)

        # ── active-ROI drag-move ──────────────────────────────────────────────
        elif self._move_roi_id is not None and event.buttons() & Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            dx = scene_pos.x() - self._move_scene_start.x()
            dy = scene_pos.y() - self._move_scene_start.y()
            item = self._roi_items.get(self._move_roi_id)
            if item:
                item.setPos(QPointF(
                    self._move_item_start_pos.x() + dx,
                    self._move_item_start_pos.y() + dy,
                ))
                point = item.data(1)
                if point is not None and self._move_point_start_pos is not None:
                    point.setPos(QPointF(
                        self._move_point_start_pos.x() + dx,
                        self._move_point_start_pos.y() + dy,
                    ))
            event.accept()
            return

        # ── left-button pan ───────────────────────────────────────────────────
        elif (self._tool == "pan" and self._left_pan_start is not None
              and event.buttons() & Qt.LeftButton):
            delta = event.pos() - self._left_pan_start
            self._left_pan_start = event.pos()
            self._scroll_by(-delta.x(), -delta.y())
            event.accept()
            # fall through to update pixel readout

        # ── drawing preview ───────────────────────────────────────────────────
        else:
            scene_pos = self._snap(self.mapToScene(event.pos()))
            tool = self._tool
            if event.buttons() & Qt.LeftButton:
                if tool == "rectangle" and self._draw_start is not None:
                    self._update_rect_preview(self._draw_start, scene_pos)
                elif tool == "ellipse" and self._draw_start is not None:
                    self._update_ellipse_preview(self._draw_start, scene_pos)
                elif tool == "line" and self._draw_start is not None:
                    self._update_line_preview(self._draw_start, scene_pos)
                elif tool == "freehand" and self._freehand_active:
                    self._draw_pts.append(scene_pos)
                    self._update_freehand_preview()
            if tool == "polygon" and self._draw_pts:
                self._update_polygon_preview(cursor_pos=scene_pos)

        # ── hover highlight (pan mode, no drag) ───────────────────────────────
        if self._tool == "pan" and not (event.buttons() & Qt.LeftButton):
            self._update_hover(event.pos())

        # ── pixel coordinate readout ──────────────────────────────────────────
        raw_pos = self.mapToScene(event.pos())
        col = int(raw_pos.x())
        row = int(raw_pos.y())
        if self._image_size is not None and self._raw_arr is not None:
            Nx, Ny = self._image_size
            if 0 <= col < Nx and 0 <= row < Ny:
                try:
                    val = float(self._raw_arr[row, col])
                except Exception:
                    val = None
                self.pixel_hovered.emit(col, row, val)

        # ── marker tooltip ────────────────────────────────────────────────────
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

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._mid_pan_start = None
            self.setCursor(self._cursor_for_tool(self._tool))
            event.accept()
            return

        if event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return

        # ── finish active-ROI drag-move ───────────────────────────────────────
        if self._move_roi_id is not None:
            scene_pos = self.mapToScene(event.pos())
            dx = round(scene_pos.x() - self._move_scene_start.x())
            dy = round(scene_pos.y() - self._move_scene_start.y())
            roi_id = self._move_roi_id
            # Reset item to original position (data model will rebuild)
            item = self._roi_items.get(roi_id)
            if item:
                item.setPos(self._move_item_start_pos)
                point = item.data(1)
                if point is not None and self._move_point_start_pos is not None:
                    point.setPos(self._move_point_start_pos)
            self._move_roi_id = None
            self._move_scene_start = None
            self._move_item_start_pos = QPointF(0, 0)
            self._move_point_start_pos = None
            self.setCursor(Qt.ArrowCursor)
            if dx != 0 or dy != 0:
                self.roi_move_requested.emit(roi_id, dx, dy)
            event.accept()
            return

        # ── end pan drag ──────────────────────────────────────────────────────
        if self._left_pan_start is not None:
            self._left_pan_start = None
            self.setCursor(self._cursor_for_tool(self._tool))
            event.accept()
            return

        # ── complete drawing shapes ───────────────────────────────────────────
        tool = self._tool
        scene_pos = self._snap(self.mapToScene(event.pos()))

        if tool == "rectangle" and self._draw_start is not None:
            p1, p2 = self._draw_start, scene_pos
            x = min(p1.x(), p2.x())
            y = min(p1.y(), p2.y())
            w = abs(p2.x() - p1.x())
            h = abs(p2.y() - p1.y())
            self._draw_start = None
            if w >= 2 and h >= 2:
                self._finish_roi("rectangle", {"x": float(x), "y": float(y),
                                               "width": float(w), "height": float(h)})
            else:
                self._cancel_drawing()
            event.accept()
            return

        if tool == "ellipse" and self._draw_start is not None:
            p1, p2 = self._draw_start, scene_pos
            cx = (p1.x() + p2.x()) / 2.0
            cy = (p1.y() + p2.y()) / 2.0
            rx = abs(p2.x() - p1.x()) / 2.0
            ry = abs(p2.y() - p1.y()) / 2.0
            self._draw_start = None
            if rx >= 1 and ry >= 1:
                self._finish_roi("ellipse", {"cx": float(cx), "cy": float(cy),
                                             "rx": float(rx), "ry": float(ry)})
            else:
                self._cancel_drawing()
            event.accept()
            return

        if tool == "line" and self._draw_start is not None:
            p1, p2 = self._draw_start, scene_pos
            dx = abs(p2.x() - p1.x())
            dy = abs(p2.y() - p1.y())
            self._draw_start = None
            if dx >= 1 or dy >= 1:
                self._finish_roi("line", {
                    "x1": float(p1.x()), "y1": float(p1.y()),
                    "x2": float(p2.x()), "y2": float(p2.y()),
                })
            else:
                self._cancel_drawing()
            event.accept()
            return

        if tool == "freehand" and self._freehand_active:
            self._freehand_active = False
            pts = self._draw_pts[:]
            self._draw_pts.clear()
            self._clear_preview()
            if len(pts) >= 3:
                vertices = [[p.x(), p.y()] for p in pts]
                self._finish_roi("freehand", {"vertices": vertices})
            else:
                self._tool = "pan"
                self._selection_tool = "pan"
                self.setCursor(Qt.ArrowCursor)
                self.tool_changed.emit("pan")
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._tool == "polygon" and self._draw_pts:
            # The press for this double-click already appended a duplicate vertex; remove it
            if len(self._draw_pts) > 1:
                self._draw_pts.pop()
            self._finish_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        k = event.key()
        if k == Qt.Key_Escape and (self._tool != "pan" or self._draw_pts or
                                    self._draw_start is not None):
            self.cancel_drawing()
            event.accept()
            return
        if k in (Qt.Key_Return, Qt.Key_Enter) and self._tool == "polygon" and self._draw_pts:
            self._finish_polygon()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            self.zoom_by(1.12 if delta > 0 else 1 / 1.12)
            event.accept()
        else:
            super().wheelEvent(event)

    def contextMenuEvent(self, event) -> None:
        roi_id = self._roi_at_pos(event.pos())
        if roi_id:
            self.roi_context_menu_requested.emit(roi_id, event.globalPos())
        else:
            self.context_menu_requested.emit(event.globalPos())
        event.accept()
