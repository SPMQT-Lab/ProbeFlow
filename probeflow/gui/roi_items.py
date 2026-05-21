"""
QGraphicsItem subclasses for rendering ROI objects on the image canvas.

Scene coordinates map 1:1 to image pixel coordinates:
  pixel (col, row) in the image → scene point QPointF(col, row)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsItemGroup,
    QGraphicsLineItem, QGraphicsPathItem,
    QGraphicsRectItem, QGraphicsTextItem,
)

if TYPE_CHECKING:
    from probeflow.core.roi import ROI


_PEN_INACTIVE   = QPen(QColor("#89b4fa"), 1.0)
_PEN_ACTIVE     = QPen(QColor("#22D3EE"), 1.5, Qt.DashLine)
_PEN_HOVER      = QPen(QColor("#f9e2af"), 1.5)
_BRUSH_INACTIVE = QBrush(QColor(137, 180, 250, 30))
_BRUSH_ACTIVE   = QBrush(QColor(34, 211, 238, 50))
_BRUSH_HOVER    = QBrush(QColor(249, 226, 175, 45))
_BRUSH_NONE     = QBrush(Qt.NoBrush)
_LABEL_FONT     = QFont("Helvetica", 8)
_LABEL_COLOR    = QColor("#cdd6f4")
_PEN_HANDLE     = QPen(QColor("#22D3EE"), 1.5)
_BRUSH_HANDLE   = QBrush(QColor(34, 211, 238, 200))


# ── Per-kind item builders ────────────────────────────────────────────────────

def _build_rectangle(roi: "ROI") -> QGraphicsRectItem:
    g = roi.geometry
    return QGraphicsRectItem(QRectF(g["x"], g["y"], g["width"], g["height"]))


def _build_ellipse(roi: "ROI") -> QGraphicsEllipseItem:
    g = roi.geometry
    cx, cy, rx, ry = g["cx"], g["cy"], g["rx"], g["ry"]
    return QGraphicsEllipseItem(QRectF(cx - rx, cy - ry, 2 * rx, 2 * ry))


def _build_polygon_path(roi: "ROI") -> QGraphicsPathItem:
    verts = roi.geometry.get("vertices", [])
    path = QPainterPath()
    if verts:
        path.moveTo(float(verts[0][0]), float(verts[0][1]))
        for v in verts[1:]:
            path.lineTo(float(v[0]), float(v[1]))
        path.closeSubpath()
    return QGraphicsPathItem(path)


def _build_multipolygon_path(roi: "ROI") -> QGraphicsPathItem:
    path = QPainterPath()
    path.setFillRule(Qt.OddEvenFill)
    for comp in roi.geometry.get("components", []):
        ext = comp.get("exterior", [])
        if len(ext) >= 3:
            path.moveTo(float(ext[0][0]), float(ext[0][1]))
            for v in ext[1:]:
                path.lineTo(float(v[0]), float(v[1]))
            path.closeSubpath()
        for hole in comp.get("holes", []):
            if len(hole) >= 3:
                path.moveTo(float(hole[0][0]), float(hole[0][1]))
                for v in hole[1:]:
                    path.lineTo(float(v[0]), float(v[1]))
                path.closeSubpath()
    return QGraphicsPathItem(path)


def _build_line(roi: "ROI") -> QGraphicsLineItem:
    g = roi.geometry
    return QGraphicsLineItem(g["x1"], g["y1"], g["x2"], g["y2"])


def _make_endpoint_handle(x: float, y: float) -> QGraphicsEllipseItem:
    """Fixed-screen-size filled circle for dragging a line ROI endpoint."""
    h = QGraphicsEllipseItem(-6, -6, 12, 12)
    h.setPos(QPointF(x, y))
    h.setPen(QPen(QColor("#ffffff"), 1.5))
    h.setBrush(QBrush(QColor("#22D3EE")))
    h.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
    h.setZValue(14)
    h.setToolTip("Drag to edit line endpoint")
    return h


class PointROIItem(QGraphicsItem):
    """Fixed-screen-size filled circle for a point ROI."""

    def __init__(self, x: float, y: float):
        super().__init__()
        self.setPos(QPointF(x, y))
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self._active = False

    def boundingRect(self) -> QRectF:
        return QRectF(-6, -6, 12, 12)

    def paint(self, painter: QPainter, _option, widget=None):
        color = QColor("#22D3EE") if self._active else QColor("#89b4fa")
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(0, 0), 4, 4)

    def set_active(self, active: bool):
        self._active = active
        self.update()


# ── Label helper ──────────────────────────────────────────────────────────────

def _make_label(roi: "ROI", shape_item: QGraphicsItem) -> QGraphicsTextItem:
    label = QGraphicsTextItem(roi.name)
    label.setFont(_LABEL_FONT)
    label.setDefaultTextColor(_LABEL_COLOR)

    br = shape_item.boundingRect() if not isinstance(shape_item, PointROIItem) else QRectF(0, 0, 0, 0)
    if isinstance(shape_item, PointROIItem):
        pos = shape_item.pos()
        label.setPos(pos.x() + 6, pos.y() - 10)
    elif isinstance(shape_item, QGraphicsLineItem):
        line = shape_item.line()
        label.setPos(min(line.x1(), line.x2()), min(line.y1(), line.y2()) - 12)
    else:
        label.setPos(br.x(), br.y() - 12)
    return label


def _tooltip_for_roi(roi: "ROI") -> str:
    if roi.kind == "line":
        return "Line ROI: click to select, drag active line or endpoints, right-click for profile/actions."
    if roi.kind in {"rectangle", "ellipse", "polygon", "freehand", "multipolygon"}:
        return "Area ROI: click to select, drag active ROI, right-click for mask/measure/actions."
    if roi.kind == "point":
        return "Point ROI: click to select, right-click for point actions."
    return "ROI: click to select, right-click for actions."


# ── Style helpers ─────────────────────────────────────────────────────────────

def _apply_style(shape_item: QGraphicsItem, active: bool) -> None:
    pen  = _PEN_ACTIVE if active else _PEN_INACTIVE

    if isinstance(shape_item, QGraphicsLineItem):
        shape_item.setPen(pen)
        return

    if isinstance(shape_item, PointROIItem):
        shape_item.set_active(active)
        return

    brush = _BRUSH_ACTIVE if active else _BRUSH_INACTIVE

    if isinstance(shape_item, QGraphicsRectItem):
        shape_item.setPen(pen)
        shape_item.setBrush(brush)
    elif isinstance(shape_item, QGraphicsEllipseItem):
        shape_item.setPen(pen)
        shape_item.setBrush(brush)
    elif isinstance(shape_item, QGraphicsPathItem):
        shape_item.setPen(pen)
        shape_item.setBrush(brush)


def _update_label_style(label: QGraphicsTextItem, active: bool, hover: bool) -> None:
    font = label.font()
    font.setBold(active)
    label.setFont(font)
    if active:
        label.setDefaultTextColor(QColor("#22D3EE"))
    elif hover:
        label.setDefaultTextColor(QColor("#f9e2af"))
    else:
        label.setDefaultTextColor(_LABEL_COLOR)


# ── Public API ────────────────────────────────────────────────────────────────

def make_roi_item(roi: "ROI", active: bool = False) -> QGraphicsItemGroup:
    """Build a QGraphicsItemGroup containing the shape item + name label."""
    kind = roi.kind

    if kind == "rectangle":
        shape = _build_rectangle(roi)
    elif kind == "ellipse":
        shape = _build_ellipse(roi)
    elif kind in ("polygon", "freehand"):
        shape = _build_polygon_path(roi)
    elif kind == "multipolygon":
        shape = _build_multipolygon_path(roi)
    elif kind == "line":
        shape = _build_line(roi)
    elif kind == "point":
        g = roi.geometry
        shape = PointROIItem(float(g["x"]), float(g["y"]))
    else:
        shape = QGraphicsRectItem(QRectF(0, 0, 1, 1))

    _apply_style(shape, active)
    shape.setZValue(10)

    label = _make_label(roi, shape)
    label.setZValue(11)
    _update_label_style(label, active, hover=False)

    tip = _tooltip_for_roi(roi)
    shape.setToolTip(tip)
    label.setToolTip(tip)

    group = QGraphicsItemGroup()
    group.setData(0, roi.id)
    group.setZValue(10)
    group.setToolTip(tip)

    if isinstance(shape, PointROIItem):
        # PointROIItem manages its own scene pos; don't add to group directly
        # — we store it separately so the group still exists as container.
        group.addToGroup(label)
        # Keep shape outside the group so ItemIgnoresTransformations works.
        # We attach it as a sibling and link via group data.
        group.setData(1, shape)
        shape.setToolTip(tip)
    else:
        group.addToGroup(shape)
        group.addToGroup(label)

    if kind == "line":
        g = roi.geometry
        line_width = int(g.get("width", 1))
        group.setData(4, line_width)
        if line_width > 1:
            for child in group.childItems():
                if isinstance(child, QGraphicsLineItem):
                    pen = child.pen()
                    pen.setWidthF(float(line_width))
                    child.setPen(pen)
                    break
        h1 = _make_endpoint_handle(float(g["x1"]), float(g["y1"]))
        h2 = _make_endpoint_handle(float(g["x2"]), float(g["y2"]))
        h1.setParentItem(group)
        h2.setParentItem(group)
        h1.setVisible(active)
        h2.setVisible(active)
        group.setData(2, h1)
        group.setData(3, h2)

    return group


def update_roi_item_style(item: QGraphicsItemGroup, active: bool,
                          hover: bool = False) -> None:
    """Update pen/brush on the shape inside *item* without rebuilding."""
    point_item = item.data(1)
    if point_item is not None and isinstance(point_item, PointROIItem):
        point_item.set_active(active or hover)
        for child in item.childItems():
            if isinstance(child, QGraphicsTextItem):
                _update_label_style(child, active, hover)
        return

    h1, h2 = item.data(2), item.data(3)
    line_width = item.data(4) or 1

    for child in item.childItems():
        if isinstance(child, QGraphicsTextItem):
            _update_label_style(child, active, hover)
            continue
        if child is h1 or child is h2:
            continue
        if hover and not active:
            pen = _PEN_HOVER
            brush = _BRUSH_HOVER if not isinstance(child, QGraphicsLineItem) else None
            child.setPen(pen)
            if brush is not None and hasattr(child, "setBrush"):
                child.setBrush(brush)
        else:
            _apply_style(child, active)
        if isinstance(child, QGraphicsLineItem) and line_width > 1:
            pen = child.pen()
            pen.setWidthF(float(line_width))
            child.setPen(pen)

    if h1 is not None:
        h1.setVisible(active)
    if h2 is not None:
        h2.setVisible(active)
