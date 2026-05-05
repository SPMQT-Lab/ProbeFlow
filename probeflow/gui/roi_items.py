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
    QGraphicsLineItem, QGraphicsPathItem, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsTextItem,
)

if TYPE_CHECKING:
    from probeflow.core.roi import ROI


_PEN_INACTIVE   = QPen(QColor("#89b4fa"), 1.5)
_PEN_ACTIVE     = QPen(QColor("#22D3EE"), 3.0, Qt.DashLine)
_PEN_HOVER      = QPen(QColor("#f9e2af"), 2.0)
_BRUSH_INACTIVE = QBrush(QColor(137, 180, 250, 30))
_BRUSH_ACTIVE   = QBrush(QColor(34, 211, 238, 50))
_BRUSH_HOVER    = QBrush(QColor(249, 226, 175, 45))
_BRUSH_NONE     = QBrush(Qt.NoBrush)
_LABEL_FONT     = QFont("Helvetica", 8)
_LABEL_COLOR    = QColor("#cdd6f4")


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


class PointROIItem(QGraphicsItem):
    """Fixed-screen-size filled circle for a point ROI."""

    def __init__(self, x: float, y: float):
        super().__init__()
        self.setPos(QPointF(x, y))
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self._active = False

    def boundingRect(self) -> QRectF:
        return QRectF(-6, -6, 12, 12)

    def paint(self, painter: QPainter, option, widget=None):
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

    group = QGraphicsItemGroup()
    group.setData(0, roi.id)
    group.setZValue(10)

    if isinstance(shape, PointROIItem):
        # PointROIItem manages its own scene pos; don't add to group directly
        # — we store it separately so the group still exists as container.
        group.addToGroup(label)
        # Keep shape outside the group so ItemIgnoresTransformations works.
        # We attach it as a sibling and link via group data.
        group.setData(1, shape)
    else:
        group.addToGroup(shape)
        group.addToGroup(label)

    return group


def update_roi_item_style(item: QGraphicsItemGroup, active: bool,
                          hover: bool = False) -> None:
    """Update pen/brush on the shape inside *item* without rebuilding."""
    point_item = item.data(1)
    if point_item is not None and isinstance(point_item, PointROIItem):
        point_item.set_active(active or hover)
        return

    for child in item.childItems():
        if isinstance(child, QGraphicsTextItem):
            continue
        if hover and not active:
            pen = _PEN_HOVER
            brush = _BRUSH_HOVER if not isinstance(child, QGraphicsLineItem) else None
            child.setPen(pen)
            if brush is not None and hasattr(child, "setBrush"):
                child.setBrush(brush)
        else:
            _apply_style(child, active)
