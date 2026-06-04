"""
QGraphicsItem subclasses for rendering ROI objects on the image canvas.

Scene coordinates map 1:1 to image pixel coordinates:
  pixel (col, row) in the image → scene point QPointF(col, row)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from probeflow.gui.typography import ui_font
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush, QColor, QPainter, QPainterPath, QPen,
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
_LABEL_FONT     = ui_font(8)
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


def _make_resize_handle(x: float, y: float) -> QGraphicsEllipseItem:
    """Fixed-screen-size filled circle for dragging a ROI resize handle."""
    h = QGraphicsEllipseItem(-6, -6, 12, 12)
    h.setPos(QPointF(x, y))
    h.setPen(QPen(QColor("#ffffff"), 1.5))
    h.setBrush(QBrush(QColor("#22D3EE")))
    h.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
    h.setZValue(14)
    h.setToolTip("Drag to resize")
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


# ── ROI interaction hints (single source of truth) ────────────────────────────
# One click selects the highlighted ROI; a second interaction on the now-active
# ROI edits it (select-then-edit). Each kind's hint is a title plus short
# clauses, shared by the floating item tooltips (wrapped onto multiple rows near
# the cursor) and the status-bar hint (a single concise line).

_AREA_ROI_KINDS = {"rectangle", "ellipse", "polygon", "freehand", "multipolygon"}

_ROI_HINTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "line": (
        "Line ROI",
        (
            "Click to select this line.",
            "Drag the active line or its endpoints to edit.",
            "Right-click for profile and actions.",
        ),
    ),
    "area": (
        "Area ROI",
        (
            "Click to select this ROI.",
            "Drag the active ROI to move it; drag handles to resize.",
            "Right-click for mask, measure and actions.",
        ),
    ),
    "point": (
        "Point ROI",
        (
            "Click to select this point.",
            "Right-click for point actions.",
        ),
    ),
    "roi": (
        "ROI",
        (
            "Click to select.",
            "Right-click for actions.",
        ),
    ),
}


def _roi_hint_key(kind: str) -> str:
    if kind == "line":
        return "line"
    if kind in _AREA_ROI_KINDS:
        return "area"
    if kind == "point":
        return "point"
    return "roi"


def roi_hint_text(kind: str) -> str:
    """Concise single-line interaction hint for the status bar."""
    title, clauses = _ROI_HINTS[_roi_hint_key(kind)]
    return f"{title}: " + " ".join(clauses)


def roi_tooltip_html(kind: str) -> str:
    """Rich-text tooltip that wraps onto several short rows near the cursor.

    Qt only word-wraps tooltips when the text is rich text; the explicit
    ``<br>`` breaks keep the tooltip from stretching into one wide row.
    """
    title, clauses = _ROI_HINTS[_roi_hint_key(kind)]
    body = "<br>".join(clauses)
    return f"<qt><b>{title}</b><br>{body}</qt>"


def _tooltip_for_roi(roi: "ROI") -> str:
    return roi_tooltip_html(roi.kind)


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
    """Build a QGraphicsItemGroup containing the shape item + name label.

    PySide6 lifetime invariant for setData() with QGraphicsItem values
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``QGraphicsItem.setData(key, item)`` stores only a raw C++ pointer
    inside a QVariant.  PySide6 does NOT hold a Python reference through
    QVariant storage.  If the Python wrapper for *item* reaches refcount 0
    (e.g. because the local variable goes out of scope when this function
    returns), the wrapper is freed, and — if Python owned the C++ object —
    the C++ object is deleted.  Any later call to ``.data(key)`` then passes
    a dangling pointer to ``QGraphicsItem_PTR_CppToPython_QGraphicsItem``
    → SIGSEGV.

    To keep each stored item alive, one of the following must hold:

    1. **Python reference on the group wrapper** — set an attribute such as
       ``group._foo_ref = item`` before the local variable leaves scope.
       PySide6 wrapper objects have a ``__dict__`` so arbitrary attributes
       are allowed.  The ref is released automatically when the group
       wrapper is GC'd.  Used for: key 1 (PointROIItem, not in scene yet
       when setData is called).

    2. **C++ parent ownership via setParentItem()** — calling
       ``item.setParentItem(group)`` *before* ``setData`` transfers C++
       ownership to the parent.  If the Python wrapper is subsequently
       freed its C++ object is NOT deleted (the parent owns it); the stored
       pointer therefore remains valid for the lifetime of the parent.
       Used for: keys 2 & 3 (endpoint handles).  We also store explicit
       Python refs on the group for consistency and defence-in-depth.
    """
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
        # IMPORTANT: setData() stores only a raw C++ QGraphicsItem* in a
        # QVariant — it does NOT hold a Python reference.  Without an explicit
        # Python ref the PointROIItem wrapper (and its C++ object) would be
        # garbage-collected the moment this function returns and 'shape' leaves
        # scope, leaving a dangling pointer in the QVariant.  Storing the
        # wrapper as a Python attribute on the group keeps it alive for as long
        # as the group itself is alive.
        group._point_roi_item_ref = shape
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

    # Resize handles — generic across kinds. resize_handles() returns the
    # ordered draggable handles for this ROI (rectangle: 8, ellipse: 4, line: 2;
    # empty for non-resizable kinds). Each handle item must be kept alive past
    # this function: setParentItem(group) transfers C++ ownership (invariant 2),
    # and we also keep an explicit Python dict ref (group._resize_handles) used
    # both as a name→item map for live repositioning during drag and for
    # defence-in-depth against the setData lifetime pitfall described above.
    from probeflow.core.roi import resize_handles
    handle_map: dict = {}
    for h in resize_handles(roi):
        item = _make_resize_handle(h.x, h.y)
        item.setParentItem(group)  # transfers C++ ownership to parent (invariant 2)
        item.setVisible(active)
        handle_map[h.name] = item
    group._resize_handles = handle_map

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

    handles = getattr(item, "_resize_handles", {}) or {}
    handle_items = set(handles.values())
    line_width = item.data(4) or 1

    for child in item.childItems():
        if isinstance(child, QGraphicsTextItem):
            _update_label_style(child, active, hover)
            continue
        if child in handle_items:
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

    for handle in handle_items:
        handle.setVisible(active)


def update_roi_item_geometry(item: QGraphicsItemGroup, roi: "ROI") -> None:
    """Update the shape child and resize-handle positions for new geometry.

    Used for live drag feedback without rebuilding the whole item. Mirrors the
    per-kind builders: rectangle/ellipse reset their QRectF, line resets its
    endpoints; handles are repositioned from ``resize_handles(roi)`` by name.
    """
    from probeflow.core.roi import resize_handles
    g = roi.geometry
    handles = getattr(item, "_resize_handles", {}) or {}
    handle_items = set(handles.values())
    for child in item.childItems():
        if child in handle_items:
            # Handles are themselves QGraphicsEllipseItems — never mistake one
            # for an ellipse ROI's shape.
            continue
        if isinstance(child, QGraphicsRectItem):
            child.setRect(QRectF(g["x"], g["y"], g["width"], g["height"]))
            break
        if isinstance(child, QGraphicsEllipseItem):
            cx, cy, rx, ry = g["cx"], g["cy"], g["rx"], g["ry"]
            child.setRect(QRectF(cx - rx, cy - ry, 2 * rx, 2 * ry))
            break
        if isinstance(child, QGraphicsLineItem):
            child.setLine(g["x1"], g["y1"], g["x2"], g["y2"])
            break
    for h in resize_handles(roi):
        handle = handles.get(h.name)
        if handle is not None:
            handle.setPos(QPointF(h.x, h.y))
