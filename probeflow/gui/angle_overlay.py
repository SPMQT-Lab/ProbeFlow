"""Interactive 3-point angle overlay for ImageCanvas."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsLineItem,
    QGraphicsTextItem,
)

_HANDLE_R  = 5.0
_PEN_ARM   = QPen(QColor("#22D3EE"), 1.0)
_PEN_HDL   = QPen(QColor("#22D3EE"), 1.0)
_BRUSH_HDL = QBrush(QColor(34, 211, 238, 180))


class _AngleHandle(QGraphicsEllipseItem):
    def __init__(self, pos: QPointF, on_move):
        r = _HANDLE_R
        super().__init__(-r, -r, 2 * r, 2 * r)
        self._on_move = on_move
        self.setPos(pos)
        self.setPen(_PEN_HDL)
        self.setBrush(_BRUSH_HDL)
        self.setFlags(
            QGraphicsItem.ItemIgnoresTransformations
            | QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setZValue(10)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self._on_move()
        return super().itemChange(change, value)


class AngleOverlayItem(QGraphicsItemGroup):
    """Manages two arm lines, three draggable handles, and a live angle readout."""

    def __init__(self, p1: QPointF, p2: QPointF, p3: QPointF, scene):
        super().__init__()
        self._h1 = _AngleHandle(p1, self._update)
        self._h2 = _AngleHandle(p2, self._update)
        self._h3 = _AngleHandle(p3, self._update)
        self._arm1 = QGraphicsLineItem()
        self._arm2 = QGraphicsLineItem()
        self._arm1.setPen(_PEN_ARM)
        self._arm2.setPen(_PEN_ARM)
        self._arm1.setZValue(9)
        self._arm2.setZValue(9)
        self._txt = QGraphicsTextItem()
        self._txt.setDefaultTextColor(QColor("#22D3EE"))
        f = QFont("Helvetica", 10)
        f.setBold(True)
        self._txt.setFont(f)
        self._txt.setZValue(11)
        for item in (self._arm1, self._arm2, self._txt, self._h1, self._h2, self._h3):
            scene.addItem(item)
        self._update()

    def _update(self) -> None:
        p1, p2, p3 = self._h1.pos(), self._h2.pos(), self._h3.pos()
        self._arm1.setLine(p1.x(), p1.y(), p2.x(), p2.y())
        self._arm2.setLine(p2.x(), p2.y(), p3.x(), p3.y())
        deg = self._angle_deg(p1, p2, p3)
        self._txt.setPlainText(f"{deg:.1f}°")
        self._txt.setPos(p2 + QPointF(8, -20))

    @staticmethod
    def _angle_deg(p1: QPointF, p2: QPointF, p3: QPointF) -> float:
        v1 = p1 - p2
        v2 = p3 - p2
        dot = v1.x() * v2.x() + v1.y() * v2.y()
        mag = math.hypot(v1.x(), v1.y()) * math.hypot(v2.x(), v2.y())
        if mag < 1e-9:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))

    def remove_from_scene(self, scene) -> None:
        for item in (self._arm1, self._arm2, self._txt, self._h1, self._h2, self._h3):
            scene.removeItem(item)

    @property
    def angle_deg(self) -> float:
        return self._angle_deg(self._h1.pos(), self._h2.pos(), self._h3.pos())

    @property
    def points(self) -> tuple[QPointF, QPointF, QPointF]:
        return self._h1.pos(), self._h2.pos(), self._h3.pos()
