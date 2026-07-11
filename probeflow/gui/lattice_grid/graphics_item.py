"""QGraphicsItem display layer for real-space lattice grids."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QGraphicsObject

from probeflow.analysis.lattice_grid import LatticeGrid

from .constants import (
    _COL_A,
    _COL_B,
    _COL_GRID,
    _COL_LABEL,
    _COL_ORIGIN,
    _COL_ROT,
    _COL_SCALE,
    _HANDLE_A,
    _HANDLE_B,
    _HANDLE_ORIGIN,
    _HANDLE_ROT,
    _HANDLE_SCALE,
    _HANDLE_SCREEN_R,
)

# ── LatticeGridItem — pure display ────────────────────────────────────────────

class LatticeGridItem(QGraphicsObject):
    """
    Draws the lattice grid on an ImageCanvas (QGraphicsScene).

    Scene coordinates match image pixel coordinates 1:1.
    All mouse interaction is handled by LatticeGridController.
    """

    grid_changed = Signal(object)   # emits LatticeGrid after any set_grid()

    def __init__(
        self,
        grid: LatticeGrid,
        image_w: int,
        image_h: int,
        cells: int = 12,
        parent=None,
        color=None,
    ):
        super().__init__(parent)
        self._grid = grid
        self._image_w = image_w
        self._image_h = image_h
        self._cells = cells
        self._line_width_px: float = 1.5
        # Custom line colour marks a stored (static) grid layer; the active
        # editable grid keeps the default blue with coloured basis vectors.
        self._color = QColor(color) if color is not None else None

        self.setZValue(50)
        self.setFlag(QGraphicsObject.ItemIsMovable, False)
        self.setFlag(QGraphicsObject.ItemIsFocusable, False)
        self.setAcceptHoverEvents(False)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def cells(self) -> int:
        return self._cells

    def set_cells(self, n: int) -> None:
        self._cells = max(1, int(n))
        self.update()

    def set_line_width(self, px: float) -> None:
        self._line_width_px = max(0.25, float(px))
        self.update()

    def grid(self) -> LatticeGrid:
        return self._grid

    def set_grid(self, grid: LatticeGrid) -> None:
        self._grid = grid
        self.update()
        self.grid_changed.emit(grid)

    # ── handle positions (scene / pixel coordinates) ───────────────────────────

    def handle_positions_scene(self) -> dict[int, tuple[float, float]]:
        """Return all handle positions in scene (= pixel) coordinates."""
        return _compute_handle_positions(self._grid)

    # ── QGraphicsItem interface ───────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        m = max(self._image_w, self._image_h) * (self._cells + 2.0)
        return QRectF(-m, -m, m * 3, m * 3)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if not self._grid.visible:
            return

        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(self._color if self._color is not None else _COL_GRID)
        pen.setCosmetic(True)
        pen.setWidthF(self._line_width_px)
        painter.setPen(pen)
        self._paint_grid_lines(painter)

        self._paint_basis_vectors(painter)

        if self._grid.show_labels:
            self._paint_labels(painter)

        if self._grid.show_handles:
            self._paint_handles(painter)

    def _paint_grid_lines(self, painter: QPainter) -> None:
        grid = self._grid
        ox, oy = grid.origin_px
        ax, ay = grid.a_px
        bx, by = grid.b_px
        c = self._cells

        # Lines parallel to b-vector (one line per n along a-direction)
        for n in range(-c, c + 1):
            sx = ox + n * ax - c * bx
            sy = oy + n * ay - c * by
            ex = ox + n * ax + c * bx
            ey = oy + n * ay + c * by
            painter.drawLine(QPointF(sx, sy), QPointF(ex, ey))

        # Lines parallel to a-vector (one line per m along b-direction)
        for m in range(-c, c + 1):
            sx = ox - c * ax + m * bx
            sy = oy - c * ay + m * by
            ex = ox + c * ax + m * bx
            ey = oy + c * ay + m * by
            painter.drawLine(QPointF(sx, sy), QPointF(ex, ey))

    def _paint_basis_vectors(self, painter: QPainter) -> None:
        grid = self._grid
        ox, oy = grid.origin_px
        ax, ay = grid.a_px
        bx, by = grid.b_px

        # Stored layers draw their basis in the layer colour so the per-handle
        # green/peach coding stays unique to the active grid.
        pen_a = QPen(self._color if self._color is not None else _COL_A)
        pen_a.setCosmetic(True)
        pen_a.setWidthF(2.0)
        painter.setPen(pen_a)
        painter.drawLine(QPointF(ox, oy), QPointF(ox + ax, oy + ay))

        pen_b = QPen(self._color if self._color is not None else _COL_B)
        pen_b.setCosmetic(True)
        pen_b.setWidthF(2.0)
        painter.setPen(pen_b)
        painter.drawLine(QPointF(ox, oy), QPointF(ox + bx, oy + by))

    def _paint_handles(self, painter: QPainter) -> None:
        zoom = painter.worldTransform().m11()
        r = _HANDLE_SCREEN_R / max(zoom, 0.01)

        handle_cols = {
            _HANDLE_ORIGIN: _COL_ORIGIN,
            _HANDLE_A:      _COL_A,
            _HANDLE_B:      _COL_B,
            _HANDLE_ROT:    _COL_ROT,
            _HANDLE_SCALE:  _COL_SCALE,
        }
        for hid, (hx, hy) in _compute_handle_positions(self._grid).items():
            col = handle_cols[hid]
            painter.setPen(QPen(col.darker(130)))
            painter.setBrush(QBrush(col))
            painter.drawEllipse(QPointF(hx, hy), r, r)

    def _paint_labels(self, painter: QPainter) -> None:
        zoom = painter.worldTransform().m11()
        font_pt = max(0.5, _HANDLE_SCREEN_R * 1.6 / max(zoom, 0.01))

        font = QFont("Helvetica")
        font.setPointSizeF(font_pt)
        painter.setFont(font)

        pen = QPen(_COL_LABEL)
        pen.setCosmetic(True)
        painter.setPen(pen)

        grid = self._grid
        handles = _compute_handle_positions(grid)
        off = _HANDLE_SCREEN_R * 1.2 / max(zoom, 0.01)

        a_lbl = "g1" if grid.space == "reciprocal" else "a"
        b_lbl = "g2" if grid.space == "reciprocal" else "b"

        ax_h, ay_h = handles[_HANDLE_A]
        bx_h, by_h = handles[_HANDLE_B]
        ox_h, oy_h = handles[_HANDLE_ORIGIN]

        painter.drawText(QPointF(ax_h + off, ay_h), a_lbl)
        painter.drawText(QPointF(bx_h + off, by_h), b_lbl)
        painter.drawText(QPointF(ox_h + off, oy_h), "O")


# ── handle position helper ────────────────────────────────────────────────────

def _compute_handle_positions(grid: LatticeGrid) -> dict[int, tuple[float, float]]:
    """Compute all handle positions in scene (pixel) coordinates."""
    ox, oy = grid.origin_px
    ax, ay = grid.a_px
    bx, by = grid.b_px
    la = math.hypot(ax, ay)
    lb = math.hypot(bx, by)
    avg_l = (la + lb) * 0.5 or 1.0

    if la > 1e-6:
        # Rotation handle perpendicular to a (90° CCW from a)
        perp_x = -ay / la
        perp_y =  ax / la
    else:
        perp_x, perp_y = 0.0, 1.0

    rot_dist = avg_l * 0.65

    return {
        _HANDLE_ORIGIN: (ox, oy),
        _HANDLE_A:      (ox + ax, oy + ay),
        _HANDLE_B:      (ox + bx, oy + by),
        _HANDLE_ROT:    (ox + perp_x * rot_dist, oy + perp_y * rot_dist),
        _HANDLE_SCALE:  (ox + (ax + bx) * 0.55, oy + (ay + by) * 0.55),
    }
