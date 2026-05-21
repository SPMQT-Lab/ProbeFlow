"""Viewport event-filter controller for lattice-grid handle editing."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QEvent, QObject, QPointF, Qt

from probeflow.analysis.lattice_grid import LatticeGrid

from .constants import (
    HIT_RADIUS_PX,
    _HANDLE_A,
    _HANDLE_B,
    _HANDLE_NONE,
    _HANDLE_ORIGIN,
    _HANDLE_ROT,
    _HANDLE_SCALE,
)
from .graphics_item import LatticeGridItem

if TYPE_CHECKING:
    from .real_space_panel import LatticeGridPanel

# ── LatticeGridController — viewport event filter ─────────────────────────────

class LatticeGridController(QObject):
    """
    Installs a viewport event filter on ImageCanvas for reliable handle dragging.

    Hit testing is done in screen pixels so it is zoom-invariant.
    Once a drag starts it is not stolen by the canvas's own pan/ROI handlers.
    """

    def __init__(self, item: LatticeGridItem, canvas, parent=None):
        super().__init__(parent)
        self._item = item
        self._canvas = canvas
        self._panel: Optional["LatticeGridPanel"] = None
        self._active: bool = False
        self._locked: bool = True
        self._ab_equal: bool = False
        self._dragging: bool = False
        self._drag_handle: int = _HANDLE_NONE
        self._drag_start_scene: Optional[QPointF] = None
        self._drag_grid_start: Optional[LatticeGrid] = None

    def set_panel(self, panel: "LatticeGridPanel") -> None:
        self._panel = panel

    def install(self) -> None:
        self._canvas.viewport().installEventFilter(self)

    def uninstall(self) -> None:
        self._canvas.viewport().removeEventFilter(self)
        self._canvas.viewport().setCursor(Qt.ArrowCursor)

    def set_active(self, active: bool) -> None:
        self._active = active
        if not active:
            self._dragging = False
            self._drag_handle = _HANDLE_NONE
            self._canvas.viewport().setCursor(Qt.ArrowCursor)

    def set_locked(self, locked: bool) -> None:
        self._locked = locked

    def set_ab_equal(self, equal: bool) -> None:
        self._ab_equal = equal

    # ── screen-space hit testing ───────────────────────────────────────────────

    def hit_handle_screen(self, view_pos) -> int:
        """
        Find which handle (if any) is under view_pos.

        Uses screen-pixel distance so hit radius is zoom-invariant.
        """
        if not self._item.grid().show_handles:
            return _HANDLE_NONE
        for hid, (sx, sy) in self._item.handle_positions_scene().items():
            screen = self._canvas.mapFromScene(QPointF(sx, sy))
            dist = math.hypot(view_pos.x() - screen.x(), view_pos.y() - screen.y())
            if dist <= HIT_RADIUS_PX:
                return hid
        return _HANDLE_NONE

    # ── event filter ──────────────────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if not self._active:
            return False

        t = event.type()

        if t == QEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
            vpos = event.pos()
            hid = self.hit_handle_screen(vpos)
            if hid != _HANDLE_NONE:
                self._drag_handle = hid
                self._drag_start_scene = self._canvas.mapToScene(vpos)
                self._drag_grid_start = self._item.grid()
                self._dragging = True
                event.accept()
                return True

        elif t == QEvent.Type.MouseMove:
            vpos = event.pos()
            if self._dragging:
                self._update_drag(self._canvas.mapToScene(vpos))
                event.accept()
                return True
            else:
                hid = self.hit_handle_screen(vpos)
                vp = self._canvas.viewport()
                if hid == _HANDLE_ORIGIN:
                    vp.setCursor(Qt.SizeAllCursor)
                elif hid != _HANDLE_NONE:
                    vp.setCursor(Qt.CrossCursor)
                else:
                    vp.setCursor(Qt.ArrowCursor)
                return False

        elif t == QEvent.Type.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self._dragging:
                self._dragging = False
                self._drag_handle = _HANDLE_NONE
                self._drag_start_scene = None
                self._drag_grid_start = None
                event.accept()
                return True

        return False

    def _update_drag(self, scene_pos: QPointF) -> None:
        if self._drag_start_scene is None or self._drag_grid_start is None:
            return

        dx = scene_pos.x() - self._drag_start_scene.x()
        dy = scene_pos.y() - self._drag_start_scene.y()
        g0 = self._drag_grid_start
        hid = self._drag_handle

        if hid == _HANDLE_ORIGIN:
            new_grid = g0.translate(dx, dy)

        elif hid == _HANDLE_A:
            new_a = (g0.a_px[0] + dx, g0.a_px[1] + dy)
            new_grid = g0.with_a_vector(new_a) if self._locked else replace(g0, a_px=new_a)

        elif hid == _HANDLE_B:
            new_b = (g0.b_px[0] + dx, g0.b_px[1] + dy)
            new_grid = g0.with_b_vector(new_b) if self._locked else replace(g0, b_px=new_b)

        elif hid == _HANDLE_ROT:
            ox, oy = g0.origin_px
            angle_start = math.degrees(math.atan2(
                self._drag_start_scene.y() - oy,
                self._drag_start_scene.x() - ox,
            ))
            angle_now = math.degrees(math.atan2(scene_pos.y() - oy, scene_pos.x() - ox))
            new_grid = g0.rotate(angle_now - angle_start)

        elif hid == _HANDLE_SCALE:
            ox, oy = g0.origin_px
            d0 = math.hypot(
                self._drag_start_scene.x() - ox,
                self._drag_start_scene.y() - oy,
            )
            d1 = math.hypot(scene_pos.x() - ox, scene_pos.y() - oy)
            if d0 > 1e-6:
                new_grid = g0.scale(d1 / d0)
            else:
                return
        else:
            return

        # Enforce |b| = |a| if a=b constraint is active
        if self._ab_equal:
            la = new_grid.a_length_px()
            lb = new_grid.b_length_px()
            if la > 1e-9 and lb > 1e-9 and abs(la - lb) > 1e-9:
                bx, by = new_grid.b_px
                new_grid = replace(new_grid, b_px=(bx * la / lb, by * la / lb))

        self._item.set_grid(new_grid)
        if self._panel is not None:
            self._panel.sync_from_model()
