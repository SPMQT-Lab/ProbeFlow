"""Matplotlib overlay for reciprocal-space lattice grids."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

import numpy as np

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
from .graphics_item import _compute_handle_positions

# ── FFT matplotlib overlay ────────────────────────────────────────────────────

class FFTLatticeOverlay:
    """
    Lattice grid overlay on a matplotlib FFT axes.

    Hit testing uses display (screen) pixels via ax.transData.transform(),
    so the hit radius is zoom-invariant.
    """

    def __init__(
        self,
        ax,
        canvas,
        qx_axis: np.ndarray,
        qy_axis: np.ndarray,
        image_w: int,
        image_h: int,
    ):
        self._ax = ax
        self._canvas = canvas
        self._qx = qx_axis
        self._qy = qy_axis
        self._image_w = image_w
        self._image_h = image_h
        self._artists: list = []
        self._grid: Optional[LatticeGrid] = None
        self._cells: int = 12
        self._line_width_px: float = 1.5
        self._locked: bool = True
        self._dragging: bool = False
        self._drag_handle: int = _HANDLE_NONE
        self._drag_start_q: Optional[tuple[float, float]] = None
        self._drag_grid_start: Optional[LatticeGrid] = None
        self._last_drag_display_xy: Optional[tuple[float, float]] = None
        self._on_change_cb = None
        self._on_drag_state_cb = None

        self._cid_press   = canvas.mpl_connect("button_press_event",   self._on_press)
        self._cid_release = canvas.mpl_connect("button_release_event", self._on_release)
        self._cid_motion  = canvas.mpl_connect("motion_notify_event",  self._on_motion)

    def set_grid(self, grid: LatticeGrid) -> None:
        self._grid = grid
        self.redraw()

    def grid(self) -> Optional[LatticeGrid]:
        return self._grid

    def set_cells(self, n: int) -> None:
        self._cells = max(1, int(n))
        self.redraw()

    def set_line_width(self, px: float) -> None:
        self._line_width_px = max(0.25, float(px))
        self.redraw()

    def set_locked(self, locked: bool) -> None:
        self._locked = locked

    def set_on_change(self, cb) -> None:
        self._on_change_cb = cb

    def set_drag_state_callback(self, cb) -> None:
        self._on_drag_state_cb = cb

    def is_dragging(self) -> bool:
        return self._dragging

    def disconnect(self) -> None:
        self._set_dragging(False)
        for cid in (self._cid_press, self._cid_release, self._cid_motion):
            try:
                self._canvas.mpl_disconnect(cid)
            except Exception:
                pass

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _px_to_q(self, ix: float, iy: float) -> tuple[float, float]:
        Nx, Ny = self._image_w, self._image_h
        dqx = (float(self._qx[-1]) - float(self._qx[0])) / max(1, Nx - 1)
        dqy = (float(self._qy[-1]) - float(self._qy[0])) / max(1, Ny - 1)
        return (float(self._qx[0]) + ix * dqx, float(self._qy[0]) + iy * dqy)

    def _q_to_px(self, qx: float, qy: float) -> tuple[float, float]:
        Nx, Ny = self._image_w, self._image_h
        dqx = (float(self._qx[-1]) - float(self._qx[0])) / max(1, Nx - 1)
        dqy = (float(self._qy[-1]) - float(self._qy[0])) / max(1, Ny - 1)
        ix = (qx - float(self._qx[0])) / dqx if dqx != 0 else 0.0
        iy = (qy - float(self._qy[0])) / dqy if dqy != 0 else 0.0
        return ix, iy

    def _handle_positions_q(self) -> dict[int, tuple[float, float]]:
        """Return handle positions in q-space."""
        if self._grid is None:
            return {}
        handles_px = _compute_handle_positions(self._grid)
        return {hid: self._px_to_q(px, py) for hid, (px, py) in handles_px.items()}

    def _hit_handle_display(self, mouse_x: float, mouse_y: float) -> int:
        """
        Hit test using display (screen) pixels.

        event.x / event.y are matplotlib display coordinates (pixels from bottom).
        """
        if self._grid is None:
            return _HANDLE_NONE
        for hid, (qx, qy) in self._handle_positions_q().items():
            try:
                disp = self._ax.transData.transform((qx, qy))
                dist = math.hypot(mouse_x - disp[0], mouse_y - disp[1])
                if dist <= HIT_RADIUS_PX:
                    return hid
            except Exception:
                pass
        return _HANDLE_NONE

    def hit_test_event(self, event) -> bool:
        if event.inaxes is not self._ax or self._grid is None:
            return False
        return self._hit_handle_display(event.x, event.y) != _HANDLE_NONE

    def _set_dragging(self, dragging: bool) -> None:
        dragging = bool(dragging)
        if self._dragging == dragging:
            return
        self._dragging = dragging
        if self._on_drag_state_cb is not None:
            self._on_drag_state_cb(dragging)

    # ── matplotlib events ─────────────────────────────────────────────────────

    def _on_press(self, event) -> None:
        if (
            event.inaxes is not self._ax
            or self._grid is None
            or event.xdata is None
            or event.ydata is None
        ):
            return
        hid = self._hit_handle_display(event.x, event.y)
        if hid == _HANDLE_NONE:
            return
        self._drag_handle = hid
        self._drag_start_q = (event.xdata, event.ydata)
        self._drag_grid_start = self._grid
        self._last_drag_display_xy = (float(event.x), float(event.y))
        self._set_dragging(True)

    def _on_release(self, event) -> None:
        self._set_dragging(False)
        self._drag_handle = _HANDLE_NONE
        self._drag_start_q = None
        self._drag_grid_start = None
        self._last_drag_display_xy = None

    def _on_motion(self, event) -> None:
        if (not self._dragging
                or event.inaxes is not self._ax
                or event.xdata is None
                or event.ydata is None
                or self._drag_start_q is None
                or self._drag_grid_start is None):
            return
        if self._last_drag_display_xy is not None:
            dx_display = float(event.x) - self._last_drag_display_xy[0]
            dy_display = float(event.y) - self._last_drag_display_xy[1]
            if math.hypot(dx_display, dy_display) < 1.5:
                return
        self._last_drag_display_xy = (float(event.x), float(event.y))

        dqx = event.xdata - self._drag_start_q[0]
        dqy = event.ydata - self._drag_start_q[1]

        Nx, Ny = self._image_w, self._image_h
        q_range_x = float(self._qx[-1]) - float(self._qx[0])
        q_range_y = float(self._qy[-1]) - float(self._qy[0])
        dpx = dqx * Nx / q_range_x if q_range_x != 0 else 0.0
        dpy = dqy * Ny / q_range_y if q_range_y != 0 else 0.0

        g0 = self._drag_grid_start
        hid = self._drag_handle

        if hid == _HANDLE_ORIGIN:
            self._grid = g0.translate(dpx, dpy)
        elif hid == _HANDLE_A:
            new_a = (g0.a_px[0] + dpx, g0.a_px[1] + dpy)
            self._grid = g0.with_a_vector(new_a) if self._locked else replace(g0, a_px=new_a)
        elif hid == _HANDLE_B:
            new_b = (g0.b_px[0] + dpx, g0.b_px[1] + dpy)
            self._grid = g0.with_b_vector(new_b) if self._locked else replace(g0, b_px=new_b)
        elif hid == _HANDLE_ROT:
            ox_px, oy_px = g0.origin_px
            ox_q, oy_q = self._px_to_q(ox_px, oy_px)
            angle_start = math.degrees(math.atan2(
                self._drag_start_q[1] - oy_q,
                self._drag_start_q[0] - ox_q,
            ))
            angle_now = math.degrees(math.atan2(event.ydata - oy_q, event.xdata - ox_q))
            self._grid = g0.rotate(angle_now - angle_start)
        elif hid == _HANDLE_SCALE:
            ox_px, oy_px = g0.origin_px
            ox_q, oy_q = self._px_to_q(ox_px, oy_px)
            d0 = math.hypot(self._drag_start_q[0] - ox_q, self._drag_start_q[1] - oy_q)
            d1 = math.hypot(event.xdata - ox_q, event.ydata - oy_q)
            if d0 > 1e-12:
                self._grid = g0.scale(d1 / d0)

        self.redraw()
        if self._on_change_cb is not None:
            self._on_change_cb(self._grid)

    # ── drawing ───────────────────────────────────────────────────────────────

    def redraw(self) -> None:
        for art in self._artists:
            try:
                art.remove()
            except Exception:
                pass
        self._artists.clear()

        if self._grid is None or not self._grid.visible:
            self._canvas.draw_idle()
            return

        grid = self._grid
        ox_px, oy_px = grid.origin_px
        ax_px, ay_px = grid.a_px
        bx_px, by_px = grid.b_px
        c = self._cells

        def p2q(ix, iy):
            return self._px_to_q(ix, iy)

        # Grid lines — parallel to b
        for n in range(-c, c + 1):
            s_px = (ox_px + n * ax_px - c * bx_px, oy_px + n * ay_px - c * by_px)
            e_px = (ox_px + n * ax_px + c * bx_px, oy_px + n * ay_px + c * by_px)
            sq, eq = p2q(*s_px), p2q(*e_px)
            art, = self._ax.plot(
                [sq[0], eq[0]], [sq[1], eq[1]],
                color="#89b4fa", lw=self._line_width_px, alpha=0.7, zorder=5,
            )
            self._artists.append(art)

        # Grid lines — parallel to a
        for m in range(-c, c + 1):
            s_px = (ox_px - c * ax_px + m * bx_px, oy_px - c * ay_px + m * by_px)
            e_px = (ox_px + c * ax_px + m * bx_px, oy_px + c * ay_px + m * by_px)
            sq, eq = p2q(*s_px), p2q(*e_px)
            art, = self._ax.plot(
                [sq[0], eq[0]], [sq[1], eq[1]],
                color="#89b4fa", lw=self._line_width_px, alpha=0.7, zorder=5,
            )
            self._artists.append(art)

        # Basis vectors
        oq = p2q(ox_px, oy_px)
        aq = p2q(ox_px + ax_px, oy_px + ay_px)
        bq = p2q(ox_px + bx_px, oy_px + by_px)

        art, = self._ax.plot([oq[0], aq[0]], [oq[1], aq[1]],
                             color="#a6e3a1", lw=2.0, zorder=6)
        self._artists.append(art)
        art, = self._ax.plot([oq[0], bq[0]], [oq[1], bq[1]],
                             color="#fab387", lw=2.0, zorder=6)
        self._artists.append(art)

        # Handles + labels
        handle_cols = {
            _HANDLE_ORIGIN: "#f38ba8",
            _HANDLE_A:      "#a6e3a1",
            _HANDLE_B:      "#fab387",
            _HANDLE_ROT:    "#cba6f7",
            _HANDLE_SCALE:  "#f9e2af",
        }
        handle_labels = {
            _HANDLE_A:      "g1" if grid.space == "reciprocal" else "a",
            _HANDLE_B:      "g2" if grid.space == "reciprocal" else "b",
            _HANDLE_ORIGIN: "O",
        }
        handles_q = self._handle_positions_q()
        for hid, (hqx, hqy) in handles_q.items():
            if grid.show_handles:
                art, = self._ax.plot(
                    hqx, hqy, "o",
                    color=handle_cols[hid], markersize=9,
                    markeredgewidth=1.0, zorder=8,
                )
                self._artists.append(art)
            if grid.show_labels and hid in handle_labels:
                art = self._ax.text(
                    hqx, hqy, f" {handle_labels[hid]}",
                    color="#cdd6f4", fontsize=8, va="center", zorder=9,
                )
                self._artists.append(art)

        self._canvas.draw_idle()
