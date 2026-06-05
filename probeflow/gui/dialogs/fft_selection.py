"""Interactive Fourier selections on the FFT viewer axes.

Three selection kinds, all created by **drawing** on the FFT (drag from one
corner to the opposite, as ROIs are drawn elsewhere in the app):

* ``ellipse`` — drag a bounding box; hold **Shift** for a circle.
* ``rect`` — drag a bounding box; hold **Shift** for a square.
* ``paint`` — freehand brush; the union of circular stamps forms the region.

Geometry is held in q-space (nm⁻¹), drawn as matplotlib patches / an RGBA
``AxesImage`` on the FFT axes, and hit-tested in display pixels.  Each
selection's conjugate partner (point reflection through DC) is shown — dashed
for shapes, mirrored for paint — it is what makes the inverse-FFT reconstruction
real-valued.

The overlay is GUI-state only; mask construction and the inverse transform live
in :mod:`probeflow.processing.inverse_fft`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from matplotlib.patches import Ellipse, Rectangle

_HIT_RADIUS_PX = 10.0
_HANDLE_NONE = 0
_HANDLE_CENTRE = 1
_HANDLE_RX = 2
_HANDLE_RY = 3
_MIN_DRAW_PX = 3.0   # drags smaller than this (an accidental click) are discarded

_DEFAULT_PAINT_COLOR = (137, 220, 235)   # #89dceb — matches the selection cyan


@dataclass
class FourierSelection:
    """A Fourier selection in q-space (nm⁻¹).

    ``rx_q``/``ry_q`` are the semi-axes for an ellipse or the half-extents for a
    rectangle.  Paint selections instead use ``stamps_q`` (circular-brush centres)
    + ``radius_q`` (brush radius) + ``color`` (RGB 0–255).
    """
    kind: str            # "ellipse" | "rect" | "paint"
    cx_q: float = 0.0
    cy_q: float = 0.0
    rx_q: float = 0.0
    ry_q: float = 0.0
    stamps_q: list = field(default_factory=list)
    radius_q: float = 0.0
    color: tuple = _DEFAULT_PAINT_COLOR


class FFTSelectionOverlay:
    """Manage a list of Fourier selections drawn on a matplotlib FFT axes."""

    def __init__(self, ax, qx, qy, image_shape, on_change=None):
        self._ax = ax
        self._qx = np.asarray(qx, dtype=np.float64)
        self._qy = np.asarray(qy, dtype=np.float64)
        self._shape = (int(image_shape[0]), int(image_shape[1]))
        self._on_change = on_change
        self._sels: list[FourierSelection] = []
        self._selected: int = -1
        self._artists: list = []
        self._drag_handle = _HANDLE_NONE
        self._drag_idx = -1
        self._drag_last = None        # last cursor (x_q, y_q) for body/paint drags
        # Draw-to-create state.
        self._tool: str | None = None        # None | "ellipse" | "rect" | "paint"
        self._draw_anchor = None             # (x_q, y_q) anchor for an in-progress shape
        self._drawing = False
        self._painting = False
        # Paint controls.
        self._brush_px = 8.0
        self._paint_color = _DEFAULT_PAINT_COLOR

    # ── geometry helpers ──────────────────────────────────────────────────────

    def _dq(self) -> tuple[float, float]:
        nx, ny = self._shape[1], self._shape[0]
        dqx = (float(self._qx[-1]) - float(self._qx[0])) / max(1, nx - 1)
        dqy = (float(self._qy[-1]) - float(self._qy[0])) / max(1, ny - 1)
        return dqx, dqy

    def set_qaxes(self, qx, qy, image_shape) -> None:
        self._qx = np.asarray(qx, dtype=np.float64)
        self._qy = np.asarray(qy, dtype=np.float64)
        self._shape = (int(image_shape[0]), int(image_shape[1]))

    def _brush_radius_q(self) -> float:
        return self._brush_px * abs(self._dq()[0])

    # ── tool / paint configuration ────────────────────────────────────────────

    def set_tool(self, tool: str | None) -> None:
        self._tool = tool if tool in ("ellipse", "rect", "paint") else None

    def tool(self) -> str | None:
        return self._tool

    def set_brush_radius_px(self, px: float) -> None:
        self._brush_px = max(0.5, float(px))

    def set_paint_color(self, rgb) -> None:
        self._paint_color = tuple(int(c) for c in rgb)
        if (0 <= self._selected < len(self._sels)
                and self._sels[self._selected].kind == "paint"):
            self._sels[self._selected].color = self._paint_color
            self._notify()

    # ── public API ────────────────────────────────────────────────────────────

    def delete_selected(self) -> None:
        if 0 <= self._selected < len(self._sels):
            self._sels.pop(self._selected)
            self._selected = min(self._selected, len(self._sels) - 1)
            self._notify()

    def clear(self) -> None:
        if self._sels:
            self._sels = []
            self._selected = -1
            self._notify()

    def count(self) -> int:
        return len(self._sels)

    def to_regions(self) -> list[dict]:
        """Selections as FFT-pixel region dicts (+ q-space provenance), for mask
        building and the ``inverse_fft_filter`` op.  Each dict carries a ``kind``
        of ``"ellipse"``, ``"rect"`` or ``"paint"``."""
        dqx, dqy = self._dq()
        out: list[dict] = []
        for s in self._sels:
            if s.kind == "paint":
                stamps = [[x / dqx if dqx else 0.0, y / dqy if dqy else 0.0]
                          for (x, y) in s.stamps_q]
                out.append({
                    "kind": "paint",
                    "stamps": stamps,
                    "radius": max(s.radius_q / abs(dqx), 0.5) if dqx else 1.0,
                    "stamps_q": [list(p) for p in s.stamps_q],
                    "radius_q": s.radius_q,
                    "color": list(s.color),
                })
                continue
            d = {
                "kind": s.kind,
                "dx": s.cx_q / dqx if dqx else 0.0,
                "dy": s.cy_q / dqy if dqy else 0.0,
                "angle_deg": 0.0,
                "cx_q": s.cx_q, "cy_q": s.cy_q, "rx_q": s.rx_q, "ry_q": s.ry_q,
            }
            if s.kind == "rect":
                d["half_w"] = max(s.rx_q / abs(dqx), 0.5) if dqx else 1.0
                d["half_h"] = max(s.ry_q / abs(dqy), 0.5) if dqy else 1.0
            else:  # ellipse
                d["rx"] = max(s.rx_q / abs(dqx), 0.5) if dqx else 1.0
                d["ry"] = max(s.ry_q / abs(dqy), 0.5) if dqy else 1.0
            out.append(d)
        return out

    # ── drawing ────────────────────────────────────────────────────────────────

    def draw(self) -> None:
        """(Re)draw all selections + conjugates + handles for the selected one."""
        self._artists = []   # axes were cla()'d by the host redraw
        for i, s in enumerate(self._sels):
            chosen = (i == self._selected)
            if s.kind == "paint":
                self._draw_paint(s, chosen)
                continue
            edge = "#89dceb" if chosen else "#89b4fa"
            lw = 1.6 if chosen else 1.2
            for cx, cy, dashed in ((s.cx_q, s.cy_q, False), (-s.cx_q, -s.cy_q, True)):
                if s.kind == "rect":
                    patch = Rectangle((cx - s.rx_q, cy - s.ry_q), 2 * s.rx_q, 2 * s.ry_q,
                                      fill=False, edgecolor=edge, lw=lw,
                                      ls="--" if dashed else "-", zorder=11)
                else:
                    patch = Ellipse((cx, cy), 2 * s.rx_q, 2 * s.ry_q, angle=0.0,
                                    fill=False, edgecolor=edge, lw=lw,
                                    ls="--" if dashed else "-", zorder=11)
                self._ax.add_patch(patch)
                self._artists.append(patch)
            if chosen:
                for hx, hy in ((s.cx_q, s.cy_q), (s.cx_q + s.rx_q, s.cy_q),
                               (s.cx_q, s.cy_q + s.ry_q)):
                    art, = self._ax.plot(hx, hy, "s", color="#f9e2af",
                                         markersize=7, markeredgewidth=1.0, zorder=12)
                    self._artists.append(art)

    def _draw_paint(self, s: FourierSelection, chosen: bool) -> None:
        rgba = self._rasterize_paint(s, chosen)
        if rgba is None:
            return
        extent = [float(self._qx[0]), float(self._qx[-1]),
                  float(self._qy[-1]), float(self._qy[0])]
        # imshow can autoscale the axes; keep the current view (zoom/pan) fixed.
        xlim, ylim = self._ax.get_xlim(), self._ax.get_ylim()
        im = self._ax.imshow(rgba, origin="upper", extent=extent, aspect="auto",
                             interpolation="nearest", zorder=10.5 if chosen else 10)
        self._ax.set_xlim(*xlim)
        self._ax.set_ylim(*ylim)
        self._artists.append(im)

    def _rasterize_paint(self, s: FourierSelection, chosen: bool):
        """Stamp the painted discs (and their conjugates) into an RGBA overlay
        matching the FFT pixel grid, so it aligns with the applied mask."""
        if not s.stamps_q:
            return None
        ny, nx = self._shape
        dqx, dqy = self._dq()
        if not dqx or not dqy:
            return None
        cx0, cy0 = nx // 2, ny // 2
        rpx = max(s.radius_q / abs(dqx), 0.5)
        rpy = max(s.radius_q / abs(dqy), 0.5)
        mask = np.zeros((ny, nx), dtype=bool)
        for (x, y) in s.stamps_q:
            dx, dy = x / dqx, y / dqy
            for sx, sy in ((dx, dy), (-dx, -dy)):   # show the conjugate too
                ix, iy = int(round(cx0 + sx)), int(round(cy0 + sy))
                x0, x1 = max(0, ix - int(rpx) - 1), min(nx, ix + int(rpx) + 2)
                y0, y1 = max(0, iy - int(rpy) - 1), min(ny, iy + int(rpy) + 2)
                if x0 >= x1 or y0 >= y1:
                    continue
                ys, xs = np.ogrid[y0:y1, x0:x1]
                disc = ((xs - ix) / rpx) ** 2 + ((ys - iy) / rpy) ** 2 <= 1.0
                mask[y0:y1, x0:x1] |= disc
        if not mask.any():
            return None
        rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
        rgba[mask] = (*[int(c) for c in s.color], 200 if chosen else 130)
        return rgba

    # ── hit-testing + drag ──────────────────────────────────────────────────────

    def _handle_at(self, event) -> tuple[int, int]:
        """Return (selection_index, handle_id) under the cursor, or (-1, NONE)."""
        order = ([self._selected] if 0 <= self._selected < len(self._sels) else []) + \
                [i for i in range(len(self._sels)) if i != self._selected]
        for i in order:
            s = self._sels[i]
            if s.kind == "paint":
                if (event.xdata is not None and event.ydata is not None
                        and any(math.hypot(event.xdata - x, event.ydata - y) <= s.radius_q
                                for (x, y) in s.stamps_q)):
                    return i, _HANDLE_CENTRE
                continue
            handles = {
                _HANDLE_RX: (s.cx_q + s.rx_q, s.cy_q),
                _HANDLE_RY: (s.cx_q, s.cy_q + s.ry_q),
                _HANDLE_CENTRE: (s.cx_q, s.cy_q),
            }
            for hid, (qx, qy) in handles.items():
                try:
                    disp = self._ax.transData.transform((qx, qy))
                    if math.hypot(event.x - disp[0], event.y - disp[1]) <= _HIT_RADIUS_PX:
                        return i, hid
                except Exception:
                    pass
            # Body hit (inside the shape, in q-space).
            if event.xdata is not None and event.ydata is not None:
                if s.kind == "rect":
                    if (abs(event.xdata - s.cx_q) <= s.rx_q
                            and abs(event.ydata - s.cy_q) <= s.ry_q):
                        return i, _HANDLE_CENTRE
                else:
                    dx = (event.xdata - s.cx_q) / max(s.rx_q, 1e-9)
                    dy = (event.ydata - s.cy_q) / max(s.ry_q, 1e-9)
                    if dx * dx + dy * dy <= 1.0:
                        return i, _HANDLE_CENTRE
        return -1, _HANDLE_NONE

    def _shift_held(self, event) -> bool:
        ge = getattr(event, "guiEvent", None)
        if ge is not None:
            try:
                from PySide6.QtCore import Qt
                return bool(ge.modifiers() & Qt.ShiftModifier)
            except Exception:
                pass
        return "shift" in (getattr(event, "key", None) or "")

    def on_press(self, event) -> bool:
        """Begin editing a hit selection, or draw a new one if a tool is active.
        Returns True if the event was consumed."""
        if event.inaxes is not self._ax:
            return False
        idx, hid = self._handle_at(event)
        if idx >= 0:
            self._selected = idx
            self._drag_idx = idx
            self._drag_handle = hid
            self._drag_last = (event.xdata, event.ydata)
            self._notify()
            return True
        if self._tool is not None and event.xdata is not None and event.ydata is not None:
            return self._begin_draw(float(event.xdata), float(event.ydata))
        return False

    def _begin_draw(self, x: float, y: float) -> bool:
        if self._tool == "paint":
            sel = FourierSelection("paint", radius_q=self._brush_radius_q(),
                                   color=self._paint_color)
            sel.stamps_q.append((x, y))
            self._sels.append(sel)
            self._selected = self._drag_idx = len(self._sels) - 1
            self._painting = True
        else:
            sel = FourierSelection(self._tool, cx_q=x, cy_q=y)
            self._sels.append(sel)
            self._selected = self._drag_idx = len(self._sels) - 1
            self._draw_anchor = (x, y)
            self._drawing = True
        self._notify()
        return True

    def on_motion(self, event) -> bool:
        if not (self._drawing or self._painting or self._drag_handle != _HANDLE_NONE):
            return False
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            try:
                xdata, ydata = self._ax.transData.inverted().transform(
                    (float(event.x), float(event.y)))
            except Exception:
                return True
        xdata, ydata = float(xdata), float(ydata)

        if self._painting and 0 <= self._drag_idx < len(self._sels):
            sel = self._sels[self._drag_idx]
            step = max(sel.radius_q * 0.4, abs(self._dq()[0]))
            if (not sel.stamps_q
                    or math.hypot(xdata - sel.stamps_q[-1][0],
                                  ydata - sel.stamps_q[-1][1]) >= step):
                sel.stamps_q.append((xdata, ydata))
                self._notify()
            return True

        if self._drawing and 0 <= self._drag_idx < len(self._sels):
            self._update_draw(self._sels[self._drag_idx], xdata, ydata,
                              self._shift_held(event))
            self._notify()
            return True

        s = self._sels[self._drag_idx]
        if s.kind == "paint":   # body drag: translate every stamp
            if self._drag_last is not None:
                lx, ly = self._drag_last
                s.stamps_q = [(px + (xdata - lx), py + (ydata - ly))
                              for (px, py) in s.stamps_q]
            self._drag_last = (xdata, ydata)
        elif self._drag_handle == _HANDLE_CENTRE:
            s.cx_q, s.cy_q = xdata, ydata
        elif self._drag_handle == _HANDLE_RX:
            s.rx_q = max(abs(xdata - s.cx_q), abs(self._dq()[0]))
            if self._shift_held(event):
                s.ry_q = s.rx_q
        elif self._drag_handle == _HANDLE_RY:
            s.ry_q = max(abs(ydata - s.cy_q), abs(self._dq()[1]))
            if self._shift_held(event):
                s.rx_q = s.ry_q
        self._notify()
        return True

    def _update_draw(self, s: FourierSelection, x: float, y: float, regular: bool) -> None:
        ax0, ay0 = self._draw_anchor
        sx, sy = x - ax0, y - ay0
        if regular:                       # Shift → circle / square
            m = max(abs(sx), abs(sy))
            sx = math.copysign(m, sx) if sx else m
            sy = math.copysign(m, sy) if sy else m
        s.cx_q = ax0 + sx / 2.0
        s.cy_q = ay0 + sy / 2.0
        # No size floor here: a zero drag stays zero so on_release discards an
        # accidental click. to_regions()/the mask builder floor the final size.
        s.rx_q = abs(sx) / 2.0
        s.ry_q = abs(sy) / 2.0

    def on_release(self, event) -> None:
        finishing_draw = self._drawing
        idx = self._drag_idx
        self._drag_handle = _HANDLE_NONE
        self._drag_idx = -1
        self._drag_last = None
        self._draw_anchor = None
        self._drawing = False
        self._painting = False
        # Discard a shape that was only clicked, not dragged.
        if finishing_draw and 0 <= idx < len(self._sels):
            s = self._sels[idx]
            try:
                p0 = self._ax.transData.transform((s.cx_q - s.rx_q, s.cy_q - s.ry_q))
                p1 = self._ax.transData.transform((s.cx_q + s.rx_q, s.cy_q + s.ry_q))
                if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < _MIN_DRAW_PX:
                    self._sels.pop(idx)
                    self._selected = min(self._selected, len(self._sels) - 1)
                    self._notify()
            except Exception:
                pass

    def is_dragging(self) -> bool:
        return (self._drag_handle != _HANDLE_NONE or self._drawing or self._painting)

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()
