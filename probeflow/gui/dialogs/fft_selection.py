"""Interactive circle / ellipse Fourier selections on the FFT viewer axes.

Mirrors the drag/hit-test pattern of
:class:`probeflow.gui.lattice_grid.fft_overlay.FFTLatticeOverlay`: geometry is
held in q-space (nm⁻¹), drawn as matplotlib patches on the FFT axes, and
hit-tested in display pixels.  Each selection's conjugate partner (point
reflection through DC) is drawn dashed — it is what makes the inverse-FFT
reconstruction real-valued.

The overlay is GUI-state only; mask construction and the inverse transform live
in :mod:`probeflow.processing.inverse_fft`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from matplotlib.patches import Ellipse

_HIT_RADIUS_PX = 10.0
_HANDLE_NONE = 0
_HANDLE_CENTRE = 1
_HANDLE_RX = 2
_HANDLE_RY = 3


@dataclass
class FourierSelection:
    """A circle/ellipse selection, centre + semi-axes in q-space (nm⁻¹)."""
    kind: str            # "circle" | "ellipse"
    cx_q: float
    cy_q: float
    rx_q: float
    ry_q: float


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

    def _default_radius_q(self) -> float:
        # ~6 % of the qx half-range, a comfortable starting spot size.
        half = max(abs(float(self._qx[0])), abs(float(self._qx[-1])))
        return max(half * 0.06, 3.0 * abs(self._dq()[0]))

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, kind: str = "circle") -> None:
        r = self._default_radius_q()
        # Place a little off-DC so it is visible and not on the central spot.
        self._sels.append(FourierSelection(kind, cx_q=3 * r, cy_q=0.0, rx_q=r, ry_q=r))
        self._selected = len(self._sels) - 1
        self._notify()

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

    def to_fft_ellipses(self) -> list[dict]:
        """Selections as FFT-pixel ellipse dicts {dx,dy,rx,ry,angle_deg} (+ q for
        provenance), for mask building / the inverse_fft_filter op."""
        dqx, dqy = self._dq()
        out: list[dict] = []
        for s in self._sels:
            out.append({
                "dx": s.cx_q / dqx if dqx else 0.0,
                "dy": s.cy_q / dqy if dqy else 0.0,
                "rx": max(s.rx_q / abs(dqx), 0.5) if dqx else 1.0,
                "ry": max(s.ry_q / abs(dqy), 0.5) if dqy else 1.0,
                "angle_deg": 0.0,
                "cx_q": s.cx_q, "cy_q": s.cy_q, "rx_q": s.rx_q, "ry_q": s.ry_q,
                "kind": s.kind,
            })
        return out

    # ── drawing ────────────────────────────────────────────────────────────────

    def draw(self) -> None:
        """(Re)draw all selections + conjugates + handles for the selected one."""
        self._artists = []   # axes were cla()'d by the host redraw
        for i, s in enumerate(self._sels):
            chosen = (i == self._selected)
            for cx, cy, dashed in ((s.cx_q, s.cy_q, False), (-s.cx_q, -s.cy_q, True)):
                e = Ellipse((cx, cy), 2 * s.rx_q, 2 * s.ry_q, angle=0.0,
                            fill=False, edgecolor="#89dceb" if chosen else "#89b4fa",
                            lw=1.6 if chosen else 1.2,
                            ls="--" if dashed else "-", zorder=11)
                self._ax.add_patch(e)
                self._artists.append(e)
            if chosen:
                for hx, hy in ((s.cx_q, s.cy_q), (s.cx_q + s.rx_q, s.cy_q),
                               (s.cx_q, s.cy_q + s.ry_q)):
                    art, = self._ax.plot(hx, hy, "s", color="#f9e2af",
                                         markersize=7, markeredgewidth=1.0, zorder=12)
                    self._artists.append(art)

    # ── hit-testing + drag ──────────────────────────────────────────────────────

    def _handle_at(self, event) -> tuple[int, int]:
        """Return (selection_index, handle_id) under the cursor, or (-1, NONE)."""
        # Prefer handles of the currently-selected item, then any body.
        order = ([self._selected] if 0 <= self._selected < len(self._sels) else []) + \
                [i for i in range(len(self._sels)) if i != self._selected]
        for i in order:
            s = self._sels[i]
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
            # Body hit (inside the ellipse, in q-space).
            if event.xdata is not None and event.ydata is not None:
                dx = (event.xdata - s.cx_q) / max(s.rx_q, 1e-9)
                dy = (event.ydata - s.cy_q) / max(s.ry_q, 1e-9)
                if dx * dx + dy * dy <= 1.0:
                    return i, _HANDLE_CENTRE
        return -1, _HANDLE_NONE

    def on_press(self, event) -> bool:
        """Begin a drag if a selection/handle is hit. Returns True if consumed."""
        if event.inaxes is not self._ax:
            return False
        idx, hid = self._handle_at(event)
        if idx < 0:
            return False
        self._selected = idx
        self._drag_idx = idx
        self._drag_handle = hid
        self._notify()
        return True

    def on_motion(self, event) -> bool:
        if self._drag_handle == _HANDLE_NONE or self._drag_idx < 0:
            return False
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            try:
                xdata, ydata = self._ax.transData.inverted().transform(
                    (float(event.x), float(event.y)))
            except Exception:
                return True
        s = self._sels[self._drag_idx]
        if self._drag_handle == _HANDLE_CENTRE:
            s.cx_q, s.cy_q = float(xdata), float(ydata)
        elif self._drag_handle == _HANDLE_RX:
            s.rx_q = max(abs(float(xdata) - s.cx_q), abs(self._dq()[0]))
            if s.kind == "circle":
                s.ry_q = s.rx_q
        elif self._drag_handle == _HANDLE_RY:
            s.ry_q = max(abs(float(ydata) - s.cy_q), abs(self._dq()[1]))
            if s.kind == "circle":
                s.rx_q = s.ry_q
        self._notify()
        return True

    def on_release(self, event) -> None:
        self._drag_handle = _HANDLE_NONE
        self._drag_idx = -1

    def is_dragging(self) -> bool:
        return self._drag_handle != _HANDLE_NONE

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()
