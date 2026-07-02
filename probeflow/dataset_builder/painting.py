"""Mask painting primitives for Dataset Builder."""

from __future__ import annotations

import numpy as np


def paint_mask(
    mask: np.ndarray,
    *,
    x: int,
    y: int,
    radius: int,
    value: bool,
) -> tuple[np.ndarray, bool]:
    """Return a copy of *mask* with one circular brush stamp applied."""

    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"paint_mask expects a 2-D mask, got shape {arr.shape}")
    h, w = arr.shape
    if h <= 0 or w <= 0:
        return arr.copy(), False

    cx = int(round(x))
    cy = int(round(y))
    r = max(1, int(round(radius)))
    if cx < -r or cy < -r or cx >= w + r or cy >= h + r:
        return arr.copy(), False

    x0 = max(0, cx - r)
    x1 = min(w, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(h, cy + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r

    out = arr.copy()
    before = out[y0:y1, x0:x1].copy()
    out[y0:y1, x0:x1][disk] = bool(value)
    changed = not np.array_equal(before, out[y0:y1, x0:x1])
    return out, changed
