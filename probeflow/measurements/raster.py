"""Point-rasterization primitives used by feature & FFT helpers.

Review arch-backend #10 (2026-05-28): two near-identical disk-dilation
loops lived in ``analysis.feature_finder.feature_points_to_image`` and
``measurements.fft_points.points_to_mask``.  Both painted a circular
footprint of a given radius around each integer-rounded point.  This
module owns the inner loop so future bug fixes (anti-aliased edges,
sub-pixel centres, etc.) need to be applied in only one place.
"""

from __future__ import annotations

import numpy as np


__all__ = ["paint_point", "paint_disk", "paint_square"]


def paint_disk(
    arr: np.ndarray,
    cy: int,
    cx: int,
    radius_px: float,
    value: float | bool = 1.0,
) -> None:
    """Paint a filled disk of ``radius_px`` centred at ``(cy, cx)`` into ``arr``.

    Pixels with ``(row-cy)**2 + (col-cx)**2 <= radius_px**2`` are set to
    ``value``.  The disk is clipped to the array bounds.  When
    ``radius_px <= 0`` only the centre pixel (if in bounds) is painted.

    ``arr`` is modified in place.
    """
    ny, nx = arr.shape[:2]
    if radius_px <= 0:
        if 0 <= cy < ny and 0 <= cx < nx:
            arr[cy, cx] = value
        return
    r_int = int(radius_px) + 1
    r2 = float(radius_px) ** 2
    for row in range(max(0, cy - r_int), min(ny, cy + r_int + 1)):
        for col in range(max(0, cx - r_int), min(nx, cx + r_int + 1)):
            if (row - cy) ** 2 + (col - cx) ** 2 <= r2:
                arr[row, col] = value


def paint_square(
    arr: np.ndarray,
    cy: int,
    cx: int,
    radius_px: int,
    value: float | bool = 1.0,
) -> None:
    """Paint a filled square of half-side ``radius_px`` into ``arr``.

    All pixels in ``[cy-r, cy+r] x [cx-r, cx+r]`` (clipped to the array
    bounds) are set to ``value``.  When ``radius_px <= 0`` only the
    centre pixel (if in bounds) is painted.

    ``arr`` is modified in place.
    """
    ny, nx = arr.shape[:2]
    r = int(max(0, radius_px))
    if r == 0:
        if 0 <= cy < ny and 0 <= cx < nx:
            arr[cy, cx] = value
        return
    arr[
        max(0, cy - r):min(ny, cy + r + 1),
        max(0, cx - r):min(nx, cx + r + 1),
    ] = value


def paint_point(
    arr: np.ndarray,
    x_px: float,
    y_px: float,
    *,
    radius_px: float = 0.0,
    shape_mode: str = "disk",
    value: float | bool = 1.0,
) -> None:
    """Paint one rasterized point into ``arr`` at sub-pixel ``(x_px, y_px)``.

    ``shape_mode`` is ``"disk"`` (default, circular footprint) or
    ``"square"`` (full square footprint).  Coordinates are
    integer-rounded before rasterization.  Out-of-bounds points are
    silently skipped (matches the prior helpers' contract).
    """
    mode = (shape_mode or "disk").strip().lower()
    if mode not in {"disk", "square"}:
        raise ValueError("shape_mode must be 'disk' or 'square'")
    cy = int(round(y_px))
    cx = int(round(x_px))
    if mode == "square":
        paint_square(arr, cy, cx, int(max(0, radius_px)), value)
    else:
        paint_disk(arr, cy, cx, float(radius_px), value)
