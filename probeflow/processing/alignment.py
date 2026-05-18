"""Row alignment and facet levelling operations."""

from __future__ import annotations

import math

import numpy as np

from ._image_utils import _finite_median
from .background import subtract_background


# ═════════════════════════════════════════════════════════════════════════════
# 4.  align_rows  (Gwyddion: "Align Rows")
# ═════════════════════════════════════════════════════════════════════════════

def align_rows(arr: np.ndarray, method: str = 'median') -> np.ndarray:
    """
    Fix inter-line DC offsets by subtracting a per-row reference value.

    method='median'  — subtract each row's median (robust to tip crashes)
    method='mean'    — subtract each row's mean
    method='linear'  — fit and subtract a per-row linear trend (slope + offset)

    This is the most effective first step for raw STM data where each scan
    line has an independent height offset due to thermal drift or tip jumps.
    """
    if method not in {"median", "mean", "linear"}:
        raise ValueError("method must be 'median', 'mean', or 'linear'")

    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    if method == 'median':
        for r in range(Ny):
            row = arr[r]
            finite = row[np.isfinite(row)]
            if finite.size:
                arr[r] -= float(np.median(finite))

    elif method == 'mean':
        for r in range(Ny):
            row = arr[r]
            finite = row[np.isfinite(row)]
            if finite.size:
                arr[r] -= float(finite.mean())

    elif method == 'linear':
        xs = np.linspace(-1.0, 1.0, Nx)
        for r in range(Ny):
            row = arr[r]
            fin = np.isfinite(row)
            if fin.sum() < 2:
                continue
            coeffs = np.polyfit(xs[fin], row[fin], 1)
            arr[r] -= np.polyval(coeffs, xs)
    return arr


# ═════════════════════════════════════════════════════════════════════════════
# 4.  facet_level  (Gwyddion: "Facet Level")
# ═════════════════════════════════════════════════════════════════════════════

def facet_level(
    arr: np.ndarray,
    threshold_deg: float = 3.0,
    *,
    pixel_size_x_m: float = 1.0,
    pixel_size_y_m: float = 1.0,
) -> np.ndarray:
    """
    Level the image using only the nearly-flat (horizontal) pixels as
    the reference plane.

    Local slopes are estimated via finite differences.  Pixels with a slope
    angle below *threshold_deg* are treated as part of flat terraces and used
    for the plane fit.  The fitted plane is then subtracted from the whole image.
    This avoids step edges biasing the background correction — essential for
    Au(111), Si(111) and other stepped surfaces.

    Parameters
    ----------
    pixel_size_x_m, pixel_size_y_m
        Physical size of one pixel in metres along x and y respectively.
        Passing the true pixel sizes makes the threshold_deg comparison
        physically meaningful: the gradient is divided by the pixel size so
        the slope is dimensionless (height / lateral distance) before being
        compared against tan(threshold_deg).  Defaults to 1.0, which preserves
        backward-compatible behaviour for callers that work in pixel units.
    """
    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    if Ny < 3 or Nx < 3:
        return arr

    finite = np.isfinite(arr)
    if not finite.any():
        return arr

    # Estimate local gradient via central differences (pixel units). NaNs are
    # filled only for gradient estimation; they are still excluded from fitting.
    grad_arr = np.where(finite, arr, _finite_median(arr))
    gy, gx = np.gradient(grad_arr)

    # Convert gradient from data-units/pixel to dimensionless rise/run by
    # dividing by the physical pixel size in metres.  This makes the
    # tan(threshold_deg) comparison physically meaningful.
    psx = max(float(pixel_size_x_m), 1e-30)
    psy = max(float(pixel_size_y_m), 1e-30)
    slope_mag = np.sqrt((gx / psx) ** 2 + (gy / psy) ** 2)

    tan_thresh = math.tan(math.radians(threshold_deg))
    flat_mask = (slope_mag < tan_thresh) & np.isfinite(arr)

    if flat_mask.sum() < 3:
        # Not enough flat pixels — fall back to full-image plane
        return subtract_background(arr, order=1)

    ys = np.linspace(-1.0, 1.0, Ny)
    xs = np.linspace(-1.0, 1.0, Nx)
    Xg, Yg = np.meshgrid(xs, ys)

    flat_x = Xg[flat_mask]
    flat_y = Yg[flat_mask]
    flat_z = arr[flat_mask]

    A = np.column_stack([flat_x, flat_y, np.ones(flat_x.size)])
    coeffs, _, _, _ = np.linalg.lstsq(A, flat_z, rcond=None)
    bg = coeffs[0]*Xg + coeffs[1]*Yg + coeffs[2]

    return arr - bg
