"""
ProbeFlow — image processing pipeline for STM/SXM data.

All functions operate on raw float32/float64 2-D arrays (physical units).
They are intentionally free of any GUI dependency so they can be called from
worker threads or batch scripts without importing PySide6.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    label as _nd_label,
    laplace as _nd_laplace,
)

# ── Font path for scale-bar labels ────────────────────────────────────────────
_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def _finite_mean(arr: np.ndarray, default: float = 0.0) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    return float(finite.mean()) if finite.size else float(default)


def _finite_median(arr: np.ndarray, default: float = 0.0) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    return float(np.median(finite)) if finite.size else float(default)


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _nan_normalized_gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that ignores NaNs instead of mean-filling them."""
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        return np.full(a.shape, np.nan, dtype=np.float64)
    sigma = max(float(sigma), 0.0)
    values = np.where(finite, a, 0.0)
    weights = finite.astype(np.float64)
    blurred_values = gaussian_filter(values, sigma=sigma, mode="reflect")
    blurred_weights = gaussian_filter(weights, sigma=sigma, mode="reflect")
    out = np.full(a.shape, np.nan, dtype=np.float64)
    np.divide(
        blurred_values,
        blurred_weights,
        out=out,
        where=blurred_weights > 1e-12,
    )
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 1.  remove_bad_lines
# ═════════════════════════════════════════════════════════════════════════════

def remove_bad_lines(
    arr: np.ndarray,
    threshold_mad: float = 5.0,
    *,
    method: str = "mad",
) -> np.ndarray:
    """Replace scan-line artefacts via interpolation from neighbours.

    Two detection strategies are available:

    ``method="mad"`` (default)
        Detects *entire bad rows* by their median height.  A row is flagged
        when ``|row_median − overall_median| > threshold_mad × MAD``, where
        MAD is the median absolute deviation of all per-row medians.  Bad rows
        are replaced by a distance-weighted column-wise blend of the nearest
        good rows above and below.  Self-calibrating — no physical unit
        threshold needed.  Works best when the bad line shifts the entire row
        by a systematic offset (tip crash, vibration burst).

    ``method="step"``
        Detects bad pixels *per column* by looking for abrupt step
        transitions that exceed an auto-computed threshold
        (``threshold_mad × MAD`` of all column-wise row differences).  For
        each column, the region between a positive crossing and the following
        negative crossing is marked bad and interpolated linearly from the
        nearest good pixels above and below.  Handles *partial* bad lines
        (only some columns affected) and isolated pixel artefacts that a
        row-level detector would miss.
    """
    if method not in {"mad", "step"}:
        raise ValueError(f"method must be 'mad' or 'step', got {method!r}")
    if method == "step":
        return _remove_bad_lines_step(arr, threshold_mad)

    threshold_mad = _nonnegative_finite(threshold_mad, "threshold_mad")
    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    row_meds = np.full(Ny, np.nan, dtype=np.float64)
    for r in range(Ny):
        finite = arr[r][np.isfinite(arr[r])]
        if finite.size:
            row_meds[r] = float(np.median(finite))

    finite_meds = row_meds[np.isfinite(row_meds)]
    if finite_meds.size == 0:
        return arr

    overall_med = float(np.median(finite_meds))
    mad = float(np.median(np.abs(finite_meds - overall_med)))
    if mad == 0.0:
        tol = np.finfo(np.float64).eps * max(abs(overall_med), 1.0) * 16.0
        bad_mask = np.abs(row_meds - overall_med) > tol
    else:
        bad_mask = np.abs(row_meds - overall_med) > threshold_mad * mad
    bad_rows = np.where(bad_mask)[0]

    if bad_rows.size == 0:
        return arr

    good_rows = np.where(~bad_mask)[0]
    if good_rows.size == 0:
        return arr

    for r in bad_rows:
        above = good_rows[good_rows < r]
        below = good_rows[good_rows > r]

        if above.size > 0 and below.size > 0:
            ra, rb = int(above[-1]), int(below[0])
            da, db = r - ra, rb - r
            wa = db / (da + db)
            wb = da / (da + db)
            arr[r] = wa * arr[ra] + wb * arr[rb]
        elif above.size > 0:
            arr[r] = arr[int(above[-1])]
        else:
            arr[r] = arr[int(below[0])]

    return arr


def _remove_bad_lines_step(arr: np.ndarray, threshold_factor: float) -> np.ndarray:
    """Column-level step detection and per-column linear interpolation.

    For each column the algorithm scans rows top-to-bottom looking for
    abrupt transitions.  A "bright line" begins at the row where the
    column value jumps up by more than *threshold*, and ends at the row
    where it jumps back down.  The flagged pixels are replaced by linear
    interpolation between the nearest good pixels above and below in the
    same column.

    The threshold is ``threshold_factor × MAD`` of all finite column-wise
    row differences, computed automatically from the image data.
    """
    threshold_factor = _nonnegative_finite(threshold_factor, "threshold_mad")
    a = arr.astype(np.float64, copy=True)
    Ny, Nx = a.shape

    # Auto-threshold from the distribution of row-to-row column differences
    diffs_raw = np.diff(a, axis=0)
    finite_diffs = np.abs(diffs_raw[np.isfinite(diffs_raw)])
    if finite_diffs.size == 0:
        return a
    mad = float(np.median(np.abs(finite_diffs - np.median(finite_diffs))))
    if mad == 0.0:
        mad = max(float(np.median(finite_diffs)) * 1e-3, 1e-15)
    threshold = threshold_factor * mad

    bad = np.zeros((Ny, Nx), dtype=bool)
    for c in range(Nx):
        col = a[:, c]
        in_bad = False
        for r in range(1, Ny):
            if not (np.isfinite(col[r]) and np.isfinite(col[r - 1])):
                continue
            step = col[r] - col[r - 1]
            if step > threshold:
                in_bad = True
            elif step < -threshold:
                in_bad = False
            if in_bad:
                bad[r, c] = True

    # Per-column linear interpolation of flagged pixels
    for c in range(Nx):
        bad_rows = np.where(bad[:, c])[0]
        if bad_rows.size == 0:
            continue
        good_rows = np.where(~bad[:, c] & np.isfinite(a[:, c]))[0]
        if good_rows.size == 0:
            continue
        for r in bad_rows:
            above = good_rows[good_rows < r]
            below = good_rows[good_rows > r]
            if above.size and below.size:
                ra, rb = int(above[-1]), int(below[0])
                da, db = r - ra, rb - r
                a[r, c] = (db * a[ra, c] + da * a[rb, c]) / (da + db)
            elif above.size:
                a[r, c] = a[int(above[-1]), c]
            else:
                a[r, c] = a[int(below[0]), c]

    return a


# ═════════════════════════════════════════════════════════════════════════════
# 2.  subtract_background
# ═════════════════════════════════════════════════════════════════════════════

def _poly_terms(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Return an (N, M) design matrix for a 2D polynomial up to total degree `order`.

    All terms x^i * y^j where i + j <= order, ordered by total degree then by
    increasing power of x.  The exact column order is deterministic and used
    consistently in both fitting and reconstruction.
    """
    cols = []
    for total in range(order + 1):
        for i in range(total + 1):
            j = total - i
            cols.append((x ** i) * (y ** j))
    return np.column_stack(cols)


def subtract_background(
    arr: np.ndarray,
    order: int = 1,
    *,
    fit_roi: "Any | None" = None,
    apply_roi: "Any | None" = None,
    exclude_roi: "Any | None" = None,
    step_tolerance: bool = False,
    step_threshold_deg: float = 3.0,
    fit_rect: Optional[tuple[int, int, int, int]] = None,
    fit_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit and subtract a 2-D polynomial background from an image.

    Parameters
    ----------
    arr:
        2-D image array.
    order:
        Total polynomial order. Supported values are 1, 2, 3, 4.
    fit_roi:
        ROI whose pixels are used to *estimate* the background.  Pixels
        outside this region do not influence the polynomial fit, but the
        fitted background is extrapolated and (optionally) subtracted over
        the full image or ``apply_roi``.  ``None`` means all pixels.
    apply_roi:
        ROI whose pixels are *modified* by subtracting the fitted background.
        Pixels outside this region are returned unchanged.  ``None`` means
        subtract the background everywhere.

        .. note::
            Where ``apply_roi`` ends, there will be a discontinuity in the
            output equal to the local background value.  This is correct
            behaviour for the "patch-only" correction case.  For per-terrace
            fits where a smooth boundary matters, ensure ``fit_roi`` includes
            the full terrace and ``apply_roi = None``.

    exclude_roi:
        ROI whose pixels are *removed from the fit*.  Applied after
        ``fit_roi``; the effective fit region is
        ``fit_roi AND NOT exclude_roi``.  If ``exclude_roi`` is partially
        outside ``fit_roi``, only the overlapping part is excluded.
        ``None`` means no exclusion.

    Canonical use patterns
    ----------------------
    Global plane fit (standard background removal)::

        subtract_background(img)

    Fit on a clean terrace, subtract globally::

        subtract_background(img, fit_roi=terrace_roi)

    Fit on a terrace, apply only to an adjacent region::

        subtract_background(img, fit_roi=terrace_roi, apply_roi=target_roi)

    Fit globally but exclude contaminated molecules::

        subtract_background(img, exclude_roi=molecules_roi)

    Correct only a small patch (fit and apply to the same region)::

        subtract_background(img, fit_roi=patch_roi, apply_roi=patch_roi)

    Legacy parameters (still accepted, combined with ROI parameters)
    -----------------------------------------------------------------
    step_tolerance:
        When True, use a step-tolerant surface mask: pixels whose finite-
        difference gradient exceeds ``tan(step_threshold_deg)`` are excluded
        from the fit. Falls back to a full-pixel fit when fewer than
        ``n_terms`` pixels remain after masking.
    step_threshold_deg:
        Slope angle (degrees) above which a pixel is treated as a step edge.
    fit_rect:
        Optional inclusive pixel rectangle ``(x0, y0, x1, y1)`` restricting
        the fit region.  Combined with ``fit_roi`` (intersection).
    fit_mask:
        Optional boolean mask restricting the fit region.  Combined with
        ``fit_roi`` and ``fit_rect`` (intersection).

    Coordinates are normalised to [-1, 1] for numerical stability. Only
    finite pixels participate in the least-squares fit. NaNs in the input
    are preserved in the output.
    """
    if order < 1 or order > 4:
        raise ValueError(f"order must be 1..4, got {order}")
    if step_tolerance and not np.isfinite(step_threshold_deg):
        raise ValueError("step_threshold_deg must be finite")

    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    ys = np.linspace(-1.0, 1.0, Ny)
    xs = np.linspace(-1.0, 1.0, Nx)
    Xg, Yg = np.meshgrid(xs, ys)

    flat_x = Xg.ravel()
    flat_y = Yg.ravel()
    flat_z = arr.ravel()

    finite = np.isfinite(flat_z)
    n_terms = (order + 1) * (order + 2) // 2
    # Graceful fallback when the image itself lacks enough finite pixels
    if finite.sum() < n_terms:
        return arr

    # ── Build fit mask from new ROI parameters ────────────────────────────────
    user_fit_mask = fit_mask  # legacy param
    has_explicit_region = (
        fit_roi is not None or exclude_roi is not None
        or fit_rect is not None or user_fit_mask is not None
    )

    # Merge fit_roi and exclude_roi into a single ROI-level mask
    roi_fit_mask: Optional[np.ndarray] = None
    if fit_roi is not None:
        roi_fit_mask = np.asarray(fit_roi.to_mask(arr.shape), dtype=bool)
    if exclude_roi is not None:
        excl = np.asarray(exclude_roi.to_mask(arr.shape), dtype=bool)
        roi_fit_mask = (~excl) if roi_fit_mask is None else (roi_fit_mask & ~excl)

    # ── Accumulate fit_mask (intersection of all constraints) ─────────────────
    fit_mask_acc = finite.copy()

    if fit_rect is not None:
        try:
            x0r, y0r, x1r, y1r = [int(v) for v in fit_rect]
        except (TypeError, ValueError):
            return arr
        x0r = max(0, min(Nx - 1, x0r))
        x1r = max(0, min(Nx - 1, x1r))
        y0r = max(0, min(Ny - 1, y0r))
        y1r = max(0, min(Ny - 1, y1r))
        if x1r <= x0r or y1r <= y0r:
            return arr
        rect_mask = np.zeros(arr.shape, dtype=bool)
        rect_mask[y0r:y1r + 1, x0r:x1r + 1] = True
        fit_mask_acc &= rect_mask.ravel()

    if user_fit_mask is not None:
        try:
            um = np.asarray(user_fit_mask, dtype=bool)
        except (TypeError, ValueError):
            return arr
        if um.shape != arr.shape:
            return arr
        fit_mask_acc &= um.ravel()

    if roi_fit_mask is not None:
        fit_mask_acc &= roi_fit_mask.ravel()

    if fit_mask_acc.sum() < n_terms:
        if has_explicit_region:
            raise ValueError(
                f"Fit region has only {int(fit_mask_acc.sum())} pixel(s), but order "
                f"{order} requires at least {n_terms} pixels."
            )
        return arr  # graceful fallback for whole-image degenerate cases

    # ── Step-tolerance masking ────────────────────────────────────────────────
    if step_tolerance and Ny >= 3 and Nx >= 3:
        gy, gx = np.gradient(np.where(np.isfinite(arr), arr, _finite_median(arr)))
        slope_mag = np.sqrt(gx ** 2 + gy ** 2).ravel()
        tan_thresh = math.tan(math.radians(step_threshold_deg))
        candidate = finite & (slope_mag < tan_thresh) & fit_mask_acc
        if candidate.sum() >= n_terms:
            fit_mask_acc = candidate

    # ── Fit polynomial ────────────────────────────────────────────────────────
    A = _poly_terms(flat_x[fit_mask_acc], flat_y[fit_mask_acc], order)
    b = flat_z[fit_mask_acc]
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    bg = (_poly_terms(flat_x, flat_y, order) @ coeffs).reshape(Ny, Nx)

    # ── Apply background subtraction ──────────────────────────────────────────
    if apply_roi is not None:
        apply_mask = np.asarray(apply_roi.to_mask(arr.shape), dtype=bool)
        arr[apply_mask] -= bg[apply_mask]
        return arr
    return arr - bg


# ═════════════════════════════════════════════════════════════════════════════
# 3.  stm_line_background
# ═════════════════════════════════════════════════════════════════════════════

def _modal_shift(values: np.ndarray, *, bins: int = 128) -> Optional[float]:
    """Estimate the dominant value from the peak of a 1-D histogram."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    if values.size == 1:
        return float(values[0])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmin == vmax:
        return vmin
    try:
        # numpy>=2 raises ValueError when (vmax-vmin)/n_bins underflows; fall
        # back to the median for residuals whose range is below float64 precision.
        hist, edges = np.histogram(values, bins=min(bins, max(8, values.size)))
    except ValueError:
        return float(np.nanmedian(values))
    if hist.size == 0 or int(hist.max()) == 0:
        return float(np.nanmedian(values))
    peak = int(np.argmax(hist))
    in_peak = (values >= edges[peak]) & (values <= edges[peak + 1])
    if np.any(in_peak):
        return float(np.nanmedian(values[in_peak]))
    return float(0.5 * (edges[peak] + edges[peak + 1]))


def stm_line_background(arr: np.ndarray, mode: str = "step_tolerant") -> np.ndarray:
    """Subtract an STM-style step-tolerant line background.

    The step-tolerant mode estimates each adjacent scan-line offset from the
    modal peak of a histogram of pixelwise row differences.  This follows the
    dominant terrace-to-terrace shift rather than the mean height, so partial
    step edges do not dominate the correction.
    """
    if mode != "step_tolerant":
        raise ValueError(f"mode must be 'step_tolerant', got {mode!r}")

    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape
    if Ny < 2 or Nx < 1:
        return arr

    shifts = np.zeros(Ny, dtype=np.float64)
    prev_shift = 0.0
    have_shift = False

    for r in range(1, Ny):
        diff = arr[r] - arr[r - 1]
        diff = diff[np.isfinite(diff)]
        if diff.size == 0:
            shifts[r] = shifts[r - 1]
            continue

        if have_shift:
            residual = diff - prev_shift
            delta = _modal_shift(residual)
            shift = prev_shift if delta is None else prev_shift + delta
        else:
            modal = _modal_shift(diff)
            if modal is None:
                shifts[r] = shifts[r - 1]
                continue
            shift = modal
            have_shift = True

        prev_shift = shift
        shifts[r] = shifts[r - 1] + shift

    if not have_shift:
        return arr

    profile = shifts - float(np.nanmedian(shifts))
    return arr - profile[:, None]


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

def facet_level(arr: np.ndarray, threshold_deg: float = 3.0) -> np.ndarray:
    """
    Level the image using only the nearly-flat (horizontal) pixels as
    the reference plane.

    Local slopes are estimated via finite differences.  Pixels with a slope
    angle below *threshold_deg* are treated as part of flat terraces and used
    for the plane fit.  The fitted plane is then subtracted from the whole image.
    This avoids step edges biasing the background correction — essential for
    Au(111), Si(111) and other stepped surfaces.
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

    # Convert threshold from degrees to tangent value
    tan_thresh = math.tan(math.radians(threshold_deg))
    slope_mag = np.sqrt(gx**2 + gy**2)
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


# ═════════════════════════════════════════════════════════════════════════════
# 5.  fourier_filter
# ═════════════════════════════════════════════════════════════════════════════

def fourier_filter(
    arr:    np.ndarray,
    mode:   str   = 'low_pass',
    cutoff: float = 0.1,
    window: str   = 'hanning',
) -> np.ndarray:
    """
    Apply a simple global radial 2-D FFT filter.

    cutoff  — fraction of Nyquist [0, 1].
    mode    — 'low_pass' | 'high_pass'

    This is a coarse circular frequency cutoff, not an ImageJ-style periodic
    spot/vector filter.
    """
    if mode not in {"low_pass", "high_pass"}:
        raise ValueError("mode must be 'low_pass' or 'high_pass'")
    if not np.isfinite(cutoff) or not 0.0 <= float(cutoff) <= 1.0:
        raise ValueError("cutoff must be finite and in [0, 1]")
    window_key = str(window or "none").lower()
    if window_key not in {"hanning", "hamming", "none", "rectangular", "boxcar"}:
        raise ValueError("window must be 'hanning', 'hamming', or 'none'")

    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    nan_mask = ~np.isfinite(arr)
    if nan_mask.all():
        return arr

    mean_val = _finite_mean(arr)
    filled = np.where(nan_mask, mean_val, arr)
    centered = filled - mean_val

    if mode == "low_pass" and float(cutoff) >= 1.0:
        out = filled
        out[nan_mask] = np.nan
        return out
    if mode == "high_pass" and float(cutoff) <= 0.0:
        out = centered
        out[nan_mask] = np.nan
        return out

    if window_key == 'hanning':
        wy = np.hanning(Ny)
        wx = np.hanning(Nx)
    elif window_key == 'hamming':
        wy = np.hamming(Ny)
        wx = np.hamming(Nx)
    else:
        wy = np.ones(Ny)
        wx = np.ones(Nx)

    win2d = np.outer(wy, wx)
    windowed = centered * win2d

    F = np.fft.fft2(windowed)
    F = np.fft.fftshift(F)

    cy, cx = Ny / 2.0, Nx / 2.0
    yr = (np.arange(Ny) - cy) / max(cy, 1e-9)
    xr = (np.arange(Nx) - cx) / max(cx, 1e-9)
    Xr, Yr = np.meshgrid(xr, yr)
    R = np.sqrt(Xr**2 + Yr**2)

    if mode == 'low_pass':
        mask = (R <= cutoff).astype(np.float64)
    elif mode == 'high_pass':
        mask = (R >= cutoff).astype(np.float64)

    F_filtered = F * mask
    F_filtered = np.fft.ifftshift(F_filtered)
    result = np.fft.ifft2(F_filtered).real
    if mode == "low_pass":
        result = result + mean_val
    result[nan_mask] = np.nan
    return result


def gaussian_high_pass(arr: np.ndarray, sigma_px: float = 8.0) -> np.ndarray:
    """Subtract a broad Gaussian-blurred background from an image.

    This mirrors the common ImageJ-style high-pass workflow: estimate large
    structures with a Gaussian blur, then subtract that smooth component while
    preserving high-frequency detail.
    """
    sigma_px = _nonnegative_finite(sigma_px, "sigma_px")
    a = arr.astype(np.float64, copy=True)
    nan_mask = ~np.isfinite(a)
    if nan_mask.all():
        return a
    bg = _nan_normalized_gaussian(a, max(sigma_px, 0.1))
    out = a - bg
    out[nan_mask] = np.nan
    return out


def periodic_notch_filter(
    arr: np.ndarray,
    peaks: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    radius_px: float = 3.0,
) -> np.ndarray:
    """Suppress selected periodic FFT peaks and their conjugates.

    ``peaks`` are integer ``(dx, dy)`` offsets from the centred FFT origin in
    pixels. A Gaussian notch is applied at each offset and its conjugate.
    """
    radius_px = _nonnegative_finite(radius_px, "radius_px")
    a = arr.astype(np.float64, copy=True)
    Ny, Nx = a.shape
    if Ny < 2 or Nx < 2 or not peaks:
        return a

    nan_mask = ~np.isfinite(a)
    mean_val = float(np.nanmean(a)) if (~nan_mask).any() else 0.0
    filled = np.where(nan_mask, mean_val, a)
    F = np.fft.fftshift(np.fft.fft2(filled - mean_val))

    cy, cx = Ny // 2, Nx // 2
    yy, xx = np.mgrid[:Ny, :Nx]
    notch = np.ones((Ny, Nx), dtype=np.float64)
    sigma = max(radius_px, 0.5)

    for peak in peaks:
        try:
            dx, dy = int(peak[0]), int(peak[1])
        except (TypeError, ValueError, IndexError):
            continue
        if dx == 0 and dy == 0:
            continue
        for sx, sy in ((dx, dy), (-dx, -dy)):
            px = cx + sx
            py = cy + sy
            if 0 <= px < Nx and 0 <= py < Ny:
                r2 = (xx - px) ** 2 + (yy - py) ** 2
                notch *= 1.0 - np.exp(-0.5 * r2 / (sigma ** 2))

    out = np.fft.ifft2(np.fft.ifftshift(F * notch)).real + mean_val
    out[nan_mask] = np.nan
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 6.  gaussian_smooth  (Gwyddion: "Gaussian filter")
# ═════════════════════════════════════════════════════════════════════════════

def gaussian_smooth(arr: np.ndarray, sigma_px: float = 1.0) -> np.ndarray:
    """
    Apply a 2-D isotropic Gaussian smoothing filter.

    sigma_px — standard deviation in pixels.  Typical STM values: 0.5–3.
    NaN pixels are handled by weighted normalisation so they don't propagate.
    """
    sigma_px = _nonnegative_finite(sigma_px, "sigma_px")
    arr = arr.astype(np.float64, copy=True)
    nan_mask = ~np.isfinite(arr)
    if nan_mask.all():
        return arr
    smoothed = _nan_normalized_gaussian(arr, sigma_px)
    if nan_mask.any():
        smoothed[nan_mask] = np.nan

    return smoothed


# ═════════════════════════════════════════════════════════════════════════════
# 7.  edge_detect  (Gwyddion/Tycoon: Laplacian, DoG, LoG)
# ═════════════════════════════════════════════════════════════════════════════

def edge_detect(
    arr:    np.ndarray,
    method: str   = 'laplacian',
    sigma:  float = 1.0,
    sigma2: float = 2.0,
) -> np.ndarray:
    """
    Edge / feature enhancement using Laplacian-family filters.

    method='laplacian' — discrete Laplacian (2nd derivative, no smoothing)
    method='log'       — Laplacian of Gaussian  (σ = sigma)
    method='dog'       — Difference of Gaussians (σ₁=sigma, σ₂=sigma2)

    Returns the filter response — positive = bright edges/peaks,
    negative = dark edges/valleys.  Useful for atomic-resolution contrast
    enhancement and finding adsorption sites.
    """
    sigma = _nonnegative_finite(sigma, "sigma")
    sigma2 = _nonnegative_finite(sigma2, "sigma2")
    a = arr.astype(np.float64, copy=True)
    nan_mask = ~np.isfinite(a)
    if nan_mask.all():
        return a
    if nan_mask.any():
        a[nan_mask] = _finite_mean(a)

    if method == 'laplacian':
        result = _nd_laplace(a)

    elif method == 'log':
        # Pre-smooth then Laplacian
        smoothed = gaussian_filter(a, sigma=max(sigma, 0.1), mode='reflect')
        result   = _nd_laplace(smoothed)

    elif method == 'dog':
        g1 = gaussian_filter(a, sigma=max(sigma,  0.1), mode='reflect')
        g2 = gaussian_filter(a, sigma=max(sigma2, sigma + 0.1), mode='reflect')
        result = g1 - g2

    else:
        raise ValueError(f"Unknown edge_detect method: {method!r}")

    if nan_mask.any():
        result[nan_mask] = np.nan

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 8.  gmm_autoclip  (UniMR: GMM auto-thresholding)
# ═════════════════════════════════════════════════════════════════════════════

def gmm_autoclip(arr: np.ndarray, n_samples: int = 2000) -> tuple[float, float]:
    """
    Estimate optimal clip_low / clip_high percentiles using a 2-component
    Gaussian Mixture Model fitted to the image histogram.

    Returns (clip_low_pct, clip_high_pct) as percentile values [0–100],
    suitable for passing directly to np.percentile.

    The approach mirrors the UniMR project's gmm_threshold():  fit two
    Gaussians to the value distribution, find their intersection, and map
    the lower-component tail and upper-component tail to clip percentiles.
    Falls back to (1.0, 99.0) if the fit is degenerate.
    """
    data = arr.astype(np.float64).ravel()
    data = data[np.isfinite(data)]
    if data.size < 10:
        return 1.0, 99.0

    # Subsample for speed
    if data.size > n_samples:
        rng = np.random.default_rng(0)
        data = rng.choice(data, size=n_samples, replace=False)

    # EM for a 2-component 1-D GMM (numpy-only implementation)
    data_min, data_max = data.min(), data.max()
    if data_max <= data_min:
        return 1.0, 99.0

    # Initialise: split at median
    med = float(np.median(data))
    mu1, mu2 = float(data[data <= med].mean()), float(data[data > med].mean())
    s1 = s2 = float(data.std()) / 2.0 + 1e-10
    pi1 = pi2 = 0.5

    for _ in range(60):
        # E-step: responsibilities
        def _gauss(x, mu, s):
            return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))

        r1 = pi1 * _gauss(data, mu1, s1)
        r2 = pi2 * _gauss(data, mu2, s2)
        denom = r1 + r2 + 1e-300
        r1 /= denom
        r2 /= denom

        # M-step
        n1, n2 = r1.sum(), r2.sum()
        if n1 < 1e-6 or n2 < 1e-6:
            break
        mu1_new = (r1 * data).sum() / n1
        mu2_new = (r2 * data).sum() / n2
        s1_new  = math.sqrt((r1 * (data - mu1_new)**2).sum() / n1) + 1e-10
        s2_new  = math.sqrt((r2 * (data - mu2_new)**2).sum() / n2) + 1e-10
        pi1_new = n1 / (n1 + n2)
        pi2_new = n2 / (n1 + n2)

        if (abs(mu1_new - mu1) < 1e-8 * abs(data_max - data_min) and
                abs(mu2_new - mu2) < 1e-8 * abs(data_max - data_min)):
            mu1, mu2, s1, s2, pi1, pi2 = mu1_new, mu2_new, s1_new, s2_new, pi1_new, pi2_new
            break
        mu1, mu2, s1, s2, pi1, pi2 = mu1_new, mu2_new, s1_new, s2_new, pi1_new, pi2_new

    # Ensure mu1 < mu2
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        s1,  s2  = s2,  s1
        pi1, pi2 = pi2, pi1

    # Convert GMM component extents to percentiles on the full dataset
    full_data = arr.astype(np.float64).ravel()
    full_data = full_data[np.isfinite(full_data)]
    if full_data.size == 0:
        return 1.0, 99.0

    # Low clip: 2σ below lower component mean
    low_val  = mu1 - 2.0 * s1
    # High clip: 2σ above upper component mean
    high_val = mu2 + 2.0 * s2

    clip_low  = float(np.sum(full_data <  low_val)  / full_data.size * 100.0)
    clip_high = float(np.sum(full_data <= high_val) / full_data.size * 100.0)

    # Clamp to sane range
    clip_low  = max(0.0, min(clip_low,  10.0))
    clip_high = min(100.0, max(clip_high, 90.0))

    return clip_low, clip_high


# ═════════════════════════════════════════════════════════════════════════════
# 9.  detect_grains  (Gwyddion: "Mark Grains by Threshold / Watershed")
# ═════════════════════════════════════════════════════════════════════════════

def detect_grains(
    arr:                np.ndarray,
    threshold_pct:      float = 50.0,
    above:              bool  = True,
    min_grain_px:       int   = 5,
) -> tuple[np.ndarray, int, dict]:
    """
    Detect grains/islands by thresholding the height data.

    Parameters
    ----------
    arr             : 2-D float array (height data)
    threshold_pct   : percentile of data used as threshold (0–100)
    above           : True = grains are above threshold (islands on flat terrace)
                      False = grains are below (holes/depressions)
    min_grain_px    : grains smaller than this many pixels are discarded

    Returns
    -------
    label_map  : int32 array with each grain labelled 1, 2, 3, …  (0 = background)
    n_grains   : number of grains found
    stats      : dict with 'areas_px', 'centroids', 'mean_heights'
    """
    a = arr.astype(np.float64, copy=True)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        empty = np.zeros(a.shape, dtype=np.int32)
        return empty, 0, {}

    thresh = float(np.percentile(finite, threshold_pct))

    if above:
        binary = np.isfinite(a) & (a >= thresh)
    else:
        binary = np.isfinite(a) & (a <= thresh)

    label_map, n_raw = _nd_label(binary)

    # Remove grains below minimum size
    if min_grain_px > 1:
        for grain_id in range(1, n_raw + 1):
            mask = label_map == grain_id
            if mask.sum() < min_grain_px:
                label_map[mask] = 0

        # Re-label contiguously
        label_map, n_grains = _nd_label(label_map > 0)
    else:
        n_grains = n_raw

    # Compute per-grain statistics
    areas, centroids, heights = [], [], []
    for grain_id in range(1, n_grains + 1):
        mask = label_map == grain_id
        ys, xs = np.where(mask)
        areas.append(int(mask.sum()))
        centroids.append((float(xs.mean()), float(ys.mean())))
        vals = a[mask]
        heights.append(float(np.nanmean(vals[np.isfinite(vals)])))

    stats = {
        'areas_px':    areas,
        'centroids':   centroids,
        'mean_heights': heights,
    }

    return label_map.astype(np.int32), n_grains, stats


# ═════════════════════════════════════════════════════════════════════════════
# 10.  measure_periodicity
# ═════════════════════════════════════════════════════════════════════════════

def measure_periodicity(
    arr:           np.ndarray,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    n_peaks:       int = 5,
) -> list[dict]:
    """
    Find dominant spatial periodicities in a 2-D array using its power spectrum.

    Returns a list (length ≤ n_peaks) of dicts:
        {'period_m': float, 'angle_deg': float, 'strength': float}
    """
    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    mean_val = float(np.nanmean(arr))
    arr[~np.isfinite(arr)] = mean_val

    wy = np.hanning(Ny)
    wx = np.hanning(Nx)
    win2d = np.outer(wy, wx)

    F = np.fft.fft2(arr * win2d)
    F = np.fft.fftshift(F)
    power = np.abs(F) ** 2

    cy, cx = Ny // 2, Nx // 2

    DC_R = 2.0

    half_mask = np.zeros((Ny, Nx), dtype=bool)
    half_mask[:cy, :] = True

    yr = np.arange(Ny) - cy
    xr = np.arange(Nx) - cx
    Xr, Yr = np.meshgrid(xr.astype(float), yr.astype(float))
    R_px = np.sqrt(Xr**2 + Yr**2)
    half_mask[R_px < DC_R] = False

    search_power = power.copy()
    search_power[~half_mask] = 0.0

    results = []
    suppress_r = max(3, min(Ny, Nx) // 20)

    for _ in range(n_peaks):
        idx = int(np.argmax(search_power))
        py, px = divmod(idx, Nx)

        peak_val = float(search_power[py, px])
        if peak_val <= 0:
            break

        fy = (py - cy) / Ny
        fx = (px - cx) / Nx

        f_mag = math.sqrt(fx**2 + fy**2)
        if f_mag == 0.0:
            break

        freq_m_x = fx / pixel_size_x_m
        freq_m_y = fy / pixel_size_y_m
        freq_m   = math.sqrt(freq_m_x**2 + freq_m_y**2)
        period_m = 1.0 / freq_m if freq_m > 0 else 0.0

        angle_deg = math.degrees(math.atan2(fy * pixel_size_y_m,
                                             fx * pixel_size_x_m))

        results.append({
            'period_m':  period_m,
            'angle_deg': angle_deg,
            'strength':  peak_val,
        })

        for (rpy, rpx) in [(py, px), (Ny - py, Nx - px)]:
            for dy in range(-suppress_r, suppress_r + 1):
                for dx in range(-suppress_r, suppress_r + 1):
                    ny_ = int(rpy) + dy
                    nx_ = int(rpx) + dx
                    if 0 <= ny_ < Ny and 0 <= nx_ < Nx:
                        search_power[ny_, nx_] = 0.0

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 11.  export_png
# ═════════════════════════════════════════════════════════════════════════════

_NICE_STEPS_NM = [
    0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50,
    100, 200, 500, 1000, 2000, 5000, 10000,
]


def _pick_scalebar_length(width_m: float, image_px: int,
                          target_frac: float = 0.20,
                          unit: str = 'nm') -> tuple[float, str]:
    unit_factors = {'nm': 1e9, 'Å': 1e10, 'pm': 1e12}
    factor = unit_factors.get(unit, 1e9)

    target_m = width_m * target_frac
    target_u = target_m * factor

    best = _NICE_STEPS_NM[0]
    for s in _NICE_STEPS_NM:
        if abs(s - target_u) < abs(best - target_u):
            best = s
        if s > target_u * 2:
            break

    bar_m = best / factor

    if best == int(best):
        label = f"{int(best)} {unit}"
    else:
        label = f"{best:g} {unit}"

    return bar_m, label


def export_png(
    arr:           np.ndarray,
    out_path,
    colormap_key:  str,
    clip_low:      float,
    clip_high:     float,
    lut_fn,
    scan_range_m:  tuple,
    add_scalebar:  bool          = True,
    scalebar_unit: str           = 'nm',
    scalebar_pos:  str           = 'bottom-right',
    vmin:          float | None  = None,
    vmax:          float | None  = None,
    provenance                   = None,   # ExportProvenance | None
) -> None:
    """
    Export a full-resolution colourised image with an optional scale bar.

    lut_fn(colormap_key) must return a (256, 3) uint8 LUT array.
    scan_range_m  — (width_m, height_m); scale bar is skipped when width ≤ 0.
    If *vmin*/*vmax* are provided, they override the percentile clip.
    If *provenance* is provided, a ``<stem>.provenance.json`` sidecar is written.
    """
    from PIL import Image as _Image, ImageDraw as _IDraw, ImageFont as _IFont

    from probeflow.processing.display import array_to_uint8 as _array_to_uint8, clip_range_from_array as _clip_range

    arr = arr.astype(np.float64, copy=True)
    if vmin is None or vmax is None:
        vmin, vmax = _clip_range(arr, clip_low, clip_high)  # raises ValueError if no finite values

    u8      = _array_to_uint8(arr, vmin=vmin, vmax=vmax)
    lut     = lut_fn(colormap_key)
    colored = lut[u8]
    img     = _Image.fromarray(colored, mode="RGB")

    width_m = scan_range_m[0] if len(scan_range_m) >= 1 else 0.0
    Ny, Nx  = arr.shape

    if add_scalebar and width_m > 0:
        bar_m, bar_label = _pick_scalebar_length(
            width_m, Nx, target_frac=0.20, unit=scalebar_unit)

        bar_px = int(round(bar_m / width_m * Nx))
        bar_px = max(4, min(bar_px, Nx - 20))

        font_size = max(12, Ny // 40)
        font = None
        if _FONT_PATH.exists():
            try:
                font = _IFont.truetype(str(_FONT_PATH), size=font_size)
            except Exception:
                pass
        if font is None:
            font = _IFont.load_default()

        MARGIN      = 10
        BAR_HEIGHT  = max(4, Ny // 80)
        TEXT_GAP    = 3

        dummy_img  = _Image.new("RGB", (1, 1))
        dummy_draw = _IDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), bar_label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if scalebar_pos == 'bottom-left':
            bar_x0 = MARGIN
        else:
            bar_x0 = Nx - MARGIN - bar_px

        bar_y0 = Ny - MARGIN - BAR_HEIGHT
        bar_x1 = bar_x0 + bar_px
        bar_y1 = bar_y0 + BAR_HEIGHT

        text_x = bar_x0 + (bar_px - text_w) // 2
        text_y = bar_y0 - TEXT_GAP - text_h

        draw = _IDraw.Draw(img)

        draw.rectangle([bar_x0 - 1, bar_y0 - 1, bar_x1 + 1, bar_y1 + 1],
                       fill=(0, 0, 0))
        draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=(255, 255, 255))

        draw.text((text_x + 1, text_y + 1), bar_label, font=font,
                  fill=(0, 0, 0))
        draw.text((text_x, text_y), bar_label, font=font,
                  fill=(255, 255, 255))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="PNG")

    if provenance is not None:
        import json as _json
        sidecar = out_path.with_suffix("").with_suffix(".provenance.json")
        try:
            sidecar.write_text(
                _json.dumps(provenance.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass  # sidecar failure must never break the PNG export


# ═════════════════════════════════════════════════════════════════════════════
# 12.  tv_denoise  (Chambolle–Pock primal-dual, ported from AiSurf)
# ═════════════════════════════════════════════════════════════════════════════

# Features-tab placement note:
# ``tv_denoise`` is kept here as a GUI-free numerical kernel because it is also
# useful from the CLI and tests. When this becomes a GUI workflow, wire it from
# ``probeflow.gui.features`` as a specialized Features-tab operation, not as a
# Browse thumbnail correction and not as a normal Viewer quick-processing
# control. The intent is to keep experimental/optional add-ons isolated from
# routine browsing, conversion, and basic image manipulation dependencies.

def _nabla_apply(x: np.ndarray, Ny: int, Nx: int, comp: str) -> np.ndarray:
    """Forward gradient (periodic-edge) for TV methods.

    Returns a flattened (2*N,) vector with the x- and y-gradient stacked.
    """
    img = x.reshape(Ny, Nx)
    if comp in ("both", "x"):
        gx = np.zeros_like(img)
        gx[:, :-1] = img[:, 1:] - img[:, :-1]
    else:
        gx = np.zeros_like(img)
    if comp in ("both", "y"):
        gy = np.zeros_like(img)
        gy[:-1, :] = img[1:, :] - img[:-1, :]
    else:
        gy = np.zeros_like(img)
    return np.concatenate([gx.ravel(), gy.ravel()])


def _nabla_T_apply(p: np.ndarray, Ny: int, Nx: int, comp: str) -> np.ndarray:
    """Adjoint of the forward gradient (negative divergence)."""
    N = Ny * Nx
    px = p[:N].reshape(Ny, Nx)
    py = p[N:].reshape(Ny, Nx)

    div = np.zeros((Ny, Nx))
    if comp in ("both", "x"):
        # x-component of -div
        d = np.zeros_like(px)
        d[:, 0] = px[:, 0]
        d[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
        d[:, -1] = -px[:, -2]
        div -= d
    if comp in ("both", "y"):
        d = np.zeros_like(py)
        d[0, :] = py[0, :]
        d[1:-1, :] = py[1:-1, :] - py[:-2, :]
        d[-1, :] = -py[-2, :]
        div -= d
    return div.ravel()


def tv_denoise(
    arr: np.ndarray,
    *,
    method: str = "huber_rof",
    lam: float = 0.05,
    alpha: float = 0.05,
    tau: float = 0.25,
    max_iter: int = 500,
    tol: float = 5e-6,
    nabla_comp: str = "both",
) -> np.ndarray:
    """Edge-preserving total-variation denoising.

    Two variants are available, ported from AiSurf:

    * ``"huber_rof"``  — Huber-ROF (smooth TV). Good general-purpose default;
      preserves terraces without staircasing.
    * ``"tv_l1"``      — Isotropic TV-L1. More aggressive on impulsive noise,
      but staircases on gently curved terraces.

    Parameters
    ----------
    arr
        2-D float input (any range — no prior normalisation required).
    method
        ``"huber_rof"`` | ``"tv_l1"``.
    lam
        Data-fidelity weight λ. Larger values stay closer to the input.
    alpha
        Huber smoothing parameter (ignored for ``tv_l1``). Typical 0.01–0.1.
    tau
        Primal step size. Default 0.25 satisfies the Chambolle-Pock
        convergence condition ``τ·σ·L² ≤ 1`` (L = √8 here).
    max_iter
        Hard cap on iterations.
    tol
        RMSE convergence threshold between primal iterates (checked every 50).
    nabla_comp
        ``"both"`` (isotropic, default), ``"x"`` (removes vertical scratches),
        or ``"y"`` (removes horizontal scratches).

    Returns
    -------
    ndarray
        The denoised image, same shape and dtype as ``arr``.
    """
    if arr.ndim != 2:
        raise ValueError("tv_denoise expects a 2-D array")
    if nabla_comp not in ("both", "x", "y"):
        raise ValueError(f"nabla_comp must be 'both', 'x', or 'y', got {nabla_comp!r}")

    Ny, Nx = arr.shape
    f = arr.astype(np.float64, copy=True).ravel()
    u = f.copy()
    p = np.zeros(2 * Ny * Nx)

    L = math.sqrt(8.0)
    sigma = 1.0 / (tau * L * L)

    if method not in ("huber_rof", "tv_l1"):
        raise ValueError(f"Unknown method {method!r}")

    for it in range(max_iter + 1):
        u_old = u.copy()
        u = u - tau * _nabla_T_apply(p, Ny, Nx, nabla_comp)

        if method == "tv_l1":
            diff = u - f
            u = f + np.maximum(0.0, np.abs(diff) - tau * lam) * np.sign(diff)
            eff_alpha = 0.0
        else:  # huber_rof
            u = (u + tau * lam * f) / (1.0 + tau * lam)
            eff_alpha = alpha

        u_bar = 2.0 * u - u_old
        p = (p + sigma * _nabla_apply(u_bar, Ny, Nx, nabla_comp)) / (1.0 + sigma * eff_alpha)

        # Proximal projection onto the unit ball (isotropic TV).
        p2 = p.reshape(2, -1)
        norm = np.sqrt(p2[0] ** 2 + p2[1] ** 2)
        denom = np.maximum(1.0, norm)
        p = (p2 / denom[np.newaxis, :]).ravel()

        if it > 0 and it % 50 == 0:
            rmse = float(np.sqrt(np.mean((u - u_old) ** 2)))
            if rmse < tol:
                break

    return u.reshape(Ny, Nx).astype(arr.dtype, copy=False)


# ═════════════════════════════════════════════════════════════════════════════
# 13.  line_profile  — z values along a straight segment, with physical x-axis
# ═════════════════════════════════════════════════════════════════════════════

def line_profile(
    arr: np.ndarray,
    p0_px: "tuple[float, float] | None" = None,
    p1_px: "tuple[float, float] | None" = None,
    *,
    roi: "Any | None" = None,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    n_samples: Optional[int] = None,
    width_px: float = 1.0,
    interp: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``arr`` along a line segment.

    Parameters
    ----------
    arr
        2-D scan plane (any units).
    p0_px, p1_px
        Endpoint pixel coordinates ``(x, y)``. May be sub-pixel. Required
        when *roi* is ``None``.
    roi
        Optional :class:`probeflow.core.roi.ROI` with ``kind='line'``.  When
        provided, *p0_px* and *p1_px* are derived from the ROI geometry and
        must not be supplied separately.  Any other kind raises ``ValueError``.
    pixel_size_x_m, pixel_size_y_m
        Physical pixel spacing in metres along x and y. Used to express the
        sample axis in metres (handles non-square pixels correctly).
    n_samples
        Number of samples along the line. Default = ``ceil(geometric pixel
        length) + 1``.
    width_px
        Half-thickness of a perpendicular averaging band in pixels. ``1.0``
        samples the line itself; larger values average across a swath, which
        is useful for noisy traces.
    interp
        ``"linear"`` (default; via ``scipy.ndimage.map_coordinates`` order 1)
        or ``"nearest"`` (order 0).

    Returns
    -------
    s_m, z
        ``s_m`` — distance along the line in metres (length ``n_samples``).
        ``z`` — sampled values, one per ``s_m`` entry.
    """
    if roi is not None:
        if roi.kind != "line":
            raise ValueError(
                f"line_profile requires roi.kind='line', got {roi.kind!r}"
            )
        if p0_px is not None or p1_px is not None:
            raise ValueError(
                "Provide either roi or p0_px/p1_px, not both"
            )
        g = roi.geometry
        p0_px = (float(g["x1"]), float(g["y1"]))
        p1_px = (float(g["x2"]), float(g["y2"]))
    else:
        if p0_px is None or p1_px is None:
            raise ValueError("Either roi or both p0_px and p1_px must be provided")

    if arr.ndim != 2:
        raise ValueError("line_profile expects a 2-D array")
    if pixel_size_x_m <= 0 or pixel_size_y_m <= 0:
        raise ValueError("pixel_size_*_m must be > 0")
    if width_px < 1.0:
        raise ValueError("width_px must be >= 1.0")
    if interp not in ("linear", "nearest"):
        raise ValueError(f"Unknown interp {interp!r}")

    from scipy.ndimage import map_coordinates

    x0, y0 = float(p0_px[0]), float(p0_px[1])
    x1, y1 = float(p1_px[0]), float(p1_px[1])
    dx_px, dy_px = x1 - x0, y1 - y0
    px_len = float(np.hypot(dx_px, dy_px))
    if px_len < 1e-9:
        raise ValueError("p0 and p1 are the same point")

    if n_samples is None:
        n_samples = int(math.ceil(px_len)) + 1
    if n_samples < 2:
        n_samples = 2

    ts = np.linspace(0.0, 1.0, n_samples)
    xs = x0 + ts * dx_px
    ys = y0 + ts * dy_px

    order = 1 if interp == "linear" else 0

    if width_px <= 1.0:
        # ``map_coordinates`` takes (row, col) = (y, x).
        z = map_coordinates(arr, np.vstack([ys, xs]), order=order, mode="reflect")
    else:
        # Perpendicular unit vector in pixel space.
        nx, ny = -dy_px / px_len, dx_px / px_len
        n_perp = int(round(width_px))
        offsets = np.linspace(-(width_px - 1) / 2.0,
                              (width_px - 1) / 2.0, n_perp)
        accum = np.zeros(n_samples, dtype=np.float64)
        for off in offsets:
            ys_o = ys + off * ny
            xs_o = xs + off * nx
            accum += map_coordinates(arr, np.vstack([ys_o, xs_o]),
                                     order=order, mode="reflect")
        z = accum / n_perp

    # Physical distance: scale x and y components by their respective pixel sizes.
    dx_m = dx_px * pixel_size_x_m
    dy_m = dy_px * pixel_size_y_m
    seg_len_m = float(np.hypot(dx_m, dy_m))
    s_m = ts * seg_len_m
    return s_m, z.astype(arr.dtype, copy=False)


# ═════════════════════════════════════════════════════════════════════════════
# 14.  set_zero_point  (Gwyddion-style "Set Z=0 here")
# ═════════════════════════════════════════════════════════════════════════════

def set_zero_point(
    arr: np.ndarray,
    y_px: int,
    x_px: int,
    *,
    patch: int = 1,
) -> np.ndarray:
    """Subtract the mean of a small patch around ``(y_px, x_px)`` from the image.

    Parameters
    ----------
    arr
        2-D scan plane.
    y_px, x_px
        Pixel coordinates of the click. Coordinates outside the array are
        clipped to the nearest edge pixel.
    patch
        Half-size of the averaging window in pixels. ``patch=1`` averages a
        3×3 region (the default; matches Gwyddion's "Z=0 at pixel"). Use 0 to
        sample a single pixel.
    """
    if arr.ndim != 2:
        raise ValueError("set_zero_point expects a 2-D array")
    Ny, Nx = arr.shape
    y = max(0, min(int(y_px), Ny - 1))
    x = max(0, min(int(x_px), Nx - 1))
    p = max(0, int(patch))
    y0, y1 = max(0, y - p), min(Ny, y + p + 1)
    x0, x1 = max(0, x - p), min(Nx, x + p + 1)
    region = arr[y0:y1, x0:x1]
    finite = region[np.isfinite(region)]
    if finite.size == 0:
        return arr.astype(np.float64, copy=True)
    ref = float(finite.mean())
    return arr.astype(np.float64, copy=True) - ref


def set_zero_plane(
    arr: np.ndarray,
    points_px: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    patch: int = 1,
) -> np.ndarray:
    """Subtract the plane defined by three clicked reference points.

    ``points_px`` contains ``(x_px, y_px)`` coordinates.  The height at each
    point is estimated from a small finite-valued patch, then a plane
    ``z = ax + by + c`` is fitted through those three references and subtracted
    from the whole image.  This is a manual zero-plane operation, distinct from
    automatic polynomial/background subtraction.
    """
    if arr.ndim != 2:
        raise ValueError("set_zero_plane expects a 2-D array")
    if len(points_px) < 3:
        return arr.astype(np.float64, copy=True)

    a = arr.astype(np.float64, copy=True)
    Ny, Nx = a.shape
    p = max(0, int(patch))
    samples = []
    for point in points_px[:3]:
        try:
            x_px, y_px = int(point[0]), int(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        x_px = max(0, min(Nx - 1, x_px))
        y_px = max(0, min(Ny - 1, y_px))
        y0, y1 = max(0, y_px - p), min(Ny, y_px + p + 1)
        x0, x1 = max(0, x_px - p), min(Nx, x_px + p + 1)
        vals = a[y0:y1, x0:x1]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        samples.append((float(x_px), float(y_px), float(np.mean(finite))))

    if len(samples) < 3:
        return a

    A = np.array([[x, y, 1.0] for x, y, _z in samples], dtype=np.float64)
    if np.linalg.matrix_rank(A) < 3:
        return a
    z = np.array([z for _x, _y, z in samples], dtype=np.float64)
    coeffs = np.linalg.solve(A, z)
    yy, xx = np.mgrid[:Ny, :Nx]
    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    out = a - plane
    out[~np.isfinite(a)] = np.nan
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 15.  fft_soft_border  (ImageJ FFT_Soft_Border port)
# ═════════════════════════════════════════════════════════════════════════════

def fft_soft_border(
    arr: np.ndarray,
    *,
    mode: str = "low_pass",
    cutoff: float = 0.1,
    border_frac: float = 0.12,
) -> np.ndarray:
    """FFT filter with a Tukey-tapered border to suppress wrap-around ringing.

    Pixels within ``border_frac`` of any edge are smoothly tapered to the
    image mean before transforming, so the periodic continuation used by the
    DFT no longer carries a discontinuity. The taper is undone after the
    inverse transform so the interior is preserved.

    Parameters
    ----------
    arr
        2-D image.
    mode
        ``"low_pass"`` keeps frequencies inside ``cutoff``; ``"high_pass"``
        keeps frequencies outside.
    cutoff
        Radial frequency cut-off in fraction of Nyquist [0, 1].
    border_frac
        Fraction of each side over which the Tukey window ramps up.
    """
    if arr.ndim != 2:
        raise ValueError("fft_soft_border expects a 2-D array")
    if not 0.0 < border_frac < 0.5:
        raise ValueError("border_frac must be in (0, 0.5)")
    if mode not in {"low_pass", "high_pass"}:
        raise ValueError(f"Unknown mode: {mode!r}")
    if not np.isfinite(cutoff) or not 0.0 <= float(cutoff) <= 1.0:
        raise ValueError("cutoff must be finite and in [0, 1]")

    a = arr.astype(np.float64, copy=True)
    Ny, Nx = a.shape
    nan_mask = ~np.isfinite(a)
    mean_val = float(np.nanmean(a)) if (~nan_mask).any() else 0.0
    a[nan_mask] = mean_val

    def _tukey(n: int, frac: float) -> np.ndarray:
        w = np.ones(n)
        edge = max(1, int(round(frac * n)))
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, edge)))
        w[:edge] = ramp
        w[-edge:] = ramp[::-1]
        return w

    wy = _tukey(Ny, border_frac)
    wx = _tukey(Nx, border_frac)
    win2d = np.outer(wy, wx)

    centered = a - mean_val
    tapered = centered * win2d

    F = np.fft.fftshift(np.fft.fft2(tapered))
    cy, cx = Ny / 2.0, Nx / 2.0
    yr = (np.arange(Ny) - cy) / max(cy, 1e-9)
    xr = (np.arange(Nx) - cx) / max(cx, 1e-9)
    Xr, Yr = np.meshgrid(xr, yr)
    R = np.sqrt(Xr ** 2 + Yr ** 2)
    if mode == "low_pass":
        mask = (R <= cutoff).astype(np.float64)
    else:
        mask = (R >= cutoff).astype(np.float64)

    out = np.fft.ifft2(np.fft.ifftshift(F * mask)).real
    safe_win = np.where(win2d > 1e-6, win2d, 1.0)
    out = out / safe_win
    if mode == "low_pass":
        out = out + mean_val
    out[nan_mask] = np.nan
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 16.  fft_magnitude  — ROI-aware FFT power-spectrum visualisation
# ═════════════════════════════════════════════════════════════════════════════

def fft_magnitude(
    arr: np.ndarray,
    roi: "Any | None" = None,
    *,
    pixel_size_x_m: float = 1.0,
    pixel_size_y_m: float = 1.0,
    window: str = "hann",
    window_param: float = 0.25,
    log_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 2-D FFT magnitude spectrum of an image or a selected region.

    Parameters
    ----------
    arr
        2-D scan plane.
    roi
        Optional :class:`probeflow.core.roi.ROI`.  Three cases:

        - ``None`` — FFT the whole image.
        - ``roi.kind`` in ``{'rectangle', 'ellipse', 'polygon', 'freehand'}`` —
          crop to the bounding box, zero pixels outside the ROI mask, then FFT
          the crop.  For rectangular ROIs the mask covers the full crop.
        - ``roi.kind`` in ``{'line', 'point'}`` — raises ``ValueError``; these
          kinds do not define a 2-D region.

    pixel_size_x_m, pixel_size_y_m
        Physical pixel spacing in metres.  Used to express the k-space axes in
        reciprocal nanometres (nm⁻¹).  The output array resolution is set by
        the ROI crop, not the full image; pass the same pixel size as the parent
        scan (pixel size is invariant to cropping).
    window
        Spatial-domain windowing applied before the DFT.

        - ``"hann"`` — 2-D outer-product Hann window.  Recommended default:
          suppresses the dominant wrap-around artefact at the price of a modest
          spectral-resolution loss.
        - ``"tukey"`` — 2-D Tukey window.  ``window_param`` controls the
          plateau fraction (0 = Hann, 1 = rectangular).  Useful when the ROI
          contains a clean periodic structure that fills most of the crop.
        - ``"none"`` — no window.  Use only when the data is already periodic
          across the boundary (rare) or when comparing raw amplitudes.

    window_param
        Shape parameter for the Tukey window (ignored for other windows).
        Default 0.25.
    log_scale
        If ``True`` (default), the returned magnitude is ``log1p(|F|)``.  This
        matches the GUI convention and compresses the dynamic range for display.
        Pass ``False`` to obtain linear amplitudes for quantitative use.

    Returns
    -------
    magnitude : np.ndarray, float64
        2-D magnitude (or log-magnitude) array with the DC term centred.
        Shape matches the bounding-box crop of the ROI (or the full image when
        ``roi`` is ``None``).
    qx : np.ndarray, float64
        1-D array of k-space frequencies in nm⁻¹ along the x axis (columns),
        centred at DC.
    qy : np.ndarray, float64
        1-D array of k-space frequencies in nm⁻¹ along the y axis (rows),
        centred at DC.

    Notes
    -----
    The output resolution is ``(Ny_crop, Nx_crop)`` where ``Ny_crop × Nx_crop``
    is the bounding-box crop of the ROI.  A 64×64 ROI on a 512×512 scan
    produces a 64×64 FFT; the k-space *range* shrinks but the pixel size (and
    therefore the Nyquist limit) is unchanged.
    """
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from probeflow.core.roi import ROI

    _AREA_KINDS = {"rectangle", "ellipse", "polygon", "freehand"}
    _INVALID_KINDS = {"line", "point"}

    if roi is not None and roi.kind in _INVALID_KINDS:
        raise ValueError(
            f"fft_magnitude does not support roi.kind={roi.kind!r}. "
            "Supported kinds: rectangle, ellipse, polygon, freehand, or None."
        )

    a = arr.astype(np.float64, copy=True)
    Ny_full, Nx_full = a.shape

    # ── 1. Crop to ROI bounding box and apply mask ────────────────────────────
    if roi is None:
        crop = a
        mask = None
    else:
        r0, r1, c0, c1 = roi.bounds(a.shape)  # (row_min, row_max, col_min, col_max)
        crop = a[r0:r1 + 1, c0:c1 + 1].copy()
        if roi.kind != "rectangle":
            full_mask = roi.to_mask(a.shape)
            mask = full_mask[r0:r1 + 1, c0:c1 + 1]
        else:
            mask = None

    Ny, Nx = crop.shape

    # ── 2. Replace NaN / Inf with the local mean ──────────────────────────────
    nan_mask = ~np.isfinite(crop)
    mean_val = float(np.nanmean(crop)) if (~nan_mask).any() else 0.0
    crop = np.where(nan_mask, mean_val, crop)

    # ── 3. Zero outside non-rectangular ROI ──────────────────────────────────
    if mask is not None:
        crop = np.where(mask, crop, 0.0)

    # ── 4. Remove mean ────────────────────────────────────────────────────────
    crop = crop - crop.mean()

    # ── 5. Apply spatial window ───────────────────────────────────────────────
    wkey = str(window).lower()
    if wkey not in {"hann", "tukey", "none"}:
        raise ValueError(f"window must be 'hann', 'tukey', or 'none'; got {window!r}")

    if wkey == "hann":
        win2d = np.outer(np.hanning(Ny), np.hanning(Nx))
    elif wkey == "tukey":
        def _tukey_1d(n: int, alpha: float) -> np.ndarray:
            if n < 2:
                return np.ones(n)
            w = np.ones(n)
            edge = max(1, int(round(alpha / 2.0 * n)))
            ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, edge)))
            w[:edge] = ramp
            w[-edge:] = ramp[::-1]
            return w
        win2d = np.outer(_tukey_1d(Ny, window_param), _tukey_1d(Nx, window_param))
    else:
        win2d = np.ones((Ny, Nx), dtype=np.float64)

    windowed = crop * win2d

    # ── 6. Compute FFT and shift DC to centre ─────────────────────────────────
    F = np.fft.fftshift(np.fft.fft2(windowed))
    mag = np.abs(F)

    if log_scale:
        mag = np.log1p(mag)

    # ── 7. Frequency axes in nm⁻¹ ────────────────────────────────────────────
    dx_nm = pixel_size_x_m * 1e9
    dy_nm = pixel_size_y_m * 1e9
    qx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx_nm))
    qy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy_nm))

    return mag.astype(np.float64), qx, qy


# ═════════════════════════════════════════════════════════════════════════════
# 17.  patch_interpolate  (ImageJ Patch_Interpolation port)
# ═════════════════════════════════════════════════════════════════════════════

def patch_interpolate(
    arr: np.ndarray,
    mask: np.ndarray,
    *,
    method: str = "line_fit",
    rim_px: int = 20,
    iterations: int = 200,
) -> np.ndarray:
    """Fill masked pixels by interpolation from surrounding data.

    Two interpolation strategies are available:

    ``method="line_fit"`` (default)
        For each masked row, fits a linear function (offset + slope × x) to
        the non-masked pixels within *rim_px* columns of the masked boundary,
        then extrapolates that line through the masked columns.  This
        preserves the local surface tilt — physically correct for repairing
        STM scan lines, where nearby scan lines share the same slope.
        Algorithm after Schmid's ImageJ ``Patch_Interpolation`` plugin
        (mode: "lines with individual slopes").  Rows where no rim data is
        available fall back to row-blended interpolation from neighbours.

    ``method="laplace"``
        Iterative Jacobi relaxation of the discrete Laplace equation: each
        masked pixel converges to the average of its four neighbours.
        Isotropic and smooth, but does not preserve scan-line slope.  Use
        for non-directional patches or when the masked region spans most of
        the image.

    Parameters
    ----------
    arr
        2-D image.
    mask
        Boolean array; True for pixels to interpolate.
    method
        ``"line_fit"`` (default) or ``"laplace"``.
    rim_px
        Half-width (in columns) of the fitting region on each side of the
        masked boundary.  Used only by ``"line_fit"``.
    iterations
        Number of Jacobi relaxation passes.  Used only by ``"laplace"``.
        100–400 is normally enough.
    """
    if arr.shape != mask.shape:
        raise ValueError("arr and mask must have the same shape")
    if method not in {"line_fit", "laplace"}:
        raise ValueError(f"method must be 'line_fit' or 'laplace', got {method!r}")
    if method == "line_fit":
        return _patch_line_fit(arr, mask, rim_px=max(1, int(rim_px)))
    return _patch_laplace(arr, mask, iterations=max(1, int(iterations)))


def _patch_line_fit(arr: np.ndarray, mask: np.ndarray, rim_px: int) -> np.ndarray:
    """Per-row linear extrapolation from rim pixels (ImageJ Patch_Interpolation style)."""
    a = arr.astype(np.float64, copy=True)
    m = mask.astype(bool)
    if not m.any():
        return a

    Ny, Nx = a.shape
    xs = np.arange(Nx, dtype=np.float64)

    fill = float(np.nanmean(a[~m & np.isfinite(a)])) if (~m).any() else 0.0
    a[m | ~np.isfinite(a)] = fill

    deferred_rows: list[int] = []

    for r in range(Ny):
        row_mask = m[r]
        if not row_mask.any():
            continue

        orig_row = arr[r].astype(np.float64)
        good = ~m[r] & np.isfinite(orig_row)

        if not good.any():
            deferred_rows.append(r)
            continue

        masked_xs = xs[row_mask]
        col_lo = float(masked_xs.min())
        col_hi = float(masked_xs.max())

        rim_left  = good & (xs <= col_lo) & (xs >= max(0.0, col_lo - rim_px))
        rim_right = good & (xs >= col_hi) & (xs <= min(float(Nx - 1), col_hi + rim_px))
        rim = rim_left | rim_right

        if rim.sum() >= 2:
            fx, fz = xs[rim], orig_row[rim]
        elif good.sum() >= 2:
            fx, fz = xs[good], orig_row[good]
        else:
            a[r][row_mask] = float(orig_row[good].mean())
            continue

        if fx.size == 1:
            a[r][row_mask] = float(fz[0])
        else:
            coeffs = np.polyfit(fx, fz, 1)
            slope, intercept = float(coeffs[0]), float(coeffs[1])
            a[r][row_mask] = slope * xs[row_mask] + intercept

    for r in deferred_rows:
        above = [rr for rr in range(r - 1, -1, -1) if not m[rr].all()]
        below = [rr for rr in range(r + 1, Ny) if not m[rr].all()]
        if above and below:
            ra, rb = above[0], below[0]
            da, db = r - ra, rb - r
            a[r] = (db * a[ra] + da * a[rb]) / (da + db)
        elif above:
            a[r] = a[above[0]]
        elif below:
            a[r] = a[below[0]]

    return a


def _patch_laplace(arr: np.ndarray, mask: np.ndarray, iterations: int) -> np.ndarray:
    """Jacobi relaxation of the Laplace equation over masked pixels."""
    a = arr.astype(np.float64, copy=True)
    m = mask.astype(bool)
    if not m.any():
        return a

    fill = float(np.nanmean(a[~m & np.isfinite(a)])) if (~m).any() else 0.0
    a[m | ~np.isfinite(a)] = fill
    Ny, Nx = a.shape
    for _ in range(iterations):
        nb = np.zeros_like(a)
        nb[1:-1, 1:-1] = 0.25 * (
            a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
        )
        nb[0, :] = a[0, :]
        nb[-1, :] = a[-1, :]
        nb[:, 0] = a[:, 0]
        nb[:, -1] = a[:, -1]
        a = np.where(m, nb, a)
    return a


# ═════════════════════════════════════════════════════════════════════════════
# 17.  linear_undistort  (ImageJ Linear_Undistort port)
# ═════════════════════════════════════════════════════════════════════════════

def linear_undistort(
    arr: np.ndarray,
    *,
    shear_x: float = 0.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    """Apply an affine drift/creep correction to a scan plane.

    The forward map shifts column ``c`` by ``shear_x * (row / Ny)`` pixels and
    rescales the row coordinate by ``scale_y``. Inverse-mapped via
    ``scipy.ndimage.map_coordinates`` so every output pixel comes from one
    bilinearly-interpolated location in the input.

    Parameters
    ----------
    arr
        2-D image.
    shear_x
        Total horizontal drift across the slow-scan axis, in pixels (so
        ``shear_x = 5`` means the bottom row is sheared 5 px to the right).
    scale_y
        Multiplicative scaling of the slow-scan axis (1.0 = no change).
    """
    from scipy.ndimage import map_coordinates

    if arr.ndim != 2:
        raise ValueError("linear_undistort expects a 2-D array")
    if scale_y <= 0:
        raise ValueError("scale_y must be > 0")
    a = arr.astype(np.float64, copy=True)
    nan_mask = ~np.isfinite(a)
    if nan_mask.any():
        a[nan_mask] = float(np.nanmean(a))
    Ny, Nx = a.shape
    yy, xx = np.indices(a.shape).astype(np.float64)
    src_y = yy / max(scale_y, 1e-9)
    src_x = xx - shear_x * (yy / max(Ny - 1, 1))
    out = map_coordinates(
        a, np.vstack([src_y.ravel(), src_x.ravel()]),
        order=1, mode="reflect",
    ).reshape(Ny, Nx)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 18.  blend_forward_backward  (ImageJ Blend_Images port)
# ═════════════════════════════════════════════════════════════════════════════

def blend_forward_backward(
    fwd: np.ndarray,
    bwd: np.ndarray,
    *,
    weight: float = 0.5,
) -> np.ndarray:
    """Blend a forward-scan plane with a horizontally-mirrored backward plane.

    Parameters
    ----------
    fwd, bwd
        2-D arrays of the same shape. ``bwd`` is left-right flipped before
        blending so the same physical location overlaps in both planes.
    weight
        Weight of the forward plane in [0, 1]. ``0.5`` is a symmetric mean.
    """
    if fwd.shape != bwd.shape:
        raise ValueError("fwd and bwd must have the same shape")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    f = fwd.astype(np.float64, copy=True)
    b = np.fliplr(bwd.astype(np.float64, copy=True))
    out = weight * f + (1.0 - weight) * b
    nan_mask = ~np.isfinite(f) | ~np.isfinite(b)
    if nan_mask.any():
        out[nan_mask] = np.where(np.isfinite(f), f, b)[nan_mask]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 19.  Geometric transforms — flip and rotation
#
# ROI-under-transformation conventions (for Phase 1 reference):
#
# - flip_horizontal, flip_vertical, rotate_90_cw, rotate_180, rotate_270_cw:
#     ROI geometry is transformed to the new pixel-coordinate system. These are
#     exact, lossless transformations. Phase 1 should update ROI pixel
#     coordinates when these ops are applied.
#
# - rotate_arbitrary:
#     Existing ROIs are INVALIDATED and must be removed or marked invalid.
#     Reason: floating-point geometry + bilinear-interpolated pixels make
#     round-tripping unreliable. The caller (apply_processing_state) warns and
#     removes any roi steps when rotate_arbitrary is encountered in the state.
#
# - crop (future):
#     ROI geometry is transformed to the new coordinate system. ROIs entirely
#     outside the crop are dropped; ROIs partially outside are clipped.
# ═════════════════════════════════════════════════════════════════════════════

def flip_horizontal(arr: np.ndarray) -> np.ndarray:
    """Flip the scan left-to-right (mirror about the vertical axis)."""
    return np.fliplr(arr.astype(np.float64, copy=True))


def flip_vertical(arr: np.ndarray) -> np.ndarray:
    """Flip the scan top-to-bottom (mirror about the horizontal axis)."""
    return np.flipud(arr.astype(np.float64, copy=True))


def rotate_90_cw(arr: np.ndarray) -> np.ndarray:
    """Rotate the scan 90° clockwise. Swaps width and height."""
    return np.rot90(arr.astype(np.float64, copy=True), k=3)


def rotate_180(arr: np.ndarray) -> np.ndarray:
    """Rotate the scan 180°. Preserves width and height."""
    return np.rot90(arr.astype(np.float64, copy=True), k=2)


def rotate_270_cw(arr: np.ndarray) -> np.ndarray:
    """Rotate the scan 270° clockwise (= 90° counter-clockwise). Swaps width and height."""
    return np.rot90(arr.astype(np.float64, copy=True), k=1)


def rotate_arbitrary(
    arr: np.ndarray,
    angle_degrees: float,
    *,
    order: int = 1,
) -> np.ndarray:
    """Rotate the scan by an arbitrary angle with canvas expansion.

    Positive angles are counter-clockwise (standard mathematical convention).
    The output canvas is enlarged to contain the entire rotated image; newly
    introduced pixels are zero. No input pixels are lost.

    Uses bilinear interpolation (``order=1``) by default, which is appropriate
    for STM topography data: it preserves smooth gradients without the ringing
    that cubic introduces near step edges.

    Parameters
    ----------
    arr
        2-D scan plane.
    angle_degrees
        Rotation angle in degrees. Positive = counter-clockwise.
    order
        Interpolation order: 0=nearest, 1=bilinear (default), 2=quadratic,
        3=bicubic. Order 1 is recommended for STM topography.
    """
    if not isinstance(order, int) or order < 0 or order > 3:
        raise ValueError(f"order must be 0–3, got {order!r}")
    from scipy.ndimage import rotate as _ndimage_rotate
    a = arr.astype(np.float64, copy=True)
    return _ndimage_rotate(a, float(angle_degrees), reshape=True, order=order,
                           mode='constant', cval=0.0)
