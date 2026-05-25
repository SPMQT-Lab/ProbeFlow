"""Spatial and frequency-domain image filters."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    laplace as _nd_laplace,
    maximum_filter,
)

from ._image_utils import (
    _finite_mean,
    _finite_median,
    _nonnegative_finite,
    _positive_finite,
    _nan_normalized_gaussian,
)


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
        out = filled.copy()
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
    result = np.fft.ifft2(F_filtered).real + mean_val
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
    out = out / safe_win + mean_val
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
    # NOTE: Window reduces spectral leakage but suppresses apparent signal amplitude.
    # Hann window power is ~0.375 (2D outer product); magnitude is NOT normalised by
    # window correction. This is acceptable for qualitative display but not for
    # quantitative spectral analysis. For calibrated measurements, divide by
    # window_correction = 2.67 (approximate for Hann 2D).
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
# 17.  predicted_bragg_radius  — predicted Bragg peak radius for lattice overlay
# ═════════════════════════════════════════════════════════════════════════════

def predicted_bragg_radius(
    a_real: float,
    symmetry: str,
    scan_size_m: float,
    n_pixels: int,
    order: int = 1,
) -> float:
    """Return the predicted Bragg peak radius in FFT pixel units, measured
    from the DC bin at the centre.

    The function is unit-agnostic above metres: ``a_real`` and ``scan_size_m``
    must be in the **same** length unit (typically metres). The GUI layer is
    responsible for converting user-facing units (Å, nm) before calling this.

    Parameters
    ----------
    a_real
        Real-space lattice constant. For a hexagonal lattice this is the
        nearest-neighbour distance. Must be positive.
    symmetry
        ``"square"`` or ``"hex"`` (hexagonal/triangular).
    scan_size_m
        Physical scan size in the same unit as ``a_real``. Assumes a square
        scan; for non-square scans the caller should pass the geometric mean.
    n_pixels
        Image side in pixels (assumes a square image). Present in the
        signature for interface completeness; not used in the current formulas.
    order
        Bragg order: 1 for first-order peaks, 2 for second-order.

    Returns
    -------
    float
        Radius in FFT pixel units from the DC bin at the image centre.

    Raises
    ------
    ValueError
        On any invalid input.

    Notes
    -----
    Square lattice, order 1:  r = scan_size_m / a_real
    Square lattice, order 2:  r = scan_size_m / a_real * sqrt(2)   (diagonal)
    Hex lattice, order 1:     r = 2 * scan_size_m / (a_real * sqrt(3))
    Hex lattice, order 2:     r = 2 * scan_size_m / a_real          (= order_1 * sqrt(3))
    """
    if a_real <= 0:
        raise ValueError(f"a_real must be > 0, got {a_real!r}")
    if symmetry not in {"square", "hex"}:
        raise ValueError(f"symmetry must be 'square' or 'hex', got {symmetry!r}")
    if scan_size_m <= 0:
        raise ValueError(f"scan_size_m must be > 0, got {scan_size_m!r}")
    if n_pixels <= 0:
        raise ValueError(f"n_pixels must be > 0, got {n_pixels!r}")
    if order not in {1, 2}:
        raise ValueError(f"order must be 1 or 2, got {order!r}")

    if symmetry == "square":
        r = scan_size_m / a_real * (1.0 if order == 1 else math.sqrt(2.0))
    else:  # hex
        r = (2.0 * scan_size_m / (a_real * math.sqrt(3.0))
             * (1.0 if order == 1 else math.sqrt(3.0)))
    return r


# ═════════════════════════════════════════════════════════════════════════════
# 18.  find_bragg_peaks_in_annulus  — auto-detect Bragg peaks in FFT
# ═════════════════════════════════════════════════════════════════════════════

def find_bragg_peaks_in_annulus(
    fft_magnitude: np.ndarray,
    r_predicted_px: float,
    width_frac: float = 0.20,
    expected_count: int = 6,
    min_separation_frac: float = 0.30,
) -> np.ndarray:
    """Find local-maximum peaks inside an annulus around a predicted Bragg radius.

    Parameters
    ----------
    fft_magnitude
        2-D FFT magnitude array with DC at the centre (after fftshift). Not
        log-scaled; the raw ``|F|`` values are expected.
    r_predicted_px
        Predicted Bragg peak radius in pixel units, measured from the array
        centre.
    width_frac
        Half-width of the search annulus as a fraction of ``r_predicted_px``.
        The annulus spans ``r_predicted_px * (1 - width_frac)`` to
        ``r_predicted_px * (1 + width_frac)``.
    expected_count
        How many peaks to try to return.  Typically 4 (square) or 6 (hex).
    min_separation_frac
        Minimum angular separation between kept peaks, expressed as a fraction
        of ``2π / expected_count``.  Prevents returning two picks from the
        same broadened peak.

    Returns
    -------
    np.ndarray, shape (M, 2)
        (x, y) pixel offsets from the array centre, ordered brightest-first.
        ``M ≤ expected_count``.  Returns an empty array of shape ``(0, 2)``
        if no local maxima are found.
    """
    mag = np.asarray(fft_magnitude, dtype=np.float64)
    if mag.ndim != 2:
        raise ValueError("fft_magnitude must be a 2-D array")
    if r_predicted_px <= 0:
        return np.empty((0, 2), dtype=np.float64)

    Ny, Nx = mag.shape
    cy, cx = Ny / 2.0, Nx / 2.0

    yy, xx = np.ogrid[:Ny, :Nx]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    r_in  = max(0.0, r_predicted_px * (1.0 - width_frac))
    r_out = r_predicted_px * (1.0 + width_frac)
    annulus_mask = (r >= r_in) & (r <= r_out)

    # Suppress local noise: footprint ~ half the annulus thickness
    footprint_size = max(3, int(r_predicted_px * width_frac))
    local_max_img = maximum_filter(mag, size=footprint_size)
    candidate_mask = annulus_mask & (mag == local_max_img)

    cys, cxs = np.where(candidate_mask)
    if cys.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Sort candidates by magnitude (brightest first)
    vals = mag[cys, cxs]
    order = np.argsort(vals)[::-1]
    cys = cys[order]
    cxs = cxs[order]

    # Angular separation filter
    min_angle = min_separation_frac * (2.0 * math.pi / max(1, expected_count))
    kept_angles: list[float] = []
    results: list[tuple[float, float]] = []

    for iy, ix in zip(cys, cxs):
        angle = math.atan2(float(iy) - cy, float(ix) - cx)
        ok = True
        for prev in kept_angles:
            diff = abs(angle - prev)
            if diff > math.pi:
                diff = 2.0 * math.pi - diff
            if diff < min_angle:
                ok = False
                break
        if ok:
            results.append((float(ix) - cx, float(iy) - cy))
            kept_angles.append(angle)
            if len(results) >= expected_count:
                break

    if not results:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(results, dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# 19.  fit_axis_aligned_ellipse  — least-squares ellipse from Bragg picks
# ═════════════════════════════════════════════════════════════════════════════

def fit_axis_aligned_ellipse(
    points: np.ndarray,
) -> tuple[float, float, float]:
    """Fit an axis-aligned ellipse (x/r_x)² + (y/r_y)² = 1 to the given points.

    Uses linear least squares on the substituted variables ``u = 1/r_x²`` and
    ``v = 1/r_y²``, so the design matrix is ``[x², y²]`` and the right-hand
    side is the constant vector ``1``.

    Parameters
    ----------
    points
        Array of shape ``(M, 2)`` containing ``(x, y)`` coordinates measured
        from the ellipse centre.  Must contain at least three points.

    Returns
    -------
    r_x : float
        Fitted semi-axis along x, in the same units as the input coordinates.
    r_y : float
        Fitted semi-axis along y.
    rms_residual : float
        RMS distance (in input units) between each point and its closest point
        on the fitted ellipse, computed radially along the direction of each
        point from the origin.

    Raises
    ------
    ValueError
        If fewer than three points are supplied, or if the fit is degenerate
        (one of the estimated ``1/r²`` values is ≤ 0).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"points must have shape (M, 2), got {pts.shape}"
        )
    M = pts.shape[0]
    if M < 3:
        raise ValueError(f"Need at least 3 points to fit an ellipse, got {M}")

    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([x ** 2, y ** 2])
    b = np.ones(M, dtype=np.float64)
    (u, v), *_ = np.linalg.lstsq(A, b, rcond=None)

    if u <= 0:
        raise ValueError(
            f"Degenerate fit: 1/r_x² = {u:.6g} ≤ 0; "
            "check that picks are not all on the y-axis"
        )
    if v <= 0:
        raise ValueError(
            f"Degenerate fit: 1/r_y² = {v:.6g} ≤ 0; "
            "check that picks are not all on the x-axis"
        )

    r_x = 1.0 / math.sqrt(u)
    r_y = 1.0 / math.sqrt(v)

    # RMS residual: radial distance between each observed point and the
    # ellipse evaluated at the same angle.
    angles = np.arctan2(y, x)
    r_fit = 1.0 / np.sqrt(u * np.cos(angles) ** 2 + v * np.sin(angles) ** 2)
    r_obs = np.hypot(x, y)
    rms = float(np.sqrt(np.mean((r_obs - r_fit) ** 2)))

    return r_x, r_y, rms


# ═════════════════════════════════════════════════════════════════════════════
# 20.  piezo_correction  — convert observed/predicted radii to new piezo values
# ═════════════════════════════════════════════════════════════════════════════

def piezo_correction(
    r_x_obs: float,
    r_y_obs: float,
    r_predicted: float,
    c_x_current: float,
    c_y_current: float,
) -> tuple[float, float]:
    """Compute corrected piezo constants from observed Bragg semi-axes.

    The function is unit-agnostic: ``r_x_obs``, ``r_y_obs``, and
    ``r_predicted`` must be in the same unit (e.g., all in nm⁻¹ or all in
    FFT pixel offsets); ``c_x_current`` and ``c_y_current`` must be in the
    same unit as each other (e.g., both in Å/V).

    Parameters
    ----------
    r_x_obs
        Observed Bragg semi-axis along x from the ellipse fit.
    r_y_obs
        Observed Bragg semi-axis along y.
    r_predicted
        Predicted isotropic Bragg radius (same unit as the observed axes).
    c_x_current
        Current piezo constant for the x axis.
    c_y_current
        Current piezo constant for the y axis.

    Returns
    -------
    c_x_new, c_y_new : float, float
        Recommended corrected piezo constants.

    Raises
    ------
    ValueError
        If any input is not strictly positive.
    """
    for name, val in [
        ("r_x_obs",     r_x_obs),
        ("r_y_obs",     r_y_obs),
        ("r_predicted", r_predicted),
        ("c_x_current", c_x_current),
        ("c_y_current", c_y_current),
    ]:
        if not (val > 0):
            raise ValueError(f"{name} must be > 0, got {val!r}")

    return c_x_current * (r_x_obs / r_predicted), c_y_current * (r_y_obs / r_predicted)
