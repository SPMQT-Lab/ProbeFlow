"""Spatial and frequency-domain image filters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    laplace as _nd_laplace,
    maximum_filter,
)
from scipy.signal import windows as _sp_windows

from ._image_utils import (
    _finite_mean,
    _finite_median,
    _nonnegative_finite,
    _positive_finite,
    _nan_normalized_gaussian,
)


# ─── Shared window helper ────────────────────────────────────────────────────
#
# Both ``fourier_filter`` and ``fft_magnitude`` previously accepted slightly
# different window vocabularies ("hanning"/"hamming"/"none" vs "hann"/"tukey"/
# "none") with subtly different implementations.  The 2026-05-27 deep review
# (image-proc #9) flagged this as a reproducibility hazard — picking a window
# in the FFT viewer and then applying a filter with the "same" window in the
# pipeline silently used a different shape.  This helper centralises window
# selection on the canonical periodic-by-default ``scipy.signal.windows.*``
# vocabulary and accepts the old names as aliases for backward compatibility.
#
# Periodic vs symmetric windows: the DFT assumes the windowed segment is
# periodic, so for spectral analysis we want ``sym=False`` (the periodic
# variant).  ``np.hanning(N)`` and ``np.hamming(N)`` return the symmetric
# variant with explicit zero endpoints, which subtly biases the bin-centre
# frequency mapping and depresses two pixels at each boundary.  Review
# physics #6 (fixed 2026-05-28).

_WINDOW_ALIASES = {
    "hann": "hann",
    "hanning": "hann",       # numpy/legacy alias
    "hamming": "hamming",
    "tukey": "tukey",
    "none": "none",
    "rectangular": "none",
    "boxcar": "none",
}


def _resolve_window_name(window: "str | None") -> str:
    """Return the canonical (lower-case) window name, accepting old aliases."""
    key = str(window or "none").lower()
    if key not in _WINDOW_ALIASES:
        raise ValueError(
            f"window must be one of {sorted(set(_WINDOW_ALIASES))!r}; got {window!r}"
        )
    return _WINDOW_ALIASES[key]


def _window_1d(name: str, n: int, *, alpha: float = 0.25) -> np.ndarray:
    """Build a 1-D window of length *n*, periodic (sym=False) for DFT use.

    *name* must be a canonical name from :data:`_WINDOW_ALIASES`'s values.
    """
    if n < 1:
        return np.ones(max(n, 0), dtype=np.float64)
    if name == "hann":
        return np.asarray(_sp_windows.hann(n, sym=False), dtype=np.float64)
    if name == "hamming":
        return np.asarray(_sp_windows.hamming(n, sym=False), dtype=np.float64)
    if name == "tukey":
        return np.asarray(
            _sp_windows.tukey(n, alpha=float(alpha), sym=False),
            dtype=np.float64,
        )
    # "none" / rectangular
    return np.ones(n, dtype=np.float64)


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

    cutoff  — fraction of Nyquist [0, 1].  Defined in true cycles/pixel
              (review physics #7): a value of ``1.0`` corresponds to the
              Nyquist limit along either axis.  For rectangular images
              the radial mask is isotropic in cycles/pixel — previously
              it was normalised to half-axis, so a "circular" cutoff
              acted as an ellipse in q-space on non-square scans.
    mode    — 'low_pass' | 'high_pass'

    window  — 'hann' (alias 'hanning'), 'hamming', 'tukey', or 'none'
              (review image-proc #9 unified the vocabulary with
              :func:`fft_magnitude`).  Uses ``scipy.signal.windows``
              with ``sym=False`` (periodic) so the DFT periodicity
              assumption is satisfied (review physics #6).

    This is a coarse circular frequency cutoff, not an ImageJ-style periodic
    spot/vector filter.
    """
    if mode not in {"low_pass", "high_pass"}:
        raise ValueError("mode must be 'low_pass' or 'high_pass'")
    if not np.isfinite(cutoff) or not 0.0 <= float(cutoff) <= 1.0:
        raise ValueError("cutoff must be finite and in [0, 1]")
    window_canonical = _resolve_window_name(window)

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

    wy = _window_1d(window_canonical, Ny)
    wx = _window_1d(window_canonical, Nx)
    win2d = np.outer(wy, wx)
    windowed = centered * win2d

    F = np.fft.fft2(windowed)
    F = np.fft.fftshift(F)

    # Radial mask in true frequency units (cycles/pixel), so the cutoff
    # is isotropic in q-space regardless of image aspect ratio.  Review
    # physics #7: the old normalisation R = sqrt((Δx/cx)² + (Δy/cy)²)
    # was distorted into an ellipse on non-square scans.
    qx = np.fft.fftshift(np.fft.fftfreq(Nx))
    qy = np.fft.fftshift(np.fft.fftfreq(Ny))
    Qx, Qy = np.meshgrid(qx, qy)
    # fftfreq normalises so Nyquist = 0.5; scale so cutoff=1.0 maps to Nyquist.
    R = np.sqrt(Qx ** 2 + Qy ** 2) / 0.5

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
    if not nan_mask.any():
        return gaussian_filter(arr, sigma=sigma_px, mode="reflect")
    smoothed = _nan_normalized_gaussian(arr, sigma_px)
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
        Spatial-domain windowing applied before the DFT.  Vocabulary
        unified with :func:`fourier_filter` (review image-proc #9).

        - ``"hann"`` (alias ``"hanning"``) — 2-D outer-product Hann window,
          periodic (``scipy.signal.windows.hann(N, sym=False)``).
          Recommended default: suppresses the dominant wrap-around
          artefact at the price of a modest spectral-resolution loss.
        - ``"hamming"`` — 2-D periodic Hamming window.
        - ``"tukey"`` — 2-D periodic Tukey window.  ``window_param``
          controls the plateau fraction (0 = Hann, 1 = rectangular).
          Useful when the ROI contains a clean periodic structure that
          fills most of the crop.
        - ``"none"`` (alias ``"rectangular"``, ``"boxcar"``) — no window.
          Use only when the data is already periodic across the boundary
          (rare) or when comparing raw amplitudes.

        The returned magnitude is window-corrected by the coherent gain
        ``Ny * Nx / sum(win2d)`` (review physics #5) so switching window
        does not change the absolute magnitude scale of a given physical
        input.

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
    # The mean is computed over FINITE pixels inside the ROI when one is set,
    # so the DC-subtraction at step 4 only counts genuine signal — fixing the
    # biased subtraction the previous implementation introduced (review
    # physics #1, fixed 2026-05-28).
    nan_mask = ~np.isfinite(crop)
    if mask is not None:
        inside_finite = (~nan_mask) & mask
    else:
        inside_finite = ~nan_mask
    if inside_finite.any():
        inside_mean = float(crop[inside_finite].mean())
    else:
        inside_mean = 0.0
    # Fill NaN inside the ROI with the inside-mean (so interpolation-style
    # leakage from NaN pixels is minimised).  Outside the ROI we zero at
    # step 3 anyway.
    crop = np.where(nan_mask, inside_mean, crop)

    # ── 3. Subtract the inside-ROI mean from inside-ROI pixels only ──────────
    # Previously the code zeroed outside-ROI pixels first, then subtracted
    # ``crop.mean()`` (averaged over the full crop *including* zeros) from
    # every pixel.  That left a low-k cross/ring artefact in the magnitude
    # spectrum because the boundary discontinuity grew with the ROI fill
    # fraction.  Correct: subtract the genuine inside-mean from inside-ROI
    # pixels and leave outside-ROI at zero (review physics #1).
    if mask is not None:
        crop = np.where(mask, crop - inside_mean, 0.0)
    else:
        crop = crop - inside_mean

    # ── 4. Apply spatial window ───────────────────────────────────────────────
    # Review image-proc #9: unified window vocabulary with fourier_filter
    # via the shared ``_resolve_window_name`` / ``_window_1d`` helpers.
    # Review physics #6: ``_window_1d`` uses scipy's periodic (sym=False)
    # variants, which match the DFT periodicity assumption — the previous
    # ``np.hanning`` produced symmetric zero-endpoint windows that subtly
    # depressed two boundary pixels.
    window_canonical = _resolve_window_name(window)
    wy = _window_1d(window_canonical, Ny, alpha=float(window_param))
    wx = _window_1d(window_canonical, Nx, alpha=float(window_param))
    win2d = np.outer(wy, wx)
    windowed = crop * win2d

    # ── 5. Compute FFT and shift DC to centre ─────────────────────────────────
    F = np.fft.fftshift(np.fft.fft2(windowed))
    mag = np.abs(F)

    # ── 6. Window coherent-gain normalisation ────────────────────────────────
    # The window suppresses apparent signal amplitude by its coherent gain
    # (mean window value).  Without correction, switching window between
    # hann / tukey / none changes the *absolute* magnitude scale for the
    # same physical input — the agent flagged this as a reproducibility
    # hazard for downstream peak-finders that use absolute log-magnitude
    # thresholds (review physics #5).  Multiply by N / sum(window) so the
    # returned linear magnitude is window-corrected.
    win_sum = float(win2d.sum())
    if win_sum > 0:
        mag = mag * (float(Ny * Nx) / win_sum)

    if log_scale:
        mag = np.log1p(mag)

    # ── 7. Frequency axes in nm⁻¹ ────────────────────────────────────────────
    dx_nm = pixel_size_x_m * 1e9
    dy_nm = pixel_size_y_m * 1e9
    qx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx_nm))
    qy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy_nm))

    return mag.astype(np.float64), qx, qy




# ─── Bragg / reciprocal-lattice analysis (moved to bragg.py) ─────────────────
# Re-exported so existing ``from probeflow.processing.filters import ...`` and
# the ``image.py`` star-import keep working unchanged.
from probeflow.processing.bragg import *  # noqa: E402,F401,F403
