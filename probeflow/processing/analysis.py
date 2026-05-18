"""Image analysis: GMM auto-clipping, grain detection, periodicity measurement."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import label as _nd_label

from ._image_utils import _finite_mean


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
