"""
ProbeFlow — image processing pipeline for STM/SXM data.

All functions operate on raw float32/float64 2-D arrays (physical units).
They are intentionally free of any GUI dependency so they can be called from
worker threads or batch scripts without importing PySide6.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ── Font path for scale-bar labels ────────────────────────────────────────────
_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  remove_bad_lines
# ═════════════════════════════════════════════════════════════════════════════

def remove_bad_lines(arr: np.ndarray, threshold_mad: float = 5.0) -> np.ndarray:
    """
    Replace outlier scan lines via weighted interpolation from neighbours.

    A row is "bad" when |row_median − overall_median| > threshold_mad × MAD,
    where MAD is the median absolute deviation of per-row medians.
    Bad rows are replaced by a distance-weighted blend of the nearest good
    rows above and below (falls back to the single nearest good row when
    only one side is available).
    """
    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    # Per-row medians (ignore NaN)
    row_meds = np.array([
        float(np.nanmedian(arr[r])) for r in range(Ny)
    ])

    finite_meds = row_meds[np.isfinite(row_meds)]
    if finite_meds.size == 0:
        return arr  # nothing we can do

    overall_med = float(np.median(finite_meds))
    mad = float(np.median(np.abs(finite_meds - overall_med)))
    # If MAD is zero (all rows identical) there are no outliers
    if mad == 0.0:
        return arr

    bad_mask = np.abs(row_meds - overall_med) > threshold_mad * mad
    bad_rows = np.where(bad_mask)[0]

    if bad_rows.size == 0:
        return arr

    good_rows = np.where(~bad_mask)[0]
    if good_rows.size == 0:
        # Every row is bad — can't fix anything
        return arr

    for r in bad_rows:
        # Find nearest good row above (< r) and below (> r)
        above = good_rows[good_rows < r]
        below = good_rows[good_rows > r]

        if above.size > 0 and below.size > 0:
            ra, rb = int(above[-1]), int(below[0])
            da, db = r - ra, rb - r          # distances (both ≥ 1)
            # Inverse-distance weights
            wa = db / (da + db)
            wb = da / (da + db)
            arr[r] = wa * arr[ra] + wb * arr[rb]
        elif above.size > 0:
            arr[r] = arr[int(above[-1])]
        else:
            arr[r] = arr[int(below[0])]

    return arr


# ═════════════════════════════════════════════════════════════════════════════
# 2.  subtract_background
# ═════════════════════════════════════════════════════════════════════════════

def subtract_background(arr: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Fit and subtract a polynomial background.

    order=1  → plane    (ax + by + c)
    order=2  → full 2nd-degree (ax² + by² + cxy + dx + ey + f)

    Coordinates are normalised to [-1, 1] for numerical stability.
    Only finite pixels participate in the least-squares fit.
    """
    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    # Normalised coordinate grids
    ys = np.linspace(-1.0, 1.0, Ny)
    xs = np.linspace(-1.0, 1.0, Nx)
    Xg, Yg = np.meshgrid(xs, ys)   # shape (Ny, Nx)

    flat_x = Xg.ravel()
    flat_y = Yg.ravel()
    flat_z = arr.ravel()

    finite = np.isfinite(flat_z)
    if finite.sum() < (3 if order == 1 else 6):
        return arr  # not enough data to fit

    if order == 1:
        # Design matrix for plane: [x, y, 1]
        A = np.column_stack([flat_x[finite], flat_y[finite],
                             np.ones(finite.sum())])
    else:
        # Design matrix for 2nd-degree: [x², y², xy, x, y, 1]
        fx, fy = flat_x[finite], flat_y[finite]
        A = np.column_stack([fx**2, fy**2, fx*fy, fx, fy,
                             np.ones(finite.sum())])

    b = flat_z[finite]
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    if order == 1:
        bg = coeffs[0]*Xg + coeffs[1]*Yg + coeffs[2]
    else:
        bg = (coeffs[0]*Xg**2 + coeffs[1]*Yg**2 +
              coeffs[2]*Xg*Yg  + coeffs[3]*Xg    +
              coeffs[4]*Yg     + coeffs[5])

    return arr - bg


# ═════════════════════════════════════════════════════════════════════════════
# 3.  fourier_filter
# ═════════════════════════════════════════════════════════════════════════════

def fourier_filter(
    arr:    np.ndarray,
    mode:   str   = 'low_pass',
    cutoff: float = 0.1,
    window: str   = 'hanning',
) -> np.ndarray:
    """
    Apply a 2-D FFT filter.

    A spatial window (hanning / hamming / none) is applied before the FFT to
    suppress ringing artefacts at image boundaries.

    cutoff  — fraction of Nyquist [0, 1].  cutoff=0.1 keeps the lowest 10 %.
    mode    — 'low_pass'  keeps frequencies ≤ cutoff
              'high_pass' keeps frequencies ≥ cutoff
    """
    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    # Replace non-finite values with the array mean so the FFT is well-defined
    mean_val = float(np.nanmean(arr))
    arr[~np.isfinite(arr)] = mean_val

    # Build 2-D window
    if window == 'hanning':
        wy = np.hanning(Ny)
        wx = np.hanning(Nx)
    elif window == 'hamming':
        wy = np.hamming(Ny)
        wx = np.hamming(Nx)
    else:
        wy = np.ones(Ny)
        wx = np.ones(Nx)

    win2d = np.outer(wy, wx)
    windowed = arr * win2d

    F = np.fft.fft2(windowed)
    F = np.fft.fftshift(F)

    # Normalised radial frequency grid; 1.0 = Nyquist
    cy, cx = Ny / 2.0, Nx / 2.0
    yr = (np.arange(Ny) - cy) / cy
    xr = (np.arange(Nx) - cx) / cx
    Xr, Yr = np.meshgrid(xr, yr)
    R = np.sqrt(Xr**2 + Yr**2)

    if mode == 'low_pass':
        mask = (R <= cutoff).astype(np.float64)
    else:  # high_pass
        mask = (R >= cutoff).astype(np.float64)

    F_filtered = F * mask
    F_filtered = np.fft.ifftshift(F_filtered)
    result = np.fft.ifft2(F_filtered).real

    # Un-window: divide out the window to restore amplitude scale where
    # the window was non-trivial.  Guard against near-zero window values.
    safe_win = np.where(win2d > 1e-6, win2d, 1.0)
    result = result / safe_win

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 4.  measure_periodicity
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

    Only one half of the spectrum is inspected (the other is its conjugate
    mirror).  Found peaks are suppressed (zeroed in a small neighbourhood)
    before searching for the next one.
    """
    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    mean_val = float(np.nanmean(arr))
    arr[~np.isfinite(arr)] = mean_val

    # Hanning window reduces spectral leakage
    wy = np.hanning(Ny)
    wx = np.hanning(Nx)
    win2d = np.outer(wy, wx)

    F = np.fft.fft2(arr * win2d)
    F = np.fft.fftshift(F)
    power = np.abs(F) ** 2

    cy, cx = Ny // 2, Nx // 2

    # DC suppression radius (2 px)
    DC_R = 2.0

    # Work only in the upper half (y < cy) to avoid duplicate conjugate peaks
    # We build a mask that is True where we allow peak search
    half_mask = np.zeros((Ny, Nx), dtype=bool)
    half_mask[:cy, :] = True   # rows 0 .. cy-1

    # Also suppress a ring around DC to ignore very low-frequency drift
    yr = np.arange(Ny) - cy
    xr = np.arange(Nx) - cx
    Xr, Yr = np.meshgrid(xr.astype(float), yr.astype(float))
    R_px = np.sqrt(Xr**2 + Yr**2)
    half_mask[R_px < DC_R] = False

    search_power = power.copy()
    search_power[~half_mask] = 0.0

    results = []
    # Suppression radius around each found peak (fraction of min dimension)
    suppress_r = max(3, min(Ny, Nx) // 20)

    for _ in range(n_peaks):
        idx = int(np.argmax(search_power))
        py, px = divmod(idx, Nx)

        peak_val = float(search_power[py, px])
        if peak_val <= 0:
            break

        # Fractional frequency coordinates (cycles/pixel)
        fy = (py - cy) / Ny   # negative (upper half)
        fx = (px - cx) / Nx

        # Spatial frequency magnitude → period in metres
        f_mag = math.sqrt(fx**2 + fy**2)   # cycles/pixel
        if f_mag == 0.0:
            break

        # Convert from cycles/pixel to cycles/metre
        # period = 1 / (f_mag_cycles_per_metre)
        # For non-square images we decompose:
        freq_m_x = fx / pixel_size_x_m   # cycles/m in x
        freq_m_y = fy / pixel_size_y_m   # cycles/m in y
        freq_m   = math.sqrt(freq_m_x**2 + freq_m_y**2)
        period_m = 1.0 / freq_m if freq_m > 0 else 0.0

        # Angle: angle of the wave-vector in the image plane (0° = +x axis)
        # Use atan2 with physical coordinates
        angle_deg = math.degrees(math.atan2(fy * pixel_size_y_m,
                                             fx * pixel_size_x_m))

        results.append({
            'period_m':  period_m,
            'angle_deg': angle_deg,
            'strength':  peak_val,
        })

        # Suppress the found peak and its conjugate mirror so the next
        # iteration finds a different one
        for (rpy, rpx) in [(py, px), (Ny - py, Nx - px)]:
            for dy in range(-suppress_r, suppress_r + 1):
                for dx in range(-suppress_r, suppress_r + 1):
                    ny_ = int(rpy) + dy
                    nx_ = int(rpx) + dx
                    if 0 <= ny_ < Ny and 0 <= nx_ < Nx:
                        search_power[ny_, nx_] = 0.0

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 5.  export_png
# ═════════════════════════════════════════════════════════════════════════════

# Nice round numbers for scale bar length in each unit system
_NICE_STEPS_NM = [
    0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50,
    100, 200, 500, 1000, 2000, 5000, 10000,
]


def _pick_scalebar_length(width_m: float, image_px: int,
                          target_frac: float = 0.20,
                          unit: str = 'nm') -> tuple[float, str]:
    """
    Choose a human-friendly scale bar length.

    Returns (length_in_metres, label_string).
    target_frac  — desired fraction of image width the bar should occupy.
    """
    unit_factors = {'nm': 1e9, 'Å': 1e10, 'pm': 1e12}
    factor = unit_factors.get(unit, 1e9)

    target_m = width_m * target_frac

    # Convert target to display unit, pick nearest nice step
    target_u = target_m * factor
    steps_u = _NICE_STEPS_NM  # reused for all units (just different magnitudes)

    best = steps_u[0]
    for s in steps_u:
        if abs(s - target_u) < abs(best - target_u):
            best = s
        if s > target_u * 2:
            break

    bar_m = best / factor   # back to metres

    # Format label: omit trailing zeros
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
    add_scalebar:  bool  = True,
    scalebar_unit: str   = 'nm',
    scalebar_pos:  str   = 'bottom-right',
) -> None:
    """
    Export a full-resolution colourised image with an optional scale bar.

    lut_fn(colormap_key) must return a (256, 3) uint8 LUT array.
    scan_range_m  — (width_m, height_m); scale bar is skipped when width ≤ 0.
    scalebar_pos  — 'bottom-right' or 'bottom-left'.
    """
    from PIL import Image as _Image, ImageDraw as _IDraw, ImageFont as _IFont

    arr = arr.astype(np.float64, copy=True)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Array contains no finite values.")

    vmin = float(np.percentile(finite, clip_low))
    vmax = float(np.percentile(finite, clip_high))
    if vmax <= vmin:
        vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    safe = np.where(np.isfinite(arr), arr, vmin).astype(np.float64)
    u8   = np.clip((safe - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    lut  = lut_fn(colormap_key)          # (256, 3) uint8
    colored = lut[u8]                    # (Ny, Nx, 3)
    img = _Image.fromarray(colored, mode="RGB")

    width_m = scan_range_m[0] if len(scan_range_m) >= 1 else 0.0
    Ny, Nx  = arr.shape

    if add_scalebar and width_m > 0:
        bar_m, bar_label = _pick_scalebar_length(
            width_m, Nx, target_frac=0.20, unit=scalebar_unit)

        # Bar length in pixels
        bar_px = int(round(bar_m / width_m * Nx))
        bar_px = max(4, min(bar_px, Nx - 20))

        # Typography
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

        # Measure text
        dummy_img  = _Image.new("RGB", (1, 1))
        dummy_draw = _IDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), bar_label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Bar geometry
        if scalebar_pos == 'bottom-left':
            bar_x0 = MARGIN
        else:
            bar_x0 = Nx - MARGIN - bar_px

        bar_y0 = Ny - MARGIN - BAR_HEIGHT
        bar_x1 = bar_x0 + bar_px
        bar_y1 = bar_y0 + BAR_HEIGHT

        # Text centered over bar
        text_x = bar_x0 + (bar_px - text_w) // 2
        text_y = bar_y0 - TEXT_GAP - text_h

        draw = _IDraw.Draw(img)

        # Bar outline (black) then fill (white)
        draw.rectangle([bar_x0 - 1, bar_y0 - 1, bar_x1 + 1, bar_y1 + 1],
                       fill=(0, 0, 0))
        draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=(255, 255, 255))

        # Shadowed text: black at +1,+1 offset then white on top
        draw.text((text_x + 1, text_y + 1), bar_label, font=font,
                  fill=(0, 0, 0))
        draw.text((text_x, text_y), bar_label, font=font,
                  fill=(255, 255, 255))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="PNG")
