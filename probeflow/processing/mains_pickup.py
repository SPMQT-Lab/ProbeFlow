"""Mains-pickup (50/60 Hz) prediction and suppression for raster scans.

Mains pickup is a *time-domain* interference signal.  In a raster scan it lands
at a predictable *spatial* frequency along the fast-scan axis: a mains tone at
``f`` Hz completes ``f · T_line`` cycles per scan line, so it appears in the FFT
at index ``round(f · T_line)`` from DC along the fast axis (and its conjugate),
i.e. at ``q = f / v`` in the scan's spatial-frequency units, where

    T_line = scan_width / v        (fast-line time)
    v      = fast-scan tip speed   (m/s)

This module is GUI-free (numpy only) and deterministic.  It provides:

* :func:`estimate_fast_scan_speed_m_per_s` — read the fast-scan speed from a
  scan header (Nanonis / Createc), or return ``None``;
* :func:`predict_mains_fft_positions` — expected FFT peak positions + harmonics;
* :func:`equivalent_frequency_hz` — invert a cursor q back to a time frequency;
* :func:`mains_pickup_suppression` — apply symmetric notches at the predicted
  (optionally snapped) peaks, reusing
  :func:`probeflow.processing.filters.periodic_notch_filter`.

Only the within-line ripple (a streak on the fast-axis) is modelled; line-to-line
mains banding (low slow-axis frequencies) is out of scope.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

import numpy as np

__all__ = [
    "estimate_fast_scan_speed_m_per_s",
    "predict_mains_fft_positions",
    "equivalent_frequency_hz",
    "mains_pickup_suppression",
]


# ─── Header parsing ──────────────────────────────────────────────────────────

def _first_float(value: Any) -> Optional[float]:
    """First finite float in a header value (handles 'a   b' multi-field strings)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    for tok in re.split(r"[\s;,]+", str(value).strip()):
        try:
            f = float(tok)
        except ValueError:
            continue
        if math.isfinite(f):
            return f
    return None


def _find_header_value(header: dict, *patterns: str) -> Optional[float]:
    """First float for the first header key matching any (case-insensitive) regex."""
    if not header:
        return None
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for key, val in header.items():
            if rx.search(str(key)):
                f = _first_float(val)
                if f is not None:
                    return f
    return None


def estimate_fast_scan_speed_m_per_s(
    header: dict | None,
    *,
    scan_range_m: tuple[float, float] | None = None,
    image_shape: tuple[int, int] | None = None,
) -> Optional[float]:
    """Estimate the fast-scan tip speed (m/s) from a scan header.

    Priority: an explicit speed field → a fast-line time × scan width → a frame
    time ÷ number of lines × width.  Returns ``None`` when nothing usable is
    found, so callers can fall back to a user-entered value.

    Recognised fields (case-insensitive):
      * Nanonis  : ``Scan>speed forw. (m/s)``; or ``SCAN_TIME`` (line time) with
        ``SCAN_RANGE`` (width).
      * Createc  : ``Sec/line:`` (line time); or ``Sec/Image:`` (frame time) with
        the number of rows.
    """
    header = header or {}
    width_m = float(scan_range_m[0]) if scan_range_m else None

    # 1. Explicit fast-scan speed (Nanonis stores this directly).
    speed = _find_header_value(header, r"scan>?\s*speed\s*forw", r"\bspeed\b.*m/s")
    if speed is not None and speed > 0:
        return speed

    # 2. Fast-line time × width.
    if width_m and width_m > 0:
        line_time = _find_header_value(header, r"sec\s*/\s*line", r"\bSCAN_TIME\b")
        if line_time is not None and line_time > 0:
            return width_m / line_time

        # 3. Frame time ÷ rows × width.
        frame_time = _find_header_value(header, r"sec\s*/\s*image", r"\bACQ_TIME\b")
        n_rows = None
        if image_shape is not None:
            n_rows = int(image_shape[0])
        else:
            nr = _find_header_value(header, r"scan>?\s*lines", r"\bNum\.Y\b")
            n_rows = int(nr) if nr else None
        if frame_time and frame_time > 0 and n_rows and n_rows > 0:
            return width_m / (frame_time / n_rows)

    return None


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict_mains_fft_positions(
    image_width_px: int,
    scan_range_x_m: float,
    scan_speed_m_per_s: float | None,
    *,
    mains_frequency_hz: float = 50.0,
    harmonics: int = 3,
    fast_axis: str = "x",
) -> list[dict]:
    """Expected FFT positions of mains pickup and its harmonics.

    Returns one dict per harmonic that falls within the Nyquist limit::

        {"n", "freq_hz", "q_nm_inv", "fft_index", "dx", "dy"}

    ``dx``/``dy`` are integer pixel offsets from the fftshift-centred DC, ready
    for :func:`probeflow.processing.filters.periodic_notch_filter`.  Returns an
    empty list when the scan speed is unknown/non-positive.
    """
    if not scan_speed_m_per_s or scan_speed_m_per_s <= 0:
        return []
    if image_width_px <= 0 or scan_range_x_m <= 0:
        return []

    w_nm = scan_range_x_m * 1e9
    dx_nm = w_nm / image_width_px
    q_nyquist = 1.0 / (2.0 * dx_nm)
    v_nm_per_s = scan_speed_m_per_s * 1e9

    out: list[dict] = []
    for n in range(1, int(max(1, harmonics)) + 1):
        f = n * mains_frequency_hz
        q = f / v_nm_per_s                      # nm⁻¹
        if q > q_nyquist:
            break                               # higher harmonics only go further out
        index = int(round(q * w_nm))            # == round(f · T_line)
        if index == 0:
            continue                            # unresolved at this scan speed
        dx, dy = (index, 0) if fast_axis == "x" else (0, index)
        out.append({
            "n": n,
            "freq_hz": float(f),
            "q_nm_inv": float(q),
            "fft_index": int(index),
            "dx": int(dx),
            "dy": int(dy),
        })
    return out


def equivalent_frequency_hz(
    q_along_fast_axis_nm_inv: float, scan_speed_m_per_s: float | None
) -> Optional[float]:
    """Time-domain frequency (Hz) for a fast-axis spatial frequency q (nm⁻¹).

    ``f = q · v``.  Returns ``None`` if the scan speed is unknown.
    """
    if not scan_speed_m_per_s or scan_speed_m_per_s <= 0:
        return None
    return float(q_along_fast_axis_nm_inv) * float(scan_speed_m_per_s) * 1e9


# ─── Suppression ─────────────────────────────────────────────────────────────

def _snap_peaks_to_brightest(
    arr: np.ndarray, peaks: list[tuple[int, int]], window: int
) -> list[tuple[int, int]]:
    """Move each predicted (dx,dy) to the brightest |FFT| pixel within ±window.

    Mains pickup smears into a short streak (inter-line phase drift), so the true
    peak can sit a bin or two off the prediction; snapping makes the notch land
    on the actual energy.
    """
    finite = np.where(np.isfinite(arr), arr, 0.0)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(finite - finite.mean())))
    Ny, Nx = arr.shape
    cx, cy = Nx // 2, Ny // 2
    snapped: list[tuple[int, int]] = []
    for dx, dy in peaks:
        px, py = cx + dx, cy + dy
        x0, x1 = max(0, px - window), min(Nx, px + window + 1)
        y0, y1 = max(0, py - window), min(Ny, py + window + 1)
        sub = mag[y0:y1, x0:x1]
        if sub.size == 0:
            snapped.append((dx, dy))
            continue
        r, c = np.unravel_index(int(np.argmax(sub)), sub.shape)
        snapped.append((x0 + c - cx, y0 + r - cy))
    return snapped


def mains_pickup_suppression(
    arr: np.ndarray,
    *,
    scan_speed_m_per_s: float | None,
    scan_range_m: tuple[float, float],
    mains_frequency_hz: float = 50.0,
    harmonics: int = 3,
    notch_radius_px: float = 3.0,
    fast_axis: str = "x",
    snap_window_px: int = 2,
) -> np.ndarray:
    """Suppress predicted mains-pickup peaks with symmetric FFT notches.

    Predicts the peak positions, optionally snaps each to the brightest nearby
    |FFT| pixel, then applies Gaussian notches (and conjugates) via
    :func:`probeflow.processing.filters.periodic_notch_filter`.  Returns the
    original array unchanged when no peaks are predicted (e.g. unknown speed).
    """
    from probeflow.processing.filters import periodic_notch_filter

    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("mains_pickup_suppression expects a 2-D array")
    Ny, Nx = a.shape
    width_m = float(scan_range_m[0]) if fast_axis == "x" else float(scan_range_m[1])
    n_fast = Nx if fast_axis == "x" else Ny

    preds = predict_mains_fft_positions(
        n_fast, width_m, scan_speed_m_per_s,
        mains_frequency_hz=mains_frequency_hz, harmonics=harmonics,
        fast_axis=fast_axis,
    )
    if not preds:
        return a.copy()

    peaks = [(p["dx"], p["dy"]) for p in preds]
    if snap_window_px and snap_window_px > 0:
        peaks = _snap_peaks_to_brightest(a, peaks, int(snap_window_px))
    return periodic_notch_filter(a, peaks, radius_px=float(notch_radius_px))
