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
* :func:`mains_pickup_suppression` — apply symmetric spot or streak notches at
  the predicted (optionally snapped) peaks.

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
    harmonics: int | None = 3,
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
    if not math.isfinite(float(scan_speed_m_per_s)):
        return []
    if image_width_px <= 0 or scan_range_x_m <= 0:
        return []
    if mains_frequency_hz <= 0 or not math.isfinite(float(mains_frequency_hz)):
        return []

    w_nm = scan_range_x_m * 1e9
    max_fft_offset = image_width_px - (image_width_px // 2) - 1
    if max_fft_offset <= 0:
        return []
    v_nm_per_s = scan_speed_m_per_s * 1e9
    harmonic_index = mains_frequency_hz * w_nm / v_nm_per_s
    if harmonic_index <= 0 or not math.isfinite(harmonic_index):
        return []
    if harmonics is None:
        max_harmonic = int(math.ceil(max_fft_offset / harmonic_index)) + 1
    else:
        max_harmonic = int(max(1, harmonics))

    out: list[dict] = []
    seen_indices: set[int] = set()
    for n in range(1, max_harmonic + 1):
        f = n * mains_frequency_hz
        q = f / v_nm_per_s                      # nm⁻¹
        index = int(round(q * w_nm))            # == round(f · T_line)
        if index > max_fft_offset:
            break                               # higher harmonics only go further out
        if index == 0 or index in seen_indices:
            continue                            # unresolved at this scan speed
        seen_indices.add(index)
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


def _q_axes_nm_inv(shape: tuple[int, int], scan_range_m: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    Ny, Nx = int(shape[0]), int(shape[1])
    width_nm = float(scan_range_m[0]) * 1e9
    height_nm = float(scan_range_m[1]) * 1e9
    dx_nm = width_nm / Nx if Nx > 0 and width_nm > 0 else 1.0
    dy_nm = height_nm / Ny if Ny > 0 and height_nm > 0 else 1.0
    qx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx_nm))
    qy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy_nm))
    return qx, qy


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


def _snap_streaks_to_brightest(
    arr: np.ndarray,
    peaks: list[tuple[int, int]],
    window: int,
    *,
    scan_range_m: tuple[float, float],
    fast_axis: str,
    min_q_nm_inv: float,
) -> list[tuple[int, int]]:
    """Move streak centres to the brightest nearby fast-axis column/row."""
    finite = np.where(np.isfinite(arr), arr, 0.0)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(finite - finite.mean())))
    Ny, Nx = arr.shape
    cx, cy = Nx // 2, Ny // 2
    qx, qy = _q_axes_nm_inv(arr.shape, scan_range_m)
    qxx, qyy = np.meshgrid(qx, qy)
    keep = np.hypot(qxx, qyy) >= max(float(min_q_nm_inv), 0.0)
    snapped: list[tuple[int, int]] = []
    if fast_axis == "x":
        weights = np.where(keep, mag, 0.0)
        for dx, _dy in peaks:
            px = cx + dx
            x0, x1 = max(0, px - window), min(Nx, px + window + 1)
            if x1 <= x0:
                snapped.append((dx, 0))
                continue
            scores = weights[:, x0:x1].sum(axis=0)
            if not np.any(scores):
                scores = mag[:, x0:x1].sum(axis=0)
            best_x = x0 + int(np.argmax(scores))
            snapped.append((best_x - cx, 0))
    else:
        weights = np.where(keep, mag, 0.0)
        for _dx, dy in peaks:
            py = cy + dy
            y0, y1 = max(0, py - window), min(Ny, py + window + 1)
            if y1 <= y0:
                snapped.append((0, dy))
                continue
            scores = weights[y0:y1, :].sum(axis=1)
            if not np.any(scores):
                scores = mag[y0:y1, :].sum(axis=1)
            best_y = y0 + int(np.argmax(scores))
            snapped.append((0, best_y - cy))
    return snapped


def _mains_streak_filter(
    arr: np.ndarray,
    peaks: list[tuple[int, int]],
    *,
    scan_range_m: tuple[float, float],
    radius_px: float,
    fast_axis: str,
    min_q_nm_inv: float,
) -> np.ndarray:
    """Suppress full predicted mains streaks, optionally outside a radial q floor."""
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
    qx, qy = _q_axes_nm_inv(a.shape, scan_range_m)
    qxx, qyy = np.meshgrid(qx, qy)
    q_keep = np.hypot(qxx, qyy) >= max(float(min_q_nm_inv), 0.0)
    notch = np.ones((Ny, Nx), dtype=np.float64)
    sigma = max(float(radius_px), 0.5)

    for dx, dy in peaks:
        if fast_axis == "x":
            offsets = (int(dx), -int(dx))
            for sx in offsets:
                if sx == 0:
                    continue
                px = cx + sx
                if 0 <= px < Nx:
                    stripe = np.exp(-0.5 * ((xx - px) ** 2) / (sigma ** 2))
                    notch *= 1.0 - np.where(q_keep, stripe, 0.0)
        else:
            offsets = (int(dy), -int(dy))
            for sy in offsets:
                if sy == 0:
                    continue
                py = cy + sy
                if 0 <= py < Ny:
                    stripe = np.exp(-0.5 * ((yy - py) ** 2) / (sigma ** 2))
                    notch *= 1.0 - np.where(q_keep, stripe, 0.0)

    out = np.fft.ifft2(np.fft.ifftshift(F * notch)).real + mean_val
    out[nan_mask] = np.nan
    return out


def mains_pickup_suppression(
    arr: np.ndarray,
    *,
    scan_speed_m_per_s: float | None,
    scan_range_m: tuple[float, float],
    mains_frequency_hz: float = 50.0,
    harmonics: int | None = 3,
    notch_radius_px: float = 3.0,
    fast_axis: str = "x",
    snap_window_px: int = 2,
    notch_shape: str = "spot",
    min_q_nm_inv: float = 0.0,
) -> np.ndarray:
    """Suppress predicted mains-pickup peaks with symmetric FFT notches.

    Predicts the peak positions, optionally snaps each to the brightest nearby
    |FFT| pixel or streak, then applies Gaussian notches and their conjugates.
    ``notch_shape="spot"`` preserves the legacy circular-spot behaviour;
    ``"streak"`` removes the predicted fast-axis streak outside ``min_q_nm_inv``.
    Returns the original array unchanged when no peaks are predicted.
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
    shape = str(notch_shape or "spot").lower()
    if shape in {"spot", "circle", "circular"}:
        if snap_window_px and snap_window_px > 0:
            peaks = _snap_peaks_to_brightest(a, peaks, int(snap_window_px))
        return periodic_notch_filter(a, peaks, radius_px=float(notch_radius_px))
    if shape == "streak":
        if snap_window_px and snap_window_px > 0:
            peaks = _snap_streaks_to_brightest(
                a, peaks, int(snap_window_px),
                scan_range_m=tuple(scan_range_m),
                fast_axis=fast_axis,
                min_q_nm_inv=float(min_q_nm_inv),
            )
        return _mains_streak_filter(
            a, peaks,
            scan_range_m=tuple(scan_range_m),
            radius_px=float(notch_radius_px),
            fast_axis=fast_axis,
            min_q_nm_inv=float(min_q_nm_inv),
        )
    raise ValueError("notch_shape must be 'spot' or 'streak'")
