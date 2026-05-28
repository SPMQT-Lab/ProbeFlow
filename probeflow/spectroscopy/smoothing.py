"""Pure smoothing helpers for spectroscopy display traces."""

from __future__ import annotations

import numpy as np

from probeflow.spectroscopy._kernels import smooth_spectrum


def savgol_validation_message(
    smoothing_label: str,
    window: int,
    polyorder: int,
    point_count: int | None,
) -> str | None:
    """Return a user-facing validation error for Savitzky-Golay settings."""
    mode = (smoothing_label or "").strip().lower()
    if mode not in {"savitzky-golay", "savgol", "savitzky_golay"}:
        return None
    if window < 3:
        return "Savitzky-Golay window must be at least 3 points."
    if window % 2 == 0:
        return "Savitzky-Golay window must be odd and greater than the polynomial order."
    if polyorder < 0:
        return "Savitzky-Golay polynomial order must be non-negative."
    if polyorder >= window:
        return "Savitzky-Golay polynomial order must be smaller than the window length."
    if point_count is not None and window > point_count:
        return f"Savitzky-Golay window must not exceed available points ({point_count})."
    return None


def apply_smoothing(
    y: np.ndarray,
    *,
    mode: str = "none",
    points: int | None = None,
    polyorder: int = 2,
) -> np.ndarray:
    """Return a smoothed copy of ``y``."""
    arr = np.asarray(y, dtype=np.float64).copy()
    mode = (mode or "none").strip().lower()
    if mode in {"none", "off"}:
        return arr

    n = arr.size
    pts = int(points if points is not None else 7)
    if pts <= 0:
        raise ValueError("smoothing points must be positive")
    if pts > n:
        raise ValueError("smoothing points must not exceed the number of data points")

    if mode in {"gaussian", "gauss"}:
        sigma = max(pts / 2.355, 0.1)
        return smooth_spectrum(arr, method="gaussian", sigma=sigma)

    if mode in {"savgol", "savitzky-golay", "savitzky_golay"}:
        if pts % 2 == 0:
            raise ValueError("Savitzky-Golay window length must be odd")
        if pts <= polyorder:
            raise ValueError("Savitzky-Golay window length must exceed polynomial order")
        return smooth_spectrum(
            arr,
            method="savgol",
            window_length=pts,
            polyorder=int(polyorder),
        )

    raise ValueError(f"Unknown smoothing mode: {mode!r}")
