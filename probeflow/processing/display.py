"""Shared display-rendering helpers for ProbeFlow.

This module handles the array-to-display-image pathway only.
It does not perform scientific processing (plane subtraction, line flattening,
denoising, FFT filtering, etc.) — those live in probeflow.processing.

Typical call order
------------------
    vmin, vmax = clip_range_from_array(arr)         # or supply explicit limits
    u8 = array_to_uint8(arr, vmin=vmin, vmax=vmax)  # renders to uint8 image
"""

from __future__ import annotations

from typing import Any

import numpy as np


def finite_values(arr: np.ndarray) -> np.ndarray:
    """Return finite values from a numeric array as a 1-D float64 array."""
    a = np.asarray(arr, dtype=np.float64)
    return a[np.isfinite(a)].ravel()


def clip_range_from_array(
    arr: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> tuple[float, float]:
    """Return robust display limits using finite pixels only.

    Ignores NaN/inf and avoids single-pixel outliers dominating display
    contrast via percentile clipping.

    Parameters
    ----------
    arr:
        2-D (or any shape) numeric array.
    low_pct, high_pct:
        Percentile bounds for clipping (default 1 % / 99 %).

    Returns
    -------
    (vmin, vmax) : tuple[float, float]

    Raises
    ------
    ValueError
        If *arr* contains no finite values.
    """
    finite = finite_values(arr)
    if finite.size == 0:
        raise ValueError("Array contains no finite values — cannot compute display range.")

    vmin = float(np.percentile(finite, low_pct))
    vmax = float(np.percentile(finite, high_pct))

    if vmax <= vmin:
        vmin = float(finite.min())
        vmax = float(finite.max())
    if vmax <= vmin:
        vmin = vmin - 1.0
        vmax = vmin + 2.0

    return vmin, vmax


def normalise_array(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalise *arr* to [0, 1] for display using supplied limits.

    NaN/inf pixels are mapped to 0.0.  When *vmin == vmax* the output is all
    zeros (no divide-by-zero warning).

    Parameters
    ----------
    arr:
        Input array (any finite/non-finite mix).
    vmin, vmax:
        Display limits; the half-open interval [vmin, vmax] maps to [0, 1].

    Returns
    -------
    np.ndarray of float32, same shape as *arr*, values in [0, 1].
    """
    a = np.asarray(arr, dtype=np.float64)
    safe = np.where(np.isfinite(a), a, vmin)
    if vmax > vmin:
        out = (safe - vmin) / (vmax - vmin)
    else:
        out = np.zeros_like(safe)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def array_to_uint8(
    arr: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Convert a 2-D numeric array to uint8 display data.

    If *vmin*/*vmax* are not supplied, they are computed from
    *clip_percentiles* using :func:`clip_range_from_array`.

    Parameters
    ----------
    arr:
        2-D numeric array.
    vmin, vmax:
        Explicit display limits.  If either is None both are derived from
        *clip_percentiles*.
    clip_percentiles:
        (low_pct, high_pct) passed to :func:`clip_range_from_array` when
        vmin/vmax are not provided.

    Returns
    -------
    np.ndarray of dtype uint8, same shape as *arr*.

    Raises
    ------
    ValueError
        If *arr* contains no finite values and limits cannot be determined.
    """
    arr = np.asarray(arr)
    if vmin is None or vmax is None:
        vmin, vmax = clip_range_from_array(arr, *clip_percentiles)

    norm = normalise_array(arr, float(vmin), float(vmax))
    return (norm * 255.0 + 0.5).astype(np.uint8)


def histogram_from_array(
    arr: np.ndarray,
    *,
    roi: "Any | None" = None,
    bins: int = 256,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return histogram data using finite values only.

    The bin range matches the display clipping range so the histogram
    represents the same signal window used for rendering.

    Parameters
    ----------
    arr:
        Input array (any shape).
    roi:
        Optional :class:`probeflow.core.roi.ROI`.  When provided, only pixels
        within the ROI mask are included in the histogram.  All ROI kinds are
        accepted (a point ROI yields a single-pixel histogram; a line ROI
        yields the histogram of pixels along the 1-pixel-wide line).  The bin
        range is still derived from the *full* array so histograms from
        different ROIs on the same scan are directly comparable.
    bins:
        Number of histogram bins.
    clip_percentiles:
        (low_pct, high_pct) for the bin range, consistent with
        :func:`clip_range_from_array`.

    Returns
    -------
    (counts, edges) — same convention as :func:`numpy.histogram`.
        counts: np.ndarray of int64, length *bins*.
        edges:  np.ndarray of float64, length *bins* + 1.

    Raises
    ------
    ValueError
        If *arr* (or the ROI-masked subset) contains no finite values.
    """
    if arr.ndim != 2 and roi is not None:
        raise ValueError("roi parameter requires a 2-D array")

    if roi is not None:
        mask = roi.to_mask(arr.shape[:2])
        masked = np.asarray(arr, dtype=np.float64)[mask]
        finite = masked[np.isfinite(masked)]
    else:
        finite = finite_values(arr)

    if finite.size == 0:
        raise ValueError("Array contains no finite values — cannot compute histogram.")

    vmin, vmax = clip_range_from_array(arr, *clip_percentiles)
    counts, edges = np.histogram(finite, bins=bins, range=(vmin, vmax))
    return counts, edges


def array_to_rgba(
    arr: np.ndarray,
    *,
    colormap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Convert a 2-D numeric array to RGBA display data using a matplotlib colormap.

    Parameters
    ----------
    arr:
        2-D numeric array.
    colormap:
        Matplotlib colormap name.
    vmin, vmax:
        Explicit display limits; derived from *clip_percentiles* if None.
    clip_percentiles:
        Fallback clipping percentiles.

    Returns
    -------
    np.ndarray of dtype uint8, shape (Ny, Nx, 4).

    Raises
    ------
    ValueError
        If *arr* contains no finite values.
    """
    import matplotlib.cm as _cm

    if vmin is None or vmax is None:
        vmin, vmax = clip_range_from_array(arr, *clip_percentiles)

    norm = normalise_array(arr, float(vmin), float(vmax))
    cmap = _cm.get_cmap(colormap)
    rgba_f = cmap(norm)                            # (Ny, Nx, 4) float32/64
    return (rgba_f * 255.0 + 0.5).astype(np.uint8)
