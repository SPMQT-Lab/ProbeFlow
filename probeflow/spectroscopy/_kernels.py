"""Pure processing kernels for spectroscopy data.

All functions operate on raw numpy arrays (physical SI units).
No GUI or file-I/O dependency; safe to call from worker threads.

Moved here from ``probeflow.processing.spectroscopy`` so the spectroscopy
package is self-contained (review arch-backend #8).  The public import
surface is preserved by a thin re-export shim at
``probeflow.processing.spectroscopy`` and by the convenience exports on
``probeflow.spectroscopy``.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import savgol_filter


def smooth_spectrum(
    data: np.ndarray,
    method: str = "savgol",
    **kwargs,
) -> np.ndarray:
    """Smooth a 1-D spectrum.

    Parameters
    ----------
    data : np.ndarray
        1-D array of spectral values.
    method : str
        'savgol' (Savitzky-Golay), 'gaussian', or 'boxcar'.
    **kwargs
        savgol:   window_length (int, default 11), polyorder (int, default 3)
        gaussian: sigma (float, default 2.0)
        boxcar:   n (int, default 5)

    Returns
    -------
    np.ndarray
        Smoothed array, same length as input.
    """
    data = np.asarray(data, dtype=np.float64)
    if method == "savgol":
        window = int(kwargs.get("window_length", 11))
        polyorder = int(kwargs.get("polyorder", 3))
        n = len(data)
        # Return unchanged if the array is too short to filter meaningfully.
        if n < polyorder + 2:
            return data.copy()
        # window must be odd and strictly greater than polyorder.
        window = max(polyorder + 2 if polyorder % 2 == 0 else polyorder + 1, window)
        if window % 2 == 0:
            window += 1
        max_win = n if n % 2 == 1 else n - 1
        window = min(window, max_win)
        # Clamp to minimum valid window after the size cap.
        if window < polyorder + 1:
            window = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2
        return savgol_filter(data, window_length=window, polyorder=polyorder)
    elif method == "gaussian":
        from scipy.ndimage import gaussian_filter1d
        sigma = float(kwargs.get("sigma", 2.0))
        return gaussian_filter1d(data, sigma=sigma)
    elif method == "boxcar":
        from scipy.ndimage import uniform_filter1d
        n = int(kwargs.get("n", 5))
        n = max(1, n)
        # mode="nearest" reflects edge values instead of zero-padding.
        return uniform_filter1d(data, size=n, mode="nearest")
    else:
        raise ValueError(
            f"Unknown smoothing method: {method!r}. Choose savgol, gaussian, or boxcar."
        )


def numeric_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute dy/dx via central finite differences.

    x must be strictly monotonic (no duplicate values). Non-monotonic inputs
    — such as a forward+backward bias sweep stored in a single array — will
    produce incorrect derivatives; split the sweep first.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (e.g. bias in V or time in s). Must be monotonic.
    y : np.ndarray
        Dependent variable (e.g. current in A).

    Returns
    -------
    np.ndarray
        Derivative dy/dx, same length as x and y.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("numeric_derivative: x and y must have matching shape.")
    if x.ndim != 1:
        raise ValueError("numeric_derivative: x and y must be 1-D arrays.")
    if x.size < 2:
        raise ValueError("numeric_derivative: at least two samples are required.")
    diffs = np.diff(x)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(
            "numeric_derivative: x is not strictly monotonic. "
            "If this is a forward+backward sweep, split it before differentiating."
        )
    return np.gradient(y, x)


def normalize(data: np.ndarray, method: str = "max") -> np.ndarray:
    """Normalize a 1-D array.

    Parameters
    ----------
    data : np.ndarray
        Input array.
    method : str
        'max'    — divide by max absolute value.
        'minmax' — rescale to [0, 1].
        'zscore' — subtract mean, divide by std.
        'setpoint' — divide by the first finite non-zero value.

    Returns
    -------
    np.ndarray
        Normalized array, same length as input.
    """
    data = np.asarray(data, dtype=np.float64)
    if method == "max":
        m = float(np.nanmax(np.abs(data)))
        if m == 0.0:
            warnings.warn("normalize: all-zero input; returning zeros unchanged", stacklevel=2)
            return data.copy()
        return data / m
    elif method == "minmax":
        lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
        return (data - lo) / (hi - lo) if hi != lo else np.zeros_like(data)
    elif method == "zscore":
        mu = float(np.nanmean(data))
        sigma = float(np.nanstd(data))
        return (data - mu) / sigma if sigma != 0.0 else np.zeros_like(data)
    elif method == "setpoint":
        finite = data[np.isfinite(data) & (data != 0)]
        if finite.size == 0:
            warnings.warn(
                "normalize: no finite non-zero setpoint; returning input unchanged",
                stacklevel=2,
            )
            return data.copy()
        return data / float(finite[0])
    else:
        raise ValueError(
            f"Unknown normalization method: {method!r}. "
            "Choose max, minmax, zscore, or setpoint."
        )


def crop(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the subset of (x, y) where x_min ≤ x ≤ x_max.

    If x_min > x_max the bounds are silently swapped.

    Parameters
    ----------
    x, y : np.ndarray
        Paired 1-D arrays of the same length.
    x_min, x_max : float
        Inclusive bounds on the x range to keep.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Cropped (x, y) pair.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    mask = (x >= x_min) & (x <= x_max)
    return x[mask], y[mask]


def average_spectra(spectra: list[np.ndarray]) -> np.ndarray:
    """Element-wise mean of a list of equal-length 1-D arrays.

    Parameters
    ----------
    spectra : list[np.ndarray]
        List of 1-D arrays, all the same length. All spectra must share the
        same x-axis grid (i.e. identical x values at every index), not merely
        the same number of points. Raises ValueError if lengths differ —
        interpolate to a common x grid before calling if needed.

    Returns
    -------
    np.ndarray
        Mean array.
    """
    if not spectra:
        raise ValueError("spectra list is empty")
    arrs = [np.asarray(s, dtype=np.float64) for s in spectra]
    lengths = [a.size for a in arrs]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"average_spectra: all spectra must have the same length, "
            f"got {lengths}. Interpolate to a common x grid first."
        )
    return np.mean(np.stack(arrs, axis=0), axis=0)


def current_histogram(
    data: np.ndarray,
    bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of current values for telegraph-noise analysis.

    Return order matches numpy: (counts, bin_edges).

    Parameters
    ----------
    data : np.ndarray
        1-D array of current values (A).
    bins : int
        Number of histogram bins.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (counts, bin_edges) — counts has length bins, bin_edges has length bins+1.
    """
    data = np.asarray(data, dtype=np.float64)
    finite = data[np.isfinite(data)]
    return np.histogram(finite, bins=bins)
