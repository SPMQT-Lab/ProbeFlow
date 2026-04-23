"""Pure processing functions for Createc spectroscopy data.

All functions operate on raw numpy arrays (physical SI units).
No GUI or file-I/O dependency; safe to call from worker threads.
"""

from __future__ import annotations

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
        # window must be odd and > polyorder
        window = max(polyorder + 2 if polyorder % 2 == 0 else polyorder + 1, window)
        if window % 2 == 0:
            window += 1
        window = min(window, len(data) if len(data) % 2 == 1 else len(data) - 1)
        return savgol_filter(data, window_length=window, polyorder=polyorder)
    elif method == "gaussian":
        from scipy.ndimage import gaussian_filter1d
        sigma = float(kwargs.get("sigma", 2.0))
        return gaussian_filter1d(data, sigma=sigma)
    elif method == "boxcar":
        n = int(kwargs.get("n", 5))
        n = max(1, n)
        kernel = np.ones(n) / n
        return np.convolve(data, kernel, mode="same")
    else:
        raise ValueError(f"Unknown smoothing method: {method!r}. Choose savgol, gaussian, or boxcar.")


def numeric_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute dy/dx via central finite differences.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (e.g. bias in V or time in s).
    y : np.ndarray
        Dependent variable (e.g. current in A).

    Returns
    -------
    np.ndarray
        Derivative dy/dx, same length as x and y.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
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

    Returns
    -------
    np.ndarray
        Normalized array, same length as input.
    """
    data = np.asarray(data, dtype=np.float64)
    if method == "max":
        m = float(np.nanmax(np.abs(data)))
        return data / m if m != 0.0 else data.copy()
    elif method == "minmax":
        lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
        return (data - lo) / (hi - lo) if hi != lo else np.zeros_like(data)
    elif method == "zscore":
        mu = float(np.nanmean(data))
        sigma = float(np.nanstd(data))
        return (data - mu) / sigma if sigma != 0.0 else np.zeros_like(data)
    else:
        raise ValueError(f"Unknown normalization method: {method!r}. Choose max, minmax, or zscore.")


def crop(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the subset of (x, y) where x_min ≤ x ≤ x_max.

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
    mask = (x >= x_min) & (x <= x_max)
    return x[mask], y[mask]


def average_spectra(spectra: list[np.ndarray]) -> np.ndarray:
    """Element-wise mean of a list of equal-length 1-D arrays.

    Parameters
    ----------
    spectra : list[np.ndarray]
        List of 1-D arrays, all the same length.

    Returns
    -------
    np.ndarray
        Mean array.
    """
    if not spectra:
        raise ValueError("spectra list is empty")
    stacked = np.stack([np.asarray(s, dtype=np.float64) for s in spectra], axis=0)
    return np.mean(stacked, axis=0)


def current_histogram(
    data: np.ndarray,
    bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of current values for telegraph-noise analysis.

    Parameters
    ----------
    data : np.ndarray
        1-D array of current values (A).
    bins : int
        Number of histogram bins.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (bin_edges, counts) — bin_edges has length bins+1, counts has length bins.
    """
    data = np.asarray(data, dtype=np.float64)
    finite = data[np.isfinite(data)]
    counts, bin_edges = np.histogram(finite, bins=bins)
    return bin_edges, counts
