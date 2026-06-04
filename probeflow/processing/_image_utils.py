"""Private shared helpers used across multiple image-processing submodules."""

from __future__ import annotations


import numpy as np
from scipy.ndimage import gaussian_filter


def _finite_mean(arr: np.ndarray, default: float = 0.0) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    return float(finite.mean()) if finite.size else float(default)


def _finite_median(arr: np.ndarray, default: float = 0.0) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    return float(np.median(finite)) if finite.size else float(default)


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nan_normalized_gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that ignores NaNs instead of mean-filling them."""
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        return np.full(a.shape, np.nan, dtype=np.float64)
    sigma = max(float(sigma), 0.0)
    values = np.where(finite, a, 0.0)
    weights = finite.astype(np.float64)
    blurred_values = gaussian_filter(values, sigma=sigma, mode="constant", cval=0.0)
    blurred_weights = gaussian_filter(weights, sigma=sigma, mode="constant", cval=0.0)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    np.divide(
        blurred_values,
        blurred_weights,
        out=out,
        where=blurred_weights > 1e-12,
    )
    return out
