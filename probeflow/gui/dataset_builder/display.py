"""Display-only helpers for Dataset Builder."""

from __future__ import annotations

import numpy as np

from probeflow.processing.background import subtract_background
from probeflow.processing.display import finite_values


def flatten_display_array(arr: np.ndarray) -> np.ndarray:
    """Return a display-only globally flattened copy of *arr*."""
    raw = np.asarray(arr, dtype=np.float64)
    flat = subtract_background(raw, order=1)
    try:
        offset = float(np.nanmedian(raw))
    except Exception:
        offset = 0.0
    return np.asarray(flat, dtype=np.float64) + offset


def dataset_builder_display_array(arr: np.ndarray, *, flatten: bool) -> np.ndarray:
    """Return the array that should be shown in Dataset Builder."""
    if flatten:
        return flatten_display_array(arr)
    return np.asarray(arr, dtype=np.float64)


def percentile_value(arr: np.ndarray, pct: float) -> float:
    """Return the value at *pct* for finite pixels in *arr*."""
    finite = finite_values(arr)
    if finite.size == 0:
        raise ValueError("Array contains no finite values.")
    return float(np.percentile(finite, float(pct)))


def value_to_percentile(arr: np.ndarray, value: float) -> float:
    """Return the percentile rank of *value* within the finite values of *arr*."""
    finite = finite_values(arr)
    if finite.size == 0:
        raise ValueError("Array contains no finite values.")
    return float(100.0 * np.count_nonzero(finite <= float(value)) / float(finite.size))
