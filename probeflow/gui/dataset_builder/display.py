"""Display-only helpers for Dataset Builder."""

from __future__ import annotations

import numpy as np

from probeflow.processing.background import subtract_background


def flatten_display_array(arr: np.ndarray) -> np.ndarray:
    """Return a display-only globally flattened copy of *arr*."""
    flat = subtract_background(np.asarray(arr, dtype=np.float64), order=1)
    return np.asarray(flat, dtype=np.float64)


def dataset_builder_display_array(arr: np.ndarray, *, flatten: bool) -> np.ndarray:
    """Return the array that should be shown in Dataset Builder."""
    if flatten:
        return flatten_display_array(arr)
    return np.asarray(arr, dtype=np.float64)

