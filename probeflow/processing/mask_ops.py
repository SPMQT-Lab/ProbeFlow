"""Morphological cleanup operations for boolean masks.

Thin, boolean-in/boolean-out wrappers over ``scipy.ndimage`` (trivial
morphology) and ``skimage.morphology`` (small-object / hole removal,
skeletonisation).  These are used by the Masks manager to tidy edge-detection
output before it is used as an active mask or converted to ROIs.

All sizes/radii are in **pixels**.  Functions never introduce NaN (input is
boolean) and always return a fresh boolean array of the same shape.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as _ndi


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool)


def _disk(radius: int) -> np.ndarray:
    """Disk-shaped structuring element of the given pixel radius."""
    r = max(1, int(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2) <= r ** 2


def invert(mask: np.ndarray) -> np.ndarray:
    """Logical complement of *mask*."""
    return ~_as_bool(mask)


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Grow True regions by a disk of *radius* px."""
    return _ndi.binary_dilation(_as_bool(mask), structure=_disk(radius))


def erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Shrink True regions by a disk of *radius* px."""
    return _ndi.binary_erosion(_as_bool(mask), structure=_disk(radius))


def binary_open(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Erosion followed by dilation — removes small protrusions/specks."""
    return _ndi.binary_opening(_as_bool(mask), structure=_disk(radius))


def binary_close(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Dilation followed by erosion — closes small gaps/holes."""
    return _ndi.binary_closing(_as_bool(mask), structure=_disk(radius))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed background regions inside True components."""
    return _ndi.binary_fill_holes(_as_bool(mask))


def remove_small_objects(mask: np.ndarray, min_size: int = 16) -> np.ndarray:
    """Drop connected True components smaller than *min_size* px."""
    from skimage.morphology import remove_small_objects as _rso
    m = _as_bool(mask)
    if not m.any():
        return m
    return _rso(m, min_size=max(1, int(min_size)))


def remove_small_holes(mask: np.ndarray, area_threshold: int = 16) -> np.ndarray:
    """Fill enclosed background holes smaller than *area_threshold* px."""
    from skimage.morphology import remove_small_holes as _rsh
    m = _as_bool(mask)
    if not m.any():
        return m
    return _rsh(m, area_threshold=max(1, int(area_threshold)))


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Reduce True regions to a 1-px-wide skeleton."""
    from skimage.morphology import skeletonize as _skel
    m = _as_bool(mask)
    if not m.any():
        return m
    return _skel(m)


def remove_border_objects(mask: np.ndarray) -> np.ndarray:
    """Remove True components touching the image border."""
    from skimage.segmentation import clear_border as _clear
    return _clear(_as_bool(mask))
