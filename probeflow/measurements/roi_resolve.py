"""Canonical ROI-or-precomputed-mask → bool array resolution.

Review arch-backend #11 (2026-05-28): the codebase previously had
near-identical private helpers in ``measurements/image.py``
(``_mask_from_roi_or_mask``) and ``measurements/features.py``
(``_roi_mask``).  Both did the same thing:

    if a precomputed mask is given:
        validate shape, return a bool copy
    elif a ROI is given:
        roi.to_mask(shape), validate, return
    else:
        return all-True

The kwarg names differed (``mask`` vs ``roi_mask``), and each call
site re-invented the shape-mismatch guard.

This module provides the single canonical helper that both modules
now wrap.
"""

from __future__ import annotations

from typing import Any

import numpy as np


__all__ = ["resolve_roi_to_mask"]


def resolve_roi_to_mask(
    image_shape: tuple[int, int],
    *,
    roi: Any | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a boolean 2-D array of ``image_shape`` for ROI selection.

    Resolution order:

    1. If ``mask`` is provided, validate its shape and return a bool copy.
    2. Else if ``roi`` is provided, call ``roi.to_mask(image_shape)`` and
       validate its shape.
    3. Else return ``np.ones(image_shape, dtype=bool)`` (all-True).

    Parameters
    ----------
    image_shape:
        ``(Ny, Nx)`` shape of the parent image.  Tuples longer than 2 are
        accepted and truncated (matches the prior ``_mask_from_roi_or_mask``
        contract that took the first two elements of ``arr.shape``).
    roi:
        Optional :class:`probeflow.core.roi.ROI` to rasterize.
    mask:
        Optional precomputed boolean mask.  Takes precedence over ``roi``
        when both are supplied (matches the prior measurements/* helpers).

    Returns
    -------
    np.ndarray of dtype bool, shape ``(Ny, Nx)``.

    Raises
    ------
    ValueError
        When the supplied ``mask`` or the rasterized ROI has a shape
        that disagrees with ``image_shape``.
    """
    shape = tuple(image_shape[:2])
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != shape:
            raise ValueError(
                f"mask shape {selected.shape} must match image shape {shape}"
            )
        return selected.copy()
    if roi is not None:
        selected = np.asarray(roi.to_mask(shape), dtype=bool)
        if selected.shape != shape:
            raise ValueError(
                f"ROI mask shape {selected.shape} must match image shape {shape}"
            )
        return selected
    return np.ones(shape, dtype=bool)
