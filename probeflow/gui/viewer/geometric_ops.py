"""Geometric-operation ROI-transform helper extracted from ImageViewerDialog."""

from __future__ import annotations

from typing import Callable


def transform_roi_set_for_display_op(
    roi_set,
    op_name: str,
    params: dict | None,
    array_shape: tuple[int, int] | None,
    *,
    status_fn: Callable[[str], None] | None = None,
    roi_changed_fn: Callable[[], None] | None = None,
) -> None:
    """Apply a geometric display op to every ROI in *roi_set*.

    Removes any ROIs that cannot be transformed (e.g. freehand masks that
    flip outside the image).  Calls *status_fn* with a one-line message when
    ROIs are dropped, and always calls *roi_changed_fn* on completion so the
    canvas stays in sync.
    """
    if roi_set is None or not roi_set.rois:
        return
    if array_shape is None:
        return
    params = params or {}
    invalidated = roi_set.transform_all(op_name, params, array_shape)
    if invalidated:
        invalid = set(invalidated)
        roi_set.rois = [roi for roi in roi_set.rois if roi.id not in invalid]
        if roi_set.active_roi_id in invalid:
            roi_set.active_roi_id = None
        if status_fn is not None:
            status_fn(f"{op_name} invalidated {len(invalidated)} ROI(s); removed them.")
    if roi_changed_fn is not None:
        roi_changed_fn()


def transform_mask_set_for_display_op(
    mask_set,
    op_name: str,
    params: dict | None,
    array_shape: tuple[int, int] | None,
    *,
    status_fn: Callable[[str], None] | None = None,
    mask_changed_fn: Callable[[], None] | None = None,
) -> None:
    """Apply a geometric display op to every mask in *mask_set*.

    Twin of :func:`transform_roi_set_for_display_op`.  Masks are rasters, so a
    flip/rotation transforms the boolean array to stay pixel-aligned with the
    image; resampling / shape-changing ops (rotate_arbitrary, scale, shear,
    affine) invalidate masks, which are then removed.  Without this, a
    same-shape op (h/v flip, 180° rotate) would leave masks geometrically stale
    while still passing the shape guard.
    """
    if mask_set is None or not mask_set.masks:
        return
    if array_shape is None:
        return
    params = params or {}
    invalidated = mask_set.transform_all(op_name, params, array_shape)
    if invalidated:
        invalid = set(invalidated)
        mask_set.masks = [m for m in mask_set.masks if m.id not in invalid]
        if mask_set.active_mask_id in invalid:
            mask_set.active_mask_id = None
        if status_fn is not None:
            status_fn(f"{op_name} invalidated {len(invalidated)} mask(s); removed them.")
    if mask_changed_fn is not None:
        mask_changed_fn()
