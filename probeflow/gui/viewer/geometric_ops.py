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
