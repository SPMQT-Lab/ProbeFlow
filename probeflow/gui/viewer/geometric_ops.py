"""Geometric-operation ROI-transform helper extracted from ImageViewerDialog."""

from __future__ import annotations

from typing import Callable

from probeflow.core import op_vocab


_FRAME_CHANGING_OPS = frozenset({
    *op_vocab.LOSSLESS_OPS,
    "crop",
    "scale_image",
    "rotate_arbitrary",
    "shear",
    "linear_undistort",
    "affine_lattice_correction",
})

_INVALIDATING_OVERLAY_OPS = frozenset({
    "rotate_arbitrary",
    "shear",
    "linear_undistort",
    "affine_lattice_correction",
})


def coordinate_frame_ops(processing: dict | None) -> list[tuple[str, dict]]:
    """Return ordered GUI operations that change the image pixel frame."""
    out: list[tuple[str, dict]] = []
    state = processing or {}
    if state.get("linear_undistort"):
        shear_x = float(state.get("undistort_shear_x", 0.0))
        scale_y = float(state.get("undistort_scale_y", 1.0))
        if shear_x != 0.0 or scale_y != 1.0:
            out.append(("linear_undistort", {
                "shear_x": shear_x,
                "scale_y": scale_y,
            }))
    for item in state.get("geometric_ops") or ():
        if not isinstance(item, dict):
            continue
        op_name = op_vocab.to_short(str(item.get("op", "")))
        if op_name in _FRAME_CHANGING_OPS:
            out.append((op_name, dict(item.get("params") or {})))
    return out


def processing_changes_coordinate_frame(processing: dict | None) -> bool:
    """Whether GUI processing places overlays outside the raw pixel frame."""
    return bool(coordinate_frame_ops(processing))


def _drop_invalidated(items, active_attr: str, invalidated: list[str]) -> None:
    if not invalidated:
        return
    invalid = set(invalidated)
    if hasattr(items, "rois"):
        items.rois = [item for item in items.rois if item.id not in invalid]
    else:
        items.masks = [item for item in items.masks if item.id not in invalid]
    if getattr(items, active_attr, None) in invalid:
        setattr(items, active_attr, None)


def _transform_sets(roi_set, mask_set, op_name: str, params: dict,
                    image_shape: tuple[int, int]) -> None:
    if roi_set is not None and roi_set.rois:
        if op_name in _INVALIDATING_OVERLAY_OPS:
            invalidated = [roi.id for roi in roi_set.rois]
        else:
            invalidated = roi_set.transform_all(op_name, params, image_shape)
        _drop_invalidated(roi_set, "active_roi_id", invalidated)
    if mask_set is not None and mask_set.masks:
        if op_name in _INVALIDATING_OVERLAY_OPS:
            invalidated = [mask.id for mask in mask_set.masks]
        else:
            invalidated = mask_set.transform_all(op_name, params, image_shape)
        _drop_invalidated(mask_set, "active_mask_id", invalidated)


def _next_shape(op_name: str, params: dict,
                image_shape: tuple[int, int]) -> tuple[int, int] | None:
    ny, nx = image_shape
    if op_name in op_vocab.DIMENSION_SWAPPING_OPS:
        return nx, ny
    if op_name in op_vocab.LOSSLESS_OPS:
        return image_shape
    if op_name == "scale_image":
        return int(params["new_height"]), int(params["new_width"])
    if op_name == "crop":
        x0 = int(params["x0"]); y0 = int(params["y0"])
        x1 = int(params["x1"]); y1 = int(params["y1"])
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        if x1 < 0 or y1 < 0 or x0 > nx - 1 or y0 > ny - 1:
            return None
        x0, x1 = max(0, x0), min(x1, nx - 1)
        y0, y1 = max(0, y0), min(y1, ny - 1)
        params.update({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
        return y1 - y0 + 1, x1 - x0 + 1
    # These resampling operations invalidate every current overlay. Their exact
    # expanded shape is irrelevant once both sets are empty.
    return None


def project_overlay_sets_for_processing(
    roi_set,
    mask_set,
    processing: dict | None,
    raw_shape: tuple[int, int],
) -> None:
    """Project raw-coordinate sidecar models into the processed display frame.

    The inputs are mutated in place and must be disposable display copies, not
    the canonical models that will be written back to disk.
    """
    shape = (int(raw_shape[0]), int(raw_shape[1]))
    for op_name, raw_params in coordinate_frame_ops(processing):
        params = dict(raw_params)
        next_shape = _next_shape(op_name, params, shape)
        if op_name == "crop" and next_shape is None:
            # Processing replay skips a crop that misses the current image.
            continue
        _transform_sets(roi_set, mask_set, op_name, params, shape)
        if next_shape is None:
            return
        shape = next_shape


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
    canonical = op_vocab.to_short(op_name)
    if canonical in _INVALIDATING_OVERLAY_OPS:
        invalidated = [roi.id for roi in roi_set.rois]
    else:
        invalidated = roi_set.transform_all(canonical, params, array_shape)
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
    canonical = op_vocab.to_short(op_name)
    invalidated = mask_set.transform_all(canonical, params, array_shape)
    if invalidated:
        invalid = set(invalidated)
        mask_set.masks = [m for m in mask_set.masks if m.id not in invalid]
        if mask_set.active_mask_id in invalid:
            mask_set.active_mask_id = None
        if status_fn is not None:
            status_fn(f"{op_name} invalidated {len(invalidated)} mask(s); removed them.")
    if mask_changed_fn is not None:
        mask_changed_fn()
