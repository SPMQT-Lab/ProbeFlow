"""ROI CRUD operations and queries extracted from ImageViewerDialog."""

from __future__ import annotations

from typing import Callable


# ── CRUD operations ───────────────────────────────────────────────────────────

def activate_roi(roi_set, roi_id: str, on_changed: Callable) -> None:
    if roi_set is None:
        return
    roi_set.set_active(roi_id)
    on_changed()


def rename_roi(
    roi_set,
    roi_id: str,
    on_changed: Callable,
    parent=None,
) -> None:
    from PySide6.QtWidgets import QInputDialog
    roi = roi_set.get(roi_id) if roi_set else None
    if roi is None:
        return
    new_name, ok = QInputDialog.getText(parent, "Rename ROI", "New name:", text=roi.name)
    if ok and new_name.strip():
        roi.name = new_name.strip()
        on_changed()


def delete_roi(roi_set, roi_id: str, on_changed: Callable) -> None:
    if roi_set is None:
        return
    roi_set.remove(roi_id)
    on_changed()


def delete_active_roi(roi_set, on_changed: Callable) -> None:
    if roi_set is None:
        return
    active_id = roi_set.active_roi_id
    if active_id is not None:
        delete_roi(roi_set, active_id, on_changed)


def delete_all_rois(roi_set, on_changed: Callable) -> None:
    if roi_set is None:
        return
    for roi_id in [r.id for r in roi_set.rois]:
        roi_set.remove(roi_id)
    on_changed()


def invert_roi(
    roi_set,
    roi_id: str,
    array_shape: tuple[int, int] | None,
    on_changed: Callable,
) -> None:
    roi = roi_set.get(roi_id) if roi_set else None
    if roi is None or array_shape is None:
        return
    from probeflow.core import roi as _roi_module
    inverted = _roi_module.invert(roi, array_shape)
    roi_set.add(inverted)
    on_changed()


def invert_active_roi(
    roi_set,
    array_shape: tuple[int, int] | None,
    on_changed: Callable,
) -> None:
    if roi_set is None:
        return
    active_id = roi_set.active_roi_id
    if active_id is not None:
        invert_roi(roi_set, active_id, array_shape, on_changed)


def select_nth_roi(roi_set, n: int, on_changed: Callable) -> None:
    if roi_set is None:
        return
    rois = list(roi_set.rois)
    if 1 <= n <= len(rois):
        roi_set.set_active(rois[n - 1].id)
        on_changed()


# ── Canvas callbacks ──────────────────────────────────────────────────────────

def roi_canvas_created(
    roi_set,
    roi,
    on_changed: Callable,
    set_tool_fn: Callable[[str], None],
) -> None:
    """Handle a completed canvas drawing: add the ROI, make it active, pan."""
    if roi_set is None:
        return
    roi_set.add(roi)
    roi_set.set_active(roi.id)
    on_changed()
    set_tool_fn("pan")


def roi_canvas_moved(
    roi_set,
    roi_id: str,
    dx: int,
    dy: int,
    on_changed: Callable,
) -> None:
    """Handle a drag-move on the canvas: translate the ROI geometry."""
    if roi_set is None or (dx == 0 and dy == 0):
        return
    roi = roi_set.get(roi_id)
    if roi is None:
        return
    from probeflow.core.roi import translate as _translate
    new_roi = _translate(roi, float(dx), float(dy))
    roi_set.remove(roi_id)
    roi_set.add(new_roi)
    roi_set.set_active(new_roi.id)
    on_changed()


def roi_line_set_width(
    roi_set,
    roi_id: str,
    width: int,
    on_changed: Callable,
) -> None:
    """Update the averaging width stored in a line ROI's geometry."""
    if roi_set is None:
        return
    roi = roi_set.get(roi_id)
    if roi is None or roi.kind != "line":
        return
    from probeflow.core.roi import ROI as _ROI
    new_geom = dict(roi.geometry)
    new_geom["width"] = max(1, int(width))
    new_roi = _ROI(
        id=roi.id, name=roi.name, kind="line",
        geometry=new_geom,
        coord_system=roi.coord_system, linked_file=roi.linked_file,
    )
    roi_set.remove(roi_id)
    roi_set.add(new_roi)
    roi_set.set_active(roi_id)
    on_changed()


def roi_line_endpoint_changed(
    roi_set,
    roi_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    on_changed: Callable,
) -> None:
    """Handle an endpoint drag on a line ROI: update geometry in place."""
    if roi_set is None:
        return
    roi = roi_set.get(roi_id)
    if roi is None or roi.kind != "line":
        return
    from probeflow.core.roi import ROI as _ROI
    new_geom = {k: v for k, v in roi.geometry.items()
                if k not in ("x1", "y1", "x2", "y2")}
    new_geom.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    new_roi = _ROI(
        id=roi.id, name=roi.name, kind="line",
        geometry=new_geom,
        coord_system=roi.coord_system, linked_file=roi.linked_file,
    )
    roi_set.remove(roi_id)
    roi_set.add(new_roi)
    roi_set.set_active(roi_id)
    on_changed()


# ── Pure queries ──────────────────────────────────────────────────────────────

def active_roi_id(roi_set) -> str | None:
    if roi_set is None:
        return None
    return roi_set.active_roi_id


def active_roi(roi_set):
    """Return the active ROI object, or None."""
    aid = active_roi_id(roi_set)
    if roi_set is None or aid is None:
        return None
    return roi_set.get(aid)


def selected_or_active_roi_id(roi_set, roi_dock=None) -> str | None:
    """Return the dock-selected ROI id, falling back to the active ROI."""
    if roi_dock is not None:
        try:
            selected = roi_dock._selected_roi_id()
        except Exception:
            selected = None
        if selected:
            return selected
    return active_roi_id(roi_set)


def has_roi_aware_local_filter(state: dict) -> bool:
    """Return True if *state* contains any ROI-scoped local filter."""
    return bool(
        state.get("smooth_sigma")
        or state.get("highpass_sigma")
        or state.get("edge_method")
        or state.get("fft_mode") is not None
        or state.get("fft_soft_border")
    )
