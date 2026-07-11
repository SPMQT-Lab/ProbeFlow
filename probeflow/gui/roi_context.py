"""Small GUI-free helpers for ROI-derived viewer context."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from probeflow.core import AREA_ROI_KINDS


@dataclass(frozen=True)
class PointSource:
    """One downstream feature-point source in pixel and physical coordinates."""

    label: str
    source_type: str
    points_px: np.ndarray
    points_m: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class ROIContext:
    """Selected or active ROI resolved for a viewer action."""

    roi_id: str | None
    roi: Any | None
    source: str


def selected_roi_ids_for_context(roi_set: Any = None, roi_dock: Any = None) -> list[str]:
    """Return dock-selected ROI IDs, falling back to the active ROI ID."""
    selected = _dock_selected_roi_ids(roi_dock)
    if selected:
        return selected
    active_id = getattr(roi_set, "active_roi_id", None)
    return [active_id] if active_id else []


def selected_or_active_roi_context(
    roi_set: Any = None,
    roi_dock: Any = None,
) -> ROIContext:
    """Return the first dock-selected ROI, falling back to the active ROI."""
    selected = _dock_selected_roi_ids(roi_dock)
    if selected:
        roi_id = selected[0]
        return ROIContext(roi_id=roi_id, roi=_roi_by_id(roi_set, roi_id), source="selected")
    active_id = getattr(roi_set, "active_roi_id", None)
    return ROIContext(
        roi_id=active_id,
        roi=_roi_by_id(roi_set, active_id),
        source="active" if active_id else "none",
    )


def active_line_roi_context(roi_set: Any = None) -> ROIContext:
    """Return the active line ROI, or an empty context."""
    active_id = getattr(roi_set, "active_roi_id", None)
    roi = _roi_by_id(roi_set, active_id)
    if roi is not None and getattr(roi, "kind", None) == "line":
        return ROIContext(roi_id=active_id, roi=roi, source="active")
    return ROIContext(roi_id=None, roi=None, source="active")


def selected_or_active_area_roi_context(
    roi_set: Any = None,
    roi_dock: Any = None,
) -> ROIContext:
    """Return the selected-or-active ROI only when it is an area ROI."""
    ctx = selected_or_active_roi_context(roi_set, roi_dock)
    if ctx.roi is not None and getattr(ctx.roi, "kind", None) in AREA_ROI_KINDS:
        return ctx
    return ROIContext(roi_id=None, roi=None, source=ctx.source)


def active_area_roi_context(roi_set: Any = None) -> ROIContext:
    """Return the active ROI only when it is an area ROI."""
    active_id = getattr(roi_set, "active_roi_id", None)
    roi = _roi_by_id(roi_set, active_id)
    if roi is not None and getattr(roi, "kind", None) in AREA_ROI_KINDS:
        return ROIContext(roi_id=active_id, roi=roi, source="active")
    return ROIContext(roi_id=None, roi=None, source="active")


def selected_area_roi_contexts(
    roi_set: Any = None,
    roi_dock: Any = None,
) -> list[ROIContext]:
    """Return area ROI contexts from the current dock selection."""
    contexts: list[ROIContext] = []
    for roi_id in _dock_selected_roi_ids(roi_dock):
        roi = _roi_by_id(roi_set, roi_id)
        if roi is not None and getattr(roi, "kind", None) in AREA_ROI_KINDS:
            contexts.append(ROIContext(roi_id=roi_id, roi=roi, source="selected"))
    return contexts


def area_roi_mask(
    roi: Any,
    image_shape: tuple[int, int],
    *,
    require_non_empty: bool = True,
) -> np.ndarray | None:
    """Return a boolean mask for an area ROI, or None when invalid."""
    if roi is None or getattr(roi, "kind", None) not in AREA_ROI_KINDS:
        return None
    try:
        mask = np.asarray(roi.to_mask(image_shape), dtype=bool)
    except Exception:
        return None
    if require_non_empty and not mask.any():
        return None
    return mask


def active_area_roi_bounds(
    roi_set: Any = None,
    image_shape: tuple[int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    """Return the active area ROI's pixel bounds, or None.

    Bounds are ``(row_min, row_max, col_min, col_max)`` inclusive and clipped to
    *image_shape* (matching :meth:`probeflow.core.roi.ROI.bounds`). Returns None
    when there is no active area ROI or its mask is empty, so callers can fall
    back to the whole image.
    """
    if image_shape is None:
        return None
    ctx = active_area_roi_context(roi_set)
    mask = area_roi_mask(ctx.roi, image_shape)
    if mask is None:
        return None
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return (int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1]))


def collect_point_source_records(
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    measurement_points: Iterable[Any] = (),
    measurement_metadata: dict[str, object] | None = None,
    roi_set: Any = None,
    selected_roi_ids: Iterable[str] = (),
) -> list[PointSource]:
    """Collect available point sources with metadata for downstream tools."""
    px_x = float(pixel_size_x_m)
    px_y = float(pixel_size_y_m)
    sources: list[PointSource] = []

    measured_points = list(measurement_points or [])
    if measured_points:
        points_px = _points_to_array(measured_points)
        sources.append(PointSource(
            label="Detected feature maxima",
            source_type="feature_maxima",
            points_px=points_px,
            points_m=_scale_points(points_px, px_x, px_y),
            metadata=_metadata_with_count(measurement_metadata or {}, len(measured_points)),
        ))

    if roi_set is not None:
        selected_points = _selected_point_rois(roi_set, selected_roi_ids)
        if selected_points:
            points_px = _roi_points_to_array(selected_points)
            sources.append(PointSource(
                label="Selected point ROIs",
                source_type="selected_point_rois",
                points_px=points_px,
                points_m=_scale_points(points_px, px_x, px_y),
                metadata={
                    "selection_scope": "selected_point_rois",
                    "point_count": len(selected_points),
                },
            ))
        all_points = [roi for roi in roi_set.rois if roi.kind == "point"]
        if all_points:
            points_px = _roi_points_to_array(all_points)
            sources.append(PointSource(
                label="All point ROIs",
                source_type="all_point_rois",
                points_px=points_px,
                points_m=_scale_points(points_px, px_x, px_y),
                metadata={
                    "selection_scope": "all_point_rois",
                    "point_count": len(all_points),
                },
            ))

    return sources


def collect_point_sources_m(
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    measurement_points: Iterable[Any] = (),
    measurement_metadata: dict[str, object] | None = None,
    roi_set: Any = None,
    selected_roi_ids: Iterable[str] = (),
) -> dict[str, np.ndarray]:
    """Collect available point sources as (N, 2) arrays in metres."""
    return point_source_arrays_m(collect_point_source_records(
        pixel_size_x_m=pixel_size_x_m,
        pixel_size_y_m=pixel_size_y_m,
        measurement_points=measurement_points,
        measurement_metadata=measurement_metadata,
        roi_set=roi_set,
        selected_roi_ids=selected_roi_ids,
    ))


def collect_point_sources_px(
    *,
    pixel_size_x_m: float = 1.0,
    pixel_size_y_m: float = 1.0,
    measurement_points: Iterable[Any] = (),
    measurement_metadata: dict[str, object] | None = None,
    roi_set: Any = None,
    selected_roi_ids: Iterable[str] = (),
) -> dict[str, np.ndarray]:
    """Collect available point sources as (N, 2) arrays in pixel coordinates."""
    return point_source_arrays_px(collect_point_source_records(
        pixel_size_x_m=pixel_size_x_m,
        pixel_size_y_m=pixel_size_y_m,
        measurement_points=measurement_points,
        measurement_metadata=measurement_metadata,
        roi_set=roi_set,
        selected_roi_ids=selected_roi_ids,
    ))


def point_source_metadata(sources: Iterable[PointSource]) -> dict[str, dict[str, object]]:
    """Return source metadata keyed by point-source label."""
    return {
        source.label: {
            **source.metadata,
            "point_source_type": source.source_type,
            "point_count": int(len(source.points_px)),
        }
        for source in sources
    }


def point_source_arrays_m(sources: Iterable[PointSource]) -> dict[str, np.ndarray]:
    """Return metre-coordinate arrays keyed by point-source label."""
    return {source.label: source.points_m for source in sources}


def point_source_arrays_px(sources: Iterable[PointSource]) -> dict[str, np.ndarray]:
    """Return pixel-coordinate arrays keyed by point-source label."""
    return {source.label: source.points_px for source in sources}


def active_area_roi_area_m2(
    active_roi: Any,
    image_shape: tuple[int, int],
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> float | None:
    """Return the physical area of an active area ROI, or None."""
    mask = area_roi_mask(active_roi, image_shape)
    if mask is None:
        return None
    return float(mask.sum()) * float(pixel_size_x_m) * float(pixel_size_y_m)


def _metadata_with_count(metadata: dict[str, object], point_count: int) -> dict[str, object]:
    cleaned = {key: value for key, value in metadata.items() if value is not None}
    cleaned["point_count"] = int(point_count)
    return cleaned


def _scale_points(points_px: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    arr = np.asarray(points_px, dtype=float)
    if arr.size == 0:
        return arr.reshape((0, 2))
    scale = np.array([float(scale_x), float(scale_y)], dtype=float)
    return arr * scale


def _selected_point_rois(roi_set: Any, selected_roi_ids: Iterable[str]) -> list[Any]:
    selected: list[Any] = []
    for roi_id in selected_roi_ids:
        roi = roi_set.get(roi_id)
        if roi is not None and roi.kind == "point":
            selected.append(roi)
    return selected


def _dock_selected_roi_ids(roi_dock: Any = None) -> list[str]:
    if roi_dock is None:
        return []
    if hasattr(roi_dock, "selected_roi_ids"):
        try:
            return [str(roi_id) for roi_id in roi_dock.selected_roi_ids()]
        except Exception:
            return []
    if hasattr(roi_dock, "_selected_roi_id"):
        try:
            roi_id = roi_dock._selected_roi_id()
        except Exception:
            roi_id = None
        return [str(roi_id)] if roi_id else []
    return []


def _roi_by_id(roi_set: Any, roi_id: str | None) -> Any | None:
    if roi_set is None or not roi_id:
        return None
    return roi_set.get(roi_id)


def _points_to_array(points: Iterable[Any]) -> np.ndarray:
    return np.array([
        [float(point.x_px), float(point.y_px)]
        for point in points
    ])


def _roi_points_to_array(rois: Iterable[Any]) -> np.ndarray:
    return np.array([
        [float(roi.geometry["x"]), float(roi.geometry["y"])]
        for roi in rois
    ])
