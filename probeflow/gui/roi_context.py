"""Small GUI-free helpers for ROI-derived viewer context."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from probeflow.core import AREA_ROI_KINDS


def collect_point_sources_m(
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    feature_finder_result: Any = None,
    measurement_points: Iterable[Any] = (),
    roi_set: Any = None,
    selected_roi_ids: Iterable[str] = (),
) -> dict[str, np.ndarray]:
    """Collect available point sources as (N, 2) arrays in metres."""
    return _collect_point_sources(
        feature_finder_result=feature_finder_result,
        measurement_points=measurement_points,
        roi_set=roi_set,
        selected_roi_ids=selected_roi_ids,
        scale_x=float(pixel_size_x_m),
        scale_y=float(pixel_size_y_m),
    )


def collect_point_sources_px(
    *,
    feature_finder_result: Any = None,
    measurement_points: Iterable[Any] = (),
    roi_set: Any = None,
    selected_roi_ids: Iterable[str] = (),
) -> dict[str, np.ndarray]:
    """Collect available point sources as (N, 2) arrays in pixel coordinates."""
    return _collect_point_sources(
        feature_finder_result=feature_finder_result,
        measurement_points=measurement_points,
        roi_set=roi_set,
        selected_roi_ids=selected_roi_ids,
        scale_x=1.0,
        scale_y=1.0,
    )


def active_area_roi_area_m2(
    active_roi: Any,
    image_shape: tuple[int, int],
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> float | None:
    """Return the physical area of an active area ROI, or None."""
    if active_roi is None or getattr(active_roi, "kind", None) not in AREA_ROI_KINDS:
        return None
    try:
        mask = active_roi.to_mask(image_shape)
    except Exception:
        return None
    return float(mask.sum()) * float(pixel_size_x_m) * float(pixel_size_y_m)


def _collect_point_sources(
    *,
    feature_finder_result: Any,
    measurement_points: Iterable[Any],
    roi_set: Any,
    selected_roi_ids: Iterable[str],
    scale_x: float,
    scale_y: float,
) -> dict[str, np.ndarray]:
    sources: dict[str, np.ndarray] = {}

    ff_points = list(getattr(feature_finder_result, "points", []) or [])
    if ff_points:
        sources["Feature result"] = _points_to_array(ff_points, scale_x, scale_y)

    measured_points = list(measurement_points or [])
    if measured_points:
        sources["Detected feature maxima"] = _points_to_array(
            measured_points,
            scale_x,
            scale_y,
        )

    if roi_set is not None:
        selected_points = _selected_point_rois(roi_set, selected_roi_ids)
        if selected_points:
            sources["Selected point ROIs"] = _roi_points_to_array(
                selected_points,
                scale_x,
                scale_y,
            )
        all_points = [roi for roi in roi_set.rois if roi.kind == "point"]
        if all_points:
            sources["All point ROIs"] = _roi_points_to_array(
                all_points,
                scale_x,
                scale_y,
            )

    return sources


def _selected_point_rois(roi_set: Any, selected_roi_ids: Iterable[str]) -> list[Any]:
    selected: list[Any] = []
    for roi_id in selected_roi_ids:
        roi = roi_set.get(roi_id)
        if roi is not None and roi.kind == "point":
            selected.append(roi)
    return selected


def _points_to_array(points: Iterable[Any], scale_x: float, scale_y: float) -> np.ndarray:
    return np.array([
        [float(point.x_px) * scale_x, float(point.y_px) * scale_y]
        for point in points
    ])


def _roi_points_to_array(rois: Iterable[Any], scale_x: float, scale_y: float) -> np.ndarray:
    return np.array([
        [float(roi.geometry["x"]) * scale_x, float(roi.geometry["y"]) * scale_y]
        for roi in rois
    ])
