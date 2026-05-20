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


def collect_point_source_records(
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    feature_finder_result: Any = None,
    measurement_points: Iterable[Any] = (),
    measurement_metadata: dict[str, object] | None = None,
    roi_set: Any = None,
    selected_roi_ids: Iterable[str] = (),
) -> list[PointSource]:
    """Collect available point sources with metadata for downstream tools."""
    px_x = float(pixel_size_x_m)
    px_y = float(pixel_size_y_m)
    sources: list[PointSource] = []

    ff_points = list(getattr(feature_finder_result, "points", []) or [])
    if ff_points:
        points_px = _points_to_array(ff_points)
        sources.append(PointSource(
            label="Feature result",
            source_type="feature_finder",
            points_px=points_px,
            points_m=_scale_points(points_px, px_x, px_y),
            metadata=_feature_finder_metadata(feature_finder_result, len(ff_points)),
        ))

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
    feature_finder_result: Any = None,
    measurement_points: Iterable[Any] = (),
    measurement_metadata: dict[str, object] | None = None,
    roi_set: Any = None,
    selected_roi_ids: Iterable[str] = (),
) -> dict[str, np.ndarray]:
    """Collect available point sources as (N, 2) arrays in metres."""
    return point_source_arrays_m(collect_point_source_records(
        pixel_size_x_m=pixel_size_x_m,
        pixel_size_y_m=pixel_size_y_m,
        feature_finder_result=feature_finder_result,
        measurement_points=measurement_points,
        measurement_metadata=measurement_metadata,
        roi_set=roi_set,
        selected_roi_ids=selected_roi_ids,
    ))


def collect_point_sources_px(
    *,
    pixel_size_x_m: float = 1.0,
    pixel_size_y_m: float = 1.0,
    feature_finder_result: Any = None,
    measurement_points: Iterable[Any] = (),
    measurement_metadata: dict[str, object] | None = None,
    roi_set: Any = None,
    selected_roi_ids: Iterable[str] = (),
) -> dict[str, np.ndarray]:
    """Collect available point sources as (N, 2) arrays in pixel coordinates."""
    return point_source_arrays_px(collect_point_source_records(
        pixel_size_x_m=pixel_size_x_m,
        pixel_size_y_m=pixel_size_y_m,
        feature_finder_result=feature_finder_result,
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
    if active_roi is None or getattr(active_roi, "kind", None) not in AREA_ROI_KINDS:
        return None
    try:
        mask = active_roi.to_mask(image_shape)
    except Exception:
        return None
    return float(mask.sum()) * float(pixel_size_x_m) * float(pixel_size_y_m)


def _feature_finder_metadata(result: Any, point_count: int) -> dict[str, object]:
    return _metadata_with_count({
        "detection_mode": getattr(result, "mode", None),
        "threshold_mode": getattr(result, "threshold_mode", None),
        "threshold_low": getattr(result, "threshold_low", None),
        "threshold_high": getattr(result, "threshold_high", None),
        "min_distance_px": getattr(result, "min_distance_px", None),
        "smoothing_sigma_px": getattr(result, "smoothing_sigma_px", None),
        "message": getattr(result, "message", None),
    }, point_count)


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
