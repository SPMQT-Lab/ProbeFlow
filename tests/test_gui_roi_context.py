"""Tests for GUI-free ROI context helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from probeflow.core.roi import ROI, ROISet
from probeflow.gui.roi_context import (
    active_area_roi_area_m2,
    collect_point_source_records,
    collect_point_sources_m,
    collect_point_sources_px,
    point_source_metadata,
)


def test_collect_point_sources_from_dialog_measurements_and_rois():
    roi_set = ROISet(image_id="scan")
    selected = ROI.new("point", {"x": 3, "y": 4}, name="selected")
    other = ROI.new("point", {"x": 8, "y": 9}, name="other")
    roi_set.add(selected)
    roi_set.add(other)
    feature_result = SimpleNamespace(
        points=[SimpleNamespace(x_px=1, y_px=2)],
        mode="minima",
        threshold_mode="below",
        threshold_high=0.2,
        threshold_low=None,
        min_distance_px=4.0,
        smoothing_sigma_px=1.5,
        message="Detected",
    )
    measured_points = [SimpleNamespace(x_px=5, y_px=6)]
    measurement_metadata = {
        "selection_scope": "roi",
        "threshold_mode": "percentile",
    }

    sources_m = collect_point_sources_m(
        pixel_size_x_m=2.0,
        pixel_size_y_m=3.0,
        feature_finder_result=feature_result,
        measurement_points=measured_points,
        measurement_metadata=measurement_metadata,
        roi_set=roi_set,
        selected_roi_ids=[selected.id],
    )
    sources_px = collect_point_sources_px(
        feature_finder_result=feature_result,
        measurement_points=measured_points,
        roi_set=roi_set,
        selected_roi_ids=[selected.id],
    )

    np.testing.assert_allclose(sources_m["Feature result"], [[2.0, 6.0]])
    np.testing.assert_allclose(sources_m["Detected feature maxima"], [[10.0, 18.0]])
    np.testing.assert_allclose(sources_m["Selected point ROIs"], [[6.0, 12.0]])
    np.testing.assert_allclose(sources_m["All point ROIs"], [[6.0, 12.0], [16.0, 27.0]])
    np.testing.assert_allclose(sources_px["All point ROIs"], [[3.0, 4.0], [8.0, 9.0]])

    records = collect_point_source_records(
        pixel_size_x_m=2.0,
        pixel_size_y_m=3.0,
        feature_finder_result=feature_result,
        measurement_points=measured_points,
        measurement_metadata=measurement_metadata,
        roi_set=roi_set,
        selected_roi_ids=[selected.id],
    )
    metadata = point_source_metadata(records)
    assert metadata["Feature result"]["point_source_type"] == "feature_finder"
    assert metadata["Feature result"]["detection_mode"] == "minima"
    assert metadata["Feature result"]["threshold_mode"] == "below"
    assert metadata["Feature result"]["point_count"] == 1
    assert metadata["Detected feature maxima"]["point_source_type"] == "feature_maxima"
    assert metadata["Detected feature maxima"]["selection_scope"] == "roi"
    assert metadata["Detected feature maxima"]["threshold_mode"] == "percentile"


def test_active_area_roi_area_uses_both_pixel_axes():
    roi = ROI.new("rectangle", {"x": 1, "y": 1, "width": 2, "height": 3})

    area = active_area_roi_area_m2(
        roi,
        (8, 8),
        pixel_size_x_m=2.0,
        pixel_size_y_m=3.0,
    )

    assert area == pytest.approx(36.0)
