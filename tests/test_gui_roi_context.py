"""Tests for GUI-free ROI context helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from probeflow.core.roi import ROI, ROISet
from probeflow.gui.roi_context import (
    active_area_roi_area_m2,
    active_area_roi_bounds,
    active_area_roi_context,
    active_line_roi_context,
    area_roi_mask,
    collect_point_source_records,
    collect_point_sources_m,
    collect_point_sources_px,
    point_source_metadata,
    selected_area_roi_contexts,
    selected_or_active_area_roi_context,
    selected_or_active_roi_context,
    selected_roi_ids_for_context,
)


class _FakeROIDock:
    def __init__(self, selected_ids):
        self._selected_ids = list(selected_ids)

    def selected_roi_ids(self):
        return list(self._selected_ids)


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


def test_roi_context_prefers_selected_roi_then_active_roi():
    roi_set = ROISet(image_id="scan")
    rect = ROI.new("rectangle", {"x": 1, "y": 1, "width": 2, "height": 2})
    line = ROI.new("line", {"x1": 0, "y1": 0, "x2": 4, "y2": 0})
    roi_set.add(rect)
    roi_set.add(line)
    roi_set.set_active(line.id)

    assert selected_roi_ids_for_context(roi_set, None) == [line.id]
    active_ctx = selected_or_active_roi_context(roi_set, None)
    assert active_ctx.roi_id == line.id
    assert active_ctx.source == "active"

    selected_ctx = selected_or_active_roi_context(roi_set, _FakeROIDock([rect.id]))
    assert selected_ctx.roi_id == rect.id
    assert selected_ctx.source == "selected"


def test_line_and_area_roi_contexts_validate_expected_kind():
    roi_set = ROISet(image_id="scan")
    rect = ROI.new("rectangle", {"x": 1, "y": 1, "width": 2, "height": 3})
    line = ROI.new("line", {"x1": 0, "y1": 0, "x2": 4, "y2": 0})
    roi_set.add(rect)
    roi_set.add(line)

    roi_set.set_active(line.id)
    line_ctx = active_line_roi_context(roi_set)
    assert line_ctx.roi_id == line.id
    assert active_area_roi_context(roi_set).roi_id is None

    area_ctx = selected_or_active_area_roi_context(roi_set, _FakeROIDock([rect.id]))
    assert area_ctx.roi_id == rect.id
    assert area_ctx.roi is rect

    non_area_ctx = selected_or_active_area_roi_context(roi_set, _FakeROIDock([line.id]))
    assert non_area_ctx.roi_id is None


def test_area_roi_mask_rejects_non_area_and_empty_masks():
    rect = ROI.new("rectangle", {"x": 1, "y": 1, "width": 2, "height": 3})
    line = ROI.new("line", {"x1": 0, "y1": 0, "x2": 4, "y2": 0})
    empty = ROI.new("rectangle", {"x": 20, "y": 20, "width": 2, "height": 2})

    mask = area_roi_mask(rect, (8, 8))

    assert mask is not None
    assert mask.dtype == bool
    assert mask.sum() == 6
    assert area_roi_mask(line, (8, 8)) is None
    assert area_roi_mask(empty, (8, 8)) is None
    assert area_roi_mask(empty, (8, 8), require_non_empty=False).sum() == 0


def test_selected_area_roi_contexts_returns_only_selected_area_rois():
    roi_set = ROISet(image_id="scan")
    rect_a = ROI.new("rectangle", {"x": 0, "y": 0, "width": 2, "height": 2})
    rect_b = ROI.new("rectangle", {"x": 4, "y": 4, "width": 2, "height": 2})
    line = ROI.new("line", {"x1": 0, "y1": 0, "x2": 4, "y2": 0})
    roi_set.add(rect_a)
    roi_set.add(rect_b)
    roi_set.add(line)
    roi_set.set_active(rect_a.id)

    contexts = selected_area_roi_contexts(
        roi_set,
        _FakeROIDock([rect_a.id, line.id, rect_b.id]),
    )

    assert [ctx.roi_id for ctx in contexts] == [rect_a.id, rect_b.id]


class TestActiveAreaRoiBounds:
    def test_bounds_for_active_rectangle(self):
        roi_set = ROISet(image_id="scan")
        # x=1, y=2, width=4, height=5 → inclusive bounds rows 2..6, cols 1..4
        rect = ROI.new("rectangle", {"x": 1, "y": 2, "width": 4, "height": 5})
        roi_set.add(rect)
        roi_set.set_active(rect.id)

        assert active_area_roi_bounds(roi_set, (10, 10)) == (2, 6, 1, 4)

    def test_none_when_active_is_not_area(self):
        roi_set = ROISet(image_id="scan")
        line = ROI.new("line", {"x1": 0, "y1": 0, "x2": 4, "y2": 0})
        roi_set.add(line)
        roi_set.set_active(line.id)

        assert active_area_roi_bounds(roi_set, (10, 10)) is None

    def test_none_when_no_roi_set(self):
        assert active_area_roi_bounds(None, (10, 10)) is None

    def test_none_when_shape_missing(self):
        roi_set = ROISet(image_id="scan")
        rect = ROI.new("rectangle", {"x": 1, "y": 2, "width": 4, "height": 5})
        roi_set.add(rect)
        roi_set.set_active(rect.id)

        assert active_area_roi_bounds(roi_set, None) is None

    def test_none_when_roi_outside_image(self):
        roi_set = ROISet(image_id="scan")
        rect = ROI.new("rectangle", {"x": 100, "y": 100, "width": 2, "height": 2})
        roi_set.add(rect)
        roi_set.set_active(rect.id)

        assert active_area_roi_bounds(roi_set, (10, 10)) is None

    def test_bounds_match_area_roi_mask_extent(self):
        # The bbox must equal the True-extent of area_roi_mask for the same ROI.
        roi_set = ROISet(image_id="scan")
        ellipse = ROI.new("ellipse", {"cx": 5.0, "cy": 6.0, "rx": 3.0, "ry": 2.0})
        roi_set.add(ellipse)
        roi_set.set_active(ellipse.id)

        bounds = active_area_roi_bounds(roi_set, (16, 16))
        mask = area_roi_mask(ellipse, (16, 16))
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        assert bounds == (int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1]))
