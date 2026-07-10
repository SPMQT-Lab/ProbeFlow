"""Tests for GUI-free viewer tool launch contexts."""

from __future__ import annotations


import numpy as np
import pytest

from probeflow.core.roi import ROI
from probeflow.gui.roi_context import PointSource
from probeflow.gui.viewer.tool_launch import (
    IMAGE_REQUIRED_MESSAGE,
    POINT_SOURCE_REQUIRED_MESSAGE,
    lattice_grid_launch_context,
    pair_correlation_launch_context,
)


def _point_source() -> PointSource:
    points_px = np.array([[1.0, 2.0], [3.0, 4.0]])
    points_m = points_px * np.array([2e-9, 3e-9])
    return PointSource(
        label="Detected feature maxima",
        source_type="feature_maxima",
        points_px=points_px,
        points_m=points_m,
        metadata={"selection_scope": "roi"},
    )


def test_pair_correlation_launch_context_requires_point_sources():
    context = pair_correlation_launch_context(
        [],
        image_shape=(8, 8),
        pixel_size_x_m=2e-9,
        pixel_size_y_m=3e-9,
    )

    assert not context.ready
    assert context.status_message == POINT_SOURCE_REQUIRED_MESSAGE
    assert context.sources_m == {}


def test_pair_correlation_launch_context_records_active_area():
    roi = ROI.new("rectangle", {"x": 1, "y": 1, "width": 2, "height": 3})

    context = pair_correlation_launch_context(
        [_point_source()],
        active_area_roi=roi,
        image_shape=(8, 8),
        pixel_size_x_m=2e-9,
        pixel_size_y_m=3e-9,
    )

    assert context.ready
    np.testing.assert_allclose(
        context.sources_m["Detected feature maxima"],
        [[2e-9, 6e-9], [6e-9, 12e-9]],
    )
    assert context.source_metadata["Detected feature maxima"]["point_source_type"] == "feature_maxima"
    assert context.roi_area_m2 == pytest.approx(36e-18)




def test_lattice_grid_launch_context_requires_image():
    context = lattice_grid_launch_context(None, scan_range_m=None)

    assert not context.ready
    assert context.status_message == IMAGE_REQUIRED_MESSAGE
    assert context.image_shape is None
    assert context.scan_range_m is None


def test_lattice_grid_launch_context_uses_scan_range_or_image_shape():
    image = np.zeros((8, 12), dtype=float)

    explicit = lattice_grid_launch_context(image, scan_range_m=(5e-9, 6e-9))
    fallback = lattice_grid_launch_context(image, scan_range_m=None)

    assert explicit.ready
    assert explicit.image_shape == (8, 12)
    assert explicit.scan_range_m == pytest.approx((5e-9, 6e-9))
    assert fallback.scan_range_m == pytest.approx((12e-9, 8e-9))
