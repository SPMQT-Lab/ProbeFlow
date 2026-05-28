"""Tests for probeflow.measurements.raster.

Consolidates the two duplicate disk-rasterization loops flagged by
arch-backend #10 (2026-05-27 deep review).
"""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.measurements.raster import paint_disk, paint_point, paint_square


class TestPaintDisk:
    def test_single_pixel_at_zero_radius(self):
        arr = np.zeros((5, 5), dtype=bool)
        paint_disk(arr, 2, 2, 0, True)
        assert arr[2, 2]
        assert arr.sum() == 1

    def test_radius_one_paints_cross(self):
        arr = np.zeros((5, 5), dtype=bool)
        paint_disk(arr, 2, 2, 1, True)
        # Disk of radius 1 at (2,2) includes (1,2),(2,1),(2,2),(2,3),(3,2)
        # plus the centre.  At r=1, the 4-neighbours have d²=1 ≤ 1² ✓
        # Diagonals have d²=2 > 1 ✗
        assert arr[2, 2]
        assert arr[1, 2] and arr[3, 2]
        assert arr[2, 1] and arr[2, 3]
        assert not arr[1, 1]  # diagonal excluded
        assert arr.sum() == 5

    def test_clipping_at_boundaries(self):
        arr = np.zeros((4, 4), dtype=bool)
        paint_disk(arr, 0, 0, 2, True)
        # Only the in-bounds region is painted
        assert arr[0, 0]
        # Should not crash; bounds-clipped


class TestPaintSquare:
    def test_zero_radius_single_pixel(self):
        arr = np.zeros((4, 4), dtype=bool)
        paint_square(arr, 2, 2, 0, True)
        assert arr[2, 2]
        assert arr.sum() == 1

    def test_radius_one_paints_3x3(self):
        arr = np.zeros((5, 5), dtype=bool)
        paint_square(arr, 2, 2, 1, True)
        assert arr[1:4, 1:4].all()
        assert arr.sum() == 9


class TestPaintPoint:
    def test_disk_mode(self):
        arr = np.zeros((5, 5), dtype=bool)
        paint_point(arr, 2, 2, radius_px=1, shape_mode="disk", value=True)
        # 5-pixel cross
        assert arr.sum() == 5

    def test_square_mode(self):
        arr = np.zeros((5, 5), dtype=bool)
        paint_point(arr, 2, 2, radius_px=1, shape_mode="square", value=True)
        # 3x3 square
        assert arr.sum() == 9

    def test_subpixel_coords_rounded(self):
        arr = np.zeros((5, 5), dtype=bool)
        paint_point(arr, 2.4, 2.6, radius_px=0, value=True)
        # x_px=2.4 → cx=2, y_px=2.6 → cy=3
        assert arr[3, 2]

    def test_invalid_mode_raises(self):
        arr = np.zeros((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="shape_mode"):
            paint_point(arr, 0, 0, radius_px=1, shape_mode="kaiser")


class TestLegacyWrappersDelegate:
    """Verify the two legacy helpers still produce the same output via
    paint_point/paint_disk."""

    def test_feature_points_to_image_via_shared_paint(self):
        from probeflow.analysis.feature_finder import FeaturePoint, feature_points_to_image
        pts = [FeaturePoint(x_px=5.0, y_px=10.0, z_value=1.0)]
        out = feature_points_to_image(pts, (20, 20), radius_px=2.0, value=2.0)
        # Centre and 4-neighbours present at radius 2 (d²<=4)
        assert out[10, 5] == 2.0
        assert out[8, 5] == 2.0  # (10-8)²=4 ≤ 4
        assert out[10, 7] == 2.0
        assert out[10, 3] == 2.0

    def test_points_to_mask_via_shared_paint(self):
        from probeflow.measurements.fft_points import points_to_mask
        mask = points_to_mask([(5.0, 10.0)], (20, 20), radius_px=2, shape_mode="square")
        assert mask[10, 5]
        # 5x5 square around (5,10)
        assert mask[8:13, 3:8].all()
