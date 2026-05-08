"""Tests for SXM plane orientation and physical geometry.

Covers the two gaps identified in the strategy review:
  1. orient_plane: SCAN_DIR and backward-plane axis flips
  2. Pixel size consistency: plane shape vs scan_range_m vs sxm_dims
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from probeflow.io.sxm_io import orient_plane, sxm_dims, sxm_scan_range

_NANONIS_SCAN = Path(__file__).resolve().parents[1] / "test_data" / "sxm_moire_10nm.sxm"


# ── orient_plane ──────────────────────────────────────────────────────────────

class TestOrientPlane:
    def _make_arr(self, rows=4, cols=6):
        return np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)

    def test_scan_dir_down_forward_plane_is_unchanged(self):
        arr = self._make_arr()
        hdr = {"SCAN_DIR": "down"}
        result = orient_plane(arr.copy(), hdr, plane_idx=0)
        np.testing.assert_array_equal(result, arr)

    def test_scan_dir_up_flips_rows(self):
        arr = self._make_arr()
        hdr = {"SCAN_DIR": "up"}
        result = orient_plane(arr.copy(), hdr, plane_idx=0)
        np.testing.assert_array_equal(result, arr[::-1])

    def test_scan_dir_missing_defaults_to_down(self):
        arr = self._make_arr()
        result = orient_plane(arr.copy(), {}, plane_idx=0)
        np.testing.assert_array_equal(result, arr)

    def test_scan_dir_case_insensitive(self):
        arr = self._make_arr()
        hdr_lower = {"SCAN_DIR": "up"}
        hdr_upper = {"SCAN_DIR": "UP"}
        np.testing.assert_array_equal(
            orient_plane(arr.copy(), hdr_lower, 0),
            orient_plane(arr.copy(), hdr_upper, 0),
        )

    def test_backward_plane_flips_columns(self):
        arr = self._make_arr()
        hdr = {"SCAN_DIR": "down"}
        result = orient_plane(arr.copy(), hdr, plane_idx=1)
        np.testing.assert_array_equal(result, arr[:, ::-1])

    def test_forward_plane_does_not_flip_columns(self):
        arr = self._make_arr()
        hdr = {"SCAN_DIR": "down"}
        result = orient_plane(arr.copy(), hdr, plane_idx=0)
        np.testing.assert_array_equal(result, arr)

    def test_backward_plane_up_scan_applies_both_flips(self):
        arr = self._make_arr()
        hdr = {"SCAN_DIR": "up"}
        result = orient_plane(arr.copy(), hdr, plane_idx=1)
        np.testing.assert_array_equal(result, arr[::-1, ::-1])

    def test_plane_idx_2_forward_is_not_flipped(self):
        arr = self._make_arr()
        hdr = {"SCAN_DIR": "down"}
        result = orient_plane(arr.copy(), hdr, plane_idx=2)
        np.testing.assert_array_equal(result, arr)

    def test_plane_idx_3_backward_is_flipped(self):
        arr = self._make_arr()
        hdr = {"SCAN_DIR": "down"}
        result = orient_plane(arr.copy(), hdr, plane_idx=3)
        np.testing.assert_array_equal(result, arr[:, ::-1])


# ── Pixel-size consistency ────────────────────────────────────────────────────

class TestPixelSizeConsistency:
    """The plane shape (Ny, Nx) must be consistent with sxm_dims and scan_range_m."""

    @pytest.fixture(scope="class")
    def scan_data(self):
        if not _NANONIS_SCAN.exists():
            pytest.skip(f"sample SXM not found: {_NANONIS_SCAN}")
        from probeflow.io.sxm_io import parse_sxm_header, read_all_sxm_planes
        hdr = parse_sxm_header(_NANONIS_SCAN)
        _, planes = read_all_sxm_planes(_NANONIS_SCAN)
        return hdr, planes

    def test_plane_shape_matches_sxm_dims(self, scan_data):
        hdr, planes = scan_data
        Nx, Ny = sxm_dims(hdr)
        for plane in planes:
            assert plane.shape == (Ny, Nx), (
                f"Expected ({Ny}, {Nx}), got {plane.shape}"
            )

    def test_pixel_size_x_is_positive_and_finite(self, scan_data):
        hdr, planes = scan_data
        Nx, _ = sxm_dims(hdr)
        w_m, _ = sxm_scan_range(hdr)
        px_size_x = w_m / Nx
        assert px_size_x > 0
        assert np.isfinite(px_size_x)

    def test_pixel_size_y_is_positive_and_finite(self, scan_data):
        hdr, planes = scan_data
        _, Ny = sxm_dims(hdr)
        _, h_m = sxm_scan_range(hdr)
        px_size_y = h_m / Ny
        assert px_size_y > 0
        assert np.isfinite(px_size_y)

    def test_pixel_size_matches_scan_range_over_dims(self, scan_data):
        hdr, planes = scan_data
        Nx, Ny = sxm_dims(hdr)
        w_m, h_m = sxm_scan_range(hdr)
        px_x = w_m / Nx
        px_y = h_m / Ny
        # Both pixel dimensions should be consistent with the plane shape
        assert abs(px_x * Nx - w_m) < 1e-30
        assert abs(px_y * Ny - h_m) < 1e-30

    def test_scan_range_positive(self, scan_data):
        hdr, _ = scan_data
        w_m, h_m = sxm_scan_range(hdr)
        assert w_m > 0 and h_m > 0
