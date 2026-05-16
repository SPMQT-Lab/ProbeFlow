"""Tests for linear_undistort — shear_x, scale_y, NaN, errors, geometry."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing.image import linear_undistort


# ── helpers ───────────────────────────────────────────────────────────────────

def _stripe_image(Ny: int = 20, Nx: int = 20, col: int = 10) -> np.ndarray:
    arr = np.zeros((Ny, Nx), dtype=np.float64)
    arr[:, col] = 1.0
    return arr


def _flat(Ny: int = 10, Nx: int = 10, val: float = 1.0) -> np.ndarray:
    return np.full((Ny, Nx), val, dtype=np.float64)


# ── errors ────────────────────────────────────────────────────────────────────

class TestLinearUndistortErrors:
    def test_scale_y_zero_raises(self):
        with pytest.raises(ValueError, match="scale_y"):
            linear_undistort(np.ones((4, 4)), scale_y=0.0)

    def test_scale_y_negative_raises(self):
        with pytest.raises(ValueError, match="scale_y"):
            linear_undistort(np.ones((4, 4)), scale_y=-1.0)

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            linear_undistort(np.ones(10))

    def test_3d_input_raises(self):
        with pytest.raises(ValueError):
            linear_undistort(np.ones((4, 4, 4)))


# ── shear geometry ────────────────────────────────────────────────────────────

class TestLinearUndistortShear:
    """The inverse map is: src_x = c - shear_x * (r / (Ny-1)).

    A stripe at source column K is visible in the output at output column
    K + shear_x * (r / (Ny-1)).  At the last row that shift is shear_x.
    """

    def test_zero_shear_is_identity(self):
        arr = _stripe_image(20, 20, col=10)
        out = linear_undistort(arr, shear_x=0.0, scale_y=1.0)
        np.testing.assert_allclose(out, arr, atol=1e-10)

    def test_positive_shear_shifts_stripe_right_at_bottom(self):
        # Stripe at col=5 with shear_x=5 → appears at col=10 at the last row
        Ny, Nx = 20, 20
        arr = _stripe_image(Ny, Nx, col=5)
        out = linear_undistort(arr, shear_x=5.0, scale_y=1.0)
        # At last row, output col 10 maps to src col 10 - 5*(19/19)=5 → bright
        assert float(out[Ny - 1, 10]) > 0.5
        # And the original col=5 at last row maps to src col 5-5=0 → dark
        assert float(out[Ny - 1, 5]) < 0.5

    def test_negative_shear_shifts_stripe_left_at_bottom(self):
        # Stripe at col=10 with shear_x=-5 → appears at col=5 at last row
        Ny, Nx = 20, 20
        arr = _stripe_image(Ny, Nx, col=10)
        out = linear_undistort(arr, shear_x=-5.0, scale_y=1.0)
        # At last row, output col 5 maps to src col 5 - (-5)*1 = 10 → bright
        assert float(out[Ny - 1, 5]) > 0.5
        # At last row, output col 10 maps to src col 10 - (-5)*1 = 15 → dark
        assert float(out[Ny - 1, 10]) < 0.5

    def test_shear_no_effect_at_first_row(self):
        # At row 0, the shift is shear_x * (0/(Ny-1)) = 0, so first row unchanged
        Ny, Nx = 20, 20
        arr = _stripe_image(Ny, Nx, col=10)
        out = linear_undistort(arr, shear_x=5.0, scale_y=1.0)
        # Row 0: src_x = c - 0 = c → identical to input
        np.testing.assert_allclose(out[0, :], arr[0, :], atol=1e-10)

    def test_shear_and_negative_shear_are_symmetric(self):
        arr = _stripe_image(20, 20, col=10)
        out_pos = linear_undistort(arr, shear_x=3.0, scale_y=1.0)
        out_neg = linear_undistort(arr, shear_x=-3.0, scale_y=1.0)
        # Last row: pos shifts right, neg shifts left — patterns differ
        assert not np.allclose(out_pos[19, :], out_neg[19, :], atol=0.01)


# ── scale_y geometry ──────────────────────────────────────────────────────────

class TestLinearUndistortScaleY:
    """The inverse map is: src_y = r / scale_y.

    scale_y > 1: output row r maps to source row r/scale_y < r → top of source
    fills more output rows (content is stretched over the full output height).
    scale_y < 1: output row r maps to source row r/scale_y > r → reads deeper
    into the source faster (bottom content appears earlier in output).
    """

    def test_unit_scale_is_identity(self):
        arr = np.arange(100.0).reshape(10, 10)
        out = linear_undistort(arr, shear_x=0.0, scale_y=1.0)
        np.testing.assert_allclose(out, arr, atol=1e-10)

    def test_scale_gt1_stretches_content_down(self):
        # Bottom half of source is bright; with scale_y=2 only top half is read
        # (src_y at row 20 = 20/2 = 10 → in bottom half for a 40-row image)
        Ny = 40
        arr = np.zeros((Ny, 10), dtype=np.float64)
        arr[20:] = 1.0
        out = linear_undistort(arr, shear_x=0.0, scale_y=2.0)
        # Row 20: src_y = 10 → dark (in top half of source)
        assert float(out[9, 0]) < 0.5
        # Row 21: src_y = 10.5 → just past boundary → near 0.5 but heading bright
        # Row 40+: would be past source (reflected), so test safe interior
        # Row 38: src_y = 19 → dark (row 19 in source is dark)
        assert float(out[38, 0]) < 0.5

    def test_scale_lt1_brings_bottom_content_up(self):
        # With scale_y=0.5: output row r maps to source row 2r
        # Source: bright from row 20 on
        Ny = 40
        arr = np.zeros((Ny, 10), dtype=np.float64)
        arr[20:] = 1.0
        out = linear_undistort(arr, shear_x=0.0, scale_y=0.5)
        # Row 0: src_y = 0 → dark
        assert float(out[0, 0]) < 0.5
        # Row 10: src_y = 20 → bright
        assert float(out[10, 0]) > 0.5

    def test_output_shape_preserved(self):
        arr = np.ones((30, 20))
        out = linear_undistort(arr, scale_y=1.5)
        assert out.shape == (30, 20)


# ── NaN handling ──────────────────────────────────────────────────────────────

class TestLinearUndistortNaN:
    def test_nan_pixel_preserved_at_same_location(self):
        arr = np.ones((10, 10), dtype=np.float64)
        arr[5, 5] = np.nan
        out = linear_undistort(arr, shear_x=0.0, scale_y=1.0)
        assert np.isnan(out[5, 5])

    def test_nan_row_preserved(self):
        arr = np.ones((10, 10), dtype=np.float64)
        arr[7, :] = np.nan
        out = linear_undistort(arr, shear_x=2.0, scale_y=1.0)
        assert np.all(np.isnan(out[7, :]))

    def test_finite_pixels_remain_finite_with_nan(self):
        arr = np.ones((10, 10), dtype=np.float64)
        arr[3, 5] = np.nan
        out = linear_undistort(arr, shear_x=1.0, scale_y=1.0)
        assert np.isfinite(out[0, 0])
        assert np.isfinite(out[9, 9])

    def test_all_nan_returns_all_nan(self):
        arr = np.full((8, 8), np.nan)
        out = linear_undistort(arr, shear_x=1.0, scale_y=1.0)
        # With all-NaN, nanmean is nan; fill is nan; map_coordinates with nan input
        # The function fills nan_mask positions after the mapping with nan
        # Interior (not nan-masked originally) may be any value; nan-masked are nan
        assert np.all(np.isnan(out))

    def test_inf_treated_as_nan(self):
        arr = np.ones((10, 10), dtype=np.float64)
        arr[2, 3] = np.inf
        out = linear_undistort(arr, shear_x=0.0, scale_y=1.0)
        # inf is in the nan_mask, so it gets restored to nan in output
        assert not np.isfinite(out[2, 3])


# ── geometry: shape, non-square, combined ────────────────────────────────────

class TestLinearUndistortGeometry:
    def test_non_square_wide_shape_preserved(self):
        arr = np.ones((10, 30))
        out = linear_undistort(arr, shear_x=1.0, scale_y=1.0)
        assert out.shape == (10, 30)

    def test_non_square_tall_shape_preserved(self):
        arr = np.ones((30, 10))
        out = linear_undistort(arr, shear_x=1.0, scale_y=1.2)
        assert out.shape == (30, 10)

    def test_combined_shear_and_scale(self):
        # With both shear and scale, output should differ from identity
        arr = np.arange(100.0).reshape(10, 10)
        identity_out = linear_undistort(arr, shear_x=0.0, scale_y=1.0)
        combined_out = linear_undistort(arr, shear_x=2.0, scale_y=1.5)
        assert not np.allclose(identity_out, combined_out, atol=0.01)

    def test_output_dtype_is_float64(self):
        arr = np.ones((8, 8), dtype=np.float32)
        out = linear_undistort(arr)
        assert out.dtype == np.float64

    def test_single_row_image(self):
        # Ny=1 → Ny-1=0, so src_x = c - shear_x * (r/1) at r=0 → src_x = c
        arr = np.array([[1.0, 0.0, 0.0, 1.0]])
        out = linear_undistort(arr, shear_x=5.0, scale_y=1.0)
        assert out.shape == (1, 4)
