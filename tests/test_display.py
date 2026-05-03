"""Tests for probeflow.processing.display — shared display-rendering helpers."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing.display import (
    array_to_uint8,
    clip_range_from_array,
    finite_values,
    histogram_from_array,
    normalise_array,
)


# ── finite_values ─────────────────────────────────────────────────────────────

class TestFiniteValues:
    def test_basic(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = finite_values(arr)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_removes_nan(self):
        arr = np.array([1.0, np.nan, 3.0])
        result = finite_values(arr)
        assert result.size == 2
        assert np.nan not in result

    def test_removes_inf(self):
        arr = np.array([1.0, np.inf, -np.inf, 4.0])
        result = finite_values(arr)
        assert result.size == 2

    def test_all_nan_returns_empty(self):
        arr = np.full((3, 3), np.nan)
        assert finite_values(arr).size == 0

    def test_2d_flattened(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = finite_values(arr)
        assert result.ndim == 1
        assert result.size == 4

    def test_integer_input(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = finite_values(arr)
        assert result.dtype == np.float64


# ── clip_range_from_array ─────────────────────────────────────────────────────

class TestClipRangeFromArray:
    def test_basic_range(self):
        arr = np.linspace(0.0, 100.0, 1000)
        vmin, vmax = clip_range_from_array(arr, 1.0, 99.0)
        assert vmin < vmax
        assert np.isfinite(vmin) and np.isfinite(vmax)

    def test_nan_pixels_ignored(self):
        arr = np.linspace(0.0, 100.0, 1000)
        arr[0] = np.nan
        arr[1] = np.inf
        vmin, vmax = clip_range_from_array(arr)
        assert np.isfinite(vmin) and np.isfinite(vmax)

    def test_single_outlier_does_not_set_high_limit(self):
        rng = np.random.default_rng(42)
        arr = rng.normal(loc=1e-10, scale=1e-11, size=(100, 100))
        arr[0, 0] = 1e6  # extreme outlier
        _, vmax = clip_range_from_array(arr, 1.0, 99.0)
        assert vmax < 1e3, f"Outlier dominated vmax: {vmax}"

    def test_all_nan_raises(self):
        arr = np.full((4, 4), np.nan)
        with pytest.raises(ValueError, match="no finite values"):
            clip_range_from_array(arr)

    def test_all_inf_raises(self):
        arr = np.full((4, 4), np.inf)
        with pytest.raises(ValueError, match="no finite values"):
            clip_range_from_array(arr)

    def test_constant_array_returns_finite_range(self):
        arr = np.full((10, 10), 5.0)
        vmin, vmax = clip_range_from_array(arr)
        assert np.isfinite(vmin) and np.isfinite(vmax)
        assert vmax > vmin

    def test_vmin_less_than_vmax(self):
        arr = np.arange(100.0).reshape(10, 10)
        vmin, vmax = clip_range_from_array(arr)
        assert vmin < vmax

    def test_custom_percentiles(self):
        arr = np.arange(100.0)
        vmin5, vmax95 = clip_range_from_array(arr, 5.0, 95.0)
        vmin1, vmax99 = clip_range_from_array(arr, 1.0, 99.0)
        assert vmin5 >= vmin1
        assert vmax95 <= vmax99


# ── normalise_array ───────────────────────────────────────────────────────────

class TestNormaliseArray:
    def test_output_range_is_zero_to_one(self):
        arr = np.linspace(0.0, 10.0, 50)
        out = normalise_array(arr, 0.0, 10.0)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_nan_mapped_to_zero(self):
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        out = normalise_array(arr, 0.0, 4.0)
        assert out[0, 1] == pytest.approx(0.0)

    def test_inf_mapped_to_zero(self):
        arr = np.array([[1.0, np.inf], [3.0, 4.0]])
        out = normalise_array(arr, 0.0, 4.0)
        assert out[0, 1] == pytest.approx(0.0)

    def test_constant_vmin_vmax_returns_zeros(self):
        arr = np.full((4, 4), 5.0)
        out = normalise_array(arr, 5.0, 5.0)
        assert (out == 0.0).all(), "Degenerate range should produce all-zero output"

    def test_dtype_is_float32(self):
        arr = np.ones((4, 4))
        out = normalise_array(arr, 0.0, 2.0)
        assert out.dtype == np.float32

    def test_shape_preserved(self):
        arr = np.ones((7, 13))
        out = normalise_array(arr, 0.0, 2.0)
        assert out.shape == (7, 13)


# ── array_to_uint8 ────────────────────────────────────────────────────────────

class TestArrayToUint8:
    def test_dtype_is_uint8(self):
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        out = array_to_uint8(arr)
        assert out.dtype == np.uint8

    def test_shape_preserved(self):
        arr = np.ones((7, 13))
        out = array_to_uint8(arr, vmin=0.0, vmax=2.0)
        assert out.shape == (7, 13)

    def test_values_in_range(self):
        arr = np.linspace(0.0, 100.0, 1000).reshape(25, 40)
        out = array_to_uint8(arr)
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_explicit_vmin_vmax(self):
        arr = np.array([[0.0, 0.5], [0.75, 1.0]])
        out = array_to_uint8(arr, vmin=0.0, vmax=1.0)
        assert out[0, 0] == 0
        assert out[1, 1] == 255

    def test_nan_does_not_crash(self):
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        out = array_to_uint8(arr)
        assert out.dtype == np.uint8
        assert out.shape == arr.shape

    def test_outlier_does_not_dominate(self):
        rng = np.random.default_rng(0)
        arr = rng.normal(loc=5.0, scale=0.5, size=(50, 50))
        arr[0, 0] = 1e9
        out = array_to_uint8(arr)
        assert int(out.max()) == 255
        assert int(out.min()) >= 0

    def test_all_nan_raises(self):
        arr = np.full((4, 4), np.nan)
        with pytest.raises(ValueError, match="no finite values"):
            array_to_uint8(arr)

    def test_constant_image_no_warnings(self):
        arr = np.full((8, 8), 3.14)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = array_to_uint8(arr)
        assert out.dtype == np.uint8

    def test_custom_clip_percentiles(self):
        arr = np.linspace(0.0, 100.0, 1000).reshape(25, 40)
        out_tight = array_to_uint8(arr, clip_percentiles=(10.0, 90.0))
        out_wide  = array_to_uint8(arr, clip_percentiles=(1.0, 99.0))
        assert out_tight.dtype == np.uint8
        assert out_wide.dtype  == np.uint8


# ── histogram_from_array ──────────────────────────────────────────────────────

class TestHistogramFromArray:
    def test_returns_counts_and_edges(self):
        arr = np.linspace(0.0, 100.0, 1000)
        counts, edges = histogram_from_array(arr)
        assert len(counts) == 256
        assert len(edges)  == 257

    def test_custom_bins(self):
        arr = np.linspace(0.0, 10.0, 200)
        counts, edges = histogram_from_array(arr, bins=64)
        assert len(counts) == 64

    def test_nan_excluded(self):
        arr = np.linspace(0.0, 10.0, 100)
        arr[0] = np.nan
        arr[1] = np.inf
        counts, edges = histogram_from_array(arr)
        assert counts.sum() <= 98  # at most the finite values
        assert counts.sum() > 0

    def test_all_nan_raises(self):
        arr = np.full((4, 4), np.nan)
        with pytest.raises(ValueError, match="no finite values"):
            histogram_from_array(arr)

    def test_bin_range_matches_clip_percentiles(self):
        rng = np.random.default_rng(1)
        arr = rng.normal(loc=0.0, scale=1.0, size=(100, 100))
        arr[0, 0] = 1e9
        counts, edges = histogram_from_array(arr, clip_percentiles=(1.0, 99.0))
        vmin, vmax = clip_range_from_array(arr, 1.0, 99.0)
        assert abs(edges[0]  - vmin) < 1e-10
        assert abs(edges[-1] - vmax) < 1e-10

    def test_counts_dtype_is_integer(self):
        arr = np.ones((10, 10))
        counts, _ = histogram_from_array(arr, bins=16)
        assert np.issubdtype(counts.dtype, np.integer)
