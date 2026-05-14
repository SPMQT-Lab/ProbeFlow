"""Regression tests for display-range clipping helpers (no Qt required)."""

from __future__ import annotations

import numpy as np

from probeflow.gui import clip_range_from_arr


class TestClipRangeFromArr:
    def test_outlier_does_not_dominate_high(self):
        """A single extreme pixel must not set the high display limit."""
        rng = np.random.default_rng(0)
        arr = rng.normal(loc=1.0, scale=0.1, size=(50, 50))
        arr[0, 0] = 1e6
        _, hi = clip_range_from_arr(arr, pct_lo=1.0, pct_hi=99.0)
        assert hi is not None
        # 99th percentile of normal(1, 0.1) is ~1.23; outlier (1e6) must be excluded
        assert hi < 10.0, f"Expected high clip near 1.0, got {hi}"

    def test_outlier_does_not_dominate_low(self):
        rng = np.random.default_rng(1)
        arr = rng.normal(loc=1.0, scale=0.1, size=(50, 50))
        arr[0, 0] = -1e6
        lo, _ = clip_range_from_arr(arr, pct_lo=1.0, pct_hi=99.0)
        assert lo is not None
        assert lo > -10.0, f"Expected low clip near 1.0, got {lo}"

    def test_none_input(self):
        lo, hi = clip_range_from_arr(None)
        assert lo is None and hi is None

    def test_empty_array(self):
        lo, hi = clip_range_from_arr(np.array([]))
        assert lo is None and hi is None

    def test_all_nan(self):
        arr = np.full((10, 10), np.nan)
        lo, hi = clip_range_from_arr(arr)
        assert lo is None and hi is None

    def test_single_value_array(self):
        # One finite pixel: constant-image fallback widens the range so it renders
        arr = np.array([[np.nan, np.nan], [np.nan, 5.0]])
        lo, hi = clip_range_from_arr(arr)
        assert lo is not None and hi is not None
        assert hi > lo

    def test_constant_image_returns_range(self):
        arr = np.full((20, 20), 3.0)
        lo, hi = clip_range_from_arr(arr, 1.0, 99.0)
        # Percentiles of a constant are identical; fallback must widen the range
        assert lo is not None and hi is not None
        assert hi > lo

    def test_normal_array_percentiles_correct(self):
        arr = np.arange(100, dtype=float).reshape(10, 10)
        lo, hi = clip_range_from_arr(arr, 1.0, 99.0)
        expected_lo = float(np.percentile(arr, 1.0))
        expected_hi = float(np.percentile(arr, 99.0))
        assert abs(lo - expected_lo) < 1e-9
        assert abs(hi - expected_hi) < 1e-9

    def test_with_nan_values_excluded(self):
        arr = np.arange(100, dtype=float).reshape(10, 10)
        arr[0, :] = np.nan  # NaN row should be excluded
        lo, hi = clip_range_from_arr(arr, 0.0, 100.0)
        assert lo is not None
        assert lo >= 10.0  # first non-NaN row starts at 10

    def test_1d_array(self):
        arr = np.arange(1000, dtype=float)
        arr[0] = 1e9  # outlier at the bottom of index but huge value
        lo, hi = clip_range_from_arr(arr, 1.0, 99.0)
        assert lo is not None
        assert hi < 1e8
