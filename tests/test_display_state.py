"""Tests for probeflow.processing.display_state.DisplayRangeState."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing.display import clip_range_from_array
from probeflow.processing.display_state import DisplayRangeState


# ── Test A: percentile mode computes limits from data ─────────────────────────

class TestPercentileMode:
    def test_default_mode_is_percentile(self):
        drs = DisplayRangeState()
        assert drs.mode == "percentile"

    def test_resolve_matches_clip_range_from_array(self):
        rng = np.random.default_rng(0)
        arr = rng.normal(loc=5.0, scale=1.0, size=(50, 50))
        drs = DisplayRangeState(low_pct=1.0, high_pct=99.0)
        vmin, vmax = drs.resolve(arr)
        expected_vmin, expected_vmax = clip_range_from_array(arr, 1.0, 99.0)
        assert abs(vmin - expected_vmin) < 1e-12
        assert abs(vmax - expected_vmax) < 1e-12

    def test_custom_percentiles(self):
        arr = np.linspace(0.0, 100.0, 1000)
        drs = DisplayRangeState(low_pct=5.0, high_pct=95.0)
        vmin, vmax = drs.resolve(arr)
        expected_vmin, expected_vmax = clip_range_from_array(arr, 5.0, 95.0)
        assert abs(vmin - expected_vmin) < 1e-10
        assert abs(vmax - expected_vmax) < 1e-10

    def test_all_nan_returns_none_none(self):
        arr = np.full((4, 4), np.nan)
        drs = DisplayRangeState()
        vmin, vmax = drs.resolve(arr)
        assert vmin is None and vmax is None

    def test_set_percentile_switches_mode(self):
        drs = DisplayRangeState()
        drs.set_manual(1.0, 5.0)
        assert drs.mode == "manual"
        drs.set_percentile(2.0, 98.0)
        assert drs.mode == "percentile"
        assert drs.low_pct == 2.0
        assert drs.high_pct == 98.0


# ── Test B: manual mode uses explicit limits ──────────────────────────────────

class TestManualMode:
    def test_resolve_returns_stored_limits(self):
        arr = np.linspace(0.0, 100.0, 1000)
        drs = DisplayRangeState()
        drs.set_manual(2.0, 5.0)
        vmin, vmax = drs.resolve(arr)
        assert vmin == pytest.approx(2.0)
        assert vmax == pytest.approx(5.0)

    def test_manual_limits_independent_of_percentile_settings(self):
        arr = np.linspace(0.0, 100.0, 1000)
        drs = DisplayRangeState(low_pct=1.0, high_pct=99.0)
        drs.set_manual(30.0, 70.0)
        vmin, vmax = drs.resolve(arr)
        # Must not match the 1%/99% limits
        pct_vmin, pct_vmax = clip_range_from_array(arr, 1.0, 99.0)
        assert abs(vmin - pct_vmin) > 1.0
        assert abs(vmax - pct_vmax) > 1.0
        assert vmin == pytest.approx(30.0)
        assert vmax == pytest.approx(70.0)

    def test_set_manual_sets_mode(self):
        drs = DisplayRangeState()
        drs.set_manual(1.0, 5.0)
        assert drs.mode == "manual"
        assert drs.vmin == pytest.approx(1.0)
        assert drs.vmax == pytest.approx(5.0)

    def test_manual_limits_not_clamped_to_percentile_range(self):
        # Limits outside the typical percentile range must work.
        drs = DisplayRangeState()
        drs.set_manual(-1e6, 1e6)
        arr = np.linspace(0.0, 1.0, 100)
        vmin, vmax = drs.resolve(arr)
        assert vmin == pytest.approx(-1e6)
        assert vmax == pytest.approx(1e6)

    def test_manual_ignores_nan_array(self):
        # Manual mode should not be affected by NaN content of the array.
        arr = np.full((4, 4), np.nan)
        drs = DisplayRangeState()
        drs.set_manual(2.0, 8.0)
        vmin, vmax = drs.resolve(arr)
        assert vmin == pytest.approx(2.0)
        assert vmax == pytest.approx(8.0)


# ── Test C: set_manual switches mode, leaving percentile fields intact ────────

class TestModeTransitions:
    def test_manual_preserves_percentile_fields(self):
        drs = DisplayRangeState(low_pct=5.0, high_pct=95.0)
        drs.set_manual(1.0, 2.0)
        assert drs.low_pct == 5.0
        assert drs.high_pct == 95.0

    def test_percentile_after_manual_resets_limits(self):
        arr = np.linspace(0.0, 100.0, 1000)
        drs = DisplayRangeState()
        drs.set_manual(40.0, 60.0)
        drs.set_percentile(1.0, 99.0)
        vmin, vmax = drs.resolve(arr)
        expected_vmin, expected_vmax = clip_range_from_array(arr, 1.0, 99.0)
        assert abs(vmin - expected_vmin) < 1e-10


# ── Test D: manual limits affect rendered output ──────────────────────────────

class TestManualLimitsAffectRendering:
    def test_different_manual_ranges_give_different_renders(self):
        from probeflow.processing.display import array_to_uint8

        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)

        drs1 = DisplayRangeState()
        drs1.set_manual(0.0, 0.5)
        vmin1, vmax1 = drs1.resolve(arr)
        u8_1 = array_to_uint8(arr, vmin=vmin1, vmax=vmax1)

        drs2 = DisplayRangeState()
        drs2.set_manual(0.0, 1.0)
        vmin2, vmax2 = drs2.resolve(arr)
        u8_2 = array_to_uint8(arr, vmin=vmin2, vmax=vmax2)

        assert not np.array_equal(u8_1, u8_2), "Different manual ranges must produce different renders"

    def test_narrow_manual_range_saturates_most_pixels(self):
        from probeflow.processing.display import array_to_uint8

        arr = np.linspace(0.0, 10.0, 100).reshape(10, 10)
        drs = DisplayRangeState()
        drs.set_manual(4.9, 5.1)  # very narrow range around middle
        vmin, vmax = drs.resolve(arr)
        u8 = array_to_uint8(arr, vmin=vmin, vmax=vmax)
        # Most pixels should be clipped to 0 or 255
        clipped = np.sum((u8 == 0) | (u8 == 255))
        assert clipped > u8.size * 0.8, "Narrow range should saturate most pixels"


# ── Test E: bars cannot cross (min_sep enforcement) ──────────────────────────

class TestNoCrossing:
    def test_equal_vmin_vmax_gets_separated(self):
        drs = DisplayRangeState()
        drs.set_manual(5.0, 5.0)
        assert drs.vmax > drs.vmin

    def test_inverted_range_gets_corrected(self):
        drs = DisplayRangeState()
        drs.set_manual(10.0, 5.0)  # vmin > vmax
        assert drs.vmax > drs.vmin

    def test_very_close_values(self):
        # 1.0 + 1e-31 rounds to 1.0 in float64 — set_manual must still separate them
        drs = DisplayRangeState()
        drs.set_manual(1.0, 1.0 + 1e-31)
        assert drs.vmax > drs.vmin
        # Also test with genuinely equal inputs
        drs2 = DisplayRangeState()
        drs2.set_manual(5.0, 5.0)
        assert drs2.vmax > drs2.vmin


# ── Test F: reset restores percentile mode ────────────────────────────────────

class TestReset:
    def test_reset_sets_percentile_mode(self):
        drs = DisplayRangeState()
        drs.set_manual(1.0, 2.0)
        drs.reset()
        assert drs.mode == "percentile"

    def test_reset_clears_manual_limits(self):
        drs = DisplayRangeState()
        drs.set_manual(1.0, 2.0)
        drs.reset()
        assert drs.vmin is None
        assert drs.vmax is None

    def test_reset_uses_default_percentiles(self):
        drs = DisplayRangeState()
        drs.reset()
        assert drs.low_pct == 1.0
        assert drs.high_pct == 99.0

    def test_reset_with_custom_percentiles(self):
        drs = DisplayRangeState()
        drs.reset(low_pct=5.0, high_pct=95.0)
        assert drs.low_pct == 5.0
        assert drs.high_pct == 95.0
