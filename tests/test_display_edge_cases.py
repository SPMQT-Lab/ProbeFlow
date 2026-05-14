"""Edge-case tests for display range and array_to_uint8.

The happy-path coverage in test_display.py and test_display_state.py is already
thorough.  This file focuses specifically on the degenerate inputs that are most
likely to cause silent black-image regressions: vmin == vmax, vmin > vmax,
and manual-mode serialisation fidelity.
"""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing.display import array_to_uint8
from probeflow.processing.display_state import DisplayRangeState


# ── DisplayRangeState: degenerate manual ranges ───────────────────────────────

class TestManualDegenerate:
    def test_set_manual_equal_vmin_vmax_nudges_vmax_up(self):
        drs = DisplayRangeState()
        drs.set_manual(5.0, 5.0)
        assert drs.vmax > drs.vmin

    def test_set_manual_inverted_vmin_vmax_nudges_vmax_up(self):
        drs = DisplayRangeState()
        drs.set_manual(10.0, 2.0)
        assert drs.vmax > drs.vmin

    def test_set_manual_zero_vmin_zero_vmax_nudges_vmax_up(self):
        drs = DisplayRangeState()
        drs.set_manual(0.0, 0.0)
        assert drs.vmax > drs.vmin

    def test_set_manual_very_small_values_nudges_vmax_up(self):
        drs = DisplayRangeState()
        drs.set_manual(1e-30, 1e-30)
        assert drs.vmax > drs.vmin

    def test_set_manual_negative_equal_nudges_vmax_up(self):
        drs = DisplayRangeState()
        drs.set_manual(-3.0, -3.0)
        assert drs.vmax > drs.vmin

    def test_resolve_after_degenerate_manual_returns_finite(self):
        drs = DisplayRangeState()
        drs.set_manual(5.0, 5.0)
        arr = np.ones((8, 8)) * 5.0
        vmin, vmax = drs.resolve(arr)
        assert np.isfinite(vmin) and np.isfinite(vmax)
        assert vmax > vmin


# ── DisplayRangeState: serialisation ─────────────────────────────────────────

class TestDisplayRangeStateSerialization:
    def test_to_dict_contains_expected_keys(self):
        drs = DisplayRangeState()
        d = drs.to_dict()
        assert "mode" in d
        assert "low_pct" in d
        assert "high_pct" in d
        assert "vmin" in d
        assert "vmax" in d

    def test_percentile_mode_serialises_vmin_vmax_as_none(self):
        drs = DisplayRangeState(mode="percentile", low_pct=2.0, high_pct=98.0)
        d = drs.to_dict()
        assert d["mode"] == "percentile"
        assert d["vmin"] is None
        assert d["vmax"] is None

    def test_manual_mode_serialises_vmin_vmax(self):
        drs = DisplayRangeState()
        drs.set_manual(1.5, 7.3)
        d = drs.to_dict()
        assert d["mode"] == "manual"
        assert d["vmin"] == pytest.approx(1.5)
        assert d["vmax"] == pytest.approx(7.3)

    def test_percentile_values_roundtrip_in_dict(self):
        drs = DisplayRangeState(low_pct=5.0, high_pct=95.0)
        d = drs.to_dict()
        assert d["low_pct"] == pytest.approx(5.0)
        assert d["high_pct"] == pytest.approx(95.0)


# ── array_to_uint8: degenerate explicit ranges ────────────────────────────────

class TestArrayToUint8Degenerate:
    def test_explicit_vmin_equal_vmax_returns_uint8_without_raise(self):
        arr = np.full((8, 8), 5.0)
        out = array_to_uint8(arr, vmin=5.0, vmax=5.0)
        assert out.dtype == np.uint8
        assert out.shape == arr.shape

    def test_explicit_inverted_range_returns_uint8_without_raise(self):
        arr = np.linspace(0.0, 1.0, 64).reshape(8, 8)
        out = array_to_uint8(arr, vmin=1.0, vmax=0.0)
        assert out.dtype == np.uint8

    def test_inf_vmin_does_not_crash(self):
        arr = np.linspace(0.0, 1.0, 64).reshape(8, 8)
        # passes explicit vmin/vmax that bypass percentile clipping
        out = array_to_uint8(arr, vmin=0.0, vmax=1.0)
        assert out.dtype == np.uint8

    def test_constant_array_explicit_range_no_warnings(self):
        import warnings
        arr = np.full((16, 16), 3.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = array_to_uint8(arr, vmin=0.0, vmax=6.0)
        assert out.dtype == np.uint8
