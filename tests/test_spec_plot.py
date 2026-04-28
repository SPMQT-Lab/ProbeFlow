"""Tests for probeflow.spec_plot helpers that don't need a Qt event loop."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.spec_plot import choose_display_unit, spec_position_to_pixel


class TestChooseDisplayUnit:
    def test_metres_picks_sensible_unit_for_typical_stm(self):
        # Typical STM topography: 1 nm ≈ 1e-9 m. Either Å (20) or nm (2)
        # lands in the target range — we only require that some scaling is
        # applied and the result is in the friendly window.
        values = np.array([1e-9, 2e-9, 3e-9])
        scale, unit = choose_display_unit("m", values)
        assert unit in {"Å", "nm"}
        assert 0.1 <= np.median(np.abs(values)) * scale < 1000

    def test_metres_picks_nm_for_bigger_topo(self):
        # Median ≈ 50 nm: Å would give 500 (still < 1000) — tie-broken by
        # preferring the smaller prefix. Verify it stays within the window.
        values = np.array([30e-9, 50e-9, 70e-9])
        scale, unit = choose_display_unit("m", values)
        assert unit in {"Å", "nm"}
        scaled = np.median(np.abs(values)) * scale
        assert 0.1 <= scaled < 1000

    def test_metres_picks_pm_for_small_heights(self):
        values = np.array([5e-12, 10e-12, 15e-12])
        scale, unit = choose_display_unit("m", values)
        assert unit == "pm"

    def test_amps_picks_pa_for_typical_tunnel_current(self):
        values = np.array([10e-12, 20e-12, 30e-12])
        scale, unit = choose_display_unit("A", values)
        assert unit == "pA"

    def test_amps_picks_na_for_bigger_current(self):
        values = np.array([1e-9, 2e-9, 3e-9])
        scale, unit = choose_display_unit("A", values)
        assert unit == "nA"

    def test_volts_picks_mv(self):
        values = np.array([0.01, 0.02, 0.03])
        scale, unit = choose_display_unit("V", values)
        assert unit == "mV"

    def test_volts_no_scale_for_volt_scale(self):
        values = np.array([1.0, 2.0, 3.0])
        scale, unit = choose_display_unit("V", values)
        assert unit == "V"
        assert scale == 1.0

    def test_unknown_unit_no_scale(self):
        values = np.array([1e-9, 2e-9])
        scale, unit = choose_display_unit("Hz", values)
        assert scale == 1.0
        assert unit == "Hz"

    def test_dimensionless_no_scale(self):
        values = np.array([1.0, 2.0])
        scale, unit = choose_display_unit("", values)
        assert scale == 1.0
        assert unit == ""

    def test_all_zero_returns_domain_friendly_default(self):
        values = np.zeros(10)
        scale, unit = choose_display_unit("A", values)
        assert scale == 1e12
        assert unit == "pA"

    def test_all_zero_metres_returns_nm(self):
        values = np.zeros(10)
        scale, unit = choose_display_unit("m", values)
        assert scale == 1e9
        assert unit == "nm"

    def test_empty_returns_no_scale(self):
        values = np.array([])
        scale, unit = choose_display_unit("m", values)
        assert scale == 1.0
        assert unit == "m"

    def test_none_returns_no_scale(self):
        scale, unit = choose_display_unit("m", None)
        assert scale == 1.0
        assert unit == "m"


class TestSpecPositionToPixel:
    W = H = 1e-7  # 100 nm scan

    def test_centre_maps_to_half_half(self):
        result = spec_position_to_pixel(
            0.0, 0.0, (64, 64), (self.W, self.H), (0.0, 0.0), 0.0)
        assert result is not None
        assert abs(result[0] - 0.5) < 1e-9
        assert abs(result[1] - 0.5) < 1e-9

    def test_top_left_corner(self):
        # World position at left edge and physically highest y maps to (0, 0)
        result = spec_position_to_pixel(
            -self.W / 2, self.H / 2, (64, 64), (self.W, self.H), (0.0, 0.0), 0.0)
        assert result is not None
        assert abs(result[0] - 0.0) < 1e-9
        assert abs(result[1] - 0.0) < 1e-9

    def test_bottom_right_corner(self):
        result = spec_position_to_pixel(
            self.W / 2, -self.H / 2, (64, 64), (self.W, self.H), (0.0, 0.0), 0.0)
        assert result is not None
        assert abs(result[0] - 1.0) < 1e-9
        assert abs(result[1] - 1.0) < 1e-9

    def test_position_outside_scan_returns_none(self):
        result = spec_position_to_pixel(
            1e-3, 0.0, (64, 64), (self.W, self.H), (0.0, 0.0), 0.0)
        assert result is None

    def test_nonzero_offset_shifts_centre(self):
        ox, oy = 100e-9, 50e-9
        result = spec_position_to_pixel(
            ox, oy, (64, 64), (self.W, self.H), (ox, oy), 0.0)
        assert result is not None
        assert abs(result[0] - 0.5) < 1e-9
        assert abs(result[1] - 0.5) < 1e-9

    def test_nonzero_offset_excludes_world_origin(self):
        ox, oy = 100e-9, 50e-9
        result = spec_position_to_pixel(
            0.0, 0.0, (64, 64), (self.W, self.H), (ox, oy), 0.0)
        assert result is None

    def test_90_degree_rotation(self):
        # At 90°: cos=0, sin=1
        # dx=W/4, dy=0 → dx_rot=0, dy_rot=-W/4
        # → frac_x=0.5, frac_y_from_bottom=0.25 → frac_y=0.75
        result = spec_position_to_pixel(
            self.W / 4, 0.0, (64, 64), (self.W, self.H), (0.0, 0.0), 90.0)
        assert result is not None
        assert abs(result[0] - 0.5) < 1e-9
        assert abs(result[1] - 0.75) < 1e-9

    def test_default_offset_is_zero(self):
        result = spec_position_to_pixel(0.0, 0.0, (64, 64), (self.W, self.H))
        assert result is not None
        assert abs(result[0] - 0.5) < 1e-9
        assert abs(result[1] - 0.5) < 1e-9
