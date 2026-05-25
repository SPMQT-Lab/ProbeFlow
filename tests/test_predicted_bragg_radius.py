"""Tests for probeflow.processing.filters.predicted_bragg_radius."""

from __future__ import annotations

import math

import pytest

from probeflow.processing.filters import bragg_shells, predicted_bragg_radius


# ── known-value checks ─────────────────────────────────────────────────────────

class TestBraggShells:
    def test_square_first_shell_factors(self):
        shells = bragg_shells("square", max_shells=4)
        assert [s.label for s in shells] == ["(10)", "(11)", "(20)", "(21)"]
        assert [s.factor for s in shells] == pytest.approx(
            [1.0, math.sqrt(2.0), 2.0, math.sqrt(5.0)]
        )

    def test_hex_first_shell_factors(self):
        shells = bragg_shells("hex", max_shells=4)
        assert [s.label for s in shells] == ["(10)", "(11)", "(20)", "(21)"]
        assert [s.factor for s in shells] == pytest.approx(
            [1.0, math.sqrt(3.0), 2.0, math.sqrt(7.0)]
        )

    def test_max_factor_limits_shells(self):
        shells = bragg_shells("square", max_shells=12, max_factor=2.0)
        assert [s.label for s in shells] == ["(10)", "(11)", "(20)"]

    def test_max_shells_caps_shells(self):
        shells = bragg_shells("hex", max_shells=2)
        assert len(shells) == 2
        assert [s.label for s in shells] == ["(10)", "(11)"]

class TestKnownValues:
    def test_square_order1_basic(self):
        # a=1 nm, scan=10 nm → r = 10/1 = 10.0 pixels
        r = predicted_bragg_radius(1e-9, "square", 10e-9, 256, order=1)
        assert r == pytest.approx(10.0)

    def test_hex_order1_basic(self):
        # a=1 nm, scan=10 nm → r = 2*10/sqrt(3) ≈ 11.547
        r = predicted_bragg_radius(1e-9, "hex", 10e-9, 256, order=1)
        assert r == pytest.approx(2.0 * 10.0 / math.sqrt(3.0))

    def test_square_order2(self):
        # order=2 for square → r = 10 * sqrt(2)
        r = predicted_bragg_radius(1e-9, "square", 10e-9, 256, order=2)
        assert r == pytest.approx(10.0 * math.sqrt(2.0))

    def test_hex_order2(self):
        # order=2 for hex → r = 2*10/sqrt(3) * sqrt(3) = 20.0
        r = predicted_bragg_radius(1e-9, "hex", 10e-9, 256, order=2)
        assert r == pytest.approx(20.0)


# ── scaling laws ───────────────────────────────────────────────────────────────

class TestScaling:
    def test_doubling_a_halves_radius_square(self):
        r1 = predicted_bragg_radius(1e-9, "square", 10e-9, 256)
        r2 = predicted_bragg_radius(2e-9, "square", 10e-9, 256)
        assert r2 == pytest.approx(r1 / 2.0)

    def test_doubling_a_halves_radius_hex(self):
        r1 = predicted_bragg_radius(1e-9, "hex", 10e-9, 256)
        r2 = predicted_bragg_radius(2e-9, "hex", 10e-9, 256)
        assert r2 == pytest.approx(r1 / 2.0)

    def test_doubling_scan_size_doubles_radius_square(self):
        r1 = predicted_bragg_radius(1e-9, "square", 10e-9, 256)
        r2 = predicted_bragg_radius(1e-9, "square", 20e-9, 256)
        assert r2 == pytest.approx(2.0 * r1)

    def test_doubling_scan_size_doubles_radius_hex(self):
        r1 = predicted_bragg_radius(1e-9, "hex", 10e-9, 256)
        r2 = predicted_bragg_radius(1e-9, "hex", 20e-9, 256)
        assert r2 == pytest.approx(2.0 * r1)


# ── order=2 is larger than order=1 ────────────────────────────────────────────

class TestOrderRelation:
    def test_square_order2_larger_than_order1(self):
        r1 = predicted_bragg_radius(1e-9, "square", 10e-9, 256, order=1)
        r2 = predicted_bragg_radius(1e-9, "square", 10e-9, 256, order=2)
        assert r2 > r1

    def test_hex_order2_larger_than_order1(self):
        r1 = predicted_bragg_radius(1e-9, "hex", 10e-9, 256, order=1)
        r2 = predicted_bragg_radius(1e-9, "hex", 10e-9, 256, order=2)
        assert r2 > r1


# ── n_pixels does not affect the result ───────────────────────────────────────

class TestNPixelsIgnored:
    @pytest.mark.parametrize("n", [64, 128, 256, 512, 1024])
    def test_n_pixels_square(self, n):
        r = predicted_bragg_radius(1e-9, "square", 10e-9, n)
        assert r == pytest.approx(10.0)

    @pytest.mark.parametrize("n", [64, 256, 1024])
    def test_n_pixels_hex(self, n):
        r = predicted_bragg_radius(1e-9, "hex", 10e-9, n)
        assert r == pytest.approx(2.0 * 10.0 / math.sqrt(3.0))


# ── error handling ─────────────────────────────────────────────────────────────

class TestValidation:
    def test_invalid_symmetry_raises(self):
        with pytest.raises(ValueError, match="symmetry"):
            predicted_bragg_radius(1e-9, "cubic", 10e-9, 256)

    def test_a_real_zero_raises(self):
        with pytest.raises(ValueError, match="a_real"):
            predicted_bragg_radius(0.0, "square", 10e-9, 256)

    def test_a_real_negative_raises(self):
        with pytest.raises(ValueError, match="a_real"):
            predicted_bragg_radius(-1e-9, "square", 10e-9, 256)

    def test_scan_size_zero_raises(self):
        with pytest.raises(ValueError, match="scan_size_m"):
            predicted_bragg_radius(1e-9, "square", 0.0, 256)

    def test_scan_size_negative_raises(self):
        with pytest.raises(ValueError, match="scan_size_m"):
            predicted_bragg_radius(1e-9, "square", -10e-9, 256)

    def test_n_pixels_zero_raises(self):
        with pytest.raises(ValueError, match="n_pixels"):
            predicted_bragg_radius(1e-9, "square", 10e-9, 0)

    def test_n_pixels_negative_raises(self):
        with pytest.raises(ValueError, match="n_pixels"):
            predicted_bragg_radius(1e-9, "square", 10e-9, -1)

    def test_order_invalid_raises(self):
        with pytest.raises(ValueError, match="order"):
            predicted_bragg_radius(1e-9, "square", 10e-9, 256, order=3)

    def test_order_zero_raises(self):
        with pytest.raises(ValueError, match="order"):
            predicted_bragg_radius(1e-9, "square", 10e-9, 256, order=0)
