"""Tests for the lattice/grid measurement model and calibration."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from probeflow.analysis.lattice_grid import (
    LatticeGrid,
    LatticeGridDisplay,
    RealSpaceCalibration,
    ReciprocalCalibration,
    _fmt_angle_deg,
    direct_lattice_vectors_from_reciprocal_grid,
    format_real_space_measurements,
    format_reciprocal_measurements,
)


# ── helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")
    app = QApplication.instance()
    if app is not None:
        return app
    try:
        return QApplication([])
    except Exception as exc:
        pytest.skip(f"QApplication unavailable: {exc}")

def _angle(vx, vy):
    return math.degrees(math.atan2(vy, vx))


def _dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def _len(v):
    return math.hypot(*v)


# ── factory tests ─────────────────────────────────────────────────────────────

class TestFactories:
    def test_make_square(self):
        g = LatticeGrid.make_square(100, 100, 50)
        assert g.kind == "square"
        assert g.origin_px == (100, 100)
        assert abs(g.a_px[0] - 50) < 1e-9
        assert abs(g.a_px[1]) < 1e-9
        assert abs(g.b_px[0]) < 1e-9
        assert abs(g.b_px[1] - 50) < 1e-9

    def test_make_rectangular(self):
        g = LatticeGrid.make_rectangular(0, 0, 30, 60)
        assert g.kind == "rectangular"
        assert abs(g.a_px[0] - 30) < 1e-9
        assert abs(g.b_px[1] - 60) < 1e-9

    def test_make_hexagonal_default_angle(self):
        g = LatticeGrid.make_hexagonal(0, 0, 40)
        assert g.kind == "hexagonal"
        assert abs(_len(g.a_px) - 40) < 1e-9
        assert abs(_len(g.b_px) - 40) < 1e-9
        assert abs(g.angle_deg() - 60.0) < 1e-6

    def test_make_hexagonal_custom_angle(self):
        g = LatticeGrid.make_hexagonal(0, 0, 40, angle_deg=30)
        assert abs(_angle(*g.a_px) - 30) < 1e-9
        assert abs(_angle(*g.b_px) - 90) < 1e-9

    def test_make_square_space_default(self):
        g = LatticeGrid.make_square(0, 0, 10)
        assert g.space == "real"

    def test_make_square_space_reciprocal(self):
        g = LatticeGrid.make_square(0, 0, 10, space="reciprocal")
        assert g.space == "reciprocal"


# ── translation test ──────────────────────────────────────────────────────────

class TestTranslate:
    def test_translate_moves_origin(self):
        g = LatticeGrid.make_square(0, 0, 50)
        g2 = g.translate(10, -20)
        assert g2.origin_px == (10, -20)

    def test_translate_preserves_vectors(self):
        g = LatticeGrid.make_square(0, 0, 50)
        g2 = g.translate(99, -99)
        assert g2.a_px == g.a_px
        assert g2.b_px == g.b_px

    def test_translate_is_immutable(self):
        g = LatticeGrid.make_square(5, 5, 30)
        g.translate(100, 100)
        assert g.origin_px == (5, 5)


# ── rotation tests ────────────────────────────────────────────────────────────

class TestRotate:
    def test_rotate_90_square(self):
        g = LatticeGrid.make_square(0, 0, 50)
        g2 = g.rotate(90)
        # a = (50, 0) → should become (0, 50)
        assert abs(g2.a_px[0]) < 1e-9
        assert abs(g2.a_px[1] - 50) < 1e-9
        # b = (0, 50) → should become (-50, 0)
        assert abs(g2.b_px[0] + 50) < 1e-9
        assert abs(g2.b_px[1]) < 1e-9

    def test_rotate_preserves_lengths(self):
        g = LatticeGrid.make_hexagonal(0, 0, 40, angle_deg=0)
        g2 = g.rotate(37)
        assert abs(_len(g2.a_px) - _len(g.a_px)) < 1e-9
        assert abs(_len(g2.b_px) - _len(g.b_px)) < 1e-9

    def test_rotate_preserves_angle(self):
        g = LatticeGrid.make_hexagonal(0, 0, 40, angle_deg=0)
        g2 = g.rotate(37)
        assert abs(g2.angle_deg() - g.angle_deg()) < 1e-9

    def test_rotate_full_circle(self):
        g = LatticeGrid.make_square(0, 0, 50)
        g2 = g.rotate(360)
        assert abs(g2.a_px[0] - g.a_px[0]) < 1e-9
        assert abs(g2.a_px[1] - g.a_px[1]) < 1e-9

    def test_rotate_origin_unchanged(self):
        g = LatticeGrid.make_square(10, 20, 50)
        g2 = g.rotate(45)
        assert g2.origin_px == g.origin_px


# ── scale tests ───────────────────────────────────────────────────────────────

class TestScale:
    def test_scale_doubles_vectors(self):
        g = LatticeGrid.make_square(0, 0, 50)
        g2 = g.scale(2.0)
        assert abs(_len(g2.a_px) - 100) < 1e-9
        assert abs(_len(g2.b_px) - 100) < 1e-9

    def test_scale_halves_vectors(self):
        g = LatticeGrid.make_rectangular(0, 0, 60, 40)
        g2 = g.scale(0.5)
        assert abs(g2.a_px[0] - 30) < 1e-9
        assert abs(g2.b_px[1] - 20) < 1e-9

    def test_scale_preserves_angle(self):
        g = LatticeGrid.make_hexagonal(0, 0, 40)
        g2 = g.scale(3.0)
        assert abs(g2.angle_deg() - g.angle_deg()) < 1e-9

    def test_scale_does_not_move_origin(self):
        g = LatticeGrid.make_square(5, 7, 50)
        g2 = g.scale(2.0)
        assert g2.origin_px == g.origin_px


# ── constraint tests ──────────────────────────────────────────────────────────

class TestConstraints:
    def test_square_with_a_vector_enforces_ortho(self):
        g = LatticeGrid.make_square(0, 0, 50)
        new_a = (30, 10)
        g2 = g.with_a_vector(new_a)
        assert g2.kind == "square"
        # b must be perpendicular to new_a
        assert abs(_dot(g2.a_px, g2.b_px)) < 1e-9

    def test_square_with_a_vector_equal_lengths(self):
        g = LatticeGrid.make_square(0, 0, 50)
        g2 = g.with_a_vector((30, 10))
        assert abs(_len(g2.a_px) - _len(g2.b_px)) < 1e-9

    def test_rectangular_with_a_vector_preserves_orthogonality(self):
        g = LatticeGrid.make_rectangular(0, 0, 50, 30)
        g2 = g.with_a_vector((20, 10))
        assert abs(_dot(g2.a_px, g2.b_px)) < 1e-9

    def test_rectangular_with_a_vector_preserves_b_length(self):
        g = LatticeGrid.make_rectangular(0, 0, 50, 30)
        g2 = g.with_a_vector((20, 10))
        assert abs(_len(g2.b_px) - 30) < 1e-9

    def test_hexagonal_with_a_vector_preserves_60_deg(self):
        g = LatticeGrid.make_hexagonal(0, 0, 40)
        g2 = g.with_a_vector((25, 10))
        assert abs(g2.angle_deg() - 60.0) < 1e-6

    def test_hexagonal_with_a_vector_equal_lengths(self):
        g = LatticeGrid.make_hexagonal(0, 0, 40)
        g2 = g.with_a_vector((25, 10))
        assert abs(_len(g2.a_px) - _len(g2.b_px)) < 1e-9

    def test_square_with_b_vector_enforces_ortho(self):
        g = LatticeGrid.make_square(0, 0, 50)
        g2 = g.with_b_vector((10, 40))
        assert abs(_dot(g2.a_px, g2.b_px)) < 1e-9

    def test_rectangular_with_b_vector_preserves_a_length(self):
        g = LatticeGrid.make_rectangular(0, 0, 50, 30)
        g2 = g.with_b_vector((10, 40))
        assert abs(_len(g2.a_px) - 50) < 1e-9

    def test_hexagonal_with_b_vector_preserves_60_deg(self):
        g = LatticeGrid.make_hexagonal(0, 0, 40)
        g2 = g.with_b_vector((10, 35))
        assert abs(g2.angle_deg() - 60.0) < 1e-6


# ── numeric setter tests ──────────────────────────────────────────────────────

class TestNumericSetters:
    def test_set_a_length_px_square(self):
        g = LatticeGrid.make_square(0, 0, 50)
        g2 = g.set_a_length_px(80)
        assert abs(g2.a_length_px() - 80) < 1e-9
        assert abs(g2.b_length_px() - 80) < 1e-9  # square keeps equal lengths

    def test_set_a_length_px_rectangular(self):
        g = LatticeGrid.make_rectangular(0, 0, 50, 30)
        g2 = g.set_a_length_px(80)
        assert abs(g2.a_length_px() - 80) < 1e-9
        assert abs(g2.b_length_px() - 30) < 1e-9  # rectangular keeps b unchanged

    def test_set_a_length_px_hexagonal(self):
        g = LatticeGrid.make_hexagonal(0, 0, 40)
        g2 = g.set_a_length_px(60)
        assert abs(g2.a_length_px() - 60) < 1e-9
        assert abs(g2.b_length_px() - 60) < 1e-9  # hex keeps equal lengths
        assert abs(g2.angle_deg() - 60.0) < 1e-6

    def test_set_b_length_px_rectangular(self):
        g = LatticeGrid.make_rectangular(0, 0, 50, 30)
        g2 = g.set_b_length_px(70)
        assert abs(g2.b_length_px() - 70) < 1e-9
        assert abs(g2.a_length_px() - 50) < 1e-9  # a unchanged

    def test_set_b_length_px_square(self):
        g = LatticeGrid.make_square(0, 0, 50)
        g2 = g.set_b_length_px(80)  # scales both
        assert abs(g2.a_length_px() - 80) < 1e-9
        assert abs(g2.b_length_px() - 80) < 1e-9

    def test_set_rotation_deg(self):
        g = LatticeGrid.make_square(0, 0, 50)
        g2 = g.set_rotation_deg(45)
        assert abs(g2.a_angle_deg() - 45) < 1e-9

    def test_set_rotation_preserves_lengths(self):
        g = LatticeGrid.make_rectangular(0, 0, 60, 40)
        g2 = g.set_rotation_deg(30)
        assert abs(g2.a_length_px() - 60) < 1e-9
        assert abs(g2.b_length_px() - 40) < 1e-9

    def test_set_a_length_zero_returns_unchanged(self):
        g = LatticeGrid.make_square(0, 0, 50)
        g2 = g.set_a_length_px(0.0)
        assert g2 is g or g2.a_length_px() == g.a_length_px()

    def test_cells_param_does_not_alter_lattice(self):
        """The cells display count is independent of measured lattice constants."""
        g = LatticeGrid.make_hexagonal(0, 0, 40)
        # cells is a panel/item parameter, not on the model; verify model is unchanged
        assert abs(g.a_length_px() - 40) < 1e-9
        assert abs(g.angle_deg() - 60.0) < 1e-6


# ── measurement tests ─────────────────────────────────────────────────────────

class TestMeasurements:
    def test_a_length_px(self):
        g = LatticeGrid.make_square(0, 0, 50)
        assert abs(g.a_length_px() - 50) < 1e-9

    def test_b_length_px(self):
        g = LatticeGrid.make_square(0, 0, 50)
        assert abs(g.b_length_px() - 50) < 1e-9

    def test_angle_square(self):
        g = LatticeGrid.make_square(0, 0, 50)
        assert abs(g.angle_deg() - 90.0) < 1e-9

    def test_angle_hexagonal(self):
        g = LatticeGrid.make_hexagonal(0, 0, 40)
        assert abs(g.angle_deg() - 60.0) < 1e-6

    def test_area_square(self):
        g = LatticeGrid.make_square(0, 0, 50)
        assert abs(g.area_px2() - 2500.0) < 1e-9

    def test_area_rectangular(self):
        g = LatticeGrid.make_rectangular(0, 0, 30, 60)
        assert abs(g.area_px2() - 1800.0) < 1e-9

    def test_area_hexagonal(self):
        g = LatticeGrid.make_hexagonal(0, 0, 40)
        expected = 40 ** 2 * math.sin(math.radians(60))
        assert abs(g.area_px2() - expected) < 1e-6

    def test_a_angle_deg(self):
        g = LatticeGrid.make_hexagonal(0, 0, 40, angle_deg=30)
        assert abs(g.a_angle_deg() - 30) < 1e-9

    def test_reset_origin(self):
        g = LatticeGrid.make_square(0, 0, 50)
        g2 = g.reset_origin(100, 200)
        assert g2.origin_px == (100, 200)
        assert g2.a_px == g.a_px
        assert g2.b_px == g.b_px


# ── calibration tests ─────────────────────────────────────────────────────────

class TestRealSpaceCalibration:
    def setup_method(self):
        # 256 × 256 image, 10 nm scan range
        self.cal = RealSpaceCalibration.from_scan_range(
            scan_range_m=(10e-9, 10e-9), image_width=256, image_height=256
        )

    def test_pixel_size(self):
        expected = 10e-9 / 256
        assert abs(self.cal.px_size_x - expected) < 1e-25

    def test_vector_length_m_horizontal(self):
        # 256 pixels horizontal = full scan width = 10 nm
        cal = self.cal
        length = cal.vector_length_m((256, 0))
        assert abs(length - 10e-9) < 1e-20

    def test_vector_length_m_diagonal(self):
        cal = self.cal
        # (256, 256) → sqrt(10^2 + 10^2) nm
        length = cal.vector_length_m((256, 256))
        expected = math.hypot(10e-9, 10e-9)
        assert abs(length - expected) < 1e-20

    def test_origin_m(self):
        cal = self.cal
        ox_m, oy_m = cal.origin_m((128, 128))
        assert abs(ox_m - 5e-9) < 1e-22
        assert abs(oy_m - 5e-9) < 1e-22

    def test_from_scan_range_asymmetric(self):
        cal = RealSpaceCalibration.from_scan_range(
            scan_range_m=(20e-9, 10e-9), image_width=400, image_height=200
        )
        assert abs(cal.px_size_x - 20e-9 / 400) < 1e-25
        assert abs(cal.px_size_y - 10e-9 / 200) < 1e-25


class TestReciprocalCalibration:
    def setup_method(self):
        # 256 × 256 FFT, 10 nm image → q range ~ ±0.1 Å^-1 = ±1 nm^-1 (Nyquist)
        # Use fftfreq with d = 10/256 nm
        Nx, Ny = 256, 256
        d_nm = 10.0 / Nx
        qx = np.fft.fftshift(np.fft.fftfreq(Nx, d=d_nm))
        qy = np.fft.fftshift(np.fft.fftfreq(Ny, d=d_nm))
        self.cal = ReciprocalCalibration(
            qx_axis=qx, qy_axis=qy, image_width=Nx, image_height=Ny
        )
        self.qx = qx
        self.qy = qy

    def test_centre_px(self):
        assert self.cal.centre_px == (128.0, 128.0)

    def test_vec_px_to_q_zero(self):
        qvx, qvy = self.cal.vec_px_to_q((0.0, 0.0))
        assert abs(qvx) < 1e-12
        assert abs(qvy) < 1e-12

    def test_vec_length_q_nonzero(self):
        # A vector of 1 pixel in a 256-px FFT spanning ±12.8 nm^-1 range
        # dq = 25.6/255 nm^-1 per pixel ≈ 0.1004 nm^-1
        g = self.cal.vec_length_q((1, 0))
        assert g > 0

    def test_vec_length_q_symmetry(self):
        g1 = self.cal.vec_length_q((1, 0))
        g2 = self.cal.vec_length_q((0, 1))
        # Square FFT → symmetric axes
        assert abs(g1 - g2) < 1e-10

    def test_direct_lattice_from_square_reciprocal_grid(self):
        grid = LatticeGrid.make_square(128, 128, 10, space="reciprocal")

        a_nm, b_nm = direct_lattice_vectors_from_reciprocal_grid(grid, self.cal)

        assert a_nm == pytest.approx((1.0, 0.0))
        assert b_nm == pytest.approx((0.0, 1.0))

    def test_direct_lattice_from_hex_reciprocal_grid(self):
        grid = LatticeGrid.make_hexagonal(128, 128, 10, space="reciprocal")

        a_nm, b_nm = direct_lattice_vectors_from_reciprocal_grid(grid, self.cal)

        assert math.hypot(*a_nm) == pytest.approx(1.0 / math.sin(math.radians(60.0)))
        assert math.hypot(*b_nm) == pytest.approx(1.0 / math.sin(math.radians(60.0)))
        dot = a_nm[0] * b_nm[0] + a_nm[1] * b_nm[1]
        angle = math.degrees(math.acos(dot / (math.hypot(*a_nm) * math.hypot(*b_nm))))
        assert angle == pytest.approx(120.0)


# ── format_real_space_measurements ───────────────────────────────────────────

class TestFormatRealSpace:
    def setup_method(self):
        self.cal = RealSpaceCalibration.from_scan_range(
            scan_range_m=(10e-9, 10e-9), image_width=256, image_height=256
        )

    def test_keys_present(self):
        g = LatticeGrid.make_square(128, 128, 50)
        d = format_real_space_measurements(g, self.cal)
        for key in ("kind", "space", "origin_px", "origin_phys",
                    "a_px", "b_px", "a_length", "b_length", "angle", "area"):
            assert key in d, f"Missing key: {key}"

    def test_kind(self):
        g = LatticeGrid.make_hexagonal(0, 0, 40)
        d = format_real_space_measurements(g, self.cal)
        assert d["kind"] == "hexagonal"

    def test_angle_square(self):
        g = LatticeGrid.make_square(0, 0, 50)
        d = format_real_space_measurements(g, self.cal)
        assert "90" in d["angle"]

    def test_angle_hexagonal(self):
        g = LatticeGrid.make_hexagonal(0, 0, 50)
        d = format_real_space_measurements(g, self.cal)
        assert "60" in d["angle"]

    def test_angle_uses_non_square_pixel_calibration(self):
        cal = RealSpaceCalibration(
            px_size_x=1e-9,
            px_size_y=2e-9,
            image_width=100,
            image_height=100,
        )
        g = LatticeGrid(
            kind="rectangular",
            space="real",
            origin_px=(0, 0),
            a_px=(10.0, 0.0),
            b_px=(10.0, 10.0),
        )
        d = format_real_space_measurements(g, cal)
        assert d["angle"].startswith("63")

    def test_angle_format_does_not_use_scientific_notation(self):
        g = LatticeGrid(
            kind="rectangular",
            space="real",
            origin_px=(0, 0),
            a_px=(10.0, 0.0),
            b_px=(-5.0, 8.660254037844386),
        )
        d = format_real_space_measurements(g, self.cal)
        assert d["angle"] == "120°"

    def test_unit_in_output(self):
        g = LatticeGrid.make_square(0, 0, 50)
        d = format_real_space_measurements(g, self.cal)
        # Should have Å or nm in the physical length fields
        assert ("Å" in d["a_length"] or "nm" in d["a_length"])


class TestFormatReciprocal:
    def setup_method(self):
        Nx, Ny = 256, 256
        d_nm = 10.0 / Nx
        qx = np.fft.fftshift(np.fft.fftfreq(Nx, d=d_nm))
        qy = np.fft.fftshift(np.fft.fftfreq(Ny, d=d_nm))
        self.cal = ReciprocalCalibration(
            qx_axis=qx, qy_axis=qy, image_width=Nx, image_height=Ny
        )

    def test_keys_present(self):
        g = LatticeGrid.make_square(128, 128, 20, space="reciprocal")
        d = format_reciprocal_measurements(g, self.cal)
        for key in ("kind", "space", "origin_px", "origin_q",
                    "g1_vec", "g2_vec", "g1", "g2", "angle", "area_q",
                    "direct_a", "direct_b", "direct_angle"):
            assert key in d, f"Missing key: {key}"

    def test_g_values_contain_nm_inv(self):
        g = LatticeGrid.make_square(128, 128, 20, space="reciprocal")
        d = format_reciprocal_measurements(g, self.cal)
        assert "nm⁻¹" in d["g1"]
        assert "nm⁻¹" in d["g2"]

    def test_real_space_period_in_g1(self):
        g = LatticeGrid.make_square(128, 128, 20, space="reciprocal")
        d = format_reciprocal_measurements(g, self.cal)
        # Should show d = value in parentheses
        assert "plane d =" in d["g1"]

    def test_angle_square(self):
        g = LatticeGrid.make_square(128, 128, 20, space="reciprocal")
        d = format_reciprocal_measurements(g, self.cal)
        assert "90" in d["angle"]

    def test_angle_uses_reciprocal_axis_scaling(self):
        qx = np.linspace(-5.0, 5.0, 101)
        qy = np.linspace(-10.0, 10.0, 101)
        cal = ReciprocalCalibration(
            qx_axis=qx,
            qy_axis=qy,
            image_width=101,
            image_height=101,
        )
        g = LatticeGrid(
            kind="rectangular",
            space="reciprocal",
            origin_px=(50, 50),
            a_px=(10.0, 0.0),
            b_px=(10.0, 10.0),
        )
        d = format_reciprocal_measurements(g, cal)
        assert d["angle"].startswith("63")

    def test_angle_format_does_not_use_scientific_notation(self):
        g = LatticeGrid(
            kind="rectangular",
            space="reciprocal",
            origin_px=(50, 50),
            a_px=(10.0, 0.0),
            b_px=(-5.0, 8.660254037844386),
        )
        d = format_reciprocal_measurements(g, self.cal)
        assert d["angle"] == "120°"

    def test_direct_lattice_from_square_reciprocal_basis(self):
        qx = np.linspace(-1.0, 1.0, 101)
        qy = np.linspace(-1.0, 1.0, 101)
        cal = ReciprocalCalibration(
            qx_axis=qx,
            qy_axis=qy,
            image_width=101,
            image_height=101,
        )
        g = LatticeGrid(
            kind="square",
            space="reciprocal",
            origin_px=(50, 50),
            a_px=(25.0, 0.0),   # 0.5 nm^-1
            b_px=(0.0, 25.0),   # 0.5 nm^-1
        )
        d = format_reciprocal_measurements(g, cal)
        assert "2 nm" in d["direct_a"]
        assert "2 nm" in d["direct_b"]
        assert d["direct_angle"] == "direct angle = 90°"

    def test_direct_lattice_from_hex_reciprocal_basis(self):
        qx = np.linspace(-1.0, 1.0, 101)
        qy = np.linspace(-1.0, 1.0, 101)
        cal = ReciprocalCalibration(
            qx_axis=qx,
            qy_axis=qy,
            image_width=101,
            image_height=101,
        )
        size_px = 25.0  # 0.5 nm^-1
        g = LatticeGrid.make_hexagonal(50, 50, size_px, space="reciprocal")
        d = format_reciprocal_measurements(g, cal)
        assert "direct |a|" in d["direct_a"]
        assert "direct |b|" in d["direct_b"]
        assert d["direct_angle"] == "direct angle = 120°"


class _FakeGridItem:
    def __init__(self, grid):
        self._grid = grid
        self.cells = 12
        self._line_width_px = 1.5

    def grid(self):
        return self._grid

    def set_grid(self, grid):
        self._grid = grid

    def set_line_width(self, value):
        self._line_width_px = value

    def setVisible(self, _visible):
        pass


class _FakeController:
    def __init__(self):
        self.uninstalled = False
        self.active_values = []

    def set_active(self, _active):
        self.active_values.append(_active)

    def set_locked(self, _locked):
        pass

    def set_ab_equal(self, _checked):
        pass

    def uninstall(self):
        self.uninstalled = True


def test_ideal_lattice_presets_control_enabled_fields(qapp):
    from probeflow.gui.lattice_grid.real_space_panel import LatticeGridPanel

    grid = LatticeGrid.make_rectangular(50, 50, 20, 30)
    cal = RealSpaceCalibration.from_scan_range((10e-9, 10e-9), 100, 100)
    panel = LatticeGridPanel(_FakeGridItem(grid), _FakeController(), cal, 100, 100)
    try:
        assert panel._structure_combo.currentText() == "Hexagonal"
        assert panel._ideal_preset_combo.currentText() == "Hexagonal"
        assert "Undistort: y/x=" in panel._correction_lbl.text()

        panel._ideal_preset_combo.setCurrentText("Match grid")
        qapp.processEvents()
        assert panel._ideal_preset_combo.currentText() == "Match grid"
        assert not panel._ideal_a_spin.isEnabled()
        assert not panel._ideal_b_spin.isEnabled()
        assert not panel._ideal_angle_spin.isEnabled()

        panel._ideal_preset_combo.setCurrentText("Square")
        qapp.processEvents()
        assert panel._ideal_a_spin.isEnabled()
        assert not panel._ideal_b_spin.isEnabled()
        assert not panel._ideal_angle_spin.isEnabled()
        assert panel._ideal_b_spin.value() == pytest.approx(panel._ideal_a_spin.value())
        assert panel._ideal_angle_spin.value() == pytest.approx(90.0)

        panel._ideal_preset_combo.setCurrentText("Rectangular")
        qapp.processEvents()
        assert panel._ideal_a_spin.isEnabled()
        assert panel._ideal_b_spin.isEnabled()
        assert not panel._ideal_angle_spin.isEnabled()
        assert panel._ideal_angle_spin.value() == pytest.approx(90.0)

        panel._ideal_preset_combo.setCurrentText("Hexagonal")
        qapp.processEvents()
        assert panel._ideal_b_spin.value() == pytest.approx(panel._ideal_a_spin.value())
        assert panel._ideal_angle_spin.value() == pytest.approx(60.0)

        panel._ideal_preset_combo.setCurrentText("Custom")
        qapp.processEvents()
        assert panel._ideal_a_spin.isEnabled()
        assert panel._ideal_b_spin.isEnabled()
        assert panel._ideal_angle_spin.isEnabled()
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()


def test_real_space_known_structure_feeds_correction_target(qapp):
    from probeflow.gui.lattice_correction_ui import KnownStructure
    from probeflow.gui.lattice_grid.real_space_panel import LatticeGridPanel

    grid = LatticeGrid.make_square(50, 50, 20)
    cal = RealSpaceCalibration.from_scan_range((10e-9, 10e-9), 100, 100)
    panel = LatticeGridPanel(_FakeGridItem(grid), _FakeController(), cal, 100, 100)
    try:
        structure = KnownStructure("Square 1 nm", "square", 1.0, 1.0, 90.0, "nm")
        panel._known_structures = [structure]
        panel._refresh_structure_combo(structure.name)
        panel._on_structure_selected(0)
        qapp.processEvents()

        assert panel._active_known_structure == structure
        assert panel._ideal_preset_combo.currentText() == "Square"
        assert panel._ideal_a_spin.value() == pytest.approx(10.0)
        assert panel._ideal_b_spin.value() == pytest.approx(10.0)
        assert "Undistort: y/x=" in panel._correction_lbl.text()
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()


def test_lattice_grid_panel_cleanup_clears_preview_and_uninstalls_controller(qapp):
    from probeflow.gui.lattice_grid.real_space_panel import LatticeGridPanel

    grid = LatticeGrid.make_square(50, 50, 20)
    cal = RealSpaceCalibration.from_scan_range((10e-9, 10e-9), 100, 100)
    controller = _FakeController()
    cleared = []
    panel = LatticeGridPanel(
        _FakeGridItem(grid),
        controller,
        cal,
        100,
        100,
        clear_preview_fn=lambda: cleared.append(True),
    )
    try:
        panel._set_preview_state(True)

        panel.cleanup()

        assert cleared == [True]
        assert panel._preview_active is False
        assert controller.active_values[-1] is False
        assert controller.uninstalled is True
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()


def test_image_info_uses_display_array_without_numpy_truth_value(monkeypatch, qapp):
    from probeflow.gui.dialogs import image_info
    from probeflow.gui.viewer.image_viewer_tools_mixin import ImageViewerToolsMixin

    captured = {}

    class FakeImageInfoDialog:
        def __init__(self, *, current_shape=None, **_kwargs):
            captured["shape"] = current_shape

    monkeypatch.setattr(image_info, "ImageInfoDialog", FakeImageInfoDialog)

    class FakeViewer(ImageViewerToolsMixin):
        _track_modeless_child = lambda self, *a: None
        # Image info now opens via the modal-overlay helper rather than dlg.show().
        def _present_modal_tool(self, dlg, **_kw):
            captured["shown"] = True
            return None

    viewer = FakeViewer()
    viewer._entries = [SimpleNamespace(path=Path("/missing.sxm"))]
    viewer._idx = 0
    viewer._processing_history = None
    viewer._display_arr = np.ones((3, 4), dtype=float)
    viewer._raw_arr = np.zeros((1, 2), dtype=float)

    viewer._on_show_image_info()

    assert captured == {"shape": (3, 4), "shown": True}


# ── edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_length_with_a_returns_unchanged(self):
        g = LatticeGrid.make_square(0, 0, 50)
        g2 = g.with_a_vector((0.0, 0.0))
        # Should return unchanged
        assert g2.a_px == g.a_px

    def test_zero_length_with_b_returns_unchanged(self):
        g = LatticeGrid.make_square(0, 0, 50)
        g2 = g.with_b_vector((0.0, 0.0))
        assert g2.b_px == g.b_px

    def test_angle_zero_vectors_returns_zero(self):
        g = LatticeGrid(
            kind="square", space="real",
            origin_px=(0, 0), a_px=(0, 0), b_px=(0, 0),
        )
        assert g.angle_deg() == 0.0

    def test_area_degenerate_returns_zero(self):
        g = LatticeGrid(
            kind="square", space="real",
            origin_px=(0, 0), a_px=(10, 0), b_px=(10, 0),
        )
        assert g.area_px2() == 0.0

    def test_visible_default(self):
        g = LatticeGrid.make_square(0, 0, 50)
        assert g.visible is True

    def test_show_labels_default(self):
        g = LatticeGrid.make_square(0, 0, 50)
        assert g.show_labels is True

    def test_show_handles_default(self):
        g = LatticeGrid.make_square(0, 0, 50)
        assert g.show_handles is True


# ── _fmt_angle_deg tests ──────────────────────────────────────────────────────

class TestFmtAngleDeg:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.0, "0°"),
            (90.0, "90°"),          # trailing zeros stripped
            (90.1, "90.1°"),
            (123.456, "123.46°"),   # rounded to 2 dp
            (-45.0, "-45°"),
            (-0.004, "0°"),         # rounds to -0.00 → "-0" → caught → "0"
            (-0.5, "-0.5°"),        # real negative, not caught
            (1e-10, "0°"),          # tiny positive rounds to 0.00 → "0"
            (360.0, "360°"),
            (1.0, "1°"),
            (1.50, "1.5°"),         # one trailing zero stripped
        ],
    )
    def test_format(self, value, expected):
        assert _fmt_angle_deg(value) == expected


# ── LatticeGridDisplay tests ──────────────────────────────────────────────────

class TestLatticeGridDisplay:
    def test_defaults(self):
        d = LatticeGridDisplay()
        assert d.cells == 12
        assert d.line_width_px == 1.5
        assert d.basis_width_px == 2.0
        assert d.handle_radius_px == 7.0
        assert d.show_grid is True
        assert d.show_handles is True
        assert d.show_labels is True

    def test_custom_values(self):
        d = LatticeGridDisplay(cells=8, line_width_px=2.5, show_labels=False)
        assert d.cells == 8
        assert d.line_width_px == 2.5
        assert d.show_labels is False

    def test_independent_of_geometry(self):
        g = LatticeGrid.make_square(0, 0, 50)
        d = LatticeGridDisplay(line_width_px=3.0)
        assert g.angle_deg() == pytest.approx(90.0)
        assert d.line_width_px == 3.0


# ── export tests ──────────────────────────────────────────────────────────────

class TestExport:
    """Export tests require PySide6 but no display."""

    @pytest.mark.skipif(
        not os.environ.get("DISPLAY") and os.environ.get("CI") == "true",
        reason="No display available in headless CI",
    )
    def test_export_png_creates_file(self, tmp_path):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication
        import sys
        _ = QApplication.instance() or QApplication(sys.argv[:1])

        from PySide6.QtWidgets import QGraphicsScene
        from probeflow.gui.lattice_grid_tool import LatticeGridItem
        from probeflow.gui.lattice_export import export_grid

        scene = QGraphicsScene()
        grid = LatticeGrid.make_square(50, 50, 20)
        item = LatticeGridItem(grid, 100, 100)
        scene.addItem(item)

        out = str(tmp_path / "test.png")
        export_grid(item, out, include_grid=True, grid_only=True)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 100
