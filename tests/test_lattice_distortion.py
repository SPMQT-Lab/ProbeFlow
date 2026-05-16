"""Tests for lattice distortion analysis (Stage 1: affine matrix calculation)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from probeflow.analysis.lattice_distortion import (
    IdealLattice,
    LatticeCorrection,
    MeasuredLattice,
    compute_correction,
    ideal_vectors_nm,
)


# ── ideal_vectors_nm ──────────────────────────────────────────────────────────

class TestIdealVectors:
    def test_a_along_x(self):
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        i_a, _ = ideal_vectors_nm(ideal)
        assert abs(i_a[0] - 0.25) < 1e-12
        assert abs(i_a[1]) < 1e-12

    def test_b_at_90_deg(self):
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        _, i_b = ideal_vectors_nm(ideal)
        assert abs(i_b[0]) < 1e-12
        assert abs(i_b[1] - 0.25) < 1e-12

    def test_b_at_60_deg(self):
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=60.0)
        i_a, i_b = ideal_vectors_nm(ideal)
        la = math.hypot(*i_a)
        lb = math.hypot(*i_b)
        cos_theta = (i_a[0] * i_b[0] + i_a[1] * i_b[1]) / (la * lb)
        assert abs(cos_theta - math.cos(math.radians(60.0))) < 1e-12

    def test_b_length_preserved(self):
        ideal = IdealLattice(a_nm=0.30, b_nm=0.45, angle_deg=73.5)
        _, i_b = ideal_vectors_nm(ideal)
        assert abs(math.hypot(*i_b) - 0.45) < 1e-12

    def test_angle_45(self):
        ideal = IdealLattice(a_nm=1.0, b_nm=1.0, angle_deg=45.0)
        _, i_b = ideal_vectors_nm(ideal)
        assert abs(i_b[0] - math.cos(math.radians(45))) < 1e-12
        assert abs(i_b[1] - math.sin(math.radians(45))) < 1e-12


# ── compute_correction — identity cases ───────────────────────────────────────

class TestComputeCorrectionIdentity:
    def _square_90(self):
        """Measured and ideal both square 0.25 nm × 0.25 nm, 90°."""
        measured = MeasuredLattice(
            a_nm=(0.25, 0.0),
            b_nm=(0.0, 0.25),
        )
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        return measured, ideal

    def test_identity_matrix(self):
        measured, ideal = self._square_90()
        result = compute_correction(measured, ideal)
        assert isinstance(result, LatticeCorrection)
        assert np.allclose(result.matrix, np.eye(2), atol=1e-10)

    def test_identity_x_scale(self):
        measured, ideal = self._square_90()
        result = compute_correction(measured, ideal)
        assert abs(result.x_scale - 1.0) < 1e-10

    def test_identity_y_over_x(self):
        measured, ideal = self._square_90()
        result = compute_correction(measured, ideal)
        assert abs(result.y_over_x - 1.0) < 1e-10

    def test_identity_shear(self):
        measured, ideal = self._square_90()
        result = compute_correction(measured, ideal)
        assert abs(result.shear) < 1e-10

    def test_identity_rotation(self):
        measured, ideal = self._square_90()
        result = compute_correction(measured, ideal)
        assert abs(result.rotation_deg) < 1e-9

    def test_hexagonal_self_correction_is_identity(self):
        angle = math.radians(60)
        measured = MeasuredLattice(
            a_nm=(0.25, 0.0),
            b_nm=(0.25 * math.cos(angle), 0.25 * math.sin(angle)),
        )
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=60.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, LatticeCorrection)
        assert np.allclose(result.matrix, np.eye(2), atol=1e-10)


# ── scale distortions ─────────────────────────────────────────────────────────

class TestScaleDistortion:
    def test_uniform_x_scale(self):
        """Measured a-spacing 10% smaller than ideal → x_scale ≠ 1."""
        measured = MeasuredLattice(
            a_nm=(0.225, 0.0),
            b_nm=(0.0, 0.25),
        )
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, LatticeCorrection)
        assert abs(result.x_scale - 0.25 / 0.225) < 1e-10

    def test_uniform_scale_y_over_x_close_to_one(self):
        """Both spacings uniformly compressed → correction has y/x ≈ 1."""
        measured = MeasuredLattice(
            a_nm=(0.20, 0.0),
            b_nm=(0.0, 0.20),   # same factor as a
        )
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, LatticeCorrection)
        assert abs(result.y_over_x - 1.0) < 1e-10

    def test_anisotropic_y_over_x(self):
        """Y spacing measured 10% smaller than ideal → y/x ≠ 1."""
        measured = MeasuredLattice(
            a_nm=(0.25, 0.0),
            b_nm=(0.0, 0.225),
        )
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, LatticeCorrection)
        expected_y_over_x = 0.25 / 0.225
        assert abs(result.y_over_x - expected_y_over_x) < 1e-10

    def test_anisotropic_no_shear(self):
        """Anisotropic rectangular distortion has no shear."""
        measured = MeasuredLattice(
            a_nm=(0.25, 0.0),
            b_nm=(0.0, 0.22),
        )
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, LatticeCorrection)
        assert abs(result.shear) < 1e-10


# ── shear distortion ──────────────────────────────────────────────────────────

class TestShearDistortion:
    def test_simple_shear_nonzero(self):
        """Measured b-vector tilted slightly off +y → nonzero shear in correction."""
        measured = MeasuredLattice(
            a_nm=(0.25, 0.0),
            b_nm=(0.025, 0.25),   # 10% shear in x
        )
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, LatticeCorrection)
        assert abs(result.shear) > 1e-3

    def test_no_shear_for_orthogonal_measured(self):
        measured = MeasuredLattice(
            a_nm=(0.25, 0.0),
            b_nm=(0.0, 0.30),
        )
        ideal = IdealLattice(a_nm=0.25, b_nm=0.30, angle_deg=90.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, LatticeCorrection)
        assert abs(result.shear) < 1e-10


# ── singular / degenerate cases ───────────────────────────────────────────────

class TestSingular:
    def test_parallel_vectors_returns_error_string(self):
        measured = MeasuredLattice(
            a_nm=(0.25, 0.0),
            b_nm=(0.25, 0.0),   # collinear
        )
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, str)

    def test_zero_a_returns_error_string(self):
        measured = MeasuredLattice(
            a_nm=(0.0, 0.0),
            b_nm=(0.0, 0.25),
        )
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, str)

    def test_error_message_is_informative(self):
        measured = MeasuredLattice(
            a_nm=(0.25, 0.0),
            b_nm=(0.50, 0.0),
        )
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, str)
        assert "collinear" in result.lower() or "singular" in result.lower()


# ── matrix properties ─────────────────────────────────────────────────────────

class TestMatrixProperties:
    def test_t_maps_measured_a_to_ideal_a(self):
        measured = MeasuredLattice(
            a_nm=(0.20, 0.05),
            b_nm=(-0.02, 0.28),
        )
        ideal = IdealLattice(a_nm=0.25, b_nm=0.28, angle_deg=85.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, LatticeCorrection)
        i_a, i_b = ideal_vectors_nm(ideal)
        mapped_a = result.matrix @ np.array(measured.a_nm)
        assert np.allclose(mapped_a, i_a, atol=1e-10)

    def test_t_maps_measured_b_to_ideal_b(self):
        measured = MeasuredLattice(
            a_nm=(0.20, 0.05),
            b_nm=(-0.02, 0.28),
        )
        ideal = IdealLattice(a_nm=0.25, b_nm=0.28, angle_deg=85.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, LatticeCorrection)
        _, i_b = ideal_vectors_nm(ideal)
        mapped_b = result.matrix @ np.array(measured.b_nm)
        assert np.allclose(mapped_b, i_b, atol=1e-10)

    def test_matrix_shape(self):
        measured = MeasuredLattice(a_nm=(0.25, 0.0), b_nm=(0.0, 0.25))
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, LatticeCorrection)
        assert result.matrix.shape == (2, 2)


# ── polar decomposition ───────────────────────────────────────────────────────

class TestPolarDecomposition:
    def _generic_correction(self):
        measured = MeasuredLattice(
            a_nm=(0.20, 0.05),
            b_nm=(-0.02, 0.28),
        )
        ideal = IdealLattice(a_nm=0.25, b_nm=0.28, angle_deg=85.0)
        return compute_correction(measured, ideal)

    def test_polar_rotation_is_orthogonal(self):
        result = self._generic_correction()
        R = result.rotation_matrix
        assert np.allclose(R @ R.T, np.eye(2), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_stretch_is_symmetric(self):
        result = self._generic_correction()
        S = result.stretch_matrix
        assert np.allclose(S, S.T, atol=1e-10)

    def test_polar_decomp_reconstructs_full_matrix(self):
        result = self._generic_correction()
        reconstructed = result.rotation_matrix @ result.stretch_matrix
        assert np.allclose(reconstructed, result.matrix, atol=1e-10)

    def test_identity_has_zero_polar_rotation(self):
        measured = MeasuredLattice(a_nm=(0.25, 0.0), b_nm=(0.0, 0.25))
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, LatticeCorrection)
        assert abs(result.polar_rotation_deg) < 1e-9

    def test_pure_scale_has_zero_polar_rotation(self):
        """Pure anisotropic scale has no rotation component."""
        measured = MeasuredLattice(a_nm=(0.20, 0.0), b_nm=(0.0, 0.22))
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, LatticeCorrection)
        assert abs(result.polar_rotation_deg) < 1e-9

    def test_stretch_preserves_scale_correction(self):
        """Applying stretch_matrix to measured vectors should give ideal vectors
        modulo the rigid rotation — i.e. the corrected lengths should match."""
        measured = MeasuredLattice(a_nm=(0.20, 0.0), b_nm=(0.0, 0.22))
        ideal = IdealLattice(a_nm=0.25, b_nm=0.25, angle_deg=90.0)
        result = compute_correction(measured, ideal)
        assert isinstance(result, LatticeCorrection)
        # stretch_matrix @ measured_a should have the same length as ideal a
        stretched_a = result.stretch_matrix @ np.array(measured.a_nm)
        assert abs(math.hypot(*stretched_a) - ideal.a_nm) < 1e-10
