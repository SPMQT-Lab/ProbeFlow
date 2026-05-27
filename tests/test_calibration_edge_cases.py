"""Tests for calibration edge cases, non-square pixel conversion,
and polar decomposition robustness."""

from __future__ import annotations

import math

import numpy as np
import pytest

from probeflow.analysis.lattice_grid import RealSpaceCalibration
from probeflow.analysis.lattice_distortion import _polar_decompose
from probeflow.processing.image import affine_lattice_correction


# ── RealSpaceCalibration edge cases ───────────────────────────────────────────

class TestRealSpaceCalibrationEdge:
    """Document current behaviour for zero/negative pixel sizes.

    These tests record what the code does today.  If validation is tightened to
    raise on zero/negative input, update the assertions and the docstring.
    """

    def test_zero_scan_range_gives_zero_px_size(self):
        cal = RealSpaceCalibration.from_scan_range((0.0, 0.0), 256, 256)
        assert cal.px_size_x == 0.0
        assert cal.px_size_y == 0.0

    def test_zero_scan_range_does_not_raise(self):
        # Current behaviour: no error on zero range
        RealSpaceCalibration.from_scan_range((0.0, 0.0), 256, 256)

    def test_negative_scan_range_gives_negative_px_size(self):
        # from_scan_range divides range / width; negative range → negative px_size
        cal = RealSpaceCalibration.from_scan_range((-1e-9, 1e-9), 256, 256)
        assert cal.px_size_x < 0.0
        assert cal.px_size_y > 0.0

    def test_vector_length_m_with_zero_px_size_returns_zero(self):
        cal = RealSpaceCalibration(px_size_x=0.0, px_size_y=0.0,
                                   image_width=64, image_height=64)
        length = cal.vector_length_m((1.0, 1.0))
        assert length == 0.0

    def test_origin_m_with_negative_px_size_returns_negative_coords(self):
        cal = RealSpaceCalibration(px_size_x=-1e-10, px_size_y=1e-10,
                                   image_width=64, image_height=64)
        ox, oy = cal.origin_m((10.0, 5.0))
        assert ox < 0.0   # negative px_size_x → negative physical x
        assert oy > 0.0

    def test_normal_calibration_roundtrip(self):
        # Verify correct behaviour for standard (positive) pixel sizes
        scan_range_m = (20e-9, 20e-9)
        cal = RealSpaceCalibration.from_scan_range(scan_range_m, 200, 200)
        assert abs(cal.px_size_x - 1e-10) < 1e-14
        assert abs(cal.px_size_y - 1e-10) < 1e-14

    def test_anisotropic_normal_calibration(self):
        # x range ≠ y range with different pixel counts
        cal = RealSpaceCalibration.from_scan_range((10e-9, 20e-9), 100, 200)
        assert abs(cal.px_size_x - 1e-10) < 1e-14
        assert abs(cal.px_size_y - 1e-10) < 1e-14


# ── nm→pixel conversion for non-square pixels ─────────────────────────────────

class TestAffineWithNonSquarePixels:
    """The nm→pixel conversion T_px = S @ T_nm @ S_inv is non-trivial when
    px_size_x ≠ px_size_y.  For an off-diagonal T_nm, T_px ≠ T_nm.
    """

    def _convert(self, T_nm: np.ndarray, px_nm_x: float, px_nm_y: float) -> np.ndarray:
        """Apply the same conversion that _correction_matrix_px uses."""
        S = np.diag([1.0 / px_nm_x, 1.0 / px_nm_y])
        S_inv = np.diag([px_nm_x, px_nm_y])
        return S @ T_nm @ S_inv

    def test_identity_nm_gives_identity_px_regardless_of_aspect(self):
        # S @ I @ S_inv = S @ S_inv = I
        T_nm = np.eye(2)
        T_px = self._convert(T_nm, px_nm_x=0.25, px_nm_y=0.125)
        np.testing.assert_allclose(T_px, np.eye(2), atol=1e-12)

    def test_shear_nm_gives_different_shear_px_for_anisotropic_pixels(self):
        # Shear in nm space:  T_nm = [[1, s], [0, 1]]
        # After conversion with 2:1 pixel aspect (px_nm_x = 2 * px_nm_y):
        #   T_px[0,1] = s * px_nm_y / px_nm_x = s / 2
        px_nm_x, px_nm_y = 0.25, 0.125
        s = 0.5
        T_nm = np.array([[1.0, s], [0.0, 1.0]])
        T_px = self._convert(T_nm, px_nm_x, px_nm_y)
        expected_shear_px = s * px_nm_y / px_nm_x
        assert abs(T_px[0, 1] - expected_shear_px) < 1e-12

    def test_shear_nm_unchanged_for_square_pixels(self):
        # Square pixels: S = s*I so T_px = s*I @ T_nm @ (1/s)*I = T_nm
        T_nm = np.array([[1.0, 0.5], [0.0, 1.0]])
        T_px = self._convert(T_nm, px_nm_x=0.25, px_nm_y=0.25)
        np.testing.assert_allclose(T_px, T_nm, atol=1e-12)

    def test_anisotropic_pixel_shear_differs_from_square_pixel_shear(self):
        T_nm = np.array([[1.0, 0.6], [0.0, 1.0]])
        T_px_square = self._convert(T_nm, 0.25, 0.25)
        T_px_aniso = self._convert(T_nm, 0.25, 0.125)
        # Off-diagonal element should differ
        assert abs(T_px_square[0, 1] - T_px_aniso[0, 1]) > 0.1

    def test_identity_round_trip_with_anisotropic_pixels(self):
        # T_px = I → affine_lattice_correction with identity → output ≈ input
        arr = np.random.RandomState(42).randn(16, 16)
        out = affine_lattice_correction(arr, np.eye(2), expand_canvas=False)
        np.testing.assert_allclose(out, arr, atol=1e-10)

    def test_anisotropic_shear_correction_runs_without_error(self):
        # Use the anisotropic-pixel-converted matrix in actual correction
        T_nm = np.array([[1.05, 0.0], [0.0, 0.97]])
        T_px = self._convert(T_nm, px_nm_x=0.20, px_nm_y=0.10)
        arr = np.ones((16, 16), dtype=np.float64)
        out = affine_lattice_correction(arr, T_px, expand_canvas=False)
        assert out.ndim == 2


# ── polar decomposition edge cases ───────────────────────────────────────────

class TestPolarDecompEdgeCases:
    def test_pure_rotation_gives_identity_stretch(self):
        theta = math.radians(37.0)
        T = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
        R, S, rot_deg = _polar_decompose(T)
        np.testing.assert_allclose(S, np.eye(2), atol=1e-10)

    def test_pure_rotation_angle_matches_input(self):
        theta_deg = 37.0
        theta = math.radians(theta_deg)
        T = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
        R, S, rot_deg = _polar_decompose(T)
        assert abs(rot_deg - theta_deg) < 1e-8

    def test_near_zero_singular_value_no_nan_in_stretch(self):
        # Matrix with sigma_min ≈ 1e-8 (ill-conditioned but well-defined)
        sigma_max, sigma_min = 1.0, 1e-8
        U = np.array([[math.cos(0.3), -math.sin(0.3)],
                      [math.sin(0.3),  math.cos(0.3)]])
        Vt = np.array([[math.cos(0.7), -math.sin(0.7)],
                       [math.sin(0.7),  math.cos(0.7)]])
        T = U @ np.diag([sigma_max, sigma_min]) @ Vt
        R, S, rot_deg = _polar_decompose(T)
        assert np.all(np.isfinite(S))
        assert np.all(np.isfinite(R))

    def test_reflective_T_yields_psd_stretch_and_improper_R(self):
        # T = [[-1, 0], [0, 1]] has det = -1 (pure reflection across the
        # y-axis).  Polar decomposition T = R @ S must have S positive
        # semi-definite; R then has det(R) = -1 (improper orthogonal).
        # This is the corrected behaviour — previously the function
        # silently produced an indefinite S to force det(R) = +1.
        T = np.array([[-1.0, 0.0], [0.0, 1.0]])
        with pytest.warns(UserWarning, match="reflection component"):
            R, S, rot_deg = _polar_decompose(T)
        # S must be PSD (eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -1e-12), f"S not PSD; eigvals={eigvals}"
        # R is orthogonal but improper (det = -1)
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-10)
        assert abs(float(np.linalg.det(R)) + 1.0) < 1e-10
        # rotation_deg is undefined for an improper R
        assert math.isnan(rot_deg)
        # Reconstruction still holds: T = R @ S
        np.testing.assert_allclose(R @ S, T, atol=1e-10)

    def test_reflective_T_emits_warning(self):
        T = np.array([[-1.0, 0.0], [0.0, 1.0]])
        with pytest.warns(UserWarning, match="reflection component"):
            _polar_decompose(T)

    def test_stretch_is_symmetric(self):
        T = np.array([[1.2, 0.3], [0.1, 0.9]])
        R, S, rot_deg = _polar_decompose(T)
        np.testing.assert_allclose(S, S.T, atol=1e-10)

    def test_decomposition_reconstructs_matrix(self):
        T = np.array([[1.2, 0.3], [0.1, 0.9]])
        R, S, rot_deg = _polar_decompose(T)
        np.testing.assert_allclose(R @ S, T, atol=1e-10)

    def test_identity_gives_zero_rotation(self):
        R, S, rot_deg = _polar_decompose(np.eye(2))
        assert abs(rot_deg) < 1e-10
        np.testing.assert_allclose(S, np.eye(2), atol=1e-10)
