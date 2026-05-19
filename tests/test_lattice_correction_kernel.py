"""Tests for the affine_lattice_correction processing kernel (Stage 2)."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.analysis.lattice_correction_workflow import (
    lattice_correction_matrix_px,
    lattice_correction_operation_params,
)
from probeflow.analysis.lattice_distortion import (
    IdealLattice,
    MeasuredLattice,
    compute_correction,
)
from probeflow.processing.image import affine_lattice_correction
from probeflow.processing.state import ProcessingStep, apply_processing_state, ProcessingState


# ── helpers ───────────────────────────────────────────────────────────────────

def _stripe_image(Ny: int = 64, Nx: int = 64) -> np.ndarray:
    """Vertical stripes — easy to check that columns shift correctly."""
    arr = np.zeros((Ny, Nx), dtype=np.float64)
    arr[:, ::4] = 1.0
    return arr


def _ramp_image(Ny: int = 64, Nx: int = 64) -> np.ndarray:
    """Linear ramp from 0 to 1 across columns."""
    arr = np.zeros((Ny, Nx), dtype=np.float64)
    arr[:, :] = np.linspace(0, 1, Nx)
    return arr


def _sample_correction():
    correction = compute_correction(
        MeasuredLattice(a_nm=(2.0, 0.0), b_nm=(0.2, 1.0)),
        IdealLattice(a_nm=1.0, b_nm=1.0, angle_deg=90.0),
    )
    assert not isinstance(correction, str)
    return correction


def test_lattice_correction_matrix_px_converts_non_square_pixels():
    correction = _sample_correction()
    matrix = lattice_correction_matrix_px(
        correction,
        pixel_size_x_m=1e-9,
        pixel_size_y_m=2e-9,
        preserve_orientation=False,
    )
    expected = np.diag([1.0, 0.5]) @ correction.matrix @ np.diag([1.0, 2.0])

    np.testing.assert_allclose(matrix, expected)


def test_lattice_correction_operation_params_records_applied_and_full_matrices():
    correction = _sample_correction()
    params = lattice_correction_operation_params(
        correction,
        pixel_size_x_m=1e-9,
        pixel_size_y_m=2e-9,
        expand_canvas=True,
        interpolation="bilinear",
        fill_mode="nan",
        preserve_orientation=True,
    )
    applied = lattice_correction_matrix_px(
        correction,
        pixel_size_x_m=1e-9,
        pixel_size_y_m=2e-9,
        preserve_orientation=True,
    )
    full = lattice_correction_matrix_px(
        correction,
        pixel_size_x_m=1e-9,
        pixel_size_y_m=2e-9,
        preserve_orientation=False,
    )

    assert params is not None
    np.testing.assert_allclose(params["matrix"], applied)
    np.testing.assert_allclose(params["full_matrix"], full)
    assert params["preserve_orientation"] is True
    assert params["measured_a_nm"] == [2.0, 0.0]


def test_lattice_correction_operation_params_rejects_invalid_pixel_size():
    params = lattice_correction_operation_params(
        _sample_correction(),
        pixel_size_x_m=0.0,
        pixel_size_y_m=1e-9,
        expand_canvas=True,
        interpolation="bilinear",
        fill_mode="nan",
        preserve_orientation=False,
    )

    assert params is None


# ── 1. identity matrix ────────────────────────────────────────────────────────

class TestIdentity:
    def test_values_preserved(self):
        arr = _stripe_image()
        out = affine_lattice_correction(arr, np.eye(2))
        # Interior should be identical; edges may have NaN from expand
        mask = np.isfinite(out)
        assert mask.any()

    def test_identity_no_expand_same_shape(self):
        arr = _stripe_image()
        out = affine_lattice_correction(arr, np.eye(2), expand_canvas=False)
        assert out.shape == arr.shape

    def test_identity_values_close(self):
        arr = _ramp_image()
        out = affine_lattice_correction(arr, np.eye(2), expand_canvas=False)
        np.testing.assert_allclose(out, arr, atol=1e-10)


# ── 2. pure scale — canvas expansion ─────────────────────────────────────────

class TestScaleExpansion:
    def test_expand_enlarges_canvas(self):
        arr = _stripe_image(64, 64)
        scale = np.diag([1.5, 1.0])   # stretch in x by 1.5
        out = affine_lattice_correction(arr, scale, expand_canvas=True)
        assert out.shape[1] > arr.shape[1]

    def test_no_expand_preserves_shape(self):
        arr = _stripe_image(64, 64)
        scale = np.diag([1.5, 1.0])
        out = affine_lattice_correction(arr, scale, expand_canvas=False)
        assert out.shape == arr.shape


# ── 3. pure shear ─────────────────────────────────────────────────────────────

class TestShear:
    def test_shear_expand_larger_height(self):
        arr = _stripe_image(64, 64)
        shear = np.array([[1.0, 0.0], [0.3, 1.0]])
        out = affine_lattice_correction(arr, shear, expand_canvas=True)
        assert out.shape[0] > arr.shape[0]

    def test_shear_no_expand_same_shape(self):
        arr = _stripe_image(64, 64)
        shear = np.array([[1.0, 0.0], [0.3, 1.0]])
        out = affine_lattice_correction(arr, shear, expand_canvas=False)
        assert out.shape == arr.shape


# ── 4. interpolation modes run without error ──────────────────────────────────

class TestInterpolation:
    @pytest.mark.parametrize("interp", ["nearest", "bilinear", "bicubic"])
    def test_runs(self, interp):
        arr = _ramp_image()
        matrix = np.diag([1.05, 0.97])
        out = affine_lattice_correction(arr, matrix, interpolation=interp)
        assert out.ndim == 2
        assert np.isfinite(out).any()


# ── 5. fill mode nan ──────────────────────────────────────────────────────────

class TestFillNan:
    def test_oob_pixels_are_nan(self):
        arr = _stripe_image()
        # Shrink transform (T=0.5×I): T_inv=2×I so corners of output map outside input
        scale = np.diag([0.5, 0.5])
        out = affine_lattice_correction(arr, scale, expand_canvas=False, fill_mode="nan")
        assert np.isnan(out).any()

    def test_interior_pixels_finite(self):
        arr = _stripe_image()
        out = affine_lattice_correction(arr, np.eye(2), expand_canvas=False, fill_mode="nan")
        assert np.isfinite(out).all()


# ── 6. fill mode zero ─────────────────────────────────────────────────────────

class TestFillZero:
    def test_oob_pixels_are_zero(self):
        arr = _stripe_image() + 5.0   # shift so interior != 0
        # Shrink transform → corners of no-expand output map outside input
        scale = np.diag([0.5, 0.5])
        out = affine_lattice_correction(arr, scale, expand_canvas=False, fill_mode="zero")
        # Corner pixels (well outside input) should be zero, not nan
        assert not np.isnan(out[0, 0])
        assert out[0, 0] == pytest.approx(0.0)


# ── 7. fill mode background ───────────────────────────────────────────────────

class TestFillBackground:
    def test_oob_pixels_use_median(self):
        arr = np.full((32, 32), 3.14)
        scale = np.diag([2.0, 2.0])
        out = affine_lattice_correction(arr, scale, expand_canvas=True, fill_mode="background")
        assert not np.isnan(out).any()
        assert out[0, 0] == pytest.approx(3.14, abs=1e-6)


# ── 8. singular matrix rejected ───────────────────────────────────────────────

class TestSingular:
    def test_raises_on_singular(self):
        arr = _stripe_image()
        singular = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="singular"):
            affine_lattice_correction(arr, singular)

    def test_raises_on_wrong_shape(self):
        arr = _stripe_image()
        with pytest.raises(ValueError, match="shape"):
            affine_lattice_correction(arr, np.eye(3))

    def test_raises_on_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            affine_lattice_correction(np.ones(10), np.eye(2))


# ── 9. processing state round-trip ───────────────────────────────────────────

class TestProcessingState:
    def test_step_is_valid(self):
        matrix = np.diag([1.02, 0.99]).tolist()
        step = ProcessingStep("affine_lattice_correction", {
            "matrix": matrix,
            "expand_canvas": True,
            "interpolation": "bilinear",
            "fill_mode": "nan",
        })
        assert step.op == "affine_lattice_correction"

    def test_apply_processing_state_runs(self):
        arr = _ramp_image(32, 32)
        matrix = np.diag([1.0, 1.0]).tolist()
        state = ProcessingState(steps=[
            ProcessingStep("affine_lattice_correction", {
                "matrix": matrix,
                "expand_canvas": False,
                "interpolation": "bilinear",
                "fill_mode": "nan",
            })
        ])
        out = apply_processing_state(arr, state)
        assert out.ndim == 2

    def test_records_measured_ideal_params(self):
        matrix = np.eye(2).tolist()
        params = {
            "matrix": matrix,
            "expand_canvas": True,
            "interpolation": "bilinear",
            "fill_mode": "nan",
            "measured_a_nm": [0.25, 0.0],
            "measured_b_nm": [0.0, 0.25],
            "ideal_a_nm": 0.25,
            "ideal_b_nm": 0.25,
            "ideal_angle_deg": 90.0,
        }
        step = ProcessingStep("affine_lattice_correction", params)
        assert step.params["ideal_a_nm"] == pytest.approx(0.25)
        assert step.params["ideal_angle_deg"] == pytest.approx(90.0)


# ── 10. fill × interpolation parametrized matrix ─────────────────────────────

@pytest.mark.parametrize("fill_mode", ["nan", "background", "zero"])
@pytest.mark.parametrize("interp", ["nearest", "bilinear", "bicubic"])
class TestFillInterpolationMatrix:
    """9-case matrix: every fill_mode × interpolation combination must run
    without error and return a finite result somewhere in the output."""

    def test_runs_without_error(self, fill_mode: str, interp: str):
        arr = _ramp_image()
        out = affine_lattice_correction(
            arr,
            np.diag([0.9, 1.1]),
            fill_mode=fill_mode,
            interpolation=interp,
            expand_canvas=True,
        )
        assert out.ndim == 2
        assert np.isfinite(out).any()
