"""Tests for probeflow.processing.symmetrize — n-fold symmetrization backend."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing.symmetrize import (
    SymmetrizeResult,
    fold_axis_from_peaks,
    symmetrize_filter,
    symmetrize_image,
)

N = 128
YY, XX = np.mgrid[:N, :N].astype(np.float64)
# scipy.ndimage.rotate pivots about the (N−1)/2 point; build patterns there so
# "centred" tests exercise the register=False path honestly.
C = (N - 1) / 2.0


def _hex_lattice(center=(C, C), angle0_deg=0.0, period_px=14.0):
    """Sum of three cosines with a sixfold reciprocal set (smooth, band-limited)."""
    y0, x0 = center
    q = 2.0 * np.pi / period_px
    out = np.zeros((N, N))
    for k in range(3):
        th = np.radians(angle0_deg) + k * np.pi / 3.0
        out += np.cos(q * ((XX - x0) * np.cos(th) + (YY - y0) * np.sin(th)))
    return out


def _square_lattice(period_px=16.0):
    return (np.cos(2.0 * np.pi * (XX - C) / period_px)
            + np.cos(2.0 * np.pi * (YY - C) / period_px))


HEX = _hex_lattice()


# ─── identity and validation ────────────────────────────────────────────────

class TestBasics:
    def test_identity_returns_copy(self):
        res = symmetrize_image(HEX, 1)
        np.testing.assert_array_equal(res.result, HEX)
        assert res.n_ops == 1
        assert res.symmetry_residual_norm == 0.0
        assert not np.shares_memory(res.result, HEX)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            symmetrize_image(np.zeros(16), 4)

    def test_rejects_bad_fold(self):
        with pytest.raises(ValueError, match="n_fold"):
            symmetrize_image(HEX, 0)

    def test_rejects_all_nan(self):
        with pytest.raises(ValueError, match="finite"):
            symmetrize_image(np.full((8, 8), np.nan), 4)

    def test_rejects_bad_interpolation(self):
        with pytest.raises(ValueError, match="interpolation"):
            symmetrize_image(HEX, 4, interpolation="nearest")

    def test_filter_matches_result(self):
        np.testing.assert_array_equal(
            symmetrize_filter(HEX, 6), symmetrize_image(HEX, 6).result)

    def test_result_type(self):
        res = symmetrize_image(HEX, 6)
        assert isinstance(res, SymmetrizeResult)
        assert res.result.shape == HEX.shape
        assert res.symmetrized_fft.shape == HEX.shape
        assert res.shifts.shape == (6, 2)
        assert res.n_ops == 6


# ─── symmetry preservation ──────────────────────────────────────────────────

class TestPreservation:
    def test_c4_on_square_lattice_is_near_exact(self):
        # 90° rotations need no interpolation — the average should be exact.
        res = symmetrize_image(_square_lattice(), 4)
        assert res.symmetry_residual_norm < 1e-10

    def test_c6_on_centred_hex_lattice(self):
        res = symmetrize_image(HEX, 6, register=False)
        assert res.symmetry_residual_norm < 0.05   # interpolation loss only

    def test_no_registration_needed_when_centred(self):
        res = symmetrize_image(HEX, 6, register=True)
        assert np.abs(res.shifts).max() < 0.6

    def test_odd_sized_image(self):
        a = HEX[:127, :127]
        res = symmetrize_image(a, 6)
        assert res.result.shape == a.shape
        assert res.symmetry_residual_norm < 0.06


# ─── registration ───────────────────────────────────────────────────────────

class TestRegistration:
    """Averaging is constructive only about a symmetry centre; registration
    recovers an off-centre axis, which is the tool's default operating mode."""

    OFF = _hex_lattice(center=(C + 5.3, C - 3.1))

    def test_registered_off_centre_lattice_survives(self):
        res = symmetrize_image(self.OFF, 6, register=True)
        assert res.symmetry_residual_norm < 0.06

    def test_unregistered_off_centre_lattice_degrades(self):
        res = symmetrize_image(self.OFF, 6, register=False)
        assert res.symmetry_residual_norm > 0.5

    def test_shifts_are_reported(self):
        res = symmetrize_image(self.OFF, 6, register=True)
        assert np.abs(res.shifts[1:]).max() > 1.0
        np.testing.assert_array_equal(res.shifts[0], (0.0, 0.0))

    def test_constant_image_skips_registration(self):
        res = symmetrize_image(np.ones((64, 64)), 6, register=True)
        np.testing.assert_array_equal(res.shifts, np.zeros((6, 2)))
        assert res.symmetry_residual_norm < 1e-10


# ─── denoising / residual honesty ───────────────────────────────────────────

class TestResidual:
    def test_noise_is_reduced(self):
        rng = np.random.default_rng(42)
        noisy = HEX + rng.normal(0.0, 0.5, HEX.shape)
        res = symmetrize_image(noisy, 6)
        full = res.coverage >= res.n_ops - 1e-3
        before = np.std((noisy - HEX)[full])
        after = np.std((res.result - HEX)[full])
        assert after < 0.5 * before

    def test_defect_lands_in_residual(self):
        # A sixfold average keeps 1/6 of a lone defect, so the residual at the
        # defect carries (n−1)/n of its amplitude.
        defect = 3.0 * np.exp(-(((XX - 40) ** 2 + (YY - 80) ** 2) / 30.0))
        res = symmetrize_image(HEX + defect, 6)
        assert res.residual[80, 40] > 0.7 * 3.0 * (5.0 / 6.0)
        iy, ix = np.unravel_index(np.nanargmax(res.residual), res.residual.shape)
        assert abs(iy - 80) <= 2 and abs(ix - 40) <= 2

    def test_residual_norm_scales_with_asymmetry(self):
        sym = symmetrize_image(HEX, 6).symmetry_residual_norm
        bumped = HEX + 3.0 * np.exp(-(((XX - 40) ** 2 + (YY - 80) ** 2) / 30.0))
        asym = symmetrize_image(bumped, 6).symmetry_residual_norm
        assert asym > 5.0 * sym


# ─── coverage and NaN handling ──────────────────────────────────────────────

class TestCoverage:
    def test_corners_have_partial_coverage(self):
        res = symmetrize_image(HEX, 6)
        assert res.coverage[2, 2] < res.n_ops - 0.5
        assert res.coverage[N // 2, N // 2] == pytest.approx(res.n_ops)

    def test_strict_coverage_nans_corners(self):
        res = symmetrize_image(HEX, 6, strict_coverage=True)
        assert np.isnan(res.result[2, 2])
        assert np.isfinite(res.result[N // 2, N // 2])

    def test_default_renormalizes_corners(self):
        res = symmetrize_image(HEX, 6)
        assert np.isfinite(res.result[2, 2])

    def test_nan_hole_filled_from_symmetry(self):
        holed = HEX.copy()
        holed[60:65, 60:65] = np.nan
        res = symmetrize_image(holed, 6)
        assert np.isfinite(res.result[62, 62])
        assert abs(res.result[62, 62] - HEX[62, 62]) < 0.3
        # The hole contributes nothing, so its residual is undefined there.
        assert np.isnan(res.residual[62, 62])


# ─── mirror (dihedral) mode ─────────────────────────────────────────────────

class TestMirror:
    def test_correct_axis_preserves_pattern(self):
        lat30 = _hex_lattice(angle0_deg=30.0)   # mirror-symmetric across 30°
        res = symmetrize_image(lat30, 1, mirror=True, mirror_axis_deg=30.0)
        assert res.n_ops == 2
        assert res.symmetry_residual_norm < 0.05

    def test_wrong_axis_degrades(self):
        lat30 = _hex_lattice(angle0_deg=30.0)
        res = symmetrize_image(lat30, 1, mirror=True, mirror_axis_deg=15.0)
        assert res.symmetry_residual_norm > 0.3

    def test_d6_doubles_op_count(self):
        res = symmetrize_image(HEX, 6, mirror=True, mirror_axis_deg=0.0)
        assert res.n_ops == 12
        assert res.shifts.shape == (12, 2)


# ─── symmetrize_fft processing op ───────────────────────────────────────────

class TestProcessingOp:
    """The op is registered end-to-end: step construction, dispatch, replay."""

    @staticmethod
    def _state(params):
        from probeflow.processing.state import ProcessingState, ProcessingStep
        return ProcessingState(steps=[ProcessingStep("symmetrize_fft", params)])

    def test_step_construction_accepts_op(self):
        self._state({"n_fold": 6})

    def test_dispatch_matches_backend(self):
        from probeflow.processing.state import apply_processing_state
        out = apply_processing_state(HEX, self._state({"n_fold": 6}))
        np.testing.assert_array_equal(out, symmetrize_filter(HEX, 6))

    def test_default_params_are_identity(self):
        from probeflow.processing.state import apply_processing_state
        np.testing.assert_array_equal(
            apply_processing_state(HEX, self._state({})), HEX)

    def test_replay_is_deterministic(self):
        # Registration shifts are recomputed on replay; same data → same result.
        from probeflow.processing.state import apply_processing_state
        state = self._state({"n_fold": 6, "mirror": True, "mirror_axis_deg": 30.0})
        a = apply_processing_state(HEX, state)
        b = apply_processing_state(HEX, state)
        np.testing.assert_array_equal(a, b)

    def test_step_dict_round_trip(self):
        from probeflow.processing.state import ProcessingStep
        params = {"n_fold": 4, "mirror": True, "mirror_axis_deg": 12.5,
                  "register": False, "interpolation": "cubic",
                  "strict_coverage": True}
        step = ProcessingStep.from_dict({"op": "symmetrize_fft", "params": params})
        assert step.op == "symmetrize_fft"
        assert step.params == params

    def test_gui_adapter_passthrough_includes_op(self):
        from probeflow.processing.gui_adapter import _FILTER_OPS_PASSTHROUGH
        assert "symmetrize_fft" in _FILTER_OPS_PASSTHROUGH


# ─── fold_axis_from_peaks ───────────────────────────────────────────────────

class TestFoldAxis:
    @staticmethod
    def _peaks(theta0_deg, n, r=50.0):
        th = np.radians(theta0_deg) + np.arange(n) * 2.0 * np.pi / n
        return np.column_stack([r * np.cos(th), r * np.sin(th)])

    def test_recovers_axis(self):
        assert fold_axis_from_peaks(self._peaks(12.0, 6), 6) == pytest.approx(12.0)

    def test_result_is_modulo_fold_angle(self):
        assert fold_axis_from_peaks(self._peaks(71.0, 6), 6) == pytest.approx(11.0)

    def test_robust_to_missing_peak(self):
        peaks = self._peaks(12.0, 6)[:-1]
        assert fold_axis_from_peaks(peaks, 6) == pytest.approx(12.0)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            fold_axis_from_peaks(np.empty((0, 2)), 6)

    def test_rejects_inconsistent_angles(self):
        # Two peaks exactly half a fold apart cancel in the circular mean.
        peaks = self._peaks(0.0, 1)[:1]
        peaks = np.vstack([peaks, self._peaks(30.0, 1)[:1]])
        with pytest.raises(ValueError, match="consistent"):
            fold_axis_from_peaks(peaks, 6)
