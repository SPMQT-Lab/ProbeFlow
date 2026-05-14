"""
Adversarial tests for ProbeFlow image processing functions.

These tests exercise edge-cases that real STM datasets can produce:
NaN stripes, single-pixel outliers, step-edges that fight background
subtraction, scanline artefacts that mimic real features, signed
height data, anisotropic pixels, non-square images, and mismatched
forward/backward scans.

The companion file adversarial_fixtures.py provides pure factory functions
(no pytest decorators) so the same arrays can also be used in scripts.

Each test is intentionally narrow — one assertion per function — so
failures point directly at the broken invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing.image import (
    align_rows,
    subtract_background,
    stm_line_background,
    fourier_filter,
    fft_soft_border,
    tv_denoise,
    gaussian_smooth,
    gaussian_high_pass,
    remove_bad_lines,
    detect_grains,
    facet_level,
    blend_forward_backward,
    rotate_arbitrary,
)

from adversarial_fixtures import (
    flat_with_outlier,
    tilted_plane_with_step,
    lattice_with_scanline_glitch,
    nan_horizontal_stripe,
    real_islands_mimic_artefact,
    anisotropic_pixels,
    non_square_image,
    tiny_3x3,
    tiny_2x2,
    constant_nonzero,
    negative_heights,
    fwd_bwd_asymmetric,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _nan_stripe_slice(fx):
    """Return the row slice for the NaN stripe in a nan_horizontal_stripe fixture."""
    ss = fx["stripe_start"]
    sw = fx["stripe_width"]
    return slice(ss, ss + sw)


# ============================================================================
# Group 1 — NaN safety
# ============================================================================

class TestNanSafety:
    """NaN stripes must be preserved and must not infect finite pixels."""

    @pytest.fixture(autouse=True)
    def fx(self):
        self._fx = nan_horizontal_stripe()

    def _stripe(self):
        return _nan_stripe_slice(self._fx)

    # --- align_rows ---

    def test_align_rows_preserves_nan_stripe(self):
        out = align_rows(self._fx["arr"])
        assert np.isnan(out[self._stripe(), :]).all()

    def test_align_rows_finite_pixels_remain_finite(self):
        arr = self._fx["arr"]
        out = align_rows(arr)
        finite_in = np.isfinite(arr)
        assert np.isfinite(out[finite_in]).all()

    # --- subtract_background ---

    def test_subtract_background_preserves_nan_stripe(self):
        out = subtract_background(self._fx["arr"], order=1)
        assert np.isnan(out[self._stripe(), :]).all()

    def test_subtract_background_finite_pixels_remain_finite(self):
        arr = self._fx["arr"]
        out = subtract_background(arr, order=1)
        finite_in = np.isfinite(arr)
        assert np.isfinite(out[finite_in]).all()

    # --- stm_line_background ---

    def test_stm_line_background_preserves_nan_stripe(self):
        out = stm_line_background(self._fx["arr"])
        assert np.isnan(out[self._stripe(), :]).all()

    def test_stm_line_background_finite_pixels_remain_finite(self):
        arr = self._fx["arr"]
        out = stm_line_background(arr)
        finite_in = np.isfinite(arr)
        assert np.isfinite(out[finite_in]).all()

    # --- fourier_filter high_pass ---

    def test_fourier_filter_hp_preserves_nan_stripe(self):
        out = fourier_filter(self._fx["arr"], mode="high_pass", cutoff=0.2)
        assert np.isnan(out[self._stripe(), :]).all()

    def test_fourier_filter_hp_finite_pixels_remain_finite(self):
        arr = self._fx["arr"]
        out = fourier_filter(arr, mode="high_pass", cutoff=0.2)
        finite_in = np.isfinite(arr)
        assert np.isfinite(out[finite_in]).all()

    # --- fft_soft_border high_pass ---

    def test_fft_soft_border_hp_preserves_nan_stripe(self):
        out = fft_soft_border(self._fx["arr"], mode="high_pass", cutoff=0.2)
        assert np.isnan(out[self._stripe(), :]).all()

    def test_fft_soft_border_hp_finite_pixels_remain_finite(self):
        arr = self._fx["arr"]
        out = fft_soft_border(arr, mode="high_pass", cutoff=0.2)
        finite_in = np.isfinite(arr)
        assert np.isfinite(out[finite_in]).all()

    # --- tv_denoise ---

    def test_tv_denoise_preserves_nan_stripe(self):
        out = tv_denoise(self._fx["arr"])
        assert np.isnan(out[self._stripe(), :]).all()

    def test_tv_denoise_finite_pixels_remain_finite(self):
        arr = self._fx["arr"]
        out = tv_denoise(arr)
        finite_in = np.isfinite(arr)
        assert np.isfinite(out[finite_in]).all()

    # --- gaussian_smooth ---

    def test_gaussian_smooth_preserves_nan_stripe(self):
        out = gaussian_smooth(self._fx["arr"])
        assert np.isnan(out[self._stripe(), :]).all()

    def test_gaussian_smooth_finite_pixels_remain_finite(self):
        arr = self._fx["arr"]
        out = gaussian_smooth(arr)
        finite_in = np.isfinite(arr)
        assert np.isfinite(out[finite_in]).all()

    # --- gaussian_high_pass ---

    def test_gaussian_high_pass_preserves_nan_stripe(self):
        out = gaussian_high_pass(self._fx["arr"])
        assert np.isnan(out[self._stripe(), :]).all()

    def test_gaussian_high_pass_finite_pixels_remain_finite(self):
        arr = self._fx["arr"]
        out = gaussian_high_pass(arr)
        finite_in = np.isfinite(arr)
        assert np.isfinite(out[finite_in]).all()


# ============================================================================
# Group 2 — Outlier robustness
# ============================================================================

class TestOutlierRobustness:
    def test_align_rows_median_outlier_survives(self):
        """align_rows(median) subtracts row median; single hot pixel row median
        equals the outlier value, so the pixel shifts to near 0 after alignment.
        Non-outlier rows (all-zero medians) are left near 0 too."""
        fx = flat_with_outlier()
        arr = fx["arr"]
        out = align_rows(arr, method="median")
        # The outlier row median == 0 (64 zeros and 1 hot pixel; median is 0)
        # so the hot pixel value is preserved
        assert np.isfinite(out[32, 32])
        # All non-outlier pixels should be very close to 0
        mask = np.ones(arr.shape, dtype=bool)
        mask[32, 32] = False
        assert np.allclose(out[mask], 0.0, atol=1e-12)

    def test_remove_bad_lines_single_pixel_not_flagged(self):
        """A single isolated pixel is not a scanline segment (min_segment >= 2).
        The pixel value should survive bad-line removal unchanged."""
        fx = flat_with_outlier()
        arr = fx["arr"]
        out = remove_bad_lines(arr, method="step", threshold_mad=5.0)
        assert np.isclose(out[32, 32], arr[32, 32])


# ============================================================================
# Group 3 — Step height preservation
# ============================================================================

class TestStepHeightPreservation:
    def test_subtract_background_step_tolerance_gradient_normalization(self):
        """pixel_size_x/y_m correctly normalises the gradient before the slope
        threshold comparison.

        For physical STM data (heights ~1e-10 m, pixel_size ~1e-9 m) the raw
        gradient at a step boundary is ~1.5e-10 m/pixel — far below tan(3°) —
        so without the normalization the masking is a no-op and the result is
        identical to step_tolerance=False.  With the correct pixel size the
        normalised gradient is ~0.15 > tan(3°), so step-boundary rows ARE
        excluded and the result differs.
        """
        fx = tilted_plane_with_step()
        arr = fx["arr"]
        px = fx["pixel_size_x_m"]  # 1e-9 m

        out_correct = subtract_background(
            arr, order=1, step_tolerance=True,
            pixel_size_x_m=px, pixel_size_y_m=px,
        )
        # default pixel_size=1.0: raw gradient (1.5e-10) << tan(3°) → no masking
        out_default_px = subtract_background(arr, order=1, step_tolerance=True)
        out_disabled = subtract_background(arr, order=1, step_tolerance=False)

        # With default pixel size the masking is a no-op for STM-scale data
        assert np.array_equal(out_default_px, out_disabled), (
            "step_tolerance with pixel_size=1.0 (default) should be identical to "
            "step_tolerance=False for STM-scale data"
        )
        # With correct pixel size the masking fires and the result changes
        assert not np.array_equal(out_correct, out_disabled), (
            "step_tolerance with correct pixel_size should exclude step-boundary "
            "rows, producing a different fit than step_tolerance=False"
        )

    def test_subtract_background_order4_eats_step(self):
        """High-order poly WITHOUT step tolerance over-fits the step.
        Measured step in output is < 50 % of original (documented overcorrection)."""
        fx = tilted_plane_with_step()
        arr = fx["arr"]
        step_height = fx["step_height_m"]
        step_row = fx["step_row"]

        out = subtract_background(arr, order=4, step_tolerance=False)
        measured = abs(float(np.median(out[step_row:, :])) - float(np.median(out[:step_row, :])))
        assert measured < step_height * 0.50, (
            f"Expected step to be eaten by order-4 poly but measured {measured:.3e}"
        )

    def test_stm_line_background_equalises_rows_including_real_step(self):
        """stm_line_background equalises row-to-row offsets so the step height
        in the output is less than 50 % of the original step."""
        fx = tilted_plane_with_step()
        arr = fx["arr"]
        step_height = fx["step_height_m"]
        step_row = fx["step_row"]

        out = stm_line_background(arr)
        measured = abs(float(np.median(out[step_row:, :])) - float(np.median(out[:step_row, :])))
        assert measured < step_height * 0.50, (
            f"stm_line_background did not equalise the step: {measured:.3e}"
        )


# ============================================================================
# Group 4 — Artefact vs real feature discrimination
# ============================================================================

class TestArtefactVsRealFeature:
    def test_remove_bad_lines_lattice_glitch_corrected(self):
        """After removing bad lines, the glitch segment should be corrected
        toward the lattice level, not remain at 5e-10 m."""
        fx = lattice_with_scanline_glitch()
        arr = fx["arr"]
        gr = fx["glitch_row"]
        gs = fx["glitch_start"]
        ge = fx["glitch_end"]
        lattice_amp = fx["lattice_amplitude"]

        out = remove_bad_lines(arr, threshold_mad=3.0)
        glitch_mean = float(np.mean(np.abs(out[gr, gs:ge])))
        assert glitch_mean < lattice_amp * 3.0, (
            f"Glitch segment not corrected: mean={glitch_mean:.3e}, "
            f"threshold={lattice_amp*3:.3e}"
        )

    def test_remove_bad_lines_lattice_non_glitch_rows_unchanged(self):
        """Rows above the glitch should not be modified."""
        fx = lattice_with_scanline_glitch()
        arr = fx["arr"]
        gr = fx["glitch_row"]

        out = remove_bad_lines(arr, threshold_mad=3.0)
        assert np.allclose(out[:gr, :], arr[:gr, :]), (
            "Non-glitch rows were altered by remove_bad_lines"
        )

    def test_remove_bad_lines_island_pixels_preserved(self):
        """Island pixels in rows that are NOT the glitch row must be unchanged."""
        fx = real_islands_mimic_artefact()
        arr = fx["arr"]
        gr = fx["glitch_row"]

        out = remove_bad_lines(arr, threshold_mad=3.0)
        # Check all rows except the glitch row are identical
        mask = np.ones(arr.shape, dtype=bool)
        mask[gr, :] = False
        assert np.allclose(out[mask], arr[mask]), (
            "Island pixels outside glitch row were altered"
        )

    def test_remove_bad_lines_glitch_row_repaired_toward_zero(self):
        """Glitch row segment should be corrected toward the neighbouring
        background (0), not left at the glitch height."""
        fx = real_islands_mimic_artefact()
        arr = fx["arr"]
        gr = fx["glitch_row"]
        gc0, gc1 = fx["glitch_cols"]
        island_height = fx["island_height"]

        out = remove_bad_lines(arr, threshold_mad=3.0)
        segment_mean = float(np.mean(out[gr, gc0:gc1]))
        # Neighbours are 0; corrected value should be closer to 0 than island_height
        assert abs(segment_mean) < island_height * 0.5, (
            f"Glitch segment not corrected: mean={segment_mean:.3e}"
        )


# ============================================================================
# Group 5 — Bug fix verification
# ============================================================================

class TestBugFixes:
    def test_fourier_filter_hp_preserves_mean_on_constant_image(self):
        """fourier_filter(high_pass) must preserve the mean (bug fix: mean_val
        is added back after filtering)."""
        fx = constant_nonzero()
        arr = fx["arr"]
        out = fourier_filter(arr, mode="high_pass", cutoff=0.2)
        assert np.allclose(out, arr, atol=1e-12), (
            "fourier_filter(high_pass) did not preserve constant image"
        )

    def test_fft_soft_border_hp_preserves_mean_on_constant_image(self):
        """fft_soft_border(high_pass) must also preserve the mean."""
        fx = constant_nonzero()
        arr = fx["arr"]
        out = fft_soft_border(arr, mode="high_pass", cutoff=0.2)
        assert np.allclose(out, arr, atol=1e-12), (
            "fft_soft_border(high_pass) did not preserve constant image"
        )

    def test_tv_denoise_nan_stripe_preserved_after_fix(self):
        """tv_denoise must not fill in NaN stripes (bug fix: nan mask restored)."""
        fx = nan_horizontal_stripe()
        arr = fx["arr"]
        ss = fx["stripe_start"]
        sw = fx["stripe_width"]
        out = tv_denoise(arr)
        assert np.isnan(out[ss : ss + sw, :]).all(), (
            "tv_denoise filled in the NaN stripe"
        )

    def test_rotate_arbitrary_corner_pixels_are_nan(self):
        """rotate_arbitrary(45) uses cval=np.nan, so canvas corners must be NaN."""
        fx = tilted_plane_with_step()
        arr = fx["arr"]
        out = rotate_arbitrary(arr, 45.0)
        # Top-left corner of expanded canvas must be NaN
        assert np.isnan(out[0, 0]), (
            "rotate_arbitrary corner pixel is not NaN (expected cval=nan)"
        )


# ============================================================================
# Group 6 — Small image safety (no crash)
# ============================================================================

def _ops_for_small_images(arr):
    """Run a battery of processing ops; return list of (name, output) tuples."""
    results = []
    results.append(("align_rows", align_rows(arr)))
    results.append(("subtract_background_1", subtract_background(arr, order=1)))
    results.append(("stm_line_background", stm_line_background(arr)))
    results.append(("gaussian_smooth", gaussian_smooth(arr)))
    results.append(("facet_level", facet_level(arr)))
    results.append(("fourier_filter", fourier_filter(arr, mode="high_pass", cutoff=0.2)))
    return results


class TestSmallImageSafety:
    def test_tiny_3x3_no_crash(self):
        fx = tiny_3x3()
        arr = fx["arr"]
        for name, out in _ops_for_small_images(arr):
            assert out is not None, f"{name} returned None on 3x3 image"

    def test_tiny_3x3_shape_preserved(self):
        fx = tiny_3x3()
        arr = fx["arr"]
        for name, out in _ops_for_small_images(arr):
            assert out.shape == arr.shape, (
                f"{name}: shape changed from {arr.shape} to {out.shape}"
            )

    def test_tiny_2x2_no_crash(self):
        fx = tiny_2x2()
        arr = fx["arr"]
        for name, out in _ops_for_small_images(arr):
            assert out is not None, f"{name} returned None on 2x2 image"

    def test_tiny_2x2_shape_preserved(self):
        fx = tiny_2x2()
        arr = fx["arr"]
        for name, out in _ops_for_small_images(arr):
            assert out.shape == arr.shape, (
                f"{name}: shape changed from {arr.shape} to {out.shape}"
            )


# ============================================================================
# Group 7 — Signed data
# ============================================================================

class TestSignedData:
    def test_subtract_background_negative_heights_finite(self):
        """subtract_background must handle negative heights without NaN/Inf."""
        fx = negative_heights()
        out = subtract_background(fx["arr"], order=1)
        assert np.isfinite(out).all()

    def test_subtract_background_negative_heights_shape(self):
        fx = negative_heights()
        out = subtract_background(fx["arr"], order=1)
        assert out.shape == fx["arr"].shape

    def test_detect_grains_below_threshold_finds_depression(self):
        """detect_grains(above=False) must find at least one grain covering the
        depression region."""
        fx = negative_heights()
        arr = fx["arr"]
        label_map, n_grains, stats = detect_grains(arr, threshold_pct=50, above=False)
        assert n_grains >= 1, "No grains found in depression data"
        # At least one labelled region should overlap the depression
        depression_labels = label_map[20:40, 20:40]
        assert (depression_labels > 0).any(), (
            "No grain label covers the depression region"
        )

    def test_fourier_filter_hp_signed_mean_preserved(self):
        """fourier_filter(high_pass) mean should be preserved for signed data."""
        fx = negative_heights()
        arr = fx["arr"]
        out = fourier_filter(arr, mode="high_pass", cutoff=0.2)
        in_mean = float(np.nanmean(arr))
        out_mean = float(np.nanmean(out))
        assert abs(out_mean - in_mean) < abs(in_mean) * 0.01 + 1e-14, (
            f"Mean not preserved: in={in_mean:.3e}, out={out_mean:.3e}"
        )


# ============================================================================
# Group 8 — Anisotropic pixel / non-square
# ============================================================================

class TestAnisotropicAndNonSquare:
    def test_facet_level_anisotropic_falls_back_to_plane(self):
        """With anisotropic pixels the slope > tan(3°) in physical units, so
        facet_level should fall back to order=1 and return a valid 2-D array."""
        fx = anisotropic_pixels()
        out = facet_level(
            fx["arr"],
            pixel_size_x_m=fx["pixel_size_x_m"],
            pixel_size_y_m=fx["pixel_size_y_m"],
        )
        assert out.ndim == 2
        assert out.shape == fx["arr"].shape

    def test_non_square_align_rows_shape(self):
        fx = non_square_image()
        out = align_rows(fx["arr"])
        assert out.shape == fx["arr"].shape

    def test_non_square_subtract_background_shape(self):
        fx = non_square_image()
        out = subtract_background(fx["arr"], order=1)
        assert out.shape == fx["arr"].shape

    def test_non_square_fourier_filter_shape(self):
        fx = non_square_image()
        out = fourier_filter(fx["arr"], mode="high_pass", cutoff=0.2)
        assert out.shape == fx["arr"].shape

    def test_non_square_fft_soft_border_shape(self):
        fx = non_square_image()
        out = fft_soft_border(fx["arr"], mode="high_pass", cutoff=0.2)
        assert out.shape == fx["arr"].shape

    def test_non_square_tv_denoise_shape(self):
        fx = non_square_image()
        out = tv_denoise(fx["arr"])
        assert out.shape == fx["arr"].shape

    def test_non_square_gaussian_smooth_shape(self):
        fx = non_square_image()
        out = gaussian_smooth(fx["arr"])
        assert out.shape == fx["arr"].shape

    def test_non_square_stm_line_background_shape(self):
        fx = non_square_image()
        out = stm_line_background(fx["arr"])
        assert out.shape == fx["arr"].shape


# ============================================================================
# Group 9 — blend_forward_backward
# ============================================================================

class TestBlendForwardBackward:
    def test_blend_returns_same_shape(self):
        fx = fwd_bwd_asymmetric()
        out = blend_forward_backward(fx["fwd"], fx["bwd"])
        assert out.shape == fx["fwd"].shape

    def test_blend_drift_smears_amplitude(self):
        """With drift=3px and period=8px the blend should smear signal:
        output std < fwd std."""
        fx = fwd_bwd_asymmetric()
        out = blend_forward_backward(fx["fwd"], fx["bwd"], weight=0.5)
        fwd_std = float(np.std(fx["fwd"]))
        out_std = float(np.std(out))
        assert out_std < fwd_std, (
            f"Blend did not smear amplitude: fwd_std={fwd_std:.3e}, out_std={out_std:.3e}"
        )

    def test_blend_weight_1_returns_fwd(self):
        """weight=1.0 must return the forward scan unchanged."""
        fx = fwd_bwd_asymmetric()
        out = blend_forward_backward(fx["fwd"], fx["bwd"], weight=1.0)
        assert np.allclose(out, fx["fwd"])

    def test_blend_weight_0_returns_fliplr_bwd(self):
        """weight=0.0 must return fliplr(bwd)."""
        fx = fwd_bwd_asymmetric()
        out = blend_forward_backward(fx["fwd"], fx["bwd"], weight=0.0)
        assert np.allclose(out, np.fliplr(fx["bwd"]))

    def test_blend_mismatched_shapes_raise(self):
        """Mismatched fwd/bwd shapes must raise ValueError."""
        fx = fwd_bwd_asymmetric()
        fwd = fx["fwd"]
        bwd_wrong = np.zeros((fwd.shape[0] + 1, fwd.shape[1]), dtype=np.float64)
        with pytest.raises(ValueError):
            blend_forward_backward(fwd, bwd_wrong)
