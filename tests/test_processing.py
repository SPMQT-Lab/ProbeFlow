"""Tests for probeflow.processing — the GUI-free image processing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing import (
    align_rows,
    BadSegment,
    apply_stm_background,
    correct_bad_scanline_segments,
    compute_scanline_profile,
    detect_bad_scanline_segments,
    detect_grains,
    edge_detect,
    export_png,
    facet_level,
    fourier_filter,
    fft_soft_border,
    gaussian_high_pass,
    gaussian_smooth,
    gmm_autoclip,
    measure_periodicity,
    periodic_notch_filter,
    remove_bad_lines,
    repair_bad_scanline_segments,
    STMBackgroundParams,
    preview_stm_background,
    set_zero_plane,
    stm_line_background,
    subtract_background,
    patch_interpolate,
)
from probeflow.cli import main as cli_main


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_image():
    """A flat 32×32 surface at unit height."""
    return np.ones((32, 32), dtype=np.float64)


@pytest.fixture
def tilted_image():
    """A 32×32 plane tilted along both axes with amplitude 10."""
    Y, X = np.mgrid[:32, :32]
    return (0.2 * X + 0.3 * Y + 5.0).astype(np.float64)


@pytest.fixture
def sine_image():
    """64×64 sinusoidal pattern; period 8 px along x."""
    Y, X = np.mgrid[:64, :64]
    return np.sin(2 * np.pi * X / 8.0).astype(np.float64)


@pytest.fixture
def bimodal_image():
    """Two clearly separated Gaussian height distributions — a 'surface+islands'."""
    rng = np.random.default_rng(42)
    img = rng.normal(loc=0.0, scale=0.1, size=(40, 40))
    img[10:20, 10:30] += 2.0  # island patch
    return img.astype(np.float64)


# ─── remove_bad_lines ────────────────────────────────────────────────────────

class TestRemoveBadLines:
    def test_flat_unchanged(self, flat_image):
        out = remove_bad_lines(flat_image)
        assert np.allclose(out, flat_image)

    def test_detects_and_repairs_only_injected_scanline_segment(self):
        yy, xx = np.mgrid[:16, :24]
        arr = (0.02 * yy + 0.01 * xx).astype(np.float64)
        damaged = arr.copy()
        damaged[7, 6:14] += 10.0

        segments = detect_bad_scanline_segments(damaged, threshold=5.0, method="step")
        assert [(s.line_index, s.start_col, s.end_col) for s in segments] == [
            (7, 6, 14)
        ]

        out = remove_bad_lines(damaged, threshold_mad=5.0, method="step")

        changed = np.zeros(damaged.shape, dtype=bool)
        changed[7, 6:14] = True
        np.testing.assert_array_equal(out[~changed], damaged[~changed])
        np.testing.assert_allclose(out[7, 6:14], arr[7, 6:14], atol=1e-12)

    def test_mad_outlier_method_repairs_only_segment(self):
        yy, xx = np.mgrid[:16, :24]
        arr = (0.02 * yy + 0.01 * xx).astype(np.float64)
        damaged = arr.copy()
        damaged[8, 5:11] -= 8.0

        corrected, info = correct_bad_scanline_segments(
            damaged, threshold=5.0, method="mad", polarity="dark")

        assert len(info.segments) == 1
        seg = info.segments[0]
        assert (seg.line_index, seg.start_col, seg.end_col) == (8, 5, 11)
        outside = np.ones(damaged.shape, dtype=bool)
        outside[8, 5:11] = False
        np.testing.assert_array_equal(corrected[outside], damaged[outside])
        np.testing.assert_allclose(corrected[8, 5:11], arr[8, 5:11], atol=1e-12)

    def test_bright_polarity_detects_positive_not_negative_segment(self):
        arr = np.ones((14, 20), dtype=np.float64)
        bright = arr.copy()
        bright[6, 5:12] += 10.0
        dark = arr.copy()
        dark[6, 5:12] -= 10.0

        bright_segments = detect_bad_scanline_segments(
            bright, threshold=5.0, method="step", polarity="bright")
        dark_segments = detect_bad_scanline_segments(
            dark, threshold=5.0, method="step", polarity="bright")

        assert [(s.line_index, s.start_col, s.end_col) for s in bright_segments] == [
            (6, 5, 12)
        ]
        assert dark_segments == []

    def test_dark_polarity_detects_negative_not_positive_segment(self):
        arr = np.ones((14, 20), dtype=np.float64)
        bright = arr.copy()
        bright[6, 5:12] += 10.0
        dark = arr.copy()
        dark[6, 5:12] -= 10.0

        dark_segments = detect_bad_scanline_segments(
            dark, threshold=5.0, method="step", polarity="dark")
        bright_segments = detect_bad_scanline_segments(
            bright, threshold=5.0, method="step", polarity="dark")

        assert [(s.line_index, s.start_col, s.end_col) for s in dark_segments] == [
            (6, 5, 12)
        ]
        assert bright_segments == []

    def test_minimum_segment_length_filters_short_candidates(self):
        arr = np.ones((14, 20), dtype=np.float64)
        arr[6, 4:7] += 10.0
        arr[8, 10:18] += 10.0

        segments = detect_bad_scanline_segments(
            arr,
            threshold=5.0,
            method="step",
            polarity="bright",
            min_segment_length_px=5,
        )

        assert [(s.line_index, s.start_col, s.end_col) for s in segments] == [
            (8, 10, 18)
        ]

    def test_max_adjacent_bad_lines_skips_unsafe_group(self):
        arr = np.ones((14, 20), dtype=np.float64)
        arr[5, 4:12] += 10.0
        arr[6, 4:12] += 10.0
        arr[7, 4:12] += 10.0

        corrected, info = correct_bad_scanline_segments(
            arr,
            threshold=5.0,
            method="step",
            polarity="bright",
            max_adjacent_bad_lines=2,
        )

        assert len(info.segments) == 3
        assert len(info.skipped_segments) == 3
        assert info.corrected_segments == ()
        np.testing.assert_array_equal(corrected, arr)

    def test_high_threshold_gives_no_correction(self):
        arr = np.ones((16, 16), dtype=np.float64)
        arr[7, 4:9] += 20.0

        corrected, info = correct_bad_scanline_segments(
            arr, threshold=1e9, method="step")

        assert info.segments == ()
        np.testing.assert_array_equal(corrected, arr)

    def test_preview_detection_does_not_mutate_image_data(self):
        arr = np.ones((12, 12), dtype=np.float64)
        arr[5, 3:8] += 10.0
        before = arr.copy()

        _segments = detect_bad_scanline_segments(arr, threshold=5.0, method="step")

        np.testing.assert_array_equal(arr, before)

    def test_repair_uses_provided_preview_segments_only(self):
        arr = np.ones((10, 10), dtype=np.float64)
        arr[4, 2:5] = 20.0
        arr[6, 7:9] = 30.0
        preview_segments = [BadSegment(4, 2, 5, 10.0, "step")]

        corrected, info = correct_bad_scanline_segments(
            arr, threshold=1e9, method="step")
        repaired, repair_info = repair_bad_scanline_segments(arr, preview_segments)

        np.testing.assert_array_equal(corrected, arr)
        assert info.segments == ()
        assert repair_info.segments == tuple(preview_segments)
        np.testing.assert_array_equal(repaired[6, 7:9], arr[6, 7:9])
        assert np.allclose(repaired[4, 2:5], 1.0)

    def test_nan_input_safe(self):
        arr = np.full((8, 8), np.nan)
        out = remove_bad_lines(arr)
        assert out.shape == arr.shape  # no crash

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold_mad"):
            remove_bad_lines(np.ones((4, 4)), threshold_mad=-1.0)


# ─── STM scan-line background ────────────────────────────────────────────────

class TestSTMBackground:
    def test_linear_background_subtraction_removes_scanline_drift(self):
        yy, xx = np.mgrid[:40, :24]
        arr = 0.25 * yy + 0.02 * xx + 3.0

        params = STMBackgroundParams(model="linear", line_statistic="median")
        result = preview_stm_background(arr, params)

        assert result.fit_status == "success"
        assert result.background_image.shape == arr.shape
        assert float(np.nanstd(np.nanmedian(result.corrected, axis=1))) < 1e-10
        assert abs(float(np.nanmedian(result.corrected)) - float(np.nanmedian(arr))) < 1e-10

    def test_polynomial_background_subtraction_removes_quadratic_drift(self):
        y = np.linspace(-1.0, 1.0, 48)
        arr = (2.0 * y**2 - 0.4 * y + 5.0)[:, None] + np.zeros((48, 20))

        corrected = apply_stm_background(
            arr,
            STMBackgroundParams(model="poly2", line_statistic="median"),
        )

        assert float(np.nanstd(np.nanmedian(corrected, axis=1))) < 1e-10

    def test_low_pass_returns_smooth_background_profile(self):
        rng = np.random.default_rng(5)
        y = np.linspace(0.0, 4.0 * np.pi, 80)
        profile = np.sin(y) + rng.normal(scale=0.25, size=80)
        arr = profile[:, None] + np.zeros((80, 16))

        result = preview_stm_background(
            arr,
            STMBackgroundParams(model="low_pass", blur_length=4.0),
        )

        raw_roughness = np.nanstd(np.diff(result.line_profile))
        fit_roughness = np.nanstd(np.diff(result.fitted_profile))
        assert fit_roughness < raw_roughness

    def test_line_by_line_uses_raw_scanline_profile(self):
        profile = np.linspace(1.0, 3.0, 10)
        arr = profile[:, None] + np.zeros((10, 6))

        result = preview_stm_background(arr, STMBackgroundParams(model="line_by_line"))

        np.testing.assert_allclose(result.fitted_profile, profile)

    def test_median_line_statistic_is_robust_against_local_outlier(self):
        arr = np.ones((8, 8), dtype=float)
        arr[3, 0] = 100.0

        median_profile = compute_scanline_profile(arr, statistic="median")
        mean_profile = compute_scanline_profile(arr, statistic="mean")

        assert median_profile[3] == 1.0
        assert mean_profile[3] > 10.0

    def test_fit_roi_estimates_background_but_subtracts_full_image(self):
        yy, _xx = np.mgrid[:30, :20]
        arr = 0.2 * yy + np.zeros((30, 20))
        arr[:, 12:] += 10.0
        mask = np.zeros(arr.shape, dtype=bool)
        mask[:, :8] = True

        result = preview_stm_background(
            arr,
            STMBackgroundParams(fit_region="active_roi", model="linear"),
            mask=mask,
        )

        assert float(np.nanstd(np.nanmedian(result.corrected[:, :8], axis=1))) < 1e-10
        assert float(np.nanstd(np.nanmedian(result.corrected[:, 12:], axis=1))) < 1e-10
        assert abs(
            float(np.nanmedian(result.corrected[:, 12:]))
            - float(np.nanmedian(result.corrected[:, :8]))
            - 10.0
        ) < 1e-10

    def test_preview_does_not_modify_image_data(self):
        arr = np.arange(100, dtype=float).reshape(10, 10)
        before = arr.copy()

        _result = preview_stm_background(arr, STMBackgroundParams(model="linear"))

        np.testing.assert_array_equal(arr, before)

    def test_unsupported_creep_model_fails_without_modifying_image(self):
        arr = np.ones((10, 10), dtype=float)
        before = arr.copy()

        with pytest.raises(ValueError, match="STM background model"):
            preview_stm_background(arr, STMBackgroundParams(model="piezo_creep"))

        np.testing.assert_array_equal(arr, before)


# ─── subtract_background ────────────────────────────────────────────────────

class TestSubtractBackground:
    # ── normalised coordinate grid shared by higher-order tests ──────────────
    @staticmethod
    def _grid():
        return np.mgrid[-1:1:64j, -1:1:64j]  # yy, xx each (64, 64)

    def test_order1_removes_tilt(self, tilted_image):
        out = subtract_background(tilted_image, order=1)
        assert abs(float(np.mean(out))) < 1e-6
        assert float(np.ptp(out)) < 1e-6  # flat after plane fit

    def test_order1_removes_normalised_plane(self):
        yy, xx = self._grid()
        bg = 1.5 + 0.2 * xx - 0.4 * yy
        signal = np.zeros_like(xx)
        out = subtract_background(bg + signal, order=1)
        assert float(np.ptp(out)) < 1e-8

    def test_fit_rect_uses_selection_but_subtracts_whole_image(self):
        y = np.linspace(-1.0, 1.0, 20)
        x = np.linspace(-1.0, 1.0, 20)
        X, Y = np.meshgrid(x, y)
        arr = 2.0 * X - 0.5 * Y + 7.0
        arr[:, 12:] += 25.0

        out = subtract_background(arr, order=1, fit_rect=(0, 0, 8, 19))

        assert float(np.nanstd(out[:, :9])) < 1e-10
        assert abs(float(np.nanmedian(out[:, 12:])) - 25.0) < 1e-10

    def test_fit_mask_uses_selection_but_subtracts_whole_image(self):
        y = np.linspace(-1.0, 1.0, 20)
        x = np.linspace(-1.0, 1.0, 20)
        X, Y = np.meshgrid(x, y)
        arr = 2.0 * X - 0.5 * Y + 7.0
        arr[:, 12:] += 25.0
        mask = np.zeros(arr.shape, dtype=bool)
        mask[:, :9] = True

        out = subtract_background(arr, order=1, fit_mask=mask)

        assert float(np.nanstd(out[:, :9])) < 1e-10
        assert abs(float(np.nanmedian(out[:, 12:])) - 25.0) < 1e-10

    def test_order2_removes_quadratic(self):
        Y, X = np.mgrid[:20, :20]
        quad = (0.01 * X**2 + 0.02 * Y**2 + 0.1 * X + 3.0).astype(np.float64)
        out = subtract_background(quad, order=2)
        assert float(np.ptp(out)) < 1e-6

    def test_order2_removes_full_quadratic_surface(self):
        yy, xx = self._grid()
        bg = 1.0 + 0.2 * xx - 0.4 * yy + 0.1 * xx**2 + 0.05 * xx * yy - 0.08 * yy**2
        out = subtract_background(bg, order=2)
        assert float(np.ptp(out)) < 1e-8

    def test_order3_removes_cubic_background(self):
        yy, xx = self._grid()
        bg = (1.0 + 0.2 * xx - 0.4 * yy
              + 0.1 * xx**2 - 0.08 * yy**2
              + 0.03 * xx**3 - 0.02 * xx**2 * yy + 0.01 * xx * yy**2)
        out = subtract_background(bg, order=3)
        assert float(np.ptp(out)) < 1e-8

    def test_order3_removes_linear_background(self):
        yy, xx = self._grid()
        bg = 1.0 + 0.2 * xx - 0.4 * yy  # purely linear
        out = subtract_background(bg, order=3)
        # A cubic fit should still remove a purely linear background exactly
        assert float(np.ptp(out)) < 1e-8

    def test_order4_removes_quartic_background(self):
        yy, xx = self._grid()
        bg = (1.0 + 0.2 * xx - 0.4 * yy
              + 0.1 * xx**2 - 0.08 * yy**2
              + 0.03 * xx**3
              + 0.02 * xx**4 - 0.015 * xx**2 * yy**2)
        out = subtract_background(bg, order=4)
        assert float(np.ptp(out)) < 1e-7

    def test_invalid_order_raises(self):
        arr = np.ones((8, 8))
        with pytest.raises(ValueError, match="order must be 1..4"):
            subtract_background(arr, order=5)

    def test_order_zero_raises(self):
        arr = np.ones((8, 8))
        with pytest.raises(ValueError):
            subtract_background(arr, order=0)

    def test_negative_order_raises(self):
        arr = np.ones((8, 8))
        with pytest.raises(ValueError):
            subtract_background(arr, order=-1)

    def test_nan_preserved_in_output(self):
        yy, xx = self._grid()
        bg = 0.2 * xx - 0.4 * yy
        arr = bg.copy()
        arr[10:20, 10:20] = np.nan  # NaN patch
        out = subtract_background(arr, order=1)
        assert np.all(np.isnan(out[10:20, 10:20]))
        assert np.all(np.isfinite(out[:10, :]))

    def test_nan_fit_uses_finite_pixels_only(self):
        yy, xx = self._grid()
        bg = 0.5 * xx - 0.3 * yy
        arr = bg.copy()
        arr[30:, :] = np.nan  # mask half the image
        out = subtract_background(arr, order=1)
        # Finite half should be nearly flat
        finite_out = out[:30, :]
        assert float(np.ptp(finite_out)) < 1e-7

    def test_preserves_shape(self, tilted_image):
        assert subtract_background(tilted_image).shape == tilted_image.shape

    def test_invalid_step_threshold_raises(self):
        with pytest.raises(ValueError, match="step_threshold_deg"):
            subtract_background(
                np.ones((8, 8)),
                order=1,
                step_tolerance=True,
                step_threshold_deg=float("nan"),
            )


# ─── stm_line_background ─────────────────────────────────────────────────────

class TestStmLineBackground:
    @staticmethod
    def _stepped_drift_image():
        Ny, Nx = 48, 80
        row_steps = np.where(np.arange(Ny) % 2 == 0, 0.03, 0.01)
        row_drift = np.cumsum(row_steps)
        arr = row_drift[:, None] + np.zeros((Ny, Nx), dtype=np.float64)
        arr[:, :30] += 2.5
        return arr

    def test_step_tolerant_mode_reduces_row_drift(self):
        arr = self._stepped_drift_image()
        out = stm_line_background(arr)
        before = float(np.std(np.nanmedian(arr, axis=1)))
        after = float(np.std(np.nanmedian(out, axis=1)))
        assert after < before * 0.05

    def test_step_contrast_is_preserved(self):
        arr = self._stepped_drift_image()
        out = stm_line_background(arr)
        before = float(np.nanmedian(arr[:, :30]) - np.nanmedian(arr[:, 30:]))
        after = float(np.nanmedian(out[:, :30]) - np.nanmedian(out[:, 30:]))
        assert abs(after - before) < 1e-9

    def test_shape_dtype_and_nans_stable(self):
        arr = self._stepped_drift_image().astype(np.float32)
        arr[5, 10:20] = np.nan
        out = stm_line_background(arr)
        assert out.shape == arr.shape
        assert out.dtype == np.float64
        assert np.all(np.isnan(out[5, 10:20]))

    def test_no_usable_adjacent_differences_returns_copy(self):
        arr = np.array([[1.0, np.nan], [np.nan, 2.0]])
        out = stm_line_background(arr)
        np.testing.assert_array_equal(out, arr)
        assert out is not arr

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="step_tolerant"):
            stm_line_background(np.ones((4, 4)), mode="other")


# ─── align_rows ──────────────────────────────────────────────────────────────

class TestAlignRows:
    def test_median_zeros_row_medians(self):
        rng = np.random.default_rng(0)
        arr = rng.normal(size=(20, 30)) + np.arange(20).reshape(-1, 1) * 5.0
        out = align_rows(arr, method="median")
        for r in range(20):
            assert abs(float(np.median(out[r]))) < 1e-9

    def test_mean_zeros_row_means(self):
        arr = np.tile(np.arange(10.0), (5, 1)) + np.arange(5).reshape(-1, 1) * 7
        out = align_rows(arr, method="mean")
        for r in range(5):
            assert abs(float(np.mean(out[r]))) < 1e-9

    def test_linear_removes_per_row_slope(self):
        xs = np.linspace(-1, 1, 32)
        arr = np.stack([xs * (r + 1) for r in range(10)])
        out = align_rows(arr, method="linear")
        # After fitting+subtracting per-row linear trend, residuals ≈ 0
        assert np.allclose(out, 0.0, atol=1e-10)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            align_rows(np.ones((4, 4)), method="mode")


# ─── facet_level ─────────────────────────────────────────────────────────────

class TestFacetLevel:
    def test_flat_stays_flat(self, flat_image):
        out = facet_level(flat_image)
        assert np.allclose(out - out.mean(), 0.0)

    def test_handles_small_images(self):
        arr = np.ones((2, 2))
        out = facet_level(arr)
        assert out.shape == arr.shape


# ─── fourier_filter ──────────────────────────────────────────────────────────

class TestFourierFilter:
    def test_low_pass_preserves_constant_image(self):
        arr = np.ones((32, 32), dtype=float) * 5.0
        out = fourier_filter(arr, mode="low_pass", cutoff=0.2)
        np.testing.assert_allclose(out, arr, atol=1e-12)

    def test_high_pass_removes_constant_image(self):
        arr = np.ones((32, 32), dtype=float) * 5.0
        out = fourier_filter(arr, mode="high_pass", cutoff=0.2)
        np.testing.assert_allclose(out, np.zeros_like(arr), atol=1e-12)

    def test_radial_low_pass_reduces_high_frequency_ripple(self):
        Y, X = np.mgrid[:64, :64]
        arr = np.sin(2 * np.pi * X / 2.0)  # 2-pixel period → high freq
        out = fourier_filter(arr, mode="low_pass", cutoff=0.05)
        # Output amplitude should be much smaller
        assert float(np.std(out)) < float(np.std(arr)) * 0.5

    def test_radial_high_pass_removes_broad_dc_background(self):
        # Mostly DC + a small high-frequency ripple. After high-pass, the mean
        # should drop close to zero while the ripple amplitude survives.
        Y, X = np.mgrid[:32, :32]
        arr = 10.0 + 0.5 * np.sin(2 * np.pi * X / 2.0)
        out = fourier_filter(arr, mode="high_pass", cutoff=0.2)
        assert abs(float(np.mean(out))) < 1.0
        assert float(np.mean(out)) < float(np.mean(arr))

    def test_shape_preserved_for_non_square_images(self):
        arr = np.random.default_rng(0).normal(size=(20, 16))
        out = fourier_filter(arr, mode="low_pass", cutoff=0.3)
        assert out.shape == arr.shape

    def test_nan_input_preserves_nan_mask(self):
        arr = np.random.default_rng(0).normal(size=(20, 16))
        arr[4, 5] = np.nan
        out = fourier_filter(arr, mode="low_pass", cutoff=0.3)
        assert out.shape == arr.shape
        assert np.isnan(out[4, 5])
        assert np.all(np.isfinite(out[np.isfinite(arr)]))

    def test_invalid_parameters_raise(self):
        arr = np.ones((8, 8))
        with pytest.raises(ValueError, match="mode"):
            fourier_filter(arr, mode="bad")
        with pytest.raises(ValueError, match="cutoff"):
            fourier_filter(arr, cutoff=1.5)
        with pytest.raises(ValueError, match="window"):
            fourier_filter(arr, window="bad")


class TestGaussianHighPass:
    def test_removes_broad_background_but_keeps_ripple(self):
        Y, X = np.mgrid[:64, :64]
        broad = 5.0 + 0.05 * X
        ripple = 0.5 * np.sin(2 * np.pi * X / 4.0)
        arr = broad + ripple
        out = gaussian_high_pass(arr, sigma_px=10.0)
        assert abs(float(np.mean(out))) < 0.2
        assert float(np.std(out)) > 0.1

    def test_preserves_nan_mask(self):
        arr = np.ones((12, 12), dtype=float)
        arr[3, 4] = np.nan
        out = gaussian_high_pass(arr, sigma_px=4.0)
        assert np.isnan(out[3, 4])

    def test_all_nan_stays_nan(self):
        arr = np.full((8, 8), np.nan)
        out = gaussian_high_pass(arr, sigma_px=4.0)
        assert np.isnan(out).all()

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma_px"):
            gaussian_high_pass(np.ones((8, 8)), sigma_px=float("nan"))


class TestFftSoftBorder:
    def test_high_pass_removes_constant_image(self):
        arr = np.ones((32, 32), dtype=float) * 5.0
        out = fft_soft_border(arr, mode="high_pass", cutoff=0.2)
        np.testing.assert_allclose(out, np.zeros_like(arr), atol=1e-12)

    def test_invalid_cutoff_raises(self):
        with pytest.raises(ValueError, match="cutoff"):
            fft_soft_border(np.ones((8, 8)), cutoff=-0.1)


class TestPeriodicNotchFilter:
    def test_suppresses_selected_periodic_peak(self):
        Y, X = np.mgrid[:64, :64]
        arr = np.sin(2 * np.pi * X / 8.0)
        out = periodic_notch_filter(arr, [(8, 0)], radius_px=2.0)
        assert float(np.std(out)) < float(np.std(arr)) * 0.35

    def test_preserves_shape_and_nan_mask(self):
        arr = np.random.default_rng(4).normal(size=(20, 24))
        arr[2, 3] = np.nan
        out = periodic_notch_filter(arr, [(3, 2)], radius_px=2.0)
        assert out.shape == arr.shape
        assert np.isnan(out[2, 3])

    def test_invalid_radius_raises(self):
        with pytest.raises(ValueError, match="radius_px"):
            periodic_notch_filter(np.ones((8, 8)), [(2, 0)], radius_px=-1.0)


class TestPatchInterpolate:
    def test_fills_masked_patch_from_surroundings(self):
        arr = np.ones((16, 16), dtype=float)
        arr[6:10, 6:10] = 20.0
        mask = np.zeros_like(arr, dtype=bool)
        mask[6:10, 6:10] = True
        out = patch_interpolate(arr, mask, iterations=80)
        assert float(np.mean(out[6:10, 6:10])) < 5.0
        np.testing.assert_array_equal(out[~mask], arr[~mask])


# ─── gaussian_smooth ─────────────────────────────────────────────────────────

class TestGaussianSmooth:
    def test_reduces_variance(self):
        rng = np.random.default_rng(1)
        arr = rng.normal(size=(32, 32))
        out = gaussian_smooth(arr, sigma_px=3.0)
        assert float(np.var(out)) < float(np.var(arr)) * 0.5

    def test_handles_nan(self):
        arr = np.ones((10, 10))
        arr[5, 5] = np.nan
        out = gaussian_smooth(arr, sigma_px=1.0)
        assert np.isnan(out[5, 5])
        assert np.isfinite(out[0, 0])

    def test_all_nan_stays_nan(self):
        arr = np.full((8, 8), np.nan)
        out = gaussian_smooth(arr, sigma_px=1.0)
        assert np.isnan(out).all()

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma_px"):
            gaussian_smooth(np.ones((8, 8)), sigma_px=-0.5)


# ─── edge_detect ─────────────────────────────────────────────────────────────

class TestEdgeDetect:
    def test_laplacian_flat_is_zero(self, flat_image):
        out = edge_detect(flat_image, method="laplacian")
        assert np.allclose(out, 0.0, atol=1e-9)

    def test_log_shape(self, flat_image):
        out = edge_detect(flat_image, method="log", sigma=1.0)
        assert out.shape == flat_image.shape

    def test_dog_shape(self, flat_image):
        out = edge_detect(flat_image, method="dog", sigma=1.0, sigma2=2.0)
        assert out.shape == flat_image.shape

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            edge_detect(np.zeros((4, 4)), method="bogus")

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            edge_detect(np.ones((8, 8)), method="log", sigma=float("nan"))


# ─── gmm_autoclip ────────────────────────────────────────────────────────────

class TestGmmAutoclip:
    def test_returns_two_percentiles(self, bimodal_image):
        low, high = gmm_autoclip(bimodal_image)
        assert 0.0 <= low <= 10.0
        assert 90.0 <= high <= 100.0
        assert low < high

    def test_fallback_on_tiny_array(self):
        low, high = gmm_autoclip(np.array([1.0, 2.0]))
        assert (low, high) == (1.0, 99.0)


# ─── detect_grains ───────────────────────────────────────────────────────────

class TestDetectGrains:
    def test_finds_island(self, bimodal_image):
        _labels, n, stats = detect_grains(
            bimodal_image, threshold_pct=60.0, above=True, min_grain_px=5
        )
        assert n >= 1
        assert "areas_px" in stats
        assert all(a >= 5 for a in stats["areas_px"])

    def test_min_px_filters_noise_grains(self, bimodal_image):
        # With a very high threshold + large min_grain_px we should reject
        # noise speckles entirely.
        _labels, n, _ = detect_grains(bimodal_image,
                                       threshold_pct=99.5,
                                       above=True,
                                       min_grain_px=50)
        assert n == 0

    def test_all_nan_returns_zero(self):
        arr = np.full((10, 10), np.nan)
        _labels, n, _ = detect_grains(arr)
        assert n == 0


# ─── measure_periodicity ─────────────────────────────────────────────────────

class TestMeasurePeriodicity:
    def test_recovers_known_period(self, sine_image):
        # Image has a 8-pixel period along x; pretend 1 px = 1 nm.
        peaks = measure_periodicity(sine_image,
                                    pixel_size_x_m=1e-9,
                                    pixel_size_y_m=1e-9,
                                    n_peaks=3)
        assert len(peaks) >= 1
        # Dominant period should be close to 8 nm
        dominant = peaks[0]["period_m"]
        assert abs(dominant - 8e-9) / 8e-9 < 0.2

    def test_flat_image_empty_or_weak(self, flat_image):
        peaks = measure_periodicity(flat_image,
                                    pixel_size_x_m=1e-9,
                                    pixel_size_y_m=1e-9)
        # Either no peaks or only zero-strength ones
        for p in peaks:
            assert p["strength"] >= 0.0


# ─── export_png ──────────────────────────────────────────────────────────────

class TestExportPng:
    def test_writes_file(self, tmp_path):
        arr = np.linspace(0, 1, 32 * 32).reshape(32, 32)
        out = tmp_path / "out.png"

        def _lut(_key):
            return np.stack([np.arange(256, dtype=np.uint8)] * 3, axis=1)

        export_png(arr, out, "gray", 1.0, 99.0,
                   lut_fn=_lut, scan_range_m=(1e-8, 1e-8))
        assert out.exists() and out.stat().st_size > 0

    def test_no_scalebar_no_range(self, tmp_path):
        arr = np.linspace(0, 1, 16 * 16).reshape(16, 16)
        out = tmp_path / "out.png"

        def _lut(_key):
            return np.stack([np.arange(256, dtype=np.uint8)] * 3, axis=1)

        # Zero scan range = skip scale bar; should still write a valid PNG.
        export_png(arr, out, "gray", 1.0, 99.0,
                   lut_fn=_lut, scan_range_m=(0.0, 0.0),
                   add_scalebar=False)
        assert out.exists()

    def test_raises_on_all_nan(self, tmp_path):
        arr = np.full((8, 8), np.nan)

        def _lut(_key):
            return np.stack([np.arange(256, dtype=np.uint8)] * 3, axis=1)

        with pytest.raises(ValueError):
            export_png(arr, tmp_path / "x.png", "gray", 1.0, 99.0,
                       lut_fn=_lut, scan_range_m=(0.0, 0.0))


# ─── manual zero plane ───────────────────────────────────────────────────────

class TestSetZeroPlane:
    def test_three_points_define_plane_to_subtract(self):
        yy, xx = np.mgrid[:12, :10]
        arr = 2.0 * xx - 0.75 * yy + 6.0

        out = set_zero_plane(arr, [(0, 0), (9, 0), (0, 11)], patch=0)

        np.testing.assert_allclose(out, np.zeros_like(arr), atol=1e-12)
        assert out.dtype == np.float64

    def test_degenerate_points_leave_copy_unchanged(self):
        yy, xx = np.mgrid[:8, :8]
        arr = xx + yy

        out = set_zero_plane(arr, [(0, 0), (1, 1), (2, 2)], patch=0)

        np.testing.assert_allclose(out, arr)
        assert out is not arr

    def test_nan_pixels_are_preserved(self):
        yy, xx = np.mgrid[:8, :8]
        arr = 0.5 * xx + 1.5 * yy + 2.0
        arr = arr.astype(float)
        arr[4, 4] = np.nan

        out = set_zero_plane(arr, [(0, 0), (7, 0), (0, 7)], patch=0)

        assert np.isnan(out[4, 4])
        finite = np.isfinite(out)
        np.testing.assert_allclose(out[finite], 0.0, atol=1e-12)


# ─── CLI: plane-bg order extension ───────────────────────────────────────────

class TestPlaneBgCli:
    def test_order3_via_cli(self, first_sample_dat, tmp_path):
        out = tmp_path / "out.sxm"
        rc = cli_main(["plane-bg", str(first_sample_dat),
                       "--order", "3", "-o", str(out)])
        assert rc == 0
        assert out.exists()

    def test_order4_via_cli(self, first_sample_dat, tmp_path):
        out = tmp_path / "out.sxm"
        rc = cli_main(["plane-bg", str(first_sample_dat),
                       "--order", "4", "-o", str(out)])
        assert rc == 0
        assert out.exists()

    def test_order3_records_history(self, first_sample_dat, tmp_path):
        from unittest.mock import patch
        captured = []

        def _capture(args, scan, default_suffix):
            captured.append(scan)
            return tmp_path / "out.sxm"

        with patch("probeflow.cli._write_output", side_effect=_capture):
            cli_main(["plane-bg", str(first_sample_dat), "--order", "3"])
        assert captured[0].processing_history[0]["params"]["order"] == 3

    def test_order5_via_cli_rejected(self, first_sample_dat, tmp_path):
        import sys
        with pytest.raises(SystemExit):
            cli_main(["plane-bg", str(first_sample_dat), "--order", "5"])

    def test_pipeline_order4_accepted(self, first_sample_dat, tmp_path):
        out = tmp_path / "out.sxm"
        rc = cli_main(["pipeline", str(first_sample_dat),
                       "-o", str(out), "--steps", "plane-bg:4"])
        assert rc == 0
        assert out.exists()

    def test_pipeline_order5_rejected(self, first_sample_dat, tmp_path):
        rc = cli_main(["pipeline", str(first_sample_dat),
                       "--steps", "plane-bg:5"])
        assert rc == 2
